"""音频混音工作台 API：轨道 CRUD、素材上传、冲突/响度分析与混音预览。

所有媒体引用必须通过 existing_file 的路径安全校验（允许根：output/assets/data），
上传素材受项目存储配额约束。轨道修改与字幕一样推进 av_config_version 并取消
项目作用域任务，防止旧配置的成片或预览发布。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api import schemas
from api.claim_guard import budget_notice, claim_or_block
from api.websocket import ws_manager
from config import settings
from db import SessionLocal, get_db
from models import AudioTrack, Project, Shot
from services.av_config_service import bump_av_config_version, collect_render_config
from services.error_reporter import ERROR_RENDER, error_payload, log_failure
from services.ffmpeg_service import FFmpegService
from services.shot_dialogue import dialogue_display_text, parse_shot_dialogue
from services.security import (
    UploadLimitExceeded,
    existing_file,
    safe_filename,
    safe_path,
    save_upload_stream,
    validate_identifier,
)
from services.storage_service import StorageQuotaExceeded, StorageService
from services.task_registry import (
    cancel_scopes,
    snapshot as task_snapshot,
    start as start_task,
    update_progress as update_job_progress,
)

router = APIRouter(prefix="/api/audio-track", tags=["audio-track"])
ffmpeg_service = FFmpegService()
storage_service = StorageService()

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".wma", ".aiff"}

# 最近一次预览结果（进程内缓存；任务态以 task_registry 的持久化快照为准）。
_preview_status: dict[str, dict] = {}


class AudioTrackCreate(BaseModel):
    kind: schemas.AudioTrackKind
    name: schemas.ShortKey | None = None
    # 上传素材的绝对路径（来自 /upload 返回值），经 existing_file 校验。
    source_path: str = ""
    # kind=dialogue 时必须绑定项目内镜头。
    shot_id: schemas.OptionalIdentifier = ""
    start_ms: int = Field(default=0, ge=0, le=86_400_000)
    volume: float = Field(default=1.0, ge=0.0, le=2.0)
    pan: float = Field(default=0.0, ge=-1.0, le=1.0)
    fade_in_ms: int = Field(default=0, ge=0, le=30_000)
    fade_out_ms: int = Field(default=0, ge=0, le=30_000)
    delay_ms: int = Field(default=0, ge=0, le=30_000)
    trim_start_ms: int = Field(default=0, ge=0, le=3_600_000)
    trim_end_ms: int = Field(default=0, ge=0, le=3_600_000)
    loop: bool = False
    muted: bool = False
    duck_amount_db: float = Field(default=0.0, ge=-48.0, le=0.0)
    duck_attack_ms: int = Field(default=120, ge=1, le=5_000)
    duck_release_ms: int = Field(default=480, ge=1, le=10_000)


class AudioTrackUpdate(AudioTrackCreate):
    project_id: schemas.Identifier
    kind: schemas.AudioTrackKind | None = None
    source_path: str | None = None
    shot_id: schemas.OptionalIdentifier | None = None
    start_ms: int | None = Field(default=None, ge=0, le=86_400_000)
    volume: float | None = Field(default=None, ge=0.0, le=2.0)
    pan: float | None = Field(default=None, ge=-1.0, le=1.0)
    fade_in_ms: int | None = Field(default=None, ge=0, le=30_000)
    fade_out_ms: int | None = Field(default=None, ge=0, le=30_000)
    delay_ms: int | None = Field(default=None, ge=0, le=30_000)
    trim_start_ms: int | None = Field(default=None, ge=0, le=3_600_000)
    trim_end_ms: int | None = Field(default=None, ge=0, le=3_600_000)
    loop: bool | None = None
    muted: bool | None = None
    duck_amount_db: float | None = Field(default=None, ge=-48.0, le=0.0)
    duck_attack_ms: int | None = Field(default=None, ge=1, le=5_000)
    duck_release_ms: int | None = Field(default=None, ge=1, le=10_000)


class AudioPreviewRequest(BaseModel):
    scope: str = "full"  # full / shot
    shot_id: schemas.OptionalIdentifier = ""


def _project_or_404(db: Session, project_id: str) -> Project:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


def _track_or_404(db: Session, track_id: str, project_id: str) -> AudioTrack:
    track = db.query(AudioTrack).filter(AudioTrack.id == track_id, AudioTrack.project_id == project_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="音频轨道不存在")
    return track


def _validate_ids(track_id: str | None = None, project_id: str | None = None) -> tuple[str | None, str | None]:
    try:
        if track_id is not None:
            track_id = validate_identifier(track_id, "音频轨道 ID")
        if project_id is not None:
            project_id = validate_identifier(project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return track_id, project_id


def _serialize_track(track: AudioTrack, shot_span: tuple[int, int] | None) -> dict:
    payload = {
        "id": track.id,
        "project_id": track.project_id,
        "kind": track.kind,
        "name": track.name or "",
        "source_path": track.source_path or "",
        "source_url": storage_service.get_relative_url(track.source_path) if track.source_path else "",
        "source_duration_ms": int(track.source_duration_ms or 0),
        "shot_id": track.shot_id or "",
        "start_ms": int(track.start_ms or 0),
        "volume": float(track.volume if track.volume is not None else 1.0),
        "pan": float(track.pan or 0.0),
        "fade_in_ms": int(track.fade_in_ms or 0),
        "fade_out_ms": int(track.fade_out_ms or 0),
        "delay_ms": int(track.delay_ms or 0),
        "trim_start_ms": int(track.trim_start_ms or 0),
        "trim_end_ms": int(track.trim_end_ms or 0),
        "loop": bool(track.loop),
        "muted": bool(track.muted),
        "duck_amount_db": float(track.duck_amount_db or 0.0),
        "duck_attack_ms": int(track.duck_attack_ms or 120),
        "duck_release_ms": int(track.duck_release_ms or 480),
        "order_index": int(track.order_index or 0),
    }
    if track.kind == "dialogue" and shot_span is not None:
        # 对白轨的实际时间线位置由镜头区间派生（start_ms 仅作展示回显）。
        payload["shot_span"] = {"start_ms": shot_span[0], "end_ms": shot_span[1]}
        payload["start_ms"] = shot_span[0]
    return payload


def _shot_infos(db: Session, project_id: str) -> tuple[list[dict], dict[str, tuple[int, int]], int]:
    shots = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
    infos: list[dict] = []
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for shot in shots:
        duration_ms = int(round(max(0.0, float(shot.duration or 0.0)) * 1000))
        spans[shot.id] = (cursor, cursor + duration_ms)
        try:
            speakers = json.loads(shot.characters_in_scene) if shot.characters_in_scene else []
        except (TypeError, ValueError):
            speakers = []
        try:
            profile = json.loads(shot.continuity_profile) if shot.continuity_profile else {}
        except (TypeError, ValueError):
            profile = {}
        lines = parse_shot_dialogue(
            shot.dialogue,
            fallback_speaker=speakers[0] if speakers and isinstance(speakers[0], str) else "",
            default_emotion=shot.emotion or "neutral",
            warn_key=f"audio_track shot {shot.id}",
        )
        infos.append(
            {
                "id": shot.id,
                "sequence": int(shot.sequence or 0),
                "dialogue": dialogue_display_text(lines),
                "character_name": lines[0].speaker if lines else "",
                "has_tts": bool(shot.audio_path),
                "native_audio": profile.get("audio_source") == "native",
                "start_ms": cursor,
                "end_ms": cursor + duration_ms,
                "duration_ms": duration_ms,
            }
        )
        cursor += duration_ms
    return infos, spans, cursor


async def _touch(db: Session, project: Project) -> None:
    bump_av_config_version(db, project)
    await cancel_scopes({f"project:{project.id}"}, "音频轨道配置已修改")
    await ws_manager.send_to_project(project.id, {"type": "av_config_updated", "part": "audio"})


def _resolve_new_source(db: Session, project_id: str, source_path: str | None) -> tuple[str, int]:
    """校验并解析轨道引用的素材路径；返回 (绝对路径, 时长毫秒)。"""

    if not source_path:
        return "", 0
    resolved = existing_file(
        source_path,
        minimum_size=1,
        allowed_roots=(settings.OUTPUT_DIR, settings.ASSETS_DIR, settings.DATA_DIR),
    )
    if resolved is None:
        raise HTTPException(status_code=400, detail="音频素材不存在或不在允许的媒体目录内")
    return str(resolved), 0


@router.get("/{project_id}/tracks")
async def list_audio_tracks(project_id: str, db: Session = Depends(get_db)):
    _, project_id = _validate_ids(project_id=project_id)
    project = _project_or_404(db, project_id)
    shot_infos, spans, total_ms = _shot_infos(db, project_id)
    tracks = (
        db.query(AudioTrack)
        .filter(AudioTrack.project_id == project_id)
        .order_by(AudioTrack.order_index, AudioTrack.created_at)
        .all()
    )
    return {
        "project_id": project_id,
        "av_config_version": int(project.av_config_version or 0),
        "total_duration_ms": total_ms,
        "shots": shot_infos,
        "max_tracks": int(settings.MAX_AUDIO_TRACKS),
        "tracks": [_serialize_track(track, spans.get(track.shot_id) if track.kind == "dialogue" else None) for track in tracks],
    }


@router.post("/{project_id}/upload")
async def upload_audio_asset(project_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """上传音频素材到项目专属目录（配额内），返回可被轨道引用的绝对路径。"""

    _, project_id = _validate_ids(project_id=project_id)
    _project_or_404(db, project_id)
    try:
        _, extension = safe_filename(file.filename, allowed_extensions=_AUDIO_EXTENSIONS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"请上传支持的音频格式（{'/'.join(sorted(_AUDIO_EXTENSIONS))}）") from exc
    media_dir = safe_path(settings.OUTPUT_DIR / "projects", project_id, "audio_tracks", create_parent=True)
    target = media_dir / f"asset-{uuid.uuid4().hex}{extension}"
    try:
        storage_service.ensure_project_capacity(project_id, settings.MAX_AUDIO_UPLOAD_BYTES)
        size = await save_upload_stream(file, target, settings.MAX_AUDIO_UPLOAD_BYTES)
    except UploadLimitExceeded as exc:
        raise HTTPException(status_code=413, detail="上传音频超过大小限制") from exc
    except StorageQuotaExceeded as exc:
        raise HTTPException(status_code=413, detail="项目媒体存储空间不足") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="保存音频失败") from exc
    if size <= 1024:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上传的音频文件为空或过小")
    duration_ms = await ffmpeg_service.probe_duration_ms(target)
    if duration_ms <= 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="无法识别该文件的音频流，请确认是有效的音频文件")
    return {
        "source_path": str(target),
        "source_url": storage_service.get_relative_url(str(target)),
        "duration_ms": duration_ms,
        "size_bytes": size,
    }


@router.post("/{project_id}/tracks")
async def create_audio_track(project_id: str, data: AudioTrackCreate, db: Session = Depends(get_db)):
    _, project_id = _validate_ids(project_id=project_id)
    project = _project_or_404(db, project_id)
    existing = db.query(AudioTrack).filter(AudioTrack.project_id == project_id).count()
    if existing >= int(settings.MAX_AUDIO_TRACKS):
        raise HTTPException(status_code=400, detail=f"音频轨道数量已达上限（{settings.MAX_AUDIO_TRACKS}）")

    source_path = ""
    duration_ms = 0
    shot_id = ""
    if data.kind == "dialogue":
        if not data.shot_id:
            raise HTTPException(status_code=400, detail="对白轨道必须绑定镜头")
        shot = db.query(Shot).filter(Shot.id == data.shot_id, Shot.project_id == project_id).first()
        if not shot:
            raise HTTPException(status_code=404, detail="绑定的镜头不存在")
        shot_id = shot.id
    elif data.kind in {"music", "ambient", "sfx"}:
        if not data.source_path:
            raise HTTPException(status_code=400, detail="请先上传音频素材或选择已有素材")
        source_path, duration_ms = _resolve_new_source(db, project_id, data.source_path)
        if duration_ms <= 0:
            duration_ms = await ffmpeg_service.probe_duration_ms(source_path)
        if duration_ms <= 0:
            raise HTTPException(status_code=400, detail="无法识别素材时长，请确认是有效的音频文件")
    else:
        raise HTTPException(status_code=400, detail="未知的轨道类型")

    order_index = existing
    track = AudioTrack(
        id=uuid.uuid4().hex,
        project_id=project_id,
        kind=data.kind,
        name=(data.name or "").strip() or _default_track_name(data.kind, order_index),
        source_path=source_path,
        source_duration_ms=duration_ms,
        shot_id=shot_id,
        start_ms=data.start_ms,
        volume=data.volume,
        pan=data.pan,
        fade_in_ms=data.fade_in_ms,
        fade_out_ms=data.fade_out_ms,
        delay_ms=data.delay_ms,
        trim_start_ms=data.trim_start_ms,
        trim_end_ms=data.trim_end_ms,
        loop=data.loop,
        muted=data.muted,
        duck_amount_db=data.duck_amount_db,
        duck_attack_ms=data.duck_attack_ms,
        duck_release_ms=data.duck_release_ms,
        order_index=order_index,
    )
    db.add(track)
    db.commit()
    await _touch(db, project)
    _, spans, _ = _shot_infos(db, project_id)
    return _serialize_track(track, spans.get(shot_id) if shot_id else None)


def _default_track_name(kind: str, index: int) -> str:
    labels = {"dialogue": "对白", "music": "背景音乐", "ambient": "环境音", "sfx": "音效"}
    return f"{labels.get(kind, '音频')} {index + 1}"


@router.put("/track/{track_id}")
async def update_audio_track(track_id: str, data: AudioTrackUpdate, db: Session = Depends(get_db)):
    track_id, _ = _validate_ids(track_id=track_id)
    project = _project_or_404(db, data.project_id)
    track = _track_or_404(db, track_id, data.project_id)
    changes = data.model_dump(exclude_unset=True)
    changes.pop("project_id", None)

    if "source_path" in changes and changes["source_path"]:
        resolved, _ = _resolve_new_source(db, data.project_id, changes["source_path"])
        changes["source_path"] = resolved
        duration_ms = await ffmpeg_service.probe_duration_ms(resolved)
        if duration_ms <= 0:
            raise HTTPException(status_code=400, detail="无法识别素材时长，请确认是有效的音频文件")
        track.source_duration_ms = duration_ms
    if "shot_id" in changes and changes["shot_id"]:
        shot = db.query(Shot).filter(Shot.id == changes["shot_id"], Shot.project_id == data.project_id).first()
        if not shot:
            raise HTTPException(status_code=404, detail="绑定的镜头不存在")
    if "kind" in changes and changes["kind"]:
        kind = changes["kind"]
        if kind == "dialogue" and not (changes.get("shot_id") or track.shot_id):
            raise HTTPException(status_code=400, detail="对白轨道必须绑定镜头")
        if kind != "dialogue" and not (changes.get("source_path") or track.source_path):
            raise HTTPException(status_code=400, detail="该轨道类型需要音频素材")

    for field, value in changes.items():
        setattr(track, field, value)
    db.commit()
    await _touch(db, project)
    _, spans, _ = _shot_infos(db, data.project_id)
    return _serialize_track(track, spans.get(track.shot_id) if track.kind == "dialogue" else None)


@router.delete("/track/{track_id}")
async def delete_audio_track(track_id: str, project_id: str, db: Session = Depends(get_db)):
    track_id, project_id = _validate_ids(track_id=track_id, project_id=project_id)
    project = _project_or_404(db, project_id)
    track = _track_or_404(db, track_id, project_id)
    db.delete(track)
    db.commit()
    await _touch(db, project)
    return {"ok": True, "deleted": track_id}


@router.post("/{project_id}/analyze")
async def analyze_audio_setup(project_id: str, db: Session = Depends(get_db)):
    """结构化检查（冲突 / 越界 / 无源）+ 素材级响度削波快检。"""

    _, project_id = _validate_ids(project_id=project_id)
    _project_or_404(db, project_id)
    av_config = collect_render_config(db, project_id)
    shot_infos, spans, total_ms = _shot_infos(db, project_id)
    warnings: list[dict] = []

    def add(level: str, code: str, message: str, track_id: str = "") -> None:
        warnings.append({"level": level, "code": code, "message": message, "track_id": track_id})

    intervals: dict[str, list[tuple[int, int, str]]] = {}
    for track in av_config.audio_tracks:
        track_id = track["id"]
        label = track.get("name") or track_id
        if track["muted"]:
            add("warning", "muted", f"轨道「{label}」处于静音状态，不会参与混音", track_id)
        source = track.get("resolved_source_path") or ""
        if track["kind"] == "dialogue":
            if not track.get("shot_id"):
                add("error", "dialogue_unbound", f"对白轨「{label}」未绑定镜头", track_id)
            elif not source:
                shot_native = next((info["native_audio"] for info in shot_infos if info["id"] == track["shot_id"]), False)
                if shot_native:
                    add("warning", "dialogue_native", f"对白轨「{label}」绑定的镜头使用原生音轨，对白无法单独调整", track_id)
                else:
                    add("warning", "dialogue_no_tts", f"对白轨「{label}」绑定的镜头尚未生成配音", track_id)
        else:
            if not source:
                add("error", "source_missing", f"轨道「{label}」的素材缺失，将被剔除出混音", track_id)
        # 有效播放区间（用于重叠与越界判断）。
        span = track.get("shot_span")
        start = span[0] if span else track["start_ms"]
        delay = track["delay_ms"]
        length = track["source_duration_ms"] - track["trim_start_ms"] - track["trim_end_ms"]
        if track["loop"]:
            length = max(total_ms - start, 0)
        end = start + delay + max(length, 0)
        intervals.setdefault(track["kind"], []).append((start + delay, end, track_id))
        if total_ms > 0 and start + delay >= total_ms:
            add("error", "out_of_range", f"轨道「{label}」的起点已超出全片时长", track_id)
        elif total_ms > 0 and end > total_ms + 500 and not track["loop"]:
            add("warning", "tail_clipped", f"轨道「{label}」的结尾超出全片，超出部分会被截断", track_id)
        if track["volume"] > 1.5:
            add("warning", "hot_gain", f"轨道「{label}」音量较高（{track['volume']:.2f}），可能引发削波", track_id)
        if track["kind"] != "dialogue" and not track["muted"] and source:
            stats = await ffmpeg_service.detect_volume(Path(source))
            max_db = stats.get("max_volume_db")
            if max_db is not None and max_db > -0.5:
                add("warning", "hot_source", f"轨道「{label}」的素材峰值已达 {max_db:.1f} dB，叠加时注意削波", track_id)

    for kind, items in intervals.items():
        ordered = sorted(items)
        for index in range(1, len(ordered)):
            previous, current = ordered[index - 1], ordered[index]
            if current[0] < previous[1]:
                add("warning", "overlap", f"同一{ _KIND_LABELS.get(kind, kind) }轨上有两条轨道时间重叠（{current[0]}ms 处）", current[2])

    for subtitle in av_config.subtitle_tracks:
        label = subtitle.get("name") or subtitle["id"]
        if not subtitle["enabled"]:
            continue
        beyond = [cue for cue in subtitle["cues"] if total_ms > 0 and cue["end_ms"] > total_ms + 500]
        if beyond:
            add("warning", "subtitle_beyond", f"字幕轨「{label}」有 {len(beyond)} 条字幕超出全片时长", subtitle["id"])
        if subtitle["burn_in"] and not subtitle["cues"]:
            add("warning", "subtitle_empty", f"字幕轨「{label}」没有字幕条目", subtitle["id"])

    if total_ms > 0 and not any(
        track["kind"] != "dialogue" and not track["muted"] for track in av_config.audio_tracks
    ):
        add("info", "no_bed", "尚未配置背景音乐 / 环境音 / 音效轨，成片将只包含对白与环境底噪")

    return {"project_id": project_id, "total_duration_ms": total_ms, "warnings": warnings}


_KIND_LABELS = {"dialogue": "对白", "music": "背景音乐", "ambient": "环境音", "sfx": "音效"}


@router.post("/{project_id}/preview")
async def start_mix_preview(project_id: str, data: AudioPreviewRequest, db: Session = Depends(get_db)):
    """启动混音预览（后台任务）：与成片渲染共用同一条混音 filter。"""

    _, project_id = _validate_ids(project_id=project_id)
    _project_or_404(db, project_id)
    scope = data.scope if data.scope in {"full", "shot"} else "full"
    if scope == "shot" and not data.shot_id:
        raise HTTPException(status_code=400, detail="单镜头预览必须提供 shot_id")
    task_key = f"project:{project_id}:audio_preview"
    claim = claim_or_block(
        task_key,
        f"project:{project_id}",
        current_step="mixing",
        message="已排队，准备生成混音预览",
    )
    if not claim.claimed:
        return {"status": "mixing", "project_id": project_id, "deduplicated": True}
    start_task(task_key, _preview_task(project_id, scope, data.shot_id))
    _preview_status[project_id] = {"status": "mixing", "scope": scope, "shot_id": data.shot_id, "progress": 0}
    return {"status": "mixing", "project_id": project_id, "scope": scope, "shot_id": data.shot_id, **budget_notice(claim)}


@router.get("/{project_id}/preview/status")
async def mix_preview_status(project_id: str):
    _, project_id = _validate_ids(project_id=project_id)
    task_key = f"project:{project_id}:audio_preview"
    durable = task_snapshot(task_key)
    memory = _preview_status.get(project_id)
    if memory and memory.get("status") not in {"completed", "error"}:
        if durable and durable.get("status") in {"queued", "running"}:
            return {"project_id": project_id, **memory, "progress": durable.get("progress", memory.get("progress", 0))}
    if durable and durable.get("status") in {"queued", "running"}:
        return {"project_id": project_id, **(memory or {}), "status": "mixing", "progress": durable.get("progress", 0)}
    return {"project_id": project_id, **(memory or {"status": "idle", "progress": 0})}


def _preview_shots_payload(db: Session, project_id: str) -> list[dict]:
    shots = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
    payload: list[dict] = []
    for shot in shots:
        try:
            profile = json.loads(shot.continuity_profile) if shot.continuity_profile else {}
        except (TypeError, ValueError):
            profile = {}
        payload.append(
            {
                "shot_id": shot.id,
                "duration": float(shot.duration or 3.0),
                "audio_path": shot.audio_path or "",
                "video_path": shot.video_path or "",
                "continuity_profile": profile if isinstance(profile, dict) else {},
            }
        )
    return payload


async def _preview_task(project_id: str, scope: str, shot_id: str) -> None:
    task_key = f"project:{project_id}:audio_preview"
    try:
        db = SessionLocal()
        try:
            av_config = collect_render_config(db, project_id)
            shot_dicts = _preview_shots_payload(db, project_id)
        finally:
            db.close()
        total_ms = av_config.total_duration_ms
        window_s: tuple[float, float] | None = None
        display_scope = scope
        if scope == "shot":
            span = av_config.shot_spans.get(shot_id)
            if span is None:
                raise ValueError("预览的镜头不存在")
            window_s = (span[0] / 1000, span[1] / 1000)
            display_scope = f"shot:{shot_id}"

        update_job_progress(task_key, 30, current_step="mixing", message="正在按渲染同源规则合成音频")
        project_dir = safe_path(settings.OUTPUT_DIR / "projects", project_id, "previews", create_parent=True)
        output = project_dir / f"preview-{display_scope.replace(':', '-')}.m4a"
        result = await ffmpeg_service.render_mix_preview(
            shots=shot_dicts,
            av_tracks=av_config.audio_tracks,
            total_duration_s=total_ms / 1000,
            output_path=output,
            window_s=window_s,
        )
        payload = {
            "status": "completed",
            "scope": scope,
            "shot_id": shot_id,
            "progress": 100,
            "audio_url": storage_service.get_relative_url(result["path"]),
            "audio_path": result["path"],
            "warnings": result.get("warnings") or [],
        }
        _preview_status[project_id] = payload
        await ws_manager.send_to_project(project_id, {"type": "av_preview_ready", **payload})
    except asyncio.CancelledError:
        _preview_status[project_id] = {"status": "cancelled", "scope": scope, "shot_id": shot_id, "progress": 0}
        raise
    except Exception as exc:
        error_id = log_failure(exc, error_type=ERROR_RENDER, context={"project_id": project_id, "scope": scope})
        _preview_status[project_id] = {
            "status": "error",
            "scope": scope,
            "shot_id": shot_id,
            "progress": 0,
            "message": f"混音预览生成失败（错误编号 {error_id}）",
        }
        await ws_manager.send_to_project(
            project_id,
            error_payload(error_type=ERROR_RENDER, message="混音预览生成失败，请检查轨道素材与参数后重试。", error_id=error_id),
        )
        raise
