"""字幕工作台 API：轨道样式、逐条编辑、SRT/VTT 导入导出与自动生成。

所有修改都会推进 projects.av_config_version 并取消项目作用域的进行中任务
（渲染 / 预览），保证配置变化后不会发布基于旧字幕的成片。
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api import schemas
from api.websocket import ws_manager
from config import settings
from db import get_db
from models import Project, Shot, SubtitleCue, SubtitleTrack
from services.av_config_service import bump_av_config_version
from services.ffmpeg_service import FFmpegService
from services.security import validate_identifier
from services.shot_dialogue import dialogue_plain_text, parse_shot_dialogue
from services.story_timing import load_shot_execution_plan
from services.subtitle_service import (
    SubtitleCueData,
    SubtitleStyle,
    SubtitleValidationError,
    build_ass_document,
    cues_from_shots,
    detect_cue_overlaps,
    parse_cues,
    serialize_cues,
    validate_cues,
)
from services.task_registry import cancel_scopes

router = APIRouter(prefix="/api/subtitle", tags=["subtitle"])
ffmpeg_service = FFmpegService()


class SubtitleTrackCreate(BaseModel):
    name: schemas.ShortKey | None = None
    language: schemas.ShortKey | None = None
    burn_in: bool = True
    font_family: schemas.ShortKey | None = None
    font_size: int = Field(default=54, ge=8, le=200)
    primary_color: schemas.HexColor | None = None
    outline_color: schemas.HexColor | None = None
    outline_width: int = Field(default=3, ge=0, le=20)
    bold: bool = False
    position: schemas.SubtitlePosition = "bottom"
    safe_margin: int = Field(default=54, ge=0, le=400)


class SubtitleTrackUpdate(BaseModel):
    project_id: schemas.Identifier
    name: schemas.ShortKey | None = None
    language: schemas.ShortKey | None = None
    burn_in: bool | None = None
    enabled: bool | None = None
    font_family: schemas.ShortKey | None = None
    font_size: int | None = Field(default=None, ge=8, le=200)
    primary_color: schemas.HexColor | None = None
    outline_color: schemas.HexColor | None = None
    outline_width: int | None = Field(default=None, ge=0, le=20)
    bold: bool | None = None
    position: schemas.SubtitlePosition | None = None
    safe_margin: int | None = Field(default=None, ge=0, le=400)


class SubtitleCueInput(BaseModel):
    start_ms: int = Field(ge=0, le=86_400_000)
    end_ms: int = Field(ge=0, le=86_400_000)
    text: str
    character_name: str = ""


class SubtitleCueReplace(BaseModel):
    project_id: schemas.Identifier
    # 上限取 MAX_SUBTITLE_CUES + 余量：整表校验仍以 validate_cues 为准。
    cues: list[SubtitleCueInput] = Field(default_factory=list, max_length=4000)


class SubtitleImportRequest(BaseModel):
    project_id: schemas.Identifier
    format: schemas.SubtitleFormat
    content: str


class SubtitleGenerateRequest(BaseModel):
    project_id: schemas.Identifier


def _project_or_404(db: Session, project_id: str) -> Project:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


def _track_or_404(db: Session, track_id: str, project_id: str) -> SubtitleTrack:
    track = db.query(SubtitleTrack).filter(SubtitleTrack.id == track_id, SubtitleTrack.project_id == project_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="字幕轨道不存在")
    return track


def _serialize_track(track: SubtitleTrack, cues: list[SubtitleCue]) -> dict:
    return {
        "id": track.id,
        "project_id": track.project_id,
        "name": track.name or "",
        "language": track.language or "zh",
        "burn_in": bool(track.burn_in),
        "enabled": bool(track.enabled),
        "font_family": track.font_family or "sans-serif",
        "font_size": int(track.font_size or 54),
        "primary_color": track.primary_color or "#FFFFFF",
        "outline_color": track.outline_color or "#000000",
        "outline_width": int(track.outline_width or 0),
        "bold": bool(track.bold),
        "position": track.position or "bottom",
        "safe_margin": int(track.safe_margin or 0),
        "cues": [
            {
                "id": cue.id,
                "start_ms": int(cue.start_ms or 0),
                "end_ms": int(cue.end_ms or 0),
                "text": cue.text or "",
                "character_name": cue.character_name or "",
            }
            for cue in cues
        ],
    }


def _tracks_payload(db: Session, project: Project) -> dict:
    tracks = (
        db.query(SubtitleTrack).filter(SubtitleTrack.project_id == project.id).order_by(SubtitleTrack.created_at).all()
    )
    track_ids = [track.id for track in tracks]
    cues: list[SubtitleCue] = []
    if track_ids:
        cues = (
            db.query(SubtitleCue)
            .filter(SubtitleCue.track_id.in_(track_ids))
            .order_by(SubtitleCue.track_id, SubtitleCue.order_index)
            .all()
        )
    cues_by_track: dict[str, list[SubtitleCue]] = {}
    for cue in cues:
        cues_by_track.setdefault(cue.track_id, []).append(cue)
    return {
        "project_id": project.id,
        "av_config_version": int(project.av_config_version or 0),
        "max_tracks": int(settings.MAX_SUBTITLE_TRACKS),
        "tracks": [_serialize_track(track, cues_by_track.get(track.id, [])) for track in tracks],
    }


async def _touch(db: Session, project: Project) -> None:
    """字幕配置变化：推进版本 + 令牌失效 + 通知同项目客户端。"""

    bump_av_config_version(db, project)
    await cancel_scopes({f"project:{project.id}"}, "字幕配置已修改")
    await ws_manager.send_to_project(project.id, {"type": "av_config_updated", "part": "subtitle"})


def _validate_track_id(track_id: str) -> str:
    try:
        return validate_identifier(track_id, "字幕轨道 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{project_id}/tracks")
async def list_subtitle_tracks(project_id: str, db: Session = Depends(get_db)):
    try:
        validate_identifier(project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    project = _project_or_404(db, project_id)
    return _tracks_payload(db, project)


@router.post("/{project_id}/tracks")
async def create_subtitle_track(project_id: str, data: SubtitleTrackCreate, db: Session = Depends(get_db)):
    try:
        validate_identifier(project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    project = _project_or_404(db, project_id)
    existing = db.query(SubtitleTrack).filter(SubtitleTrack.project_id == project_id).count()
    if existing >= int(settings.MAX_SUBTITLE_TRACKS):
        raise HTTPException(status_code=400, detail=f"字幕轨道数量已达上限（{settings.MAX_SUBTITLE_TRACKS}）")
    track = SubtitleTrack(
        id=uuid.uuid4().hex,
        project_id=project_id,
        name=(data.name or "").strip() or f"字幕轨 {existing + 1}",
        language=(data.language or "zh").strip() or "zh",
        burn_in=data.burn_in,
        enabled=True,
        font_family=(data.font_family or "sans-serif").strip() or "sans-serif",
        font_size=data.font_size,
        primary_color=data.primary_color or "#FFFFFF",
        outline_color=data.outline_color or "#000000",
        outline_width=data.outline_width,
        bold=data.bold,
        position=data.position,
        safe_margin=data.safe_margin,
    )
    db.add(track)
    db.commit()
    await _touch(db, project)
    return _serialize_track(track, [])


@router.put("/track/{track_id}")
async def update_subtitle_track(track_id: str, data: SubtitleTrackUpdate, db: Session = Depends(get_db)):
    _validate_track_id(track_id)
    project = _project_or_404(db, data.project_id)
    track = _track_or_404(db, track_id, data.project_id)
    changes = data.model_dump(exclude_unset=True)
    changes.pop("project_id", None)
    for field, value in changes.items():
        setattr(track, field, value)
    db.commit()
    await _touch(db, project)
    cues = db.query(SubtitleCue).filter(SubtitleCue.track_id == track.id).order_by(SubtitleCue.order_index).all()
    return _serialize_track(track, cues)


@router.delete("/track/{track_id}")
async def delete_subtitle_track(track_id: str, project_id: str, db: Session = Depends(get_db)):
    _validate_track_id(track_id)
    try:
        validate_identifier(project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    project = _project_or_404(db, project_id)
    track = _track_or_404(db, track_id, project_id)
    db.query(SubtitleCue).filter(SubtitleCue.track_id == track.id).delete()
    db.delete(track)
    db.commit()
    await _touch(db, project)
    return {"ok": True, "deleted": track_id}


def _replace_cues(db: Session, track: SubtitleTrack, cues: list[SubtitleCueData]) -> None:
    try:
        validated = validate_cues(cues)
    except SubtitleValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.query(SubtitleCue).filter(SubtitleCue.track_id == track.id).delete()
    for cue in validated:
        db.add(
            SubtitleCue(
                id=uuid.uuid4().hex,
                track_id=track.id,
                project_id=track.project_id,
                order_index=cue.order_index,
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                text=cue.text,
                character_name=cue.character_name,
            )
        )


@router.put("/track/{track_id}/cues")
async def replace_subtitle_cues(track_id: str, data: SubtitleCueReplace, db: Session = Depends(get_db)):
    _validate_track_id(track_id)
    project = _project_or_404(db, data.project_id)
    track = _track_or_404(db, track_id, data.project_id)
    _replace_cues(
        db,
        track,
        [
            SubtitleCueData(
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                text=cue.text,
                character_name=cue.character_name,
            )
            for cue in data.cues
        ],
    )
    db.commit()
    await _touch(db, project)
    cues = db.query(SubtitleCue).filter(SubtitleCue.track_id == track.id).order_by(SubtitleCue.order_index).all()
    return _serialize_track(track, cues)


@router.post("/track/{track_id}/import")
async def import_subtitle(track_id: str, data: SubtitleImportRequest, db: Session = Depends(get_db)):
    _validate_track_id(track_id)
    project = _project_or_404(db, data.project_id)
    track = _track_or_404(db, track_id, data.project_id)
    try:
        parsed = parse_cues(data.content, data.format)
    except SubtitleValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not parsed:
        raise HTTPException(status_code=400, detail="导入内容中未找到有效字幕")
    _replace_cues(db, track, parsed)
    db.commit()
    await _touch(db, project)
    cues = db.query(SubtitleCue).filter(SubtitleCue.track_id == track.id).order_by(SubtitleCue.order_index).all()
    return _serialize_track(track, cues)


@router.get("/track/{track_id}/export")
async def export_subtitle(track_id: str, project_id: str, format: str = "srt", db: Session = Depends(get_db)):
    _validate_track_id(track_id)
    try:
        validate_identifier(project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _project_or_404(db, project_id)
    track = _track_or_404(db, track_id, project_id)
    fmt = str(format or "srt").strip().lower()
    if fmt not in {"srt", "vtt"}:
        raise HTTPException(status_code=400, detail="仅支持导出 SRT / VTT")
    cues = db.query(SubtitleCue).filter(SubtitleCue.track_id == track.id).order_by(SubtitleCue.order_index).all()
    payload = serialize_cues(
        [
            SubtitleCueData(
                start_ms=int(cue.start_ms or 0),
                end_ms=int(cue.end_ms or 0),
                text=cue.text or "",
                character_name=cue.character_name or "",
            )
            for cue in cues
        ],
        fmt,
    )
    media_type = "application/x-subrip" if fmt == "srt" else "text/vtt"
    return PlainTextResponse(
        payload,
        media_type=f"{media_type}; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="subtitle-{track_id}.{fmt}"'},
    )


@router.post("/track/{track_id}/generate")
async def generate_subtitle_from_shots(track_id: str, data: SubtitleGenerateRequest, db: Session = Depends(get_db)):
    """从镜头对白生成字幕：时长优先取 TTS 配音实际时长，且不越过镜头边界。"""

    from services.subtitle_service import DialogueLineInput, ShotDialogueInput

    _validate_track_id(track_id)
    project = _project_or_404(db, data.project_id)
    track = _track_or_404(db, track_id, data.project_id)
    shots = db.query(Shot).filter(Shot.project_id == data.project_id).order_by(Shot.sequence).all()
    if not shots:
        raise HTTPException(status_code=400, detail="项目还没有镜头，无法生成字幕")
    cursor_ms = 0
    inputs: list[ShotDialogueInput] = []
    for shot in shots:
        duration_ms = int(round(max(0.0, float(shot.duration or 0.0)) * 1000))
        # 统一执行计划：剪辑窗口与实测对白时间轴与视频生成/后期合同一份；
        # 没有计划的旧镜头回落 duration 字段（行为不变）。
        try:
            profile = json.loads(shot.continuity_profile or "{}")
        except (TypeError, ValueError):
            profile = {}
        execution_plan = load_shot_execution_plan({"continuity_profile": profile})
        if execution_plan is not None and execution_plan.effective_duration_ms > 0:
            duration_ms = int(execution_plan.effective_duration_ms)
        tts_duration_ms = 0
        if shot.audio_path:
            tts_duration_ms = await ffmpeg_service.probe_duration_ms(shot.audio_path)
        try:
            speakers = json.loads(shot.characters_in_scene) if shot.characters_in_scene else []
        except (TypeError, ValueError):
            speakers = []
        speaker = speakers[0] if speakers and isinstance(speakers[0], str) else ""
        # 结构化对白逐句出字幕：说话人 + 实测时间轴；旧版纯文本自动迁移为
        # 单条（说话人为场内第一个角色，与旧配音行为一致）。
        lines = parse_shot_dialogue(
            shot.dialogue,
            fallback_speaker=speaker,
            default_emotion=shot.emotion or "neutral",
            warn_key=f"subtitle shot {shot.id}",
        )
        inputs.append(
            ShotDialogueInput(
                shot_id=shot.id,
                sequence=int(shot.sequence or 0),
                start_ms=cursor_ms,
                duration_ms=duration_ms,
                dialogue=dialogue_plain_text(lines),
                character_name=lines[0].speaker if lines else speaker,
                tts_duration_ms=tts_duration_ms,
                lines=[
                    DialogueLineInput(
                        speaker=line.speaker,
                        line=line.line,
                        start_ms=line.start_ms,
                        end_ms=line.end_ms,
                    )
                    for line in lines
                ],
                execution_plan=execution_plan,
            )
        )
        cursor_ms += duration_ms
    generated = cues_from_shots(inputs)
    if not generated:
        raise HTTPException(status_code=400, detail="镜头对白为空，未能生成任何字幕")
    _replace_cues(db, track, generated)
    db.commit()
    await _touch(db, project)
    cues = db.query(SubtitleCue).filter(SubtitleCue.track_id == track.id).order_by(SubtitleCue.order_index).all()
    result = _serialize_track(track, cues)
    result["overlaps"] = [
        {"index_a": warn.index_a, "index_b": warn.index_b, "overlap_ms": warn.overlap_ms}
        for warn in detect_cue_overlaps(
            [SubtitleCueData(start_ms=c.start_ms, end_ms=c.end_ms, text=c.text) for c in cues]
        )
    ]
    return result


@router.get("/track/{track_id}/preview-ass")
async def preview_track_ass(
    track_id: str, project_id: str, width: int = 1080, height: int = 1920, db: Session = Depends(get_db)
):
    """渲染出该轨当前样式的 ASS 文档（供前端预览烧录效果描述）。"""

    _validate_track_id(track_id)
    try:
        validate_identifier(project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _project_or_404(db, project_id)
    track = _track_or_404(db, track_id, project_id)
    cues = db.query(SubtitleCue).filter(SubtitleCue.track_id == track.id).order_by(SubtitleCue.order_index).all()
    style = SubtitleStyle(
        font_family=track.font_family or "sans-serif",
        font_size=int(track.font_size or 54),
        primary_color=track.primary_color or "#FFFFFF",
        outline_color=track.outline_color or "#000000",
        outline_width=int(track.outline_width or 0),
        bold=bool(track.bold),
        position=track.position or "bottom",
        safe_margin=int(track.safe_margin or 0),
    )
    width = min(max(int(width), 240), 7680)
    height = min(max(int(height), 240), 4320)
    document = build_ass_document(
        style,
        [
            SubtitleCueData(
                start_ms=int(cue.start_ms or 0),
                end_ms=int(cue.end_ms or 0),
                text=cue.text or "",
                character_name=cue.character_name or "",
            )
            for cue in cues
        ],
        width,
        height,
    )
    return PlainTextResponse(document, media_type="text/plain; charset=utf-8")
