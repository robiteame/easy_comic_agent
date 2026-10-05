import asyncio
import json
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from api import schemas
from api.claim_guard import budget_notice, claim_or_block
from api.websocket import ws_manager
from config import settings
from db import SessionLocal, get_db
from models import Project
from models import Shot as ShotModel
from services.av_config_service import build_av_manifest, collect_render_config
from services.error_reporter import ERROR_RENDER, error_payload, log_failure
from services.ffmpeg_service import FFmpegService
from services.post_production_plan import CAMERA_MOVEMENT_PROMPTS, SUPPORTED_TRANSITIONS, build_post_production_plan
from services.providers.endpoint import get_endpoint
from services.providers.registry import UnknownProtocolError, get_adapter
from services.security import existing_file, validate_identifier
from services.shot_dialogue import dialogue_lines_payload, parse_shot_dialogue
from services.story_timing import (
    StoryTimingError,
    StoryTimingPlan,
    estimate_action_beats,
    provider_duration_capability,
)
from services.task_registry import snapshot as task_snapshot
from services.task_registry import start as start_task
from services.task_registry import update_progress as update_job_progress

router = APIRouter(prefix="/api/render", tags=["render"])
ffmpeg_service = FFmpegService()

_render_tasks: set[asyncio.Task] = set()
_render_status: dict[str, dict] = {}
_render_locks: dict[str, asyncio.Lock] = {}


class RenderRequest(BaseModel):
    # project_id 沿用 400 的既有约定（下面显式 validate_identifier），
    # 比例与分辨率在这里收敛到项目真实支持的档位。
    project_id: str
    output_format: schemas.OutputFormat = "9:16"
    resolution: schemas.Resolution = "1080p"


@router.post("")
async def render_video(data: RenderRequest, db=Depends(get_db)):
    try:
        validate_identifier(data.project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not db.query(Project).filter(Project.id == data.project_id).first():
        raise HTTPException(status_code=404, detail="项目不存在")
    task_key = f"project:{data.project_id}:render"
    claim = claim_or_block(
        task_key,
        f"project:{data.project_id}",
        current_step="rendering",
        message="已排队，准备导出成片",
    )
    if not claim.claimed:
        return {"status": "rendering", "project_id": data.project_id, "deduplicated": True}
    task = start_task(task_key, _render_task(data.project_id, data.output_format, data.resolution))
    _render_tasks.add(task)
    task.add_done_callback(_render_tasks.discard)
    _render_status[data.project_id] = {"status": "rendering", "progress": 0}
    return {"status": "rendering", "project_id": data.project_id, **budget_notice(claim)}


@router.get("/capabilities")
async def get_render_capabilities():
    """返回当前 Provider + FFmpeg 对镜头参数的真实支持范围。"""

    ffmpeg_caps = await ffmpeg_service.capabilities()
    try:
        endpoint = get_endpoint("video")
        adapter_cls = get_adapter("video", endpoint.protocol)
        provider_caps = getattr(adapter_cls, "effective_capabilities", lambda _model="": adapter_cls.capabilities)(
            endpoint.model or ""
        )
    except (UnknownProtocolError, RuntimeError, ValueError):
        endpoint = None
        provider_caps = None
    supported_movements = list(getattr(provider_caps, "supported_camera_movements", ()) or ())
    movement_prompt = bool(getattr(provider_caps, "camera_movement_prompt", False)) if provider_caps else False
    camera_movements = [
        {
            "value": value,
            "supported": value in supported_movements and movement_prompt,
            "reason": ""
            if value in supported_movements and movement_prompt
            else "provider_camera_movement_unsupported",
            "prompt_strategy": prompt,
        }
        for value, prompt in CAMERA_MOVEMENT_PROMPTS.items()
    ]
    transition_items = []
    for value in SUPPORTED_TRANSITIONS:
        supported = value in ffmpeg_caps.get("supported_transitions", [])
        transition_items.append(
            {
                "value": value,
                "supported": supported,
                "reason": "" if supported else "ffmpeg_transition_filter_unsupported",
                "fallback": "cut" if not supported else "",
            }
        )
    return {
        "provider": {
            "protocol": str(getattr(endpoint, "protocol", "") or ""),
            "model": str(getattr(endpoint, "model", "") or ""),
            "camera_movement_prompt": movement_prompt,
            "timed_dialogue": bool(getattr(provider_caps, "timed_dialogue", False)) if provider_caps else False,
            "fixed_duration": getattr(provider_caps, "fixed_duration", None) if provider_caps else None,
            "min_duration": getattr(provider_caps, "min_duration", None) if provider_caps else None,
            "max_duration": getattr(provider_caps, "max_duration", None) if provider_caps else None,
            "duration_step": getattr(provider_caps, "duration_step", None) if provider_caps else None,
        },
        "ffmpeg": ffmpeg_caps,
        "camera_movements": camera_movements,
        "transitions": transition_items,
        "fallback_policy": "unsupported_transition_falls_back_to_cut_with_reason",
    }


@router.get("/{project_id}/timeline")
async def get_render_timeline(project_id: str):
    """读取渲染前生成的可审查 PostProductionPlan JSON。"""

    try:
        validate_identifier(project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    timeline_path = settings.OUTPUT_DIR / "projects" / project_id / "output" / "timeline.json"
    if not timeline_path.exists():
        raise HTTPException(status_code=404, detail="尚未生成后期时间线")
    try:
        return json.loads(timeline_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="后期时间线 JSON 无法读取") from exc


@router.get("/{project_id}/status")
async def get_render_status(project_id: str):
    try:
        validate_identifier(project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    task_key = f"project:{project_id}:render"
    durable = task_snapshot(task_key)
    memory = _render_status.get(project_id)
    db = SessionLocal()
    try:
        project_status = db.query(Project.status).filter(Project.id == project_id).scalar()
    finally:
        db.close()
    final_path = settings.OUTPUT_DIR / "projects" / project_id / "output" / "final.mp4"
    valid_final = existing_file(final_path, minimum_size=1024, allowed_roots=(settings.OUTPUT_DIR,))
    if durable is None:
        if project_status == "completed" and valid_final:
            return {"project_id": project_id, "status": "completed", "progress": 100, "video_path": str(final_path)}
        if memory and memory.get("status") == "completed":
            memory = None
        return {"project_id": project_id, **(memory or {"status": "idle", "progress": 0})}

    status = durable["status"]
    if status in {"queued", "running"}:
        # The worker records its richer message in memory, while SQLite remains
        # the authority for ownership and survives a process restart.
        if memory and memory.get("status") in {"completed", "error"}:
            payload = memory
        else:
            payload = {**(memory or {}), "status": "rendering", "progress": durable["progress"]}
    elif status == "completed":
        if project_status != "completed":
            return {"project_id": project_id, "status": "idle", "progress": 0}
        payload = {"status": "completed", "progress": 100}
        if valid_final:
            payload["video_path"] = str(final_path)
        timeline_path = settings.OUTPUT_DIR / "projects" / project_id / "output" / "timeline.json"
        if timeline_path.exists():
            payload["timeline_path"] = str(timeline_path)
        if memory and memory.get("status") == "completed":
            if memory.get("video_path"):
                payload["video_path"] = memory["video_path"]
    elif status in {"failed", "interrupted"}:
        payload = {"status": "error", "progress": durable["progress"], "message": durable["error"]}
    else:
        payload = {"status": "cancelled", "progress": durable["progress"], "message": durable["error"]}
    return {"project_id": project_id, **payload}


async def _render_task(project_id: str, output_format: str, resolution: str):
    lock = _render_locks.setdefault(project_id, asyncio.Lock())
    if lock.locked():
        raise RuntimeError("项目渲染任务已在运行")
    await lock.acquire()
    staged_video: Path | None = None
    try:
        await _progress(project_id, "rendering", 0, "开始导出成片")
        db = SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = "rendering"
                db.commit()

            db_shots = db.query(ShotModel).filter(ShotModel.project_id == project_id).order_by(ShotModel.sequence).all()
            if not db_shots:
                raise ValueError("没有可导出的镜头")
            if any(not s.confirmed for s in db_shots):
                raise ValueError("仍有镜头故事板未人工审核通过，不能导出成片")
            if any(not (s.storyboard_path or s.image_path) for s in db_shots):
                raise ValueError("仍有镜头未生成定稿故事板参考图，不能导出成片")
            if any(not s.video_path for s in db_shots):
                raise ValueError("仍有镜头未逐一生成视频，请先完成每个镜头的视频生成")
            media_roots = (settings.OUTPUT_DIR, settings.ASSETS_DIR, settings.DATA_DIR)
            missing_media = [
                s.id
                for s in db_shots
                if existing_file(s.video_path, minimum_size=4096, allowed_roots=media_roots) is None
            ]
            if missing_media:
                raise ValueError(f"镜头视频文件不存在或无效: {', '.join(missing_media)}")

            shots = [
                {
                    "shot_id": s.id,
                    "shot_type": s.shot_type,
                    "scene_description": s.scene_description,
                    "characters_in_scene": json.loads(s.characters_in_scene) if s.characters_in_scene else [],
                    "character_action": s.character_action,
                    "dialogue": _render_dialogue_payload(s),
                    "camera_angle": s.camera_angle,
                    "camera_movement": s.camera_movement,
                    "emotion": s.emotion,
                    "duration": s.duration,
                    "estimated_speech_ms": s.estimated_speech_ms or 0,
                    "transition": s.transition,
                    "image_path": s.image_path or s.storyboard_path,
                    "video_path": s.video_path,
                    "audio_path": s.audio_path,
                    "status": s.status,
                    "version": s.version,
                    "scene_group_id": s.scene_group_id or s.scene_asset_id or "",
                    "continuity_profile": _json_dict(s.continuity_profile),
                }
                for s in db_shots
            ]
            for item in shots:
                profile_timing = (item.get("continuity_profile") or {}).get("timing") or {}
                item["timing"] = {
                    **profile_timing,
                    **{
                        key: value
                        for key, value in (item.get("continuity_profile") or {}).items()
                        if key in {"audio_source", "audio_mode"}
                    },
                }
            manifest = {
                s.id: (s.version or 1, bool(s.confirmed), s.video_path or "", s.audio_path or "") for s in db_shots
            }
            # 字幕/音频工作台：渲染采用当时的全部轨道与字幕配置；发布前重读比对，
            # 期间任何修改（av_config_version 亦会推进）都使本批成片作废。
            av_render_config = collect_render_config(db, project_id)
            av_manifest = build_av_manifest(av_render_config)
            media_paths = {
                str(path)
                for item in shots
                for path in (item.get("video_path"), item.get("audio_path"))
                if str(path or "")
            }
            media_paths.update(
                str(track.get("resolved_source_path") or track.get("source_path") or "")
                for track in av_render_config.audio_tracks
                if str(track.get("resolved_source_path") or track.get("source_path") or "")
            )
            av_config_payload = {
                "audio_tracks": av_render_config.audio_tracks,
                "subtitle_tracks": av_render_config.subtitle_tracks,
                "total_duration_s": av_render_config.total_duration_ms / 1000,
            }
            project_manifest = _project_manifest_tuple(project)
        finally:
            db.close()

        media_durations_ms = await _probe_media_durations(media_paths)
        planned_total_s = sum(float(item.get("duration") or 0) for item in shots)
        target_duration_s = float(project_timing_target(project_id) or planned_total_s)
        duration_capability = provider_duration_capability()
        timing_plan = StoryTimingPlan(
            target_duration_s=target_duration_s,
            provider=duration_capability,
            shot_count=len(shots),
            planned_total_duration_s=planned_total_s,
            action_beats=[beat for item in shots for beat in estimate_action_beats(item.get("character_action"))],
        )
        timing_issues = timing_plan.validate_timeline(
            shots,
            audio_tracks=av_config_payload.get("audio_tracks", []),
            media_durations_ms=media_durations_ms,
            provider=duration_capability,
            require_target=True,
        )
        zero_media_shot_ids = {
            str(item.get("shot_id") or "")
            for item in shots
            if int(media_durations_ms.get(str(item.get("video_path") or ""), 0) or 0) <= 0
        }
        blocking_timing_issues = [
            issue
            for issue in timing_issues
            if issue.code not in {"provider_duration_invalid", "target_duration_mismatch"}
            and not (issue.code == "video_shorter_than_timeline" and set(issue.shot_ids).issubset(zero_media_shot_ids))
        ]
        if blocking_timing_issues:
            raise StoryTimingError(blocking_timing_issues)

        _apply_post_profiles(shots)
        render_capabilities = await get_render_capabilities()
        timeline_plan = build_post_production_plan(
            shots,
            av_config_payload,
            project_id=project_id,
            fps=ffmpeg_service.fps,
            capabilities=render_capabilities,
        )
        # 渲染器必须消费时间线夹定后的字幕与计划总时长，不能回头读取旧边界。
        av_config_payload = {
            **av_config_payload,
            "subtitle_tracks": timeline_plan.subtitle_tracks,
            "total_duration_s": timeline_plan.total_duration_ms / 1000,
        }
        timeline_path = settings.OUTPUT_DIR / "projects" / project_id / "output" / "timeline.json"
        timeline_plan.write_json(timeline_path)

        staged_video = Path(
            await ffmpeg_service.compose_video(
                shots=shots,
                output_format=output_format,
                resolution=resolution,
                project_id=project_id,
                publish=False,
                av_config=av_config_payload,
                plan=timeline_plan.to_dict(),
            )
        )
        final_file = existing_file(staged_video, minimum_size=1024, allowed_roots=media_roots)
        if final_file is None:
            raise RuntimeError("FFmpeg 未生成有效的成片文件")

        video_path = _publish_render(project_id, staged_video, manifest, project_manifest, av_manifest)
        staged_video = None
        _render_status[project_id] = {"status": "completed", "progress": 100, "video_path": video_path}
        await ws_manager.send_to_project(
            project_id,
            {
                "type": "render_complete",
                "video_url": f"/output/projects/{project_id}/output/final.mp4",
                "duration": sum(float(s["duration"] or 0) for s in shots),
            },
        )
    except asyncio.CancelledError as exc:
        _render_status[project_id] = {"status": "cancelled", "progress": 0, "message": str(exc)}
        db = SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project and project.status == "rendering":
                project.status = "assets_ready"
                db.commit()
        finally:
            db.close()
        raise
    except Exception as exc:
        error_id = log_failure(exc, error_type=ERROR_RENDER, context={"project_id": project_id})
        if isinstance(exc, StoryTimingError):
            public_message = f"成片导出已阻止，时间线不一致：{exc}"
        else:
            public_message = "成片导出失败，本次渲染已停止。请检查镜头素材、配音与输出设置后重试。"
        # 渲染状态会经 GET /api/render/{id}/status 回显，只保留可读提示与错误编号。
        _render_status[project_id] = {
            "status": "error",
            "progress": 0,
            "message": f"{public_message}（错误编号 {error_id}）",
        }
        db = SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = "error"
                db.commit()
        finally:
            db.close()
        await ws_manager.send_to_project(
            project_id,
            error_payload(error_type=ERROR_RENDER, message=public_message, error_id=error_id),
        )
        # Do not swallow the error: automatic LangGraph callers must be able to
        # short-circuit instead of reaching END with a false success.
        raise
    finally:
        if staged_video is not None:
            staged_video.unlink(missing_ok=True)
        lock.release()


async def _progress(project_id: str, step: str, progress: int, message: str):
    _render_status[project_id] = {"status": "rendering", "progress": progress, "message": message}
    update_job_progress(f"project:{project_id}:render", progress, current_step=step, message=message)
    await ws_manager.send_to_project(
        project_id, {"type": "progress", "step": step, "progress": progress, "message": message}
    )


async def _probe_media_durations(paths: set[str]) -> dict[str, int]:
    """Probe exact media durations used by the render timing gate."""

    durations: dict[str, int] = {}
    for raw_path in sorted(paths):
        path = existing_file(
            raw_path, minimum_size=1, allowed_roots=(settings.OUTPUT_DIR, settings.ASSETS_DIR, settings.DATA_DIR)
        )
        if path is None:
            continue
        try:
            durations[raw_path] = int(await ffmpeg_service.probe_duration_ms(path))
        except Exception:
            durations[raw_path] = 0
    return durations


def project_timing_target(project_id: str) -> float:
    """Read the explicit project target; legacy rows infer it from current shots."""

    db = SessionLocal()
    try:
        row = db.query(Project).filter(Project.id == project_id).first()
        target = float(getattr(row, "target_duration", 0) or 0) if row else 0.0
        if target > 0:
            return target
        if row is not None:
            plan = _json_dict(getattr(row, "timing_plan", "{}"))
            planned_target = float(plan.get("target_duration_s") or 0)
            if planned_target > 0:
                return planned_target
        return sum(
            float(row[0] or 0) for row in db.query(ShotModel.duration).filter(ShotModel.project_id == project_id).all()
        )
    finally:
        db.close()


def _render_dialogue_payload(shot: ShotModel) -> list[dict]:
    """旧版纯文本迁移时保留场内首个说话人，并记录可追踪来源。"""

    try:
        speakers = json.loads(shot.characters_in_scene or "[]")
    except (TypeError, ValueError):
        speakers = []
    fallback = speakers[0] if speakers and isinstance(speakers[0], str) else ""
    return dialogue_lines_payload(
        parse_shot_dialogue(
            shot.dialogue,
            fallback_speaker=fallback,
            default_emotion=shot.emotion or "neutral",
            warn_key=f"render shot {shot.id}",
        )
    )


def _json_dict(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _project_manifest_tuple(project) -> tuple | None:
    if project is None:
        return None
    return (project.style, project.output_format, project.resolution, int(project.av_config_version or 0))


def _manifest_matches(recorded: tuple | None, current: tuple | None) -> bool:
    """比对项目配置 manifest；三段历史格式（无 av 版本）缺省按 0 处理。"""

    if recorded is None or current is None:
        return recorded is None and current is None
    if len(recorded) == 3:
        recorded = (*recorded, 0)
    if len(current) == 3:
        current = (*current, 0)
    return recorded == current


def _publish_render(
    project_id: str,
    staged_video: Path,
    manifest: dict[str, tuple],
    project_manifest: tuple | None,
    av_manifest: list | None = None,
) -> str:
    """Atomically validate the render inputs and promote its staged output."""

    db = SessionLocal()
    try:
        db.execute(text("BEGIN IMMEDIATE"))
        project = db.query(Project).filter(Project.id == project_id).first()
        current = db.query(ShotModel).filter(ShotModel.project_id == project_id).all()
        current_manifest = {
            shot.id: (shot.version or 1, bool(shot.confirmed), shot.video_path or "", shot.audio_path or "")
            for shot in current
        }
        current_project_manifest = _project_manifest_tuple(project)
        current_av_manifest = (
            build_av_manifest(collect_render_config(db, project_id)) if av_manifest is not None else None
        )
        if (
            not project
            or current_manifest != manifest
            or not _manifest_matches(project_manifest, current_project_manifest)
            or (av_manifest is not None and current_av_manifest != av_manifest)
        ):
            db.rollback()
            raise asyncio.CancelledError("渲染期间镜头或字幕/音频配置已发生变化")
        final_path = settings.OUTPUT_DIR / "projects" / project_id / "output" / "final.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path = final_path.with_name(f".final-{uuid.uuid4().hex}.previous")
        had_previous = final_path.exists()
        if had_previous:
            os.replace(final_path, backup_path)
        try:
            os.replace(staged_video, final_path)
            project.status = "completed"
            db.commit()
        except BaseException:
            db.rollback()
            final_path.unlink(missing_ok=True)
            if had_previous and backup_path.exists():
                os.replace(backup_path, final_path)
            raise
        try:
            backup_path.unlink(missing_ok=True)
        except OSError:
            # A committed render is valid even if best-effort cleanup is
            # delayed; leaving the backup avoids risking the published file.
            pass
        return str(final_path)
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def _apply_post_profiles(shots: list[dict]) -> None:
    for index, shot in enumerate(shots):
        previous_shot = shots[index - 1] if index > 0 else None
        next_shot = shots[index + 1] if index + 1 < len(shots) else None
        profile = shot.get("continuity_profile") or {}
        scene_group = shot.get("scene_group_id") or ""
        previous_group = previous_shot.get("scene_group_id") if previous_shot else ""
        next_group = next_shot.get("scene_group_id") if next_shot else ""
        cross_in = bool(previous_shot and previous_group != scene_group)
        cross_out = bool(next_shot and next_group != scene_group)
        shot["post_profile"] = {
            "scene_group_id": scene_group,
            "transition_in": profile.get("cross_scene_transition") if cross_in else "hard cut",
            "transition_out": profile.get("cross_scene_transition")
            if cross_out
            else profile.get("same_scene_transition", "hard cut or 0.2s fade only"),
            "same_scene_fade_seconds": 0.2,
            "cross_scene_flash_seconds": 0.35,
            "cross_scene_in": cross_in,
            "cross_scene_out": cross_out,
            "lut": profile.get("lut", "project_scene_lut_locked"),
            "saturation": profile.get("saturation", "locked per scene group"),
            "sharpness": profile.get("sharpness", "locked per scene group"),
            "ambient_audio_policy": profile.get("ambient_audio_policy", "continuous room tone"),
        }
