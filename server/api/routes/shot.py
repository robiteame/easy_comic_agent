import asyncio
import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api import schemas
from api.websocket import ws_manager
from config import settings
from db import SessionLocal, get_db
from models import Character, Project, SceneAsset, Shot, ShotVersion
from services.audio_routing import resolve_audio_mode
from services.consistency_service import ConsistencyService
from services.invalidation_service import clear_shot_media_stale, mark_shot_media_stale
from services.error_reporter import (
    ERROR_SHOT_VIDEO,
    ERROR_STORYBOARD,
    error_payload,
    log_failure,
    report_failure,
)
from services.image_service import ImageService
from services.providers.base import Dialogue
from services.providers.endpoint import get_endpoint
from services.shot_version_service import (
    apply_snapshot_to_shot,
    capture_current_snapshot,
    content_hash,
    create_version,
    diff_snapshots,
    list_versions,
    missing_asset_bindings,
    missing_media,
    parse_snapshot,
    version_detail,
)
from services.skill_config_service import (
    apply_agent_config_to_shot,
    clean_tts_text,
    resolve_effective_style,
    resolve_skill_config,
)
from services.style_templates import style_prompt_params
from services.tts_service import TTSService
from services.video_service import SeedanceVideoService
from services.security import existing_file, validate_identifier
from api.claim_guard import budget_notice, claim_or_block
from api.provider_guard import ensure_providers_ready
from services.task_registry import (
    cancel as cancel_task,
    cancel_scopes,
    claim as claim_task,
    finish as finish_task,
    start as start_task,
    update_progress as update_job_progress,
)

router = APIRouter(prefix="/api/shot", tags=["shot"])

image_service = ImageService()
tts_service = TTSService()
seedance_service = SeedanceVideoService()
consistency_service = ConsistencyService()
_regeneration_tasks: set[asyncio.Task] = set()
_shot_video_tasks: set[asyncio.Task] = set()
_project_generation_locks: dict[str, asyncio.Lock] = {}
_shot_generation_locks: dict[str, asyncio.Lock] = {}


class ShotUpdate(BaseModel):
    shot_type: schemas.ShotType | None = None
    scene_description: schemas.ShotText | None = None
    character_action: schemas.ShotText | None = None
    dialogue: schemas.ShotText | None = None
    camera_angle: schemas.CameraAngle | None = None
    camera_movement: schemas.CameraMovement | None = None
    duration: schemas.ShotDuration | None = None
    emotion: schemas.Emotion | None = None
    transition: schemas.Transition | None = None
    visual_notes: schemas.VisualNotes | None = None
    scene_asset_id: schemas.OptionalIdentifier | None = None
    character_asset_ids: schemas.CharacterAssetIdList | None = None
    audio_mode: schemas.AudioModeOverride | None = None  # 镜头级音频路径覆盖："tts" | "native" | "auto"，空串清除


class RegenerateRequest(BaseModel):
    reason: schemas.ReasonText = ""
    prompt: schemas.VisualNotes | None = None
    visual_notes: schemas.VisualNotes | None = None
    new_emotion: schemas.Emotion | None = None
    new_scene: schemas.ShotText | None = None
    new_camera_angle: schemas.CameraAngle | None = None
    shot_type: schemas.ShotType | None = None
    character_action: schemas.ShotText | None = None
    dialogue: schemas.ShotText | None = None
    duration: schemas.ShotDuration | None = None
    force_confirmed: bool = False
    # 关键镜头可一次生成 2 个候选：每个候选各自进入版本历史，人工对比后选用。
    candidates: schemas.CandidateCount = 1
    capability_mode: schemas.PipelineMode = "manual"
    confirm_capability_downgrade: bool = False


class StoryboardGenerateRequest(BaseModel):
    shot_ids: schemas.ShotIdList = Field(default_factory=list)
    capability_mode: schemas.PipelineMode = "manual"
    confirm_capability_downgrade: bool = False


class StoryboardApprovalRequest(BaseModel):
    approved: bool = True


class ShotVideoGenerateRequest(BaseModel):
    force: bool = False
    # 重试 / 续跑时复用仍然有效的配音，避免重复执行已经成功完成的阶段。
    # 前端不发送该字段，默认 False 保持既有行为不变。
    reuse_audio: bool = False
    # 仅供显式选择性队列使用：故事板 + 视频批次可以在同一批次内衔接，
    # 不把“未人工审核”误认为普通视频入口的授权。
    allow_unconfirmed: bool = False
    capability_mode: schemas.PipelineMode = "manual"
    confirm_capability_downgrade: bool = False


class ShotAudioGenerateRequest(BaseModel):
    force: bool = False
    reuse_existing: bool = False


@router.get("/{shot_id}/generation-prompt")
async def get_shot_generation_prompt(shot_id: str, db: Session = Depends(get_db)):
    shot = db.query(Shot).filter(Shot.id == shot_id).first()
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")

    project = db.query(Project).filter(Project.id == shot.project_id).first()
    skill_config = resolve_skill_config(shot.project_id, db)
    characters = _characters(db, shot.project_id)
    scenes = _scenes(db, shot.project_id)
    # 与实际生成路径保持一致：同场景优先上一镜尾帧，无则回退上一镜故事板。
    previous_reference = _previous_reference_for_shot(db, shot, prefer_last_frame=True)
    shot_data = _shot_dict(shot)
    shot_data["storyboard_prompt"] = _storyboard_notes(shot, scenes)
    shot_data.update(
        consistency_service.build_generation_context(
            shot_data,
            characters,
            scenes,
            previous_reference_path=previous_reference,
            for_video=False,
        )
    )
    apply_agent_config_to_shot(shot_data, skill_config)
    prompt, negative_prompt = image_service.build_shot_prompt(
        shot=shot_data,
        characters=characters,
        style_params=_storyboard_style_params(project, skill_config),
    )
    return {
        "shot_id": shot.id,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "scene_reference_images": shot_data.get("scene_reference_images", []),
        "character_reference_images": shot_data.get("character_reference_images", []),
        **resolve_effective_style(project.style if project else "anime", skill_config, "storyboard_agent"),
    }


@router.get("/{project_id}/shots")
async def get_project_shots(project_id: str, db: Session = Depends(get_db)):
    _validate_id_or_400(project_id, "项目 ID")
    if not db.query(Project).filter(Project.id == project_id).first():
        raise HTTPException(status_code=404, detail="Project not found")
    shots = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
    project = db.query(Project).filter(Project.id == project_id).first()
    style_meta = resolve_effective_style((project.style if project else "anime"), resolve_skill_config(project_id, db), "storyboard_agent")
    return [{**_serialize_shot(s), **style_meta} for s in shots]


@router.put("/{shot_id}")
async def update_shot(shot_id: str, data: ShotUpdate, db: Session = Depends(get_db)):
    shot = db.query(Shot).filter(Shot.id == shot_id).first()
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")

    _ensure_shot_unlocked(shot)
    project_id = shot.project_id
    result_id = shot.id
    previous_scene_key = _shot_scene_key(shot)
    changed = data.model_dump(exclude_unset=True)
    if "scene_asset_id" in changed or "character_asset_ids" in changed:
        scene_value = changed.get("scene_asset_id", shot.scene_asset_id) or ""
        character_value = changed.get("character_asset_ids", _json_list(shot.character_asset_ids))
        scene_value, character_value = _validate_asset_bindings(db, shot, scene_value, character_value)
        if "scene_asset_id" in changed:
            changed["scene_asset_id"] = scene_value
        if "character_asset_ids" in changed:
            changed["character_asset_ids"] = character_value
    if changed:
        # 被编辑替换的当前状态先进入版本历史（内容与最新记录一致时自动去重）。
        create_version(db, shot, "manual_edit")
    for key, value in changed.items():
        if key == "character_asset_ids":
            setattr(shot, key, json.dumps(value or [], ensure_ascii=False))
        elif key == "audio_mode":
            profile = _json_dict(shot.continuity_profile)
            mode = str(value or "").strip().lower()
            if mode:
                profile["audio_mode"] = mode
            else:
                profile.pop("audio_mode", None)
            shot.continuity_profile = json.dumps(profile, ensure_ascii=False)
        else:
            setattr(shot, key, value)

    if changed:
        _invalidate_storyboard_outputs(shot)
        _invalidate_downstream_media(db, shot, {previous_scene_key, _shot_scene_key(shot)}, source="manual_edit")
        shot.version = (shot.version or 1) + 1
        _mark_project_output_stale(db, project_id)

    db.commit()
    if changed:
        # The project scope also owns automatic pipelines and renders, while the
        # shot scope owns manual image/video work. Wait for both to unwind so
        # no stale worker can publish after this response.
        await cancel_scopes(
            {f"shot:{shot_id}", f"project:{project_id}"},
            "shot was edited",
        )
    return {"id": result_id, "status": "updated", "needs_render": bool(changed)}


def _capability_kwargs(capability_mode: str, confirm_capability_downgrade: bool) -> dict:
    """默认 manual/未确认时保持旧调用签名兼容；显式策略才携带新参数。"""

    if str(capability_mode or "manual").lower() == "manual" and not confirm_capability_downgrade:
        return {}
    return {
        "capability_mode": str(capability_mode or "manual").lower(),
        "confirm_capability_downgrade": bool(confirm_capability_downgrade),
    }


def _prepare_storyboard_candidate(shot: Shot, db: Session, data: RegenerateRequest) -> tuple[int, str]:
    """登记一版故事板候选：版本快照 + 失效下游 + 应用本次参数。

    返回 (本次 expected_version, 生成理由)。多个候选共用同一段登记逻辑，
    因此第 2 个候选与第 1 个候选一样会进入版本历史，可对比后再选用。
    """
    task_key = _shot_task_key(shot.id, "storyboard")
    create_version(db, shot, "regenerate", task_id=task_key)
    previous_scene_key = _shot_scene_key(shot)
    _invalidate_storyboard_outputs(shot)
    _invalidate_downstream_media(db, shot, {previous_scene_key, _shot_scene_key(shot)}, source="regenerate")
    shot.status = "pending"
    shot.storyboard_status = "queued"
    shot.version = (shot.version or 1) + 1
    if data.new_emotion:
        shot.emotion = data.new_emotion
    if data.new_scene:
        shot.scene_description = data.new_scene
    if data.new_camera_angle:
        shot.camera_angle = data.new_camera_angle
    if data.shot_type:
        shot.shot_type = data.shot_type
    if data.character_action is not None:
        shot.character_action = data.character_action
    if data.dialogue is not None:
        shot.dialogue = data.dialogue
    if data.duration is not None:
        shot.duration = data.duration
    prompt = data.prompt if data.prompt is not None else data.visual_notes
    if prompt is not None:
        shot.visual_notes = prompt
    _mark_project_output_stale(db, shot.project_id)
    db.commit()
    return int(shot.version or 1), str(data.reason or prompt or "")


async def _run_storyboard_candidates(
    shot_id: str, data: RegenerateRequest, candidates: int, first_expected_version: int, first_reason: str
) -> None:
    """顺序生成 N 个故事板候选（同一镜头锁内串行，避免版本互相踩踏）。"""
    expected_version = first_expected_version
    reason = first_reason
    for index in range(candidates):
        if index:
            db = SessionLocal()
            try:
                shot = db.query(Shot).filter(Shot.id == shot_id).first()
                if not shot:
                    return
                expected_version, reason = _prepare_storyboard_candidate(shot, db, data)
            finally:
                db.close()
        update_job_progress(
            _shot_task_key(shot_id, "storyboard"),
            40,
            current_step="regenerate_storyboard",
            message=f"正在生成故事板候选 {index + 1}/{candidates}",
        )
        await _regenerate_single_shot(
            shot_id,
            reason,
            expected_version,
            **_capability_kwargs(data.capability_mode, data.confirm_capability_downgrade),
        )


@router.post("/{shot_id}/regenerate")
async def regenerate_shot(shot_id: str, data: RegenerateRequest, db: Session = Depends(get_db)):
    shot = db.query(Shot).filter(Shot.id == shot_id).first()
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")

    _ensure_shot_unlocked(shot, force=data.force_confirmed)
    task_key = _shot_task_key(shot_id, "storyboard")
    expected_version = (shot.version or 1) + 1
    candidates = int(data.candidates or 1)
    claim = claim_or_block(
        task_key,
        f"shot:{shot_id}",
        version=expected_version,
        current_step="regenerate_storyboard",
        message=f"正在重新生成镜头 {shot.sequence} 的故事板",
    )
    if not claim.claimed:
        return {"id": shot.id, "status": "regenerating", "version": shot.version, "deduplicated": True}

    try:
        expected_version, reason = _prepare_storyboard_candidate(shot, db, data)
        if candidates > 1:
            # 关键镜头：一次提交生成 N 个候选，全部进入版本历史供人工挑选。
            task = start_task(
                task_key,
                _run_storyboard_candidates(shot_id, data, candidates, expected_version, reason),
            )
        else:
            task = start_task(
                task_key,
                _regenerate_single_shot(
                    shot_id,
                    reason,
                    expected_version,
                    **_capability_kwargs(data.capability_mode, data.confirm_capability_downgrade),
                ),
            )
    except BaseException as exc:
        db.rollback()
        finish_task(task_key, "failed", f"storyboard scheduling failed: {exc}")
        raise
    _regeneration_tasks.add(task)
    task.add_done_callback(_regeneration_tasks.discard)
    return {
        "id": shot.id,
        "status": "regenerating" if candidates <= 1 else "generating_candidates",
        "version": shot.version,
        "candidates": candidates,
    }


@router.post("/batch-regenerate")
async def batch_regenerate(shot_ids: schemas.ShotIdList, reason: schemas.ReasonText = "", db: Session = Depends(get_db)):
    shots = db.query(Shot).filter(Shot.id.in_(shot_ids)).all()
    missing = sorted(set(shot_ids) - {shot.id for shot in shots})
    if missing:
        raise HTTPException(status_code=404, detail=f"镜头不存在: {', '.join(missing)}")
    locked = [shot.id for shot in shots if shot.confirmed]
    if locked:
        raise HTTPException(status_code=423, detail=f"已审核镜头禁止重新生成: {', '.join(locked)}")
    claimed_keys: list[str] = []
    for shot in shots:
        task_key = _shot_task_key(shot.id, "storyboard")
        # 预算不足时 claim_or_block 直接抛 409（budget_exceeded），不需要回滚已占用的镜头。
        claim = claim_or_block(task_key, f"shot:{shot.id}", version=(shot.version or 1) + 1)
        if not claim.claimed:
            for claimed_key in claimed_keys:
                finish_task(claimed_key, "cancelled", "batch claim rolled back")
            raise HTTPException(status_code=409, detail=f"镜头已有生成任务: {shot.id}")
        claimed_keys.append(task_key)
    expected_versions: dict[str, int] = {}
    try:
        for shot in shots:
            create_version(db, shot, "regenerate", task_id=_shot_task_key(shot.id, "storyboard"))
            previous_scene_key = _shot_scene_key(shot)
            _invalidate_storyboard_outputs(shot)
            _invalidate_downstream_media(db, shot, {previous_scene_key, _shot_scene_key(shot)}, source="regenerate")
            shot.status = "pending"
            shot.storyboard_status = "queued"
            shot.version = (shot.version or 1) + 1
            expected_versions[shot.id] = shot.version
        for project_id in {shot.project_id for shot in shots}:
            _mark_project_output_stale(db, project_id)
        db.commit()
    except BaseException as exc:
        db.rollback()
        for task_key in claimed_keys:
            finish_task(task_key, "failed", f"batch preparation failed: {exc}")
        raise

    started_keys: set[str] = set()
    try:
        for shot in shots:
            task_key = _shot_task_key(shot.id, "storyboard")
            task = start_task(task_key, _regenerate_single_shot(shot.id, reason, expected_versions[shot.id]))
            started_keys.add(task_key)
            _regeneration_tasks.add(task)
            task.add_done_callback(_regeneration_tasks.discard)
    except BaseException:
        for task_key in started_keys:
            cancel_task(task_key)
        for task_key in set(claimed_keys) - started_keys:
            finish_task(task_key, "failed", "batch scheduling was aborted")
        raise
    return {"updated": len(shots)}


@router.post("/{project_id}/generate-storyboard")
async def generate_storyboard_images(project_id: str, data: StoryboardGenerateRequest, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    query = db.query(Shot).filter(Shot.project_id == project_id)
    if data.shot_ids:
        query = query.filter(Shot.id.in_(data.shot_ids))
    shots = query.order_by(Shot.sequence).all()
    locked = [shot.id for shot in shots if shot.confirmed]
    if locked:
        if data.shot_ids:
            raise HTTPException(status_code=423, detail=f"已审核镜头禁止重新生成: {', '.join(locked)}")
        shots = [shot for shot in shots if not shot.confirmed]
    if not shots:
        raise HTTPException(status_code=404, detail="No shots available for storyboard generation")

    task_key = _project_task_key(project_id, "storyboard")
    claim = claim_or_block(
        task_key,
        f"project:{project_id}",
        current_step="generate_storyboard_images",
        message=f"已排队，准备生成 {len(shots)} 个镜头的定稿故事板",
    )
    if not claim.claimed:
        return {"status": "storyboard_generating", "project_id": project_id, "deduplicated": True}

    try:
        expected_versions: dict[str, int] = {}
        for shot in shots:
            create_version(db, shot, "regenerate", task_id=task_key)
            previous_scene_key = _shot_scene_key(shot)
            _invalidate_storyboard_outputs(shot)
            _invalidate_downstream_media(db, shot, {previous_scene_key, _shot_scene_key(shot)}, source="regenerate")
            shot.storyboard_status = "queued"
            shot.status = "pending"
            shot.version = (shot.version or 1) + 1
            expected_versions[shot.id] = shot.version
        project.status = "storyboard_generating"
        db.commit()

        task = start_task(
            task_key,
            _run_storyboard_generation(
                project_id,
                [shot.id for shot in shots],
                expected_versions,
                **_capability_kwargs(data.capability_mode, data.confirm_capability_downgrade),
            ),
        )
    except BaseException as exc:
        db.rollback()
        finish_task(task_key, "failed", f"storyboard scheduling failed: {exc}")
        raise
    _regeneration_tasks.add(task)
    task.add_done_callback(_regeneration_tasks.discard)
    return {"status": "storyboard_started", "project_id": project_id, "shots": len(shots)}


@router.post("/{project_id}/confirm-storyboard")
async def confirm_storyboard(project_id: str, db: Session = Depends(get_db)):
    shots = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
    if not shots:
        raise HTTPException(status_code=404, detail="No storyboard shots available for confirmation")
    unfinished = [
        shot.id
        for shot in shots
        if (not shot.storyboard_path and not shot.image_path) or shot.media_stale
    ]
    if unfinished:
        raise HTTPException(
            status_code=400,
            detail="仍有镜头未生成定稿故事板或素材待重新生成（参数已变更）",
        )
    unapproved = [shot.id for shot in shots if not shot.confirmed]
    if unapproved:
        raise HTTPException(status_code=400, detail="仍有镜头未通过人工审核")

    project = db.query(Project).filter(Project.id == project_id).first()
    if project:
        project.status = "storyboard_approved"
    db.commit()

    return {"status": "storyboard_approved", "project_id": project_id, "confirmed_shots": len(shots)}


@router.post("/{shot_id}/approve-storyboard")
async def approve_storyboard(shot_id: str, data: StoryboardApprovalRequest, db: Session = Depends(get_db)):
    shot = db.query(Shot).filter(Shot.id == shot_id).first()
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    if data.approved and not (shot.storyboard_path or shot.image_path):
        raise HTTPException(status_code=400, detail="该镜头故事板尚未生成")
    if data.approved and shot.media_stale:
        # 参数已变更但旧素材仍在展示：审核的必须是「与当前参数一致」的素材。
        raise HTTPException(status_code=400, detail="该镜头参数已修改，素材待重新生成，请先重新生成再审核")
    project_id = shot.project_id
    result_id = shot.id
    was_approved = bool(shot.confirmed)
    shot.confirmed = bool(data.approved)
    shot.status = "storyboard_approved" if data.approved else "needs_review"
    revoked = was_approved and not data.approved
    if revoked:
        # 撤销审核会清空视频产物：被替换的状态先进版本历史。
        create_version(db, shot, "manual_edit")
        shot.version = (shot.version or 1) + 1
        _invalidate_video_outputs(shot)
        _invalidate_downstream_media(db, shot, source="manual_edit")
        shot.status = "needs_review"
        _mark_project_output_stale(db, project_id, status="storyboard_ready")
    db.commit()
    if revoked:
        await cancel_scopes(
            {f"shot:{shot_id}", f"project:{project_id}"},
            "storyboard approval was revoked",
        )
    return {
        "id": result_id,
        "approved": bool(data.approved),
        "status": "storyboard_approved" if data.approved else "needs_review",
    }


@router.post("/{shot_id}/generate-video")
async def generate_shot_video(shot_id: str, data: ShotVideoGenerateRequest, db: Session = Depends(get_db)):
    shot = db.query(Shot).filter(Shot.id == shot_id).first()
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    task_key = _shot_task_key(shot_id, "video")
    if not shot.confirmed and not data.allow_unconfirmed:
        raise HTTPException(status_code=400, detail="请先审核通过该镜头故事板")
    if not (shot.storyboard_path or shot.image_path):
        raise HTTPException(status_code=400, detail="该镜头尚未生成定稿故事板")
    if _can_reuse_existing_video(shot, data.force):
        return {"id": shot.id, "status": shot.status, "video_path": shot.video_path, "audio_path": shot.audio_path}

    # 启动前预检：视频端点必配；配音端点默认必配，但视频模型具备原生对白语音
    # 能力（或镜头无台词 / 镜头级显式指定 native）时不强制要求 TTS。
    ensure_providers_ready(
        "shot_video",
        has_dialogue=bool((shot.dialogue or "").strip()),
        audio_mode_override=str(_json_dict(shot.continuity_profile).get("audio_mode") or ""),
    )

    expected_version = shot.version or 1
    claim = claim_or_block(
        task_key,
        f"shot:{shot_id}",
        version=expected_version,
        current_step="generate_voice",
        message=f"已排队，准备生成镜头 {shot.sequence} 的配音与视频",
    )
    if not claim.claimed:
        return {"id": shot.id, "status": "video_generating", "deduplicated": True}
    try:
        # 视频重新生成前保存当前状态（含旧视频/配音路径），供 A/B 对比与回滚。
        create_version(db, shot, "regenerate", task_id=task_key)
        shot.status = "video_generating"
        _mark_project_output_stale(db, shot.project_id, status="storyboard_approved")
        db.commit()
        task = start_task(
            task_key,
            _run_single_shot_video(
                shot_id,
                data.force,
                expected_version,
                data.reuse_audio,
                **_capability_kwargs(data.capability_mode, data.confirm_capability_downgrade),
            ),
        )
    except BaseException as exc:
        db.rollback()
        finish_task(task_key, "failed", f"video scheduling failed: {exc}")
        raise
    _shot_video_tasks.add(task)
    task.add_done_callback(_shot_video_tasks.discard)
    return {"id": shot.id, "status": "video_generating"}


@router.post("/{shot_id}/generate-audio")
async def generate_shot_audio(shot_id: str, data: ShotAudioGenerateRequest, db: Session = Depends(get_db)):
    """只生成镜头配音；独立于视频阶段，供选择性重生成队列使用。"""
    shot = db.query(Shot).filter(Shot.id == shot_id).first()
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    if shot.confirmed and not data.force:
        raise HTTPException(status_code=423, detail="已审核锁定的镜头禁止重新生成配音")
    if not (shot.dialogue or "").strip():
        return {"id": shot.id, "status": shot.status, "audio_path": shot.audio_path, "skipped": True}
    task_key = _shot_task_key(shot_id, "audio")
    expected_version = shot.version or 1
    if data.reuse_existing and _reusable_audio_path(shot_id, expected_version, shot.audio_path):
        return {"id": shot.id, "status": shot.status, "audio_path": shot.audio_path, "skipped": True}
    # 纯配音任务本身就是 TTS 调用：语音端点未配置时直接拒绝（已有可复用配音除外）。
    ensure_providers_ready("shot_audio")
    claim = claim_or_block(
        task_key,
        f"shot:{shot_id}",
        version=expected_version,
        current_step="generate_voice",
        message=f"已排队，准备生成镜头 {shot.sequence} 的配音",
    )
    if not claim.claimed:
        return {"id": shot.id, "status": "audio_generating", "deduplicated": True}
    create_version(db, shot, "regenerate", task_id=task_key)
    shot.status = "audio_generating"
    _mark_project_output_stale(db, shot.project_id, status="storyboard_approved" if shot.confirmed else "storyboard_ready")
    db.commit()
    task = start_task(task_key, _run_single_shot_audio(shot_id, expected_version))
    _regeneration_tasks.add(task)
    task.add_done_callback(_regeneration_tasks.discard)
    return {"id": shot.id, "status": "audio_generating"}


@router.get("/{shot_id}/versions")
async def list_shot_versions(shot_id: str, db: Session = Depends(get_db)):
    """镜头版本时间线（新版本在前）。``current_version_id`` 标记与当前状态一致的记录。"""

    _validate_id_or_400(shot_id, "镜头 ID")
    shot = db.query(Shot).filter(Shot.id == shot_id).first()
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    versions = list_versions(db, shot_id)
    current_hash = content_hash(capture_current_snapshot(db, shot))
    return {
        "shot_id": shot_id,
        "versions": versions,
        "current_version_id": next((item["id"] for item in versions if item["content_hash"] == current_hash), None),
    }


@router.get("/{shot_id}/versions/compare")
async def compare_shot_versions(
    shot_id: str,
    a: str = Query(...),
    b: str = Query(...),
    db: Session = Depends(get_db),
):
    """A/B 对比两个版本：返回双方完整快照与逐字段差异，只读、不改变当前状态。"""

    _validate_id_or_400(shot_id, "镜头 ID")
    _validate_id_or_400(a, "版本 ID")
    _validate_id_or_400(b, "版本 ID")
    if not db.query(Shot).filter(Shot.id == shot_id).first():
        raise HTTPException(status_code=404, detail="Shot not found")
    if a == b:
        raise HTTPException(status_code=400, detail="对比的两个版本不能相同")
    row_a = db.query(ShotVersion).filter(ShotVersion.id == a, ShotVersion.shot_id == shot_id).first()
    row_b = db.query(ShotVersion).filter(ShotVersion.id == b, ShotVersion.shot_id == shot_id).first()
    if not row_a or not row_b:
        raise HTTPException(status_code=404, detail="Version not found")
    diff = diff_snapshots(parse_snapshot(row_a), parse_snapshot(row_b))
    return {
        "shot_id": shot_id,
        "a": version_detail(row_a),
        "b": version_detail(row_b),
        "diff": diff,
        "changed_fields": [item["field"] for item in diff if item["changed"]],
    }


@router.get("/{shot_id}/versions/{version_id}")
async def get_shot_version(shot_id: str, version_id: str, db: Session = Depends(get_db)):
    """版本详情：元数据 + 完整字段快照。"""

    _validate_id_or_400(shot_id, "镜头 ID")
    _validate_id_or_400(version_id, "版本 ID")
    if not db.query(Shot).filter(Shot.id == shot_id).first():
        raise HTTPException(status_code=404, detail="Shot not found")
    row = db.query(ShotVersion).filter(ShotVersion.id == version_id, ShotVersion.shot_id == shot_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Version not found")
    return version_detail(row)


@router.post("/{shot_id}/versions/{version_id}/restore")
async def restore_shot_version(shot_id: str, version_id: str, db: Session = Depends(get_db)):
    """把历史版本恢复为当前版本。

    恢复只追加：先保存被替换的当前状态，再按快照写回镜头并追加「恢复后」的
    新版本记录；历史记录不改写。恢复前校验快照引用的媒体与资产仍然有效，
    缺失时返回 409 与明确原因。已审核锁定的镜头禁止恢复。
    """

    _validate_id_or_400(shot_id, "镜头 ID")
    _validate_id_or_400(version_id, "版本 ID")
    shot = db.query(Shot).filter(Shot.id == shot_id).first()
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    _ensure_shot_unlocked(shot)
    row = db.query(ShotVersion).filter(ShotVersion.id == version_id, ShotVersion.shot_id == shot_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Version not found")
    snapshot = parse_snapshot(row)

    missing_files = missing_media(snapshot)
    if missing_files:
        raise HTTPException(
            status_code=409,
            detail="版本引用的媒体文件已缺失，无法恢复: " + ", ".join(missing_files),
        )
    missing_assets = missing_asset_bindings(db, shot, snapshot)
    if missing_assets:
        raise HTTPException(
            status_code=409,
            detail="版本绑定的资产已不存在，无法恢复: " + ", ".join(missing_assets),
        )

    project_id = shot.project_id
    create_version(db, shot, "restore")
    apply_snapshot_to_shot(shot, snapshot)
    shot.version = (shot.version or 1) + 1
    # 恢复后的媒体与恢复后的参数一致：清除过期标记，旧素材重新作为当前素材。
    clear_shot_media_stale(shot)
    _mark_project_output_stale(db, project_id)
    # 恢复必须留下新版本记录（内容与被恢复版本一致，来源标记为 restore）。
    restored_row = create_version(db, shot, "restore", force=True)
    db.commit()

    # 版本号已递增 + 作用域取消双保险：在途任务既过不了版本校验，也被等待退出，
    # 不会把恢复后的镜头再覆盖掉。
    await cancel_scopes(
        {f"shot:{shot_id}", f"project:{project_id}"},
        "shot version was restored",
    )
    payload = _shot_update_payload(shot)
    payload["restored_from_version_id"] = row.id
    await ws_manager.send_to_project(project_id, payload)
    return {
        "id": shot.id,
        "status": "updated",
        "version": shot.version,
        "restored_from_version_id": row.id,
        "new_version_id": restored_row.id if restored_row is not None else None,
        "shot": _serialize_shot(shot),
    }


async def _run_storyboard_generation(
    project_id: str,
    shot_ids: list[str],
    expected_versions: dict[str, int] | None = None,
    *,
    capability_mode: str = "manual",
    confirm_capability_downgrade: bool = False,
) -> None:
    """Run one project storyboard job at a time."""

    lock = _project_generation_locks.setdefault(project_id, asyncio.Lock())
    if lock.locked():
        raise RuntimeError("项目故事板生成任务已在运行")
    if expected_versions is None:
        db = SessionLocal()
        try:
            expected_versions = {
                shot_id: version or 1
                for shot_id, version in db.query(Shot.id, Shot.version)
                .filter(Shot.project_id == project_id, Shot.id.in_(shot_ids))
                .all()
            }
        finally:
            db.close()
    async with lock:
        await _run_storyboard_generation_impl(
            project_id,
            shot_ids,
            expected_versions,
            **_capability_kwargs(capability_mode, confirm_capability_downgrade),
        )


async def _run_storyboard_generation_impl(
    project_id: str,
    shot_ids: list[str],
    expected_versions: dict[str, int],
    *,
    capability_mode: str = "manual",
    confirm_capability_downgrade: bool = False,
) -> None:
    try:
        db = SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            skill_config = resolve_skill_config(project_id, db)
            scenes = _scenes(db, project_id)
        finally:
            db.close()
        await _ensure_scene_baselines(project_id, project, scenes, skill_config)

        db = SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            characters = _characters(db, project_id)
            scenes = _scenes(db, project_id)
        finally:
            db.close()

        for index, shot_id in enumerate(shot_ids):
            db = SessionLocal()
            try:
                shot = db.query(Shot).filter(Shot.id == shot_id, Shot.project_id == project_id).first()
                if not shot:
                    continue
                expected_version = expected_versions.get(shot.id)
                if expected_version is not None and (shot.version or 1) != expected_version:
                    continue
                shot_data = _shot_dict(shot)
                shot_data["visual_notes"] = _storyboard_notes(shot, scenes)
                if project:
                    shot_data["output_format"] = project.output_format or "9:16"
                # 同场景优先用上一镜尾帧（Seedance return_last_frame 产物）作为续帧参考：
                # 该帧尚无视频时自动回退到上一镜已审核故事板，首次全量出图行为不变。
                previous_reference = _previous_reference_for_shot(db, shot, prefer_last_frame=True)
                shot_data.update(
                    consistency_service.build_generation_context(
                        shot_data,
                        characters,
                        scenes,
                        previous_reference_path=previous_reference,
                        for_video=False,
                    )
                )
                apply_agent_config_to_shot(shot_data, skill_config)
                style_params = _storyboard_style_params(project, skill_config)
                seed = 42 + (shot.version or 1) * 100
            finally:
                db.close()

            await _progress(
                project_id,
                "generate_storyboard_images",
                48 + min(index * 4, 35),
                f"正在生成镜头 {shot.sequence} 的定稿故事板参考图",
                job_keys=(f"project:{project_id}:storyboard", f"shot:{shot_id}:storyboard"),
            )
            _materialize_control_references(project_id, shot_data, skill_config)
            image_path = await image_service.generate_shot_image(
                shot=shot_data,
                characters=characters,
                style_params=style_params,
                project_id=project_id,
                seed=seed,
                **_capability_kwargs(capability_mode, confirm_capability_downgrade),
            )

            db = SessionLocal()
            try:
                shot = db.query(Shot).filter(Shot.id == shot_id, Shot.project_id == project_id).first()
                if not shot or (expected_version is not None and (shot.version or 1) != expected_version):
                    continue
                shot.scene_group_id = shot_data.get("scene_group_id", shot.scene_group_id)
                shot.consistency_context = shot_data.get("consistency_context", shot.consistency_context)
                shot.reference_weights = json.dumps(shot_data.get("reference_weights", {}), ensure_ascii=False)
                storyboard_profile = shot_data.get("continuity_profile", {}) or {}
                # 图像 Provider 真实能力如实落库：provider/model/reference_mode、
                # references_validated（已校验数）与 references_sent（实际发送数）。
                image_meta = dict(image_service.last_generation_metadata or {})
                storyboard_profile.update(
                    {
                        "provider": image_meta.get("provider", ""),
                        "model": image_meta.get("model", ""),
                        "provider_source": image_meta.get("provider_source", ""),
                        "reference_mode": image_meta.get("reference_mode", ""),
                        "references_validated": image_meta.get("references_validated", 0),
                        "references_sent": image_meta.get("references_sent", 0),
                        "references_sent_detail": image_meta.get("references_sent_detail", []),
                        "control_types_sent": image_meta.get("control_types_sent", []),
                        "provider_capabilities": image_meta.get("provider_capabilities", {}),
                        "reference_weight_policy": image_meta.get("reference_weight_policy", "text_only_policy"),
                        "consistency_metrics": image_meta.get("consistency_metrics", {}),
                        "generation_report": image_meta,
                        "references_unsupported": bool(image_meta.get("references_unsupported")),
                        "reference_capability_warning": image_meta.get("reference_capability_warning", ""),
                        "prompt_trimmed_fields": list(image_service.last_prompt_trimmed_fields or []),
                        "requested_style": shot_data.get("requested_style", shot_data.get("style", "anime")),
                        "effective_style": shot_data.get("effective_style", shot_data.get("style", "anime")),
                        "style_source": shot_data.get("style_source", "project_request"),
                    }
                )
                shot.continuity_profile = json.dumps(storyboard_profile, ensure_ascii=False)
                shot.continuity_reference_path = previous_reference
                shot.pose_reference_path = ""
                shot.depth_reference_path = ""
                shot.image_path = image_path
                shot.storyboard_path = image_path
                shot.storyboard_status = "done"
                shot.status = "storyboard_done"
                # 新故事板原子替换旧路径；若旧视频/配音仍引用旧故事板，保持
                # stale 提示用户重生成下游媒体，否则过期标记就此清除。
                shot.media_stale = bool(shot.video_path or shot.audio_path or shot.last_frame_path)
                shot.style_fingerprint = hashlib.sha256(
                    str(shot_data.get("effective_style") or shot_data.get("style") or "anime").encode()
                ).hexdigest()[:16]
                # 生成结果同样入版本历史（与版本校验同事务，过期任务写不进来）。
                create_version(db, shot, "regenerate", task_id=f"project:{project_id}:storyboard")
                db.commit()
                update = _shot_update_payload(shot)
            finally:
                db.close()
            await ws_manager.send_to_project(
                project_id,
                update,
            )

        db = SessionLocal()
        try:
            if expected_versions:
                stale_or_missing = [
                    item.id
                    for item in db.query(Shot)
                    .filter(Shot.project_id == project_id, Shot.id.in_(shot_ids))
                    .all()
                    if (item.version or 1) != expected_versions.get(item.id)
                    or item.storyboard_status != "done"
                ]
                requested_ids = set(expected_versions)
                found_ids = {item.id for item in db.query(Shot.id).filter(Shot.project_id == project_id, Shot.id.in_(shot_ids)).all()}
                stale_or_missing.extend(sorted(requested_ids - found_ids))
                if stale_or_missing:
                    project = db.query(Project).filter(Project.id == project_id).first()
                    if project and project.status == "storyboard_generating":
                        project.status = "assets_ready"
                        db.commit()
                    raise asyncio.CancelledError(f"故事板任务版本已变化: {', '.join(stale_or_missing)}")

            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = "storyboard_ready"
                db.commit()
            project_style = (project.style if project else "anime") or "anime"
        finally:
            db.close()
        await _progress(
            project_id,
            "wait_storyboard_approval",
            72,
            "定稿故事板参考图已生成，等待人工审核",
            job_keys=(f"project:{project_id}:storyboard",),
        )
        style_meta = resolve_effective_style(project_style, skill_config, "storyboard_agent")
        await ws_manager.send_to_project(
            project_id,
            {
                "type": "storyboard_ready",
                "project_id": project_id,
                **style_meta,
            },
        )
    except Exception as exc:
        db = SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = "error"
            queued = db.query(Shot).filter(Shot.project_id == project_id, Shot.storyboard_status == "queued").all()
            for shot in queued:
                shot.storyboard_status = "failed"
                if shot.status == "pending":
                    shot.status = "failed"
            db.commit()
        finally:
            db.close()
        await ws_manager.send_to_project(
            project_id,
            report_failure(
                exc,
                error_type=ERROR_STORYBOARD,
                message="定稿故事板生成失败，本次任务已停止。请检查镜头参数与模型配置后重试。",
                context={"project_id": project_id},
            ),
        )
        raise


async def _regenerate_single_shot(
    shot_id: str,
    reason: str = "",
    expected_version: int | None = None,
    *,
    capability_mode: str = "manual",
    confirm_capability_downgrade: bool = False,
):
    lock = _shot_generation_locks.setdefault(shot_id, asyncio.Lock())
    if lock.locked():
        raise RuntimeError("镜头故事板生成任务已在运行")
    await lock.acquire()
    project_id = ""
    try:
        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if not shot or (expected_version is not None and (shot.version or 1) != expected_version):
                raise asyncio.CancelledError("镜头版本已变化")
            project_id = shot.project_id
            project = db.query(Project).filter(Project.id == project_id).first()
            skill_config = resolve_skill_config(project_id, db)
            scenes = _scenes(db, project_id)
        finally:
            db.close()

        await _ensure_scene_baselines(project_id, project, scenes, skill_config)

        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if not shot or (expected_version is not None and (shot.version or 1) != expected_version):
                raise asyncio.CancelledError("镜头版本已变化")
            project = db.query(Project).filter(Project.id == project_id).first()
            characters = _characters(db, project_id)
            scenes = _scenes(db, project_id)
            shot_data = _shot_dict(shot)
            shot_data["visual_notes"] = reason or _storyboard_notes(shot, scenes)
            if project:
                shot_data["output_format"] = project.output_format or "9:16"
            # 同场景优先用上一镜尾帧（Seedance return_last_frame 产物）作为续帧参考：
            # 该帧尚无视频时自动回退到上一镜已审核故事板，因此首次全量出图行为不变。
            previous_reference = _previous_reference_for_shot(db, shot, prefer_last_frame=True)
            shot_data.update(
                consistency_service.build_generation_context(
                    shot_data,
                    characters,
                    scenes,
                    previous_reference_path=previous_reference,
                    for_video=False,
                )
            )
            apply_agent_config_to_shot(shot_data, skill_config)
            style_params = _storyboard_style_params(project, skill_config)
            seed = 42 + (shot.version or 1) * 100
        finally:
            db.close()

        update_job_progress(
            f"shot:{shot_id}:storyboard",
            55,
            current_step="regenerate_storyboard",
            message="正在重新生成镜头故事板参考图",
        )
        _materialize_control_references(project_id, shot_data, skill_config)
        image_path = await image_service.generate_shot_image(
            shot=shot_data,
            characters=characters,
            style_params=style_params,
            project_id=project_id,
            seed=seed,
            **_capability_kwargs(capability_mode, confirm_capability_downgrade),
        )

        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if not shot or (expected_version is not None and (shot.version or 1) != expected_version):
                raise asyncio.CancelledError("镜头版本已变化")
            shot.scene_group_id = shot_data.get("scene_group_id", shot.scene_group_id)
            shot.consistency_context = shot_data.get("consistency_context", shot.consistency_context)
            shot.reference_weights = json.dumps(shot_data.get("reference_weights", {}), ensure_ascii=False)
            storyboard_profile = shot_data.get("continuity_profile", {}) or {}
            image_meta = dict(image_service.last_generation_metadata or {})
            storyboard_profile.update(
                {
                    "provider": image_meta.get("provider", ""),
                    "model": image_meta.get("model", ""),
                    "reference_mode": image_meta.get("reference_mode", ""),
                    "references_validated": image_meta.get("references_validated", 0),
                    "references_sent": image_meta.get("references_sent", 0),
                    "references_sent_detail": image_meta.get("references_sent_detail", []),
                    "control_types_sent": image_meta.get("control_types_sent", []),
                    "provider_capabilities": image_meta.get("provider_capabilities", {}),
                    "reference_weight_policy": image_meta.get("reference_weight_policy", "text_only_policy"),
                    "consistency_metrics": image_meta.get("consistency_metrics", {}),
                    "generation_report": image_meta,
                    "prompt_trimmed_fields": list(image_service.last_prompt_trimmed_fields or []),
                    "requested_style": shot_data.get("requested_style", shot_data.get("style", "anime")),
                    "effective_style": shot_data.get("effective_style", shot_data.get("style", "anime")),
                    "style_source": shot_data.get("style_source", "project_request"),
                }
            )
            shot.continuity_profile = json.dumps(storyboard_profile, ensure_ascii=False)
            shot.continuity_reference_path = previous_reference
            shot.pose_reference_path = ""
            shot.depth_reference_path = ""
            shot.image_path = image_path
            shot.storyboard_path = image_path
            shot.storyboard_status = "done"
            shot.status = "storyboard_done"
            # 新故事板原子替换旧路径；若旧视频/配音仍引用旧故事板，保持
            # stale 提示用户重生成下游媒体，否则过期标记就此清除。
            shot.media_stale = bool(shot.video_path or shot.audio_path or shot.last_frame_path)
            shot.style_fingerprint = hashlib.sha256(
                str(shot_data.get("effective_style") or shot_data.get("style") or "anime").encode()
            ).hexdigest()[:16]
            create_version(db, shot, "regenerate", task_id=f"shot:{shot_id}:storyboard")
            db.commit()
            update = _shot_update_payload(shot)
        finally:
            db.close()
        await ws_manager.send_to_project(project_id, update)
    except Exception as exc:
        error_id = log_failure(exc, error_type=ERROR_STORYBOARD, context={"shot_id": shot_id})
        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if shot and (expected_version is None or (shot.version or 1) == expected_version):
                shot.status = "failed"
                shot.storyboard_status = "failed"
                # 落库的失败备注会回显到界面，只保留简短提示与错误编号。
                shot.visual_notes = f"重新生成故事板失败（错误编号 {error_id}）"
                project_id = shot.project_id
                should_notify = True
                db.commit()
            else:
                should_notify = False
        finally:
            db.close()
        if should_notify:
            await ws_manager.send_to_project(
                project_id,
                error_payload(
                    error_type=ERROR_STORYBOARD,
                    message="镜头故事板重新生成失败。请检查镜头参数、素材绑定与模型配置后重试。",
                    error_id=error_id,
                ),
            )
        raise
    finally:
        lock.release()


async def _run_single_shot_audio(shot_id: str, expected_version: int) -> None:
    """配音阶段的最小独立 worker；写入前再次校验版本，避免取消后的迟到发布。"""
    project_id = ""
    try:
        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if not shot or (shot.version or 1) != expected_version:
                raise asyncio.CancelledError("镜头版本已变化")
            project_id = shot.project_id
            dialogue = clean_tts_text(shot.dialogue or "", resolve_skill_config(project_id, db))
            speaker = (_json_list(shot.characters_in_scene) or [""])[0]
            characters = _characters(db, project_id)
            voice_id = next((item.get("voice_id", "") for item in characters if item.get("name") == speaker), "")
            emotion = shot.emotion or "neutral"
        finally:
            db.close()
        audio_path = await tts_service.generate_dialogue(
            text=dialogue,
            voice_id=voice_id,
            emotion=emotion,
            project_id=project_id,
            shot_id=_versioned_media_id(shot_id, expected_version),
        )
        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if not shot or (shot.version or 1) != expected_version:
                raise asyncio.CancelledError("镜头版本已变化")
            shot.audio_path = audio_path
            shot.status = "video_done" if shot.video_path else ("storyboard_approved" if shot.confirmed else "storyboard_done")
            create_version(db, shot, "regenerate", task_id=f"shot:{shot_id}:audio")
            db.commit()
            await ws_manager.send_to_project(project_id, _shot_update_payload(shot))
        finally:
            db.close()
    except Exception as exc:
        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if shot and (shot.version or 1) == expected_version:
                shot.status = "failed"
                db.commit()
        finally:
            db.close()
        raise exc


async def _run_single_shot_video(
    shot_id: str,
    force: bool = False,
    expected_version: int | None = None,
    reuse_audio: bool = False,
    *,
    capability_mode: str = "manual",
    confirm_capability_downgrade: bool = False,
) -> None:
    lock = _shot_generation_locks.setdefault(shot_id, asyncio.Lock())
    if lock.locked():
        raise RuntimeError("镜头视频生成任务已在运行")
    await lock.acquire()
    project_id = ""
    shot_sequence = 0
    try:
        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if not shot:
                return
            if expected_version is None:
                expected_version = shot.version or 1
            elif (shot.version or 1) != expected_version:
                raise asyncio.CancelledError("镜头版本已变化")
            if _can_reuse_existing_video(shot, force):
                return
            project_id = shot.project_id
            project = db.query(Project).filter(Project.id == project_id).first()
            skill_config = resolve_skill_config(project_id, db)
            scenes = _scenes(db, project_id)
        finally:
            db.close()

        await _ensure_scene_baselines(project_id, project, scenes, skill_config)

        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if not shot or (expected_version is not None and (shot.version or 1) != expected_version):
                raise asyncio.CancelledError("镜头版本已变化")
            project = db.query(Project).filter(Project.id == project_id).first()
            characters = _characters(db, project_id)
            scenes = _scenes(db, project_id)
            shot_data = _shot_dict(shot)
            shot_data["storyboard_prompt"] = _storyboard_notes(shot, scenes)
            if project:
                shot_data["output_format"] = project.output_format or "9:16"
                shot_data["resolution"] = project.resolution or "720p"
                shot_data["style"] = project.style or "anime"
            previous_reference = _previous_reference_for_shot(db, shot, prefer_last_frame=True)
            shot_data.update(
                consistency_service.build_generation_context(
                    shot_data,
                    characters,
                    scenes,
                    previous_reference_path=previous_reference,
                    for_video=True,
                )
            )
            apply_agent_config_to_shot(shot_data, skill_config)
            # 镜头级 audio_mode 覆盖存于 continuity_profile，但一致性上下文会重建
            # profile，这里从数据库存档提升为 shot 顶级字段，保证覆盖不被冲掉。
            stored_audio_mode = _json_dict(shot.continuity_profile).get("audio_mode")
            if stored_audio_mode:
                shot_data["audio_mode"] = str(stored_audio_mode).strip().lower()
            shot_sequence = shot.sequence
            dialogue = shot.dialogue
            emotion = shot.emotion or "neutral"
            speakers = _json_list(shot.characters_in_scene)
            speaker = speakers[0] if speakers else ""
        finally:
            db.close()
        _materialize_control_references(project_id, shot_data, skill_config)

        await _progress(
            project_id,
            "generate_voice",
            82,
            f"正在准备镜头 {shot_sequence} 的音频（音频路由决策中）",
            job_keys=(f"shot:{shot_id}:video",),
        )
        media_id = _versioned_media_id(shot_id, expected_version)
        audio_path = shot_data.get("audio_path", "")
        audio_mode = resolve_audio_mode(shot_data)
        native_routed = audio_mode == "native"
        dialogues = None
        if native_routed:
            # 原生音频路径：对白交给视频模型经 prompt 生成并随视频直出，
            # 跳过独立 TTS 配音与后续 ffmpeg 音轨合成。
            audio_path = ""
            shot_data["audio_path"] = ""
            if dialogue:
                dialogues = [
                    Dialogue(role=speaker, text=clean_tts_text(dialogue, skill_config), emotion=emotion)
                ]
        elif dialogue:
            reusable_audio = _reusable_audio_path(shot_id, expected_version, audio_path) if reuse_audio else ""
            if reusable_audio:
                # 续跑 / 重试：该版本已经有有效配音，不再重复调用 TTS。
                audio_path = reusable_audio
                shot_data["audio_path"] = audio_path
                await _progress(
                    project_id,
                    "generate_voice",
                    84,
                    f"镜头 {shot_sequence} 已有有效配音，复用后继续生成视频",
                    job_keys=(f"shot:{shot_id}:video",),
                )
            else:
                await _progress(
                    project_id,
                    "generate_voice",
                    84,
                    f"正在生成镜头 {shot_sequence} 的配音",
                    job_keys=(f"shot:{shot_id}:video",),
                )
                voice_id = next((item.get("voice_id", "") for item in characters if item.get("name") == speaker), "")
                audio_path = await tts_service.generate_dialogue(
                    text=clean_tts_text(dialogue, skill_config),
                    voice_id=voice_id,
                    emotion=emotion,
                    project_id=project_id,
                    shot_id=media_id,
                )
                shot_data["audio_path"] = audio_path

        await _progress(
            project_id,
            "generate_seedance_video",
            90,
            f"正在生成镜头 {shot_sequence} 的视频",
            job_keys=(f"shot:{shot_id}:video",),
        )
        video_shot_data = {**shot_data, "shot_id": media_id, "dialogues": dialogues}
        result = await seedance_service.generate_shot_video(
            video_shot_data,
            characters,
            scenes,
            project_id,
            **_capability_kwargs(capability_mode, confirm_capability_downgrade),
        )
        if native_routed and not result.get("native_audio"):
            raise RuntimeError("视频适配器未按原生音频模式返回带音轨视频，已阻止无声成品")
        continuity_profile = shot_data.get("continuity_profile", {}) or {}
        video_report = dict(result.get("generation_report") or seedance_service.last_generation_metadata or {})
        if result.get("reference_payload_mode") or video_report:
            continuity_profile["seedance_reference_payload_mode"] = result.get("reference_payload_mode", "")
            continuity_profile["reference_mode"] = video_report.get("reference_mode", "")
            continuity_profile["references_validated"] = len(video_shot_data.get("seedance_reference_manifest") or [])
            continuity_profile["references_sent"] = video_report.get("references_sent", [])
            continuity_profile["references_sent_detail"] = video_report.get("references_sent_detail", [])
            continuity_profile["control_types_sent"] = video_report.get("control_types_sent", [])
            continuity_profile["provider_capabilities"] = video_report.get("provider_capabilities", {})
            continuity_profile["reference_weight_policy"] = video_report.get("reference_weight_policy", "text_only_policy")
            continuity_profile["consistency_metrics"] = video_report.get("consistency_metrics", {})
            continuity_profile["generation_report"] = video_report
            continuity_profile["provider"] = video_report.get("provider", get_endpoint("video").protocol)
            continuity_profile["model"] = video_report.get("model", get_endpoint("video").model)
            continuity_profile["requested_style"] = video_shot_data.get("requested_style", video_shot_data.get("style", "anime"))
            continuity_profile["effective_style"] = video_shot_data.get("effective_style", video_shot_data.get("style", "anime"))
            continuity_profile["style_source"] = video_shot_data.get("style_source", "project_request")
            shot_data["continuity_profile"] = continuity_profile
        if native_routed:
            # 标记音轨来源为视频自带，渲染时沿用视频音轨而非叠加 TTS。
            continuity_profile["audio_source"] = "native"
            shot_data["continuity_profile"] = continuity_profile
        if shot_data.get("audio_mode"):
            # 镜头级覆盖写回存档，保证后续重新生成时依然生效。
            continuity_profile["audio_mode"] = str(shot_data["audio_mode"]).strip().lower()
            shot_data["continuity_profile"] = continuity_profile

        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if not shot or (expected_version is not None and (shot.version or 1) != expected_version):
                raise asyncio.CancelledError("镜头版本已变化")
            shot.scene_group_id = shot_data.get("scene_group_id", shot.scene_group_id)
            shot.consistency_context = shot_data.get("consistency_context", shot.consistency_context)
            shot.reference_weights = json.dumps(shot_data.get("reference_weights", {}), ensure_ascii=False)
            shot.continuity_profile = json.dumps(shot_data.get("continuity_profile", {}), ensure_ascii=False)
            shot.continuity_reference_path = previous_reference
            shot.pose_reference_path = shot_data.get("pose_reference_path", "")
            shot.depth_reference_path = shot_data.get("depth_reference_path", "")
            shot.audio_path = audio_path
            shot.video_path = result["video_path"]
            shot.last_frame_path = result.get("frame_path", "")
            if not shot.image_path:
                shot.image_path = result.get("frame_path", "")
            shot.status = "video_done"
            # 视频 + 配音 + 尾帧全部基于当前参数重新生成：过期标记就此清除。
            clear_shot_media_stale(shot)
            create_version(db, shot, "regenerate", task_id=f"shot:{shot_id}:video")
            db.commit()
            update = _shot_update_payload(shot)
        finally:
            db.close()

        await ws_manager.send_to_project(project_id, update)
    except Exception as exc:
        error_id = log_failure(exc, error_type=ERROR_SHOT_VIDEO, context={"shot_id": shot_id})
        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if shot and (expected_version is None or (shot.version or 1) == expected_version):
                shot.status = "failed"
                shot.visual_notes = f"单镜头视频生成失败（错误编号 {error_id}）"
                project_id = shot.project_id
                shot_sequence = shot.sequence
                db.commit()
                should_notify = True
            else:
                should_notify = False
        finally:
            db.close()
        if should_notify:
            await ws_manager.send_to_project(
                project_id,
                error_payload(
                    error_type=ERROR_SHOT_VIDEO,
                    message=f"镜头 {shot_sequence} 视频生成失败。请检查该镜头的故事板、配音与视频模型配置后重试。",
                    error_id=error_id,
                ),
            )
        raise
    finally:
        lock.release()


def _serialize_shot(s: Shot) -> dict:
    return {
        "id": s.id,
        "project_id": s.project_id,
        "sequence": s.sequence,
        "shot_type": s.shot_type,
        "scene_description": s.scene_description,
        "character_action": s.character_action,
        "dialogue": s.dialogue,
        "camera_angle": s.camera_angle,
        "camera_movement": s.camera_movement or "静止",
        "duration": s.duration,
        "emotion": s.emotion,
        "transition": s.transition,
        "visual_notes": s.visual_notes or "",
        "image_path": s.image_path,
        "storyboard_path": s.storyboard_path,
        "video_path": s.video_path,
        "audio_path": s.audio_path,
        "status": s.status,
        "storyboard_status": s.storyboard_status,
        "version": s.version,
        "confirmed": s.confirmed,
        "media_stale": bool(s.media_stale),
        "characters_in_scene": json.loads(s.characters_in_scene) if s.characters_in_scene else [],
        "scene_asset_id": s.scene_asset_id or "",
        "character_asset_ids": json.loads(s.character_asset_ids) if s.character_asset_ids else [],
        "scene_group_id": s.scene_group_id or "",
        "consistency_context": s.consistency_context or "",
        "reference_weights": _json_dict(s.reference_weights),
        "continuity_profile": _json_dict(s.continuity_profile),
        "continuity_reference_path": s.continuity_reference_path or "",
        "pose_reference_path": s.pose_reference_path or "",
        "depth_reference_path": s.depth_reference_path or "",
        "last_frame_path": s.last_frame_path or "",
        "style_fingerprint": s.style_fingerprint or "",
    }


def _shot_update_payload(shot: Shot) -> dict:
    return {
        "type": "shot_update",
        "shot_id": shot.id,
        "status": shot.status,
        "storyboard_status": shot.storyboard_status or "pending",
        # 版本号随更新下发：前端用它丢弃旧任务的迟到响应（expected_version 语义）。
        "version": shot.version or 1,
        "confirmed": bool(shot.confirmed),
        "media_stale": bool(shot.media_stale),
        "image_path": shot.image_path,
        "storyboard_path": shot.storyboard_path,
        "audio_path": shot.audio_path,
        "video_path": shot.video_path,
        "last_frame_path": shot.last_frame_path,
        "scene_group_id": shot.scene_group_id,
        "reference_weights": _json_dict(shot.reference_weights),
        "continuity_profile": _json_dict(shot.continuity_profile),
        "continuity_reference_path": shot.continuity_reference_path,
        "pose_reference_path": shot.pose_reference_path,
        "depth_reference_path": shot.depth_reference_path,
        "reference_mode": _json_dict(shot.continuity_profile).get("reference_mode", ""),
        "references_validated": _json_dict(shot.continuity_profile).get("references_validated", False),
        "references_sent": _json_dict(shot.continuity_profile).get("references_sent", []),
        "references_sent_detail": _json_dict(shot.continuity_profile).get("references_sent_detail", []),
        "control_types_sent": _json_dict(shot.continuity_profile).get("control_types_sent", []),
        "provider_capabilities": _json_dict(shot.continuity_profile).get("provider_capabilities", {}),
        "reference_weight_policy": _json_dict(shot.continuity_profile).get("reference_weight_policy", "text_only_policy"),
        "consistency_metrics": _json_dict(shot.continuity_profile).get("consistency_metrics", {}),
        "generation_report": _json_dict(shot.continuity_profile).get("generation_report", {}),
        "provider": _json_dict(shot.continuity_profile).get("provider", ""),
        "model": _json_dict(shot.continuity_profile).get("model", ""),
        "requested_style": _json_dict(shot.continuity_profile).get("requested_style", ""),
        "effective_style": _json_dict(shot.continuity_profile).get("effective_style", ""),
        "style_source": _json_dict(shot.continuity_profile).get("style_source", ""),
        "provider_source": _json_dict(shot.continuity_profile).get("provider_source", ""),
        "references_unsupported": bool(_json_dict(shot.continuity_profile).get("references_unsupported")),
        "reference_capability_warning": _json_dict(shot.continuity_profile).get("reference_capability_warning", ""),
    }


def _shot_dict(s: Shot) -> dict:
    data = _serialize_shot(s)
    data["shot_id"] = data.pop("id")
    data["camera_movement"] = s.camera_movement or "静止"
    data["seed"] = 42
    data["visual_notes"] = s.visual_notes or ""
    data["consistency_context"] = s.consistency_context or ""
    data["scene_group_id"] = s.scene_group_id or ""
    data["reference_weights"] = _json_dict(s.reference_weights)
    data["continuity_profile"] = _json_dict(s.continuity_profile)
    data["continuity_reference_path"] = s.continuity_reference_path or ""
    data["pose_reference_path"] = s.pose_reference_path or ""
    data["depth_reference_path"] = s.depth_reference_path or ""
    data["last_frame_path"] = s.last_frame_path or ""
    if not data["image_path"] and data.get("storyboard_path"):
        data["image_path"] = data["storyboard_path"]
    data["video_path"] = s.video_path or ""
    return data


def _asset_project_id(db, project_id: str) -> str:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return project_id
    return project.parent_project_id or project.id


def _json_list(raw: str | None) -> list:
    try:
        value = json.loads(raw or "[]")
        return value if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


def _validate_asset_bindings(db: Session, shot: Shot, scene_asset_id, character_asset_ids) -> tuple[str, list[str]]:
    """Validate all asset references against the shot's owning project."""

    asset_project_id = _asset_project_id(db, shot.project_id)
    scene_id = str(scene_asset_id or "").strip()
    if scene_id:
        scene = db.query(SceneAsset).filter(SceneAsset.id == scene_id, SceneAsset.project_id == asset_project_id).first()
        if not scene:
            raise HTTPException(status_code=400, detail="场景资产不属于该项目")
    ids = list(dict.fromkeys(str(item).strip() for item in (character_asset_ids or []) if str(item).strip()))
    if ids:
        found = {
            item.id
            for item in db.query(Character).filter(Character.id.in_(ids), Character.project_id == asset_project_id).all()
        }
        if found != set(ids):
            raise HTTPException(status_code=400, detail="角色资产不属于该项目")
    return scene_id, ids


def _validate_id_or_400(value: str, field: str) -> str:
    try:
        return validate_identifier(value, field)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _shot_task_key(shot_id: str, kind: str) -> str:
    return f"shot:{shot_id}:{kind}"


def _project_task_key(project_id: str, kind: str) -> str:
    return f"project:{project_id}:{kind}"


def _versioned_media_id(shot_id: str, version: int | None) -> str:
    suffix = f"_v{version or 1}"
    candidate = f"{shot_id}{suffix}"
    if len(candidate) <= 128:
        return candidate
    digest = hashlib.sha256(shot_id.encode("utf-8")).hexdigest()[:20]
    return f"{shot_id[:96]}_{digest}{suffix}"[:128]


def _characters(db, project_id: str) -> list[dict]:
    asset_project_id = _asset_project_id(db, project_id)
    chars = db.query(Character).filter(Character.project_id == asset_project_id).all()
    result: list[dict] = []
    for c in chars:
        # stale 资产（风格已切换或来源未知）不再作为参考图复用；
        # 只保留文字画像，参考图等待按当前风格重建。
        stale = str(c.asset_status or "active") == "stale"
        references = json.loads(c.reference_images) if c.reference_images else []
        result.append(
            {
                "id": c.id,
                "name": c.name,
                "appearance": json.loads(c.appearance) if c.appearance else {},
                "personality": c.personality or "",
                "visual_prompt": c.visual_prompt or "",
                "negative_prompt": c.negative_prompt or "",
                "voice_id": c.voice_id or "",
                "key_features": json.loads(c.key_features) if c.key_features else [],
                "emotion_variants": json.loads(c.emotion_variants) if c.emotion_variants else {},
                "reference_images": [] if stale else references,
                "default_outfit": c.default_outfit or "",
                "lora_profile": "",
                "ip_adapter_profile": "",
                "wardrobe_lock": c.wardrobe_lock or "",
                "seed": int(c.seed) if c.seed and c.seed.isdigit() else 42,
                "asset_status": str(c.asset_status or "active"),
                "style_fingerprint": c.style_fingerprint or "",
            }
        )
    return result


def _scenes(db, project_id: str) -> dict[str, dict]:
    asset_project_id = _asset_project_id(db, project_id)
    scenes = db.query(SceneAsset).filter(SceneAsset.project_id == asset_project_id).all()
    result: dict[str, dict] = {}
    for item in scenes:
        stale = str(item.asset_status or "active") == "stale"
        references = json.loads(item.reference_images) if item.reference_images else []
        result[item.id] = {
            "id": item.id,
            "name": item.name,
            "description": item.description,
            "visual_prompt": item.visual_prompt,
            "negative_prompt": item.negative_prompt,
            "key_features": json.loads(item.key_features) if item.key_features else [],
            "reference_images": [] if stale else references,
            "scene_group_key": item.scene_group_key or item.id,
            "time_of_day": item.time_of_day or "",
            "baseline_image_path": "" if stale else (item.baseline_image_path or ""),
            "consistency_profile": json.loads(item.consistency_profile) if item.consistency_profile else {},
            "prop_lock": item.prop_lock or "",
            "seed": item.seed or 1200,
            "asset_status": str(item.asset_status or "active"),
            "style_fingerprint": item.style_fingerprint or "",
        }
    return result


def _json_dict(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _ensure_shot_unlocked(shot: Shot, *, force: bool = False) -> None:
    if shot.confirmed and not force:
        raise HTTPException(status_code=423, detail="已审核锁定的镜头禁止修改或重新生成")


def _invalidate_storyboard_outputs(shot: Shot) -> None:
    """编辑 / 重生成排队时标记素材过期待重生成。

    旧故事板与全部下游媒体路径必须保留（预览、对比、回滚都依赖它们），
    直到新素材生成成功的写回原子替换；这里只撤销审核并打上 stale 标记。
    """

    mark_shot_media_stale(shot)


def _reusable_audio_path(shot_id: str, version: int | None, audio_path: str | None) -> str:
    """已经生成且仍然对应当前镜头版本的有效配音路径；否则返回空串。

    只认文件名与 ``_versioned_media_id`` 完全一致的产物，避免复用旧版本配音。
    """

    if not audio_path:
        return ""
    if Path(audio_path).stem != _versioned_media_id(shot_id, version):
        return ""
    resolved = existing_file(
        audio_path,
        minimum_size=1024,
        allowed_roots=(settings.OUTPUT_DIR, settings.ASSETS_DIR, settings.DATA_DIR),
    )
    return str(resolved) if resolved is not None else ""


def _can_reuse_existing_video(shot: Shot, force: bool = False) -> bool:
    if force or getattr(shot, "media_stale", False) or shot.status != "video_done" or not shot.video_path:
        return False
    return existing_file(
        shot.video_path,
        minimum_size=4096,
        allowed_roots=(settings.OUTPUT_DIR, settings.ASSETS_DIR, settings.DATA_DIR),
    ) is not None


def _invalidate_video_outputs(shot: Shot, reset_status: bool = True) -> None:
    """撤销审核等显式操作后的失效：只标记 stale，不清空任何媒体路径。

    旧视频 / 配音 / 尾帧保留在镜头上供预览与回滚；一致性档案（含镜头级
    audio_mode 用户设置）原样保留，由下一次生成成功的写回整体替换。
    """

    mark_shot_media_stale(shot, reset_confirmed=False)
    if not reset_status:
        return
    if shot.storyboard_path or shot.image_path:
        shot.status = "storyboard_approved" if shot.confirmed else "storyboard_done"
    else:
        shot.status = "pending"


def _invalidate_downstream_media(
    db: Session,
    shot: Shot,
    scene_keys: set[str] | None = None,
    source: str = "manual_edit",
) -> None:
    """上游镜头变更后，同场景下游镜头标记待重生成（旧素材保留）。

    下游视频 / 续帧引用了上游产物，上游重新生成后它们需要跟进；版本号 +1
    隔离在途生成任务（迟到任务写不回过期版本）。变更前先进版本历史，
    保证下游旧素材随时可回滚。
    """
    keys = {key for key in (scene_keys or {_shot_scene_key(shot)}) if key}
    if not keys:
        return
    downstream = (
        db.query(Shot)
        .filter(Shot.project_id == shot.project_id, Shot.sequence > shot.sequence)
        .order_by(Shot.sequence)
        .all()
    )
    for item in downstream:
        if _shot_scene_key(item) in keys:
            if any(
                (
                    item.audio_path,
                    item.video_path,
                    item.last_frame_path,
                    item.continuity_reference_path,
                    item.pose_reference_path,
                    item.depth_reference_path,
                )
            ):
                create_version(db, item, source)
            mark_shot_media_stale(item)
            item.version = (getattr(item, "version", 1) or 1) + 1


def _mark_project_output_stale(db: Session, project_id: str, status: str = "assets_ready") -> None:
    project = db.query(Project).filter(Project.id == project_id).first()
    if project:
        project.status = status


def _shot_scene_key(shot: Shot) -> str:
    return shot.scene_group_id or shot.scene_asset_id or ""


def _shot_scene_keys(shot: Shot, db: Session | None = None) -> set[str]:
    keys = {str(value) for value in (shot.scene_group_id, shot.scene_asset_id) if value}
    if db is not None and shot.scene_asset_id:
        try:
            scene = db.query(SceneAsset).filter(SceneAsset.id == shot.scene_asset_id).first()
            if scene and scene.scene_group_key:
                keys.add(scene.scene_group_key)
        except Exception:
            pass
    return keys


def _materialize_control_references(project_id: str, shot_data: dict, skill_config: dict | None = None) -> None:
    """连续性控制参考的如实归一化。

    没有接入真实的 OpenPose / Depth 模型（``materialize_continuity_controls``
    已停用，不再产出边缘图/灰度图冒充控制图），画像必须统一标记
    ``unsupported``，且不允许出现 openpose/depth 类参考资产，避免任何
    「控制已生效」的虚假声明。
    """
    profile = shot_data.get("continuity_profile") or {}
    profile["openpose_lock"] = "unsupported"
    profile["depth_lock"] = "unsupported"
    profile["pose_control_model"] = "unsupported"
    profile["depth_control_model"] = "unsupported"
    profile.pop("pose_reference_path", None)
    profile.pop("depth_reference_path", None)
    shot_data["continuity_profile"] = profile
    shot_data["pose_reference_path"] = ""
    shot_data["depth_reference_path"] = ""
    assets = [
        asset
        for asset in (shot_data.get("reference_assets") or [])
        if isinstance(asset, dict) and str(asset.get("type") or "") not in {"openpose_source_frame", "depth_source_frame"}
    ]
    shot_data["reference_assets"] = assets


def _storyboard_notes(shot: Shot, scenes: dict[str, dict]) -> str:
    scene = scenes.get(shot.scene_asset_id or "")
    parts = [
        "finished approved storyboard keyframe",
        "full color final-look reference image",
        "lock character identity, costume, face, hairstyle and scene palette",
    ]
    if scene:
        parts.extend([scene.get("visual_prompt", ""), scene.get("description", "")])
        parts.extend([scene.get("prop_lock", ""), scene.get("baseline_image_path", "") and "preserve scene baseline reference"])
        if scene.get("reference_images"):
            parts.append("strictly preserve the approved scene asset reference")
    if shot.visual_notes:
        parts.append(shot.visual_notes)
    return ", ".join(part for part in parts if part)


def _storyboard_style_params(project: Project | None, skill_config: dict | None = None) -> dict:
    requested = (project.style if project else "anime") or "anime"
    style = resolve_effective_style(requested, skill_config, "storyboard_agent")["effective_style"]
    params = style_prompt_params(style)
    params["prompt_prefix"] = (
        f"{params.get('prompt_prefix', '')}, production-ready storyboard reference, "
        "not sketch, not rough line art, no monochrome, no grayscale"
    )
    return params


async def _ensure_scene_baselines(
    project_id: str,
    project: Project | None,
    scenes: dict[str, dict],
    skill_config: dict | None = None,
) -> None:
    if not project:
        return
    style = resolve_effective_style(project.style or "anime", skill_config, "storyboard_agent")["effective_style"]
    asset_project_id = project.parent_project_id or project_id
    skill_append = ""
    if skill_config:
        from services.skill_config_service import agent_prompt_append

        skill_append = agent_prompt_append(skill_config, "storyboard_agent")
    for scene_id, scene in scenes.items():
        if scene.get("baseline_image_path") or scene.get("reference_images"):
            continue
        try:
            scene_payload = dict(scene)
            if skill_append:
                scene_payload["visual_prompt"] = ", ".join(part for part in [scene.get("visual_prompt", ""), skill_append] if part)
            ref_path = await image_service.generate_scene_baseline_reference(
                scene=scene_payload,
                style=style,
                project_id=asset_project_id,
                seed=int(scene.get("seed") or 1200),
            )
            db = SessionLocal()
            try:
                model = (
                    db.query(SceneAsset)
                    .filter(SceneAsset.id == scene_id, SceneAsset.project_id == asset_project_id)
                    .first()
                )
                if model and not model.baseline_image_path:
                    model.baseline_image_path = ref_path
                    model.reference_images = json.dumps([ref_path], ensure_ascii=False)
                    db.commit()
            finally:
                db.close()
        except Exception:
            continue


def _previous_reference_for_shot(db: Session, shot: Shot, prefer_last_frame: bool = False) -> str:
    current_keys = _shot_scene_keys(shot, db)
    if not current_keys:
        return ""
    previous_shots = (
        db.query(Shot)
        .filter(Shot.project_id == shot.project_id, Shot.sequence < shot.sequence)
        .order_by(Shot.sequence.desc())
        .all()
    )
    for previous in previous_shots:
        if not (current_keys & _shot_scene_keys(previous, db)):
            continue
        if prefer_last_frame and previous.last_frame_path:
            return previous.last_frame_path
        return previous.storyboard_path or previous.image_path or previous.last_frame_path or ""
    return ""


async def _progress(project_id: str, step: str, progress: int, message: str, *, job_keys: tuple[str, ...] = ()):
    """推送项目进度，并把同一份进度写进任务中心的 durable 记录。

    ``job_keys`` 传候选键即可：只有真正持有当前 run token 的那个会写入成功，其余
    会被 task_registry 静默忽略，因此旧尝试的迟到回调不会覆盖新尝试。
    """

    for key in job_keys:
        update_job_progress(key, progress, current_step=step, message=message)
    await ws_manager.send_to_project(project_id, {"type": "progress", "step": step, "progress": progress, "message": message})
