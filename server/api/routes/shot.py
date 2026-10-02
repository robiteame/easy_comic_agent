import asyncio
import copy
from datetime import datetime
import hashlib
import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agent.checkpoints import CheckpointStore
from agent.contracts import (
    DecisionTrace,
    StageName,
    VideoCandidateRecord,
    VideoCandidateSelection,
    VideoCandidateStatus,
    score_video_candidate,
    select_video_candidate,
)
from api import schemas
from api.websocket import ws_manager
from config import settings
from db import SessionLocal, get_db
from models import Character, Project, SceneAsset, Shot, ShotVersion, ShotVideoCandidate
from services.audio_routing import resolve_audio_mode
from services.consistency_service import ConsistencyService, normalize_continuity_mode
from services.dialogue_audio import generate_dialogue_track
from services.invalidation_service import clear_shot_media_stale, mark_shot_media_stale
from services.reference_readiness_service import (
    blocking_report,
    build_manifest_for_shot,
    ensure_generation_gate,
    refresh_project_reference_state,
)
from services.shot_dialogue import (
    dialogue_lines_payload,
    parse_shot_dialogue,
    serialize_dialogue_lines,
    warn_unknown_speakers,
)
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
from services.providers.registry import UnknownProtocolError, get_adapter
from services.quality_review_service import quality_review_service
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
from services.structural_validation import probe_media_duration, validate_video_file
from services.story_timing import (
    ShotExecutionPlan,
    StoryTimingError,
    dialogue_text,
    estimate_speech_ms,
    provider_duration_capability,
    usable_speech_ms,
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
    dialogue: schemas.ShotDialogueList | None = None
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
    dialogue: schemas.ShotDialogueList | None = None
    duration: schemas.ShotDuration | None = None
    force_confirmed: bool = False
    # 关键镜头可一次生成 2 个候选：每个候选各自进入版本历史，人工对比后选用。
    candidates: schemas.CandidateCount = 1
    capability_mode: schemas.PipelineMode = "manual"
    confirm_capability_downgrade: bool = False


class StoryboardGenerateRequest(BaseModel):
    shot_ids: schemas.ShotIdList = Field(default_factory=list)
    # 手动模式专用：用户必须在 UI 明确确认后才允许参考素材降级。
    confirm_degraded: bool = False
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
    # 与实际生成路径保持一致：只有 continuous_action 才自动读取上一镜末帧。
    previous_shot_model = _previous_shot_for_continuity(db, shot)
    previous_shot_data = _shot_dict(previous_shot_model) if previous_shot_model else None
    shot_data = _shot_dict(shot)
    shot_data["storyboard_prompt"] = _storyboard_notes(shot, scenes)
    shot_data.update(
        consistency_service.build_generation_context(
            shot_data,
            characters,
            scenes,
            previous_shot=previous_shot_data,
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
    estimated_speech_ms = _validate_timing_edit(shot, changed, db)
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
        elif key == "dialogue":
            shot.dialogue = _serialize_dialogue_input(value, shot, db)
        else:
            setattr(shot, key, value)

    if changed:
        if "dialogue" in changed or "duration" in changed:
            shot.estimated_speech_ms = estimated_speech_ms
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


def _validate_timing_edit(shot: Shot, changed: dict, db: Session) -> int:
    """Validate dialogue capacity before an edit is recorded.

    Manual edits may temporarily describe an out-of-provider duration so version
    comparison/restore remains lossless; actual video generation and rendering
    enforce the live Provider contract and refuse such shots.
    """

    duration = float(changed.get("duration") if changed.get("duration") is not None else shot.duration or 0)
    if "dialogue" in changed:
        dialogue_raw = _serialize_dialogue_input(changed.get("dialogue"), shot, db)
    else:
        dialogue_raw = shot.dialogue
    estimated_speech_ms = estimate_speech_ms(dialogue_text(dialogue_raw))
    action = str(changed.get("character_action") if changed.get("character_action") is not None else shot.character_action or "")
    available_ms = usable_speech_ms({"character_action": action}, duration)
    if estimated_speech_ms > available_ms:
        raise HTTPException(
            status_code=409,
            detail=(
                f"镜头 {shot.id} 对白预计 {estimated_speech_ms} 毫秒，超过镜头可用时长 "
                f"{available_ms} 毫秒；请拆分台词、合并镜头或在 Provider 能力内延长该镜头"
            ),
        )
    return estimated_speech_ms


def _validate_shot_duration_for_provider(shot: Shot) -> None:
    """生成入口的时长能力校验。

    固定档 Provider（如固定 5 秒）允许更短的故事时长：按固定秒数生成后，由
    执行计划在成片中裁剪到故事时长；只有不超出固定档/上限的镜头可以放行。
    非固定档 Provider 仍要求时长落在能力范围与步长网格上。
    """

    capability = provider_duration_capability()
    narrative_s = float(shot.duration or 0)
    if capability.is_fixed:
        if narrative_s <= 0 or narrative_s > capability.max_duration + capability.tolerance_s:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"镜头 {shot.id} 时长 {narrative_s:g} 秒超出固定档视频 Provider "
                    f"{capability.protocol or '<unknown>'} 的 {capability.describe()}；"
                    "请拆分镜头到 Provider 允许的时长后逐镜生成"
                ),
            )
        return
    try:
        capability.validate(narrative_s, shot_id=shot.id)
    except StoryTimingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _persist_execution_plan(shot: Shot, plan: ShotExecutionPlan) -> None:
    """把统一执行计划写入 continuity_profile JSON（复用现有字段，无迁移）。

    实测对白把故事时长延展到剪辑区间时同步 ``shot.duration``，保证数据库、
    视频 Prompt 与后期合成读到的时长一致；只在延展时调整，绝不悄悄缩短
    故事时长（超出 Provider 能力的镜头由入口校验拒绝）。
    """

    profile = _json_dict(shot.continuity_profile)
    profile["execution_plan"] = plan.to_dict()
    shot.continuity_profile = json.dumps(profile, ensure_ascii=False)
    effective_s = round(plan.effective_duration_ms / 1000, 3)
    if effective_s > 0 and round(float(shot.duration or 0), 3) < effective_s:
        shot.duration = effective_s


def _capability_kwargs(capability_mode: str, confirm_capability_downgrade: bool) -> dict[str, object]:
    """Normalize capability policy before passing it to generation services.

    ``confirm_degraded`` remains a separate readiness gate for storyboard assets;
    this helper only controls Provider capability downgrade behavior.
    """

    mode = str(capability_mode or "manual").strip().lower()
    if mode not in {"manual", "auto"}:
        mode = "manual"
    # Keep legacy internal adapters/callables compatible for the default policy.
    if mode == "manual" and not confirm_capability_downgrade:
        return {}
    return {
        "capability_mode": mode,
        "confirm_capability_downgrade": bool(confirm_capability_downgrade),
    }


def _prepare_storyboard_candidate(shot: Shot, db: Session, data: RegenerateRequest) -> tuple[int, str]:
    """登记一版故事板候选：版本快照 + 失效下游 + 应用本次参数。

    返回 (本次 expected_version, 生成理由)。多个候选共用同一段登记逻辑，
    因此第 2 个候选与第 1 个候选一样会进入版本历史，可对比后再选用。
    """
    task_key = _shot_task_key(shot.id, "storyboard")
    candidate_changes = data.model_dump(exclude_unset=True)
    estimated_speech_ms = _validate_timing_edit(shot, candidate_changes, db)
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
        shot.dialogue = _serialize_dialogue_input(data.dialogue, shot, db)
    if data.duration is not None:
        shot.duration = data.duration
    if data.dialogue is not None or data.duration is not None:
        shot.estimated_speech_ms = estimated_speech_ms
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


def prepare_storyboard_quality_retry(shot_id: str, review) -> bool:
    """按质量审核结果登记一轮故事板重试。

    审核修正指令合入 ``visual_notes``（替换上一轮的修正块，不叠加），随后走
    与人工重生成相同的登记口径：版本快照 + 失效下游 + 版本号 +1。返回 False
    表示无法登记（镜头不存在 / 已确认锁定 / 没有修正指令），调用方跳过重生成。
    """
    from services.quality_review_service import merge_quality_fix_notes

    directives = [str(item) for item in ((review.fix or {}).get("directives") or []) if str(item).strip()]
    if not directives:
        return False
    db = SessionLocal()
    try:
        shot = db.query(Shot).filter(Shot.id == shot_id).first()
        if shot is None or shot.confirmed:
            return False
        task_key = _shot_task_key(shot.id, "storyboard")
        create_version(db, shot, "quality_retry", task_id=task_key)
        previous_scene_key = _shot_scene_key(shot)
        _invalidate_storyboard_outputs(shot)
        _invalidate_downstream_media(db, shot, {previous_scene_key, _shot_scene_key(shot)}, source="regenerate")
        shot.status = "pending"
        shot.storyboard_status = "queued"
        shot.version = (shot.version or 1) + 1
        shot.visual_notes = merge_quality_fix_notes(shot.visual_notes or "", directives)
        _mark_project_output_stale(db, shot.project_id)
        db.commit()
        return True
    finally:
        db.close()


def prepare_video_quality_retry(shot_id: str, review) -> bool:
    """按视频质量审核结果登记一轮视频重试（只改 prompt 指令，故事板不动）。"""
    from services.quality_review_service import merge_quality_fix_notes

    directives = [str(item) for item in ((review.fix or {}).get("directives") or []) if str(item).strip()]
    if not directives:
        return False
    db = SessionLocal()
    try:
        shot = db.query(Shot).filter(Shot.id == shot_id).first()
        if shot is None:
            return False
        create_version(db, shot, "quality_retry", task_id=_shot_task_key(shot.id, "video"))
        shot.visual_notes = merge_quality_fix_notes(shot.visual_notes or "", directives)
        shot.version = (shot.version or 1) + 1
        db.commit()
        return True
    finally:
        db.close()


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

    gate = ensure_generation_gate(db, project_id, allow_degraded=bool(getattr(data, "confirm_degraded", False)), shot_ids=[shot.id for shot in shots])
    if gate.get("blocking"):
        if not bool(getattr(data, "confirm_degraded", False)):
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "一致性参考素材未就绪，故事板生成已阻止",
                    "requires_manual_review": True,
                    "consistency_report": gate,
                    "affected_shot_ids": gate.get("affected_shot_ids", []),
                    "shot_range": gate.get("shot_range", ""),
                },
            )
        # 手动模式只有在显式确认后才允许降级；每个未就绪项都留下 degraded 状态。
        from services.reference_readiness_service import mark_reference_degraded
        for item in gate.get("blocking_items", []):
            model = Character if item.get("kind") == "character" else SceneAsset
            row = db.query(model).filter(model.id == item.get("asset_id")).first()
            if row:
                mark_reference_degraded(db, item.get("kind"), row, reason="用户在故事板生成前确认降级")
        db.commit()
        gate = refresh_project_reference_state(db, project_id)
        if gate.get("blocking"):
            raise HTTPException(status_code=409, detail={"message": "参考素材降级确认后仍有阻断项", "consistency_report": gate})

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
                allow_degraded=True,
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
    _validate_timing_edit(shot, {}, db)
    _validate_shot_duration_for_provider(shot)

    # 启动前预检：视频端点必配；配音端点默认必配，但视频模型具备原生对白语音
    # 能力（或镜头无台词 / 镜头级显式指定 native）时不强制要求 TTS。
    ensure_providers_ready(
        "shot_video",
        has_dialogue=bool(_shot_dialogue_lines(shot)),
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


@router.get("/{shot_id}/video-candidates")
async def list_video_candidates(shot_id: str, db: Session = Depends(get_db)):
    """列出候选保存记录；失败候选与成功候选并列返回，不把失败藏进日志。"""

    if not db.query(Shot).filter(Shot.id == shot_id).first():
        raise HTTPException(status_code=404, detail="Shot not found")
    rows = (
        db.query(ShotVideoCandidate)
        .filter(ShotVideoCandidate.shot_id == shot_id)
        .order_by(ShotVideoCandidate.shot_version.desc(), ShotVideoCandidate.created_at)
        .all()
    )
    return {"shot_id": shot_id, "candidates": [_video_candidate_payload(row) for row in rows]}


async def _retry_failed_video_candidate(
    shot_id: str,
    candidate_id: str,
    expected_version: int,
) -> dict:
    """只补拍指定失败候选；旧失败行不修改，成功候选继续参与默认选择。"""

    return await _run_single_shot_video(
        shot_id,
        force=True,
        expected_version=int(expected_version),
        reuse_audio=True,
        candidate_count=1,
        strict_structural_selection=True,
        retry_of_candidate_id=str(candidate_id),
    )


@router.post("/{shot_id}/video-candidates/{candidate_id}/retry")
async def retry_video_candidate(shot_id: str, candidate_id: str, db: Session = Depends(get_db)):
    shot = db.query(Shot).filter(Shot.id == shot_id).first()
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    candidate = db.query(ShotVideoCandidate).filter(ShotVideoCandidate.candidate_id == candidate_id).first()
    if not candidate or candidate.shot_id != shot_id:
        raise HTTPException(status_code=404, detail="Video candidate not found")
    if candidate.status != VideoCandidateStatus.FAILED.value:
        raise HTTPException(status_code=409, detail="只有失败候选允许重试")
    if int(candidate.shot_version or 0) != int(shot.version or 1):
        raise HTTPException(status_code=409, detail="候选所属镜头版本已变化")
    expected_version = int(shot.version or 1)
    task_key = _shot_task_key(shot_id, f"video-candidate-retry:{candidate_id}")
    claim = claim_or_block(
        task_key,
        f"shot:{shot_id}",
        version=expected_version,
        current_step="video_candidate_retry",
        message=f"已排队重试视频候选 {candidate_id}",
    )
    if not claim.claimed:
        return {"shot_id": shot_id, "candidate_id": candidate_id, "status": "retrying", "deduplicated": True}
    try:
        task = start_task(
            task_key,
            _retry_failed_video_candidate(shot_id, candidate_id, expected_version),
        )
    except BaseException as exc:
        finish_task(task_key, "failed", f"video candidate retry scheduling failed: {exc}")
        raise
    _shot_video_tasks.add(task)
    task.add_done_callback(_shot_video_tasks.discard)
    return {"shot_id": shot_id, "candidate_id": candidate_id, "status": "retrying"}


@router.post("/{shot_id}/generate-audio")
async def generate_shot_audio(shot_id: str, data: ShotAudioGenerateRequest, db: Session = Depends(get_db)):
    """只生成镜头配音；独立于视频阶段，供选择性重生成队列使用。"""
    shot = db.query(Shot).filter(Shot.id == shot_id).first()
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    if shot.confirmed and not data.force:
        raise HTTPException(status_code=423, detail="已审核锁定的镜头禁止重新生成配音")
    if not _shot_dialogue_lines(shot):
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


def _ensure_reference_provider_capability(db: Session, project_id: str, *, allow_degraded: bool = False) -> dict:
    """参考图 Provider 不支持参考图时显式降级或阻止，绝不静默纯文本。"""

    adapter, endpoint = image_service._resolve_route()
    try:
        adapter, endpoint, _source = image_service._upgrade_to_reference_provider(adapter, endpoint)
    except Exception:
        pass
    if getattr(adapter.capabilities, "reference_images", False):
        return {}
    warning = (
        f"图像 Provider {endpoint.protocol}/{endpoint.model or 'default'} 不支持参考图，"
        "本次角色三视图/场景基准图不会发送给模型，已切换为纯文本生成。"
    )
    report = refresh_project_reference_state(db, project_id)
    for item in report.get("items", []):
        if not item.get("reference_images"):
            continue
        model = Character if item.get("kind") == "character" else SceneAsset
        row = db.query(model).filter(model.id == item.get("asset_id")).first()
        if not row:
            continue
        if allow_degraded:
            from services.reference_readiness_service import mark_reference_degraded

            mark_reference_degraded(db, item.get("kind"), row, reason="用户确认 Provider 纯文本降级", capability_warning=warning)
        else:
            from services.reference_readiness_service import mark_reference_unsupported

            mark_reference_unsupported(db, item.get("kind"), row, warning=warning)
    db.commit()
    report = refresh_project_reference_state(db, project_id)
    report["capability_warning"] = warning
    if not allow_degraded:
        raise RuntimeError(warning)
    return report


async def _run_storyboard_generation(
    project_id: str,
    shot_ids: list[str],
    expected_versions: dict[str, int] | None = None,
    *,
    allow_degraded: bool = False,
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
            allow_degraded=allow_degraded,
            capability_mode=capability_mode,
            confirm_capability_downgrade=confirm_capability_downgrade,
        )


class StoryboardVersionConflict(RuntimeError):
    """故事板任务与镜头版本不一致：过期结果丢弃，项目保持可编辑状态。

    版本冲突是用户编辑与后台任务的竞态，不是生成失败；恢复逻辑已在抛出前
    把项目状态还原，通用异常处理不得再覆盖成 error。
    """


async def _run_storyboard_generation_impl(
    project_id: str,
    shot_ids: list[str],
    expected_versions: dict[str, int],
    *,
    emit_project_result: bool = True,
    allow_degraded: bool = False,
    provider_override: str = "",
    preferred_size: str = "",
    capability_mode: str = "manual",
    confirm_capability_downgrade: bool = False,
    seed_override: int | None = None,
    recovery_revisions: list[dict] | None = None,
) -> None:
    """生成指定镜头；``emit_project_result=False`` 时作为 fan-out worker 使用。

    worker 模式只更新镜头和版本历史，不在单镜头失败时把整个项目标成 error，
    也不发布项目级完成事件；调用方负责 fan-in、局部恢复和最终状态。
    """
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
                # 只有 continuous_action 才自动读取上一镜 last_frame_path；
                # same_scene 仅继承场景/角色身份，其他模式不继承上一镜具体画面。
                previous_shot_model = _previous_shot_for_continuity(db, shot)
                previous_shot_data = _shot_dict(previous_shot_model) if previous_shot_model else None
                shot_data.update(
                    consistency_service.build_generation_context(
                        shot_data,
                        characters,
                        scenes,
                        previous_shot=previous_shot_data,
                        for_video=False,
                    )
                )
                shot_data["reference_manifest"] = build_manifest_for_shot(
                    db,
                    shot,
                    stage="storyboard",
                    continuity_profile=shot_data.get("continuity_profile") or {},
                    continuity_reference_path=shot_data.get("continuity_reference_path", ""),
                )
                shot_data["reference_versions"] = {
                    str(item.get("asset_id")): item.get("version")
                    for item in shot_data["reference_manifest"]
                }
                apply_agent_config_to_shot(shot_data, skill_config)
                _apply_recovery_revisions(shot_data, recovery_revisions, shot_id=shot.id, stage="image_generation")
                style_params = _storyboard_style_params(project, skill_config)
                seed = int(seed_override) if seed_override is not None else 42 + (shot.version or 1) * 100
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
                provider_override=provider_override,
                preferred_size=preferred_size,
                capability_mode=capability_mode,
                confirm_capability_downgrade=confirm_capability_downgrade,
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
                shot.continuity_reference_path = shot_data.get("continuity_reference_path", "")
                shot.storyboard_reference_manifest = json.dumps(shot_data.get("reference_manifest", []), ensure_ascii=False)
                shot.reference_capability_warning = str(image_meta.get("reference_capability_warning", ""))
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
                    raise StoryboardVersionConflict(f"故事板任务版本已变化: {', '.join(stale_or_missing)}")

            project = db.query(Project).filter(Project.id == project_id).first()
            if project and emit_project_result:
                project.status = "storyboard_ready"
                db.commit()
            project_style = (project.style if project else "anime") or "anime"
        finally:
            db.close()
        if not emit_project_result:
            return
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
            # 版本冲突已在上抛前还原项目状态：不覆盖成 error，也不把
            # 批次内其它已完成镜头标成 failed。
            version_conflict = isinstance(exc, StoryboardVersionConflict)
            if project and emit_project_result and not version_conflict:
                project.status = "error"
            queued = (
                db.query(Shot)
                .filter(Shot.project_id == project_id, Shot.id.in_(list(expected_versions)))
                .all()
            )
            for shot in queued:
                expected_version = expected_versions.get(shot.id)
                if expected_version is not None and (shot.version or 1) != expected_version:
                    continue
                if version_conflict:
                    continue
                shot.storyboard_status = "failed"
                if shot.status in {"pending", "storyboard_generating"}:
                    shot.status = "failed"
            db.commit()
        finally:
            db.close()
        if emit_project_result:
            await ws_manager.send_to_project(
                project_id,
                report_failure(
                    exc,
                    error_type=ERROR_STORYBOARD,
                    message=(
                        "故事板任务因镜头版本已变化而停止：检测到用户编辑，本次结果已丢弃，请重新发起生成。"
                        if version_conflict
                        else "定稿故事板生成失败，本次任务已停止。请检查镜头参数与模型配置后重试。"
                    ),
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
                raise RuntimeError("镜头版本已变化")
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
                raise RuntimeError("镜头版本已变化")
            project = db.query(Project).filter(Project.id == project_id).first()
            characters = _characters(db, project_id)
            scenes = _scenes(db, project_id)
            shot_data = _shot_dict(shot)
            shot_data["visual_notes"] = reason or _storyboard_notes(shot, scenes)
            if project:
                shot_data["output_format"] = project.output_format or "9:16"
            # 只有 continuous_action 才自动读取上一镜 last_frame_path。
            previous_shot_model = _previous_shot_for_continuity(db, shot)
            previous_shot_data = _shot_dict(previous_shot_model) if previous_shot_model else None
            shot_data.update(
                consistency_service.build_generation_context(
                    shot_data,
                    characters,
                    scenes,
                    previous_shot=previous_shot_data,
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
            capability_mode=capability_mode,
            confirm_capability_downgrade=confirm_capability_downgrade,
        )

        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if not shot or (expected_version is not None and (shot.version or 1) != expected_version):
                raise RuntimeError("镜头版本已变化")
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
            shot.continuity_reference_path = shot_data.get("continuity_reference_path", "")
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


async def _prepare_shot_audio(shot_id: str, expected_version: int) -> dict[str, object]:
    """准备当前镜头版本的音频，供独立音频阶段和一键视频入口复用。

    该步骤是幂等的：无对白或 native 路由直接跳过 TTS；当前版本已有有效音频
    时始终复用；只有缺少有效音频的 TTS 镜头才会调用外部配音服务。
    """
    db = SessionLocal()
    try:
        shot = db.query(Shot).filter(Shot.id == shot_id).first()
        if not shot or (shot.version or 1) != expected_version:
            raise RuntimeError("镜头版本已变化")
        project_id = shot.project_id
        skill_config = resolve_skill_config(project_id, db)
        lines = _shot_dialogue_lines(shot)
        characters = _characters(db, project_id)
        emotion = shot.emotion or "neutral"
        shot_data = _shot_dict(shot)
        audio_mode = resolve_audio_mode(shot_data)
        native_routed = audio_mode == "native"
        existing_audio = _reusable_audio_path(shot_id, expected_version, shot.audio_path)
    finally:
        db.close()

    # native 音频由视频模型负责；没有对白的镜头也不应触发外部 TTS。
    if native_routed or not lines:
        return {
            "project_id": project_id,
            "characters": characters,
            "skill_config": skill_config,
            "dialogue_lines": lines,
            "timed_lines": [],
            "audio_path": "",
            "native_routed": native_routed,
            "reused": False,
            "skipped": True,
        }

    if existing_audio:
        return {
            "project_id": project_id,
            "characters": characters,
            "skill_config": skill_config,
            "dialogue_lines": lines,
            "timed_lines": [],
            "audio_path": existing_audio,
            "native_routed": False,
            "reused": True,
            "skipped": False,
        }

    audio_path, timed_lines = await generate_dialogue_track(
        lines,
        characters=characters,
        project_id=project_id,
        media_id=_versioned_media_id(shot_id, expected_version),
        default_emotion=emotion,
        text_cleaner=lambda text: clean_tts_text(text, skill_config),
    )
    db = SessionLocal()
    try:
        shot = db.query(Shot).filter(Shot.id == shot_id).first()
        if not shot or (shot.version or 1) != expected_version:
            raise RuntimeError("镜头版本已变化")
        shot.audio_path = audio_path
        shot.dialogue = serialize_dialogue_lines(timed_lines)
        # TTS 实测时间轴落库的同时刷新统一执行计划（先于视频生成），视频
        # Prompt 与后期合成后续都消费这一份计划。
        _persist_execution_plan(
            shot,
            ShotExecutionPlan.derive(
                shot_data,
                provider=provider_duration_capability(),
                audio_mode="tts",
                dialogue_timing=[line.as_dict() for line in timed_lines],
                dialogue_timing_source="tts_measured",
            ),
        )
        shot.status = "video_done" if shot.video_path else ("storyboard_approved" if shot.confirmed else "storyboard_done")
        create_version(db, shot, "regenerate", task_id=f"shot:{shot_id}:audio")
        db.commit()
    finally:
        db.close()
    return {
        "project_id": project_id,
        "characters": characters,
        "skill_config": skill_config,
        "dialogue_lines": lines,
        "timed_lines": timed_lines,
        "audio_path": audio_path,
        "native_routed": False,
        "reused": False,
        "skipped": False,
    }


async def _run_single_shot_audio(shot_id: str, expected_version: int) -> None:
    """独立音频 worker；准备步骤本身负责版本校验、复用和跳过策略。"""
    lock = _shot_generation_locks.setdefault(shot_id, asyncio.Lock())
    if lock.locked():
        raise RuntimeError("镜头音频生成任务已在运行")
    await lock.acquire()
    project_id = ""
    try:
        result = await _prepare_shot_audio(shot_id, expected_version)
        project_id = str(result.get("project_id") or "")
        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if shot and (shot.version or 1) == expected_version:
                await ws_manager.send_to_project(project_id, _shot_update_payload(shot))
        finally:
            db.close()
    except Exception:
        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if shot and (shot.version or 1) == expected_version:
                shot.status = "failed"
                db.commit()
        finally:
            db.close()
        raise
    finally:
        lock.release()


async def _generate_shot_video(
    shot_data: dict,
    characters: list[dict],
    scenes: list[dict],
    project_id: str,
    *,
    provider_override: str = "",
    resolution_override: str = "",
    capability_mode: str = "manual",
    confirm_capability_downgrade: bool = False,
) -> dict:
    """调用视频 Provider 的纯视频步骤；音频准备由 ``_prepare_shot_audio`` 完成。"""
    video_options = {}
    if provider_override:
        video_options["provider_override"] = provider_override
    if resolution_override:
        video_options["resolution_override"] = resolution_override
    video_options.update(_capability_kwargs(capability_mode, confirm_capability_downgrade))
    return await seedance_service.generate_shot_video(
        shot_data,
        characters,
        scenes,
        project_id,
        **video_options,
    )


def _video_candidate_media_id(
    media_id: str,
    candidate_index: int,
    retry_of_candidate_id: str = "",
    batch_id: str = "",
) -> str:
    """为候选/重试生成不会覆盖旧文件的稳定媒体名。"""

    batch_suffix = f"_b{hashlib.sha256(str(batch_id).encode('utf-8')).hexdigest()[:8]}" if batch_id else ""
    if retry_of_candidate_id:
        retry_suffix = hashlib.sha256(str(retry_of_candidate_id).encode("utf-8")).hexdigest()[:8]
        stem = f"{media_id}_retry_{retry_suffix}{batch_suffix}_c{int(candidate_index)}"
    else:
        stem = f"{media_id}{batch_suffix}_c{int(candidate_index)}"
    if len(stem) <= 128:
        return stem
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:16]
    return f"{stem[:105]}_{digest}"[:128]


def _apply_recovery_revisions(shot_data: dict, revisions: list[dict] | None, *, shot_id: str, stage: str) -> None:
    """Apply only explicit, stage/shot-scoped patches to the actual generation input."""
    for revision in revisions or []:
        if not isinstance(revision, dict) or str(revision.get("shot_id") or "") not in {"", str(shot_id)}:
            continue
        for raw in revision.get("patches") or []:
            if not isinstance(raw, dict) or str(raw.get("target_stage") or stage) != stage:
                continue
            if str(raw.get("shot_id") or "") not in {"", str(shot_id)}:
                continue
            field = str(raw.get("field") or "")
            op = str(raw.get("op") or "set")
            value = raw.get("value")
            if field in {"visual_prompt", "storyboard_prompt", "visual_notes"}:
                if isinstance(value, dict) and isinstance(value.get("rule"), str):
                    value = value["rule"]
                if isinstance(value, str) and value.strip():
                    existing = str(shot_data.get(field) or "")
                    shot_data[field] = value.strip() if op == "replace" else f"{existing}\\n{value.strip()}".strip()
                    # Both providers consume these fields through their prompt builders.
                    if field == "visual_prompt":
                        shot_data["visual_notes"] = f"{shot_data.get('visual_notes', '')}\\n{value.strip()}".strip()
                        shot_data["storyboard_prompt"] = f"{shot_data.get('storyboard_prompt', '')}\\n{value.strip()}".strip()
            elif field == "negative_prompt" and isinstance(value, str):
                existing = str(shot_data.get(field) or "")
                shot_data[field] = value if op == "replace" else f"{existing}, {value}".strip(", ")
            elif field == "reference_images" and op == "replace":
                _replace_recovery_reference(shot_data, stage=stage, source=value if isinstance(value, dict) else {})
            elif field == "resolution" and isinstance(value, str):
                shot_data["resolution"] = value


def _replace_recovery_reference(shot_data: dict, *, stage: str, source: dict) -> None:
    """Choose a real scene baseline/continuity frame; never invent a reference path."""
    requested = str(source.get("source") or "scene_baseline_or_previous_tail_frame")
    manifest = list(shot_data.get("reference_manifest") or [])
    valid = [
        item for item in manifest
        if isinstance(item, dict)
        and str(item.get("path") or "")
        and Path(str(item.get("path"))).is_file()
        and str(item.get("status") or "ready") not in {"failed", "stale", "unsupported"}
    ]
    priorities = ("scene_baseline", "continuity_frame", "previous_last_frame", "approved_storyboard_first_frame")
    chosen = next((item for kind in priorities for item in valid if str(item.get("type") or "") == kind), None)
    if chosen is None:
        shot_data["recovery_reference"] = {"requested": requested, "applied": False, "reason": "no_valid_reference"}
        return
    if stage == "video_generation":
        # Video generation always retains the approved storyboard first frame as its required anchor.
        first = next((item for item in valid if item.get("type") == "approved_storyboard_first_frame"), None)
        selected = [first] if first and first is not chosen else []
        selected.append(chosen)
    else:
        selected = [chosen]
    selected = list({str(item.get("path")): item for item in selected}.values())
    shot_data["reference_manifest"] = selected
    shot_data["reference_assets"] = [
        {"type": item.get("type", "reference_image"), "role": item.get("name") or item.get("type", "reference_image"), "path": item["path"], "version": item.get("version", "")}
        for item in selected
    ]
    shot_data["scene_reference_images"] = [item["path"] for item in selected if item.get("type") == "scene_baseline"]
    shot_data["character_reference_images"] = []
    continuity = next((item["path"] for item in selected if item.get("type") in {"continuity_frame", "previous_last_frame"}), "")
    if continuity:
        shot_data["continuity_reference_path"] = continuity
    shot_data["recovery_reference"] = {"requested": requested, "applied": True, "path": chosen["path"], "type": chosen.get("type", "")}


def _candidate_seed(media_id: str, candidate_index: int, retry_of_candidate_id: str = "") -> int:
    """为每个候选生成可追溯的稳定 seed；Provider 不支持 seed 时仍记录 recipe seed。"""

    payload = f"{media_id}:{candidate_index}:{retry_of_candidate_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31 - 1)


def _candidate_recipe_hash(
    *,
    provider: str,
    model: str,
    seed: int,
    execution_plan_hash: str,
    reference_manifest: list,
    prompt: str,
) -> str:
    payload = {
        "provider": str(provider or ""),
        "model": str(model or ""),
        "seed": int(seed),
        "execution_plan_hash": str(execution_plan_hash or ""),
        "reference_manifest": reference_manifest,
        "prompt": str(prompt or ""),
    }
    return _execution_plan_hash(payload)


def _candidate_decision_trace(
    *,
    project_id: str,
    shot_id: str,
    expected_version: int,
    batch_id: str,
    candidates: list[dict],
    selection: VideoCandidateSelection,
) -> dict:
    return DecisionTrace(
        trace_id=f"trace:video_candidate:{shot_id}:{batch_id}",
        project_id=project_id,
        shot_id=shot_id,
        shot_version=int(expected_version),
        run_id=f"shot:{shot_id}:video",
        stage=StageName.VIDEO_GENERATION,
        mode="auto",
        reason=selection.reason or "candidate_selection",
        selected_video_candidate_id=str(selection.candidate_id or ""),
        video_candidates=[dict(item) for item in candidates],
        candidate_selection=selection.model_dump(mode="json"),
    ).model_dump(mode="json")


def _new_video_candidate_id(media_id: str, candidate_index: int, batch_id: str, retry_of_candidate_id: str = "") -> str:
    stem = _video_candidate_media_id(media_id, candidate_index, retry_of_candidate_id)
    return f"{stem}_{batch_id}_{int(candidate_index)}"


def _output_format_ratio(value: str | None) -> float | None:
    """把 ``9:16`` 这类画幅字符串解析成宽高比；无法解析返回 None。"""
    width, _, height = str(value or "").partition(":")
    try:
        ratio = float(width) / float(height)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return ratio if ratio > 0 else None


def _execution_plan_hash(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _video_candidate_model(data: dict) -> ShotVideoCandidate:
    failure = _json_dict(data.get("failure"))
    if not failure and (data.get("failure_kind") or data.get("failure_message")):
        failure = {"kind": data.get("failure_kind"), "message": data.get("failure_message")}
    metrics = _json_dict(data.get("metrics") or data.get("structural_metrics"))
    path = str(data.get("path") or data.get("video_path") or "")
    last_frame_path = str(data.get("last_frame_path") or data.get("tail_frame_path") or "")
    recipe_hash = str(data.get("recipe_hash") or data.get("execution_plan_hash") or "")
    seed_raw = data.get("seed")
    try:
        seed = int(seed_raw) if seed_raw is not None and str(seed_raw) != "" else None
    except (TypeError, ValueError):
        seed = None
    return ShotVideoCandidate(
        candidate_id=str(data["candidate_id"]),
        shot_id=str(data["shot_id"]),
        project_id=str(data.get("project_id") or ""),
        shot_version=int(data.get("shot_version") or 1),
        batch_id=str(data.get("batch_id") or ""),
        candidate_index=int(data.get("candidate_index") or 1),
        status=str(data.get("status") or VideoCandidateStatus.PENDING.value),
        path=path,
        last_frame_path=last_frame_path,
        provider=str(data.get("provider") or ""),
        model=str(data.get("model") or ""),
        seed=seed,
        recipe_hash=recipe_hash,
        reference_manifest=json.dumps(data.get("reference_manifest") or [], ensure_ascii=False),
        generation_duration_ms=int(data.get("generation_duration_ms") or 0),
        score=float(data.get("score") or 0.0),
        metrics=json.dumps(metrics, ensure_ascii=False),
        failure=json.dumps(failure, ensure_ascii=False),
        video_path=path,
        tail_frame_path=last_frame_path,
        execution_plan_hash=recipe_hash,
        structural_passed=data.get("structural_passed"),
        structural_metrics=json.dumps(metrics, ensure_ascii=False),
        failure_kind=str(failure.get("kind") or data.get("failure_kind") or ""),
        failure_message=str(failure.get("message") or data.get("failure_message") or ""),
        retry_of_candidate_id=str(data.get("retry_of_candidate_id") or ""),
        selected=bool(data.get("selected") or False),
        selection_reason=str(data.get("selection_reason") or ""),
    )


def _video_candidate_payload(row: ShotVideoCandidate) -> dict:
    failure = _json_dict(row.failure)
    if not failure and (row.failure_kind or row.failure_message):
        failure = {"kind": row.failure_kind, "message": row.failure_message, "stage": "video_generation", "shot_id": row.shot_id}
    failure = failure or None
    metrics = _json_dict(row.metrics) or _json_dict(row.structural_metrics)
    path = row.path or row.video_path or ""
    last_frame_path = row.last_frame_path or row.tail_frame_path or ""
    return {
        "candidate_id": row.candidate_id,
        "shot_id": row.shot_id,
        "project_id": row.project_id or "",
        "shot_version": int(row.shot_version or 1),
        "batch_id": row.batch_id or "",
        "candidate_index": int(row.candidate_index or 1),
        "status": row.status,
        "path": path,
        "last_frame_path": last_frame_path,
        "provider": row.provider or "",
        "model": row.model or "",
        "seed": row.seed,
        "recipe_hash": row.recipe_hash or row.execution_plan_hash or "",
        "reference_manifest": _json_list(row.reference_manifest),
        "generation_duration_ms": int(row.generation_duration_ms or 0),
        "score": float(row.score or 0.0),
        "metrics": metrics,
        "failure": failure,
        "video_path": path,
        "tail_frame_path": last_frame_path,
        "execution_plan_hash": row.recipe_hash or row.execution_plan_hash or "",
        "structural_passed": row.structural_passed,
        "structural_metrics": metrics,
        "retry_of_candidate_id": row.retry_of_candidate_id or "",
        "selected": bool(row.selected),
        "selection_reason": row.selection_reason or "",
    }


def _save_video_candidate(data: dict) -> dict:
    row = _video_candidate_model(data)
    db = SessionLocal()
    try:
        merged = db.merge(row)
        db.commit()
        db.refresh(merged)
        return _video_candidate_payload(merged)
    finally:
        db.close()


def _shot_video_candidates(
    shot_id: str,
    *,
    shot_version: int | None = None,
    statuses: set[str] | None = None,
) -> list[dict]:
    db = SessionLocal()
    try:
        query = db.query(ShotVideoCandidate).filter(ShotVideoCandidate.shot_id == str(shot_id))
        if shot_version is not None:
            query = query.filter(ShotVideoCandidate.shot_version == int(shot_version))
        if statuses is not None:
            query = query.filter(ShotVideoCandidate.status.in_(sorted(statuses)))
        rows = query.order_by(ShotVideoCandidate.candidate_index, ShotVideoCandidate.created_at).all()
        return [_video_candidate_payload(row) for row in rows]
    finally:
        db.close()


def _persist_video_candidate_selection(
    db: Session,
    shot: Shot,
    selected: VideoCandidateSelection,
    *,
    expected_version: int,
) -> ShotVideoCandidate:
    if not selected.candidate_id:
        raise RuntimeError(selected.reason or "没有可发布的视频候选")
    if int(shot.version or 1) != int(expected_version):
        raise RuntimeError("镜头版本已变化")
    rows = (
        db.query(ShotVideoCandidate)
        .filter(ShotVideoCandidate.shot_id == shot.id, ShotVideoCandidate.shot_version == int(expected_version))
        .all()
    )
    matched = None
    for row in rows:
        is_selected = row.candidate_id == selected.candidate_id
        if is_selected:
            if row.status != VideoCandidateStatus.SUCCEEDED.value:
                raise RuntimeError("只能选择成功的视频候选")
            if row.structural_passed is False:
                raise RuntimeError("结构检查失败的候选不能被选择")
            matched = row
        row.selected = is_selected
        row.selection_reason = selected.reason if is_selected else ""
        row.selected_at = None
    if matched is None:
        raise RuntimeError("视频候选不存在或版本不匹配")
    matched.selected_at = datetime.utcnow()
    return matched

async def _run_single_shot_video(
    shot_id: str,
    force: bool = False,
    expected_version: int | None = None,
    reuse_audio: bool = False,
    provider_override: str = "",
    resolution_override: str = "",
    *,
    capability_mode: str = "manual",
    confirm_capability_downgrade: bool = False,
    candidate_count: int = 1,
    recovery_budget: int = 1,
    strict_structural_selection: bool = False,
    retry_of_candidate_id: str = "",
    seed_override: int | None = None,
    recovery_revisions: list[dict] | None = None,
) -> dict:
    candidate_count = max(1, min(3, int(candidate_count or 1)))
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
                return {"skipped": True, "reason": "shot_not_found", "video_candidates": []}
            if expected_version is None:
                expected_version = shot.version or 1
            elif (shot.version or 1) != expected_version:
                raise RuntimeError("镜头版本已变化")
            if _can_reuse_existing_video(shot, force):
                return {"skipped": True, "reason": "existing_video_reused", "video_candidates": []}
            video_gate = ensure_generation_gate(db, shot.project_id, allow_degraded=True, shot_ids=[shot.id])
            if video_gate.get("blocking"):
                raise RuntimeError(
                    "一致性参考素材未就绪，视频生成已阻止："
                    + ", ".join(item.get("name") or item.get("asset_id") for item in video_gate.get("blocking_items", []))
                )
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
                raise RuntimeError("镜头版本已变化")
            project = db.query(Project).filter(Project.id == project_id).first()
            characters = _characters(db, project_id)
            scenes = _scenes(db, project_id)
            shot_data = _shot_dict(shot)
            shot_data["storyboard_prompt"] = _storyboard_notes(shot, scenes)
            if project:
                shot_data["output_format"] = project.output_format or "9:16"
                shot_data["resolution"] = project.resolution or "720p"
                shot_data["style"] = project.style or "anime"
            previous_shot_model = _previous_shot_for_continuity(db, shot)
            previous_shot_data = _shot_dict(previous_shot_model) if previous_shot_model else None
            shot_data.update(
                consistency_service.build_generation_context(
                    shot_data,
                    characters,
                    scenes,
                    previous_shot=previous_shot_data,
                    for_video=True,
                )
            )
            previous_reference = shot_data.get("continuity_reference_path", "")
            shot_data["reference_manifest"] = build_manifest_for_shot(
                db,
                shot,
                stage="video",
                continuity_profile=shot_data.get("continuity_profile") or {},
                continuity_reference_path=previous_reference,
            )
            shot_data["reference_versions"] = {
                str(item.get("asset_id")): item.get("version")
                for item in shot_data["reference_manifest"]
            }
            apply_agent_config_to_shot(shot_data, skill_config)
            _apply_recovery_revisions(shot_data, recovery_revisions, shot_id=shot.id, stage="video_generation")
            # 镜头级 audio_mode 覆盖存于 continuity_profile，但一致性上下文会重建
            # profile，这里从数据库存档提升为 shot 顶级字段，保证覆盖不被冲掉。
            stored_audio_mode = _json_dict(shot.continuity_profile).get("audio_mode")
            if stored_audio_mode:
                shot_data["audio_mode"] = str(stored_audio_mode).strip().lower()
            shot_sequence = shot.sequence
            dialogue_lines = _shot_dialogue_lines(shot)
            emotion = shot.emotion or "neutral"
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
        prepared_audio = await _prepare_shot_audio(shot_id, expected_version)
        audio_path = str(prepared_audio.get("audio_path") or "")
        native_routed = bool(prepared_audio.get("native_routed"))
        timed_lines = list(prepared_audio.get("timed_lines") or [])
        shot_data["audio_path"] = audio_path
        if dialogue_lines and native_routed:
            # 原生音频由视频模型负责；对白随 prompt 传入，不调用独立 TTS。
            dialogues = [
                Dialogue(
                    role=line.speaker or "角色",
                    text=clean_tts_text(line.line, skill_config),
                    emotion=line.emotion or emotion,
                    start_ms=int(line.start_ms or 0),
                    end_ms=int(line.end_ms or line.start_ms or 0),
                )
                for line in dialogue_lines
            ]
        else:
            dialogues = None
            if prepared_audio.get("reused"):
                await _progress(
                    project_id,
                    "generate_voice",
                    84,
                    f"镜头 {shot_sequence} 已有有效配音，复用后继续生成视频",
                    job_keys=(f"shot:{shot_id}:video",),
                )

        # 统一执行计划先行落库，再进入视频生成：TTS 实测时间轴（或复用配音的
        # 库内时间轴 / native prompt 时间）与 Provider 生成时长在此收敛成一份
        # 裁剪区间，视频 Prompt 与后期合成都消费它，不再各自推导。候选数与
        # 恢复预算来自调用方质量档位，一并写入计划供恢复决策追踪。
        if native_routed and dialogues:
            timing_payload = [
                {"speaker": item.role, "text": item.text, "start_ms": item.start_ms, "end_ms": item.end_ms}
                for item in dialogues
            ]
            timing_source = "native_prompt"
        elif timed_lines:
            timing_payload = [line.as_dict() for line in timed_lines]
            timing_source = "tts_measured"
        else:
            timing_payload = None
            timing_source = ""
        shot_data["candidate_count"] = candidate_count
        shot_data["recovery_budget"] = max(0, int(recovery_budget))
        execution_plan = ShotExecutionPlan.derive(
            shot_data,
            provider=provider_duration_capability(),
            audio_mode="native" if native_routed else "tts",
            dialogue_timing=timing_payload,
            dialogue_timing_source=timing_source,
            candidate_count=candidate_count,
            recovery_budget=max(0, int(recovery_budget)),
        )
        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if not shot or (expected_version is not None and (shot.version or 1) != expected_version):
                raise RuntimeError("镜头版本已变化")
            _persist_execution_plan(shot, execution_plan)
            db.commit()
        finally:
            db.close()
        shot_data["execution_plan"] = execution_plan.to_dict()
        shot_data["duration"] = round(execution_plan.effective_duration_ms / 1000, 3)

        await _progress(
            project_id,
            "generate_seedance_video",
            90,
            f"正在生成镜头 {shot_sequence} 的视频",
            job_keys=(f"shot:{shot_id}:video",),
        )
        batch_id = uuid.uuid4().hex[:12]
        execution_plan_hash = _execution_plan_hash(execution_plan.to_dict())
        # 候选验证上下文：执行计划时长、目标画幅、实测配音时长。缺失的维度
        # 在校验器里记 skipped，不猜测也不冒充通过。
        expected_aspect_ratio = _output_format_ratio(str(shot_data.get("output_format") or ""))
        audio_duration_s = await probe_media_duration(audio_path) if audio_path else None
        candidate_rows: list[dict] = []
        candidate_results: dict[str, dict] = {}
        candidate_specs: list[dict] = []
        for candidate_index in range(1, candidate_count + 1):
            candidate_id = _new_video_candidate_id(media_id, candidate_index, batch_id, retry_of_candidate_id)
            candidate_media_id = _video_candidate_media_id(media_id, candidate_index, retry_of_candidate_id, batch_id)
            if seed_override is not None:
                seed_payload = f"{int(seed_override)}:{candidate_index}:{batch_id}:{retry_of_candidate_id}".encode("utf-8")
                seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:4], "big") % (2**31 - 1)
            else:
                seed = _candidate_seed(media_id, candidate_index, retry_of_candidate_id)
            reference_manifest = list(shot_data.get("reference_manifest") or [])
            recipe_hash = _candidate_recipe_hash(
                provider=provider_override or get_endpoint("video").protocol,
                model=get_endpoint("video").model,
                seed=seed,
                execution_plan_hash=execution_plan_hash,
                reference_manifest=reference_manifest,
                prompt=str(shot_data.get("visual_notes") or shot_data.get("storyboard_prompt") or ""),
            )
            spec = {
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "candidate_media_id": candidate_media_id,
                "seed": seed,
                "recipe_hash": recipe_hash,
                "reference_manifest": reference_manifest,
            }
            candidate_specs.append(spec)
            _save_video_candidate({
                "candidate_id": candidate_id,
                "shot_id": shot_id,
                "project_id": project_id,
                "shot_version": int(expected_version),
                "batch_id": batch_id,
                "candidate_index": candidate_index,
                "status": VideoCandidateStatus.PENDING.value,
                "path": "",
                "last_frame_path": "",
                "provider": provider_override or get_endpoint("video").protocol,
                "model": get_endpoint("video").model,
                "seed": seed,
                "recipe_hash": recipe_hash,
                "reference_manifest": reference_manifest,
                "generation_duration_ms": 0,
                "score": 0.0,
                "metrics": {},
                "failure": {},
                "retry_of_candidate_id": retry_of_candidate_id,
            })

        async def generate_candidate(spec: dict) -> dict:
            candidate_id = str(spec["candidate_id"])
            candidate_index = int(spec["candidate_index"])
            candidate_media_id = str(spec["candidate_media_id"])
            candidate_started = time.monotonic()
            endpoint = get_endpoint("video")
            candidate_shot_data = copy.deepcopy(shot_data)
            candidate_shot_data.update({"shot_id": candidate_media_id, "dialogues": dialogues, "seed": spec["seed"]})
            try:
                _save_video_candidate({
                    "candidate_id": candidate_id,
                    "shot_id": shot_id,
                    "project_id": project_id,
                    "shot_version": int(expected_version),
                    "batch_id": batch_id,
                    "candidate_index": candidate_index,
                    "status": VideoCandidateStatus.RUNNING.value,
                    "path": "",
                    "last_frame_path": "",
                    "provider": provider_override or endpoint.protocol,
                    "model": endpoint.model,
                    "seed": spec["seed"],
                    "recipe_hash": spec["recipe_hash"],
                    "reference_manifest": spec["reference_manifest"],
                    "retry_of_candidate_id": retry_of_candidate_id,
                })
                candidate_result = await _generate_shot_video(
                    candidate_shot_data,
                    characters,
                    scenes,
                    project_id,
                    provider_override=provider_override,
                    resolution_override=resolution_override,
                    capability_mode=capability_mode,
                    confirm_capability_downgrade=confirm_capability_downgrade,
                )
                if native_routed and not candidate_result.get("native_audio"):
                    raise RuntimeError("视频适配器未按原生音频模式返回带音轨视频，已阻止无声成品")
                reference_manifest = list(
                    candidate_result.get("reference_manifest")
                    or candidate_shot_data.get("reference_manifest")
                    or spec["reference_manifest"]
                    or []
                )
                video_report = dict(candidate_result.get("generation_report") or {})
                structural = await validate_video_file(
                    str(candidate_result.get("video_path") or ""),
                    expected_duration_s=float(execution_plan.provider_generation_duration_s),
                    expected_aspect_ratio=expected_aspect_ratio,
                    audio_duration_s=audio_duration_s,
                    tail_frame_path=str(candidate_result.get("frame_path") or "") or None,
                )
                structural_passed = bool(structural.get("passed"))
                # 测试/兼容 Provider 若不落可探测媒体，保留生成成功记录但不获得结构分；
                # 严格自动候选只选择 structural_passed=True 的行。
                if not Path(str(candidate_result.get("video_path") or "")).exists():
                    structural_passed = None
                candidate_data = {
                    "candidate_id": candidate_id,
                    "shot_id": shot_id,
                    "project_id": project_id,
                    "shot_version": int(expected_version),
                    "batch_id": batch_id,
                    "candidate_index": candidate_index,
                    "status": VideoCandidateStatus.SUCCEEDED.value,
                    "path": str(candidate_result.get("video_path") or ""),
                    "last_frame_path": str(candidate_result.get("frame_path") or ""),
                    "provider": str(video_report.get("provider") or provider_override or endpoint.protocol),
                    "model": str(video_report.get("model") or endpoint.model),
                    "seed": spec["seed"],
                    "recipe_hash": spec["recipe_hash"],
                    "reference_manifest": reference_manifest,
                    "generation_duration_ms": int((time.monotonic() - candidate_started) * 1000),
                    "metrics": structural,
                    "structural_passed": structural_passed,
                    "retry_of_candidate_id": retry_of_candidate_id,
                }
                candidate_data["score"] = score_video_candidate({**candidate_data, "score": 0.0})
                saved = _save_video_candidate(candidate_data)
                return {
                    "candidate_index": candidate_index,
                    "row": saved,
                    "result": {
                        **candidate_result,
                        "reference_manifest": reference_manifest,
                        "generation_report": video_report,
                        "candidate_shot_data": candidate_shot_data,
                    },
                }
            except Exception as candidate_exc:
                failure = {
                    "kind": "video_failed",
                    "stage": "video_generation",
                    "shot_id": shot_id,
                    "message": str(candidate_exc),
                    "retryable": True,
                    "details": {"candidate_index": candidate_index, "candidate_id": candidate_id},
                }
                saved = _save_video_candidate({
                    "candidate_id": candidate_id,
                    "shot_id": shot_id,
                    "project_id": project_id,
                    "shot_version": int(expected_version),
                    "batch_id": batch_id,
                    "candidate_index": candidate_index,
                    "status": VideoCandidateStatus.FAILED.value,
                    "path": "",
                    "last_frame_path": "",
                    "provider": provider_override or endpoint.protocol,
                    "model": endpoint.model,
                    "seed": spec["seed"],
                    "recipe_hash": spec["recipe_hash"],
                    "reference_manifest": list(spec["reference_manifest"] or []),
                    "generation_duration_ms": int((time.monotonic() - candidate_started) * 1000),
                    "score": 0.0,
                    "metrics": {"passed": False, "issues": [str(candidate_exc)]},
                    "failure": failure,
                    "retry_of_candidate_id": retry_of_candidate_id,
                })
                return {"candidate_index": candidate_index, "row": saved, "result": {}}

        # 候选之间互不等待：一个 Provider 失败只生成自己的 failed 记录，
        # 其它候选继续完成，最后统一评分与选择。
        generated = await asyncio.gather(*(generate_candidate(spec) for spec in candidate_specs))
        for item in sorted(generated, key=lambda value: int(value["candidate_index"])):
            saved = item["row"]
            candidate_rows.append(saved)
            if item.get("result"):
                candidate_results[str(saved["candidate_id"])] = item["result"]

        # 重试时把同镜头版本的历史成功候选一并纳入，避免新候选直接挤掉更好的旧结果。
        selection_pool = (
            _shot_video_candidates(shot_id, shot_version=int(expected_version))
            if retry_of_candidate_id
            else [item for item in candidate_rows if item.get("batch_id") == batch_id]
        )
        selection_pool = [
            {**item, "score": score_video_candidate(item)}
            for item in selection_pool
        ]
        selection = select_video_candidate(
            selection_pool,
            require_structural=strict_structural_selection,
            allow_single_candidate_fallback=(not strict_structural_selection and candidate_count == 1),
        )
        decision_trace = _candidate_decision_trace(
            project_id=project_id,
            shot_id=shot_id,
            expected_version=int(expected_version),
            batch_id=batch_id,
            candidates=selection_pool,
            selection=selection,
        )
        CheckpointStore.get(project_id, f"shot:{shot_id}:video").add_decision(decision_trace)
        if not selection.candidate_id:
            raise RuntimeError(selection.reason or "没有可发布的视频候选")
        selected_candidate_id = str(selection.candidate_id)
        result = candidate_results.get(selected_candidate_id, {})
        if not result:
            persisted = next(item for item in selection_pool if item.get("candidate_id") == selected_candidate_id)
            result = {
                "video_path": persisted.get("video_path", ""),
                "frame_path": persisted.get("tail_frame_path", ""),
                "reference_manifest": persisted.get("reference_manifest", []),
                "generation_report": {
                    "provider": persisted.get("provider", ""),
                    "model": persisted.get("model", ""),
                },
                "candidate_shot_data": dict(shot_data),
            }
        video_shot_data = result.get("candidate_shot_data") or {**shot_data, "shot_id": media_id, "dialogues": dialogues}
        if result.get("reference_manifest"):
            # 请求级 manifest 含已审核首帧、实际候选参考和 continuity 决策/缺帧原因。
            shot_data["reference_manifest"] = list(result["reference_manifest"])
        continuity_profile = shot_data.get("continuity_profile", {}) or {}
        video_report = dict(result.get("generation_report") or {})
        if video_report:
            continuity_profile.update(
                {
                    "provider": video_report.get("provider", ""),
                    "model": video_report.get("model", ""),
                    "provider_source": video_report.get("provider_source", ""),
                    "reference_mode": video_report.get("reference_mode", ""),
                    "references_validated": video_report.get("references_validated", 0),
                    "references_sent": video_report.get("references_sent", []),
                    "references_sent_detail": video_report.get("references_sent_detail", []),
                    "control_types_sent": video_report.get("control_types_sent", []),
                    "provider_capabilities": video_report.get("provider_capabilities", {}),
                    "reference_weight_policy": video_report.get("reference_weight_policy", "text_only_policy"),
                    "consistency_metrics": video_report.get("consistency_metrics", {}),
                    "generation_report": video_report,
                }
            )
        continuity_profile["reference_capability_warning"] = result.get("reference_capability_warning", "")
        continuity_profile["reference_manifest"] = result.get("reference_manifest", shot_data.get("reference_manifest", []))
        if result.get("reference_payload_mode"):
            continuity_profile["seedance_reference_payload_mode"] = result["reference_payload_mode"]
            continuity_profile.setdefault("reference_mode", "first_frame_only")
            continuity_profile.setdefault("references_validated", len(video_shot_data.get("seedance_reference_manifest") or []))
            continuity_profile.setdefault(
                "references_sent",
                ["approved_storyboard_first_frame"] if result["reference_payload_mode"] == "first_frame_reference" else [],
            )
            continuity_profile.setdefault("provider", get_endpoint("video").protocol)
            continuity_profile.setdefault("model", get_endpoint("video").model)
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
                raise RuntimeError("镜头版本已变化")
            selected_candidate_row = _persist_video_candidate_selection(
                db,
                shot,
                selection,
                expected_version=int(expected_version),
            )
            shot.scene_group_id = shot_data.get("scene_group_id", shot.scene_group_id)
            shot.consistency_context = shot_data.get("consistency_context", shot.consistency_context)
            shot.reference_weights = json.dumps(shot_data.get("reference_weights", {}), ensure_ascii=False)
            # 执行计划随本次生成结果一起写回存档；后期合成直接读取这一份计划。
            # 生成路由在候选上下文里按可加载素材覆写过能力清单与 video_mode，
            # 合并进最终计划（对白实测时间轴与 shot_id 仍以路由级计划为准）。
            final_profile = dict(shot_data.get("continuity_profile") or {})
            final_plan = dict(execution_plan.to_dict())
            candidate_plan = dict((video_shot_data or {}).get("execution_plan") or {})
            for key in ("required_capabilities", "video_mode", "candidate_count", "recovery_budget"):
                if key in candidate_plan:
                    final_plan[key] = candidate_plan[key]
            final_plan["shot_id"] = shot_id
            final_profile["execution_plan"] = final_plan
            shot.continuity_profile = json.dumps(final_profile, ensure_ascii=False)
            shot.continuity_reference_path = shot_data.get("continuity_reference_path", "")
            shot.video_reference_manifest = json.dumps(shot_data.get("reference_manifest", []), ensure_ascii=False)
            shot.reference_capability_warning = str(shot_data.get("continuity_profile", {}).get("reference_capability_warning", ""))
            shot.pose_reference_path = shot_data.get("pose_reference_path", "")
            shot.depth_reference_path = shot_data.get("depth_reference_path", "")
            shot.audio_path = audio_path
            if timed_lines:
                # 新配音的逐句实测时间轴随镜头落库，字幕与时间线按此计算。
                shot.dialogue = serialize_dialogue_lines(timed_lines)
            shot.video_path = result["video_path"]
            shot.last_frame_path = result.get("frame_path", "")
            if not shot.image_path:
                shot.image_path = result.get("frame_path", "")
            shot.status = "video_done"
            # 视频 + 配音 + 尾帧全部基于当前参数重新生成：过期标记就此清除。
            clear_shot_media_stale(shot)
            create_version(
                db,
                shot,
                "regenerate",
                task_id=f"shot:{shot_id}:video",
                candidate_selection=selection.model_dump(mode="json"),
                decision_trace=decision_trace,
                force=True,
            )
            db.commit()
            update = _shot_update_payload(shot)
        finally:
            db.close()

        await ws_manager.send_to_project(project_id, update)
        return {
            "shot_id": shot_id,
            "shot_version": int(expected_version),
            "selected_video_candidate_id": selected_candidate_id,
            "candidate_selection": selection.model_dump(mode="json"),
            "decision_trace": decision_trace,
            "video_candidates": _shot_video_candidates(shot_id, shot_version=int(expected_version)),
            "video_path": str(result.get("video_path") or ""),
            "tail_frame_path": str(result.get("frame_path") or ""),
        }
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


def _shot_dialogue_lines(s: Shot) -> list:
    """读取镜头对白为结构化列表；旧版纯文本自动迁移为单条（说话人取场内
    第一个角色并记录 warning，保持旧配音行为可追溯）。"""

    speakers = _json_list(s.characters_in_scene)
    fallback = speakers[0] if speakers and isinstance(speakers[0], str) else ""
    return parse_shot_dialogue(
        s.dialogue,
        fallback_speaker=fallback,
        default_emotion=s.emotion or "neutral",
        warn_key=f"shot {s.id}",
    )


def _serialize_dialogue_input(value, shot: Shot, db: Session) -> str:
    """把请求 DTO 的对白列表序列化入库，并做说话人可追踪校验。"""

    lines = parse_shot_dialogue(value)
    if lines:
        warn_unknown_speakers(lines, _characters(db, shot.project_id), context=f"update shot {shot.id}")
    if (
        len(lines) == 1
        and not lines[0].speaker
        and not lines[0].action
        and lines[0].start_ms is None
        and lines[0].end_ms is None
    ):
        # 单句无说话人/时间轴的旧版输入保持纯文本存储；完整结构化输入仍原样 JSON 化。
        return str(lines[0].line)
    return serialize_dialogue_lines(lines)


def _serialize_shot(s: Shot) -> dict:
    profile = _json_dict(s.continuity_profile)
    return {
        "id": s.id,
        "project_id": s.project_id,
        "sequence": s.sequence,
        "shot_type": s.shot_type,
        "scene_description": s.scene_description,
        "character_action": s.character_action,
        "dialogue": dialogue_lines_payload(_shot_dialogue_lines(s)),
        "camera_angle": s.camera_angle,
        "camera_movement": s.camera_movement or "静止",
        "duration": s.duration,
        "estimated_speech_ms": s.estimated_speech_ms or 0,
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
        "consistency_status": getattr(s, "consistency_status", "pending"),
        "consistency_report": _json_dict(getattr(s, "consistency_report", "{}")),
        "storyboard_reference_manifest": _json_list_raw(getattr(s, "storyboard_reference_manifest", "[]")),
        "video_reference_manifest": _json_list_raw(getattr(s, "video_reference_manifest", "[]")),
        "reference_capability_warning": getattr(s, "reference_capability_warning", ""),
        "characters_in_scene": json.loads(s.characters_in_scene) if s.characters_in_scene else [],
        "scene_asset_id": s.scene_asset_id or "",
        "character_asset_ids": json.loads(s.character_asset_ids) if s.character_asset_ids else [],
        "scene_group_id": s.scene_group_id or "",
        "consistency_context": s.consistency_context or "",
        "reference_weights": _json_dict(s.reference_weights),
        "continuity_mode": normalize_continuity_mode(profile.get("continuity_mode"), default=""),
        "continuity_mode_source": str(profile.get("continuity_mode_source") or ""),
        "continuity_profile": profile,
        "continuity_reference_path": s.continuity_reference_path or "",
        "pose_reference_path": s.pose_reference_path or "",
        "depth_reference_path": s.depth_reference_path or "",
        "last_frame_path": s.last_frame_path or "",
        "style_fingerprint": s.style_fingerprint or "",
        # 质量审核摘要（最新一轮）：verdict/passed/score/degraded/未检测维度。
        # 结构检查（storyboard_status 等）与质量审核严格分开展示。
        "quality_review": quality_review_service.shot_review_summary(s.id),
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
        "duration": shot.duration,
        "estimated_speech_ms": shot.estimated_speech_ms or 0,
        "consistency_status": getattr(shot, "consistency_status", "pending"),
        "consistency_report": _json_dict(getattr(shot, "consistency_report", "{}")),
        "storyboard_reference_manifest": _json_list_raw(getattr(shot, "storyboard_reference_manifest", "[]")),
        "video_reference_manifest": _json_list_raw(getattr(shot, "video_reference_manifest", "[]")),
        "reference_capability_warning": getattr(shot, "reference_capability_warning", ""),
        "image_path": shot.image_path,
        "storyboard_path": shot.storyboard_path,
        "audio_path": shot.audio_path,
        "video_path": shot.video_path,
        "last_frame_path": shot.last_frame_path,
        "scene_group_id": shot.scene_group_id,
        "reference_weights": _json_dict(shot.reference_weights),
        "continuity_mode": normalize_continuity_mode(_json_dict(shot.continuity_profile).get("continuity_mode"), default=""),
        "continuity_mode_source": _json_dict(shot.continuity_profile).get("continuity_mode_source", ""),
        "continuity_profile": _json_dict(shot.continuity_profile),
        "continuity_reference_path": shot.continuity_reference_path,
        "pose_reference_path": shot.pose_reference_path,
        "depth_reference_path": shot.depth_reference_path,
        "reference_mode": _json_dict(shot.continuity_profile).get("reference_mode", ""),
        "references_validated": _json_dict(shot.continuity_profile).get("references_validated", False),
        "references_sent": _json_dict(shot.continuity_profile).get("references_sent", []),
        "provider": _json_dict(shot.continuity_profile).get("provider", ""),
        "model": _json_dict(shot.continuity_profile).get("model", ""),
        "requested_style": _json_dict(shot.continuity_profile).get("requested_style", ""),
        "effective_style": _json_dict(shot.continuity_profile).get("effective_style", ""),
        "style_source": _json_dict(shot.continuity_profile).get("style_source", ""),
        "provider_source": _json_dict(shot.continuity_profile).get("provider_source", ""),
        "references_unsupported": bool(_json_dict(shot.continuity_profile).get("references_unsupported")),
        "reference_capability_warning": _json_dict(shot.continuity_profile).get("reference_capability_warning", ""),
        "quality_review": quality_review_service.shot_review_summary(shot.id),
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


def _json_list_raw(raw: str | None) -> list:
    try:
        value = json.loads(raw or "[]")
        return value if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


def _json_dict(raw: str | dict | None) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
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
                    from services.reference_readiness_service import mark_reference_success

                    mark_reference_success(db, "scene", model, ref_path)
                    db.commit()
            finally:
                db.close()
        except Exception as exc:
            from services.error_reporter import log_failure
            from services.reference_readiness_service import mark_reference_failure

            error_id = log_failure(exc, error_type=ERROR_STORYBOARD, context={"asset_type": "scene", "asset_id": scene_id})
            db = SessionLocal()
            try:
                model = (
                    db.query(SceneAsset)
                    .filter(SceneAsset.id == scene_id, SceneAsset.project_id == asset_project_id)
                    .first()
                )
                if model:
                    mark_reference_failure(db, "scene", model, exc, error_id=error_id)
                    db.commit()
                    refresh_project_reference_state(db, project_id)
            finally:
                db.close()


def _previous_shot_for_continuity(db: Session, shot: Shot) -> Shot | None:
    """只取时间线上的紧邻上一镜；连续性不跨越中间镜头回溯旧素材。"""

    return (
        db.query(Shot)
        .filter(Shot.project_id == shot.project_id, Shot.sequence < shot.sequence)
        .order_by(Shot.sequence.desc())
        .first()
    )


def _previous_reference_for_shot(db: Session, shot: Shot, prefer_last_frame: bool = False) -> str:
    """兼容入口：返回策略实际选中的上一镜末帧，不再回退故事板/首帧。"""

    del prefer_last_frame  # 历史参数保留签名；候选类型现在完全由 continuity_mode 决定。
    previous = _previous_shot_for_continuity(db, shot)
    decision = consistency_service.resolve_continuity(
        _shot_dict(shot),
        _shot_dict(previous) if previous else None,
    )
    return str(decision.get("continuity_reference_path") or "")


async def _progress(project_id: str, step: str, progress: int, message: str, *, job_keys: tuple[str, ...] = (), report: dict | None = None):
    """推送项目进度，并把同一份进度写进任务中心的 durable 记录。

    ``job_keys`` 传候选键即可：只有真正持有当前 run token 的那个会写入成功，其余
    会被 task_registry 静默忽略，因此旧尝试的迟到回调不会覆盖新尝试。
    """

    for key in job_keys:
        update_job_progress(key, progress, current_step=step, message=message, report=report)
    await ws_manager.send_to_project(project_id, {"type": "progress", "step": step, "progress": progress, "message": message})
