import asyncio
import json
import os
import uuid
import hashlib
from pathlib import Path
from typing import Literal

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from agent.nodes import script_parser, storyboard_gen
from api import schemas
from api.websocket import ws_manager
from config import settings
from db import SessionLocal
from models import Character as CharacterModel
from models import Project, SceneAsset, Shot as ShotModel
from models import ShotVersion
from services.consistency_service import ConsistencyService
from services.reference_readiness_service import (
    mark_reference_failure,
    mark_reference_success,
    refresh_project_reference_state,
)
from services.error_reporter import ERROR_PIPELINE, log_failure, redact, report_failure
from services.image_service import ImageService
from services.llm_service import LLMService
from services.shot_version_service import create_version
from services.skill_config_service import agent_prompt_append, resolve_effective_style, resolve_skill_config
from services.style_templates import style_prompt_params, style_template
from services.tts_service import normalize_mimo_voice
from services.security import UploadLimitExceeded, safe_filename, save_upload_stream, validate_identifier, validate_script_upload
from api.claim_guard import budget_notice, claim_or_block
from api.provider_guard import ensure_providers_ready
from services.task_registry import claim as claim_task, start as start_task, update_progress as update_job_progress

router = APIRouter(prefix="/api/script", tags=["script"])

_background_tasks: set[asyncio.Task] = set()
llm_service = LLMService()
image_service = ImageService()
consistency_service = ConsistencyService()


class ScriptGenerateRequest(BaseModel):
    project_id: schemas.OptionalIdentifier | None = None
    prompt: schemas.GenerationPrompt = "校园成长短剧"
    style: schemas.StyleId = "anime"
    genre: schemas.Genre = "原创短剧"
    target_duration: schemas.TargetDuration = 45
    characters_hint: schemas.CharactersHint = ""


class ScriptParseRequest(BaseModel):
    project_id: schemas.Identifier
    user_input: schemas.ScriptText
    input_type: schemas.InputType = "text"
    style: schemas.StyleId = "anime"
    output_format: schemas.OutputFormat = "9:16"
    resolution: schemas.Resolution = "1080p"
    platform: schemas.Platform = "douyin"
    target_duration: schemas.TargetDuration = 45
    mode: schemas.PipelineMode = "manual"
    quality_profile: Literal["draft", "standard", "finishing"] = "standard"


@router.post("/generate")
async def generate_script(data: ScriptGenerateRequest):
    skill_config = _resolved_skill_config(data.project_id or "")
    style_meta = resolve_effective_style(data.style, skill_config, "script_agent")
    script = await _generate_script_text(data, skill_config)
    return {
        "project_id": data.project_id,
        "title": _script_title(script),
        "script": script,
        "style": style_meta["effective_style"],
        **style_meta,
        "genre": data.genre,
        "target_duration": data.target_duration,
    }


@router.post("/parse")
async def parse_script(data: ScriptParseRequest):
    if not data.user_input.strip():
        raise HTTPException(status_code=400, detail="请输入剧本内容")
    if len(data.user_input) > settings.MAX_SCRIPT_TEXT_CHARS:
        raise HTTPException(status_code=413, detail="剧本文本超过长度限制")
    _require_project(data.project_id)
    # 启动前预检：LLM（手动模式）/ 全链路 Provider（自动模式）未配置时直接拒绝。
    ensure_providers_ready("script_pipeline", mode=data.mode)
    skill_config = _resolved_skill_config(data.project_id)
    style_meta = resolve_effective_style(data.style, skill_config, "script_agent")
    task = _spawn_pipeline(
        data.project_id,
        _initial_state(data.model_dump(), skill_config),
        data.mode,
        data.output_format,
        data.resolution,
    )
    if task is None:
        return {"status": "already_running", "project_id": data.project_id, "mode": data.mode, "deduplicated": True, **style_meta}
    return {
        "status": "started",
        "project_id": data.project_id,
        "mode": data.mode,
        "quality_profile": data.quality_profile,
        **style_meta,
        **getattr(task, "budget_notice", {}),
    }


@router.post("/upload")
async def upload_script(
    project_id: str = Form(...),
    file: UploadFile = File(...),
    style: schemas.StyleId = Form("anime"),
    output_format: schemas.OutputFormat = Form("9:16"),
    resolution: schemas.Resolution = Form("1080p"),
    platform: schemas.Platform = Form("douyin"),
    target_duration: schemas.TargetDuration = Form(45),
    mode: schemas.PipelineMode = Form("manual"),
    quality_profile: Literal["draft", "standard", "finishing"] = Form("standard"),
):
    try:
        safe_project_id = validate_identifier(project_id, "项目 ID")
        _require_project(safe_project_id)
        _, file_ext = safe_filename(file.filename or "script.txt", {".txt", ".md", ".markdown", ".docx"})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Store each upload under a project-scoped directory with a server-generated
    # name.  This avoids traversal, filename collisions and symlink surprises.
    upload_dir = settings.DATA_DIR / "uploads" / safe_project_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"script_{uuid.uuid4().hex}{file_ext or '.txt'}"
    try:
        await save_upload_stream(file, file_path, settings.MAX_SCRIPT_UPLOAD_BYTES)
    except UploadLimitExceeded as exc:
        raise HTTPException(status_code=413, detail="上传剧本超过大小限制") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="保存上传文件失败") from exc

    try:
        validate_script_upload(file_path, file_ext, getattr(file, "content_type", None))
    except (OSError, ValueError) as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if file_ext.lower() == ".docx":
        from docx import Document

        doc = Document(str(file_path))
        user_input = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            user_input = await f.read(settings.MAX_SCRIPT_TEXT_CHARS + 1)

    if len(user_input) > settings.MAX_SCRIPT_TEXT_CHARS:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="剧本文本超过长度限制")

    payload = {
        "project_id": safe_project_id,
        "user_input": user_input,
        "input_type": "file",
        "uploaded_file_path": str(file_path),
        "file_type": file_ext.lstrip("."),
        "style": style,
        "output_format": output_format,
        "resolution": resolution,
        "platform": platform,
        "target_duration": target_duration,
        "quality_profile": quality_profile,
    }
    # 与 /parse 同口径的启动预检：缺 Provider 配置时拒绝排队，删除临时上传文件。
    try:
        ensure_providers_ready("script_pipeline", mode=mode)
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    skill_config = _resolved_skill_config(safe_project_id)
    style_meta = resolve_effective_style(style, skill_config, "script_agent")
    task = _spawn_pipeline(safe_project_id, _initial_state(payload, skill_config), mode, output_format, resolution)
    if task is None:
        return {"status": "already_running", "project_id": safe_project_id, "mode": mode, "deduplicated": True, **style_meta}
    return {
        "status": "started",
        "project_id": safe_project_id,
        "mode": mode,
        "quality_profile": quality_profile,
        "file": file.filename,
        "script": user_input,
        **style_meta,
        **getattr(task, "budget_notice", {}),
    }


async def _generate_script_text(data: ScriptGenerateRequest, skill_config: dict | None = None) -> str:
    if not llm_service.available:
        raise RuntimeError("未配置可用的 Mimo/LLM API Key，无法生成真实剧本")

    script_style = resolve_effective_style(data.style, skill_config, "script_agent")["effective_style"]
    script_append = agent_prompt_append(skill_config, "script_agent")
    try:
        script = await llm_service.call(
            "你是漫剧编剧。请输出完整中文漫剧剧本，包含标题、人物、场景、动作、对白和情绪，不要输出解释。",
            f"""
创作方向：{data.prompt}
类型：{data.genre}
画风：{style_template(script_style)["label"]}（{script_style}）
目标时长：{data.target_duration} 秒
人物提示：{data.characters_hint or "自行设计 1 到 3 名角色"}
Skill 配置：{script_append}

请用清晰结构输出：标题、人物、场景一、动作、对白、情绪、场景二……
""",
            temperature=0.75,
        )
    except Exception as exc:
        raise RuntimeError(f"Mimo 剧本生成失败: {exc}") from exc

    if not script.strip():
        raise RuntimeError("Mimo 剧本生成返回为空")
    return script


def _initial_state(data: dict, skill_config: dict | None = None) -> dict:
    style_meta = resolve_effective_style(data.get("style", "anime"), skill_config, "script_agent")
    style = style_meta["effective_style"]
    return {
        "project_id": data["project_id"],
        "user_input": data.get("user_input", ""),
        "input_type": data.get("input_type", "text"),
        "uploaded_file_path": data.get("uploaded_file_path", ""),
        "file_type": data.get("file_type", ""),
        "script_title": "",
        "genre": "",
        "style_suggestion": style,
        "characters": [],
        "raw_script": "",
        "script_scenes": [],
        "logic_issues": [],
        "shots": [],
        "style": style,
        **style_meta,
        "style_params": style_prompt_params(style),
        "skill_config": skill_config or {},
        "skill_prompt_append": agent_prompt_append(skill_config, "script_agent"),
        "output_format": data.get("output_format", "9:16"),
        "resolution": data.get("resolution", "1080p"),
        "platform": data.get("platform", "douyin"),
        "target_duration": data.get("target_duration", 45),
        "quality_profile": data.get("quality_profile") or "standard",
        "timing_plan": {},
        "video_path": "",
        "current_step": "",
        "errors": [],
        "human_feedback": "",
        "needs_human_review": False,
        "storyboard_confirmed": False,
        "rag_context": [],
        "narrative_context": {},
        "generation_preferences": {},
    }


def _resolved_skill_config(project_id: str) -> dict:
    db = SessionLocal()
    try:
        return resolve_skill_config(project_id, db)
    finally:
        db.close()


def _require_project(project_id: str) -> Project:
    """Validate a project before creating files or background work."""

    try:
        safe_id = validate_identifier(project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == safe_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="项目不存在")
        return project
    finally:
        db.close()


def _spawn_pipeline(project_id: str, initial_state: dict, mode: str, output_format: str, resolution: str) -> asyncio.Task | None:
    """根据 mode 启动手动(逐步)或自动(LangGraph 端到端)流水线,均后台异步执行。"""
    task_key = f"project:{project_id}:pipeline:{mode}"
    # 抢占即预算闸门：硬预算不足时这里会抛 409（error_code=budget_exceeded）。
    claim = claim_or_block(
        task_key,
        f"project:{project_id}",
        current_step="parse_script",
        message="已排队，正在解析剧本",
    )
    if not claim.claimed:
        return None
    _pending_budget_notice = budget_notice(claim)
    if mode == "auto":
        coro = _run_auto_pipeline(
            project_id,
            initial_state,
            output_format,
            resolution,
            quality_profile=str(initial_state.get("quality_profile") or "standard"),
        )
    else:
        coro = _run_storyboard_phase(project_id, initial_state)
    task = start_task(task_key, coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    # 软预算提示跟随本次任务对象返回给路由层（未触发时为空 dict）。
    task.budget_notice = _pending_budget_notice  # type: ignore[attr-defined]
    return task


async def _run_auto_pipeline(
    project_id: str,
    initial_state: dict,
    output_format: str,
    resolution: str,
    *,
    quality_profile: str = "standard",
):
    """自动模式:经 LangGraph 一次 ainvoke 从解析跑到成片,节点复用 route 步骤函数。"""
    from agent.graph import get_graph

    try:
        result = await get_graph().ainvoke(
            {
                "project_id": project_id,
                "mode": "auto",
                "initial_state": initial_state,
                "output_format": output_format,
                "resolution": resolution,
                "quality_profile": quality_profile,
                "run_id": "auto",
                "current_step": "",
                "errors": [],
            }
        )
        if result.get("errors"):
            db = SessionLocal()
            try:
                _mark_project_error(db, project_id)
            finally:
                db.close()
            raise RuntimeError("自动流程失败: " + "\n".join(result["errors"]))
    except Exception as exc:
        db = SessionLocal()
        try:
            _mark_project_error(db, project_id)
        finally:
            db.close()
        await ws_manager.send_to_project(
            project_id,
            report_failure(
                exc,
                error_type=ERROR_PIPELINE,
                message="自动流程中断，本次生成已停止。请检查运行日志后重试，或改用「手动审核」逐步生成。",
                context={"project_id": project_id, "mode": "auto"},
            ),
        )
        raise


async def _run_storyboard_phase(project_id: str, state: dict, mode: str = "manual"):
    try:
        await _progress(project_id, "parse_script", 10, "正在解析剧本")
        state.update(await script_parser.run(state))
        state["characters"] = [
            consistency_service.enrich_character(character, index)
            for index, character in enumerate(state.get("characters", []))
        ]
        state["script_scenes"] = [
            consistency_service.enrich_scene(scene, index)
            for index, scene in enumerate(state.get("script_scenes", []))
        ]

        await _progress(project_id, "generate_storyboard", 32, "正在生成分镜列表")
        state.update(await storyboard_gen.run(state))

        await _progress(project_id, "wait_asset_confirm", 40, "正在生成角色三视图与项目素材板")
        db = SessionLocal()
        try:
            asset_project_id = _resolve_asset_project_id(db, project_id)
        finally:
            db.close()
        await _ensure_character_reference_images(asset_project_id, state)
        await _ensure_scene_baseline_images(asset_project_id, state)

        db = SessionLocal()
        try:
            project_title = _persist_phase1(db, project_id, state, status="assets_ready")
            consistency_report = refresh_project_reference_state(db, project_id)
        finally:
            db.close()
        if mode == "auto" and consistency_report.get("blocking"):
            await _progress(
                project_id,
                "needs_reference_review",
                45,
                "一致性参考素材未达到自动成片要求，已转人工审核",
                report=consistency_report,
            )
            await ws_manager.send_to_project(
                project_id,
                {
                    "type": "reference_review_required",
                    "project_id": project_id,
                    "consistency_report": consistency_report,
                    "affected_shot_ids": consistency_report.get("affected_shot_ids", []),
                    "shot_range": consistency_report.get("shot_range", ""),
                    "message": "自动模式已阻止生成：请补齐或明确降级一致性参考素材后重试。",
                },
            )
            return
        await _progress(
            project_id,
            "wait_asset_confirm",
            45,
            "角色板、场景板与分镜已生成，请确认素材后生成故事板",
            report=consistency_report,
        )
        await ws_manager.send_to_project(
            project_id,
            {
                "type": "complete",
                "project_id": project_id,
                # 根据剧本自动命名的项目标题：客户端据此同步项目名与项目列表。
                "title": project_title,
                "shots": state.get("shots", []),
                "video_path": "",
                "asset_board_ready": True,
                "requested_style": state.get("requested_style", state.get("style")),
                "effective_style": state.get("effective_style", state.get("style")),
                "style_source": state.get("style_source", "project_request"),
                "consistency_report": consistency_report,
                "consistency_status": consistency_report.get("status", "ready"),
                "affected_shot_ids": consistency_report.get("affected_shot_ids", []),
                "shot_range": consistency_report.get("shot_range", ""),
            },
        )
    except Exception as exc:
        db = SessionLocal()
        try:
            db.rollback()
            _mark_project_error(db, project_id)
        finally:
            db.close()
        await ws_manager.send_to_project(
            project_id,
            report_failure(
                exc,
                error_type=ERROR_PIPELINE,
                message="剧本与分镜生成失败，本次生成已停止。请确认剧本内容与模型配置后重试。",
                context={"project_id": project_id, "mode": "manual"},
            ),
        )
        raise


def _persist_phase1(db, project_id: str, state: dict, status: str = "assets_ready") -> str:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise RuntimeError("项目不存在")
    asset_project_id = project.parent_project_id or project.id

    # 根据剧本自动命名：优先用 LLM 解析出的剧名，其次回退到剧本文本本身的标题行。
    project.title = state.get("script_title") or _script_title(state.get("user_input", "")) or project.title
    project.genre = state.get("genre") or project.genre
    # The request/state style is authoritative. Never fall back through a
    # parser suggestion or legacy Skill default that can silently become anime.
    project.style = state.get("effective_style") or state.get("style") or project.style
    project.input_text = state.get("user_input", "")
    project.input_type = state.get("input_type", "text")
    project.output_format = state.get("output_format", project.output_format)
    project.resolution = state.get("resolution", project.resolution)
    project.platform = state.get("platform", project.platform)
    project.target_duration = int(state.get("target_duration") or project.target_duration or 0)
    project.timing_plan = json.dumps(state.get("timing_plan") or {}, ensure_ascii=False)
    project.consistency_config = json.dumps(consistency_service.project_config(), ensure_ascii=False)
    project.status = status

    character_ids = _upsert_characters(db, asset_project_id, state.get("characters", []), state.get("effective_style") or state.get("style"))
    scene_ids = _upsert_scenes(db, asset_project_id, state.get("script_scenes", []), state.get("effective_style") or state.get("style"))

    existing_shots = db.query(ShotModel).filter(ShotModel.project_id == project_id).all()
    if any(shot.confirmed or shot.video_path for shot in existing_shots):
        raise RuntimeError("项目已存在已确认/已出片的镜头，重新解析会丢失这些成果，请先清空项目后再解析")
    db.query(ShotVersion).filter(ShotVersion.project_id == project_id).delete()
    db.query(ShotModel).filter(ShotModel.project_id == project_id).delete()
    shots = state.get("shots", [])
    for index, shot in enumerate(shots):
        model = _shot_model(project_id, shot, index + 1, character_ids, scene_ids, state.get("script_scenes", []), state.get("characters", []))
        model.style_fingerprint = hashlib.sha256(str(state.get("effective_style") or state.get("style") or "anime").encode()).hexdigest()[:16]
        db.add(model)
        # 剧本解析生成的镜头以 import 来源进入版本历史，作为时间线的 v1。
        create_version(db, model, "import")

    db.commit()
    return project.title


async def _ensure_character_reference_images(asset_project_id: str, state: dict) -> None:
    """生成角色三视图，并把成功/失败状态写回 state，绝不静默吞掉异常。"""
    style = state.get("effective_style") or state.get("style") or state.get("style_suggestion") or "anime"
    skill_append = agent_prompt_append(state.get("skill_config"), "script_agent")
    characters = state.get("characters", [])
    semaphore = asyncio.Semaphore(3)

    async def ensure_one(index: int, character: dict) -> None:
        character.setdefault("id", _character_asset_id(asset_project_id, index))
        refs = character.get("reference_images")
        if isinstance(refs, list) and refs:
            character["reference_status"] = "ready"
            return
        character["reference_retry_count"] = int(character.get("reference_retry_count") or 0) + 1
        try:
            async with semaphore:
                character_payload = dict(character)
                if skill_append:
                    character_payload["visual_prompt"] = ", ".join(
                        part for part in [character.get("visual_prompt", ""), skill_append] if part
                    )
                ref_path = await image_service.generate_character_reference(
                    character=character_payload,
                    style=style,
                    project_id=asset_project_id,
                    seed=int(character.get("seed") or 42) + 7000 + index,
                )
            character["reference_images"] = [ref_path]
            character["reference_status"] = "ready"
            character["reference_version"] = int(character.get("reference_version") or 1)
            character["reference_failure_reason"] = ""
            character["reference_error_id"] = ""
        except Exception as exc:
            error_id = log_failure(exc, error_type=ERROR_PIPELINE, context={"asset_type": "character", "asset_id": character.get("id")})
            character["reference_images"] = list(refs or [])
            character["reference_status"] = "stale" if refs else "failed"
            character["reference_failure_reason"] = redact(str(exc), limit=500)
            character["reference_error_id"] = error_id

    await asyncio.gather(*(ensure_one(index, character) for index, character in enumerate(characters)))


async def _ensure_scene_baseline_images(asset_project_id: str, state: dict) -> None:
    """生成场景基准图，并把成功/失败状态写回 state，绝不静默吞掉异常。"""
    style = state.get("effective_style") or state.get("style") or state.get("style_suggestion") or "anime"
    skill_append = agent_prompt_append(state.get("skill_config"), "script_agent")
    for index, scene in enumerate(state.get("script_scenes", [])):
        scene.setdefault("id", _scene_asset_id(asset_project_id, index))
        refs = scene.get("reference_images")
        baseline = scene.get("baseline_image_path", "")
        if baseline or (isinstance(refs, list) and refs):
            scene["reference_status"] = "ready"
            continue
        scene["reference_retry_count"] = int(scene.get("reference_retry_count") or 0) + 1
        try:
            scene_payload = dict(scene)
            if skill_append:
                scene_payload["visual_prompt"] = ", ".join(part for part in [scene.get("visual_prompt", ""), skill_append] if part)
            ref_path = await image_service.generate_scene_baseline_reference(
                scene=scene_payload,
                style=style,
                project_id=asset_project_id,
                seed=int(scene.get("seed") or 1200) + index,
            )
            scene["baseline_image_path"] = ref_path
            scene["reference_images"] = [ref_path]
            scene["reference_status"] = "ready"
            scene["reference_version"] = int(scene.get("reference_version") or 1)
            scene["reference_failure_reason"] = ""
            scene["reference_error_id"] = ""
        except Exception as exc:
            error_id = log_failure(exc, error_type=ERROR_PIPELINE, context={"asset_type": "scene", "asset_id": scene.get("id")})
            scene["baseline_image_path"] = baseline
            scene["reference_images"] = list(refs or [])
            scene["reference_status"] = "stale" if (baseline or refs) else "failed"
            scene["reference_failure_reason"] = redact(str(exc), limit=500)
            scene["reference_error_id"] = error_id


def _copy_reference_state(item, source: dict) -> None:
    for key in (
        "reference_status",
        "reference_version",
        "reference_retry_count",
        "reference_failure_reason",
        "reference_error_id",
        "reference_skip_reason",
        "reference_capability_warning",
    ):
        if key in source:
            setattr(item, key, source[key])


def _resolve_asset_project_id(db, project_id: str) -> str:
    project = db.query(Project).filter(Project.id == project_id).first()
    return (project.parent_project_id or project.id) if project else project_id


def _upsert_characters(db, asset_project_id: str, characters: list[dict], style: str = "anime") -> dict[str, str]:
    ids: dict[str, str] = {}
    for index, char in enumerate(characters):
        name = char.get("name") or f"角色{index + 1}"
        char_id = char.get("id") or _character_asset_id(asset_project_id, index)
        char["id"] = char_id
        existing = db.query(CharacterModel).filter(CharacterModel.project_id == asset_project_id, CharacterModel.id == char_id).first()
        item = existing or CharacterModel(id=char_id, project_id=asset_project_id, name=name)
        item.name = name
        fingerprint = hashlib.sha256(str(style or "anime").encode()).hexdigest()[:16]
        # 空指纹 = 历史遗留、来源未知：不猜测它属于哪种风格，一律视为与当前
        # 风格不一致，标记 stale 并清空引用（不删除旧文件）。
        style_changed = existing is not None and (item.style_fingerprint or "") != fingerprint
        if style_changed:
            item.asset_status = "stale"
            item.reference_images = "[]"
        item.style_fingerprint = fingerprint
        item.appearance = json.dumps(char.get("appearance", {}), ensure_ascii=False)
        item.personality = char.get("personality", "")
        item.visual_prompt = char.get("visual_prompt", "")
        item.negative_prompt = char.get("negative_prompt", "")
        item.voice_id = normalize_mimo_voice(char.get("voice_id", ""))
        item.emotion_variants = json.dumps(char.get("emotion_variants", {}), ensure_ascii=False)
        item.key_features = json.dumps(char.get("key_features", []), ensure_ascii=False)
        appearance = char.get("appearance", {}) if isinstance(char.get("appearance"), dict) else {}
        item.default_outfit = char.get("default_outfit") or appearance.get("default_outfit", "")
        item.lora_profile = ""
        item.ip_adapter_profile = ""
        item.wardrobe_lock = char.get("wardrobe_lock", "")
        next_refs = char.get("reference_images", [])
        if next_refs:
            # 本次运行内按当前生效风格新生成的参考图：直接置为 active。
            item.reference_images = json.dumps(next_refs, ensure_ascii=False)
            item.asset_status = "active"
        elif not existing:
            item.reference_images = "[]"
            item.asset_status = "active"
        _copy_reference_state(item, char)
        item.seed = str(char.get("seed", 42 + index))
        if not existing:
            db.add(item)
        ids[name] = item.id
    return ids


def _character_asset_id(asset_project_id: str, index: int) -> str:
    return f"{asset_project_id}_char_{index + 1:04d}"


def _upsert_scenes(db, asset_project_id: str, scenes: list[dict], style: str = "anime") -> list[str]:
    ids: list[str] = []
    for index, scene in enumerate(scenes):
        location = scene.get("location") or scene.get("name") or f"场景{index + 1}"
        time_of_day = scene.get("time_of_day") or "locked"
        scene_group_key = scene.get("scene_group_key", "")
        name = scene.get("name") or f"{location}-{time_of_day}"
        scene_id = scene.get("id") or _scene_asset_id(asset_project_id, index)
        scene["id"] = scene_id
        base_query = db.query(SceneAsset).filter(SceneAsset.project_id == asset_project_id)
        existing = base_query.filter(SceneAsset.scene_group_key == scene_group_key).first() if scene_group_key else None
        if not existing:
            existing = base_query.filter(SceneAsset.name == name).first()
        item = existing or SceneAsset(id=scene_id, project_id=asset_project_id, name=name)
        item.name = name
        fingerprint = hashlib.sha256(str(style or "anime").encode()).hexdigest()[:16]
        # 与角色资产同口径：指纹为空（来源未知）或与当前风格不一致都标记
        # stale 并清空基准图引用（不删除旧文件），等待按当前风格重建。
        style_changed = existing is not None and (item.style_fingerprint or "") != fingerprint
        if style_changed:
            item.asset_status = "stale"
            item.baseline_image_path = ""
            item.reference_images = "[]"
        item.style_fingerprint = fingerprint
        item.description = scene.get("actions") or scene.get("description") or ""
        item.visual_prompt = scene.get("visual_prompt") or item.visual_prompt or f"{name}, consistent {style} background, clean composition"
        item.negative_prompt = "watermark, subtitles, text artifacts, low quality"
        item.key_features = json.dumps([scene.get("emotion", "neutral"), scene.get("camera_suggestion", "medium")], ensure_ascii=False)
        item.scene_group_key = scene_group_key
        item.time_of_day = time_of_day
        next_refs = scene.get("reference_images", [])
        if scene.get("baseline_image_path") or next_refs:
            item.baseline_image_path = scene.get("baseline_image_path", "")
            item.reference_images = json.dumps(next_refs, ensure_ascii=False)
            item.asset_status = "active"
        elif not existing:
            item.baseline_image_path = ""
            item.reference_images = "[]"
            item.asset_status = "active"
        _copy_reference_state(item, scene)
        item.consistency_profile = json.dumps(scene.get("consistency_profile", {}), ensure_ascii=False)
        item.prop_lock = scene.get("prop_lock", "")
        item.seed = 1200 + index
        if not existing:
            db.add(item)
        ids.append(item.id)
    return ids


def _scene_asset_id(asset_project_id: str, index: int) -> str:
    return f"{asset_project_id}_scene_{index + 1:04d}"


def _shot_model(
    project_id: str,
    shot: dict,
    sequence: int,
    character_ids: dict[str, str],
    scene_ids: list[str],
    scenes: list[dict],
    characters: list[dict],
) -> ShotModel:
    characters_in_scene = shot.get("characters_in_scene", [])
    scene_number = int(shot.get("scene_number") or shot.get("source_scene_number") or min(sequence, max(len(scene_ids), 1)))
    scene_index = max(0, min(scene_number - 1, len(scene_ids) - 1)) if scene_ids else 0
    scene_asset_id = scene_ids[scene_index] if scene_ids else ""
    scene_meta = scenes[scene_index] if scenes and scene_index < len(scenes) else {}
    scene_cards: dict[str, dict] = {}
    for item_index, item_id in enumerate(scene_ids):
        item_meta = scenes[item_index] if item_index < len(scenes) else {}
        scene_cards.setdefault(
            item_id,
            {
                **item_meta,
                "id": item_id,
                "name": item_meta.get("name") or item_meta.get("location") or item_id,
            },
        )
    character_cards = []
    for character in characters:
        card = dict(character)
        name = card.get("name", "")
        if name in character_ids:
            card["id"] = character_ids[name]
        character_cards.append(card)
    shot_context = {
        **shot,
        "shot_id": shot.get("shot_id") or f"{project_id}_shot_{sequence:04d}",
        "scene_asset_id": scene_asset_id,
        "scene_group_id": scene_meta.get("scene_group_key", scene_asset_id),
        "characters_in_scene": characters_in_scene,
        "character_asset_ids": [character_ids[name] for name in characters_in_scene if name in character_ids],
    }
    consistency = consistency_service.build_generation_context(shot_context, character_cards, scene_cards)
    return ShotModel(
        id=shot_context["shot_id"],
        project_id=project_id,
        sequence=sequence,
        shot_type=shot.get("shot_type", "medium"),
        scene_description=shot.get("scene_description", ""),
        character_action=shot.get("character_action", ""),
        # 状态里的对白是结构化列表；ORM validates 钩子负责序列化入库。
        dialogue=shot.get("dialogue", ""),
        camera_angle=shot.get("camera_angle", "正面"),
        camera_movement=shot.get("camera_movement", "静止"),
        duration=shot.get("duration", 3.0),
        estimated_speech_ms=int(shot.get("estimated_speech_ms") or 0),
        emotion=shot.get("emotion", "neutral"),
        transition=shot.get("transition", "cut"),
        image_path="",
        audio_path="",
        status="pending",
        version=shot.get("version", 1),
        confirmed=False,
        characters_in_scene=json.dumps(characters_in_scene, ensure_ascii=False),
        character_asset_ids=json.dumps(shot_context["character_asset_ids"], ensure_ascii=False),
        scene_asset_id=scene_asset_id,
        scene_group_id=consistency.get("scene_group_id", scene_meta.get("scene_group_key", scene_asset_id)),
        consistency_context=consistency.get("consistency_context", ""),
        reference_weights=json.dumps(consistency.get("reference_weights", {}), ensure_ascii=False),
        continuity_profile=json.dumps(
            {
                **consistency.get("continuity_profile", {}),
                **({"timing": shot.get("timing")} if shot.get("timing") else {}),
            },
            ensure_ascii=False,
        ),
        continuity_reference_path="",
        pose_reference_path="",
        depth_reference_path="",
        last_frame_path="",
        storyboard_path="",
        storyboard_status="pending",
        visual_notes=shot.get("visual_notes", ""),
    )


async def _persist_shot_update(project_id: str, shot: dict) -> None:
    db = SessionLocal()
    try:
        shot_id = shot.get("shot_id") or shot.get("id")
        db_shot = db.query(ShotModel).filter(ShotModel.id == shot_id, ShotModel.project_id == project_id).first()
        if db_shot:
            db_shot.image_path = shot.get("image_path", db_shot.image_path)
            db_shot.storyboard_path = shot.get("storyboard_path") or shot.get("image_path", db_shot.storyboard_path)
            db_shot.audio_path = shot.get("audio_path", db_shot.audio_path)
            db_shot.status = shot.get("status", db_shot.status)
            db_shot.storyboard_status = shot.get("storyboard_status", db_shot.storyboard_status)
            db_shot.version = shot.get("version", db_shot.version)
            db_shot.visual_notes = shot.get("visual_notes", db_shot.visual_notes)
            db.commit()
        await ws_manager.send_to_project(
            project_id,
            {
                "type": "shot_update",
                "shot_id": shot_id,
                "status": shot.get("status", "done"),
                "image_path": shot.get("image_path", ""),
                "storyboard_path": shot.get("storyboard_path") or shot.get("image_path", ""),
                "audio_path": shot.get("audio_path", ""),
            },
        )
    finally:
        db.close()


async def _progress(project_id: str, step: str, progress: int, message: str, report: dict | None = None) -> None:
    # 两个键都写一次：只有真正持有当前 run token 的那个会成功，另一个会被静默忽略，
    # 因此不需要在这里判断本次是 manual 还是 auto。
    for key in (f"project:{project_id}:pipeline:auto", f"project:{project_id}:pipeline:manual"):
        update_job_progress(key, progress, current_step=step, message=message, report=report)
    await ws_manager.send_to_project(project_id, {"type": "progress", "step": step, "progress": progress, "message": message})


def _mark_project_error(db, project_id: str) -> None:
    project = db.query(Project).filter(Project.id == project_id).first()
    if project:
        project.status = "error"
        db.commit()


def _script_title(script: str) -> str:
    for line in script.splitlines():
        clean = line.strip()
        if clean.startswith("标题"):
            return clean.replace("标题：", "").replace("标题:", "").strip()[:30] or "未命名项目"
    return (script.strip().splitlines()[0] if script.strip() else "未命名项目")[:30]
