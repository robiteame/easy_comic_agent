import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api import schemas
from api.websocket import ws_manager
from config import settings
from db import SessionLocal, get_db
from models import Character, Project, SceneAsset, Shot, ShotVersion
from services.image_service import ImageService
from services.invalidation_service import mark_shot_media_stale
from services.reference_readiness_service import (
    mark_reference_failure,
    mark_reference_success,
    refresh_project_reference_state,
)
from services.sample_project_service import create_sample_project as build_sample_project
from services.security import (
    UploadLimitExceeded,
    safe_path,
    save_upload_stream,
    validate_identifier,
    validate_video_upload,
)
from services.skill_config_service import agent_prompt_append, resolve_effective_style, resolve_skill_config
from services.storage_service import StorageQuotaExceeded, StorageService
from services.task_registry import ScopeCancellation, cancel_scopes, release_scope_block
from services.task_registry import claim as claim_task
from services.task_registry import finish as finish_task
from services.task_registry import start as start_task

router = APIRouter(prefix="/api/project", tags=["project"])
storage_service = StorageService()


@dataclass(frozen=True)
class _StagedProjectPath:
    source: Path
    trash: Path


class ProjectCreate(BaseModel):
    title: schemas.ProjectTitle = "未命名项目"
    first_episode_title: schemas.EpisodeTitle = ""
    parent_project_id: schemas.OptionalIdentifier = ""
    project_type: schemas.ProjectType = "series"
    episode_number: schemas.EpisodeNumber = 0
    genre: schemas.Genre = ""
    style: schemas.StyleId = "anime"
    input_text: schemas.ScriptText = ""
    input_type: schemas.InputType = "text"
    output_format: schemas.OutputFormat = "9:16"
    resolution: schemas.Resolution = "1080p"
    platform: schemas.Platform = "douyin"
    target_duration: schemas.TargetDuration = 45


class ProjectUpdate(BaseModel):
    title: schemas.OptionalTitle | None = None
    parent_project_id: schemas.OptionalIdentifier | None = None
    project_type: schemas.ProjectType | None = None
    episode_number: schemas.EpisodeNumber | None = None
    genre: schemas.Genre | None = None
    style: schemas.StyleId | None = None
    output_format: schemas.OutputFormat | None = None
    resolution: schemas.Resolution | None = None
    platform: schemas.Platform | None = None
    target_duration: schemas.TargetDuration | None = None


@router.post("")
async def create_project(data: ProjectCreate, db: Session = Depends(get_db)):
    payload = data.model_dump()
    first_episode_title = payload.pop("first_episode_title", "")
    project_type, parent_id, episode_number = _resolve_project_tree(
        db,
        project_type=payload.get("project_type", "series"),
        parent_project_id=payload.get("parent_project_id", ""),
        episode_number=payload.get("episode_number", 0),
    )
    payload["project_type"] = project_type
    payload["parent_project_id"] = parent_id
    payload["episode_number"] = episode_number
    project = Project(id=str(uuid.uuid4()), **payload)
    db.add(project)
    first_episode = None
    if project.project_type == "series":
        first_episode = Project(
            id=str(uuid.uuid4()),
            title=first_episode_title.strip() or "第 1 集",
            parent_project_id=project.id,
            project_type="episode",
            episode_number=1,
            genre=project.genre,
            style=project.style,
            input_text="",
            input_type="text",
            output_format=project.output_format,
            resolution=project.resolution,
            platform=project.platform,
            target_duration=project.target_duration,
            consistency_config=project.consistency_config,
        )
        db.add(first_episode)
    db.commit()
    db.refresh(project)
    if first_episode:
        db.refresh(first_episode)
    parent_titles = {project.id: project.title}
    result = _serialize_project(project, parent_titles)
    if first_episode:
        result["first_episode"] = _serialize_project(first_episode, parent_titles)
    return result


@router.post("/sample")
async def create_sample_project(db: Session = Depends(get_db)):
    """创建内置示例项目：固定剧本直接落库 + PIL 占位图，不调用任何外部 API。

    返回形状与 POST /api/project 一致（series + first_episode），前端可直接
    按 first_episode 打开工作台浏览分镜与故事板。
    """
    try:
        return await build_sample_project(db)
    except OSError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="示例项目素材写入失败，请检查磁盘空间") from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"示例项目创建失败: {exc}") from exc


@router.get("/{project_id}")
async def get_project(project_id: str, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    if project.status == "deleting":
        raise HTTPException(status_code=409, detail="项目正在删除")
    _sync_completed_status(db, [project])
    result = _serialize_project(project, _parent_titles(db, [project]))
    result.update(
        resolve_effective_style(project.style or "anime", resolve_skill_config(project_id, db), "storyboard_agent")
    )
    return result


@router.put("/{project_id}")
async def update_project(project_id: str, data: ProjectUpdate, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    changed = data.model_dump(exclude_unset=True)
    generation_fields = {"style", "output_format", "resolution", "target_duration"}
    generation_changed = any(
        key in generation_fields and getattr(project, key) != value for key, value in changed.items()
    )
    tree_fields = {"project_type", "parent_project_id", "episode_number"}
    if tree_fields & changed.keys():
        next_type, next_parent, next_number = _resolve_project_tree(
            db,
            project_type=changed.get("project_type", project.project_type or "series"),
            parent_project_id=changed.get("parent_project_id", project.parent_project_id or ""),
            episode_number=changed.get("episode_number", project.episode_number or 0),
            project_id=project.id,
        )
        child_episodes = _episode_count(db, project.id)
        if next_type != "series" and child_episodes:
            raise HTTPException(
                status_code=409,
                detail=f"该项目下已有 {child_episodes} 集剧集，不能改为剧集类型",
            )
        if next_parent and _is_descendant(db, project.id, next_parent):
            raise HTTPException(status_code=400, detail="不能将项目移动到自己的子项目下")
        if next_parent != (project.parent_project_id or ""):
            changed["parent_project_id"] = next_parent
        if next_type != project.project_type or "project_type" in changed:
            changed["project_type"] = next_type
        if next_number != (project.episode_number or 0) or "episode_number" in changed:
            changed["episode_number"] = next_number
    for key, value in changed.items():
        setattr(project, key, value)
    if generation_changed:
        _invalidate_project_generation(db, project)
        if "style" in changed:
            # 风格切换使旧的角色三视图/场景基准图全部失效（标记 stale、清空引用，
            # 不删除文件），需要通过「重建资产」或重新解析按新风格重生成。
            # 剧集项目例外：资产归父项目所有且多集共享，单集切换风格不得清空。
            _invalidate_assets_for_style_change(db, project)
    project.updated_at = datetime.utcnow()
    db.commit()
    if generation_changed:
        await cancel_scopes({f"project:{project_id}"}, "project generation settings changed")
    style_meta = resolve_effective_style(
        project.style or "anime", resolve_skill_config(project_id, db), "storyboard_agent"
    )
    assets_stale = bool("style" in changed) and not project.parent_project_id
    return {
        "id": project_id,
        "status": "updated",
        **style_meta,
        "assets_stale": assets_stale,
        "shots_media_stale": bool(generation_changed),
        "asset_rebuild_endpoint": f"/api/project/{project_id}/assets/rebuild" if assets_stale else "",
    }


_asset_rebuild_tasks: set = set()


@router.post("/{project_id}/assets/rebuild")
async def rebuild_project_assets(project_id: str, db: Session = Depends(get_db)):
    """按项目当前生效风格重建角色三视图与场景基准图。

    风格切换后旧资产已标记 stale；本端点是明确的重建路径——逐个按当前
    effective style 重新生成参考图并置回 active。旧文件不删除，仅解除引用。
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    task_key = f"project:{project_id}:assets-rebuild"
    if not claim_task(
        task_key,
        f"project:{project_id}",
        current_step="rebuild_assets",
        message="已排队，准备按当前风格重建资产",
    ):
        return {"status": "assets_rebuilding", "project_id": project_id, "deduplicated": True}
    try:
        db.commit()
        task = start_task(task_key, _run_asset_rebuild(project_id))
    except BaseException as exc:
        finish_task(task_key, "failed", f"asset rebuild scheduling failed: {exc}")
        raise
    _asset_rebuild_tasks.add(task)
    task.add_done_callback(_asset_rebuild_tasks.discard)
    return {"status": "assets_rebuild_started", "project_id": project_id}


async def _run_asset_rebuild(project_id: str) -> None:
    """后台重建 stale 资产：角色三视图并发(3)、场景基准图串行。"""
    import asyncio
    import hashlib

    image_service = ImageService()

    def _load():
        db = SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                return None, None, [], []
            skill_config = resolve_skill_config(project_id, db)
            asset_project_id = project.parent_project_id or project.id
            style = resolve_effective_style(project.style or "anime", skill_config, "storyboard_agent")[
                "effective_style"
            ]
            characters = (
                db.query(Character)
                .filter(Character.project_id == asset_project_id, Character.asset_status == "stale")
                .all()
            )
            scenes = (
                db.query(SceneAsset)
                .filter(SceneAsset.project_id == asset_project_id, SceneAsset.asset_status == "stale")
                .all()
            )
            return (
                style,
                skill_config,
                [{"id": c.id, "index": index, "name": c.name} for index, c in enumerate(characters)],
                [{"id": s.id, "index": index, "name": s.name} for index, s in enumerate(scenes)],
            )
        finally:
            db.close()

    async def _fetch_character_payload(item: dict) -> dict | None:
        db = SessionLocal()
        try:
            character = db.query(Character).filter(Character.id == item["id"]).first()
            if not character:
                return None
            return {
                "id": character.id,
                "name": character.name,
                "visual_prompt": character.visual_prompt or "",
                "personality": character.personality or "",
                "negative_prompt": character.negative_prompt or "",
                "appearance": json.loads(character.appearance) if character.appearance else {},
                "key_features": json.loads(character.key_features) if character.key_features else [],
                "seed": int(character.seed) if character.seed and character.seed.isdigit() else 42,
            }
        finally:
            db.close()

    style, skill_config, characters, scenes = _load()
    if style is None:
        finish_task(f"project:{project_id}:assets-rebuild", "failed", "项目不存在")
        return
    await ws_manager.send_to_project(
        project_id,
        {"type": "progress", "step": "rebuild_assets", "progress": 5, "message": "开始按当前风格重建资产"},
    )
    skill_append = agent_prompt_append(skill_config, "storyboard_agent")
    fingerprint = hashlib.sha256(str(style).encode()).hexdigest()[:16]
    failures: list[str] = []
    rebuilt_characters = 0
    rebuilt_scenes = 0
    semaphore = asyncio.Semaphore(3)

    async def rebuild_character(item: dict) -> bool:
        async with semaphore:
            payload = await _fetch_character_payload(item)
            if payload is None:
                return False
            if skill_append:
                payload["visual_prompt"] = ", ".join(part for part in [payload["visual_prompt"], skill_append] if part)
            ref_path = await image_service.generate_character_reference(
                character=payload,
                style=style,
                project_id=project_id,
                seed=int(payload["seed"]) + 7000 + int(item["index"]),
            )
            db = SessionLocal()
            try:
                row = db.query(Character).filter(Character.id == item["id"]).first()
                if row:
                    mark_reference_success(db, "character", row, ref_path)
                    row.asset_status = "active"
                    row.style_fingerprint = fingerprint
                    db.commit()
                return True
            finally:
                db.close()

    results = await asyncio.gather(*(rebuild_character(item) for item in characters), return_exceptions=True)
    for item, result in zip(characters, results, strict=True):
        if isinstance(result, BaseException):
            failures.append(f"角色 {item['name']}: {result}")
            db = SessionLocal()
            try:
                row = db.query(Character).filter(Character.id == item["id"]).first()
                if row:
                    mark_reference_failure(db, "character", row, result)
                    db.commit()
            finally:
                db.close()
        elif result:
            rebuilt_characters += 1

    for item in scenes:
        try:
            db = SessionLocal()
            try:
                scene_row = db.query(SceneAsset).filter(SceneAsset.id == item["id"]).first()
                if not scene_row:
                    continue
                scene_payload = {
                    "id": scene_row.id,
                    "name": scene_row.name,
                    "location": scene_row.name,
                    "visual_prompt": scene_row.visual_prompt or "",
                    "actions": scene_row.description or "",
                    "seed": scene_row.seed or 1200,
                }
            finally:
                db.close()
            if skill_append:
                scene_payload["visual_prompt"] = ", ".join(
                    part for part in [scene_payload["visual_prompt"], skill_append] if part
                )
            ref_path = await image_service.generate_scene_baseline_reference(
                scene=scene_payload,
                style=style,
                project_id=project_id,
                seed=int(scene_payload["seed"]) + int(item["index"]),
            )
            db = SessionLocal()
            try:
                row = db.query(SceneAsset).filter(SceneAsset.id == item["id"]).first()
                if row:
                    row.baseline_image_path = ref_path
                    mark_reference_success(db, "character", row, ref_path)
                    row.asset_status = "active"
                    row.style_fingerprint = fingerprint
                    db.commit()
            finally:
                db.close()
            rebuilt_scenes += 1
        except Exception as exc:
            failures.append(f"场景 {item['name']}: {exc}")
            db = SessionLocal()
            try:
                row = db.query(SceneAsset).filter(SceneAsset.id == item["id"]).first()
                if row:
                    mark_reference_failure(db, "scene", row, exc)
                    db.commit()
            finally:
                db.close()

    report_db = SessionLocal()
    try:
        consistency_report = refresh_project_reference_state(report_db, project_id)
    finally:
        report_db.close()
    status = "failed" if failures else "completed"
    finish_task(
        f"project:{project_id}:assets-rebuild",
        status,
        "；".join(failures)[:500] or "资产重建完成",
        report=consistency_report,
    )
    await ws_manager.send_to_project(
        project_id,
        {
            "type": "assets_rebuilt",
            "project_id": project_id,
            "style": style,
            "rebuilt_characters": rebuilt_characters,
            "rebuilt_scenes": rebuilt_scenes,
            "failures": failures,
            "consistency_report": consistency_report,
        },
    )


@router.delete("/{project_id}")
async def delete_project(project_id: str, db: Session = Depends(get_db)):
    try:
        validate_identifier(project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    delete_ids = _descendant_ids(db, project.id)
    previous_statuses = {
        target_id: status
        for target_id, status in db.query(Project.id, Project.status).filter(Project.id.in_(delete_ids)).all()
    }
    shot_ids = [shot_id for (shot_id,) in db.query(Shot.id).filter(Shot.project_id.in_(delete_ids)).all()]
    db.query(Project).filter(Project.id.in_(delete_ids)).update(
        {Project.status: "deleting", Project.updated_at: datetime.utcnow()},
        synchronize_session=False,
    )
    # End the read transaction before the registry takes its SQLite write lock.
    # Keep already-loaded identity values usable for callers after the bulk
    # deletion, matching SQLAlchemy's normal synchronized-delete behavior.
    expire_on_commit = db.expire_on_commit
    db.expire_on_commit = False
    db.commit()
    scopes = {f"project:{target_id}" for target_id in delete_ids}
    scopes.update(f"shot:{shot_id}" for shot_id in shot_ids)
    cancellation: ScopeCancellation | None = None
    staged_paths: list[_StagedProjectPath] = []
    trash_token = uuid.uuid4().hex
    try:
        cancellation = await cancel_scopes(scopes, "project was deleted", keep_blocked=True)
        if not isinstance(cancellation, ScopeCancellation):
            raise RuntimeError("项目任务作用域锁定失败")

        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return {"status": "deleted", "deleted_project_ids": delete_ids, "cleared_output_project_ids": []}

        staged_paths = _stage_project_paths(delete_ids, trash_token, previous_statuses)

        for target_id in delete_ids:
            # 版本快照随镜头一并清理（append-only 只限制改写，不限制项目级清理）。
            db.query(ShotVersion).filter(ShotVersion.project_id == target_id).delete()
            db.query(Shot).filter(Shot.project_id == target_id).delete()
            db.query(Character).filter(Character.project_id == target_id).delete()
            db.query(SceneAsset).filter(SceneAsset.project_id == target_id).delete()

        deleted_files = [
            path.source.relative_to((settings.OUTPUT_DIR / "projects").resolve()).parts[0]
            for path in staged_paths
            if path.source.parent == (settings.OUTPUT_DIR / "projects").resolve()
        ]

        db.query(Project).filter(Project.id.in_(delete_ids)).delete()
        db.commit()
        _cleanup_staged_project_paths(staged_paths, trash_token)
        return {"status": "deleted", "deleted_project_ids": delete_ids, "cleared_output_project_ids": deleted_files}
    except BaseException:
        db.rollback()
        _restore_staged_project_paths(staged_paths)
        _remove_delete_manifest(trash_token)
        _prune_trash_parents(staged_paths)
        _prune_empty_trash_roots()
        _restore_project_statuses(previous_statuses)
        raise
    finally:
        if cancellation is not None:
            release_scope_block(cancellation)
        db.expire_on_commit = expire_on_commit


@router.post("/{project_id}/import-video")
async def import_final_video(project_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        validate_identifier(project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    suffix = Path(file.filename or "final.mp4").suffix.lower()
    if suffix not in {".mp4", ".m4v"}:
        raise HTTPException(status_code=400, detail="请上传 mp4 或 m4v 视频文件")
    mime = getattr(file, "content_type", None)

    candidate: Path | None = None
    cancellation: ScopeCancellation | None = None
    try:
        output_root = (settings.OUTPUT_DIR / "projects").resolve()
        output_dir = safe_path(output_root, project_id, "output", create_parent=True)
        target = output_dir / "final.mp4"
        candidate = output_dir / f".final-{uuid.uuid4().hex}.candidate"
        # Keep an existing final video intact until the replacement passes
        # signature checks, so the temporary candidate must fit alongside it.
        available = storage_service.ensure_project_capacity(project_id, replacing=target)
        size = await save_upload_stream(file, candidate, min(settings.MAX_VIDEO_UPLOAD_BYTES, available))
    except UploadLimitExceeded as exc:
        raise HTTPException(status_code=413, detail="上传视频超过大小或项目存储配额") from exc
    except StorageQuotaExceeded as exc:
        raise HTTPException(status_code=413, detail="项目媒体存储空间不足") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        if candidate is not None:
            candidate.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="保存视频失败") from exc
    if size <= 1024:
        candidate.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上传的视频文件为空或过小")
    try:
        validate_video_upload(candidate, mime)
    except (OSError, ValueError):
        candidate.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上传文件不是有效的 MP4 视频") from None

    # Hold the project scope while promoting the candidate. This cancels and
    # waits for an in-flight render before it can publish over the import.
    db.rollback()
    try:
        cancellation = await cancel_scopes(
            {f"project:{project_id}"},
            "final video was imported",
            keep_blocked=True,
        )
        if not isinstance(cancellation, ScopeCancellation):
            raise RuntimeError("项目渲染作用域锁定失败")
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project or project.status == "deleting":
            raise HTTPException(status_code=409, detail="项目正在删除或已不存在")
        output_dir = safe_path((settings.OUTPUT_DIR / "projects").resolve(), project_id, "output", create_parent=True)
        target = output_dir / "final.mp4"
        # Recheck against the post-upload tree while the project blocker is
        # held. The first quota check ran before the candidate was written and
        # could race with a render or another media writer.
        usage = storage_service.project_usage_bytes(project_id)
        replaced_bytes = target.stat().st_size if target.is_file() else 0
        if usage - replaced_bytes > int(settings.PROJECT_STORAGE_QUOTA_BYTES):
            raise HTTPException(status_code=413, detail="项目媒体存储空间不足")
    except BaseException:
        candidate.unlink(missing_ok=True)
        if cancellation is not None:
            release_scope_block(cancellation)
            cancellation = None
        raise

    backup = target.with_name(f".final-{uuid.uuid4().hex}.previous")
    had_previous = target.exists()
    try:
        if had_previous:
            os.replace(target, backup)
        os.replace(candidate, target)
        project.status = "completed"
        project.updated_at = datetime.utcnow()
        db.commit()
    except BaseException as exc:
        db.rollback()
        target.unlink(missing_ok=True)
        if had_previous and backup.exists():
            os.replace(backup, target)
        elif backup.exists():
            backup.unlink(missing_ok=True)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail="发布视频失败") from exc
    finally:
        if cancellation is not None:
            release_scope_block(cancellation)
    try:
        backup.unlink(missing_ok=True)
    except OSError:
        pass
    return {**_serialize_project(project, _parent_titles(db, [project])), "imported": True}


@router.get("")
async def list_projects(db: Session = Depends(get_db)):
    projects = db.query(Project).order_by(Project.updated_at.desc()).all()
    _sync_completed_status(db, projects)
    parent_titles = _parent_titles(db, projects)
    return [_serialize_project(project, parent_titles) for project in projects]


@router.get("/{project_id}/episodes")
async def list_episodes(project_id: str, db: Session = Depends(get_db)):
    episodes = (
        db.query(Project)
        .filter(Project.parent_project_id == project_id, Project.project_type == "episode")
        .order_by(Project.episode_number.asc(), Project.created_at.asc())
        .all()
    )
    _sync_completed_status(db, episodes)
    parent_titles = _parent_titles(db, episodes)
    return [_serialize_project(project, parent_titles) for project in episodes]


def _serialize_project(project: Project, parent_titles: dict[str, str] | None = None) -> dict:
    video_path = _final_video_path(project.id)
    parent_title = (parent_titles or {}).get(project.parent_project_id or "", "")
    return {
        "id": project.id,
        "parent_project_id": project.parent_project_id or "",
        "parent_project_title": parent_title,
        "project_type": project.project_type or "series",
        "episode_number": project.episode_number or 0,
        "title": project.title,
        "genre": project.genre,
        "style": project.style,
        "status": project.status,
        "video_path": (
            f"/output/projects/{project.id}/output/final.mp4"
            if project.status == "completed" and _has_file(video_path)
            else ""
        ),
        "input_text": project.input_text,
        "output_format": project.output_format,
        "resolution": project.resolution,
        "platform": project.platform,
        "target_duration": project.target_duration,
        "timing_plan": json.loads(project.timing_plan) if project.timing_plan else {},
        "consistency_config": json.loads(project.consistency_config) if project.consistency_config else {},
        "consistency_report": json.loads(project.consistency_report) if project.consistency_report else {},
        "is_sample": bool(project.is_sample),
        "created_at": project.created_at.isoformat(),
        "updated_at": project.updated_at.isoformat(),
    }


PROJECT_TYPES = ("series", "episode")


def _resolve_project_tree(
    db: Session,
    *,
    project_type: str,
    parent_project_id: str,
    episode_number: int,
    project_id: str | None = None,
) -> tuple[str, str, int]:
    """校验并归一化项目树字段，返回 (project_type, parent_project_id, episode_number)。

    不变量：series 是根节点（无父级），episode 必须挂在真实存在的 series 下，
    禁止自引用与未知类型，集号不能为负；集号为 0 时自动取下一个可用集号。
    """

    normalized_type = str(project_type or "").strip()
    if normalized_type not in PROJECT_TYPES:
        raise HTTPException(status_code=400, detail=f"未知的项目类型: {normalized_type or '(空)'}")
    parent_id = str(parent_project_id or "").strip()
    if project_id and parent_id and parent_id == project_id:
        raise HTTPException(status_code=400, detail="项目不能成为自己的父项目")
    if normalized_type == "series":
        if parent_id:
            raise HTTPException(status_code=400, detail="大项目不能挂在其他项目下")
        return "series", "", 0

    if not parent_id:
        raise HTTPException(status_code=400, detail="剧集必须指定所属大项目")
    try:
        validate_identifier(parent_id, "父项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    parent = db.query(Project).filter(Project.id == parent_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="父项目不存在")
    if parent.status == "deleting":
        raise HTTPException(status_code=409, detail="父项目正在删除")
    if (parent.project_type or "series") != "series":
        raise HTTPException(status_code=400, detail="剧集只能挂在系列项目下")

    number = int(episode_number or 0)
    if number < 0:
        raise HTTPException(status_code=400, detail="集号不能为负数")
    if number == 0:
        number = _next_episode_number(db, parent_id)
    return "episode", parent_id, number


def _episode_count(db: Session, project_id: str) -> int:
    """该项目直属的剧集数量，用于阻止会把剧集变成孤儿的类型变更。"""

    return db.query(Project).filter(Project.parent_project_id == project_id, Project.project_type == "episode").count()


def _next_episode_number(db: Session, parent_project_id: str) -> int:
    if not parent_project_id:
        return 1
    existing = (
        db.query(Project)
        .filter(Project.parent_project_id == parent_project_id, Project.project_type == "episode")
        .order_by(Project.episode_number.desc())
        .first()
    )
    return int(existing.episode_number or 0) + 1 if existing else 1


def _final_video_path(project_id: str) -> Path:
    return settings.OUTPUT_DIR / "projects" / project_id / "output" / "final.mp4"


def _stage_project_paths(
    project_ids: list[str], token: str, previous_statuses: dict[str, str]
) -> list[_StagedProjectPath]:
    """Move project trees into same-filesystem trash before deleting rows."""

    staged: list[_StagedProjectPath] = []
    roots = ((settings.OUTPUT_DIR / "projects").resolve(), (settings.DATA_DIR / "uploads").resolve())
    try:
        _write_delete_manifest(token, project_ids, previous_statuses)
        for root in roots:
            for project_id in dict.fromkeys(project_ids):
                try:
                    validate_identifier(project_id, "项目 ID")
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                source = (root / project_id).resolve()
                if root not in source.parents:
                    raise HTTPException(status_code=400, detail="项目资产路径异常，已拒绝删除")
                if not source.exists():
                    continue
                trash = (root / ".trash" / token / project_id).resolve()
                trash.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, trash)
                staged.append(_StagedProjectPath(source=source, trash=trash))
    except BaseException:
        _restore_staged_project_paths(staged)
        _remove_delete_manifest(token)
        _prune_trash_parents(staged)
        raise
    return staged


def _restore_staged_project_paths(staged: list[_StagedProjectPath]) -> None:
    for item in reversed(staged):
        if not item.trash.exists():
            continue
        item.source.parent.mkdir(parents=True, exist_ok=True)
        if item.source.exists():
            raise RuntimeError(f"无法恢复项目目录，目标已存在: {item.source}")
        os.replace(item.trash, item.source)
    _prune_trash_parents(staged)


def _cleanup_staged_project_paths(staged: list[_StagedProjectPath], token: str) -> None:
    for item in staged:
        shutil.rmtree(item.trash, ignore_errors=True)
    roots = {
        (settings.OUTPUT_DIR / "projects").resolve(),
        (settings.DATA_DIR / "uploads").resolve(),
    }
    for root in roots:
        trash_dir = root / ".trash" / token
        shutil.rmtree(trash_dir, ignore_errors=True)
    _remove_delete_manifest(token)
    _prune_trash_parents(staged)


def _prune_trash_parents(staged: list[_StagedProjectPath]) -> None:
    for item in staged:
        for directory in (item.trash.parent, item.trash.parent.parent):
            try:
                directory.rmdir()
            except OSError:
                pass


def _delete_manifest_path(token: str) -> Path:
    return (settings.OUTPUT_DIR / "projects" / ".trash" / token / "manifest.json").resolve()


def _write_delete_manifest(token: str, project_ids: list[str], previous_statuses: dict[str, str]) -> None:
    manifest = _delete_manifest_path(token)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(f".{manifest.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "version": 1,
        "project_ids": list(dict.fromkeys(project_ids)),
        "previous_statuses": previous_statuses,
    }
    try:
        # fsync requires a writable handle; a read-only descriptor raises EBADF
        # on Windows, so the manifest is written and flushed in one open call.
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_delete_manifest(token: str) -> None:
    manifest = _delete_manifest_path(token)
    manifest.unlink(missing_ok=True)
    try:
        manifest.parent.rmdir()
    except OSError:
        pass


def recover_staged_project_deletions() -> int:
    """Recover or finish project deletions interrupted by process death."""

    trash_root = (settings.OUTPUT_DIR / "projects" / ".trash").resolve()
    if not trash_root.exists():
        return 0
    from db import SessionLocal

    recovered = 0
    db = SessionLocal()
    try:
        for manifest in sorted(trash_root.glob("*/manifest.json")):
            token = manifest.parent.name
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                project_ids = [validate_identifier(item, "项目 ID") for item in payload.get("project_ids", [])]
                previous_statuses = {
                    validate_identifier(key, "项目 ID"): str(value)
                    for key, value in (payload.get("previous_statuses") or {}).items()
                }
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            existing = {
                project_id: status
                for project_id, status in db.query(Project.id, Project.status).filter(Project.id.in_(project_ids)).all()
            }
            if existing:
                staged: list[_StagedProjectPath] = []
                for root in ((settings.OUTPUT_DIR / "projects").resolve(), (settings.DATA_DIR / "uploads").resolve()):
                    for project_id in project_ids:
                        source = root / project_id
                        trash = root / ".trash" / token / project_id
                        if trash.exists() and not source.exists():
                            source.parent.mkdir(parents=True, exist_ok=True)
                            os.replace(trash, source)
                            staged.append(_StagedProjectPath(source=source, trash=trash))
                for project_id, status in previous_statuses.items():
                    project = db.query(Project).filter(Project.id == project_id).first()
                    if project and project.status == "deleting":
                        project.status = status
                db.commit()
                _remove_delete_manifest(token)
                shutil.rmtree(manifest.parent, ignore_errors=True)
                _prune_empty_trash_roots()
                recovered += 1
            else:
                # The row deletion committed before the process died; the
                # database is authoritative, so only discard staged files.
                for root in ((settings.OUTPUT_DIR / "projects").resolve(), (settings.DATA_DIR / "uploads").resolve()):
                    shutil.rmtree(root / ".trash" / token, ignore_errors=True)
                _prune_empty_trash_roots()
                recovered += 1
        db.commit()
    finally:
        db.close()
    return recovered


def _prune_empty_trash_roots() -> None:
    for root in (
        (settings.OUTPUT_DIR / "projects" / ".trash").resolve(),
        (settings.DATA_DIR / "uploads" / ".trash").resolve(),
    ):
        if root.exists():
            # Remove only empty token/parent directories. Any unknown file or
            # manifest remains for a later explicit recovery decision.
            for child in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if child.is_dir():
                    try:
                        child.rmdir()
                    except OSError:
                        pass
        try:
            root.rmdir()
        except OSError:
            pass


def _restore_project_statuses(previous_statuses: dict[str, str]) -> None:
    if not previous_statuses:
        return
    restore_db = None
    try:
        from db import SessionLocal

        restore_db = SessionLocal()
        for project_id, status in previous_statuses.items():
            project = restore_db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = status
        restore_db.commit()
    except Exception:
        if restore_db is not None:
            restore_db.rollback()
    finally:
        if restore_db is not None:
            restore_db.close()


def _descendant_ids(db: Session, root_id: str) -> list[str]:
    """Return a project and nested descendants using one indexed query."""

    children_by_parent: dict[str, list[str]] = {}
    for child_id, parent_id in db.query(Project.id, Project.parent_project_id).all():
        children_by_parent.setdefault(parent_id or "", []).append(child_id)
    result: list[str] = []
    seen: set[str] = set()
    pending = [root_id]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        result.append(current)
        pending.extend(children_by_parent.get(current, ()))
    return result


def _is_descendant(db: Session, root_id: str, candidate_id: str) -> bool:
    return candidate_id in set(_descendant_ids(db, root_id)[1:])


def _parent_titles(db: Session, projects: list[Project]) -> dict[str, str]:
    parent_ids = {project.parent_project_id for project in projects if project.parent_project_id}
    if not parent_ids:
        return {}
    return dict(db.query(Project.id, Project.title).filter(Project.id.in_(parent_ids)).all())


def _has_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _sync_completed_status(db: Session, projects: list[Project]) -> None:
    changed = False
    nonfinal_statuses = {
        "draft",
        "pending",
        "assets_ready",
        "storyboard_generating",
        "storyboard_ready",
        "storyboard_approved",
        "rendering",
        "error",
        "failed",
        "cancelled",
        "interrupted",
        "deleting",
    }
    for project in projects:
        if (
            project.status not in nonfinal_statuses
            and project.status != "completed"
            and _has_file(_final_video_path(project.id))
        ):
            project.status = "completed"
            project.updated_at = datetime.utcnow()
            changed = True
    if changed:
        db.commit()


def _invalidate_project_generation(db: Session, project: Project) -> None:
    """生成配置（画风 / 画幅 / 分辨率）变更：全部镜头标记待重生成，素材保留。

    旧故事板、视频、配音与尾帧继续保留引用，用户可预览、对比并通过版本
    历史回滚；新素材生成成功后由写回原子替换。版本号 +1 隔离在途任务。
    """
    project.status = "assets_ready"
    for shot in db.query(Shot).filter(Shot.project_id == project.id).all():
        mark_shot_media_stale(shot)
        shot.version = (shot.version or 1) + 1


def _invalidate_assets_for_style_change(db: Session, project: Project) -> None:
    """风格切换后旧参考资产标记 stale：清空引用、保留文件。

    不猜测旧资产原本是哪种风格：指纹与切换后的风格必然不一致，直接失效，
    等待用户走「重建资产」或重新解析按当前风格重生成。

    只处理项目自有资产；剧集（有父项目）的资产由父项目持有、多集共享，
    单集风格切换不得清空共享资产（其它剧集仍在使用），本集镜头的 stale
    标记已由 ``_invalidate_project_generation`` 完成。
    """
    if project.parent_project_id:
        return
    asset_project_id = project.id
    for character in db.query(Character).filter(Character.project_id == asset_project_id).all():
        character.asset_status = "stale"
        character.reference_status = "stale"
        character.reference_failure_reason = "画风已切换，参考素材需按新画风重建"
        character.reference_images = "[]"
        character.lora_profile = ""
        character.ip_adapter_profile = ""
    for scene in db.query(SceneAsset).filter(SceneAsset.project_id == asset_project_id).all():
        scene.asset_status = "stale"
        scene.reference_status = "stale"
        scene.reference_failure_reason = "画风已切换，参考素材需按新画风重建"
        scene.baseline_image_path = ""
        scene.reference_images = "[]"
