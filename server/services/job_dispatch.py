"""把「重试 / 续跑」重新派发回现有的任务入口。

关键点：这里不重新实现任何业务步骤，而是复用 ``api.routes.script`` /
``api.routes.shot`` / ``api.routes.render`` 里已经被手动模式、全自动模式、前端
按钮共用的那批入口函数。抢占（claim）、作用域互斥、run token 与幂等键语义全部
沿用 ``services.task_registry``，因此重试天然满足：

- 同一作用域不会同时跑两个互斥任务；
- 已经完成且仍然有效的阶段 / 产物会被跳过（剧本解析结果、已生成配音、镜头视频）；
- 上一次尝试会被归档为 ``key#attempt-N``，历史失败记录与 attempt 全部保留；
- 新尝试拿到新的 run token，旧尝试的迟到回调无法覆盖它。

两种模式的区别：

- **重试（retry）**：重新执行该任务对应的操作。剧本解析 / 分镜这类「批次」任务
  会重做整批未确认镜头，单镜头任务会重新生成目标产物；但已经成功完成的上游阶段
  （解析结果、有效配音、已生成的镜头视频）不会被重复执行。
- **续跑（resume）**：只补「缺失或损坏」的中间产物。如果目标产物已经存在，返回
  明确错误（job_not_resumable），而不是静默从头重跑。
"""

from __future__ import annotations

from dataclasses import dataclass

from config import settings
from db import SessionLocal
from models import BackgroundJob, Project, Shot
from services.job_types import (
    ERROR_CODE_NOT_FOUND,
    ERROR_CODE_NOT_RESUMABLE,
    ERROR_CODE_NOT_RETRYABLE,
    ERROR_CODE_UNSUPPORTED,
    JOB_TYPE_RENDER,
    JOB_TYPE_SCRIPT_PIPELINE,
    JOB_TYPE_SHOT_IMAGE,
    JOB_TYPE_SHOT_VIDEO,
    JOB_TYPE_STORYBOARD,
    parse_job_key,
)
from services.security import existing_file

# 媒体有效性的最低大小：与既有渲染 / 视频复用判断保持一致。
_MEDIA_MIN_BYTES = 1024
_VIDEO_MIN_BYTES = 4096


@dataclass(frozen=True)
class DispatchResult:
    status: str  # started | deduplicated | rejected
    message: str
    job_key: str = ""
    error_code: str = ""

    @property
    def started(self) -> bool:
        return self.status == "started"


def _rejected(code: str, message: str) -> DispatchResult:
    return DispatchResult(status="rejected", message=message, error_code=code)


def _deduplicated(message: str = "该任务已有正在执行的尝试，本次请求已合并") -> DispatchResult:
    return DispatchResult(status="deduplicated", message=message)


def _started(key: str, message: str = "已重新派发") -> DispatchResult:
    return DispatchResult(status="started", message=message, job_key=key)


async def redispatch(job: BackgroundJob, mode: str) -> DispatchResult:
    """按任务类型重新派发一次执行。mode: ``retry`` 或 ``resume``。"""

    key = parse_job_key(job.idempotency_key).canonical
    job_type = str(job.job_type or "")
    try:
        if job_type == JOB_TYPE_SCRIPT_PIPELINE:
            return await _dispatch_pipeline(key, mode)
        if job_type == JOB_TYPE_STORYBOARD:
            return await _dispatch_storyboard(key, mode)
        if job_type == JOB_TYPE_SHOT_IMAGE:
            return await _dispatch_shot_image(key, mode)
        if job_type == JOB_TYPE_SHOT_VIDEO:
            return await _dispatch_shot_video(key, mode)
        if job_type == JOB_TYPE_RENDER:
            return await _dispatch_render(key, mode)
    except Exception as exc:  # noqa: BLE001 - 统一转成可读拒绝原因，不把堆栈给前端
        return _rejected(ERROR_CODE_NOT_RETRYABLE, f"重新派发失败：{type(exc).__name__}")
    return _rejected(ERROR_CODE_UNSUPPORTED, "该任务类型暂不支持重新派发")


# --- 剧本解析 / 分镜批次 ---------------------------------------------------


def _pending_storyboard_shots(shots: list[Shot], *, full: bool) -> list[Shot]:
    """需要生成故事板的镜头。

    ``full=True``（完整重试）取全部未确认镜头；``full=False``（续跑）只取
    状态不是 done、或产物缺失/损坏的镜头，绝不重复生成已经完成的镜头。
    """

    pending: list[Shot] = []
    for shot in shots:
        if shot.confirmed:
            continue
        if full:
            pending.append(shot)
        elif shot.storyboard_status != "done" or _reusable_storyboard(shot) is None:
            pending.append(shot)
    return pending


def _reusable_storyboard(shot: Shot) -> str:
    for candidate in (shot.storyboard_path, shot.image_path):
        if not candidate:
            continue
        resolved = existing_file(
            candidate,
            minimum_size=_MEDIA_MIN_BYTES,
            allowed_roots=(settings.OUTPUT_DIR, settings.ASSETS_DIR, settings.DATA_DIR),
        )
        if resolved is not None:
            return str(resolved)
    return ""


async def _dispatch_pipeline(key: str, mode: str) -> DispatchResult:
    from api.routes import script as script_route

    identity = parse_job_key(key)
    project_id = identity.owner_id
    pipeline_mode = identity.qualifier if identity.qualifier in {"manual", "auto"} else "manual"
    db = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project is None:
            return _rejected(ERROR_CODE_NOT_FOUND, "项目不存在或已被删除")
        shots = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
        if shots:
            pending = _pending_storyboard_shots(shots, full=(mode == "retry"))
            if not pending:
                return _rejected(
                    ERROR_CODE_NOT_RESUMABLE if mode == "resume" else ERROR_CODE_NOT_RETRYABLE,
                    "剧本与分镜阶段已经全部完成，没有需要重新执行的步骤",
                )
            return await _start_storyboard([shot.id for shot in pending], db, project_id, key)
        if mode == "resume":
            return _rejected(
                ERROR_CODE_NOT_RESUMABLE,
                "剧本解析阶段没有可复用的中间产物，无法安全续跑；请改用完整重试",
            )
        data = {
            "project_id": project_id,
            "user_input": project.input_text or "",
            "input_type": project.input_type or "text",
            "style": project.style or "anime",
            "output_format": project.output_format or "9:16",
            "resolution": project.resolution or "1080p",
            "platform": project.platform or "douyin",
            "target_duration": 45,
        }
        if not str(data["user_input"]).strip():
            return _rejected(ERROR_CODE_NOT_RETRYABLE, "项目没有保存原始剧本内容，无法重新解析")
        initial_state = script_route._initial_state(data, script_route._resolved_skill_config(project_id))
        task = script_route._spawn_pipeline(
            project_id,
            initial_state,
            pipeline_mode,
            data["output_format"],
            data["resolution"],
        )
        if task is None:
            return _deduplicated()
        return _started(key, "已重新执行剧本解析与分镜生成")
    finally:
        db.close()


async def _start_storyboard(shot_ids: list[str], db, project_id: str, key: str) -> DispatchResult:
    from api.routes import shot as shot_route

    result = await shot_route.generate_storyboard_images(
        project_id,
        shot_route.StoryboardGenerateRequest(shot_ids=list(shot_ids)),
        db,
    )
    if isinstance(result, dict) and result.get("deduplicated"):
        return _deduplicated()
    return _started(key, f"已重新派发 {len(shot_ids)} 个镜头的故事板生成")


# --- 项目级定稿故事板 ------------------------------------------------------


async def _dispatch_storyboard(key: str, mode: str) -> DispatchResult:
    identity = parse_job_key(key)
    project_id = identity.owner_id
    db = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project is None:
            return _rejected(ERROR_CODE_NOT_FOUND, "项目不存在或已被删除")
        shots = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
        if not shots:
            return _rejected(ERROR_CODE_NOT_RETRYABLE, "项目还没有分镜，无法生成故事板")
        pending = _pending_storyboard_shots(shots, full=(mode == "retry"))
        if not pending:
            return _rejected(
                ERROR_CODE_NOT_RESUMABLE if mode == "resume" else ERROR_CODE_NOT_RETRYABLE,
                "所有镜头的定稿故事板都已生成，没有需要重新执行的镜头",
            )
        return await _start_storyboard([shot.id for shot in pending], db, project_id, key)
    finally:
        db.close()


# --- 单镜头故事板 ----------------------------------------------------------


async def _dispatch_shot_image(key: str, mode: str) -> DispatchResult:
    from api.routes import shot as shot_route

    shot_id = parse_job_key(key).owner_id
    db = SessionLocal()
    try:
        shot = db.query(Shot).filter(Shot.id == shot_id).first()
        if shot is None:
            return _rejected(ERROR_CODE_NOT_FOUND, "镜头不存在或已被删除")
        if shot.confirmed and shot.storyboard_status == "done":
            return _rejected(ERROR_CODE_NOT_RESUMABLE, "该镜头故事板已审核通过，请先撤销审核再重新生成")
        if mode == "resume" and _reusable_storyboard(shot):
            return _rejected(ERROR_CODE_NOT_RESUMABLE, "该镜头故事板已存在且可用，无需续跑；如需重新生成请使用重试")
        result = await shot_route.regenerate_shot(shot_id, shot_route.RegenerateRequest(), db)
        if isinstance(result, dict) and result.get("deduplicated"):
            return _deduplicated()
        return _started(key, f"已重新生成镜头 {shot.sequence} 的故事板")
    finally:
        db.close()


# --- 单镜头视频 ------------------------------------------------------------


async def _dispatch_shot_video(key: str, mode: str) -> DispatchResult:
    from api.routes import shot as shot_route

    shot_id = parse_job_key(key).owner_id
    db = SessionLocal()
    try:
        shot = db.query(Shot).filter(Shot.id == shot_id).first()
        if shot is None:
            return _rejected(ERROR_CODE_NOT_FOUND, "镜头不存在或已被删除")
        if not shot.confirmed or not (shot.storyboard_path or shot.image_path):
            return _rejected(ERROR_CODE_NOT_RETRYABLE, "请先为该镜头生成并审核通过定稿故事板")
        reusable_video = shot_route._can_reuse_existing_video(shot, False)
        if mode == "resume" and reusable_video:
            return _rejected(ERROR_CODE_NOT_RESUMABLE, "该镜头视频已存在且可用，无需续跑；如需重新生成请使用重试")
        result = await shot_route.generate_shot_video(
            shot_id,
            shot_route.ShotVideoGenerateRequest(force=mode != "resume", reuse_audio=True),
            db,
        )
        if isinstance(result, dict) and result.get("deduplicated"):
            return _deduplicated()
        return _started(key, f"已重新生成镜头 {shot.sequence} 的视频")
    finally:
        db.close()


# --- 成片渲染 --------------------------------------------------------------


def _final_video(project_id: str) -> str:
    path = settings.OUTPUT_DIR / "projects" / project_id / "output" / "final.mp4"
    resolved = existing_file(path, minimum_size=_MEDIA_MIN_BYTES, allowed_roots=(settings.OUTPUT_DIR,))
    return str(resolved) if resolved is not None else ""


async def _dispatch_render(key: str, mode: str) -> DispatchResult:
    from api.routes import render as render_route

    project_id = parse_job_key(key).owner_id
    db = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project is None:
            return _rejected(ERROR_CODE_NOT_FOUND, "项目不存在或已被删除")
        if mode == "resume" and _final_video(project_id):
            return _rejected(ERROR_CODE_NOT_RESUMABLE, "成片已存在，无需续跑；如需重新导出请使用重试")
        shots = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
        if not shots:
            return _rejected(ERROR_CODE_NOT_RETRYABLE, "没有可导出的镜头")
        if any(not shot.confirmed for shot in shots):
            return _rejected(ERROR_CODE_NOT_RETRYABLE, "仍有镜头故事板未通过人工审核，不能导出成片")
        if any(not (shot.storyboard_path or shot.image_path) for shot in shots):
            return _rejected(ERROR_CODE_NOT_RETRYABLE, "仍有镜头未生成定稿故事板，不能导出成片")
        if any(not shot.video_path for shot in shots):
            return _rejected(ERROR_CODE_NOT_RETRYABLE, "仍有镜头未生成视频，不能导出成片")
        result = await render_route.render_video(
            render_route.RenderRequest(
                project_id=project_id,
                output_format=project.output_format or "9:16",
                resolution=project.resolution or "1080p",
            ),
            db,
        )
        if isinstance(result, dict) and result.get("deduplicated"):
            return _deduplicated()
        return _started(key, "已重新开始导出成片")
    finally:
        db.close()


__all__ = ["DispatchResult", "redispatch"]
