"""任务中心的写操作：取消、重试、续跑、删除。

所有写操作都先把「能不能做、做完是什么状态」算清楚，再调用既有基础设施：

- 取消复用 ``task_registry.cancel()``（同一把作用域锁与 run token 校验），
  重复取消返回幂等结果，不会 500；
- 重试 / 续跑复用 ``services.job_dispatch```，由它调回现有 route 入口；
- 删除只允许终态任务，运行中的任务必须先取消。

返回结构统一为 ``ActionOutcome```，路由只负责把它翻译成 HTTP 响应，因此
「服务端先落库、前端再合并状态」这条链路不会被前端的乐观更新绕过。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from models import BackgroundJob, Shot
from services import job_center
from services.job_dispatch import redispatch
from services.job_types import (
    ACTIVE_STATUSES,
    DISPATCHABLE_JOB_TYPES,
    ERROR_CODE_NOT_FOUND,
    ERROR_CODE_NOT_RETRYABLE,
    ERROR_CODE_SCOPE_CONFLICT,
    ERROR_CODE_UNSUPPORTED,
    JOB_TYPE_SCRIPT_PIPELINE,
    JOB_TYPE_SHOT_VIDEO,
    RETRYABLE_STATUSES,
    STATUS_CANCELLING,
    TERMINAL_STATUSES,
    parse_job_key,
)
from services.provider_readiness import CODE_PROVIDER_NOT_CONFIGURED, format_message, missing_providers
from services.task_registry import cancel as cancel_job_task
from services.shot_dialogue import parse_shot_dialogue

RETRY_MODE = "retry"
RESUME_MODE = "resume"


@dataclass(frozen=True)
class ActionOutcome:
    """一次写操作的结果；http_status 由路由直接采用。"""

    ok: bool
    http_status: int
    status: str
    message: str
    job: dict[str, Any] | None = None
    error_code: str = ""
    idempotent: bool = False

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "ok": self.ok,
            "status": self.status,
            "message": self.message,
            "idempotent": self.idempotent,
        }
        if self.error_code:
            body["error_code"] = self.error_code
        if self.job is not None:
            body["job"] = self.job
        return body


def _not_found() -> ActionOutcome:
    return ActionOutcome(
        ok=False,
        http_status=404,
        status="not_found",
        message="任务不存在或已被清理",
        error_code=ERROR_CODE_NOT_FOUND,
    )


def _active_successor(db: Session, job: BackgroundJob) -> BackgroundJob | None:
    """同一条操作链上正在执行的新尝试（重试后点击第二次会命中这里）。"""

    canonical = parse_job_key(job.idempotency_key).canonical
    successor = (
        db.query(BackgroundJob)
        .filter(
            BackgroundJob.idempotency_key == canonical,
            BackgroundJob.id != job.id,
            BackgroundJob.status.in_(ACTIVE_STATUSES),
        )
        .first()
    )
    if successor is not None:
        return successor
    return (
        db.query(BackgroundJob)
        .filter(
            BackgroundJob.retry_of == job.id,
            BackgroundJob.status.in_(ACTIVE_STATUSES),
        )
        .first()
    )


def cancel_job(db: Session, job: BackgroundJob) -> ActionOutcome:
    """取消任务：queued/running/cancelling 都能正确处理，且幂等。"""

    status = str(job.status)
    if status in TERMINAL_STATUSES:
        return ActionOutcome(
            ok=True,
            http_status=200,
            status=status,
            message="任务已结束，无需取消",
            job=job_center.job_detail(db, job),
            idempotent=True,
        )
    if status == STATUS_CANCELLING:
        return ActionOutcome(
            ok=True,
            http_status=200,
            status=status,
            message="取消请求已受理，正在等待当前步骤安全退出",
            job=job_center.job_detail(db, job),
            idempotent=True,
        )

    handled = cancel_job_task(str(job.idempotency_key))
    db.expire_all()
    fresh = job_center.get_job(db, str(job.id))
    if fresh is None:
        return _not_found()
    fresh_status = str(fresh.status)
    return ActionOutcome(
        ok=bool(handled),
        http_status=200 if handled else 409,
        status=fresh_status,
        message="已请求取消，等待当前步骤安全退出" if handled else "取消请求未能生效，任务可能已结束",
        job=job_center.job_detail(db, fresh),
        idempotent=False,
    )


async def retry_job(db: Session, job: BackgroundJob) -> ActionOutcome:
    """完整重试：重新执行该任务的操作（已完成且仍然有效的阶段会被跳过）。"""

    return await _dispatch(db, job, RETRY_MODE)


async def resume_job(db: Session, job: BackgroundJob) -> ActionOutcome:
    """续跑：只补缺失或损坏的中间产物；无法安全续跑时给出明确错误。"""

    return await _dispatch(db, job, RESUME_MODE)


def _provider_block(db: Session, job: BackgroundJob) -> ActionOutcome | None:
    """重试 / 续跑前同样做 Provider 预检，避免任务重新排队后在中途再次失败。

    仅覆盖会直接消耗模型能力的任务类型；其余交给各自入口的既有校验。
    """

    job_type = str(job.job_type or "")
    identity = parse_job_key(job.idempotency_key)
    if job_type == JOB_TYPE_SCRIPT_PIPELINE:
        mode = identity.qualifier if identity.qualifier in {"manual", "auto"} else "manual"
        missing = missing_providers("script_pipeline", mode=mode)
    elif job_type == JOB_TYPE_SHOT_VIDEO:
        shot = db.query(Shot).filter(Shot.id == identity.owner_id).first()
        if shot is None:
            return None
        profile = _json_dict(shot.continuity_profile)
        missing = missing_providers(
            "shot_video",
            has_dialogue=bool(parse_shot_dialogue(shot.dialogue)),
            audio_mode_override=str(profile.get("audio_mode") or ""),
        )
    else:
        return None
    if not missing:
        return None
    return ActionOutcome(
        ok=False,
        http_status=409,
        status="provider_not_configured",
        message=format_message(missing),
        error_code=CODE_PROVIDER_NOT_CONFIGURED,
    )


def _json_dict(raw: Any) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


async def _dispatch(db: Session, job: BackgroundJob, mode: str) -> ActionOutcome:
    job_type = str(job.job_type or "")
    canonical = parse_job_key(job.idempotency_key).canonical
    action = "重试" if mode == RETRY_MODE else "续跑"

    if job_type not in DISPATCHABLE_JOB_TYPES:
        return ActionOutcome(
            ok=False,
            http_status=409,
            status="unsupported",
            message=f"该任务类型暂不支持{action}",
            error_code=ERROR_CODE_UNSUPPORTED,
            job=job_center.job_detail(db, job),
        )

    if str(job.status) not in RETRYABLE_STATUSES:
        successor = _active_successor(db, job)
        if successor is not None:
            return ActionOutcome(
                ok=True,
                http_status=200,
                status="already_running",
                message="该任务已有正在执行的新尝试，本次请求已合并",
                job=job_center.job_detail(db, successor),
                idempotent=True,
            )
        return ActionOutcome(
            ok=False,
            http_status=409,
            status="not_retryable",
            message=f"只有失败、已取消或已中断的任务可以{action}",
            error_code=ERROR_CODE_NOT_RETRYABLE,
            job=job_center.job_detail(db, job),
        )

    successor = _active_successor(db, job)
    if successor is not None:
        return ActionOutcome(
            ok=True,
            http_status=200,
            status="already_running",
            message="该任务已有正在执行的新尝试，本次请求已合并",
            job=job_center.job_detail(db, successor),
            idempotent=True,
        )

    target = db.query(BackgroundJob).filter(BackgroundJob.idempotency_key == canonical).first()
    if target is not None and str(target.status) in ACTIVE_STATUSES and target.id != job.id:
        return ActionOutcome(
            ok=True,
            http_status=200,
            status="already_running",
            message="该任务已有正在执行的新尝试，本次请求已合并",
            job=job_center.job_detail(db, target),
            idempotent=True,
        )

    blocked = _provider_block(db, job)
    if blocked is not None:
        return blocked

    result = await redispatch(job, mode)
    db.expire_all()
    fresh = db.query(BackgroundJob).filter(BackgroundJob.idempotency_key == canonical).first()
    if result.status == "started":
        if fresh is None:
            return _not_found()
        return ActionOutcome(
            ok=True,
            http_status=200,
            status="started",
            message=result.message,
            job=job_center.job_detail(db, fresh),
        )
    if result.status == "deduplicated":
        # 只有「同一条操作链的新尝试」才算幂等合并；被别的任务占着作用域时要
        # 明确告诉用户是互斥冲突，而不是含糊地说「已在运行」。
        if fresh is not None and str(fresh.status) in ACTIVE_STATUSES and fresh.id != job.id:
            return ActionOutcome(
                ok=True,
                http_status=200,
                status="already_running",
                message=result.message,
                job=job_center.job_detail(db, fresh),
                idempotent=True,
            )
        return ActionOutcome(
            ok=False,
            http_status=409,
            status="scope_conflict",
            message="同一项目或镜头下已有其它任务在运行，请等待其结束后再试",
            error_code=ERROR_CODE_SCOPE_CONFLICT,
            job=job_center.job_detail(db, job),
        )
    return ActionOutcome(
        ok=False,
        http_status=409,
        status="rejected",
        message=result.message,
        error_code=result.error_code or ERROR_CODE_NOT_RETRYABLE,
        job=job_center.job_detail(db, job),
    )


def delete_job(db: Session, job: BackgroundJob) -> ActionOutcome:
    """删除历史任务记录；运行中的任务必须先取消，避免删除仍在写入的所有者。"""

    if str(job.status) not in TERMINAL_STATUSES:
        return ActionOutcome(
            ok=False,
            http_status=409,
            status="active",
            message="任务仍在执行，请先取消再删除",
            error_code=ERROR_CODE_NOT_RETRYABLE,
            job=job_center.job_detail(db, job),
        )
    detail = job_center.job_detail(db, job)
    if not job_center.delete_job(db, job):
        return ActionOutcome(
            ok=False,
            http_status=409,
            status="active",
            message="任务仍在执行，请先取消再删除",
            error_code=ERROR_CODE_NOT_RETRYABLE,
            job=detail,
        )
    return ActionOutcome(ok=True, http_status=200, status="deleted", message="任务记录已清理", job=detail)


__all__ = [
    "ActionOutcome",
    "RESUME_MODE",
    "RETRY_MODE",
    "cancel_job",
    "delete_job",
    "resume_job",
    "retry_job",
]
