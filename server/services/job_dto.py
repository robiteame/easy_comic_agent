"""任务中心对外的稳定 DTO。

前端只依赖这里的字段名与取值，不直接序列化 SQLAlchemy 对象。两条硬约束：

- 永不返回 ``run_token``：它是服务端内部的抢占令牌，只在服务端比较；
- 永不返回完整堆栈、供应商原始响应或本地绝对路径：失败只回稳定错误码 + 短消息。

预计剩余时间（ETA）只有在同一任务类型存在足够多的真实历史耗时样本时才给出，
否则固定为 ``None``，避免编造进度。
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from services.error_reporter import summarize
from services.job_types import (
    ACTIVE_STATUSES,
    JOB_TYPE_UNKNOWN,
    RESUMABLE_JOB_TYPES,
    RETRYABLE_STATUSES,
    TERMINAL_STATUSES,
    error_code_for_status,
    error_code_label,
    job_type_label,
    status_label,
)

# 面向用户的短消息上限：数据库里的 error 列保留更长文本供诊断，DTO 只给摘要。
MESSAGE_MAX_CHARS = 240


def as_utc(value: datetime | None) -> datetime | None:
    """把存储/传入的时间统一成 UTC 时区感知对象。

    数据库里的历史列都是 naive UTC（SQLite 不保留时区），这里显式补上 UTC，
    避免序列化成不带时区的 ISO 字符串后被前端按本地时间解释（东八区会差 8 小时）。
    已带时区的值原样换算到 UTC。
    """

    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _as_iso(value: datetime | None) -> str | None:
    moment = as_utc(value)
    return moment.isoformat() if moment is not None else None


def _clamp_progress(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


def _short_message(value: str | None) -> str:
    """防御性再脱敏一次：历史行里可能残留未清理的堆栈文本。"""

    return summarize(value or "", limit=MESSAGE_MAX_CHARS)


def _json_list(value: Any) -> list[str]:
    import json

    try:
        parsed = json.loads(value or "[]") if isinstance(value, str) else value
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def _report(value: Any) -> dict[str, Any] | None:
    """解析任务结果报告；空/非法历史行返回 None。"""
    import json

    if not value:
        return None
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _error_detail(value: Any) -> dict[str, Any] | None:
    """解析落库的失败分析 JSON；无效或为空返回 None（字段白名单 + 长度兜底）。"""

    import json

    if not value:
        return None
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    summary = _short_message(str(parsed.get("summary") or ""))
    suggestion = _short_message(str(parsed.get("suggestion") or ""))
    if not summary and not suggestion:
        return None
    source = str(parsed.get("source") or "")
    return {
        "summary": summary,
        "suggestion": suggestion,
        "source": source if source in {"rule", "llm"} else "",
        "model": _short_message(str(parsed.get("model") or "")),
    }


def job_duration_seconds(job: Any, *, now: datetime | None = None) -> int:
    """已运行时长（秒）。排队中且未开始的按 0 处理。"""

    moment = as_utc(now) if now is not None else datetime.now(UTC)
    started = as_utc(job.started_at or job.created_at)
    if started is None:
        return 0
    end = as_utc(job.finished_at) if job.status in TERMINAL_STATUSES and job.finished_at else moment
    try:
        return max(0, int((end - started).total_seconds()))
    except TypeError:  # 防御：意外类型组合时不要让列表接口 500
        return 0


def estimate_eta_seconds(durations: Iterable[float], progress: int, *, minimum_samples: int = 3) -> int | None:
    """按同类型历史耗时中位数估算剩余秒数；样本不足时返回 None。"""

    samples = [float(item) for item in durations if isinstance(item, (int, float)) and item > 0]
    if len(samples) < max(1, int(minimum_samples)):
        return None
    remaining_ratio = (100 - _clamp_progress(progress)) / 100
    if remaining_ratio <= 0:
        return 0
    return max(0, int(round(statistics.median(samples) * remaining_ratio)))


def action_flags(
    status: str,
    job_type: str,
    *,
    has_active_successor: bool = False,
    dispatchable: bool | None = None,
) -> dict[str, Any]:
    """可执行动作与禁用原因；前端按钮的 disabled 状态直接来自这里。"""

    retryable = status in RETRYABLE_STATUSES
    blocked = has_active_successor
    dispatch_ok = (job_type in RESUMABLE_JOB_TYPES) if dispatchable is None else bool(dispatchable)

    can_cancel = status in ACTIVE_STATUSES
    can_retry = retryable and dispatch_ok and not blocked
    can_resume = retryable and dispatch_ok and not blocked and job_type in RESUMABLE_JOB_TYPES
    can_delete = status in TERMINAL_STATUSES

    retry_reason = ""
    if not retryable:
        retry_reason = "只有失败、已取消或已中断的任务可以重试"
    elif blocked:
        retry_reason = "该任务已有正在执行的新尝试"
    elif not dispatch_ok:
        retry_reason = "该任务类型暂不支持重新派发"

    resume_reason = ""
    if not retryable:
        resume_reason = "只有失败、已取消或已中断的任务可以续跑"
    elif blocked:
        resume_reason = "该任务已有正在执行的新尝试"
    elif not dispatch_ok:
        resume_reason = "该任务类型暂不支持续跑"

    return {
        "can_cancel": can_cancel,
        "can_retry": can_retry,
        "can_resume": can_resume,
        "can_delete": can_delete,
        "retry_blocked_reason": retry_reason,
        "resume_blocked_reason": resume_reason,
    }


def cost_dto(usage: dict[str, Any] | None, estimate: dict[str, Any] | None = None) -> dict[str, Any]:
    """任务成本快照：单次实际成本 + 启动前估算（两者分表存储，这里只做展示合并）。

    cost_known=False 时 cost_micro 必为 None —— 前端据此显示「成本未知」，而不是
    把未知当成 0 元。
    """

    usage = usage or {}
    estimate = estimate or {}
    cost_micro = usage.get("cost_micro")
    return {
        "currency": str(usage.get("currency") or estimate.get("currency") or "CNY"),
        "cost_micro": int(cost_micro) if cost_micro is not None else None,
        "cost_known": bool(usage.get("cost_known", False)) and cost_micro is not None,
        "call_count": int(usage.get("call_count") or 0),
        "unknown_call_count": int(usage.get("unknown_call_count") or 0),
        "failed_call_count": int(usage.get("failed_call_count") or 0),
        "provider_seconds": int(usage.get("provider_seconds") or 0),
        "by_capability": list(usage.get("by_capability") or []),
        "estimated_cost_micro": (
            int(estimate["estimated_cost_micro"]) if estimate.get("estimated_cost_micro") is not None else None
        ),
        "estimated_cost_known": bool(estimate.get("cost_known", False)),
        "estimated_seconds": int(estimate["estimated_seconds"])
        if estimate.get("estimated_seconds") is not None
        else None,
        "duration_source": str(estimate.get("duration_source") or ""),
        "has_usage": bool(usage.get("call_count")),
    }


def job_dto(
    job: Any,
    *,
    now: datetime | None = None,
    has_active_successor: bool = False,
    eta_seconds: int | None = None,
    dispatchable: bool | None = None,
    usage: dict[str, Any] | None = None,
    estimate: dict[str, Any] | None = None,
    include_report: bool = False,
) -> dict[str, Any]:
    """把一条任务行转成稳定 DTO。绝不包含 run_token。"""

    moment = now if now is not None else datetime.now(UTC)
    status = str(job.status or "")
    job_type = str(job.job_type or JOB_TYPE_UNKNOWN)
    error_message = _short_message(job.error_message or job.error)
    error_code = str(job.error_code or "") or (error_code_for_status(status, error_message) if error_message else "")
    if status in {*RETRYABLE_STATUSES} and not error_message:
        error_message = "任务未正常完成"

    dto: dict[str, Any] = {
        "id": str(job.id),
        "scope": str(job.scope or ""),
        "project_id": str(job.project_id or ""),
        "job_type": job_type,
        "job_type_label": job_type_label(job_type),
        "display_name": str(job.display_name or "") or job_type_label(job_type),
        "status": status,
        "status_label": status_label(status),
        "progress": _clamp_progress(job.progress),
        "current_step": str(job.current_step or ""),
        "message": _short_message(job.message),
        "error_code": error_code,
        "error_code_label": error_code_label(error_code),
        "error_message": error_message,
        "error_detail": _error_detail(getattr(job, "error_detail", "")),
        **({"report": _report(getattr(job, "report", ""))} if include_report else {}),
        "debug_revision": max(0, int(getattr(job, "debug_revision", 0) or 0)),
        "attempt": max(1, int(job.attempt or 1)),
        "retry_of": str(job.retry_of) if job.retry_of else None,
        "version": int(job.version or 0),
        "created_at": _as_iso(job.created_at),
        "started_at": _as_iso(job.started_at),
        "updated_at": _as_iso(job.updated_at),
        "finished_at": _as_iso(job.finished_at),
        "cancel_requested_at": _as_iso(job.cancel_requested_at),
        "duration_seconds": job_duration_seconds(job, now=moment),
        "eta_seconds": eta_seconds if status in ACTIVE_STATUSES else None,
        "is_active": status in ACTIVE_STATUSES,
        "is_terminal": status in TERMINAL_STATUSES,
        "has_active_successor": bool(has_active_successor),
        "cost": cost_dto(usage, estimate),
    }
    if str(getattr(job, "queue_batch_id", "") or "") or str(getattr(job, "queue_stage", "") or ""):
        dto.update(
            {
                "batch_id": str(getattr(job, "queue_batch_id", "") or "") or None,
                "queue_position": max(0, int(getattr(job, "queue_position", 0) or 0)),
                "priority": int(getattr(job, "queue_priority", 0) or 0),
                "queue_order": max(0, int(getattr(job, "queue_order", 0) or 0)),
                "stage": str(getattr(job, "queue_stage", "") or ""),
                "shot_id": str(getattr(job, "queue_shot_id", "") or "") or None,
                "dependency_ids": _json_list(getattr(job, "queue_dependency_ids", "[]")),
                "blocked_reason": str(getattr(job, "queue_blocked_reason", "") or ""),
                "queue_concurrency": max(1, int(getattr(job, "queue_concurrency", 1) or 1)),
                "paused": bool(getattr(job, "queue_paused", False)),
                "resume_missing": bool(getattr(job, "queue_resume_missing", False)),
                "reuse_audio": bool(getattr(job, "queue_reuse_audio", False)),
                "force_confirmed": bool(getattr(job, "queue_force_confirmed", False)),
                "requested_version": max(0, int(getattr(job, "queue_requested_version", 0) or 0)),
            }
        )
    dto.update(
        action_flags(
            status,
            job_type,
            has_active_successor=has_active_successor,
            dispatchable=dispatchable,
        )
    )
    return dto


__all__ = [
    "MESSAGE_MAX_CHARS",
    "action_flags",
    "as_utc",
    "cost_dto",
    "estimate_eta_seconds",
    "job_dto",
    "job_duration_seconds",
]
