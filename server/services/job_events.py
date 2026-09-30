"""任务中心实时事件的发布入口。

事件只携带任务 DTO（见 ``services.job_dto``），因此天然不含 run token、完整
堆栈、API Key 或供应商原始响应。发布是「尽力而为」：没有事件循环、没有连接或
广播失败都不允许影响正在运行的生成任务。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

EVENT_JOB_CREATED = "job.created"
EVENT_JOB_UPDATED = "job.updated"
EVENT_JOB_PROGRESS = "job.progress"
EVENT_JOB_COMPLETED = "job.completed"
EVENT_JOB_FAILED = "job.failed"
EVENT_JOB_CANCELLED = "job.cancelled"
EVENT_JOB_INTERRUPTED = "job.interrupted"
EVENT_JOB_RETRY_STARTED = "job.retry_started"
EVENT_JOB_DEBUG = "job.debug"

JOB_EVENT_TYPES = (
    EVENT_JOB_CREATED,
    EVENT_JOB_UPDATED,
    EVENT_JOB_PROGRESS,
    EVENT_JOB_COMPLETED,
    EVENT_JOB_FAILED,
    EVENT_JOB_CANCELLED,
    EVENT_JOB_INTERRUPTED,
    EVENT_JOB_RETRY_STARTED,
    EVENT_JOB_DEBUG,
)

_TERMINAL_EVENT_BY_STATUS = {
    "completed": EVENT_JOB_COMPLETED,
    "failed": EVENT_JOB_FAILED,
    "cancelled": EVENT_JOB_CANCELLED,
    "interrupted": EVENT_JOB_INTERRUPTED,
}


def terminal_event_for(status: str) -> str:
    return _TERMINAL_EVENT_BY_STATUS.get(status, EVENT_JOB_UPDATED)


def event_envelope(event_type: str, job_payload: dict[str, Any]) -> dict[str, Any]:
    """构造事件信封；只放 DTO 与事件元信息。"""

    return {
        "type": event_type,
        "job": job_payload,
        "job_id": job_payload.get("id", ""),
        "project_id": job_payload.get("project_id", ""),
        "sent_at": datetime.utcnow().isoformat(),
    }


def debug_event_envelope(
    job_id: str,
    project_id: str,
    event: dict[str, Any],
    revision: int,
) -> dict[str, Any]:
    """调试日志事件只携带单条脱敏记录，不复制完整任务 DTO。"""

    return {
        "type": EVENT_JOB_DEBUG,
        "job_id": str(job_id or ""),
        "project_id": str(project_id or ""),
        "event": event,
        "debug_revision": max(0, int(revision or 0)),
        "sent_at": datetime.utcnow().isoformat(),
    }


def publish_job_debug_event(
    *,
    job_id: str,
    project_id: str,
    event: dict[str, Any],
    revision: int,
) -> None:
    """推送一条任务调试日志；失败只记录 debug，不影响生成任务。"""

    if not job_id or not event:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    envelope = debug_event_envelope(job_id, project_id, event, revision)
    try:
        from api.websocket import jobs_manager

        loop.create_task(jobs_manager.broadcast(envelope))
    except Exception:  # noqa: BLE001
        logger.debug("任务调试事件推送失败", exc_info=True)


def has_job_listeners() -> bool:
    """是否至少有一个任务中心 WebSocket 连接；没有监听者时跳过昂贵的 DTO 查询。"""

    try:
        from api.websocket import jobs_manager

        return jobs_manager.has_connections()
    except Exception:  # noqa: BLE001
        return False


def publish_job_event(event_type: str, job_payload: dict[str, Any]) -> None:
    """线程内同步入口：把事件投递到全局任务 WebSocket。

    调用方可能是同步的数据库辅助函数，因此在没有运行中的事件循环时直接返回；
    广播本身是异步的，不会阻塞 asyncio event loop 上的业务协程。
    """

    if not job_payload:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    envelope = event_envelope(event_type, job_payload)
    try:
        from api.websocket import jobs_manager

        loop.create_task(jobs_manager.broadcast(envelope))
    except Exception:  # noqa: BLE001 - 事件推送失败不能影响生成任务
        logger.debug("任务事件推送失败: %s", event_type, exc_info=True)


__all__ = [
    "EVENT_JOB_CANCELLED",
    "EVENT_JOB_COMPLETED",
    "EVENT_JOB_CREATED",
    "EVENT_JOB_DEBUG",
    "EVENT_JOB_FAILED",
    "EVENT_JOB_INTERRUPTED",
    "EVENT_JOB_PROGRESS",
    "EVENT_JOB_RETRY_STARTED",
    "EVENT_JOB_UPDATED",
    "JOB_EVENT_TYPES",
    "debug_event_envelope",
    "event_envelope",
    "has_job_listeners",
    "publish_job_debug_event",
    "publish_job_event",
    "terminal_event_for",
]
