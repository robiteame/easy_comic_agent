"""任务调试日志：记录进度与对外 API 请求的可诊断快照。

约束：
- 只记录任务执行上下文中的业务参数与提示词，不读取或回显 API Key、Authorization、
  本地绝对路径、参考图 data URL / 签名 URL；
- 日志随 ``background_jobs`` 持久化并限制条数，任务重试会创建新的任务行，因此
  新尝试不会污染旧尝试的调试轨迹；
- 调试记录只用于诊断，写入/推送失败不得影响真实生成任务。
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from services.error_reporter import redact
from services.usage_service import current_scope

logger = logging.getLogger(__name__)

DEBUG_EVENT_CREATED = "job.debug"
MAX_DEBUG_EVENTS = 120
MAX_DEBUG_JSON_CHARS = 400_000
MAX_PROMPT_CHARS = 12_000
MAX_PARAM_CHARS = 6_000
_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "token",
    "secret",
    "password",
    "credential",
    "credentials",
    "x-api-key",
}
_MEDIA_KEYS = {
    "image",
    "images",
    "image_url",
    "image_urls",
    "reference_image",
    "reference_images",
    "reference_assets",
    "content",
    "output_video_path",
    "output_frame_path",
    "audio_path",
    "video_path",
    "frame_path",
}


def _safe_value(value: Any, *, limit: int) -> Any:
    if isinstance(value, str):
        return redact(value, limit=limit)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key, raw_value in list(value.items())[:80]:
            key = str(raw_key)
            lowered = key.lower()
            if lowered in _SECRET_KEYS:
                result[key] = "[已脱敏]"
            elif lowered in _MEDIA_KEYS:
                if isinstance(raw_value, (list, tuple)):
                    result[key] = f"[{len(raw_value)} 项参考素材，不记录内容]"
                elif raw_value:
                    result[key] = "[媒体引用，不记录内容]"
                else:
                    result[key] = ""
            else:
                result[key] = _safe_value(raw_value, limit=min(limit, MAX_PARAM_CHARS))
        return result
    if isinstance(value, (list, tuple)):
        clipped = list(value)[:80]
        if len(clipped) >= 3 and all(isinstance(item, str) for item in clipped):
            return [_safe_value(item, limit=min(limit, 2_000)) for item in clipped]
        return [_safe_value(item, limit=min(limit, MAX_PARAM_CHARS)) for item in clipped]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(value, limit=min(limit, MAX_PARAM_CHARS))


def make_event(
    kind: str,
    message: str,
    *,
    step: str = "",
    progress: int | None = None,
    api: str = "",
    provider: str = "",
    model: str = "",
    params: Any = None,
    prompt: Any = None,
    status: str = "info",
    request_id: str = "",
    detail: Any = None,
) -> dict[str, Any]:
    prompt_payload = _safe_value(prompt, limit=MAX_PROMPT_CHARS) if prompt not in (None, "") else ""
    return {
        "id": uuid.uuid4().hex,
        "request_id": str(request_id or ""),
        "timestamp": datetime.utcnow().isoformat(),
        "kind": str(kind or "log")[:40],
        "level": str(status or "info")[:20],
        "step": str(step or "")[:120],
        "progress": None if progress is None else max(0, min(100, int(progress))),
        "message": redact(message, limit=600),
        "api": str(api or "")[:80],
        "provider": str(provider or "")[:80],
        "model": str(model or "")[:120],
        "params": _safe_value(params, limit=MAX_PARAM_CHARS) if params not in (None, "") else "",
        "prompt": prompt_payload,
        "detail": _safe_value(detail, limit=MAX_PARAM_CHARS) if detail not in (None, "") else "",
    }


def parse_events(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def append_event(raw: Any, event: dict[str, Any], *, revision: int = 0) -> tuple[str, int]:
    """把事件追加到 JSON 文本并返回（新 JSON，新 revision）。"""

    try:
        current = parse_events(raw)
        revision = max(0, int(revision or 0))
        current.append(event)
        if len(current) > MAX_DEBUG_EVENTS:
            current = current[-MAX_DEBUG_EVENTS:]
        payload = json.dumps(current, ensure_ascii=False, separators=(",", ":"))
        while len(payload) > MAX_DEBUG_JSON_CHARS and len(current) > 1:
            current.pop(0)
            payload = json.dumps(current, ensure_ascii=False, separators=(",", ":"))
        return payload, revision + 1
    except Exception:  # noqa: BLE001 - 调试日志绝不能阻断任务
        logger.debug("调试日志序列化失败", exc_info=True)
        return "[]", 1


def publish_debug_event(job: Any, event: dict[str, Any], revision: int) -> None:
    """向任务中心 WebSocket 推送单条调试事件（不含任务 DTO）。"""

    try:
        from services.job_events import publish_job_debug_event

        publish_job_debug_event(
            job_id=str(job.id),
            project_id=str(job.project_id or ""),
            event=event,
            revision=revision,
        )
    except Exception:  # noqa: BLE001
        logger.debug("调试日志事件推送失败", exc_info=True)


def record_api_request(
    *,
    api: str,
    provider: str,
    model: str,
    params: Any,
    prompt: Any,
    message: str = "",
    detail: Any = None,
) -> str:
    """记录一次对外 API 请求参数与提示词；不在后台任务上下文时安全跳过。"""

    scope = current_scope()
    if not scope.job_id:
        return ""
    request_id = uuid.uuid4().hex
    event = make_event(
        "api_request",
        message or f"发起 {api} API 请求",
        api=api,
        provider=provider,
        model=model,
        params=params,
        prompt=prompt,
        status="request",
        request_id=request_id,
        detail=detail,
    )
    _record_for_current_job(event)
    return request_id


def record_api_result(
    request_id: str,
    *,
    status: str,
    message: str,
    detail: Any = None,
    api: str = "",
) -> None:
    """补充 API 请求结果；与请求事件通过 request_id 关联。"""

    if not request_id:
        return
    event = make_event(
        "api_result",
        message,
        api=api,
        status=status,
        request_id=request_id,
        detail=detail,
    )
    _record_for_current_job(event)


def _record_for_current_job(event: dict[str, Any]) -> None:
    scope = current_scope()
    if not scope.job_id:
        return
    from db import SessionLocal
    from models import BackgroundJob

    db = SessionLocal()
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.id == scope.job_id).first()
        if job is None:
            return
        if not event.get("step"):
            event["step"] = str(job.current_step or "")
        if event.get("progress") is None:
            event["progress"] = int(job.progress or 0)
        raw, revision = append_event(job.debug_events, event, revision=int(job.debug_revision or 0))
        job.debug_events = raw
        job.debug_revision = revision
        job.updated_at = datetime.utcnow()
        db.commit()
        publish_debug_event(job, event, revision)
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.debug("调试日志写入失败", exc_info=True)
    finally:
        db.close()


__all__ = [
    "DEBUG_EVENT_CREATED",
    "MAX_DEBUG_EVENTS",
    "append_event",
    "make_event",
    "parse_events",
    "publish_debug_event",
    "record_api_request",
    "record_api_result",
]
