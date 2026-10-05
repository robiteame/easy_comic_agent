"""后台任务失败的统一上报与脱敏。

后台任务（剧本解析、分镜、逐镜头视频、成片导出）失败时，堆栈只应留在服务端：

- 服务端用 logger.exception 记录完整异常，并附带一个短错误 ID；
- 通过 WebSocket 只发送稳定的错误类型、简短中文提示和该错误 ID；
- 写入数据库的失败备注同样只保留简短提示 + 错误 ID。

日志写入前会先做脱敏，避免把 API Key、完整请求体或供应商原始响应写进普通
日志：密钥样式串会被替换，长文本会被截断。
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import traceback
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

# 稳定错误类型：前端按类型分支，不依赖异常文本。
ERROR_PIPELINE = "pipeline_error"
ERROR_STORYBOARD = "storyboard_error"
ERROR_SHOT_VIDEO = "shot_video_error"
ERROR_RENDER = "render_error"
ERROR_BACKGROUND_JOB = "background_job"

_MAX_LOG_CHARS = 400
_MAX_MESSAGE_CHARS = 160

# 日志脱敏过滤器标记，避免重复安装。
_FILTER_FLAG = "_comic_agent_redacting"

# 密钥、鉴权头与常见供应商凭据样式。
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{6,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|apikey|authorization|x-api-key|access[_-]?token)\b\s*[:=]?\s*[\"']?[A-Za-z0-9_\-\.]{8,}"
    ),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}\b"),
)
# 本地绝对路径（macOS / Linux / Windows）不应出现在面向用户的错误里。
_LOCAL_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:\\\\[^\s\"']+|/(?:Users|home|private|var|tmp|opt|root|mnt|srv|etc)/[^\s\"']*)"
)


def new_error_id() -> str:
    """生成短且唯一的错误编号（correlation id）。"""

    return secrets.token_hex(4)


def redact_secrets(value: Any, *, limit: int = _MAX_LOG_CHARS) -> str:
    """只脱敏密钥样式串，保留本地路径等调试信息（用于服务端日志）。"""

    text = value if isinstance(value, str) else str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[已脱敏]", text)
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def redact(value: Any, *, limit: int = _MAX_LOG_CHARS) -> str:
    """脱敏后的文本：去掉密钥与本地路径，并限制长度（用于回显给界面）。"""

    return _LOCAL_PATH_PATTERN.sub("[本地路径]", redact_secrets(value, limit=limit))


# 堆栈噪声：面向用户的错误摘要里不应出现任何行号、文件名或帧分隔符。
_TRACEBACK_NOISE = (
    re.compile(r"(?m)^\s*Traceback \(most recent call last\):\s*$"),
    re.compile(r"(?m)^\s*File \"[^\"]*\".*$"),
    re.compile(r"(?m)^\s*[~^]+\s*$"),
    re.compile(r"(?m)^\s*During handling of the above exception.*$"),
    re.compile(r"(?m)^\s*The above exception was the direct cause.*$"),
)


def summarize(value: Any, *, limit: int = _MAX_MESSAGE_CHARS) -> str:
    """把异常文本压成可回显的摘要：去掉堆栈帧、密钥、本地路径并截断。

    完整堆栈只应留在服务端日志；落库与回给前端的失败说明一律走这里。
    """

    text = redact(value, limit=8000)
    for pattern in _TRACEBACK_NOISE:
        text = pattern.sub("", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = " ".join(lines).strip()
    if len(summary) > limit:
        summary = summary[: max(0, limit - 1)] + "…"
    return summary


def _format_traceback(exc: BaseException) -> str:
    """完整堆栈文本（已脱敏）。"""

    try:
        rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    except Exception:  # noqa: BLE001 - 兜底：格式化失败也不能吞掉错误处理
        rendered = f"{type(exc).__name__}: {exc}"
    return redact_secrets(rendered, limit=8000)


def describe_context(context: Mapping[str, Any] | None) -> str:
    """把结构化上下文压成一行日志（键值均脱敏，不含请求体全文）。"""

    if not context:
        return "{}"
    try:
        payload = json.dumps(
            {str(key): redact(value, limit=120) for key, value in context.items()},
            ensure_ascii=False,
        )
    except (TypeError, ValueError):
        payload = redact(context)
    return redact(payload)


def log_failure(
    exc: BaseException,
    *,
    error_type: str,
    error_id: str | None = None,
    context: Mapping[str, Any] | None = None,
    log: logging.Logger | None = None,
) -> str:
    """在服务端记录完整异常（含堆栈），返回本次错误编号。

    堆栈先脱敏再写入 record.exc_text：直接依赖 logger.exception 的自动堆栈渲染
    会把异常文本原样打进日志，供应商响应里夹带的密钥就会随之落到磁盘。
    """

    identifier = error_id or new_error_id()
    target = log or logger
    if target.isEnabledFor(logging.ERROR):
        record = target.makeRecord(
            target.name,
            logging.ERROR,
            "(comic-agent)",
            0,
            "后台任务失败 [%s] type=%s context=%s error=%s",
            (identifier, error_type, describe_context(context), redact(exc)),
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        record.exc_text = _format_traceback(exc)
        target.handle(record)
    return identifier


class SecretRedactingFilter(logging.Filter):
    """兜底过滤器：清掉任何 handler 输出中的密钥样式串。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and not record.exc_text:
            try:
                record.exc_text = _format_traceback(record.exc_info[1])
            except Exception:  # noqa: BLE001
                record.exc_text = "【异常堆栈已隐藏】"
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact_secrets(item) if isinstance(item, str) else item for item in record.args)
        elif isinstance(record.args, dict):
            record.args = {
                key: redact_secrets(value) if isinstance(value, str) else value for key, value in record.args.items()
            }
        return True


def install_log_redaction() -> None:
    """给当前已注册的 handler 安装密钥脱敏过滤器（幂等）。"""

    handlers = list(logging.getLogger().handlers)
    if not handlers and logging.lastResort is not None:
        handlers = [logging.lastResort]
    for handler in handlers:
        if getattr(handler, _FILTER_FLAG, False):
            continue
        handler.addFilter(SecretRedactingFilter())
        setattr(handler, _FILTER_FLAG, True)


def user_message(text: str) -> str:
    """面向用户的单行提示（不含堆栈/路径/密钥）。"""

    return summarize(text, limit=_MAX_MESSAGE_CHARS)


def error_payload(
    *,
    error_type: str,
    message: str,
    error_id: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 WebSocket 错误消息：只有稳定类型、简短提示与错误编号。"""

    payload: dict[str, Any] = {
        "type": "error",
        "error_type": error_type,
        "message": user_message(message),
        "error_id": error_id,
    }
    if extra:
        for key, value in extra.items():
            payload[str(key)] = value
    return payload


def report_failure(
    exc: BaseException,
    *,
    error_type: str,
    message: str,
    context: Mapping[str, Any] | None = None,
    log: logging.Logger | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """记录完整异常并返回可直接推送的 WebSocket 错误消息。"""

    error_id = log_failure(exc, error_type=error_type, context=context, log=log)
    return error_payload(error_type=error_type, message=message, error_id=error_id, extra=extra)


def failure_note(exc: BaseException, *, prefix: str, error_type: str, log: logging.Logger | None = None) -> str:
    """写入数据库的失败备注：简短提示 + 错误编号，不含堆栈。"""

    error_id = log_failure(exc, error_type=error_type, log=log)
    return f"{prefix}（错误编号 {error_id}）"
