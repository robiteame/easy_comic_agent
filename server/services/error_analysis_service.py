"""任务失败的自动归因：规则分类 + LLM 异步增强。

任务失败终态落库后由 ``schedule_failure_analysis`` 调度本服务，best-effort 地
细化 ``error_code`` 并写入 ``error_detail``（JSON：summary/suggestion/source/
model/analyzed_at），完成后发布 ``job.updated`` 事件，任务中心据此展示具体
失败原因（额度不足 / 触发限流 / API 参数错误 / ...）与修复建议。

三层防护确保分析永不影响任务本身：

- 规则先行：``job_types.classify_error_code`` 的确定性分类零成本且即时可见，
  LLM 只做增强，且只允许把泛化码（job_failed / provider_error）细化为具体码；
- 全程兜底：LLM 未配置、超时或输出不合法时静默保留规则结果，只写服务端日志；
- 幂等与去重：同一任务只分析一次（error_detail 非空即跳过），相同错误文本走
  指纹缓存，批量任务集体失败时只真正调用一次 LLM。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from typing import Any

from config import settings
from db import SessionLocal
from models import BackgroundJob
from services.error_reporter import redact
from services.job_dto import job_dto
from services.job_events import EVENT_JOB_UPDATED, publish_job_event
from services.job_types import (
    ERROR_CODE_BUDGET_EXCEEDED,
    ERROR_CODE_BUDGET_SOFT_EXCEEDED,
    ERROR_CODE_CONFIG,
    ERROR_CODE_DEPENDENCY_FAILED,
    ERROR_CODE_INVALID_REQUEST,
    ERROR_CODE_JOB_CANCELLED,
    ERROR_CODE_JOB_FAILED,
    ERROR_CODE_JOB_INTERRUPTED,
    ERROR_CODE_PROVIDER,
    ERROR_CODE_QUOTA_EXCEEDED,
    ERROR_CODE_RATE_LIMITED,
    ERROR_CODE_SERVER_RESTART,
    ERROR_CODE_STORAGE,
    ERROR_CODE_TIMEOUT,
    STATUS_FAILED,
    classify_error_code,
    job_type_label,
)
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

llm_service = LLMService()

# 单次分析（含 LLM 调用）的整体上限；超时按失败处理，保留规则结果。
_ANALYSIS_TIMEOUT_SECONDS = 45
# 送入 LLM 的错误文本上限（列内最长 2000 字符，取前 1500 足够归因）。
_ERROR_INPUT_CHARS = 1500
_SUMMARY_MAX_CHARS = 120
_SUGGESTION_MAX_CHARS = 160
# 相同错误的 LLM 结果缓存上限：批量失败（如整批额度不足）只调一次 LLM。
_CACHE_LIMIT = 64

# 语义已经明确的错误码（业务/环境类，消息本身就是人话）：不再调 LLM。
_LLM_SKIPPED_CODES = frozenset(
    {
        ERROR_CODE_JOB_CANCELLED,
        ERROR_CODE_JOB_INTERRUPTED,
        ERROR_CODE_SERVER_RESTART,
        ERROR_CODE_BUDGET_EXCEEDED,
        ERROR_CODE_BUDGET_SOFT_EXCEEDED,
        ERROR_CODE_DEPENDENCY_FAILED,
        ERROR_CODE_STORAGE,
    }
)

# 泛化码：规则只给了「失败/供应商错误」时，允许 LLM 依据全文细化为具体类别。
# 规则命中的确定性分类（额度/限流/参数/鉴权等）永远优先，不被 LLM 覆盖。
_UPGRADABLE_CODES = frozenset({ERROR_CODE_JOB_FAILED, ERROR_CODE_PROVIDER})

# LLM 允许输出的类别枚举，与 job_types 的稳定错误码保持一致。
_LLM_CATEGORY_ENUM = (
    ERROR_CODE_QUOTA_EXCEEDED,
    ERROR_CODE_RATE_LIMITED,
    ERROR_CODE_INVALID_REQUEST,
    ERROR_CODE_CONFIG,
    ERROR_CODE_PROVIDER,
    ERROR_CODE_TIMEOUT,
    ERROR_CODE_STORAGE,
    ERROR_CODE_JOB_FAILED,
)

_SYSTEM_PROMPT = (
    "你是后台任务失败原因诊断助手。根据任务信息与错误文本判断失败类别，"
    "并生成面向用户的中文解释与修复建议。\n"
    "category 只能是以下之一：\n"
    "- provider_quota_exceeded：API 额度不足或账户欠费\n"
    "- provider_rate_limited：API 限流（请求过于频繁）\n"
    "- provider_invalid_request：API 调用参数错误\n"
    "- provider_config_error：API Key 无效、未配置或无权限\n"
    "- provider_error：其他 API 调用失败（网络、服务商故障等）\n"
    "- timeout：调用超时\n"
    "- storage_error：本地存储或磁盘问题\n"
    "- job_failed：其他未知原因\n"
    '只输出一个 JSON 对象：{"category": string, "confidence": number, "summary": string, "suggestion": string}\n'
    "要求：summary 与 suggestion 都用中文，分别不超过 60 与 80 字；"
    "只依据错误文本中的信息，不要编造；输出中不得包含 API Key、密钥或本地路径。"
)

# 分析结果指纹缓存与执行中的任务集合（进程内即可：分析只做增强，重复执行无害）。
_analysis_cache: dict[str, dict[str, Any]] = {}
_in_flight: set[str] = set()
_pending_tasks: set[asyncio.Task] = set()
# 相同错误指纹的并发分析合并为一次 LLM 调用：后来者等待首个调用的结果。
_pending_fingerprints: dict[str, asyncio.Future] = {}


def schedule_failure_analysis(job_id: str) -> None:
    """任务失败落库后的同步调度入口；无事件循环时静默跳过。"""

    if not job_id:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if job_id in _in_flight:
        return
    task = loop.create_task(analyze_job_failure(job_id))
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)


async def analyze_job_failure(job_id: str) -> None:
    """best-effort 分析一个失败任务；任何异常都不向外传播。"""

    if job_id in _in_flight:
        return
    _in_flight.add(job_id)
    try:
        await asyncio.wait_for(_analyze(job_id), timeout=_ANALYSIS_TIMEOUT_SECONDS)
    except TimeoutError:
        logger.warning("失败原因分析超时: job_id=%s", job_id)
    except Exception:  # noqa: BLE001 - 分析失败绝不影响任务状态
        logger.warning("失败原因分析失败: job_id=%s", job_id, exc_info=True)
    finally:
        _in_flight.discard(job_id)


async def _analyze(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
        if job is None or job.status != STATUS_FAILED or job.error_detail:
            return
        rule_code = str(job.error_code or "") or classify_error_code(job.error or job.error_message or "")
        error_text = (job.error or job.error_message or "").strip()
        snapshot = {
            "job_type": str(job.job_type or ""),
            "display_name": str(job.display_name or ""),
            "idempotency_key": str(job.idempotency_key or ""),
        }
    finally:
        db.close()

    # LLM 调用放在数据库会话之外，避免长时间占用连接。
    detail = None
    if error_text and rule_code not in _LLM_SKIPPED_CODES:
        detail = await _analyze_with_llm(snapshot, rule_code, error_text)

    final_code = rule_code
    if detail is not None and rule_code in _UPGRADABLE_CODES:
        upgraded = detail.get("category")
        if upgraded in _LLM_CATEGORY_ENUM and upgraded != rule_code:
            final_code = str(upgraded)

    db = SessionLocal()
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
        if job is None or job.status != STATUS_FAILED or job.error_detail:
            return
        changed = False
        if final_code and final_code != str(job.error_code or ""):
            job.error_code = final_code
            changed = True
        if detail is not None:
            job.error_detail = json.dumps(
                {
                    "summary": detail.get("summary", ""),
                    "suggestion": detail.get("suggestion", ""),
                    "source": "llm",
                    "model": detail.get("model", ""),
                    "analyzed_at": datetime.utcnow().isoformat(),
                },
                ensure_ascii=False,
            )
            changed = True
        if not changed:
            return
        job.updated_at = datetime.utcnow()
        payload = _dto_with_cost(db, job)
        db.commit()
        publish_job_event(EVENT_JOB_UPDATED, payload)
    finally:
        db.close()


def _dto_with_cost(db, job: BackgroundJob) -> dict[str, Any]:
    from services import usage_service

    try:
        key = str(job.idempotency_key or "")
        usage = usage_service.job_usage_map(db, [key]).get(key)
        estimate = usage_service.job_estimate_map(db, [key]).get(key)
    except Exception:  # noqa: BLE001 - 成本快照缺失不影响分析事件
        usage, estimate = None, None
    return job_dto(job, usage=usage, estimate=estimate)


async def _analyze_with_llm(snapshot: dict[str, Any], rule_code: str, error_text: str) -> dict[str, Any] | None:
    if not settings.ERROR_ANALYSIS_LLM_ENABLED or not llm_service.available:
        return None
    cache_key = _fingerprint(rule_code, error_text)
    cached = _analysis_cache.get(cache_key)
    if cached is not None:
        return dict(cached)
    pending = _pending_fingerprints.get(cache_key)
    if pending is not None:
        # 同一错误已有并发分析在跑（如队列占位行与真实任务几乎同时失败）：
        # 等它完成后直接复用缓存，不再发第二次 LLM 调用。
        try:
            await pending
        except Exception:  # noqa: BLE001 - 首个调用失败时后来者各自返回 None
            return None
        cached = _analysis_cache.get(cache_key)
        return dict(cached) if cached is not None else None
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    _pending_fingerprints[cache_key] = future
    try:
        user_prompt = (
            f"任务类型：{job_type_label(snapshot.get('job_type', ''))}\n"
            f"任务名称：{snapshot.get('display_name') or '-'}\n"
            f"初步分类（关键词规则）：{rule_code or ERROR_CODE_JOB_FAILED}\n"
            f"错误文本：\n{redact(error_text, limit=_ERROR_INPUT_CHARS)}"
        )
        raw = await llm_service.call_json(_SYSTEM_PROMPT, user_prompt, temperature=0.1)
        detail = _validate_llm_result(raw)
        if detail is None:
            return None
        detail["model"] = str(getattr(llm_service, "last_provider_used", "") or "")
        _remember(cache_key, detail)
        if not future.done():
            future.set_result(True)
        return dict(detail)
    except BaseException as exc:
        if not future.done():
            # 不把 CancelledError 直接塞给等待者（会被 asyncio 当成等待者自身被取消），
            # 统一包成普通异常，等待者捕获后回退规则结果。
            future.set_exception(RuntimeError(str(exc) or exc.__class__.__name__))
        raise
    finally:
        _pending_fingerprints.pop(cache_key, None)


def _validate_llm_result(raw: Any) -> dict[str, Any] | None:
    """只接受类别合法且带有效摘要的输出；summary/suggestion 再脱敏一次。"""

    if not isinstance(raw, dict):
        return None
    category = str(raw.get("category") or "").strip()
    if category not in _LLM_CATEGORY_ENUM:
        return None
    summary = redact(str(raw.get("summary") or "").strip(), limit=_SUMMARY_MAX_CHARS)
    if not summary:
        return None
    suggestion = redact(str(raw.get("suggestion") or "").strip(), limit=_SUGGESTION_MAX_CHARS)
    return {"category": category, "summary": summary, "suggestion": suggestion}


def _fingerprint(rule_code: str, error_text: str) -> str:
    return hashlib.sha1(f"{rule_code}|{error_text[:_ERROR_INPUT_CHARS]}".encode()).hexdigest()


def _remember(key: str, value: dict[str, Any]) -> None:
    if len(_analysis_cache) >= _CACHE_LIMIT:
        _analysis_cache.pop(next(iter(_analysis_cache)), None)
    _analysis_cache[key] = value


__all__ = ["analyze_job_failure", "schedule_failure_analysis"]
