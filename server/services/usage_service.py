"""用量与费用记账（实际值）。

职责边界：

- **实际用量**只写在这里（usage_records 表）；任务启动前的估算由 budget_service
  写入 cost_estimates 表，两者分表存储、互不覆盖（验收要求「估算值和实际 usage
  分开保存」）；
- 每次供应商调用只允许入账一次：usage_key 唯一 + 事务内先查后插，重复回调/重复
  代码路径不会重复计费；
- 失败与取消同样落库：调用失败记一条 status=failed 的记录（用量未知时金额为
  NULL），保证「任务失败后仍能查询已发生的调用成本」；
- 不写入供应商原始响应、API Key、本地绝对路径。

用量上下文（project / 剧集 / 镜头 / 任务）通过 contextvar 传递：任务注册表在启动
协程时绑定一次，服务层（LLM/图像/视频/TTS/FFmpeg）不需要层层透传参数即可正确归属。
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from db import SessionLocal
from models import CostEstimate, Project, Shot, UsageRecord
from models.pricing import DEFAULT_CURRENCY
from services import pricing_service
from services.pricing_service import (
    COST_SOURCE_LOCAL,
    COST_SOURCE_PRICING,
    COST_SOURCE_UNKNOWN,
)
from services.providers.usage import (
    CAPABILITY_BASE_UNITS,
    CAPABILITY_LABELS,
    ERROR_CODE_PROVIDER_CALL_FAILED,
    UsageMetadata,
    unknown_usage,
)

logger = logging.getLogger(__name__)

# 单次调用自身的结局。
CALL_SUCCEEDED = "succeeded"
CALL_FAILED = "failed"
CALL_CANCELLED = "cancelled"
CALL_STATUSES = (CALL_SUCCEEDED, CALL_FAILED, CALL_CANCELLED)

# 统计维度。
GROUP_BY_CAPABILITY = "capability"
GROUP_BY_JOB_TYPE = "job_type"
GROUP_BY_SHOT = "shot"
GROUP_BY_PROJECT = "project"
GROUP_BY_EPISODE = "episode"
GROUP_BY_PROVIDER = "provider"
GROUP_BY_MODEL = "model"
GROUP_BY_STATUS = "status"
GROUP_BY_DIMENSIONS = (
    GROUP_BY_CAPABILITY,
    GROUP_BY_JOB_TYPE,
    GROUP_BY_SHOT,
    GROUP_BY_PROJECT,
    GROUP_BY_EPISODE,
    GROUP_BY_PROVIDER,
    GROUP_BY_MODEL,
    GROUP_BY_STATUS,
)

_GROUP_COLUMNS = {
    GROUP_BY_CAPABILITY: UsageRecord.capability,
    GROUP_BY_JOB_TYPE: UsageRecord.job_type,
    GROUP_BY_SHOT: UsageRecord.shot_id,
    GROUP_BY_PROJECT: UsageRecord.project_id,
    GROUP_BY_EPISODE: UsageRecord.project_id,
    GROUP_BY_PROVIDER: UsageRecord.provider,
    GROUP_BY_MODEL: UsageRecord.model,
    GROUP_BY_STATUS: UsageRecord.status,
}

_MAX_KEY_CHARS = 200


@dataclass(frozen=True)
class UsageScope:
    """一次调用所属的业务上下文。"""

    project_id: str = ""
    series_id: str = ""
    shot_id: str = ""
    job_key: str = ""
    job_id: str = ""
    job_type: str = ""

    def merged(self, **kwargs: Any) -> UsageScope:
        values = {field: getattr(self, field) for field in self.__dataclass_fields__}  # type: ignore[attr-defined]
        for key, value in kwargs.items():
            if key in values and value not in (None, ""):
                values[key] = str(value)
        return UsageScope(**values)

    def to_dict(self) -> dict[str, str]:
        return {
            "project_id": self.project_id,
            "series_id": self.series_id,
            "shot_id": self.shot_id,
            "job_key": self.job_key,
            "job_id": self.job_id,
            "job_type": self.job_type,
        }


_EMPTY_SCOPE = UsageScope()
_scope_var: ContextVar[UsageScope] = ContextVar("comic_agent_usage_scope", default=_EMPTY_SCOPE)


def current_scope() -> UsageScope:
    return _scope_var.get()


@contextmanager
def usage_scope(**kwargs: Any) -> Iterator[UsageScope]:
    """在 with 块内绑定用量上下文；退出时恢复原值。"""

    token = _scope_var.set(current_scope().merged(**kwargs))
    try:
        yield _scope_var.get()
    finally:
        _scope_var.reset(token)


def bind_scope(scope: UsageScope) -> Any:
    """把上下文绑定到当前 context（供 asyncio 任务在创建前调用），返回 reset token。"""

    return _scope_var.set(scope)


def reset_scope(token: Any) -> None:
    try:
        _scope_var.reset(token)
    except (ValueError, LookupError):  # 跨 context reset 时忽略
        pass


def resolve_scope(db: Session, scope: UsageScope | None = None, **overrides: Any) -> UsageScope:
    """补齐 scope 的 project / series / shot 归属信息。"""

    merged = (scope or current_scope()).merged(**overrides)
    if merged.project_id and merged.series_id:
        return merged
    project_id = merged.project_id
    if not project_id and merged.shot_id:
        project_id = str(db.query(Shot.project_id).filter(Shot.id == merged.shot_id).scalar() or "")
    series_id = merged.series_id
    if project_id and not series_id:
        row = db.query(Project.project_type, Project.parent_project_id).filter(Project.id == project_id).first()
        if row is not None:
            project_type, parent_id = str(row[0] or ""), str(row[1] or "")
            series_id = parent_id if project_type == "episode" and parent_id else project_id
        else:
            series_id = project_id
    return replace(merged, project_id=project_id, series_id=series_id)


def _clip(value: str, limit: int = _MAX_KEY_CHARS) -> str:
    return str(value or "")[:limit]


# 失败/取消时仍然可信的用量来源：provider=供应商回报，local=本地实测。
# request=按请求推导但调用没成功，不能当作已发生的消耗。
_TRUSTED_SOURCES_ON_FAILURE = frozenset({"provider", "local"})


def _outcome_metadata(metadata: UsageMetadata, status: str) -> UsageMetadata:
    """失败/取消时的用量口径。

    只有「供应商真的回报过用量」（source=provider）或「本地实测」（source=local）
    才保留数量；按请求推导的数量（source=request）在调用失败后一律记为未知
    —— 调用失败了，我们并不知道供应商是否计费，不能凭空填数字。
    """

    if status == CALL_SUCCEEDED:
        return metadata
    if metadata.known and metadata.source in _TRUSTED_SOURCES_ON_FAILURE:
        return metadata
    return unknown_usage(
        metadata.capability,
        metadata.provider,
        metadata.model,
        duration_ms=metadata.duration_ms,
    )


def _is_local_free(capability: str, provider: str) -> bool:
    return (str(capability), str(provider)) in pricing_service.LOCAL_ZERO_COST_PROVIDERS


def _price_outcome(
    db: Session,
    metadata: UsageMetadata,
    *,
    status: str,
) -> tuple[int | None, bool, str, pricing_service.PriceResolution]:
    """算出这次调用的金额（micro）与可信度。"""

    price = pricing_service.resolve_price(
        db,
        metadata.capability,
        metadata.provider,
        metadata.model,
        resolution=metadata.resolution,
    )
    if price.priced and metadata.known:
        cost = price.cost_micro(metadata.quantity, metadata.secondary_quantity)
        return cost, True, COST_SOURCE_PRICING, price
    if _is_local_free(metadata.capability, metadata.provider):
        # 本地能力（占位图 / 本地编码）：数量照记，费用恒为 0，不是「未知」。
        return 0, True, COST_SOURCE_LOCAL, price
    return None, False, COST_SOURCE_UNKNOWN, price


def record_metadata(
    metadata: UsageMetadata,
    *,
    status: str = CALL_SUCCEEDED,
    error_code: str = "",
    duration_ms: int = 0,
    scope: UsageScope | None = None,
    dedupe_key: str = "",
    extra_units: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """记录一次供应商调用的用量与费用。

    返回落库概要（含 usage_key / cost_micro / cost_known）；重复的 usage_key 不会
    再次计费，直接返回既有记录（duplicate=True）。记账失败只写日志，不打断生成流程。
    """

    normalized_status = status if status in CALL_STATUSES else CALL_SUCCEEDED
    effective = _outcome_metadata(metadata, normalized_status)
    effective = replace(effective, duration_ms=int(duration_ms or effective.duration_ms or 0))
    units = effective.units()
    if extra_units:
        units.update({key: value for key, value in extra_units.items() if value not in (None, "", 0)})
    units["call_status"] = normalized_status

    db = SessionLocal()
    try:
        resolved = resolve_scope(db, scope)
        key = _clip(dedupe_key) or _clip(f"{resolved.job_key or 'manual'}:{effective.capability}:{uuid.uuid4().hex}")
        cost, known, source, price = _price_outcome(db, effective, status=normalized_status)
        now = datetime.utcnow()
        db.execute(text("BEGIN IMMEDIATE"))
        existing = db.query(UsageRecord).filter(UsageRecord.usage_key == key).first()
        if existing is not None:
            db.rollback()
            return {**_record_dto(existing), "duplicate": True}
        record = UsageRecord(
            id=uuid.uuid4().hex,
            usage_key=key,
            job_key=resolved.job_key,
            job_id=resolved.job_id,
            job_type=resolved.job_type,
            job_status="",
            project_id=resolved.project_id,
            series_id=resolved.series_id,
            shot_id=resolved.shot_id,
            capability=effective.capability,
            provider=effective.provider,
            model=effective.model,
            quantity=int(effective.quantity),
            secondary_quantity=int(effective.secondary_quantity),
            resolution=str(effective.resolution or "")[:64],
            units=json.dumps(units, ensure_ascii=False),
            status=normalized_status,
            error_code=_clip(error_code, 64),
            cost_micro=cost,
            currency=price.currency or DEFAULT_CURRENCY,
            cost_known=bool(known),
            cost_source=source,
            price_snapshot=json.dumps(price.snapshot(), ensure_ascii=False),
            duration_ms=max(0, int(effective.duration_ms or 0)),
            started_at=now,
            finished_at=now,
            created_at=now,
            updated_at=now,
        )
        db.add(record)
        db.commit()
        return {**_record_dto(record), "duplicate": False}
    except Exception:  # noqa: BLE001 - 记账不能打断付费调用本身的业务流程
        db.rollback()
        logger.warning("用量记账失败（已跳过，不影响生成流程）: capability=%s", effective.capability, exc_info=True)
        return None
    finally:
        db.close()


def record_usage(
    capability: str,
    *,
    provider: str = "",
    model: str = "",
    quantity: int = 0,
    secondary_quantity: int = 0,
    resolution: str = "",
    units: dict[str, Any] | None = None,
    known: bool = True,
    billable: bool = True,
    source: str = "",
    status: str = CALL_SUCCEEDED,
    error_code: str = "",
    duration_ms: int = 0,
    scope: UsageScope | None = None,
    dedupe_key: str = "",
) -> dict[str, Any] | None:
    """直接用数量记账（FFmpeg 等本地能力、或调用方已自行统计数量的场景）。

    source 缺省时按能力推断：本地能力记为 local（本地实测，失败也保留数量），
    其余记为 provider（外部调用）。
    """

    resolved_source = str(source or "").strip() or (
        "local" if (not billable or str(provider) in {"local", "ffmpeg"}) else "provider"
    )
    metadata = UsageMetadata(
        capability=str(capability),
        provider=str(provider or ""),
        model=str(model or ""),
        known=bool(known),
        billable=bool(billable),
        source=resolved_source,
        duration_ms=int(duration_ms or 0),
        extra=dict(units or {}),
    )
    if capability in (pricing_service.CAPABILITY_LLM,):
        metadata = replace(metadata, input_tokens=int(quantity or 0), output_tokens=int(secondary_quantity or 0))
    elif capability == pricing_service.CAPABILITY_IMAGE:
        metadata = replace(metadata, images=int(quantity or 0), resolution=str(resolution or ""))
    elif capability in (pricing_service.CAPABILITY_VIDEO,):
        metadata = replace(metadata, seconds=float(quantity or 0), resolution=str(resolution or ""))
    elif capability == pricing_service.CAPABILITY_TTS:
        metadata = replace(metadata, characters=int(quantity or 0))
    elif capability == pricing_service.CAPABILITY_FFMPEG:
        metadata = replace(metadata, seconds=float(quantity or 0), resolution=str(resolution or ""))
    return record_metadata(
        metadata,
        status=status,
        error_code=error_code,
        duration_ms=duration_ms,
        scope=scope,
        dedupe_key=dedupe_key,
        extra_units=units,
    )


def record_failure(
    metadata: UsageMetadata,
    *,
    capability: str = "",
    error_code: str = ERROR_CODE_PROVIDER_CALL_FAILED,
    duration_ms: int = 0,
    scope: UsageScope | None = None,
    dedupe_key: str = "",
) -> dict[str, Any] | None:
    """记录一次失败的供应商调用（用量未知时金额保持 NULL）。"""

    return record_metadata(
        metadata if capability == "" else replace(metadata, capability=capability),
        status=CALL_FAILED,
        error_code=error_code,
        duration_ms=duration_ms,
        scope=scope,
        dedupe_key=dedupe_key,
    )


def record_cancelled(
    metadata: UsageMetadata,
    *,
    error_code: str = "provider_call_cancelled",
    duration_ms: int = 0,
    scope: UsageScope | None = None,
    dedupe_key: str = "",
) -> dict[str, Any] | None:
    """记录一次被取消的供应商调用（任务取消也要留痕，金额通常为未知）。"""

    return record_metadata(
        metadata,
        status=CALL_CANCELLED,
        error_code=error_code,
        duration_ms=duration_ms,
        scope=scope,
        dedupe_key=dedupe_key,
    )


def _record_dto(record: UsageRecord) -> dict[str, Any]:
    try:
        units = json.loads(record.units or "{}")
    except (TypeError, ValueError):
        units = {}
    currency = str(record.currency or DEFAULT_CURRENCY)
    return {
        "id": str(record.id),
        "usage_key": str(record.usage_key),
        "job_key": str(record.job_key or ""),
        "job_id": str(record.job_id or ""),
        "job_type": str(record.job_type or ""),
        "job_status": str(record.job_status or ""),
        "project_id": str(record.project_id or ""),
        "series_id": str(record.series_id or ""),
        "shot_id": str(record.shot_id or ""),
        "capability": str(record.capability),
        "capability_label": CAPABILITY_LABELS.get(str(record.capability), str(record.capability)),
        "provider": str(record.provider or ""),
        "model": str(record.model or ""),
        "quantity": int(record.quantity or 0),
        "secondary_quantity": int(record.secondary_quantity or 0),
        "base_unit": CAPABILITY_BASE_UNITS.get(str(record.capability), ""),
        "resolution": str(record.resolution or ""),
        "units": units,
        "status": str(record.status or ""),
        "error_code": str(record.error_code or ""),
        "cost_micro": int(record.cost_micro) if record.cost_micro is not None else None,
        "cost_known": bool(record.cost_known),
        "cost_source": str(record.cost_source or COST_SOURCE_UNKNOWN),
        "currency": currency,
        "duration_ms": int(record.duration_ms or 0),
        "created_at": record.created_at.isoformat() if isinstance(record.created_at, datetime) else None,
    }


def save_estimate(
    *,
    estimate_key: str,
    job_key: str = "",
    job_id: str = "",
    job_type: str = "",
    project_id: str = "",
    series_id: str = "",
    shot_id: str = "",
    currency: str = DEFAULT_CURRENCY,
    estimated_cost_micro: int | None = None,
    cost_known: bool = False,
    estimated_seconds: int | None = None,
    duration_source: str = "unknown",
    components: list[dict[str, Any]] | None = None,
    unknown_components: list[dict[str, Any]] | None = None,
    note: str = "",
) -> dict[str, Any]:
    """写入/更新一条任务估算（与 usage_records 分表，绝不覆盖实际用量）。"""

    db = SessionLocal()
    try:
        row = db.query(CostEstimate).filter(CostEstimate.estimate_key == estimate_key).first()
        if row is None:
            row = CostEstimate(id=uuid.uuid4().hex, estimate_key=_clip(estimate_key))
            db.add(row)
        row.job_key = _clip(job_key)
        row.job_id = _clip(job_id)
        row.job_type = _clip(job_type, 40)
        row.project_id = _clip(project_id)
        row.series_id = _clip(series_id)
        row.shot_id = _clip(shot_id)
        row.currency = str(currency or DEFAULT_CURRENCY)[:8]
        row.estimated_cost_micro = int(estimated_cost_micro) if estimated_cost_micro is not None else None
        row.cost_known = bool(cost_known)
        row.estimated_seconds = int(estimated_seconds) if estimated_seconds is not None else None
        row.duration_source = _clip(duration_source, 24)
        row.components = json.dumps(components or [], ensure_ascii=False)
        row.unknown_components = json.dumps(unknown_components or [], ensure_ascii=False)
        row.note = _clip(note, 200)
        db.commit()
        return estimate_dto(row)
    finally:
        db.close()


def estimate_dto(row: CostEstimate) -> dict[str, Any]:
    def _load(value: str) -> Any:
        try:
            return json.loads(value or "[]")
        except (TypeError, ValueError):
            return []

    return {
        "estimate_key": str(row.estimate_key),
        "job_key": str(row.job_key or ""),
        "job_id": str(row.job_id or ""),
        "job_type": str(row.job_type or ""),
        "project_id": str(row.project_id or ""),
        "series_id": str(row.series_id or ""),
        "shot_id": str(row.shot_id or ""),
        "estimated_cost_micro": int(row.estimated_cost_micro) if row.estimated_cost_micro is not None else None,
        "cost_known": bool(row.cost_known),
        "estimated_seconds": int(row.estimated_seconds) if row.estimated_seconds is not None else None,
        "duration_source": str(row.duration_source or "unknown"),
        "currency": str(row.currency or DEFAULT_CURRENCY),
        "components": _load(row.components),
        "unknown_components": _load(row.unknown_components),
        "note": str(row.note or ""),
        "created_at": row.created_at.isoformat() if isinstance(row.created_at, datetime) else None,
        "updated_at": row.updated_at.isoformat() if isinstance(row.updated_at, datetime) else None,
    }


def finalize_job(job_key: str, status: str, *, job_id: str = "") -> int:
    """把某个任务的用量记录标记为「归属任务已终结」。

    失败 / 取消的任务同样会被标记，其已发生的调用成本依旧可按 job_id / job_key
    查询——这是验收要求「任务失败后仍能查询已发生的调用成本」的落地点。
    """

    if not job_key:
        return 0
    db = SessionLocal()
    try:
        values: dict[Any, Any] = {
            UsageRecord.job_status: _clip(status, 24),
            UsageRecord.updated_at: datetime.utcnow(),
        }
        if job_id:
            values[UsageRecord.job_id] = _clip(job_id)
        updated = db.query(UsageRecord).filter(UsageRecord.job_key == job_key).update(values, synchronize_session=False)
        db.commit()
        return int(updated or 0)
    finally:
        db.close()


def bind_job_id(job_key: str, job_id: str) -> int:
    """任务抢占后把 job_id 补写到已落库的用量行（用量可能先于 job_id 落库）。"""

    if not job_key or not job_id:
        return 0
    db = SessionLocal()
    try:
        updated = (
            db.query(UsageRecord)
            .filter(UsageRecord.job_key == job_key, UsageRecord.job_id == "")
            .update({UsageRecord.job_id: _clip(job_id)}, synchronize_session=False)
        )
        db.commit()
        return int(updated or 0)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 统计查询
# ---------------------------------------------------------------------------


def _filters(
    query,
    *,
    project_id: str = "",
    series_id: str = "",
    shot_id: str = "",
    job_id: str = "",
    job_key: str = "",
    capability: str = "",
    job_type: str = "",
):
    if project_id:
        query = query.filter(UsageRecord.project_id == project_id)
    if series_id:
        query = query.filter(UsageRecord.series_id == series_id)
    if shot_id:
        query = query.filter(UsageRecord.shot_id == shot_id)
    if job_id:
        query = query.filter(UsageRecord.job_id == job_id)
    if job_key:
        query = query.filter(UsageRecord.job_key == job_key)
    if capability:
        query = query.filter(UsageRecord.capability == capability)
    if job_type:
        query = query.filter(UsageRecord.job_type == job_type)
    return query


def summarize(db: Session, **filters: Any) -> dict[str, Any]:
    """按给定维度汇总实际用量与成本。

    cost_micro 只统计「成本已知」的记录；未知成本的调用数量单独给出
    （unknown_call_count），前端据此显示「成本未知」而不是把它当成 0。
    """

    query = _filters(db.query(UsageRecord), **filters)
    rows = query.all()
    totals = {
        "call_count": len(rows),
        "cost_micro": 0,
        "cost_known": True,
        "unknown_call_count": 0,
        "failed_call_count": 0,
        "duration_ms": 0,
        "currency": DEFAULT_CURRENCY,
    }
    by_capability: dict[str, dict[str, Any]] = {}
    for row in rows:
        currency = str(row.currency or DEFAULT_CURRENCY)
        totals["currency"] = currency
        if row.cost_known and row.cost_micro is not None:
            totals["cost_micro"] += int(row.cost_micro)
        else:
            totals["unknown_call_count"] += 1
            totals["cost_known"] = False
        if str(row.status) != CALL_SUCCEEDED:
            totals["failed_call_count"] += 1
        totals["duration_ms"] += int(row.duration_ms or 0)

        entry = by_capability.setdefault(
            str(row.capability),
            {
                "capability": str(row.capability),
                "label": CAPABILITY_LABELS.get(str(row.capability), str(row.capability)),
                "call_count": 0,
                "cost_micro": 0,
                "cost_known": True,
                "unknown_call_count": 0,
                "quantity": 0,
                "secondary_quantity": 0,
                "seconds": 0,
                "base_unit": CAPABILITY_BASE_UNITS.get(str(row.capability), ""),
                "currency": currency,
            },
        )
        entry["call_count"] += 1
        entry["quantity"] += int(row.quantity or 0)
        entry["secondary_quantity"] += int(row.secondary_quantity or 0)
        if str(row.capability) in (pricing_service.CAPABILITY_VIDEO, pricing_service.CAPABILITY_FFMPEG):
            entry["seconds"] += int(row.quantity or 0)
        if row.cost_known and row.cost_micro is not None:
            entry["cost_micro"] += int(row.cost_micro)
        else:
            entry["unknown_call_count"] += 1
            entry["cost_known"] = False

    totals["by_capability"] = sorted(by_capability.values(), key=lambda item: item["capability"])
    return totals


def group_usage(db: Session, dimension: str, **filters: Any) -> list[dict[str, Any]]:
    """按维度分组统计（项目 / 剧集 / 镜头 / 任务类型 / 能力 / provider / 模型）。"""

    column = _GROUP_COLUMNS.get(str(dimension))
    if column is None:
        raise ValueError(f"不支持的分组维度: {dimension}，可选值: {', '.join(GROUP_BY_DIMENSIONS)}")
    query = _filters(db.query(UsageRecord), **filters)
    rows = query.all()
    if dimension == GROUP_BY_EPISODE:
        rows = [row for row in rows if str(row.project_id or "")]
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(getattr(row, column.key) or "")
        entry = grouped.setdefault(
            key,
            {
                "key": key,
                "call_count": 0,
                "cost_micro": 0,
                "cost_known": True,
                "unknown_call_count": 0,
                "currency": str(row.currency or DEFAULT_CURRENCY),
            },
        )
        entry["call_count"] += 1
        if row.cost_known and row.cost_micro is not None:
            entry["cost_micro"] += int(row.cost_micro)
        else:
            entry["unknown_call_count"] += 1
            entry["cost_known"] = False
    return sorted(grouped.values(), key=lambda item: (-item["cost_micro"], item["key"]))


def job_usage_map(db: Session, job_keys: list[str]) -> dict[str, dict[str, Any]]:
    """批量取多个任务的成本/耗时聚合（任务中心列表用，避免 N+1）。"""

    keys = [str(key) for key in job_keys if key]
    if not keys:
        return {}
    rows = db.query(UsageRecord).filter(UsageRecord.job_key.in_(keys)).all()
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = result.setdefault(
            str(row.job_key),
            {
                "call_count": 0,
                "cost_micro": 0,
                "cost_known": True,
                "unknown_call_count": 0,
                "failed_call_count": 0,
                "provider_seconds": 0,
                "provider_duration_ms": 0,
                "currency": str(row.currency or DEFAULT_CURRENCY),
                "by_capability": {},
            },
        )
        entry["call_count"] += 1
        if row.cost_known and row.cost_micro is not None:
            entry["cost_micro"] += int(row.cost_micro)
        else:
            entry["unknown_call_count"] += 1
            entry["cost_known"] = False
        if str(row.status) != CALL_SUCCEEDED:
            entry["failed_call_count"] += 1
        if str(row.capability) in (pricing_service.CAPABILITY_VIDEO, pricing_service.CAPABILITY_FFMPEG):
            entry["provider_seconds"] += int(row.quantity or 0)
        entry["provider_duration_ms"] += int(row.duration_ms or 0)
        per_capability = entry["by_capability"].setdefault(
            str(row.capability),
            {"capability": str(row.capability), "call_count": 0, "cost_micro": 0, "cost_known": True},
        )
        per_capability["call_count"] += 1
        if row.cost_known and row.cost_micro is not None:
            per_capability["cost_micro"] += int(row.cost_micro)
        else:
            per_capability["cost_known"] = False
    for entry in result.values():
        entry["by_capability"] = sorted(entry["by_capability"].values(), key=lambda item: item["capability"])
    return result


def job_estimate_map(db: Session, job_keys: list[str]) -> dict[str, dict[str, Any]]:
    """批量取多个任务的估算行（任务中心显示「预计 vs 实际」）。"""

    keys = [str(key) for key in job_keys if key]
    if not keys:
        return {}
    rows = db.query(CostEstimate).filter(CostEstimate.job_key.in_(keys)).all()
    return {str(row.job_key): estimate_dto(row) for row in rows}


def list_records(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    **filters: Any,
) -> dict[str, Any]:
    """分页列出实际用量明细。"""

    size = max(1, min(200, int(page_size or 20)))
    current = max(1, int(page or 1))
    query = _filters(db.query(UsageRecord), **filters)
    total = query.count()
    rows = (
        query.order_by(UsageRecord.created_at.desc(), UsageRecord.id.desc())
        .offset((current - 1) * size)
        .limit(size)
        .all()
    )
    return {
        "items": [_record_dto(row) for row in rows],
        "total": int(total),
        "page": current,
        "page_size": size,
        "pages": max(1, (int(total) + size - 1) // size),
    }


def usage_counts_by_project(db: Session, project_ids: list[str]) -> dict[str, dict[str, Any]]:
    """按项目批量汇总（项目列表页显示已用成本）。"""

    ids = [str(item) for item in project_ids if item]
    if not ids:
        return {}
    rows = (
        db.query(
            UsageRecord.project_id,
            func.count(UsageRecord.id),
            func.coalesce(func.sum(UsageRecord.cost_micro), 0),
            func.sum(func.coalesce(UsageRecord.cost_known, 0)),
        )
        .filter(UsageRecord.project_id.in_(ids))
        .group_by(UsageRecord.project_id)
        .all()
    )
    return {
        str(project_id): {
            "call_count": int(count or 0),
            "cost_micro": int(cost or 0),
            "cost_known": int(known or 0) >= int(count or 0),
            "unknown_call_count": max(0, int(count or 0) - int(known or 0)),
        }
        for project_id, count, cost, known in rows
    }


__all__ = [
    "CALL_CANCELLED",
    "CALL_FAILED",
    "CALL_STATUSES",
    "CALL_SUCCEEDED",
    "GROUP_BY_DIMENSIONS",
    "UsageScope",
    "bind_job_id",
    "bind_scope",
    "current_scope",
    "estimate_dto",
    "finalize_job",
    "group_usage",
    "job_estimate_map",
    "job_usage_map",
    "list_records",
    "record_cancelled",
    "record_failure",
    "record_metadata",
    "record_usage",
    "reset_scope",
    "resolve_scope",
    "save_estimate",
    "summarize",
    "usage_counts_by_project",
    "usage_scope",
]
