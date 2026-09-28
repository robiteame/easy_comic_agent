"""任务中心的查询、统计与详情层。

这一层只做「读」：把 background_jobs 行转成稳定 DTO、做筛选分页排序、统计活动
与失败数量、组装详情（含历次尝试）。写操作（取消/重试/续跑/清理）在
services.job_actions，重新派发在 services.job_dispatch；这里不复制业务逻辑，也
不新建第二套任务表。

安全约束（与 services.job_dto 一致）：

- DTO 不含 run_token；
- 失败只回稳定错误码 + 脱敏截断后的短消息，完整堆栈留在服务端日志；
- 作用域占位行（scope-block:）不是用户任务，一律不出现在列表/详情里。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, or_
from sqlalchemy.orm import Query, Session

from config import settings
from models import BackgroundJob
from services import usage_service
from services.job_dto import as_utc, estimate_eta_seconds, job_dto
from services.job_types import (
    ACTIVE_STATUSES,
    DISPATCHABLE_JOB_TYPES,
    JOB_STATUSES,
    JOB_TYPES,
    TERMINAL_STATUSES,
    error_code_label,
    parse_job_key,
)
from services.task_registry import SCOPE_BLOCK_KEY_PREFIX

# 列表默认排序：最近更新的任务在最前。
_LIST_ORDER = (BackgroundJob.updated_at.desc(), BackgroundJob.created_at.desc())
# 搜索命中的字段：只搜用户可见的文本，不搜错误堆栈或内部键。
_SEARCH_COLUMNS = ("display_name", "current_step", "message", "error_message", "project_id")
# 估算 ETA 时最多参考同类型最近多少次真实耗时。
_ETA_SAMPLE_LIMIT = 20
_ESCAPE_CHAR = "!"


def _escape_like(term: str) -> str:
    return term.replace(_ESCAPE_CHAR, _ESCAPE_CHAR * 2).replace("%", f"{_ESCAPE_CHAR}%").replace("_", f"{_ESCAPE_CHAR}_")


def _iso_utc(value: datetime | None) -> str | None:
    """任务中心对外的时刻一律带 UTC 时区（历史 naive 行按 UTC 解释）。"""

    moment = as_utc(value)
    return moment.isoformat() if moment is not None else None


def _not_scope_block(query: Query) -> Query:
    return query.filter(~BackgroundJob.idempotency_key.startswith(SCOPE_BLOCK_KEY_PREFIX))


def _prefix_clause(canonical: str):
    """匹配规范键本身以及它的 #attempt-N 归档行（不用 LIKE，避免通配符歧义）。"""

    return func.substr(BackgroundJob.idempotency_key, 1, len(canonical)) == canonical


@dataclass(frozen=True)
class JobQuery:
    project_id: str = ""
    statuses: tuple[str, ...] = ()
    job_types: tuple[str, ...] = ()
    scope: str = ""
    active_only: bool = False
    search: str = ""
    page: int = 1
    page_size: int = 20

    def normalized(self) -> "JobQuery":
        page = max(1, int(self.page or 1))
        size = int(self.page_size or settings.JOB_LIST_DEFAULT_PAGE_SIZE)
        size = max(1, min(settings.JOB_LIST_MAX_PAGE_SIZE, size))
        return JobQuery(
            project_id=(self.project_id or "").strip(),
            statuses=tuple(item for item in self.statuses if item in JOB_STATUSES),
            job_types=tuple(item for item in self.job_types if item in JOB_TYPES),
            scope=(self.scope or "").strip(),
            active_only=bool(self.active_only),
            search=(self.search or "").strip()[:120],
            page=page,
            page_size=size,
        )


def build_query(db: Session, query: JobQuery, *, drop_status: bool = False, drop_type: bool = False) -> Query:
    """构造基础查询。drop_* 用于统计「忽略该筛选维度」的计数。"""

    q = _not_scope_block(db.query(BackgroundJob))
    if query.project_id:
        q = q.filter(BackgroundJob.project_id == query.project_id)
    if query.scope:
        q = q.filter(BackgroundJob.scope == query.scope)
    if not drop_status:
        if query.active_only:
            q = q.filter(BackgroundJob.status.in_(ACTIVE_STATUSES))
        elif query.statuses:
            q = q.filter(BackgroundJob.status.in_(query.statuses))
    if query.job_types and not drop_type:
        q = q.filter(BackgroundJob.job_type.in_(query.job_types))
    if query.search:
        pattern = f"%{_escape_like(query.search)}%"
        q = q.filter(
            or_(*[getattr(BackgroundJob, column).ilike(pattern, escape=_ESCAPE_CHAR) for column in _SEARCH_COLUMNS])
        )
    return q


def active_successor_ids(db: Session, job_ids: list[str]) -> set[str]:
    """返回「已有正在执行的新尝试」的任务 id 集合。"""

    if not job_ids:
        return set()
    rows = (
        db.query(BackgroundJob.retry_of)
        .filter(BackgroundJob.retry_of.in_(job_ids), BackgroundJob.status.in_(ACTIVE_STATUSES))
        .all()
    )
    return {str(row[0]) for row in rows if row[0]}


def _eta_samples(db: Session, job_types: set[str]) -> dict[str, list[float]]:
    if not job_types:
        return {}
    rows = (
        db.query(BackgroundJob.job_type, BackgroundJob.started_at, BackgroundJob.finished_at)
        .filter(
            BackgroundJob.job_type.in_(job_types),
            BackgroundJob.status == "completed",
            BackgroundJob.started_at.isnot(None),
            BackgroundJob.finished_at.isnot(None),
        )
        .order_by(BackgroundJob.finished_at.desc())
        .limit(_ETA_SAMPLE_LIMIT * max(1, len(job_types)))
        .all()
    )
    samples: dict[str, list[float]] = {}
    for job_type, started, finished in rows:
        try:
            seconds = float((finished - started).total_seconds())
        except TypeError:
            continue
        if seconds > 0:
            samples.setdefault(str(job_type), []).append(seconds)
    return samples


def _dto(
    row: BackgroundJob,
    *,
    now: datetime,
    successors: set[str],
    samples: dict[str, list[float]],
    usages: dict[str, dict[str, Any]] | None = None,
    estimates: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    eta = None
    if str(row.status) in ACTIVE_STATUSES:
        eta = estimate_eta_seconds(
            samples.get(str(row.job_type), []),
            int(row.progress or 0),
            minimum_samples=settings.JOB_ETA_MIN_SAMPLES,
        )
    canonical = parse_job_key(str(row.idempotency_key)).canonical
    usage = (usages or {}).get(canonical)
    estimate = (estimates or {}).get(canonical)
    return job_dto(
        row,
        now=now,
        has_active_successor=row.id in successors,
        eta_seconds=eta,
        dispatchable=str(row.job_type) in DISPATCHABLE_JOB_TYPES,
        usage=usage,
        estimate=estimate,
    )


def _cost_maps(
    db: Session, rows: list[BackgroundJob]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """批量取本页任务的成本与估算（按幂等键聚合，避免 N+1 查询）。"""

    keys = [parse_job_key(str(row.idempotency_key)).canonical for row in rows]
    return usage_service.job_usage_map(db, keys), usage_service.job_estimate_map(db, keys)


def list_jobs(db: Session, query: JobQuery) -> dict[str, Any]:
    """分页返回任务列表、总数、活动数与状态/类型统计。"""

    normalized = query.normalized()
    base = build_query(db, normalized)
    total = base.order_by(None).count()
    rows = (
        base.order_by(*_LIST_ORDER)
        .offset((normalized.page - 1) * normalized.page_size)
        .limit(normalized.page_size)
        .all()
    )

    successors = active_successor_ids(db, [row.id for row in rows])
    samples = _eta_samples(db, {str(row.job_type) for row in rows if str(row.status) in ACTIVE_STATUSES})
    usages, estimates = _cost_maps(db, rows)
    now = datetime.now(timezone.utc)
    items = [
        _dto(row, now=now, successors=successors, samples=samples, usages=usages, estimates=estimates)
        for row in rows
    ]

    active_count = (
        build_query(db, normalized, drop_status=True).filter(BackgroundJob.status.in_(ACTIVE_STATUSES)).count()
    )
    status_counts = {
        str(status): int(count)
        for status, count in build_query(db, normalized, drop_status=True)
        .with_entities(BackgroundJob.status, func.count(BackgroundJob.id))
        .group_by(BackgroundJob.status)
        .all()
    }
    type_counts = {
        str(job_type): int(count)
        for job_type, count in build_query(db, normalized, drop_type=True)
        .with_entities(BackgroundJob.job_type, func.count(BackgroundJob.id))
        .group_by(BackgroundJob.job_type)
        .all()
    }

    return {
        "items": items,
        "total": int(total),
        "page": normalized.page,
        "page_size": normalized.page_size,
        "pages": max(1, (int(total) + normalized.page_size - 1) // normalized.page_size),
        "active_count": int(active_count),
        "status_counts": status_counts,
        "job_type_counts": type_counts,
        "generated_at": now.isoformat(),
    }


def job_stats(db: Session, *, project_id: str = "") -> dict[str, Any]:
    """轻量统计：活动数、失败数、最近一次任务，供任务入口徽标使用。"""

    query = _not_scope_block(db.query(BackgroundJob))
    if project_id:
        query = query.filter(BackgroundJob.project_id == project_id)
    counts = {
        str(status): int(count)
        for status, count in query.with_entities(BackgroundJob.status, func.count(BackgroundJob.id))
        .group_by(BackgroundJob.status)
        .all()
    }
    latest = query.order_by(*_LIST_ORDER).first()
    usages, estimates = _cost_maps(db, [latest] if latest is not None else [])
    now = datetime.now(timezone.utc)
    usage_summary = usage_service.summarize(db, project_id=project_id) if project_id else usage_service.summarize(db)
    return {
        "project_id": project_id,
        "active_count": sum(counts.get(status, 0) for status in ACTIVE_STATUSES),
        "failed_count": counts.get("failed", 0),
        "total": sum(counts.values()),
        "status_counts": counts,
        "latest_job": job_dto(latest, now=now, usage=usages, estimate=estimates) if latest is not None else None,
        "cost": {
            "currency": str(usage_summary.get("currency") or "CNY"),
            "cost_micro": int(usage_summary.get("cost_micro") or 0),
            "cost_known": bool(usage_summary.get("cost_known", True)),
            "unknown_call_count": int(usage_summary.get("unknown_call_count") or 0),
            "call_count": int(usage_summary.get("call_count") or 0),
        },
        "generated_at": now.isoformat(),
    }


def get_job(db: Session, job_id: str) -> BackgroundJob | None:
    if not job_id:
        return None
    return _not_scope_block(db.query(BackgroundJob)).filter(BackgroundJob.id == job_id).first()


def attempt_history(db: Session, job: BackgroundJob) -> list[dict[str, Any]]:
    """同一操作的历次尝试（含被归档的失败记录），按创建时间升序。"""

    canonical = parse_job_key(job.idempotency_key).canonical
    rows = (
        db.query(BackgroundJob)
        .filter(_prefix_clause(canonical))
        .order_by(BackgroundJob.created_at.asc())
        .all()
    )
    history = [
        row
        for row in rows
        if row.idempotency_key == canonical or row.idempotency_key.rsplit("#attempt-", 1)[-1].isdigit()
    ]
    now = datetime.utcnow()
    return [
        {
            "id": str(row.id),
            "attempt": max(1, int(row.attempt or 1)),
            "status": str(row.status),
            "error_code": str(row.error_code or ""),
            "error_code_label": error_code_label(str(row.error_code or "")),
            "error_message": str(row.error_message or row.error or "")[:240],
            "started_at": _iso_utc(row.started_at),
            "finished_at": _iso_utc(row.finished_at),
            "updated_at": _iso_utc(row.updated_at),
            "duration_seconds": max(
                0,
                int(((row.finished_at or now) - (row.started_at or row.created_at or now)).total_seconds()),
            ),
            "retry_of": str(row.retry_of) if row.retry_of else None,
        }
        for row in history
    ]


def _retry_of_attempt(db: Session, job: BackgroundJob) -> int | None:
    if not job.retry_of:
        return None
    previous = db.query(BackgroundJob.attempt).filter(BackgroundJob.id == job.retry_of).scalar()
    return int(previous) if previous is not None else None


def job_detail(db: Session, job: BackgroundJob) -> dict[str, Any]:
    """详情 = DTO + 尝试历史 + 重试关系。DTO 已含可执行动作与禁用原因。"""

    canonical = parse_job_key(job.idempotency_key).canonical
    successor_rows = (
        db.query(BackgroundJob.id)
        .filter(BackgroundJob.retry_of == job.id, BackgroundJob.status.in_(ACTIVE_STATUSES))
        .all()
    )
    latest = db.query(BackgroundJob).filter(BackgroundJob.idempotency_key == canonical).first()
    samples = _eta_samples(db, {str(job.job_type)})
    usages, estimates = _cost_maps(db, [job])
    dto = _dto(
        job,
        now=datetime.now(timezone.utc),
        successors={str(row[0]) for row in successor_rows},
        samples=samples,
        usages=usages,
        estimates=estimates,
    )
    dto["attempts"] = attempt_history(db, job)
    dto["latest_attempt_job_id"] = str(latest.id) if latest is not None else None
    dto["retry_relationship"] = {
        "retry_of": dto["retry_of"],
        "retry_of_attempt": _retry_of_attempt(db, job),
        "attempt": dto["attempt"],
    }
    return dto


def purge_jobs(db: Session, query: JobQuery) -> int:
    """清理历史任务：只删终态行，一次最多删除配置上限的行数。"""

    normalized = query.normalized()
    rows = (
        build_query(db, normalized, drop_status=True)
        .filter(BackgroundJob.status.in_(TERMINAL_STATUSES))
        .order_by(*_LIST_ORDER)
        .limit(settings.JOB_CLEANUP_MAX_ROWS)
        .all()
    )
    for row in rows:
        db.delete(row)
    db.commit()
    return len(rows)


def delete_job(db: Session, job: BackgroundJob) -> bool:
    """删除单条历史任务；运行中的任务必须先取消。"""

    if str(job.status) not in TERMINAL_STATUSES:
        return False
    db.delete(job)
    db.commit()
    return True


__all__ = [
    "JobQuery",
    "active_successor_ids",
    "attempt_history",
    "build_query",
    "delete_job",
    "get_job",
    "job_detail",
    "job_stats",
    "list_jobs",
    "purge_jobs",
]
