"""Durable claim and lifecycle handling for background jobs.

The local task map is only an execution convenience. SQLite records are the
authority, which makes competing requests and server restarts explicit.

任务中心复用同一张 `background_jobs` 表与同一套抢占逻辑，不引入第二套任务
注册系统。新增能力都向后兼容：

- `claim()` 在同一幂等键的上一次尝试已进入终态时，把它归档为
  `key#attempt-N` 并新建 `attempt+1` 的新行——历史失败记录不会被覆盖，规范
  键依旧归最新尝试所有，因此所有既有调用方（进度上报、完成回调）无需改动；
- 状态迁移集中到 `services.job_types.ALLOWED_TRANSITIONS`，终态是吸收态，
  任何执行路径都不能把终态直接改回 running；
- 所有写入都按 run token 过滤，旧尝试的迟到回调无法覆盖新尝试；
- 状态变化会向任务中心推送事件（事件只含 DTO，不含 run token）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Coroutine, Coroutine as _Coroutine

from sqlalchemy import text

from config import settings
from db import SessionLocal
from models import BackgroundJob, Project, Shot
from services import budget_service, usage_service
from services.job_debug import (
    append_event as append_debug_event,
    make_event as make_debug_event,
    parse_events as parse_debug_events,
    publish_debug_event,
)
from services.error_reporter import ERROR_BACKGROUND_JOB, log_failure, redact, summarize
from services.job_dto import job_dto
from services.job_events import (
    EVENT_JOB_CREATED,
    EVENT_JOB_PROGRESS,
    EVENT_JOB_RETRY_STARTED,
    EVENT_JOB_UPDATED,
    has_job_listeners,
    publish_job_event,
    terminal_event_for,
)
from services.job_types import (
    ACTIVE_STATUSES,
    ERROR_CODE_BUDGET_EXCEEDED,
    STATUS_CANCELLED,
    STATUS_CANCELLING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    TERMINAL_STATUSES,
    JobKey,
    archived_key,
    can_transition,
    error_code_for_status,
    job_type_label,
    parse_job_key,
)


logger = logging.getLogger(__name__)

# 展示名称与落库失败文本的上限：完整堆栈只写服务端日志。
_DISPLAY_NAME_CHARS = 80
_ERROR_TEXT_CHARS = 2000
_ERROR_MESSAGE_CHARS = 240

# 作用域占位行的幂等键前缀：它们只用于在删除/重跑期间占住作用域，不是用户任务。
SCOPE_BLOCK_KEY_PREFIX = "scope-block:"

_tasks: dict[str, asyncio.Task] = {}
_claim_tokens: dict[str, str] = {}
_task_tokens: dict[asyncio.Task, str] = {}
# 任务 -> 用量归属（project / shot / job）。由 start() 在协程内绑定 contextvar，
# 这样 LLM/图像/视频/TTS/FFmpeg 服务无需层层透传参数即可把用量记到正确的项目与镜头。
_claim_scopes: dict[str, usage_service.UsageScope] = {}


def unique_archived_key(db, key: str, attempt: int, *, exclude_id: str | None = None) -> str:
    """Return an unused historical key, preserving the normal attempt suffix."""
    candidate_attempt = max(1, int(attempt or 1))
    while True:
        candidate = archived_key(key, candidate_attempt)
        query = db.query(BackgroundJob.id).filter(BackgroundJob.idempotency_key == candidate)
        if exclude_id is not None:
            query = query.filter(BackgroundJob.id != exclude_id)
        if query.first() is None:
            return candidate
        candidate_attempt += 1


@dataclass(frozen=True)
class ScopeCancellation:
    cancelled_jobs: int
    blocker_ids: tuple[str, ...]


@dataclass(frozen=True)
class ClaimResult:
    """一次抢占的结果：成功 / 作用域被占 / 被硬预算拦下。

    路由层据此区分「已有任务在跑（409 deduplicated）」与「预算不足（409
    budget_exceeded）」，后者必须给出可见的错误码，而不是静默合并成一次去重。
    """

    claimed: bool
    reason: str = ""  # "" | scope_busy | budget_exceeded
    error_code: str = ""
    message: str = ""
    budget: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked_by_budget(self) -> bool:
        return bool(self.error_code) and self.error_code == ERROR_CODE_BUDGET_EXCEEDED


def claim(
    key: str,
    scope: str,
    *,
    version: int = 0,
    job_type: str | None = None,
    project_id: str | None = None,
    display_name: str | None = None,
    current_step: str = "",
    message: str = "",
) -> bool:
    """抢占任务（兼容既有调用方）；预算被拒时同样返回 False。"""

    return claim_job(
        key,
        scope,
        version=version,
        job_type=job_type,
        project_id=project_id,
        display_name=display_name,
        current_step=current_step,
        message=message,
    ).claimed


def claim_job(
    key: str,
    scope: str,
    *,
    version: int = 0,
    job_type: str | None = None,
    project_id: str | None = None,
    display_name: str | None = None,
    current_step: str = "",
    message: str = "",
) -> ClaimResult:
    """Atomically acquire an operation and its project/shot scope.

    新尝试永远占规范幂等键 ``key``。若上一次尝试已经进入终态，则先把旧行改名为
    ``key#attempt-N`` 归档（保留失败记录与 attempt 信息），再插入 `attempt+1`
    的新行：既满足「不覆盖旧任务」，又让既有调用方继续用同一个 key 上报进度。

    抢占之前先做预算检查（估算 + 预留），硬预算超限时直接拒绝启动；作用域忙碌时
    不会留下任何预留 —— 预留只在真正抢占成功后才有归属。
    """

    identity = parse_job_key(key)
    resolved_type = job_type or identity.job_type
    resolved_project = project_id if project_id is not None else _project_id_for_scope_readonly(scope, identity)
    if _scope_busy(scope, key):
        return ClaimResult(False, reason="scope_busy", message="该作用域已有任务在执行")

    decision = budget_service.check_and_reserve(
        job_key=key,
        job_type=resolved_type,
        project_id=resolved_project or "",
        shot_id=identity.owner_id if identity.owner_type == "shot" else "",
    )
    if not decision.allowed:
        return ClaimResult(
            False,
            reason="budget_exceeded",
            error_code=decision.code or ERROR_CODE_BUDGET_EXCEEDED,
            message=decision.message or "已超出项目硬预算，任务未启动",
            budget=decision.to_dict(),
        )

    result = _claim_row(
        key,
        scope,
        version=version,
        job_type=resolved_type,
        project_id=resolved_project,
        display_name=display_name,
        current_step=current_step,
        message=message,
    )
    if not result.claimed:
        # 抢占失败（作用域被别的进程占住）：释放刚建立的预留，避免额度被永久占着。
        budget_service.release_reservation(key)
        return result
    return replace(result, budget=decision.to_dict())


def _project_id_for_scope_readonly(scope: str, identity: JobKey) -> str:
    if identity.owner_type == "project" and identity.owner_id:
        return identity.owner_id
    if identity.owner_type == "shot" and identity.owner_id:
        db = SessionLocal()
        try:
            return str(db.query(Shot.project_id).filter(Shot.id == identity.owner_id).scalar() or "")
        finally:
            db.close()
    return ""


def _scope_busy(scope: str, key: str) -> bool:
    """只读预检：该作用域是否已被别的任务占用（真正的判定仍在事务里）。"""

    db = SessionLocal()
    try:
        owner = _scope_owner(db, scope)
        existing = db.query(BackgroundJob).filter(BackgroundJob.idempotency_key == key).first()
        if owner is not None and owner.idempotency_key != key:
            # 同一镜头的其它 queued queue-items 只是占位，不应阻止当前阶段
            # 用同一作用域取得真实 run token；真实运行任务仍然严格互斥。
            if not (_is_queue_placeholder(existing) and _is_queue_placeholder(owner)):
                return True
        return bool(existing and existing.status in ACTIVE_STATUSES and not _is_queue_placeholder(existing))
    except Exception:  # noqa: BLE001 - 预检失败不阻断抢占，交给事务内判定
        return False
    finally:
        db.close()


def _claim_row(
    key: str,
    scope: str,
    *,
    version: int,
    job_type: str,
    project_id: str | None,
    display_name: str | None,
    current_step: str,
    message: str,
) -> ClaimResult:
    """真正写库的抢占：作用域互斥、尝试归档与 run token 全部在这一步完成。"""

    now = datetime.utcnow()
    run_token = uuid.uuid4().hex
    db = SessionLocal()
    try:
        # Serializes the scope read and claim for all local server processes.
        db.execute(text("BEGIN IMMEDIATE"))
        existing = db.query(BackgroundJob).filter(BackgroundJob.idempotency_key == key).first()
        if existing and existing.status in ACTIVE_STATUSES and not _is_queue_placeholder(existing):
            db.rollback()
            return ClaimResult(False, reason="scope_busy", message="该任务已有正在执行的尝试")
        scope_owner = _scope_owner(db, scope)
        if scope_owner and scope_owner.idempotency_key != key:
            if not (_is_queue_placeholder(existing) and _is_queue_placeholder(scope_owner)):
                db.rollback()
                return ClaimResult(False, reason="scope_busy", message="该作用域已有任务在执行")

        identity = parse_job_key(key)
        resolved_type = job_type or identity.job_type
        resolved_project = project_id if project_id is not None else _project_id_for_scope(db, scope)
        attempt = 1
        retry_of: str | None = None
        queue_metadata: dict[str, Any] = {}
        if existing is not None:
            attempt = max(1, int(existing.attempt or 1)) + 1
            retry_of = existing.id
            # 队列项在真正派发时会被归档；把队列元数据带到新的运行行，
            # 任务中心因此仍能按批次、阶段和镜头关联展示活动任务。
            for field_name in (
                "queue_batch_id", "queue_position", "queue_priority", "queue_order",
                "queue_stage", "queue_shot_id", "queue_dependency_ids", "queue_blocked_reason",
                "queue_concurrency", "queue_paused", "queue_resume_missing",
                "queue_reuse_audio", "queue_force_confirmed", "queue_requested_version",
            ):
                if hasattr(existing, field_name):
                    queue_metadata[field_name] = getattr(existing, field_name)
            # 归档的只是键与身份：状态、错误与时间戳全部保持原样。
            #
            # 用 Core UPDATE 显式写入 updated_at，而不是改 ORM 属性：后者不会把该列
            # 放进 SET 子句，列的 onupdate 会把归档行的时间戳刷成「现在」，让历史失败
            # 记录看起来刚刚发生过（并顶到按 updated_at 排序的列表最前面）。
            db.query(BackgroundJob).filter(BackgroundJob.id == existing.id).update(
                {
                    BackgroundJob.idempotency_key: unique_archived_key(
                        db, existing.idempotency_key, int(existing.attempt or 1), exclude_id=existing.id
                    ),
                    BackgroundJob.updated_at: existing.updated_at,
                },
                synchronize_session=False,
            )
            db.expunge(existing)
            db.flush()
        job = BackgroundJob(
            id=uuid.uuid4().hex,
            idempotency_key=key,
            scope=scope,
            status="running",
            progress=0,
            version=version,
            run_token=run_token,
            created_at=now,
            started_at=now,
            updated_at=now,
            project_id=resolved_project,
            job_type=resolved_type,
            display_name=display_name or _display_name(db, identity, resolved_project),
            current_step=current_step[:120],
            message=message[:_ERROR_MESSAGE_CHARS],
            attempt=attempt,
            retry_of=retry_of,
            **queue_metadata,
        )
        db.add(job)
        debug_raw, debug_revision = append_debug_event(
            "[]",
            make_debug_event(
                "lifecycle",
                message or "任务已创建，等待执行",
                step=current_step,
                progress=0,
                status="request",
            ),
        )
        job.debug_events = debug_raw
        job.debug_revision = debug_revision
        usage, estimate = _cost_context(db, key)
        payload = job_dto(job, now=now, usage=usage, estimate=estimate)
        db.commit()
        publish_debug_event(job, parse_debug_events(debug_raw)[-1], debug_revision)
        _claim_tokens[key] = run_token
        _record_claim_scope(key, scope, resolved_type, job, identity)
        # 已经把 job_id 补进估算 / 预留 / 已落库的用量行，任务中心可据此直查成本。
        budget_service.attach_job_id(key, job.id)
        usage_service.bind_job_id(key, job.id)
        publish_job_event(EVENT_JOB_RETRY_STARTED if retry_of else EVENT_JOB_CREATED, payload)
        return ClaimResult(True, message="任务已启动")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _cost_context(db, key: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """事件负载里的成本快照：实际用量 + 启动前估算（两者分表存储）。

    任务中心的事件是增量的（不做全量刷新），因此每个事件都带上成本，避免界面在
    任务进行中把已经显示的成本又退回「暂无用量」。
    """

    try:
        usage = usage_service.job_usage_map(db, [key]).get(key)
        estimate = usage_service.job_estimate_map(db, [key]).get(key)
    except Exception:  # noqa: BLE001 - 成本查询失败不能影响任务事件本身
        logger.warning("任务成本快照读取失败: key=%s", key, exc_info=True)
        return None, None
    return usage, estimate


def _record_claim_scope(key: str, scope: str, job_type: str, job: BackgroundJob, identity: JobKey) -> None:
    """记住这次任务的用量归属，供 start() 在协程内绑定 contextvar。"""

    shot_id = identity.owner_id if identity.owner_type == "shot" else ""
    _claim_scopes[key] = usage_service.UsageScope(
        project_id=str(job.project_id or ""),
        shot_id=shot_id,
        job_key=key,
        job_id=str(job.id),
        job_type=str(job_type or ""),
    )


def _project_id_for_scope(db, scope: str) -> str:
    """从作用域解析所属项目：``project:<id>`` 直接取，``shot:<id>`` 查一次镜头表。"""

    owner_type, _, owner_id = (scope or "").partition(":")
    if owner_type == "project":
        return owner_id
    if owner_type == "shot" and owner_id:
        return db.query(Shot.project_id).filter(Shot.id == owner_id).scalar() or ""
    return ""


def _display_name(db, identity: JobKey, project_id: str) -> str:
    """生成不含密钥与完整 prompt 的短任务名。"""

    base = job_type_label(identity.job_type)
    title = ""
    if project_id:
        title = (db.query(Project.title).filter(Project.id == project_id).scalar() or "").strip()
    if identity.owner_type == "project" and identity.operation == "pipeline":
        mode = "全自动" if identity.qualifier == "auto" else "手动"
        base = f"{base}（{mode}）"
    elif identity.owner_type == "shot" and identity.owner_id:
        sequence = db.query(Shot.sequence).filter(Shot.id == identity.owner_id).scalar()
        if sequence is not None:
            base = f"镜头 {sequence} · {base}"
    label = f"{base} · {title}" if title else base
    return label[:_DISPLAY_NAME_CHARS]


def active(key: str) -> bool:
    task = _tasks.get(key)
    if task is not None and not task.done():
        return True
    db = SessionLocal()
    try:
        return bool(
            db.query(BackgroundJob)
            .filter(BackgroundJob.idempotency_key == key, BackgroundJob.status.in_(ACTIVE_STATUSES))
            .first()
        )
    finally:
        db.close()


def _report_json(report: dict[str, Any] | str | None) -> str:
    if isinstance(report, str):
        return report[:20000]
    try:
        return json.dumps(report or {}, ensure_ascii=False)[:20000]
    except (TypeError, ValueError):
        return "{}"


def update_progress(
    key: str,
    progress: int,
    *,
    run_token: str | None = None,
    current_step: str | None = None,
    message: str | None = None,
    report: dict[str, Any] | str | None = None,
) -> bool:
    """写入进度与可读步骤；``current_step`` / ``message`` 为可选增量。

    只有持有当前 run token 的调用方才写得进去，旧尝试的迟到回调会被静默丢弃。
    """

    token = run_token or _run_token_for_current_task(key)
    if not token:
        return False
    progress_value = max(0, min(100, int(progress)))
    db = SessionLocal()
    try:
        job = (
            db.query(BackgroundJob)
            .filter(
                BackgroundJob.idempotency_key == key,
                BackgroundJob.run_token == token,
                BackgroundJob.status.in_(ACTIVE_STATUSES),
            )
            .first()
        )
        if job is None:
            return False
        values: dict[Any, Any] = {
            BackgroundJob.progress: progress_value,
            BackgroundJob.updated_at: datetime.utcnow(),
        }
        resolved_step = str(current_step if current_step is not None else job.current_step or "")[:120]
        resolved_message = summarize(message, limit=_ERROR_MESSAGE_CHARS) if message is not None else str(job.message or "")
        if current_step is not None:
            values[BackgroundJob.current_step] = resolved_step
        if message is not None:
            values[BackgroundJob.message] = resolved_message
        if report is not None:
            values[BackgroundJob.report] = _report_json(report)
        debug_event = make_debug_event(
            "progress",
            resolved_message or "进度已更新",
            step=resolved_step,
            progress=progress_value,
            status="progress",
        )
        debug_raw, debug_revision = append_debug_event(
            job.debug_events,
            debug_event,
            revision=int(job.debug_revision or 0),
        )
        values[BackgroundJob.debug_events] = debug_raw
        values[BackgroundJob.debug_revision] = debug_revision
        updated = (
            db.query(BackgroundJob)
            .filter(BackgroundJob.id == job.id, BackgroundJob.run_token == token)
            .update(values, synchronize_session=False)
        )
        db.commit()
        if updated:
            db.refresh(job)
            publish_debug_event(job, debug_event, debug_revision)
            if has_job_listeners():
                usage, estimate = _cost_context(db, key)
                publish_job_event(EVENT_JOB_PROGRESS, job_dto(job, usage=usage, estimate=estimate))
        return bool(updated)
    finally:
        db.close()


def snapshot(key: str) -> dict[str, Any] | None:
    """Return a detached, durable view of a job's latest state."""

    db = SessionLocal()
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.idempotency_key == key).first()
        if not job:
            return None
        return {
            "idempotency_key": job.idempotency_key,
            "scope": job.scope,
            "status": job.status,
            "progress": job.progress,
            "error": job.error,
            "version": job.version,
            "run_token": job.run_token,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "updated_at": job.updated_at,
        }
    finally:
        db.close()


def finish(
    key: str,
    status: str,
    error: str = "",
    *,
    run_token: str | None = None,
    error_code: str | None = None,
    report: dict[str, Any] | str | None = None,
) -> bool:
    """收敛到终态。终态是吸收态，且只有持有当前 run token 的尝试可以写入。

    ``cancelling`` 状态下取消永远优先：即使协程在被取消后仍跑到了正常结尾，也
    记为 ``cancelled``，避免留下一个「已完成」的幻影所有者与作用域取消打架。
    """

    if status not in TERMINAL_STATUSES:
        raise ValueError(f"invalid terminal job status: {status}")
    token = run_token or _run_token_for_current_task(key)
    if not token:
        return False
    db = SessionLocal()
    try:
        # Serialize validation and completion so a reclaimed row cannot change
        # attempts between the token check and the terminal update.
        db.execute(text("BEGIN IMMEDIATE"))
        job = (
            db.query(BackgroundJob)
            .filter(BackgroundJob.idempotency_key == key, BackgroundJob.run_token == token)
            .first()
        )
        if not job:
            db.rollback()
            return False
        target = status
        if job.status == STATUS_CANCELLING and status in {STATUS_COMPLETED, STATUS_CANCELLED}:
            target = STATUS_CANCELLED
        if not can_transition(job.status, target):
            db.rollback()
            return False
        # 落库的失败说明会经 API 回显到界面：去掉堆栈帧、密钥与本地路径后再截断。
        safe_error = summarize(error, limit=_ERROR_TEXT_CHARS) if error else ""
        job.status = target
        job.error = safe_error
        job.error_code = error_code or error_code_for_status(target, safe_error)
        job.error_message = _short_error(safe_error, target)
        if report is not None:
            job.report = _report_json(report)
        job.current_step = "" if target == STATUS_COMPLETED else job.current_step
        job.message = _terminal_message(target, job.message)
        job.progress = 100 if target == STATUS_COMPLETED else job.progress
        job.finished_at = datetime.utcnow()
        job.updated_at = job.finished_at
        terminal_debug = make_debug_event(
            "lifecycle",
            _terminal_message(target, job.message),
            step=str(job.current_step or ""),
            progress=100 if target == STATUS_COMPLETED else int(job.progress or 0),
            status="success" if target == STATUS_COMPLETED else ("cancelled" if target == STATUS_CANCELLED else "error"),
            detail={"error_code": str(job.error_code or "")} if target != STATUS_COMPLETED else None,
        )
        debug_raw, debug_revision = append_debug_event(
            job.debug_events,
            terminal_debug,
            revision=int(job.debug_revision or 0),
        )
        job.debug_events = debug_raw
        job.debug_revision = debug_revision
        if target != STATUS_COMPLETED:
            _reconcile_abandoned_work(db, job, target)
        usage, estimate = _cost_context(db, key)
        payload = job_dto(job, usage=usage, estimate=estimate)
        job_id = str(job.id)
        db.commit()
        if _claim_tokens.get(key) == token:
            _claim_tokens.pop(key, None)
        _settle(key, target, job_id=job_id)
        publish_debug_event(job, terminal_debug, debug_revision)
        publish_job_event(terminal_event_for(target), payload)
        if target == STATUS_FAILED:
            # 失败归因在后台异步进行（规则 + LLM），完成后再推 job.updated；
            # 惰性 import 避免 task_registry 与分析服务在启动期互相依赖。
            from services.error_analysis_service import schedule_failure_analysis

            schedule_failure_analysis(job_id)
        return True
    finally:
        db.close()


def _settle(key: str, status: str, *, job_id: str = "") -> None:
    """任务进入终态后的收尾：标记用量归属 + 释放预算预留。

    无论任务是成功、失败、取消还是被服务重启中断，都会走到这里，因此
    「任务失败后仍能查询已发生的调用成本」与「并发任务不会重复计费（预留必然释放）」
    两条都能成立。
    """

    try:
        usage_service.finalize_job(key, status, job_id=job_id)
    except Exception:  # noqa: BLE001 - 收尾失败不影响任务状态写入
        logger.warning("用量收尾失败: key=%s", key, exc_info=True)
    try:
        budget_service.release_reservation(key)
    except Exception:  # noqa: BLE001
        logger.warning("预算预留释放失败: key=%s", key, exc_info=True)
    _claim_scopes.pop(key, None)


def _short_error(text: str, status: str) -> str:
    if status == STATUS_COMPLETED:
        return ""
    clean = (text or "").strip()
    if not clean:
        return "任务未正常完成"
    return clean[:_ERROR_MESSAGE_CHARS]


def _terminal_message(status: str, current: str) -> str:
    if status == STATUS_COMPLETED:
        return "任务已完成"
    if status == STATUS_CANCELLED:
        return "任务已取消"
    if status == STATUS_INTERRUPTED:
        return "服务重启中断了该任务"
    return current[:_ERROR_MESSAGE_CHARS] or "任务失败"


def recover_interrupted() -> int:
    """Make work abandoned by a process restart visible and retryable.

    恢复出的任务在任务中心显示为 ``interrupted``（终态），因此不会被误认为仍在
    运行，也可以由用户手动重试或续跑。
    """

    db = SessionLocal()
    try:
        db.execute(text("BEGIN IMMEDIATE"))
        now = datetime.utcnow()
        jobs = db.query(BackgroundJob).filter(BackgroundJob.status.in_(ACTIVE_STATUSES)).all()
        payloads: list[dict[str, Any]] = []
        settled: list[tuple[str, str]] = []
        for job in jobs:
            if not can_transition(job.status, STATUS_INTERRUPTED):
                continue
            job.status = STATUS_INTERRUPTED
            job.error = "server restarted before background job finished"
            job.error_code = error_code_for_status(STATUS_INTERRUPTED)
            job.error_message = "服务重启中断了该任务，可手动重试或续跑"
            job.message = job.error_message
            job.finished_at = now
            job.updated_at = now
            _reconcile_abandoned_work(db, job, STATUS_INTERRUPTED)
            payloads.append(job_dto(job, now=now))
            settled.append((job.idempotency_key, str(job.id)))
        db.commit()
        for key, job_id in settled:
            _settle(key, STATUS_INTERRUPTED, job_id=job_id)
        for payload in payloads:
            publish_job_event(terminal_event_for(STATUS_INTERRUPTED), payload)
        return len(payloads)
    finally:
        db.close()


async def _with_usage_scope(key: str, coroutine: Coroutine[Any, Any, Any]) -> Any:
    """在任务协程内绑定用量归属（项目 / 剧集 / 镜头 / 任务类型）。

    服务层（LLM / 图像 / 视频 / TTS / FFmpeg）在记账时读取这个上下文，因此任何
    生成调用都会被记到正确的项目与镜头下，不需要一路透传参数。
    """

    scope = _claim_scopes.get(key)
    if scope is None:
        return await coroutine
    token = usage_service.bind_scope(scope)
    try:
        return await coroutine
    finally:
        usage_service.reset_scope(token)


def register(key: str, task: asyncio.Task, *, run_token: str | None = None) -> bool:
    """Attach durable completion bookkeeping to a claimed asyncio task."""

    token = run_token or _claim_tokens.get(key)
    if not token:
        return False
    current = _tasks.get(key)
    if current is not None and not current.done():
        return False
    _tasks[key] = task
    _task_tokens[task] = token

    def _cleanup(done: asyncio.Task) -> None:
        if _tasks.get(key) is done:
            _tasks.pop(key, None)
        _task_tokens.pop(done, None)
        if done.cancelled():
            finish(key, "cancelled", "background job was cancelled", run_token=token)
            return
        try:
            error = done.exception()
        except asyncio.CancelledError:
            finish(key, "cancelled", "background job was cancelled", run_token=token)
            return
        if error is None:
            finish(key, "completed", run_token=token)
        else:
            # 任务表里的错误会经 API 回显到界面，先脱敏再落库；完整堆栈写进
            # 服务端日志（已脱敏），保证任务失败一定留下可诊断记录。
            log_failure(error, error_type=ERROR_BACKGROUND_JOB, context={"key": key}, log=logger)
            finish(key, "failed", redact(error, limit=2000), run_token=token)

    task.add_done_callback(_cleanup)
    return True


def start(key: str, coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task:
    """Create and register a previously claimed coroutine without leaking its claim."""

    run_token = _claim_tokens.get(key)
    if not run_token:
        coroutine.close()
        raise RuntimeError("后台任务缺少有效的运行令牌")
    current = _tasks.get(key)
    if current is not None and not current.done():
        coroutine.close()
        raise RuntimeError("后台任务注册失败")
    # 显式保存包装协程：create_task 失败时它和原始协程都还没启动，必须分别关闭，
    # 否则 GC 时会留下 "coroutine was never awaited" 的 RuntimeWarning。
    wrapper = _with_usage_scope(key, coroutine)
    try:
        task = asyncio.create_task(wrapper)
    except BaseException as exc:
        # create_task 抛错说明任务从未被调度，包装协程体（含 await coroutine）
        # 没有执行过，因此这里关闭两个协程各一次，既不重复也不执行原始协程。
        wrapper.close()
        coroutine.close()
        finish(key, "failed", f"background task could not be created: {exc}", run_token=run_token)
        raise
    try:
        if not register(key, task, run_token=run_token):
            task.cancel()
            # A different local task already owns this attempt. Do not mark
            # its durable row failed just because a duplicate caller tried to
            # attach another coroutine.
            raise RuntimeError("后台任务注册失败")
    except BaseException as exc:
        if not task.done():
            task.cancel()
        if _tasks.get(key) is task:
            unregister(key, task)
            finish(key, "failed", f"background task could not be registered: {exc}", run_token=run_token)
        raise
    return task


def cancel(key: str) -> bool:
    """Request cancellation of a single job. 重复调用是幂等的。

    ``queued``/``running`` 先进入 ``cancelling``（任务中心立即可见，作用域仍被
    锁定），协程真正退出后由完成回调动到 ``cancelled``；没有本地协程时直接收敛。
    """

    task = _tasks.get(key)
    if task is not None and not task.done():
        task.cancel()
        # Cancellation callbacks run on the next event-loop turn. Mark the
        # durable row as cancelling now while retaining the claim until the
        # coroutine has actually unwound.
        return _enter_cancelling(key, _task_tokens.get(task))
    return _cancel_without_local_task(key)


def _enter_cancelling(key: str, token: str | None) -> bool:
    """把 durable 行标记为 cancelling 并记录取消请求时间。"""

    db = SessionLocal()
    try:
        db.execute(text("BEGIN IMMEDIATE"))
        query = db.query(BackgroundJob).filter(
            BackgroundJob.idempotency_key == key,
            BackgroundJob.status.in_(ACTIVE_STATUSES),
        )
        if token:
            query = query.filter(BackgroundJob.run_token == token)
        job = query.first()
        if not job:
            db.rollback()
            return False
        now = datetime.utcnow()
        job.cancel_requested_at = now
        job.updated_at = now
        if can_transition(job.status, STATUS_CANCELLING):
            job.status = STATUS_CANCELLING
        job.message = "已请求取消，等待当前步骤安全退出"
        _reconcile_abandoned_work(db, job, STATUS_CANCELLED)
        payload = job_dto(job, now=now)
        db.commit()
        publish_job_event(EVENT_JOB_UPDATED, payload)
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _cancel_without_local_task(key: str) -> bool:
    """没有本地协程可取消时的收敛路径：直接进入 cancelled（幂等）。"""

    db = SessionLocal()
    try:
        db.execute(text("BEGIN IMMEDIATE"))
        job = (
            db.query(BackgroundJob)
            .filter(BackgroundJob.idempotency_key == key, BackgroundJob.status.in_(ACTIVE_STATUSES))
            .first()
        )
        if not job:
            db.rollback()
            return False
        now = datetime.utcnow()
        job.cancel_requested_at = job.cancel_requested_at or now
        job.status = STATUS_CANCELLED
        job.error = job.error or "job was cancelled"
        job.error_code = error_code_for_status(STATUS_CANCELLED)
        job.error_message = "任务已取消"
        job.message = job.error_message
        job.finished_at = now
        job.updated_at = now
        _reconcile_abandoned_work(db, job, STATUS_CANCELLED)
        payload = job_dto(job, now=now)
        cancelled_key = str(job.idempotency_key)
        cancelled_id = str(job.id)
        db.commit()
        _settle(cancelled_key, STATUS_CANCELLED, job_id=cancelled_id)
        publish_job_event(EVENT_JOB_UPDATED, payload)
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def cancel_scopes(
    scopes: set[str] | list[str] | tuple[str, ...],
    reason: str = "scope was cancelled",
    *,
    keep_blocked: bool = False,
) -> int | ScopeCancellation:
    """Block scopes, cancel local work, and wait until it cannot publish again."""

    scope_set = {scope for scope in scopes if scope}
    if not scope_set:
        return 0

    db = SessionLocal()
    try:
        db.execute(text("BEGIN IMMEDIATE"))
        active_jobs = db.query(BackgroundJob).filter(BackgroundJob.status.in_(ACTIVE_STATUSES)).all()
        jobs_by_id = {job.id: job for job in active_jobs if job.scope in scope_set}
        project_ids = {scope.split(":", 1)[1] for scope in scope_set if scope.startswith("project:")}
        shot_ids = {scope.split(":", 1)[1] for scope in scope_set if scope.startswith("shot:")}
        if project_ids:
            project_shot_ids = {
                shot_id
                for (shot_id,) in db.query(Shot.id).filter(Shot.project_id.in_(project_ids)).all()
            }
            for job in active_jobs:
                if job.scope.startswith("shot:") and job.scope.split(":", 1)[1] in project_shot_ids:
                    jobs_by_id[job.id] = job
        if shot_ids:
            shot_project_ids = {
                project_id
                for (project_id,) in db.query(Shot.project_id).filter(Shot.id.in_(shot_ids)).all()
            }
            for job in active_jobs:
                if job.scope.startswith("project:") and job.scope.split(":", 1)[1] in shot_project_ids:
                    jobs_by_id[job.id] = job
        jobs = list(jobs_by_id.values())
        job_count = len(jobs)
        owned_scopes = {job.scope for job in jobs}
        safe_reason = summarize(reason, limit=_ERROR_TEXT_CHARS)
        requested_at = datetime.utcnow()
        for job in jobs:
            job.status = STATUS_CANCELLING
            job.error = safe_reason
            job.message = "已请求取消，等待当前步骤安全退出"
            job.cancel_requested_at = job.cancel_requested_at or requested_at
            job.updated_at = requested_at
            _reconcile_abandoned_work(db, job, STATUS_CANCELLED)
        # Empty scopes also need a durable owner so another process cannot claim
        # them while deletion waits for existing local tasks to unwind.
        blocker_ids: list[str] = []
        scopes_to_block = scope_set if keep_blocked else scope_set - owned_scopes
        for scope in scopes_to_block:
            blocker_id = uuid.uuid4().hex
            blocker_ids.append(blocker_id)
            db.add(
                BackgroundJob(
                    id=blocker_id,
                    # 作用域占位行不是用户任务，任务中心按前缀过滤掉它们。
                    idempotency_key=f"{SCOPE_BLOCK_KEY_PREFIX}{uuid.uuid4().hex}",
                    scope=scope,
                    status=STATUS_CANCELLING,
                    error=safe_reason,
                    run_token=uuid.uuid4().hex,
                    created_at=requested_at,
                    started_at=requested_at,
                    updated_at=requested_at,
                )
            )
        db.commit()
        keys_to_cancel = {job.idempotency_key for job in jobs}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    local_tasks = []
    for key in keys_to_cancel:
        task = _tasks.get(key)
        if task is not None and not task.done():
            task.cancel()
            local_tasks.append(task)
    if local_tasks:
        _, pending = await asyncio.wait(
            local_tasks,
            timeout=max(1, int(settings.BACKGROUND_TASK_CANCEL_TIMEOUT_SECONDS)),
        )
        if pending:
            # Fail closed: ``cancelling`` remains an active durable owner, so a
            # caller cannot delete data or launch replacement work while an old
            # coroutine may still publish.
            raise TimeoutError("等待后台任务取消超时；作用域保持锁定")

    db = SessionLocal()
    try:
        now = datetime.utcnow()
        cancelling = (
            db.query(BackgroundJob)
            .filter(
                BackgroundJob.id.in_([job.id for job in jobs]),
                BackgroundJob.status == "cancelling",
            )
            .all()
        )
        payloads: list[dict[str, Any]] = []
        for job in cancelling:
            job.status = STATUS_CANCELLED
            job.error = safe_reason
            job.error_code = error_code_for_status(STATUS_CANCELLED)
            job.error_message = safe_reason[:_ERROR_MESSAGE_CHARS] or "任务已取消"
            job.message = job.error_message
            job.cancel_requested_at = job.cancel_requested_at or now
            job.finished_at = now
            job.updated_at = now
            payloads.append(job_dto(job, now=now))
        settled = [(job.idempotency_key, str(job.id)) for job in cancelling]
        if blocker_ids and not keep_blocked:
            db.query(BackgroundJob).filter(BackgroundJob.id.in_(blocker_ids)).delete(synchronize_session=False)
        db.commit()
        for key, job_id in settled:
            _settle(key, STATUS_CANCELLED, job_id=job_id)
        for payload in payloads:
            publish_job_event(terminal_event_for(STATUS_CANCELLED), payload)
        if keep_blocked:
            return ScopeCancellation(job_count, tuple(blocker_ids))
        return job_count
    finally:
        db.close()


def release_scope_block(cancellation: ScopeCancellation) -> None:
    if not cancellation.blocker_ids:
        return
    db = SessionLocal()
    try:
        db.query(BackgroundJob).filter(BackgroundJob.id.in_(cancellation.blocker_ids)).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def unregister(key: str, task: asyncio.Task | None = None) -> None:
    if task is None or _tasks.get(key) is task:
        removed = _tasks.pop(key, None)
        if removed is not None:
            _task_tokens.pop(removed, None)


def keys() -> set[str]:
    return {key for key in list(_tasks) if active(key)}


def _run_token_for_current_task(key: str) -> str | None:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    if task is not None:
        # A coroutine that is not the registered owner must never fall back to
        # the latest claim token: doing so would let an old attempt update a
        # reclaimed row. Synchronous legacy callers still use the latest token
        # below for API compatibility.
        return _task_tokens.get(task)
    return _claim_tokens.get(key)


def _reconcile_abandoned_work(db, job: BackgroundJob, terminal_status: str) -> None:
    """Remove business rows from transient states after work stops unexpectedly."""

    parts = job.idempotency_key.split(":")
    if len(parts) < 3:
        return
    owner_type, owner_id, operation = parts[0], parts[1], parts[2]
    cancelled = terminal_status == "cancelled"

    if owner_type == "shot":
        shot = db.query(Shot).filter(Shot.id == owner_id).first()
        if not shot or (job.version and (shot.version or 1) != job.version):
            return
        if operation == "storyboard" and shot.storyboard_status == "queued":
            shot.storyboard_status = "pending" if cancelled else "failed"
            if shot.status == "pending":
                shot.status = "pending" if cancelled else "failed"
        elif operation == "video" and shot.status == "video_generating":
            if cancelled:
                shot.status = "storyboard_approved" if shot.confirmed else "storyboard_done"
            else:
                shot.status = "failed"
        return

    if owner_type != "project":
        return
    project = db.query(Project).filter(Project.id == owner_id).first()
    if operation in {"storyboard", "pipeline"}:
        queued = db.query(Shot).filter(Shot.project_id == owner_id, Shot.storyboard_status == "queued").all()
        for shot in queued:
            shot.storyboard_status = "pending" if cancelled else "failed"
            if shot.status == "pending":
                shot.status = "pending" if cancelled else "failed"
    if operation == "pipeline":
        videos = db.query(Shot).filter(Shot.project_id == owner_id, Shot.status == "video_generating").all()
        for shot in videos:
            shot.status = "storyboard_approved" if cancelled and shot.confirmed else "failed"

    if not project:
        return
    if project.status == "storyboard_generating":
        project.status = "assets_ready" if cancelled else "error"
    elif project.status == "rendering":
        project.status = "error"


def _scope_owner(db, scope: str) -> BackgroundJob | None:
    """Find an active owner, including the project/shot scope hierarchy."""

    active_jobs = db.query(BackgroundJob).filter(BackgroundJob.status.in_(ACTIVE_STATUSES)).all()
    exact = next((job for job in active_jobs if job.scope == scope), None)
    if exact:
        return exact

    owner_type, separator, owner_id = scope.partition(":")
    if not separator:
        return None
    if owner_type == "shot":
        project_id = db.query(Shot.project_id).filter(Shot.id == owner_id).scalar()
        if project_id:
            return next((job for job in active_jobs if job.scope == f"project:{project_id}"), None)
        return None
    if owner_type == "project":
        active_shot_jobs = {
            job.scope.split(":", 1)[1]: job
            for job in active_jobs
            if job.scope.startswith("shot:")
        }
        if not active_shot_jobs:
            return None
        shot_id = (
            db.query(Shot.id)
            .filter(Shot.project_id == owner_id, Shot.id.in_(active_shot_jobs))
            .limit(1)
            .scalar()
        )
        return active_shot_jobs.get(shot_id) if shot_id else None
    return None


def _is_queue_placeholder(job: BackgroundJob | None) -> bool:
    """队列项在拿到真实 run token 前允许自身 claim 穿透作用域预占。"""

    return bool(job is not None and getattr(job, "queue_batch_id", "") and not getattr(job, "run_token", ""))


__all__ = [
    "ACTIVE_STATUSES", "SCOPE_BLOCK_KEY_PREFIX", "ScopeCancellation", "TERMINAL_STATUSES", "active", "cancel",
    "cancel_scopes", "claim", "finish", "keys", "recover_interrupted", "register", "release_scope_block",
    "snapshot", "start", "unregister", "update_progress",
]
