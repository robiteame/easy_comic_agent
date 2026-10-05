"""选择性重生成队列。

队列项就是 ``BackgroundJob``：提交时是 queued，真正执行时回到现有镜头路由，
由 task_registry 取得 run token 和作用域锁。这样批量调度不会绕过旧任务的版本围栏，
也不会因为某个镜头失败而把同批其它镜头写成失败。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from db import SessionLocal
from models import BackgroundJob, Project, Shot, ShotVersion
from services.job_dto import job_dto
from services.job_types import (
    ACTIVE_STATUSES,
    ERROR_CODE_DEPENDENCY_FAILED,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    TERMINAL_STATUSES,
    error_code_for_status,
    parse_job_key,
)
from services.security import existing_file
from services.task_registry import cancel as cancel_task
from services.task_registry import unique_archived_key

STAGE_STORYBOARD = "storyboard"
STAGE_AUDIO = "audio"
STAGE_VIDEO = "video"
STAGES = (STAGE_STORYBOARD, STAGE_AUDIO, STAGE_VIDEO)
STAGE_JOB_TYPES = {STAGE_STORYBOARD: "shot_image", STAGE_AUDIO: "shot_audio", STAGE_VIDEO: "shot_video"}
_MIN_MEDIA = {STAGE_STORYBOARD: 1024, STAGE_AUDIO: 1024, STAGE_VIDEO: 4096}
_batch_tasks: dict[str, asyncio.Task] = {}
_item_tasks: set[asyncio.Task] = set()
_running_item_ids: set[str] = set()


@dataclass(frozen=True)
class QueueSubmission:
    batch_id: str
    items: list[dict[str, Any]]
    blocked: list[dict[str, Any]]


def _json_ids(value: str | None) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def _artifact_ok(shot: Shot, stage: str) -> bool:
    if shot.media_stale:
        # 参数已变更：旧产物虽存在但已过期，续跑不允许跳过重生成。
        return False
    path = {
        STAGE_STORYBOARD: shot.storyboard_path or shot.image_path,
        STAGE_AUDIO: shot.audio_path,
        STAGE_VIDEO: shot.video_path,
    }.get(stage, "")
    if not path:
        return False
    return existing_file(path, minimum_size=_MIN_MEDIA[stage]) is not None


def _canonical_key(shot_id: str, stage: str) -> str:
    return f"shot:{shot_id}:{stage}"


def _active_or_queued(db: Session, key: str) -> BackgroundJob | None:
    return (
        db.query(BackgroundJob)
        .filter(BackgroundJob.idempotency_key == key, BackgroundJob.status.in_((*ACTIVE_STATUSES,)))
        .order_by(BackgroundJob.created_at.desc())
        .first()
    )


def _archive_terminal_key(db: Session, key: str) -> None:
    """释放规范幂等键，同时保留旧尝试记录。"""
    old = db.query(BackgroundJob).filter(BackgroundJob.idempotency_key == key).first()
    if old is None:
        return
    if old.status in ACTIVE_STATUSES:
        return
    old.idempotency_key = unique_archived_key(db, key, int(old.attempt or 1), exclude_id=old.id)
    db.flush()


def submit(
    db: Session,
    project_id: str,
    shot_ids: list[str],
    stages: list[str],
    *,
    priority: int = 0,
    concurrency: int = 1,
    order: str = "shot",
    reuse_audio: bool = False,
    resume_missing: bool = False,
    force_confirmed: bool = False,
    version_map: dict[str, int] | None = None,
) -> QueueSubmission:
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    requested_stages = list(dict.fromkeys(str(stage).strip().lower() for stage in stages))
    if not requested_stages or any(stage not in STAGES for stage in requested_stages):
        raise HTTPException(status_code=422, detail="阶段必须是 storyboard、audio 或 video")
    # 阶段依赖是固定的：故事板 -> 配音 -> 视频。用户的执行顺序只影响镜头间
    # 排队，不允许把视频放到故事板之前造成同一镜头的作用域竞争。
    normalized_stages = [stage for stage in STAGES if stage in requested_stages]
    if not shot_ids or len(shot_ids) > 100:
        raise HTTPException(status_code=422, detail="至少选择一个镜头，最多 100 个")
    concurrency = max(1, min(8, int(concurrency)))
    priority = max(-10, min(10, int(priority)))
    shots = db.query(Shot).filter(Shot.project_id == project_id, Shot.id.in_(shot_ids)).all()
    by_id = {shot.id: shot for shot in shots}
    missing = [shot_id for shot_id in shot_ids if shot_id not in by_id]
    if missing:
        raise HTTPException(status_code=404, detail=f"镜头不存在: {', '.join(missing)}")
    for version_shot_id, version_number in (version_map or {}).items():
        if version_shot_id not in by_id or int(version_number or 0) < 1:
            raise HTTPException(status_code=422, detail="版本重生成必须引用已选择镜头的正整数版本")
        version_exists = (
            db.query(ShotVersion.id)
            .filter(ShotVersion.shot_id == version_shot_id, ShotVersion.number == int(version_number))
            .first()
        )
        if version_exists is None:
            raise HTTPException(status_code=404, detail=f"镜头 {version_shot_id} 的版本 v{version_number} 不存在")

    batch_id = uuid.uuid4().hex
    requested = list(shot_ids)
    if order == "sequence":
        requested.sort(key=lambda shot_id: by_id[shot_id].sequence)
    elif order == "reverse":
        requested.reverse()

    created: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    deduplicated_batch_ids: list[str] = []
    has_new_items = False
    order_index = 0
    for shot_id in requested:
        shot = by_id[shot_id]
        for stage in normalized_stages:
            key = _canonical_key(shot_id, stage)
            existing = _active_or_queued(db, key)
            if existing is not None:
                created.append({"id": existing.id, "shot_id": shot_id, "stage": stage, "deduplicated": True})
                if existing.queue_batch_id:
                    deduplicated_batch_ids.append(existing.queue_batch_id)
                continue
            if shot.confirmed and not force_confirmed:
                blocked.append({"shot_id": shot_id, "stage": stage, "reason": "镜头已确认锁定，请显式解锁或强制确认"})
                continue
            if resume_missing and _artifact_ok(shot, stage):
                blocked.append({"shot_id": shot_id, "stage": stage, "reason": "产物已存在且有效，续跑已跳过"})
                continue
            previous_ids = [item["id"] for item in created if item.get("shot_id") == shot_id]
            _archive_terminal_key(db, key)
            job = BackgroundJob(
                id=uuid.uuid4().hex,
                idempotency_key=key,
                scope=f"shot:{shot_id}",
                status="queued",
                progress=0,
                version=int(shot.version or 1),
                run_token="",
                project_id=project_id,
                job_type=STAGE_JOB_TYPES[stage],
                display_name=f"镜头 {shot.sequence} · {stage}",
                current_step=stage,
                message="等待队列调度",
                attempt=1,
                queue_batch_id=batch_id,
                queue_position=len(created),
                queue_priority=priority,
                queue_order=order_index,
                queue_stage=stage,
                queue_shot_id=shot_id,
                queue_dependency_ids=json.dumps(previous_ids, ensure_ascii=False),
                queue_concurrency=concurrency,
                queue_resume_missing=resume_missing,
                queue_reuse_audio=reuse_audio,
                queue_force_confirmed=force_confirmed,
                queue_requested_version=int((version_map or {}).get(shot_id) or 0),
            )
            db.add(job)
            db.flush()
            created.append({"id": job.id, "shot_id": shot_id, "stage": stage, "deduplicated": False})
            has_new_items = True
            order_index += 1
    # 纯幂等合并不创建新批次，否则客户端拿到的 batch_id 无法被 GET 到。
    if not has_new_items and deduplicated_batch_ids:
        batch_id = deduplicated_batch_ids[0]
    db.commit()
    if has_new_items and batch_id not in _batch_tasks:
        task = asyncio.create_task(_run_batch(batch_id))
        _batch_tasks[batch_id] = task
        task.add_done_callback(lambda done, bid=batch_id: _batch_tasks.pop(bid, None))
    return QueueSubmission(batch_id=batch_id, items=created, blocked=blocked)


def _batch_jobs(db: Session, batch_id: str) -> list[BackgroundJob]:
    return (
        db.query(BackgroundJob)
        .filter(BackgroundJob.queue_batch_id == batch_id)
        .order_by(BackgroundJob.queue_priority.desc(), BackgroundJob.queue_order.asc(), BackgroundJob.created_at.asc())
        .all()
    )


def _is_archived_attempt(job: BackgroundJob) -> bool:
    """判断任务行是否已被 task_registry 归档为历史 attempt。"""

    key = str(job.idempotency_key or "")
    return key.endswith(f"#attempt-{int(job.attempt or 1)}")


def _visible_jobs(jobs: list[BackgroundJob]) -> list[BackgroundJob]:
    """按规范幂等键去重，只返回每个队列项当前可操作的任务行。"""

    visible_by_key: dict[str, BackgroundJob] = {}
    for job in jobs:
        canonical = parse_job_key(str(job.idempotency_key or "")).canonical
        current = visible_by_key.get(canonical)
        if current is None:
            visible_by_key[canonical] = job
            continue
        current_archived = _is_archived_attempt(current)
        job_archived = _is_archived_attempt(job)
        if current_archived and not job_archived:
            visible_by_key[canonical] = job
        elif current_archived == job_archived and job.created_at > current.created_at:
            visible_by_key[canonical] = job
    return list(visible_by_key.values())


async def _run_batch(batch_id: str) -> None:
    while True:
        db = SessionLocal()
        try:
            jobs = _batch_jobs(db, batch_id)
            current_jobs = _visible_jobs(jobs)
            queued = [job for job in current_jobs if job.status == "queued"]
            active = [job for job in current_jobs if job.status in ACTIVE_STATUSES and job.status != "queued"]
            if not queued and not active:
                return
            paused = any(
                bool(job.queue_paused) for job in current_jobs if job.status in {"queued", "running", "cancelling"}
            )
            if paused:
                await asyncio.sleep(0.2)
                continue
            limit = max(1, int(next((job.queue_concurrency for job in current_jobs if job.queue_concurrency), 1)))
            running_count = sum(1 for job in current_jobs if job.status in {"running", "cancelling"})
            capacity = max(0, limit - running_count)
            selected: list[BackgroundJob] = []
            for job in queued:
                if job.id in _running_item_ids:
                    continue
                dependencies = _json_ids(job.queue_dependency_ids)
                dep_rows = {item.id: item for item in jobs}
                missing_dependency_ids = [dep_id for dep_id in dependencies if dep_id not in dep_rows]
                if missing_dependency_ids:
                    dependency_rows = db.query(BackgroundJob).filter(BackgroundJob.id.in_(missing_dependency_ids)).all()
                    dep_rows.update({item.id: item for item in dependency_rows})
                if any(
                    dep_rows.get(dep_id) and dep_rows[dep_id].status in {STATUS_FAILED, STATUS_CANCELLED}
                    for dep_id in dependencies
                ):
                    job.status = STATUS_FAILED
                    job.error_code = ERROR_CODE_DEPENDENCY_FAILED
                    job.error_message = "前置阶段失败，当前镜头未执行"
                    job.queue_blocked_reason = "前置阶段失败"
                    job.finished_at = datetime.utcnow()
                    job.updated_at = datetime.utcnow()
                    continue
                if any(
                    dep_rows.get(dep_id) and dep_rows[dep_id].status not in {STATUS_COMPLETED}
                    for dep_id in dependencies
                ):
                    job.queue_blocked_reason = "等待前置阶段完成"
                    continue
                if capacity <= 0:
                    break
                selected.append(job)
                capacity -= 1
            db.commit()
            selected_ids = [job.id for job in selected]
        finally:
            db.close()
        if selected_ids:
            db = SessionLocal()
            try:
                for job_id in selected_ids:
                    persisted = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
                    if persisted is not None:
                        persisted.status = "running"
                        persisted.message = "正在调度"
                        persisted.updated_at = datetime.utcnow()
                db.commit()
            finally:
                db.close()
        for job_id in selected_ids:
            _running_item_ids.add(job_id)
            task = asyncio.create_task(_run_item(job_id))
            _item_tasks.add(task)
            task.add_done_callback(_item_tasks.discard)
        await asyncio.sleep(0.15)


async def _run_item(queue_job_id: str) -> None:
    db = SessionLocal()
    try:
        queued = db.query(BackgroundJob).filter(BackgroundJob.id == queue_job_id).first()
        if queued is None or queued.status not in {"queued", "running"}:
            return
        shot_id, stage, batch_id = queued.queue_shot_id, queued.queue_stage, queued.queue_batch_id
        force = bool(queued.queue_force_confirmed)
        reuse_audio = bool(queued.queue_reuse_audio)
        requested_version = int(queued.queue_requested_version or 0)
        dependency_ids = _json_ids(queued.queue_dependency_ids)
        key = _canonical_key(shot_id, stage)
    finally:
        db.close()
    try:
        from api.routes import shot as shot_route

        run_db = SessionLocal()
        try:
            shot = run_db.query(Shot).filter(Shot.id == shot_id).first()
            if shot is None:
                raise RuntimeError("镜头已删除")
            # 历史快照只在该镜头批次的第一个阶段恢复一次；后续音频/视频阶段
            # 依赖前置阶段生成的新状态，不能再次把故事板覆盖回旧版本。
            if requested_version and not dependency_ids:
                from services.shot_version_service import apply_snapshot_to_shot, create_version, parse_snapshot

                version_row = (
                    run_db.query(ShotVersion)
                    .filter(ShotVersion.shot_id == shot_id, ShotVersion.number == requested_version)
                    .order_by(ShotVersion.created_at.desc())
                    .first()
                )
                if version_row is None:
                    raise RuntimeError(f"镜头版本 v{requested_version} 不存在")
                create_version(run_db, shot, "regenerate", task_id=key)
                snapshot = parse_snapshot(version_row)
                apply_snapshot_to_shot(shot, snapshot)
                # 选择性重生成面向新结构化对白：历史纯文本快照恢复时迁移到单条结构。
                if isinstance(snapshot.get("dialogue"), str):
                    from services.shot_dialogue import parse_shot_dialogue, serialize_dialogue_lines

                    shot.dialogue = serialize_dialogue_lines(parse_shot_dialogue(snapshot.get("dialogue")))
                shot.version = int(shot.version or 1) + 1
                run_db.commit()
            if bool(queued.queue_resume_missing) and _artifact_ok(shot, stage):
                queued = run_db.query(BackgroundJob).filter(BackgroundJob.id == queue_job_id).first()
                queued.status, queued.progress, queued.finished_at = STATUS_COMPLETED, 100, datetime.utcnow()
                queued.message = "有效产物已存在，未重复生成"
                run_db.commit()
                return
            if stage == STAGE_STORYBOARD:
                result = await shot_route.regenerate_shot(
                    shot_id,
                    shot_route.RegenerateRequest(reason="selective_queue", force_confirmed=force),
                    run_db,
                )
            elif stage == STAGE_VIDEO:
                result = await shot_route.generate_shot_video(
                    shot_id,
                    shot_route.ShotVideoGenerateRequest(force=True, reuse_audio=reuse_audio, allow_unconfirmed=True),
                    run_db,
                )
            else:
                result = await shot_route.generate_shot_audio(
                    shot_id,
                    shot_route.ShotAudioGenerateRequest(force=True, reuse_existing=reuse_audio),
                    run_db,
                )
        finally:
            run_db.close()
        if isinstance(result, dict) and result.get("deduplicated"):
            # 路由层对「作用域忙」和「同阶段已有真实任务」都返回去重。
            # 若规范键仍指向本队列的无 token 占位行，说明项目级/其它镜头任务
            # 抢占了作用域；不能把占位行当成真实 attempt 等待，否则批次会永久卡住。
            active_db = SessionLocal()
            try:
                active = (
                    active_db.query(BackgroundJob)
                    .filter(BackgroundJob.idempotency_key == key)
                    .order_by(BackgroundJob.created_at.desc())
                    .first()
                )
                blocked_by_scope = active is None or active.id == queue_job_id or not active.run_token
            finally:
                active_db.close()
            if blocked_by_scope:
                _mark_queue_waiting(queue_job_id, "等待项目或镜头作用域释放")
                return
        if isinstance(result, dict) and result.get("skipped"):
            _mark_queue_job(queue_job_id, STATUS_COMPLETED, "阶段无需生成或已复用有效配音")
            return
        active_id = None
        check = SessionLocal()
        try:
            active = (
                check.query(BackgroundJob)
                .filter(BackgroundJob.idempotency_key == key)
                .order_by(BackgroundJob.created_at.desc())
                .first()
            )
            if active is not None:
                active_id = active.id
                active.queue_batch_id = batch_id
                check.commit()
        finally:
            check.close()
        if active_id is None:
            _mark_queue_job(queue_job_id, STATUS_COMPLETED, "阶段已完成")
            return
        await _wait_for_job(active_id)
        final = SessionLocal()
        try:
            running = final.query(BackgroundJob).filter(BackgroundJob.id == active_id).first()
            status = running.status if running else STATUS_FAILED
            message = running.error_message or running.message if running else "任务未找到"
            error_code = str(running.error_code or "") if running else ""
        finally:
            final.close()
        _mark_queue_job(queue_job_id, status, message, error_code=error_code)
    except asyncio.CancelledError:
        _mark_queue_job(queue_job_id, STATUS_CANCELLED, "队列任务已取消")
        raise
    except Exception as exc:  # 只标记当前镜头，不影响同批其它项
        _mark_queue_job(queue_job_id, STATUS_FAILED, str(exc)[:240])
    finally:
        _running_item_ids.discard(queue_job_id)


async def _wait_for_job(job_id: str) -> None:
    for _ in range(2400):
        db = SessionLocal()
        try:
            job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
            if job is None or job.status in TERMINAL_STATUSES:
                return
        finally:
            db.close()
        await asyncio.sleep(0.25)


def _mark_queue_job(job_id: str, status: str, message: str, *, error_code: str = "") -> None:
    from services.error_analysis_service import schedule_failure_analysis

    db = SessionLocal()
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
        if job is None:
            return
        if job.status in TERMINAL_STATUSES:
            return
        job.status = status
        job.progress = 100 if status == STATUS_COMPLETED else job.progress
        job.message = message[:240]
        job.error_message = "" if status == STATUS_COMPLETED else message[:240]
        # 队列占位行此前不写 error_code；真实任务的错误码优先，缺省按状态/文本推导。
        job.error_code = error_code or error_code_for_status(status, message)[:64]
        job.finished_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()
    if status == STATUS_FAILED:
        schedule_failure_analysis(job_id)


def _mark_queue_waiting(job_id: str, reason: str) -> None:
    """把被项目级/其它镜头作用域挡住的占位项放回队列。"""

    db = SessionLocal()
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
        if job is None or job.status in TERMINAL_STATUSES:
            return
        job.status = "queued"
        job.queue_blocked_reason = "scope_busy"
        job.message = reason[:240]
        job.updated_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


def batch_snapshot(db: Session, batch_id: str) -> dict[str, Any]:
    jobs = _batch_jobs(db, batch_id)
    if not jobs:
        raise HTTPException(status_code=404, detail="队列批次不存在")
    # task_registry 会保留「queued 占位 -> 真实 attempt」两行。批次视图只展示
    # 当前规范键，历史 attempt 仍可从任务详情的 attempt history 查看。
    visible_jobs = _visible_jobs(jobs)
    visible_jobs.sort(key=lambda job: (-int(job.queue_priority or 0), int(job.queue_order or 0), job.created_at))
    return {
        "batch_id": batch_id,
        "paused": any(bool(job.queue_paused) for job in visible_jobs),
        "items": [job_dto(job) for job in visible_jobs],
        "summary": {
            "total": len(visible_jobs),
            "queued": sum(job.status == "queued" for job in visible_jobs),
            "running": sum(job.status in {"running", "cancelling"} for job in visible_jobs),
            "completed": sum(job.status == STATUS_COMPLETED for job in visible_jobs),
            "failed": sum(job.status == STATUS_FAILED for job in visible_jobs),
            "cancelled": sum(job.status == STATUS_CANCELLED for job in visible_jobs),
        },
    }


def pause(db: Session, batch_id: str) -> dict[str, Any]:
    jobs = _batch_jobs(db, batch_id)
    if not jobs:
        raise HTTPException(status_code=404, detail="队列批次不存在")
    for job in _visible_jobs(jobs):
        if job.status in ACTIVE_STATUSES:
            job.queue_paused = True
    db.commit()
    return batch_snapshot(db, batch_id)


def resume(db: Session, batch_id: str) -> dict[str, Any]:
    jobs = _batch_jobs(db, batch_id)
    if not jobs:
        raise HTTPException(status_code=404, detail="队列批次不存在")
    for job in _visible_jobs(jobs):
        job.queue_paused = False
    db.commit()
    if batch_id not in _batch_tasks:
        task = asyncio.create_task(_run_batch(batch_id))
        _batch_tasks[batch_id] = task
        task.add_done_callback(lambda done, bid=batch_id: _batch_tasks.pop(bid, None))
    return batch_snapshot(db, batch_id)


def cancel(db: Session, batch_id: str) -> dict[str, Any]:
    jobs = _batch_jobs(db, batch_id)
    if not jobs:
        raise HTTPException(status_code=404, detail="队列批次不存在")
    active_keys: list[str] = []
    for job in _visible_jobs(jobs):
        if job.status == "queued":
            job.status = STATUS_CANCELLED
            job.finished_at = datetime.utcnow()
            job.message = "队列批次已取消"
        elif job.status in ACTIVE_STATUSES:
            active_keys.append(job.idempotency_key)
    db.commit()
    for key in active_keys:
        cancel_task(key)
    return batch_snapshot(db, batch_id)


def retry(db: Session, batch_id: str) -> dict[str, Any]:
    jobs = _batch_jobs(db, batch_id)
    if not jobs:
        raise HTTPException(status_code=404, detail="队列批次不存在")
    failed = [job for job in _visible_jobs(jobs) if job.status in {STATUS_FAILED, STATUS_CANCELLED, STATUS_INTERRUPTED}]
    if not failed:
        return batch_snapshot(db, batch_id)
    shot_ids = list(dict.fromkeys(job.queue_shot_id for job in failed if job.queue_shot_id))
    stages = list(dict.fromkeys(job.queue_stage for job in failed if job.queue_stage))
    priority = max((int(job.queue_priority or 0) for job in failed), default=0)
    concurrency = max((int(job.queue_concurrency or 1) for job in failed), default=1)
    version_map = {
        job.queue_shot_id: int(job.queue_requested_version)
        for job in failed
        if job.queue_shot_id and int(job.queue_requested_version or 0) > 0
    }
    options = submit(
        db,
        str(failed[0].project_id),
        shot_ids,
        stages,
        priority=priority,
        concurrency=concurrency,
        order="shot",
        reuse_audio=any(bool(job.queue_reuse_audio) for job in failed),
        resume_missing=any(bool(job.queue_resume_missing) for job in failed),
        force_confirmed=any(bool(job.queue_force_confirmed) for job in failed),
        version_map=version_map,
    )
    return batch_snapshot(db, options.batch_id)


def delete(db: Session, batch_id: str) -> dict[str, Any]:
    jobs = _batch_jobs(db, batch_id)
    if not jobs:
        raise HTTPException(status_code=404, detail="队列批次不存在")
    if any(job.status in ACTIVE_STATUSES for job in _visible_jobs(jobs)):
        raise HTTPException(status_code=409, detail="运行中的队列必须先取消")
    deleted = len(jobs)
    for job in jobs:
        db.delete(job)
    db.commit()
    return {"batch_id": batch_id, "deleted": deleted}


__all__ = [
    "STAGES",
    "STAGE_STORYBOARD",
    "STAGE_AUDIO",
    "STAGE_VIDEO",
    "QueueSubmission",
    "submit",
    "batch_snapshot",
    "pause",
    "resume",
    "cancel",
    "retry",
    "delete",
]
