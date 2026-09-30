"""后台任务中心 API。

只暴露前端需要的稳定 DTO：不返回 run_token、不返回内部异常堆栈、不直接序列化
SQLAlchemy 对象。所有写操作都先把服务端状态落库，再返回结果，前端据此刷新或
合并状态，而不是只做乐观更新。

响应形状对成功/失败保持一致（ok / status / message / job / error_code），前端
只需要一套解析逻辑；失败时用真实 HTTP 状态码表达语义（404 / 409）。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from api import schemas
from db import get_db
from models import BackgroundJob
from services import job_actions, job_center

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# 任务 ID 是服务端生成的十六进制串；这里只做长度上限，实际存在性由查询决定，
# 因此不存在的 ID 统一返回 404 而不是 422。
_JOB_ID_MAX_CHARS = 64


@router.get("")
async def list_jobs(
    project_id: Annotated[schemas.OptionalIdentifier, Query()] = "",
    status: Annotated[list[schemas.JobStatus] | None, Query()] = None,
    job_type: Annotated[list[schemas.JobType] | None, Query()] = None,
    scope: Annotated[schemas.JobScope, Query()] = "",
    active_only: Annotated[bool, Query()] = False,
    q: Annotated[schemas.JobSearch | None, Query()] = None,
    page: Annotated[int, Query(ge=1, le=10_000)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    db: Session = Depends(get_db),
):
    """任务列表：项目/状态/类型/作用域筛选、关键词搜索与分页。

    默认按 updated_at 倒序；同时返回任务总数、活动任务数与状态统计。
    """

    query = job_center.JobQuery(
        project_id=project_id,
        statuses=tuple(status or ()),
        job_types=tuple(job_type or ()),
        scope=scope,
        active_only=active_only,
        search=q or "",
        page=page,
        page_size=page_size,
    )
    return job_center.list_jobs(db, query)


@router.get("/stats")
async def job_stats(
    project_id: Annotated[schemas.OptionalIdentifier, Query()] = "",
    db: Session = Depends(get_db),
):
    """任务入口徽标所需的轻量统计：活动数、失败数与最近一次任务。"""

    return job_center.job_stats(db, project_id=project_id)


@router.delete("")
async def cleanup_jobs(
    project_id: Annotated[schemas.OptionalIdentifier, Query()] = "",
    status: Annotated[list[schemas.JobStatus] | None, Query()] = None,
    job_type: Annotated[list[schemas.JobType] | None, Query()] = None,
    scope: Annotated[schemas.JobScope, Query()] = "",
    q: Annotated[schemas.JobSearch | None, Query()] = None,
    db: Session = Depends(get_db),
):
    """清理历史任务：只删除终态行，运行中的任务不会被清理。"""

    query = job_center.JobQuery(
        project_id=project_id,
        statuses=tuple(status or ()),
        job_types=tuple(job_type or ()),
        scope=scope,
        search=q or "",
    )
    deleted = job_center.purge_jobs(db, query)
    return {"deleted": deleted, "stats": job_center.job_stats(db, project_id=project_id)}


def _load_job(db: Session, job_id: str) -> BackgroundJob:
    if not job_id or len(job_id) > _JOB_ID_MAX_CHARS:
        job = None
    else:
        job = job_center.get_job(db, job_id)
    if job is None:
        return None  # type: ignore[return-value]
    return job


def _missing() -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"ok": False, "status": "not_found", "message": "任务不存在或已被清理", "error_code": "job_not_found"},
    )


@router.get("/{job_id}")
async def get_job_detail(job_id: str, db: Session = Depends(get_db)):
    """任务详情：状态、进度、步骤、可读消息、时长、attempt 与可执行动作。"""

    job = _load_job(db, job_id)
    if job is None:
        return _missing()
    return job_center.job_detail(db, job)


@router.get("/{job_id}/debug")
async def get_job_debug_log(job_id: str, db: Session = Depends(get_db)):
    """任务调试日志：进度、当前步骤、API 请求参数与脱敏提示词。"""

    job = _load_job(db, job_id)
    if job is None:
        return _missing()
    return job_center.job_debug_log(db, job)


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str, db: Session = Depends(get_db)):
    """取消任务：queued / running / cancelling 均可处理，重复调用幂等。"""

    job = _load_job(db, job_id)
    if job is None:
        return _missing()
    return _respond(job_actions.cancel_job(db, job))


@router.post("/{job_id}/retry")
async def retry_job(job_id: str, db: Session = Depends(get_db)):
    """完整重试：只有 failed / cancelled / interrupted 可以重试，且幂等。"""

    job = _load_job(db, job_id)
    if job is None:
        return _missing()
    return _respond(await job_actions.retry_job(db, job))


@router.post("/{job_id}/resume")
async def resume_job(job_id: str, db: Session = Depends(get_db)):
    """续跑：从已有数据库状态 / 中间产物继续，无法安全续跑时返回明确错误。"""

    job = _load_job(db, job_id)
    if job is None:
        return _missing()
    return _respond(await job_actions.resume_job(db, job))


@router.delete("/{job_id}")
async def delete_job(job_id: str, db: Session = Depends(get_db)):
    """删除单条任务记录；运行中的任务必须先取消。"""

    job = _load_job(db, job_id)
    if job is None:
        return _missing()
    return _respond(job_actions.delete_job(db, job))


def _respond(outcome: job_actions.ActionOutcome):
    payload = outcome.payload()
    if outcome.ok:
        return payload
    return JSONResponse(status_code=outcome.http_status, content=payload)
