"""成本、用量与预算 API。

对外只暴露稳定 DTO，且遵守三条安全约束：

- 不返回 API Key、端点凭据或供应商原始响应：用量记录里只有归一化后的数量与金额；
- 金额一律是「货币最小单位的整数倍」（micro），未知成本返回 cost_micro=null 且
  cost_known=false，前端据此显示「成本未知」，而不是 0；
- 估算（cost_estimates）与实际用量（usage_records）分接口返回，前端可并排对比，
  但服务端绝不会把两者写进同一张表。

错误码：/`budget_exceeded`（硬预算超限，任务无法启动）由 task_registry.claim_job
返回，见 services/budget_service.CODE_BUDGET_EXCEEDED。
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db import get_db
from models import BackgroundJob, Project, Shot
from services import budget_service, pricing_service, usage_service
from services.job_dto import job_duration_seconds
from services.job_types import JOB_TYPE_UNKNOWN, JOB_TYPES, parse_job_key
from services.security import validate_identifier

router = APIRouter(prefix="/api/budget", tags=["budget"])

_MAX_ITEMS = 200


class PricingItem(BaseModel):
    capability: str
    provider: str = ""
    model: str = ""
    # 金额一律整数（最小货币单位的倍数，本项目管理到 10^-6 个货币单位）。
    # 这里用 int | float 接收，是为了让浮点金额落到业务校验里报 400 + 中文原因，
    # 而不是被 Pydantic 直接拦成 422（用户看不出哪里错了）。
    unit_price_micro: int | float | None = None
    unit_price_secondary_micro: int | float | None = None
    resolution_multipliers: dict[str, int] = Field(default_factory=dict)
    configured: bool | None = None
    currency: str | None = None
    note: str = ""


class PricingSave(BaseModel):
    currency: str | None = None
    items: list[PricingItem] = Field(default_factory=list)


class BudgetSave(BaseModel):
    scope_type: str = "project"
    scope_id: str = ""
    currency: str | None = None
    # 同上：金额只接受整数最小货币单位，浮点由业务校验明确拒绝。
    soft_cost_micro: int | float | None = None
    hard_cost_micro: int | float | None = None
    soft_seconds: int | float | None = None
    hard_seconds: int | float | None = None
    enabled: bool = True
    note: str = ""


class EstimateRequest(BaseModel):
    job_type: str
    project_id: str = ""
    shot_id: str = ""
    shot_ids: list[str] = Field(default_factory=list)


def _project_or_404(db: Session, project_id: str) -> Project:
    try:
        safe_id = validate_identifier(project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    project = db.query(Project).filter(Project.id == safe_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


def _validated_job_type(value: str) -> str:
    job_type = str(value or "").strip()
    if job_type not in JOB_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"未知任务类型: {job_type or '<empty>'}，可选值: {', '.join((*JOB_TYPES, JOB_TYPE_UNKNOWN))}",
        )
    return job_type


# ---------------------------------------------------------------------------
# 模型价格配置（系统设置）
# ---------------------------------------------------------------------------


@router.get("/pricing")
async def get_pricing(db: Session = Depends(get_db)):
    """价目表：每类能力的计价单位、单位换算与已配置单价。"""

    return pricing_service.list_pricing(db)


@router.put("/pricing")
async def update_pricing(payload: PricingSave, db: Session = Depends(get_db)):
    """保存价目表。金额必须是整数（拒绝浮点），未配置价格的能力保持「成本未知」。"""

    items = [item.model_dump() for item in payload.items][:_MAX_ITEMS]
    try:
        return pricing_service.save_pricing(db, {"currency": payload.currency, "items": items})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# 预算配置与状态
# ---------------------------------------------------------------------------


@router.get("/config")
async def get_budget_config(
    project_id: Annotated[str, Query(max_length=64)] = "",
    db: Session = Depends(get_db),
):
    """预算配置：全局 + 该项目（含父系列）的行与生效值。"""

    if project_id:
        _project_or_404(db, project_id)
    return budget_service.get_budget(db, project_id=project_id)


@router.put("/config")
async def update_budget_config(payload: BudgetSave, db: Session = Depends(get_db)):
    """保存预算配置（项目级或全局）。软预算只提示，硬预算会阻止任务启动。"""

    try:
        return budget_service.save_budget(db, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/status")
async def budget_status(
    project_id: Annotated[str, Query(max_length=64)] = "",
    db: Session = Depends(get_db),
):
    """项目预算状态（已用 / 已预留 / 限额），供提交前提示与页面徽标使用。"""

    if not project_id:
        raise HTTPException(status_code=400, detail="必须提供 project_id")
    _project_or_404(db, project_id)
    return budget_service.budget_state(db, project_id=project_id)


# ---------------------------------------------------------------------------
# 估算（提交前确认）
# ---------------------------------------------------------------------------


@router.post("/estimate")
async def estimate_task(payload: EstimateRequest, db: Session = Depends(get_db)):
    """按项目当前状态估算一次任务的成本与耗时（只读，不占额度）。

    这是「任务提交前显示估算确认」的数据源；真正的硬预算拦截发生在任务抢占时
    （task_registry.claim_job），因此这里的结论只是提示。
    """

    job_type = _validated_job_type(payload.job_type)
    project_id = str(payload.project_id or "")
    shot_id = str(payload.shot_id or "")
    if project_id:
        _project_or_404(db, project_id)
    if shot_id:
        shot = db.query(Shot).filter(Shot.id == shot_id).first()
        if shot is None:
            raise HTTPException(status_code=404, detail="镜头不存在")
    estimate = budget_service.estimate_job(
        db,
        job_type=job_type,
        project_id=project_id,
        shot_id=shot_id,
        shot_ids=[str(item) for item in payload.shot_ids][:200] or None,
    )
    state = budget_service.budget_state(
        db,
        project_id=project_id,
        estimate_cost_micro=estimate.get("estimated_cost_micro"),
        estimate_seconds=estimate.get("estimated_seconds"),
        cost_known=bool(estimate.get("cost_known")),
    )
    return {
        "job_type": job_type,
        "project_id": project_id,
        "shot_id": shot_id,
        "estimate": estimate,
        "budget": state,
        "blocked": state.get("level") == budget_service.LEVEL_HARD_EXCEEDED,
        "warning": state.get("message") if state.get("level") == budget_service.LEVEL_SOFT_EXCEEDED else "",
    }


# ---------------------------------------------------------------------------
# 统计与明细
# ---------------------------------------------------------------------------


@router.get("/summary")
async def cost_summary(
    project_id: Annotated[str, Query(max_length=64)] = "",
    series_id: Annotated[str, Query(max_length=64)] = "",
    db: Session = Depends(get_db),
):
    """项目 / 剧集页所需的预算、已用成本、预计成本与预计耗时。"""

    if not project_id and not series_id:
        raise HTTPException(status_code=400, detail="必须提供 project_id 或 series_id")
    if project_id:
        _project_or_404(db, project_id)
    elif db.query(Project.id).filter(Project.id == series_id).first() is None:
        raise HTTPException(status_code=404, detail="项目不存在")
    summary = budget_service.project_summary(db, project_id=project_id, series_id=series_id)
    summary["episodes"] = budget_service.series_episode_breakdown(db, summary.get("series_id") or series_id)
    return summary


@router.get("/usage")
async def usage_detail(
    project_id: Annotated[str, Query(max_length=64)] = "",
    series_id: Annotated[str, Query(max_length=64)] = "",
    shot_id: Annotated[str, Query(max_length=64)] = "",
    job_id: Annotated[str, Query(max_length=64)] = "",
    job_key: Annotated[str, Query(max_length=200)] = "",
    capability: Annotated[str, Query(max_length=32)] = "",
    job_type: Annotated[str, Query(max_length=32)] = "",
    group_by: Annotated[list[str] | None, Query()] = None,
    page: Annotated[int, Query(ge=1, le=10_000)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 20,
    db: Session = Depends(get_db),
):
    """实际用量明细 + 多维度统计（项目 / 剧集 / 镜头 / 任务类型 / 能力）。"""

    filters: dict[str, Any] = {
        "project_id": project_id,
        "series_id": series_id,
        "shot_id": shot_id,
        "job_id": job_id,
        "job_key": job_key,
        "capability": capability,
        "job_type": job_type,
    }
    summary = usage_service.summarize(db, **filters)
    dimensions = [item for item in (group_by or list(usage_service.GROUP_BY_DIMENSIONS[:4])) if item]
    groups: dict[str, list[dict[str, Any]]] = {}
    for dimension in dimensions:
        try:
            groups[dimension] = usage_service.group_usage(db, dimension, **filters)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "filters": {key: value for key, value in filters.items() if value},
        "summary": summary,
        "groups": groups,
        "records": usage_service.list_records(db, page=page, page_size=page_size, **filters),
        "data_sources": {
            "actual": "usage_records（实际用量，按调用逐条落库）",
            "estimate": "cost_estimates（启动前估算，与实际上分开存储）",
        },
    }


@router.get("/jobs/{job_id}")
async def job_cost_detail(job_id: str, db: Session = Depends(get_db)):
    """单个任务的成本明细：实际用量记录 + 启动前估算。

    失败 / 已取消的任务同样可查——它们的调用成本已经落库，这正是「任务失败后仍能
    查询已发生的调用成本」的查询入口。
    """

    try:
        safe_id = validate_identifier(job_id, "任务 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job = db.query(BackgroundJob).filter(BackgroundJob.id == safe_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    summary = usage_service.summarize(db, job_id=safe_id)
    records = usage_service.list_records(db, page=1, page_size=500, job_id=safe_id)
    if not records["items"]:
        # 兜底：早期或人工触发的调用可能没有写回 job_id，用幂等键再查一次。
        canonical = parse_job_key(str(job.idempotency_key)).canonical
        summary = usage_service.summarize(db, job_key=canonical)
        records = usage_service.list_records(db, page=1, page_size=500, job_key=canonical)
    canonical = parse_job_key(str(job.idempotency_key)).canonical
    estimates = usage_service.job_estimate_map(db, [canonical])
    return {
        "job_id": safe_id,
        "job_key": canonical,
        "job_type": str(job.job_type or ""),
        "status": str(job.status or ""),
        "duration_seconds": job_duration_seconds(job),
        "summary": summary,
        # 估算与实际分别返回：cost_estimates 与 usage_records 是两张表。
        "estimate": estimates.get(canonical),
        "records": records,
    }


__all__ = ["router"]
