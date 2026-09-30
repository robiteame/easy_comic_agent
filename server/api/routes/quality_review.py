"""质量审核（quality review）查询与手动重审接口。

只读历史 + 能力状态查询 + 单镜头手动重审。自动模式的审核由 LangGraph
质量门禁节点驱动，不在这里重复实现；本路由服务于界面的评分/问题/证据/
历史候选展示，以及人工对某条素材的即时复检。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db
from models import Project, Shot
from services.quality_review_service import (
    STAGE_STORYBOARD,
    STAGE_VIDEO,
    quality_review_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/quality-review", tags=["quality-review"])


class RerunRequest(BaseModel):
    stage: str = STAGE_STORYBOARD  # storyboard | video


@router.get("/capability")
async def review_capability():
    """审核能力与门禁配置：VLM / 身份 embedding 是否配置、阈值与降级策略。

    界面用它如实展示「哪些维度可检测、哪些未配置」，未配置的能力绝不
    显示为已通过。
    """
    return quality_review_service.capability_summary()


@router.get("/project/{project_id}")
async def project_reviews(project_id: str, db: Session = Depends(get_db)):
    if not db.query(Project).filter(Project.id == project_id).first():
        raise HTTPException(status_code=404, detail="Project not found")
    rows = quality_review_service.rows_for_project(project_id)
    return {
        "project_id": project_id,
        "reviews": rows,
        "gate": {
            "storyboard": quality_review_service.storyboard_gate_status(project_id),
            "video": quality_review_service.video_gate_status(project_id),
        },
    }


@router.get("/shot/{shot_id}")
async def shot_reviews(shot_id: str, db: Session = Depends(get_db)):
    """单镜头的审核历史（新在前），含每轮评分、问题、证据与修正建议。"""
    if not db.query(Shot).filter(Shot.id == shot_id).first():
        raise HTTPException(status_code=404, detail="Shot not found")
    rows = quality_review_service.rows_for_shot(shot_id)
    return {"shot_id": shot_id, "reviews": rows}


@router.post("/shot/{shot_id}/rerun")
async def rerun_shot_review(shot_id: str, data: RerunRequest, db: Session = Depends(get_db)):
    """人工触发一次即时复检（不改变门禁状态，只追加一条审核记录）。"""
    shot = db.query(Shot).filter(Shot.id == shot_id).first()
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    stage = (data.stage or STAGE_STORYBOARD).strip().lower()
    if stage == STAGE_VIDEO:
        if not (shot.video_path or ""):
            raise HTTPException(status_code=400, detail="该镜头尚无视频产物，无法审核")
        review = await quality_review_service.review_video_shot(shot_id)
    elif stage == STAGE_STORYBOARD:
        if not (shot.storyboard_path or shot.image_path):
            raise HTTPException(status_code=400, detail="该镜头尚无故事板素材，无法审核")
        review = await quality_review_service.review_storyboard_shot(shot_id)
    else:
        raise HTTPException(status_code=400, detail="stage 仅支持 storyboard / video")
    return {"shot_id": shot_id, "review": review.to_dict()}
