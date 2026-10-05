from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db import get_db
from services import regeneration_queue

router = APIRouter(tags=["regeneration-queue"])


class QueueSubmitRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    shot_ids: list[str] = Field(min_length=1, max_length=100)
    # storyboard_only / storyboard+video 等 UI 文案统一在前端映射到这些稳定值。
    stages: list[str] = Field(min_length=1, max_length=3)
    priority: int = Field(default=0, ge=-10, le=10)
    concurrency: int = Field(default=1, ge=1, le=8)
    order: str = Field(default="shot", pattern="^(shot|sequence|reverse)$")
    reuse_audio: bool = False
    resume_missing: bool = False
    force_confirmed: bool = False
    version_map: dict[str, int] = Field(default_factory=dict)


def _response(submission: regeneration_queue.QueueSubmission) -> dict[str, Any]:
    return {
        "ok": True,
        "batch_id": submission.batch_id,
        "items": submission.items,
        "blocked": submission.blocked,
        "merged": sum(1 for item in submission.items if item.get("deduplicated")),
    }


@router.post("/regeneration-queue")
async def submit_queue(data: QueueSubmitRequest, db: Session = Depends(get_db)):
    submission = regeneration_queue.submit(
        db,
        data.project_id,
        data.shot_ids,
        data.stages,
        priority=data.priority,
        concurrency=data.concurrency,
        order=data.order,
        reuse_audio=data.reuse_audio,
        resume_missing=data.resume_missing,
        force_confirmed=data.force_confirmed,
        version_map=data.version_map,
    )
    return _response(submission)


@router.get("/regeneration-queue/{batch_id}")
async def queue_detail(batch_id: str, db: Session = Depends(get_db)):
    return regeneration_queue.batch_snapshot(db, batch_id)


@router.post("/regeneration-queue/{batch_id}/pause")
async def pause_queue(batch_id: str, db: Session = Depends(get_db)):
    return regeneration_queue.pause(db, batch_id)


@router.post("/regeneration-queue/{batch_id}/resume")
async def resume_queue(batch_id: str, db: Session = Depends(get_db)):
    return regeneration_queue.resume(db, batch_id)


@router.post("/regeneration-queue/{batch_id}/continue")
async def continue_queue(batch_id: str, db: Session = Depends(get_db)):
    return regeneration_queue.resume(db, batch_id)


@router.post("/regeneration-queue/{batch_id}/cancel")
async def cancel_queue(batch_id: str, db: Session = Depends(get_db)):
    return regeneration_queue.cancel(db, batch_id)


@router.post("/regeneration-queue/{batch_id}/retry")
async def retry_queue(batch_id: str, db: Session = Depends(get_db)):
    return regeneration_queue.retry(db, batch_id)


@router.post("/regeneration-queue/{batch_id}/resume-failed")
async def resume_failed_queue(batch_id: str, db: Session = Depends(get_db)):
    return regeneration_queue.retry(db, batch_id)


@router.delete("/regeneration-queue/{batch_id}")
async def delete_queue(batch_id: str, db: Session = Depends(get_db)):
    return regeneration_queue.delete(db, batch_id)
