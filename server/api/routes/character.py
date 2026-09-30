import json

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.schemas import CharacterName, JsonText, OptionalIdentifier, ShortKey, ShotText, VisualNotes
from db import get_db
from models import Character, Project
from services.invalidation_service import invalidate_asset_consumers
from services.reference_readiness_service import refresh_project_reference_state
from services.security import validate_identifier
from services.task_registry import cancel_scopes

router = APIRouter(prefix="/api/character", tags=["character"])


class CharacterUpdate(BaseModel):
    project_id: OptionalIdentifier | None = None
    name: CharacterName | None = None
    appearance: JsonText | None = None
    personality: ShotText | None = None
    visual_prompt: VisualNotes | None = None
    negative_prompt: VisualNotes | None = None
    voice_id: ShortKey | None = None
    emotion_variants: JsonText | None = None
    key_features: JsonText | None = None
    default_outfit: ShotText | None = None
    lora_profile: ShortKey | None = None
    ip_adapter_profile: ShortKey | None = None
    wardrobe_lock: ShotText | None = None
    seed: ShortKey | None = None


@router.get("/{project_id}/characters")
async def get_project_characters(project_id: str, db: Session = Depends(get_db)):
    asset_project_id = _asset_project_id(db, project_id)
    chars = db.query(Character).filter(Character.project_id == asset_project_id).all()
    return [
        {
            "id": c.id,
            "project_id": c.project_id,
            "name": c.name,
            "appearance": json.loads(c.appearance) if c.appearance else {},
            "personality": c.personality,
            "visual_prompt": c.visual_prompt,
            "negative_prompt": c.negative_prompt,
            "voice_id": c.voice_id,
            "emotion_variants": json.loads(c.emotion_variants) if c.emotion_variants else {},
            "key_features": json.loads(c.key_features) if c.key_features else [],
            "default_outfit": c.default_outfit,
            "reference_images": json.loads(c.reference_images) if c.reference_images else [],
            "lora_profile": c.lora_profile or "",
            "ip_adapter_profile": c.ip_adapter_profile or "",
            "wardrobe_lock": c.wardrobe_lock or "",
            "seed": c.seed,
            "reference_status": getattr(c, "reference_status", "stale"),
            "reference_version": int(getattr(c, "reference_version", 1) or 1),
            "reference_retry_count": int(getattr(c, "reference_retry_count", 0) or 0),
            "reference_failure_reason": getattr(c, "reference_failure_reason", ""),
            "reference_error_id": getattr(c, "reference_error_id", ""),
            "reference_skip_reason": getattr(c, "reference_skip_reason", ""),
            "reference_capability_warning": getattr(c, "reference_capability_warning", ""),
            "reference_impact": json.loads(c.reference_impact) if getattr(c, "reference_impact", "") else {},
        }
        for c in chars
    ]


@router.put("/{character_id}")
async def update_character(
    character_id: str,
    data: CharacterUpdate,
    db: Session = Depends(get_db),
    project_id: str | None = Query(default=None),
):
    owner_id = data.project_id or project_id
    if not owner_id:
        raise HTTPException(status_code=400, detail="更新角色必须提供 project_id")
    try:
        validate_identifier(owner_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    owner = db.query(Project).filter(Project.id == owner_id).first()
    if not owner:
        raise HTTPException(status_code=404, detail="项目不存在")
    asset_project_id = owner.parent_project_id or owner.id
    char = db.query(Character).filter(Character.id == character_id, Character.project_id == asset_project_id).first()
    if not char:
        raise HTTPException(status_code=404, detail="角色不存在")
    changed = data.model_dump(exclude_unset=True)
    changed.pop("project_id", None)
    result_id = char.id
    for key, value in changed.items():
        setattr(char, key, value)
    affected_scopes = invalidate_asset_consumers(db, asset_project_id, character_id=character_id) if changed else set()
    if changed:
        char.reference_status = "stale"
        char.reference_failure_reason = "角色参数已修改，参考素材需重新生成"
    db.commit()
    if changed:
        refresh_project_reference_state(db, owner_id)
    if affected_scopes:
        affected_project = db.query(Project).filter(Project.id == owner_id).first()
        if affected_project:
            affected_project.status = "assets_ready"
        db.commit()
    if affected_scopes:
        await cancel_scopes(affected_scopes, "character was edited")
    return {"id": result_id, "status": "updated"}


def _asset_project_id(db: Session, project_id: str) -> str:
    project = db.query(Project).filter(Project.id == project_id).first()
    return (project.parent_project_id or project.id) if project else project_id
