import json
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.schemas import (
    CharacterAssetIdList,
    CharacterName,
    Identifier,
    JsonFieldInput,
    KeyFeatureInput,
    OptionalIdentifier,
    ShortKey,
    ShotText,
    VisualNotes,
)
from db import get_db
from models import Character, Project, SceneAsset, Shot
from services.image_service import ImageService
from services.invalidation_service import invalidate_asset_consumers, mark_shot_media_stale
from services.security import validate_identifier
from services.shot_version_service import create_version
from services.task_registry import cancel_scopes

router = APIRouter(prefix="/api/asset", tags=["asset"])
image_service = ImageService()


class ShotAssetUpdate(BaseModel):
    project_id: Identifier
    scene_asset_id: OptionalIdentifier = ""
    character_asset_ids: CharacterAssetIdList = Field(default_factory=list)


class CharacterAssetUpdate(BaseModel):
    # Required for direct asset updates so an ID from another project cannot be
    # edited accidentally (or by a guessed identifier).
    project_id: OptionalIdentifier | None = None
    name: CharacterName | None = None
    appearance: JsonFieldInput | None = None
    personality: ShotText | None = None
    visual_prompt: VisualNotes | None = None
    negative_prompt: VisualNotes | None = None
    voice_id: ShortKey | None = None
    emotion_variants: JsonFieldInput | None = None
    key_features: KeyFeatureInput | None = None
    default_outfit: ShotText | None = None
    lora_profile: ShortKey | None = None
    ip_adapter_profile: ShortKey | None = None
    wardrobe_lock: ShotText | None = None
    seed: ShortKey | None = None
    regenerate: bool = False


class SceneAssetUpdate(BaseModel):
    project_id: OptionalIdentifier | None = None
    name: CharacterName | None = None
    description: ShotText | None = None
    visual_prompt: VisualNotes | None = None
    negative_prompt: VisualNotes | None = None
    key_features: KeyFeatureInput | None = None
    # 场景分组键与时段允许中文（例如「教室-morning」「清晨」），不做 identifier 校验。
    scene_group_key: ShortKey | None = None
    time_of_day: ShortKey | None = None
    consistency_profile: JsonFieldInput | None = None
    prop_lock: ShotText | None = None
    seed: Annotated[int, Field(ge=0, le=2_147_483_647)] | None = None
    regenerate: bool = False


@router.get("/{project_id}/board")
async def get_asset_board(project_id: str, db: Session = Depends(get_db)):
    asset_project_id = _asset_project_id(db, project_id)
    return {
        "project_id": project_id,
        "asset_project_id": asset_project_id,
        "characters": [_serialize_character(item) for item in db.query(Character).filter(Character.project_id == asset_project_id).all()],
        "scenes": [_serialize_scene(item) for item in db.query(SceneAsset).filter(SceneAsset.project_id == asset_project_id).all()],
    }


@router.put("/shot/{shot_id}")
async def update_shot_assets(shot_id: str, data: ShotAssetUpdate, db: Session = Depends(get_db)):
    try:
        validate_identifier(shot_id, "镜头 ID")
        validate_identifier(data.project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Bind the caller's project context directly into the lookup.  A parent
    # series may own shared assets, but it must not be usable as authority to
    # mutate a child episode's shot.
    shot = db.query(Shot).filter(Shot.id == shot_id, Shot.project_id == data.project_id).first()
    if not shot:
        raise HTTPException(status_code=404, detail="镜头不存在")
    if shot.confirmed:
        raise HTTPException(status_code=423, detail="已审核锁定的镜头不能更换资产")
    asset_project_id = _asset_project_id(db, shot.project_id)

    scene_asset_id = str(data.scene_asset_id or "").strip()
    if scene_asset_id:
        scene = (
            db.query(SceneAsset)
            .filter(SceneAsset.id == scene_asset_id, SceneAsset.project_id == asset_project_id)
            .first()
        )
        if not scene:
            raise HTTPException(status_code=400, detail="场景资产不属于该项目")

    character_ids = list(dict.fromkeys(str(item).strip() for item in (data.character_asset_ids or []) if str(item).strip()))
    if character_ids:
        characters = (
            db.query(Character)
            .filter(Character.id.in_(character_ids), Character.project_id == asset_project_id)
            .all()
        )
        found = {item.id for item in characters}
        if found != set(character_ids):
            raise HTTPException(status_code=400, detail="角色资产不属于该项目")

    changed = shot.scene_asset_id != scene_asset_id or _json_list(shot.character_asset_ids) != character_ids
    shot_project_id = shot.project_id
    result_id = shot.id
    if changed:
        # 资产换绑会使现有素材与绑定不一致：被替换的当前状态先进版本历史。
        create_version(db, shot, "manual_edit")
    shot.scene_asset_id = scene_asset_id
    shot.character_asset_ids = json.dumps(character_ids, ensure_ascii=False)
    if changed:
        # Asset rebinding invalidates every downstream artifact, including audio
        # and the last-frame continuity reference — but only as a stale marker:
        # the old media stays referenced and previewable until regenerated.
        mark_shot_media_stale(shot)
        shot.storyboard_status = "pending"
        shot.version = (shot.version or 1) + 1
        project = db.query(Project).filter(Project.id == shot_project_id).first()
        if project:
            project.status = "assets_ready"
    db.commit()
    if changed:
        await cancel_scopes(
            {f"shot:{shot_id}", f"project:{shot_project_id}"},
            "shot assets were edited",
        )
    return {"id": result_id, "status": "updated"}


@router.put("/character/{character_id}")
async def update_character_asset(
    character_id: str,
    data: CharacterAssetUpdate,
    db: Session = Depends(get_db),
    project_id: str | None = Query(default=None),
):
    owner_id = _required_asset_project_id(db, data.project_id or project_id)
    item = db.query(Character).filter(Character.id == character_id, Character.project_id == owner_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="角色资产不存在")

    changed = data.model_dump(exclude_unset=True)
    changed.pop("project_id", None)
    regenerate = bool(changed.pop("regenerate", False))
    mutation_requested = bool(changed) or regenerate
    _assign_json_field(item, changed, "appearance")
    _assign_json_field(item, changed, "emotion_variants")
    _assign_json_field(item, changed, "key_features")
    for key, value in changed.items():
        setattr(item, key, value)

    mutation_time = datetime.utcnow()
    item.updated_at = mutation_time
    affected_scopes = invalidate_asset_consumers(db, owner_id, character_id=character_id) if mutation_requested else set()
    character_payload = _serialize_character(item)
    generation_project_id = item.project_id
    generation_seed = _int_seed(item.seed, 42)
    style = _project_style(db, generation_project_id) if regenerate else ""
    db.commit()
    db.close()
    if affected_scopes:
        await cancel_scopes(affected_scopes, "character asset was edited")

    if regenerate:
        ref_path = await image_service.generate_character_reference(
            character=character_payload,
            style=style,
            project_id=generation_project_id,
            seed=generation_seed,
        )
        current = db.query(Character).filter(Character.id == character_id, Character.project_id == owner_id).first()
        if not current:
            raise HTTPException(status_code=404, detail="角色资产不存在")
        if current.updated_at == mutation_time:
            current.reference_images = json.dumps([ref_path], ensure_ascii=False)
            current.updated_at = datetime.utcnow()
            db.commit()
        return _serialize_character(current)
    return character_payload


@router.put("/scene/{scene_id}")
async def update_scene_asset(
    scene_id: str,
    data: SceneAssetUpdate,
    db: Session = Depends(get_db),
    project_id: str | None = Query(default=None),
):
    owner_id = _required_asset_project_id(db, data.project_id or project_id)
    item = db.query(SceneAsset).filter(SceneAsset.id == scene_id, SceneAsset.project_id == owner_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="场景资产不存在")

    changed = data.model_dump(exclude_unset=True)
    changed.pop("project_id", None)
    regenerate = bool(changed.pop("regenerate", False))
    mutation_requested = bool(changed) or regenerate
    _assign_json_field(item, changed, "key_features")
    _assign_json_field(item, changed, "consistency_profile")
    for key, value in changed.items():
        setattr(item, key, value)

    mutation_time = datetime.utcnow()
    item.updated_at = mutation_time
    affected_scopes = invalidate_asset_consumers(db, owner_id, scene_id=scene_id) if mutation_requested else set()
    scene_payload = _serialize_scene(item)
    generation_project_id = item.project_id
    generation_seed = int(item.seed or 1200)
    style = _project_style(db, generation_project_id) if regenerate else ""
    db.commit()
    db.close()
    if affected_scopes:
        await cancel_scopes(affected_scopes, "scene asset was edited")

    if regenerate:
        ref_path = await image_service.generate_scene_baseline_reference(
            scene=scene_payload,
            style=style,
            project_id=generation_project_id,
            seed=generation_seed,
        )
        current = db.query(SceneAsset).filter(SceneAsset.id == scene_id, SceneAsset.project_id == owner_id).first()
        if not current:
            raise HTTPException(status_code=404, detail="场景资产不存在")
        if current.updated_at == mutation_time:
            current.baseline_image_path = ref_path
            current.reference_images = json.dumps([ref_path], ensure_ascii=False)
            current.updated_at = datetime.utcnow()
            db.commit()
        return _serialize_scene(current)
    return scene_payload


def _asset_project_id(db: Session, project_id: str) -> str:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project.parent_project_id or project.id


def _required_asset_project_id(db: Session, project_id: str | None) -> str:
    """Resolve and require the project context for direct asset mutations."""

    if not project_id:
        raise HTTPException(status_code=400, detail="更新资产必须提供 project_id")
    try:
        validate_identifier(project_id, "项目 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _asset_project_id(db, project_id)


def _assign_json_field(item, data: dict, key: str) -> None:
    if key not in data:
        return
    value = data.pop(key)
    if isinstance(value, str):
        setattr(item, key, value)
    else:
        setattr(item, key, json.dumps(value, ensure_ascii=False))


def _json_list(raw: str | None) -> list:
    try:
        value = json.loads(raw or "[]")
        return list(value) if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


def _project_style(db: Session, project_id: str) -> str:
    project = db.query(Project).filter(Project.id == project_id).first()
    return (project.style if project else "anime") or "anime"


def _int_seed(value: str | None, fallback: int) -> int:
    try:
        return int(value or fallback)
    except (TypeError, ValueError):
        return fallback


def _serialize_character(item: Character) -> dict:
    return {
        "id": item.id,
        "project_id": item.project_id,
        "type": "character",
        "name": item.name,
        "appearance": json.loads(item.appearance) if item.appearance else {},
        "personality": item.personality,
        "visual_prompt": item.visual_prompt,
        "negative_prompt": item.negative_prompt,
        "voice_id": item.voice_id,
        "key_features": json.loads(item.key_features) if item.key_features else [],
        "reference_images": json.loads(item.reference_images) if item.reference_images else [],
        "default_outfit": item.default_outfit or "",
        "lora_profile": item.lora_profile or "",
        "ip_adapter_profile": item.ip_adapter_profile or "",
        "wardrobe_lock": item.wardrobe_lock or "",
        "seed": item.seed,
        "asset_status": str(item.asset_status or "active"),
    }


def _serialize_scene(item: SceneAsset) -> dict:
    return {
        "id": item.id,
        "project_id": item.project_id,
        "type": "scene",
        "name": item.name,
        "description": item.description,
        "visual_prompt": item.visual_prompt,
        "negative_prompt": item.negative_prompt,
        "key_features": json.loads(item.key_features) if item.key_features else [],
        "reference_images": json.loads(item.reference_images) if item.reference_images else [],
        "scene_group_key": item.scene_group_key or item.id,
        "time_of_day": item.time_of_day or "",
        "baseline_image_path": item.baseline_image_path or "",
        "consistency_profile": json.loads(item.consistency_profile) if item.consistency_profile else {},
        "prop_lock": item.prop_lock or "",
        "seed": item.seed,
        "asset_status": str(item.asset_status or "active"),
    }
