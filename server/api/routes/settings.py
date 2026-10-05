import re
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, StringConstraints

from api.schemas import StyleKeywords, StyleLabel
from config import settings
from services.atomic_json import read_json_file
from services.model_config_service import get_model_config, save_model_config
from services.model_discovery_service import ModelDiscoveryError, discover_models
from services.prompts import validate_system_prompt
from services.providers.capability_matrix import capability_report
from services.providers.endpoint import KNOWN_PROTOCOLS, endpoint_identity, get_endpoint
from services.skill_config_service import list_skill_templates, save_skill_template, set_skill_bindings
from services.style_templates import create_custom_style_template, style_options

router = APIRouter(prefix="/api/settings", tags=["settings"])


class StyleTemplateCreate(BaseModel):
    label: StyleLabel
    keywords: StyleKeywords
    negative_prompt: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)] = ""


class SkillTemplateSave(BaseModel):
    id: str | None = None
    name: str
    script_agent: dict
    storyboard_agent: dict


class SkillBindingsSave(BaseModel):
    global_default_template_id: str | None = None
    project_bindings: dict[str, str] | None = None
    episode_bindings: dict[str, str] | None = None


class ModelConfigSave(BaseModel):
    categories: dict[str, dict] | None = None
    script: dict | None = None
    image: dict | None = None
    video: dict | None = None
    voice: dict | None = None


class ModelDiscoveryRequest(BaseModel):
    category: str
    base_url: str
    api_key: str = ""
    protocol: str = ""
    auth_style: str = "bearer"


@router.get("/style-templates")
async def get_style_templates():
    return {"templates": style_options()}


@router.post("/style-templates")
async def create_style_template(data: StyleTemplateCreate):
    label = data.label.strip()
    keywords = data.keywords.strip()
    if not label or not keywords:
        raise HTTPException(status_code=400, detail="模板名称和画风关键词不能为空")

    key = _template_key(label)
    template = create_custom_style_template(
        key=key,
        label=label,
        keywords=keywords,
        negative_prompt=data.negative_prompt.strip(),
        created_at=datetime.utcnow().isoformat(),
    )
    return template


@router.get("/skill-configs")
async def get_skill_configs():
    return list_skill_templates()


@router.post("/skill-configs")
async def save_skill_config(data: SkillTemplateSave):
    # system_prompt 的类型/长度/控制字符校验在这里显式拒绝（400），
    # 不让非法值落盘、也不以裸异常冒泡。空字符串合法 = 使用默认提示词。
    for agent in ("script_agent", "storyboard_agent"):
        config = getattr(data, agent)
        if isinstance(config, dict) and "system_prompt" in config:
            try:
                validate_system_prompt(config.get("system_prompt"), field=f"{agent}.system_prompt")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
    return save_skill_template(data.model_dump(exclude_none=True))


@router.put("/skill-configs/bindings")
async def update_skill_bindings(data: SkillBindingsSave):
    try:
        return set_skill_bindings(data.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/provider-capabilities")
async def get_provider_capabilities():
    """完整 Provider Capability Matrix；不含密钥，不把提示词策略当模型能力。"""

    providers: dict[str, list[dict]] = {}
    for capability in ("image", "video"):
        providers[capability] = [
            capability_report(capability, protocol) for protocol in KNOWN_PROTOCOLS.get(capability, ())
        ]
    return {"matrix_version": 1, "providers": providers}


@router.get("/model-configs")
async def get_model_configs():
    return get_model_config()


@router.put("/model-configs")
async def update_model_configs(data: ModelConfigSave):
    try:
        return save_model_config(data.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/model-configs/discover")
async def discover_model_configs(data: ModelDiscoveryRequest):
    try:
        payload = data.model_dump()
        # GET /model-configs intentionally masks secrets. Reuse the stored key
        # only when the caller is still targeting the same endpoint, so a URL
        # change can never accidentally send credentials to a new host.
        if not str(payload.get("api_key") or "").strip() or payload["api_key"].strip() == "********":
            configured = get_endpoint(payload["category"])
            if endpoint_identity(configured.base_url) == endpoint_identity(payload["base_url"]):
                payload["api_key"] = configured.api_key
        return await discover_models(**payload)
    except ModelDiscoveryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _template_key(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip().lower()).strip("_")
    if not slug:
        slug = "custom"
    path = settings.DATA_DIR / "custom_style_templates.json"
    data = read_json_file(path, default=None)
    existing = data if isinstance(data, dict) else {}
    key = f"custom_{slug}"
    if key not in existing:
        return key
    index = 2
    while f"{key}_{index}" in existing:
        index += 1
    return f"{key}_{index}"
