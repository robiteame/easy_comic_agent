from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from typing import Any

from config import settings
from services.atomic_json import atomic_write_json, path_lock, read_json_file


DEFAULT_AGENT_CONFIG: dict[str, Any] = {
    # Style belongs to the project/request. A Skill only overrides it when the
    # user explicitly enables the override switch.
    "style_template_id": "",
    "style_override_enabled": False,
    "custom_style_keywords": "",
    "filter_tts_instruction_text": True,
    "camera_composition": "medium shot, vertical 9:16, clear subject staging, readable foreground and background layers",
    "force_character_scene_references": True,
    "prompt_auto_assembly": True,
    "openpose_lock_enabled": False,
    "style_reference_weight": 0.45,
    "action_reference_weight": 0.30,
    "continuity_enabled": True,
}


DEFAULT_TEMPLATE: dict[str, Any] = {
    "id": "default",
    "name": "默认 Skill 方案",
    "script_agent": copy.deepcopy(DEFAULT_AGENT_CONFIG),
    "storyboard_agent": copy.deepcopy(DEFAULT_AGENT_CONFIG),
    "created_at": "",
    "updated_at": "",
}


def list_skill_templates() -> dict[str, Any]:
    store = _load_store()
    return {
        "templates": list(store["templates"].values()),
        "global_default_template_id": store["global_default_template_id"],
        "project_bindings": store["project_bindings"],
        "episode_bindings": store["episode_bindings"],
    }


def save_skill_template(template: dict[str, Any]) -> dict[str, Any]:
    # 同一路径的锁覆盖「读 -> 改 -> 写」整段，避免并发保存互相覆盖。
    with path_lock(_store_path()):
        store = _load_store()
        template_id = _template_id(template.get("id") or template.get("name") or "skill")
        existing = store["templates"].get(template_id, {})
        now = datetime.utcnow().isoformat()
        normalized = _normalize_template(
            {
                **existing,
                **template,
                "id": template_id,
                "created_at": existing.get("created_at") or now,
                "updated_at": now,
            }
        )
        store["templates"][template_id] = normalized
        if not store.get("global_default_template_id"):
            store["global_default_template_id"] = template_id
        _save_store(store)
    return normalized


def set_skill_bindings(data: dict[str, Any]) -> dict[str, Any]:
    with path_lock(_store_path()):
        store = _load_store()
        templates = store["templates"]

        global_default = data.get("global_default_template_id")
        if global_default:
            _ensure_template(templates, global_default)
            store["global_default_template_id"] = global_default

        for key, target in (("project_bindings", "project_bindings"), ("episode_bindings", "episode_bindings")):
            if key not in data:
                continue
            next_bindings = {}
            for project_id, template_id in (data.get(key) or {}).items():
                if not project_id or not template_id:
                    continue
                _ensure_template(templates, template_id)
                next_bindings[str(project_id)] = str(template_id)
            store[target] = next_bindings

        _save_store(store)
    return list_skill_templates()


def resolve_skill_config(project_id: str = "", db=None) -> dict[str, Any]:
    store = _load_store()
    template_id = ""
    scope = "global"
    project = None

    if project_id and db is not None:
        from models import Project

        project = db.query(Project).filter(Project.id == project_id).first()

    if project_id and project_id in store["episode_bindings"]:
        template_id = store["episode_bindings"][project_id]
        scope = "episode"
    if not template_id and project is not None:
        project_key = project.parent_project_id if project.project_type == "episode" else project.id
        if project_key in store["project_bindings"]:
            template_id = store["project_bindings"][project_key]
            scope = "project"
    if not template_id and project_id in store["project_bindings"]:
        template_id = store["project_bindings"][project_id]
        scope = "project"
    if not template_id:
        template_id = store["global_default_template_id"]

    template = copy.deepcopy(store["templates"].get(template_id) or store["templates"]["default"])
    template["binding_scope"] = scope
    template["resolved_template_id"] = template.get("id", template_id)
    return template


def resolve_effective_style(
    requested_style: str | None,
    skill_config: dict[str, Any] | None = None,
    agent: str = "storyboard_agent",
    *,
    fallback: str = "anime",
) -> dict[str, str]:
    """Resolve one style for every generation stage.

    Legacy skill files often contain ``style_template_id=anime``. That value
    is treated as inherit unless ``style_override_enabled`` is explicitly true,
    preventing historical defaults from silently changing a realistic request.
    """
    requested = str(requested_style or fallback or "anime").strip() or fallback
    config = _agent_config(skill_config, agent)
    override = bool(config.get("style_override_enabled"))
    skill_style = str(config.get("style_template_id") or "").strip()
    effective = skill_style if override and skill_style else requested
    source = "skill_override" if override and skill_style else "project_request"
    return {
        "requested_style": requested,
        "effective_style": effective,
        "style_source": source,
    }


def agent_style_id(skill_config: dict[str, Any] | None, agent: str, fallback: str = "anime") -> str:
    return resolve_effective_style(fallback, skill_config, agent, fallback=fallback)["effective_style"]


def agent_prompt_append(skill_config: dict[str, Any] | None, agent: str) -> str:
    config = _agent_config(skill_config, agent)
    if not config.get("prompt_auto_assembly", True):
        return ""
    parts = [
        config.get("camera_composition", ""),
        config.get("custom_style_keywords", ""),
    ]
    if config.get("force_character_scene_references", True):
        parts.append("strictly use bound character assets and scene baseline references when available")
    if config.get("continuity_enabled", True):
        parts.append("continue from the previous shot frame for pose, eye-line, axis and lighting continuity")
    return ", ".join(str(part).strip() for part in parts if str(part or "").strip())


def apply_agent_config_to_shot(shot_data: dict[str, Any], skill_config: dict[str, Any] | None, agent: str = "storyboard_agent") -> None:
    config = _agent_config(skill_config, agent)
    metadata = resolve_effective_style(shot_data.get("style") or shot_data.get("effective_style"), skill_config, agent)
    shot_data.update(metadata)
    shot_data["style"] = metadata["effective_style"]
    weights = shot_data.get("reference_weights") if isinstance(shot_data.get("reference_weights"), dict) else {}
    weights["environment"] = float(config.get("style_reference_weight") or 0.45)
    weights["style"] = float(config.get("style_reference_weight") or 0.45)
    weights["action"] = float(config.get("action_reference_weight") or 0.30)
    shot_data["reference_weights"] = weights
    shot_data["skill_prompt_append"] = agent_prompt_append(skill_config, agent)
    shot_data["skill_config_snapshot"] = copy.deepcopy(config)
    if not config.get("continuity_enabled", True):
        shot_data["continuity_reference_path"] = ""
    if not config.get("force_character_scene_references", True):
        shot_data["scene_reference_images"] = []
        shot_data["character_reference_images"] = []
        shot_data["reference_assets"] = []


def should_materialize_openpose(skill_config: dict[str, Any] | None, agent: str = "storyboard_agent") -> bool:
    # No real pose estimator/control model is integrated. Historical switches
    # must not reactivate PIL edge images as fake OpenPose controls.
    return False


def clean_tts_text(text: str, skill_config: dict[str, Any] | None) -> str:
    if not _agent_config(skill_config, "storyboard_agent").get("filter_tts_instruction_text", True):
        return text
    cleaned = re.sub(r"[\(（\[].*?(?:语气|口吻|旁白|动作|镜头|指令|停顿).*?[\)）\]]", "", text or "")
    cleaned = re.sub(r"^(台词|旁白|对白)\s*[:：]", "", cleaned.strip())
    return cleaned.strip()


def _agent_config(skill_config: dict[str, Any] | None, agent: str) -> dict[str, Any]:
    if not skill_config:
        return copy.deepcopy(DEFAULT_AGENT_CONFIG)
    return _normalize_agent_config(skill_config.get(agent) or {})


def _load_store() -> dict[str, Any]:
    path = _store_path()
    with path_lock(path):
        data = read_json_file(path, default=None)
        if not isinstance(data, dict):
            data = {}
        store = _normalize_store(data)
        # 文件不存在或刚被判定损坏（已备份为 .corrupt）时落一份默认配置。
        if not path.exists():
            _save_store(store)
        return store


def _save_store(store: dict[str, Any]) -> None:
    path = _store_path()
    with path_lock(path):
        atomic_write_json(path, _normalize_store(store))


def _normalize_store(data: dict[str, Any]) -> dict[str, Any]:
    now = datetime.utcnow().isoformat()
    default = copy.deepcopy(DEFAULT_TEMPLATE)
    default["created_at"] = data.get("templates", {}).get("default", {}).get("created_at") or now
    default["updated_at"] = data.get("templates", {}).get("default", {}).get("updated_at") or now
    templates = {"default": default}
    for key, template in (data.get("templates") or {}).items():
        templates[str(key)] = _normalize_template({**template, "id": str(key)})
    global_default = data.get("global_default_template_id") or "default"
    if global_default not in templates:
        global_default = "default"
    return {
        "templates": templates,
        "global_default_template_id": global_default,
        "project_bindings": dict(data.get("project_bindings") or {}),
        "episode_bindings": dict(data.get("episode_bindings") or {}),
    }


def _normalize_template(template: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(template.get("id") or "default"),
        "name": str(template.get("name") or "未命名 Skill 方案"),
        "script_agent": _normalize_agent_config(template.get("script_agent") or {}),
        "storyboard_agent": _normalize_agent_config(template.get("storyboard_agent") or {}),
        "created_at": str(template.get("created_at") or ""),
        "updated_at": str(template.get("updated_at") or ""),
    }


def _normalize_agent_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = {**DEFAULT_AGENT_CONFIG, **(config or {})}
    # Files written by older releases used anime as an implicit default. Keep
    # that file readable, but do not interpret it as an opt-in override.
    if "style_override_enabled" not in config:
        normalized["style_override_enabled"] = False
    if not normalized.get("style_override_enabled") and normalized.get("style_template_id") == "anime":
        normalized["style_template_id"] = ""
    normalized["openpose_lock_enabled"] = False
    normalized["style_reference_weight"] = _clamp_float(normalized.get("style_reference_weight"), 0, 1, 0.45)
    normalized["action_reference_weight"] = _clamp_float(normalized.get("action_reference_weight"), 0, 1, 0.30)
    for key in (
        "filter_tts_instruction_text",
        "force_character_scene_references",
        "prompt_auto_assembly",
        "openpose_lock_enabled",
        "continuity_enabled",
    ):
        normalized[key] = bool(normalized.get(key))
    return normalized


def _template_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value).strip().lower()).strip("_") or "skill"
    return slug if slug.startswith("skill_") or slug == "default" else f"skill_{slug}"


def _ensure_template(templates: dict[str, Any], template_id: str) -> None:
    if template_id not in templates:
        raise ValueError(f"Skill template not found: {template_id}")


def _clamp_float(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    try:
        number = float(value)
    except Exception:
        return fallback
    return max(minimum, min(maximum, number))


def _store_path():
    return settings.DATA_DIR / "skill_config_templates.json"
