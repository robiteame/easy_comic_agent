"""内置示例项目：零外部 API 创建一个完整可浏览的项目。

数据来源是随代码分发的固定剧本 ``server/samples/sample_project.json``：
分镜 / 角色 / 场景直接落库，所有图像（角色三视图、场景基准图、故事板）
通过 PIL 占位图适配器在本地生成——即使配置了真实图像 Provider 也绝不
外呼，保证「无 Key 全程零报错」的承诺成立。

持久化复用剧本解析流程的同一套写库辅助（``api.routes.script`` 的
``_upsert_characters`` / ``_upsert_scenes`` / ``_shot_model``），示例项目
与真实项目在数据形状上完全一致，可正常浏览、编辑剧本、删除。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from pathlib import Path

from config import settings
from services.consistency_service import ConsistencyService
from services.providers.base import ImageRequest
from services.providers.endpoint import EndpointConfig
from services.providers.image_placeholder import PlaceholderImageAdapter
from services.security import atomic_write_bytes, safe_path, validate_identifier
from services.shot_version_service import create_version
from services.storage_service import StorageService

logger = logging.getLogger(__name__)

SAMPLE_CONTENT_PATH = Path(__file__).resolve().parent.parent / "samples" / "sample_project.json"

SAMPLE_SERIES_TITLE_PREFIX = "示例项目"

# 占位图尺寸：保持与真实产物相近的观感，同时把 PIL 渲染控制在毫秒级。
_CHARACTER_SHEET_SIZE = "1024x1024"
_SCENE_BASELINE_SIZE = "1024x1024"
_STORYBOARD_SIZE_BY_FORMAT = {
    "9:16": "720x1280",
    "3:4": "864x1152",
    "1:1": "1024x1024",
    "4:3": "1152x864",
    "16:9": "1280x720",
}


def load_sample_content() -> dict:
    """读取随代码分发的示例剧本内容。"""

    data = json.loads(SAMPLE_CONTENT_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("示例剧本内容格式非法")
    return data


async def _render_placeholder(label: str, prompt: str, size: str) -> bytes:
    """用 PIL 占位图适配器本地渲染一张图，不经过任何网络调用。"""

    adapter = PlaceholderImageAdapter(EndpointConfig(protocol="placeholder"))
    return await adapter.generate(ImageRequest(prompt=prompt, label=label, size=size))


def _write_project_image(storage: StorageService, project_id: str, image_path: Path, data: bytes) -> None:
    storage.ensure_project_capacity(project_id, len(data), replacing=image_path)
    atomic_write_bytes(image_path, data, minimum_size=1024)


async def _generate_character_sheet(storage: StorageService, project_id: str, character: dict, style: str) -> str:
    safe_project_id = validate_identifier(project_id, "项目 ID")
    ref_dir = safe_path(settings.OUTPUT_DIR / "projects", safe_project_id, "characters", create_parent=True)
    identity_key = character.get("id") or character.get("name", "character")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(identity_key)).strip("_") or "character"
    character_dir = ref_dir / safe_name
    character_dir.mkdir(parents=True, exist_ok=True)
    image_path = character_dir / "three_view_original.png"
    data = await _render_placeholder(
        "SAMPLE CHARACTER",
        f"{character.get('name', '')}, {character.get('visual_prompt', '')}, style: {style}",
        _CHARACTER_SHEET_SIZE,
    )
    _write_project_image(storage, project_id, image_path, data)
    return str(image_path)


async def _generate_scene_baseline(storage: StorageService, project_id: str, scene: dict, style: str) -> str:
    safe_project_id = validate_identifier(project_id, "项目 ID")
    ref_dir = safe_path(settings.OUTPUT_DIR / "projects", safe_project_id, "scenes", create_parent=True)
    identity_key = scene.get("id") or scene.get("name", "scene")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(identity_key)).strip("_") or "scene"
    scene_dir = ref_dir / safe_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    image_path = scene_dir / "baseline_original.png"
    data = await _render_placeholder(
        "SAMPLE SCENE",
        f"{scene.get('name', '')}, {scene.get('visual_prompt', '')}, style: {style}",
        _SCENE_BASELINE_SIZE,
    )
    _write_project_image(storage, project_id, image_path, data)
    return str(image_path)


async def _generate_storyboard(
    storage: StorageService, project_id: str, shot: dict, sequence: int, style: str, output_format: str
) -> str:
    safe_project_id = validate_identifier(project_id, "项目 ID")
    shot_dir = safe_path(settings.OUTPUT_DIR / "projects", safe_project_id, "shots", create_parent=True)
    shot_id = shot.get("shot_id") or f"{safe_project_id}_shot_{sequence:04d}"
    image_path = shot_dir / f"{shot_id}_v1.png"
    prompt = ", ".join(
        part for part in [style, shot.get("scene_description", ""), shot.get("character_action", "")] if part
    )
    data = await _render_placeholder(
        f"SAMPLE SHOT {sequence:02d}", prompt, _STORYBOARD_SIZE_BY_FORMAT.get(output_format, "720x1280")
    )
    _write_project_image(storage, project_id, image_path, data)
    return str(image_path)


async def create_sample_project(db) -> dict:
    """创建示例项目（series + 第一集），返回与 POST /api/project 同形的序列化结果。"""

    # 延迟导入：与 script 路由共享同一套持久化辅助，避免模块级循环依赖。
    from api.routes.project import _serialize_project
    from api.routes.script import _shot_model, _upsert_characters, _upsert_scenes
    from models import Project
    from services.reference_readiness_service import refresh_project_reference_state

    content = load_sample_content()
    style = str(content.get("style") or "anime")
    output_format = str(content.get("output_format") or "9:16")
    storage = StorageService()
    consistency_service = ConsistencyService()

    series_id = str(uuid.uuid4())
    episode_id = str(uuid.uuid4())
    series = Project(
        id=series_id,
        title=content.get("title") or f"{SAMPLE_SERIES_TITLE_PREFIX} · 未命名",
        project_type="series",
        episode_number=0,
        genre=content.get("genre") or "",
        style=style,
        input_text=content.get("script_text") or "",
        input_type="text",
        output_format=output_format,
        resolution=content.get("resolution") or "1080p",
        platform=content.get("platform") or "douyin",
        target_duration=int(content.get("target_duration") or 30),
        consistency_config=json.dumps(consistency_service.project_config(), ensure_ascii=False),
        status="storyboard_ready",
        is_sample=True,
    )
    episode = Project(
        id=episode_id,
        title=content.get("episode_title") or "示例剧集",
        parent_project_id=series_id,
        project_type="episode",
        episode_number=1,
        genre=series.genre,
        style=style,
        input_text="",
        input_type="text",
        output_format=output_format,
        resolution=series.resolution,
        platform=series.platform,
        target_duration=series.target_duration,
        consistency_config=series.consistency_config,
        status="storyboard_ready",
        is_sample=True,
    )
    db.add(series)
    db.add(episode)

    # 角色三视图与场景基准图：挂到 series（资产归属规则与剧本解析一致）。
    characters = [dict(item) for item in content.get("characters", [])]
    for index, character in enumerate(characters):
        character.setdefault("id", f"{series_id}_char_{index + 1:04d}")
        character["reference_images"] = [await _generate_character_sheet(storage, series_id, character, style)]
        character["reference_status"] = "ready"
        character["reference_version"] = 1
    scenes = [dict(item) for item in content.get("script_scenes", []) or content.get("scenes", [])]
    for index, scene in enumerate(scenes):
        scene.setdefault("id", f"{series_id}_scene_{index + 1:04d}")
        baseline = await _generate_scene_baseline(storage, series_id, scene, style)
        scene["baseline_image_path"] = baseline
        scene["reference_images"] = [baseline]
        scene["reference_status"] = "ready"
        scene["reference_version"] = 1

    character_ids = _upsert_characters(db, series_id, characters, style)
    scene_ids = _upsert_scenes(db, series_id, scenes, style)

    fingerprint = hashlib.sha256(style.encode()).hexdigest()[:16]
    shots = [dict(item) for item in content.get("shots", [])]
    for index, shot in enumerate(shots):
        # 镜头 ID 必须随项目唯一：剧本里的短 shot_id 只用于阅读，落库前统一
        # 改写为与真实解析流程一致的 {episode}_shot_NNNN，否则第二次创建示例
        # 项目会撞主键。
        shot["shot_id"] = f"{episode_id}_shot_{index + 1:04d}"
        storyboard_path = await _generate_storyboard(storage, episode_id, shot, index + 1, style, output_format)
        model = _shot_model(
            episode_id,
            shot,
            index + 1,
            character_ids,
            scene_ids,
            scenes,
            characters,
        )
        # 故事板已用占位图生成：直接落在「故事板完成」状态，打开即可浏览。
        model.image_path = storyboard_path
        model.storyboard_path = storyboard_path
        model.storyboard_status = "done"
        model.status = "storyboard_done"
        model.style_fingerprint = fingerprint
        db.add(model)
        create_version(db, model, "import")

    db.commit()
    db.refresh(series)
    db.refresh(episode)
    consistency_report = refresh_project_reference_state(db, episode_id)

    parent_titles = {series_id: series.title}
    result = _serialize_project(series, parent_titles)
    result["first_episode"] = _serialize_project(episode, parent_titles)
    result["sample"] = True
    result["consistency_report"] = consistency_report
    logger.info("示例项目已创建: series=%s episode=%s shots=%s", series_id, episode_id, len(shots))
    return result
