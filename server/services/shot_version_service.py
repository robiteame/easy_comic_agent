"""镜头版本历史：不可变快照的捕获、对比与恢复。

版本记录只追加（表级触发器兜底禁止 UPDATE），每条记录描述「镜头当时的完整
状态」以及「为什么被记录」（manual_edit / regenerate / restore / import）。

写入约定：
- 任何破坏性变更（编辑、重新生成、资产换绑、恢复）先对被替换的当前状态补一条
  快照，再做变更；
- 后台生成任务成功写回媒体时，把生成结果也记为一条快照（task_id 记任务幂等键），
  这样 A/B 对比能看到「重新生成前 / 重新生成后」两个版本；
- 追加按内容哈希去重：当前状态与最新记录完全一致时跳过，避免重复噪音；恢复
  例外（``force=True``），恢复必须留下新版本记录。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Iterable

from sqlalchemy.orm import Session

from config import settings
from models import Project, SceneAsset, Character, Shot, ShotVersion
from services.security import existing_file
from services.shot_dialogue import dialogue_lines_payload, parse_shot_dialogue, serialize_dialogue_lines

VERSION_SOURCES = ("manual_edit", "regenerate", "restore", "import")

# 快照直接取值的镜头标量字段。confirmed 也会被记录，但恢复时永远写回 False。
_SIMPLE_FIELDS: tuple[str, ...] = (
    "shot_type",
    "scene_description",
    "character_action",
    "camera_angle",
    "camera_movement",
    "duration",
    "estimated_speech_ms",
    "emotion",
    "transition",
    "visual_notes",
    "image_path",
    "storyboard_path",
    "video_path",
    "audio_path",
    "last_frame_path",
    "status",
    "storyboard_status",
    "scene_asset_id",
    "scene_group_id",
    "consistency_context",
    "continuity_reference_path",
    "pose_reference_path",
    "depth_reference_path",
    "confirmed",
)
_JSON_LIST_FIELDS: tuple[str, ...] = ("characters_in_scene", "character_asset_ids")
_JSON_DICT_FIELDS: tuple[str, ...] = ("reference_weights", "continuity_profile")

# 参与媒体有效性校验与存储保护的路径字段。
MEDIA_FIELDS: tuple[str, ...] = ("image_path", "storyboard_path", "audio_path", "video_path", "last_frame_path")

# A/B 对比时的字段展示顺序（快照里其余字段按字母序排在后面）。
_DIFF_FIELD_ORDER: tuple[str, ...] = (
    "visual_notes",
    "prompt",
    "negative_prompt",
    "scene_description",
    "character_action",
    "dialogue",
    "shot_type",
    "camera_angle",
    "camera_movement",
    "duration",
    "estimated_speech_ms",
    "emotion",
    "transition",
    "scene_asset_id",
    "character_asset_ids",
    "image_path",
    "storyboard_path",
    "video_path",
    "audio_path",
    "last_frame_path",
    "status",
    "storyboard_status",
    "consistency_context",
    "reference_weights",
    "continuity_profile",
    "continuity_reference_path",
    "pose_reference_path",
    "depth_reference_path",
    "scene_group_id",
    "characters_in_scene",
    "version",
)

_ALLOWED_MEDIA_ROOTS = (settings.OUTPUT_DIR, settings.ASSETS_DIR, settings.DATA_DIR)


def _json_list(raw: Any) -> list:
    try:
        value = json.loads(raw or "[]")
        return value if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


def _json_dict(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def _resolve_negative_prompt(db: Session, shot: Shot, override: str | None) -> str:
    """快照里的 negative prompt：调用方已知时直接用，否则按项目画风模板解析。"""

    if override is not None:
        return str(override)
    try:
        from services.skill_config_service import agent_style_id, resolve_skill_config
        from services.style_templates import style_prompt_params

        project = db.query(Project).filter(Project.id == shot.project_id).first()
        style = agent_style_id(
            resolve_skill_config(shot.project_id, db),
            "storyboard_agent",
            (project.style if project else "anime") or "anime",
        )
        return str(style_prompt_params(style).get("negative_prompt", ""))
    except Exception:
        # 快照永远不能因为风格解析失败而阻塞主流程。
        return ""


def capture_snapshot(shot: Shot, *, negative_prompt: str = "") -> dict:
    """捕获镜头的完整字段快照（含生成 Prompt 与资产绑定）。

    dialogue 以结构化列表入快照（说话人、逐句时间轴完整保留）；旧版纯文本
    镜头读取时迁移为单条对白，说话人如实取场内第一个角色。
    """

    data: dict[str, Any] = {}
    for field in _SIMPLE_FIELDS:
        value = getattr(shot, field, None)
        if field == "duration":
            data[field] = float(value if value is not None else 3.0)
        elif field == "estimated_speech_ms":
            data[field] = int(value or 0)
        elif field == "confirmed":
            data[field] = bool(value)
        else:
            data[field] = str(value or "")
    raw_dialogue = getattr(shot, "dialogue", None)
    if isinstance(raw_dialogue, str) and raw_dialogue.strip() and not raw_dialogue.lstrip().startswith(("[", "{")):
        # 旧版纯文本快照保持原样，版本历史是不可变记录，不在读取/恢复时偷偷改写。
        data["dialogue"] = raw_dialogue
    else:
        speakers = _json_list(getattr(shot, "characters_in_scene", None))
        fallback_speaker = speakers[0] if speakers and isinstance(speakers[0], str) else ""
        parsed_dialogue = parse_shot_dialogue(
            raw_dialogue,
            fallback_speaker=fallback_speaker,
            default_emotion=str(getattr(shot, "emotion", "") or "neutral"),
            warn_key=f"snapshot shot {shot.id}",
        )
        if (
            len(parsed_dialogue) == 1
            and not parsed_dialogue[0].speaker
            and not parsed_dialogue[0].action
            and parsed_dialogue[0].start_ms is None
            and parsed_dialogue[0].end_ms is None
        ):
            data["dialogue"] = str(parsed_dialogue[0].line)
        else:
            data["dialogue"] = dialogue_lines_payload(parsed_dialogue)
    for field in _JSON_LIST_FIELDS:
        data[field] = _json_list(getattr(shot, field, None))
    for field in _JSON_DICT_FIELDS:
        data[field] = _json_dict(getattr(shot, field, None))
    # visual_notes 是用户可编辑的生成 Prompt（界面上的「镜头 Prompt」输入框）。
    data["prompt"] = str(shot.visual_notes or "")
    data["negative_prompt"] = str(negative_prompt or "")
    data["version"] = int(shot.version or 1)
    return data


def capture_current_snapshot(db: Session, shot: Shot) -> dict:
    """捕获当前状态并解析风格负向词（与入库快照同一口径，供「当前版本」判断）。"""

    return capture_snapshot(shot, negative_prompt=_resolve_negative_prompt(db, shot, None))


def content_hash(snapshot: dict) -> str:
    payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def latest_version(db: Session, shot_id: str) -> ShotVersion | None:
    return (
        db.query(ShotVersion)
        .filter(ShotVersion.shot_id == shot_id)
        .order_by(ShotVersion.number.desc(), ShotVersion.created_at.desc())
        .first()
    )


def create_version(
    db: Session,
    shot: Shot,
    source: str,
    *,
    task_id: str = "",
    negative_prompt: str | None = None,
    force: bool = False,
) -> ShotVersion | None:
    """追加一条当前镜头状态的版本快照。

    当前状态与最新记录内容一致时跳过（返回已有记录），保证时间线不因重复的
    无效变更产生噪音；``force=True`` 供恢复使用——恢复必须留下新的版本记录。
    """

    if source not in VERSION_SOURCES:
        raise ValueError(f"unknown shot version source: {source}")
    snapshot = capture_snapshot(shot, negative_prompt=_resolve_negative_prompt(db, shot, negative_prompt))
    digest = content_hash(snapshot)
    head = latest_version(db, shot.id)
    if head is not None and not force and head.content_hash == digest:
        return head
    row = ShotVersion(
        id=uuid.uuid4().hex,
        shot_id=shot.id,
        project_id=shot.project_id,
        number=(int(head.number or 0) + 1) if head is not None else 1,
        version=int(shot.version or 1),
        source=source,
        task_id=str(task_id or "")[:200],
        parent_version_id=head.id if head is not None else "",
        content_hash=digest,
        snapshot=json.dumps(snapshot, ensure_ascii=False),
    )
    db.add(row)
    # flush 让同一事务内的连续追加（恢复流程先存「被替换状态」再存「恢复后状态」）
    # 都能读到最新 head，序号与父版本链不会重复/断链。
    db.flush()
    return row


def parse_snapshot(row: ShotVersion) -> dict:
    try:
        value = json.loads(row.snapshot or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def version_summary(row: ShotVersion) -> dict:
    snapshot = parse_snapshot(row)
    return {
        "id": row.id,
        "shot_id": row.shot_id,
        "number": int(row.number or 0),
        "version": int(row.version or 1),
        "source": row.source,
        "task_id": row.task_id or "",
        "parent_version_id": row.parent_version_id or "",
        "content_hash": row.content_hash or "",
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "has_image": bool(snapshot.get("storyboard_path") or snapshot.get("image_path")),
        "has_video": bool(snapshot.get("video_path")),
    }


def version_detail(row: ShotVersion) -> dict:
    detail = version_summary(row)
    detail["snapshot"] = parse_snapshot(row)
    return detail


def list_versions(db: Session, shot_id: str) -> list[dict]:
    rows = (
        db.query(ShotVersion)
        .filter(ShotVersion.shot_id == shot_id)
        .order_by(ShotVersion.number.desc(), ShotVersion.created_at.desc())
        .all()
    )
    return [version_summary(row) for row in rows]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def diff_snapshots(a: dict, b: dict) -> list[dict]:
    """逐字段对比两份快照；返回按展示顺序排列的字段差异行。"""

    ordered = [key for key in _DIFF_FIELD_ORDER if key in a or key in b]
    extras = sorted((set(a) | set(b)) - set(ordered))
    rows = []
    for key in ordered + extras:
        default: Any = [] if (key in _JSON_LIST_FIELDS or key == "dialogue") else ({} if key in _JSON_DICT_FIELDS else "")
        value_a = a.get(key, default)
        value_b = b.get(key, default)
        changed = _canonical(value_a) != _canonical(value_b)
        rows.append({"field": key, "a": value_a, "b": value_b, "changed": bool(changed)})
    return rows


def snapshot_media_paths(snapshot: dict) -> dict[str, str]:
    return {
        field: str(snapshot.get(field, "") or "").strip()
        for field in MEDIA_FIELDS
        if str(snapshot.get(field, "") or "").strip()
    }


def iter_snapshot_media_paths(raw_snapshot: str | None) -> Iterable[str]:
    """从快照 JSON 文本提取媒体路径（供存储清理保护使用）。"""

    try:
        value = json.loads(raw_snapshot or "{}")
    except (TypeError, ValueError):
        return []
    if not isinstance(value, dict):
        return []
    return [path for path in snapshot_media_paths(value).values()]


def missing_media(snapshot: dict) -> list[str]:
    """快照引用但本地已不存在的媒体路径；空路径不校验。"""

    missing: list[str] = []
    for path in snapshot_media_paths(snapshot).values():
        if existing_file(path, minimum_size=1, allowed_roots=_ALLOWED_MEDIA_ROOTS) is None:
            missing.append(path)
    return missing


def missing_asset_bindings(db: Session, shot: Shot, snapshot: dict) -> list[str]:
    """快照仍引用、但项目中已不存在的资产 ID（场景 / 角色）。"""

    project = db.query(Project).filter(Project.id == shot.project_id).first()
    asset_project_id = (project.parent_project_id if project else None) or shot.project_id
    missing: list[str] = []
    scene_id = str(snapshot.get("scene_asset_id", "") or "").strip()
    if scene_id and not db.query(SceneAsset).filter(SceneAsset.id == scene_id, SceneAsset.project_id == asset_project_id).first():
        missing.append(scene_id)
    character_ids = [str(item).strip() for item in _json_list(snapshot.get("character_asset_ids")) if str(item).strip()]
    if character_ids:
        found = {
            item.id
            for item in db.query(Character)
            .filter(Character.id.in_(character_ids), Character.project_id == asset_project_id)
            .all()
        }
        missing.extend(sorted(set(character_ids) - found))
    return missing


def apply_snapshot_to_shot(shot: Shot, snapshot: dict) -> list[str]:
    """把快照字段写回镜头（恢复）。返回写回的字段名列表。

    恢复一律回到未审核态：``confirmed`` 强制 False、审核状态降级，瞬态过程态
    （queued / video_generating）归一为稳定状态，避免恢复带回一个「生成中」
    的假状态。版本号由调用方递增，不从快照回写。
    """

    restored: list[str] = []
    for field in _SIMPLE_FIELDS:
        if field == "confirmed" or field not in snapshot:
            continue
        setattr(shot, field, snapshot[field])
        restored.append(field)
    # 对白恢复：快照可能是结构化列表（新）或旧版纯文本（迁移为单条），
    # 说话人与逐句时间轴原样写回，不做任何「第一个角色」兜底改写。
    snapshot_dialogue = snapshot.get("dialogue")
    if isinstance(snapshot_dialogue, str):
        shot.dialogue = snapshot_dialogue
    else:
        speakers = _json_list(snapshot.get("characters_in_scene"))
        fallback_speaker = speakers[0] if speakers and isinstance(speakers[0], str) else ""
        shot.dialogue = serialize_dialogue_lines(
            parse_shot_dialogue(
                snapshot_dialogue,
                fallback_speaker=fallback_speaker,
                warn_key=f"restore shot {shot.id}",
            )
        )
    restored.append("dialogue")
    shot.characters_in_scene = json.dumps(_json_list(snapshot.get("characters_in_scene")), ensure_ascii=False)
    shot.character_asset_ids = json.dumps(_json_list(snapshot.get("character_asset_ids")), ensure_ascii=False)
    shot.reference_weights = json.dumps(_json_dict(snapshot.get("reference_weights")), ensure_ascii=False)
    shot.continuity_profile = json.dumps(_json_dict(snapshot.get("continuity_profile")), ensure_ascii=False)
    restored.extend([*_JSON_LIST_FIELDS, *_JSON_DICT_FIELDS])
    shot.confirmed = False
    if shot.storyboard_status == "queued":
        shot.storyboard_status = "pending"
    if shot.status == "video_generating":
        shot.status = (
            "video_done"
            if shot.video_path
            else ("storyboard_done" if (shot.storyboard_path or shot.image_path) else "pending")
        )
    elif shot.status == "storyboard_approved":
        # 审核标记不随快照恢复，状态同步降级等待重新审核。
        shot.status = "storyboard_done"
    return restored
