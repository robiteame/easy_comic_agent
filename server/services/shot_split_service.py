"""Persisted, version-fenced shot splitting for automatic recovery.

The timing planner is deliberately pure.  This module is the small persistence
boundary used by recovery code when a split must be applied to an already
persisted timeline.  It does not commit the caller's SQLAlchemy session: the
caller owns the surrounding transaction and must commit or roll back it.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Mapping

from sqlalchemy.orm import Session

from models import Project, Shot
from services.shot_version_service import create_version
from services.story_timing import ShotExecutionPlan, load_shot_execution_plan, split_shot


class ShotSplitError(RuntimeError):
    """Base error for a split which was not persisted."""


class ShotSplitNotFound(ShotSplitError):
    """The requested project or shot does not exist."""


class ShotSplitVersionConflict(ShotSplitError):
    """The worker attempted to mutate a stale shot version."""


class ShotSplitUnsafe(ShotSplitError):
    """The split would be ambiguous, destructive, or not idempotent."""


_MEDIA_FIELDS = (
    "image_path",
    "storyboard_path",
    "audio_path",
    "video_path",
    "last_frame_path",
)
# Fields which describe the source shot and are safe to copy to a new timeline
# part.  Generated media and review locks are reset below instead of inherited.
_COPY_FIELDS = (
    "shot_type",
    "scene_description",
    "character_action",
    "camera_angle",
    "camera_movement",
    "estimated_speech_ms",
    "emotion",
    "transition",
    "characters_in_scene",
    "scene_asset_id",
    "character_asset_ids",
    "scene_group_id",
    "consistency_context",
    "reference_weights",
    "continuity_reference_path",
    "pose_reference_path",
    "depth_reference_path",
    "style_fingerprint",
    "reference_capability_warning",
    # The prompt is part of the split source and must survive on every part;
    # only generated media and review state are intentionally reset.
    "visual_notes",
)


def persist_split_shot(
    db: Session,
    *,
    project_id: str,
    shot_id: str,
    expected_version: int,
    parts: int = 2,
    reason: str = "自动恢复拆镜",
    operation_id: str = "",
) -> dict[str, Any]:
    """Split one persisted shot atomically within the caller's transaction.

    The first part keeps ``shot_id`` so existing references and version history
    remain addressable.  Other parts receive deterministic ``_part_NN`` IDs.
    Existing media is captured in the original shot's immutable version history
    and cleared from all newly renderable parts.  The function flushes, but does
    not commit, the supplied session.

    Repeating the same operation with the same operation ID returns the prior
    result without changing the timeline.  A different operation against a
    changed version raises :class:`ShotSplitVersionConflict`.
    """
    project_id = str(project_id or "").strip()
    shot_id = str(shot_id or "").strip()
    if not project_id or not shot_id:
        raise ShotSplitNotFound("project_id and shot_id are required")
    try:
        requested_parts = int(parts)
    except (TypeError, ValueError) as exc:
        raise ShotSplitUnsafe("parts must be an integer") from exc
    if requested_parts < 2 or requested_parts > 32:
        raise ShotSplitUnsafe("parts must be between 2 and 32")

    project = db.query(Project).filter(Project.id == project_id).first()
    shot = (
        db.query(Shot)
        .filter(Shot.project_id == project_id, Shot.id == shot_id)
        .first()
    )
    if project is None or shot is None:
        raise ShotSplitNotFound(f"shot not found: {shot_id}")

    operation_id = str(operation_id or "").strip() or _operation_id(
        project_id, shot_id, int(expected_version), requested_parts, reason
    )
    existing = _split_metadata(shot)
    if existing.get("operation_id") == operation_id:
        declared_ids = [str(item) for item in (existing.get("shot_ids") or []) if str(item)]
        actual_parts = int(existing.get("parts") or 0)
        recorded_requested_parts = int(existing.get("requested_parts") or actual_parts)
        prior = _existing_operation_result(
            db,
            project_id,
            shot_id,
            operation_id,
            expected_parts=actual_parts,
            expected_ids=declared_ids,
        )
        if (
            recorded_requested_parts != requested_parts
            or actual_parts < requested_parts
            or len(declared_ids) != actual_parts
            or declared_ids[0:1] != [shot_id]
            or prior["shot_ids"] != declared_ids
            or not prior["shot_ids"]
        ):
            raise ShotSplitUnsafe("split operation metadata is incomplete or mismatched")
        return {**prior, "status": "already_applied", "operation_id": operation_id}

    if int(shot.version or 1) != int(expected_version):
        raise ShotSplitVersionConflict(
            f"shot version changed: expected {expected_version}, got {int(shot.version or 1)}"
        )
    if str(shot.status or "").lower() in {"generating", "video_generating", "rendering"}:
        raise ShotSplitUnsafe("cannot split a shot while generation is active")

    source = _shot_mapping(shot)
    source_plan = load_shot_execution_plan(source)
    all_rows = (
        db.query(Shot)
        .filter(Shot.project_id == project_id)
        .order_by(Shot.sequence, Shot.id)
        .all()
    )
    existing_ids = {str(row.id) for row in all_rows}
    planned = split_shot(source, requested_parts, reason=str(reason or "自动恢复拆镜"))
    if len(planned) < requested_parts:
        raise ShotSplitUnsafe("timing planner returned too few parts")
    # Do not let split_shot silently suffix an ID on a collision: that would
    # make a retry produce a different timeline and defeat recovery idempotency.
    if any(str(item.get("shot_id") or "") in existing_ids for item in planned):
        raise ShotSplitUnsafe("a deterministic split child ID already exists")

    source_sequence = int(shot.sequence or 0)
    operation_key = operation_id[:160]
    # Freeze all source values before applying part one.  In particular, the
    # first-part update changes character_action, duration, and continuity
    # metadata; children must still inherit the original source context.
    source_values = {
        field: getattr(shot, field, None)
        for field in (*_COPY_FIELDS, "project_id", "version", "id", "continuity_profile")
    }
    source_profile = _json_dict(source_values.get("continuity_profile"))
    source_context: dict[str, Any] = {
        **source,
        **source_values,
        "id": str(shot.id),
        "project_id": str(shot.project_id),
        "version": int(shot.version or 1),
        "continuity_profile": source_profile,
    }
    final_ids = [shot_id, *[str(item.get("shot_id") or "") for item in planned[1:]]]
    if len(final_ids) != len(set(final_ids)) or any(not item for item in final_ids):
        raise ShotSplitUnsafe("split result contains duplicate or empty IDs")
    # Put the operation marker on the source before capturing the old snapshot.
    # This makes the immutable pre-split record self-describing for recovery
    # inspection, while the media paths are still the original paths.
    marked_profile = dict(source_profile)
    marked_profile["split_recovery"] = {
        "operation_id": operation_key,
        "source_shot_id": shot_id,
        "requested_parts": requested_parts,
        "parts": len(planned),
        "shot_ids": final_ids,
    }
    shot.continuity_profile = json.dumps(marked_profile, ensure_ascii=False)
    create_version(
        db,
        shot,
        "quality_retry",
        task_id=f"split:{operation_key}",
        force=True,
    )

    # Move later rows from the back so the operation remains safe even if a
    # future schema adds a uniqueness constraint on (project_id, sequence).
    later = [row for row in all_rows if int(row.sequence or 0) > source_sequence]
    for row in sorted(later, key=lambda item: int(item.sequence or 0), reverse=True):
        row.sequence = int(row.sequence or 0) + len(planned) - 1

    persisted: list[Shot] = []
    for index, item in enumerate(planned):
        item = dict(item)
        if index == 0:
            item["shot_id"] = shot.id
            item["timing"] = dict(item.get("timing") or {})
            # split_shot's generated name is useful for children, but the
            # retained first part must keep the original history address.
            structure = item["timing"].get("structure_change")
            if isinstance(structure, dict):
                after = dict(structure.get("after") or {})
                after["shot_id"] = shot.id
                structure["after"] = after
                item["timing"]["structure_change"] = structure
            target = shot
        else:
            target = Shot(id=str(item.get("shot_id") or ""), project_id=project_id)
            db.add(target)
        profile = _profile_for_part(source_context, item, source_plan, operation_key, index + 1, len(planned))
        _apply_part(target, source_context, item, profile, sequence=source_sequence + index)
        persisted.append(target)

    # Flush before recording child snapshots so all foreign-key-like identity
    # fields are materialized and the append-only version chain is complete.
    db.flush()
    create_version(
        db,
        shot,
        "quality_retry",
        task_id=f"split:{operation_key}:part:1",
        force=True,
    )
    for index, child in enumerate(persisted[1:], start=2):
        create_version(
            db,
            child,
            "import",
            task_id=f"split:{operation_key}:part:{index}",
            force=True,
        )

    project.status = "assets_ready"
    db.flush()
    result = {
        "status": "applied",
        "operation_id": operation_id,
        "project_id": project_id,
        "source_shot_id": shot_id,
        "shot_ids": [str(item.id) for item in persisted],
        "parts": len(persisted),
        "source_version": int(expected_version),
        "versions": {str(item.id): int(item.version or 1) for item in persisted},
        "sequence": {str(item.id): int(item.sequence or 0) for item in persisted},
        "reason": str(reason or ""),
    }
    # Store an idempotency marker in every part's continuity profile.  It is
    # intentionally ordinary JSON so old databases need no schema migration.
    for item in persisted:
        profile = _json_dict(item.continuity_profile)
        split_meta = dict(profile.get("split_recovery") or {})
        split_meta.update(
            {
                "operation_id": operation_id,
                "source_shot_id": shot_id,
                "requested_parts": requested_parts,
                "parts": len(persisted),
                "shot_ids": result["shot_ids"],
            }
        )
        profile["split_recovery"] = split_meta
        item.continuity_profile = json.dumps(profile, ensure_ascii=False)
    db.flush()
    return result


def _operation_id(project_id: str, shot_id: str, version: int, parts: int, reason: str) -> str:
    payload = f"{project_id}\0{shot_id}\0{version}\0{parts}\0{reason}"
    return "split-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _split_metadata(shot: Shot) -> dict[str, Any]:
    profile = _json_dict(getattr(shot, "continuity_profile", "{}"))
    value = profile.get("split_recovery")
    return dict(value) if isinstance(value, dict) else {}


def _existing_operation_result(
    db: Session,
    project_id: str,
    source_id: str,
    operation_id: str,
    *,
    expected_parts: int | None = None,
    expected_ids: list[str] | None = None,
) -> dict[str, Any]:
    rows = (
        db.query(Shot)
        .filter(Shot.project_id == project_id)
        .order_by(Shot.sequence, Shot.id)
        .all()
    )
    selected = [row for row in rows if _split_metadata(row).get("operation_id") == operation_id]
    selected.sort(key=lambda row: int(row.sequence or 0))
    selected_ids = [str(row.id) for row in selected]
    if expected_parts is not None and len(selected) != int(expected_parts):
        return {"project_id": project_id, "source_shot_id": source_id, "shot_ids": [], "parts": 0, "versions": {}, "sequence": {}}
    if expected_ids is not None and selected_ids != list(expected_ids):
        return {"project_id": project_id, "source_shot_id": source_id, "shot_ids": [], "parts": 0, "versions": {}, "sequence": {}}
    # Every marked part must still be pending/recoverable and carry the same
    # operation marker.  A partially edited or externally regenerated part is
    # not safe to silently accept as an idempotent replay.
    if any(
        str(_split_metadata(row).get("source_shot_id") or "") != source_id
        or int(_split_metadata(row).get("parts") or 0) != len(selected)
        or int(_split_metadata(row).get("requested_parts") or len(selected)) > len(selected)
        for row in selected
    ):
        return {"project_id": project_id, "source_shot_id": source_id, "shot_ids": [], "parts": 0, "versions": {}, "sequence": {}}
    return {
        "project_id": project_id,
        "source_shot_id": source_id,
        "shot_ids": [str(row.id) for row in selected],
        "parts": len(selected),
        "source_version": int(selected[0].version or 1),
        "versions": {str(row.id): int(row.version or 1) for row in selected},
        "sequence": {str(row.id): int(row.sequence or 0) for row in selected},
    }


def _shot_mapping(shot: Shot) -> dict[str, Any]:
    profile = _json_dict(getattr(shot, "continuity_profile", "{}"))
    return {
        "shot_id": str(shot.id),
        "shot_type": str(shot.shot_type or "medium"),
        "scene_description": str(shot.scene_description or ""),
        "character_action": str(shot.character_action or ""),
        "dialogue": _json_value(shot.dialogue, []),
        "camera_angle": str(shot.camera_angle or "正面"),
        "camera_movement": str(shot.camera_movement or "静止"),
        "duration": float(shot.duration or 3.0),
        "emotion": str(shot.emotion or "neutral"),
        "transition": str(shot.transition or "cut"),
        "characters_in_scene": _json_value(shot.characters_in_scene, []),
        "scene_asset_id": str(shot.scene_asset_id or ""),
        "scene_group_id": str(shot.scene_group_id or ""),
        "continuity_profile": profile,
        "timing": dict(profile.get("timing") or {}),
        "version": int(shot.version or 1),
    }


def _profile_for_part(
    source: Mapping[str, Any],
    item: dict[str, Any],
    source_plan: ShotExecutionPlan | None,
    operation_id: str,
    part: int,
    total: int,
) -> dict[str, Any]:
    profile = _json_dict(source.get("continuity_profile"))
    profile["timing"] = dict(item.get("timing") or {})
    plan = ShotExecutionPlan.derive(
        {
            **item,
            "continuity_profile": profile,
        },
        shot_id=str(item.get("shot_id") or ""),
        audio_mode=source_plan.audio_mode if source_plan else "",
        candidate_count=source_plan.candidate_count if source_plan else 0,
        required_capabilities=source_plan.required_capabilities if source_plan else None,
        recovery_budget=source_plan.recovery_budget if source_plan else -1,
    )
    profile["execution_plan"] = plan.to_dict()
    profile["split_recovery"] = {
        "operation_id": operation_id,
        "source_shot_id": str(source.get("id") or source.get("shot_id") or ""),
        "part": part,
        "parts": total,
    }
    return profile


def _apply_part(target: Shot, source: Mapping[str, Any], item: dict[str, Any], profile: dict[str, Any], *, sequence: int) -> None:
    target.project_id = str(source.get("project_id") or target.project_id or "")
    target.sequence = sequence
    for field in _COPY_FIELDS:
        setattr(target, field, source.get(field))
    target.shot_type = str(item.get("shot_type") or source.get("shot_type") or "medium")
    target.scene_description = str(item.get("scene_description") or "")
    target.character_action = str(item.get("character_action") or "")
    target.dialogue = item.get("dialogue") or ""
    target.camera_angle = str(item.get("camera_angle") or source.get("camera_angle") or "正面")
    target.camera_movement = str(item.get("camera_movement") or source.get("camera_movement") or "静止")
    target.duration = float(item.get("duration") or 0.001)
    target.estimated_speech_ms = int(item.get("estimated_speech_ms") or 0)
    target.emotion = str(item.get("emotion") or source.get("emotion") or "neutral")
    target.transition = str(item.get("transition") or source.get("transition") or "cut")
    target.characters_in_scene = json.dumps(item.get("characters_in_scene") or _json_value(source.get("characters_in_scene"), []), ensure_ascii=False)
    target.continuity_profile = json.dumps(profile, ensure_ascii=False)
    target.version = int(source.get("version") or 1) + 1 if target.id == str(source.get("id") or "") else 1
    target.confirmed = False
    target.status = "pending"
    target.storyboard_status = "pending"
    target.media_stale = True
    if hasattr(target, "consistency_status"):
        target.consistency_status = "pending"
    for field in _MEDIA_FIELDS:
        setattr(target, field, "")
    # A new part must not accidentally reuse a prior rendered candidate/manifest.
    if hasattr(target, "storyboard_reference_manifest"):
        target.storyboard_reference_manifest = "[]"
    if hasattr(target, "video_reference_manifest"):
        target.video_reference_manifest = "[]"


def _json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _json_value(raw: Any, default: Any) -> Any:
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw or json.dumps(default))
    except (TypeError, ValueError):
        return default


__all__ = [
    "ShotSplitError",
    "ShotSplitNotFound",
    "ShotSplitUnsafe",
    "ShotSplitVersionConflict",
    "persist_split_shot",
]
