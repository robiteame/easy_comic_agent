"""一致性参考素材的生命周期、影响范围与下游门禁。

这个模块是参考素材状态的唯一口径。角色三视图、场景基准图都必须留下
``ready / failed / degraded / unsupported / stale`` 状态、失败原因、重试次数、
错误编号和受影响镜头范围；下游故事板/视频通过 manifest 记录实际使用的版本。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Iterable

from sqlalchemy.orm import Session

from models import Character, Project, SceneAsset, Shot
from services.error_reporter import redact

REFERENCE_STATUSES = ("ready", "failed", "degraded", "unsupported", "stale")
BLOCKING_STATUSES = frozenset({"failed", "unsupported", "stale"})
ALLOWED_AFTER_CONFIRMATION = frozenset({"degraded"})


def _json(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        parsed = json.loads(value or "")
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback
    return parsed if isinstance(parsed, type(fallback)) else fallback


def _json_list(value: Any) -> list[str]:
    parsed = _json(value, [])
    return [str(item) for item in parsed if str(item).strip()]


def _paths(kind: str, item: Any) -> list[str]:
    if kind == "scene":
        values = [getattr(item, "baseline_image_path", "")]
        values.extend(_json_list(getattr(item, "reference_images", "[]")))
    else:
        values = _json_list(getattr(item, "reference_images", "[]"))
    return list(dict.fromkeys(str(path) for path in values if str(path).strip()))


def _status_for_item(kind: str, item: Any) -> str:
    status = str(getattr(item, "reference_status", "") or "").strip().lower()
    paths = _paths(kind, item)
    # 迁移前的旧资产没有状态列；已有可用参考图时按 ready 处理。
    if status == "stale" and paths and not str(getattr(item, "reference_failure_reason", "") or "") and str(getattr(item, "asset_status", "") or "") != "stale":
        return "ready"
    if status in REFERENCE_STATUSES:
        return status
    return "ready" if paths else "stale"


def _project_tree_ids(db: Session, root_id: str) -> set[str]:
    children: dict[str, list[str]] = {}
    for project_id, parent_id in db.query(Project.id, Project.parent_project_id).all():
        children.setdefault(str(parent_id or ""), []).append(str(project_id))
    result: set[str] = set()
    pending = [str(root_id or "")]
    while pending:
        current = pending.pop()
        if not current or current in result:
            continue
        result.add(current)
        pending.extend(children.get(current, []))
    return result


def _shot_impact(db: Session, kind: str, item: Any) -> list[dict[str, Any]]:
    project_ids = _project_tree_ids(db, str(getattr(item, "project_id", "") or ""))
    if not project_ids:
        return []
    shots = (
        db.query(Shot)
        .filter(Shot.project_id.in_(project_ids))
        .order_by(Shot.project_id, Shot.sequence, Shot.id)
        .all()
    )
    item_id = str(getattr(item, "id", "") or "")
    item_name = str(getattr(item, "name", "") or "")
    group_key = str(getattr(item, "scene_group_key", "") or "")
    impact: list[dict[str, Any]] = []
    for shot in shots:
        if kind == "character":
            ids = _json_list(shot.character_asset_ids)
            names = _json_list(shot.characters_in_scene)
            matched = item_id in ids or (item_name and item_name in names)
        else:
            scene_id = str(shot.scene_asset_id or "")
            scene_group = str(shot.scene_group_id or "")
            matched = scene_id == item_id or (group_key and scene_group == group_key)
        if matched:
            impact.append(
                {
                    "shot_id": str(shot.id),
                    "project_id": str(shot.project_id),
                    "sequence": int(shot.sequence or 0),
                }
            )
    return impact


def _shot_range(impact: Iterable[dict[str, Any]]) -> str:
    sequences = sorted({int(item.get("sequence") or 0) for item in impact if int(item.get("sequence") or 0) > 0})
    if not sequences:
        return ""
    if len(sequences) == 1:
        return f"镜头 {sequences[0]}"
    contiguous = sequences == list(range(sequences[0], sequences[-1] + 1))
    return f"镜头 {sequences[0]}-{sequences[-1]}" if contiguous else "镜头 " + "、".join(str(value) for value in sequences)


def item_record(kind: str, item: Any, impact: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    impact = impact if impact is not None else []
    paths = _paths(kind, item)
    status = _status_for_item(kind, item)
    return {
        "kind": "character" if kind == "character" else "scene",
        "reference_type": "character_three_view" if kind == "character" else "scene_baseline",
        "asset_id": str(getattr(item, "id", "") or ""),
        "name": str(getattr(item, "name", "") or ""),
        "status": status,
        "reference_version": int(getattr(item, "reference_version", 1) or 1),
        "retry_count": int(getattr(item, "reference_retry_count", 0) or 0),
        "failure_reason": str(getattr(item, "reference_failure_reason", "") or ""),
        "error_id": str(getattr(item, "reference_error_id", "") or ""),
        "skip_reason": str(getattr(item, "reference_skip_reason", "") or ""),
        "capability_warning": str(getattr(item, "reference_capability_warning", "") or ""),
        "reference_images": paths,
        "affected_shot_ids": [str(row["shot_id"]) for row in impact],
        "affected_shot_count": len(impact),
        "shot_range": _shot_range(impact),
        "shots": impact,
    }


def build_report(db: Session, asset_project_id: str, *, persist: bool = True) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for item in db.query(Character).filter(Character.project_id == asset_project_id).order_by(Character.id).all():
        impact = _shot_impact(db, "character", item)
        items.append(item_record("character", item, impact))
        if persist:
            item.reference_impact = json.dumps(
                {"shot_ids": [row["shot_id"] for row in impact], "shot_range": _shot_range(impact)},
                ensure_ascii=False,
            )
    for item in db.query(SceneAsset).filter(SceneAsset.project_id == asset_project_id).order_by(SceneAsset.id).all():
        impact = _shot_impact(db, "scene", item)
        items.append(item_record("scene", item, impact))
        if persist:
            item.reference_impact = json.dumps(
                {"shot_ids": [row["shot_id"] for row in impact], "shot_range": _shot_range(impact)},
                ensure_ascii=False,
            )

    statuses = [item["status"] for item in items]
    if any(status == "failed" for status in statuses):
        overall = "failed"
    elif any(status == "unsupported" for status in statuses):
        overall = "unsupported"
    elif any(status == "stale" for status in statuses):
        overall = "stale"
    elif any(status == "degraded" for status in statuses):
        overall = "degraded"
    else:
        overall = "ready"
    affected_ids = sorted({shot_id for item in items for shot_id in item["affected_shot_ids"]})
    report = {
        "status": overall,
        "blocking": overall in BLOCKING_STATUSES,
        "degraded": overall in {"degraded", "unsupported"} or any(item["status"] == "degraded" for item in items),
        "items": items,
        "affected_shot_ids": affected_ids,
        "affected_shot_count": len(affected_ids),
        "shot_range": _shot_range({"shot_id": shot_id, "sequence": index + 1} for index, shot_id in enumerate(affected_ids)),
        "capability_warnings": sorted({item["capability_warning"] for item in items if item["capability_warning"]}),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    # 重新计算真实序列范围（上面的占位 sequence 不可靠）。
    sequence_by_shot = {
        str(shot.id): int(shot.sequence or 0)
        for shot in db.query(Shot).filter(Shot.id.in_(affected_ids)).all()
    } if affected_ids else {}
    report["shot_range"] = _shot_range(
        {"shot_id": shot_id, "sequence": sequence_by_shot.get(shot_id, 0)} for shot_id in affected_ids
    )
    return report


def refresh_project_reference_state(db: Session, project_id: str, *, persist: bool = True) -> dict[str, Any]:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return {"status": "failed", "blocking": True, "items": [], "affected_shot_ids": []}
    asset_project_id = project.parent_project_id or project.id
    report = build_report(db, asset_project_id, persist=persist)
    item_status_by_shot: dict[str, list[str]] = {}
    item_report_by_shot: dict[str, list[dict[str, Any]]] = {}
    for item in report["items"]:
        for shot_id in item["affected_shot_ids"]:
            item_status_by_shot.setdefault(shot_id, []).append(item["status"])
            item_report_by_shot.setdefault(shot_id, []).append(item)

    for shot in db.query(Shot).filter(Shot.project_id.in_(_project_tree_ids(db, project_id))).all():
        statuses = item_status_by_shot.get(str(shot.id), [])
        if any(status == "failed" for status in statuses):
            shot_status = "failed"
        elif any(status == "unsupported" for status in statuses):
            shot_status = "unsupported"
        elif any(status == "stale" for status in statuses):
            shot_status = "stale"
        elif any(status == "degraded" for status in statuses):
            shot_status = "degraded"
        else:
            shot_status = "ready" if statuses else "pending"
        shot.consistency_status = shot_status
        shot.consistency_report = json.dumps(
            {
                "status": shot_status,
                "items": item_report_by_shot.get(str(shot.id), []),
                "affected_shot_ids": [str(shot.id)],
            },
            ensure_ascii=False,
        )

    project.consistency_report = json.dumps(report, ensure_ascii=False)
    if report["blocking"] and project.status not in {"error", "needs_review"}:
        project.status = "needs_review"
    elif not report["blocking"] and report["degraded"] and project.status not in {"error"}:
        project.status = "degraded"
    if persist:
        db.commit()
    return report


def mark_reference_success(
    db: Session,
    kind: str,
    item: Any,
    path: str,
    *,
    capability_warning: str = "",
    bump_version: bool = True,
) -> None:
    current = _paths(kind, item)
    next_paths = [str(path)] if path else current
    changed = next_paths != current
    item.reference_images = json.dumps(next_paths, ensure_ascii=False)
    if kind == "scene":
        item.baseline_image_path = str(path or item.baseline_image_path or "")
    item.reference_status = "ready"
    item.reference_retry_count = int(getattr(item, "reference_retry_count", 0) or 0) + 1
    if bump_version:
        previous_version = int(getattr(item, "reference_version", 1) or 1)
        if current:
            item.reference_version = previous_version + 1
        else:
            item.reference_version = max(1, previous_version)
    item.reference_failure_reason = ""
    item.reference_error_id = ""
    item.reference_skip_reason = ""
    item.reference_capability_warning = str(capability_warning or "")
    item.asset_status = "active"


def mark_reference_failure(
    db: Session,
    kind: str,
    item: Any,
    exc: BaseException | str,
    *,
    error_id: str = "",
    capability_warning: str = "",
) -> None:
    existing = _paths(kind, item)
    item.reference_status = "stale" if existing else "failed"
    item.reference_retry_count = int(getattr(item, "reference_retry_count", 0) or 0) + 1
    item.reference_failure_reason = redact(str(exc), limit=500)
    item.reference_error_id = str(error_id or "")
    item.reference_capability_warning = str(capability_warning or "")
    # 旧参考图继续保留，避免重试失败时把可回滚素材也清空。


def mark_reference_degraded(
    db: Session,
    kind: str,
    item: Any,
    *,
    reason: str,
    capability_warning: str = "",
) -> None:
    item.reference_status = "degraded"
    item.reference_skip_reason = redact(reason, limit=500)
    item.reference_capability_warning = str(capability_warning or item.reference_capability_warning or "")
    item.asset_status = "active"


def mark_reference_unsupported(
    db: Session,
    kind: str,
    item: Any,
    *,
    warning: str,
    error_id: str = "",
) -> None:
    item.reference_status = "unsupported"
    item.reference_capability_warning = redact(warning, limit=500)
    item.reference_error_id = str(error_id or "")


def build_manifest_for_shot(db: Session, shot: Shot, *, stage: str = "storyboard") -> list[dict[str, Any]]:
    """记录本次下游生成实际绑定的参考素材及版本。"""

    project = db.query(Project).filter(Project.id == shot.project_id).first()
    asset_project_id = (project.parent_project_id if project else "") or shot.project_id
    manifest: list[dict[str, Any]] = []
    character_ids = _json_list(shot.character_asset_ids)
    if character_ids:
        for item in db.query(Character).filter(Character.project_id == asset_project_id, Character.id.in_(character_ids)).all():
            for path in _paths("character", item):
                manifest.append(
                    {
                        "type": "character_three_view",
                        "asset_id": str(item.id),
                        "name": str(item.name or ""),
                        "path": path,
                        "version": int(item.reference_version or 1),
                        "status": _status_for_item("character", item),
                        "usage": "direct_reference" if stage == "storyboard" else "upstream_dependency",
                        "sent": stage == "storyboard",
                    }
                )
    if shot.scene_asset_id:
        item = db.query(SceneAsset).filter(SceneAsset.project_id == asset_project_id, SceneAsset.id == shot.scene_asset_id).first()
        if item:
            for path in _paths("scene", item):
                manifest.append(
                    {
                        "type": "scene_baseline",
                        "asset_id": str(item.id),
                        "name": str(item.name or ""),
                        "path": path,
                        "version": int(item.reference_version or 1),
                        "status": _status_for_item("scene", item),
                        "usage": "direct_reference" if stage == "storyboard" else "upstream_dependency",
                        "sent": stage == "storyboard",
                    }
                )
    continuity = str(shot.continuity_reference_path or "")
    if continuity:
        manifest.append(
            {
                "type": "continuity_frame",
                "asset_id": "continuity",
                "name": "previous-shot-frame",
                "path": continuity,
                "version": hashlib.sha256(continuity.encode()).hexdigest()[:16],
                "status": "ready",
                "usage": "direct_reference" if stage == "storyboard" else "upstream_dependency",
                "sent": stage == "storyboard",
            }
        )
    if stage == "video":
        storyboard = str(shot.storyboard_path or shot.image_path or "")
        if storyboard:
            manifest.append(
                {
                    "type": "approved_storyboard_first_frame",
                    "asset_id": "storyboard",
                    "name": "approved-storyboard",
                    "path": storyboard,
                    "version": hashlib.sha256(storyboard.encode()).hexdigest()[:16],
                    "status": "ready",
                    "usage": "direct_reference",
                    "sent": True,
                }
            )
    return manifest


def blocking_report(report: dict[str, Any], *, allow_degraded: bool = False) -> dict[str, Any]:
    items = [
        item
        for item in report.get("items", [])
        if item.get("status") in BLOCKING_STATUSES or (not allow_degraded and item.get("status") == "degraded")
    ]
    return {
        **report,
        "blocking": bool(items),
        "blocking_items": items,
    }


def ensure_generation_gate(
    db: Session,
    project_id: str,
    *,
    allow_degraded: bool = False,
    shot_ids: list[str] | None = None,
) -> dict[str, Any]:
    report = refresh_project_reference_state(db, project_id)
    scoped = report
    if shot_ids is not None:
        wanted = set(str(value) for value in shot_ids)
        scoped = {
            **report,
            "items": [item for item in report.get("items", []) if wanted.intersection(item.get("affected_shot_ids", []))],
        }
        scoped["affected_shot_ids"] = sorted({shot_id for item in scoped["items"] for shot_id in item.get("affected_shot_ids", [])})
    return blocking_report(scoped, allow_degraded=allow_degraded)


def accept_degraded_reference(db: Session, kind: str, asset_id: str, *, reason: str) -> dict[str, Any]:
    model = Character if kind == "character" else SceneAsset
    item = db.query(model).filter(model.id == asset_id).first()
    if not item:
        raise ValueError("参考素材不存在")
    mark_reference_degraded(db, kind, item, reason=reason)
    db.commit()
    project_id = str(item.project_id or "")
    report = refresh_project_reference_state(db, project_id)
    return item_record(kind, item, _shot_impact(db, kind, item)) | {"project_report": report}
