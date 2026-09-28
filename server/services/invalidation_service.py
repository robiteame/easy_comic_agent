"""素材失效的唯一口径：标记过期待重生成，绝不清空媒体路径。

镜头的旧素材（故事板 / 视频 / 配音 / 尾帧）必须一直保留到新素材生成成功
并原子替换；参数编辑、资产换绑、画风切换等操作只把 ``media_stale`` 置为
True 并撤销审核锁定，用户仍可预览、对比与回滚旧素材。
"""

import json

from sqlalchemy.orm import Session

from models import Project, Shot


def mark_shot_media_stale(shot: Shot, *, reset_confirmed: bool = True) -> None:
    """把镜头当前素材标记为「参数已变更，待重新生成」。

    只做标记与审核解锁；媒体路径、一致性档案（含镜头级 audio_mode 覆盖）、
    场景组等用户可见状态全部保留，由后续生成成功的写回原子替换。
    """

    shot.media_stale = True
    if reset_confirmed:
        shot.confirmed = False


def clear_shot_media_stale(shot: Shot) -> None:
    """素材已与当前参数重新一致（生成成功 / 版本恢复）时清除过期标记。"""

    shot.media_stale = False


def invalidate_asset_consumers(
    db: Session,
    asset_project_id: str,
    *,
    character_id: str = "",
    scene_id: str = "",
) -> set[str]:
    """Invalidate shots that consume an edited series or episode asset."""

    children_by_parent: dict[str, list[str]] = {}
    for project_id, parent_id in db.query(Project.id, Project.parent_project_id).all():
        children_by_parent.setdefault(parent_id or "", []).append(project_id)
    project_ids: set[str] = set()
    pending = [asset_project_id]
    while pending:
        current = pending.pop()
        if current in project_ids:
            continue
        project_ids.add(current)
        pending.extend(children_by_parent.get(current, ()))

    candidates = db.query(Shot).filter(Shot.project_id.in_(project_ids)).all()
    affected = [
        shot
        for shot in candidates
        if (scene_id and shot.scene_asset_id == scene_id)
        or (character_id and character_id in _json_list(shot.character_asset_ids))
    ]
    affected_projects = {shot.project_id for shot in affected}
    for shot in affected:
        # 资产描述变更后，消费该资产的镜头素材全部过期：保留旧素材可预览，
        # 版本号 +1 隔离在途生成任务（迟到任务写不回过期版本）。
        mark_shot_media_stale(shot)
        shot.version = (shot.version or 1) + 1
    if affected_projects:
        for project in db.query(Project).filter(Project.id.in_(affected_projects)).all():
            project.status = "assets_ready"

    scopes = {f"shot:{shot.id}" for shot in affected}
    scopes.update(f"project:{project_id}" for project_id in affected_projects)
    return scopes


def _json_list(raw: str | None) -> list:
    try:
        value = json.loads(raw or "[]")
        return list(value) if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []
