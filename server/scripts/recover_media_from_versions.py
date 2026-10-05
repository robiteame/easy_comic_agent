"""从镜头版本历史安全恢复被清空的媒体路径。

适用场景：数据库中 shots 的 image_path / storyboard_path / video_path /
audio_path / last_frame_path 因历史缺陷被清空，但素材文件仍在
``output/`` 下、且 ``shot_versions.snapshot`` 保留了路径。

恢复规则（与 ``POST /api/shot/{shot_id}/versions/{version_id}/restore`` 同口径）：
- 只处理「五类媒体路径全部为空」的镜头——任何仍有引用的镜头绝不动；
- 从最新版本向前找到第一个「快照中至少一个媒体文件真实存在」的版本；
- 恢复前校验文件存在（缺失则跳过并报告，不修改当前镜头）；
- 追加 ``restore`` 版本记录，不修改任何历史版本；
- 恢复后 ``media_stale`` 置 True：旧素材与当前参数可能不一致，界面会
  明确提示「待重新生成」，不会冒充新生成结果。

用法::

    python scripts/recover_media_from_versions.py                     # dry-run
    python scripts/recover_media_from_versions.py --apply            # 真正恢复
    python scripts/recover_media_from_versions.py --apply --project <id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from db import SessionLocal, init_db  # noqa: E402
from models import Project, Shot, ShotVersion  # noqa: E402
from services.invalidation_service import mark_shot_media_stale  # noqa: E402
from services.shot_version_service import (  # noqa: E402
    apply_snapshot_to_shot,
    create_version,
    missing_media,
    parse_snapshot,
    snapshot_media_paths,
)


def _has_any_media(shot: Shot) -> bool:
    return any(
        str(value or "").strip()
        for value in (shot.image_path, shot.storyboard_path, shot.video_path, shot.audio_path, shot.last_frame_path)
    )


def _recoverable_version(db: SessionLocal, shot: Shot) -> ShotVersion | None:
    rows = (
        db.query(ShotVersion)
        .filter(ShotVersion.shot_id == shot.id)
        .order_by(ShotVersion.number.desc(), ShotVersion.created_at.desc())
        .all()
    )
    for row in rows:
        snapshot = parse_snapshot(row)
        if not snapshot_media_paths(snapshot):
            continue
        if missing_media(snapshot):
            # 快照引用的文件已缺失：不能拿它恢复，继续找更早的版本。
            continue
        return row
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="从版本历史恢复被清空的镜头媒体路径")
    parser.add_argument("--project", default="", help="只处理指定项目 ID（默认全部项目）")
    parser.add_argument("--apply", action="store_true", help="真正写库（默认 dry-run 只报告）")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    plan: list[tuple[Shot, ShotVersion]] = []
    skipped: list[tuple[Shot, str]] = []
    try:
        query = db.query(Shot).order_by(Shot.project_id, Shot.sequence)
        if args.project:
            if not db.query(Project).filter(Project.id == args.project).first():
                print(f"项目不存在: {args.project}")
                return 2
            query = query.filter(Shot.project_id == args.project)
        for shot in query.all():
            if _has_any_media(shot):
                continue  # 仍有引用的镜头绝不重置——禁止静默改写。
            version = _recoverable_version(db, shot)
            if version is None:
                skipped.append((shot, "无可恢复版本（历史快照为空或文件已缺失）"))
                continue
            plan.append((shot, version))

        if not plan and not skipped:
            print("所有镜头媒体路径均有效，无需恢复。")
            return 0

        for shot, version in plan:
            print(
                f"[可恢复] 项目 {shot.project_id} 镜头 {shot.sequence} ({shot.id}) ← 版本 #{version.number} ({version.id})"
            )
        for shot, reason in skipped:
            print(f"[跳过]   项目 {shot.project_id} 镜头 {shot.sequence} ({shot.id}): {reason}")

        if not args.apply:
            print("\ndry-run：以上为恢复计划，加 --apply 执行。")
            return 0

        restored = 0
        for shot, version in plan:
            snapshot = parse_snapshot(version)
            create_version(db, shot, "restore")
            apply_snapshot_to_shot(shot, snapshot)
            shot.version = (shot.version or 1) + 1
            # 旧素材与当前参数可能已不一致：标记待重生成，界面明确提示。
            mark_shot_media_stale(shot)
            create_version(db, shot, "restore", force=True)
            db.commit()
            restored += 1
            print(f"[已恢复] {shot.id} ← 版本 #{version.number}")
        print(f"\n完成：恢复 {restored} 个镜头；历史版本仅追加，未改写。")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
