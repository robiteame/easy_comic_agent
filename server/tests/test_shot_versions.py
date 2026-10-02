"""镜头版本历史（版本快照 / A-B 对比 / 恢复）的回归测试。

覆盖验收标准：
- 编辑、重新生成、恢复都会生成正确的版本链（来源、任务 ID、父版本）；
- 版本记录只追加：SQLite 触发器拒绝任何 UPDATE；
- A/B 对比是只读操作，不改变当前版本；
- 恢复追加新版本记录，旧版本仍可查看；恢复前校验媒体与资产有效性；
- 已审核锁定的镜头禁止恢复；
- 过期任务的写回不能追加版本或覆盖当前状态；
- 下游失效与剧本导入同样进入版本历史；存储清理不误删版本引用的媒体。
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT  # noqa: F401,E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from api.routes import script as script_route  # noqa: E402
from api.routes import shot as shot_route  # noqa: E402
from config import settings  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from main import app  # noqa: E402
from models import BackgroundJob, Character, Project, SceneAsset, Shot, ShotVersion  # noqa: E402
from services.shot_version_service import (  # noqa: E402
    create_version,
    list_versions,
    parse_snapshot,
    version_detail,
)


def _media_file(name: str) -> str:
    path = Path(settings.OUTPUT_DIR) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"media-bytes")
    return str(path)


class ShotVersionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()
        cls.client = TestClient(app)

    def setUp(self) -> None:
        self.db = SessionLocal()
        self.db.query(BackgroundJob).delete()
        self.db.query(ShotVersion).delete()
        self.db.query(Shot).delete()
        self.db.query(Character).delete()
        self.db.query(SceneAsset).delete()
        self.db.query(Project).delete()
        self.db.commit()

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()

    def make_project_and_shot(self, shot_id: str = "shot-ver-shot", **shot_overrides) -> tuple[Project, Shot]:
        project = Project(id="shot-ver-project", title="版本测试")
        defaults = dict(id=shot_id, project_id=project.id, sequence=1, version=1, dialogue="one")
        defaults.update(shot_overrides)
        shot = Shot(**defaults)
        self.db.add_all([project, shot])
        self.db.commit()
        return project, shot

    def rows(self, shot_id: str) -> list[ShotVersion]:
        return (
            self.db.query(ShotVersion)
            .filter(ShotVersion.shot_id == shot_id)
            .order_by(ShotVersion.number)
            .all()
        )


class EditVersionChainTests(ShotVersionTestCase):
    def test_edit_appends_pre_change_snapshot_and_parent_chain(self) -> None:
        _, shot = self.make_project_and_shot()
        asyncio.run(shot_route.update_shot(shot.id, shot_route.ShotUpdate(dialogue="two"), self.db))
        asyncio.run(shot_route.update_shot(shot.id, shot_route.ShotUpdate(dialogue="three"), self.db))

        rows = self.rows(shot.id)
        self.assertEqual([row.number for row in rows], [1, 2])
        self.assertTrue(all(row.source == "manual_edit" for row in rows))
        self.assertEqual(parse_snapshot(rows[0])["dialogue"], "one")
        self.assertEqual(parse_snapshot(rows[1])["dialogue"], "two")
        self.assertEqual(rows[1].parent_version_id, rows[0].id)
        self.assertEqual(rows[0].parent_version_id, "")

        self.db.expire_all()
        self.assertEqual(self.db.get(Shot, shot.id).version, 3)

    def test_update_without_changes_never_appends(self) -> None:
        _, shot = self.make_project_and_shot()
        asyncio.run(shot_route.update_shot(shot.id, shot_route.ShotUpdate(), self.db))
        self.assertEqual(self.rows(shot.id), [])
        self.db.expire_all()
        self.assertEqual(self.db.get(Shot, shot.id).version, 1)

    def test_repeated_snapshots_of_identical_state_are_deduplicated(self) -> None:
        _, shot = self.make_project_and_shot()
        first = create_version(self.db, shot, "manual_edit")
        second = create_version(self.db, shot, "manual_edit")
        self.db.commit()
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.rows(shot.id)), 1)

    def test_downstream_media_invalidation_snapshots_affected_shot(self) -> None:
        project = Project(id="shot-ver-project", title="版本测试")
        upstream = Shot(id="shot-ver-up", project_id=project.id, sequence=1, version=1, scene_group_id="room")
        downstream = Shot(
            id="shot-ver-down",
            project_id=project.id,
            sequence=2,
            version=4,
            scene_group_id="room",
            storyboard_path="story.png",
            image_path="story.png",
            video_path="clip.mp4",
            audio_path="voice.wav",
            status="video_done",
        )
        self.db.add_all([project, upstream, downstream])
        self.db.commit()

        asyncio.run(shot_route.update_shot(upstream.id, shot_route.ShotUpdate(scene_description="changed"), self.db))

        rows = self.rows(downstream.id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].source, "manual_edit")
        snapshot = parse_snapshot(rows[0])
        self.assertEqual(snapshot["video_path"], "clip.mp4")
        self.assertEqual(snapshot["audio_path"], "voice.wav")
        self.db.expire_all()
        refreshed = self.db.get(Shot, downstream.id)
        # 素材只标记过期（media_stale），媒体路径必须保留供预览与回滚。
        self.assertEqual(refreshed.video_path, "clip.mp4")
        self.assertEqual(refreshed.audio_path, "voice.wav")
        self.assertTrue(refreshed.media_stale)


class RegenerationVersionTests(ShotVersionTestCase):
    def test_storyboard_regeneration_records_pre_state_and_result_with_task_id(self) -> None:
        _, shot = self.make_project_and_shot()
        image_path = _media_file("regen-result.png")
        task_key = f"shot:{shot.id}:storyboard"
        # 模拟 endpoint 在启动任务前的快照与版本递增。
        shot.version = 2
        shot.storyboard_status = "queued"
        create_version(self.db, shot, "regenerate", task_id=task_key)
        self.db.commit()

        async def fake_image(**_kwargs):
            return image_path

        with patch.object(shot_route.image_service, "generate_shot_image", side_effect=fake_image):
            asyncio.run(shot_route._regenerate_single_shot(shot.id, "reason", expected_version=2))

        rows = self.rows(shot.id)
        self.assertEqual([row.number for row in rows], [1, 2])
        self.assertEqual([row.source for row in rows], ["regenerate", "regenerate"])
        self.assertEqual(rows[0].task_id, task_key)
        self.assertEqual(rows[1].task_id, task_key)
        self.assertEqual(rows[1].parent_version_id, rows[0].id)
        self.assertEqual(parse_snapshot(rows[1])["storyboard_path"], image_path)

    def test_stale_task_writeback_never_appends_or_overwrites(self) -> None:
        _, shot = self.make_project_and_shot()
        create_version(self.db, shot, "regenerate", task_id=f"shot:{shot.id}:storyboard")
        self.db.commit()
        before = self.rows(shot.id)

        async def generate_then_bump_version(**_kwargs):
            other = SessionLocal()
            try:
                current = other.get(Shot, shot.id)
                current.version = 9
                current.dialogue = "edited elsewhere"
                other.commit()
            finally:
                other.close()
            return str(TEST_ROOT / "stale-result.png")

        with patch.object(shot_route.image_service, "generate_shot_image", side_effect=generate_then_bump_version):
            with self.assertRaisesRegex(RuntimeError, "版本已变化"):
                asyncio.run(shot_route._regenerate_single_shot(shot.id, "reason", expected_version=1))

        self.assertEqual(self.rows(shot.id), before)
        self.db.expire_all()
        current = self.db.get(Shot, shot.id)
        self.assertEqual(current.version, 9)
        self.assertEqual(current.dialogue, "edited elsewhere")
        self.assertEqual(current.storyboard_path, "")

    def test_regenerate_endpoint_records_pre_state_snapshot(self) -> None:
        _, shot = self.make_project_and_shot(visual_notes="origin prompt")
        response = self.client.post(
            f"/api/shot/{shot.id}/regenerate",
            json={"prompt": "new prompt", "reason": "retry"},
        )
        self.assertEqual(response.status_code, 200, response.text)

        rows = self.rows(shot.id)
        # 本地占位图生成得很快：请求返回时结果版本可能已经写入，但第一条
        # 一定是「重新生成前」的快照。
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0].source, "regenerate")
        self.assertEqual(rows[0].task_id, f"shot:{shot.id}:storyboard")
        snapshot = parse_snapshot(rows[0])
        self.assertEqual(snapshot["visual_notes"], "origin prompt")
        self.assertIn("negative_prompt", snapshot)
        for row in rows:
            self.assertEqual(row.task_id, f"shot:{shot.id}:storyboard")


class ImportVersionTests(ShotVersionTestCase):
    def test_persist_phase1_imports_initial_versions_and_replaces_them_on_reparse(self) -> None:
        project = Project(id="import-project", title="导入")
        self.db.add(project)
        self.db.commit()
        state = {
            "script_title": "导入剧本",
            "user_input": "剧本内容",
            "characters": [],
            "script_scenes": [],
            "shots": [
                {"scene_description": "s1", "dialogue": "d1"},
                {"scene_description": "s2", "dialogue": "d2"},
            ],
        }
        script_route._persist_phase1(self.db, project.id, state)
        rows = self.db.query(ShotVersion).filter(ShotVersion.project_id == project.id).all()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.source == "import" for row in rows))
        self.assertTrue(all(row.number == 1 for row in rows))

        script_route._persist_phase1(self.db, project.id, state)
        rows = self.db.query(ShotVersion).filter(ShotVersion.project_id == project.id).all()
        self.assertEqual(len(rows), 2)


class CompareVersionTests(ShotVersionTestCase):
    def test_compare_is_readonly_and_reports_changed_fields(self) -> None:
        _, shot = self.make_project_and_shot()
        asyncio.run(shot_route.update_shot(shot.id, shot_route.ShotUpdate(dialogue="two", duration=6.5), self.db))
        rows = self.rows(shot.id)
        self.assertEqual(len(rows), 1)

        response = self.client.get(
            f"/api/shot/{shot.id}/versions/compare",
            params={"a": rows[0].id, "b": rows[0].id},
        )
        self.assertEqual(response.status_code, 400, response.text)

        head_snapshot_row = create_version(self.db, shot, "manual_edit")
        self.db.commit()

        response = self.client.get(
            f"/api/shot/{shot.id}/versions/compare",
            params={"a": rows[0].id, "b": head_snapshot_row.id},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("dialogue", payload["changed_fields"])
        self.assertIn("duration", payload["changed_fields"])
        dialogue_diff = next(item for item in payload["diff"] if item["field"] == "dialogue")
        self.assertEqual(dialogue_diff["a"], "one")
        self.assertEqual(dialogue_diff["b"], "two")
        self.assertTrue(dialogue_diff["changed"])

        # 对比是只读的：版本数与当前镜头状态都不变。
        self.assertEqual(len(self.rows(shot.id)), 2)
        self.db.expire_all()
        current = self.db.get(Shot, shot.id)
        self.assertEqual(current.dialogue, "two")
        self.assertEqual(current.version, 2)

    def test_list_and_detail_endpoints(self) -> None:
        _, shot = self.make_project_and_shot()
        asyncio.run(shot_route.update_shot(shot.id, shot_route.ShotUpdate(dialogue="two"), self.db))
        head = create_version(self.db, shot, "manual_edit")
        self.db.commit()

        listing = self.client.get(f"/api/shot/{shot.id}/versions")
        self.assertEqual(listing.status_code, 200, listing.text)
        payload = listing.json()
        self.assertEqual(len(payload["versions"]), 2)
        self.assertEqual(payload["versions"][0]["id"], head.id)
        self.assertEqual(payload["current_version_id"], head.id)

        detail = self.client.get(f"/api/shot/{shot.id}/versions/{head.id}")
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(detail.json()["snapshot"]["dialogue"], "two")

        missing = self.client.get(f"/api/shot/{shot.id}/versions/does-not-exist")
        self.assertEqual(missing.status_code, 404)


class RestoreVersionTests(ShotVersionTestCase):
    def test_restore_appends_new_version_and_restores_previous_state(self) -> None:
        _, shot = self.make_project_and_shot()
        create_version(self.db, shot, "manual_edit")  # v1: dialogue "one"
        asyncio.run(shot_route.update_shot(shot.id, shot_route.ShotUpdate(dialogue="two"), self.db))

        rows = self.rows(shot.id)
        self.assertEqual(len(rows), 1)

        response = self.client.post(f"/api/shot/{shot.id}/versions/{rows[0].id}/restore")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["restored_from_version_id"], rows[0].id)
        self.assertIsNotNone(payload["new_version_id"])

        self.db.expire_all()
        current = self.db.get(Shot, shot.id)
        self.assertEqual(current.dialogue, "one")
        self.assertEqual(current.version, 3)
        self.assertFalse(current.confirmed)

        after = self.rows(shot.id)
        self.assertEqual([row.source for row in after], ["manual_edit", "restore", "restore"])
        # 旧版本仍可查看，且内容未被改写。
        self.assertEqual(parse_snapshot(after[0])["dialogue"], "one")
        detail = self.client.get(f"/api/shot/{shot.id}/versions/{after[0].id}")
        self.assertEqual(detail.status_code, 200)

        listing = self.client.get(f"/api/shot/{shot.id}/versions").json()
        self.assertEqual(listing["current_version_id"], after[-1].id)

    def test_restore_rejects_confirmed_shot(self) -> None:
        _, shot = self.make_project_and_shot(confirmed=True, storyboard_path="story.png", image_path="story.png")
        row = create_version(self.db, shot, "manual_edit")
        self.db.commit()

        response = self.client.post(f"/api/shot/{shot.id}/versions/{row.id}/restore")
        self.assertEqual(response.status_code, 423)
        self.assertEqual(self.rows(shot.id), [row])

    def test_restore_rejects_missing_media_with_explicit_error(self) -> None:
        media_path = _media_file("restored-media.png")
        _, shot = self.make_project_and_shot(image_path=media_path, storyboard_path=media_path)
        row = create_version(self.db, shot, "manual_edit")
        shot.image_path = ""
        shot.storyboard_path = ""
        self.db.commit()
        Path(media_path).unlink()

        response = self.client.post(f"/api/shot/{shot.id}/versions/{row.id}/restore")
        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("媒体文件已缺失", response.json()["detail"])

    def test_restore_rejects_deleted_asset_binding(self) -> None:
        project = Project(id="shot-ver-project", title="版本测试")
        scene = SceneAsset(id="shot-ver-scene", project_id=project.id, name="教室")
        shot = Shot(id="shot-ver-shot", project_id=project.id, sequence=1, version=1, dialogue="one", scene_asset_id="shot-ver-scene")
        self.db.add_all([project, scene, shot])
        self.db.commit()
        row = create_version(self.db, shot, "manual_edit")
        shot.scene_asset_id = ""
        self.db.delete(scene)
        self.db.commit()

        response = self.client.post(f"/api/shot/{shot.id}/versions/{row.id}/restore")
        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("资产已不存在", response.json()["detail"])

    def test_restore_normalizes_transient_states(self) -> None:
        media_path = _media_file("transient-storyboard.png")
        _, shot = self.make_project_and_shot(
            storyboard_path=media_path,
            image_path=media_path,
            status="video_generating",
            storyboard_status="queued",
        )
        row = create_version(self.db, shot, "regenerate", task_id="shot:shot-ver-shot:storyboard")
        shot.status = "failed"
        shot.storyboard_status = "failed"
        self.db.commit()

        response = self.client.post(f"/api/shot/{shot.id}/versions/{row.id}/restore")
        self.assertEqual(response.status_code, 200, response.text)
        self.db.expire_all()
        current = self.db.get(Shot, shot.id)
        # 恢复不能带回 queued / video_generating 这类过程态。
        self.assertEqual(current.storyboard_status, "pending")
        self.assertEqual(current.status, "storyboard_done")


class ImmutabilityAndStorageTests(ShotVersionTestCase):
    def test_shot_versions_reject_in_place_updates(self) -> None:
        _, shot = self.make_project_and_shot()
        row = create_version(self.db, shot, "manual_edit")
        self.db.commit()

        with self.assertRaises(Exception):
            self.db.execute(
                text("UPDATE shot_versions SET source = 'hacked', snapshot = '{}' WHERE id = :id"),
                {"id": row.id},
            )
        self.db.rollback()
        self.db.expire_all()
        intact = self.db.get(ShotVersion, row.id)
        self.assertEqual(intact.source, "manual_edit")
        self.assertEqual(parse_snapshot(intact)["dialogue"], "one")

    def test_storage_cleanup_keeps_media_referenced_by_versions(self) -> None:
        from services.storage_service import StorageService

        project = Project(id="gc-version-project", title="GC")
        shot = Shot(id="gc-version-shot", project_id=project.id, sequence=1, version=1)
        self.db.add_all([project, shot])
        self.db.commit()

        service = StorageService()
        project_dir = service.get_project_dir("gc-version-project")
        shots_dir = project_dir / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)
        retained = shots_dir / "shot_v1.png"
        for version in range(1, 4):
            (shots_dir / f"shot_v{version}.png").write_bytes(b"image")
        shot.image_path = str(retained)
        create_version(self.db, shot, "manual_edit")
        shot.image_path = str(shots_dir / "shot_v3.png")
        self.db.commit()

        old_ttl = settings.PROJECT_TEMP_FILE_TTL_SECONDS
        old_retention = settings.PROJECT_VERSION_RETENTION_COUNT
        settings.PROJECT_TEMP_FILE_TTL_SECONDS = 0
        settings.PROJECT_VERSION_RETENTION_COUNT = 1
        try:
            service.cleanup_project("gc-version-project")
        finally:
            settings.PROJECT_TEMP_FILE_TTL_SECONDS = old_ttl
            settings.PROJECT_VERSION_RETENTION_COUNT = old_retention

        # retention=1 只保留最新 shot_v3，但版本快照引用的 shot_v1 受保护。
        self.assertTrue(retained.exists())
        self.assertTrue((shots_dir / "shot_v3.png").exists())
        self.assertFalse((shots_dir / "shot_v2.png").exists())


class ApiValidationTests(ShotVersionTestCase):
    def test_invalid_identifiers_are_rejected(self) -> None:
        _, shot = self.make_project_and_shot()
        create_version(self.db, shot, "manual_edit")
        self.db.commit()

        long_id = "a" * 129
        self.assertEqual(self.client.get(f"/api/shot/{long_id}/versions").status_code, 400)
        self.assertEqual(
            self.client.get(
                f"/api/shot/{shot.id}/versions/compare",
                params={"a": "../escape", "b": "abc"},
            ).status_code,
            400,
        )
        self.assertEqual(self.client.get("/api/shot/missing-shot/versions").status_code, 404)


if __name__ == "__main__":
    unittest.main()
