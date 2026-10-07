"""内置示例项目（POST /api/project/sample）与首启向导标志的验收测试。

覆盖：
- 示例项目零外部 API 创建：series + 第一集 + 角色/场景/分镜落库，媒体为
  PIL 占位图文件（磁盘上真实存在），标题带「示例项目」标注，``is_sample``
  标记为真；
- 示例项目的数据形状与真实项目一致：GET shots / asset board 可正常读取，
  故事板状态为 done；
- 示例项目可正常删除，且删除不影响向导完成标志；
- 向导标志 GET/PUT /api/settings/onboarding 的持久化读写。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient  # noqa: E402

from db import SessionLocal, init_db  # noqa: E402
from main import app  # noqa: E402
from models import Character, Project, SceneAsset, Shot  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402

init_db()

client = TestClient(app)


class SampleProjectTests(unittest.TestCase):
    def setUp(self):
        self.db = SessionLocal()

    def tearDown(self):
        self.db.rollback()
        self.db.close()

    def _cleanup_projects(self, series_id: str):
        episode_ids = [
            row_id for (row_id,) in self.db.query(Project.id).filter(Project.parent_project_id == series_id).all()
        ]
        for project_id in [series_id, *episode_ids]:
            self.db.query(Shot).filter(Shot.project_id == project_id).delete()
            self.db.query(Character).filter(Character.project_id == project_id).delete()
            self.db.query(SceneAsset).filter(SceneAsset.project_id == project_id).delete()
            self.db.query(Project).filter(Project.id == project_id).delete()
        self.db.commit()

    def test_create_sample_project_zero_external_api(self):
        response = client.post("/api/project/sample")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        series_id = payload["id"]
        self.addCleanup(self._cleanup_projects, series_id)

        self.assertIn("示例项目", payload["title"])
        self.assertTrue(payload["is_sample"])
        episode = payload.get("first_episode") or {}
        self.assertTrue(episode.get("is_sample"))
        self.assertEqual(episode.get("project_type"), "episode")
        self.assertEqual(episode.get("parent_project_id"), series_id)
        self.assertEqual(payload["status"], "storyboard_ready")

        # 角色 / 场景落在 series（资产归属规则与剧本解析一致），三视图与
        # 基准图必须是磁盘上真实存在的占位图文件。
        characters = self.db.query(Character).filter(Character.project_id == series_id).all()
        scenes = self.db.query(SceneAsset).filter(SceneAsset.project_id == series_id).all()
        self.assertGreaterEqual(len(characters), 1)
        self.assertGreaterEqual(len(scenes), 1)
        for character in characters:
            refs = json.loads(character.reference_images or "[]")
            self.assertTrue(refs, "示例角色必须带占位三视图引用")
            self.assertTrue(Path(refs[0]).is_file(), f"角色占位图缺失: {refs[0]}")
            self.assertEqual(character.reference_status, "ready")
            self.assertEqual(character.asset_status, "active")
        for scene in scenes:
            self.assertTrue(scene.baseline_image_path)
            self.assertTrue(Path(scene.baseline_image_path).is_file(), "场景基准图缺失")

        # 分镜直接落库且故事板为占位图完成态。
        episode_id = episode["id"]
        shots = self.db.query(Shot).filter(Shot.project_id == episode_id).order_by(Shot.sequence).all()
        self.assertGreaterEqual(len(shots), 3)
        self.assertLessEqual(len(shots), 5)
        for shot in shots:
            self.assertEqual(shot.storyboard_status, "done")
            self.assertEqual(shot.status, "storyboard_done")
            self.assertTrue(shot.image_path)
            self.assertTrue(Path(shot.image_path).is_file(), f"故事板占位图缺失: {shot.image_path}")
        dialogue_shots = [shot for shot in shots if json.loads(shot.dialogue or "[]")]
        self.assertTrue(dialogue_shots, "示例分镜应包含结构化对白")

    def test_sample_project_readable_via_regular_apis(self):
        created = client.post("/api/project/sample").json()
        series_id = created["id"]
        episode_id = created["first_episode"]["id"]
        self.addCleanup(self._cleanup_projects, series_id)

        shots_response = client.get(f"/api/shot/{episode_id}/shots")
        self.assertEqual(shots_response.status_code, 200, shots_response.text)
        shots = shots_response.json()
        self.assertTrue(isinstance(shots, list) and shots)
        self.assertTrue(all(shot["storyboard_path"] for shot in shots))

        board_response = client.get(f"/api/asset/{episode_id}/board")
        self.assertEqual(board_response.status_code, 200, board_response.text)
        board = board_response.json()
        self.assertTrue(board["characters"])
        self.assertTrue(board["scenes"])

        project_response = client.get(f"/api/project/{episode_id}")
        self.assertEqual(project_response.status_code, 200)
        self.assertTrue(project_response.json()["is_sample"])

    def test_sample_project_deletable_without_touching_onboarding(self):
        client.put("/api/settings/onboarding", json={"completed": True})
        created = client.post("/api/project/sample").json()
        series_id = created["id"]

        delete_response = client.delete(f"/api/project/{series_id}")
        self.assertEqual(delete_response.status_code, 200, delete_response.text)

        remaining = self.db.query(Project).filter((Project.id == series_id) | (Project.parent_project_id == series_id))
        self.assertEqual(remaining.count(), 0, "示例项目（含剧集）必须可被正常删除")

        status = client.get("/api/settings/onboarding").json()
        self.assertTrue(status["completed"], "删除示例项目不得回退向导完成标志")

    def test_user_created_project_is_not_sample(self):
        created = client.post("/api/project", json={"title": "普通项目"}).json()
        self.addCleanup(self._cleanup_projects, created["id"])
        self.assertFalse(created["is_sample"])
        self.assertFalse(created["first_episode"]["is_sample"])

    def test_second_sample_project_coexists_with_distinct_shot_ids(self):
        """两次创建示例项目（如重新运行向导后再开一个）必须都成功。"""
        first = client.post("/api/project/sample")
        self.assertEqual(first.status_code, 200, first.text)
        second = client.post("/api/project/sample")
        self.assertEqual(second.status_code, 200, second.text)
        self.addCleanup(self._cleanup_projects, first.json()["id"])
        self.addCleanup(self._cleanup_projects, second.json()["id"])
        self.assertNotEqual(first.json()["id"], second.json()["id"])

        first_shots = {
            shot["id"] for shot in client.get(f"/api/shot/{first.json()['first_episode']['id']}/shots").json()
        }
        second_shots = {
            shot["id"] for shot in client.get(f"/api/shot/{second.json()['first_episode']['id']}/shots").json()
        }
        self.assertTrue(first_shots)
        self.assertFalse(first_shots & second_shots, "两次示例项目的镜头 ID 不得冲突")


class OnboardingStatusTests(unittest.TestCase):
    def test_default_and_roundtrip(self):
        # 未写入前：completed 为 False（全新环境首启出现向导）。
        status = client.get("/api/settings/onboarding").json()
        self.assertIn("completed", status)
        self.assertIsInstance(status["completed"], bool)

        put_response = client.put("/api/settings/onboarding", json={"completed": True})
        self.assertEqual(put_response.status_code, 200, put_response.text)
        saved = put_response.json()
        self.assertTrue(saved["completed"])
        self.assertTrue(saved["completed_at"])

        reread = client.get("/api/settings/onboarding").json()
        self.assertTrue(reread["completed"], "向导完成标志必须持久化（第二次启动不再出现）")

        # 重置回未完成（供后续测试 / 手动重新触发向导）。
        client.put("/api/settings/onboarding", json={"completed": False})
        reread = client.get("/api/settings/onboarding").json()
        self.assertFalse(reread["completed"])
        self.assertEqual(reread["completed_at"], "")


if __name__ == "__main__":
    unittest.main()
