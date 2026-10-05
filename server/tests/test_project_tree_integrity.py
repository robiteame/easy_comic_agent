"""P2: 项目树数据完整性的回归测试。

覆盖创建、更新、父子关系、循环引用、删除与数据库层约束：
- episode 必须有真实存在的 series 父级，series 不能挂在任何项目下；
- 禁止自引用与未知 project_type；
- 集号不能为负，缺省时自动取下一个可用集号；
- 删除父项目不会留下孤儿 episode；
- 存量库通过触发器补约束，脏数据在启动时被修复。
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402

from api.routes import project as project_route  # noqa: E402
from db import SessionLocal, engine, init_db  # noqa: E402
from db.database import _repair_project_tree  # noqa: E402
from main import app  # noqa: E402
from models import BackgroundJob, Character, Project, SceneAsset, Shot  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402


def _orphan_episode_count(db) -> int:
    """统计父级指向不存在项目的剧集（孤儿）。"""

    return db.execute(
        text(
            "SELECT COUNT(*) FROM projects WHERE parent_project_id IS NOT NULL "
            "AND parent_project_id <> '' AND parent_project_id NOT IN (SELECT id FROM projects)"
        )
    ).scalar()


class ProjectTreeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()
        cls.client = TestClient(app)

    def setUp(self) -> None:
        self.db = SessionLocal()
        self.db.query(BackgroundJob).delete()
        self.db.query(Shot).delete()
        self.db.query(Character).delete()
        self.db.query(SceneAsset).delete()
        self.db.query(Project).delete()
        self.db.commit()

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()

    def create_series(self, title: str = "大项目") -> str:
        response = self.client.post("/api/project", json={"title": title, "project_type": "series"})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["id"]

    def first_episode_id(self, series_id: str) -> str:
        response = self.client.get(f"/api/project/{series_id}/episodes")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()[0]["id"]


class ProjectTreeCreationTests(ProjectTreeTestCase):
    def test_series_creation_also_creates_first_episode(self) -> None:
        series_id = self.create_series()
        self.db.expire_all()
        series = self.db.get(Project, series_id)
        self.assertEqual(series.project_type, "series")
        self.assertEqual(series.parent_project_id or "", "")
        episodes = self.db.query(Project).filter(Project.parent_project_id == series_id).all()
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].project_type, "episode")
        self.assertEqual(episodes[0].episode_number, 1)

    def test_episode_requires_a_parent(self) -> None:
        response = self.client.post("/api/project", json={"title": "孤儿集", "project_type": "episode"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("剧集必须指定所属大项目", response.json()["detail"])
        self.assertEqual(self.db.query(Project).count(), 0)

    def test_episode_parent_must_be_a_series(self) -> None:
        series_id = self.create_series()
        episode_id = self.first_episode_id(series_id)
        response = self.client.post(
            "/api/project",
            json={"title": "孙集", "project_type": "episode", "parent_project_id": episode_id},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("剧集只能挂在系列项目下", response.json()["detail"])
        self.assertEqual(self.db.query(Project).filter(Project.id != series_id).count(), 1)

    def test_episode_parent_must_exist(self) -> None:
        response = self.client.post(
            "/api/project",
            json={"title": "悬空集", "project_type": "episode", "parent_project_id": "missing-parent"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.db.query(Project).count(), 0)

    def test_series_cannot_have_a_parent(self) -> None:
        series_id = self.create_series()
        response = self.client.post(
            "/api/project",
            json={"title": "子系列", "project_type": "series", "parent_project_id": series_id},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("大项目不能挂在其他项目下", response.json()["detail"])

    def test_unknown_project_type_reports_an_explicit_error(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(
                project_route.create_project(
                    project_route.ProjectCreate.model_construct(title="未知类型", project_type="season"),
                    self.db,
                )
            )
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("未知的项目类型", raised.exception.detail)

    def test_episode_number_is_assigned_automatically_and_increments(self) -> None:
        series_id = self.create_series()
        second = self.client.post(
            "/api/project",
            json={"title": "第二集", "project_type": "episode", "parent_project_id": series_id},
        )
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["episode_number"], 2)
        third = self.client.post(
            "/api/project",
            json={"title": "指定集", "project_type": "episode", "parent_project_id": series_id, "episode_number": 7},
        )
        self.assertEqual(third.status_code, 200, third.text)
        self.assertEqual(third.json()["episode_number"], 7)

    def test_parent_pending_deletion_is_rejected(self) -> None:
        series_id = self.create_series()
        self.db.query(Project).filter(Project.id == series_id).update({Project.status: "deleting"})
        self.db.commit()
        response = self.client.post(
            "/api/project",
            json={"title": "删除中的父项目", "project_type": "episode", "parent_project_id": series_id},
        )
        self.assertEqual(response.status_code, 409)


class ProjectTreeUpdateTests(ProjectTreeTestCase):
    def test_project_cannot_become_its_own_parent(self) -> None:
        series_id = self.create_series()
        episode_id = self.first_episode_id(series_id)
        response = self.client.put(f"/api/project/{episode_id}", json={"parent_project_id": episode_id})
        self.assertEqual(response.status_code, 400)
        self.assertIn("不能成为自己的父项目", response.json()["detail"])

    def test_series_with_episodes_cannot_become_an_episode(self) -> None:
        series_id = self.create_series()
        other_series = self.create_series("另一个大项目")
        response = self.client.put(
            f"/api/project/{series_id}",
            json={"project_type": "episode", "parent_project_id": other_series},
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("不能改为剧集类型", response.json()["detail"])
        self.db.expire_all()
        self.assertEqual(self.db.get(Project, series_id).project_type, "series")
        self.assertEqual(_orphan_episode_count(self.db), 0)

    def test_series_cannot_be_moved_under_another_project(self) -> None:
        series_id = self.create_series()
        other_series = self.create_series("另一个大项目")
        response = self.client.put(f"/api/project/{series_id}", json={"parent_project_id": other_series})
        self.assertEqual(response.status_code, 400)
        self.db.expire_all()
        self.assertEqual(self.db.get(Project, series_id).parent_project_id or "", "")

    def test_episode_can_move_to_another_series(self) -> None:
        first = self.create_series("大项目一")
        second = self.create_series("大项目二")
        episode_id = self.first_episode_id(first)
        response = self.client.put(f"/api/project/{episode_id}", json={"parent_project_id": second})
        self.assertEqual(response.status_code, 200, response.text)
        self.db.expire_all()
        self.assertEqual(self.db.get(Project, episode_id).parent_project_id, second)

    def test_episode_cannot_move_under_another_episode(self) -> None:
        series_id = self.create_series()
        episode_id = self.first_episode_id(series_id)
        second = self.client.post(
            "/api/project",
            json={"title": "第二集", "project_type": "episode", "parent_project_id": series_id},
        ).json()["id"]
        response = self.client.put(f"/api/project/{episode_id}", json={"parent_project_id": second})
        self.assertEqual(response.status_code, 400)
        self.db.expire_all()
        self.assertEqual(self.db.get(Project, episode_id).parent_project_id, series_id)

    def test_invalid_parent_identifier_is_rejected(self) -> None:
        series_id = self.create_series()
        episode_id = self.first_episode_id(series_id)
        response = self.client.put(f"/api/project/{episode_id}", json={"parent_project_id": "../escape"})
        self.assertEqual(response.status_code, 422)

    def test_series_episode_number_is_normalized_to_zero(self) -> None:
        series_id = self.create_series()
        response = self.client.put(f"/api/project/{series_id}", json={"episode_number": 5})
        self.assertEqual(response.status_code, 200, response.text)
        self.db.expire_all()
        self.assertEqual(self.db.get(Project, series_id).episode_number, 0)

    def test_episode_update_keeps_it_under_a_series(self) -> None:
        series_id = self.create_series()
        episode_id = self.first_episode_id(series_id)
        response = self.client.put(f"/api/project/{episode_id}", json={"title": "改名", "episode_number": 3})
        self.assertEqual(response.status_code, 200, response.text)
        self.db.expire_all()
        episode = self.db.get(Project, episode_id)
        self.assertEqual(episode.episode_number, 3)
        self.assertEqual(episode.parent_project_id, series_id)
        self.assertEqual(_orphan_episode_count(self.db), 0)


class ProjectTreeDeletionTests(ProjectTreeTestCase):
    def test_deleting_a_series_removes_its_episodes(self) -> None:
        series_id = self.create_series()
        episode_id = self.first_episode_id(series_id)
        second = self.client.post(
            "/api/project",
            json={"title": "第二集", "project_type": "episode", "parent_project_id": series_id},
        ).json()["id"]

        result = asyncio.run(project_route.delete_project(series_id, self.db))

        self.assertEqual(result["status"], "deleted")
        self.assertIn(episode_id, result["deleted_project_ids"])
        self.assertIn(second, result["deleted_project_ids"])
        self.db.expire_all()
        self.assertIsNone(self.db.get(Project, series_id))
        self.assertIsNone(self.db.get(Project, episode_id))
        self.assertEqual(_orphan_episode_count(self.db), 0)

    def test_deleting_an_episode_keeps_the_series(self) -> None:
        series_id = self.create_series()
        episode_id = self.first_episode_id(series_id)

        asyncio.run(project_route.delete_project(episode_id, self.db))

        self.db.expire_all()
        self.assertIsNotNone(self.db.get(Project, series_id))
        self.assertIsNone(self.db.get(Project, episode_id))
        self.assertEqual(_orphan_episode_count(self.db), 0)


class ProjectTreeConstraintTests(ProjectTreeTestCase):
    def test_fresh_database_has_project_tree_checks(self) -> None:
        with engine.connect() as connection:
            ddl = connection.execute(
                text("SELECT sql FROM sqlite_master WHERE type='table' AND name='projects'")
            ).scalar()
        self.assertIn("ck_projects_type", ddl)
        self.assertIn("ck_projects_parent_shape", ddl)
        self.assertIn("ck_projects_no_self_parent", ddl)
        self.assertIn("ck_projects_episode_number", ddl)

    def test_legacy_database_backfills_tree_triggers(self) -> None:
        with engine.connect() as connection:
            triggers = {
                row[0]
                for row in connection.execute(text("SELECT name FROM sqlite_master WHERE type='trigger'")).fetchall()
            }
        self.assertIn("trg_projects_insert_tree", triggers)
        self.assertIn("trg_projects_update_tree", triggers)

    def test_database_rejects_moving_a_series_under_a_parent(self) -> None:
        series_id = self.create_series()
        other_series = self.create_series("另一个大项目")
        with self.assertRaises(IntegrityError):
            self.db.execute(
                text("UPDATE projects SET parent_project_id = :parent WHERE id = :project"),
                {"parent": other_series, "project": series_id},
            )
            self.db.commit()
        self.db.rollback()
        self.db.expire_all()
        self.assertEqual(self.db.get(Project, series_id).parent_project_id or "", "")

    def test_database_rejects_episode_parented_by_an_episode(self) -> None:
        series_id = self.create_series()
        episode_id = self.first_episode_id(series_id)
        # 形状合法（有父级），但父级类型非法，应在 UPDATE 时被触发器拦下。
        self.db.add(Project(id="legacy-child", title="历史孙集", project_type="episode", parent_project_id=series_id))
        self.db.commit()
        with self.assertRaises(IntegrityError):
            self.db.execute(
                text("UPDATE projects SET parent_project_id = :parent WHERE id = 'legacy-child'"),
                {"parent": episode_id},
            )
            self.db.commit()
        self.db.rollback()

    def test_status_updates_on_valid_rows_still_work(self) -> None:
        series_id = self.create_series()
        episode_id = self.first_episode_id(series_id)
        self.db.execute(
            text("UPDATE projects SET status = 'assets_ready' WHERE id IN (:series, :episode)"),
            {"series": series_id, "episode": episode_id},
        )
        self.db.commit()
        self.db.expire_all()
        self.assertEqual(self.db.get(Project, series_id).status, "assets_ready")
        self.assertEqual(self.db.get(Project, episode_id).status, "assets_ready")


class ProjectTreeRepairTests(ProjectTreeTestCase):
    def _insert_legacy_rows(self, statements: list[tuple[str, dict]]) -> None:
        """模拟升级前的历史库：临时去掉触发器/CHECK 约束后直接写脏数据。"""

        with engine.begin() as connection:
            connection.execute(text("DROP TRIGGER IF EXISTS trg_projects_insert_tree"))
            connection.execute(text("DROP TRIGGER IF EXISTS trg_projects_update_tree"))
        with engine.begin() as connection:
            connection.execute(text("PRAGMA ignore_check_constraints=ON"))
            try:
                for sql, params in statements:
                    connection.execute(text(sql), params)
            finally:
                connection.execute(text("PRAGMA ignore_check_constraints=OFF"))
        self.db.expire_all()

    def test_orphan_episode_is_promoted_to_a_series(self) -> None:
        # 直接落库绕过 API 校验，模拟历史脏数据。
        self.db.add(Project(id="orphan-episode", title="孤儿集", project_type="episode", parent_project_id="gone"))
        self.db.commit()

        repaired = _repair_project_tree()

        self.assertEqual(repaired, 1)
        self.db.expire_all()
        orphan = self.db.get(Project, "orphan-episode")
        self.assertEqual(orphan.project_type, "series")
        self.assertEqual(orphan.parent_project_id or "", "")
        self.assertEqual(_orphan_episode_count(self.db), 0)
        # 幂等：再次执行不会重复修复。
        self.assertEqual(_repair_project_tree(), 0)

    def test_negative_episode_number_is_clamped(self) -> None:
        series_id = self.create_series()
        try:
            self._insert_legacy_rows(
                [
                    (
                        "INSERT INTO projects (id, title, project_type, parent_project_id, episode_number) "
                        "VALUES ('negative-episode', '负集号', 'episode', :parent, -4)",
                        {"parent": series_id},
                    )
                ]
            )

            repaired = _repair_project_tree()

            self.assertEqual(repaired, 1)
            self.db.expire_all()
            self.assertEqual(self.db.get(Project, "negative-episode").episode_number, 0)
            self.assertEqual(self.db.get(Project, "negative-episode").project_type, "episode")
        finally:
            # 恢复触发器，避免影响后续用例。
            init_db()

    def test_unknown_project_type_is_normalized(self) -> None:
        try:
            self._insert_legacy_rows(
                [
                    (
                        "INSERT INTO projects (id, title, project_type, parent_project_id, episode_number) "
                        "VALUES ('legacy-type', '未知类型', 'season', '', 0)",
                        {},
                    )
                ]
            )

            repaired = _repair_project_tree()

            self.assertGreaterEqual(repaired, 1)
            self.db.expire_all()
            self.assertEqual(self.db.get(Project, "legacy-type").project_type, "series")
        finally:
            init_db()

    def test_startup_migration_repairs_and_restores_constraints(self) -> None:
        """模拟完整启动流程：脏数据被修复，触发器同时被补齐。"""

        try:
            self._insert_legacy_rows(
                [
                    (
                        "INSERT INTO projects (id, title, project_type, parent_project_id, episode_number) "
                        "VALUES ('startup-orphan', '启动孤儿集', 'episode', 'gone', -2)",
                        {},
                    )
                ]
            )

            init_db()

            self.db.expire_all()
            orphan = self.db.get(Project, "startup-orphan")
            self.assertEqual(orphan.project_type, "series")
            self.assertEqual(orphan.episode_number, 0)
            self.assertEqual(_orphan_episode_count(self.db), 0)
            with engine.connect() as connection:
                triggers = {
                    row[0]
                    for row in connection.execute(
                        text("SELECT name FROM sqlite_master WHERE type='trigger'")
                    ).fetchall()
                }
            self.assertIn("trg_projects_insert_tree", triggers)
            self.assertIn("trg_projects_update_tree", triggers)
        finally:
            init_db()


if __name__ == "__main__":
    unittest.main()
