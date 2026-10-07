"""风格切换使旧资产失效的验收测试。

- 同风格重新解析：资产保持 active、参考图保留；
- 切换风格：旧资产标记 stale、引用清空（文件不删除）；
- 历史资产指纹为空（来源未知）：不猜测旧风格，视为不一致 → stale；
- 本次运行内新生成的参考图（当前风格）→ 置回 active；
- stale 资产在生成上下文中不再提供参考图。
"""

from __future__ import annotations

import hashlib
import unittest

from api.routes import script as script_route  # noqa: E402
from api.routes.shot import _characters, _scenes  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import Character, Project, SceneAsset  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


def _fingerprint(style: str) -> str:
    return hashlib.sha256(style.encode()).hexdigest()[:16]


def _seed_project(db, project_id: str = "style_inval_project") -> str:
    project = Project(id=project_id, title="t", style="anime", project_type="series")
    db.add(project)
    return project_id


class AssetStyleInvalidationTests(unittest.TestCase):
    def setUp(self) -> None:
        init_db()
        self.db = SessionLocal()
        for model in (Character, SceneAsset, Project):
            rows = self.db.query(model).filter(model.id.like("style_inval%") if hasattr(model, "id") else True)
            for row in rows:
                if hasattr(row, "project_id") and str(row.project_id).startswith("style_inval"):
                    self.db.delete(row)
        self.db.commit()

    def tearDown(self) -> None:
        self.db.rollback()
        for model in (Character, SceneAsset):
            for row in self.db.query(model).filter(model.project_id == "style_inval_project"):
                self.db.delete(row)
        for row in self.db.query(Project).filter(Project.id.like("style_inval%")):
            self.db.delete(row)
        self.db.commit()
        self.db.close()

    def test_style_switch_marks_existing_assets_stale(self) -> None:
        project_id = _seed_project(self.db)
        # 旧资产：anime 指纹 + 参考图。
        character = Character(
            id="style_inval_char_0001",
            project_id=project_id,
            name="林晚",
            style_fingerprint=_fingerprint("anime"),
            asset_status="active",
            reference_images='["/old/three_view.png"]',
        )
        scene = SceneAsset(
            id="style_inval_scene_0001",
            project_id=project_id,
            name="天台",
            style_fingerprint=_fingerprint("anime"),
            asset_status="active",
            baseline_image_path="/old/baseline.png",
            reference_images='["/old/baseline.png"]',
        )
        self.db.add_all([character, scene])
        self.db.commit()

        # 重新解析（本次未生成新参考图），风格为 realistic。
        script_route._upsert_characters(
            self.db, project_id, [{"id": character.id, "name": "林晚", "reference_images": []}], "realistic"
        )
        script_route._upsert_scenes(
            self.db, project_id, [{"id": scene.id, "name": "天台", "scene_group_key": "rooftop-night"}], "realistic"
        )
        self.db.commit()

        self.db.refresh(character)
        self.db.refresh(scene)
        self.assertEqual(character.asset_status, "stale")
        self.assertEqual(character.reference_images, "[]")
        self.assertEqual(scene.asset_status, "stale")
        self.assertEqual(scene.baseline_image_path, "")
        self.assertEqual(scene.reference_images, "[]")
        # 旧文件不被删除（这里只验证引用被清空，删除从未发生）。
        self.assertEqual(character.style_fingerprint, _fingerprint("realistic"))

    def test_unknown_legacy_fingerprint_is_treated_as_mismatch(self) -> None:
        project_id = _seed_project(self.db, "style_inval_project2")
        character = Character(
            id="style_inval_char_0002",
            project_id=project_id,
            name="陈默",
            style_fingerprint="",  # 历史遗留：来源未知
            asset_status="active",
            reference_images='["/legacy/three_view.png"]',
        )
        self.db.add(character)
        self.db.commit()

        script_route._upsert_characters(
            self.db, project_id, [{"id": character.id, "name": "陈默", "reference_images": []}], "realistic"
        )
        self.db.commit()
        self.db.refresh(character)
        self.assertEqual(character.asset_status, "stale")
        self.assertEqual(character.reference_images, "[]")

    def test_fresh_references_mark_asset_active_again(self) -> None:
        project_id = _seed_project(self.db, "style_inval_project3")
        character = Character(
            id="style_inval_char_0003",
            project_id=project_id,
            name="林晚",
            style_fingerprint=_fingerprint("anime"),
            asset_status="active",
            reference_images='["/old/three_view.png"]',
        )
        self.db.add(character)
        self.db.commit()

        script_route._upsert_characters(
            self.db,
            project_id,
            [{"id": character.id, "name": "林晚", "reference_images": ["/new/three_view.png"]}],
            "realistic",
        )
        self.db.commit()
        self.db.refresh(character)
        self.assertEqual(character.asset_status, "active")
        self.assertIn("/new/three_view.png", character.reference_images)
        self.assertNotIn("/old/three_view.png", character.reference_images)

    def test_same_style_reparse_keeps_references(self) -> None:
        project_id = _seed_project(self.db, "style_inval_project4")
        character = Character(
            id="style_inval_char_0004",
            project_id=project_id,
            name="林晚",
            style_fingerprint=_fingerprint("realistic"),
            asset_status="active",
            reference_images='["/keep/three_view.png"]',
        )
        self.db.add(character)
        self.db.commit()

        script_route._upsert_characters(
            self.db,
            project_id,
            [{"id": character.id, "name": "林晚", "reference_images": ["/keep/three_view.png"]}],
            "realistic",
        )
        self.db.commit()
        self.db.refresh(character)
        self.assertEqual(character.asset_status, "active")
        self.assertIn("/keep/three_view.png", character.reference_images)

    def test_stale_assets_do_not_feed_generation_context(self) -> None:
        project_id = _seed_project(self.db, "style_inval_project5")
        self.db.add(
            Character(
                id="style_inval_char_0005",
                project_id=project_id,
                name="林晚",
                style_fingerprint=_fingerprint("anime"),
                asset_status="stale",
                reference_images='["/old/three_view.png"]',
            )
        )
        self.db.add(
            SceneAsset(
                id="style_inval_scene_0005",
                project_id=project_id,
                name="天台",
                style_fingerprint=_fingerprint("anime"),
                asset_status="stale",
                baseline_image_path="/old/baseline.png",
                reference_images='["/old/baseline.png"]',
            )
        )
        self.db.commit()

        characters = _characters(self.db, project_id)
        scenes = _scenes(self.db, project_id)
        self.assertEqual(characters[0]["reference_images"], [])
        self.assertEqual(characters[0]["asset_status"], "stale")
        self.assertEqual(scenes["style_inval_scene_0005"]["reference_images"], [])
        self.assertEqual(scenes["style_inval_scene_0005"]["baseline_image_path"], "")


if __name__ == "__main__":
    unittest.main()
