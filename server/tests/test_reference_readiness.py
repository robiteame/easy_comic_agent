"""一致性参考失败、手动降级与下游版本追踪的验收测试。"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT  # noqa: E402

from PIL import Image  # noqa: E402

from agent import graph  # noqa: E402
from api.routes import script as script_route  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import BackgroundJob, Character, Project, SceneAsset, Shot  # noqa: E402
from services.invalidation_service import invalidate_asset_consumers  # noqa: E402
from services.job_dto import job_dto  # noqa: E402
from services.reference_readiness_service import (  # noqa: E402
    accept_degraded_reference,
    build_manifest_for_shot,
    ensure_generation_gate,
    mark_reference_success,
    refresh_project_reference_state,
)


def _image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 960), (190, 180, 170)).save(path)
    return str(path)


class ReferenceReadinessTestCase(unittest.TestCase):
    prefix = "reference_readiness_tests"

    def setUp(self) -> None:
        init_db()
        self.db = SessionLocal()
        self.project_ids: list[str] = []

    def tearDown(self) -> None:
        self.db.rollback()
        for project_id in self.project_ids:
            self.db.query(Shot).filter(Shot.project_id == project_id).delete()
        self.db.query(Character).filter(Character.project_id.in_(self.project_ids)).delete()
        self.db.query(SceneAsset).filter(SceneAsset.project_id.in_(self.project_ids)).delete()
        self.db.query(BackgroundJob).filter(BackgroundJob.project_id.in_(self.project_ids)).delete()
        self.db.query(Project).filter(Project.id.in_(self.project_ids)).delete()
        self.db.commit()
        self.db.close()

    def _project(self, name: str) -> str:
        project_id = f"{self.prefix}_{name}"
        self.project_ids.append(project_id)
        self.db.add(Project(id=project_id, title=name, style="anime", project_type="series"))
        self.db.commit()
        return project_id

    def _shot(self, project_id: str, sequence: int, *, character_id: str = "", scene_id: str = "") -> Shot:
        shot = Shot(
            id=f"{project_id}_shot_{sequence}",
            project_id=project_id,
            sequence=sequence,
            character_asset_ids=json.dumps([character_id] if character_id else []),
            scene_asset_id=scene_id,
            storyboard_path=_image(TEST_ROOT / "output" / project_id / f"shot_{sequence}.png"),
            status="storyboard_done",
            storyboard_status="done",
        )
        self.db.add(shot)
        self.db.commit()
        return shot

    def test_generation_failure_is_recorded_and_not_silently_continued(self) -> None:
        state = {"characters": [{"id": "char_failed", "name": "林晚", "visual_prompt": "blue coat"}], "style": "anime"}
        with patch.object(
            script_route.image_service,
            "generate_character_reference",
            new=AsyncMock(side_effect=RuntimeError("provider exploded")),
        ):
            asyncio.run(script_route._ensure_character_reference_images(self.prefix, state))

        item = state["characters"][0]
        self.assertEqual(item["reference_status"], "failed")
        self.assertEqual(item["reference_images"], [])
        self.assertEqual(item["reference_retry_count"], 1)
        self.assertIn("provider exploded", item["reference_failure_reason"])
        self.assertTrue(item["reference_error_id"])

    def test_auto_mode_never_approves_storyboard_with_failed_reference(self) -> None:
        project_id = self._project("auto_reference_gate")
        character = Character(
            id=f"{project_id}_char",
            project_id=project_id,
            name="林晚",
            reference_images="[]",
            reference_status="failed",
            reference_failure_reason="三视图生成失败",
            reference_error_id="deadbeef",
        )
        scene = SceneAsset(
            id=f"{project_id}_scene",
            project_id=project_id,
            name="教室",
            baseline_image_path=_image(TEST_ROOT / "output" / project_id / "scene.png"),
            reference_images=json.dumps([str(TEST_ROOT / "output" / project_id / "scene.png")]),
            reference_status="ready",
        )
        self.db.add_all([character, scene])
        self.db.commit()
        self._shot(project_id, 1, character_id=character.id, scene_id=scene.id)
        refresh_project_reference_state(self.db, project_id)

        result = asyncio.run(graph._auto_approve_storyboard({"project_id": project_id}))
        self.assertTrue(result.get("errors"))
        self.assertEqual(result.get("current_step"), "aborted")
        self.db.expire_all()
        shot = self.db.query(Shot).filter(Shot.project_id == project_id).first()
        self.assertFalse(shot.confirmed)
        self.assertNotEqual(shot.status, "storyboard_approved")

    def test_manual_degraded_state_and_task_report_show_impact_range(self) -> None:
        project_id = self._project("manual_degraded_report")
        character = Character(
            id=f"{project_id}_char",
            project_id=project_id,
            name="林晚",
            reference_images="[]",
            reference_status="failed",
            reference_failure_reason="三视图生成失败",
            reference_error_id="cafebabe",
        )
        scene = SceneAsset(
            id=f"{project_id}_scene",
            project_id=project_id,
            name="天台",
            baseline_image_path=_image(TEST_ROOT / "output" / project_id / "scene.png"),
            reference_images=json.dumps([str(TEST_ROOT / "output" / project_id / "scene.png")]),
            reference_status="ready",
        )
        self.db.add_all([character, scene])
        self.db.commit()
        self._shot(project_id, 1, character_id=character.id, scene_id=scene.id)
        self._shot(project_id, 2, character_id=character.id, scene_id=scene.id)
        refresh_project_reference_state(self.db, project_id)

        item = accept_degraded_reference(self.db, "character", character.id, reason="用户明确确认降级")["project_report"]
        self.assertEqual(item["status"], "degraded")
        self.assertEqual(item["shot_range"], "镜头 1-2")
        self.assertEqual(item["affected_shot_count"], 2)
        self.db.expire_all()
        project = self.db.get(Project, project_id)
        self.assertEqual(project.status, "degraded")
        for shot in self.db.query(Shot).filter(Shot.project_id == project_id).all():
            self.assertEqual(shot.consistency_status, "degraded")
            self.assertIn("林晚", shot.consistency_report)

        job = BackgroundJob(
            id=f"{project_id}_job",
            idempotency_key=f"project:{project_id}:storyboard",
            scope=f"project:{project_id}",
            project_id=project_id,
            job_type="storyboard",
            status="completed",
            report=json.dumps(item, ensure_ascii=False),
        )
        self.db.add(job)
        self.db.commit()
        dto = job_dto(job, include_report=True)
        self.assertEqual(dto["report"]["shot_range"], "镜头 1-2")
        self.assertEqual(dto["report"]["affected_shot_ids"], [f"{project_id}_shot_1", f"{project_id}_shot_2"])

    def test_reference_version_update_marks_downstream_media_stale(self) -> None:
        project_id = self._project("reference_manifest_stale")
        old_path = _image(TEST_ROOT / "output" / project_id / "old_ref.png")
        character = Character(
            id=f"{project_id}_char",
            project_id=project_id,
            name="林晚",
            reference_images=json.dumps([old_path]),
            reference_status="ready",
            reference_version=1,
        )
        self.db.add(character)
        self.db.commit()
        shot = self._shot(project_id, 1, character_id=character.id)
        old_manifest = build_manifest_for_shot(self.db, shot)
        self.assertEqual(old_manifest[0]["version"], 1)
        shot.storyboard_reference_manifest = json.dumps(old_manifest, ensure_ascii=False)
        self.db.commit()

        new_path = _image(TEST_ROOT / "output" / project_id / "new_ref.png")
        mark_reference_success(self.db, "character", character, new_path)
        invalidate_asset_consumers(self.db, project_id, character_id=character.id)
        self.db.commit()

        self.db.expire_all()
        shot = self.db.get(Shot, shot.id)
        self.assertTrue(shot.media_stale)
        self.assertEqual(json.loads(shot.storyboard_reference_manifest or "[]"), old_manifest)
        self.assertEqual(character.reference_version, 2)
