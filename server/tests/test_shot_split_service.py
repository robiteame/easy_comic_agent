"""Regression tests for transaction-scoped persisted shot splitting."""

from __future__ import annotations

import json
import unittest

from db import SessionLocal, init_db  # noqa: E402
from models import Character, Project, SceneAsset, Shot, ShotVersion  # noqa: E402
from services.shot_split_service import (  # noqa: E402
    ShotSplitUnsafe,
    ShotSplitVersionConflict,
    persist_split_shot,
)
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


class PersistedShotSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    def setUp(self) -> None:
        self.db = SessionLocal()
        self.db.query(ShotVersion).delete()
        self.db.query(Shot).delete()
        self.db.query(Character).delete()
        self.db.query(SceneAsset).delete()
        self.db.query(Project).delete()
        self.db.commit()
        self.project = Project(id="split-project", title="split")
        self.source = Shot(
            id="split-project_shot_0001",
            project_id=self.project.id,
            sequence=1,
            version=3,
            duration=6.0,
            character_action="走到窗边然后回头",
            dialogue="甲：你好。乙：再见。",
            image_path="/tmp/old.png",
            storyboard_path="/tmp/old-storyboard.png",
            audio_path="/tmp/old.wav",
            video_path="/tmp/old.mp4",
            confirmed=True,
            status="video_done",
            storyboard_status="approved",
            continuity_profile=json.dumps({"timing": {"audio_mode": "tts"}}),
        )
        self.later = Shot(
            id="split-project_shot_0002",
            project_id=self.project.id,
            sequence=2,
            version=1,
            duration=3.0,
        )
        self.db.add_all([self.project, self.source, self.later])
        self.db.commit()

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()

    def test_split_retains_first_id_shifts_timeline_and_resets_media(self) -> None:
        result = persist_split_shot(
            self.db,
            project_id=self.project.id,
            shot_id=self.source.id,
            expected_version=3,
            parts=2,
            reason="对白过长",
            operation_id="split-op-1",
        )
        self.db.commit()
        rows = self.db.query(Shot).filter(Shot.project_id == self.project.id).order_by(Shot.sequence).all()
        self.assertEqual([row.id for row in rows], [self.source.id, f"{self.source.id}_part_02", self.later.id])
        self.assertEqual([row.sequence for row in rows], [1, 2, 3])
        self.assertEqual(result["shot_ids"], [self.source.id, f"{self.source.id}_part_02"])
        self.assertEqual(rows[0].version, 4)
        self.assertEqual(rows[1].version, 1)
        for row in rows[:2]:
            self.assertFalse(row.confirmed)
            self.assertEqual(row.status, "pending")
            self.assertEqual(row.storyboard_status, "pending")
            self.assertTrue(row.media_stale)
            self.assertFalse(row.video_path)
            self.assertFalse(row.audio_path)
        self.assertEqual(
            rows[0].continuity_profile and json.loads(rows[0].continuity_profile)["split_recovery"]["operation_id"],
            "split-op-1",
        )
        self.assertGreaterEqual(self.db.query(ShotVersion).filter(ShotVersion.shot_id == self.source.id).count(), 2)
        self.assertEqual(self.project.status, "assets_ready")

    def test_same_operation_is_idempotent_and_stale_version_is_rejected(self) -> None:
        first = persist_split_shot(
            self.db,
            project_id=self.project.id,
            shot_id=self.source.id,
            expected_version=3,
            parts=2,
            operation_id="split-op-2",
        )
        before = self.db.query(Shot).count()
        again = persist_split_shot(
            self.db,
            project_id=self.project.id,
            shot_id=self.source.id,
            expected_version=4,
            parts=2,
            operation_id="split-op-2",
        )
        self.assertEqual(again["status"], "already_applied")
        self.assertEqual(again["shot_ids"], first["shot_ids"])
        self.assertEqual(self.db.query(Shot).count(), before)
        with self.assertRaises(ShotSplitVersionConflict):
            persist_split_shot(
                self.db,
                project_id=self.project.id,
                shot_id=self.later.id,
                expected_version=99,
                parts=2,
            )

    def test_complex_planner_expansion_is_idempotent(self) -> None:
        self.source.character_action = "准备起身，走到窗边，打开窗户，回头看向门口"
        self.db.commit()
        first = persist_split_shot(
            self.db,
            project_id=self.project.id,
            shot_id=self.source.id,
            expected_version=3,
            parts=2,
            operation_id="split-op-expanded",
        )
        self.assertGreater(first["parts"], 2)
        self.assertEqual(first["parts"], len(first["shot_ids"]))
        again = persist_split_shot(
            self.db,
            project_id=self.project.id,
            shot_id=self.source.id,
            expected_version=4,
            parts=2,
            operation_id="split-op-expanded",
        )
        self.assertEqual(again["status"], "already_applied")
        self.assertEqual(again["shot_ids"], first["shot_ids"])
        metadata = json.loads(self.db.query(Shot).filter(Shot.id == self.source.id).one().continuity_profile)[
            "split_recovery"
        ]
        self.assertEqual(metadata["requested_parts"], 2)
        self.assertEqual(metadata["parts"], first["parts"])

    def test_active_generation_and_id_collision_are_rejected(self) -> None:
        self.source.status = "video_generating"
        self.db.commit()
        with self.assertRaises(ShotSplitUnsafe):
            persist_split_shot(
                self.db,
                project_id=self.project.id,
                shot_id=self.source.id,
                expected_version=3,
                parts=2,
            )


if __name__ == "__main__":
    unittest.main()
