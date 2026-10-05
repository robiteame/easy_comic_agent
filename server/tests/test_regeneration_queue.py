from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes import shot as shot_route  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import BackgroundJob, Project, Shot  # noqa: E402
from services import (
    regeneration_queue,  # noqa: E402
    task_registry,  # noqa: E402
)
from services.shot_version_service import create_version  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402


class SelectiveRegenerationQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.db = SessionLocal()
        self.db.query(BackgroundJob).delete()
        self.db.query(Shot).delete()
        self.db.query(Project).delete()
        self.db.commit()
        self.db.add(Project(id="queue-project", title="Queue Project"))
        for index in range(1, 11):
            self.db.add(Shot(id=f"queue-shot-{index}", project_id="queue-project", sequence=index, version=1))
        self.db.commit()

    def tearDown(self):
        self.db.rollback()
        self.db.close()

    def _submit(self, shot_ids, stages=("storyboard",), **kwargs):
        with patch.object(regeneration_queue.asyncio, "create_task") as create_task:
            create_task.side_effect = lambda coroutine: (coroutine.close(), MagicMock())[1]
            create_task.return_value.add_done_callback = lambda callback: None
            return regeneration_queue.submit(
                self.db,
                "queue-project",
                list(shot_ids),
                list(stages),
                **kwargs,
            )

    def test_only_selected_shots_are_queued(self):
        result = self._submit(["queue-shot-2", "queue-shot-7"], stages=("storyboard", "video"), concurrency=2)
        rows = self.db.query(BackgroundJob).filter(BackgroundJob.queue_batch_id == result.batch_id).all()
        self.assertEqual({row.queue_shot_id for row in rows}, {"queue-shot-2", "queue-shot-7"})
        self.assertEqual({row.queue_stage for row in rows}, {"storyboard", "video"})
        self.assertEqual({row.queue_concurrency for row in rows}, {2})
        self.assertEqual({row.queue_priority for row in rows}, {0})
        self.assertEqual(self.db.query(Shot).filter(Shot.storyboard_status == "pending").count(), 10)
        self.assertTrue(all(row.queue_dependency_ids != "[]" for row in rows if row.queue_stage == "video"))

    def test_unselected_shots_keep_version_and_media_state(self):
        untouched = self.db.query(Shot).filter(Shot.id == "queue-shot-10").one()
        untouched.version = 4
        untouched.image_path = "projects/queue-shot-10/v4.png"
        untouched.video_path = "projects/queue-shot-10/v4.mp4"
        untouched.audio_path = "projects/queue-shot-10/v4.wav"
        self.db.commit()
        result = self._submit(["queue-shot-2", "queue-shot-7"], stages=("storyboard",))
        self.assertNotEqual(result.batch_id, "")
        self.db.expire_all()
        untouched_after = self.db.query(Shot).filter(Shot.id == untouched.id).one()
        self.assertEqual(untouched_after.version, 4)
        self.assertEqual(untouched_after.image_path, "projects/queue-shot-10/v4.png")
        self.assertEqual(untouched_after.video_path, "projects/queue-shot-10/v4.mp4")
        self.assertEqual(untouched_after.audio_path, "projects/queue-shot-10/v4.wav")

    def test_duplicate_active_submission_is_merged(self):
        first = self._submit(["queue-shot-2"], stages=("video",))
        second = self._submit(["queue-shot-2"], stages=("video",))
        self.assertTrue(any(item["deduplicated"] for item in second.items))
        self.assertEqual(second.batch_id, first.batch_id)
        self.assertEqual(regeneration_queue.batch_snapshot(self.db, second.batch_id)["summary"]["total"], 1)
        self.assertEqual(
            self.db.query(BackgroundJob)
            .filter(BackgroundJob.queue_shot_id == "queue-shot-2", BackgroundJob.queue_stage == "video")
            .count(),
            1,
        )
        self.assertNotEqual(first.batch_id, "")

    def test_confirmed_shot_requires_explicit_force(self):
        shot = self.db.query(Shot).filter(Shot.id == "queue-shot-2").one()
        shot.confirmed = True
        self.db.commit()
        blocked = self._submit([shot.id], stages=("storyboard",))
        self.assertEqual(len(blocked.items), 0)
        self.assertEqual(blocked.blocked[0]["shot_id"], shot.id)
        forced = self._submit([shot.id], stages=("storyboard",), force_confirmed=True)
        self.assertEqual(len(forced.items), 1)

    def test_retry_preserves_queue_execution_options(self):
        shot = self.db.query(Shot).filter(Shot.id == "queue-shot-2").one()
        shot.confirmed = True
        self.db.commit()
        first = self._submit(
            [shot.id],
            stages=("storyboard", "video"),
            priority=5,
            concurrency=3,
            reuse_audio=True,
            resume_missing=True,
            force_confirmed=True,
        )
        self.db.query(BackgroundJob).filter(BackgroundJob.queue_batch_id == first.batch_id).update({"status": "failed"})
        self.db.commit()
        with patch.object(regeneration_queue.asyncio, "create_task") as create_task:
            create_task.side_effect = lambda coroutine: (coroutine.close(), MagicMock())[1]
            create_task.return_value.add_done_callback = lambda callback: None
            retried = regeneration_queue.retry(self.db, first.batch_id)
        rows = self.db.query(BackgroundJob).filter(BackgroundJob.queue_batch_id == retried["batch_id"]).all()
        self.assertEqual({row.queue_priority for row in rows}, {5})
        self.assertEqual({row.queue_concurrency for row in rows}, {3})
        self.assertTrue(all(row.queue_reuse_audio for row in rows))
        self.assertTrue(all(row.queue_resume_missing for row in rows))
        self.assertTrue(all(row.queue_force_confirmed for row in rows))

    def test_audio_stage_forwards_reuse_audio(self):
        captured = {}

        async def fake_audio(shot_id, data, db):
            captured["shot_id"] = shot_id
            captured["reuse_existing"] = data.reuse_existing
            return {"id": shot_id, "skipped": True}

        with patch.object(shot_route, "generate_shot_audio", fake_audio):
            submission = self._submit(["queue-shot-2"], stages=("audio",), reuse_audio=True)
            asyncio.run(regeneration_queue._run_item(submission.items[0]["id"]))

        self.assertEqual(captured, {"shot_id": "queue-shot-2", "reuse_existing": True})
        snapshot = regeneration_queue.batch_snapshot(self.db, submission.batch_id)
        self.assertEqual(snapshot["summary"]["completed"], 1)

    def test_version_snapshot_is_restored_once_for_multi_stage_batch(self):
        shot = self.db.query(Shot).filter(Shot.id == "queue-shot-2").one()
        shot.dialogue = "历史对白"
        create_version(self.db, shot, "manual_edit")
        shot.dialogue = "当前对白"
        shot.version = 2
        self.db.commit()
        seen = []

        async def fake_storyboard(shot_id, data, db):
            seen.append(("storyboard", db.get(Shot, shot_id).dialogue))
            current = db.get(Shot, shot_id)
            current.dialogue = "新故事板对白"
            db.commit()
            return {"id": shot_id, "skipped": True}

        async def fake_video(shot_id, data, db):
            seen.append(("video", db.get(Shot, shot_id).dialogue))
            return {"id": shot_id, "skipped": True}

        with patch.object(regeneration_queue.asyncio, "create_task") as create_task:
            create_task.side_effect = lambda coroutine: (coroutine.close(), MagicMock())[1]
            create_task.return_value.add_done_callback = lambda callback: None
            submission = regeneration_queue.submit(
                self.db,
                "queue-project",
                [shot.id],
                ["storyboard", "video"],
                force_confirmed=True,
                version_map={shot.id: 1},
            )
        with (
            patch.object(shot_route, "regenerate_shot", fake_storyboard),
            patch.object(shot_route, "generate_shot_video", fake_video),
        ):
            asyncio.run(regeneration_queue._run_item(submission.items[0]["id"]))
            video_id = next(item["id"] for item in submission.items if item["stage"] == "video")
            asyncio.run(regeneration_queue._run_item(video_id))

        # 恢复后的对白是结构化 JSON（旧字符串迁移为单条），语义仍是历史台词。
        self.assertEqual([stage for stage, _ in seen], ["storyboard", "video"])
        self.assertEqual(json.loads(seen[0][1])[0]["line"], "历史对白")
        self.assertEqual(seen[1][1], "新故事板对白")

    def test_cancel_batch_only_cancels_its_queue_items(self):
        result = self._submit(["queue-shot-2", "queue-shot-7"], stages=("audio",))
        snapshot = regeneration_queue.cancel(self.db, result.batch_id)
        self.assertEqual(snapshot["summary"]["cancelled"], 2)
        self.assertEqual(self.db.query(BackgroundJob).filter(BackgroundJob.status == "queued").count(), 0)

    def test_pause_and_resume_batch_updates_queue_state(self):
        result = self._submit(["queue-shot-2"], stages=("audio",))
        paused = regeneration_queue.pause(self.db, result.batch_id)
        self.assertTrue(paused["paused"])
        resumed = regeneration_queue.resume(self.db, result.batch_id)
        self.assertFalse(resumed["paused"])
        regeneration_queue.cancel(self.db, result.batch_id)

    def test_scheduler_runs_stage_dependencies_and_keeps_batch_view_deduplicated(self):
        async def fake_stage(stage, shot_id, db):
            key = f"shot:{shot_id}:{stage}"
            job_type = "shot_image" if stage == "storyboard" else "shot_video"
            claim = task_registry.claim_job(
                key,
                f"shot:{shot_id}",
                version=1,
                job_type=job_type,
                project_id="queue-project",
                current_step=stage,
            )
            self.assertTrue(claim.claimed, claim.message)

            async def worker():
                await asyncio.sleep(0.01)

            task_registry.start(key, worker())
            return {"id": shot_id}

        async def run():
            with (
                patch.object(
                    shot_route, "regenerate_shot", lambda shot_id, data, db: fake_stage("storyboard", shot_id, db)
                ),
                patch.object(
                    shot_route, "generate_shot_video", lambda shot_id, data, db: fake_stage("video", shot_id, db)
                ),
            ):
                submission = regeneration_queue.submit(
                    self.db,
                    "queue-project",
                    ["queue-shot-2"],
                    ["storyboard", "video"],
                    concurrency=1,
                    force_confirmed=True,
                )
                for _ in range(120):
                    await asyncio.sleep(0.02)
                    snapshot = regeneration_queue.batch_snapshot(self.db, submission.batch_id)
                    if snapshot["summary"]["completed"] == 2:
                        return snapshot
                return regeneration_queue.batch_snapshot(self.db, submission.batch_id)

        snapshot = asyncio.run(run())
        self.assertEqual(snapshot["summary"]["total"], 2)
        self.assertEqual(snapshot["summary"]["completed"], 2)
        self.assertEqual([item["stage"] for item in snapshot["items"]], ["storyboard", "video"])
        # task_registry 会保留已归档 attempt；它们不能让已完成批次继续被视为活动，
        # 也不能阻止批量删除。
        deleted = regeneration_queue.delete(self.db, snapshot["batch_id"])
        self.assertGreaterEqual(deleted["deleted"], 2)

    def test_scheduler_honors_concurrency_limit(self):
        active = 0
        maximum = 0

        async def fake_storyboard(shot_id, data, db):
            nonlocal active, maximum
            key = f"shot:{shot_id}:storyboard"
            claim = task_registry.claim_job(
                key,
                f"shot:{shot_id}",
                version=1,
                job_type="shot_image",
                project_id="queue-project",
                current_step="storyboard",
            )
            self.assertTrue(claim.claimed, claim.message)
            active += 1
            maximum = max(maximum, active)

            async def worker():
                nonlocal active
                await asyncio.sleep(0.05)
                active -= 1

            task_registry.start(key, worker())
            return {"id": shot_id}

        async def run():
            with patch.object(shot_route, "regenerate_shot", fake_storyboard):
                submission = regeneration_queue.submit(
                    self.db,
                    "queue-project",
                    ["queue-shot-2", "queue-shot-7", "queue-shot-8"],
                    ["storyboard"],
                    concurrency=2,
                    force_confirmed=True,
                )
                for _ in range(180):
                    await asyncio.sleep(0.02)
                    snapshot = regeneration_queue.batch_snapshot(self.db, submission.batch_id)
                    if snapshot["summary"]["completed"] == 3:
                        return snapshot
                return regeneration_queue.batch_snapshot(self.db, submission.batch_id)

        snapshot = asyncio.run(run())
        self.assertEqual(snapshot["summary"]["completed"], 3)
        self.assertLessEqual(maximum, 2)

    def test_one_failed_shot_does_not_fail_other_queue_items(self):
        async def fake_storyboard(shot_id, data, db):
            if shot_id == "queue-shot-2":
                raise RuntimeError("one shot failed")
            key = f"shot:{shot_id}:storyboard"
            claim = task_registry.claim_job(
                key,
                f"shot:{shot_id}",
                version=1,
                job_type="shot_image",
                project_id="queue-project",
                current_step="storyboard",
            )
            self.assertTrue(claim.claimed, claim.message)

            async def worker():
                await asyncio.sleep(0.01)

            task_registry.start(key, worker())
            return {"id": shot_id}

        async def run():
            with patch.object(shot_route, "regenerate_shot", fake_storyboard):
                submission = regeneration_queue.submit(
                    self.db,
                    "queue-project",
                    ["queue-shot-2", "queue-shot-7"],
                    ["storyboard"],
                    concurrency=2,
                    force_confirmed=True,
                )
                for _ in range(120):
                    await asyncio.sleep(0.02)
                    snapshot = regeneration_queue.batch_snapshot(self.db, submission.batch_id)
                    if snapshot["summary"]["failed"] == 1 and snapshot["summary"]["completed"] == 1:
                        return snapshot
                return regeneration_queue.batch_snapshot(self.db, submission.batch_id)

        snapshot = asyncio.run(run())
        self.assertEqual(snapshot["summary"]["failed"], 1)
        self.assertEqual(snapshot["summary"]["completed"], 1)
        untouched = self.db.query(Shot).filter(Shot.id == "queue-shot-8").one()
        self.assertEqual(untouched.version, 1)

    def test_cancel_running_queue_stops_worker_without_touching_shot(self):
        async def fake_storyboard(shot_id, data, db):
            key = f"shot:{shot_id}:storyboard"
            claim = task_registry.claim_job(
                key,
                f"shot:{shot_id}",
                version=1,
                job_type="shot_image",
                project_id="queue-project",
                current_step="fake",
            )
            self.assertTrue(claim.claimed, claim.message)

            async def worker():
                await asyncio.sleep(5)

            task_registry.start(key, worker())
            return {"id": shot_id}

        async def run():
            with patch.object(shot_route, "regenerate_shot", fake_storyboard):
                submission = regeneration_queue.submit(
                    self.db,
                    "queue-project",
                    ["queue-shot-2"],
                    ["storyboard"],
                    force_confirmed=True,
                )
                for _ in range(40):
                    await asyncio.sleep(0.02)
                    active_db = SessionLocal()
                    active = (
                        active_db.query(BackgroundJob)
                        .filter(BackgroundJob.queue_batch_id == submission.batch_id, BackgroundJob.status == "running")
                        .count()
                    )
                    active_db.close()
                    if active >= 2:
                        break
                cancel_db = SessionLocal()
                regeneration_queue.cancel(cancel_db, submission.batch_id)
                cancel_db.close()
                await asyncio.sleep(0.15)
                detail_db = SessionLocal()
                snapshot = regeneration_queue.batch_snapshot(detail_db, submission.batch_id)
                shot = detail_db.query(Shot).filter(Shot.id == "queue-shot-2").one()
                detail_db.close()
                return snapshot, shot

        snapshot, shot = asyncio.run(run())
        self.assertEqual(snapshot["summary"]["cancelled"], 1)
        self.assertEqual(shot.version, 1)
        self.assertEqual(shot.storyboard_path, "")

    def test_project_scope_busy_returns_queue_item_to_waiting(self):
        project_key = "project:queue-project:storyboard"
        claim = task_registry.claim_job(
            project_key,
            "project:queue-project",
            version=1,
            job_type="storyboard",
            project_id="queue-project",
            current_step="project_storyboard",
        )
        self.assertTrue(claim.claimed, claim.message)

        async def project_worker():
            await asyncio.sleep(0.3)

        async def run():
            task_registry.start(project_key, project_worker())
            with patch.object(regeneration_queue.asyncio, "create_task") as create_task:
                create_task.side_effect = lambda coroutine: (coroutine.close(), MagicMock())[1]
                create_task.return_value.add_done_callback = lambda callback: None
                submission = regeneration_queue.submit(
                    self.db,
                    "queue-project",
                    ["queue-shot-2"],
                    ["storyboard"],
                    force_confirmed=True,
                )
            # 直接运行一个队列项，覆盖真实路由层的 scope_busy -> deduplicated 分支。
            with patch.object(shot_route, "regenerate_shot", return_value={"deduplicated": True}):
                await regeneration_queue._run_item(submission.items[0]["id"])
            detail_db = SessionLocal()
            try:
                return regeneration_queue.batch_snapshot(detail_db, submission.batch_id)
            finally:
                detail_db.close()

        snapshot = asyncio.run(run())
        self.assertEqual(snapshot["summary"]["queued"], 1)
        self.assertEqual(snapshot["items"][0]["blocked_reason"], "scope_busy")
        task_registry.cancel(project_key)


if __name__ == "__main__":
    unittest.main()
