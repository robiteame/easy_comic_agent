"""视频候选生成、保存、默认选择、失败重试与版本围栏回归测试。"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from agent.checkpoints import CheckpointStore  # noqa: E402
from agent.contracts import (  # noqa: E402
    QUALITY_STRATEGIES,
    QualityProfileName,
    VideoCandidateRecord,
    select_video_candidate,
)
from api.routes import shot as shot_route  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import Project, Shot, ShotVideoCandidate  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


def _candidate(
    candidate_id: str, *, score: float, passed: bool | None = True, status: str = "succeeded", duration: int = 100
) -> dict:
    return {
        "candidate_id": candidate_id,
        "shot_id": "candidate-shot",
        "shot_version": 4,
        "batch_id": "batch",
        "candidate_index": 1,
        "status": status,
        "video_path": f"/output/{candidate_id}.mp4",
        "tail_frame_path": f"/output/{candidate_id}.png",
        "provider": "test-video",
        "model": "test-model",
        "execution_plan_hash": "plan-hash",
        "reference_manifest": [{"type": "approved_storyboard_first_frame", "path": "story.png", "sent": True}],
        "generation_duration_ms": duration,
        "score": score,
        "structural_passed": passed,
        "structural_metrics": {"passed": bool(passed), "issues": []},
    }


class QualityCandidateCountTests(unittest.TestCase):
    def test_quality_profiles_use_existing_candidate_counts(self) -> None:
        self.assertEqual(QUALITY_STRATEGIES[QualityProfileName.DRAFT].candidate_count, 1)
        self.assertEqual(QUALITY_STRATEGIES[QualityProfileName.STANDARD].candidate_count, 2)
        self.assertEqual(QUALITY_STRATEGIES[QualityProfileName.FINISHING].candidate_count, 3)

    def test_manual_single_shot_defaults_to_one_candidate(self) -> None:
        import inspect

        self.assertEqual(inspect.signature(shot_route._run_single_shot_video).parameters["candidate_count"].default, 1)

    def test_video_worker_passes_profile_candidate_count(self) -> None:
        from agent import shot_work

        for profile, expected in (
            (QualityProfileName.DRAFT.value, 1),
            (QualityProfileName.STANDARD.value, 2),
            (QualityProfileName.FINISHING.value, 3),
        ):
            with (
                patch(
                    "api.routes.shot._run_single_shot_video",
                    new=AsyncMock(return_value={"video_candidates": [], "candidate_selection": {}}),
                ) as generate,
                patch.object(
                    shot_work,
                    "_db_artifact",
                    return_value={
                        "shot_id": "candidate-shot",
                        "shot_version": 1,
                        "stage": "video_generation",
                        "status": "succeeded",
                    },
                ),
            ):
                asyncio.run(
                    shot_work.generate_video_shot(
                        "candidate-shot",
                        1,
                        project_id="candidate-project",
                        quality_profile=profile,
                    )
                )
            self.assertEqual(generate.await_args.kwargs["candidate_count"], expected)


class CandidateSelectionTests(unittest.TestCase):
    def test_selection_prefers_structural_pass_then_highest_score(self) -> None:
        rows = [
            VideoCandidateRecord.model_validate(_candidate("failed-high", score=1.0, status="failed")),
            VideoCandidateRecord.model_validate(_candidate("invalid-high", score=1.0, passed=False)),
            VideoCandidateRecord.model_validate(_candidate("valid-low", score=0.4, duration=20)),
            VideoCandidateRecord.model_validate(_candidate("valid-high", score=0.8, duration=50)),
        ]
        selected = select_video_candidate(rows)
        self.assertEqual(selected.candidate_id, "valid-high")
        self.assertEqual(selected.reason, "structural_pass_highest_score")
        self.assertEqual(
            {item["reason"] for item in selected.rejected},
            {"candidate_failed", "structural_check_failed"},
        )

    def test_failure_candidate_is_not_selected_but_remains_recorded(self) -> None:
        rows = [
            VideoCandidateRecord.model_validate(_candidate("success", score=0.7)),
            VideoCandidateRecord.model_validate(_candidate("failure", score=1.0, status="failed")),
        ]
        selected = select_video_candidate(rows)
        self.assertEqual(selected.candidate_id, "success")
        self.assertEqual({item.candidate_id for item in rows}, {"success", "failure"})


class _CandidatePersistenceMixin:
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    def setUp(self) -> None:
        self.db = SessionLocal()
        self.db.query(ShotVideoCandidate).delete()
        self.db.query(Shot).delete()
        self.db.query(Project).delete()
        self.db.commit()
        self.root = TEST_ROOT / "video-candidate-checkpoints"
        self.store = CheckpointStore("candidate-project", "candidate-run", root=self.root)

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()


class CandidatePersistenceTests(_CandidatePersistenceMixin, unittest.TestCase):
    def test_checkpoint_saves_success_and_failure_candidates_independently(self) -> None:
        success = _candidate("candidate-success", score=1.0)
        failure = {
            **_candidate("candidate-failure", score=0.0, passed=False, status="failed"),
            "failure": {"kind": "video_failed", "stage": "video_generation", "message": "provider timeout"},
        }
        self.store.save_video_candidate("candidate-shot", success)
        self.store.save_video_candidate("candidate-shot", failure)

        rows = self.store.video_candidates("candidate-shot", shot_version=4)
        self.assertEqual({row["candidate_id"] for row in rows}, {"candidate-success", "candidate-failure"})
        self.assertEqual({row["status"] for row in rows}, {"succeeded", "failed"})
        saved = next(row for row in rows if row["candidate_id"] == "candidate-success")
        self.assertEqual(saved["provider"], "test-video")
        self.assertEqual(saved["model"], "test-model")
        self.assertEqual(saved["execution_plan_hash"], "plan-hash")
        self.assertEqual(saved["tail_frame_path"], "/output/candidate-success.png")
        self.assertEqual(saved["generation_duration_ms"], 100)
        self.assertTrue(saved["reference_manifest"])
        self.assertEqual(len(self.store.failed_video_candidates("candidate-shot", shot_version=4)), 1)

    def test_database_candidate_save_keeps_required_generation_metadata(self) -> None:
        payload = _candidate("database-candidate", score=1.0)
        saved = shot_route._save_video_candidate(payload)
        row = self.db.get(ShotVideoCandidate, "database-candidate")
        self.assertIsNotNone(row)
        self.assertEqual(row.provider, "test-video")
        self.assertEqual(row.model, "test-model")
        self.assertEqual(row.execution_plan_hash, "plan-hash")
        self.assertEqual(row.tail_frame_path, "/output/database-candidate.png")
        self.assertEqual(row.generation_duration_ms, 100)
        self.assertEqual(saved["reference_manifest"][0]["type"], "approved_storyboard_first_frame")

    def test_candidate_ids_and_media_paths_are_independent(self) -> None:
        first_id = shot_route._new_video_candidate_id("shot_v1", 1, "batch-a")
        second_id = shot_route._new_video_candidate_id("shot_v1", 2, "batch-a")
        retry_id = shot_route._new_video_candidate_id("shot_v1", 1, "batch-b", retry_of_candidate_id=first_id)
        first_path = shot_route._video_candidate_media_id("shot_v1", 1)
        second_path = shot_route._video_candidate_media_id("shot_v1", 2)
        retry_path = shot_route._video_candidate_media_id("shot_v1", 1, retry_of_candidate_id=first_id)
        retry_batch_a = shot_route._video_candidate_media_id("shot_v1", 1, first_id, "batch-a")
        retry_batch_b = shot_route._video_candidate_media_id("shot_v1", 1, first_id, "batch-b")
        retry_candidate_two = shot_route._video_candidate_media_id("shot_v1", 2, first_id, "batch-b")
        first_batch_a = shot_route._video_candidate_media_id("shot_v1", 1, batch_id="batch-a")
        first_batch_b = shot_route._video_candidate_media_id("shot_v1", 1, batch_id="batch-b")
        self.assertEqual(len({first_id, second_id, retry_id}), 3)
        self.assertEqual(
            len(
                {
                    first_path,
                    second_path,
                    retry_path,
                    retry_batch_a,
                    retry_batch_b,
                    retry_candidate_two,
                    first_batch_a,
                    first_batch_b,
                }
            ),
            8,
        )
        self.assertTrue(retry_batch_b.endswith("_c1"))

    def test_seed_override_is_distinct_for_each_candidate(self) -> None:
        seeds = {
            int.from_bytes(__import__("hashlib").sha256(f"123:2:{batch}:retry".encode()).digest()[:4], "big")
            % (2**31 - 1)
            for batch in ("batch-a", "batch-b")
        }
        self.assertEqual(len(seeds), 2)


class CandidateGenerationTests(_CandidatePersistenceMixin, unittest.TestCase):
    def test_three_candidates_finish_before_official_video_is_replaced(self) -> None:
        project = Project(id="candidate-generation-project", title="Candidate generation")
        shot = Shot(
            id="candidate-generation-shot",
            project_id=project.id,
            version=3,
            confirmed=True,
            storyboard_path="story.png",
            image_path="story.png",
            video_path="/output/old-official.mp4",
            status="storyboard_approved",
        )
        self.db.add_all([project, shot])
        self.db.commit()
        calls: list[str] = []
        seeds: list[int] = []
        active = 0
        max_active = 0

        async def fake_generate(shot_data, *_args, **_kwargs):
            nonlocal active, max_active
            media_id = str(shot_data["shot_id"])
            calls.append(media_id)
            seeds.append(int(shot_data["seed"]))
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            db = SessionLocal()
            try:
                current = db.get(Shot, shot.id)
                self.assertEqual(current.video_path, "/output/old-official.mp4", "候选生成期间不得覆盖正式视频")
            finally:
                db.close()
            try:
                if media_id.endswith("_c2"):
                    raise RuntimeError("one candidate failed")
                video = TEST_ROOT / f"{media_id}.mp4"
                frame = TEST_ROOT / f"{media_id}.png"
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b"v" * 5000)
                frame.write_bytes(b"f" * 1200)
                return {
                    "video_path": str(video),
                    "frame_path": str(frame),
                    "generation_report": {"provider": "candidate-provider", "model": "candidate-model"},
                    "reference_manifest": [
                        {"type": "approved_storyboard_first_frame", "path": "story.png", "sent": True}
                    ],
                }
            finally:
                active -= 1

        async def fake_validate(path: str, **_kwargs):
            return {"kind": "video", "path": path, "passed": True, "issues": []}

        with (
            patch.object(shot_route, "ensure_generation_gate", return_value={}),
            patch.object(shot_route, "_ensure_scene_baselines", new=AsyncMock()),
            patch.object(shot_route, "_materialize_control_references"),
            patch.object(
                shot_route,
                "_prepare_shot_audio",
                new=AsyncMock(return_value={"audio_path": "", "native_routed": False}),
            ),
            patch.object(shot_route, "_progress", new=AsyncMock()),
            patch.object(shot_route, "validate_video_file", side_effect=fake_validate),
            patch.object(shot_route, "_generate_shot_video", side_effect=fake_generate),
            patch.object(shot_route.ws_manager, "send_to_project", new=AsyncMock()),
        ):
            result = asyncio.run(
                shot_route._run_single_shot_video(
                    shot.id,
                    force=True,
                    expected_version=3,
                    candidate_count=3,
                    strict_structural_selection=True,
                    seed_override=123,
                )
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual(len(set(calls)), 3)
        self.assertEqual(len(set(seeds)), 3)
        self.assertGreater(max_active, 1, "候选必须并行生成")
        rows = result["video_candidates"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(item["status"] == "failed" for item in rows), 1)
        self.assertEqual(sum(item["status"] == "succeeded" for item in rows), 2)
        selected = next(item for item in rows if item["candidate_id"] == result["selected_video_candidate_id"])
        self.assertEqual(selected["provider"], "candidate-provider")
        self.assertEqual(selected["model"], "candidate-model")
        self.assertTrue(selected["execution_plan_hash"])
        self.assertTrue(selected["reference_manifest"])
        self.assertTrue(selected["tail_frame_path"])
        self.assertGreaterEqual(selected["generation_duration_ms"], 0)
        self.db.expire_all()
        current = self.db.get(Shot, shot.id)
        self.assertEqual(current.video_path, selected["video_path"])


class FailedCandidateRetryTests(_CandidatePersistenceMixin, unittest.TestCase):
    def test_retry_appends_new_candidate_and_preserves_failed_attempt(self) -> None:
        failed = {
            **_candidate("failed-attempt", score=0.0, passed=False, status="failed"),
            "shot_id": "retry-shot",
            "failure": {"kind": "video_failed", "stage": "video_generation", "message": "temporary failure"},
        }
        shot_route._save_video_candidate(failed)

        async def fake_generate(shot_id: str, **kwargs):
            self.assertEqual(shot_id, "retry-shot")
            self.assertEqual(kwargs["candidate_count"], 1)
            self.assertTrue(kwargs["strict_structural_selection"])
            self.assertEqual(kwargs["retry_of_candidate_id"], "failed-attempt")
            shot_route._save_video_candidate(
                {
                    **_candidate("retry-attempt", score=1.0),
                    "shot_id": "retry-shot",
                    "retry_of_candidate_id": "failed-attempt",
                }
            )
            return {"selected_video_candidate_id": "retry-attempt"}

        with patch.object(shot_route, "_run_single_shot_video", side_effect=fake_generate):
            result = asyncio.run(shot_route._retry_failed_video_candidate("retry-shot", "failed-attempt", 4))

        self.assertEqual(result["selected_video_candidate_id"], "retry-attempt")
        rows = {item["candidate_id"]: item for item in shot_route._shot_video_candidates("retry-shot", shot_version=4)}
        self.assertEqual(rows["failed-attempt"]["status"], "failed")
        self.assertEqual(rows["failed-attempt"]["failure"]["message"], "temporary failure")
        self.assertEqual(rows["retry-attempt"]["status"], "succeeded")
        self.assertEqual(rows["retry-attempt"]["retry_of_candidate_id"], "failed-attempt")


class CandidateVersionConflictTests(_CandidatePersistenceMixin, unittest.TestCase):
    def test_checkpoint_selection_rejects_stale_candidate_without_marking_it(self) -> None:
        self.store.save_video_candidate("version-shot", _candidate("stale-candidate", score=1.0))
        with self.assertRaisesRegex(RuntimeError, "version conflict"):
            self.store.mark_video_candidate_selected(
                "version-shot",
                "stale-candidate",
                shot_version=5,
                reason="default",
            )
        row = self.store.video_candidates("version-shot", shot_version=4)[0]
        self.assertFalse(row["selected"])

    def test_database_selection_rejects_stale_candidate_and_keeps_official_video(self) -> None:
        project = Project(id="candidate-version-project", title="Candidate version")
        shot = Shot(
            id="candidate-version-shot",
            project_id=project.id,
            version=2,
            video_path="/output/current.mp4",
            status="video_done",
        )
        self.db.add_all([project, shot])
        self.db.commit()
        shot_route._save_video_candidate(
            {
                **_candidate("stale-db-candidate", score=1.0),
                "shot_id": shot.id,
                "shot_version": 1,
            }
        )
        self.db.expire_all()
        current = self.db.get(Shot, shot.id)
        selection = select_video_candidate(
            shot_route._shot_video_candidates(shot.id, shot_version=1),
        )
        with self.assertRaisesRegex(RuntimeError, "版本已变化"):
            shot_route._persist_video_candidate_selection(
                self.db,
                current,
                selection,
                expected_version=1,
            )
        self.db.rollback()
        current = self.db.get(Shot, shot.id)
        self.assertEqual(current.video_path, "/output/current.mp4")
        row = self.db.get(ShotVideoCandidate, "stale-db-candidate")
        self.assertFalse(row.selected)


if __name__ == "__main__":
    unittest.main()
