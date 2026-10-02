"""逐镜头并行、候选管理、版本围栏和断点恢复的稳定契约回归测试。"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT  # noqa: F401,E402

from agent.checkpoints import CheckpointStore  # noqa: E402
from agent.contracts import (  # noqa: E402
    DecisionTrace,
    StageName,
    StageStatus,
    VideoCandidateSelection,
    select_video_candidate,
)
from agent.shot_work import fan_in_shot_results, run_shot_fanout  # noqa: E402
from api.routes import shot as shot_route  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import Project, Shot, ShotVersion  # noqa: E402
from services.shot_version_service import create_version, version_detail  # noqa: E402


class ParallelFanoutContractTests(unittest.TestCase):
    def test_parallel_success_partial_failure_and_resume_are_independent(self) -> None:
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as root:
                store = CheckpointStore("parallel-project", "parallel-run", root=Path(root))
                started: set[str] = set()
                both_started = asyncio.Event()

                async def worker(shot_id: str, version: int) -> dict:
                    started.add(shot_id)
                    if len(started) == 2:
                        both_started.set()
                    await asyncio.wait_for(both_started.wait(), timeout=1)
                    if shot_id == "failed":
                        raise RuntimeError("镜头版本已变化")
                    path = Path(root) / f"{shot_id}.mp4"
                    path.write_bytes(b"v" * 5000)
                    return {"shot_id": shot_id, "shot_version": version, "status": "succeeded", "path": str(path)}

                result = await run_shot_fanout(
                    project_id="parallel-project",
                    shot_versions={"success": 1, "failed": 1},
                    stage=StageName.VIDEO_GENERATION,
                    worker=worker,
                    checkpoint=store,
                    concurrency=2,
                    run_id="parallel-run",
                    input_fingerprint="fp",
                )
                self.assertEqual(started, {"success", "failed"})
                self.assertEqual({item["shot_id"] for item in result["successes"]}, {"success"})
                self.assertEqual({item["shot_id"] for item in result["failures"]}, {"failed"})
                self.assertEqual(result["pending"], [])
                self.assertEqual(
                    {"successes", "failures", "degraded", "skipped", "pending", "artifacts"},
                    set(result) & {"successes", "failures", "degraded", "skipped", "pending", "artifacts"},
                )

                calls: list[str] = []

                async def resume_worker(shot_id: str, version: int) -> dict:
                    calls.append(shot_id)
                    return {"shot_id": shot_id, "shot_version": version, "status": "succeeded", "path": str(Path(root) / f"{shot_id}.mp4")}

                resumed = await run_shot_fanout(
                    project_id="parallel-project",
                    shot_versions={"success": 1, "failed": 1},
                    stage=StageName.VIDEO_GENERATION,
                    worker=resume_worker,
                    checkpoint=store,
                    concurrency=2,
                    run_id="parallel-run",
                    input_fingerprint="fp",
                )
                self.assertEqual(calls, ["failed"])
                self.assertEqual({item["shot_id"] for item in resumed["skipped"]}, {"success"})

        asyncio.run(scenario())

    def test_fan_in_always_returns_all_status_groups(self) -> None:
        result = fan_in_shot_results([
            {"shot_id": "ok", "status": StageStatus.SUCCEEDED.value},
            {"shot_id": "bad", "status": StageStatus.FAILED.value},
            {"shot_id": "warn", "status": StageStatus.DEGRADED.value},
            {"shot_id": "skip", "status": StageStatus.SKIPPED.value},
            {"shot_id": "wait", "status": StageStatus.RUNNING.value},
        ])
        self.assertTrue({"successes", "failures", "degraded", "skipped", "pending", "artifacts"} <= set(result))
        self.assertEqual([item["shot_id"] for item in result["pending"]], ["wait"])
        self.assertEqual(len(result["artifacts"]), 5)


class CandidateContractAndHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    def setUp(self) -> None:
        self.db = SessionLocal()
        self.db.query(ShotVersion).delete()
        self.db.query(Shot).delete()
        self.db.query(Project).delete()
        self.db.commit()

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()

    def test_candidate_metadata_and_structural_failure_selection(self) -> None:
        payload = {
            "candidate_id": "candidate-contract",
            "shot_id": "candidate-shot",
            "shot_version": 2,
            "status": "failed",
            "path": "/output/candidate.mp4",
            "last_frame_path": "/output/candidate.png",
            "provider": "provider-a",
            "model": "model-a",
            "seed": 12345,
            "recipe_hash": "recipe-a",
            "reference_manifest": [{"type": "approved_storyboard_first_frame", "path": "story.png"}],
            "score": 0.0,
            "metrics": {"passed": False, "issues": ["structural"]},
            "failure": {"kind": "video_failed", "stage": "video_generation", "message": "broken"},
            "structural_passed": False,
        }
        shot_route._save_video_candidate(payload)
        saved = shot_route._shot_video_candidates("candidate-shot", shot_version=2)[0]
        for key in ("candidate_id", "shot_id", "shot_version", "provider", "model", "seed", "recipe_hash", "reference_manifest", "path", "last_frame_path", "score", "metrics", "failure"):
            self.assertIn(key, saved)
        self.assertEqual(saved["seed"], 12345)
        self.assertEqual(saved["failure"]["message"], "broken")
        selected = select_video_candidate([
            {**payload, "candidate_id": "invalid", "status": "succeeded", "structural_passed": False, "score": 1.0},
            {**payload, "candidate_id": "valid", "status": "succeeded", "structural_passed": True, "score": 0.5},
        ])
        self.assertEqual(selected.candidate_id, "valid")
        self.assertIn("structural_check_failed", {item["reason"] for item in selected.rejected})

    def test_selection_result_is_written_to_version_history_and_decision_trace(self) -> None:
        project = Project(id="candidate-history-project", title="Candidate history")
        shot = Shot(id="candidate-history-shot", project_id=project.id, version=1)
        self.db.add_all([project, shot])
        self.db.commit()
        selection = VideoCandidateSelection(candidate_id="best", shot_version=1, score=0.9, reason="structural_pass_highest_score")
        trace = DecisionTrace(
            trace_id="trace:candidate-history",
            project_id=project.id,
            shot_id=shot.id,
            shot_version=1,
            run_id="shot:candidate-history-shot:video",
            stage=StageName.VIDEO_GENERATION,
            mode="auto",
            selected_video_candidate_id="best",
            candidate_selection=selection.model_dump(mode="json"),
            video_candidates=[{"candidate_id": "best", "score": 0.9}],
            reason="structural_pass_highest_score",
        ).model_dump(mode="json")
        row = create_version(
            self.db,
            shot,
            "regenerate",
            task_id="shot:candidate-history-shot:video",
            candidate_selection=selection.model_dump(mode="json"),
            decision_trace=trace,
            force=True,
        )
        self.db.commit()
        detail = version_detail(row)
        self.assertEqual(detail["candidate_selection"]["candidate_id"], "best")
        self.assertEqual(detail["decision_trace"]["selected_video_candidate_id"], "best")


if __name__ == "__main__":
    unittest.main()
