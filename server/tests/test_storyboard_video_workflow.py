"""生成工作流验收测试：结构检查门禁、单镜头重试与尾帧续接。

- 自动模式不因「有图」就批量批准：结构检查不合格的故事板既不会被批准，
  也只重生成失败镜头；
- 视频生成失败只重试当前镜头，不重跑整个项目；
- 同场景故事板优先使用上一镜尾帧（Seedance return_last_frame 产物）做续帧参考。
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT  # noqa: F401,E402

from PIL import Image  # noqa: E402

from agent import graph  # noqa: E402
from api.routes import shot as shot_route  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import Project, Shot  # noqa: E402
from services.shot_version_service import create_version, list_versions  # noqa: E402


def _noise_image(path: Path, size: tuple[int, int] = (512, 512)) -> str:
    """写一张有噪点的图：足够大且高于结构检查的最小字节数。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3)).save(path)
    return str(path)


class WorkflowTestCase(unittest.TestCase):
    prefix = "workflow_tests"

    def setUp(self) -> None:
        init_db()
        self.db = SessionLocal()
        self.project_ids: list[str] = []

    def tearDown(self) -> None:
        self.db.rollback()
        for project_id in self.project_ids:
            self.db.query(Shot).filter(Shot.project_id == project_id).delete()
            row = self.db.query(Project).filter(Project.id == project_id).first()
            if row:
                self.db.delete(row)
        self.db.commit()
        self.db.close()

    def _seed(self, name: str, storyboards: dict[int, bool | None]) -> str:
        """创建项目与镜头；storyboards 用 {sequence: 结构是否合格} 描述分镜图。"""
        project_id = f"{self.prefix}_{name}"
        self.project_ids.append(project_id)
        self.db.add(Project(id=project_id, title=name, style="realistic", project_type="series"))
        for sequence, valid in storyboards.items():
            path = ""
            if valid is True:
                path = _noise_image(TEST_ROOT / "output" / project_id / f"shot_{sequence}.png")
            elif valid is False:
                broken = TEST_ROOT / "output" / project_id / f"shot_{sequence}.png"
                broken.parent.mkdir(parents=True, exist_ok=True)
                broken.write_bytes(b"not-an-image")
                path = str(broken)
            self.db.add(
                Shot(
                    id=f"{project_id}_shot_{sequence}",
                    project_id=project_id,
                    sequence=sequence,
                    scene_group_id=f"{project_id}_scene_1",
                    scene_description=f"scene {sequence}",
                    dialogue="",
                    status="storyboard_done",
                    storyboard_path=path,
                    image_path=path,
                )
            )
        self.db.commit()
        return project_id

    def _statuses(self, project_id: str) -> list[Shot]:
        self.db.expire_all()
        return self.db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()


class AutoApproveStructuralGateTests(WorkflowTestCase):
    def test_structurally_invalid_storyboards_are_never_auto_approved(self) -> None:
        project_id = self._seed("auto_approve_invalid", {1: False, 2: True})
        regenerated: list[list[str]] = []

        async def fake_regenerate(pid, shot_ids):  # noqa: ANN001
            regenerated.append(list(shot_ids))

        with patch.object(shot_route, "_run_storyboard_generation", fake_regenerate):
            with self.assertLogs("agent.graph", level=logging.ERROR) as captured:
                result = asyncio.run(graph._auto_approve_storyboard({"project_id": project_id}))

        # 只重生成结构不合格的那一个镜头，仍失败则中止。
        self.assertEqual(regenerated, [[f"{project_id}_shot_1"]])
        self.assertTrue(result["errors"])
        self.assertEqual(result["current_step"], "aborted")
        self.assertIn(f"{project_id}_shot_1", "\n".join(captured.output))
        for shot in self._statuses(project_id):
            self.assertFalse(shot.confirmed, "结构检查不合格的故事板不得被自动批准")
            self.assertNotEqual(shot.status, "storyboard_approved")

    def test_valid_storyboards_are_approved_without_regeneration(self) -> None:
        project_id = self._seed("auto_approve_valid", {1: True, 2: True})
        regenerated: list[list[str]] = []

        async def fake_regenerate(pid, shot_ids):  # noqa: ANN001
            regenerated.append(list(shot_ids))

        with patch.object(shot_route, "_run_storyboard_generation", fake_regenerate):
            result = asyncio.run(graph._auto_approve_storyboard({"project_id": project_id}))

        self.assertEqual(regenerated, [])
        self.assertNotIn("errors", result)
        shots = self._statuses(project_id)
        self.assertEqual(len(shots), 2)
        for shot in shots:
            self.assertTrue(shot.confirmed)
            self.assertEqual(shot.status, "storyboard_approved")


class PerShotVideoRetryTests(WorkflowTestCase):
    def _run(self, project_id: str, shot_ids: list[str], handler) -> tuple[dict, list]:  # noqa: ANN001
        calls: list[tuple[str, bool]] = []

        async def fake_single(shot_id, force=False, **kwargs):  # noqa: ANN001
            calls.append((shot_id, force))
            handler(shot_id, force)

        with (
            patch.object(shot_route, "_run_single_shot_video", fake_single),
            patch.object(graph, "_shot_ids", lambda pid: shot_ids),
            patch.object(graph, "_has_unfinished_videos", lambda pid: False),
        ):
            result = asyncio.run(graph._generate_shot_videos({"project_id": project_id}))
        return result, calls

    def test_only_the_failed_shot_is_retried(self) -> None:
        project_id = self._seed("video_retry", {1: True, 2: True, 3: True})
        shot_ids = [f"{project_id}_shot_{index}" for index in (1, 2, 3)]

        def handler(shot_id: str, force: bool) -> None:
            if shot_id.endswith("shot_2") and not force:
                raise RuntimeError("Seedance 创建任务失败")

        result, calls = self._run(project_id, shot_ids, handler)

        first_pass = [call for call in calls if not call[1]]
        retries = [call for call in calls if call[1]]
        # 首轮每个镜头各一次；重试只针对失败的那一个镜头，不重跑整个项目。
        self.assertEqual(first_pass, [(shot_id, False) for shot_id in shot_ids])
        self.assertEqual(retries, [(f"{project_id}_shot_2", True)])
        self.assertNotIn("errors", result)

    def test_persistent_failure_retries_once_per_shot_then_aborts(self) -> None:
        project_id = self._seed("video_retry_fail", {1: True})
        shot_ids = [f"{project_id}_shot_1"]

        def handler(shot_id: str, force: bool) -> None:
            raise RuntimeError("Seedance 任务失败")

        with self.assertLogs("agent.graph", level=logging.ERROR) as captured:
            result, calls = self._run(project_id, shot_ids, handler)

        self.assertEqual(calls, [(shot_ids[0], False), (shot_ids[0], True)])
        self.assertTrue(result["errors"])
        self.assertEqual(result["current_step"], "aborted")
        self.assertIn(shot_ids[0], "\n".join(captured.output))


class LastFrameContinuityTests(WorkflowTestCase):
    def test_same_scene_storyboard_prefers_previous_last_frame(self) -> None:
        project_id = self._seed("last_frame", {1: True, 2: True})
        last_frame = _noise_image(TEST_ROOT / "output" / project_id / "shot_1_last.png", (256, 256))
        previous_storyboard = _noise_image(TEST_ROOT / "output" / project_id / "shot_1_story.png", (256, 256))
        self.db.query(Shot).filter(Shot.id == f"{project_id}_shot_1").update(
            {"last_frame_path": last_frame, "storyboard_path": previous_storyboard}
        )
        self.db.commit()
        second = self.db.query(Shot).filter(Shot.id == f"{project_id}_shot_2").first()

        resolved = shot_route._previous_reference_for_shot(self.db, second, prefer_last_frame=True)
        self.assertEqual(resolved, last_frame)

        # 尚无尾帧时回退上一镜故事板，行为与首次全量出图一致。
        self.db.query(Shot).filter(Shot.id == f"{project_id}_shot_1").update({"last_frame_path": ""})
        self.db.commit()
        self.db.expire_all()
        second = self.db.query(Shot).filter(Shot.id == f"{project_id}_shot_2").first()
        resolved = shot_route._previous_reference_for_shot(self.db, second, prefer_last_frame=True)
        self.assertEqual(resolved, previous_storyboard)

    def test_storyboard_generation_path_requests_last_frame_preference(self) -> None:
        """出图路径确实以 prefer_last_frame=True 调用（否则尾帧永远不会被用上）。"""

        import inspect

        source = inspect.getsource(shot_route._run_storyboard_generation_impl)
        self.assertIn("prefer_last_frame=True", source)
        self.assertIn("prefer_last_frame=True", inspect.getsource(shot_route._regenerate_single_shot))


class StoryboardCandidateTests(WorkflowTestCase):
    """关键镜头可一次生成 2 个故事板候选，两个候选都进入版本历史。"""

    def _request(self) -> object:
        from api.routes.shot import RegenerateRequest

        return RegenerateRequest(reason="关键反转镜头")

    def test_candidate_count_is_bounded(self) -> None:
        from pydantic import ValidationError

        from api.routes.shot import RegenerateRequest

        self.assertEqual(RegenerateRequest(candidates=1).candidates, 1)
        self.assertEqual(RegenerateRequest(candidates=2).candidates, 2)
        for invalid in (0, 3):
            with self.assertRaises(ValidationError):
                RegenerateRequest(candidates=invalid)

    def test_two_candidates_each_land_in_version_history(self) -> None:
        project_id = self._seed("candidates", {1: True})
        shot_id = f"{project_id}_shot_1"
        seen_versions: list = []

        async def fake_generate(target_shot_id, reason="", expected_version=None):  # noqa: ANN001
            seen_versions.append(expected_version)
            # 模拟真实生成：写入产物，create_version 才能记录出不同的候选快照。
            path = _noise_image(TEST_ROOT / "output" / project_id / f"cand_{len(seen_versions)}.png", (256, 256))
            db = SessionLocal()
            try:
                row = db.query(Shot).filter(Shot.id == target_shot_id).first()
                row.storyboard_path = path
                row.image_path = path
                row.status = "storyboard_done"
                create_version(db, row, "regenerate", task_id="test")
                db.commit()
            finally:
                db.close()

        with patch.object(shot_route, "_regenerate_single_shot", fake_generate):
            asyncio.run(
                shot_route._run_storyboard_candidates(shot_id, self._request(), 2, 1, "key shot")
            )

        # 两个候选使用不同的 expected_version（各自独立抢占镜头版本）。
        self.assertEqual(len(seen_versions), 2)
        self.assertEqual(len(set(seen_versions)), 2)

        self.db.expire_all()
        versions = list_versions(self.db, shot_id)
        self.assertGreaterEqual(len(versions), 2, "两个候选都必须进入版本历史供挑选")

if __name__ == "__main__":
    unittest.main()
