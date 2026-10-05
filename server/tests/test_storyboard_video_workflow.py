"""生成工作流验收测试：结构检查门禁、单镜头重试与尾帧续接。

- 自动模式不因「有图」就批量批准：结构检查不合格的故事板既不会被批准，
  也只重生成失败镜头；
- 视频生成失败只重试当前镜头，不重跑整个项目；
- 只有 continuous_action 自动使用上一镜尾帧；same_scene 不继承上一镜具体画面。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from PIL import Image  # noqa: E402

from agent import graph  # noqa: E402
from api.routes import shot as shot_route  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import Project, Shot  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402


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
        """结构合格 + 质量审核全通过才批准；本测试桩掉质量审核为全通过。"""
        from tests.test_auto_quality_gate import passing_review_patch  # noqa: E402

        project_id = self._seed("auto_approve_valid", {1: True, 2: True})
        regenerated: list[list[str]] = []

        async def fake_regenerate(pid, shot_ids):  # noqa: ANN001
            regenerated.append(list(shot_ids))

        with (
            patch.object(shot_route, "_run_storyboard_generation", fake_regenerate),
            passing_review_patch(),
        ):
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
        from tests.test_auto_quality_gate import video_gate_passing_patch  # noqa: E402

        calls: list[tuple[str, bool]] = []

        async def fake_single(shot_id, force=False, **kwargs):  # noqa: ANN001
            calls.append((shot_id, force))
            handler(shot_id, force)

        with (
            patch.object(shot_route, "_run_single_shot_video", fake_single),
            patch.object(graph, "_shot_ids", lambda pid: shot_ids),
            patch.object(graph, "_has_unfinished_videos", lambda pid: False),
            video_gate_passing_patch(),
        ):
            result = asyncio.run(graph._generate_shot_videos({"project_id": project_id}))
        return result, calls

    def test_only_the_failed_shot_is_retried(self) -> None:
        project_id = self._seed("video_retry", {1: True, 2: True, 3: True})
        shot_ids = [f"{project_id}_shot_{index}" for index in (1, 2, 3)]

        def handler(shot_id: str, force: bool) -> None:
            if shot_id.endswith("shot_2") and not force:
                # 瞬时错误（超时）才允许重试；重试只针对失败的那一个镜头。
                raise RuntimeError("Seedance 创建任务超时 timeout")

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
            # 持续超时属于瞬时错误证据，每镜头允许一次重试，重试后仍失败则终止。
            raise RuntimeError("Seedance 任务超时 timeout")

        with self.assertLogs("agent.graph", level=logging.ERROR) as captured:
            result, calls = self._run(project_id, shot_ids, handler)

        self.assertEqual(calls, [(shot_ids[0], False), (shot_ids[0], True)])
        self.assertTrue(result["errors"])
        self.assertEqual(result["current_step"], "aborted")
        self.assertIn(shot_ids[0], "\n".join(captured.output))


class LastFrameContinuityTests(WorkflowTestCase):
    def test_continuous_action_uses_previous_last_frame(self) -> None:
        project_id = self._seed("last_frame", {1: True, 2: True})
        last_frame = _noise_image(TEST_ROOT / "output" / project_id / "shot_1_last.png", (256, 256))
        previous_storyboard = _noise_image(TEST_ROOT / "output" / project_id / "shot_1_story.png", (256, 256))
        self.db.query(Shot).filter(Shot.id == f"{project_id}_shot_1").update(
            {"last_frame_path": last_frame, "storyboard_path": previous_storyboard}
        )
        self.db.query(Shot).filter(Shot.id == f"{project_id}_shot_2").update(
            {"continuity_profile": json.dumps({"continuity_mode": "continuous_action"}, ensure_ascii=False)}
        )
        self.db.commit()
        second = self.db.query(Shot).filter(Shot.id == f"{project_id}_shot_2").first()

        resolved = shot_route._previous_reference_for_shot(self.db, second)

        self.assertEqual(resolved, last_frame)

    def test_same_scene_does_not_inherit_previous_concrete_image(self) -> None:
        project_id = self._seed("same_scene_identity", {1: True, 2: True})
        last_frame = _noise_image(TEST_ROOT / "output" / project_id / "shot_1_last.png", (256, 256))
        previous_storyboard = _noise_image(TEST_ROOT / "output" / project_id / "shot_1_story.png", (256, 256))
        self.db.query(Shot).filter(Shot.id == f"{project_id}_shot_1").update(
            {"last_frame_path": last_frame, "storyboard_path": previous_storyboard}
        )
        self.db.query(Shot).filter(Shot.id == f"{project_id}_shot_2").update(
            {"continuity_profile": json.dumps({"continuity_mode": "same_scene"}, ensure_ascii=False)}
        )
        self.db.commit()
        second = self.db.query(Shot).filter(Shot.id == f"{project_id}_shot_2").first()

        resolved = shot_route._previous_reference_for_shot(self.db, second)

        self.assertEqual(resolved, "")

    def test_storyboard_generation_resolves_previous_shot_by_policy(self) -> None:
        source = inspect.getsource(shot_route._run_storyboard_generation_impl)
        self.assertIn("previous_shot=", source)
        self.assertNotIn("previous.storyboard_path or previous.image_path", source)


if __name__ == "__main__":
    unittest.main()
