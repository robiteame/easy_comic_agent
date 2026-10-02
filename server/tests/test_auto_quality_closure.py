"""无需人工审核的自动质量闭环回归测试。

锁定四件事：

1. 三个质量分类的真实边界——structural_validity / technical_quality /
   visual_quality_pending，其中空帧是独立检测项（不再与黑帧混为一谈）；
2. 没有真实视觉模型时视觉质量只能是 pending，任何路径都不得伪造成 passed；
3. 显式 auto 模式下，视觉能力缺失不再让闭环卡死：按结构 + 技术门禁自动继续，
   成片以 degraded 发布，且绝不进入 human_gate / waiting_human / needs_human_review；
4. 结构或技术不合格时仍然拦截——自动继续不是无条件放行。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT  # noqa: F401,E402

from PIL import Image  # noqa: E402

from agent import graph  # noqa: E402
from agent import nodes as agent_nodes  # noqa: E402
from agent.contracts import StageName, StageStatus  # noqa: E402
from agent.critic import build_final_report, critique_final, critique_images, critique_videos  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import Character, Project, QualityReview, SceneAsset, Shot  # noqa: E402
from services import structural_validation as sv  # noqa: E402
from services.quality_review_providers import VLMCapability  # noqa: E402
from services.quality_review_service import STAGE_STORYBOARD, STAGE_VIDEO, quality_review_service  # noqa: E402

FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
MEDIA_ROOT = TEST_ROOT / "auto-quality-closure"

VISUAL_KEYS = (
    "first_frame_storyboard_similarity",
    "reference_match",
    "motion_stability",
    "shot_continuity",
    "action_completion",
)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True, capture_output=True)


def _make_video(name: str, *, source: str = "testsrc2", duration: int = 3, color: str = "", audio: bool = False) -> Path:
    """生成测试视频。

    纯色视频会被 x264 压到 MIN_VIDEO_BYTES 以下，先触发「文件过小」而不是帧内容
    检查；因此纯色素材一律带音轨，保证文件足够大。
    """

    path = MEDIA_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = f"color={color}:size=540x960:rate=15:duration={duration}" if color else f"{source}=size=540x960:rate=15:duration={duration}"
    cmd = ["-f", "lavfi", "-i", spec]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-shortest", "-c:a", "aac"]
    _ffmpeg(*cmd, "-pix_fmt", "yuv420p", "-c:v", "libx264", str(path))
    return path


def _noise_image(path: Path, size: tuple[int, int] = (320, 320)) -> str:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3)).save(path)
    return str(path)


def _unsupported_vlm():
    """让质量审核能力如实报告为 unsupported（等价于没配视觉模型）。"""

    class _Unsupported:
        def capability(self) -> VLMCapability:
            return VLMCapability(supported=False, reason="VLM 未配置（测试桩）")

        async def judge(self, system_prompt, user_prompt, image_paths):  # noqa: ANN001, ARG002
            raise AssertionError("视觉模型未配置时不应被调用")

    return _Unsupported()


# ---------------------------------------------------------------------------
# 1. 三个分类的边界
# ---------------------------------------------------------------------------


@unittest.skipUnless(FFMPEG_AVAILABLE, "需要 ffmpeg/ffprobe")
class StructuralValidationCategoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.valid = _make_video("valid.mp4", audio=True)
        # 纯色视频必须带音轨：x264 会把静态画面压到 MIN_VIDEO_BYTES 以下，
        # 那样先触发的是「文件过小」而不是帧内容检查。
        cls.white = _make_video("white.mp4", color="white", audio=True)
        cls.gray = _make_video("gray.mp4", color="gray", audio=True)
        cls.black = _make_video("black.mp4", color="black", audio=True)

    def _report(self, path: Path, **kwargs):
        return asyncio.run(sv.validate_video_file(str(path), **kwargs))

    def test_all_five_visual_dimensions_are_pending(self) -> None:
        report = self._report(self.valid)
        category = report["categories"]["visual_quality_pending"]
        self.assertEqual(category["status"], "pending")
        self.assertIsNone(category["passed"])
        self.assertEqual(tuple(item["key"] for item in category["dimensions"]), VISUAL_KEYS)
        for dimension in category["dimensions"]:
            self.assertEqual(dimension["status"], "pending")
            self.assertIsNone(dimension["passed"], f"{dimension['key']} 不得给出通过结论")
        self.assertTrue(report["visual_quality_pending"])

    def test_visual_dimensions_cannot_be_forced_to_pass(self) -> None:
        """即使调用方手工塞入 passed=True，汇总时也必须被纠正回 pending。"""

        categories = sv._empty_categories()
        categories[sv.CATEGORY_VISUAL_PENDING]["passed"] = True
        categories[sv.CATEGORY_VISUAL_PENDING]["status"] = "passed"
        categories[sv.CATEGORY_VISUAL_PENDING]["dimensions"][0]["passed"] = True
        result = sv._finalize_video_result({"categories": categories})
        pending = result["categories"][sv.CATEGORY_VISUAL_PENDING]
        self.assertEqual(pending["status"], "pending")
        self.assertIsNone(pending["passed"])
        self.assertTrue(all(item["passed"] is None for item in pending["dimensions"]))

    def test_solid_white_and_gray_frames_are_empty_not_black(self) -> None:
        for label, path in (("white", self.white), ("gray", self.gray)):
            report = self._report(path)
            technical = report["categories"]["technical_quality"]
            codes = {item["code"] for item in technical["issues"]}
            self.assertIn("video_empty_frames", codes, f"{label} 空帧必须被检出")
            self.assertNotIn("video_black_frames", codes, f"{label} 不是黑帧")
            self.assertGreater(report["frame_scan"]["blank_seconds"], 0)
            self.assertTrue(report["categories"]["structural_validity"]["passed"], f"{label} 结构上仍可播放")

    def test_black_frames_are_not_double_counted_as_empty(self) -> None:
        report = self._report(self.black)
        codes = {item["code"] for item in report["categories"]["technical_quality"]["issues"]}
        self.assertIn("video_black_frames", codes)
        self.assertNotIn("video_empty_frames", codes, "与黑帧重叠的区间不得重复计为空帧")

    def test_healthy_video_has_no_empty_or_black_frames(self) -> None:
        report = self._report(self.valid)
        codes = {item["code"] for item in report["categories"]["technical_quality"]["issues"]}
        self.assertNotIn("video_empty_frames", codes)
        self.assertNotIn("video_black_frames", codes)
        self.assertEqual(report["frame_scan"]["blank_seconds"], 0)

    def test_output_too_small_is_reported_under_technical_quality(self) -> None:
        tiny = MEDIA_ROOT / "tiny.mp4"
        tiny.parent.mkdir(parents=True, exist_ok=True)
        tiny.write_bytes(b"\x00" * 128)
        report = self._report(tiny)
        technical = report["categories"]["technical_quality"]
        self.assertIn("video_output_too_small", {item["code"] for item in technical["issues"]})
        self.assertIn("video_file_too_small", {item["code"] for item in report["categories"]["structural_validity"]["issues"]})

    def test_audio_track_checked_only_when_expected(self) -> None:
        silent = _make_video("silent.mp4", audio=False)
        # 期望音轨但视频没有 → 结构问题（音视频轨检查）。
        expected = self._report(silent, expect_audio=True)
        self.assertIn(
            "audio_stream_missing",
            {item["code"] for item in expected["categories"]["structural_validity"]["issues"]},
        )
        self.assertFalse(expected["checks"]["audio_track"])
        # 未声明期望时记为 skipped：外部 TTS 模式下单镜头无音轨是正常的。
        unknown = self._report(silent)
        self.assertIn("audio_track", unknown["categories"]["technical_quality"]["skipped"])
        self.assertNotIn(
            "audio_stream_missing",
            {item["code"] for item in unknown["categories"]["structural_validity"]["issues"]},
        )
        # 有音轨且期望音轨 → 无问题。
        ok = self._report(self.valid, expect_audio=True)
        self.assertTrue(ok["checks"]["audio_track"])
        self.assertNotIn(
            "audio_stream_missing",
            {item["code"] for item in ok["categories"]["structural_validity"]["issues"]},
        )


# ---------------------------------------------------------------------------
# 2. Critic 不得伪造视觉结论
# ---------------------------------------------------------------------------


@unittest.skipUnless(FFMPEG_AVAILABLE, "需要 ffmpeg/ffprobe")
class CriticHonestyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.valid = _make_video("critic_valid.mp4")
        cls.white = _make_video("critic_white.mp4", color="white", audio=True)

    def test_video_critic_emits_named_pending_dimensions(self) -> None:
        report = critique_videos([{"shot_id": "s1", "path": str(self.valid)}])
        by_name = {metric.name: metric for metric in report.metrics}
        for key in VISUAL_KEYS:
            self.assertIn(key, by_name)
            self.assertIsNone(by_name[key].passed, f"{key} 必须是 pending")
        self.assertIsNone(by_name["visual_quality_pending"].passed)

    def test_pending_issue_is_localized_to_shot_and_stage(self) -> None:
        report = critique_videos([{"shot_id": "s7", "path": str(self.valid)}])
        issue = next(item for item in report.issues if item.code == "visual_quality_pending")
        self.assertEqual(issue.details["stage"], StageName.VIDEO_GENERATION.value)
        self.assertEqual(issue.details["shot_ids"], ["s7"])
        self.assertEqual(tuple(issue.details["dimensions"]), VISUAL_KEYS)
        per_shot = [item for item in report.issues if item.code.startswith("visual_pending:")]
        self.assertTrue(per_shot, "每个视觉维度都应逐镜头登记")
        self.assertTrue(all(item.shot_id == "s7" for item in per_shot))

    def test_structural_and_technical_pass_does_not_imply_visual_pass(self) -> None:
        report = critique_videos([{"shot_id": "s1", "path": str(self.valid)}])
        self.assertTrue(report.passed)  # 可测量的门禁通过
        visual = [metric for metric in report.metrics if metric.name in VISUAL_KEYS]
        self.assertTrue(all(metric.passed is None for metric in visual))

    def test_empty_frames_surface_as_blocking_issue(self) -> None:
        report = critique_videos([{"shot_id": "s2", "path": str(self.white)}])
        issue = next(item for item in report.issues if item.code == "video_empty_frames")
        self.assertEqual(issue.severity, "error")
        self.assertEqual(issue.shot_id, "s2")
        self.assertTrue(issue.recommendation)
        self.assertFalse(report.passed)

    def test_image_critic_does_not_fabricate_candidate_score(self) -> None:
        image = _noise_image(MEDIA_ROOT / "storyboard.png")
        no_score = critique_images([{"shot_id": "s1", "path": image}])
        metric = next(item for item in no_score.metrics if item.name == "candidate_score")
        self.assertIsNone(metric.passed, "没有评分证据时不得宣称候选评分通过")
        self.assertIsNone(metric.value)

        scored = critique_images([{"shot_id": "s1", "path": image, "score": 0.4}])
        metric = next(item for item in scored.metrics if item.name == "candidate_score")
        self.assertIs(metric.passed, False)
        self.assertEqual(metric.value, 0.4)


class FinalReportClosureTests(unittest.TestCase):
    def test_final_report_localizes_pending_visual_risk_and_marks_degraded(self) -> None:
        state = {
            "project_id": "final-report-closure",
            "run_id": "auto",
            "mode": "auto",
            "quality_threshold": 0.72,
            "output_path": "/tmp/final.mp4",
            "degraded_published": True,
            "degraded_reason": "视觉质量未评估",
            "visual_quality_pending": True,
            "critiques": [{
                "stage": StageName.VIDEO_REVIEW.value,
                "passed": True,
                "score": 1.0,
                "metrics": [{"name": "visual_quality_pending", "passed": None, "detail": "未接入视觉模型"}],
                "issues": [
                    {
                        "code": "visual_pending:motion_stability",
                        "severity": "info",
                        "message": "镜头 s3 运动稳定度待审",
                        "shot_id": "s3",
                        "details": {"dimension": "motion_stability", "stage": StageName.VIDEO_GENERATION.value},
                    },
                    {
                        "code": "visual_quality_pending",
                        "severity": "info",
                        "message": "视觉质量待审",
                        "details": {
                            "stage": StageName.VIDEO_REVIEW.value,
                            "shot_ids": ["s3"],
                            "dimensions": list(VISUAL_KEYS),
                        },
                    },
                ],
            }],
            "shot_artifacts": [{
                "shot_id": "s3",
                "stage": StageName.VIDEO_GENERATION.value,
                "status": "succeeded",
                "path": "/tmp/s3.mp4",
                "structural_passed": True,
            }],
        }
        report = critique_final(state)
        final = next(item["report"] for item in report.evidence if item["kind"] == "final_report")
        risks = [item for item in final["unresolved_risks"] if item["code"] == "visual_quality_pending"]
        self.assertTrue(risks)
        localized = [item for item in risks if item["shot_ids"]]
        self.assertTrue(localized, "视觉待审风险必须能定位到镜头")
        self.assertIn("s3", localized[0]["shot_ids"])
        self.assertTrue(all(item["stage"] for item in risks), "视觉待审风险必须带阶段")
        dimensions = {dim for item in risks for dim in item.get("dimensions", [])}
        self.assertTrue(set(VISUAL_KEYS).issubset(dimensions), f"缺少视觉维度: {VISUAL_KEYS}")
        self.assertEqual(final["final_choice"]["visual_quality"]["status"], "pending")
        self.assertTrue(final["final_choice"]["degraded"], "视觉未验证的成片必须标为降级")

    def test_final_report_lists_repairs_and_degradations_with_shot_and_stage(self) -> None:
        state = {
            "project_id": "final-report-trace",
            "run_id": "auto",
            "quality_threshold": 0.72,
            "decision_traces": [{
                "trace_id": "t1",
                "stage": StageName.VIDEO_GENERATION.value,
                "reason": "视频黑帧",
                "selected": {
                    "strategy": "change_seed",
                    "target_stage": StageName.VIDEO_GENERATION.value,
                    "shot_ids": ["s1"],
                    "seed": 4321,
                },
                "failure": {"kind": "video_failed", "message": "黑帧", "shot_id": "s1"},
                "critique": {"score": 0.2, "affected_shot_ids": ["s1"]},
            }],
            "recovery_history": [{
                "stage": StageName.VIDEO_GENERATION.value,
                "strategy": "change_seed",
                "trace_id": "t1",
                "shot_ids": ["s1"],
            }],
            "shot_artifacts": [{
                "shot_id": "s1",
                "stage": StageName.VIDEO_GENERATION.value,
                "status": "succeeded",
                "path": "/tmp/s1.mp4",
                "structural_passed": True,
            }],
        }
        final = build_final_report(state)
        self.assertTrue(final["automatic_repairs"])
        repair = final["automatic_repairs"][0]
        self.assertEqual(repair["action"], "change_seed")
        self.assertEqual(repair["stage"], StageName.VIDEO_GENERATION.value)
        self.assertEqual(repair["shot_ids"], ["s1"])
        self.assertEqual(repair["seed"], 4321)
        self.assertIn("automatic_repairs", final)
        self.assertIn("degradations", final)
        self.assertIn("unresolved_risks", final)
        self.assertIn("final_choice", final)


# ---------------------------------------------------------------------------
# 3. auto 模式：视觉能力缺失也要自动收口
# ---------------------------------------------------------------------------


class VisualPendingPolicyTests(unittest.TestCase):
    def test_policy_only_applies_to_explicit_auto_mode(self) -> None:
        self.assertTrue(graph._visual_pending_continue({"mode": "auto"}))
        self.assertFalse(graph._visual_pending_continue({"mode": "manual"}))
        self.assertFalse(graph._visual_pending_continue({"project_id": "x"}))
        with patch.object(graph, "_visual_pending_continue", return_value=False):
            self.assertEqual(graph._visual_pending_gate_kwargs({"mode": "manual"}), {})
        self.assertEqual(
            graph._visual_pending_gate_kwargs({"mode": "auto"}),
            {"allow_visual_pending": True},
        )

    def test_block_policy_keeps_fail_closed(self) -> None:
        from config import settings

        with patch.object(settings, "QUALITY_VISUAL_PENDING_POLICY", "block"):
            self.assertFalse(graph._visual_pending_continue({"mode": "auto"}))


class AutoClosureGraphTests(unittest.TestCase):
    prefix = "auto_quality_closure"

    def setUp(self) -> None:
        init_db()
        self.db = SessionLocal()
        self.project_ids: list[str] = []

    def tearDown(self) -> None:
        self.db.rollback()
        for project_id in self.project_ids:
            self.db.query(QualityReview).filter(QualityReview.project_id == project_id).delete()
            self.db.query(Shot).filter(Shot.project_id == project_id).delete()
            self.db.query(Character).filter(Character.project_id == project_id).delete()
            self.db.query(SceneAsset).filter(SceneAsset.project_id == project_id).delete()
            row = self.db.query(Project).filter(Project.id == project_id).first()
            if row:
                self.db.delete(row)
        self.db.commit()
        self.db.close()

    def _seed(self, name: str, *, storyboard: str = "", video: str = "") -> str:
        project_id = f"{self.prefix}_{name}"
        self.project_ids.append(project_id)
        self.db.add(Project(id=project_id, title=name, style="realistic", project_type="series"))
        image = storyboard or _noise_image(TEST_ROOT / "output" / project_id / "shot_1.png")
        self.db.add(
            Shot(
                id=f"{project_id}_shot_1",
                project_id=project_id,
                sequence=1,
                scene_group_id=f"{project_id}_scene_1",
                scene_description="scene 1",
                character_action="走进房间",
                dialogue="",
                characters_in_scene="[]",
                status="storyboard_done",
                storyboard_path=image,
                image_path=image,
                video_path=video,
            )
        )
        self.db.commit()
        return project_id

    def _patch_unsupported(self):
        return patch.object(quality_review_service, "_vlm", _unsupported_vlm())

    def test_storyboard_gate_continues_as_visual_pending_in_auto(self) -> None:
        project_id = self._seed("storyboard_ok")
        with self._patch_unsupported():
            gate = asyncio.run(graph._run_storyboard_quality_gate(project_id, allow_visual_pending=True))
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["visual_pending"])
        self.assertFalse(gate.get("errors"))
        self.assertNotIn("needs_human_review", gate)

    def test_storyboard_gate_still_blocks_broken_structure(self) -> None:
        project_id = self._seed("storyboard_broken", storyboard=str(MEDIA_ROOT / "does-not-exist.png"))
        with self._patch_unsupported():
            gate = asyncio.run(graph._run_storyboard_quality_gate(project_id, allow_visual_pending=True))
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["visual_pending"])
        self.assertTrue(gate.get("errors"), "结构不合格不得放行")

    def test_storyboard_gate_default_still_fails_closed(self) -> None:
        project_id = self._seed("storyboard_strict")
        with self._patch_unsupported():
            gate = asyncio.run(graph._run_storyboard_quality_gate(project_id))
        self.assertFalse(gate["passed"])
        self.assertTrue(gate.get("errors"))

    @unittest.skipUnless(FFMPEG_AVAILABLE, "需要 ffmpeg/ffprobe")
    def test_video_review_continues_as_visual_pending_in_auto(self) -> None:
        video = _make_video("closure_ok.mp4")
        project_id = self._seed("video_ok", video=str(video))
        state = {"project_id": project_id, "mode": "auto", "run_id": "auto"}
        with self._patch_unsupported():
            result = asyncio.run(graph._review_shot_videos(state))
        self.assertFalse(result.get("errors"), "auto 模式不应因缺视觉模型而卡死")
        self.assertTrue(result["visual_pending"])
        self.assertNotIn("needs_human_review", result)

    @unittest.skipUnless(FFMPEG_AVAILABLE, "需要 ffmpeg/ffprobe")
    def test_video_review_blocks_broken_media_even_in_auto(self) -> None:
        blank = _make_video("closure_blank.mp4", color="white", audio=True)
        project_id = self._seed("video_blank", video=str(blank))
        state = {"project_id": project_id, "mode": "auto", "run_id": "auto"}
        with self._patch_unsupported():
            result = asyncio.run(graph._review_shot_videos(state))
        self.assertTrue(result.get("errors"), "空帧视频必须被拦截")
        self.assertFalse(result["visual_pending"])

    @unittest.skipUnless(FFMPEG_AVAILABLE, "需要 ffmpeg/ffprobe")
    def test_video_review_default_still_fails_closed_and_never_asks_human(self) -> None:
        video = _make_video("closure_strict.mp4")
        project_id = self._seed("video_strict", video=str(video))
        with self._patch_unsupported():
            # 兼容入口（未携带 mode）沿用旧的 fail-closed 语义。
            result = asyncio.run(graph._review_shot_videos({"project_id": project_id}))
        self.assertTrue(result.get("errors"))
        with self._patch_unsupported():
            result = asyncio.run(graph._review_shot_videos({"project_id": project_id, "mode": "auto"}))
        # 显式 auto 且策略为 continue 时不再终止。
        self.assertFalse(result.get("errors"))

    def test_gate_status_accepts_unsupported_only_when_allowed(self) -> None:
        project_id = self._seed("gate_policy")
        shot_id = f"{project_id}_shot_1"
        asyncio.run(
            quality_review_service.record_unsupported_reviews(
                project_id, [shot_id], STAGE_STORYBOARD, "VLM 未配置（测试桩）"
            )
        )
        strict = quality_review_service.storyboard_gate_status(project_id)
        self.assertFalse(strict["ok"])
        self.assertNotIn("degraded", strict)

        relaxed = quality_review_service.storyboard_gate_status(project_id, allow_visual_pending=True)
        self.assertTrue(relaxed["ok"], "视觉能力缺失应按结构门禁降级放行")
        self.assertTrue(relaxed["visual_pending"])
        self.assertTrue(relaxed["degraded"])
        self.assertEqual(relaxed["degraded"][0]["shot_id"], shot_id)
        self.assertEqual(relaxed["degraded"][0]["verdict"], "unsupported")

    def test_visual_pending_update_marks_degraded_without_human(self) -> None:
        update = graph._visual_pending_update(
            {"project_id": "p", "mode": "auto"},
            stage=StageName.VIDEO_REVIEW,
            reason="VLM 未配置",
            shot_ids=["s1"],
        )
        self.assertTrue(update["visual_quality_pending"])
        self.assertTrue(update["degraded_published"])
        self.assertIn("pending", update["visual_pending_reason"])
        self.assertNotIn("needs_human_review", update)
        self.assertNotIn("human_reason", update)
        self.assertEqual(update["stage_status"][StageName.VIDEO_REVIEW.value], StageStatus.DEGRADED.value)

    def test_auto_mode_never_requests_human_review_when_visual_pending(self) -> None:
        """显式 auto 下，整条视觉待审路径不得产生人工等待标记。"""

        for state in ({"mode": "auto"}, {"mode": "auto", "human_gate_policy": "manual"}):
            self.assertFalse(graph._human_allowed(state))
            self.assertFalse(graph._legacy_human_allowed(state))
        update = graph._visual_pending_update(
            {"mode": "auto"}, stage=StageName.QUALITY_REVIEW, reason="unsupported"
        )
        self.assertFalse(update.get("needs_human_review", False))


class ReferenceStatusAutoGuardTests(unittest.TestCase):
    prefix = "auto_quality_closure_ref"

    def setUp(self) -> None:
        init_db()
        self.db = SessionLocal()
        self.project_ids: list[str] = []

    def tearDown(self) -> None:
        self.db.rollback()
        for project_id in self.project_ids:
            self.db.query(QualityReview).filter(QualityReview.project_id == project_id).delete()
            self.db.query(Shot).filter(Shot.project_id == project_id).delete()
            self.db.query(Character).filter(Character.project_id == project_id).delete()
            self.db.query(SceneAsset).filter(SceneAsset.project_id == project_id).delete()
            row = self.db.query(Project).filter(Project.id == project_id).first()
            if row:
                self.db.delete(row)
        self.db.commit()
        self.db.close()

    def test_blocking_reference_does_not_write_needs_review_in_auto(self) -> None:
        from services.reference_readiness_service import refresh_project_reference_state

        project_id = f"{self.prefix}_blocking"
        self.project_ids.append(project_id)
        self.db.add(Project(id=project_id, title="blocking", style="realistic", project_type="series"))
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
            baseline_image_path=_noise_image(TEST_ROOT / "output" / project_id / "scene.png"),
            reference_images=json.dumps([_noise_image(TEST_ROOT / "output" / project_id / "scene.png")]),
            reference_status="ready",
        )
        self.db.add_all([character, scene])
        self.db.commit()
        self.db.add(
            Shot(
                id=f"{project_id}_shot_1",
                project_id=project_id,
                sequence=1,
                scene_asset_id=scene.id,
                character_asset_ids=json.dumps([character.id]),
                status="storyboard_done",
                storyboard_path=_noise_image(TEST_ROOT / "output" / project_id / "shot_1.png"),
            )
        )
        self.db.commit()

        manual = refresh_project_reference_state(self.db, project_id)
        self.assertTrue(manual.get("blocking"), "夹具必须真的构成阻塞")
        self.db.expire_all()
        self.assertEqual(
            self.db.query(Project).filter(Project.id == project_id).first().status,
            "needs_review",
        )

        # 用 auto 语义再跑一次：不得写 needs_review，但阻塞必须如实可见。
        self.db.query(Project).filter(Project.id == project_id).update({"status": "draft"})
        self.db.commit()
        auto = refresh_project_reference_state(self.db, project_id, allow_needs_review=False)
        self.assertTrue(auto.get("blocking"))
        self.db.expire_all()
        status = self.db.query(Project).filter(Project.id == project_id).first().status
        self.assertNotEqual(status, "needs_review", "auto 模式不得把项目写成等待人工审核")
        self.assertEqual(status, "degraded", "阻塞状态仍须对用户可见")


# ---------------------------------------------------------------------------
# 4. 端到端：没有视觉模型时自动闭环也能跑完并如实降级
# ---------------------------------------------------------------------------


@unittest.skipUnless(FFMPEG_AVAILABLE, "需要 ffmpeg/ffprobe")
class AutoLoopClosesWithoutVisionModelTests(unittest.TestCase):
    """整图 ainvoke：真实质量门禁 + 未配置 VLM，验证闭环不卡人工。"""

    prefix = "auto_quality_closure_e2e"

    def setUp(self) -> None:
        import tempfile

        init_db()
        self.db = SessionLocal()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.project_ids: list[str] = []

    def tearDown(self) -> None:
        self.db.rollback()
        for project_id in self.project_ids:
            self.db.query(QualityReview).filter(QualityReview.project_id == project_id).delete()
            self.db.query(Shot).filter(Shot.project_id == project_id).delete()
            self.db.query(Character).filter(Character.project_id == project_id).delete()
            self.db.query(Project).filter(Project.id == project_id).delete()
        self.db.commit()
        self.db.close()

    def _seed(self, project_id: str, storyboards: list[str], videos: list[str]) -> None:
        """视频路径直接落库：mock worker 不写 DB，门禁读的是 Shot 行。"""

        self.project_ids.append(project_id)
        self.db.add(Project(id=project_id, title=project_id))
        for index, (storyboard, video) in enumerate(zip(storyboards, videos), start=1):
            self.db.add(
                Shot(
                    id=f"shot-{index}",
                    project_id=project_id,
                    sequence=index,
                    version=1,
                    duration=3.0,
                    dialogue="",
                    storyboard_path=storyboard,
                    image_path=storyboard,
                    video_path=video,
                )
            )
        self.db.commit()

    def test_auto_run_completes_degraded_without_vision_model(self) -> None:
        import types

        from agent.checkpoints import CheckpointStore
        from api.routes import render as render_route
        from api.routes import script as script_route
        from tests.test_agent_e2e_integration import _base_state, _parsed_payload, _storyboard_payload

        project_id = f"{self.prefix}_complete"
        images = [_noise_image(self.root / "auto-1.png"), _noise_image(self.root / "auto-2.png")]
        videos = [_make_video("auto-1.mp4"), _make_video("auto-2.mp4")]
        self._seed(project_id, images, [str(item) for item in videos])
        store = CheckpointStore(project_id, "auto", root=self.root)

        async def parser(state: dict) -> dict:
            return _parsed_payload()

        async def storyboard(state: dict) -> dict:
            return _storyboard_payload()

        async def image_worker(shot_id: str, version: int, **kwargs) -> dict:
            return {
                "shot_id": shot_id, "shot_version": version, "status": "succeeded",
                "path": images[0] if shot_id == "shot-1" else images[1],
                "provider": "mock-image",
            }

        async def video_worker(shot_id: str, version: int, **kwargs) -> dict:
            path = str(videos[0] if shot_id == "shot-1" else videos[1])
            return {
                "shot_id": shot_id, "shot_version": version, "status": "succeeded",
                "path": path, "video_path": path, "provider": "mock-video", "model": "mock-video-v1",
            }

        initial = _base_state(project_id, run_id="auto", output_format="9:16", resolution="720p")
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(agent_nodes, "script_parser", types.SimpleNamespace(run=parser), create=True),
            patch.object(agent_nodes, "storyboard_gen", types.SimpleNamespace(run=storyboard), create=True),
            patch.object(script_route, "_ensure_character_reference_images", new=AsyncMock()),
            patch.object(script_route, "_ensure_scene_baseline_images", new=AsyncMock()),
            patch.object(graph, "_persist_phase1_idempotent", new=AsyncMock()),
            patch.object(graph, "refresh_project_reference_state_for_graph", return_value={"blocking": False}),
            patch.object(graph, "_reference_gate", return_value={}),
            patch.object(graph, "_reference_gate_for_state", return_value={}),
            patch.object(graph, "provider_profiles", lambda *a, **k: [types.SimpleNamespace(supports_reference_images=True, available=True)]),
            patch.object(graph, "generate_storyboard_shot", side_effect=image_worker),
            patch.object(graph, "generate_video_shot", side_effect=video_worker),
            patch.object(render_route, "_render_task", new=AsyncMock()),
            patch.object(render_route, "_render_status", {project_id: {"status": "completed", "video_path": str(videos[0])}}),
            # 关键：不 patch 质量门禁与 _review_shot_videos，只让能力如实报告 unsupported。
            patch.object(quality_review_service, "_vlm", _unsupported_vlm()),
            patch.object(quality_review_service, "_identity", None),
        ):
            result = asyncio.run(graph.get_graph().ainvoke(initial, config={"recursion_limit": 200}))

        # 闭环跑完：没有错误、没有人工卡点。
        self.assertEqual(result.get("errors"), [], f"自动闭环不应失败: {result.get('errors')}")
        self.assertFalse(result.get("needs_human_review"), "auto 模式不得请求人工审核")
        self.assertNotEqual(result.get("run_status"), "waiting_human")
        self.assertNotEqual(result.get("current_step"), "human_gate")
        self.assertEqual(result.get("stage_status", {}).get(StageName.FINAL_REVIEW.value), StageStatus.SUCCEEDED.value)

        # 视觉未验证：必须如实标记 pending + degraded，而不是伪装成质量通过。
        self.assertTrue(result.get("visual_quality_pending"), "缺视觉模型时必须标记视觉待审")
        self.assertTrue(result.get("degraded_published"))
        self.assertIn("pending", str(result.get("visual_pending_reason") or ""))
        final_report = result.get("final_report") or {}
        self.assertTrue(final_report, "必须生成最终报告")
        self.assertEqual(final_report["final_choice"]["visual_quality"]["status"], "pending")
        self.assertTrue(final_report["final_choice"]["degraded"])
        risks = [item for item in final_report["unresolved_risks"] if item["code"] == "visual_quality_pending"]
        self.assertTrue(risks, "最终报告必须列出视觉待审这一未解决风险")
        self.assertTrue(any(item["shot_ids"] for item in risks), "风险必须能定位到镜头")
        self.assertTrue(any(item["stage"] for item in risks), "风险必须能定位到阶段")

        # 追踪里不得出现任何人工卡点事件或人工等待状态。
        events = {str(item.get("event") or "") for item in (store.snapshot().get("events") or [])}
        self.assertNotIn("human_gate", events)
        summary = store.trace_summary()
        self.assertNotEqual(summary["run"]["status"], "waiting_human")
        for row in summary["stages"]:
            self.assertNotEqual(row["status"], StageStatus.WAITING_HUMAN.value, f"{row['stage']} 不得停在人工等待")
