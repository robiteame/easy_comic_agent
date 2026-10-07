"""视频质量检查（结构/技术/视觉待审）回归测试。

覆盖四类真实媒体：有效视频、黑帧视频、冻结视频、音视频时长不一致，
以及时长-执行计划、分辨率/比例、尾帧、不可播放与 Critic 映射行为。
全部检查只用 ffprobe/ffmpeg 本地能力，不接入任何外部视觉模型。
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

from agent import graph  # noqa: E402
from agent.contracts import (  # noqa: E402
    STAGE_CONTRACTS,
    FailureKind,
    ShotArtifact,
    StageName,
    StageStatus,
)
from agent.critic import critique_videos  # noqa: E402
from services.structural_validation import (  # noqa: E402
    probe_media_duration_sync,
    validate_video_file,
    validate_video_sync,
)
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402

FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
MEDIA_ROOT = TEST_ROOT / "video-quality-checks"


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True, capture_output=True)


def _make_valid_video(name: str = "valid.mp4", *, size: str = "540x960", duration: int = 4, audio: bool = True) -> Path:
    path = MEDIA_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=15:duration={duration}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-shortest", "-c:a", "aac"]
    _ffmpeg(*cmd, "-pix_fmt", "yuv420p", "-c:v", "libx264", str(path))
    return path


def _make_static_video(color: str, name: str) -> Path:
    path = MEDIA_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"color={color}:size=540x960:rate=15:duration=4",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(path),
    )
    return path


def _make_av_mismatch_video(name: str = "av_mismatch.mp4") -> Path:
    path = MEDIA_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=540x960:rate=15:duration=3",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=6",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(path),
    )
    return path


def _make_long_audio(name: str = "long_tts.wav") -> Path:
    path = MEDIA_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=300:duration=6", str(path))
    return path


def _make_short_audio(name: str = "short_tts.wav") -> Path:
    path = MEDIA_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=300:duration=1", str(path))
    return path


def _extract_tail_frame(video: Path, name: str = "tail.png") -> Path:
    path = MEDIA_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-sseof", "-0.1", "-i", str(video), "-frames:v", "1", str(path))
    return path


def _codes(report: dict, category: str) -> list[str]:
    return [str(item.get("code")) for item in report["categories"][category]["issues"]]


@unittest.skipUnless(FFMPEG_AVAILABLE, "需要本地 ffmpeg/ffprobe")
class VideoValidatorCategoryTests(unittest.TestCase):
    """三分类结构与每项廉价检查的行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.valid = _make_valid_video()
        cls.black = _make_static_video("black", "black.mp4")
        cls.frozen = _make_static_video("red", "frozen.mp4")
        cls.av_mismatch = _make_av_mismatch_video()
        cls.long_audio = _make_long_audio()
        cls.short_audio = _make_short_audio()
        cls.tail = _extract_tail_frame(cls.valid)

    def test_valid_video_passes_structure_and_technical_but_visual_stays_pending(self) -> None:
        import asyncio

        report = asyncio.run(
            validate_video_file(
                str(self.valid),
                expected_duration_s=4.0,
                expected_aspect_ratio=9 / 16,
                audio_duration_s=3.5,
                tail_frame_path=str(self.tail),
            )
        )
        self.assertTrue(report["passed"])
        self.assertTrue(report["categories"]["structural_validity"]["passed"])
        self.assertTrue(report["categories"]["technical_quality"]["passed"])
        # 结构与技术通过 ≠ 视觉质量通过：视觉维度必须保持待审。
        pending = report["categories"]["visual_quality_pending"]
        self.assertEqual(pending["status"], "pending")
        self.assertIsNone(pending["passed"])
        self.assertIn("未接入", pending["reason"])
        self.assertTrue(report.get("checks", {}).get("decoded_ok"))
        self.assertTrue(report.get("tail_frame_ok"))

    def test_black_video_passes_structural_but_fails_technical(self) -> None:
        import asyncio

        report = asyncio.run(validate_video_file(str(self.black)))
        self.assertTrue(report["categories"]["structural_validity"]["passed"], "黑帧视频在结构层仍应可播放")
        self.assertFalse(report["passed"])
        self.assertIn("video_black_frames", _codes(report, "technical_quality"))

    def test_frozen_video_is_detected(self) -> None:
        import asyncio

        report = asyncio.run(validate_video_file(str(self.frozen)))
        self.assertFalse(report["passed"])
        self.assertIn("video_frozen", _codes(report, "technical_quality"))
        self.assertGreaterEqual(report["frame_scan"]["freeze_seconds"], 2.0)

    def test_muxed_audio_longer_than_picture_is_detected(self) -> None:
        import asyncio

        report = asyncio.run(validate_video_file(str(self.av_mismatch)))
        self.assertFalse(report["passed"])
        self.assertIn("audio_exceeds_picture", _codes(report, "technical_quality"))
        self.assertIsNotNone(report["video_duration_seconds"])
        self.assertGreater(report["audio"]["duration_seconds"], report["video_duration_seconds"])

    def test_external_audio_longer_than_picture_is_detected(self) -> None:
        import asyncio

        audio_duration = probe_media_duration_sync(str(self.long_audio))
        self.assertGreater(audio_duration, 5.0)
        report = asyncio.run(validate_video_file(str(self.valid), audio_duration_s=audio_duration))
        self.assertFalse(report["passed"])
        self.assertIn("audio_exceeds_picture", _codes(report, "technical_quality"))

    def test_external_audio_shorter_than_picture_is_detected(self) -> None:
        import asyncio

        audio_duration = probe_media_duration_sync(str(self.short_audio))
        self.assertLess(audio_duration, 2.0)
        report = asyncio.run(validate_video_file(str(self.valid), audio_duration_s=audio_duration))
        self.assertFalse(report["passed"])
        self.assertIn("audio_shorter_than_picture", _codes(report, "technical_quality"))

    def test_missing_frame_context_is_not_reported_as_overall_pass(self) -> None:
        import asyncio

        report = asyncio.run(validate_video_file(str(self.valid), deep_scan=False))
        self.assertIsNone(report["categories"]["technical_quality"]["passed"])
        self.assertFalse(report["passed"])
        self.assertIn("frame_scan", report["categories"]["technical_quality"]["skipped"])

        import asyncio

        shorter = asyncio.run(validate_video_file(str(self.valid), expected_duration_s=5.0))
        self.assertIn("video_duration_shorter_than_plan", _codes(shorter, "technical_quality"))
        longer = asyncio.run(validate_video_file(str(self.valid), expected_duration_s=3.5))
        self.assertNotIn("video_duration_shorter_than_plan", _codes(longer, "technical_quality"))

    def test_resolution_below_minimum_is_reported(self) -> None:
        import asyncio

        low = _make_valid_video("lowres.mp4", size="180x320", audio=False)
        report = asyncio.run(validate_video_file(str(low)))
        self.assertIn("video_resolution_below_minimum", _codes(report, "technical_quality"))

    def test_aspect_ratio_mismatch_is_reported_with_recovery(self) -> None:
        import asyncio

        report = asyncio.run(validate_video_file(str(self.valid), expected_aspect_ratio=16 / 9))
        codes = _codes(report, "technical_quality")
        self.assertIn("video_aspect_mismatch", codes)
        entry = next(
            item
            for item in report["categories"]["technical_quality"]["issues"]
            if item["code"] == "video_aspect_mismatch"
        )
        self.assertTrue(entry["recommendation"])

    def test_slight_aspect_deviation_warns_without_failing(self) -> None:
        import asyncio

        # 9:16=0.5625，预期 0.58 偏差约 3%，属于 warning 区间。
        report = asyncio.run(validate_video_file(str(self.valid), expected_aspect_ratio=0.58))
        self.assertNotIn("video_aspect_mismatch", _codes(report, "technical_quality"))
        warning_codes = [item["code"] for item in report.get("warnings", [])]
        self.assertIn("video_aspect_slight_mismatch", warning_codes)
        self.assertTrue(report["passed"])
        # Critic 把 warning 透出为非阻断 issue。
        critique = critique_videos([{"shot_id": "shot-slight", "path": str(self.valid), "expected_aspect_ratio": 0.58}])
        warning_issues = [issue for issue in critique.issues if issue.code == "video_aspect_slight_mismatch"]
        self.assertTrue(warning_issues)
        self.assertEqual(warning_issues[0].severity, "warning")
        self.assertTrue(warning_issues[0].recommendation)
        self.assertTrue(critique.passed, "轻微比例偏差不得阻断阶段通过")

    def test_unplayable_file_fails_structural_validity(self) -> None:
        import asyncio

        broken = MEDIA_ROOT / "broken.mp4"
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_bytes(b"\x00" * 9000)
        report = asyncio.run(validate_video_file(str(broken)))
        self.assertFalse(report["passed"])
        self.assertFalse(report["categories"]["structural_validity"]["passed"])
        self.assertTrue(_codes(report, "structural_validity"))

    def test_first_frame_context_is_validated(self) -> None:
        import asyncio

        missing = MEDIA_ROOT / "missing-first.png"
        report = asyncio.run(validate_video_file(str(self.valid), first_frame_path=str(missing)))
        self.assertFalse(report["passed"])
        self.assertIn("first_frame_missing", _codes(report, "technical_quality"))

        import asyncio

        report = asyncio.run(validate_video_file(str(self.valid), tail_frame_path=str(MEDIA_ROOT / "missing.png")))
        self.assertFalse(report["passed"])
        self.assertIn("tail_frame_missing", _codes(report, "technical_quality"))

    def test_every_issue_has_code_and_recommendation(self) -> None:
        import asyncio

        for target, kwargs in (
            (self.black, {}),
            (self.frozen, {}),
            (self.av_mismatch, {}),
            (self.valid, {"expected_duration_s": 9.0, "tail_frame_path": str(MEDIA_ROOT / "none.png")}),
        ):
            report = asyncio.run(validate_video_file(str(target), **kwargs))
            for category in ("structural_validity", "technical_quality"):
                for entry in report["categories"][category]["issues"]:
                    self.assertTrue(entry.get("code"), f"{category} 问题缺少 code: {entry}")
                    self.assertTrue(entry.get("recommendation"), f"{entry.get('code')} 缺少恢复建议")

    def test_sync_wrapper_is_safe_inside_running_loop(self) -> None:
        import asyncio

        async def run() -> dict:
            return validate_video_sync(str(self.valid))

        report = asyncio.run(run())
        self.assertTrue(report["passed"])


@unittest.skipUnless(FFMPEG_AVAILABLE, "需要本地 ffmpeg/ffprobe")
class CriticVideoReportTests(unittest.TestCase):
    """critique_videos 的三维度指标、issue code 与诚实性。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.valid = _make_valid_video("critic-valid.mp4")
        cls.black = _make_static_video("black", "critic-black.mp4")

    def test_valid_artifacts_pass_without_claiming_visual_quality(self) -> None:
        report = critique_videos(
            [
                {
                    "shot_id": "shot-1",
                    "path": str(self.valid),
                    "expected_duration_s": 4.0,
                    "expected_aspect_ratio": 9 / 16,
                }
            ]
        )
        self.assertTrue(report.passed)
        names = {metric.name for metric in report.metrics}
        self.assertIn("structural_validity", names)
        self.assertIn("technical_quality", names)
        self.assertIn("visual_quality_pending", names)
        pending = next(metric for metric in report.metrics if metric.name == "visual_quality_pending")
        self.assertIsNone(pending.passed, "视觉维度不得输出通过结论")
        # 未接入识别模型时不得宣称角色一致性/连续性通过。
        self.assertNotIn("character_consistency", names)
        self.assertNotIn("continuity", names)
        pending_issues = [issue for issue in report.issues if issue.code == "visual_quality_pending"]
        self.assertTrue(pending_issues and pending_issues[0].severity == "info")

    def test_black_frame_artifact_fails_with_issue_code_and_recovery(self) -> None:
        report = critique_videos([{"shot_id": "shot-2", "path": str(self.black)}])
        self.assertFalse(report.passed)
        issue = next(issue for issue in report.issues if issue.code == "video_black_frames")
        self.assertEqual(issue.shot_id, "shot-2")
        self.assertTrue(issue.recommendation)
        structural = next(metric for metric in report.metrics if metric.name == "structural_validity")
        technical = next(metric for metric in report.metrics if metric.name == "technical_quality")
        self.assertTrue(structural.passed)
        self.assertFalse(technical.passed)
        self.assertTrue(any("video_black_frames" in change for change in report.proposed_changes))

    def test_missing_file_reports_structural_issue_code(self) -> None:
        report = critique_videos([{"shot_id": "shot-3", "path": str(MEDIA_ROOT / "ghost.mp4")}])
        self.assertFalse(report.passed)
        codes = {issue.code for issue in report.issues}
        self.assertTrue(codes & {"video_file_missing", "video_unreadable", "video_not_playable"})

    def test_generation_failure_kept_as_independent_issue(self) -> None:
        report = critique_videos(
            [
                {
                    "shot_id": "shot-4",
                    "path": "",
                    "failure": {"kind": "video_failed", "stage": "video_generation", "message": "provider down"},
                }
            ]
        )
        self.assertFalse(report.passed)
        self.assertTrue(any(issue.code == "video_generation_failure" for issue in report.issues))

    def test_empty_artifacts_do_not_claim_any_quality(self) -> None:
        report = critique_videos([])
        self.assertEqual(report.issues, [])
        names = {metric.name for metric in report.metrics}
        self.assertIn("visual_quality_pending", names)


class GraphWiringTests(unittest.TestCase):
    """阶段契约与失败码映射。"""

    def test_video_stage_contract_declares_three_dimensions(self) -> None:
        metrics = STAGE_CONTRACTS[StageName.VIDEO_GENERATION].quality_metrics
        for name in ("structural_validity", "technical_quality", "visual_quality_pending"):
            self.assertIn(name, metrics)
        self.assertNotIn("motion_quality", metrics)
        self.assertNotIn("continuity", metrics)

    def test_video_review_node_meta_describes_three_dimensions(self) -> None:
        meta = graph.GRAPH_NODE_META["video_review"]["description"]
        for keyword in ("structural_validity", "technical_quality", "visual_quality_pending"):
            self.assertIn(keyword, meta)

    def test_new_issue_codes_map_to_recovery_kinds(self) -> None:
        critique = {
            "issues": [
                {"code": "video_black_frames", "severity": "error", "message": "黑帧", "shot_id": "s1"},
            ]
        }
        failure = graph._failure_from_critique(StageName.VIDEO_GENERATION, critique)
        self.assertIsNotNone(failure)
        self.assertEqual(failure.kind, FailureKind.VIDEO_FAILED)

        tail_critique = {
            "issues": [
                {"code": "tail_frame_missing", "severity": "error", "message": "尾帧缺失", "shot_id": "s1"},
            ]
        }
        tail_failure = graph._failure_from_critique(StageName.VIDEO_GENERATION, tail_critique)
        self.assertEqual(tail_failure.kind, FailureKind.STORAGE_FAILED)

    def test_shot_artifact_carries_check_context(self) -> None:
        artifact = ShotArtifact(
            shot_id="shot-ctx",
            shot_version=1,
            stage=StageName.VIDEO_GENERATION,
            status=StageStatus.SUCCEEDED,
            path="/output/ctx.mp4",
            expected_duration_s=5.0,
            expected_aspect_ratio=0.5625,
            audio_path="/output/ctx.wav",
            tail_frame_path="/output/ctx_tail.png",
        )
        dumped = artifact.model_dump(mode="json")
        self.assertEqual(dumped["expected_duration_s"], 5.0)
        self.assertEqual(dumped["tail_frame_path"], "/output/ctx_tail.png")


if __name__ == "__main__":
    unittest.main()
