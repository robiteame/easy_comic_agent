"""视频工作流 A/B 评估 fixture 流程与确定性报告回归测试。"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from config import settings  # noqa: E402
from services.video_ab_evaluation import (  # noqa: E402
    SHOT_TYPE_ORDER,
    STRATEGY_ORDER,
    VideoABEvaluationError,
    build_evaluation_report,
    load_evaluation_payload,
)
from test_environment import TEST_ROOT  # noqa: F401,E402

FIXTURE_PATH = _SERVER_DIR / "tests" / "fixtures" / "video_ab_evaluation" / "fixture.json"
CLI_PATH = _SERVER_DIR / "scripts" / "video_ab_evaluate.py"


def _run_cli(output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--input",
            str(FIXTURE_PATH),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


class VideoABFixtureFlowTests(unittest.TestCase):
    def test_fixture_cli_generates_readable_and_byte_reproducible_reports(self) -> None:
        with tempfile.TemporaryDirectory() as first_name, tempfile.TemporaryDirectory() as second_name:
            first = Path(first_name)
            second = Path(second_name)
            first_run = _run_cli(first)
            second_run = _run_cli(second)

            self.assertIn("VIDEO_AB_EVALUATION_OK", first_run.stdout)
            self.assertIn("VIDEO_AB_EVALUATION_OK", second_run.stdout)
            self.assertEqual(
                (first / "video_ab_evaluation.json").read_bytes(),
                (second / "video_ab_evaluation.json").read_bytes(),
            )
            self.assertEqual(
                (first / "video_ab_evaluation.md").read_bytes(),
                (second / "video_ab_evaluation.md").read_bytes(),
            )

            report = json.loads((first / "video_ab_evaluation.json").read_text(encoding="utf-8"))
            markdown = (first / "video_ab_evaluation.md").read_text(encoding="utf-8")

        self.assertEqual(report["evaluation_id"], "fixture-video-ab-001")
        self.assertEqual(report["batch_id"], "fixture-batch-2026-10")
        self.assertTrue(report["comparison_scope"]["complete_matrix"])
        self.assertEqual(tuple(report["comparison_scope"]["strategy_order"]), STRATEGY_ORDER)
        self.assertEqual(tuple(report["comparison_scope"]["shot_type_order"]), SHOT_TYPE_ORDER)
        self.assertEqual(report["shot_count"], 5)
        self.assertEqual(len(report["results"]), 20)

        by_key = {(item["shot_id"], item["strategy_id"]): item for item in report["results"]}
        record = by_key[("dialogue_001", "C")]
        self.assertEqual(record["provider"], "fixture-r2v")
        self.assertEqual(record["model"], "fixture-r2v-1.0")
        self.assertEqual(record["cost"], {"cost_known": True, "cost_micro": 160000, "currency": "CNY"})
        self.assertEqual(record["elapsed_ms"], 5900)
        self.assertEqual(record["output_path"], "fixture/dialogue_001/C/output.mp4")
        self.assertEqual(len(record["reference_manifest"]), 4)

        self.assertEqual(report["strategies"]["A"]["overall"]["metrics"]["rates"]["structural_pass_rate"], 0.8)
        self.assertEqual(report["strategies"]["B"]["overall"]["metrics"]["rates"]["freeze_rate"], 0.0)
        self.assertEqual(report["strategies"]["B"]["overall"]["metrics"]["manual_acceptance"]["pending_count"], 1)
        self.assertEqual(report["strategies"]["C"]["overall"]["metrics"]["rates"]["first_frame_match_score"], 0.874)
        self.assertIn("### 对白（`dialogue`）", markdown)
        self.assertIn("### 跨场景（`cross_scene`）", markdown)
        self.assertIn("## 执行审计", markdown)

    def test_same_execution_plan_is_enforced_for_every_strategy(self) -> None:
        payload = load_evaluation_payload(FIXTURE_PATH)
        payload = copy.deepcopy(payload)
        payload["results"][1]["execution_plan_hash"] = "different-plan"

        with self.assertRaisesRegex(VideoABEvaluationError, "使用了不同执行计划"):
            build_evaluation_report(payload)

    def test_incomplete_strategy_matrix_is_rejected(self) -> None:
        payload = copy.deepcopy(load_evaluation_payload(FIXTURE_PATH))
        payload["results"] = [
            item for item in payload["results"] if not (item["shot_id"] == "cross_001" and item["strategy_id"] == "D")
        ]

        with self.assertRaisesRegex(VideoABEvaluationError, "四策略必须覆盖同一批镜头"):
            build_evaluation_report(payload)

    def test_report_generation_does_not_change_default_provider_or_quality_threshold(self) -> None:
        provider_before = settings.VIDEO_PROVIDER
        threshold_before = settings.QUALITY_REVIEW_PASS_SCORE

        report = build_evaluation_report(load_evaluation_payload(FIXTURE_PATH))

        self.assertEqual(settings.VIDEO_PROVIDER, provider_before)
        self.assertEqual(settings.QUALITY_REVIEW_PASS_SCORE, threshold_before)
        self.assertFalse(report["policy_safety"]["auto_policy_mutation_enabled"])
        self.assertFalse(report["policy_safety"]["default_provider_changed"])
        self.assertFalse(report["policy_safety"]["quality_threshold_changed"])
        self.assertEqual(report["policy_safety"]["applied_policy_changes"], [])


if __name__ == "__main__":
    unittest.main()
