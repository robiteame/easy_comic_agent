"""结果反思（Critic）与恢复决策（Decision）机制的验收。

覆盖：
- CritiqueReport 必须携带完整反思字段（失败分类、证据、受影响镜头、建议策略等）；
- 13 类失败分类的枚举与消息分类；
- Decision 生成多候选（失败类型/质量分/候选结果/Provider 能力/预算/剩余重试）；
- 候选字段完整性与结构化 Prompt patch；
- 按综合收益选择且预算允许，禁止无条件 RETRY；
- DecisionTrace 记录候选、淘汰原因与最终选择；
- 自动模式无法继续时 degraded_publish / terminal_failure，只有 manual 可选 human_review；
- 图接线：degraded_publish 节点、路由与兼容节点的条件重试。
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from agent import graph  # noqa: E402
from agent.contracts import (  # noqa: E402
    CritiqueReport,
    FailureKind,
    FailureRecord,
    RecoveryCandidate,
    RecoveryStrategy,
    StageName,
)
from agent.critic import (  # noqa: E402
    MAX_DIALOGUE_CHARS_PER_SHOT,
    critique_assets,
    critique_audio,
    critique_compose,
    critique_director,
    critique_final,
    critique_images,
    critique_llm_failure,
    critique_storyboard,
    critique_videos,
)
from agent.decision import (  # noqa: E402
    choose_recovery,
    classify_failure,
    recovery_candidates,
)

_UNLIMITED_BUDGET = {"remaining_cost_micro": 10**9, "remaining_seconds": 10**6, "unlimited": True}
_ZERO_BUDGET = {"remaining_cost_micro": 0, "remaining_seconds": 0}

REQUIRED_FAILURE_KINDS = {
    "llm_invalid_output": "输出 JSON 结构无法解析",
    "image_failed": "image provider 生成失败",
    "video_failed": "seedance 视频任务失败",
    "audio_failed": "tts 配音 voice 失败",
    "dialogue_too_long": "镜头对白过长无法配音",
    "shot_too_complex": "镜头过于复杂包含多个动作节拍",
    "provider_reference_unsupported": "当前 provider 不支持参考图 references_sent=0",
    "provider_capability_mismatch": "provider capability mismatch 不支持该分辨率",
    "quality_below_threshold": "整体质量低于阈值",
    "version_conflict": "检测到 stale version 过期",
    "budget_exceeded": "项目预算 quota 已耗尽",
    "storage_failed": "存储写入失败 尾帧缺失",
    "timeout": "上游请求 timeout 超时",
}

REQUIRED_CRITIQUE_FIELDS = (
    "stage",
    "score",
    "metrics",
    "issues",
    "evidence",
    "failure_kind",
    "recoverable",
    "proposed_changes",
    "affected_shot_ids",
    "recommended_strategy",
)


def _failure(
    kind: FailureKind, stage: StageName = StageName.IMAGE_GENERATION, shot_id: str = "s1", message: str = "boom"
) -> FailureRecord:
    return FailureRecord(kind=kind, stage=stage, shot_id=shot_id, message=message, provider="failing-provider")


class FailureClassificationTests(unittest.TestCase):
    def test_all_required_failure_kinds_exist_and_classify_messages(self) -> None:
        for value, message in REQUIRED_FAILURE_KINDS.items():
            self.assertEqual(FailureKind(value).value, value)
            record = classify_failure(stage=StageName.IMAGE_GENERATION, message=message)
            self.assertEqual(record.kind, FailureKind(value), f"消息应分类为 {value}: {message}")

    def test_non_recoverable_kinds_are_not_retryable(self) -> None:
        for kind in (FailureKind.BUDGET_EXCEEDED, FailureKind.CANCELLED):
            record = classify_failure(stage=StageName.VIDEO_GENERATION, message="x", kind=kind)
            self.assertFalse(record.retryable)


class CriticReflectionTests(unittest.TestCase):
    def _all_critique_reports(self) -> list[CritiqueReport]:
        return [
            critique_director({"characters": [{"name": "林夏"}], "script_scenes": [{"id": "scene-1"}]}),
            critique_storyboard(
                {"shots": [{"shot_id": "s1", "dialogue": "长" * (MAX_DIALOGUE_CHARS_PER_SHOT + 1), "duration": 3.0}]}
            ),
            critique_assets({"characters": [], "script_scenes": []}, reference_supported=True),
            critique_images([{"shot_id": "s1", "path": "", "score": 0.2}]),
            critique_videos([{"shot_id": "s1", "path": ""}]),
            critique_audio([{"shot_id": "s1", "dialogue": "你好"}], []),
            critique_compose("p1", [{"shot_id": "s1", "video_path": ""}], ""),
            critique_final({"shot_artifacts": [], "quality_threshold": 0.72}),
            critique_llm_failure("JSON 解析失败", stage=StageName.DIRECTOR_PLANNING),
        ]

    def test_every_critique_returns_full_reflection_fields(self) -> None:
        for report in self._all_critique_reports():
            dump = report.model_dump()
            for field in REQUIRED_CRITIQUE_FIELDS:
                self.assertIn(field, dump)
            self.assertTrue(report.metrics or report.evidence, "报告必须带指标或证据")
            self.assertIsInstance(report.recoverable, bool)

    def test_dialogue_failure_is_classified_with_affected_shot_and_strategy(self) -> None:
        report = critique_storyboard(
            {"shots": [{"shot_id": "s9", "dialogue": "长" * (MAX_DIALOGUE_CHARS_PER_SHOT + 1), "duration": 3.0}]}
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.failure_kind, FailureKind.DIALOGUE_TOO_LONG)
        self.assertEqual(report.affected_shot_ids, ["s9"])
        self.assertEqual(report.recommended_strategy, RecoveryStrategy.SPLIT_SHOT)
        self.assertTrue(report.recoverable)
        self.assertTrue(any(entry.get("kind") == "metric" for entry in report.evidence))
        self.assertTrue(any(entry.get("kind") == "issue" for entry in report.evidence))

    def test_media_failures_map_to_provider_failure_kinds(self) -> None:
        image = critique_images([{"shot_id": "i1", "path": "", "score": 0.1}])
        self.assertEqual(image.failure_kind, FailureKind.IMAGE_FAILED)
        self.assertEqual(image.recommended_strategy, RecoveryStrategy.REGENERATE_FAILED_SHOTS)
        self.assertEqual(image.affected_shot_ids, ["i1"])
        video = critique_videos([{"shot_id": "v1", "path": ""}])
        self.assertEqual(video.failure_kind, FailureKind.VIDEO_FAILED)
        audio = critique_audio([{"shot_id": "a1", "dialogue": "台词"}], [])
        self.assertEqual(audio.failure_kind, FailureKind.AUDIO_FAILED)
        compose = critique_compose("p", [{"shot_id": "c1", "video_path": ""}], "")
        self.assertEqual(compose.failure_kind, FailureKind.STORAGE_FAILED)

    def test_llm_failure_critique_classifies_invalid_output(self) -> None:
        report = critique_llm_failure("模型输出 JSON 非法", stage=StageName.STORYBOARD_DESIGN)
        self.assertEqual(report.failure_kind, FailureKind.LLM_INVALID_OUTPUT)
        self.assertFalse(report.passed)
        self.assertEqual(report.recommended_strategy, RecoveryStrategy.REVISE_PROMPT)
        other = critique_llm_failure("模型服务暂时不可用", stage=StageName.STORYBOARD_DESIGN)
        self.assertEqual(other.failure_kind, FailureKind.UNKNOWN)


class RecoveryCandidateGenerationTests(unittest.TestCase):
    def test_candidates_are_complete_and_multiple(self) -> None:
        candidates = recovery_candidates(
            _failure(FailureKind.IMAGE_FAILED),
            budget=_UNLIMITED_BUDGET,
            retries_remaining=2,
            quality_score=0.6,
            provider_profiles_by_capability={},
        )
        self.assertGreaterEqual(len({item.strategy for item in candidates}), 5)
        for candidate in candidates:
            self.assertIsNotNone(candidate.estimated_cost_micro)
            self.assertIsNotNone(candidate.estimated_seconds)
            self.assertGreaterEqual(candidate.quality_gain, 0.0)
            self.assertIsInstance(candidate.provider_capability_ok, bool)
            self.assertIsInstance(candidate.budget_fit, bool)
            self.assertEqual(candidate.retries_remaining, 2)
            self.assertTrue(candidate.rationale)
        self.assertTrue(all(item.budget_fit for item in candidates))

    def test_unknown_failure_never_proposes_unconditional_retry(self) -> None:
        candidates = recovery_candidates(
            _failure(FailureKind.UNKNOWN), budget=_UNLIMITED_BUDGET, provider_profiles_by_capability={}
        )
        strategies = {item.strategy for item in candidates}
        self.assertNotIn(RecoveryStrategy.RETRY, strategies)
        timeout = recovery_candidates(
            _failure(FailureKind.TIMEOUT), budget=_UNLIMITED_BUDGET, provider_profiles_by_capability={}
        )
        self.assertIn(RecoveryStrategy.RETRY, {item.strategy for item in timeout})

    def test_prompt_modifications_are_structured_patches(self) -> None:
        candidates = recovery_candidates(
            _failure(FailureKind.LLM_INVALID_OUTPUT, stage=StageName.DIRECTOR_PLANNING),
            budget=_UNLIMITED_BUDGET,
            provider_profiles_by_capability={},
        )
        revise = next(item for item in candidates if item.strategy is RecoveryStrategy.REVISE_PROMPT)
        self.assertTrue(revise.prompt_patches)
        for patch_entry in revise.prompt_patches:
            self.assertTrue(patch_entry.field)
            self.assertTrue(patch_entry.op)
        fields = {patch_entry.field for patch_entry in revise.prompt_patches}
        self.assertIn("output_format", fields)
        self.assertEqual(revise.prompt_changes["shot_id"], "s1")
        self.assertEqual(
            revise.prompt_changes["patches"],
            [patch_entry.model_dump(mode="json") for patch_entry in revise.prompt_patches],
        )
        split = next(
            item
            for item in recovery_candidates(
                _failure(FailureKind.DIALOGUE_TOO_LONG, stage=StageName.STORYBOARD_DESIGN),
                budget=_UNLIMITED_BUDGET,
                provider_profiles_by_capability={},
            )
            if item.strategy is RecoveryStrategy.SPLIT_SHOT
        )
        self.assertEqual(split.prompt_patches[0].field, "dialogue")
        self.assertEqual(split.prompt_patches[0].op, "split")

    def test_candidate_results_boost_provider_switch_and_regen(self) -> None:
        all_failed = [{"status": "failed", "provider": "p1"}, {"status": "failed", "provider": "p1"}]
        candidates = recovery_candidates(
            _failure(FailureKind.VIDEO_FAILED, stage=StageName.VIDEO_GENERATION),
            budget=_UNLIMITED_BUDGET,
            provider_profiles_by_capability={},
            candidate_results=all_failed,
        )
        switch = next(item for item in candidates if item.strategy is RecoveryStrategy.SWITCH_PROVIDER)
        self.assertGreater(switch.quality_gain, 0.30)  # 同 Provider 全失败证据提升切换收益
        some_success = [{"status": "succeeded", "provider": "p1"}, {"status": "failed", "provider": "p1"}]
        regen_candidates = recovery_candidates(
            _failure(FailureKind.VIDEO_FAILED, stage=StageName.VIDEO_GENERATION),
            budget=_UNLIMITED_BUDGET,
            provider_profiles_by_capability={},
            candidate_results=some_success,
        )
        regen = next(item for item in regen_candidates if item.strategy is RecoveryStrategy.REGENERATE_FAILED_SHOTS)
        self.assertGreater(regen.quality_gain, 0.25)


class RecoverySelectionTests(unittest.TestCase):
    def test_selects_highest_scoring_feasible_candidate_and_records_rejections(self) -> None:
        candidates = [
            RecoveryCandidate(
                strategy=RecoveryStrategy.SWITCH_PROVIDER,
                quality_gain=0.5,
                estimated_cost_micro=999_999,
                estimated_seconds=500,
                provider_capability_ok=True,
                budget_fit=False,
            ),
            RecoveryCandidate(
                strategy=RecoveryStrategy.REVISE_PROMPT,
                quality_gain=0.3,
                estimated_cost_micro=100,
                estimated_seconds=10,
                provider_capability_ok=True,
                budget_fit=True,
            ),
            RecoveryCandidate(
                strategy=RecoveryStrategy.LOWER_RESOLUTION,
                quality_gain=0.1,
                estimated_cost_micro=100,
                estimated_seconds=10,
                provider_capability_ok=False,
                budget_fit=True,
            ),
        ]
        trace = choose_recovery(
            _failure(FailureKind.IMAGE_FAILED),
            stage=StageName.IMAGE_GENERATION,
            candidates=candidates,
            mode="auto",
            retries_remaining=2,
            quality_score=0.6,
        )
        self.assertIsNotNone(trace.selected)
        self.assertEqual(trace.selected.strategy, RecoveryStrategy.REVISE_PROMPT)
        rejected = {item["strategy"]: item["reason"] for item in trace.considered_rejected}
        self.assertIn("预算", rejected[RecoveryStrategy.SWITCH_PROVIDER.value])
        self.assertIn("能力", rejected[RecoveryStrategy.LOWER_RESOLUTION.value])
        self.assertEqual(trace.mode, "auto")
        self.assertEqual(trace.retries_remaining, 2)
        self.assertEqual(trace.quality_score, 0.6)
        self.assertTrue(trace.reason)

    def test_exhausted_retries_in_auto_mode_choose_degraded_or_terminal(self) -> None:
        # 无任何可用部分结果 -> terminal_failure
        trace = choose_recovery(
            _failure(FailureKind.IMAGE_FAILED),
            stage=StageName.IMAGE_GENERATION,
            mode="auto",
            retries_remaining=0,
            budget=_ZERO_BUDGET,
            provider_profiles_by_capability={},
        )
        self.assertEqual(trace.selected.strategy, RecoveryStrategy.TERMINAL_FAILURE)
        # 未提供结构证据时，质量分不能冒充可发布产物 -> terminal
        degraded = choose_recovery(
            _failure(FailureKind.IMAGE_FAILED),
            stage=StageName.IMAGE_GENERATION,
            mode="auto",
            retries_remaining=0,
            budget=_ZERO_BUDGET,
            provider_profiles_by_capability={},
            critique={"stage": "image_generation", "passed": False, "score": 0.45},
        )
        self.assertEqual(degraded.selected.strategy, RecoveryStrategy.TERMINAL_FAILURE)
        structurally_complete = choose_recovery(
            _failure(FailureKind.IMAGE_FAILED),
            stage=StageName.IMAGE_GENERATION,
            mode="auto",
            retries_remaining=0,
            budget=_ZERO_BUDGET,
            provider_profiles_by_capability={},
            candidate_results=[{"status": "succeeded", "path": "/tmp/x.png", "structural_passed": True}],
        )
        self.assertEqual(structurally_complete.selected.strategy, RecoveryStrategy.DEGRADED_PUBLISH)
        for item in degraded.considered_rejected:
            if item["strategy"] == RecoveryStrategy.HUMAN_REVIEW.value:
                self.assertIn("human_review", item["reason"])

    def test_human_review_is_only_selectable_in_manual_mode(self) -> None:
        manual = choose_recovery(
            _failure(FailureKind.IMAGE_FAILED),
            stage=StageName.IMAGE_GENERATION,
            mode="manual",
            retries_remaining=0,
            budget=_ZERO_BUDGET,
            provider_profiles_by_capability={},
        )
        self.assertEqual(manual.selected.strategy, RecoveryStrategy.HUMAN_REVIEW)
        auto = choose_recovery(
            _failure(FailureKind.IMAGE_FAILED),
            stage=StageName.IMAGE_GENERATION,
            mode="auto",
            retries_remaining=0,
            budget=_ZERO_BUDGET,
            provider_profiles_by_capability={},
        )
        self.assertNotEqual(auto.selected.strategy, RecoveryStrategy.HUMAN_REVIEW)

    def test_budget_exceeded_selects_terminal_without_new_spend(self) -> None:
        trace = choose_recovery(
            _failure(FailureKind.BUDGET_EXCEEDED, stage=StageName.VIDEO_GENERATION),
            stage=StageName.VIDEO_GENERATION,
            mode="auto",
            retries_remaining=3,
            budget=_ZERO_BUDGET,
            provider_profiles_by_capability={},
            candidate_results=[
                {"status": "succeeded", "provider": "p1", "path": "/tmp/video.mp4", "structural_passed": True}
            ],
        )
        self.assertEqual(trace.selected.strategy, RecoveryStrategy.DEGRADED_PUBLISH)
        self.assertEqual(trace.selected.estimated_cost_micro, 0)
        for item in trace.candidates:
            if item.strategy not in {
                RecoveryStrategy.DEGRADED_PUBLISH,
                RecoveryStrategy.TERMINAL_FAILURE,
                RecoveryStrategy.HUMAN_REVIEW,
            }:
                self.assertFalse(item.budget_fit)

    def test_selection_prefers_primary_strategy_for_failure_kind(self) -> None:
        trace = choose_recovery(
            _failure(FailureKind.PROVIDER_REFERENCE_UNSUPPORTED),
            stage=StageName.IMAGE_GENERATION,
            mode="auto",
            retries_remaining=2,
            budget=_UNLIMITED_BUDGET,
            provider_profiles_by_capability={},
        )
        # 无 Provider 画像时 switch 不可行，退而选择参考替换/改写而非终止。
        self.assertIn(trace.selected.strategy, {RecoveryStrategy.REPLACE_REFERENCE, RecoveryStrategy.REVISE_PROMPT})
        dialogue = choose_recovery(
            _failure(FailureKind.DIALOGUE_TOO_LONG, stage=StageName.STORYBOARD_DESIGN),
            stage=StageName.STORYBOARD_DESIGN,
            mode="auto",
            retries_remaining=2,
            budget=_UNLIMITED_BUDGET,
            provider_profiles_by_capability={},
        )
        self.assertEqual(dialogue.selected.strategy, RecoveryStrategy.REVISE_PROMPT)
        after_patch = choose_recovery(
            _failure(FailureKind.DIALOGUE_TOO_LONG, stage=StageName.STORYBOARD_DESIGN),
            stage=StageName.STORYBOARD_DESIGN,
            mode="auto",
            retries_remaining=1,
            attempted_strategies=[RecoveryStrategy.REVISE_PROMPT],
            budget=_UNLIMITED_BUDGET,
            provider_profiles_by_capability={},
        )
        self.assertEqual(after_patch.selected.strategy, RecoveryStrategy.SPLIT_SHOT)


class GraphRecoveryWiringTests(unittest.TestCase):
    def test_graph_has_degraded_publish_node_and_edges(self) -> None:
        drawable = graph.get_graph().get_graph()
        self.assertIn("degraded_publish", drawable.nodes)
        edges = {(edge.source, edge.target, edge.data) for edge in drawable.edges}
        self.assertIn(("image_decision", "degraded_publish", "degraded"), edges)
        self.assertIn(("degraded_publish", "__end__", None), edges)

    def test_decision_node_records_terminal_strategy_when_retries_exhausted(self) -> None:
        state = {
            "project_id": "p1",
            "run_id": "r1",
            "mode": "auto",
            "quality_profile": "standard",
            "critiques": [
                {
                    "stage": StageName.IMAGE_GENERATION.value,
                    "passed": False,
                    "score": 0.4,
                    "failure_kind": FailureKind.IMAGE_FAILED.value,
                }
            ],
            "shot_artifacts": [
                {
                    "stage": StageName.IMAGE_GENERATION.value,
                    "shot_id": "bad",
                    "status": "succeeded",
                    "path": "/tmp/good.png",
                    "structural_passed": True,
                }
            ],
            "recovery_attempts": {StageName.IMAGE_GENERATION.value: 9},
        }
        update = graph._decision_node(state, StageName.IMAGE_GENERATION, next_target="quality_review")
        self.assertEqual(update["selected_strategy"], RecoveryStrategy.DEGRADED_PUBLISH.value)
        routed = graph._route_decision({**state, **update}, StageName.IMAGE_GENERATION, "next", "recover")
        self.assertEqual(routed, "degraded")

    def test_degraded_publish_node_finishes_as_degraded_run(self) -> None:
        state = {
            "current_stage": StageName.IMAGE_GENERATION.value,
            "degraded_reason": "预算耗尽，保留已成功镜头",
            "shot_artifacts": [
                {"stage": StageName.IMAGE_GENERATION.value, "shot_id": "good", "status": "succeeded"},
                {"stage": StageName.IMAGE_GENERATION.value, "shot_id": "bad", "status": "failed"},
            ],
        }
        update = asyncio.run(graph._degraded_publish(state))
        self.assertEqual(update["run_status"], "degraded")
        self.assertTrue(update["degraded_published"])
        self.assertEqual(update["stage_status"][StageName.IMAGE_GENERATION.value], "degraded")
        self.assertEqual(update["successful_shot_ids"], ["good"])

    def test_recovery_node_auto_mode_never_waits_for_human(self) -> None:
        base = {
            "project_id": "p1",
            "run_id": "r1",
            "quality_profile": "standard",
            "critiques": [{"stage": StageName.IMAGE_GENERATION.value, "passed": False, "score": 0.0}],
            "shot_artifacts": [],
            "recovery_attempts": {StageName.IMAGE_GENERATION.value: 9},
        }
        auto = graph._recovery_node(
            {**base, "mode": "auto"}, StageName.IMAGE_GENERATION, default_target="image_generation"
        )
        self.assertNotIn("needs_human_review", auto)
        self.assertEqual(auto["run_status"], "failed")
        manual = graph._recovery_node(
            {**base, "mode": "manual", "human_gate_policy": "manual"},
            StageName.IMAGE_GENERATION,
            default_target="image_generation",
        )
        self.assertTrue(manual.get("needs_human_review"))

    def test_terminal_strategy_routes_to_failed(self) -> None:
        state = {
            "mode": "auto",
            "selected_strategy": RecoveryStrategy.TERMINAL_FAILURE.value,
            "stage_status": {StageName.VIDEO_GENERATION.value: "failed"},
            "critiques": [{"stage": StageName.VIDEO_GENERATION.value, "passed": False}],
        }
        self.assertEqual(graph._route_decision(state, StageName.VIDEO_GENERATION, "next", "recover"), "failed")
        self.assertEqual(graph._route_video_generation_recovery(state), "failed")


class ConditionalRetryTests(unittest.TestCase):
    def _run_generate_shot_videos(self, message: str):
        call = AsyncMock(side_effect=RuntimeError(message))
        with (
            patch.object(graph, "_shot_ids", return_value=["s1"]),
            patch.object(graph, "_has_unfinished_videos", return_value=False),
            patch(
                "services.quality_review_service.quality_review_service.storyboard_gate_status",
                return_value={"ok": True},
            ),
            patch("api.routes.shot._run_single_shot_video", new=call),
        ):
            result = asyncio.run(graph._generate_shot_videos({"project_id": "retry-project"}))
        return result, call

    def test_deterministic_failure_is_not_retried_unconditionally(self) -> None:
        result, call = self._run_generate_shot_videos("provider capability mismatch 不支持 4k 分辨率")
        self.assertTrue(result["errors"])
        self.assertEqual(call.await_count, 1)  # 确定性失败只尝试一次，不无条件重试

    def test_transient_timeout_is_retried_once(self) -> None:
        result, call = self._run_generate_shot_videos("上游视频任务 timeout")
        self.assertTrue(result["errors"])
        self.assertEqual(call.await_count, 2)  # 瞬时失败允许一次重试


if __name__ == "__main__":
    unittest.main()
