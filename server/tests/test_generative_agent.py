"""生成 Agent 状态机、决策恢复、逐镜头 fan-out 和断点续跑验收。"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from PIL import Image  # noqa: E402

from agent import graph, shot_work  # noqa: E402
from agent import nodes as agent_nodes  # noqa: E402
from agent.checkpoints import CheckpointStore  # noqa: E402
from agent.contracts import (  # noqa: E402
    QUALITY_STRATEGIES,
    STAGE_CONTRACTS,
    FailureKind,
    QualityProfileName,
    RecoveryStrategy,
    RunStatus,
    StageName,
    StageStatus,
    transition_allowed,
)
from agent.critic import (  # noqa: E402
    MAX_DIALOGUE_CHARS_PER_SHOT,
    critique_assets,
    critique_audio,
    critique_storyboard,
    split_dialogue,
)
from agent.decision import choose_recovery, recovery_candidates  # noqa: E402
from agent.shot_work import run_shot_fanout  # noqa: E402


def _image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 384), (120, 180, 220)).save(path)
    # 结构检查要求超过 1KB，纯色 PNG 在部分编码器下过小。
    if path.stat().st_size < 1100:
        path.write_bytes(path.read_bytes() + b"\0" * (1100 - path.stat().st_size))
    return str(path)


class ContractAndStateMachineTests(unittest.TestCase):
    def test_automatic_graph_prepares_audio_before_video(self) -> None:
        self.assertLess(
            graph.GRAPH_STAGE_ORDER.index(StageName.AUDIO_PRODUCTION.value),
            graph.GRAPH_STAGE_ORDER.index(StageName.VIDEO_GENERATION.value),
        )
        edges = {
            (edge.source, edge.target, edge.data)
            for edge in graph.build_graph().compile().get_graph().edges
        }
        self.assertIn(("quality_decision", "audio_production", "next"), edges)
        self.assertIn(("audio_decision", "video_generation", "next"), edges)
        self.assertIn(("video_decision", "edit_composition", "next"), edges)

    def test_all_stages_have_structured_contracts(self) -> None:
        self.assertEqual(
            set(STAGE_CONTRACTS),
            {
                StageName.DIRECTOR_PLANNING,
                StageName.STORYBOARD_DESIGN,
                StageName.ASSET_PREPARATION,
                StageName.IMAGE_GENERATION,
                StageName.QUALITY_REVIEW,
                StageName.AUDIO_PRODUCTION,
                StageName.VIDEO_GENERATION,
                StageName.VIDEO_REVIEW,
                StageName.EDIT_COMPOSITION,
                StageName.FINAL_REVIEW,
            },
        )
        self.assertEqual(
            graph.GRAPH_STAGE_ORDER,
            (
                StageName.DIRECTOR_PLANNING.value,
                StageName.STORYBOARD_DESIGN.value,
                StageName.ASSET_PREPARATION.value,
                StageName.IMAGE_GENERATION.value,
                StageName.QUALITY_REVIEW.value,
                StageName.AUDIO_PRODUCTION.value,
                StageName.VIDEO_GENERATION.value,
                StageName.VIDEO_REVIEW.value,
                StageName.EDIT_COMPOSITION.value,
                StageName.FINAL_REVIEW.value,
            ),
        )
        for contract in STAGE_CONTRACTS.values():
            self.assertTrue(contract.input_model)
            self.assertTrue(contract.output_model)
            self.assertTrue(contract.quality_metrics)
            self.assertTrue(contract.failure_classes)
            self.assertTrue(contract.checkpoint_key)
        self.assertTrue(STAGE_CONTRACTS[StageName.IMAGE_GENERATION].fan_out)
        self.assertTrue(STAGE_CONTRACTS[StageName.VIDEO_GENERATION].fan_out)
        self.assertTrue(STAGE_CONTRACTS[StageName.AUDIO_PRODUCTION].fan_out)

    def test_quality_profiles_declare_cost_and_human_policy(self) -> None:
        self.assertEqual(set(QUALITY_STRATEGIES), set(QualityProfileName))
        self.assertLess(
            QUALITY_STRATEGIES[QualityProfileName.DRAFT].cost_multiplier,
            QUALITY_STRATEGIES[QualityProfileName.STANDARD].cost_multiplier,
        )
        self.assertLess(
            QUALITY_STRATEGIES[QualityProfileName.STANDARD].cost_multiplier,
            QUALITY_STRATEGIES[QualityProfileName.FINISHING].cost_multiplier,
        )
        for item in QUALITY_STRATEGIES.values():
            self.assertEqual(item.human_intervention.value, "disabled")
            self.assertGreater(item.expected_duration, 0)
            self.assertTrue(item.publish_policy.value)
            self.assertGreaterEqual(item.max_recovery_attempts, 1)

    def test_run_state_machine_rejects_silent_completed_restart(self) -> None:
        self.assertTrue(transition_allowed(RunStatus.RUNNING, RunStatus.RECOVERING))
        self.assertTrue(transition_allowed(RunStatus.RECOVERING, RunStatus.WAITING_HUMAN))
        self.assertFalse(transition_allowed(RunStatus.COMPLETED, RunStatus.RUNNING))
        self.assertFalse(transition_allowed(RunStatus.CANCELLED, RunStatus.RUNNING))


class DecisionRecoveryTests(unittest.TestCase):
    def test_image_failure_produces_specific_recovery_candidates(self) -> None:
        candidates = recovery_candidates(
            type("F", (), {"kind": FailureKind.IMAGE_FAILED, "stage": StageName.IMAGE_GENERATION, "shot_id": "s1"})(),
            quality=QualityProfileName.STANDARD,
        )
        strategies = {item.strategy for item in candidates}
        self.assertIn(RecoveryStrategy.REVISE_PROMPT, strategies)
        self.assertIn(RecoveryStrategy.SWITCH_PROVIDER, strategies)
        self.assertIn(RecoveryStrategy.REPLACE_REFERENCE, strategies)
        self.assertIn(RecoveryStrategy.LOWER_RESOLUTION, strategies)
        self.assertIn(RecoveryStrategy.REGENERATE_FAILED_SHOTS, strategies)
        self.assertIn(RecoveryStrategy.HUMAN_REVIEW, strategies)

    def test_dialogue_too_long_prefers_shot_split(self) -> None:
        candidates = recovery_candidates(
            type("F", (), {"kind": FailureKind.DIALOGUE_TOO_LONG, "stage": StageName.STORYBOARD_DESIGN, "shot_id": "s1"})(),
            quality=QualityProfileName.STANDARD,
        )
        self.assertEqual(candidates[0].strategy, RecoveryStrategy.SPLIT_SHOT)
        self.assertIn("拆", candidates[0].prompt_changes["instruction"])

    def test_provider_reference_unsupported_can_switch_and_record_budget(self) -> None:
        trace = choose_recovery(
            type(
                "F",
                (),
                {
                    "kind": FailureKind.PROVIDER_REFERENCE_UNSUPPORTED,
                    "stage": StageName.IMAGE_GENERATION,
                    "shot_id": "s1",
                    "message": "不支持参考图",
                },
            )(),
            stage=StageName.IMAGE_GENERATION,
            project_id="decision-project",
            quality=QualityProfileName.FINISHING,
        )
        self.assertIsNotNone(trace.selected)
        self.assertTrue(trace.candidates)
        self.assertIn("remaining_cost_micro", trace.budget_snapshot)
        self.assertTrue(any(item.estimated_cost_micro is not None for item in trace.candidates))
        self.assertTrue(any(item.estimated_seconds is not None for item in trace.candidates))
        self.assertTrue(trace.reason)


class CriticTests(unittest.TestCase):
    def test_storyboard_and_audio_critic_find_long_dialogue(self) -> None:
        shot = {"shot_id": "s1", "dialogue": "长" * (MAX_DIALOGUE_CHARS_PER_SHOT + 1), "duration": 3.0}
        storyboard = critique_storyboard({"shots": [shot]})
        audio = critique_audio([shot], [])
        self.assertTrue(any(item.code == "dialogue_too_long" for item in storyboard.issues))
        self.assertTrue(any(item.code == "dialogue_too_long" for item in audio.issues))
        parts = split_dialogue("第一句。第二句，很长很长很长很长很长。第三句")
        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(all(parts))

    def test_asset_critic_reports_reference_capability_degradation(self) -> None:
        report = critique_assets(
            {
                "characters": [{"name": "林夏", "reference_images": []}],
                "script_scenes": [{"id": "scene-1", "baseline_image_path": ""}],
            },
            reference_supported=False,
        )
        codes = {item.code for item in report.issues}
        self.assertIn("provider_reference_unsupported", codes)
        self.assertIn("missing_character_reference", codes)
        self.assertTrue(report.proposed_changes)


class CheckpointResumeTests(unittest.TestCase):
    def test_user_change_invalidates_only_changed_shots_and_resume_reuses_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore("change-project", "auto", root=Path(root))
            store.detect_changes({"s1": 1, "s2": 1, "__av_config_version__": 0})
            store.save_shot_artifact(
                "s1",
                StageName.IMAGE_GENERATION.value,
                shot_version=1,
                status=StageStatus.SUCCEEDED.value,
                output_fingerprint="ok",
            )
            changes = store.detect_changes({"s1": 2, "s2": 1, "__av_config_version__": 0})
            self.assertEqual(changes["changed_shot_ids"], ["s1"])
            self.assertIsNone(store.reusable_shot_artifact("s1", StageName.IMAGE_GENERATION.value, shot_version=1))
            self.assertIsNone(store.reusable_shot_artifact("s1", StageName.IMAGE_GENERATION.value, shot_version=2))

            # 同版本 + 真实图片允许断点续跑跳过生成。
            image = _image(Path(root) / "resume.png")
            store.save_shot_artifact(
                "s2",
                StageName.IMAGE_GENERATION.value,
                shot_version=1,
                status=StageStatus.SUCCEEDED.value,
                path=image,
                output_fingerprint="ok2",
            )
            self.assertIsNotNone(store.reusable_shot_artifact("s2", StageName.IMAGE_GENERATION.value, shot_version=1))

            calls = []

            async def worker(shot_id, version):
                calls.append((shot_id, version))
                return {"shot_id": shot_id, "shot_version": version, "status": "succeeded", "path": image}

            result = asyncio.run(
                run_shot_fanout(
                    project_id="change-project",
                    shot_versions={"s2": 1},
                    stage=StageName.IMAGE_GENERATION,
                    worker=worker,
                    checkpoint=store,
                )
            )
            self.assertEqual(calls, [])
            self.assertEqual(len(result["successes"]), 1)

    def test_av_change_invalidates_compose_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore("av-project", "auto", root=Path(root))
            store.detect_changes({"s1": 1, "__av_config_version__": 1})
            store.save_stage(StageName.EDIT_COMPOSITION.value, status="succeeded", input_fingerprint="fp")
            changes = store.detect_changes({"s1": 1, "__av_config_version__": 2})
            self.assertTrue(changes["av_config_changed"])
            self.assertEqual(store.stage(StageName.EDIT_COMPOSITION.value)["status"], "invalidated")


class FanoutIntegrationTests(unittest.TestCase):
    def test_one_failed_image_does_not_invalidate_success(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore("image-fanout", "auto", root=Path(root))
            image = _image(Path(root) / "ok.png")

            async def worker(shot_id, version):
                if shot_id == "bad":
                    raise RuntimeError("image provider failed")
                return {"shot_id": shot_id, "shot_version": version, "status": "succeeded", "path": image, "score": 0.9}

            result = asyncio.run(
                run_shot_fanout(
                    project_id="image-fanout",
                    shot_versions={"good": 1, "bad": 1},
                    stage=StageName.IMAGE_GENERATION,
                    worker=worker,
                    checkpoint=store,
                )
            )
            self.assertEqual([item["shot_id"] for item in result["successes"]], ["good"])
            self.assertEqual([item["shot_id"] for item in result["failures"]], ["bad"])
            self.assertIsNotNone(store.reusable_shot_artifact("good", StageName.IMAGE_GENERATION.value, shot_version=1))
            self.assertEqual(store.shot_artifact("bad", StageName.IMAGE_GENERATION.value)["status"], "failed")

    def test_one_failed_video_keeps_success_and_classifies_video_failure(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore("video-fanout", "auto", root=Path(root))
            video = Path(root) / "ok.mp4"
            video.write_bytes(b"v" * 5000)

            async def worker(shot_id, version):
                if shot_id == "bad":
                    raise RuntimeError("Seedance video task failed")
                return {"shot_id": shot_id, "shot_version": version, "status": "succeeded", "path": str(video)}

            result = asyncio.run(
                run_shot_fanout(
                    project_id="video-fanout",
                    shot_versions={"good": 1, "bad": 1},
                    stage=StageName.VIDEO_GENERATION,
                    worker=worker,
                    checkpoint=store,
                )
            )
            self.assertEqual([item["shot_id"] for item in result["successes"]], ["good"])
            self.assertEqual(result["failures"][0]["failure"]["kind"], FailureKind.VIDEO_FAILED.value)


class GraphIntegrationTests(unittest.TestCase):
    def test_graph_exposes_all_agent_stages_and_recovery_nodes(self) -> None:
        drawable = graph.get_graph().get_graph()
        node_ids = set(drawable.nodes)
        for stage in StageName:
            self.assertIn(stage.value, node_ids)
        for node in ("director_review", "quality_review", "video_review", "final_review", "human_gate"):
            self.assertIn(node, node_ids)

    def test_llm_invalid_output_creates_revision_decision(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            fake_store = CheckpointStore("llm-invalid", "auto", root=Path(root))

            async def invalid_run(_state):
                raise RuntimeError("Mimo 剧本解析结果无法使用: JSON 结构非法")

            fake_parser = types.SimpleNamespace(run=invalid_run)
            with (
                patch.object(graph.CheckpointStore, "get", return_value=fake_store),
                patch.object(agent_nodes, "script_parser", fake_parser, create=True),
            ):
                state = asyncio.run(
                    graph._director_planning(
                        {
                            "project_id": "llm-invalid",
                            "run_id": "auto",
                            "quality_profile": "standard",
                            "initial_state": {"project_id": "llm-invalid", "user_input": "一段剧本"},
                        }
                    )
                )
            self.assertEqual(state["stage_status"]["director_planning"], StageStatus.FAILED.value)
            self.assertTrue(state["decision_traces"])
            selected = state["decision_traces"][-1]["selected"]
            self.assertIsNotNone(selected)
            self.assertIn(
                selected["strategy"],
                {RecoveryStrategy.REVISE_PROMPT.value, RecoveryStrategy.SWITCH_PROVIDER.value, RecoveryStrategy.HUMAN_REVIEW.value},
            )
            self.assertEqual(fake_store.decisions()[-1]["stage"], StageName.DIRECTOR_PLANNING.value)

    def test_image_fanout_graph_node_preserves_partial_success(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            fake_store = CheckpointStore("image-node", "auto", root=Path(root))
            image = _image(Path(root) / "ok.png")

            async def fake_generate(shot_id, expected_version, *, project_id, provider_override="", preferred_size="", seed_override=None, recovery_revisions=None):
                if shot_id == "bad":
                    raise RuntimeError("image generation failed")
                return {"shot_id": shot_id, "shot_version": expected_version, "status": "succeeded", "path": image}

            with (
                patch.object(graph.CheckpointStore, "get", return_value=fake_store),
                patch.object(graph, "_reference_gate", return_value={}),
                patch.object(graph, "_shot_versions", return_value={"good": 1, "bad": 1}),
                patch.object(graph, "generate_storyboard_shot", side_effect=fake_generate),
            ):
                state = asyncio.run(
                    graph._image_generation_fan_out(
                        {
                            "project_id": "image-node",
                            "run_id": "auto",
                            "quality_profile": "standard",
                            "initial_state": {},
                        }
                    )
                )
            self.assertEqual(state["successful_shot_ids"], ["good"])
            self.assertEqual(state["failed_shot_ids"], ["bad"])
            self.assertEqual(state["stage_status"]["image_generation"], StageStatus.DEGRADED.value)


if __name__ == "__main__":
    unittest.main()
