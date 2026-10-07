"""十阶段生成 Agent 的结构、依赖、局部恢复和断点续跑验收。"""

from __future__ import annotations

import asyncio
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent import graph, shot_work  # noqa: E402
from agent import nodes as agent_nodes  # noqa: E402
from agent.checkpoints import CheckpointStore  # noqa: E402
from agent.contracts import RecoveryStrategy, StageName, StageStatus  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import Project, Shot, ShotVersion  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


class PhaseGraphStructureTests(unittest.TestCase):
    def test_ten_stage_order_and_process_review_decision_recovery_roles(self) -> None:
        self.assertEqual(
            graph.GRAPH_STAGE_ORDER,
            (
                "director_planning",
                "storyboard_design",
                "asset_preparation",
                "image_generation",
                "quality_review",
                "audio_production",
                "video_generation",
                "video_review",
                "edit_composition",
                "final_review",
            ),
        )
        self.assertEqual(set(graph.GRAPH_STAGE_NODE_NAMES), set(graph.GRAPH_STAGE_ORDER))
        drawable = graph.get_graph().get_graph()
        for stage, roles in graph.GRAPH_STAGE_NODE_NAMES.items():
            self.assertEqual(set(roles), {"process", "critic", "decision", "recovery"}, stage)
            for role, node_id in roles.items():
                self.assertIn(node_id, drawable.nodes)
                self.assertEqual(graph.GRAPH_NODE_META[node_id]["type"], role)

    def test_external_tts_dependency_is_audio_check_before_video_check(self) -> None:
        edges = {(edge.source, edge.target, edge.data) for edge in graph.get_graph().get_graph().edges}
        self.assertIn(("quality_decision", "audio_production", "next"), edges)
        self.assertIn(("audio_production", "audio_review", None), edges)
        self.assertIn(("audio_decision", "video_generation", "next"), edges)
        self.assertIn(("video_generation_decision", "video_review", "next"), edges)
        self.assertIn(("video_decision", "edit_composition", "next"), edges)
        self.assertIn(("edit_decision", "final_review", "next"), edges)

    def test_final_feedback_routes_by_issue_code_and_shot(self) -> None:
        state = {
            "critiques": [
                {
                    "stage": "final_review",
                    "passed": False,
                    "issues": [
                        {
                            "severity": "error",
                            "code": "video_frozen",
                            "shot_id": "s2",
                            "details": {"source_stage": "video_generation"},
                        },
                        {"severity": "info", "code": "visual_quality_pending"},
                    ],
                }
            ]
        }
        self.assertEqual(graph._final_feedback_target(state), "video_generation")
        self.assertEqual(graph._final_feedback_shot_ids(state, "video_generation"), ["s2"])
        self.assertEqual(graph._final_feedback_shot_ids(state, "image_generation"), [])

    def test_native_audio_plan_can_skip_external_tts_dependency(self) -> None:
        shots = [
            {
                "shot_id": "native-1",
                "dialogue": "你好",
                "shot_type": "close-up",
                "continuity_profile": {"audio_mode": "native"},
            },
        ]
        with (
            patch.object(graph, "_db_shots", return_value=shots),
            patch.object(graph, "_resolve_audio_mode", return_value="native"),
        ):
            plan = graph._audio_execution_plan("native-project", {"mode": "auto"})
        self.assertEqual(plan["mode"], "native_audio")
        self.assertEqual(plan["native_audio_shot_ids"], ["native-1"])
        self.assertEqual(plan["external_tts_shot_ids"], [])
        self.assertNotIn("audio_production", plan["dependency"])

    def test_native_audio_shot_never_calls_external_tts_route(self) -> None:
        async def run() -> dict:
            with (
                patch.object(
                    shot_work,
                    "_shot_audio_context",
                    return_value={
                        "dialogue": "你好",
                        "shot_type": "close-up",
                        "audio_mode": "native",
                        "continuity_profile": {},
                    },
                ),
                patch.object(shot_work, "_resolved_audio_mode", return_value="native"),
                patch("api.routes.shot._run_single_shot_audio", new_callable=AsyncMock) as tts,
            ):
                result = await shot_work.generate_audio_shot("native-1", 1, project_id="native-project")
            self.assertEqual(tts.await_count, 0)
            return result

        artifact = asyncio.run(run())
        self.assertEqual(artifact["status"], StageStatus.SKIPPED.value)
        self.assertIn({"name": "audio_source", "value": "native"}, artifact["metrics"])

    def test_final_feedback_routes_only_to_requested_local_stages(self) -> None:
        targets = {
            "image": "image_generation",
            "audio": "audio_production",
            "video": "video_generation",
            "edit": "edit_composition",
        }
        for explicit, expected in targets.items():
            state = {"mode": "auto", "final_recovery_target": explicit, "pending_recovery_target": expected}
            self.assertEqual(graph._final_feedback_target(state), expected)
            self.assertEqual(
                graph._route_final_recovery(state),
                {
                    "image_generation": "image",
                    "audio_production": "audio",
                    "video_generation": "video",
                    "edit_composition": "compose",
                }[expected],
            )


class AutomaticRoutingTests(unittest.TestCase):
    def test_auto_never_routes_to_human_gate(self) -> None:
        state = {
            "mode": "auto",
            "needs_human_review": True,
            "human_gate_policy": "manual",
            "stage_status": {"image_generation": StageStatus.FAILED.value},
            "recovery_attempts": {"image_generation": 99},
            "pending_recovery_target": "image_generation",
        }
        self.assertEqual(graph._route_decision(state, StageName.IMAGE_GENERATION, "next", "recover"), "failed")
        self.assertEqual(graph._route_image_recovery(state), "failed")

    def test_manual_human_gate_requires_explicit_policy(self) -> None:
        base = {
            "mode": "manual",
            "needs_human_review": True,
            "stage_status": {"image_generation": StageStatus.FAILED.value},
            "recovery_attempts": {"image_generation": 99},
        }
        self.assertEqual(graph._route_decision(base, StageName.IMAGE_GENERATION, "next", "recover"), "failed")
        self.assertEqual(
            graph._route_decision(
                {**base, "human_gate_policy": "manual"}, StageName.IMAGE_GENERATION, "next", "recover"
            ),
            "human",
        )


class RecoveryAndResumeTests(unittest.TestCase):
    def test_generation_revisions_only_reach_matching_shot_and_stage(self) -> None:
        state = {
            "prompt_revisions": [
                {
                    "shot_id": "a",
                    "patches": [
                        {
                            "field": "visual_prompt",
                            "op": "replace",
                            "value": {"rule": "close-up"},
                            "shot_id": "a",
                            "target_stage": "image_generation",
                        },
                        {"field": "seed", "op": "set", "value": 27, "shot_id": "a", "target_stage": "image_generation"},
                        {
                            "field": "visual_prompt",
                            "op": "replace",
                            "value": "other",
                            "shot_id": "b",
                            "target_stage": "image_generation",
                        },
                        {
                            "field": "visual_prompt",
                            "op": "replace",
                            "value": "video",
                            "shot_id": "a",
                            "target_stage": "video_generation",
                        },
                    ],
                },
            ]
        }
        image = graph._shot_revisions(state, StageName.IMAGE_GENERATION, "a")
        self.assertEqual([patch["value"] for patch in image[0]["patches"]], [{"rule": "close-up"}])
        self.assertEqual(graph._seed_override(state, StageName.IMAGE_GENERATION, "a"), 27)
        self.assertEqual(
            graph._shot_revisions(state, StageName.IMAGE_GENERATION, "b")[0]["patches"][0]["value"], "other"
        )
        self.assertEqual(
            graph._shot_revisions(state, StageName.VIDEO_GENERATION, "a")[0]["patches"][0]["value"], "video"
        )
        self.assertEqual(graph._shot_revisions(state, StageName.VIDEO_GENERATION, "b"), [])

    def test_image_fanout_passes_scoped_patches_and_seed_to_only_failed_shot(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore("scoped-image", "auto", root=Path(root))
            from PIL import Image

            image = Path(root) / "shot.png"
            Image.new("RGB", (256, 384), (120, 160, 200)).save(image)
            if image.stat().st_size < 1100:
                image.write_bytes(image.read_bytes() + b"\0" * (1100 - image.stat().st_size))
            calls = {}

            async def generate(shot_id, version, **kwargs):
                calls[shot_id] = kwargs
                return {"shot_id": shot_id, "shot_version": version, "status": "succeeded", "path": str(image)}

            state = {
                "project_id": "scoped-image",
                "run_id": "auto",
                "mode": "auto",
                "prompt_revisions": [
                    {
                        "shot_id": "bad",
                        "patches": [
                            {
                                "field": "visual_prompt",
                                "op": "replace",
                                "value": "close-up",
                                "shot_id": "bad",
                                "target_stage": "image_generation",
                            },
                            {
                                "field": "seed",
                                "op": "set",
                                "value": {"seed": 321},
                                "shot_id": "bad",
                                "target_stage": "image_generation",
                            },
                        ],
                    }
                ],
            }
            with (
                patch.object(graph.CheckpointStore, "get", return_value=store),
                patch.object(graph, "_reference_gate", return_value={}),
                patch.object(graph, "_shot_versions", return_value={"good": 1, "bad": 1}),
                patch.object(graph, "generate_storyboard_shot", side_effect=generate),
            ):
                result = asyncio.run(graph._image_generation_fan_out(state))
        self.assertEqual(result["successful_shot_ids"], ["bad", "good"])
        self.assertEqual(calls["bad"]["seed_override"], 321)
        self.assertEqual(calls["bad"]["recovery_revisions"][0]["patches"][0]["value"], "close-up")
        self.assertIsNone(calls["good"]["seed_override"])
        self.assertEqual(calls["good"]["recovery_revisions"], [])

    def test_recovery_seed_handles_prior_scalar_patch(self) -> None:
        state = {
            "project_id": "p",
            "run_id": "r",
            "prompt_revisions": [{"shot_id": "a", "patches": [{"field": "seed", "value": 123}]}],
        }
        seed = graph._recovery_seed(state, StageName.IMAGE_GENERATION, "a", [])
        self.assertNotEqual(seed, 123)
        self.assertEqual(seed, graph._recovery_seed(state, StageName.IMAGE_GENERATION, "a", []))

    def test_video_resolution_recovery_uses_supported_provider_tier(self) -> None:
        selected = types.SimpleNamespace(
            strategy=RecoveryStrategy.LOWER_RESOLUTION,
            shot_ids=["s1"],
            prompt_changes={},
            provider="",
        )
        trace = types.SimpleNamespace(
            selected=selected,
            candidates=[],
            trace_id="resolution-trace",
            budget_snapshot={},
            input_fingerprint="fp",
            reason="lower resolution",
            shot_id="",
            model_dump=lambda mode: {"selected": {"strategy": "lower_resolution"}},
        )
        store = types.SimpleNamespace(add_decision=lambda *_: None, add_event=lambda *_, **__: None)
        state = {
            "mode": "auto",
            "project_id": "p",
            "run_id": "r",
            "quality_profile": "standard",
            "shot_artifacts": [],
            "failed_shot_ids": ["s1"],
        }
        with (
            patch.object(graph, "choose_recovery", return_value=trace),
            patch.object(graph.CheckpointStore, "get", return_value=store),
        ):
            update = graph._recovery_node(state, StageName.VIDEO_GENERATION, default_target="video_generation")
        self.assertEqual(update["provider_switch"]["video_generation:resolution"], "480p")
        self.assertEqual(update["pending_shot_ids"], ["s1"])

    def test_video_split_persists_failed_shot_and_routes_back_through_image(self) -> None:
        selected = types.SimpleNamespace(
            strategy=RecoveryStrategy.SPLIT_SHOT,
            shot_ids=["bad"],
            prompt_changes={},
            provider="",
        )
        trace = types.SimpleNamespace(
            selected=selected,
            candidates=[],
            trace_id="split-trace",
            budget_snapshot={},
            input_fingerprint="fp",
            reason="complex motion",
            shot_id="",
            model_dump=lambda mode: {"selected": {"strategy": "split_shot"}},
        )
        store = types.SimpleNamespace(
            add_decision=lambda *_: None,
            add_event=lambda *_, **__: None,
            detect_changes=lambda: {"changed_shot_ids": ["bad", "bad_part_02"]},
        )
        state = {
            "mode": "auto",
            "project_id": "p",
            "run_id": "r",
            "shot_artifacts": [
                {"shot_id": "good", "stage": "video_generation", "status": "succeeded"},
                {"shot_id": "bad", "stage": "video_generation", "status": "failed"},
            ],
            "failed_shot_ids": ["bad"],
        }
        split_result = {"operation_id": "split-trace:bad", "shot_ids": ["bad", "bad_part_02"]}
        with (
            patch.object(graph, "choose_recovery", return_value=trace),
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(graph, "_persist_recovery_splits", return_value=[split_result]) as persist,
        ):
            update = graph._recovery_node(state, StageName.VIDEO_GENERATION, default_target="video_generation")
        persist.assert_called_once()
        self.assertEqual(persist.call_args.args[:2], ("p", ["bad"]))
        self.assertEqual(update["pending_recovery_target"], "image_generation")
        self.assertEqual(update["pending_shot_ids"], ["bad", "bad_part_02"])
        self.assertEqual(update["split_recovery_shot_ids"], ["bad", "bad_part_02"])
        self.assertEqual(graph._route_video_generation_recovery(update), "image")
        self.assertNotIn("needs_human_review", update)
        self.assertEqual(update["recovery_history"][0]["split_results"], [split_result])

    def test_failed_split_is_explicit_failure_not_fake_storyboard_retry(self) -> None:
        selected = types.SimpleNamespace(
            strategy=RecoveryStrategy.SPLIT_SHOT,
            shot_ids=["bad"],
            prompt_changes={},
            provider="",
        )
        trace = types.SimpleNamespace(
            selected=selected,
            candidates=[],
            trace_id="unsafe-split",
            budget_snapshot={},
            input_fingerprint="fp",
            reason="complex motion",
            shot_id="",
            model_dump=lambda mode: {"selected": {"strategy": "split_shot"}},
        )
        store = types.SimpleNamespace(add_decision=lambda *_: None, add_event=lambda *_, **__: None)
        state = {"mode": "auto", "project_id": "p", "run_id": "r", "failed_shot_ids": ["bad"]}
        with (
            patch.object(graph, "choose_recovery", return_value=trace),
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(graph, "_persist_recovery_splits", side_effect=RuntimeError("version conflict")),
        ):
            update = graph._recovery_node(state, StageName.VIDEO_GENERATION, default_target="video_generation")
        self.assertEqual(update["selected_strategy"], RecoveryStrategy.TERMINAL_FAILURE.value)
        self.assertEqual(update["run_status"], "failed")
        self.assertEqual(graph._route_video_generation_recovery(update), "failed")
        self.assertNotIn("needs_human_review", update)

    def test_persisted_split_commits_and_invalidates_changed_shot_checkpoints(self) -> None:
        init_db()
        project_id = "graph-split-integration"
        with tempfile.TemporaryDirectory() as root:
            db = SessionLocal()
            try:
                db.query(ShotVersion).filter(ShotVersion.shot_id.like(f"{project_id}%")).delete(
                    synchronize_session=False
                )
                db.query(Shot).filter(Shot.project_id == project_id).delete(synchronize_session=False)
                db.query(Project).filter(Project.id == project_id).delete(synchronize_session=False)
                project = Project(id=project_id, title="split")
                bad = Shot(
                    id=f"{project_id}-bad",
                    project_id=project_id,
                    sequence=1,
                    version=2,
                    duration=6.0,
                    character_action="走到窗边然后回头",
                    storyboard_path="old-storyboard.png",
                    video_path="old-video.mp4",
                )
                good = Shot(
                    id=f"{project_id}-good",
                    project_id=project_id,
                    sequence=2,
                    version=1,
                    storyboard_path="good-storyboard.png",
                    video_path="good-video.mp4",
                )
                db.add_all([project, bad, good])
                db.commit()
                store = CheckpointStore(project_id, "split-run", root=Path(root))
                store.detect_changes()
                for shot_id, version in ((bad.id, 2), (good.id, 1)):
                    store.save_shot_artifact(
                        shot_id,
                        "video_generation",
                        shot_version=version,
                        status="succeeded",
                        path="old.mp4",
                        input_fingerprint="input",
                    )
                result = graph._persist_recovery_splits(
                    project_id, [bad.id], trace_id="integration-split", reason="complex motion"
                )
                changes = store.detect_changes()
                self.assertEqual(set(changes["changed_shot_ids"]), set(result[0]["shot_ids"]))
                self.assertEqual(store.shot_artifact(bad.id, "video_generation")["status"], "invalidated")
                self.assertEqual(store.shot_artifact(good.id, "video_generation")["status"], "succeeded")
                db.expire_all()
                rows = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
                self.assertEqual([row.id for row in rows], [*result[0]["shot_ids"], good.id])
                self.assertEqual([row.sequence for row in rows], list(range(1, len(rows) + 1)))
                self.assertEqual(rows[0].video_path, "")
                self.assertEqual(rows[-1].video_path, "good-video.mp4")
            finally:
                db.query(ShotVersion).filter(ShotVersion.shot_id.like(f"{project_id}%")).delete(
                    synchronize_session=False
                )
                db.query(Shot).filter(Shot.project_id == project_id).delete(synchronize_session=False)
                db.query(Project).filter(Project.id == project_id).delete(synchronize_session=False)
                db.commit()
                db.close()

    def test_failed_shot_recovery_queue_does_not_rerun_successful_shot(self) -> None:
        calls: list[str] = []

        async def worker(shot_id: str, version: int) -> dict:
            calls.append(shot_id)
            if shot_id == "bad":
                raise RuntimeError("provider failed")
            return {"shot_id": shot_id, "shot_version": version, "status": "succeeded", "path": ""}

        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore("partial-recovery", "run-1", root=Path(root))
            versions = {"good": 1, "bad": 1}
            first = asyncio.run(
                shot_work.run_shot_fanout(
                    project_id="partial-recovery",
                    shot_versions=versions,
                    stage=StageName.AUDIO_PRODUCTION,
                    worker=worker,
                    checkpoint=store,
                    run_id="run-1",
                    input_fingerprint="input-1",
                )
            )
            second = asyncio.run(
                shot_work.run_shot_fanout(
                    project_id="partial-recovery",
                    shot_versions=versions,
                    stage=StageName.AUDIO_PRODUCTION,
                    worker=worker,
                    checkpoint=store,
                    run_id="run-1",
                    input_fingerprint="input-1",
                )
            )

        self.assertEqual(calls.count("good"), 1)
        self.assertEqual(calls.count("bad"), 2)
        self.assertEqual([item["shot_id"] for item in first["successes"]], ["good"])
        self.assertEqual([item["shot_id"] for item in second["successes"]], ["good"])
        self.assertEqual([item["shot_id"] for item in second["failures"]], ["bad"])

    def test_stage_checkpoint_resumes_without_recalling_provider(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore("resume-stage", "run-1", root=Path(root))
            calls = 0

            async def parser(state: dict) -> dict:
                nonlocal calls
                calls += 1
                return {
                    "script_title": "标题",
                    "genre": "剧情",
                    "characters": [{"name": "主角"}],
                    "script_scenes": [{"name": "场景"}],
                    "logic_issues": [],
                }

            fake_parser = types.SimpleNamespace(run=parser)
            initial = {
                "project_id": "resume-stage",
                "run_id": "run-1",
                "mode": "auto",
                "quality_profile": "standard",
                "initial_state": {"project_id": "resume-stage", "user_input": "剧本"},
            }
            with (
                patch.object(graph.CheckpointStore, "get", return_value=store),
                patch.object(agent_nodes, "script_parser", fake_parser, create=True),
            ):
                first = asyncio.run(graph._director_planning(initial))
                resumed = asyncio.run(graph._director_planning({**initial, **first}))

        self.assertEqual(calls, 1)
        self.assertEqual(first["stage_status"]["director_planning"], StageStatus.SUCCEEDED.value)
        self.assertEqual(resumed["stage_status"]["director_planning"], StageStatus.SUCCEEDED.value)
        self.assertEqual(resumed["current_step"], "director_planning")


if __name__ == "__main__":
    unittest.main()
