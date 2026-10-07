"""生成 Agent 端到端集成测试：失败注入全部使用 Mock Provider/Worker。

覆盖十三个验收场景：LLM 非法输出、图片/视频 Provider 失败、对白超时长、
Provider 不支持参考图、局部失败保留成功、用户修改后的版本冲突、进程退出后
断点续跑、候选自动选择、自动模式完成成片、自动模式不进人工卡点、预算不足
自动降级/终止、成片复审反馈局部重算，以及可解释追踪汇总的字段完整性。
"""

from __future__ import annotations

import asyncio
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image  # noqa: E402

from agent import graph, shot_work  # noqa: E402
from agent import nodes as agent_nodes  # noqa: E402
from agent.checkpoints import CheckpointStore  # noqa: E402
from agent.contracts import (  # noqa: E402
    RecoveryStrategy,
    RunStatus,
    StageName,
    StageStatus,
    select_video_candidate,
)
from db import SessionLocal, init_db  # noqa: E402
from models import Project, Shot  # noqa: E402


def _image_file(root: Path, name: str) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 384), (110, 170, 210)).save(path)
    if path.stat().st_size < 1100:
        path.write_bytes(path.read_bytes() + b"\0" * (1100 - path.stat().st_size))
    return str(path)


def _video_file(root: Path, name: str) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * 8192)
    return str(path)


def _base_state(project_id: str, run_id: str = "auto", **extra) -> dict:
    return {
        "project_id": project_id,
        "run_id": run_id,
        "mode": "auto",
        "quality_profile": "standard",
        "initial_state": {"project_id": project_id, "user_input": "测试剧本"},
        "errors": [],
        **extra,
    }


def _parsed_payload() -> dict:
    return {
        "script_title": "测试剧",
        "genre": "剧情",
        "style_suggestion": "写实",
        "characters": [{"name": "主角", "reference_images": ["a.png"]}],
        "raw_script": "剧本",
        "script_scenes": [{"id": "scene-1", "name": "房间", "baseline_image_path": "base.png"}],
        "logic_issues": [],
        "rag_context": [],
        "requested_style": "",
        "effective_style": "",
        "style_source": "",
    }


def _storyboard_payload() -> dict:
    return {
        "shots": [
            {
                "shot_id": "shot-1",
                "shot_type": "medium",
                "duration": 4.0,
                "dialogue": "",
                "character_action": "看向窗外",
                "scene_description": "房间",
            },
            {
                "shot_id": "shot-2",
                "shot_type": "close-up",
                "duration": 3.0,
                "dialogue": "",
                "character_action": "微笑",
                "scene_description": "房间",
            },
        ],
        "timing_plan": {},
    }


def _seed_project_shots(project_id: str, shots: list[dict]) -> None:
    init_db()
    db = SessionLocal()
    try:
        db.query(Shot).filter(Shot.project_id == project_id).delete(synchronize_session=False)
        db.query(Project).filter(Project.id == project_id).delete(synchronize_session=False)
        db.add(Project(id=project_id, title=project_id))
        for index, item in enumerate(shots, start=1):
            db.add(
                Shot(
                    id=item["id"],
                    project_id=project_id,
                    sequence=index,
                    version=int(item.get("version", 1)),
                    duration=float(item.get("duration", 4.0)),
                    dialogue=item.get("dialogue", ""),
                    storyboard_path=item.get("storyboard_path", ""),
                    video_path=item.get("video_path", ""),
                )
            )
        db.commit()
    finally:
        db.close()


def _cleanup_project(project_id: str) -> None:
    db = SessionLocal()
    try:
        db.query(Shot).filter(Shot.project_id == project_id).delete(synchronize_session=False)
        db.query(Project).filter(Project.id == project_id).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _shot_versions_filter(rows: dict[str, int]):
    """模拟真实 _shot_versions：按 pending 恢复范围过滤镜头。"""

    def lookup(project_id: str, only_ids: list[str] | None = None, require_storyboard: bool = False):
        if only_ids:
            wanted = {str(item) for item in only_ids if item}
            return {key: value for key, value in rows.items() if key in wanted}
        return dict(rows)

    return lookup


async def _run_image_stage(state: dict) -> dict:
    """按图结构执行 image_generation -> fan_in -> review -> decision。"""

    state = {**state, **await graph._image_generation_fan_out(state)}
    state = {**state, **await graph._image_generation_fan_in(state)}
    state = {**state, **await graph._image_review(state)}
    state = {**state, **await graph._image_decision(state)}
    return state


async def _run_video_stage(state: dict) -> dict:
    state = {**state, **await graph._video_generation_fan_out(state)}
    state = {**state, **await graph._video_generation_fan_in(state)}
    state = {**state, **await graph._video_generation_review(state)}
    state = {**state, **await graph._video_generation_decision(state)}
    return state


def _last_decision(store: CheckpointStore) -> dict:
    decisions = store.decisions()
    assert decisions, "决策记录不应为空"
    return decisions[-1]


class _TempRootTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def store(self, project_id: str, run_id: str = "auto") -> CheckpointStore:
        return CheckpointStore(project_id, run_id, root=self.root)


class Scenario1LLMInvalidJSONTests(_TempRootTestCase):
    def test_invalid_json_fails_over_to_prompt_revision_and_recovers(self) -> None:
        project_id = "e2e-llm-invalid"
        store = self.store(project_id)
        calls = {"parser": 0}

        async def broken_parser(state: dict) -> dict:
            calls["parser"] += 1
            raise RuntimeError("LLM 输出不是合法 JSON：json decode error at line 1")

        fake_parser = types.SimpleNamespace(run=broken_parser)
        state = _base_state(project_id)
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(agent_nodes, "script_parser", fake_parser, create=True),
        ):
            failed = asyncio.run(graph._director_planning(state))
        self.assertEqual(failed["stage_status"][StageName.DIRECTOR_PLANNING.value], StageStatus.FAILED.value)
        self.assertEqual(failed["run_status"], RunStatus.RECOVERING.value)
        decision = _last_decision(store)
        self.assertEqual(decision["failure"]["kind"], "llm_invalid_output")
        self.assertEqual(decision["selected"]["strategy"], RecoveryStrategy.REVISE_PROMPT.value)
        patches = decision["selected"]["prompt_patches"]
        self.assertTrue(any(p["field"] == "output_format" and p["value"] == "strict_json" for p in patches))

        with patch.object(graph.CheckpointStore, "get", return_value=store):
            recovery = graph._recovery_node(
                {**state, **failed}, StageName.DIRECTOR_PLANNING, default_target="director_planning"
            )
        self.assertEqual(recovery["selected_strategy"], RecoveryStrategy.REVISE_PROMPT.value)
        self.assertTrue(recovery["prompt_revisions"])
        self.assertEqual(recovery["recovery_attempts"][StageName.DIRECTOR_PLANNING.value], 1)
        self.assertEqual(graph._route_director_recovery(recovery), "retry")

        async def fixed_parser(state: dict) -> dict:
            calls["parser"] += 1
            return _parsed_payload()

        fixed = types.SimpleNamespace(run=fixed_parser)
        retried_state = {**state, **failed, **recovery}
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(agent_nodes, "script_parser", fixed, create=True),
        ):
            recovered = asyncio.run(graph._director_planning(retried_state))
        self.assertEqual(calls["parser"], 2)
        self.assertEqual(recovered["stage_status"][StageName.DIRECTOR_PLANNING.value], StageStatus.SUCCEEDED.value)
        self.assertTrue(any(event["event"] == "recovery_selected" for event in store.snapshot()["events"]))
        self.assertFalse(
            any(item["selected"] and item["selected"]["strategy"] == "human_review" for item in store.decisions())
        )


class Scenario2ImageProviderFailureTests(_TempRootTestCase):
    def test_all_images_fail_produces_traced_recovery_without_human(self) -> None:
        project_id = "e2e-image-fail"
        store = self.store(project_id)

        async def worker(shot_id: str, version: int, **kwargs) -> dict:
            raise RuntimeError(f"image provider request failed for {shot_id}")

        state = _base_state(project_id)
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(graph, "_reference_gate", return_value={}),
            patch.object(graph, "_shot_versions", return_value={"shot-1": 1, "shot-2": 1}),
            patch.object(graph, "generate_storyboard_shot", side_effect=worker),
        ):
            result = asyncio.run(_run_image_stage(state))
        self.assertEqual(result["stage_status"][StageName.IMAGE_GENERATION.value], StageStatus.FAILED.value)
        self.assertEqual(sorted(result["failed_shot_ids"]), ["shot-1", "shot-2"])
        decision = _last_decision(store)
        self.assertEqual(decision["failure"]["kind"], "image_failed")
        self.assertIsNotNone(decision["selected"])
        self.assertNotEqual(decision["selected"]["strategy"], RecoveryStrategy.HUMAN_REVIEW.value)
        self.assertGreater(len(decision["candidates"]), 1)
        self.assertEqual(graph._route_image_decision(result), "recover")
        row = store.shot_artifact("shot-1", StageName.IMAGE_GENERATION.value)
        self.assertEqual(row["status"], StageStatus.FAILED.value)
        self.assertEqual(row["failure"]["kind"], "image_failed")


class Scenario3VideoProviderFailureTests(_TempRootTestCase):
    def test_video_provider_failure_is_classified_and_routed_to_recovery(self) -> None:
        project_id = "e2e-video-fail"
        store = self.store(project_id)
        video = _video_file(self.root, "shot-1.mp4")

        async def worker(shot_id: str, version: int, **kwargs) -> dict:
            if shot_id == "shot-2":
                raise RuntimeError("video provider 渲染失败")
            return {
                "shot_id": shot_id,
                "shot_version": version,
                "status": "succeeded",
                "path": video,
                "provider": "mock-video",
            }

        state = _base_state(project_id)
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(graph, "_reference_gate", return_value={}),
            patch.object(graph, "_shot_versions", return_value={"shot-1": 1, "shot-2": 1}),
            patch.object(graph, "generate_video_shot", side_effect=worker),
        ):
            gate = types.SimpleNamespace(storyboard_gate_status=lambda pid, **_kwargs: {"ok": True})
            from services.quality_review_service import quality_review_service

            with patch.object(quality_review_service, "storyboard_gate_status", gate.storyboard_gate_status):
                result = asyncio.run(_run_video_stage(state))
        self.assertEqual(result["stage_status"][StageName.VIDEO_GENERATION.value], StageStatus.DEGRADED.value)
        self.assertEqual(result["successful_shot_ids"], ["shot-1"])
        self.assertEqual(result["failed_shot_ids"], ["shot-2"])
        decision = _last_decision(store)
        self.assertEqual(decision["failure"]["kind"], "video_failed")
        self.assertNotEqual(decision["selected"]["strategy"], RecoveryStrategy.HUMAN_REVIEW.value)
        self.assertEqual(graph._route_video_generation_decision(result), "recover")
        self.assertEqual(
            store.shot_artifact("shot-2", StageName.VIDEO_GENERATION.value)["status"], StageStatus.FAILED.value
        )


class Scenario4DialogueTooLongTests(_TempRootTestCase):
    def test_overlong_dialogue_ladder_moves_from_revision_to_split_patch(self) -> None:
        from agent.critic import MAX_DIALOGUE_CHARS_PER_SHOT, critique_audio

        project_id = "e2e-dialogue"
        store = self.store(project_id)
        long_dialogue = "字" * (MAX_DIALOGUE_CHARS_PER_SHOT + 40)
        _seed_project_shots(project_id, [{"id": "shot-1", "dialogue": long_dialogue, "storyboard_path": "a.png"}])
        self.addCleanup(_cleanup_project, project_id)

        critique = critique_audio([{"shot_id": "shot-1", "dialogue": long_dialogue}], [])
        self.assertFalse(critique.passed)
        self.assertEqual(critique.issues[0].code, "dialogue_too_long")

        state = _base_state(project_id, critiques=[critique.model_dump(mode="json")])
        with patch.object(graph.CheckpointStore, "get", return_value=store):
            # 恢复阶梯第一步：先收紧 Prompt（修正），而不是直接拆镜头。
            first = graph._recovery_node(state, StageName.AUDIO_PRODUCTION, default_target="audio_production")
        self.assertEqual(first["selected_strategy"], RecoveryStrategy.REVISE_PROMPT.value)
        self.assertTrue(first["prompt_revisions"])
        self.assertEqual(first["recovery_attempts"][StageName.AUDIO_PRODUCTION.value], 1)

        ladder_state = {**state, **first}
        with patch.object(graph.CheckpointStore, "get", return_value=store):
            second = graph._recovery_node(ladder_state, StageName.AUDIO_PRODUCTION, default_target="audio_production")
        self.assertEqual(second["selected_strategy"], RecoveryStrategy.SPLIT_SHOT.value)
        split_decision = _last_decision(store)
        patch_row = next(p for p in split_decision["selected"]["prompt_patches"] if p["field"] == "dialogue")
        self.assertEqual(patch_row["op"], "split")
        self.assertEqual(patch_row["value"]["max_chars"], 90)
        # 拆镜真实落库（版本围栏内），恢复范围收敛到拆分后的镜头。
        db = SessionLocal()
        try:
            self.assertGreater(db.query(Shot).filter(Shot.project_id == project_id).count(), 1)
        finally:
            db.close()
        self.assertTrue(second["pending_shot_ids"])


class Scenario5ReferenceUnsupportedTests(_TempRootTestCase):
    def test_reference_unsupported_ladder_is_fully_traced(self) -> None:
        from agent import decision as decision_module
        from agent.contracts import ProviderProfile

        project_id = "e2e-ref-unsupported"
        store = self.store(project_id)
        no_ref_profiles = [
            ProviderProfile(
                capability="image",
                provider="mock-no-ref",
                available=True,
                supports_reference_images=False,
                reliability=0.9,
            )
        ]
        with_ref_profiles = no_ref_profiles + [
            ProviderProfile(
                capability="image",
                provider="mock-with-ref",
                available=True,
                supports_reference_images=True,
                reliability=0.95,
            )
        ]
        artifacts = [
            {
                "shot_id": "shot-1",
                "shot_version": 1,
                "stage": StageName.IMAGE_GENERATION.value,
                "status": StageStatus.FAILED.value,
                "path": "",
                "failure": {
                    "kind": "provider_reference_unsupported",
                    "stage": StageName.IMAGE_GENERATION.value,
                    "shot_id": "shot-1",
                    "message": "references_sent=0 当前 Provider 不支持参考图",
                },
            }
        ]

        def make_state(**extra) -> dict:
            return _base_state(
                project_id,
                shot_artifacts=artifacts,
                failed_shot_ids=["shot-1"],
                stage_status={StageName.IMAGE_GENERATION.value: StageStatus.FAILED.value},
                critiques=[
                    {
                        "stage": StageName.IMAGE_GENERATION.value,
                        "passed": False,
                        "score": 0.2,
                        "issues": [
                            {
                                "code": "provider_reference_unsupported",
                                "severity": "error",
                                "message": "不支持参考图",
                                "shot_id": "shot-1",
                            }
                        ],
                    }
                ],
                **extra,
            )

        unlimited = {"level": "unlimited", "remaining_cost_micro": None, "remaining_seconds": None}
        # 场景 A：没有任何支持参考图的 Provider 时，切换候选被明确淘汰并记录原因。
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(decision_module, "provider_profiles", lambda capability, **kwargs: list(no_ref_profiles)),
            patch.object(decision_module, "budget_snapshot", lambda project_id="": dict(unlimited)),
        ):
            no_ref_decision = asyncio.run(graph._image_decision(make_state()))
        decision = _last_decision(store)
        self.assertEqual(decision["failure"]["kind"], "provider_reference_unsupported")
        rejected = {item["strategy"]: item["reason"] for item in decision["considered_rejected"]}
        self.assertIn(RecoveryStrategy.SWITCH_PROVIDER.value, rejected)
        self.assertIn("Provider 能力不匹配", rejected[RecoveryStrategy.SWITCH_PROVIDER.value])
        self.assertIn(
            no_ref_decision["selected_strategy"],
            {RecoveryStrategy.REPLACE_REFERENCE.value, RecoveryStrategy.REVISE_PROMPT.value},
        )
        self.assertNotEqual(no_ref_decision["selected_strategy"], RecoveryStrategy.HUMAN_REVIEW.value)

        # 场景 B：存在支持参考图的 Provider 时，阶梯依次修正 Prompt → 替换参考 → 切换 Provider。
        # attempted_strategies 来自 recovery_history，每轮把上一轮的恢复记录累积进状态。
        def ladder_state(history: list[dict], **extra) -> dict:
            return make_state(recovery_history=history, **extra)

        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(decision_module, "provider_profiles", lambda capability, **kwargs: list(with_ref_profiles)),
            patch.object(decision_module, "budget_snapshot", lambda project_id="": dict(unlimited)),
        ):
            history: list[dict] = []
            step1 = graph._recovery_node(
                ladder_state(history), StageName.IMAGE_GENERATION, default_target="image_generation"
            )
            history += step1["recovery_history"]
            step2 = graph._recovery_node(
                ladder_state(history), StageName.IMAGE_GENERATION, default_target="image_generation"
            )
            history += step2["recovery_history"]
            step3 = graph._recovery_node(
                ladder_state(history, provider_switch={StageName.IMAGE_GENERATION.value: "mock-no-ref"}),
                StageName.IMAGE_GENERATION,
                default_target="image_generation",
            )
        self.assertEqual(step1["selected_strategy"], RecoveryStrategy.REVISE_PROMPT.value)
        self.assertEqual(step2["selected_strategy"], RecoveryStrategy.REPLACE_REFERENCE.value)
        self.assertEqual(step3["selected_strategy"], RecoveryStrategy.SWITCH_PROVIDER.value)
        self.assertEqual(step3["provider_switch"][StageName.IMAGE_GENERATION.value], "mock-with-ref")
        patch_row = next(p for p in step2["prompt_revisions"][0]["patches"] if p["field"] == "reference_images")
        self.assertEqual(patch_row["op"], "replace")
        switch_decision = _last_decision(store)
        self.assertEqual(switch_decision["selected"]["strategy"], RecoveryStrategy.SWITCH_PROVIDER.value)
        self.assertEqual(switch_decision["selected"]["provider"], "mock-with-ref")


class Scenario6PartialFailureTests(_TempRootTestCase):
    def test_one_failed_shot_keeps_successes_and_only_reruns_failed(self) -> None:
        project_id = "e2e-partial"
        store = self.store(project_id)
        image = _image_file(self.root, "ok.png")
        calls: list[str] = []
        fail_first = {"shot-2": True}

        async def worker(shot_id: str, version: int, **kwargs) -> dict:
            calls.append(shot_id)
            if shot_id == "shot-2" and fail_first["shot-2"]:
                raise RuntimeError("image provider failed")
            return {
                "shot_id": shot_id,
                "shot_version": version,
                "status": "succeeded",
                "path": image,
                "provider": "mock-image",
            }

        state = _base_state(project_id)
        shot_versions = _shot_versions_filter({"shot-1": 1, "shot-2": 1})
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(graph, "_reference_gate", return_value={}),
            patch.object(graph, "_shot_versions", side_effect=shot_versions),
            patch.object(graph, "generate_storyboard_shot", side_effect=worker),
        ):
            first = asyncio.run(_run_image_stage(state))
        self.assertEqual(first["stage_status"][StageName.IMAGE_GENERATION.value], StageStatus.DEGRADED.value)
        self.assertEqual(first["successful_shot_ids"], ["shot-1"])
        self.assertEqual(first["failed_shot_ids"], ["shot-2"])
        self.assertEqual(graph._route_image_decision(first), "recover")
        self.assertEqual(
            store.shot_artifact("shot-1", StageName.IMAGE_GENERATION.value)["status"], StageStatus.SUCCEEDED.value
        )

        fail_first["shot-2"] = False
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(graph, "_reference_gate", return_value={}),
            patch.object(graph, "_shot_versions", side_effect=shot_versions),
            patch.object(graph, "generate_storyboard_shot", side_effect=worker),
        ):
            second = asyncio.run(graph._image_generation_fan_out({**state, "pending_shot_ids": ["shot-2"]}))
        # 局部补拍只覆盖失败镜头；成功镜头不会被重算。
        self.assertEqual(calls.count("shot-1"), 1)
        self.assertEqual(calls.count("shot-2"), 2)
        self.assertEqual(second["successful_shot_ids"], ["shot-2"])
        # 检查点视角下两个镜头的最终状态都收敛为成功（fan-in 聚合口径）。
        rows = [store.shot_artifact(shot_id, StageName.IMAGE_GENERATION.value) for shot_id in ("shot-1", "shot-2")]
        merged = shot_work.fan_in_shot_results(rows)
        self.assertEqual([item["shot_id"] for item in merged["successes"]], ["shot-1", "shot-2"])
        self.assertEqual(merged["failure_count"], 0)


class Scenario7VersionConflictTests(_TempRootTestCase):
    def test_user_edit_invalidates_changed_shot_and_recovery_selects_resume(self) -> None:
        project_id = "e2e-version-conflict"
        _seed_project_shots(
            project_id,
            [
                {"id": "shot-1", "storyboard_path": "a.png"},
                {"id": "shot-2", "storyboard_path": "b.png"},
            ],
        )
        self.addCleanup(_cleanup_project, project_id)
        store = self.store(project_id)
        image = _image_file(self.root, "storyboard.png")
        calls: list[tuple[str, int]] = []
        conflict = {"on": False}

        async def worker(shot_id: str, version: int, **kwargs) -> dict:
            calls.append((shot_id, version))
            if conflict["on"] and shot_id == "shot-2":
                raise RuntimeError("shot version changed: expected 1, got 2")
            return {
                "shot_id": shot_id,
                "shot_version": version,
                "status": "succeeded",
                "path": image,
                "provider": "mock-image",
            }

        state = _base_state(project_id)
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(graph, "_reference_gate", return_value={}),
            patch.object(graph, "generate_storyboard_shot", side_effect=worker),
        ):
            asyncio.run(_run_image_stage(state))

            # 用户修改镜头 2：版本 +1，触发版本围栏与检查点失效。
            db = SessionLocal()
            try:
                row = db.query(Shot).filter(Shot.id == "shot-2").first()
                row.version = int(row.version or 1) + 1
                row.dialogue = "用户改写的台词"
                db.commit()
            finally:
                db.close()
            changes = store.detect_changes()
            self.assertEqual(changes["changed_shot_ids"], ["shot-2"])
            self.assertEqual(store.shot_artifact("shot-2", StageName.IMAGE_GENERATION.value)["status"], "invalidated")
            self.assertEqual(store.shot_artifact("shot-1", StageName.IMAGE_GENERATION.value)["status"], "succeeded")

            conflict["on"] = True
            conflicted = asyncio.run(_run_image_stage(state))
        decision = _last_decision(store)
        self.assertEqual(decision["failure"]["kind"], "version_conflict")
        self.assertEqual(decision["selected"]["strategy"], RecoveryStrategy.RESUME_CHECKPOINT.value)
        self.assertEqual(conflicted["stage_status"][StageName.IMAGE_GENERATION.value], StageStatus.DEGRADED.value)
        self.assertEqual(
            store.shot_artifact("shot-1", StageName.IMAGE_GENERATION.value)["status"], StageStatus.SUCCEEDED.value
        )

        conflict["on"] = False
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(graph, "_reference_gate", return_value={}),
            patch.object(graph, "generate_storyboard_shot", side_effect=worker),
        ):
            repaired = asyncio.run(graph._image_generation_fan_out({**state, "pending_shot_ids": ["shot-2"]}))
        self.assertEqual(repaired["successful_shot_ids"], ["shot-2"])
        self.assertIn(("shot-2", 2), calls)
        self.assertEqual(
            store.shot_artifact("shot-2", StageName.IMAGE_GENERATION.value)["status"], StageStatus.SUCCEEDED.value
        )


class Scenario8ProcessRestartTests(_TempRootTestCase):
    def test_stage_and_shot_checkpoints_survive_process_restart(self) -> None:
        project_id = "e2e-restart"
        image = _image_file(self.root, "restart.png")
        parser_calls = {"count": 0}
        worker_calls = {"count": 0}

        async def parser(state: dict) -> dict:
            parser_calls["count"] += 1
            return _parsed_payload()

        async def worker(shot_id: str, version: int, **kwargs) -> dict:
            worker_calls["count"] += 1
            return {
                "shot_id": shot_id,
                "shot_version": version,
                "status": "succeeded",
                "path": image,
                "provider": "mock-image",
            }

        state = _base_state(project_id, run_id="run-1")
        store_a = self.store(project_id, "run-1")
        fake_parser = types.SimpleNamespace(run=parser)
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store_a),
            patch.object(agent_nodes, "script_parser", fake_parser, create=True),
        ):
            first = asyncio.run(graph._director_planning(state))
        self.assertTrue(store_a.path.exists())

        # 模拟进程退出：从磁盘重建 store（绕过进程内缓存），续跑同一 run。
        store_b = CheckpointStore(project_id, "run-1", root=self.root)
        self.assertEqual(store_b.stage(StageName.DIRECTOR_PLANNING.value)["status"], StageStatus.SUCCEEDED.value)
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store_b),
            patch.object(agent_nodes, "script_parser", fake_parser, create=True),
        ):
            resumed = asyncio.run(graph._director_planning({**state, **first}))
        self.assertEqual(parser_calls["count"], 1)
        self.assertEqual(resumed["stage_status"][StageName.DIRECTOR_PLANNING.value], StageStatus.SUCCEEDED.value)

        with (
            patch.object(graph.CheckpointStore, "get", return_value=store_a),
            patch.object(graph, "_reference_gate", return_value={}),
            patch.object(graph, "_shot_versions", return_value={"shot-1": 1, "shot-2": 1}),
            patch.object(graph, "generate_storyboard_shot", side_effect=worker),
        ):
            asyncio.run(graph._image_generation_fan_out(state))

        store_c = CheckpointStore(project_id, "run-1", root=self.root)
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store_c),
            patch.object(graph, "_reference_gate", return_value={}),
            patch.object(graph, "_shot_versions", return_value={"shot-1": 1, "shot-2": 1}),
            patch.object(graph, "generate_storyboard_shot", side_effect=worker),
        ):
            resumed_shots = asyncio.run(graph._image_generation_fan_out(state))
        self.assertEqual(worker_calls["count"], 2)
        self.assertEqual(sorted(resumed_shots["successful_shot_ids"]), ["shot-1", "shot-2"])


class Scenario9CandidateSelectionTests(_TempRootTestCase):
    def test_structural_filter_autoselects_highest_score_candidate(self) -> None:
        project_id = "e2e-candidates"
        store = self.store(project_id)
        video = _video_file(self.root, "candidate.mp4")

        async def worker(shot_id: str, version: int, **kwargs) -> dict:
            return {
                "shot_id": shot_id,
                "shot_version": version,
                "status": "succeeded",
                "path": video,
                "provider": "mock-video",
                "video_candidates": [
                    {
                        "candidate_id": "c-broken",
                        "shot_id": shot_id,
                        "shot_version": version,
                        "status": "succeeded",
                        "path": video,
                        "provider": "mock-video",
                        "score": 0.95,
                        "structural_passed": False,
                    },
                    {
                        "candidate_id": "c-good",
                        "shot_id": shot_id,
                        "shot_version": version,
                        "status": "succeeded",
                        "path": video,
                        "provider": "mock-video",
                        "score": 0.8,
                        "structural_passed": True,
                    },
                    {
                        "candidate_id": "c-failed",
                        "shot_id": shot_id,
                        "shot_version": version,
                        "status": "failed",
                        "path": "",
                        "provider": "mock-video",
                        "score": 0.0,
                        "structural_passed": None,
                    },
                ],
                "selected_video_candidate_id": "c-good",
                "candidate_selection": {
                    "candidate_id": "c-good",
                    "score": 0.8,
                    "reason": "structural_pass_highest_score",
                    "considered": ["c-broken", "c-good", "c-failed"],
                    "rejected": [{"candidate_id": "c-broken", "reason": "structural_check_failed"}],
                },
            }

        state = _base_state(project_id)
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(graph, "_reference_gate", return_value={}),
            patch.object(graph, "_shot_versions", return_value={"shot-1": 1}),
            patch.object(graph, "generate_video_shot", side_effect=worker),
        ):
            from services.quality_review_service import quality_review_service

            with patch.object(quality_review_service, "storyboard_gate_status", lambda pid, **_kwargs: {"ok": True}):
                result = asyncio.run(_run_video_stage(state))
        artifact = next(item for item in result["shot_artifacts"] if item["shot_id"] == "shot-1")
        self.assertEqual(artifact["selected_video_candidate_id"], "c-good")
        selection = select_video_candidate([dict(item, shot_version=1) for item in artifact["video_candidates"]])
        self.assertEqual(selection.candidate_id, "c-good")
        self.assertEqual(selection.reason, "structural_pass_highest_score")
        rejected = {item["candidate_id"]: item["reason"] for item in selection.rejected}
        self.assertEqual(rejected["c-broken"], "structural_check_failed")

        row = store.shot_artifact("shot-1", StageName.VIDEO_GENERATION.value)
        self.assertEqual(row["selected_video_candidate_id"], "c-good")
        # 候选历史也落在检查点的候选存储里：自动选择可被标记并防止版本冲突。
        for candidate in artifact["video_candidates"]:
            store.save_video_candidate("shot-1", candidate)
        store.mark_video_candidate_selected("shot-1", "c-good", shot_version=1, reason="structural_pass_highest_score")
        persisted = next(item for item in store.video_candidates("shot-1") if item["candidate_id"] == "c-good")
        self.assertTrue(persisted["selected"])
        self.assertEqual(persisted["selection_reason"], "structural_pass_highest_score")
        with self.assertRaises(RuntimeError):
            store.mark_video_candidate_selected("shot-1", "c-good", shot_version=2, reason="wrong version")


class Scenario10AutoCompletionTests(_TempRootTestCase):
    def test_auto_mode_completes_final_video_end_to_end(self) -> None:
        from agent.critic import CritiqueReport
        from api.routes import render as render_route
        from api.routes import script as script_route
        from services.quality_review_service import quality_review_service

        project_id = "e2e-auto-complete"
        image_a = _image_file(self.root, "auto-1.png")
        image_b = _image_file(self.root, "auto-2.png")
        video_a = _video_file(self.root, "auto-1.mp4")
        video_b = _video_file(self.root, "auto-2.mp4")
        final_path = _video_file(self.root, "final.mp4")
        _seed_project_shots(
            project_id,
            [
                {"id": "shot-1", "storyboard_path": image_a, "video_path": video_a},
                {"id": "shot-2", "storyboard_path": image_b, "video_path": video_b},
            ],
        )
        self.addCleanup(_cleanup_project, project_id)
        store = self.store(project_id, "auto")

        async def parser(state: dict) -> dict:
            return _parsed_payload()

        async def storyboard(state: dict) -> dict:
            return _storyboard_payload()

        async def image_worker(shot_id: str, version: int, **kwargs) -> dict:
            return {
                "shot_id": shot_id,
                "shot_version": version,
                "status": "succeeded",
                "path": image_a if shot_id == "shot-1" else image_b,
                "provider": "mock-image",
                "cost_micro": 100,
                "duration_ms": 900,
            }

        async def video_worker(shot_id: str, version: int, **kwargs) -> dict:
            return {
                "shot_id": shot_id,
                "shot_version": version,
                "status": "succeeded",
                "path": video_a if shot_id == "shot-1" else video_b,
                "provider": "mock-video",
                "model": "mock-video-v1",
                "cost_micro": 200,
                "duration_ms": 1500,
                "score": 0.9,
                "video_candidates": [
                    {
                        "candidate_id": f"{shot_id}-c1",
                        "shot_id": shot_id,
                        "shot_version": version,
                        "status": "succeeded",
                        "path": video_a if shot_id == "shot-1" else video_b,
                        "provider": "mock-video",
                        "model": "mock-video-v1",
                        "score": 0.9,
                        "structural_passed": True,
                        "selected": True,
                        "selection_reason": "structural_pass_highest_score",
                        "reference_manifest": [{"kind": "character", "name": "主角三视图"}],
                    }
                ],
                "selected_video_candidate_id": f"{shot_id}-c1",
                "candidate_selection": {"candidate_id": f"{shot_id}-c1", "reason": "structural_pass_highest_score"},
            }

        def passing_video_critique(artifacts) -> CritiqueReport:
            # stage 必须是 video_generation：决策节点按阶段名检索最新 Critique。
            return CritiqueReport(stage=StageName.VIDEO_GENERATION.value, passed=True, score=1.0)

        fake_parser = types.SimpleNamespace(run=parser)
        fake_storyboard = types.SimpleNamespace(run=storyboard)
        initial = _base_state(project_id, run_id="auto", output_format="9:16", resolution="720p")
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(agent_nodes, "script_parser", fake_parser, create=True),
            patch.object(agent_nodes, "storyboard_gen", fake_storyboard, create=True),
            patch.object(script_route, "_ensure_character_reference_images", new_callable=AsyncMock),
            patch.object(script_route, "_ensure_scene_baseline_images", new_callable=AsyncMock),
            patch.object(graph, "_persist_phase1_idempotent", new_callable=AsyncMock),
            patch.object(graph, "refresh_project_reference_state_for_graph", return_value={"blocking": False}),
            patch.object(
                graph,
                "provider_profiles",
                lambda *args, **kwargs: [types.SimpleNamespace(supports_reference_images=True, available=True)],
            ),
            patch.object(graph, "_reference_gate", return_value={}),
            patch.object(
                graph,
                "_run_storyboard_quality_gate",
                new_callable=AsyncMock,
                return_value={"passed": True, "reviews": {}},
            ),
            patch.object(graph, "generate_storyboard_shot", side_effect=image_worker),
            patch.object(quality_review_service, "storyboard_gate_status", lambda pid, **_kwargs: {"ok": True}),
            patch.object(quality_review_service, "video_gate_status", lambda pid, **_kwargs: {"ok": True}),
            patch.object(graph, "generate_video_shot", side_effect=video_worker),
            patch.object(graph, "critique_videos", side_effect=passing_video_critique),
            patch.object(graph, "_review_shot_videos", new_callable=AsyncMock, return_value={}),
            patch.object(render_route, "_render_task", new_callable=AsyncMock),
            patch.object(
                render_route, "_render_status", {project_id: {"status": "completed", "video_path": final_path}}
            ),
        ):
            result = asyncio.run(graph.get_graph().ainvoke(initial, config={"recursion_limit": 160}))

        self.assertEqual(result.get("errors"), [])
        self.assertEqual(result["stage_status"][StageName.FINAL_REVIEW.value], StageStatus.SUCCEEDED.value)
        self.assertEqual(result["video_path"], final_path)
        self.assertTrue(result.get("storyboard_confirmed"))
        self.assertIn("final_report", result)
        summary = store.trace_summary()
        self.assertEqual(summary["run"]["current_stage"], StageName.FINAL_REVIEW.value)
        self.assertTrue(all(row["status"] == StageStatus.SUCCEEDED.value for row in summary["stages"]))
        self.assertEqual(summary["totals"]["cost_micro"], 600)
        self.assertEqual(summary["totals"]["selected_video_candidates"], 2)
        self.assertFalse(summary["degradations"])


class Scenario11NoHumanGateTests(_TempRootTestCase):
    def test_persistent_auto_failure_ends_in_auto_abort_never_human(self) -> None:
        project_id = "e2e-auto-abort"
        store = self.store(project_id, "auto")

        async def broken_parser(state: dict) -> dict:
            raise RuntimeError("LLM 输出不是合法 JSON")

        fake_parser = types.SimpleNamespace(run=broken_parser)
        initial = _base_state(project_id, run_id="auto")
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(agent_nodes, "script_parser", fake_parser, create=True),
        ):
            result = asyncio.run(graph.get_graph().ainvoke(initial, config={"recursion_limit": 80}))

        self.assertEqual(result["run_status"], RunStatus.FAILED.value)
        self.assertEqual(result["current_step"], "auto_abort")
        self.assertNotIn("needs_human_review", {key for key, value in result.items() if value})
        snapshot = store.snapshot()
        self.assertEqual(snapshot["status"], RunStatus.FAILED.value)
        self.assertTrue(any(event["event"] == "auto_abort" for event in snapshot["events"]))
        self.assertFalse(any(event["event"] == "human_gate" for event in snapshot["events"]))
        for decision in store.decisions():
            selected = decision.get("selected") or {}
            self.assertNotEqual(selected.get("strategy"), RecoveryStrategy.HUMAN_REVIEW.value)


class Scenario12BudgetExhaustedTests(_TempRootTestCase):
    def test_zero_budget_terminates_or_degrades_with_recorded_reasons(self) -> None:
        from agent import decision as decision_module

        project_id = "e2e-budget"
        store = self.store(project_id)
        empty_budget = {"level": "project", "remaining_cost_micro": 0, "remaining_seconds": 0}

        failed_only = [
            {
                "shot_id": "shot-1",
                "shot_version": 1,
                "stage": StageName.IMAGE_GENERATION.value,
                "status": StageStatus.FAILED.value,
                "path": "",
                "failure": {
                    "kind": "image_failed",
                    "stage": StageName.IMAGE_GENERATION.value,
                    "shot_id": "shot-1",
                    "message": "image provider failed",
                },
            }
        ]
        state = _base_state(
            project_id,
            shot_artifacts=failed_only,
            failed_shot_ids=["shot-1"],
            stage_status={StageName.IMAGE_GENERATION.value: StageStatus.FAILED.value},
            critiques=[
                {
                    "stage": StageName.IMAGE_GENERATION.value,
                    "passed": False,
                    "score": 0.1,
                    "issues": [
                        {
                            "code": "image_generation_failure",
                            "severity": "error",
                            "message": "生成失败",
                            "shot_id": "shot-1",
                        }
                    ],
                }
            ],
        )
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(decision_module, "budget_snapshot", lambda project_id="": dict(empty_budget)),
        ):
            recovery = graph._recovery_node(state, StageName.IMAGE_GENERATION, default_target="image_generation")
        self.assertEqual(recovery["selected_strategy"], RecoveryStrategy.TERMINAL_FAILURE.value)
        self.assertEqual(recovery["run_status"], RunStatus.FAILED.value)
        decision = _last_decision(store)
        rejected = {item["strategy"]: item["reason"] for item in decision["considered_rejected"]}
        self.assertTrue(any("超出剩余预算" in reason for reason in rejected.values()))
        self.assertNotEqual(decision["selected"]["strategy"], RecoveryStrategy.HUMAN_REVIEW.value)

        # 存在结构完整可用结果时：预算耗尽必须走降级发布而不是悄悄失败。
        usable = [
            {
                "shot_id": "shot-2",
                "shot_version": 1,
                "stage": StageName.IMAGE_GENERATION.value,
                "status": StageStatus.SUCCEEDED.value,
                "path": "usable.png",
                "provider": "mock-image",
                "structural_passed": True,
                "technical_passed": True,
            }
        ]
        degraded_state = _base_state(
            project_id,
            shot_artifacts=usable,
            stage_status={StageName.IMAGE_GENERATION.value: StageStatus.FAILED.value},
            critiques=[
                {
                    "stage": StageName.IMAGE_GENERATION.value,
                    "passed": False,
                    "score": 0.4,
                    "failure_kind": "budget_exceeded",
                    "issues": [{"code": "budget_exceeded", "severity": "error", "message": "预算不足"}],
                }
            ],
        )
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(decision_module, "budget_snapshot", lambda project_id="": dict(empty_budget)),
        ):
            degraded_decision = asyncio.run(graph._image_decision(degraded_state))
            degraded_state = {**degraded_state, **degraded_decision}
            published = asyncio.run(graph._degraded_publish(degraded_state))
        self.assertEqual(degraded_decision["selected_strategy"], RecoveryStrategy.DEGRADED_PUBLISH.value)
        self.assertTrue(published["degraded_published"])
        snapshot = store.snapshot()
        self.assertEqual(snapshot["status"], RunStatus.DEGRADED.value)
        self.assertTrue(any(event["event"] == "degraded_publish" for event in snapshot["events"]))
        summary = store.trace_summary()
        self.assertTrue(summary["degradations"])


class Scenario13FinalFeedbackTests(_TempRootTestCase):
    def test_final_review_feedback_recomputes_only_affected_shot(self) -> None:
        project_id = "e2e-final-feedback"
        store = self.store(project_id)
        video = _video_file(self.root, "feedback.mp4")
        critique = {
            "stage": StageName.FINAL_REVIEW.value,
            "passed": False,
            "score": 0.5,
            "failure_kind": "quality_below_threshold",
            "issues": [
                {
                    "code": "video_frozen",
                    "severity": "error",
                    "message": "镜头 shot-2 画面冻结",
                    "shot_id": "shot-2",
                    "details": {"source_stage": StageName.VIDEO_GENERATION.value},
                },
            ],
        }
        state = _base_state(
            project_id,
            critiques=[critique],
            shot_artifacts=[
                {
                    "shot_id": "shot-1",
                    "shot_version": 1,
                    "stage": StageName.VIDEO_GENERATION.value,
                    "status": "succeeded",
                    "path": video,
                    "provider": "mock-video",
                },
                {
                    "shot_id": "shot-2",
                    "shot_version": 1,
                    "stage": StageName.VIDEO_GENERATION.value,
                    "status": "succeeded",
                    "path": video,
                    "provider": "mock-video",
                },
            ],
            stage_status={StageName.FINAL_REVIEW.value: StageStatus.FAILED.value},
        )
        calls: list[str] = []

        async def worker(shot_id: str, version: int, **kwargs) -> dict:
            calls.append(shot_id)
            return {
                "shot_id": shot_id,
                "shot_version": version,
                "status": "succeeded",
                "path": video,
                "provider": "mock-video",
            }

        from services.quality_review_service import quality_review_service

        shot_versions = _shot_versions_filter({"shot-1": 1, "shot-2": 1})
        with (
            patch.object(graph.CheckpointStore, "get", return_value=store),
            patch.object(quality_review_service, "storyboard_gate_status", lambda pid, **_kwargs: {"ok": True}),
            patch.object(graph, "_shot_versions", side_effect=shot_versions),
            patch.object(graph, "generate_video_shot", side_effect=worker),
        ):
            recovery = asyncio.run(graph._final_recovery(state))
            self.assertEqual(recovery["pending_recovery_target"], StageName.VIDEO_GENERATION.value)
            self.assertEqual(recovery["pending_shot_ids"], ["shot-2"])
            self.assertEqual(graph._route_final_recovery(recovery), "video")
            asyncio.run(graph._video_generation_fan_out({**state, **recovery}))
        self.assertEqual(calls, ["shot-2"])
        events = [event for event in store.snapshot()["events"] if event["event"] == "recovery_selected"]
        self.assertTrue(any(event["stage"] == StageName.FINAL_REVIEW.value for event in events))


if __name__ == "__main__":
    unittest.main()
