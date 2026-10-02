"""可解释追踪汇总契约测试：/api/graph/runs/:id/trace 必须展示的字段完整性。"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from db import SessionLocal, init_db  # noqa: E402
from models import Project, Shot  # noqa: E402
from agent.checkpoints import CheckpointStore, summarize_trace  # noqa: E402
from agent.contracts import StageName  # noqa: E402
from api.routes import graph as graph_route  # noqa: E402


def _seed_trace_data(store: CheckpointStore) -> None:
    store.add_event("stage_entered", stage=StageName.IMAGE_GENERATION.value, recovery=False)
    store.save_stage(
        StageName.IMAGE_GENERATION.value,
        status="degraded",
        input_fingerprint="fp-1",
        critique={
            "stage": StageName.IMAGE_GENERATION.value, "passed": False, "score": 0.42,
            "issues": [{"code": "image_invalid", "severity": "error", "message": "镜头 s2 图片结构不合格", "shot_id": "s2", "recommendation": "只重生成该镜头"}],
            "proposed_changes": ["收紧 Prompt"],
        },
    )
    store.save_shot_artifact(
        "s1", StageName.IMAGE_GENERATION.value, shot_version=1, status="succeeded",
        path="s1.png", provider="mock-image", cost_micro=100, duration_ms=1200, score=0.9,
        input_fingerprint="fp-1",
        extra={
            "selected_video_candidate_id": "s1-c1",
            "candidate_selection": {"candidate_id": "s1-c1", "reason": "structural_pass_highest_score"},
            "video_candidates": [{
                "candidate_id": "s1-c1", "shot_id": "s1", "shot_version": 1, "status": "succeeded",
                "path": "s1.mp4", "provider": "mock-video", "model": "mock-video-v1", "score": 0.88,
                "generation_duration_ms": 8000, "structural_passed": True, "selected": True,
                "selection_reason": "structural_pass_highest_score",
                "reference_manifest": [{"kind": "character", "name": "主角三视图", "path": "ref.png"}],
            }],
        },
    )
    store.save_shot_artifact(
        "s2", StageName.IMAGE_GENERATION.value, shot_version=1, status="failed",
        path="", provider="mock-image", cost_micro=30, duration_ms=400,
        input_fingerprint="fp-1",
        failure={"kind": "image_failed", "stage": StageName.IMAGE_GENERATION.value, "shot_id": "s2", "message": "provider failed"},
    )
    store.add_decision({
        "trace_id": "trace-1", "stage": StageName.IMAGE_GENERATION.value, "mode": "auto", "shot_id": "s2",
        "failure": {"kind": "image_failed", "message": "provider failed"},
        "quality_score": 0.42, "retries_remaining": 1, "reason": "自动恢复：优先策略=revise_prompt",
        "selected": {
            "strategy": "revise_prompt", "provider": "", "target_stage": StageName.IMAGE_GENERATION.value,
            "shot_ids": ["s2"], "estimated_cost_micro": 13000, "estimated_seconds": 6, "score": 0.71,
            "prompt_patches": [{"field": "visual_prompt", "op": "replace", "value": {"rule": "close-up"}, "shot_id": "s2", "target_stage": StageName.IMAGE_GENERATION.value, "reason": "结果不合格，收紧描述"}],
        },
        "candidates": [{"strategy": "switch_provider", "provider": "mock-b", "score": 0.66, "estimated_cost_micro": 20000, "estimated_seconds": 8}],
        "considered_rejected": [{"strategy": "switch_provider", "score": 0.66, "reason": "综合排序低于选中策略（选中=revise_prompt，rank=[0, 0.66, 0.95]）"}],
        "budget_snapshot": {"level": "project", "remaining_cost_micro": 50000, "remaining_seconds": 600},
        "provider_profiles": [{"capability": "image", "provider": "mock-image", "model": "mock-image-v2", "available": True, "supports_reference_images": True}],
    })
    store.add_event("recovery_selected", stage=StageName.IMAGE_GENERATION.value, shot_ids=["s2"], strategy="revise_prompt", trace_id="trace-1")
    store.set_status("degraded", reason="自动恢复无法继续，按降级结果发布")


class TraceSummaryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.store = CheckpointStore("trace-project", "auto", root=self.root)

    def test_summary_contains_every_required_display_field(self) -> None:
        _seed_trace_data(self.store)
        summary = self.store.trace_summary()

        # 当前阶段 + 运行状态
        self.assertEqual(summary["run"]["current_stage"], StageName.IMAGE_GENERATION.value)
        self.assertEqual(summary["run"]["status"], "degraded")
        self.assertEqual(summary["run"]["status_reason"], "自动恢复无法继续，按降级结果发布")

        # 阶段质量分 + Critic 问题
        image_stage = next(row for row in summary["stages"] if row["stage"] == StageName.IMAGE_GENERATION.value)
        self.assertEqual(image_stage["quality"]["score"], 0.42)
        self.assertEqual(image_stage["quality"]["issues"][0]["code"], "image_invalid")

        # 镜头状态 / Provider / 成本 / 实际耗时 / 候选结果 / 自动选择
        s1 = next(row for row in summary["shots"] if row["shot_id"] == "s1")
        self.assertEqual(s1["stages"][StageName.IMAGE_GENERATION.value]["provider"], "mock-image")
        self.assertEqual(s1["cost_micro"], 100)
        self.assertEqual(s1["duration_ms"], 1200)
        self.assertEqual(s1["selected_video_candidate_id"], "s1-c1")
        candidate = s1["video_candidates"][0]
        self.assertEqual(candidate["provider"], "mock-video")
        self.assertEqual(candidate["model"], "mock-video-v1")
        self.assertEqual(candidate["generation_duration_ms"], 8000)
        self.assertEqual(candidate["reference_manifest"][0]["name"], "主角三视图")
        s2 = next(row for row in summary["shots"] if row["shot_id"] == "s2")
        self.assertEqual(s2["stages"][StageName.IMAGE_GENERATION.value]["failure_kind"], "image_failed")

        # 最终决策 / RecoveryCandidate / 淘汰原因 / Prompt 修改 / 预计成本与时长
        decision = summary["decisions"][0]
        self.assertEqual(decision["selected"]["strategy"], "revise_prompt")
        self.assertEqual(decision["selected"]["estimated_cost_micro"], 13000)
        self.assertEqual(decision["selected"]["estimated_seconds"], 6)
        self.assertEqual(decision["candidates"][0]["provider"], "mock-b")
        self.assertIn("综合排序低于选中策略", decision["considered_rejected"][0]["reason"])
        self.assertEqual(decision["provider_profiles"][0]["model"], "mock-image-v2")
        self.assertEqual(summary["prompt_changes"][0]["field"], "visual_prompt")
        self.assertEqual(summary["prompt_changes"][0]["shot_id"], "s2")

        # 自动降级原因 / 检查点与恢复次数
        self.assertTrue(any("降级" in item["reason"] for item in summary["degradations"]))
        self.assertGreaterEqual(summary["counters"]["checkpoint_records"], 3)
        self.assertEqual(summary["counters"]["decisions"], 1)
        self.assertEqual(summary["counters"]["recoveries"], 1)
        self.assertEqual(summary["totals"]["cost_micro"], 130)
        self.assertEqual(summary["totals"]["selected_video_candidates"], 1)

    def test_summary_is_empty_but_well_formed_for_fresh_run(self) -> None:
        summary = self.store.trace_summary()
        self.assertEqual(summary["run"]["status"], "pending")
        self.assertEqual(summary["run"]["current_stage"], "")
        self.assertEqual(summary["stages"], [])
        self.assertEqual(summary["shots"], [])
        self.assertEqual(summary["decisions"], [])
        self.assertEqual(summary["counters"]["checkpoint_records"], 0)

    def test_trace_route_merges_db_reference_manifests(self) -> None:
        init_db()
        project_id = "trace-route-merge"
        store = CheckpointStore(project_id, "auto", root=self.root)
        _seed_trace_data(store)
        db = SessionLocal()
        try:
            db.query(Shot).filter(Shot.project_id == project_id).delete(synchronize_session=False)
            db.query(Project).filter(Project.id == project_id).delete(synchronize_session=False)
            db.add(Project(id=project_id, title="trace"))
            db.add(Shot(
                id="s1", project_id=project_id, sequence=1, version=3, status="video_ready", confirmed=True,
                storyboard_path="s1.png",
                storyboard_reference_manifest='[{"name": "主角三视图", "kind": "character"}]',
                video_reference_manifest='[{"name": "上一镜尾帧", "kind": "tail_frame"}, {"name": "场景基准图", "kind": "scene"}]',
            ))
            db.add(Shot(id="s2", project_id=project_id, sequence=2, version=1, status="failed"))
            db.commit()
        finally:
            db.close()
        self.addCleanup(self._cleanup, project_id)

        with _patch_store(store):
            response = asyncio.run(graph_route.get_agent_trace(project_id, "auto"))

        summary = response["summary"]
        s1 = next(row for row in summary["shots"] if row["shot_id"] == "s1")
        # 实际发送参考图：DB manifest 并入追踪（图像 1 张、视频 2 张）。
        self.assertEqual(len(s1["references_sent"]["image_generation"]), 1)
        self.assertEqual(len(s1["references_sent"]["video_generation"]), 2)
        self.assertEqual(s1["db_shot_version"], 3)
        self.assertEqual(s1["db_status"], "video_ready")
        self.assertTrue(s1["confirmed"])
        # 兼容字段仍在：旧消费者读取 stages/decisions/shots/events。
        self.assertIn("stages", response)
        self.assertIn("decisions", response)
        self.assertIn("mermaid", response)
        s2 = next(row for row in summary["shots"] if row["shot_id"] == "s2")
        # 数据库里没有发送记录的镜头，参考图清单为空数组而不是缺失键。
        self.assertEqual(s2.get("references_sent"), {"image_generation": [], "video_generation": []})

    def _cleanup(self, project_id: str) -> None:
        db = SessionLocal()
        try:
            db.query(Shot).filter(Shot.project_id == project_id).delete(synchronize_session=False)
            db.query(Project).filter(Project.id == project_id).delete(synchronize_session=False)
            db.commit()
        finally:
            db.close()


class _patch_store:
    """把路由内 _store(...) 替换为返回固定检查点实例的上下文管理器。"""

    def __init__(self, store: CheckpointStore) -> None:
        self.store = store

    def __enter__(self):
        import unittest.mock as mock

        self._patcher = mock.patch.object(graph_route, "_store", return_value=self.store)
        return self._patcher.__enter__()

    def __exit__(self, *args):
        return self._patcher.__exit__(*args)


if __name__ == "__main__":
    unittest.main()
