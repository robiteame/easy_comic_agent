"""自动模式质量闭环验收测试：生成—审核—修改—重试。

覆盖验收标准的三个硬性要求：
1. 坏图（语义不符/伪影，结构检查合格但 VLM 低分）不能自动通过，必须按
   建议修正 prompt 后重试，重试耗尽转 needs_review；
2. 人物身份漂移（embedding 相似度低于阈值）必须触发重试；
3. 审核能力未配置时（VLM 缺失 / 维度无法检测）不得自动批准，整体裁决
   为 unsupported 并如实落库，绝不假装通过。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from PIL import Image  # noqa: E402

from agent import graph  # noqa: E402
from api.routes import shot as shot_route  # noqa: E402
from config import settings as app_settings  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import Character, Project, QualityReview, Shot  # noqa: E402
from services.quality_review_providers import MediaProbe, SimilarityReport, VLMCapability  # noqa: E402
from services.quality_review_service import (  # noqa: E402
    DIMENSIONS,
    QUALITY_FIX_TAG,
    STAGE_STORYBOARD,
    merge_quality_fix_notes,
    quality_review_service,
)
from test_environment import TEST_ROOT  # noqa: F401,E402


def _noise_image(path: Path, size: tuple[int, int] = (512, 512)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3)).save(path)
    return str(path)


# ---------------------------------------------------------------------------
# Provider stub
# ---------------------------------------------------------------------------


class StubVLM:
    """可控评分的 VLM 桩：scores 是 {维度key: 0~1}，缺省维度给默认分。"""

    def __init__(
        self,
        default_score: float = 0.95,
        scores: dict | None = None,
        issues: dict | None = None,
        supported: bool = True,
    ):
        self.default_score = default_score
        self.scores = scores or {}
        self.issues = issues or {}
        self.supported = supported
        self.calls: list[list[str]] = []

    def capability(self) -> VLMCapability:
        if not self.supported:
            return VLMCapability(supported=False, reason="VLM 未配置（stub）")
        return VLMCapability(supported=True, provider="vlm:stub")

    async def judge(self, system_prompt: str, user_prompt: str, image_paths: list[str]) -> dict:
        self.calls.append(list(image_paths))
        dimensions = {}
        for key in DIMENSIONS:
            score = self.scores.get(key, self.default_score)
            dimensions[key] = {
                "score": int(round(score * 10)),
                "issues": list(self.issues.get(key, [])),
                "evidence": ["stub 评审依据"],
            }
        return {"dimensions": dimensions, "suggestion": "stub 建议"}


class StubIdentityEmbedding:
    """可控相似度的身份 embedding 桩。"""

    def __init__(self, status: str = "scored", min_score: float | None = 0.95):
        self.status = status
        self.min_score = min_score

    def capability(self) -> VLMCapability:
        if self.status == "unsupported":
            return VLMCapability(supported=False, reason="身份 embedding 未配置（stub）")
        return VLMCapability(supported=True, provider="identity-embedding:stub")

    async def similarity(self, subject_path: str, reference_paths: list[dict]) -> SimilarityReport:
        if self.status != "scored":
            return SimilarityReport(status=self.status, error="stub 状态")
        similarities = [{"label": ref["label"], "score": self.min_score} for ref in reference_paths]
        return SimilarityReport(
            status="scored",
            similarities=similarities,
            min_score=self.min_score,
            provider="identity-embedding:stub",
        )


def _provider_patches(vlm: StubVLM | None, identity: StubIdentityEmbedding | None) -> list:
    patches = []
    if vlm is not None:
        patches.append(patch.object(quality_review_service, "_vlm", vlm))
    if identity is not None:
        patches.append(patch.object(quality_review_service, "_identity", identity))
    return patches


def passing_review_patch():
    """结构合格之外，质量审核全通过（供旧工作流测试复用）。"""
    return ExitStackWith(
        _provider_patches(StubVLM(default_score=0.95), StubIdentityEmbedding(status="scored", min_score=0.95))
    )


def video_gate_passing_patch():
    """跳过视频节点的门禁预检（只测视频生成重试本身，供旧测试复用）。"""
    return patch.object(quality_review_service, "storyboard_gate_status", lambda pid: {"ok": True, "failed": []})


class ExitStackWith(ExitStack):
    """把一组 patcher 包装成单个 contextmanager，便于复用。"""

    def __init__(self, patches: list):
        super().__init__()
        self._patches = patches

    def __enter__(self):
        for item in self._patches:
            self.enter_context(item)
        return self


# ---------------------------------------------------------------------------
# 测试基类
# ---------------------------------------------------------------------------


class QualityGateTestCase(unittest.TestCase):
    prefix = "quality_gate_tests"

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
            row = self.db.query(Project).filter(Project.id == project_id).first()
            if row:
                self.db.delete(row)
        self.db.commit()
        self.db.close()

    def _seed(
        self, name: str, sequences=(1,), *, with_characters: bool = True, dialogue: str = "", visual_notes: str = ""
    ) -> str:
        project_id = f"{self.prefix}_{name}"
        self.project_ids.append(project_id)
        self.db.add(Project(id=project_id, title=name, style="realistic", project_type="series"))
        ref_path = ""
        if with_characters:
            ref_path = _noise_image(TEST_ROOT / "output" / project_id / "character_ref.png", (256, 256))
            self.db.add(
                Character(
                    id=f"{project_id}_char_1",
                    project_id=project_id,
                    name="小明",
                    appearance="黑色短发，蓝色外套",
                    default_outfit="蓝色外套",
                    reference_images=json.dumps([ref_path]),
                )
            )
        for sequence in sequences:
            path = _noise_image(TEST_ROOT / "output" / project_id / f"shot_{sequence}.png")
            self.db.add(
                Shot(
                    id=f"{project_id}_shot_{sequence}",
                    project_id=project_id,
                    sequence=sequence,
                    scene_group_id=f"{project_id}_scene_1",
                    scene_description=f"scene {sequence}",
                    character_action="小明 走进房间",
                    dialogue=dialogue,
                    characters_in_scene=json.dumps(["小明"]) if with_characters else "[]",
                    status="storyboard_done",
                    storyboard_path=path,
                    image_path=path,
                    visual_notes=visual_notes,
                )
            )
        self.db.commit()
        return project_id

    def _shots(self, project_id: str) -> list[Shot]:
        self.db.expire_all()
        return self.db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()

    def _reviews(self, project_id: str, stage: str = STAGE_STORYBOARD) -> list[QualityReview]:
        self.db.expire_all()
        return (
            self.db.query(QualityReview)
            .filter(QualityReview.project_id == project_id, QualityReview.stage == stage)
            .order_by(QualityReview.created_at)
            .all()
        )


# ---------------------------------------------------------------------------
# 验收 1：能力未配置不得自动批准
# ---------------------------------------------------------------------------


class CapabilityUnconfiguredTests(QualityGateTestCase):
    def test_unconfigured_vlm_blocks_auto_approve(self) -> None:
        """VLM 未配置：整体 unsupported，镜头不得批准、转 needs_review、如实落库。"""
        project_id = self._seed("no_vlm")
        unsupported_vlm = StubVLM(supported=False)

        regenerated: list[list[str]] = []

        async def fake_regenerate(pid, shot_ids):  # noqa: ANN001
            regenerated.append(list(shot_ids))

        with (
            patch.object(shot_route, "_run_storyboard_generation", fake_regenerate),
            ExitStackWith(_provider_patches(unsupported_vlm, StubIdentityEmbedding(status="unsupported"))),
            self.assertLogs("agent.graph", level=logging.ERROR),
        ):
            result = asyncio.run(graph._auto_approve_storyboard({"project_id": project_id}))

        self.assertTrue(result["errors"], "审核能力未配置时必须中止")
        self.assertTrue(result.get("needs_human_review"))
        self.assertEqual(regenerated, [], "能力未配置时不应该重生成素材")
        for shot in self._shots(project_id):
            self.assertFalse(shot.confirmed, "审核能力未配置不得自动批准")
            self.assertEqual(shot.status, "needs_review")
        rows = self._reviews(project_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].verdict, "unsupported")
        self.assertFalse(rows[0].passed)
        unsupported_labels = json.loads(rows[0].unsupported_dimensions)
        self.assertIn(DIMENSIONS["character_identity"].label, unsupported_labels)

    def test_unconfigured_vlm_blocks_video_review(self) -> None:
        project_id = self._seed("no_vlm_video")
        with ExitStackWith(_provider_patches(StubVLM(supported=False), None)):
            result = asyncio.run(graph._review_shot_videos({"project_id": project_id}))
        self.assertTrue(result["errors"])
        self.assertTrue(result.get("needs_human_review"))
        rows = self._reviews(project_id, stage="video")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].verdict, "unsupported")


# ---------------------------------------------------------------------------
# 验收 2：坏图不能自动通过（重试 + needs_review）
# ---------------------------------------------------------------------------


class BadImageRetryTests(QualityGateTestCase):
    def test_semantically_bad_image_is_never_approved_and_retries(self) -> None:
        project_id = self._seed("bad_image", (1, 2))
        bad_vlm = StubVLM(
            default_score=0.3,
            issues={"scene_match": ["画面与场景描述明显不符"]},
        )
        regenerated: list[list[str]] = []

        async def fake_regenerate(pid, shot_ids):  # noqa: ANN001
            regenerated.append(list(shot_ids))

        max_retries = int(app_settings.QUALITY_STORYBOARD_MAX_RETRIES)
        with (
            patch.object(shot_route, "_run_storyboard_generation", fake_regenerate),
            ExitStackWith(_provider_patches(bad_vlm, StubIdentityEmbedding(status="unsupported"))),
            self.assertLogs("agent.graph", level=logging.ERROR),
        ):
            result = asyncio.run(graph._auto_approve_storyboard({"project_id": project_id}))

        self.assertTrue(result["errors"], "坏图必须被质量门禁拦下")
        self.assertTrue(result.get("needs_human_review"))
        # 重试次数受配置约束：1 次初评 + max_retries 次修正重生成后再评。
        self.assertEqual(len(regenerated), max_retries)
        for call_ids in regenerated:
            self.assertEqual(call_ids, [f"{project_id}_shot_1", f"{project_id}_shot_2"])
        rows = self._reviews(project_id)
        self.assertEqual(len(rows), 2 * (max_retries + 1))
        self.assertTrue(all(row.verdict == "failed" for row in rows))
        for shot in self._shots(project_id):
            self.assertFalse(shot.confirmed, "质量不达标的镜头绝不能自动批准")
            self.assertEqual(shot.status, "needs_review")
        # 修正指令已写入 visual_notes（替换式，不叠加）。
        for shot in self._shots(project_id):
            self.assertIn(QUALITY_FIX_TAG, shot.visual_notes)
            self.assertIn("画面必须严格呈现场景", shot.visual_notes)
            self.assertGreaterEqual(shot.version or 1, 1 + max_retries)

    def test_structural_pass_is_not_quality_pass(self) -> None:
        """结构检查合格但没有任何审核记录时，视频生成被门禁拦下。"""
        project_id = self._seed("gate_block", (1,))
        calls: list[str] = []

        async def fake_video(shot_id, force=False, **kwargs):  # noqa: ANN001
            calls.append(shot_id)

        with patch.object(shot_route, "_run_single_shot_video", fake_video):
            result = asyncio.run(graph._generate_shot_videos({"project_id": project_id}))

        self.assertTrue(result["errors"])
        self.assertEqual(calls, [], "故事板质量门禁未通过时不得生成视频")

    def test_compose_blocked_without_video_quality_gate(self) -> None:
        project_id = self._seed("compose_block", (1,))
        rendered: list[str] = []

        async def fake_render(pid, output_format, resolution):  # noqa: ANN001
            rendered.append(pid)

        from api.routes import render as render_route

        with patch.object(render_route, "_render_task", fake_render):
            result = asyncio.run(graph._compose({"project_id": project_id}))

        self.assertTrue(result["errors"], "视频质量门禁未通过时不得导出成片")
        self.assertEqual(rendered, [])


# ---------------------------------------------------------------------------
# 验收 3：人物身份漂移必须重试
# ---------------------------------------------------------------------------


class IdentityDriftTests(QualityGateTestCase):
    def test_identity_embedding_drift_triggers_retry(self) -> None:
        """VLM 全高分但 embedding 相似度低于阈值：身份维度失败 → 必须重试。"""
        project_id = self._seed("identity_drift")
        drifting_vlm = StubVLM(default_score=0.95)
        drifted_identity = StubIdentityEmbedding(status="scored", min_score=0.42)
        regenerated: list[list[str]] = []

        async def fake_regenerate(pid, shot_ids):  # noqa: ANN001
            regenerated.append(list(shot_ids))

        with (
            patch.object(shot_route, "_run_storyboard_generation", fake_regenerate),
            ExitStackWith(_provider_patches(drifting_vlm, drifted_identity)),
        ):
            result = asyncio.run(graph._auto_approve_storyboard({"project_id": project_id}))

        self.assertTrue(result["errors"], "身份漂移必须被质量门禁拦下")
        self.assertGreaterEqual(len(regenerated), 1, "身份漂移必须触发重试")
        shot = self._shots(project_id)[0]
        self.assertFalse(shot.confirmed)
        self.assertEqual(shot.status, "needs_review")
        # 修正指令包含身份锁定要求。
        self.assertIn("脸型五官必须与角色参考图完全一致", shot.visual_notes)
        # 审核记录包含 embedding 证据与低于阈值的问题。
        rows = self._reviews(project_id)
        identity_dim = next(
            dimension
            for row in rows
            for dimension in json.loads(row.dimensions)
            if dimension["key"] == "character_identity"
        )
        embedding = identity_dim["evidence"].get("identity_embedding", {})
        self.assertEqual(embedding.get("status"), "scored")
        self.assertEqual(
            embedding.get("similarities"),
            [{"label": "小明", "score": drifted_identity.min_score}],
        )
        self.assertTrue(any("低于阈值" in issue for issue in identity_dim["issues"]))

    def test_identity_score_capped_by_similarity(self) -> None:
        """身份维度得分 = min(VLM 分, 相似度折算分)，漂移时不得高于折算分。"""
        project_id = self._seed("identity_cap")
        with ExitStackWith(
            _provider_patches(
                StubVLM(default_score=0.95),
                StubIdentityEmbedding(status="scored", min_score=0.5),
            )
        ):
            review = asyncio.run(quality_review_service.review_storyboard_shot(f"{project_id}_shot_1"))
        identity = next(d for d in review.dimensions if d.key == "character_identity")
        threshold = float(app_settings.QUALITY_IDENTITY_SIMILARITY_THRESHOLD)
        self.assertAlmostEqual(identity.score, min(0.95, 0.5 / threshold), places=3)


# ---------------------------------------------------------------------------
# 通过路径与降级策略
# ---------------------------------------------------------------------------


class PassAndDegradeTests(QualityGateTestCase):
    def test_passing_review_approves_storyboard(self) -> None:
        project_id = self._seed("all_pass")
        with ExitStackWith(
            _provider_patches(
                StubVLM(default_score=0.95),
                StubIdentityEmbedding(status="scored", min_score=0.95),
            )
        ):
            result = asyncio.run(graph._auto_approve_storyboard({"project_id": project_id}))
        self.assertNotIn("errors", result)
        for shot in self._shots(project_id):
            self.assertTrue(shot.confirmed)
            self.assertEqual(shot.status, "storyboard_approved")
        rows = self._reviews(project_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].verdict, "passed")
        self.assertFalse(rows[0].degraded)

    def test_missing_reference_blocks_under_strict_but_lenient_degrades_honestly(self) -> None:
        """有角色但缺参考图：strict=unsupported 不得放行；lenient=可过但必须标 degraded。"""
        project_id = self._seed("no_refs", with_characters=True)
        # 有 characters_in_scene 但删掉参考图 → 身份/服装维度无法检测。
        self.db.query(Character).filter(Character.project_id == project_id).delete()
        self.db.commit()

        vlm = StubVLM(default_score=0.95)
        with ExitStackWith(_provider_patches(vlm, StubIdentityEmbedding(status="scored", min_score=0.95))):
            result = asyncio.run(graph._auto_approve_storyboard({"project_id": project_id}))
        self.assertTrue(result["errors"], "strict 策略下未检测维度必须拦截")
        self.assertTrue(result.get("needs_human_review"))
        rows = self._reviews(project_id)
        self.assertEqual(rows[0].verdict, "unsupported")

        # 换 lenient：允许在已检测维度上通过，但必须如实标记 degraded。
        project_id2 = self._seed("no_refs_lenient", with_characters=True)
        self.db.query(Character).filter(Character.project_id == project_id2).delete()
        self.db.commit()
        with (
            patch.object(app_settings, "QUALITY_DEGRADATION_POLICY", "lenient"),
            ExitStackWith(
                _provider_patches(StubVLM(default_score=0.95), StubIdentityEmbedding(status="scored", min_score=0.95))
            ),
        ):
            result = asyncio.run(graph._auto_approve_storyboard({"project_id": project_id2}))
        self.assertNotIn("errors", result)
        rows = self._reviews(project_id2)
        self.assertEqual(rows[0].verdict, "passed")
        self.assertTrue(rows[0].degraded, "lenient 放行必须标记 degraded 供界面提示")
        self.assertTrue(json.loads(rows[0].unsupported_dimensions))

    def test_dimension_floor_blocks_even_if_average_passes(self) -> None:
        """均分过阈值但单维度击穿下限（如伪影）时不得通过。"""
        project_id = self._seed("artifact_floor")
        vlm = StubVLM(default_score=0.95, scores={"artifacts": 0.65}, issues={"artifacts": ["手指数量错误"]})
        with ExitStackWith(_provider_patches(vlm, StubIdentityEmbedding(status="scored", min_score=0.95))):
            review = asyncio.run(quality_review_service.review_storyboard_shot(f"{project_id}_shot_1"))
        self.assertEqual(review.verdict, "failed")
        self.assertFalse(review.passed)
        self.assertTrue(any("伪影" in directive for directive in review.fix["directives"]))

    def test_vlm_error_fails_closed_without_regeneration(self) -> None:
        """VLM 调用失败（error）必须 fail-closed，且不浪费素材重生成。"""
        project_id = self._seed("vlm_error")

        class ErrorVLM(StubVLM):
            async def judge(self, system_prompt, user_prompt, image_paths):  # noqa: ANN001
                raise RuntimeError("VLM 网关超时")

        regenerated: list[list[str]] = []

        async def fake_regenerate(pid, shot_ids):  # noqa: ANN001
            regenerated.append(list(shot_ids))

        with (
            patch.object(shot_route, "_run_storyboard_generation", fake_regenerate),
            ExitStackWith(_provider_patches(ErrorVLM(), StubIdentityEmbedding(status="scored", min_score=0.95))),
        ):
            result = asyncio.run(graph._auto_approve_storyboard({"project_id": project_id}))
        self.assertTrue(result["errors"])
        self.assertEqual(regenerated, [], "Provider 故障不应触发素材重生成")
        rows = self._reviews(project_id)
        self.assertEqual(rows[0].verdict, "error")


# ---------------------------------------------------------------------------
# 视频审核节点与维度
# ---------------------------------------------------------------------------


class _FakeVideoReview:
    def __init__(self, project_id: str, passed: bool, verdict: str = "failed"):
        self.project_id = project_id
        self.passed = passed
        self.verdict = verdict
        self.calls: list[str] = []

    async def __call__(self, shot_id: str, previous_frame_path: str = ""):
        from services.quality_review_service import ShotReview

        self.calls.append(shot_id)
        review = ShotReview(
            shot_id=shot_id,
            project_id=self.project_id,
            stage="video",
            verdict=self.verdict,
            passed=self.passed,
            overall_score=0.95 if self.passed else 0.3,
        )
        review.fix = {"directives": ["动作连贯流畅，避免瞬移、抖动与画面闪烁"], "summary": "..."}
        review.issues = [] if self.passed else ["[运动连贯性] 动作瞬移"]
        # 与真实路径一致：审核结果落库（shot_summary / 门禁查询都依赖记录存在）。
        await quality_review_service._persist_and_notify(review)  # noqa: SLF001
        return review


class VideoReviewNodeTests(QualityGateTestCase):
    def _seed_with_video(self, name: str) -> str:
        project_id = self._seed(name, (1,))
        path = _noise_image(TEST_ROOT / "output" / project_id / "shot_1.mp4.png", (256, 256))
        self.db.query(Shot).filter(Shot.id == f"{project_id}_shot_1").update({"video_path": path})
        self.db.commit()
        return project_id

    def test_video_quality_fail_retries_then_needs_human(self) -> None:
        project_id = self._seed_with_video("video_fail")
        fake = _FakeVideoReview(project_id, passed=False)
        video_calls: list[tuple[str, bool]] = []

        async def fake_video(shot_id, force=False, **kwargs):  # noqa: ANN001
            video_calls.append((shot_id, force))

        max_retries = int(app_settings.QUALITY_VIDEO_MAX_RETRIES)
        with (
            patch.object(quality_review_service, "review_video_shot", fake),
            patch.object(shot_route, "_run_single_shot_video", fake_video),
            ExitStackWith(_provider_patches(StubVLM(), None)),
        ):
            result = asyncio.run(graph._review_shot_videos({"project_id": project_id}))

        self.assertTrue(result["errors"], "视频质量不达标必须阻断成片")
        self.assertTrue(result.get("needs_human_review"))
        self.assertEqual(len(fake.calls), max_retries + 1)
        self.assertEqual(video_calls, [(f"{project_id}_shot_1", True)] * max_retries)
        shot = self._shots(project_id)[0]
        self.assertEqual(shot.status, "needs_review")
        self.assertIn("避免瞬移", shot.visual_notes)

    def test_video_quality_pass_allows_next_stage(self) -> None:
        project_id = self._seed_with_video("video_pass")
        fake = _FakeVideoReview(project_id, passed=True, verdict="passed")
        with (
            patch.object(quality_review_service, "review_video_shot", fake),
            ExitStackWith(_provider_patches(StubVLM(), None)),
        ):
            result = asyncio.run(graph._review_shot_videos({"project_id": project_id}))
        self.assertNotIn("errors", result)
        self.assertEqual(result.get("current_step"), "review_shot_videos")
        # 门禁状态应放行 compose。
        gate = quality_review_service.video_gate_status(project_id)
        self.assertTrue(gate["ok"])


class VideoDimensionTests(unittest.TestCase):
    """音画同步 / 音频清晰度维度的降级与打分（不依赖真实 ffmpeg）。"""

    def setUp(self) -> None:
        self.service = quality_review_service

    def test_no_audio_track_marks_skipped_not_fake_pass(self):
        probe = MediaProbe(status="skipped", has_audio=False, issues=["无音频轨（该镜头没有配音）"])
        dimension = self.service._audio_sync_dimension(probe)
        self.assertEqual(dimension.status, "skipped")
        self.assertIsNone(dimension.score, "跳过的维度不得有分数")

    def test_ffprobe_missing_marks_unsupported(self):
        probe = MediaProbe(status="unsupported", error="ffprobe 不可用（unsupported）")
        dimension = self.service._audio_sync_dimension(probe)
        self.assertEqual(dimension.status, "unsupported")
        self.assertIsNone(dimension.score)

    def test_sync_drift_scores_low(self):
        probe = MediaProbe(
            status="scored",
            has_audio=True,
            video_duration=5.0,
            audio_duration=3.8,
            issues=["音画时长差 1.20s（视频 5.00s / 音频 3.80s）"],
        )
        dimension = self.service._audio_sync_dimension(probe)
        self.assertEqual(dimension.status, "scored")
        self.assertLessEqual(dimension.score, 0.3)

    def test_clarity_issues_reduce_score(self):
        async def fake_clarity(path):  # noqa: ANN001
            return MediaProbe(
                status="scored",
                mean_volume_db=-50.0,
                silence_ratio=0.8,
                issues=["平均音量过低（-50.0 dB）", "大部分时间是无声（静音占比 80%）"],
            )

        probe = MediaProbe(status="scored", has_audio=True, video_duration=5.0, audio_duration=5.0)
        with patch("services.quality_review_service.analyze_audio_clarity", fake_clarity):
            dimension = asyncio.run(self.service._audio_clarity_dimension("x.wav", probe))
        self.assertEqual(dimension.status, "scored")
        self.assertLessEqual(dimension.score, 0.3)
        self.assertEqual(len(dimension.issues), 2)


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


class FixNotesTests(unittest.TestCase):
    def test_fix_block_replaces_instead_of_accumulating(self):
        notes = merge_quality_fix_notes("用户手写备注", ["指令A"])
        self.assertEqual(notes, f"用户手写备注\n{QUALITY_FIX_TAG}指令A")
        again = merge_quality_fix_notes(notes, ["指令B"])
        self.assertEqual(again, f"用户手写备注\n{QUALITY_FIX_TAG}指令B")

    def test_no_directives_keeps_base(self):
        self.assertEqual(merge_quality_fix_notes(f"备注\n{QUALITY_FIX_TAG}旧", []), "备注")


if __name__ == "__main__":
    unittest.main()
