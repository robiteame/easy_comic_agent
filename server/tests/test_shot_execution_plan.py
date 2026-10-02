"""镜头统一执行计划（ShotExecutionPlan）的行为验收。

覆盖四类场景：
- 固定时长 Provider：生成满固定秒数，narrative 全量使用；
- 短镜头裁剪：固定 5 秒 Provider 生成 5 秒后，成片按 trim 区间裁成更短故事时长；
- 对白超时：TTS 实测时间轴把故事时长延展到剪辑区间（超出生成容量时封顶并告警）；
- 旧数据兼容：没有执行计划的镜头从 duration/dialogue/continuity_profile 推导，
  后期合成时间线与历史行为一致；持久化计划 JSON 可无损往返。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

# 必须先于 services/config 导入，保证 settings 绑定测试沙箱目录，
# 不污染同批运行的其它模块（参考图 allowed_roots 校验依赖 OUTPUT_DIR）。
from test_environment import TEST_ROOT  # noqa: F401,E402

from services.post_production_plan import build_post_production_plan  # noqa: E402
from services.story_timing import (  # noqa: E402
    ProviderDurationCapability,
    ShotExecutionPlan,
    StoryTimingPlan,
    load_shot_execution_plan,
    merge_shots,
    plan_required_capabilities,
    resolve_shot_execution_plan,
    split_shot,
)
from services.video_service import VideoService  # noqa: E402


def _fixed_5s_provider() -> ProviderDurationCapability:
    return ProviderDurationCapability(
        protocol="fixed-test",
        fixed_duration=5,
        min_duration=5,
        max_duration=5,
        duration_step=5,
    )


def _flex_provider() -> ProviderDurationCapability:
    return ProviderDurationCapability(
        protocol="flex-test",
        fixed_duration=None,
        min_duration=2,
        max_duration=6,
        duration_step=1,
    )


def _shot(**overrides) -> dict:
    base = {
        "shot_id": "plan_0001",
        "duration": 5.0,
        "character_action": "主角转身看向门口",
        "dialogue": [{"speaker": "主角", "line": "来了。"}],
        "camera_movement": "静止",
        "camera_angle": "正面",
        "shot_type": "medium",
        "emotion": "neutral",
        "storyboard_path": "/refs/storyboard.png",
        "continuity_profile": {"control_source": "previous_final_frame"},
    }
    base.update(overrides)
    return base


class FixedDurationPlanTests(unittest.TestCase):
    def test_fixed_provider_full_narrative_generates_fixed_clip(self) -> None:
        plan = ShotExecutionPlan.derive(_shot(duration=5.0), provider=_fixed_5s_provider())

        self.assertEqual(plan.narrative_duration_ms, 5000)
        self.assertEqual(plan.provider_generation_duration_s, 5.0)
        self.assertEqual((plan.trim_start_ms, plan.trim_end_ms), (0, 5000))
        self.assertEqual(plan.effective_duration_ms, 5000)
        self.assertEqual(plan.warnings, ())
        self.assertTrue(plan.recipe_hash)
        self.assertEqual(plan.audio_mode, "tts")
        self.assertEqual(plan.continuity_mode, "continuous_action")
        self.assertEqual(plan.video_mode, "first_frame_reference")

    def test_fixed_provider_short_narrative_trims_generated_clip(self) -> None:
        """固定 5 秒 Provider 允许生成 5 秒后在成片中裁剪为更短的故事时长。"""

        plan = ShotExecutionPlan.derive(_shot(duration=4.5), provider=_fixed_5s_provider())

        self.assertEqual(plan.narrative_duration_ms, 4500)
        self.assertEqual(plan.provider_generation_duration_s, 5.0)
        self.assertEqual((plan.trim_start_ms, plan.trim_end_ms), (0, 4500))
        self.assertEqual(plan.effective_duration_ms, 4500)
        self.assertEqual(plan.warnings, ())

    def test_overlong_narrative_keeps_story_time_and_warns(self) -> None:
        """超出 Provider 能力的故事时长不被静默截短：narrative 保持原值并告警。"""

        plan = ShotExecutionPlan.derive(_shot(duration=12.0), provider=_fixed_5s_provider())

        self.assertEqual(plan.narrative_duration_ms, 12000)
        self.assertEqual(plan.provider_generation_duration_s, 5.0)
        self.assertEqual(plan.trim_end_ms, 5000)
        self.assertIn("narrative_exceeds_provider_clip", plan.warnings)


class DialogueOverrunTests(unittest.TestCase):
    def test_measured_dialogue_extends_cut_within_generated_clip(self) -> None:
        """TTS 实测对白超出故事时间但在生成片段内：剪辑区间延展到实测结束。"""

        plan = ShotExecutionPlan.derive(
            _shot(duration=4.5),
            provider=_fixed_5s_provider(),
            dialogue_timing=[{"speaker": "主角", "text": "来了。", "start_ms": 300, "end_ms": 4800}],
            dialogue_timing_source="tts_measured",
        )

        self.assertEqual(plan.narrative_duration_ms, 4800)
        self.assertEqual(plan.trim_end_ms, 4800)
        self.assertEqual(plan.effective_duration_ms, 4800)
        self.assertIn("narrative_extended_for_dialogue", plan.warnings)
        self.assertTrue(plan.has_measured_dialogue)
        self.assertEqual(plan.dialogue_end_ms, 4800)

    def test_measured_dialogue_beyond_clip_is_capped_with_warning(self) -> None:
        plan = ShotExecutionPlan.derive(
            _shot(duration=4.5),
            provider=_fixed_5s_provider(),
            dialogue_timing=[{"speaker": "主角", "text": "来了。", "start_ms": 300, "end_ms": 5200}],
            dialogue_timing_source="tts_measured",
        )

        self.assertEqual(plan.narrative_duration_ms, 5200)
        self.assertEqual(plan.provider_generation_duration_s, 5.0)
        self.assertEqual(plan.trim_end_ms, 5000)
        self.assertIn("dialogue_exceeds_provider_clip", plan.warnings)

    def test_flexible_provider_snaps_extended_cut_to_duration_step(self) -> None:
        """非固定档 Provider 的延展结果必须落在步长网格上，避免下次生成被拒。"""

        plan = ShotExecutionPlan.derive(
            _shot(duration=4.0),
            provider=_flex_provider(),
            dialogue_timing=[{"speaker": "主角", "text": "来了。", "start_ms": 0, "end_ms": 4300}],
            dialogue_timing_source="tts_measured",
        )

        self.assertEqual(plan.provider_generation_duration_s, 5.0)
        self.assertEqual(plan.effective_duration_ms, 5000)
        _flex_provider().validate(plan.effective_duration_ms / 1000)

    def test_native_mode_does_not_extend_cut_for_dialogue(self) -> None:
        """native 路径对白由视频模型自带，剪辑区间不按 prompt 时间延展。"""

        plan = ShotExecutionPlan.derive(
            _shot(duration=4.5),
            provider=_fixed_5s_provider(),
            audio_mode="native",
            dialogue_timing=[{"speaker": "主角", "text": "来了。", "start_ms": 0, "end_ms": 4800}],
            dialogue_timing_source="native_prompt",
        )

        self.assertEqual(plan.audio_mode, "native")
        self.assertEqual(plan.narrative_duration_ms, 4500)
        self.assertEqual(plan.trim_end_ms, 4500)
        self.assertNotIn("narrative_extended_for_dialogue", plan.warnings)


class LegacyDataCompatibilityTests(unittest.TestCase):
    def test_legacy_shot_without_plan_derives_from_old_fields(self) -> None:
        shot = {
            "shot_id": "legacy_0001",
            "duration": 3.0,
            "dialogue": "旧版纯文本对白",
            "continuity_profile": {"audio_source": "native"},
        }

        plan = ShotExecutionPlan.derive(shot)

        self.assertEqual(plan.narrative_duration_ms, 3000)
        self.assertEqual(plan.provider_generation_duration_s, 3.0)
        self.assertEqual((plan.trim_start_ms, plan.trim_end_ms), (0, 3000))
        self.assertEqual(plan.audio_mode, "native")
        self.assertEqual(plan.dialogue_timing_source, "none")
        self.assertEqual(plan.video_mode, "text_only")

    def test_stored_dialogue_timings_are_used_without_remeasurement(self) -> None:
        shot = _shot(
            duration=4.5,
            dialogue=[{"speaker": "主角", "line": "来了。", "start_ms": 200, "end_ms": 1600}],
        )

        plan = ShotExecutionPlan.derive(shot, provider=_fixed_5s_provider())

        self.assertEqual(plan.dialogue_timing_source, "stored")
        self.assertEqual(plan.dialogue_end_ms, 1600)
        self.assertEqual(plan.narrative_duration_ms, 4500)

    def test_load_returns_none_and_resolve_derives_for_old_profile(self) -> None:
        shot = {"shot_id": "legacy_0002", "duration": 2.0, "dialogue": "", "continuity_profile": {"lut": "locked"}}

        self.assertIsNone(load_shot_execution_plan(shot))
        plan = resolve_shot_execution_plan(shot)
        self.assertEqual(plan.narrative_duration_ms, 2000)

    def test_persisted_plan_json_roundtrips_and_is_preferred(self) -> None:
        shot = _shot(duration=4.5)
        derived = ShotExecutionPlan.derive(shot, provider=_fixed_5s_provider())
        persisted_shot = {**shot, "continuity_profile": {"execution_plan": derived.to_dict()}}

        loaded = load_shot_execution_plan(persisted_shot)
        resolved = resolve_shot_execution_plan({**persisted_shot, "duration": 5.0})

        self.assertEqual(loaded, derived)
        self.assertEqual(resolved.recipe_hash, derived.recipe_hash)

    def test_post_production_keeps_legacy_durations_without_plan(self) -> None:
        shots = [
            {"shot_id": "s1", "sequence": 1, "duration": 3.0, "scene_group_id": "scene-a", "dialogue": "", "transition": "cut"},
            {"shot_id": "s2", "sequence": 2, "duration": 2.0, "scene_group_id": "scene-a", "dialogue": ""},
        ]

        plan = build_post_production_plan(shots, project_id="p1")

        self.assertEqual([entry.duration_ms for entry in plan.shots], [3000, 2000])
        self.assertEqual([(entry.source_in_ms, entry.source_out_ms) for entry in plan.shots], [(0, 3000), (0, 2000)])
        self.assertEqual(plan.total_duration_ms, 5000)


class PlanConsumptionTests(unittest.TestCase):
    def test_video_prompt_uses_plan_durations_not_stale_estimates(self) -> None:
        """视频 Prompt 的有效时长与对白预算来自执行计划，不依赖过期估算值。"""

        execution_plan = ShotExecutionPlan.derive(
            _shot(duration=4.5),
            provider=_fixed_5s_provider(),
            dialogue_timing=[{"speaker": "主角", "text": "来了。", "start_ms": 250, "end_ms": 1800}],
            dialogue_timing_source="tts_measured",
        )
        shot = _shot(
            duration=4.5,
            estimated_speech_ms=9999,
            execution_plan=execution_plan.to_dict(),
        )

        with patch("services.video_service.get_endpoint", return_value=SimpleNamespace(protocol="fixed-test")):
            prompt = VideoService()._build_prompt(shot, [], {})

        self.assertIn("actual generated clip duration: 5 seconds", prompt)
        self.assertIn("estimated speech duration: 1800 ms", prompt)
        self.assertNotIn("9999", prompt)
        self.assertIn("requested source duration 4.500s", prompt)
        self.assertIn("[250-1800ms]", prompt)

    def test_post_production_uses_plan_trim_window_and_dialogue_timing(self) -> None:
        execution_plan = ShotExecutionPlan.derive(
            _shot(duration=4.5),
            provider=_fixed_5s_provider(),
            dialogue_timing=[{"speaker": "主角", "text": "来了。", "start_ms": 200, "end_ms": 1400}],
            dialogue_timing_source="tts_measured",
        )
        shots = [
            {
                "shot_id": "s1",
                "sequence": 1,
                "duration": 4.5,
                "scene_group_id": "scene-a",
                "transition": "cut",
                "dialogue": [{"speaker": "主角", "line": "来了。"}],
                "continuity_profile": {"execution_plan": execution_plan.to_dict()},
            },
            {"shot_id": "s2", "sequence": 2, "duration": 2.0, "scene_group_id": "scene-a", "dialogue": ""},
        ]

        plan = build_post_production_plan(shots, project_id="p1")

        first, second = plan.shots
        self.assertEqual(first.duration_ms, 4500)
        self.assertEqual((first.source_in_ms, first.source_out_ms), (0, 4500))
        self.assertEqual((first.timeline_start_ms, first.timeline_end_ms), (0, 4500))
        # 后一镜紧贴执行计划给出的时长排布，而不是各自推导。
        self.assertEqual(second.timeline_start_ms, 4500)
        self.assertEqual(plan.total_duration_ms, 6500)
        # 对白时间轴直接取计划里的实测值，不按字符比例二次推导。
        self.assertEqual((first.dialogue[0].start_ms, first.dialogue[0].end_ms), (200, 1400))
        self.assertEqual(first.audio[0].clip_start_ms, 200)

    def test_render_validation_accepts_planned_fixed_clip_trim(self) -> None:
        """固定 5 秒片段裁成 4.2 秒在执行计划里是有意行为，不算违规裁剪。"""

        execution_plan = ShotExecutionPlan.derive(_shot(duration=4.2), provider=_fixed_5s_provider())
        shot = {
            "shot_id": "trim_0001",
            "duration": 4.2,
            "dialogue": [],
            "video_path": "/video/trim_0001.mp4",
            "execution_plan": execution_plan.to_dict(),
        }
        plan = StoryTimingPlan(target_duration_s=4.2, provider=_fixed_5s_provider())

        with_plan = plan.validate_timeline([shot], media_durations_ms={"/video/trim_0001.mp4": 5000}, require_target=False)
        without_plan = plan.validate_timeline(
            [{key: value for key, value in shot.items() if key != "execution_plan"}],
            media_durations_ms={"/video/trim_0001.mp4": 5000},
            require_target=False,
        )

        self.assertNotIn("video_will_be_trimmed", {issue.code for issue in with_plan})
        self.assertIn("video_will_be_trimmed", {issue.code for issue in without_plan})


class PlanFieldContractTests(unittest.TestCase):
    """执行计划契约：候选数、能力清单、恢复预算与实测对白时间轴齐备可回放。"""

    def test_plan_carries_candidates_capabilities_and_recovery_budget(self) -> None:
        plan = ShotExecutionPlan.derive(
            _shot(characters_in_scene=["主角"], scene_description="雨夜天台"),
            provider=_fixed_5s_provider(),
            candidate_count=3,
            recovery_budget=2,
        )

        self.assertEqual(plan.candidate_count, 3)
        self.assertEqual(plan.recovery_budget, 2)
        # 能力清单按镜头语义推导：首帧 + 角色身份 + 场景基准 → 多参考。
        self.assertEqual(
            plan.required_capabilities,
            ("first_frame", "character_identity", "scene_reference", "multiple_reference_images"),
        )

    def test_plan_serializes_actual_dialogue_timing_and_roundtrips(self) -> None:
        plan = ShotExecutionPlan.derive(
            _shot(duration=4.5),
            provider=_fixed_5s_provider(),
            dialogue_timing=[{"speaker": "主角", "text": "来了。", "start_ms": 200, "end_ms": 1400}],
            dialogue_timing_source="tts_measured",
            candidate_count=2,
            recovery_budget=2,
        )

        payload = plan.to_dict()
        self.assertEqual(
            payload["actual_dialogue_timing"],
            [{"speaker": "主角", "text": "来了。", "start_ms": 200, "end_ms": 1400}],
        )
        self.assertEqual(payload["dialogue_timing"], payload["actual_dialogue_timing"])
        self.assertEqual(payload["candidate_count"], 2)
        self.assertEqual(payload["recovery_budget"], 2)

        reloaded = ShotExecutionPlan.from_mapping(payload)
        self.assertEqual(reloaded, plan)
        # 旧消费者只写 actual_dialogue_timing 时同样可解析。
        legacy_only = {key: value for key, value in payload.items() if key != "dialogue_timing"}
        self.assertEqual(ShotExecutionPlan.from_mapping(legacy_only), plan)

    def test_planner_capabilities_follow_shot_semantics(self) -> None:
        minimal = plan_required_capabilities({"shot_id": "s1"})
        self.assertEqual(minimal, ["first_frame"])

        continuous = plan_required_capabilities(
            {
                "shot_id": "s2",
                "characters_in_scene": ["主角"],
                "scene_description": "雨夜天台",
                "continuity_mode": "continuous_action",
            }
        )
        self.assertIn("character_identity", continuous)
        self.assertIn("scene_reference", continuous)
        self.assertIn("multiple_reference_images", continuous)

    def test_dialogue_overrun_plan_still_carries_execution_metadata(self) -> None:
        """对白超出固定档容量的计划：警告齐备且候选/恢复预算不丢失。"""

        plan = ShotExecutionPlan.derive(
            _shot(duration=4.5),
            provider=_fixed_5s_provider(),
            dialogue_timing=[{"speaker": "主角", "text": "来了。", "start_ms": 0, "end_ms": 5200}],
            dialogue_timing_source="tts_measured",
            candidate_count=3,
            recovery_budget=3,
        )

        self.assertIn("dialogue_exceeds_provider_clip", plan.warnings)
        self.assertEqual(plan.candidate_count, 3)
        self.assertEqual(plan.recovery_budget, 3)
        self.assertEqual(plan.trim_end_ms, 5000)


class SplitMergeAuditTests(unittest.TestCase):
    """拆镜/合镜审计：原因、前后结构与 timing_plan 调整记录同源可追溯。"""

    def test_complex_action_split_records_reason_and_before_after(self) -> None:
        shot = _shot(
            duration=9.0,
            version=3,
            character_action="主角追逐敌人，翻滚闪避后跳跃翻越障碍",
            dialogue=[],
        )

        parts = split_shot(shot, 3, reason="complex_motion")

        self.assertEqual(len(parts), 3)
        phases = [part["action_beats"][0]["phase"] for part in parts]
        self.assertEqual(phases, ["preparation", "action", "reaction"])
        for part in parts:
            change = part["timing"]["structure_change"]
            self.assertEqual(change["kind"], "split")
            self.assertEqual(change["reason"], "complex_motion")
            self.assertEqual(change["before"], {"shot_id": "plan_0001", "duration": 9.0, "version": 3})
            self.assertEqual(change["after"]["shot_id"], part["shot_id"])
            self.assertEqual(change["after"]["part_count"], 3)
            self.assertEqual(part["continuity_mode"], "continuous_action")

    def test_merge_records_reason_and_source_versions(self) -> None:
        left = _shot(
            shot_id="m_0001",
            duration=1.5,
            version=2,
            character_action="她抬头",
            dialogue=[],
            scene_asset_id="scene-a",
        )
        right = _shot(
            shot_id="m_0002",
            duration=1.5,
            version=1,
            character_action="她抬头",
            dialogue=[],
            scene_asset_id="scene-a",
        )

        merged = merge_shots(left, right, reason="short_adjacent_beats")

        change = merged["timing"]["structure_change"]
        self.assertEqual(change["kind"], "merge")
        self.assertEqual(change["reason"], "short_adjacent_beats")
        self.assertEqual([item["shot_id"] for item in change["before"]], ["m_0001", "m_0002"])
        self.assertEqual(change["after"]["duration"], 3.0)
        self.assertEqual(merged["timing"]["merged_from"], ["m_0001", "m_0002"])

    def test_rebalance_adjustments_carry_before_after_snapshots(self) -> None:
        shots = [
            _shot(
                shot_id="audit_0001",
                duration=9.0,
                version=1,
                character_action="主角追逐敌人，翻滚闪避后跳跃翻越障碍",
                dialogue=[],
            )
        ]

        plan = StoryTimingPlan.from_shots(12.0, shots, _fixed_5s_provider())
        split_adjustments = [item for item in plan.adjustments if item.code == "shot_split"]

        self.assertTrue(split_adjustments)
        first = split_adjustments[0]
        self.assertIn("complex_motion", first.reason)
        self.assertIn("action_beat_count", first.reason)
        self.assertEqual(first.before[0]["shot_id"], "audit_0001")
        self.assertEqual({item["shot_id"] for item in first.after}, set(first.shot_ids))
        serialized = first.to_dict()
        self.assertIn("reason", serialized)
        self.assertEqual(len(serialized["after"]), len(first.shot_ids))


class StoryboardPlanTests(unittest.TestCase):
    """分镜阶段：每个镜头（含自动拆镜产物）都携带可执行计划。"""

    def test_attach_execution_plans_uses_quality_profile_budget(self) -> None:
        from agent.nodes.storyboard_gen import _attach_execution_plans

        shots = [
            _shot(shot_id="sb_0001", duration=4.5),
            _shot(shot_id="sb_0002", duration=5.0, characters_in_scene=["主角", "同伴"]),
        ]

        result = _attach_execution_plans(shots, _fixed_5s_provider(), "finishing")

        for shot in result:
            payload = shot["execution_plan"]
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["candidate_count"], 3)
            self.assertEqual(payload["recovery_budget"], 3)
            self.assertEqual(payload["provider_generation_duration_s"], 5.0)
            self.assertIn("first_frame", payload["required_capabilities"])
            # 计划可直接还原（持久化/消费同一 schema）。
            self.assertIsNotNone(ShotExecutionPlan.from_mapping(payload))

    def test_auto_split_parts_each_get_their_own_plan(self) -> None:
        from agent.nodes.storyboard_gen import _attach_execution_plans, _ensure_one_action_beat_per_shot

        shots = [
            _shot(
                shot_id="complex_0001",
                duration=6.0,
                character_action="主角追逐敌人，翻滚闪避后跳跃翻越障碍",
                dialogue=[],
            )
        ]

        parts = _ensure_one_action_beat_per_shot(shots, _fixed_5s_provider())
        planned = _attach_execution_plans(parts, _fixed_5s_provider(), "standard")

        self.assertGreater(len(planned), 1)
        for part in planned:
            change = part["timing"]["structure_change"]
            self.assertEqual(change["kind"], "split")
            self.assertIn(change["reason"], {"complex_motion+action_beat_count", "complex_motion"})
            payload = part["execution_plan"]
            self.assertEqual(payload["candidate_count"], 2)
            self.assertEqual(payload["recovery_budget"], 2)
            self.assertEqual(payload["continuity_mode"], "continuous_action")
            self.assertGreater(payload["narrative_duration_ms"], 0)


class SubtitlePlanTests(unittest.TestCase):
    """字幕与视频/合成共用同一执行计划：窗口与逐句时间轴取计划实测值。"""

    def test_cues_use_plan_window_and_measured_dialogue_timing(self) -> None:
        from services.subtitle_service import DialogueLineInput, ShotDialogueInput, cues_from_shots

        execution_plan = ShotExecutionPlan.derive(
            _shot(duration=4.5),
            provider=_fixed_5s_provider(),
            dialogue_timing=[{"speaker": "主角", "text": "来了。", "start_ms": 200, "end_ms": 1400}],
            dialogue_timing_source="tts_measured",
        )
        shot_input = ShotDialogueInput(
            shot_id="s1",
            sequence=1,
            start_ms=0,
            duration_ms=4500,
            dialogue="来了。",
            lines=[
                # 旧口径的粗略时间轴（整镜），计划实测值必须覆盖它。
                DialogueLineInput(speaker="主角", line="来了。", start_ms=0, end_ms=4500)
            ],
            execution_plan=execution_plan,
        )

        cues = cues_from_shots([shot_input])

        self.assertEqual(len(cues), 1)
        self.assertEqual((cues[0].start_ms, cues[0].end_ms), (200, 1400))

    def test_cues_without_plan_keep_legacy_duration_behavior(self) -> None:
        from services.subtitle_service import DialogueLineInput, ShotDialogueInput, cues_from_shots

        shot_input = ShotDialogueInput(
            shot_id="s2",
            sequence=1,
            start_ms=1000,
            duration_ms=3000,
            dialogue="来了。",
            lines=[DialogueLineInput(speaker="主角", line="来了。", start_ms=0, end_ms=3000)],
        )

        cues = cues_from_shots([shot_input])

        self.assertEqual((cues[0].start_ms, cues[0].end_ms), (1000, 4000))


if __name__ == "__main__":
    unittest.main()
