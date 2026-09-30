"""回归测试：分镜时长预算必须真正约束生成与导出时间线。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from services.story_timing import (  # noqa: E402
    ProviderDurationCapability,
    StoryTimingError,
    StoryTimingPlan,
    estimate_speech_ms,
)


class StoryTimingPlanTests(unittest.TestCase):
    def test_30_second_target_is_rebalanced_to_provider_duration(self) -> None:
        capability = ProviderDurationCapability(
            protocol="fixed-test",
            fixed_duration=5,
            min_duration=5,
            max_duration=5,
            duration_step=5,
        )
        shots = [
            {"shot_id": f"shot_{index}", "scene_number": index, "duration": 1.0, "character_action": "缓慢看向门口", "dialogue": []}
            for index in range(8)
        ]

        plan = StoryTimingPlan(target_duration_s=30, provider=capability)
        result = plan.rebalance(shots)

        self.assertEqual(len(result), 6)
        self.assertAlmostEqual(sum(item["duration"] for item in result), 30.0, places=3)
        self.assertTrue(plan.target_feasible)
        self.assertTrue(all(item["duration"] == 5 for item in result))
        self.assertEqual(plan.to_dict()["target_duration_s"], 30)

    def test_six_second_complex_action_is_split_into_continuous_shots(self) -> None:
        capability = ProviderDurationCapability(
            protocol="flexible-test",
            fixed_duration=None,
            min_duration=2,
            max_duration=6,
            duration_step=1,
        )
        shot = {
            "shot_id": "complex_001",
            "scene_number": 1,
            "duration": 6.0,
            "character_action": "主角冲刺，翻滚，躲闪，挥剑反击",
            "dialogue": [{"speaker": "主角", "line": "来了！"}],
        }

        plan = StoryTimingPlan(target_duration_s=6, provider=capability)
        result = plan.rebalance([shot])

        self.assertEqual(len(result), 2)
        self.assertEqual([item["timing"]["split_part"] for item in result], [1, 2])
        self.assertEqual(sum(item["duration"] for item in result), 6.0)
        self.assertTrue(all(capability.contains(item["duration"]) for item in result))

    def test_dialogue_that_cannot_fit_is_rejected_with_shot_id(self) -> None:
        capability = ProviderDurationCapability(
            protocol="speech-test",
            fixed_duration=None,
            min_duration=1,
            max_duration=3,
            duration_step=1,
        )
        dialogue = [{"speaker": "主角", "line": "这是一段无论如何都无法在三秒内完整说出的长对白，必须被明确提示而不是裁掉。"}]
        shot = {
            "shot_id": "speech_001",
            "scene_number": 1,
            "duration": 2.0,
            "character_action": "站定",
            "dialogue": dialogue,
        }
        plan = StoryTimingPlan(target_duration_s=3, provider=capability)

        with self.assertRaises(StoryTimingError) as raised:
            plan.rebalance([shot])

        self.assertIn("speech_001", str(raised.exception))
        self.assertGreater(estimate_speech_ms(dialogue), 3000)

    def test_short_dialogue_shot_is_extended_instead_of_truncated(self) -> None:
        capability = ProviderDurationCapability(
            protocol="speech-test",
            fixed_duration=None,
            min_duration=4,
            max_duration=8,
            duration_step=1,
        )
        shot = {
            "shot_id": "short_001",
            "scene_number": 1,
            "duration": 1.0,
            "character_action": "",
            "dialogue": [{"speaker": "主角", "line": "听我说，这件事非常重要。"}],
        }
        plan = StoryTimingPlan(target_duration_s=4, provider=capability)

        result = plan.rebalance([shot])

        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(result[0]["duration"], 4.0)
        self.assertGreater(result[0]["estimated_speech_ms"], 0)


if __name__ == "__main__":
    unittest.main()


class RenderTimingValidationTests(unittest.TestCase):
    def test_render_validation_reports_target_and_dialogue_audio_conflicts(self) -> None:
        capability = ProviderDurationCapability(
            protocol="render-test",
            fixed_duration=None,
            min_duration=2,
            max_duration=8,
            duration_step=1,
        )
        plan = StoryTimingPlan(target_duration_s=10, provider=capability)
        shots = [
            {
                "shot_id": "render_001",
                "duration": 4.0,
                "character_action": "",
                "dialogue": [{"speaker": "主角", "line": "这句对白很长，需要完整播放。"}],
                "estimated_speech_ms": estimate_speech_ms("这句对白很长，需要完整播放。"),
                "audio_path": "/audio/dialogue.wav",
                "video_path": "/video/shot.mp4",
                "timing": {"audio_source": "tts"},
            }
        ]

        issues = plan.validate_timeline(
            shots,
            audio_tracks=[
                {
                    "id": "track_001",
                    "kind": "dialogue",
                    "shot_id": "render_001",
                    "start_ms": 0,
                    "source_duration_ms": 6500,
                    "resolved_source_path": "/audio/dialogue.wav",
                }
            ],
            media_durations_ms={
                "/audio/dialogue.wav": 6500,
                "/video/shot.mp4": 4000,
            },
        )
        codes = {issue.code for issue in issues}

        self.assertIn("target_duration_mismatch", codes)
        self.assertIn("dialogue_audio_exceeds_shot", codes)
        self.assertIn("dialogue_track_exceeds_span", codes)
        self.assertTrue(any("render_001" in issue.message for issue in issues))

    def test_provider_rejects_shot_outside_fixed_duration(self) -> None:
        capability = ProviderDurationCapability(
            protocol="fixed-test",
            fixed_duration=5,
            min_duration=5,
            max_duration=5,
            duration_step=5,
        )
        plan = StoryTimingPlan(target_duration_s=5, provider=capability)

        issues = plan.validate_shots(
            [{"shot_id": "invalid_001", "duration": 4.0, "dialogue": []}],
            require_target=False,
        )

        self.assertEqual(issues[0].code, "provider_duration_invalid")
        self.assertIn("invalid_001", issues[0].message)
