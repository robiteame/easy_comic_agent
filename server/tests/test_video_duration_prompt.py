"""视频 Prompt 必须携带真实生成片段时长，不能默认按 5 秒编动作。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.video_service import VideoService  # noqa: E402


class VideoDurationPromptTests(unittest.TestCase):
    def test_prompt_contains_exact_generation_duration_and_speech_budget(self) -> None:
        service = VideoService()
        shot = {
            "shot_id": "prompt_001",
            "duration": 5.0,
            "generation_duration_s": 5.0,
            "estimated_speech_ms": 1800,
            "character_action": "主角转身看向门口",
            "dialogue": [{"speaker": "主角", "line": "来了。"}],
            "camera_movement": "静止",
            "camera_angle": "正面",
            "shot_type": "medium",
            "emotion": "neutral",
        }

        with patch("services.video_service.get_endpoint", return_value=SimpleNamespace(protocol="fixed-test")):
            with patch(
                "services.video_service.get_video_generation_duration_s",
                return_value=5.0,
            ):
                prompt = service._build_prompt(shot, [], {})

        self.assertIn("actual generated clip duration: 5 seconds", prompt)
        self.assertIn("estimated speech duration: 1800 ms", prompt)


if __name__ == "__main__":
    unittest.main()
