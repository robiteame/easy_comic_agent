"""Prompt 预算裁剪与关键字段保底的验收测试。

- 裁剪只丢低优先级字段，不做末尾硬截断；
- 人物身份、动作、情绪、运镜在极端超预算时仍必须留在 Prompt；
- 正负向合并时去重并移除冲突词。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT  # noqa: F401,E402

from services.image_service import ImageService  # noqa: E402
from services.prompt_budget import (  # noqa: E402
    assemble_prompt,
    dedupe_terms,
    ensure_critical_fields,
    remove_conflicting_terms,
)
from services.style_templates import style_prompt_params  # noqa: E402
from services.video_service import VideoService  # noqa: E402


class AssemblePromptTests(unittest.TestCase):
    def test_drops_lowest_priority_fields_first(self) -> None:
        fields = [
            ("style", "cinematic realism"),
            ("identity", "Lin Wan, black long hair"),
            ("action", "turns toward the door"),
            ("sop", "low priority " * 200),
        ]
        prompt, dropped = assemble_prompt(fields, max_chars=80)
        self.assertIn("Lin Wan", prompt)
        self.assertIn("turns toward the door", prompt)
        self.assertIn("sop", dropped)
        self.assertLessEqual(len(prompt), 80)

    def test_never_cuts_a_field_in_the_middle(self) -> None:
        fields = [("identity", "A" * 50), ("sop", "B" * 100)]
        prompt, dropped = assemble_prompt(fields, max_chars=60)
        self.assertIn("A" * 50, prompt)
        self.assertEqual(dropped, ["sop"])

    def test_conflicting_negative_terms_removed_from_positive(self) -> None:
        positive, removed = remove_conflicting_terms(
            "live-action cinematic realism, cartoon style girl, natural skin",
            ["cartoon style girl", "watermark"],
        )
        self.assertNotIn("cartoon", positive)
        self.assertIn("natural skin", positive)
        self.assertEqual(removed, ["cartoon style girl"])

    def test_dedupe_terms(self) -> None:
        self.assertEqual(dedupe_terms(["watermark", "Watermark", "blurry"]), ["watermark", "blurry"])


class EnsureCriticalFieldsTests(unittest.TestCase):
    def test_missing_critical_field_is_readded(self) -> None:
        prompt, readded = ensure_critical_fields(
            "style only",
            [("character_identity", "Lin Wan, black long hair"), ("camera_movement", "camera movement: 平移")],
            max_chars=200,
        )
        self.assertIn("Lin Wan", prompt)
        self.assertIn("平移", prompt)
        self.assertIn("character_identity", readded)
        self.assertIn("camera_movement", readded)

    def test_tail_is_trimmed_to_make_room_for_critical_fact(self) -> None:
        prompt = "style, " + "filler, " * 40
        prompt, readded = ensure_critical_fields(
            prompt,
            [("emotion", "emotional tone: angry")],
            max_chars=120,
        )
        self.assertIn("emotional tone: angry", prompt)
        self.assertLessEqual(len(prompt), 120)
        self.assertEqual(readded, ["emotion"])


class ServicePromptBudgetTests(unittest.TestCase):
    """极端预算下，身份/动作/情绪/运镜四类关键字段必须幸存。"""

    def test_video_prompt_keeps_identity_action_emotion_camera(self) -> None:
        service = VideoService()
        shot = {
            "shot_id": "p_shot_0001",
            "style": "realistic",
            "scene_description": "场景 " * 400,
            "character_action": "她攥紧伞柄转身冲向门口",
            "emotion": "angry",
            "camera_movement": "跟随",
            "camera_angle": "侧面",
            "shot_type": "medium",
            "storyboard_prompt": "storyboard " * 200,
            "visual_notes": "note " * 200,
            "consistency_context": "sop " * 400,
            "skill_prompt_append": "append " * 200,
        }
        characters = [
            {
                "id": "c1",
                "name": "林晚",
                "visual_prompt": "Lin Wan, 26-year-old woman, oval face, black long hair, beige trench coat",
                "key_features": ["黑长直"],
                "appearance": {"hair": "black long hair"},
            }
        ]
        with patch_budget(2000):
            prompt = service._build_prompt(shot, characters, {})
        self.assertIn("Lin Wan", prompt)
        self.assertIn("她攥紧伞柄转身冲向门口", prompt)
        self.assertIn("angry", prompt)
        self.assertIn("跟随", prompt)
        self.assertTrue(service.last_prompt_trimmed_fields or service.last_prompt_readded_fields)

    def test_image_prompt_keeps_identity_and_action(self) -> None:
        service = ImageService()
        shot = {
            "shot_id": "p_shot_0002",
            "scene_description": "scene " * 300,
            "character_action": "他扶住门框低头",
            "shot_type": "close-up",
            "camera_angle": "正面",
            "camera_movement": "静止",
            "visual_notes": "note " * 200,
            "consistency_context": "sop " * 300,
        }
        characters = [
            {
                "id": "c1",
                "name": "陈默",
                "visual_prompt": "Chen Mo, 30-year-old man, short black hair, gray hoodie",
                "key_features": ["短黑发"],
                "appearance": {},
            }
        ]
        # 6000 字预算下填充超长低优先级内容，身份与动作仍必须在场。
        prompt, _ = service._build_prompt(shot, characters, style_prompt_params("realistic"))
        self.assertIn("Chen Mo", prompt)
        self.assertIn("他扶住门框低头", prompt)


import unittest.mock as _mock  # noqa: E402


class patch_budget:
    """临时把 video prompt 的 6000 字预算压小，验证关键字段保底。"""

    def __init__(self, max_chars: int):
        self.max_chars = max_chars
        self._patcher = None

    def __enter__(self):
        import services.video_service as vs

        original = vs.assemble_prompt

        def small(fields, *, max_chars=6000):
            return original(fields, max_chars=self.max_chars)

        self._patcher = _mock.patch.object(vs, "assemble_prompt", side_effect=small)
        return self._patcher.__enter__()

    def __exit__(self, *exc):
        return self._patcher.__exit__(*exc)


if __name__ == "__main__":
    unittest.main()
