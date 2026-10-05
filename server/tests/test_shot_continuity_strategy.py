"""镜头连续性模式与上一镜末帧使用策略测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from PIL import Image  # noqa: E402

from agent.output_schemas import parse_storyboard_output  # noqa: E402
from services.consistency_service import ConsistencyService  # noqa: E402
from services.reference_readiness_service import build_manifest_for_shot  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402


def _png(name: str) -> str:
    path = TEST_ROOT / "output" / "continuity_strategy" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (120, 80, 40)).save(path)
    return str(path)


class ShotContinuityStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ConsistencyService()
        self.last_frame = _png("previous_last.png")
        self.previous = {
            "shot_id": "shot_1",
            "sequence": 1,
            "scene_number": 1,
            "scene_group_id": "scene-a",
            "character_action": "她向前跑",
            "storyboard_path": _png("previous_storyboard.png"),
            "last_frame_path": self.last_frame,
        }

    def _context(self, current: dict, previous: dict | None = "auto"):
        previous_shot = self.previous if previous == "auto" else previous
        return self.service.build_generation_context(
            shot=current,
            characters=[
                {
                    "id": "char-1",
                    "name": "林夏",
                    "reference_images": [_png("character.png")],
                }
            ],
            scenes={
                "scene-1": {
                    "scene_group_key": "scene-a",
                    "baseline_image_path": _png("scene.png"),
                }
            },
            previous_shot=previous_shot,
            for_video=True,
        )

    def test_same_scene_continuous_action_uses_previous_last_frame(self) -> None:
        context = self._context(
            {
                "shot_id": "shot_2",
                "scene_number": 1,
                "scene_asset_id": "scene-1",
                "characters_in_scene": ["林夏"],
                "continuity_mode": "continuous_action",
                "character_action": "继续向前跑",
            }
        )

        profile = context["continuity_profile"]
        self.assertEqual(profile["continuity_mode"], "continuous_action")
        self.assertTrue(profile["continuity_reference_used"])
        self.assertEqual(context["continuity_reference_path"], self.last_frame)
        self.assertIn("continuity_frame", {item["type"] for item in context["reference_assets"]})

    def test_same_scene_defaults_to_identity_only_without_previous_image(self) -> None:
        context = self._context(
            {
                "shot_id": "shot_2",
                "scene_number": 1,
                "scene_asset_id": "scene-1",
                "characters_in_scene": ["林夏"],
                "character_action": "停下看向窗外",
            }
        )

        profile = context["continuity_profile"]
        self.assertEqual(profile["continuity_mode"], "same_scene")
        self.assertTrue(profile["inherits_scene_identity"])
        self.assertTrue(profile["inherits_character_identity"])
        self.assertFalse(profile["inherits_previous_image"])
        self.assertEqual(context["continuity_reference_path"], "")
        types = {item["type"] for item in context["reference_assets"]}
        self.assertIn("scene_baseline", types)
        self.assertIn("character_three_view", types)
        self.assertNotIn("continuity_frame", types)

    def test_reverse_shot_never_inherits_previous_last_frame(self) -> None:
        context = self._context(
            {
                "shot_id": "shot_2",
                "scene_number": 1,
                "continuity_mode": "反打",
                "character_action": "看向林夏",
            }
        )

        profile = context["continuity_profile"]
        self.assertEqual(profile["continuity_mode"], "reverse_shot")
        self.assertFalse(profile["continuity_reference_used"])
        self.assertEqual(profile["continuity_reference_reason"], "not_required_by_continuity_mode:reverse_shot")
        self.assertEqual(context["continuity_reference_path"], "")
        self.assertNotIn("continuity_frame", {item["type"] for item in context["reference_assets"]})

    def test_scene_transition_never_inherits_previous_last_frame(self) -> None:
        current = {
            "shot_id": "shot_2",
            "scene_number": 2,
            "character_action": "走进新房间",
        }
        context = self._context(current)

        profile = context["continuity_profile"]
        self.assertEqual(profile["continuity_mode"], "scene_transition")
        self.assertEqual(profile["continuity_mode_source"], "deterministic_rule")
        self.assertFalse(profile["inherits_previous_image"])
        self.assertEqual(context["continuity_reference_path"], "")

    def test_continuous_action_without_previous_last_frame_records_manifest_reason(self) -> None:
        previous = {**self.previous, "last_frame_path": ""}
        context = self._context(
            {
                "shot_id": "shot_2",
                "scene_number": 1,
                "continuity_mode": "continuous_action",
                "character_action": "继续向前跑",
            },
            previous=previous,
        )

        item = context["continuity_manifest_item"]
        self.assertEqual(item["type"], "continuity_reference")
        self.assertFalse(item["used"])
        self.assertEqual(item["status"], "unavailable")
        self.assertEqual(item["reason"], "previous_last_frame_missing")
        self.assertEqual(item["not_used_reason"], "previous_last_frame_missing")
        self.assertNotIn("continuity_frame", {asset["type"] for asset in context["reference_assets"]})

    def test_missing_previous_shot_records_manifest_reason(self) -> None:
        context = self._context(
            {
                "shot_id": "shot_1",
                "scene_number": 1,
                "continuity_mode": "continuous_action",
                "character_action": "向前跑",
            },
            previous=None,
        )

        self.assertEqual(context["continuity_manifest_item"]["reason"], "previous_shot_missing")
        self.assertEqual(context["continuity_reference_path"], "")

    def test_deterministic_rule_detects_continuous_action(self) -> None:
        previous = {**self.previous, "character_action": "她向前跑"}
        current = {
            "shot_id": "shot_2",
            "scene_number": 1,
            "character_action": "继续向前跑",
        }

        decision = self.service.resolve_continuity(current, previous)

        self.assertEqual(decision["continuity_mode"], "continuous_action")
        self.assertEqual(decision["continuity_mode_source"], "deterministic_rule")
        self.assertEqual(decision["continuity_reference_path"], self.last_frame)

    def test_persisted_video_manifest_records_missing_reference_reason(self) -> None:
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        shot = SimpleNamespace(
            project_id="project-1",
            character_asset_ids="[]",
            scene_asset_id="",
            continuity_profile="{}",
            continuity_reference_path="",
            storyboard_path=_png("approved_first.png"),
            image_path="",
        )
        profile = {
            "continuity_mode": "continuous_action",
            "continuity_mode_source": "storyboard",
            "continuity_reference_used": False,
            "continuity_reference_reason": "previous_last_frame_missing",
        }

        manifest = build_manifest_for_shot(
            db,
            shot,
            stage="video",
            continuity_profile=profile,
            continuity_reference_path="",
        )

        continuity = next(item for item in manifest if item["type"] == "continuity_reference")
        self.assertEqual(continuity["reason"], "previous_last_frame_missing")
        self.assertEqual(continuity["not_used_reason"], "previous_last_frame_missing")
        self.assertIn("approved_storyboard_first_frame", {item["type"] for item in manifest})

    def test_storyboard_mode_aliases_are_normalized(self) -> None:
        parsed = parse_storyboard_output(
            {
                "shots": [
                    {"scene_number": 1, "continuity_mode": "match-on-action"},
                    {"scene_number": 1, "continuity_mode": "反打"},
                    {"scene_number": 1, "continuity_mode": "跨场景"},
                    {"scene_number": 1, "continuity_mode": "unknown-mode"},
                ]
            }
        )

        self.assertEqual(
            [item.continuity_mode for item in parsed.shots],
            ["continuous_action", "reverse_shot", "scene_transition", ""],
        )


if __name__ == "__main__":
    unittest.main()
