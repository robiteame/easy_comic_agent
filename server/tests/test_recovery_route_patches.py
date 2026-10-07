"""Generation-input application for scoped automatic recovery patches."""

from __future__ import annotations

import unittest

from api.routes import shot as shot_route  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: E402


class RecoveryRoutePatchTests(unittest.TestCase):
    def test_visual_prompt_rule_is_applied_only_to_matching_shot_and_stage(self) -> None:
        shot_data = {
            "visual_prompt": "主体站立",
            "visual_notes": "原始备注",
            "storyboard_prompt": "原始分镜",
        }
        revisions = [
            {
                "shot_id": "shot-1",
                "patches": [
                    {
                        "shot_id": "shot-1",
                        "target_stage": "image_generation",
                        "field": "visual_prompt",
                        "op": "replace",
                        "value": {"rule": "主体、动作、镜头运动、光线各一句"},
                    },
                    {
                        "shot_id": "shot-1",
                        "target_stage": "video_generation",
                        "field": "visual_prompt",
                        "op": "replace",
                        "value": {"rule": "不应应用到图像阶段"},
                    },
                    {
                        "shot_id": "other-shot",
                        "target_stage": "image_generation",
                        "field": "visual_prompt",
                        "op": "replace",
                        "value": {"rule": "不应应用到当前镜头"},
                    },
                ],
            }
        ]

        shot_route._apply_recovery_revisions(
            shot_data,
            revisions,
            shot_id="shot-1",
            stage="image_generation",
        )

        self.assertEqual(shot_data["visual_prompt"], "主体、动作、镜头运动、光线各一句")
        self.assertIn("主体、动作、镜头运动、光线各一句", shot_data["visual_notes"])
        self.assertIn("主体、动作、镜头运动、光线各一句", shot_data["storyboard_prompt"])
        self.assertNotIn("不应应用", shot_data["visual_prompt"])

    def test_reference_replacement_uses_real_scene_baseline_and_records_application(self) -> None:
        baseline = TEST_ROOT / "recovery-scene-baseline.png"
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_bytes(b"baseline")
        shot_data = {
            "reference_manifest": [
                {
                    "type": "character_three_view",
                    "path": str(TEST_ROOT / "missing-character.png"),
                    "status": "ready",
                },
                {
                    "type": "scene_baseline",
                    "path": str(baseline),
                    "status": "ready",
                    "version": 3,
                },
            ],
            "reference_assets": [],
            "scene_reference_images": [],
            "character_reference_images": ["old-character.png"],
        }

        shot_route._apply_recovery_revisions(
            shot_data,
            [
                {
                    "shot_id": "shot-1",
                    "patches": [
                        {
                            "shot_id": "shot-1",
                            "target_stage": "image_generation",
                            "field": "reference_images",
                            "op": "replace",
                            "value": {
                                "source": "scene_baseline_or_previous_tail_frame",
                                "record_weight": True,
                            },
                        }
                    ],
                }
            ],
            shot_id="shot-1",
            stage="image_generation",
        )

        self.assertEqual([item["path"] for item in shot_data["reference_manifest"]], [str(baseline)])
        self.assertEqual(shot_data["scene_reference_images"], [str(baseline)])
        self.assertEqual(shot_data["character_reference_images"], [])
        self.assertTrue(shot_data["recovery_reference"]["applied"])
        self.assertEqual(shot_data["recovery_reference"]["path"], str(baseline))

    def test_reference_replacement_is_safe_when_no_valid_file_exists(self) -> None:
        manifest = [{"type": "scene_baseline", "path": str(TEST_ROOT / "not-there.png"), "status": "ready"}]
        shot_data = {"reference_manifest": manifest, "reference_assets": [{"path": "keep"}]}

        shot_route._apply_recovery_revisions(
            shot_data,
            [
                {
                    "shot_id": "shot-1",
                    "patches": [
                        {
                            "target_stage": "video_generation",
                            "field": "reference_images",
                            "op": "replace",
                            "value": {"source": "scene_baseline_or_previous_tail_frame"},
                        }
                    ],
                }
            ],
            shot_id="shot-1",
            stage="video_generation",
        )

        self.assertEqual(shot_data["reference_manifest"], manifest)
        self.assertEqual(shot_data["reference_assets"], [{"path": "keep"}])
        self.assertFalse(shot_data["recovery_reference"]["applied"])
        self.assertEqual(shot_data["recovery_reference"]["reason"], "no_valid_reference")

    def test_video_reference_replacement_retains_approved_storyboard_anchor(self) -> None:
        baseline = TEST_ROOT / "recovery-video-baseline.png"
        storyboard = TEST_ROOT / "recovery-approved-storyboard.png"
        baseline.write_bytes(b"baseline")
        storyboard.write_bytes(b"storyboard")
        shot_data = {
            "reference_manifest": [
                {"type": "scene_baseline", "path": str(baseline), "status": "ready"},
                {"type": "approved_storyboard_first_frame", "path": str(storyboard), "status": "ready"},
            ]
        }

        shot_route._apply_recovery_revisions(
            shot_data,
            [
                {
                    "shot_id": "shot-1",
                    "patches": [
                        {
                            "target_stage": "video_generation",
                            "field": "reference_images",
                            "op": "replace",
                            "value": {"source": "scene_baseline_or_previous_tail_frame"},
                        }
                    ],
                }
            ],
            shot_id="shot-1",
            stage="video_generation",
        )

        self.assertEqual(
            {item["type"] for item in shot_data["reference_manifest"]},
            {"scene_baseline", "approved_storyboard_first_frame"},
        )
        self.assertNotIn("references_sent", shot_data)


if __name__ == "__main__":
    unittest.main()
