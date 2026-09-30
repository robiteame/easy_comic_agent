from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.ffmpeg_service import FFmpegService
from services.post_production_plan import build_post_production_plan


class PostProductionPlanTests(unittest.TestCase):
    def test_dissolve_is_rendered_as_xfade_not_hard_cut(self) -> None:
        shots = [
            {
                "shot_id": "s1",
                "sequence": 1,
                "duration": 3.0,
                "scene_group_id": "scene-a",
                "transition": "dissolve",
                "dialogue": "",
            },
            {"shot_id": "s2", "sequence": 2, "duration": 2.0, "scene_group_id": "scene-a", "dialogue": ""},
        ]
        plan = build_post_production_plan(shots, project_id="p1")
        boundary = plan.transitions[0]
        self.assertEqual(boundary.effective, "dissolve")
        self.assertGreater(boundary.duration_ms, 0)

        captured: list[list[str]] = []

        async def fake_run(_self, args: list[str]) -> None:
            captured.append(args)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            clips = [root / "a.mp4", root / "b.mp4"]
            with patch.object(FFmpegService, "_run", new=fake_run):
                asyncio.run(FFmpegService()._concat_clips(clips, root, [boundary.to_dict()], [3.0, 2.0]))
        filter_complex = next(arg for index, arg in enumerate(captured[0]) if captured[0][index - 1] == "-filter_complex")
        self.assertIn("xfade=transition=dissolve", filter_complex)
        self.assertNotIn("concat=n=2:v=1:a=1", filter_complex)

    def test_cross_scene_white_flash_uses_configured_duration(self) -> None:
        shots = [
            {
                "shot_id": "s1",
                "sequence": 1,
                "duration": 3.0,
                "scene_group_id": "scene-a",
                "dialogue": "",
                "continuity_profile": {"cross_scene_flash_seconds": 0.72},
            },
            {"shot_id": "s2", "sequence": 2, "duration": 2.0, "scene_group_id": "scene-b", "dialogue": ""},
        ]
        plan = build_post_production_plan(shots, project_id="p1")
        boundary = plan.transitions[0]
        self.assertEqual(boundary.scene_relation, "cross_scene")
        self.assertEqual(boundary.effective, "white_flash")
        self.assertEqual(boundary.duration_ms, 720)

        captured: list[list[str]] = []

        async def fake_run(_self, args: list[str]) -> None:
            captured.append(args)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with patch.object(FFmpegService, "_run", new=fake_run):
                asyncio.run(
                    FFmpegService()._concat_clips(
                        [root / "a.mp4", root / "b.mp4"], root, [boundary.to_dict()], [3.0, 2.0]
                    )
                )
        filter_complex = next(arg for index, arg in enumerate(captured[0]) if captured[0][index - 1] == "-filter_complex")
        self.assertIn("duration=0.720", filter_complex)
        self.assertIn("color=white@1", filter_complex)

    def test_dialogue_subtitle_never_crosses_shot_boundary(self) -> None:
        shots = [
            {
                "shot_id": "s1",
                "sequence": 1,
                "duration": 2.0,
                "scene_group_id": "scene-a",
                "transition": "cut",
                "dialogue": "",
            },
            {"shot_id": "s2", "sequence": 2, "duration": 2.0, "scene_group_id": "scene-b", "dialogue": ""},
        ]
        av_config = {
            "audio_tracks": [],
            "subtitle_tracks": [
                {
                    "id": "sub1",
                    "enabled": True,
                    "cues": [
                        {"start_ms": 1500, "end_ms": 2600, "text": "越界字幕", "character_name": "小明"}
                    ],
                }
            ],
        }
        plan = build_post_production_plan(shots, av_config, project_id="p1")
        cue = plan.subtitle_tracks[0]["cues"][0]
        self.assertEqual(cue["shot_id"], "s1")
        self.assertGreaterEqual(cue["start_ms"], plan.shots[0].timeline_start_ms)
        self.assertLessEqual(cue["end_ms"], plan.shots[0].timeline_end_ms)
        self.assertTrue(cue["clamped"])
        self.assertTrue(any("越过镜头" in warning for warning in plan.warnings))

    def test_unsupported_transition_falls_back_to_cut_with_reason(self) -> None:
        shots = [
            {
                "shot_id": "s1",
                "sequence": 1,
                "duration": 2.0,
                "scene_group_id": "scene-a",
                "transition": "star_warp",
                "dialogue": "",
            },
            {"shot_id": "s2", "sequence": 2, "duration": 2.0, "scene_group_id": "scene-b", "dialogue": ""},
        ]
        boundary = build_post_production_plan(shots, project_id="p1").transitions[0]
        self.assertFalse(boundary.supported)
        self.assertEqual(boundary.effective, "cut")
        self.assertEqual(boundary.fallback_reason, "unsupported_transition:star_warp")

    def test_timeline_json_contains_reviewable_shot_audio_subtitle_and_effects(self) -> None:
        shots = [
            {
                "shot_id": "s1",
                "sequence": 1,
                "duration": 3.0,
                "scene_group_id": "scene-a",
                "transition": "push",
                "dialogue": [
                    {
                        "speaker": "小明",
                        "line": "出发。",
                        "emotion": "happy",
                        "start_ms": 200,
                        "end_ms": 1200,
                    }
                ],
                "camera_movement": "推",
                "camera_angle": "侧面",
                "emotion": "happy",
            },
            {"shot_id": "s2", "sequence": 2, "duration": 2.0, "scene_group_id": "scene-b", "dialogue": ""},
        ]
        plan = build_post_production_plan(shots, project_id="p1")
        with tempfile.TemporaryDirectory() as raw:
            path = plan.write_json(Path(raw) / "timeline.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
        entry = payload["shots"][0]
        self.assertEqual(entry["timeline_start_ms"], 0)
        self.assertEqual(entry["timeline_end_ms"], 3000)
        self.assertEqual(entry["transition_out"]["effective"], "push")
        self.assertEqual(entry["dialogue"][0]["speaker"], "小明")
        self.assertEqual(entry["audio"][0]["clip_start_ms"], 200)
        self.assertEqual(entry["camera"]["camera_movement"], "推")
        self.assertEqual(entry["effects"][0]["type"], "camera_movement")

    def test_user_transition_has_priority_over_cross_scene_rule(self) -> None:
        shots = [
            {
                "shot_id": "s1",
                "sequence": 1,
                "duration": 3.0,
                "scene_group_id": "scene-a",
                "transition": "dissolve",
                "dialogue": "",
                "continuity_profile": {
                    "cross_scene_transition": "white_flash",
                    "cross_scene_flash_seconds": 0.8,
                },
            },
            {"shot_id": "s2", "sequence": 2, "duration": 2.0, "scene_group_id": "scene-b", "dialogue": ""},
        ]
        boundary = build_post_production_plan(shots, project_id="p1").transitions[0]
        self.assertEqual(boundary.source, "user_shot_transition")
        self.assertEqual(boundary.effective, "dissolve")
        self.assertNotEqual(boundary.duration_ms, 800)
