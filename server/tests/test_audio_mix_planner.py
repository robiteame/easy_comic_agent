"""混音规划器测试：filter_complex 生成、Ducking 拓扑与可复现 manifest。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from services.audio_mix_planner import (  # noqa: E402
    MixTrackInput,
    manifest_digest,
    plan_audio_mix,
)
from test_environment import TEST_ROOT as _TEST_ROOT  # noqa: E402,F401


def _statement(plan, marker: str) -> str:
    """返回包含 marker 的那条 filter 语句（避免依赖语句顺序）。"""

    for statement in plan.filter_complex.split(";"):
        if marker in statement:
            return statement
    raise AssertionError(f"filter 中没有包含 {marker} 的语句: {plan.filter_complex}")


def _music(**overrides) -> MixTrackInput:
    defaults = dict(
        id="bgm1",
        kind="music",
        media_path="/tmp/media/bgm.mp3",
        media_duration_ms=30_000,
        start_ms=0,
        volume=0.8,
    )
    defaults.update(overrides)
    return MixTrackInput(**defaults)


class PlanShapeTests(unittest.TestCase):
    def test_no_tracks_returns_none_for_render(self) -> None:
        self.assertIsNone(plan_audio_mix([], 30.0))

    def test_allow_empty_produces_main_only_plan_for_preview(self) -> None:
        plan = plan_audio_mix([], 30.0, allow_empty=True)
        self.assertIsNotNone(plan)
        self.assertIn("[main0]anull[voicepre]", plan.filter_complex)
        self.assertIn("anoisesrc=color=pink", plan.filter_complex)
        self.assertEqual(plan.inputs, [])

    def test_single_track_chain_order(self) -> None:
        plan = plan_audio_mix(
            [_music(volume=0.8, pan=0.5, fade_in_ms=500, fade_out_ms=1000, start_ms=2000)],
            60.0,
            include_ambient_bed=False,
        )
        self.assertIsNotNone(plan)
        chain = _statement(plan, "[t0]")
        for fragment in (
            "aresample=44100",
            "volume=0.8",
            "pan=stereo|c0=0.5*c0|c1=1*c1",
            "afade=t=in:st=0:d=0.5",
            "afade=t=out:st=29:d=1",
            "adelay=2000:all=1",
            "[t0]",
        ):
            self.assertIn(fragment, chain, f"filter 缺少片段: {fragment}")
        final = plan.filter_complex.split(";")[-1]
        self.assertIn("amix=inputs=2:duration=first:normalize=0", final)
        self.assertIn("atrim=duration=60", final)

    def test_trim_and_loop(self) -> None:
        trimmed = plan_audio_mix([_music(trim_start_ms=1000, trim_end_ms=2000)], 60.0, include_ambient_bed=False)
        chain = _statement(trimmed, "[t0]")
        self.assertIn("atrim=start=1", chain)
        self.assertIn("atrim=end=28", chain)
        self.assertIn("asetpts=PTS-STARTPTS", chain)

        looped = plan_audio_mix([_music(loop=True)], 60.0, include_ambient_bed=False)
        self.assertIn("aloop=loop=-1:size=", looped.filter_complex)

    def test_ambient_bed_matches_legacy_parameters(self) -> None:
        plan = plan_audio_mix([_music()], 60.0)
        self.assertIn("anoisesrc=color=pink:amplitude=0.008:sample_rate=44100", plan.filter_complex)
        self.assertIn("volume=0.08[amb]", plan.filter_complex)
        self.assertIn("[voice_main][t0][amb]", plan.filter_complex)


class DuckingTests(unittest.TestCase):
    def test_ducked_track_uses_sidechaincompress(self) -> None:
        plan = plan_audio_mix(
            [_music(duck_amount_db=-12.0, duck_attack_ms=100, duck_release_ms=400)],
            60.0,
            include_ambient_bed=False,
        )
        self.assertIn("[voicepre]asplit=2[voice_main][sc0]", plan.filter_complex)
        self.assertIn(
            "[t0][sc0]sidechaincompress=threshold=0.02:ratio=7:attack=100:release=400[td0]",
            plan.filter_complex,
        )
        self.assertIn("[voice_main][td0]", plan.filter_complex)
        self.assertEqual(plan.ducked_track_ids, ["bgm1"])

    def test_multiple_duck_tracks_split_voice_for_each(self) -> None:
        plan = plan_audio_mix(
            [
                _music(id="m1", duck_amount_db=-6.0),
                _music(id="m2", duck_amount_db=-6.0),
            ],
            60.0,
            include_ambient_bed=False,
        )
        self.assertIn("asplit=3[voice_main][sc0][sc1]", plan.filter_complex)
        self.assertEqual(plan.ducked_track_ids, ["m1", "m2"])

    def test_zero_duck_disables_sidechain(self) -> None:
        plan = plan_audio_mix([_music(duck_amount_db=0.0)], 60.0, include_ambient_bed=False)
        self.assertNotIn("sidechaincompress", plan.filter_complex)
        self.assertEqual(plan.ducked_track_ids, [])

    def test_dialogue_joins_voice_and_is_capped_to_shot(self) -> None:
        dialogue = MixTrackInput(
            id="d1",
            kind="dialogue",
            media_path="/tmp/media/tts.wav",
            media_duration_ms=2500,
            start_ms=3000,
            delay_ms=200,
            clip_limit_ms=3000,
        )
        plan = plan_audio_mix([dialogue, _music(duck_amount_db=-12.0)], 60.0, include_ambient_bed=False)
        self.assertIn("[main0][dlg0]amix=inputs=2:duration=first:normalize=0[voicepre]", plan.filter_complex)
        chain = _statement(plan, "[dlg0]")
        self.assertIn("adelay=3200:all=1", chain)
        self.assertIn("atrim=duration=2.8", chain)


class ReproducibilityTests(unittest.TestCase):
    def test_same_input_same_filter_and_manifest(self) -> None:
        tracks = [
            _music(duck_amount_db=-9.0),
            MixTrackInput(id="sfx", kind="sfx", media_path="/x/a.wav", media_duration_ms=900, start_ms=5_000),
        ]
        first = plan_audio_mix(tracks, 42.0)
        second = plan_audio_mix(tracks, 42.0)
        self.assertEqual(first.filter_complex, second.filter_complex)
        self.assertEqual(first.track_manifest, second.track_manifest)
        self.assertEqual(manifest_digest(first.track_manifest), manifest_digest(second.track_manifest))

    def test_manifest_changes_when_parameters_change(self) -> None:
        base = plan_audio_mix([_music()], 30.0)
        changed = plan_audio_mix([_music(volume=0.4)], 30.0)
        self.assertNotEqual(manifest_digest(base.track_manifest), manifest_digest(changed.track_manifest))

    def test_unsafe_numbers_are_clamped_not_rejected(self) -> None:
        plan = plan_audio_mix(
            [_music(volume=99.0, pan=5.0, fade_in_ms=-3, delay_ms=-9)], 30.0, include_ambient_bed=False
        )
        chain = _statement(plan, "[t0]")
        self.assertIn("volume=4", chain)
        self.assertIn("pan=stereo|c0=0*c0|c1=1*c1", chain)
        self.assertNotIn("afade=t=in", chain)
        self.assertNotIn("adelay=-", chain)

    def test_muted_and_invalid_tracks_are_skipped_with_warning(self) -> None:
        plan = plan_audio_mix(
            [_music(muted=True), MixTrackInput(id="bad", kind="sfx", media_path="", media_duration_ms=0)],
            30.0,
        )
        self.assertIsNone(plan)

        warned = plan_audio_mix(
            [_music(muted=True), MixTrackInput(id="bad", kind="sfx", media_path="", media_duration_ms=0)],
            30.0,
            allow_empty=True,
        )
        self.assertIsNotNone(warned)
        self.assertEqual(len(warned.warnings), 1)
        self.assertEqual(warned.inputs, [])


if __name__ == "__main__":
    unittest.main()
