"""字幕工作台纯文本层的测试：SRT/VTT 解析序列化、校验、自动生成与 ASS。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from config import settings  # noqa: E402
from services.subtitle_service import (  # noqa: E402
    ShotDialogueInput,
    SubtitleCueData,
    SubtitleStyle,
    SubtitleValidationError,
    build_ass_document,
    cues_from_shots,
    detect_cue_overlaps,
    format_srt_time,
    format_vtt_time,
    parse_cues,
    parse_srt,
    parse_timecode,
    parse_vtt,
    serialize_srt,
    serialize_vtt,
    validate_cue_text,
    validate_cues,
)
from test_environment import TEST_ROOT as _TEST_ROOT  # noqa: E402,F401


def _cue(start: int, end: int, text: str, name: str = "") -> SubtitleCueData:
    return SubtitleCueData(start_ms=start, end_ms=end, text=text, character_name=name)


class SrtVttRoundTripTests(unittest.TestCase):
    def test_srt_round_trip_is_lossless(self) -> None:
        cues = [
            _cue(0, 1500, "你好，世界"),
            _cue(2_000, 3_500, "第二行\n多行文本"),
            _cue(61_500, 62_000, "roll 'em!"),
        ]
        parsed = parse_srt(serialize_srt(cues))
        self.assertEqual(
            [(c.start_ms, c.end_ms, c.text) for c in parsed], [(c.start_ms, c.end_ms, c.text) for c in cues]
        )

    def test_vtt_round_trip_is_lossless(self) -> None:
        cues = [_cue(120, 9_800, "第一句"), _cue(10_000, 12_500, "line A\nline B")]
        parsed = parse_vtt(serialize_vtt(cues))
        self.assertEqual(
            [(c.start_ms, c.end_ms, c.text) for c in parsed], [(c.start_ms, c.end_ms, c.text) for c in cues]
        )

    def test_parse_srt_tolerates_bom_crlf_and_missing_index(self) -> None:
        raw = "\ufeff1\r\n00:00:01,000 --> 00:00:02,000\r\nhello\r\n\r\n00:00:03,000 --> 00:00:04,500\r\nworld\r\n"
        cues = parse_srt(raw)
        self.assertEqual(len(cues), 2)
        self.assertEqual((cues[0].start_ms, cues[0].end_ms, cues[0].text), (1000, 2000, "hello"))
        self.assertEqual((cues[1].start_ms, cues[1].end_ms), (3000, 4500))

    def test_parse_vtt_skips_note_blocks_and_position_suffix(self) -> None:
        raw = "WEBVTT\n\nNOTE 这是一段注释\n跨行也没有关系\n\n00:00:00.500 --> 00:00:02.000 line:85%\n带定位后缀\n"
        cues = parse_vtt(raw)
        self.assertEqual(len(cues), 1)
        self.assertEqual((cues[0].start_ms, cues[0].end_ms, cues[0].text), (500, 2000, "带定位后缀"))

    def test_parse_vtt_requires_header(self) -> None:
        with self.assertRaises(SubtitleValidationError):
            parse_cues("00:00 --> 00:01\nx", "vtt")

    def test_srt_accepts_dot_milliseconds_separator(self) -> None:
        cues = parse_srt("1\n00:00:01.250 --> 00:00:02.750\ndot\n")
        self.assertEqual((cues[0].start_ms, cues[0].end_ms), (1250, 2750))

    def test_timecode_formatting(self) -> None:
        self.assertEqual(format_srt_time(3_723_456), "01:02:03,456")
        self.assertEqual(format_vtt_time(3_723_456), "01:02:03.456")
        self.assertEqual(parse_timecode("01:02:03,456"), 3_723_456)
        self.assertIsNone(parse_timecode("nonsense"))


class CueValidationTests(unittest.TestCase):
    def test_rejects_control_characters(self) -> None:
        with self.assertRaises(SubtitleValidationError):
            validate_cue_text("包含\x01控制符")

    def test_rejects_oversized_text(self) -> None:
        original = settings.MAX_SUBTITLE_CUE_CHARS
        settings.MAX_SUBTITLE_CUE_CHARS = 5
        try:
            with self.assertRaises(SubtitleValidationError):
                validate_cue_text("超过五个字符的文本")
        finally:
            settings.MAX_SUBTITLE_CUE_CHARS = original

    def test_rejects_blank_and_reversed_times(self) -> None:
        with self.assertRaises(SubtitleValidationError):
            validate_cue_text("   ")
        with self.assertRaises(SubtitleValidationError):
            validate_cues([_cue(1000, 1000, "同时")])
        with self.assertRaises(SubtitleValidationError):
            validate_cues([_cue(2000, 1000, "倒序")])

    def test_rejects_too_short_cues(self) -> None:
        with self.assertRaises(SubtitleValidationError):
            validate_cues([_cue(0, 10, "过短")])

    def test_validate_cues_sorts_and_indexes(self) -> None:
        validated = validate_cues([_cue(5000, 6000, "B"), _cue(0, 900, "A")])
        self.assertEqual([cue.order_index for cue in validated], [0, 1])
        self.assertEqual([cue.text for cue in validated], ["A", "B"])

    def test_cue_count_limit(self) -> None:
        original = settings.MAX_SUBTITLE_CUES
        settings.MAX_SUBTITLE_CUES = 2
        try:
            with self.assertRaises(SubtitleValidationError):
                validate_cues([_cue(0, 100, "a"), _cue(200, 300, "b"), _cue(400, 500, "c")])
        finally:
            settings.MAX_SUBTITLE_CUES = original


class GenerateFromShotsTests(unittest.TestCase):
    def test_cues_follow_shot_spans_and_tts_duration(self) -> None:
        shots = [
            ShotDialogueInput("s1", 1, 0, 3000, "对白一", "小明", tts_duration_ms=1800),
            ShotDialogueInput("s2", 2, 3000, 2000, "对白二", "小红", tts_duration_ms=0),
            ShotDialogueInput("s3", 3, 5000, 2500, "", "小刚"),
        ]
        cues = cues_from_shots(shots)
        self.assertEqual(len(cues), 2)
        self.assertEqual((cues[0].start_ms, cues[0].end_ms), (0, 1800))
        self.assertEqual(cues[0].character_name, "小明")
        # TTS 时长未知时使用镜头时长，不越出镜头边界。
        self.assertEqual((cues[1].start_ms, cues[1].end_ms), (3000, 5000))

    def test_tts_longer_than_shot_is_clamped(self) -> None:
        shots = [ShotDialogueInput("s1", 1, 0, 1200, "很长", "小明", tts_duration_ms=9000)]
        cues = cues_from_shots(shots)
        self.assertEqual(cues[0].end_ms, 1200)

    def test_minimum_display_duration(self) -> None:
        shots = [ShotDialogueInput("s1", 1, 0, 3000, "短句", "小明", tts_duration_ms=200)]
        cues = cues_from_shots(shots)
        self.assertEqual(cues[0].end_ms - cues[0].start_ms, 800)


class AssDocumentTests(unittest.TestCase):
    def test_style_and_escaping(self) -> None:
        style = SubtitleStyle(
            font_family="Noto Sans",
            font_size=54,
            primary_color="#FFFFFF",
            outline_color="#101010",
            outline_width=3,
            bold=True,
            position="top",
            safe_margin=54,
        )
        cues = [_cue(0, 1000, "换行\n与{花括号}和\\反斜杠", "小明")]
        doc = build_ass_document(style, cues, 1080, 1920)
        self.assertIn("PlayResX: 1080", doc)
        self.assertIn("PlayResY: 1920", doc)
        self.assertIn("Noto Sans,54", doc)
        self.assertIn("&H00FFFFFF", doc)  # 白色主文字（ASS 的 &HAABBGGRR）
        self.assertIn("&H00101010", doc)  # 描边色 #101010 → BGR 倒序
        self.assertIn(",3,0,8,54,54,54,1", doc)  # Outline=3, Shadow=0, Alignment=8(top), 边距=54
        self.assertIn("\\N", doc)
        self.assertIn("\\{花括号\\}", doc)
        self.assertIn("0:00:00.00,0:00:01.00", doc)

    def test_resolution_scaling(self) -> None:
        scaled = SubtitleStyle(font_size=54, safe_margin=54, outline_width=4).scaled(540, 960)
        self.assertEqual(scaled.font_size, 27)
        self.assertEqual(scaled.safe_margin, 27)
        self.assertEqual(scaled.outline_width, 2)

    def test_unknown_position_falls_back_to_bottom(self) -> None:
        scaled = SubtitleStyle(position="diagonal").scaled(1080, 1920)
        self.assertEqual(scaled.position, "bottom")


class OverlapDetectionTests(unittest.TestCase):
    def test_detects_and_reports_overlaps(self) -> None:
        cues = [_cue(0, 2000, "A"), _cue(1500, 3000, "B"), _cue(3000, 4000, "C")]
        overlaps = detect_cue_overlaps(cues)
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0].overlap_ms, 500)

    def test_no_overlap_when_touching(self) -> None:
        self.assertEqual(detect_cue_overlaps([_cue(0, 1000, "A"), _cue(1000, 2000, "B")]), [])


if __name__ == "__main__":
    unittest.main()
