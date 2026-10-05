"""P2: LLM 输出强 schema 的回归测试。

覆盖畸形 JSON、缺字段、错误类型、非法 duration、超长数组与非法枚举，并确认：
- 顶层结构错误抛出可读业务错误，而不是 KeyError/TypeError/ValueError；
- 字段级问题被归一化或按条目拒绝，并记录字段路径与原因；
- 诊断信息不包含 API Key / 完整 prompt / 本地绝对路径。
"""

from __future__ import annotations

import asyncio
import logging
import sys
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from agent import output_schemas  # noqa: E402
from agent.nodes import script_parser, storyboard_gen  # noqa: E402
from agent.output_schemas import (  # noqa: E402
    LLMOutputError,
    parse_script_output,
    parse_storyboard_output,
)
from config import settings  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402


def _script_payload(**overrides) -> dict:
    payload = {
        "title": "测试剧",
        "genre": "甜宠",
        "style_suggestion": "anime",
        "characters": [{"name": "林夏", "appearance": {"hair": "black"}, "voice_type": "少女"}],
        "script_scenes": [
            {
                "scene_number": 1,
                "location": "教室",
                "characters_in_scene": ["林夏"],
                "actions": "林夏走进教室",
                "dialogue": [{"character": "林夏", "line": "早上好", "emotion": "happy"}],
                "emotion": "happy",
                "camera_suggestion": "medium",
            }
        ],
    }
    payload.update(overrides)
    return payload


class TopLevelStructureTests(unittest.TestCase):
    def test_non_object_payload_raises_readable_error(self) -> None:
        for payload in (None, "不是对象", 42, [1, 2, 3]):
            with self.subTest(payload=payload):
                with self.assertRaises(LLMOutputError) as raised:
                    parse_script_output(payload)
                self.assertIn("JSON 对象", str(raised.exception))
                self.assertNotIn("Traceback", str(raised.exception))

    def test_missing_characters_or_scenes_are_reported_by_caller(self) -> None:
        parsed = parse_script_output(_script_payload(characters=None))
        self.assertEqual(parsed.characters, [])
        parsed = parse_script_output(_script_payload(script_scenes=None))
        self.assertEqual(parsed.script_scenes, [])

    def test_scenes_alias_is_accepted(self) -> None:
        payload = _script_payload()
        payload["scenes"] = payload.pop("script_scenes")
        parsed = parse_script_output(payload)
        self.assertEqual(len(parsed.script_scenes), 1)

    def test_mimo_character_aliases_are_normalized(self) -> None:
        payload = _script_payload(
            characters=[{"角色名": "林夏", "appearance": {"hair": "black"}}, {"character": "顾言"}]
        )
        parsed = parse_script_output(payload)
        self.assertEqual([item.name for item in parsed.characters], ["林夏", "顾言"])

    def test_mimo_character_name_mapping_is_normalized(self) -> None:
        payload = _script_payload(characters={"林夏": {"personality": "勇敢"}, "顾言": "戴眼镜"})
        parsed = parse_script_output(payload)
        self.assertEqual([item.name for item in parsed.characters], ["林夏", "顾言"])

    def test_invalid_appearance_type_is_dropped_without_crashing(self) -> None:
        parsed = parse_script_output(_script_payload(characters=[{"name": "林夏", "appearance": []}]))
        self.assertEqual(parsed.characters[0].appearance, {})

    def test_storyboard_accepts_object_or_array(self) -> None:
        shot = {"scene_number": 1, "scene_description": "画面", "duration": 3.5}
        self.assertEqual(len(parse_storyboard_output({"shots": [shot]}).shots), 1)
        self.assertEqual(len(parse_storyboard_output([shot]).shots), 1)

    def test_storyboard_rejects_unusable_top_level(self) -> None:
        for payload in (None, "shots", 7):
            with self.subTest(payload=payload):
                with self.assertRaises(LLMOutputError) as raised:
                    parse_storyboard_output(payload)
                self.assertIn("JSON 对象或数组", str(raised.exception))


class FieldNormalizationTests(unittest.TestCase):
    def test_missing_fields_fall_back_to_defaults(self) -> None:
        parsed = parse_storyboard_output({"shots": [{}]})
        shot = parsed.shots[0]
        self.assertEqual(shot.shot_type, "medium")
        self.assertEqual(shot.emotion, "neutral")
        self.assertEqual(shot.transition, "cut")
        self.assertEqual(shot.camera_angle, "正面")
        self.assertEqual(shot.camera_movement, "静止")
        self.assertEqual(shot.status, "pending")
        self.assertEqual(shot.version, 1)
        self.assertEqual(shot.duration, 3.0)
        self.assertIsNone(shot.seed)

    def test_wrong_field_types_do_not_escape_as_type_errors(self) -> None:
        parsed = parse_storyboard_output(
            {
                "shots": [
                    {
                        "scene_description": {"unexpected": "object"},
                        "dialogue": ["数组形式"],
                        "duration": "不是数字",
                        "scene_number": "abc",
                    }
                ]
            }
        )
        shot = parsed.shots[0]
        self.assertEqual(shot.scene_description, "")
        # 对白为结构化列表：字符串数组形态迁移为逐句对象（speaker 为空）。
        self.assertEqual(len(shot.dialogue), 1)
        self.assertEqual(shot.dialogue[0].line, "数组形式")
        self.assertEqual(shot.dialogue[0].speaker, "")
        self.assertEqual(shot.duration, 3.0)
        self.assertEqual(shot.scene_number, 1)

    def test_illegal_duration_is_clamped(self) -> None:
        parsed = parse_storyboard_output(
            {
                "shots": [
                    {"duration": -12},
                    {"duration": 10**9},
                    {"duration": float("inf")},
                    {"duration": float("nan")},
                ]
            }
        )
        self.assertEqual(parsed.shots[0].duration, settings.MIN_SHOT_DURATION_SECONDS)
        self.assertEqual(parsed.shots[1].duration, settings.MAX_SHOT_DURATION_SECONDS)
        self.assertEqual(parsed.shots[2].duration, 3.0)
        self.assertEqual(parsed.shots[3].duration, 3.0)

    def test_illegal_enums_are_normalized(self) -> None:
        parsed = parse_storyboard_output(
            {
                "shots": [
                    {
                        "shot_type": "extreme close-up",
                        "camera_angle": "low angle",
                        "camera_movement": "slow push",
                        "emotion": "tense",
                        "transition": "白闪",
                        "status": "视频完成",
                        "version": 0,
                    }
                ]
            }
        )
        shot = parsed.shots[0]
        self.assertEqual(shot.shot_type, "extreme_close")
        self.assertEqual(shot.camera_angle, "仰视")
        self.assertEqual(shot.camera_movement, "缓慢推进")
        self.assertEqual(shot.emotion, "angry")
        self.assertEqual(shot.transition, "white_flash")
        self.assertEqual(shot.status, "pending")
        self.assertEqual(shot.version, 1)

    def test_style_suggestion_only_accepts_known_styles(self) -> None:
        self.assertEqual(parse_script_output(_script_payload(style_suggestion="chinese")).style_suggestion, "chinese")
        self.assertEqual(parse_script_output(_script_payload(style_suggestion="赛博朋克")).style_suggestion, "anime")
        self.assertEqual(
            parse_script_output(_script_payload(style_suggestion=""), fallback_style="realistic").style_suggestion,
            "realistic",
        )

    def test_nested_value_types_are_bounded(self) -> None:
        parsed = parse_script_output(
            _script_payload(
                characters=[
                    {
                        "name": " 林夏 ",
                        "appearance": {"hair": "black", "nested": {"bad": True}, "long": "x" * 900},
                        "key_features": [" 清爽造型 ", 5],
                        "seed": "不是数字",
                    }
                ]
            )
        )
        character = parsed.characters[0]
        self.assertEqual(character.name, "林夏")
        self.assertNotIn("nested", character.appearance)
        self.assertEqual(len(character.appearance["long"]), output_schemas.MAX_APPEARANCE_VALUE_CHARS)
        self.assertEqual(character.key_features, ["清爽造型", "5"])
        self.assertIsNone(character.seed)


class BoundedCollectionTests(unittest.TestCase):
    def test_oversized_arrays_are_truncated_to_the_configured_caps(self) -> None:
        with self.assertLogs("agent.output_schemas", level=logging.WARNING) as captured:
            parsed = parse_script_output(
                _script_payload(
                    characters=[{"name": f"角色{index}"} for index in range(50)],
                    script_scenes=[
                        {
                            "scene_number": index + 1,
                            "dialogue": [{"line": f"第{line}句"} for line in range(60)],
                        }
                        for index in range(40)
                    ],
                    logic_issues=["问题"] * 100,
                )
            )
        self.assertEqual(len(parsed.characters), settings.LLM_MAX_CHARACTERS)
        self.assertEqual(len(parsed.script_scenes), settings.LLM_MAX_SCENES)
        self.assertEqual(len(parsed.script_scenes[0].dialogue), settings.LLM_MAX_DIALOGUE_LINES)
        self.assertEqual(len(parsed.logic_issues), output_schemas.MAX_LOGIC_ISSUES)
        self.assertTrue(any("超长已截断" in line for line in captured.output))

    def test_shots_are_truncated_to_the_configured_cap(self) -> None:
        parsed = parse_storyboard_output({"shots": [{"scene_number": index + 1} for index in range(80)]})
        self.assertEqual(len(parsed.shots), settings.LLM_MAX_SHOTS)

    def test_overlong_text_is_truncated_and_logged(self) -> None:
        with self.assertLogs("agent.output_schemas", level=logging.WARNING) as captured:
            parsed = parse_storyboard_output({"shots": [{"scene_description": "景" * 9000}]})
        self.assertEqual(len(parsed.shots[0].scene_description), settings.LLM_MAX_TEXT_CHARS)
        self.assertTrue(any("field=scene_description" in line for line in captured.output))


class RejectedItemTests(unittest.TestCase):
    def test_non_object_items_are_dropped_with_path_and_reason(self) -> None:
        with self.assertLogs("agent.output_schemas", level=logging.WARNING) as captured:
            parsed = parse_storyboard_output({"shots": [{"scene_number": 1}, "坏数据", 42, {"scene_number": 2}]})
        self.assertEqual([shot.scene_number for shot in parsed.shots], [1, 2])
        messages = " ".join(captured.output)
        self.assertIn("path=shots[1]", messages)
        self.assertIn("path=shots[2]", messages)
        self.assertIn("应为对象", messages)

    def test_rejected_dialogue_records_field_path(self) -> None:
        with self.assertLogs("agent.output_schemas", level=logging.WARNING) as captured:
            parsed = parse_script_output(
                _script_payload(
                    script_scenes=[
                        {
                            "scene_number": 1,
                            "dialogue": [
                                {"line": "正常"},
                                {"line": {"nested": "object"}},
                                "不是对象",
                            ],
                        }
                    ]
                )
            )
        # 非对象条目被拒绝（无法修复），字段类型错误则被置空并记录字段名。
        self.assertEqual(len(parsed.script_scenes[0].dialogue), 2)
        self.assertEqual(parsed.script_scenes[0].dialogue[0].line, "正常")
        self.assertEqual(parsed.script_scenes[0].dialogue[1].line, "")
        messages = " ".join(captured.output)
        self.assertIn("path=dialogue[2]", messages)
        self.assertIn("field=line", messages)
        self.assertIn("应为对象", messages)

    def test_diagnostics_never_contain_secrets_or_local_paths(self) -> None:
        secret = "sk-abcdefghijklmnop0123456789"
        with self.assertLogs("agent.output_schemas", level=logging.WARNING) as captured:
            parse_storyboard_output({"shots": [secret, {"scene_description": "/Users/someone/private/script.txt"}]})
        messages = " ".join(captured.output)
        self.assertNotIn(secret, messages)
        self.assertNotIn("/Users/someone", messages)


class NodeIntegrationTests(unittest.TestCase):
    """节点层：模型返回脏数据时只抛可读的 RuntimeError。"""

    class _StubLLM:
        def __init__(self, payload):
            self.payload = payload
            self.available = True

        async def call_json(self, *args, **kwargs):
            if isinstance(self.payload, Exception):
                raise self.payload
            return self.payload

    def _run_script_parser(self, payload) -> dict:
        original = script_parser.llm_service
        script_parser.llm_service = self._StubLLM(payload)
        try:
            return asyncio.run(
                script_parser.run(
                    {
                        "project_id": "llm-schema-project",
                        "user_input": "林夏走进教室，和同学打招呼。",
                        "style": "anime",
                    }
                )
            )
        finally:
            script_parser.llm_service = original

    def _run_storyboard(self, payload) -> dict:
        original = storyboard_gen.llm_service
        storyboard_gen.llm_service = self._StubLLM(payload)
        try:
            return asyncio.run(
                storyboard_gen.run(
                    {
                        "project_id": "llm-schema-project",
                        "characters": [{"name": "林夏"}],
                        "script_scenes": [{"scene_number": 1, "actions": "走进教室"}],
                        "style": "anime",
                    }
                )
            )
        finally:
            storyboard_gen.llm_service = original

    def test_script_parser_survives_malformed_payload(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            self._run_script_parser("完全不是 JSON 对象")
        self.assertIn("剧本解析结果无法使用", str(raised.exception))
        self.assertNotIn("Traceback", str(raised.exception))

    def test_script_parser_reports_readable_error_when_everything_is_dropped(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            self._run_script_parser({"characters": ["坏"], "script_scenes": ["坏"]})
        self.assertIn("缺少可用角色", str(raised.exception))

    def test_script_parser_builds_characters_and_scenes_from_messy_output(self) -> None:
        result = self._run_script_parser(
            {
                "title": "  测试剧  ",
                "genre": "甜宠",
                "style_suggestion": "未知",
                "characters": [
                    {"name": "林夏", "appearance": {"features": "清爽造型、短发"}, "voice_type": "少女"},
                    "坏数据",
                ],
                "script_scenes": [
                    {
                        "scene_number": -5,
                        "location": "教室",
                        "actions": "走进教室",
                        "emotion": "开心",
                        "camera_suggestion": "closeup",
                        "dialogue": [{"character": "林夏", "line": "早上好", "emotion": "tense"}],
                    }
                ],
            }
        )
        self.assertEqual(result["script_title"], "测试剧")
        self.assertEqual(result["style_suggestion"], "anime")
        self.assertEqual(len(result["characters"]), 1)
        character = result["characters"][0]
        self.assertEqual(character["key_features"], ["清爽造型", "短发"])
        self.assertIn("emotion_variants", character)
        self.assertEqual(character["seed"], 42)
        scene = result["script_scenes"][0]
        self.assertEqual(scene["scene_number"], 1)
        self.assertEqual(scene["emotion"], "happy")
        self.assertEqual(scene["camera_suggestion"], "close-up")
        self.assertEqual(scene["dialogue"][0]["emotion"], "angry")

    def test_storyboard_parser_survives_unusable_payload(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            self._run_storyboard({"shots": "不是数组"})
        self.assertIn("分镜生成结果无法使用", str(raised.exception))

    def test_storyboard_parser_builds_shots_with_bounded_values(self) -> None:
        result = self._run_storyboard(
            {
                "shots": [
                    {
                        "scene_number": 0,
                        "shot_type": "extreme close-up",
                        "duration": -3,
                        "emotion": "tense",
                        "camera_angle": "low",
                        "camera_movement": "slow push",
                        "transition": "白闪",
                        "characters_in_scene": "林夏",
                    },
                    "坏数据",
                ]
            }
        )
        self.assertEqual(len(result["shots"]), 1)
        shot = result["shots"][0]
        self.assertEqual(shot["shot_id"], "llm-schema-project_shot_0001")
        self.assertEqual(shot["scene_number"], 1)
        self.assertEqual(shot["shot_type"], "extreme_close")
        self.assertEqual(shot["duration"], settings.MIN_SHOT_DURATION_SECONDS)
        self.assertEqual(shot["emotion"], "angry")
        self.assertEqual(shot["camera_angle"], "仰视")
        self.assertEqual(shot["camera_movement"], "缓慢推进")
        self.assertEqual(shot["transition"], "white_flash")
        self.assertEqual(shot["characters_in_scene"], ["林夏"])
        self.assertEqual(shot["seed"], 42)


if __name__ == "__main__":
    unittest.main()
