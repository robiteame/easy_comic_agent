"""Agent 系统提示词可配置（Skill 方案 system_prompt）的验收测试。

覆盖需求：
1. 默认配置下，script_parser / storyboard_gen 使用与历史硬编码逐字一致的原默认 Prompt；
2. 自定义 script_agent.system_prompt 后，剧本生成与剧本解析调用中出现该 Prompt；
3. 自定义 storyboard_agent.system_prompt 后，分镜生成调用中出现该 Prompt；
4. system_prompt 为空（空串/空白/缺失/非法类型）时回退默认 Prompt；
5. 旧配置文件没有 system_prompt 字段时可正常读取并补默认值；
6. 项目绑定与单剧集绑定解析到不同 Skill 方案的 system_prompt；
7. 用户自定义 Prompt 不会删除固定 JSON 输出契约，解析结果仍通过 schema 校验；
8. 保存/另存为模板往返保留 system_prompt，重置语义 = 空串默认值；
9. 超长、空值（null）、非法类型在保存入口返回明确的 400 错误。
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from fastapi import HTTPException  # noqa: E402

from agent.nodes import script_parser, storyboard_gen  # noqa: E402
from agent.output_schemas import parse_script_output, parse_storyboard_output  # noqa: E402
from api.routes import script as script_route  # noqa: E402
from api.routes import settings as settings_route  # noqa: E402
from api.routes.settings import SkillTemplateSave  # noqa: E402
from services.prompts import (  # noqa: E402
    MAX_SYSTEM_PROMPT_LENGTH,
    SCRIPT_GENERATION_SYSTEM_PROMPT,
    SCRIPT_PARSE_JSON_CONTRACT,
    SCRIPT_PARSE_SYSTEM_PROMPT,
    STORYBOARD_JSON_CONTRACT,
    STORYBOARD_SYSTEM_PROMPT,
    resolve_json_system_prompt,
    resolve_system_prompt,
    validate_system_prompt,
)
from services.skill_config_service import (  # noqa: E402
    DEFAULT_AGENT_CONFIG,
    _normalize_agent_config,
    list_skill_templates,
    resolve_skill_config,
    save_skill_template,
)
from test_environment import TEST_ROOT  # noqa: F401,E402

# 历史硬编码的原文（逐字节），用于断言默认行为不变。
LEGACY_SCRIPT_GENERATION_PROMPT = (
    "你是漫剧编剧。请输出完整中文漫剧剧本，包含标题、人物、场景、动作、对白和情绪，不要输出解释。"
)
LEGACY_SCRIPT_PARSE_PROMPT = (
    "你是资深漫剧编导。请把用户输入解析成角色、场景、对白和情绪，输出严格 JSON，不要输出 Markdown。"
)
LEGACY_STORYBOARD_PROMPT = (
    "你是专业漫剧分镜师。根据剧本场景输出可执行分镜 JSON，不要输出 Markdown。"
    "每个镜头的对白是结构化数组，每句台词都必须写明说话人 speaker。"
)

CUSTOM_SCRIPT_PROMPT = "你是古风漫剧编剧，对白讲究韵律与留白。"
CUSTOM_STORYBOARD_PROMPT = "你是快节奏短视频分镜师，偏好高信息密度切镜。"

PARSE_PAYLOAD = {
    "title": "青檐",
    "genre": "古风",
    "characters": [{"name": "沈砚", "appearance": {"hair": "墨发束起"}, "voice_type": "少年"}],
    "script_scenes": [
        {
            "scene_number": 1,
            "location": "书院",
            "characters_in_scene": ["沈砚"],
            "actions": "沈砚在檐下收伞",
            "dialogue": [{"character": "沈砚", "line": "雨停了。", "emotion": "neutral"}],
            "emotion": "neutral",
            "camera_suggestion": "medium",
        }
    ],
    "logic_issues": [],
}

STORYBOARD_PAYLOAD = {
    "shots": [
        {
            "scene_number": 1,
            "shot_type": "medium",
            "scene_description": "沈砚在檐下收伞",
            "characters_in_scene": ["沈砚"],
            "character_action": "收伞抬头",
            "duration": 3.0,
        }
    ]
}


class _RecordingLLM:
    """记录 system prompt 的 LLM stub；按调用顺序返回预设结果。"""

    def __init__(self, results: list) -> None:
        self.results = list(results)
        self.calls: list[dict] = []
        self.available = True

    async def call_json(self, system_prompt, user_prompt, temperature=0.3, **kwargs):
        self.calls.append({"system": system_prompt, "user": user_prompt, "kwargs": kwargs})
        return self._next()

    async def call(self, system_prompt, user_prompt, temperature=0.3, **kwargs):
        self.calls.append({"system": system_prompt, "user": user_prompt, "kwargs": kwargs})
        return self._next()

    def _next(self):
        if not self.results:
            raise AssertionError("意外的额外 LLM 调用")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _skill_config(script_prompt: str = "", storyboard_prompt: str = "") -> dict:
    return {
        "script_agent": {**DEFAULT_AGENT_CONFIG, "system_prompt": script_prompt},
        "storyboard_agent": {**DEFAULT_AGENT_CONFIG, "system_prompt": storyboard_prompt},
    }


def _run_parser(stub, state_extra: dict | None = None) -> dict:
    original_llm = script_parser.llm_service
    original_memory = script_parser.project_memory
    script_parser.llm_service = stub
    script_parser.project_memory = type(
        "MemoryStub",
        (),
        {"save_characters": lambda *a, **k: None, "save_narrative_context": lambda *a, **k: None},
    )()
    try:
        state = {
            "project_id": "prompt-project",
            "user_input": "沈砚走进书院，收伞抬头说：雨停了。",
            "style": "anime",
        }
        state.update(state_extra or {})
        return asyncio.run(script_parser.run(state))
    finally:
        script_parser.llm_service = original_llm
        script_parser.project_memory = original_memory


def _run_storyboard(stub, state_extra: dict | None = None) -> dict:
    original_llm = storyboard_gen.llm_service
    storyboard_gen.llm_service = stub
    try:
        state = {
            "project_id": "prompt-project",
            "characters": [{"name": "沈砚"}],
            "script_scenes": [{"scene_number": 1, "actions": "檐下收伞"}],
            "style": "anime",
        }
        state.update(state_extra or {})
        return asyncio.run(storyboard_gen.run(state))
    finally:
        storyboard_gen.llm_service = original_llm


def _run_generation(stub, skill_config: dict | None) -> str:
    original_llm = script_route.llm_service
    script_route.llm_service = stub
    try:
        data = script_route.ScriptGenerateRequest(prompt="古风校园", style="anime")
        return asyncio.run(script_route._generate_script_text(data, skill_config))
    finally:
        script_route.llm_service = original_llm


class _StoreFileGuard:
    """备份/还原沙箱里的 skill_config_templates.json，避免用例互相污染。"""

    def __enter__(self):
        from config import settings

        self.path = settings.DATA_DIR / "skill_config_templates.json"
        self.backup = self.path.read_text(encoding="utf-8") if self.path.exists() else None
        return self

    def __exit__(self, *exc_info):
        if self.backup is None:
            self.path.unlink(missing_ok=True)
        else:
            self.path.write_text(self.backup, encoding="utf-8")
        return False


def _find_template(listed: dict, template_id: str) -> dict:
    for template in listed["templates"]:
        if template["id"] == template_id:
            return template
    raise AssertionError(f"模板未找到: {template_id}")


class DefaultPromptTests(unittest.TestCase):
    """验收 1/4：默认与空配置使用原默认 Prompt，行为不变。"""

    def test_default_prompts_match_legacy_hardcoded_text(self) -> None:
        self.assertEqual(SCRIPT_GENERATION_SYSTEM_PROMPT, LEGACY_SCRIPT_GENERATION_PROMPT)
        self.assertEqual(SCRIPT_PARSE_SYSTEM_PROMPT, LEGACY_SCRIPT_PARSE_PROMPT)
        self.assertEqual(STORYBOARD_SYSTEM_PROMPT, LEGACY_STORYBOARD_PROMPT)
        self.assertEqual(script_parser._load_system_prompt(), LEGACY_SCRIPT_PARSE_PROMPT)
        self.assertEqual(storyboard_gen._load_system_prompt(), LEGACY_STORYBOARD_PROMPT)

    def test_parser_node_uses_default_prompt_without_skill_config(self) -> None:
        stub = _RecordingLLM([PARSE_PAYLOAD])
        _run_parser(stub)
        self.assertEqual(stub.calls[0]["system"], LEGACY_SCRIPT_PARSE_PROMPT)

    def test_storyboard_node_uses_default_prompt_without_skill_config(self) -> None:
        stub = _RecordingLLM([STORYBOARD_PAYLOAD])
        _run_storyboard(stub)
        self.assertEqual(stub.calls[0]["system"], LEGACY_STORYBOARD_PROMPT)

    def test_generation_uses_default_prompt_without_skill_config(self) -> None:
        stub = _RecordingLLM(["标题：青檐\n场景一……"])
        _run_generation(stub, None)
        self.assertEqual(stub.calls[0]["system"], LEGACY_SCRIPT_GENERATION_PROMPT)

    def test_empty_or_invalid_config_falls_back_to_default(self) -> None:
        for skill_config in (
            {},
            {"script_agent": {}},
            {"script_agent": {"system_prompt": ""}},
            {"script_agent": {"system_prompt": "   \n\t  "}},
            {"script_agent": {"system_prompt": 123}},
            {"script_agent": None},
        ):
            self.assertEqual(
                script_parser._load_system_prompt(skill_config),
                LEGACY_SCRIPT_PARSE_PROMPT,
                f"空/非法配置必须回落默认解析 Prompt: {skill_config!r}",
            )
            self.assertEqual(
                storyboard_gen._load_system_prompt(skill_config),
                LEGACY_STORYBOARD_PROMPT,
                f"空/非法配置必须回落默认分镜 Prompt: {skill_config!r}",
            )
        self.assertEqual(
            resolve_system_prompt(None, "script_agent", fallback=SCRIPT_GENERATION_SYSTEM_PROMPT),
            LEGACY_SCRIPT_GENERATION_PROMPT,
        )

    def test_default_agent_config_uses_empty_prompt_semantics(self) -> None:
        self.assertEqual(DEFAULT_AGENT_CONFIG["system_prompt"], "")
        normalized = _normalize_agent_config({})
        self.assertEqual(normalized["system_prompt"], "")


class CustomPromptTests(unittest.TestCase):
    """验收 2/3：自定义 Prompt 真实进入剧本生成、剧本解析、分镜生成调用。"""

    def test_generation_call_contains_custom_script_prompt(self) -> None:
        stub = _RecordingLLM(["标题：青檐\n场景一……"])
        script = _run_generation(stub, _skill_config(script_prompt=CUSTOM_SCRIPT_PROMPT))
        self.assertEqual(stub.calls[0]["system"], CUSTOM_SCRIPT_PROMPT)
        # 剧本生成输出纯文本：不追加 JSON 契约。
        self.assertNotIn("JSON", stub.calls[0]["system"])
        self.assertIn("青檐", script)

    def test_parser_call_contains_custom_script_prompt_and_contract(self) -> None:
        stub = _RecordingLLM([PARSE_PAYLOAD])
        result = _run_parser(stub, {"skill_config": _skill_config(script_prompt=CUSTOM_SCRIPT_PROMPT)})
        system = stub.calls[0]["system"]
        self.assertIn(CUSTOM_SCRIPT_PROMPT, system)
        self.assertIn(SCRIPT_PARSE_JSON_CONTRACT, system)
        # 契约拼装在用户提示词之后：与用户内容冲突时契约为最终结构约束。
        self.assertGreater(system.index(SCRIPT_PARSE_JSON_CONTRACT), system.index(CUSTOM_SCRIPT_PROMPT))
        self.assertIn("雨停了", json.dumps(result["script_scenes"], ensure_ascii=False))

    def test_storyboard_call_contains_custom_storyboard_prompt_and_contract(self) -> None:
        stub = _RecordingLLM([STORYBOARD_PAYLOAD])
        result = _run_storyboard(stub, {"skill_config": _skill_config(storyboard_prompt=CUSTOM_STORYBOARD_PROMPT)})
        system = stub.calls[0]["system"]
        self.assertIn(CUSTOM_STORYBOARD_PROMPT, system)
        self.assertIn(STORYBOARD_JSON_CONTRACT, system)
        self.assertGreater(system.index(STORYBOARD_JSON_CONTRACT), system.index(CUSTOM_STORYBOARD_PROMPT))
        self.assertEqual(len(result["shots"]), 1)

    def test_segmented_parsing_uses_same_resolved_prompt(self) -> None:
        scenes = 6
        long_script = "\n\n".join(
            f"第{i}场\n" + "沈砚在书院中踱步，檐外雨声渐歇，他望着远山出神。" * 110 + f"\n沈砚：这是第{i}场。"
            for i in range(1, scenes + 1)
        )
        segment_payload = {
            "title": "",
            "genre": "",
            "characters": [{"name": "沈砚", "voice_type": "少年"}],
            "script_scenes": [
                {
                    "scene_number": 1,
                    "location": "书院",
                    "characters_in_scene": ["沈砚"],
                    "actions": "踱步",
                    "dialogue": [{"character": "沈砚", "line": "分段台词", "emotion": "neutral"}],
                }
            ],
        }
        stub = _RecordingLLM([segment_payload] * 3)
        _run_parser(
            stub, {"user_input": long_script, "skill_config": _skill_config(script_prompt=CUSTOM_SCRIPT_PROMPT)}
        )
        self.assertEqual(len(stub.calls), 3, "长剧本按场次分段解析")
        for call in stub.calls:
            self.assertEqual(call["system"], stub.calls[0]["system"], "整段与分段必须使用同一个已解析 Prompt")
            self.assertIn(CUSTOM_SCRIPT_PROMPT, call["system"])
            self.assertIn(SCRIPT_PARSE_JSON_CONTRACT, call["system"])

    def test_custom_prompt_still_passes_schema_validation(self) -> None:
        """验收 7：自定义 Prompt 不破坏 schema——节点跑通且结果通过统一校验。"""
        stub = _RecordingLLM([PARSE_PAYLOAD])
        parsed = _run_parser(stub, {"skill_config": _skill_config(script_prompt="输出所有字段为英文也必须遵守结构。")})
        reparsed = parse_script_output(
            {
                "characters": [
                    {"name": item["name"], "appearance": item["appearance"]} for item in parsed["characters"]
                ],
                "script_scenes": parsed["script_scenes"],
            },
            fallback_style="anime",
        )
        self.assertEqual(len(reparsed.script_scenes), 1)

        stub_shot = _RecordingLLM([STORYBOARD_PAYLOAD])
        shots = _run_storyboard(stub_shot, {"skill_config": _skill_config(storyboard_prompt="用镜头语言讲故事。")})
        reparsed_shots = parse_storyboard_output({"shots": shots})
        self.assertEqual(len(reparsed_shots.shots), 1)
        # 固定契约保留在代码中：自定义提示词无法移除结构约束关键词。
        composed = resolve_json_system_prompt(
            {"storyboard_agent": {"system_prompt": "随意发挥，不需要 JSON。"}},
            "storyboard_agent",
            fallback=STORYBOARD_SYSTEM_PROMPT,
            contract=STORYBOARD_JSON_CONTRACT,
        )
        self.assertIn("严格合法的 JSON", composed)
        self.assertIn("禁止输出 Markdown", composed)


class LegacyStoreTests(unittest.TestCase):
    """验收 5：旧配置文件没有 system_prompt 字段时正常读取并补默认值。"""

    def test_legacy_store_file_without_system_prompt_is_readable(self) -> None:
        from config import settings

        legacy_store = {
            "templates": {
                "default": {
                    "id": "default",
                    "name": "旧默认方案",
                    "script_agent": {
                        "style_template_id": "",
                        "custom_style_keywords": "水彩",
                        "camera_composition": "wide shot",
                    },
                    "storyboard_agent": {},
                }
            },
            "global_default_template_id": "default",
            "project_bindings": {},
            "episode_bindings": {},
        }
        with _StoreFileGuard():
            store_path = settings.DATA_DIR / "skill_config_templates.json"
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store_path.write_text(json.dumps(legacy_store, ensure_ascii=False), encoding="utf-8")

            listed = list_skill_templates()
            template = _find_template(listed, "default")
            self.assertEqual(template["script_agent"]["system_prompt"], "")
            self.assertEqual(template["storyboard_agent"]["system_prompt"], "")
            self.assertEqual(template["script_agent"]["custom_style_keywords"], "水彩")

            resolved = resolve_skill_config()
            self.assertEqual(resolved["script_agent"]["system_prompt"], "")
            # 空配置回落默认提示词，而不是把空串发给模型。
            self.assertEqual(
                script_parser._load_system_prompt(resolved),
                LEGACY_SCRIPT_PARSE_PROMPT,
            )
            self.assertEqual(
                storyboard_gen._load_system_prompt(resolved),
                LEGACY_STORYBOARD_PROMPT,
            )

    def test_binding_scopes_resolve_different_prompts(self) -> None:
        """验收 6：项目绑定与单剧集绑定解析到不同 Skill 方案的 system_prompt。"""
        with _StoreFileGuard():
            save_skill_template(
                {
                    "id": "skill_project_style",
                    "name": "项目方案",
                    "script_agent": {**DEFAULT_AGENT_CONFIG, "system_prompt": "项目级剧本提示词。"},
                    "storyboard_agent": {**DEFAULT_AGENT_CONFIG, "system_prompt": "项目级分镜提示词。"},
                }
            )
            save_skill_template(
                {
                    "id": "skill_episode_style",
                    "name": "剧集方案",
                    "script_agent": {**DEFAULT_AGENT_CONFIG, "system_prompt": "剧集级剧本提示词。"},
                    "storyboard_agent": {**DEFAULT_AGENT_CONFIG, "system_prompt": "剧集级分镜提示词。"},
                }
            )
            from services.skill_config_service import set_skill_bindings

            set_skill_bindings(
                {
                    "project_bindings": {"proj-main": "skill_project_style"},
                    "episode_bindings": {"proj-main-ep02": "skill_episode_style"},
                }
            )

            episode_config = resolve_skill_config("proj-main-ep02")
            self.assertEqual(episode_config["resolved_template_id"], "skill_episode_style")
            self.assertEqual(episode_config["binding_scope"], "episode")
            self.assertIn("剧集级剧本提示词。", script_parser._load_system_prompt(episode_config))
            self.assertIn("剧集级分镜提示词。", storyboard_gen._load_system_prompt(episode_config))

            project_config = resolve_skill_config("proj-main")
            self.assertEqual(project_config["resolved_template_id"], "skill_project_style")
            self.assertEqual(project_config["binding_scope"], "project")
            self.assertIn("项目级剧本提示词。", script_parser._load_system_prompt(project_config))
            self.assertIn("项目级分镜提示词。", storyboard_gen._load_system_prompt(project_config))

            # 未绑定的项目回落全局默认（默认方案 system_prompt 为空 → 内置默认）。
            global_config = resolve_skill_config("proj-other")
            self.assertEqual(global_config["binding_scope"], "global")
            self.assertEqual(script_parser._load_system_prompt(global_config), LEGACY_SCRIPT_PARSE_PROMPT)


class SaveRoundTripTests(unittest.TestCase):
    """验收 8：保存/另存为模板往返保留 system_prompt；重置语义 = 空串默认。"""

    def test_save_and_save_as_preserve_system_prompt(self) -> None:
        with _StoreFileGuard():
            saved = save_skill_template(
                {
                    "id": "skill_prompt_a",
                    "name": "提示词方案A",
                    "script_agent": {**DEFAULT_AGENT_CONFIG, "system_prompt": f"  {CUSTOM_SCRIPT_PROMPT}  "},
                    "storyboard_agent": {**DEFAULT_AGENT_CONFIG, "system_prompt": CUSTOM_STORYBOARD_PROMPT},
                }
            )
            # 去首尾空白但保留换行后落盘。
            self.assertEqual(saved["script_agent"]["system_prompt"], CUSTOM_SCRIPT_PROMPT)
            self.assertEqual(saved["storyboard_agent"]["system_prompt"], CUSTOM_STORYBOARD_PROMPT)

            listed = list_skill_templates()
            stored = _find_template(listed, "skill_prompt_a")
            self.assertEqual(stored["script_agent"]["system_prompt"], CUSTOM_SCRIPT_PROMPT)

            copy = save_skill_template(
                {
                    "id": "skill_prompt_b",
                    "name": "提示词方案B",
                    "script_agent": stored["script_agent"],
                    "storyboard_agent": stored["storyboard_agent"],
                }
            )
            self.assertEqual(copy["script_agent"]["system_prompt"], CUSTOM_SCRIPT_PROMPT)
            self.assertEqual(copy["storyboard_agent"]["system_prompt"], CUSTOM_STORYBOARD_PROMPT)

    def test_multi_line_prompt_is_preserved(self) -> None:
        multi_line = "第一行角色定位。\n第二行创作要求。\n\n第四行语言语气。"
        with _StoreFileGuard():
            saved = save_skill_template(
                {
                    "id": "skill_multiline",
                    "name": "多行提示词",
                    "script_agent": {**DEFAULT_AGENT_CONFIG, "system_prompt": multi_line},
                    "storyboard_agent": {**DEFAULT_AGENT_CONFIG},
                }
            )
            self.assertEqual(saved["script_agent"]["system_prompt"], multi_line)
            self.assertIn(multi_line, script_parser._load_system_prompt(saved))


class ValidationTests(unittest.TestCase):
    """验收 9：超长、空值（null）、非法类型返回明确的 400。"""

    def test_validate_system_prompt_rejects_bad_values(self) -> None:
        with self.assertRaises(ValueError):
            validate_system_prompt(None, field="script_agent.system_prompt")
        with self.assertRaises(ValueError):
            validate_system_prompt(123, field="script_agent.system_prompt")
        with self.assertRaises(ValueError):
            validate_system_prompt({"hacked": True}, field="script_agent.system_prompt")
        with self.assertRaises(ValueError) as raised:
            validate_system_prompt("长" * (MAX_SYSTEM_PROMPT_LENGTH + 1), field="script_agent.system_prompt")
        self.assertIn("20000", str(raised.exception))
        with self.assertRaises(ValueError):
            validate_system_prompt("\x00\x01\x02", field="script_agent.system_prompt")
        # 合法值：空串（= 默认提示词）、普通文本、多行文本。
        self.assertEqual(validate_system_prompt("", field="f"), "")
        self.assertEqual(validate_system_prompt(f"  {CUSTOM_SCRIPT_PROMPT}  ", field="f"), CUSTOM_SCRIPT_PROMPT)
        self.assertEqual(validate_system_prompt("第一行\n第二行", field="f"), "第一行\n第二行")

    def _save_via_route(self, script_prompt, storyboard_prompt=None):
        payload = SkillTemplateSave(
            id="skill_validation_case",
            name="校验用例",
            script_agent={"system_prompt": script_prompt},
            storyboard_agent={"system_prompt": storyboard_prompt if storyboard_prompt is not None else ""},
        )
        with _StoreFileGuard():
            return asyncio.run(settings_route.save_skill_config(payload))

    def test_route_returns_400_for_invalid_system_prompts(self) -> None:
        cases = [
            ("长" * (MAX_SYSTEM_PROMPT_LENGTH + 1), None, "script_agent 超长"),
            (None, None, "script_agent 空值 null"),
            (12345, None, "script_agent 非法类型"),
            ("ok", None if False else "\x00\x01", "storyboard_agent 纯控制字符"),
            ("ok", {"hacked": True}, "storyboard_agent 非法类型"),
        ]
        for script_prompt, storyboard_prompt, label in cases:
            with self.subTest(label=label):
                with self.assertRaises(HTTPException) as raised:
                    self._save_via_route(script_prompt, storyboard_prompt)
                self.assertEqual(raised.exception.status_code, 400)
                self.assertTrue(str(raised.exception.detail), "400 必须携带明确的错误信息")

    def test_route_accepts_empty_and_valid_system_prompts(self) -> None:
        saved = self._save_via_route("")
        self.assertEqual(saved["script_agent"]["system_prompt"], "")
        saved = self._save_via_route(CUSTOM_SCRIPT_PROMPT, CUSTOM_STORYBOARD_PROMPT)
        self.assertEqual(saved["script_agent"]["system_prompt"], CUSTOM_SCRIPT_PROMPT)
        self.assertEqual(saved["storyboard_agent"]["system_prompt"], CUSTOM_STORYBOARD_PROMPT)


if __name__ == "__main__":
    unittest.main()
