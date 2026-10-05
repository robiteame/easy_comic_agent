"""effective style 解析与全链路一致性的验收测试。

- 项目/本次请求的 style 是默认权威来源；Skill 只有在显式开启
  ``style_override_enabled`` 时才能覆盖；
- 历史默认 ``style_template_id=anime`` 不得把 realistic 请求静默改写；
- script → storyboard → image → video 全链路解析出同一个 effective style。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from api.routes import script as script_route  # noqa: E402
from services.skill_config_service import (  # noqa: E402
    DEFAULT_AGENT_CONFIG,
    apply_agent_config_to_shot,
    resolve_effective_style,
)
from services.style_templates import style_prompt_params, style_template  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402

LEGACY_SKILL = {
    "script_agent": {**DEFAULT_AGENT_CONFIG, "style_template_id": "anime"},
    "storyboard_agent": {**DEFAULT_AGENT_CONFIG, "style_template_id": "anime"},
}


class ResolveEffectiveStyleTests(unittest.TestCase):
    def test_legacy_anime_skill_does_not_override_realistic(self) -> None:
        meta = resolve_effective_style("realistic", LEGACY_SKILL, "storyboard_agent")
        self.assertEqual(meta["effective_style"], "realistic")
        self.assertEqual(meta["requested_style"], "realistic")
        self.assertEqual(meta["style_source"], "project_request")

    def test_explicit_override_wins_and_reports_source(self) -> None:
        skill = {
            "script_agent": {**DEFAULT_AGENT_CONFIG, "style_template_id": "anime", "style_override_enabled": True},
            "storyboard_agent": {**DEFAULT_AGENT_CONFIG, "style_template_id": "anime", "style_override_enabled": True},
        }
        meta = resolve_effective_style("realistic", skill, "storyboard_agent")
        self.assertEqual(meta["effective_style"], "anime")
        self.assertEqual(meta["style_source"], "skill_override")

    def test_explicit_override_with_realistic_skill_keeps_realistic(self) -> None:
        skill = {
            "script_agent": {**DEFAULT_AGENT_CONFIG, "style_template_id": "realistic", "style_override_enabled": True},
            "storyboard_agent": {
                **DEFAULT_AGENT_CONFIG,
                "style_template_id": "realistic",
                "style_override_enabled": True,
            },
        }
        meta = resolve_effective_style("anime", skill, "script_agent")
        self.assertEqual(meta["effective_style"], "realistic")
        self.assertEqual(meta["style_source"], "skill_override")

    def test_missing_request_falls_back_without_skill(self) -> None:
        meta = resolve_effective_style("", LEGACY_SKILL, "script_agent")
        self.assertEqual(meta["effective_style"], "anime")
        self.assertEqual(meta["style_source"], "project_request")


class PipelineConsistencyTests(unittest.TestCase):
    def test_initial_state_carries_style_meta_for_every_stage(self) -> None:
        state = script_route._initial_state(
            {"project_id": "p1", "user_input": "text", "style": "realistic"},
            LEGACY_SKILL,
        )
        self.assertEqual(state["style"], "realistic")
        self.assertEqual(state["requested_style"], "realistic")
        self.assertEqual(state["effective_style"], "realistic")
        self.assertEqual(state["style_source"], "project_request")
        # 图像/视频阶段读取的 style_params 必须来自同一个 effective style。
        self.assertEqual(state["style_params"]["prompt_prefix"], style_prompt_params("realistic")["prompt_prefix"])

    def test_storyboard_stage_uses_same_style(self) -> None:
        state = script_route._initial_state(
            {"project_id": "p1", "user_input": "text", "style": "realistic"},
            LEGACY_SKILL,
        )
        shot_data = {"style": state["effective_style"]}
        apply_agent_config_to_shot(shot_data, LEGACY_SKILL, "storyboard_agent")
        self.assertEqual(shot_data["style"], "realistic")
        self.assertEqual(shot_data["effective_style"], "realistic")
        # 视频服务按 shot["style"] 取模板，必须与项目一致。
        self.assertEqual(style_template(shot_data["style"])["label"], style_template("realistic")["label"])

    def test_persist_phase1_writes_effective_style_not_skill_default(self) -> None:
        # _persist_phase1 落库 project.style 用 state 的 effective_style；
        # 这里直接验证初始状态在 legacy skill 下仍是 realistic（不再写回 anime）。
        state = script_route._initial_state(
            {"project_id": "p1", "user_input": "text", "style": "realistic"},
            LEGACY_SKILL,
        )
        persisted_style = state.get("effective_style") or state.get("style")
        self.assertEqual(persisted_style, "realistic")


if __name__ == "__main__":
    unittest.main()
