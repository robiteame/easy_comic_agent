"""子 Agent 系统提示词的唯一事实源。

默认提示词原先散落在 script.py / script_parser.py / storyboard_gen.py，
现在集中到这里；Skill 方案里的 ``system_prompt`` 为空时各调用点回落到
本模块的内置默认值，保证默认行为与历史版本完全一致。

JSON 输出契约是固定文本：用户自定义 system_prompt 只能改角色定位与
创作要求，最终拼装时契约追加在用户提示词之后，输出结构约束不可删除。
"""

from __future__ import annotations

from typing import Any

# 与各节点历史硬编码逐字一致：空配置回落时必须发送同样的文本。
SCRIPT_GENERATION_SYSTEM_PROMPT = (
    "你是漫剧编剧。请输出完整中文漫剧剧本，包含标题、人物、场景、动作、对白和情绪，不要输出解释。"
)
SCRIPT_PARSE_SYSTEM_PROMPT = (
    "你是资深漫剧编导。请把用户输入解析成角色、场景、对白和情绪，输出严格 JSON，不要输出 Markdown。"
)
STORYBOARD_SYSTEM_PROMPT = (
    "你是专业漫剧分镜师。根据剧本场景输出可执行分镜 JSON，不要输出 Markdown。"
    "每个镜头的对白是结构化数组，每句台词都必须写明说话人 speaker。"
)

# 固定 JSON 输出契约：仅在用户自定义 system_prompt 时追加，防止用户提示词
# 弱化结构约束导致解析结果偏离 schema。默认提示词已内含同等约束，追加会
# 改变默认请求文本，因此默认路径原样返回。
SCRIPT_PARSE_JSON_CONTRACT = (
    "输出契约（固定，不可被自定义提示词覆盖）：只输出一个严格合法的 JSON 对象，"
    "禁止输出 Markdown 代码块、注释或任何解释文字。"
    "JSON 结构必须与任务提示中的结构完全一致："
    "title、genre、style_suggestion、characters、script_scenes、logic_issues；"
    "characters 内每个角色包含 name 与 appearance；"
    "script_scenes 内每场包含 scene_number、location、characters_in_scene、actions、dialogue；"
    "dialogue 内每句对白包含 character 与 line 字段。"
)
STORYBOARD_JSON_CONTRACT = (
    "输出契约（固定，不可被自定义提示词覆盖）：只输出一个严格合法的 JSON 对象，"
    "禁止输出 Markdown 代码块、注释或任何解释文字。"
    "JSON 结构必须与任务提示中的结构完全一致：顶层为 shots 数组，"
    "每个镜头包含 scene_number、scene_description、characters_in_scene、"
    "character_action、action_beats、dialogue 等字段；"
    "dialogue 是结构化数组，每句台词都必须写明说话人 speaker，不得合并或省略。"
)

# 每个 Agent 槽位的代表默认值（UI 展示/兜底用）。剧本生成与剧本解析共用
# script_agent 槽位，但内置默认提示词不同，由调用点通过 fallback 指定。
DEFAULT_SYSTEM_PROMPTS: dict[str, str] = {
    "script_agent": SCRIPT_GENERATION_SYSTEM_PROMPT,
    "storyboard_agent": STORYBOARD_SYSTEM_PROMPT,
}

MAX_SYSTEM_PROMPT_LENGTH = 20000


def resolve_system_prompt(
    skill_config: dict[str, Any] | None,
    agent: str,
    *,
    fallback: str = "",
) -> str:
    """解析当前 Skill 方案中某个 Agent 的系统提示词。

    ``skill_config`` 必须是任务启动时快照进 AgentState 的那份配置，节点
    不得重新读取全局配置文件。配置值为空（空串/空白/缺失/非法类型）时
    返回 ``fallback``，fallback 也为空时返回该槽位的内置默认提示词。
    """
    custom = custom_system_prompt(skill_config, agent)
    if custom:
        return custom
    default = str(fallback or "").strip()
    return default or DEFAULT_SYSTEM_PROMPTS.get(agent, "")


def resolve_json_system_prompt(
    skill_config: dict[str, Any] | None,
    agent: str,
    *,
    fallback: str,
    contract: str,
) -> str:
    """解析依赖严格 JSON schema 的调用点的系统提示词。

    用户自定义提示词时在末尾追加固定 JSON 输出契约（与用户提示词冲突时
    契约为最终输出结构约束）；配置为空时原样返回内置默认提示词，保证
    默认行为与历史版本一致。
    """
    custom = custom_system_prompt(skill_config, agent)
    if not custom:
        return str(fallback or "").strip()
    fixed = str(contract or "").strip()
    if not fixed:
        return custom
    return f"{custom}\n\n{fixed}"


def custom_system_prompt(skill_config: dict[str, Any] | None, agent: str) -> str:
    """读取并清洗用户配置的 system_prompt；不可用配置一律视为空。"""

    if not isinstance(skill_config, dict):
        return ""
    config = skill_config.get(agent)
    if not isinstance(config, dict):
        return ""
    return sanitize_system_prompt(config.get("system_prompt"))


def sanitize_system_prompt(value: Any) -> str:
    """归一化 system_prompt：非字符串、超长、纯控制字符都视为空。

    去除首尾空白但保留换行；清空输入框等价于使用内置默认提示词。
    """
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    if not cleaned:
        return ""
    if len(cleaned) > MAX_SYSTEM_PROMPT_LENGTH:
        return ""
    if all(ord(char) < 32 or ord(char) == 127 for char in cleaned):
        return ""
    return cleaned


def validate_system_prompt(value: Any, *, field: str) -> str:
    """保存入口的严格校验：类型/长度/纯控制字符不合法时抛 ValueError。

    空字符串合法，表示使用内置默认提示词；路由层捕获后返回 400。
    """
    if value is None:
        raise ValueError(f"{field} 必须是字符串（留空表示使用默认系统提示词）")
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须是字符串，当前类型为 {type(value).__name__}")
    cleaned = value.strip()
    if len(cleaned) > MAX_SYSTEM_PROMPT_LENGTH:
        raise ValueError(f"{field} 超过最大长度 {MAX_SYSTEM_PROMPT_LENGTH} 字符")
    if cleaned and all(ord(char) < 32 or ord(char) == 127 for char in cleaned):
        raise ValueError(f"{field} 不能只包含控制字符")
    return cleaned
