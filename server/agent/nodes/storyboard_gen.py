import json

from agent.output_schemas import LLMOutputError, ShotOutput, parse_storyboard_output
from agent.state import AgentState
from config import settings
from services.llm_service import LLMService

llm_service = LLMService()


async def run(state: AgentState) -> dict:
    """Generate storyboard shots from parsed scenes."""
    if state.get("storyboard_confirmed") and state.get("shots"):
        return {"current_step": "generate_storyboard"}

    if not llm_service.available:
        raise RuntimeError("未配置可用的 Mimo/LLM API Key，无法生成真实分镜")

    try:
        result = await llm_service.call_json(
            _load_system_prompt(),
            _build_task_prompt(
                script_scenes=state.get("script_scenes", []),
                characters=state.get("characters", []),
                style=state.get("effective_style") or state.get("style", "anime"),
                platform=state.get("platform", "douyin"),
                target_duration=state.get("target_duration", 30),
            ),
            temperature=0.35,
        )
    except Exception as exc:
        raise RuntimeError(f"Mimo 分镜生成失败: {exc}") from exc

    # 模型可能返回 {"shots": [...]} 或直接返回数组；两种情况都经过强 schema 校验，
    # 非法枚举/负时长/超长数组在这里被归一化或丢弃，不会冒泡成未捕获异常。
    try:
        parsed = parse_storyboard_output(result)
    except LLMOutputError as exc:
        raise RuntimeError(f"Mimo 分镜生成结果无法使用: {exc}") from exc
    if not parsed.shots:
        raise RuntimeError("Mimo 分镜生成结果缺少 shots")

    shots = _build_shots(parsed.shots, state)
    if not shots:
        raise RuntimeError("Mimo 分镜生成结果缺少可用镜头")

    return {
        "shots": shots,
        "current_step": "generate_storyboard",
    }


def _build_shots(items: list[ShotOutput], state: AgentState) -> list[dict]:
    """把已校验的镜头输出补全为可落库的分镜（shot_id、seed、默认角色）。"""

    project_id = state["project_id"]
    characters = state.get("characters", [])
    default_character = characters[0]["name"] if characters else "主角"
    scenes = state.get("script_scenes", [])

    shots: list[dict] = []
    for index, item in enumerate(items[: settings.LLM_MAX_SHOTS]):
        scene_number = item.scene_number or item.source_scene_number or min(index + 1, max(len(scenes), 1))
        shots.append(
            {
                "shot_id": item.shot_id or f"{project_id}_shot_{index + 1:04d}",
                "scene_number": scene_number,
                "shot_type": item.shot_type,
                "scene_description": item.scene_description or "角色推进剧情",
                "characters_in_scene": item.characters_in_scene or [default_character],
                "character_action": item.character_action,
                "dialogue": item.dialogue,
                "camera_angle": item.camera_angle,
                "camera_movement": item.camera_movement,
                "emotion": item.emotion,
                "duration": item.duration,
                "transition": item.transition,
                "image_path": item.image_path,
                "audio_path": item.audio_path,
                "status": item.status,
                "confirmed": item.confirmed,
                "version": item.version,
                "seed": item.seed if item.seed is not None else 42 + index,
                "visual_notes": item.visual_notes,
            }
        )
    return shots


def _load_system_prompt() -> str:
    return "你是专业漫剧分镜师。根据剧本场景输出可执行分镜 JSON，不要输出 Markdown。"


def _build_task_prompt(script_scenes: list, characters: list, style: str, platform: str, target_duration: int) -> str:
    return f"""
剧本场景：
{json.dumps(script_scenes, ensure_ascii=False, indent=2)}

角色：
{json.dumps(characters, ensure_ascii=False, indent=2)}

项目实际生效画风（必须严格遵守）：{style}
平台：{platform}
目标时长：{target_duration} 秒
输出 JSON：
{{
  "shots": [
    {{
      "scene_number": 1,
      "shot_type": "wide/medium/close-up/extreme_close",
      "scene_description": "画面描述",
      "characters_in_scene": ["角色名"],
      "character_action": "动作",
      "dialogue": "台词",
      "camera_angle": "正面/侧面/俯视/仰视",
      "camera_movement": "静止/缓慢推进/平移/跟随",
      "emotion": "neutral/happy/shy/sad/angry/surprised",
      "duration": 3.5,
      "transition": "cut/fade/dissolve",
      "visual_notes": "画面注意事项"
    }}
  ]
}}

镜头设计硬性规则：
1. 每个镜头只包含一个主体、一个主要动作和一个镜头运动；不要在一个镜头里堆叠多个动作节拍。
2. 复杂动作（追逐、打斗、转身走位等）必须拆成多个短镜头，靠剪辑衔接。
3. 每个镜头时长控制在 2 到 5 秒之间；视频模型按约 5 秒固定档出片，超出时长的镜头会被拒绝生成。
4. 关键情绪/反转镜头可以标记 visual_notes 建议"生成 2 个候选供挑选"。

风格与一致性规则：同一 scene_number 属于同场景组。不得让同场景镜头出现昼夜、冷暖、光源方向、道具位置、人物站位和180度轴线跳变；Agent 会在后续生成阶段强制以场景组基准图、角色三视图和上一镜头末尾帧覆盖单镜头自定义参数。
"""
