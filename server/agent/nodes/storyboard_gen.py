import json
import logging

from agent.output_schemas import LLMOutputError, ShotOutput, parse_storyboard_output
from agent.state import AgentState
from config import settings
from services.llm_service import LLMService
from services.shot_dialogue import dialogue_plain_text, warn_unknown_speakers
from services.story_timing import (
    StoryTimingPlan,
    estimate_action_beats,
    estimate_speech_ms,
    provider_duration_capability,
)

llm_service = LLMService()
logger = logging.getLogger(__name__)


async def run(state: AgentState) -> dict:
    """Generate storyboard shots from parsed scenes."""
    if state.get("storyboard_confirmed") and state.get("shots"):
        return {"current_step": "generate_storyboard"}

    if not llm_service.available:
        raise RuntimeError("未配置可用的 Mimo/LLM API Key，无法生成真实分镜")

    duration_capability = provider_duration_capability()
    target_duration = float(state.get("target_duration", 30) or 30)
    has_explicit_target = state.get("target_duration") is not None
    try:
        result = await llm_service.call_json(
            _load_system_prompt(),
            _build_task_prompt(
                script_scenes=state.get("script_scenes", []),
                characters=state.get("characters", []),
                style=state.get("effective_style") or state.get("style", "anime"),
                platform=state.get("platform", "douyin"),
                target_duration=target_duration,
                duration_capability=duration_capability,
                revision_notes=_revision_notes(state),
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

    # LLM 时长只是初稿。这里按叙事节拍、对白容量和当前 Provider 的真实生成能力
    # 自动拆分/合并/重分配，并把不可行时长作为具体错误阻断，绝不只写进 Prompt。
    timing_plan = StoryTimingPlan(target_duration_s=target_duration, provider=duration_capability)
    if has_explicit_target:
        shots = timing_plan.rebalance(shots)
    else:
        # schema/节点单测没有传目标时长时只做字段归一化，不凭空把一个镜头扩展成整集时长。
        timing_plan.shot_count = len(shots)
        timing_plan.planned_total_duration_s = round(sum(float(item.get("duration") or 0) for item in shots), 3)
        timing_plan.dialogue_duration_s = round(
            sum(estimate_speech_ms(dialogue_plain_text(item.get("dialogue"))) for item in shots) / 1000.0,
            3,
        )
        timing_plan.action_beats = [
            beat
            for item in shots
            for beat in estimate_action_beats(item.get("character_action"))
        ]
    logger.info(
        "分镜时长预算已校正: target=%.3fs planned=%.3fs shots=%d dialogue=%.3fs beats=%d provider=%s",
        timing_plan.target_duration_s,
        timing_plan.planned_total_duration_s,
        timing_plan.shot_count,
        timing_plan.dialogue_duration_s,
        len(timing_plan.action_beats),
        duration_capability.describe(),
    )

    return {
        "shots": shots,
        "timing_plan": timing_plan.to_dict(),
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
        shot_id = item.shot_id or f"{project_id}_shot_{index + 1:04d}"
        dialogue_lines = [
            {
                "speaker": line.speaker,
                "line": line.line,
                "emotion": line.emotion,
                "action": line.action,
                "start_ms": line.start_ms,
                "end_ms": line.end_ms,
            }
            for line in item.dialogue
        ]
        if dialogue_lines:
            # 说话人校验：未登记说话人只告警不丢弃（后续配音阶段同口径兜底），
            # 绝不静默把台词记到第一个角色名下。
            warn_unknown_speakers(
                item.dialogue,
                characters,
                context=f"storyboard shot={shot_id}",
            )
        shots.append(
            {
                "shot_id": shot_id,
                "scene_number": scene_number,
                "shot_type": item.shot_type,
                "scene_description": item.scene_description or "角色推进剧情",
                "characters_in_scene": item.characters_in_scene or [default_character],
                "character_action": item.character_action,
                "dialogue": dialogue_lines,
                "camera_angle": item.camera_angle,
                "camera_movement": item.camera_movement,
                "emotion": item.emotion,
                "duration": item.duration,
                "estimated_speech_ms": estimate_speech_ms(dialogue_plain_text(item.dialogue)),
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


def _revision_notes(state: AgentState) -> str:
    items = [str(item.get("instruction") or item) for item in state.get("prompt_revisions") or [] if item]
    revision = str(state.get("prompt_revision") or "").strip()
    if revision:
        items.append(revision)
    return "\n".join(f"- {item}" for item in items[-3:])


def _load_system_prompt() -> str:
    return (
        "你是专业漫剧分镜师。根据剧本场景输出可执行分镜 JSON，不要输出 Markdown。"
        "每个镜头的对白是结构化数组，每句台词都必须写明说话人 speaker。"
    )


def _build_task_prompt(
    script_scenes: list,
    characters: list,
    style: str,
    platform: str,
    target_duration: float,
    duration_capability,
    revision_notes: str = "",
) -> str:
    return f"""
剧本场景：
{json.dumps(script_scenes, ensure_ascii=False, indent=2)}

角色：
{json.dumps(characters, ensure_ascii=False, indent=2)}

项目实际生效画风（必须严格遵守）：{style}
平台：{platform}
目标总时长：{target_duration:g} 秒
视频 Provider 时长能力：{duration_capability.describe()}（fixed_duration={duration_capability.fixed_duration}，min_duration={duration_capability.min_duration:g}，max_duration={duration_capability.max_duration:g}）
输出 JSON：
{{
  "shots": [
    {{
      "scene_number": 1,
      "shot_type": "wide/medium/close-up/extreme_close",
      "scene_description": "画面描述",
      "characters_in_scene": ["角色名"],
      "character_action": "动作",
      "dialogue": [
        {{
          "speaker": "说话角色名",
          "line": "台词原文",
          "emotion": "neutral/happy/shy/sad/angry/surprised",
          "action": "说话时的动作或表情",
          "start_ms": 0,
          "end_ms": 1500
        }}
      ],
      "camera_angle": "正面/侧面/俯视/仰视",
      "camera_movement": "静止/缓慢推进/平移/跟随",
      "emotion": "neutral/happy/shy/sad/angry/surprised",
      "duration": 3.5,
      "estimated_speech_ms": 1800,
      "transition": "cut/fade/dissolve",
      "visual_notes": "画面注意事项"
    }}
  ]
}}

镜头设计硬性规则：
1. 每个镜头只包含一个主体、一个主要动作和一个镜头运动；不要在一个镜头里堆叠多个动作节拍。
2. 复杂动作（追逐、打斗、转身走位等）必须拆成多个连续短镜头，靠剪辑衔接，不能在固定时长后截断。
3. 所有镜头 duration 必须严格落在 Provider 时长能力内；不要生成 Provider 无法生成或会触发 FFmpeg 裁短的镜头。
4. estimated_speech_ms 必须覆盖全部 speaker/line 对白的预计 TTS 时长；对白必须能完整放入镜头可用时间。
5. 所有镜头 duration 之和必须接近目标总时长；请合理分配动作节拍、对白停顿和镜头数量。
6. 关键情绪/反转镜头可以标记 visual_notes 建议"生成 2 个候选供挑选"。

对白硬性规则：
1. dialogue 是数组，镜头内每句台词单独一项，按说话顺序排列；没有台词时输出空数组 []。
2. 每句台词的 speaker 必须写明说话角色的名字，且必须与「角色」列表或 characters_in_scene 中的名字完全一致；旁白/画外音使用 "旁白"。
3. start_ms / end_ms 是该句台词相对镜头开头的毫秒时间估计，必须按说话顺序递增且落在 0 到 duration*1000 之间。
4. 绝不允许把多个角色的台词合并成一句或省略 speaker；配音音色完全由 speaker 决定。

{("Critic 修订要求：" + revision_notes) if revision_notes else ""}

风格与一致性规则：同一 scene_number 属于同场景组。不得让同场景镜头出现昼夜、冷暖、光源方向、道具位置、人物站位和180度轴线跳变；Agent 会在后续生成阶段强制以场景组基准图、角色三视图和上一镜头末尾帧覆盖单镜头自定义参数。
"""
