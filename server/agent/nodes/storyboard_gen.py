import json
import logging
import math

from agent.contracts import QualityProfileName, default_quality_profile
from agent.output_schemas import LLMOutputError, ShotOutput, parse_storyboard_output
from agent.state import AgentState
from config import settings
from services.consistency_service import ConsistencyService
from services.llm_service import LLMService, large_json_max_tokens
from services.prompts import STORYBOARD_JSON_CONTRACT, STORYBOARD_SYSTEM_PROMPT, resolve_json_system_prompt
from services.shot_dialogue import dialogue_plain_text, warn_unknown_speakers
from services.story_timing import (
    COMPLEX_ACTION_MAX_SECONDS,
    EXECUTION_PLAN_PROFILE_KEY,
    ShotExecutionPlan,
    StoryTimingPlan,
    estimate_speech_ms,
    normalize_action_beats,
    provider_duration_capability,
    split_shot,
)

llm_service = LLMService()
consistency_service = ConsistencyService()
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
    # 分镜 JSON（12 镜 × 动作节拍/对白/机位）同样可能超过端点默认 4096 输出上限，
    # 与剧本解析共用大 JSON 输出额度；截断会被 llm_service 明确识别并上抛。
    prefer_fallback = str((state.get("provider_switch") or {}).get("storyboard_design") or "") == "script_fallback"
    # Skill 方案快照决定系统提示词：自定义时追加固定 JSON 契约，为空时回落
    # 与历史硬编码逐字一致的默认提示词。节点不回读全局配置文件。
    system_prompt = _load_system_prompt(state.get("skill_config"))
    try:
        result = await llm_service.call_json(
            system_prompt,
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
            max_tokens=large_json_max_tokens(),
            prefer_fallback=prefer_fallback,
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

    shots = _ensure_one_action_beat_per_shot(_build_shots(parsed.shots, state), duration_capability)
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
            for beat in normalize_action_beats(
                item.get("action_beats"),
                fallback_text=item.get("character_action"),
            )
        ]
    shots = _resolve_continuity_modes(shots)
    shots = _attach_execution_plans(shots, duration_capability, state.get("quality_profile"))
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
                "action_beats": [beat.model_dump() for beat in item.action_beats],
                "gaze_direction": item.gaze_direction,
                "screen_axis": item.screen_axis,
                "action_entry_state": item.action_entry_state,
                "action_exit_state": item.action_exit_state,
                "dialogue": dialogue_lines,
                "camera_angle": item.camera_angle,
                "camera_movement": item.camera_movement,
                "emotion": item.emotion,
                "duration": item.duration,
                "estimated_speech_ms": estimate_speech_ms(dialogue_plain_text(item.dialogue)),
                "transition": item.transition,
                "continuity_mode": item.continuity_mode,
                "continuity_mode_source": "storyboard" if item.continuity_mode else "",
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


def _attach_execution_plans(
    shots: list[dict],
    duration_capability,
    quality_profile: QualityProfileName | str | None,
) -> list[dict]:
    """为每个镜头生成初始统一执行计划（视频/字幕/合成的单一事实源）。

    计划在分镜阶段先落一份可执行基线：故事时长、Provider 生成时长、连续性
    模式与质量档位决定的候选数/恢复预算；TTS 实测与视频路由会在后续阶段
    按同一 schema 覆写对白时间轴与能力清单，不会退回旧字段推导。
    """

    strategy = default_quality_profile(quality_profile)
    for shot in shots:
        shot[EXECUTION_PLAN_PROFILE_KEY] = ShotExecutionPlan.derive(
            shot,
            provider=duration_capability,
            candidate_count=strategy.candidate_count,
            recovery_budget=strategy.max_recovery_attempts,
        ).to_dict()
    return shots


def _ensure_one_action_beat_per_shot(shots: list[dict], duration_capability=None) -> list[dict]:
    """复杂/多节拍动作自动拆镜；普通镜头也显式落一份 action_beats。"""

    output: list[dict] = []
    used = {str(item.get("shot_id") or "") for item in shots}
    for shot in shots:
        beats = normalize_action_beats(shot.get("action_beats"), fallback_text=shot.get("character_action"))
        first_frame_only = bool(getattr(duration_capability, "first_frame_only", False))
        duration_parts = 1
        if first_frame_only:
            duration_parts = max(1, math.ceil(float(shot.get("duration") or 0) / COMPLEX_ACTION_MAX_SECONDS))
        needs_split = len(beats) > 1 or any(item.complex_motion for item in beats) or duration_parts > 1
        if not needs_split:
            shot["action_beats"] = [item.to_dict() for item in beats] or [
                {"text": "", "phase": "continuation", "complex_motion": False, "entry_state": "", "exit_state": ""}
            ]
            output.append(shot)
            continue
        split_reasons = ["complex_motion" if any(item.complex_motion for item in beats) else "action_beat_count"]
        if duration_parts > 1:
            split_reasons.append("first_frame_duration_limit" if first_frame_only else "duration_over_provider_limit")
        parts = split_shot(shot, max(len(beats), duration_parts), existing_ids=used, reason="+".join(split_reasons))
        used.update(item["shot_id"] for item in parts)
        if first_frame_only:
            for part in parts:
                timing = dict(part.get("timing") or {})
                timing.update({"video_mode": "first_frame_i2v", "short_shot": True})
                part["timing"] = timing
        output.extend(parts)
    return output


def _resolve_continuity_modes(shots: list[dict]) -> list[dict]:
    """把分镜显式模式与确定性规则兜底收敛为规范值并落到镜头。"""

    resolved: list[dict] = []
    previous: dict | None = None
    for item in shots:
        current = dict(item)
        decision = consistency_service.resolve_continuity(current, previous)
        current["continuity_mode"] = decision["continuity_mode"]
        current["continuity_mode_source"] = decision["continuity_mode_source"]
        resolved.append(current)
        previous = current
    return resolved


def _revision_notes(state: AgentState) -> str:
    items = [str(item.get("instruction") or item) for item in state.get("prompt_revisions") or [] if item]
    revision = str(state.get("prompt_revision") or "").strip()
    if revision:
        items.append(revision)
    return "\n".join(f"- {item}" for item in items[-3:])


def _load_system_prompt(skill_config: dict | None = None) -> str:
    """解析分镜生成的系统提示词。

    Skill 方案自定义了 storyboard_agent.system_prompt 时使用用户提示词并
    追加固定 JSON 输出契约；为空时返回与历史硬编码逐字一致的默认提示词。
    """
    return resolve_json_system_prompt(
        skill_config,
        "storyboard_agent",
        fallback=STORYBOARD_SYSTEM_PROMPT,
        contract=STORYBOARD_JSON_CONTRACT,
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
      "action_beats": [
        {{
          "text": "本镜头唯一主要动作节拍",
          "phase": "preparation/action/reaction/continuation",
          "complex_motion": false,
          "entry_state": "角色身体、视线、朝向及动作进入状态",
          "exit_state": "角色身体、视线、朝向及动作退出状态"
        }}
      ],
      "gaze_direction": "视线方向与注视对象",
      "screen_axis": "人物站位和180度轴线关系",
      "action_entry_state": "动作进入前姿态、重心、朝向",
      "action_exit_state": "动作完成后姿态、重心、朝向",
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
      "continuity_mode": "independent/same_scene/continuous_action/reverse_shot/scene_transition",
      "visual_notes": "画面注意事项"
    }}
  ]
}}

镜头设计硬性规则：
1. action_beats 必须显式输出；每个镜头默认只有一个主要动作节拍，character_action 与唯一 action_beat.text 保持一致。
2. 复杂动作（追逐、打斗、奔跑、翻滚、连续转身、转身走位等）必须拆成 preparation（准备）、action（动作）、reaction（反应）连续短镜头，靠剪辑衔接，不能在固定时长后截断。
3. 拆镜必须逐镜保留 characters_in_scene、scene_description、gaze_direction、screen_axis、action_entry_state、action_exit_state；相邻镜头的上一镜 exit_state 与下一镜 entry_state 必须连续。
4. first_frame_only Provider 使用短镜头和当前审核首帧；不要依赖上一镜画面替代首帧。
5. 所有镜头 duration 必须严格落在 Provider 时长能力内；不要生成 Provider 无法生成或会触发 FFmpeg 裁短的镜头。
6. estimated_speech_ms 必须覆盖全部 speaker/line 对白的预计 TTS 时长；对白必须能完整放入镜头可用时间。
7. 所有镜头 duration 之和必须接近目标总时长；请合理分配动作节拍、对白停顿和镜头数量。
8. 关键情绪/反转镜头可以标记 visual_notes 建议"生成 2 个候选供挑选"。

对白硬性规则：
1. dialogue 是数组，镜头内每句台词单独一项，按说话顺序排列；没有台词时输出空数组 []。
2. 每句台词的 speaker 必须写明说话角色的名字，且必须与「角色」列表或 characters_in_scene 中的名字完全一致；旁白/画外音使用 "旁白"。
3. start_ms / end_ms 是该句台词相对镜头开头的毫秒时间估计，必须按说话顺序递增且落在 0 到 duration*1000 之间。
4. 绝不允许把多个角色的台词合并成一句或省略 speaker；配音音色完全由 speaker 决定。

镜头连续性硬性规则：
1. continuity_mode 只能取 independent、same_scene、continuous_action、reverse_shot、scene_transition。
2. continuous_action 只用于同一动作节拍跨镜接续；其余模式绝不继承上一镜具体画面。
3. same_scene 只继承场景与角色身份；reverse_shot 保持视线和180度轴线；scene_transition 用于跨场景；independent 用于无须继承上一镜的镜头。
4. 可省略 continuity_mode，Agent 会按相邻镜头的场景、反打提示与动作连续性确定性兜底。

{("Critic 修订要求：" + revision_notes) if revision_notes else ""}

风格与一致性规则：同一 scene_number 属于同场景组。不得让同场景镜头出现昼夜、冷暖、光源方向、道具位置、人物站位和180度轴线跳变；Agent 会以场景组基准图和角色三视图保持场景/身份一致，只有 continuous_action 才允许自动使用上一镜 last_frame_path。
"""
