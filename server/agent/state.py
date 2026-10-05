"""LangGraph AgentState。

保留旧字段供手动/自动路由复用，同时加入生成 Agent 的阶段状态、检查点、
逐镜头 fan-out 结果、Critic/Reviewer 报告、恢复决策和人工卡点。
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict


def _merge_dicts(left: dict | None, right: dict | None) -> dict:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def _merge_attempts(left: dict | None, right: dict | None) -> dict:
    merged = dict(left or {})
    for key, value in (right or {}).items():
        merged[key] = max(int(merged.get(key, 0)), int(value or 0))
    return merged


class CharacterCard(TypedDict):
    name: str
    appearance: dict
    personality: str
    visual_prompt: str
    negative_prompt: str
    voice_id: str
    key_features: list[str]
    emotion_variants: dict[str, str]
    seed: int


class Shot(TypedDict):
    shot_id: str
    shot_type: Literal["wide", "medium", "close-up", "extreme_close"]
    scene_description: str
    characters_in_scene: list[str]
    character_action: str
    dialogue: str
    camera_angle: str
    camera_movement: str
    emotion: str
    duration: float
    transition: str
    image_path: str
    audio_path: str
    confirmed: bool
    status: Literal["pending", "generating", "done", "failed", "needs_review"]
    version: int
    seed: int
    visual_notes: str


class StyleParams(TypedDict):
    style_id: str
    style_name: str
    prompt_prefix: str
    negative_prefix: str
    color_palette: dict
    camera_preferences: dict


class ExecutionIdentity(TypedDict):
    """任何运行/阶段/镜头状态都必须携带的稳定身份与输入指纹。"""

    project_id: str
    shot_version: int
    run_id: str
    input_fingerprint: str


class AgentState(ExecutionIdentity, TypedDict, total=False):
    # 项目标识
    project_id: str
    shot_version: int
    run_id: str

    # 用户输入
    user_input: str
    input_type: Literal["text", "file", "ip"]
    uploaded_file_path: str
    file_type: str

    # 脚本解析结果
    script_title: str
    genre: str
    style_suggestion: str
    characters: list[CharacterCard]
    raw_script: str
    script_scenes: list[dict]
    logic_issues: list[dict]

    # 分镜
    shots: Annotated[list[Shot], add]

    # 风格
    style: str
    style_params: StyleParams
    requested_style: str
    effective_style: str
    style_source: str
    skill_config: dict
    skill_prompt_append: str

    # 渲染参数
    output_format: Literal["9:16", "16:9", "1:1"]
    resolution: str
    platform: Literal["douyin", "kuaishou", "bilibili", "custom"]
    target_duration: int

    # 输出
    video_path: str
    output_path: str
    final_feedback: str
    final_report: dict

    # 流程控制
    current_step: str
    errors: Annotated[list[str], add]
    human_feedback: str
    needs_human_review: bool
    human_reason: str
    storyboard_confirmed: bool
    consistency_report: dict
    affected_shot_ids: list[str]
    audio_mode: Literal["tts", "native", "auto"]
    audio_execution_plan: dict
    external_tts_required: bool
    human_gate_policy: Literal["disabled", "manual"]
    final_recovery_target: str

    # 生成 Agent 运行时
    mode: Literal["manual", "auto"]
    resume: bool
    run_status: str
    current_stage: str
    stage_history: Annotated[list[dict], add]
    quality_profile: str
    quality_threshold: float
    initial_state: dict
    input_fingerprint: str
    version_snapshot: dict[str, int]
    shot_versions: dict[str, int]
    changed_shot_ids: list[str]
    budget_snapshot: dict
    provider_profiles: dict[str, list[dict]]
    stage_status: Annotated[dict, _merge_dicts]
    stage_outputs: Annotated[dict, _merge_dicts]
    critiques: Annotated[list[dict], add]
    decision_traces: Annotated[list[dict], add]
    shot_artifacts: Annotated[list[dict], add]
    # 稳定 fan-in 聚合契约；fan-in 节点直接返回这些分组，避免调用方猜测状态。
    successes: list[dict]
    failures: list[dict]
    degraded: list[dict]
    skipped: list[dict]
    pending: list[dict]
    artifacts: list[dict]
    successful_shot_ids: list[str]
    failed_shot_ids: list[str]
    degraded_shot_ids: list[str]
    recovery_attempts: Annotated[dict, _merge_attempts]
    recovery_history: Annotated[list[dict], add]
    recovery_plan: dict
    recovery_candidates: list[dict]
    selected_strategy: str
    degraded_published: bool
    degraded_reason: str
    # 视觉质量未验证（pending）：无真实视觉模型时按结构+技术门禁自动继续的显式留痕。
    visual_quality_pending: bool
    visual_pending_reason: str
    visual_pending_stages: Annotated[list[str], add]
    visual_pending_shot_ids: Annotated[list[str], add]
    pending_recovery_stage: str
    pending_recovery_target: str
    pending_shot_ids: list[str]
    split_recovery_shot_ids: list[str]
    provider_switch: dict
    prompt_revisions: Annotated[list[dict], add]
    manual_interventions: Annotated[list[dict], add]
    checkpoint_version: int
    checkpoint_key: str

    # 记忆上下文（从 RAG 和记忆系统注入）
    rag_context: list[str]
    narrative_context: dict
    generation_preferences: dict


__all__ = ["AgentState", "CharacterCard", "ExecutionIdentity", "Shot", "StyleParams"]
