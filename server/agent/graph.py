"""可分析、决策、修改、恢复的 LangGraph 生成 Agent。

旧的五节点线性路径已由兼容入口替代；本版把流程拆成十个明确阶段，并在阶段间插入
Critic/Reviewer 与恢复决策。逐镜头生成使用 fan-out/fan-in：一个镜头失败只会
进入自己的恢复队列，已经成功的结果保留且不会重算。

生成、审核和恢复节点通过 ``agent.checkpoints.CheckpointStore`` 保存输入指纹、输出指纹、
Shot.version、候选、评分、失败分类和 DecisionTrace，因此可幂等恢复、任务续跑、
用户中途修改检测和局部重算。可视化结构来自本文件的 GRAPH_NODE_META + build_graph。
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from services.error_reporter import ERROR_PIPELINE, log_failure, new_error_id, redact

from .checkpoints import CheckpointStore, fingerprint
from .contracts import (
    STAGE_ORDER,
    FailureKind,
    FailureRecord,
    PromptPatch,
    QualityProfileName,
    RecoveryStrategy,
    RunStatus,
    StageName,
    StageStatus,
    default_quality_profile,
    ensure_stage_transition,
    stage_contract,
)
from .critic import (
    critique_assets,
    critique_audio,
    critique_compose,
    critique_director,
    critique_final,
    critique_images,
    critique_llm_failure,
    critique_storyboard,
    critique_videos,
    extract_final_report,
)
from .decision import TRANSIENT_FAILURES, choose_recovery, classify_failure, provider_profiles
from .shot_work import (
    fan_in_shot_results,
    generate_audio_shot,
    generate_storyboard_shot,
    generate_video_shot,
    run_shot_fanout,
)
from .state import AgentState

logger = logging.getLogger(__name__)

# 音频必须在视频之前完成，视频阶段只消费已准备好的音频（或 native 音频能力）。
GRAPH_STAGE_ORDER: tuple[str, ...] = tuple(stage.value for stage in STAGE_ORDER)

GRAPH_STAGE_NODE_NAMES: dict[str, dict[str, str]] = {
    StageName.DIRECTOR_PLANNING.value: {
        "process": "director_planning",
        "critic": "director_review",
        "decision": "director_decision",
        "recovery": "director_recovery",
    },
    StageName.STORYBOARD_DESIGN.value: {
        "process": "storyboard_design",
        "critic": "storyboard_review",
        "decision": "storyboard_decision",
        "recovery": "storyboard_recovery",
    },
    StageName.ASSET_PREPARATION.value: {
        "process": "asset_preparation",
        "critic": "asset_review",
        "decision": "asset_decision",
        "recovery": "asset_recovery",
    },
    StageName.IMAGE_GENERATION.value: {
        "process": "image_generation",
        "critic": "image_review",
        "decision": "image_decision",
        "recovery": "image_recovery",
    },
    StageName.QUALITY_REVIEW.value: {
        "process": "quality_review",
        "critic": "quality_critic",
        "decision": "quality_decision",
        "recovery": "quality_recovery",
    },
    StageName.AUDIO_PRODUCTION.value: {
        "process": "audio_production",
        "critic": "audio_review",
        "decision": "audio_decision",
        "recovery": "audio_recovery",
    },
    StageName.VIDEO_GENERATION.value: {
        "process": "video_generation",
        "critic": "video_generation_review",
        "decision": "video_generation_decision",
        "recovery": "video_generation_recovery",
    },
    StageName.VIDEO_REVIEW.value: {
        "process": "video_review",
        "critic": "video_critic",
        "decision": "video_decision",
        "recovery": "video_recovery",
    },
    StageName.EDIT_COMPOSITION.value: {
        "process": "edit_composition",
        "critic": "edit_review",
        "decision": "edit_decision",
        "recovery": "edit_recovery",
    },
    StageName.FINAL_REVIEW.value: {
        "process": "final_review",
        "critic": "final_critic",
        "decision": "final_decision",
        "recovery": "final_recovery",
    },
}

GRAPH_NODE_META: dict[str, dict] = {
    "director_planning": {
        "label": "导演规划",
        "type": "process",
        "description": "解析剧本、人物、场景、叙事目标与预算约束，产出导演计划",
    },
    "director_review": {
        "label": "导演 Critic",
        "type": "critic",
        "description": "检查角色/场景覆盖和逻辑问题，提出具体修改",
    },
    "director_decision": {
        "label": "导演决策",
        "type": "decision",
        "description": "按失败分类、Provider 能力、成本和剩余预算选择恢复策略",
    },
    "director_recovery": {
        "label": "导演恢复",
        "type": "recovery",
        "description": "只修改受影响输入并回到导演规划局部重算",
    },
    "storyboard_design": {
        "label": "分镜设计",
        "type": "process",
        "description": "生成镜头并自动拆合动作节拍、对白和时长",
    },
    "storyboard_review": {
        "label": "分镜 Critic",
        "type": "critic",
        "description": "检查镜头规则、对白密度、连续性和可执行性",
    },
    "storyboard_decision": {"label": "分镜决策", "type": "decision", "description": "选择拆分、修改、合并或局部恢复"},
    "storyboard_recovery": {
        "label": "分镜恢复",
        "type": "recovery",
        "description": "把 Critic 修改写回受影响镜头/Prompt 后局部重算",
    },
    "asset_preparation": {
        "label": "素材准备",
        "type": "process",
        "description": "生成角色三视图、场景基准图和连续性参考，并记录能力限制",
    },
    "asset_review": {"label": "素材 Critic", "type": "critic", "description": "检查素材覆盖、参考图兼容性与版本一致性"},
    "asset_decision": {"label": "素材决策", "type": "decision", "description": "替换参考、切换 Provider 或局部恢复"},
    "asset_recovery": {"label": "素材恢复", "type": "recovery", "description": "只补生成缺失参考，不重跑已成功素材"},
    "image_generation": {
        "label": "图像生成",
        "type": "process",
        "description": "按镜头版本逐镜头生成故事板，失败镜头独立进入恢复队列",
    },
    "image_generation_fan_out": {
        "label": "图像 fan-out（兼容）",
        "type": "process",
        "description": "兼容入口，与图像生成阶段执行同一逐镜头 worker",
    },
    "image_generation_fan_in": {
        "label": "图像 fan-in",
        "type": "process",
        "description": "聚合成功、失败、降级和跳过镜头，保留局部成功",
    },
    "image_review": {"label": "图像 Critic", "type": "critic", "description": "检查图片结构、失败镜头和候选完整性"},
    "image_decision": {
        "label": "图像决策",
        "type": "decision",
        "description": "只把失败/低分镜头送入恢复，成功镜头直接进入质量审核",
    },
    "image_recovery": {
        "label": "图像恢复",
        "type": "recovery",
        "description": "改 Prompt、换参考、换 Provider 或只补拍失败镜头",
    },
    "quality_review": {
        "label": "质量审核",
        "type": "process",
        "description": "执行故事板结构与真实质量门禁，产出逐镜头审核记录",
    },
    "quality_critic": {
        "label": "质量 Critic",
        "type": "critic",
        "description": "复核质量报告、证据和修改建议，避免把未检测维度伪装成通过",
    },
    "quality_decision": {
        "label": "质量决策",
        "type": "decision",
        "description": "确认分镜并进入音频，或只恢复失败/低分镜头",
    },
    "quality_recovery": {
        "label": "质量恢复",
        "type": "recovery",
        "description": "按失败镜头局部补拍或回分镜修改，不重跑成功镜头",
    },
    "audio_production": {
        "label": "音频制作",
        "type": "process",
        "description": "外部 TTS 镜头逐镜头生成/复用配音；native audio 镜头跳过外部 TTS",
    },
    "audio_review": {
        "label": "音频检查",
        "type": "critic",
        "description": "检查对白长度、TTS 产物、native audio 跳过依据和混音准备度",
    },
    "audio_decision": {
        "label": "音频决策",
        "type": "decision",
        "description": "音频检查通过后才允许视频生成，失败镜头只局部重配音",
    },
    "audio_recovery": {
        "label": "音频恢复",
        "type": "recovery",
        "description": "拆句、局部重配音或切换 Provider，保留成功音频",
    },
    "video_generation": {
        "label": "视频生成",
        "type": "process",
        "description": "逐镜头生成视频候选，失败镜头独立重试/补拍",
    },
    "video_generation_fan_out": {
        "label": "视频 fan-out（兼容）",
        "type": "process",
        "description": "兼容入口，与视频生成阶段执行同一逐镜头 worker",
    },
    "video_generation_fan_in": {
        "label": "视频 fan-in",
        "type": "process",
        "description": "聚合视频结果，不让一个失败镜头使成功镜头失效",
    },
    "video_generation_review": {
        "label": "视频生成 Critic",
        "type": "critic",
        "description": "检查生成产物、候选选择和逐镜头失败，不覆盖成功视频",
    },
    "video_generation_decision": {
        "label": "视频生成决策",
        "type": "decision",
        "description": "只恢复失败视频，成功视频进入独立视频检查阶段",
    },
    "video_generation_recovery": {
        "label": "视频生成恢复",
        "type": "recovery",
        "description": "只补拍失败镜头并复用成功视频",
    },
    "video_review": {
        "label": "视频检查",
        "type": "process",
        "description": "三维度检查：structural_validity（可播放/视频轨）、technical_quality（时长/分辨率/黑帧/冻结/音画/尾帧）、visual_quality_pending（未接入视觉模型，视觉质量待审）",
    },
    "video_critic": {
        "label": "视频检查 Critic",
        "type": "critic",
        "description": "复核视频质量报告与证据，保持未检测视觉维度为待审",
    },
    "video_decision": {
        "label": "视频决策",
        "type": "decision",
        "description": "视频检查通过后进入剪辑合成，失败镜头只局部补拍",
    },
    "video_recovery": {
        "label": "视频恢复",
        "type": "recovery",
        "description": "改 Prompt、换 Provider、降分辨率或只补拍失败视频",
    },
    "edit_composition": {
        "label": "剪辑合成",
        "type": "process",
        "description": "按版本校验后的镜头清单合成成片，支持明确降级状态",
    },
    "compose": {
        "label": "剪辑合成（兼容）",
        "type": "process",
        "description": "兼容入口，与剪辑合成阶段执行同一渲染任务",
    },
    "edit_review": {
        "label": "剪辑 Critic",
        "type": "critic",
        "description": "检查镜头完整率、音画同步、时长和成片可播放性",
    },
    "edit_decision": {
        "label": "剪辑决策",
        "type": "decision",
        "description": "成片可复审时进入 final_review，否则只修复剪辑问题",
    },
    "edit_recovery": {
        "label": "剪辑恢复",
        "type": "recovery",
        "description": "重新合成或只补拍缺失镜头，不重算完整成功链路",
    },
    "final_review": {
        "label": "成片复审",
        "type": "process",
        "description": "从叙事、视觉、音频、节奏和反馈层面复审成片",
    },
    "final_critic": {
        "label": "成片 Critic",
        "type": "critic",
        "description": "把成片反馈拆成图像、音频、视频或剪辑的具体修改",
    },
    "final_decision": {
        "label": "成片决策",
        "type": "decision",
        "description": "通过则结束，否则路由到明确的局部重算阶段",
    },
    "final_recovery": {
        "label": "成片恢复",
        "type": "recovery",
        "description": "反馈可路由回图像、音频、视频或剪辑，不重跑已成功镜头",
    },
    "auto_abort": {
        "label": "自动终止",
        "type": "output",
        "description": "自动模式恢复耗尽且无可用结果时记录失败并结束，绝不进入人工节点",
    },
    "degraded_publish": {
        "label": "降级发布",
        "type": "output",
        "description": "自动模式恢复无法继续但存在可用部分结果：按降级结果结束并保留失败清单",
    },
    "human_gate": {
        "label": "人工审核",
        "type": "human",
        "description": "仅 manual 模式且显式 human_gate_policy=manual 时允许进入",
    },
}


def build_graph() -> StateGraph:
    """只编排阶段节点和条件路由；业务生成由 route/service worker 执行。"""

    graph = StateGraph(AgentState)
    for names in GRAPH_STAGE_NODE_NAMES.values():
        for node_name in names.values():
            graph.add_node(node_name, _NODE_FUNCTIONS[node_name])
    for node_name in (
        "image_generation_fan_in",
        "image_generation_fan_out",
        "video_generation_fan_in",
        "video_generation_fan_out",
        "compose",
        "auto_abort",
        "degraded_publish",
        "human_gate",
    ):
        graph.add_node(node_name, _NODE_FUNCTIONS[node_name])

    graph.add_edge(START, "director_planning")

    # 每个主要阶段固定 process -> critic/reviewer -> decision -> recovery。
    # failed=terminal_failure；degraded=degraded_publish（存在可用部分结果时优先于终止）。
    graph.add_edge("director_planning", "director_review")
    graph.add_edge("director_review", "director_decision")
    graph.add_conditional_edges(
        "director_decision",
        _route_director_decision,
        {
            "next": "storyboard_design",
            "recover": "director_recovery",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )
    graph.add_conditional_edges(
        "director_recovery",
        _route_director_recovery,
        {"retry": "director_planning", "failed": "auto_abort", "degraded": "degraded_publish", "human": "human_gate"},
    )

    graph.add_edge("storyboard_design", "storyboard_review")
    graph.add_edge("storyboard_review", "storyboard_decision")
    graph.add_conditional_edges(
        "storyboard_decision",
        _route_storyboard_decision,
        {
            "next": "asset_preparation",
            "recover": "storyboard_recovery",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )
    graph.add_conditional_edges(
        "storyboard_recovery",
        _route_storyboard_recovery,
        {"retry": "storyboard_design", "failed": "auto_abort", "degraded": "degraded_publish", "human": "human_gate"},
    )

    graph.add_edge("asset_preparation", "asset_review")
    graph.add_edge("asset_review", "asset_decision")
    graph.add_conditional_edges(
        "asset_decision",
        _route_asset_decision,
        {
            "next": "image_generation",
            "recover": "asset_recovery",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )
    graph.add_conditional_edges(
        "asset_recovery",
        _route_asset_recovery,
        {
            "retry": "asset_preparation",
            "storyboard": "storyboard_design",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )

    graph.add_edge("image_generation", "image_generation_fan_in")
    graph.add_edge("image_generation_fan_out", "image_generation_fan_in")
    graph.add_edge("image_generation_fan_in", "image_review")
    graph.add_edge("image_review", "image_decision")
    graph.add_conditional_edges(
        "image_decision",
        _route_image_decision,
        {
            "next": "quality_review",
            "recover": "image_recovery",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )
    graph.add_conditional_edges(
        "image_recovery",
        _route_image_recovery,
        {
            "retry": "image_generation",
            "storyboard": "storyboard_design",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )

    graph.add_edge("quality_review", "quality_critic")
    graph.add_edge("quality_critic", "quality_decision")
    graph.add_conditional_edges(
        "quality_decision",
        _route_quality_decision,
        {
            "next": "audio_production",
            "recover": "quality_recovery",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )
    graph.add_conditional_edges(
        "quality_recovery",
        _route_quality_recovery,
        {
            "retry": "image_generation",
            "storyboard": "storyboard_design",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )

    # 外部 TTS：分镜确认 -> 音频制作 -> 音频检查 -> 视频生成。
    graph.add_edge("audio_production", "audio_review")
    graph.add_edge("audio_review", "audio_decision")
    graph.add_conditional_edges(
        "audio_decision",
        _route_audio_decision,
        {
            "next": "video_generation",
            "recover": "audio_recovery",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )
    graph.add_conditional_edges(
        "audio_recovery",
        _route_audio_recovery,
        {
            "retry": "audio_production",
            "image": "image_generation",
            "storyboard": "storyboard_design",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )

    graph.add_edge("video_generation", "video_generation_fan_in")
    graph.add_edge("video_generation_fan_out", "video_generation_fan_in")
    graph.add_edge("video_generation_fan_in", "video_generation_review")
    graph.add_edge("video_generation_review", "video_generation_decision")
    graph.add_conditional_edges(
        "video_generation_decision",
        _route_video_generation_decision,
        {
            "next": "video_review",
            "recover": "video_generation_recovery",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )
    graph.add_conditional_edges(
        "video_generation_recovery",
        _route_video_generation_recovery,
        {
            "retry": "video_generation",
            "image": "image_generation",
            "audio": "audio_production",
            "storyboard": "storyboard_design",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )

    graph.add_edge("video_review", "video_critic")
    graph.add_edge("video_critic", "video_decision")
    graph.add_conditional_edges(
        "video_decision",
        _route_video_decision,
        {
            "next": "edit_composition",
            "recover": "video_recovery",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )
    graph.add_conditional_edges(
        "video_recovery",
        _route_video_recovery,
        {
            "retry": "video_generation",
            "image": "image_generation",
            "audio": "audio_production",
            "storyboard": "storyboard_design",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )

    graph.add_edge("edit_composition", "edit_review")
    graph.add_edge("compose", "edit_review")
    graph.add_edge("edit_review", "edit_decision")
    graph.add_conditional_edges(
        "edit_decision",
        _route_edit_decision,
        {
            "next": "final_review",
            "recover": "edit_recovery",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )
    graph.add_conditional_edges(
        "edit_recovery",
        _route_edit_recovery,
        {
            "retry": "edit_composition",
            "image": "image_generation",
            "audio": "audio_production",
            "video": "video_generation",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )

    graph.add_edge("final_review", "final_critic")
    graph.add_edge("final_critic", "final_decision")
    graph.add_conditional_edges(
        "final_decision",
        _route_final_decision,
        {
            "done": END,
            "recover": "final_recovery",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )
    graph.add_conditional_edges(
        "final_recovery",
        _route_final_recovery,
        {
            "image": "image_generation",
            "audio": "audio_production",
            "video": "video_generation",
            "compose": "edit_composition",
            "failed": "auto_abort",
            "degraded": "degraded_publish",
            "human": "human_gate",
        },
    )

    graph.add_edge("auto_abort", END)
    graph.add_edge("degraded_publish", END)
    graph.add_edge("human_gate", END)
    return graph


# --- stage nodes ---


def _reference_gate(
    project_id: str, *, allow_degraded: bool = False, allow_needs_review: bool = True
) -> dict[str, Any]:
    from db import SessionLocal
    from services.reference_readiness_service import ensure_generation_gate

    db = SessionLocal()
    try:
        return ensure_generation_gate(
            db, project_id, allow_degraded=allow_degraded, allow_needs_review=allow_needs_review
        )
    finally:
        db.close()


def _reference_gate_for_state(state: AgentState, *, allow_degraded: bool = False) -> dict[str, Any]:
    """auto 模式不得把项目状态写成 needs_review（等价于等待人工）。"""

    return _reference_gate(
        str(state.get("project_id") or ""),
        allow_degraded=allow_degraded,
        allow_needs_review=not _auto_mode(state),
    )


def _reference_review_update(
    project_id: str, report: dict[str, Any], state: AgentState | None = None
) -> dict[str, Any]:
    """参考素材阻断时保留可追踪原因；只有显式 manual 才写人工等待状态。"""

    affected = report.get("affected_shot_ids", [])
    manual = bool(state is not None and _human_allowed(state))
    reason = (
        "一致性参考素材未达到自动成片要求"
        + ("，等待人工审核" if manual else "，自动恢复无法继续")
        + (f"；影响{report.get('shot_range', '')}" if report.get("shot_range") else "")
    )
    update = {
        "consistency_report": report,
        "affected_shot_ids": affected,
        "stage_status": {
            StageName.ASSET_PREPARATION.value: StageStatus.DEGRADED.value if manual else StageStatus.FAILED.value
        },
        "current_step": "asset_review",
    }
    if manual:
        update.update({"needs_human_review": True, "human_reason": reason})
    else:
        update["recovery_plan"] = {
            "stage": StageName.ASSET_PREPARATION.value,
            "reason": reason,
            "affected_shot_ids": affected,
        }
    return update


async def _director_planning(state: AgentState) -> dict:
    project_id, store, base = _stage_context(state, StageName.DIRECTOR_PLANNING)
    if store.stage_is_reusable(StageName.DIRECTOR_PLANNING.value, base["input_fingerprint"]):
        return _restore_stage(state, store, StageName.DIRECTOR_PLANNING)
    try:
        from agent.nodes import script_parser

        parsed = await script_parser.run(
            {**base["initial_state"], **state, "prompt_revisions": state.get("prompt_revisions") or []}
        )
        payload = {
            "script_title": parsed.get("script_title", ""),
            "genre": parsed.get("genre", ""),
            "style_suggestion": parsed.get("style_suggestion", ""),
            "characters": parsed.get("characters", []),
            "raw_script": parsed.get("raw_script", ""),
            "script_scenes": parsed.get("script_scenes", []),
            "logic_issues": parsed.get("logic_issues", []),
            "rag_context": parsed.get("rag_context", []),
            "requested_style": parsed.get("requested_style", ""),
            "effective_style": parsed.get("effective_style", ""),
            "style_source": parsed.get("style_source", ""),
        }
        critique = critique_director(payload)
        return _save_stage(state, store, StageName.DIRECTOR_PLANNING, base, payload, critique=critique)
    except Exception as exc:
        return _failed_stage(
            state,
            store,
            StageName.DIRECTOR_PLANNING,
            base,
            exc,
            critique=critique_llm_failure(exc, stage=StageName.DIRECTOR_PLANNING),
        )


async def _director_review(state: AgentState) -> dict:
    critique = critique_director(state)
    return {"critiques": [critique.model_dump(mode="json")], "current_step": "director_review"}


async def _director_decision(state: AgentState) -> dict:
    return _decision_node(state, StageName.DIRECTOR_PLANNING, next_target="storyboard_design")


async def _director_recovery(state: AgentState) -> dict:
    return _recovery_node(state, StageName.DIRECTOR_PLANNING, default_target="director_planning")


async def _storyboard_design(state: AgentState) -> dict:
    project_id, store, base = _stage_context(state, StageName.STORYBOARD_DESIGN)
    if store.stage_is_reusable(StageName.STORYBOARD_DESIGN.value, base["input_fingerprint"]):
        return _restore_stage(state, store, StageName.STORYBOARD_DESIGN)
    try:
        from agent.nodes import storyboard_gen

        generated = await storyboard_gen.run(
            {**base["initial_state"], **state, "prompt_revisions": state.get("prompt_revisions") or []}
        )
        payload = {"shots": generated.get("shots", []), "timing_plan": generated.get("timing_plan", {})}
        critique = critique_storyboard({"shots": payload["shots"]})
        return _save_stage(state, store, StageName.STORYBOARD_DESIGN, base, payload, critique=critique)
    except Exception as exc:
        return _failed_stage(
            state,
            store,
            StageName.STORYBOARD_DESIGN,
            base,
            exc,
            critique=critique_llm_failure(exc, stage=StageName.STORYBOARD_DESIGN),
        )


async def _storyboard_review(state: AgentState) -> dict:
    critique = critique_storyboard({"shots": state.get("shots", [])})
    return {"critiques": [critique.model_dump(mode="json")], "current_step": "storyboard_review"}


async def _storyboard_decision(state: AgentState) -> dict:
    return _decision_node(state, StageName.STORYBOARD_DESIGN, next_target="asset_preparation")


async def _storyboard_recovery(state: AgentState) -> dict:
    return _recovery_node(state, StageName.STORYBOARD_DESIGN, default_target="storyboard_design")


async def _asset_preparation(state: AgentState) -> dict:
    project_id, store, base = _stage_context(state, StageName.ASSET_PREPARATION)
    if store.stage_is_reusable(StageName.ASSET_PREPARATION.value, base["input_fingerprint"]):
        return _restore_stage(state, store, StageName.ASSET_PREPARATION)
    try:
        from api.routes import script as script_route

        working = {**base["initial_state"], **state}
        asset_project_id = _asset_project_id(project_id)
        await script_route._ensure_character_reference_images(asset_project_id, working)
        await script_route._ensure_scene_baseline_images(asset_project_id, working)
        await _persist_phase1_idempotent(project_id, working)
        reference_report = refresh_project_reference_state_for_graph(project_id, allow_needs_review=False)
        reference_supported = any(
            item.supports_reference_images
            for item in provider_profiles("image", reference_required=True)
            if item.available
        )
        critique = critique_assets(working, reference_supported=reference_supported)
        payload = {
            "characters": working.get("characters", []),
            "script_scenes": working.get("script_scenes", []),
            "shots": working.get("shots", []),
            "reference_supported": reference_supported,
            "consistency_report": reference_report,
        }
        return _save_stage(state, store, StageName.ASSET_PREPARATION, base, payload, critique=critique)
    except Exception as exc:
        return _failed_stage(
            state,
            store,
            StageName.ASSET_PREPARATION,
            base,
            exc,
            critique=critique_assets(state, reference_supported=False),
        )


async def _asset_review(state: AgentState) -> dict:
    reference_supported = any(
        item.supports_reference_images for item in provider_profiles("image", reference_required=True) if item.available
    )
    critique = critique_assets(state, reference_supported=reference_supported)
    report = state.get("consistency_report") or _reference_gate_for_state(state)
    if report.get("blocking"):
        return {
            "critiques": [critique.model_dump(mode="json")],
            **_reference_review_update(str(state.get("project_id") or ""), report, state),
        }
    return {
        "critiques": [critique.model_dump(mode="json")],
        "consistency_report": report,
        "current_step": "asset_review",
    }


async def _asset_decision(state: AgentState) -> dict:
    report = state.get("consistency_report") or _reference_gate_for_state(state)
    if report.get("blocking"):
        return _reference_review_update(str(state.get("project_id") or ""), report, state)
    return _decision_node(state, StageName.ASSET_PREPARATION, next_target="image_generation")


async def _asset_recovery(state: AgentState) -> dict:
    return _recovery_node(state, StageName.ASSET_PREPARATION, default_target="asset_preparation")


async def _image_generation_fan_out(state: AgentState) -> dict:
    project_id, store, base = _stage_context(state, StageName.IMAGE_GENERATION)
    reference_gate = _reference_gate_for_state(state)
    if reference_gate.get("blocking"):
        return _reference_review_update(project_id, reference_gate, state)
    shot_versions = _shot_versions(project_id, state.get("pending_shot_ids"))
    if state.get("pending_shot_ids") and not set(shot_versions).issuperset(state["pending_shot_ids"]):
        return {
            **_identity_update(state, StageName.IMAGE_GENERATION, base),
            "stage_status": {StageName.IMAGE_GENERATION.value: StageStatus.FAILED.value},
            "failed_shot_ids": list(state["pending_shot_ids"]),
            "current_step": "image_generation_fan_out",
        }
    if not shot_versions:
        return {
            **_identity_update(state, StageName.IMAGE_GENERATION, base),
            "stage_status": {StageName.IMAGE_GENERATION.value: StageStatus.FAILED.value},
            "current_step": "image_generation_fan_out",
        }
    provider = str((state.get("provider_switch") or {}).get(StageName.IMAGE_GENERATION.value) or "")
    preferred_size = _preferred_image_size(state)
    worker = partial(
        generate_storyboard_shot, project_id=project_id, provider_override=provider, preferred_size=preferred_size
    )

    async def generate_one(shot_id: str, version: int) -> dict[str, Any]:
        return await worker(
            shot_id,
            version,
            seed_override=_seed_override(state, StageName.IMAGE_GENERATION, shot_id),
            recovery_revisions=_shot_revisions(state, StageName.IMAGE_GENERATION, shot_id),
        )

    result = await run_shot_fanout(
        project_id=project_id,
        shot_versions=shot_versions,
        stage=StageName.IMAGE_GENERATION,
        worker=generate_one,
        checkpoint=store,
        concurrency=3,
        reuse=not bool(state.get("pending_shot_ids")),
        run_id=base["run_id"],
        input_fingerprint=base["input_fingerprint"],
    )
    critique = critique_images(result["artifacts"])
    status = _fanout_status(result)
    row = store.save_stage(
        StageName.IMAGE_GENERATION.value,
        status=status.value,
        input_fingerprint=base["input_fingerprint"],
        output_fingerprint=fingerprint(result),
        payload={"result": result, "provider": provider, "preferred_size": preferred_size},
        critique=critique,
        shot_artifacts=result["artifacts"],
    )
    return {
        **_identity_update(state, StageName.IMAGE_GENERATION, base, row),
        "stage_status": {StageName.IMAGE_GENERATION.value: status.value},
        "stage_outputs": {StageName.IMAGE_GENERATION.value: row},
        "shot_artifacts": result["artifacts"],
        "decision_traces": [item["decision_trace"] for item in result["artifacts"] if item.get("decision_trace")],
        "successful_shot_ids": [item["shot_id"] for item in result["successes"]],
        "failed_shot_ids": [item["shot_id"] for item in result["failures"]],
        "degraded_shot_ids": [item["shot_id"] for item in result["degraded"]],
        # Recovery scope must survive image -> quality -> audio -> video.  The
        # fan-out itself is synchronous, so result["pending"] is execution
        # status, not the cross-stage recovery queue.
        "current_step": "image_generation_fan_out",
    }


async def _image_generation_fan_in(state: AgentState) -> dict:
    artifacts = _latest_artifacts(state, StageName.IMAGE_GENERATION)
    result = fan_in_shot_results(artifacts)
    return {
        **result,
        "shot_artifacts": artifacts,
        "successful_shot_ids": [item.get("shot_id") for item in result["successes"]],
        "failed_shot_ids": [item.get("shot_id") for item in result["failures"]],
        "degraded_shot_ids": [item.get("shot_id") for item in result["degraded"]],
        "skipped_shot_ids": [item.get("shot_id") for item in result["skipped"]],
        "pending_shot_ids": [item.get("shot_id") for item in result["pending"]],
        "current_step": "image_generation_fan_in",
    }


async def _image_review(state: AgentState) -> dict:
    critique = critique_images(_latest_artifacts(state, StageName.IMAGE_GENERATION))
    return {"critiques": [critique.model_dump(mode="json")], "current_step": "image_review"}


async def _image_decision(state: AgentState) -> dict:
    return _decision_node(
        state, StageName.IMAGE_GENERATION, next_target="quality_review", artifacts_stage=StageName.IMAGE_GENERATION
    )


async def _image_recovery(state: AgentState) -> dict:
    return _recovery_node(
        state, StageName.IMAGE_GENERATION, default_target="image_generation", artifacts_stage=StageName.IMAGE_GENERATION
    )


async def _quality_critic(state: AgentState) -> dict:
    reports = [item for item in state.get("critiques", []) if item.get("stage") == StageName.QUALITY_REVIEW.value]
    critique = (
        reports[-1]
        if reports
        else critique_images(_latest_artifacts(state, StageName.IMAGE_GENERATION)).model_dump(mode="json")
    )
    return {"critiques": [critique], "current_step": "quality_critic"}


async def _quality_review(state: AgentState) -> dict:
    project_id, store, base = _stage_context(state, StageName.QUALITY_REVIEW)
    if store.stage_is_reusable(StageName.QUALITY_REVIEW.value, base["input_fingerprint"]):
        return _restore_stage(state, store, StageName.QUALITY_REVIEW)
    reference_gate = _reference_gate_for_state(state)
    if reference_gate.get("blocking"):
        return _reference_review_update(str(state.get("project_id") or ""), reference_gate, state)
    gate_shot_ids = state.get("pending_shot_ids") or state.get("split_recovery_shot_ids")
    if gate_shot_ids and not set(_shot_versions(project_id)).issuperset(gate_shot_ids):
        return {
            "stage_status": {StageName.QUALITY_REVIEW.value: StageStatus.FAILED.value},
            "failed_shot_ids": list(gate_shot_ids),
            "current_step": "quality_review",
        }
    artifacts = _latest_artifacts(state, StageName.IMAGE_GENERATION)
    if gate_shot_ids:
        artifacts = [item for item in artifacts if item.get("shot_id") in gate_shot_ids]
    structural = critique_images(artifacts)
    if not structural.passed:
        return {
            **_identity_update(state, StageName.QUALITY_REVIEW, base),
            "critiques": [structural.model_dump(mode="json")],
            "stage_status": {StageName.QUALITY_REVIEW.value: StageStatus.FAILED.value},
            "current_step": "quality_review",
        }
    allow_human = _legacy_human_allowed(state)
    allow_visual_pending = _visual_pending_continue(state)
    gate_shot_ids = state.get("pending_shot_ids") or state.get("split_recovery_shot_ids")
    gate = await _run_storyboard_quality_gate(
        project_id,
        gate_shot_ids,
        allow_human=allow_human,
        allow_visual_pending=allow_visual_pending,
    )
    reviews = gate.get("reviews") or {}
    visual_pending = bool(gate.get("visual_pending"))
    passed = bool(gate.get("passed")) and not gate.get("errors")
    failed_reviews = {shot_id: review for shot_id, review in reviews.items() if not review.passed}
    overall_score = round(sum(float(review.overall_score) for review in reviews.values()) / max(1, len(reviews)), 3)
    quality_critique = {
        "stage": StageName.QUALITY_REVIEW.value,
        "passed": passed,
        "score": overall_score,
        "metrics": [
            {
                "name": "review_score",
                "value": overall_score,
                "threshold": default_quality_profile(state.get("quality_profile")).quality_threshold,
                "passed": passed,
            },
            {
                "name": "reviewed_shot_count",
                "value": len(reviews),
                "threshold": len(reviews),
                "passed": not failed_reviews,
            },
            # 结构门禁收口时视觉质量仍是 pending：这里显式暴露，不伪装成已审核。
            {
                "name": "visual_quality_pending",
                "passed": None,
                "detail": "视觉模型缺失，仅结构门禁收口；视觉质量未评估",
            },
        ]
        if visual_pending
        else [
            {
                "name": "review_score",
                "value": overall_score,
                "threshold": default_quality_profile(state.get("quality_profile")).quality_threshold,
                "passed": passed,
            },
            {
                "name": "reviewed_shot_count",
                "value": len(reviews),
                "threshold": len(reviews),
                "passed": not failed_reviews,
            },
        ],
        "issues": [
            {
                "code": "quality_review_failed",
                "severity": "error",
                "message": f"镜头 {shot_id} 未通过质量审核: {review.suggestion or '请查看质量审核记录'}",
                "shot_id": shot_id,
            }
            for shot_id, review in sorted(failed_reviews.items())
        ]
        + (
            [
                {
                    "code": "visual_quality_pending",
                    "severity": "info",
                    "message": f"故事板视觉质量待审：{gate.get('reason') or '未配置视觉模型'}；已按结构门禁自动继续",
                    "details": {
                        "stage": StageName.QUALITY_REVIEW.value,
                        "status": "pending",
                        "shot_ids": [str(item) for item in (gate.get("shot_ids") or []) if item],
                    },
                }
            ]
            if visual_pending
            else []
        ),
        "evidence": [{"kind": "metric", "name": "review_score", "value": overall_score, "passed": passed}]
        + [
            {
                "kind": "issue",
                "code": "quality_review_failed",
                "shot_id": shot_id,
                "details": {"overall_score": float(review.overall_score)},
            }
            for shot_id, review in sorted(failed_reviews.items())
        ]
        + ([{"kind": "structural_only_gate", **(gate.get("evidence") or {})}] if visual_pending else []),
        "failure_kind": FailureKind.QUALITY_BELOW_THRESHOLD.value if not passed else None,
        "recoverable": True,
        "proposed_changes": [review.suggestion for review in reviews.values() if review.suggestion],
        "affected_shot_ids": sorted(failed_reviews),
        "recommended_strategy": RecoveryStrategy.REVISE_PROMPT.value if not passed else None,
        "source": "quality_review_service",
    }
    human_allowed = _human_allowed(state)
    if passed and visual_pending:
        status = StageStatus.DEGRADED.value
    else:
        status = (
            StageStatus.SUCCEEDED.value
            if passed
            else (StageStatus.WAITING_HUMAN.value if human_allowed else StageStatus.FAILED.value)
        )
    update: dict[str, Any] = {
        **_identity_update(state, StageName.QUALITY_REVIEW, base),
        "critiques": [structural.model_dump(mode="json"), quality_critique],
        "stage_status": {StageName.QUALITY_REVIEW.value: status},
        "storyboard_confirmed": bool(passed),
        "current_step": "quality_review",
    }
    if passed:
        # 结构门禁收口时也必须确认镜头，否则下游视频阶段会被 confirmed 门禁挡住；
        # 视觉待审通过 degraded 状态与未解决风险如实暴露。
        await _confirm_storyboard_shots(project_id, artifacts)
    if visual_pending and passed:
        update.update(
            _visual_pending_update(
                state,
                stage=StageName.QUALITY_REVIEW,
                reason=str(gate.get("reason") or "未配置视觉模型"),
                shot_ids=[str(item) for item in (gate.get("shot_ids") or []) if item],
            )
        )
        update["stage_status"] = {StageName.QUALITY_REVIEW.value: StageStatus.DEGRADED.value}
    if gate.get("errors"):
        allowed_keys = {"errors", "needs_human_review", "human_reason"} if human_allowed else set()
        update.update({key: value for key, value in gate.items() if key in allowed_keys})
        if not human_allowed:
            update["recovery_plan"] = {
                "stage": StageName.QUALITY_REVIEW.value,
                "reason": gate.get("errors"),
                "affected_shot_ids": sorted(failed_reviews) or list(reviews),
            }
    quality_row = _save_quality_review_checkpoint(state, quality_critique, reviews, status=status, base=base)
    update.update(
        {
            "stage_outputs": {StageName.QUALITY_REVIEW.value: quality_row},
            **_identity_update(state, StageName.QUALITY_REVIEW, base, quality_row),
        }
    )
    return update


async def _quality_decision(state: AgentState) -> dict:
    return _decision_node(
        state, StageName.QUALITY_REVIEW, next_target="audio_production", artifacts_stage=StageName.IMAGE_GENERATION
    )


async def _quality_recovery(state: AgentState) -> dict:
    return _recovery_node(
        state, StageName.QUALITY_REVIEW, default_target="image_generation", artifacts_stage=StageName.IMAGE_GENERATION
    )


async def _video_generation_fan_out(state: AgentState) -> dict:
    project_id, store, base = _stage_context(state, StageName.VIDEO_GENERATION)
    from services.quality_review_service import quality_review_service

    storyboard_gate = quality_review_service.storyboard_gate_status(project_id, **_visual_pending_gate_kwargs(state))
    if not storyboard_gate.get("ok"):
        return {
            "stage_status": {StageName.VIDEO_GENERATION.value: StageStatus.FAILED.value},
            "recovery_plan": {
                "stage": StageName.QUALITY_REVIEW.value,
                "reason": _gate_failure_summary(storyboard_gate),
                "affected_shot_ids": [
                    str(item.get("shot_id")) for item in storyboard_gate.get("failed") or [] if item.get("shot_id")
                ],
            },
            "current_step": "video_generation",
        }
    shot_versions = _shot_versions(
        project_id, state.get("pending_shot_ids") or state.get("split_recovery_shot_ids"), require_storyboard=True
    )
    required_shots = state.get("pending_shot_ids") or state.get("split_recovery_shot_ids") or []
    if required_shots and not set(shot_versions).issuperset(required_shots):
        return {
            "stage_status": {StageName.VIDEO_GENERATION.value: StageStatus.FAILED.value},
            "failed_shot_ids": list(required_shots),
            "current_step": "video_generation_fan_out",
        }
    if not shot_versions:
        return {
            "stage_status": {StageName.VIDEO_GENERATION.value: StageStatus.FAILED.value},
            "current_step": "video_generation_fan_out",
        }
    provider = str((state.get("provider_switch") or {}).get(StageName.VIDEO_GENERATION.value) or "")
    resolution = str(
        (state.get("provider_switch") or {}).get(f"{StageName.VIDEO_GENERATION.value}:resolution")
        or state.get("resolution")
        or "720p"
    )
    # 音频阶段已经先行完成；视频 worker 只准备/复用当前版本音频，不重复调用 TTS。
    quality_profile = str(state.get("quality_profile") or QualityProfileName.STANDARD.value)
    retry_candidate = str((state.get("recovery_plan") or {}).get("retry_of_candidate_id") or "")
    worker = partial(
        generate_video_shot,
        project_id=project_id,
        provider_override=provider,
        resolution_override=resolution,
        quality_profile=quality_profile,
        retry_of_candidate_id=retry_candidate,
    )

    async def generate_one(shot_id: str, version: int) -> dict[str, Any]:
        return await worker(
            shot_id,
            version,
            seed_override=_seed_override(state, StageName.VIDEO_GENERATION, shot_id),
            recovery_revisions=_shot_revisions(state, StageName.VIDEO_GENERATION, shot_id),
        )

    result = await run_shot_fanout(
        project_id=project_id,
        shot_versions=shot_versions,
        stage=StageName.VIDEO_GENERATION,
        worker=generate_one,
        checkpoint=store,
        concurrency=2,
        reuse=not bool(state.get("pending_shot_ids")),
        run_id=base["run_id"],
        input_fingerprint=base["input_fingerprint"],
    )
    critique = critique_videos(result["artifacts"])
    status = _fanout_status(result)
    row = store.save_stage(
        StageName.VIDEO_GENERATION.value,
        status=status.value,
        input_fingerprint=base["input_fingerprint"],
        output_fingerprint=fingerprint(result),
        payload={"result": result, "provider": provider, "resolution": resolution},
        critique=critique,
        shot_artifacts=result["artifacts"],
    )
    return {
        **_identity_update(state, StageName.VIDEO_GENERATION, base, row),
        "stage_status": {StageName.VIDEO_GENERATION.value: status.value},
        "stage_outputs": {StageName.VIDEO_GENERATION.value: row},
        "shot_artifacts": result["artifacts"],
        "decision_traces": [item["decision_trace"] for item in result["artifacts"] if item.get("decision_trace")],
        "successful_shot_ids": [item["shot_id"] for item in result["successes"]],
        "failed_shot_ids": [item["shot_id"] for item in result["failures"]],
        "degraded_shot_ids": [item["shot_id"] for item in result["degraded"]],
        "pending_shot_ids": [],
        "current_step": "video_generation_fan_out",
    }


async def _video_generation_fan_in(state: AgentState) -> dict:
    artifacts = _latest_artifacts(state, StageName.VIDEO_GENERATION)
    result = fan_in_shot_results(artifacts)
    return {
        **result,
        "shot_artifacts": artifacts,
        "successful_shot_ids": [item.get("shot_id") for item in result["successes"]],
        "failed_shot_ids": [item.get("shot_id") for item in result["failures"]],
        "degraded_shot_ids": [item.get("shot_id") for item in result["degraded"]],
        "skipped_shot_ids": [item.get("shot_id") for item in result["skipped"]],
        "pending_shot_ids": [item.get("shot_id") for item in result["pending"]],
        "current_step": "video_generation_fan_in",
    }


async def _video_generation_review(state: AgentState) -> dict:
    critique = critique_videos(_latest_artifacts(state, StageName.VIDEO_GENERATION))
    return {"critiques": [critique.model_dump(mode="json")], "current_step": "video_generation_review"}


async def _video_generation_decision(state: AgentState) -> dict:
    return _decision_node(
        state, StageName.VIDEO_GENERATION, next_target="video_review", artifacts_stage=StageName.VIDEO_GENERATION
    )


async def _video_generation_recovery(state: AgentState) -> dict:
    return _recovery_node(
        state, StageName.VIDEO_GENERATION, default_target="video_generation", artifacts_stage=StageName.VIDEO_GENERATION
    )


async def _video_review(state: AgentState) -> dict:
    project_id, store, base = _stage_context(state, StageName.VIDEO_REVIEW)
    if store.stage_is_reusable(StageName.VIDEO_REVIEW.value, base["input_fingerprint"]):
        return _restore_stage(state, store, StageName.VIDEO_REVIEW)
    quality = await _review_shot_videos(state, allow_retry=False)
    visual_pending = bool(quality.get("visual_pending"))
    artifacts = _latest_artifacts(state, StageName.VIDEO_GENERATION)
    critique = critique_videos(artifacts)
    passed = not quality.get("errors") and critique.passed
    error_shot_ids = sorted({issue.shot_id for issue in critique.issues if issue.severity == "error" and issue.shot_id})
    review_critique = {
        "stage": StageName.VIDEO_REVIEW.value,
        "passed": passed,
        "score": critique.score,
        "metrics": [metric.model_dump(mode="json") for metric in critique.metrics],
        "issues": [
            {"code": "video_review_failed", "severity": "error", "message": str(message)}
            for message in (quality.get("errors") or [])
        ]
        + [issue.model_dump(mode="json") for issue in critique.issues if issue.severity == "error"],
        "evidence": list(critique.evidence),
        "failure_kind": (
            critique.failure_kind.value if critique.failure_kind else FailureKind.QUALITY_BELOW_THRESHOLD.value
        )
        if not passed
        else None,
        "recoverable": critique.recoverable,
        "proposed_changes": critique.proposed_changes,
        "affected_shot_ids": sorted(
            set(critique.affected_shot_ids)
            | set(error_shot_ids)
            | set(_shot_ids(project_id) if quality.get("errors") else [])
        ),
        "recommended_strategy": (
            critique.recommended_strategy.value
            if critique.recommended_strategy
            else RecoveryStrategy.REGENERATE_FAILED_SHOTS.value
        )
        if not passed
        else None,
        "source": "quality_review_service",
    }
    if passed and visual_pending:
        # 结构 + 技术可用但视觉未验证：只能算降级通过，绝不写成 succeeded。
        status = StageStatus.DEGRADED
        review_critique["metrics"] = [
            *review_critique["metrics"],
            {
                "name": "visual_quality_pending",
                "passed": None,
                "detail": "视觉模型缺失，仅结构+技术门禁收口；视觉质量未评估",
            },
        ]
        review_critique["issues"] = [
            *review_critique["issues"],
            {
                "code": "visual_quality_pending",
                "severity": "info",
                "message": f"视频视觉质量待审：{quality.get('reason') or '未配置视觉模型'}；已按结构+技术门禁自动继续",
                "details": {
                    "stage": StageName.VIDEO_REVIEW.value,
                    "status": "pending",
                    "shot_ids": [str(item) for item in (quality.get("shot_ids") or []) if item],
                },
            },
        ]
    else:
        status = (
            StageStatus.SUCCEEDED
            if passed
            else (StageStatus.WAITING_HUMAN if _human_allowed(state) else StageStatus.FAILED)
        )
    row = store.save_stage(
        StageName.VIDEO_REVIEW.value,
        status=status.value,
        input_fingerprint=base["input_fingerprint"],
        output_fingerprint=fingerprint({"quality": quality, "critique": critique.model_dump(mode="json")}),
        payload={"quality": quality, "artifacts": artifacts},
        critique=review_critique,
        shot_artifacts=artifacts,
    )
    update: dict[str, Any] = {
        **_identity_update(state, StageName.VIDEO_REVIEW, base, row),
        "stage_status": {StageName.VIDEO_REVIEW.value: status.value},
        "stage_outputs": {StageName.VIDEO_REVIEW.value: row},
        "critiques": [critique.model_dump(mode="json"), review_critique],
        "failed_shot_ids": [
            item.get("shot_id") for item in artifacts if item.get("status") == StageStatus.FAILED.value
        ],
        "current_step": "video_review",
    }
    if passed and visual_pending:
        update.update(
            _visual_pending_update(
                state,
                stage=StageName.VIDEO_REVIEW,
                reason=str(quality.get("reason") or "未配置视觉模型"),
                shot_ids=[str(item) for item in (quality.get("shot_ids") or []) if item],
            )
        )
        update["stage_status"] = {StageName.VIDEO_REVIEW.value: StageStatus.DEGRADED.value}
    return update


async def _video_critic(state: AgentState) -> dict:
    critique = critique_videos(_latest_artifacts(state, StageName.VIDEO_GENERATION))
    return {"critiques": [critique.model_dump(mode="json")], "current_step": "video_critic"}


async def _video_decision(state: AgentState) -> dict:
    return _decision_node(
        state, StageName.VIDEO_REVIEW, next_target="edit_composition", artifacts_stage=StageName.VIDEO_GENERATION
    )


async def _video_recovery(state: AgentState) -> dict:
    return _recovery_node(
        state, StageName.VIDEO_REVIEW, default_target="video_generation", artifacts_stage=StageName.VIDEO_GENERATION
    )


async def _audio_production(state: AgentState) -> dict:
    project_id, store, base = _stage_context(state, StageName.AUDIO_PRODUCTION)
    plan = _audio_execution_plan(project_id, state)
    if plan["mode"] == "external_tts" and not (state.get("storyboard_confirmed") or plan.get("storyboard_confirmed")):
        return _abort("audio_production", "外部 TTS 依赖分镜确认；分镜未确认前不得生成音频")
    shot_versions = _audio_shot_versions(
        project_id, state.get("pending_shot_ids") or state.get("split_recovery_shot_ids")
    )
    scope_ids = state.get("pending_shot_ids") or state.get("split_recovery_shot_ids") or []
    if scope_ids and not set(_shot_versions(project_id)).issuperset(scope_ids):
        return {
            "stage_status": {StageName.AUDIO_PRODUCTION.value: StageStatus.FAILED.value},
            "failed_shot_ids": list(scope_ids),
            "current_step": "audio_production",
        }
    worker = partial(generate_audio_shot, project_id=project_id)
    result = await run_shot_fanout(
        project_id=project_id,
        shot_versions=shot_versions,
        stage=StageName.AUDIO_PRODUCTION,
        worker=worker,
        checkpoint=store,
        concurrency=3,
        reuse=not bool(state.get("pending_shot_ids")),
        run_id=base["run_id"],
        input_fingerprint=base["input_fingerprint"],
    )
    critique = critique_audio(_db_shots(project_id), result["artifacts"])
    status = _fanout_status(result)
    row = store.save_stage(
        StageName.AUDIO_PRODUCTION.value,
        status=status.value,
        input_fingerprint=base["input_fingerprint"],
        output_fingerprint=fingerprint(result),
        payload={"result": result},
        critique=critique,
        shot_artifacts=result["artifacts"],
    )
    return {
        **_identity_update(state, StageName.AUDIO_PRODUCTION, base, row),
        "stage_status": {StageName.AUDIO_PRODUCTION.value: status.value},
        "stage_outputs": {StageName.AUDIO_PRODUCTION.value: row},
        "shot_artifacts": result["artifacts"],
        "decision_traces": [item["decision_trace"] for item in result["artifacts"] if item.get("decision_trace")],
        "successful_shot_ids": [item["shot_id"] for item in result["successes"]],
        "failed_shot_ids": [item["shot_id"] for item in result["failures"]],
        "pending_shot_ids": [],
        "audio_execution_plan": plan,
        "external_tts_required": plan["mode"] == "external_tts",
        "current_step": "audio_production",
    }


async def _audio_review(state: AgentState) -> dict:
    critique = critique_audio(
        _db_shots(state.get("project_id", "")), _latest_artifacts(state, StageName.AUDIO_PRODUCTION)
    )
    return {"critiques": [critique.model_dump(mode="json")], "current_step": "audio_review"}


async def _audio_decision(state: AgentState) -> dict:
    return _decision_node(
        state, StageName.AUDIO_PRODUCTION, next_target="video_generation", artifacts_stage=StageName.AUDIO_PRODUCTION
    )


async def _audio_recovery(state: AgentState) -> dict:
    return _recovery_node(
        state, StageName.AUDIO_PRODUCTION, default_target="audio_production", artifacts_stage=StageName.AUDIO_PRODUCTION
    )


async def _generate_shot_videos(state: AgentState) -> dict:
    """兼容旧自动视频入口：逐镜头独立尝试；只有瞬时失败才补一次重试。"""

    if state.get("errors"):
        return {}
    from api.routes.shot import _run_single_shot_video
    from services.quality_review_service import quality_review_service

    project_id = state["project_id"]
    gate = quality_review_service.storyboard_gate_status(project_id, **_visual_pending_gate_kwargs(state))
    if not gate.get("ok"):
        return _abort(
            "generate_shot_videos", "故事板质量门禁未通过，自动模式禁止生成视频: " + _gate_failure_summary(gate)
        )
    failures: dict[str, str] = {}
    transient: dict[str, bool] = {}
    for shot_id in _shot_ids(project_id):
        try:
            await _run_single_shot_video(shot_id, force=False, capability_mode="auto")
        except Exception as exc:
            failures[shot_id] = str(exc)
            transient[shot_id] = (
                classify_failure(stage=StageName.VIDEO_GENERATION, message=str(exc)).kind in TRANSIENT_FAILURES
            )
    if failures:
        # 禁止无条件「重试一次」：只重试被分类为瞬时失败的错误（超时/存储抖动等），
        # 确定性失败交给恢复决策去修改输入或切换 Provider。
        retried = [shot_id for shot_id in sorted(failures) if transient.get(shot_id)]
        for shot_id in retried:
            try:
                await _run_single_shot_video(shot_id, force=True, capability_mode="auto")
                failures.pop(shot_id, None)
            except Exception as exc:
                failures[shot_id] = str(exc)
    if failures:
        summary = "; ".join(f"{shot_id}: {reason[:120]}" for shot_id, reason in sorted(failures.items()))
        retried_note = f"（其中 {len(retried)} 个瞬时失败已单独重试）" if retried else ""
        return _abort("generate_shot_videos", f"以下镜头视频生成失败{retried_note}: {summary}")
    if _has_unfinished_videos(project_id):
        return _abort("generate_shot_videos", "仍有镜头视频未生成")
    return {"current_step": "generate_shot_videos"}


def _video_structural_fallback(project_id: str, shot_ids: list[str], *, reason: str) -> dict[str, Any]:
    """视觉模型缺失时用可测量的结构 + 技术门禁收口视频阶段。

    只依据 ``critique_videos``（可播放性、时长、分辨率、比例、音轨、黑帧、空帧、
    冻结、音画时长、尾帧、文件大小）判定是否可用；视觉五项维度保持 pending。
    结构/技术不合格时保持失败，交给恢复阶梯继续处理。
    """

    wanted = {str(item) for item in shot_ids if item}
    artifacts = [
        item for item in _shot_artifacts_from_db(project_id) if not wanted or str(item.get("shot_id") or "") in wanted
    ]
    critique = critique_videos(artifacts)
    failed = sorted({issue.shot_id for issue in critique.issues if issue.severity == "error" and issue.shot_id})
    if not critique.passed:
        return {
            **_abort(
                "review_shot_videos",
                f"视觉模型缺失且视频结构/技术检查未通过（{reason}）；受影响镜头: {', '.join(failed[:10]) or '未知'}",
            ),
            "passed": False,
            "visual_pending": False,
        }
    return {
        "passed": True,
        "visual_pending": True,
        "degraded": True,
        "reason": reason,
        "shot_ids": list(shot_ids),
        "critique": critique.model_dump(mode="json"),
        "evidence": {
            "kind": "structural_only_gate",
            "checked": "video_structural_technical",
            "shot_count": len(artifacts),
        },
        "current_step": "review_shot_videos",
    }


def _shot_artifacts_from_db(project_id: str) -> list[dict[str, Any]]:
    """把已落库的镜头视频产物转成 critic 可消费的 artifacts（仅结构/技术所需字段）。"""

    from db import SessionLocal
    from models import Shot

    db = SessionLocal()
    try:
        rows = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
        return [
            {
                "shot_id": str(row.id),
                "stage": StageName.VIDEO_GENERATION.value,
                "status": StageStatus.SUCCEEDED.value if row.video_path else StageStatus.FAILED.value,
                "path": str(row.video_path or ""),
                "video_path": str(row.video_path or ""),
                "audio_path": str(row.audio_path or ""),
                "tail_frame_path": str(row.last_frame_path or ""),
                "expected_duration_s": float(row.duration) if row.duration else None,
            }
            for row in rows
        ]
    finally:
        db.close()


async def _review_shot_videos(state: AgentState, *, allow_retry: bool = True) -> dict:
    """真实执行视频检查；兼容入口可内部重试，阶段化图只执行一轮并交给 recovery。

    视觉模型缺失时：默认 fail-closed 明确终止；显式 auto 模式 + 策略允许时改走
    ``_video_structural_fallback``——只按可测量的结构/技术门禁收口，视觉质量保持
    pending 并降级，绝不宣称审核通过。
    """
    if state.get("errors"):
        return {}
    from api.routes.shot import _run_single_shot_video, prepare_video_quality_retry
    from config import settings
    from services.quality_review_service import STAGE_VIDEO, quality_review_service

    project_id = str(state.get("project_id") or "")
    shot_ids = _shot_ids(project_id)
    allow_human = _legacy_human_allowed(state)
    allow_visual_pending = _visual_pending_continue(state)
    capability = quality_review_service.video_capability()
    if not capability.get("supported"):
        reason = str(capability.get("reason") or "unsupported")
        # 先如实落库 unsupported 审核记录，再决定终止还是按结构门禁降级继续。
        await quality_review_service.record_unsupported_reviews(project_id, shot_ids, STAGE_VIDEO, reason)
        if allow_visual_pending:
            return _video_structural_fallback(project_id, shot_ids, reason=reason)
        if allow_human:
            await quality_review_service.mark_shots_needs_human_review(shot_ids)
        result = _abort("review_shot_videos", f"视频质量审核能力未配置，自动模式不得导出成片（{reason}）")
        if allow_human:
            result["needs_human_review"] = True
        return result
    max_retries = max(0, int(settings.QUALITY_VIDEO_MAX_RETRIES)) if allow_retry else 0
    previous_frames = _previous_last_frames(project_id)
    last_reviews: dict[str, Any] = {}
    for attempt in range(max_retries + 1):
        reviews = {
            shot_id: await quality_review_service.review_video_shot(
                shot_id, previous_frame_path=previous_frames.get(shot_id, "")
            )
            for shot_id in shot_ids
        }
        last_reviews = reviews
        errored = {shot_id: review for shot_id, review in reviews.items() if review.verdict == "error"}
        if errored:
            return {
                **_abort("review_shot_videos", f"视频质量审核 Provider 调用失败: {_review_summary(errored)}"),
                "stage_status": {StageName.VIDEO_GENERATION.value: StageStatus.FAILED.value},
            }
        unsupported = {shot_id: review for shot_id, review in reviews.items() if review.verdict == "unsupported"}
        if unsupported:
            if allow_human:
                await quality_review_service.mark_shots_needs_human_review(sorted(unsupported))
            result = _abort("review_shot_videos", f"视频审核存在未检测维度: {_review_summary(unsupported)}")
            if allow_human:
                result["needs_human_review"] = True
            return result
        failed = {shot_id: review for shot_id, review in reviews.items() if not review.passed}
        if not failed:
            return {"current_step": "review_shot_videos"}
        if attempt >= max_retries:
            break
        retry_ids = [shot_id for shot_id in sorted(failed) if prepare_video_quality_retry(shot_id, failed[shot_id])]
        if not retry_ids:
            break
        for shot_id in retry_ids:
            try:
                await _run_single_shot_video(shot_id, force=True, capability_mode="auto")
            except Exception as exc:
                return _abort("review_shot_videos", exc)
    if allow_human:
        await quality_review_service.mark_shots_needs_human_review(sorted(last_reviews))
    suffix = f"（已重试 {max_retries} 次）" if allow_retry else ""
    result = _abort("review_shot_videos", f"以下镜头视频未通过质量审核{suffix}: {_review_summary(last_reviews)}")
    if allow_human:
        result["needs_human_review"] = True
    return result


async def _compose(state: AgentState) -> dict:
    if state.get("errors"):
        return {}
    project_id, store, base = _stage_context(state, StageName.EDIT_COMPOSITION)
    if store.stage_is_reusable(StageName.EDIT_COMPOSITION.value, base["input_fingerprint"]):
        return _restore_stage(state, store, StageName.EDIT_COMPOSITION)
    try:
        from services.quality_review_service import quality_review_service

        video_gate = quality_review_service.video_gate_status(project_id, **_visual_pending_gate_kwargs(state))
        if not video_gate.get("ok"):
            return _abort("compose", "视频质量门禁未通过，自动模式禁止导出成片: " + _gate_failure_summary(video_gate))
        from api.routes.render import _render_status, _render_task

        await _render_task(project_id, state.get("output_format") or "9:16", state.get("resolution") or "1080p")
        render_status = _render_status.get(project_id, {})
        if render_status.get("status") != "completed":
            raise RuntimeError(render_status.get("message") or "成片导出未完成")
        output_path = str(render_status.get("video_path") or _final_path(project_id))
        critique = critique_compose(project_id, _db_shots(project_id), output_path)
        payload = {"output_path": output_path, "video_path": output_path}
        return _save_stage(state, store, StageName.EDIT_COMPOSITION, base, payload, critique=critique)
    except Exception as exc:
        return _failed_stage(
            state,
            store,
            StageName.EDIT_COMPOSITION,
            base,
            exc,
            critique=critique_compose(project_id, _db_shots(project_id), ""),
        )


async def _edit_review(state: AgentState) -> dict:
    project_id = str(state.get("project_id") or "")
    critique = critique_compose(
        project_id, _db_shots(project_id), str(state.get("output_path") or state.get("video_path") or "")
    )
    return {"critiques": [critique.model_dump(mode="json")], "current_step": "edit_review"}


async def _edit_decision(state: AgentState) -> dict:
    return _decision_node(state, StageName.EDIT_COMPOSITION, next_target="final_review")


async def _edit_recovery(state: AgentState) -> dict:
    return _recovery_node(state, StageName.EDIT_COMPOSITION, default_target="edit_composition")


async def _final_review(state: AgentState) -> dict:
    _project_id, store, base = _stage_context(state, StageName.FINAL_REVIEW)
    critique = critique_final({**state, "shot_artifacts": _latest_artifacts(state, StageName.VIDEO_GENERATION)})
    failure = (
        None if critique.passed else _failure_from_critique(StageName.FINAL_REVIEW, critique.model_dump(mode="json"))
    )
    update = _save_stage(
        state,
        store,
        StageName.FINAL_REVIEW,
        base,
        {"overall_score": critique.score, "passed": critique.passed, "final_report": extract_final_report(critique)},
        critique=critique,
        failure=failure,
    )
    update["final_report"] = extract_final_report(critique)
    return update


async def _final_critic(state: AgentState) -> dict:
    critique = critique_final({**state, "shot_artifacts": _latest_artifacts(state, StageName.VIDEO_GENERATION)})
    return {
        "critiques": [critique.model_dump(mode="json")],
        "final_report": extract_final_report(critique),
        "current_step": "final_critic",
    }


async def _final_decision(state: AgentState) -> dict:
    return _decision_node(
        state, StageName.FINAL_REVIEW, next_target="completed", artifacts_stage=StageName.VIDEO_GENERATION
    )


async def _final_recovery(state: AgentState) -> dict:
    update = _recovery_node(
        state,
        StageName.FINAL_REVIEW,
        default_target=_final_feedback_target(state),
        artifacts_stage=StageName.VIDEO_GENERATION,
    )
    if (
        update.get("run_status") not in {RunStatus.FAILED.value, RunStatus.DEGRADED.value}
        and not update.get("needs_human_review")
        and update.get("selected_strategy") != RecoveryStrategy.SPLIT_SHOT.value
    ):
        update["pending_recovery_target"] = _final_feedback_target(state)
    return update


def _record_terminal(state: AgentState, *, status: str, event: str, reason: str, **fields: Any) -> None:
    """终态节点把运行结论写入检查点：进程重启后追踪仍能看到最终决策与原因。"""

    project_id = str(state.get("project_id") or "")
    if not project_id:
        return
    try:
        store = CheckpointStore.get(project_id, str(state.get("run_id") or "auto"))
        store.set_status(status, reason=reason)
        store.add_event(event, **{"stage": str(state.get("current_stage") or ""), **fields})
    except Exception:  # noqa: BLE001 - 终态记录失败不能阻断流程结束
        logger.exception("记录运行终态失败 project=%s status=%s", project_id, status)


def _latest_failure_message(state: AgentState) -> str:
    """提取最近一次阶段失败的可读消息（阶段失败记录优先，其次 Critic 问题）。"""

    for name in reversed(GRAPH_STAGE_ORDER):
        failure = ((state.get("stage_outputs") or {}).get(name) or {}).get("failure")
        if isinstance(failure, dict):
            message = str(failure.get("message") or "")
            if message:
                return message
    for critique in reversed(state.get("critiques") or []):
        for issue in critique.get("issues") or []:
            if issue.get("severity") == "error" and issue.get("message"):
                return str(issue["message"])
    return ""


async def _auto_abort(state: AgentState) -> dict:
    reason = str(state.get("human_reason") or "自动恢复已耗尽，流程明确失败；已成功镜头保留在检查点中")
    # 终止原因必须携带真实失败信息（如输出截断诊断）：决策节点直接选 terminal_failure
    # 时不会经过恢复节点，human_reason 只是兜底文案。
    failure_message = _latest_failure_message(state)
    if failure_message and failure_message[:200] not in reason:
        reason = f"{reason}（最近失败：{failure_message[:300]}）"
    _record_terminal(state, status=RunStatus.FAILED.value, event="auto_abort", reason=reason)
    return {
        "run_status": RunStatus.FAILED.value,
        "current_step": "auto_abort",
        "errors": [f"[auto_abort] {reason}"],
    }


async def _degraded_publish(state: AgentState) -> dict:
    """自动模式恢复无法继续但存在可用部分结果：按降级结果结束并保留失败清单。"""

    reason = str(state.get("degraded_reason") or state.get("human_reason") or "自动恢复无法继续，已按降级结果发布")
    stage = str(state.get("current_stage") or "")
    successful = sorted(
        {
            str(item.get("shot_id") or "")
            for item in (state.get("shot_artifacts") or [])
            if item.get("status") == StageStatus.SUCCEEDED.value
        }
    )
    _record_terminal(
        state,
        status=RunStatus.DEGRADED.value,
        event="degraded_publish",
        reason=reason,
        shot_ids=successful,
        failed_shot_ids=[str(item or "") for item in (state.get("failed_shot_ids") or []) if item],
    )
    update: dict[str, Any] = {
        "run_status": RunStatus.DEGRADED.value,
        "degraded_published": True,
        "degraded_reason": reason,
        "current_step": "degraded_publish",
        "pending_recovery_target": "",
    }
    if stage:
        update["stage_status"] = {stage: StageStatus.DEGRADED.value}
    if successful:
        update["successful_shot_ids"] = successful
    return update


async def _human_gate(state: AgentState) -> dict:
    if not _human_allowed(state):
        return await _auto_abort(state)
    project_id = state.get("project_id", "")
    _set_project_status(project_id, "needs_review")
    reason = state.get("human_reason") or "Agent 已保留成功结果，等待人工确认后从检查点续跑"
    _record_terminal(state, status=RunStatus.WAITING_HUMAN.value, event="human_gate", reason=reason)
    return {
        "run_status": RunStatus.WAITING_HUMAN.value,
        "needs_human_review": True,
        "human_reason": reason,
        "current_step": "human_gate",
    }


# --- routing ---


def _route_director_review(state: AgentState) -> str:
    return "decision"


def _route_storyboard_review(state: AgentState) -> str:
    return "decision"


def _route_asset_review(state: AgentState) -> str:
    return "decision"


def _route_quality_review(state: AgentState) -> str:
    return "decision"


def _route_video_review(state: AgentState) -> str:
    return "decision"


def _route_audio_review(state: AgentState) -> str:
    return "decision"


def _route_final_review(state: AgentState) -> str:
    return "decision"


def _route_director_decision(state: AgentState) -> str:
    return _route_decision(state, StageName.DIRECTOR_PLANNING, "next", "recover")


def _route_storyboard_decision(state: AgentState) -> str:
    return _route_decision(state, StageName.STORYBOARD_DESIGN, "next", "recover")


def _route_asset_decision(state: AgentState) -> str:
    return _route_decision(state, StageName.ASSET_PREPARATION, "next", "recover")


def _route_image_decision(state: AgentState) -> str:
    return _route_decision(state, StageName.IMAGE_GENERATION, "next", "recover")


def _route_quality_decision(state: AgentState) -> str:
    return _route_decision(state, StageName.QUALITY_REVIEW, "next", "recover")


def _route_audio_decision(state: AgentState) -> str:
    return _route_decision(state, StageName.AUDIO_PRODUCTION, "next", "recover")


def _route_video_generation_decision(state: AgentState) -> str:
    return _route_decision(state, StageName.VIDEO_GENERATION, "next", "recover")


def _route_video_decision(state: AgentState) -> str:
    return _route_decision(state, StageName.VIDEO_REVIEW, "next", "recover")


def _route_edit_decision(state: AgentState) -> str:
    return _route_decision(state, StageName.EDIT_COMPOSITION, "next", "recover")


def _route_final_decision(state: AgentState) -> str:
    if state.get("needs_human_review") and _human_allowed(state):
        return "human"
    critique = _latest_critique(state, StageName.FINAL_REVIEW)
    if critique and critique.get("passed"):
        return "done"
    return _route_decision(state, StageName.FINAL_REVIEW, "done", "recover")


def _route_director_recovery(state: AgentState) -> str:
    return _route_recovery(state, {"director_planning": "retry"})


def _route_storyboard_recovery(state: AgentState) -> str:
    return _route_recovery(state, {"storyboard_design": "retry"})


def _route_asset_recovery(state: AgentState) -> str:
    return _route_recovery(state, {"asset_preparation": "retry", "storyboard_design": "storyboard"})


def _route_image_recovery(state: AgentState) -> str:
    return _route_recovery(state, {"image_generation": "retry", "storyboard_design": "storyboard"})


def _route_quality_recovery(state: AgentState) -> str:
    return _route_recovery(
        state, {"image_generation": "retry", "quality_review": "retry", "storyboard_design": "storyboard"}
    )


def _route_audio_recovery(state: AgentState) -> str:
    return _route_recovery(
        state, {"audio_production": "retry", "image_generation": "image", "storyboard_design": "storyboard"}
    )


def _route_video_generation_recovery(state: AgentState) -> str:
    return _route_recovery(
        state,
        {
            "video_generation": "retry",
            "image_generation": "image",
            "audio_production": "audio",
            "storyboard_design": "storyboard",
        },
    )


def _route_video_recovery(state: AgentState) -> str:
    return _route_recovery(
        state,
        {
            "video_generation": "retry",
            "video_review": "retry",
            "image_generation": "image",
            "audio_production": "audio",
            "storyboard_design": "storyboard",
        },
    )


def _route_edit_recovery(state: AgentState) -> str:
    return _route_recovery(
        state,
        {
            "edit_composition": "retry",
            "image_generation": "image",
            "audio_production": "audio",
            "video_generation": "video",
        },
    )


def _route_final_recovery(state: AgentState) -> str:
    return _route_recovery(
        state,
        {
            "image_generation": "image",
            "audio_production": "audio",
            "video_generation": "video",
            "edit_composition": "compose",
            "final_review": "compose",
        },
    )


def _route_decision(state: AgentState, stage: StageName | str, next_key: str, recover_key: str) -> str:
    if state.get("needs_human_review") and _human_allowed(state):
        return "human"
    key = StageName(stage).value
    status = str((state.get("stage_status") or {}).get(key, StageStatus.PENDING.value))
    critique = _latest_critique(state, stage)
    passed = bool(critique and critique.get("passed"))
    if status in {StageStatus.SUCCEEDED.value, StageStatus.DEGRADED.value, StageStatus.SKIPPED.value} and passed:
        return next_key
    # 决策节点选择的终止策略优先于次数兜底：恢复无法继续时按可用性降级发布或终止。
    selected = str(state.get("selected_strategy") or "")
    if selected == RecoveryStrategy.DEGRADED_PUBLISH.value:
        return "degraded"
    if selected == RecoveryStrategy.TERMINAL_FAILURE.value:
        return "failed"
    attempts = int((state.get("recovery_attempts") or {}).get(key, 0))
    max_attempts = default_quality_profile(state.get("quality_profile")).max_recovery_attempts
    if attempts < max_attempts:
        return recover_key
    return "human" if _human_allowed(state) else "failed"


def _route_recovery(state: AgentState, targets: dict[str, str]) -> str:
    if state.get("needs_human_review"):
        return "human" if _human_allowed(state) else "failed"
    selected = str(state.get("selected_strategy") or "")
    if selected == RecoveryStrategy.DEGRADED_PUBLISH.value:
        return "degraded"
    if selected == RecoveryStrategy.TERMINAL_FAILURE.value:
        return "failed"
    target = str(state.get("pending_recovery_target") or "")
    return targets.get(target, "failed")


# --- common helpers ---


def _human_allowed(state: AgentState) -> bool:
    """只有 manual 且显式 human_gate_policy=manual 才能进入人工节点。"""

    return (
        str(state.get("mode") or "").lower() == "manual"
        and str(state.get("human_gate_policy") or "").lower() == "manual"
    )


def _legacy_human_allowed(state: AgentState) -> bool:
    """旧兼容入口未携带 mode 时沿用手动语义；显式 auto 永不放行。"""
    if "mode" not in state and "auto" not in state:
        return True
    return _human_allowed(state)


VISUAL_PENDING_REASON_TEMPLATE = (
    "未配置真实视觉模型（{reason}）；已按结构 + 技术门禁自动继续，"
    "首帧相似度/参考匹配/运动稳定/镜头连续/动作完成五项视觉维度保持 pending，未验证"
)


def _auto_mode(state: AgentState) -> bool:
    """显式 mode=auto（生产自动流程的唯一模式标记）。"""

    return str(state.get("mode") or "").lower() == "auto"


def _visual_pending_continue(state: AgentState) -> bool:
    """显式 auto 模式 + 策略允许时，视觉能力缺失不阻断闭环。

    只有 ``mode=auto`` 且 ``QUALITY_VISUAL_PENDING_POLICY=continue`` 才返回 True：
    manual/legacy 调用（含未携带 mode 的兼容入口）保持原有 fail-closed 语义。
    """

    if str(state.get("mode") or "").lower() != "auto":
        return False
    try:
        from config import settings

        policy = str(getattr(settings, "QUALITY_VISUAL_PENDING_POLICY", "continue") or "continue").strip().lower()
    except Exception:  # noqa: BLE001 - 配置不可读时按更保守的 block 处理
        policy = "block"
    return policy == "continue"


def _visual_pending_gate_kwargs(state: AgentState) -> dict[str, Any]:
    """仅在策略生效时向门禁传参，避免改变 manual/legacy 调用的调用签名。"""

    return {"allow_visual_pending": True} if _visual_pending_continue(state) else {}


def _visual_pending_update(
    state: AgentState, *, stage: StageName, reason: str, shot_ids: list[str] | None = None
) -> dict[str, Any]:
    """视觉能力缺失时的降级继续记录：只降级，不冒充质量通过。"""

    detail = VISUAL_PENDING_REASON_TEMPLATE.format(reason=reason or "未配置")
    return {
        "visual_quality_pending": True,
        "visual_pending_reason": detail,
        "visual_pending_stages": [stage.value],
        "visual_pending_shot_ids": list(shot_ids or []),
        "degraded_published": True,
        "degraded_reason": detail,
        "stage_status": {stage.value: StageStatus.DEGRADED.value},
        "current_step": f"{stage.value}_visual_pending",
    }


def _final_feedback_target(state: AgentState) -> str:
    explicit = str(state.get("final_recovery_target") or "").strip().lower()
    aliases = {
        "image": StageName.IMAGE_GENERATION.value,
        "image_generation": StageName.IMAGE_GENERATION.value,
        "audio": StageName.AUDIO_PRODUCTION.value,
        "audio_production": StageName.AUDIO_PRODUCTION.value,
        "video": StageName.VIDEO_GENERATION.value,
        "video_generation": StageName.VIDEO_GENERATION.value,
        "edit": StageName.EDIT_COMPOSITION.value,
        "compose": StageName.EDIT_COMPOSITION.value,
        "edit_composition": StageName.EDIT_COMPOSITION.value,
    }
    if explicit in aliases:
        return aliases[explicit]
    issues = (_latest_critique(state, StageName.FINAL_REVIEW) or {}).get("issues") or []
    for issue in issues:
        if issue.get("severity") not in {"error", "warning"}:
            continue
        issue_stage = str((issue.get("details") or {}).get("source_stage") or "").strip().lower()
        if issue_stage in aliases:
            return aliases[issue_stage]
        code = str(issue.get("code") or "").lower()
        if code.startswith(("audio_", "tts_", "voice_", "dialogue_")):
            return StageName.AUDIO_PRODUCTION.value
        if code.startswith(("video_", "tail_frame_", "black_frame", "frozen_frame")):
            return StageName.VIDEO_GENERATION.value
        if code.startswith(("image_", "storyboard_", "reference_")):
            return StageName.IMAGE_GENERATION.value
        if code.startswith(("render_", "compose_", "edit_", "subtitle_")):
            return StageName.EDIT_COMPOSITION.value
    text = " ".join(
        str(value or "")
        for value in (
            state.get("final_feedback"),
            state.get("human_feedback"),
            *(
                item.get("message", "")
                for item in (_latest_critique(state, StageName.FINAL_REVIEW) or {}).get("issues", [])
            ),
        )
    ).lower()
    if any(word in text for word in ("audio", "tts", "voice", "配音", "音轨", "声音", "台词", "对白")):
        return StageName.AUDIO_PRODUCTION.value
    if any(word in text for word in ("video", "闪烁", "冻结", "动态", "运动", "播放", "视频")):
        return StageName.VIDEO_GENERATION.value
    if any(word in text for word in ("edit", "compose", "剪辑", "合成", "转场", "字幕", "节奏", "时长")):
        return StageName.EDIT_COMPOSITION.value
    return StageName.IMAGE_GENERATION.value


def _final_feedback_shot_ids(state: AgentState, target: str) -> list[str]:
    """只向所选阶段回流带有镜头标识的失败反馈。"""

    aliases = {
        StageName.IMAGE_GENERATION.value: ("image_", "storyboard_", "reference_"),
        StageName.AUDIO_PRODUCTION.value: ("audio_", "tts_", "voice_", "dialogue_"),
        StageName.VIDEO_GENERATION.value: ("video_", "tail_frame_", "black_frame", "frozen_frame"),
        StageName.EDIT_COMPOSITION.value: ("render_", "compose_", "edit_", "subtitle_"),
    }
    ids = set()
    for issue in (_latest_critique(state, StageName.FINAL_REVIEW) or {}).get("issues") or []:
        if issue.get("severity") not in {"error", "warning"}:
            continue
        source_stage = str((issue.get("details") or {}).get("source_stage") or "")
        code = str(issue.get("code") or "").lower()
        if source_stage == target or (not source_stage and code.startswith(aliases.get(target, ()))):
            if issue.get("shot_id"):
                ids.add(str(issue["shot_id"]))
            ids.update(str(item) for item in (issue.get("details") or {}).get("shot_ids") or [] if item)
    return sorted(ids)


def _audio_execution_plan(project_id: str, state: AgentState) -> dict[str, Any]:
    """按镜头 audio_mode 计算外部 TTS/native audio 依赖，不调用 Provider。"""

    shots = _db_shots(project_id)
    external: list[str] = []
    native: list[str] = []
    for shot in shots:
        if not shot.get("dialogue"):
            continue
        mode = _resolve_audio_mode(shot)
        (external if mode == "tts" else native).append(str(shot.get("shot_id") or ""))
    mode = "external_tts" if external else ("native_audio" if native else "none")
    return {
        "mode": mode,
        "external_tts_shot_ids": external,
        "native_audio_shot_ids": native,
        "storyboard_confirmed": bool(state.get("storyboard_confirmed")),
        "dependency": [
            "storyboard_confirmed",
            "audio_production",
            "audio_review",
            "video_generation",
            "video_review",
            "edit_composition",
            "final_review",
        ]
        if mode == "external_tts"
        else ["storyboard_confirmed", "video_generation", "video_review", "edit_composition", "final_review"],
    }


def _resolve_audio_mode(shot: dict[str, Any]) -> str:
    from services.audio_routing import resolve_audio_mode

    return resolve_audio_mode(shot)


# auto 流水线的阶段进度权重：任务中心在解析（细粒度 5%~18%）之后仍能看到
# 流程真实推进，而不是一直停在 0%「已排队」。manual 流程沿用 route 层的上报。
_STAGE_PROGRESS_WEIGHTS: dict[str, int] = {
    StageName.DIRECTOR_PLANNING.value: 5,
    StageName.STORYBOARD_DESIGN.value: 20,
    StageName.ASSET_PREPARATION.value: 35,
    StageName.IMAGE_GENERATION.value: 50,
    StageName.QUALITY_REVIEW.value: 58,
    StageName.AUDIO_PRODUCTION.value: 65,
    StageName.VIDEO_GENERATION.value: 75,
    StageName.VIDEO_REVIEW.value: 85,
    StageName.EDIT_COMPOSITION.value: 93,
    StageName.FINAL_REVIEW.value: 97,
}


def _report_stage_progress(project_id: str, stage: StageName) -> None:
    """阶段进入时向任务中心上报粗粒度进度（auto/manual 键都写，仅持有者生效）。"""

    weight = _STAGE_PROGRESS_WEIGHTS.get(stage.value)
    if weight is None:
        return
    label = (GRAPH_NODE_META.get(stage.value) or {}).get("label") or stage.value
    try:
        from services.task_registry import update_progress as update_job_progress

        for key in (f"project:{project_id}:pipeline:auto", f"project:{project_id}:pipeline:manual"):
            update_job_progress(key, weight, current_step=stage.value, message=f"{label}进行中")
    except Exception:  # noqa: BLE001 - 进度上报失败不能阻断生成流程
        logger.debug("阶段进度上报失败: project=%s stage=%s", project_id, stage.value)


def _stage_context(state: AgentState, stage: StageName) -> tuple[str, CheckpointStore, dict[str, Any]]:
    project_id = str(state.get("project_id") or "")
    if not project_id:
        raise RuntimeError("缺少 project_id")
    run_id = str(state.get("run_id") or "auto")
    store = CheckpointStore.get(project_id, run_id)
    changes = store.detect_changes()
    recovery = bool(state.get("pending_recovery_target")) or str(
        (state.get("stage_status") or {}).get(str(state.get("current_stage") or ""), "")
    ) in {
        StageStatus.RECOVERING.value,
        StageStatus.FAILED.value,
    }
    # stage_entered 事件是「当前阶段」的稳定事实来源：进程重启后仍能从检查点
    # 文件还原最近一次执行到的阶段，而不依赖内存 state。
    store.add_event(
        "stage_entered", stage=stage.value, recovery=recovery, shot_version=int(state.get("shot_version") or 0)
    )
    _report_stage_progress(project_id, stage)
    initial_state = dict(state.get("initial_state") or {})
    current_stage = state.get("current_stage")
    ensure_stage_transition(current_stage, stage, recovery=recovery)
    shot_version = int(state.get("shot_version") or 0)
    upstream_outputs = {
        name: str((state.get("stage_outputs") or {}).get(name, {}).get("output_fingerprint") or "")
        for name in GRAPH_STAGE_ORDER[: GRAPH_STAGE_ORDER.index(stage.value)]
    }
    input_fingerprint = fingerprint(
        {
            "project_id": project_id,
            "run_id": run_id,
            "stage": stage.value,
            "shot_version": shot_version,
            "initial_state": initial_state,
            "quality_profile": state.get("quality_profile") or QualityProfileName.STANDARD.value,
            "prompt_revisions": state.get("prompt_revisions") or [],
            "provider_switch": state.get("provider_switch") or {},
            "resolution": state.get("resolution") or "",
            "versions": store.data.get("version_snapshot", {}),
            "upstream_outputs": upstream_outputs,
        }
    )
    return (
        project_id,
        store,
        {
            "project_id": project_id,
            "run_id": run_id,
            "shot_version": shot_version,
            "input_fingerprint": input_fingerprint,
            "checkpoint_key": stage_contract(stage).checkpoint_key,
            "initial_state": initial_state,
            "changes": changes,
            "recovery": recovery,
        },
    )


def _identity_update(
    state: AgentState, stage: StageName, base: dict[str, Any], row: dict[str, Any] | None = None
) -> dict[str, Any]:
    source = row or base
    transition = {
        "from": str(state.get("current_stage") or ""),
        "to": stage.value,
        "recovery": bool(base.get("recovery")),
        "input_fingerprint": str(source.get("input_fingerprint") or base.get("input_fingerprint") or ""),
    }
    return {
        "project_id": str(source.get("project_id") or base.get("project_id") or ""),
        "run_id": str(source.get("run_id") or base.get("run_id") or "auto"),
        "shot_version": int(source.get("shot_version") or base.get("shot_version") or 0),
        "input_fingerprint": str(source.get("input_fingerprint") or base.get("input_fingerprint") or ""),
        "checkpoint_key": str(source.get("checkpoint_key") or base.get("checkpoint_key") or ""),
        "current_stage": stage.value,
        "stage_history": [transition],
    }


def _save_stage(
    state: AgentState,
    store: CheckpointStore,
    stage: StageName,
    base: dict[str, Any],
    payload: dict[str, Any],
    *,
    critique: Any = None,
    failure: Any = None,
) -> dict:
    status = StageStatus.DEGRADED if failure else StageStatus.SUCCEEDED
    output_fp = fingerprint(payload)
    row = store.save_stage(
        stage.value,
        status=status.value,
        input_fingerprint=base["input_fingerprint"],
        output_fingerprint=output_fp,
        payload=payload,
        critique=critique,
        failure=failure,
        shot_version=base.get("shot_version", 0),
    )
    strategy = default_quality_profile(state.get("quality_profile"))
    update = {
        "stage_status": {stage.value: status.value},
        "stage_outputs": {stage.value: row},
        "current_step": stage.value,
        "run_status": RunStatus.RUNNING.value,
        "quality_threshold": strategy.quality_threshold,
    }
    update.update(_identity_update(state, stage, base, row))
    update.update(_payload_state_fields(stage, payload))
    if critique is not None:
        update["critiques"] = [critique.model_dump(mode="json")]
    return update


def _failed_stage(
    state: AgentState,
    store: CheckpointStore,
    stage: StageName,
    base: dict[str, Any],
    exc: Exception,
    *,
    critique: Any = None,
) -> dict:
    failure = classify_failure(stage=stage, message=str(exc))
    row = store.save_stage(
        stage.value,
        status=StageStatus.FAILED.value,
        input_fingerprint=base["input_fingerprint"],
        failure=failure,
        critique=critique,
        shot_version=base.get("shot_version", 0),
    )
    trace = choose_recovery(
        failure,
        stage=stage,
        run_id=base.get("run_id", "auto"),
        quality=state.get("quality_profile"),
        project_id=base.get("project_id", ""),
        shot_version=base.get("shot_version", 0),
        critique=critique,
        input_fingerprint=base["input_fingerprint"],
    )
    store.add_decision(trace)
    return {
        "stage_status": {stage.value: StageStatus.FAILED.value},
        "stage_outputs": {stage.value: row},
        "critiques": [critique.model_dump(mode="json")] if critique else [],
        "decision_traces": [trace.model_dump(mode="json")],
        "current_step": stage.value,
        "run_status": RunStatus.RECOVERING.value,
        **_identity_update(state, stage, base, row),
    }


def _restore_stage(state: AgentState, store: CheckpointStore, stage: StageName) -> dict:
    row = store.stage(stage.value)
    payload = dict(row.get("payload") or {})
    base = {
        "project_id": row.get("project_id") or state.get("project_id") or "",
        "run_id": row.get("run_id") or state.get("run_id") or "auto",
        "shot_version": row.get("shot_version") or state.get("shot_version") or 0,
        "input_fingerprint": row.get("input_fingerprint") or state.get("input_fingerprint") or "",
        "checkpoint_key": row.get("checkpoint_key") or stage_contract(stage).checkpoint_key,
        "recovery": False,
    }
    update = {
        "stage_status": {stage.value: str(row.get("status") or StageStatus.SUCCEEDED.value)},
        "stage_outputs": {stage.value: row},
        "current_step": stage.value,
        **_identity_update(state, stage, base, row),
    }
    update.update(_payload_state_fields(stage, payload))
    return update


def _payload_state_fields(stage: StageName, payload: dict[str, Any]) -> dict[str, Any]:
    fields = {
        StageName.DIRECTOR_PLANNING: (
            "script_title",
            "genre",
            "style_suggestion",
            "characters",
            "raw_script",
            "script_scenes",
            "logic_issues",
            "rag_context",
            "requested_style",
            "effective_style",
            "style_source",
        ),
        StageName.STORYBOARD_DESIGN: ("shots", "timing_plan"),
        StageName.ASSET_PREPARATION: (
            "characters",
            "script_scenes",
            "shots",
            "reference_supported",
            "consistency_report",
        ),
        StageName.EDIT_COMPOSITION: ("output_path", "video_path"),
        StageName.FINAL_REVIEW: ("final_report",),
    }.get(stage, ())
    update = {key: payload[key] for key in fields if key in payload}
    if stage is StageName.QUALITY_REVIEW:
        update["storyboard_confirmed"] = bool(payload.get("passed") or payload.get("storyboard_confirmed"))
    return update


def _remaining_retries(state: AgentState, stage: StageName) -> int:
    profile = default_quality_profile(state.get("quality_profile"))
    used = int((state.get("recovery_attempts") or {}).get(stage.value, 0))
    return max(0, profile.max_recovery_attempts - used)


def _candidate_results(state: AgentState, artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把逐镜头产物（含视频候选历史）聚合成决策可用的候选结果证据。"""

    rows: list[dict[str, Any]] = []
    for item in artifacts:
        candidates = [entry for entry in (item.get("video_candidates") or []) if isinstance(entry, dict)]
        if candidates:
            rows.extend(
                {
                    "status": str(entry.get("status") or ""),
                    "provider": str(entry.get("provider") or ""),
                    "structural_passed": entry.get("structural_passed"),
                    "technical_passed": ((entry.get("metrics") or {}).get("categories") or {})
                    .get("technical_quality", {})
                    .get("passed"),
                    "path": str(entry.get("path") or entry.get("video_path") or ""),
                }
                for entry in candidates
            )
        else:
            path = str(item.get("path") or "")
            structural_passed = item.get("structural_passed")
            technical_passed = item.get("technical_passed")
            if structural_passed is None and str(item.get("stage") or "") == StageName.IMAGE_GENERATION.value and path:
                from services.structural_validation import validate_image_file

                structural_passed = validate_image_file(path).get("passed")
            if structural_passed is None and str(item.get("stage") or "") == StageName.VIDEO_GENERATION.value and path:
                from services.structural_validation import validate_video_sync

                report = validate_video_sync(path)
                categories = report.get("categories") or {}
                structural_passed = (categories.get("structural_validity") or {}).get("passed")
                technical_passed = (categories.get("technical_quality") or {}).get("passed")
            rows.append(
                {
                    "status": str(item.get("status") or ""),
                    "provider": str(item.get("provider") or ""),
                    "structural_passed": structural_passed,
                    "technical_passed": technical_passed,
                    "path": path,
                }
            )
    return rows


def _attempted_strategies(state: AgentState, stage: StageName) -> list[str]:
    """从持久化决策历史读取本阶段已经执行过的动作。"""
    attempted: list[str] = []
    for row in state.get("recovery_history") or []:
        if str(row.get("stage") or "") != stage.value:
            continue
        strategy = str(row.get("strategy") or row.get("selected_strategy") or "")
        if strategy and strategy not in attempted:
            attempted.append(strategy)
    return attempted


def _decision_context(state: AgentState, stage: StageName, critique: dict[str, Any] | None) -> dict[str, Any]:
    """决策节点的公共上下文：模式、剩余重试次数、质量分与候选结果。"""

    score = None
    if critique:
        try:
            score = float(critique.get("score"))
        except (TypeError, ValueError):
            score = None
    return {
        "mode": str(state.get("mode") or "auto"),
        "retries_remaining": _remaining_retries(state, stage),
        "quality_score": score,
        "attempted_strategies": _attempted_strategies(state, stage),
    }


def _decision_node(
    state: AgentState, stage: StageName, *, next_target: str, artifacts_stage: StageName | None = None
) -> dict:
    critique = _latest_critique(state, stage)
    artifacts = _latest_artifacts(state, artifacts_stage or stage)
    failure = (
        (_failure_from_artifacts(stage, artifacts) if artifacts else None)
        or _failure_from_stage_output(state, stage)
        or _failure_from_critique(stage, critique)
    )
    context = _decision_context(state, stage, critique)
    trace = choose_recovery(
        failure,
        stage=stage,
        run_id=str(state.get("run_id") or "auto"),
        quality=state.get("quality_profile"),
        project_id=str(state.get("project_id") or ""),
        shot_version=int(state.get("shot_version") or 0),
        critique=critique,
        input_fingerprint=str(
            state.get("input_fingerprint")
            or fingerprint({"stage": stage.value, "state": state.get("stage_outputs", {})})
        ),
        mode=context["mode"],
        retries_remaining=context["retries_remaining"],
        quality_score=context["quality_score"],
        candidate_results=_candidate_results(state, artifacts),
        attempted_strategies=context["attempted_strategies"],
    )
    CheckpointStore.get(str(state.get("project_id") or ""), str(state.get("run_id") or "auto")).add_decision(trace)
    update: dict[str, Any] = {
        "decision_traces": [trace.model_dump(mode="json")],
        "recovery_candidates": [item.model_dump(mode="json") for item in trace.candidates],
        "current_step": f"{stage.value}_decision",
        "next_target": next_target,
    }
    # 只有阶段未通过时才暴露选中策略，供路由执行降级发布/明确终止。
    if not bool(critique and critique.get("passed")) and trace.selected is not None:
        update["selected_strategy"] = trace.selected.strategy.value
    return update


def _recovery_node(
    state: AgentState, stage: StageName, *, default_target: str, artifacts_stage: StageName | None = None
) -> dict:
    artifacts = _latest_artifacts(state, artifacts_stage or stage)
    critique = _latest_critique(state, stage)
    failure = (
        (_failure_from_artifacts(stage, artifacts) if artifacts else None)
        or _failure_from_stage_output(state, stage)
        or _failure_from_critique(stage, critique)
    )
    context = _decision_context(state, stage, critique)
    trace = choose_recovery(
        failure,
        stage=stage,
        run_id=str(state.get("run_id") or "auto"),
        quality=state.get("quality_profile"),
        project_id=str(state.get("project_id") or ""),
        shot_version=int(state.get("shot_version") or 0),
        critique=critique,
        input_fingerprint=str(
            state.get("input_fingerprint")
            or fingerprint({"stage": stage.value, "attempts": state.get("recovery_attempts", {})})
        ),
        mode=context["mode"],
        retries_remaining=context["retries_remaining"],
        quality_score=context["quality_score"],
        candidate_results=_candidate_results(state, artifacts),
        attempted_strategies=context["attempted_strategies"],
    )
    selected = trace.selected
    selected_strategy = selected.strategy if selected else RecoveryStrategy.TERMINAL_FAILURE
    prompt_revisions = []
    provider_switch = dict(state.get("provider_switch") or {})
    pending_ids = list(state.get("pending_shot_ids") or [])
    failed_ids = {
        str(item.get("shot_id") or "") for item in artifacts if item.get("status") == StageStatus.FAILED.value
    }
    if stage is not StageName.FINAL_REVIEW:
        failed_ids.update(str(item or "") for item in (state.get("failed_shot_ids") or []))
    critique_data = critique or {}
    failed_ids.update(str(item or "") for item in (critique_data.get("affected_shot_ids") or []))
    failed_ids.update(
        str(item.get("shot_id") or "")
        for item in critique_data.get("issues", [])
        if item.get("severity") == "error" and item.get("shot_id")
    )
    failed_ids.discard("")
    pending_ids = [shot_id for shot_id in sorted(failed_ids)] or pending_ids
    if selected and selected.shot_ids and stage is not StageName.FINAL_REVIEW:
        pending_ids = sorted(set(pending_ids) | {str(shot_id) for shot_id in selected.shot_ids if shot_id})
    if selected_strategy is RecoveryStrategy.CHANGE_SEED and not pending_ids:
        pending_ids = sorted({str(item.get("shot_id")) for item in artifacts if item.get("shot_id")})
    target = default_target
    generation_stage = {
        StageName.QUALITY_REVIEW: StageName.IMAGE_GENERATION,
        StageName.VIDEO_REVIEW: StageName.VIDEO_GENERATION,
    }.get(stage, stage)
    if stage is StageName.FINAL_REVIEW:
        generation_stage = StageName(_final_feedback_target(state))
        feedback_ids = _final_feedback_shot_ids(state, generation_stage.value)
        pending_ids = feedback_ids if feedback_ids else []
    if selected_strategy is RecoveryStrategy.SPLIT_SHOT and stage is StageName.STORYBOARD_DESIGN:
        # Planning still operates on in-memory shots; only persisted downstream
        # stages can safely mutate an existing timeline under a DB version fence.
        selected_strategy = RecoveryStrategy.TERMINAL_FAILURE
    if selected_strategy is RecoveryStrategy.SPLIT_SHOT and not pending_ids:
        selected_strategy = RecoveryStrategy.TERMINAL_FAILURE
    if selected_strategy is RecoveryStrategy.REVISE_PROMPT:
        prompt_revisions = [_scope_revision(selected.prompt_changes, generation_stage, pending_ids)]
    elif selected_strategy is RecoveryStrategy.CHANGE_SEED:
        for shot_id in pending_ids:
            seed = _recovery_seed(state, generation_stage, shot_id, artifacts)
            prompt_revisions.append(
                {
                    "shot_id": shot_id,
                    "instruction": selected.prompt_changes.get("instruction", ""),
                    "patches": [
                        {
                            "field": "seed",
                            "op": "set",
                            "value": {"seed": seed},
                            "shot_id": shot_id,
                            "target_stage": generation_stage.value,
                            "reason": "自动更换随机种子",
                        }
                    ],
                }
            )
    elif selected_strategy is RecoveryStrategy.SWITCH_PROVIDER:
        provider_switch[generation_stage.value] = selected.provider
    elif selected_strategy is RecoveryStrategy.LOWER_RESOLUTION:
        provider_switch[f"{generation_stage.value}:resolution"] = (
            "480p" if generation_stage is StageName.VIDEO_GENERATION else "540p"
        )
    elif selected_strategy is RecoveryStrategy.SPLIT_SHOT:
        # Storyboard design intentionally does not rewrite existing Shot rows.
        # Persist the split under a version fence before regenerating the parts.
        target = StageName.IMAGE_GENERATION.value
    elif selected_strategy is RecoveryStrategy.MERGE_SHOTS:
        # There is no persisted merge transaction. Do not report a prompt-only
        # storyboard retry as though it had changed the database timeline.
        selected_strategy = RecoveryStrategy.TERMINAL_FAILURE
    elif selected_strategy in {
        RecoveryStrategy.REGENERATE_FAILED_SHOTS,
        RecoveryStrategy.REPLACE_REFERENCE,
        RecoveryStrategy.RETRY,
        RecoveryStrategy.RESUME_CHECKPOINT,
    }:
        if stage is not StageName.FINAL_REVIEW:
            pending_ids = [
                item.get("shot_id") for item in artifacts if item.get("status") == StageStatus.FAILED.value
            ] or pending_ids
    if selected_strategy is RecoveryStrategy.REPLACE_REFERENCE:
        prompt_revisions = [_scope_revision(selected.prompt_changes, generation_stage, pending_ids)]
    if selected_strategy is RecoveryStrategy.HUMAN_REVIEW and not _human_allowed(state):
        selected_strategy = RecoveryStrategy.TERMINAL_FAILURE
    trace.shot_id = str((pending_ids or [""])[0])
    if selected is not None and selected_strategy is RecoveryStrategy.CHANGE_SEED:
        selected.shot_ids = list(pending_ids)
        selected.prompt_changes = {
            "shot_id": trace.shot_id,
            "instruction": selected.prompt_changes.get("instruction", ""),
            "revisions": prompt_revisions,
        }
        selected.prompt_patches = [
            PromptPatch.model_validate(patch) for revision in prompt_revisions for patch in revision["patches"]
        ]
    common = {
        "decision_traces": [trace.model_dump(mode="json")],
        "recovery_history": [
            {
                "stage": stage.value,
                "strategy": selected_strategy.value,
                "trace_id": trace.trace_id,
                "shot_ids": pending_ids,
                "shot_version": int(state.get("shot_version") or 0),
                "input_fingerprint": trace.input_fingerprint,
                "budget": trace.budget_snapshot,
                "candidate_history": [item.model_dump(mode="json") for item in trace.candidates],
                "prompt_revisions": prompt_revisions,
            }
        ],
        "selected_strategy": selected_strategy.value,
        "current_step": f"{stage.value}_recovery",
    }
    store = CheckpointStore.get(str(state.get("project_id") or ""), str(state.get("run_id") or "auto"))
    store.add_decision(trace)
    store.add_event(
        "recovery_selected",
        stage=stage.value,
        shot_ids=pending_ids,
        strategy=selected_strategy.value,
        trace_id=trace.trace_id,
        input_fingerprint=trace.input_fingerprint,
        shot_version=int(state.get("shot_version") or 0),
    )
    if selected_strategy is RecoveryStrategy.HUMAN_REVIEW:
        return {**common, "needs_human_review": True, "human_reason": trace.reason}
    if selected_strategy is RecoveryStrategy.DEGRADED_PUBLISH:
        return {
            **common,
            "run_status": RunStatus.DEGRADED.value,
            "degraded_published": True,
            "degraded_reason": trace.reason,
            "pending_recovery_target": "",
            "stage_status": {stage.value: StageStatus.DEGRADED.value},
        }
    if selected_strategy is RecoveryStrategy.TERMINAL_FAILURE:
        reason = (
            trace.reason
            if selected_strategy is (selected.strategy if selected else None)
            else "选定的恢复动作无法安全执行；已明确终止"
        )
        # 终止原因必须保留原始失败信息（如输出截断诊断），否则任务中心只能看到
        # 「没有可行候选」，无法据此给出可执行建议。
        failure_message = str(getattr(failure, "message", "") or "").strip()
        if failure_message:
            reason = f"{reason}（最近失败：{failure_message[:300]}）"
        return {**common, "run_status": RunStatus.FAILED.value, "pending_recovery_target": "", "human_reason": reason}
    if selected_strategy is RecoveryStrategy.SPLIT_SHOT:
        project_id = str(state.get("project_id") or "")
        try:
            if not pending_ids:
                raise ValueError("拆镜恢复缺少失败镜头 ID")
            split_results = _persist_recovery_splits(
                project_id,
                pending_ids,
                trace_id=trace.trace_id,
                reason=str(failure.message if failure else "自动恢复拆镜"),
            )
        except Exception as exc:
            error_id = log_failure(exc, error_type=ERROR_PIPELINE, context={"node": "split_recovery"}, log=logger)
            store.add_event(
                "recovery_apply_failed",
                stage=stage.value,
                shot_ids=pending_ids,
                strategy=selected_strategy.value,
                trace_id=trace.trace_id,
                error_id=error_id,
            )
            common["recovery_history"][0]["outcome"] = "failed"
            common["recovery_history"][0]["error_id"] = error_id
            return {
                **common,
                "selected_strategy": RecoveryStrategy.TERMINAL_FAILURE.value,
                "pending_recovery_target": "",
                "run_status": RunStatus.FAILED.value,
                "human_reason": f"拆镜恢复未能安全落库（错误编号 {error_id}）",
            }
        pending_ids = [shot_id for result in split_results for shot_id in result["shot_ids"]]
        store.detect_changes()
        store.add_event(
            "recovery_split_applied",
            stage=stage.value,
            shot_ids=pending_ids,
            strategy=selected_strategy.value,
            trace_id=trace.trace_id,
            operations=[result["operation_id"] for result in split_results],
        )
        common["recovery_history"][0]["shot_ids"] = pending_ids
        common["recovery_history"][0]["outcome"] = "applied"
        common["recovery_history"][0]["split_results"] = split_results
    attempts = dict(state.get("recovery_attempts") or {})
    attempts[stage.value] = int(attempts.get(stage.value, 0)) + 1
    return {
        **common,
        "prompt_revisions": prompt_revisions,
        "provider_switch": provider_switch,
        "pending_shot_ids": pending_ids,
        "split_recovery_shot_ids": pending_ids
        if selected_strategy is RecoveryStrategy.SPLIT_SHOT
        else state.get("split_recovery_shot_ids", []),
        "pending_recovery_target": target,
        "recovery_attempts": attempts,
        "stage_status": {stage.value: StageStatus.RECOVERING.value},
    }


def _persist_recovery_splits(
    project_id: str, shot_ids: list[str], *, trace_id: str, reason: str
) -> list[dict[str, Any]]:
    """Commit all scoped shot splits together, or leave the timeline unchanged."""

    from db import SessionLocal
    from models import Shot
    from services.shot_split_service import persist_split_shot

    db = SessionLocal()
    try:
        versions = {
            str(row.id): int(row.version or 1)
            for row in db.query(Shot).filter(Shot.project_id == project_id, Shot.id.in_(shot_ids)).all()
        }
        if set(versions) != set(shot_ids):
            raise ValueError("拆镜恢复的镜头已不存在")
        results = [
            persist_split_shot(
                db,
                project_id=project_id,
                shot_id=shot_id,
                expected_version=versions[shot_id],
                parts=2,
                reason=reason,
                operation_id=f"{trace_id}:{shot_id}",
            )
            for shot_id in shot_ids
        ]
        db.commit()
        return results
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _failure_from_stage_output(state: AgentState, stage: StageName) -> FailureRecord | None:
    """读取阶段检查点里保存的失败记录（含原始异常消息与失败分类）。

    Critic 对 LLM 失败只给出泛化消息；输出截断这类失败的诊断信息
    （finish_reason/token 数/max_tokens）保存在阶段失败记录里。
    """

    row = (state.get("stage_outputs") or {}).get(stage.value) or {}
    data = row.get("failure")
    if isinstance(data, dict):
        try:
            return FailureRecord(**data)
        except Exception:
            return None
    return None


def _failure_from_artifacts(stage: StageName, artifacts: list[dict[str, Any]]) -> FailureRecord | None:
    for item in reversed(artifacts):
        if item.get("failure"):
            data = dict(item["failure"])
            try:
                return FailureRecord(**data)
            except Exception:
                return FailureRecord(
                    kind=FailureKind(data.get("kind", FailureKind.UNKNOWN.value)),
                    stage=stage,
                    shot_id=str(item.get("shot_id") or ""),
                    message=str(data.get("message") or ""),
                )
        if item.get("status") == StageStatus.FAILED.value:
            return FailureRecord(
                kind=FailureKind.IMAGE_FAILED if stage is StageName.IMAGE_GENERATION else FailureKind.VIDEO_FAILED,
                stage=stage,
                shot_id=str(item.get("shot_id") or ""),
                message="镜头产物失败",
            )
    return None


def _failure_from_critique(stage: StageName, critique: dict[str, Any] | None) -> FailureRecord | None:
    data = critique or {}
    issues = list(data.get("issues") or [])
    first_error = next((item for item in issues if item.get("severity") == "error"), None)
    # Critic 已给出失败分类时直接采信，并携带受影响镜头与可恢复性证据。
    kind_value = str(data.get("failure_kind") or "")
    if kind_value:
        try:
            return FailureRecord(
                kind=FailureKind(kind_value),
                stage=stage,
                shot_id=str((first_error or {}).get("shot_id") or ""),
                message=str((first_error or {}).get("message") or kind_value),
                details={
                    "affected_shot_ids": list(data.get("affected_shot_ids") or []),
                    "recoverable": bool(data.get("recoverable", True)),
                },
            )
        except ValueError:
            pass
    for issue in issues:
        if issue.get("severity") == "error":
            code = str(issue.get("code") or "")
            kind = {
                "dialogue_too_long": FailureKind.DIALOGUE_TOO_LONG,
                "dialogue_duration_ratio": FailureKind.DIALOGUE_TOO_LONG,
                "shot_too_complex": FailureKind.SHOT_TOO_COMPLEX,
                "provider_reference_unsupported": FailureKind.PROVIDER_REFERENCE_UNSUPPORTED,
                "provider_capability_mismatch": FailureKind.PROVIDER_CAPABILITY_MISMATCH,
                "llm_invalid_output": FailureKind.LLM_INVALID_OUTPUT,
                "llm_output_truncated": FailureKind.LLM_OUTPUT_TRUNCATED,
                "llm_failed": FailureKind.LLM_INVALID_OUTPUT,
                "image_invalid": FailureKind.IMAGE_FAILED,
                "audio_missing": FailureKind.AUDIO_FAILED,
                # 视频结构/技术检查失败码（structural_validity + technical_quality）
                "video_invalid": FailureKind.VIDEO_FAILED,
                "video_file_missing": FailureKind.VIDEO_FAILED,
                "video_file_too_small": FailureKind.VIDEO_FAILED,
                "video_unreadable": FailureKind.VIDEO_FAILED,
                "video_stream_missing": FailureKind.VIDEO_FAILED,
                "video_duration_invalid": FailureKind.VIDEO_FAILED,
                "video_not_playable": FailureKind.VIDEO_FAILED,
                "video_duration_shorter_than_plan": FailureKind.VIDEO_FAILED,
                "video_resolution_below_minimum": FailureKind.VIDEO_FAILED,
                "video_aspect_mismatch": FailureKind.VIDEO_FAILED,
                "video_black_frames": FailureKind.VIDEO_FAILED,
                "video_frozen": FailureKind.VIDEO_FAILED,
                "audio_exceeds_picture": FailureKind.VIDEO_FAILED,
                "tail_frame_missing": FailureKind.STORAGE_FAILED,
                "render_missing": FailureKind.STORAGE_FAILED,
                "video_generation_failure": FailureKind.VIDEO_FAILED,
            }.get(code, FailureKind.QUALITY_BELOW_THRESHOLD)
            return FailureRecord(
                kind=kind, stage=stage, shot_id=str(issue.get("shot_id") or ""), message=str(issue.get("message") or "")
            )
    if not data.get("passed", True):
        return FailureRecord(
            kind=FailureKind.QUALITY_BELOW_THRESHOLD,
            stage=stage,
            message=str(data.get("failure_kind") or "阶段未通过质量门禁"),
        )
    return None


def _latest_critique(state: AgentState, stage: StageName | str) -> dict[str, Any] | None:
    key = StageName(stage).value
    rows = [item for item in state.get("critiques", []) if str(item.get("stage")) == key]
    return rows[-1] if rows else None


def _latest_artifacts(state: AgentState, stage: StageName | str) -> list[dict[str, Any]]:
    key = StageName(stage).value
    latest: dict[str, dict[str, Any]] = {}
    for item in state.get("shot_artifacts", []):
        if str(item.get("stage")) == key:
            latest[str(item.get("shot_id"))] = dict(item)
    return [latest[key_id] for key_id in sorted(latest)]


def _fanout_status(result: dict[str, Any]) -> StageStatus:
    if result.get("pending"):
        return StageStatus.RUNNING
    if result.get("degraded") or (result.get("failures") and (result.get("successes") or result.get("skipped"))):
        return StageStatus.DEGRADED
    if result.get("failures"):
        return StageStatus.FAILED
    return StageStatus.SUCCEEDED


def _shot_versions(
    project_id: str, only_ids: list[str] | None = None, *, require_storyboard: bool = False
) -> dict[str, int]:
    from db import SessionLocal
    from models import Shot

    db = SessionLocal()
    try:
        query = db.query(Shot).filter(Shot.project_id == project_id)
        if only_ids:
            query = query.filter(Shot.id.in_([str(item) for item in only_ids if item]))
        rows = query.order_by(Shot.sequence).all()
        return {
            row.id: int(row.version or 1)
            for row in rows
            if (not require_storyboard or bool(row.storyboard_path or row.image_path))
        }
    finally:
        db.close()


def _audio_shot_versions(project_id: str, only_ids: list[str] | None = None) -> dict[str, int]:
    from db import SessionLocal
    from models import Shot

    db = SessionLocal()
    try:
        query = db.query(Shot).filter(Shot.project_id == project_id)
        if only_ids:
            query = query.filter(Shot.id.in_([str(item) for item in only_ids if item]))
        return {row.id: int(row.version or 1) for row in query.order_by(Shot.sequence).all() if bool(row.dialogue)}
    finally:
        db.close()


def _db_shots(project_id: str) -> list[dict[str, Any]]:
    from db import SessionLocal
    from models import Shot

    if not project_id:
        return []
    db = SessionLocal()
    try:
        return [
            {
                "shot_id": row.id,
                "dialogue": row.dialogue,
                "shot_type": row.shot_type,
                "audio_mode": getattr(row, "audio_mode", ""),
                "continuity_profile": json.loads(row.continuity_profile or "{}"),
                "video_path": row.video_path,
                "audio_path": row.audio_path,
                "status": row.status,
                "version": row.version,
                "confirmed": bool(row.confirmed),
            }
            for row in db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
        ]
    finally:
        db.close()


def _scope_revision(revision: dict[str, Any], stage: StageName, shot_ids: list[str]) -> dict[str, Any]:
    """将选中补丁限定到本次失败镜头和实际执行阶段。"""

    patches = []
    for patch in revision.get("patches") or []:
        patch_shot = str(patch.get("shot_id") or "")
        for shot_id in [patch_shot] if patch_shot else shot_ids or [""]:
            patches.append({**patch, "shot_id": shot_id, "target_stage": stage.value})
    return {**revision, "shot_id": str((shot_ids or [revision.get("shot_id") or ""])[0]), "patches": patches}


def _shot_revisions(state: AgentState, stage: StageName, shot_id: str) -> list[dict[str, Any]]:
    """只把目标阶段、目标镜头的生成补丁交给对应 worker。"""

    revisions = []
    for revision in state.get("prompt_revisions") or []:
        patches = [
            patch
            for patch in revision.get("patches") or []
            if str(patch.get("target_stage") or "") == stage.value
            and str(patch.get("shot_id") or "") == shot_id
            and str(patch.get("field") or "") != "seed"
        ]
        if patches:
            revisions.append({**revision, "shot_id": shot_id, "patches": patches})
    return revisions


def _recovery_seed(state: AgentState, stage: StageName, shot_id: str, artifacts: list[dict[str, Any]]) -> int:
    """为当前镜头派生稳定 seed，并避开当前版本已有候选 seed。"""

    def shot_seed(item: dict[str, Any]) -> int | None:
        for patch in item.get("patches") or []:
            if patch.get("field") != "seed":
                continue
            value = patch.get("value")
            value = value.get("seed") if isinstance(value, dict) else value
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                continue
        return None

    previous = {
        int(candidate["seed"])
        for item in artifacts
        if str(item.get("shot_id") or "") == shot_id
        for candidate in (item.get("video_candidates") or [])
        if isinstance(candidate, dict) and candidate.get("seed") is not None
    }
    version = next(
        (int(item.get("shot_version") or 1) for item in artifacts if str(item.get("shot_id") or "") == shot_id),
        int(state.get("shot_version") or 1),
    )
    if stage is StageName.IMAGE_GENERATION:
        previous.add(42 + version * 100)
    previous.update(
        seed
        for revision in (state.get("prompt_revisions") or [])
        if str(revision.get("shot_id") or "") == shot_id
        for seed in [shot_seed(revision)]
        if seed is not None
    )
    for nonce in range(8):
        data = f"{state.get('project_id', '')}:{stage.value}:{shot_id}:{state.get('run_id', 'auto')}:{version}:{(state.get('recovery_attempts') or {}).get(stage.value, 0)}:{nonce}"
        seed = int.from_bytes(hashlib.sha256(data.encode("utf-8")).digest()[:4], "big") % (2**31 - 1)
        if seed not in previous:
            return seed
    return (seed + 1) % (2**31 - 1)


def _seed_override(state: AgentState, stage: StageName, shot_id: str = "") -> int | None:
    """只有本阶段选中过 seed 恢复时，为每个镜头派生稳定且不同的 seed。"""

    for revision in reversed(state.get("prompt_revisions") or []):
        for item in revision.get("patches") or []:
            if item.get("field") != "seed" or str(item.get("target_stage") or "") != stage.value:
                continue
            scoped_shot = str(item.get("shot_id") or "")
            if scoped_shot and scoped_shot != shot_id:
                continue
            value = item.get("value")
            if isinstance(value, dict) and value.get("seed") is not None:
                try:
                    return int(value["seed"])
                except (TypeError, ValueError):
                    continue
            if isinstance(value, int):
                return value
    return None


def _preferred_image_size(state: AgentState) -> str:
    switch = state.get("provider_switch") or {}
    if switch.get(f"{StageName.IMAGE_GENERATION.value}:resolution") == "540p":
        return "540x960"
    quality = default_quality_profile(state.get("quality_profile"))
    return {"540p": "540x960", "720p": "720x1280", "1080p": "1080x1920", "4k": "2160x3840"}.get(quality.resolution, "")


async def _persist_phase1_idempotent(project_id: str, state: dict[str, Any]) -> None:
    """只在没有旧成果时创建镜头；已有镜头时复用，保证断点续跑不会销毁结果。"""

    from db import SessionLocal
    from models import Shot

    db = SessionLocal()
    try:
        rows = db.query(Shot).filter(Shot.project_id == project_id).all()
        if rows and not any(row.confirmed or row.video_path for row in rows):
            return
        if rows:
            return
    finally:
        db.close()
    from api.routes.script import _persist_phase1

    db = SessionLocal()
    try:
        _persist_phase1(db, project_id, state, status="assets_ready")
    finally:
        db.close()


async def _confirm_storyboard_shots(project_id: str, artifacts: list[dict[str, Any]]) -> None:
    if not project_id:
        return
    from db import SessionLocal
    from models import Shot

    ids = {
        str(item.get("shot_id"))
        for item in artifacts
        if item.get("status") == StageStatus.SUCCEEDED.value and item.get("path")
    }
    if not ids:
        return
    db = SessionLocal()
    try:
        for shot in db.query(Shot).filter(Shot.project_id == project_id, Shot.id.in_(ids)).all():
            shot.confirmed = True
            shot.status = "storyboard_approved"
        db.commit()
    finally:
        db.close()


def _asset_project_id(project_id: str) -> str:
    from db import SessionLocal
    from models import Project

    db = SessionLocal()
    try:
        row = db.query(Project).filter(Project.id == project_id).first()
        return str(row.parent_project_id or row.id) if row else project_id
    finally:
        db.close()


def refresh_project_reference_state_for_graph(project_id: str, *, allow_needs_review: bool = True) -> dict[str, Any]:
    """图内刷新参考素材状态；auto 模式传 allow_needs_review=False 以免写入人工卡点态。"""

    from db import SessionLocal
    from services.reference_readiness_service import refresh_project_reference_state

    db = SessionLocal()
    try:
        return refresh_project_reference_state(db, project_id, allow_needs_review=allow_needs_review)
    finally:
        db.close()


def _set_project_status(project_id: str, status: str) -> None:
    if not project_id:
        return
    from db import SessionLocal
    from models import Project

    db = SessionLocal()
    try:
        row = db.query(Project).filter(Project.id == project_id).first()
        if row:
            row.status = status
            db.commit()
    finally:
        db.close()


def _final_path(project_id: str) -> str:
    from config import settings

    return str(settings.OUTPUT_DIR / "projects" / project_id / "output" / "final.mp4")


def _abort(node: str, exc) -> dict:
    """兼容旧调用：记录完整异常，只把可读的一行摘要交给状态机。"""

    if isinstance(exc, BaseException):
        error_id = log_failure(exc, error_type=ERROR_PIPELINE, context={"node": node}, log=logger)
    else:
        error_id = new_error_id()
        logger.error("自动流程节点失败 [%s] node=%s reason=%s", error_id, node, redact(exc))
    return {"errors": [f"[{node}] 执行失败（错误编号 {error_id}）"], "current_step": "aborted"}


def _review_summary(reviews: dict[str, Any]) -> str:
    parts = []
    for shot_id, review in sorted(reviews.items()):
        issues = getattr(review, "issues", None) or []
        top_issue = issues[0] if issues else getattr(review, "verdict", "failed")
        score = float(getattr(review, "overall_score", 0.0) or 0.0)
        parts.append(f"{shot_id}[{getattr(review, 'verdict', 'failed')} {score:.2f}] {top_issue}")
    return "; ".join(parts)[:500]


def _gate_failure_summary(gate: dict[str, Any]) -> str:
    failed = gate.get("failed") or []
    if not failed:
        return str(gate.get("reason") or "门禁未通过")
    return "; ".join(f"{item.get('shot_id', '?')}: {item.get('reason', '门禁未通过')}" for item in failed[:10])


async def _run_storyboard_generation_auto(project_id: str, shot_ids: list[str]) -> None:
    """Run automatic storyboard retries while tolerating legacy callables."""

    import inspect

    from api.routes import shot as shot_route

    func = shot_route._run_storyboard_generation
    kwargs = {"capability_mode": "auto"} if "capability_mode" in inspect.signature(func).parameters else {}
    await func(project_id, shot_ids, **kwargs)


def _save_quality_review_checkpoint(
    state: AgentState,
    critique: dict[str, Any],
    reviews: dict[str, Any],
    *,
    status: str,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the real per-shot quality results in the resumable stage store."""

    if base is None:
        _project_id, store, stage_base = _stage_context(state, StageName.QUALITY_REVIEW)
    else:
        _project_id = str(state.get("project_id") or "")
        store = CheckpointStore.get(_project_id, str(state.get("run_id") or "auto"))
        stage_base = base
    payload = {
        "reviews": {
            shot_id: review.to_dict() if hasattr(review, "to_dict") else dict(review)
            for shot_id, review in reviews.items()
        },
        "review_count": len(reviews),
    }
    return store.save_stage(
        StageName.QUALITY_REVIEW.value,
        status=status,
        input_fingerprint=stage_base["input_fingerprint"],
        output_fingerprint=fingerprint(payload),
        payload=payload,
        critique=critique,
        shot_version=stage_base.get("shot_version", 0),
    )


def _storyboard_structural_fallback(project_id: str, shot_ids: list[str], *, reason: str) -> dict[str, Any]:
    """视觉模型缺失时用可测量的结构门禁收口故事板阶段。

    只依据 ``validate_image_file`` 判断故事板结构可用（存在、可解码、尺寸达标）。
    结构合格即允许自动继续，但结果标记为 ``visual_pending``：视觉质量仍未评估，
    绝不冒充审核通过。结构不合格时保持失败，交给恢复阶梯处理。
    """

    from db import SessionLocal
    from models import Shot
    from services.structural_validation import validate_image_file

    db = SessionLocal()
    try:
        rows = {
            str(row.id): row
            for row in db.query(Shot).filter(Shot.project_id == project_id, Shot.id.in_(shot_ids)).all()
        }
    finally:
        db.close()
    failed: list[dict[str, str]] = []
    for shot_id in shot_ids:
        row = rows.get(shot_id)
        if row is None:
            failed.append({"shot_id": shot_id, "reason": "镜头不存在"})
            continue
        path = str(row.storyboard_path or row.image_path or "")
        check = validate_image_file(path)
        if not check.get("passed"):
            failed.append({"shot_id": shot_id, "reason": "故事板结构不合格: " + "; ".join(check.get("issues") or [])})
    if failed:
        summary = "；".join(f"{item['shot_id']}: {item['reason'][:80]}" for item in failed[:10])
        return {
            **_abort("quality_review", f"视觉模型缺失且故事板结构不合格（{reason}）: {summary}"),
            "passed": False,
            "reviews": {},
            "visual_pending": False,
        }
    return {
        "passed": True,
        "reviews": {},
        "visual_pending": True,
        "degraded": True,
        "reason": reason,
        "shot_ids": list(shot_ids),
        "evidence": {
            "kind": "structural_only_gate",
            "checked": "storyboard_image_structure",
            "shot_count": len(shot_ids),
        },
    }


async def _run_storyboard_quality_gate(
    project_id: str,
    shot_ids: list[str] | None = None,
    *,
    allow_human: bool = False,
    allow_visual_pending: bool = False,
) -> dict[str, Any]:
    """Run persisted storyboard quality service.

    默认 fail-closed：审核能力未配置时既不批准也不放行。``allow_visual_pending``
    （仅显式 auto 模式 + 策略允许）时，改用可测量的结构门禁收口：故事板结构可用
    即继续，视觉质量如实记为 pending 并降级，绝不宣称视觉通过。
    """
    from api.routes.shot import prepare_storyboard_quality_retry
    from config import settings
    from services.quality_review_service import STAGE_STORYBOARD, quality_review_service

    shot_ids = list(shot_ids or _shot_ids(project_id))
    if not shot_ids:
        return {**_abort("quality_review", "无镜头可进行故事板质量审核"), "passed": False, "reviews": {}}
    capability = quality_review_service.storyboard_capability()
    if not capability.get("supported"):
        reason = str(capability.get("reason") or "unsupported")
        # 先如实落库 unsupported 审核记录（界面可见原因），再决定是终止还是降级继续。
        await quality_review_service.record_unsupported_reviews(project_id, shot_ids, STAGE_STORYBOARD, reason)
        if allow_visual_pending:
            return _storyboard_structural_fallback(project_id, shot_ids, reason=reason)
        if allow_human:
            await quality_review_service.mark_shots_needs_human_review(shot_ids)
        result = {
            **_abort("quality_review", f"质量审核能力未配置，自动模式不得批准故事板（{reason}）"),
            "passed": False,
            "reviews": {},
        }
        if allow_human:
            result["needs_human_review"] = True
        return result
    max_retries = max(0, int(settings.QUALITY_STORYBOARD_MAX_RETRIES))
    last_reviews: dict[str, Any] = {}
    for attempt in range(max_retries + 1):
        reviews = {shot_id: await quality_review_service.review_storyboard_shot(shot_id) for shot_id in shot_ids}
        last_reviews = reviews
        errored = {shot_id: review for shot_id, review in reviews.items() if review.verdict == "error"}
        if errored:
            return {
                **_abort("quality_review", f"质量审核 Provider 调用失败: {_review_summary(errored)}"),
                "passed": False,
                "reviews": reviews,
            }
        unsupported = {shot_id: review for shot_id, review in reviews.items() if review.verdict == "unsupported"}
        if unsupported:
            if allow_human:
                await quality_review_service.mark_shots_needs_human_review(sorted(unsupported))
            result = {
                **_abort("quality_review", f"质量审核存在未检测维度: {_review_summary(unsupported)}"),
                "passed": False,
                "reviews": reviews,
            }
            if allow_human:
                result["needs_human_review"] = True
            return result
        failed = {shot_id: review for shot_id, review in reviews.items() if not review.passed}
        if not failed:
            return {"passed": True, "reviews": reviews, "attempt": attempt}
        if attempt >= max_retries:
            break
        retry_ids = [
            shot_id for shot_id in sorted(failed) if prepare_storyboard_quality_retry(shot_id, failed[shot_id])
        ]
        if not retry_ids:
            break
        try:
            await _run_storyboard_generation_auto(project_id, retry_ids)
        except Exception as exc:
            return {**_abort("quality_review", exc), "passed": False, "reviews": reviews}
    if allow_human:
        await quality_review_service.mark_shots_needs_human_review(sorted(last_reviews))
    result = {
        **_abort(
            "quality_review", f"以下镜头未通过质量审核（已重试 {max_retries} 次）: {_review_summary(last_reviews)}"
        ),
        "passed": False,
        "reviews": last_reviews,
    }
    if allow_human:
        result["needs_human_review"] = True
    return result


def _shot_ids(project_id: str) -> list[str]:
    return list(_shot_versions(project_id))


def _previous_last_frames(project_id: str) -> dict[str, str]:
    from db import SessionLocal
    from models import Shot

    db = SessionLocal()
    try:
        shots = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
        frames: dict[str, str] = {}
        previous = ""
        for shot in shots:
            if previous:
                frames[shot.id] = previous
            previous = shot.last_frame_path or shot.storyboard_path or shot.image_path or ""
        return frames
    finally:
        db.close()


def _project_failed(project_id: str) -> bool:
    from db import SessionLocal
    from models import Project

    db = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        return bool(project and project.status == "error")
    finally:
        db.close()


def _has_unfinished_videos(project_id: str) -> bool:
    from db import SessionLocal
    from models import Shot

    db = SessionLocal()
    try:
        shots = db.query(Shot).filter(Shot.project_id == project_id).all()
        return (not shots) or any(not s.video_path for s in shots)
    finally:
        db.close()


async def _auto_approve_storyboard(state: AgentState) -> dict:
    """兼容旧自动审核入口：结构门禁 + 单镜头重算，不自动批准不合格结果。"""

    if state.get("errors"):
        return {}
    from db import SessionLocal
    from models import Shot
    from services.structural_validation import validate_image_file

    project_id = state["project_id"]
    reference_gate = _reference_gate_for_state(state)
    if reference_gate.get("blocking"):
        return _abort(
            "auto_approve_storyboard",
            "一致性参考素材未达到自动成片要求，已阻止故事板批准："
            + ", ".join(item.get("name") or item.get("asset_id") for item in reference_gate.get("blocking_items", [])),
        )

    def _structural_failures(shot_rows) -> list[str]:
        failed = []
        for shot in shot_rows:
            path = shot.storyboard_path or shot.image_path
            if not path or not validate_image_file(path)["passed"]:
                failed.append(shot.id)
        return failed

    db = SessionLocal()
    try:
        shots = db.query(Shot).filter(Shot.project_id == project_id).all()
        without_image = [shot.id for shot in shots if not (shot.storyboard_path or shot.image_path)]
        if without_image:
            return _abort("auto_approve_storyboard", f"仍有镜头未生成故事板: {', '.join(without_image)}")
        stale_shots = [shot.id for shot in shots if shot.media_stale]
        if stale_shots:
            return _abort("auto_approve_storyboard", f"以下镜头参数已修改、素材待重新生成: {', '.join(stale_shots)}")
        failed_once = _structural_failures(shots)
    finally:
        db.close()

    if failed_once:
        try:
            await _run_storyboard_generation_auto(project_id, list(failed_once))
        except Exception as exc:
            return _abort("auto_approve_storyboard", exc)

        # A retry is only a chance to repair the file. Re-run the structural
        # check before invoking any semantic/VLM provider; malformed output
        # must fail locally and must never trigger a network review call.
        db = SessionLocal()
        try:
            shots = db.query(Shot).filter(Shot.project_id == project_id).all()
            failed_after_retry = _structural_failures(shots)
        finally:
            db.close()
        if failed_after_retry:
            return _abort(
                "auto_approve_storyboard",
                f"以下镜头故事板未通过结构检查（已重试一次仍失败）: {', '.join(failed_after_retry)}",
            )

    allow_human = _legacy_human_allowed(state)
    gate = await _run_storyboard_quality_gate(project_id, allow_human=allow_human)
    if gate.get("errors") or not gate.get("passed"):
        return gate

    db = SessionLocal()
    try:
        shots = db.query(Shot).filter(Shot.project_id == project_id).all()
        failed = _structural_failures(shots)
        if failed:
            return _abort(
                "auto_approve_storyboard", f"以下镜头故事板未通过结构检查（已重试一次仍失败）: {', '.join(failed)}"
            )
        for shot in shots:
            shot.confirmed = True
            shot.status = "storyboard_approved"
        db.commit()
    finally:
        db.close()
    return {"current_step": "auto_approve_storyboard"}


_NODE_FUNCTIONS = {
    "director_planning": _director_planning,
    "director_review": _director_review,
    "director_decision": _director_decision,
    "director_recovery": _director_recovery,
    "storyboard_design": _storyboard_design,
    "storyboard_review": _storyboard_review,
    "storyboard_decision": _storyboard_decision,
    "storyboard_recovery": _storyboard_recovery,
    "asset_preparation": _asset_preparation,
    "asset_review": _asset_review,
    "asset_decision": _asset_decision,
    "asset_recovery": _asset_recovery,
    "image_generation": _image_generation_fan_out,
    "image_generation_fan_out": _image_generation_fan_out,
    "image_generation_fan_in": _image_generation_fan_in,
    "image_review": _image_review,
    "image_decision": _image_decision,
    "image_recovery": _image_recovery,
    "quality_review": _quality_review,
    "quality_critic": _quality_critic,
    "quality_decision": _quality_decision,
    "quality_recovery": _quality_recovery,
    "audio_production": _audio_production,
    "audio_review": _audio_review,
    "audio_decision": _audio_decision,
    "audio_recovery": _audio_recovery,
    "video_generation": _video_generation_fan_out,
    "video_generation_fan_out": _video_generation_fan_out,
    "video_generation_fan_in": _video_generation_fan_in,
    "video_generation_review": _video_generation_review,
    "video_generation_decision": _video_generation_decision,
    "video_generation_recovery": _video_generation_recovery,
    "video_review": _video_review,
    "video_critic": _video_critic,
    "video_decision": _video_decision,
    "video_recovery": _video_recovery,
    "edit_composition": _compose,
    "compose": _compose,
    "edit_review": _edit_review,
    "edit_decision": _edit_decision,
    "edit_recovery": _edit_recovery,
    "final_review": _final_review,
    "final_critic": _final_critic,
    "final_decision": _final_decision,
    "final_recovery": _final_recovery,
    "auto_abort": _auto_abort,
    "degraded_publish": _degraded_publish,
    "human_gate": _human_gate,
}


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph().compile()
    return _graph


__all__ = ["GRAPH_NODE_META", "GRAPH_STAGE_NODE_NAMES", "GRAPH_STAGE_ORDER", "build_graph", "get_graph"]
