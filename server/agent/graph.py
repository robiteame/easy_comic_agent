"""可分析、决策、修改、恢复的 LangGraph 生成 Agent。

旧版是五个节点线性调用；本版把流程拆成九个明确阶段，并在阶段间插入
Critic/Reviewer 与恢复决策。逐镜头生成使用 fan-out/fan-in：一个镜头失败只会
进入自己的恢复队列，已经成功的结果保留且不会重算。

所有节点通过 ``agent.checkpoints.CheckpointStore`` 保存输入指纹、输出指纹、
Shot.version、候选、评分、失败分类和 DecisionTrace，因此可幂等恢复、任务续跑、
用户中途修改检测和局部重算。可视化结构来自本文件的 GRAPH_NODE_META + build_graph。
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from services.error_reporter import ERROR_PIPELINE, log_failure, new_error_id, redact

from .checkpoints import CheckpointStore, fingerprint
from .contracts import (
    FailureKind,
    FailureRecord,
    QualityProfileName,
    RecoveryStrategy,
    RunStatus,
    StageName,
    StageStatus,
    default_quality_profile,
    ensure_transition,
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
)
from .decision import budget_snapshot, choose_recovery, provider_profiles
from .shot_work import (
    generate_audio_shot,
    generate_storyboard_shot,
    generate_video_shot,
    run_shot_fanout,
)
from .state import AgentState

logger = logging.getLogger(__name__)

GRAPH_STAGE_ORDER: tuple[str, ...] = tuple(stage.value for stage in StageName)

GRAPH_NODE_META: dict[str, dict] = {
    "director_planning": {"label": "导演规划", "type": "process", "description": "解析剧本、人物、场景、叙事目标与预算约束，产出导演计划"},
    "director_review": {"label": "导演 Critic", "type": "critic", "description": "检查角色/场景覆盖和逻辑问题，提出具体修改"},
    "director_decision": {"label": "导演决策", "type": "decision", "description": "按失败分类、Provider 能力、成本和剩余预算选择恢复策略"},
    "director_recovery": {"label": "导演恢复", "type": "recovery", "description": "修改 Prompt、切换 Provider 或转人工后局部重算"},
    "storyboard_design": {"label": "分镜设计", "type": "process", "description": "生成镜头并自动拆合动作节拍、对白和时长"},
    "storyboard_review": {"label": "分镜 Critic", "type": "critic", "description": "检查镜头规则、对白密度、连续性和可执行性"},
    "storyboard_decision": {"label": "分镜决策", "type": "decision", "description": "选择拆分镜头、修改 Prompt、合并镜头或人工审核"},
    "storyboard_recovery": {"label": "分镜恢复", "type": "recovery", "description": "把 Critic 修改写回镜头/Prompt 后重算受影响部分"},
    "asset_preparation": {"label": "素材准备", "type": "process", "description": "生成角色三视图、场景基准图和连续性参考，并记录能力限制"},
    "asset_review": {"label": "素材 Critic", "type": "critic", "description": "检查素材覆盖、参考图兼容性与版本一致性"},
    "asset_decision": {"label": "素材决策", "type": "decision", "description": "替换参考、切换支持参考图的 Provider 或降级/人工审核"},
    "asset_recovery": {"label": "素材恢复", "type": "recovery", "description": "局部补生成缺失参考，不重跑已成功素材"},
    "image_generation": {"label": "图像生成 fan-out", "type": "fanout", "description": "按镜头版本分发图像任务，逐镜头保存检查点"},
    "image_generation_fan_out": {"label": "图像 fan-out（兼容）", "type": "fanout", "description": "兼容入口，与图像生成阶段执行同一逐镜头 worker"},
    "image_generation_fan_in": {"label": "图像 fan-in", "type": "fanin", "description": "聚合成功、失败、降级和跳过镜头，保留局部成功"},
    "quality_review": {"label": "质量审核", "type": "critic", "description": "对故事板候选评分，给出证据和可执行修改"},
    "quality_decision": {"label": "质量决策", "type": "decision", "description": "只重生成失败/低分镜头，或进入视频阶段"},
    "quality_recovery": {"label": "质量恢复", "type": "recovery", "description": "改 Prompt、换参考、换 Provider、降分辨率或局部补拍"},
    "video_generation": {"label": "视频生成 fan-out", "type": "fanout", "description": "逐镜头生成视频，失败镜头独立重试/补拍"},
    "video_generation_fan_out": {"label": "视频 fan-out（兼容）", "type": "fanout", "description": "兼容入口，与视频生成阶段执行同一逐镜头 worker"},
    "video_generation_fan_in": {"label": "视频 fan-in", "type": "fanin", "description": "聚合视频结果，不让一个失败镜头使成功镜头失效"},
    "video_review": {"label": "视频 Critic", "type": "critic", "description": "检查视频结构、时长、连续性和 Provider 能力"},
    "video_decision": {"label": "视频决策", "type": "decision", "description": "切换 Provider、降分辨率、只补拍失败镜头或人工审核"},
    "video_recovery": {"label": "视频恢复", "type": "recovery", "description": "执行视频恢复策略并记录成本/时长/选择原因"},
    "audio_production": {"label": "音频制作", "type": "process", "description": "逐镜头生成/复用配音，过长对白自动拆句或转人工"},
    "audio_review": {"label": "音频 Critic", "type": "critic", "description": "检查对白长度、TTS 产物、音色和混音准备度"},
    "audio_decision": {"label": "音频决策", "type": "decision", "description": "拆句、局部重配音、替换 Provider 或人工改词"},
    "audio_recovery": {"label": "音频恢复", "type": "recovery", "description": "只补拍失败音频并保持其它镜头不变"},
    "edit_composition": {"label": "剪辑合成", "type": "output", "description": "按版本校验后的镜头清单合成成片，支持降级状态"},
    "compose": {"label": "剪辑合成（兼容）", "type": "output", "description": "兼容入口，与剪辑合成阶段执行同一渲染任务"},
    "final_review": {"label": "成片复审", "type": "critic", "description": "从叙事、视觉、音频、节奏和人工反馈层面复审成片"},
    "final_decision": {"label": "成片决策", "type": "decision", "description": "发布、局部重算或明确转人工"},
    "final_recovery": {"label": "成片恢复", "type": "recovery", "description": "把成片级反馈转成具体镜头/音频/剪辑重算"},
    "human_gate": {"label": "人工审核", "type": "human", "description": "明确暂停，保留检查点和待处理原因，人工确认后续跑"},
}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("director_planning", _director_planning)
    graph.add_node("director_review", _director_review)
    graph.add_node("director_decision", _director_decision)
    graph.add_node("director_recovery", _director_recovery)
    graph.add_node("storyboard_design", _storyboard_design)
    graph.add_node("storyboard_review", _storyboard_review)
    graph.add_node("storyboard_decision", _storyboard_decision)
    graph.add_node("storyboard_recovery", _storyboard_recovery)
    graph.add_node("asset_preparation", _asset_preparation)
    graph.add_node("asset_review", _asset_review)
    graph.add_node("asset_decision", _asset_decision)
    graph.add_node("asset_recovery", _asset_recovery)
    graph.add_node("image_generation", _image_generation_fan_out)
    graph.add_node("image_generation_fan_out", _image_generation_fan_out)
    graph.add_node("image_generation_fan_in", _image_generation_fan_in)
    graph.add_node("quality_review", _quality_review)
    graph.add_node("quality_decision", _quality_decision)
    graph.add_node("quality_recovery", _quality_recovery)
    graph.add_node("video_generation", _video_generation_fan_out)
    graph.add_node("video_generation_fan_out", _video_generation_fan_out)
    graph.add_node("video_generation_fan_in", _video_generation_fan_in)
    graph.add_node("video_review", _video_review)
    graph.add_node("video_decision", _video_decision)
    graph.add_node("video_recovery", _video_recovery)
    graph.add_node("audio_production", _audio_production)
    graph.add_node("audio_review", _audio_review)
    graph.add_node("audio_decision", _audio_decision)
    graph.add_node("audio_recovery", _audio_recovery)
    graph.add_node("edit_composition", _compose)
    graph.add_node("compose", _compose)
    graph.add_node("final_review", _final_review)
    graph.add_node("final_decision", _final_decision)
    graph.add_node("final_recovery", _final_recovery)
    graph.add_node("human_gate", _human_gate)

    graph.add_edge(START, "director_planning")
    graph.add_edge("director_planning", "director_review")
    graph.add_conditional_edges("director_review", _route_director_review, {"decision": "director_decision"})
    graph.add_conditional_edges("director_decision", _route_director_decision, {"recover": "director_recovery", "next": "storyboard_design", "human": "human_gate"})
    graph.add_conditional_edges("director_recovery", _route_director_recovery, {"retry": "director_planning", "human": "human_gate"})

    graph.add_edge("storyboard_design", "storyboard_review")
    graph.add_conditional_edges("storyboard_review", _route_storyboard_review, {"decision": "storyboard_decision"})
    graph.add_conditional_edges("storyboard_decision", _route_storyboard_decision, {"recover": "storyboard_recovery", "next": "asset_preparation", "human": "human_gate"})
    graph.add_conditional_edges("storyboard_recovery", _route_storyboard_recovery, {"retry": "storyboard_design", "human": "human_gate"})

    graph.add_edge("asset_preparation", "asset_review")
    graph.add_conditional_edges("asset_review", _route_asset_review, {"decision": "asset_decision"})
    graph.add_conditional_edges("asset_decision", _route_asset_decision, {"recover": "asset_recovery", "next": "image_generation", "human": "human_gate"})
    graph.add_conditional_edges("asset_recovery", _route_asset_recovery, {"retry": "asset_preparation", "human": "human_gate"})

    graph.add_edge("image_generation", "image_generation_fan_in")
    graph.add_edge("image_generation_fan_out", "image_generation_fan_in")
    graph.add_edge("image_generation_fan_in", "quality_review")
    graph.add_conditional_edges("quality_review", _route_quality_review, {"decision": "quality_decision"})
    graph.add_conditional_edges("quality_decision", _route_quality_decision, {"recover": "quality_recovery", "next": "video_generation", "human": "human_gate"})
    graph.add_conditional_edges("quality_recovery", _route_quality_recovery, {"retry": "image_generation", "storyboard": "storyboard_design", "human": "human_gate"})

    graph.add_edge("video_generation", "video_generation_fan_in")
    graph.add_edge("video_generation_fan_out", "video_generation_fan_in")
    graph.add_edge("video_generation_fan_in", "video_review")
    graph.add_conditional_edges("video_review", _route_video_review, {"decision": "video_decision"})
    graph.add_conditional_edges("video_decision", _route_video_decision, {"recover": "video_recovery", "next": "audio_production", "human": "human_gate"})
    graph.add_conditional_edges("video_recovery", _route_video_recovery, {"retry": "video_generation", "storyboard": "storyboard_design", "human": "human_gate"})

    graph.add_edge("audio_production", "audio_review")
    graph.add_conditional_edges("audio_review", _route_audio_review, {"decision": "audio_decision"})
    graph.add_conditional_edges("audio_decision", _route_audio_decision, {"recover": "audio_recovery", "next": "edit_composition", "human": "human_gate"})
    graph.add_conditional_edges("audio_recovery", _route_audio_recovery, {"retry": "audio_production", "storyboard": "storyboard_design", "human": "human_gate"})

    graph.add_edge("edit_composition", "final_review")
    graph.add_edge("compose", "final_review")
    graph.add_conditional_edges("final_review", _route_final_review, {"decision": "final_decision"})
    graph.add_conditional_edges("final_decision", _route_final_decision, {"recover": "final_recovery", "done": END, "human": "human_gate"})
    graph.add_conditional_edges("final_recovery", _route_final_recovery, {"image": "image_generation", "video": "video_generation", "audio": "audio_production", "compose": "compose", "human": "human_gate"})
    graph.add_edge("human_gate", END)
    return graph


# --- stage nodes ---


def _reference_gate(project_id: str, *, allow_degraded: bool = False) -> dict[str, Any]:
    from db import SessionLocal
    from services.reference_readiness_service import ensure_generation_gate

    db = SessionLocal()
    try:
        return ensure_generation_gate(db, project_id, allow_degraded=allow_degraded)
    finally:
        db.close()


def _reference_review_update(project_id: str, report: dict[str, Any]) -> dict[str, Any]:
    """自动流程遇到缺失参考时转人工，不把失败伪装成阶段成功。"""

    affected = report.get("affected_shot_ids", [])
    reason = (
        "一致性参考素材未达到自动成片要求，已转人工审核"
        + (f"；影响{report.get('shot_range', '')}" if report.get("shot_range") else "")
    )
    return {
        "needs_human_review": True,
        "human_reason": reason,
        "consistency_report": report,
        "affected_shot_ids": affected,
        "stage_status": {StageName.ASSET_PREPARATION.value: StageStatus.DEGRADED.value},
        "current_step": "asset_review",
    }



async def _director_planning(state: AgentState) -> dict:
    project_id, store, base = _stage_context(state, StageName.DIRECTOR_PLANNING)
    if store.stage_is_reusable(StageName.DIRECTOR_PLANNING.value, base["input_fingerprint"]):
        return _restore_stage(state, store, StageName.DIRECTOR_PLANNING)
    try:
        from agent.nodes import script_parser

        parsed = await script_parser.run({**base["initial_state"], **state, "prompt_revisions": state.get("prompt_revisions") or []})
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
        return _failed_stage(state, store, StageName.DIRECTOR_PLANNING, base, exc, critique=critique_llm_failure(exc, stage=StageName.DIRECTOR_PLANNING))


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

        generated = await storyboard_gen.run({**base["initial_state"], **state, "prompt_revisions": state.get("prompt_revisions") or []})
        payload = {"shots": generated.get("shots", []), "timing_plan": generated.get("timing_plan", {})}
        critique = critique_storyboard({"shots": payload["shots"]})
        return _save_stage(state, store, StageName.STORYBOARD_DESIGN, base, payload, critique=critique)
    except Exception as exc:
        return _failed_stage(state, store, StageName.STORYBOARD_DESIGN, base, exc, critique=critique_llm_failure(exc, stage=StageName.STORYBOARD_DESIGN))


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
        reference_report = refresh_project_reference_state_for_graph(project_id)
        reference_supported = any(item.supports_reference_images for item in provider_profiles("image", reference_required=True) if item.available)
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
        return _failed_stage(state, store, StageName.ASSET_PREPARATION, base, exc, critique=critique_assets(state, reference_supported=False))


async def _asset_review(state: AgentState) -> dict:
    reference_supported = any(item.supports_reference_images for item in provider_profiles("image", reference_required=True) if item.available)
    critique = critique_assets(state, reference_supported=reference_supported)
    report = state.get("consistency_report") or _reference_gate(str(state.get("project_id") or ""))
    if report.get("blocking"):
        return {
            "critiques": [critique.model_dump(mode="json")],
            "consistency_report": report,
            "needs_human_review": True,
            "human_reason": "一致性参考素材未达到自动成片要求，已转人工审核",
            "current_step": "asset_review",
        }
    return {"critiques": [critique.model_dump(mode="json")], "consistency_report": report, "current_step": "asset_review"}


async def _asset_decision(state: AgentState) -> dict:
    report = state.get("consistency_report") or _reference_gate(str(state.get("project_id") or ""))
    if report.get("blocking"):
        return _reference_review_update(str(state.get("project_id") or ""), report)
    return _decision_node(state, StageName.ASSET_PREPARATION, next_target="image_generation")


async def _asset_recovery(state: AgentState) -> dict:
    return _recovery_node(state, StageName.ASSET_PREPARATION, default_target="asset_preparation")


async def _image_generation_fan_out(state: AgentState) -> dict:
    project_id, store, base = _stage_context(state, StageName.IMAGE_GENERATION)
    reference_gate = _reference_gate(project_id)
    if reference_gate.get("blocking"):
        return _reference_review_update(project_id, reference_gate)
    shot_versions = _shot_versions(project_id, state.get("pending_shot_ids"))
    if not shot_versions:
        return {"stage_status": {StageName.IMAGE_GENERATION.value: StageStatus.FAILED.value}, "current_step": "image_generation_fan_out"}
    provider = str((state.get("provider_switch") or {}).get(StageName.IMAGE_GENERATION.value) or "")
    preferred_size = _preferred_image_size(state)
    worker = partial(generate_storyboard_shot, project_id=project_id, provider_override=provider, preferred_size=preferred_size)
    result = await run_shot_fanout(
        project_id=project_id,
        shot_versions=shot_versions,
        stage=StageName.IMAGE_GENERATION,
        worker=worker,
        checkpoint=store,
        concurrency=3,
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
        "stage_status": {StageName.IMAGE_GENERATION.value: status.value},
        "stage_outputs": {StageName.IMAGE_GENERATION.value: row},
        "shot_artifacts": result["artifacts"],
        "successful_shot_ids": [item["shot_id"] for item in result["successes"]],
        "failed_shot_ids": [item["shot_id"] for item in result["failures"]],
        "degraded_shot_ids": [item["shot_id"] for item in result["degraded"]],
        "pending_shot_ids": [],
        "current_step": "image_generation_fan_out",
    }


async def _image_generation_fan_in(state: AgentState) -> dict:
    artifacts = _latest_artifacts(state, StageName.IMAGE_GENERATION)
    return {
        "shot_artifacts": artifacts,
        "successful_shot_ids": [item.get("shot_id") for item in artifacts if item.get("status") == StageStatus.SUCCEEDED.value],
        "failed_shot_ids": [item.get("shot_id") for item in artifacts if item.get("status") == StageStatus.FAILED.value],
        "degraded_shot_ids": [item.get("shot_id") for item in artifacts if item.get("status") == StageStatus.DEGRADED.value],
        "current_step": "image_generation_fan_in",
    }


async def _quality_review(state: AgentState) -> dict:
    project_id = str(state.get("project_id") or "")
    reference_gate = _reference_gate(str(state.get("project_id") or ""))
    if reference_gate.get("blocking"):
        return _reference_review_update(str(state.get("project_id") or ""), reference_gate)
    artifacts = _latest_artifacts(state, StageName.IMAGE_GENERATION)
    structural = critique_images(artifacts)
    if not structural.passed:
        return {
            "critiques": [structural.model_dump(mode="json")],
            "stage_status": {StageName.QUALITY_REVIEW.value: StageStatus.FAILED.value},
            "current_step": "quality_review",
        }
    gate = await _run_storyboard_quality_gate(project_id)
    reviews = gate.get("reviews") or {}
    passed = bool(gate.get("passed")) and not gate.get("errors")
    quality_critique = {
        "stage": StageName.QUALITY_REVIEW.value,
        "passed": passed,
        "score": round(sum(float(review.overall_score) for review in reviews.values()) / max(1, len(reviews)), 3),
        "issues": [
            {"code": "quality_review_failed", "severity": "error", "message": f"镜头 {shot_id} 未通过质量审核: {review.suggestion or '请查看质量审核记录'}", "shot_id": shot_id}
            for shot_id, review in reviews.items() if not review.passed
        ],
        "proposed_changes": [review.suggestion for review in reviews.values() if review.suggestion],
        "source": "quality_review_service",
    }
    status = StageStatus.SUCCEEDED.value if passed else StageStatus.WAITING_HUMAN.value
    update: dict[str, Any] = {
        "critiques": [structural.model_dump(mode="json"), quality_critique],
        "stage_status": {StageName.QUALITY_REVIEW.value: status},
        "current_step": "quality_review",
    }
    if passed:
        await _confirm_storyboard_shots(project_id, artifacts)
    if gate.get("errors"):
        update.update({key: value for key, value in gate.items() if key in {"errors", "needs_human_review", "human_reason"}})
    _save_quality_review_checkpoint(state, quality_critique, reviews, status=status)
    return update


async def _quality_decision(state: AgentState) -> dict:
    return _decision_node(state, StageName.QUALITY_REVIEW, next_target="video_generation", artifacts_stage=StageName.IMAGE_GENERATION)


async def _quality_recovery(state: AgentState) -> dict:
    return _recovery_node(state, StageName.QUALITY_REVIEW, default_target="image_generation", artifacts_stage=StageName.IMAGE_GENERATION)


async def _video_generation_fan_out(state: AgentState) -> dict:
    project_id, store, base = _stage_context(state, StageName.VIDEO_GENERATION)
    from services.quality_review_service import quality_review_service

    storyboard_gate = quality_review_service.storyboard_gate_status(project_id)
    if not storyboard_gate.get("ok"):
        return {
            **_abort("video_generation", "故事板质量门禁未通过，自动模式禁止生成视频: " + _gate_failure_summary(storyboard_gate)),
            "stage_status": {StageName.VIDEO_GENERATION.value: StageStatus.WAITING_HUMAN.value},
            "needs_human_review": True,
            "human_reason": "故事板质量门禁未通过",
            "current_step": "video_generation",
        }
    shot_versions = _shot_versions(project_id, state.get("pending_shot_ids"), require_storyboard=True)
    if not shot_versions:
        return {"stage_status": {StageName.VIDEO_GENERATION.value: StageStatus.FAILED.value}, "current_step": "video_generation_fan_out"}
    provider = str((state.get("provider_switch") or {}).get(StageName.VIDEO_GENERATION.value) or "")
    resolution = str(state.get("resolution") or "720p")
    worker = partial(generate_video_shot, project_id=project_id, provider_override=provider, resolution_override=resolution)
    result = await run_shot_fanout(project_id=project_id, shot_versions=shot_versions, stage=StageName.VIDEO_GENERATION, worker=worker, checkpoint=store, concurrency=2)
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
        "stage_status": {StageName.VIDEO_GENERATION.value: status.value},
        "stage_outputs": {StageName.VIDEO_GENERATION.value: row},
        "shot_artifacts": result["artifacts"],
        "successful_shot_ids": [item["shot_id"] for item in result["successes"]],
        "failed_shot_ids": [item["shot_id"] for item in result["failures"]],
        "degraded_shot_ids": [item["shot_id"] for item in result["degraded"]],
        "pending_shot_ids": [],
        "current_step": "video_generation_fan_out",
    }


async def _video_generation_fan_in(state: AgentState) -> dict:
    artifacts = _latest_artifacts(state, StageName.VIDEO_GENERATION)
    return {
        "shot_artifacts": artifacts,
        "successful_shot_ids": [item.get("shot_id") for item in artifacts if item.get("status") == StageStatus.SUCCEEDED.value],
        "failed_shot_ids": [item.get("shot_id") for item in artifacts if item.get("status") == StageStatus.FAILED.value],
        "degraded_shot_ids": [item.get("shot_id") for item in artifacts if item.get("status") == StageStatus.DEGRADED.value],
        "current_step": "video_generation_fan_in",
    }


async def _video_review(state: AgentState) -> dict:
    quality = await _review_shot_videos(state)
    critique = critique_videos(_latest_artifacts(state, StageName.VIDEO_GENERATION))
    update = dict(quality)
    update.setdefault("critiques", []).append(critique.model_dump(mode="json"))
    update["current_step"] = "video_review"
    if quality.get("errors"):
        update["stage_status"] = {StageName.VIDEO_GENERATION.value: StageStatus.WAITING_HUMAN.value}
    else:
        update["stage_status"] = {StageName.VIDEO_GENERATION.value: StageStatus.SUCCEEDED.value}
    return update


async def _video_decision(state: AgentState) -> dict:
    return _decision_node(state, StageName.VIDEO_GENERATION, next_target="audio_production", artifacts_stage=StageName.VIDEO_GENERATION)


async def _video_recovery(state: AgentState) -> dict:
    return _recovery_node(state, StageName.VIDEO_GENERATION, default_target="video_generation", artifacts_stage=StageName.VIDEO_GENERATION)


async def _audio_production(state: AgentState) -> dict:
    project_id, store, base = _stage_context(state, StageName.AUDIO_PRODUCTION)
    shot_versions = _audio_shot_versions(project_id, state.get("pending_shot_ids"))
    worker = partial(generate_audio_shot, project_id=project_id)
    result = await run_shot_fanout(project_id=project_id, shot_versions=shot_versions, stage=StageName.AUDIO_PRODUCTION, worker=worker, checkpoint=store, concurrency=3)
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
        "stage_status": {StageName.AUDIO_PRODUCTION.value: status.value},
        "stage_outputs": {StageName.AUDIO_PRODUCTION.value: row},
        "shot_artifacts": result["artifacts"],
        "successful_shot_ids": [item["shot_id"] for item in result["successes"]],
        "failed_shot_ids": [item["shot_id"] for item in result["failures"]],
        "pending_shot_ids": [],
        "current_step": "audio_production",
    }


async def _audio_review(state: AgentState) -> dict:
    critique = critique_audio(_db_shots(state.get("project_id", "")), _latest_artifacts(state, StageName.AUDIO_PRODUCTION))
    return {"critiques": [critique.model_dump(mode="json")], "current_step": "audio_review"}


async def _audio_decision(state: AgentState) -> dict:
    return _decision_node(state, StageName.AUDIO_PRODUCTION, next_target="compose", artifacts_stage=StageName.AUDIO_PRODUCTION)


async def _audio_recovery(state: AgentState) -> dict:
    return _recovery_node(state, StageName.AUDIO_PRODUCTION, default_target="audio_production", artifacts_stage=StageName.AUDIO_PRODUCTION)


async def _generate_shot_videos(state: AgentState) -> dict:
    """兼容旧自动视频入口：逐镜头独立尝试一次，再只重试失败镜头。"""

    if state.get("errors"):
        return {}
    from api.routes.shot import _run_single_shot_video
    from services.quality_review_service import quality_review_service

    project_id = state["project_id"]
    gate = quality_review_service.storyboard_gate_status(project_id)
    if not gate.get("ok"):
        return _abort("generate_shot_videos", "故事板质量门禁未通过，自动模式禁止生成视频: " + _gate_failure_summary(gate))
    failures: dict[str, str] = {}
    for shot_id in _shot_ids(project_id):
        try:
            await _run_single_shot_video(shot_id, force=False, capability_mode="auto")
        except Exception as exc:
            failures[shot_id] = str(exc)
    if failures:
        retried = list(failures)
        for shot_id in retried:
            try:
                await _run_single_shot_video(shot_id, force=True, capability_mode="auto")
                failures.pop(shot_id, None)
            except Exception as exc:
                failures[shot_id] = str(exc)
    if failures:
        summary = "; ".join(f"{shot_id}: {reason[:120]}" for shot_id, reason in sorted(failures.items()))
        return _abort("generate_shot_videos", f"以下镜头视频生成失败（已单独重试一次）: {summary}")
    if _has_unfinished_videos(project_id):
        return _abort("generate_shot_videos", "仍有镜头视频未生成")
    return {"current_step": "generate_shot_videos"}


async def _review_shot_videos(state: AgentState) -> dict:
    """兼容旧自动入口：真实审核视频，按镜头重试，失败即转人工。"""
    if state.get("errors"):
        return {}
    from api.routes.shot import _run_single_shot_video, prepare_video_quality_retry
    from config import settings
    from services.quality_review_service import STAGE_VIDEO, quality_review_service

    project_id = str(state.get("project_id") or "")
    shot_ids = _shot_ids(project_id)
    capability = quality_review_service.video_capability()
    if not capability.get("supported"):
        await quality_review_service.record_unsupported_reviews(project_id, shot_ids, STAGE_VIDEO, capability.get("reason") or "未配置")
        await quality_review_service.mark_shots_needs_human_review(shot_ids)
        return {**_abort("review_shot_videos", f"视频质量审核能力未配置，自动模式不得导出成片（{capability.get('reason') or 'unsupported'}）"), "needs_human_review": True}
    max_retries = max(0, int(settings.QUALITY_VIDEO_MAX_RETRIES))
    previous_frames = _previous_last_frames(project_id)
    last_reviews: dict[str, Any] = {}
    for attempt in range(max_retries + 1):
        reviews = {shot_id: await quality_review_service.review_video_shot(shot_id, previous_frame_path=previous_frames.get(shot_id, "")) for shot_id in shot_ids}
        last_reviews = reviews
        errored = {shot_id: review for shot_id, review in reviews.items() if review.verdict == "error"}
        if errored:
            return {**_abort("review_shot_videos", f"视频质量审核 Provider 调用失败: {_review_summary(errored)}"), "stage_status": {StageName.VIDEO_GENERATION.value: StageStatus.FAILED.value}}
        unsupported = {shot_id: review for shot_id, review in reviews.items() if review.verdict == "unsupported"}
        if unsupported:
            await quality_review_service.mark_shots_needs_human_review(sorted(unsupported))
            return {**_abort("review_shot_videos", f"视频审核存在未检测维度，转人工: {_review_summary(unsupported)}"), "needs_human_review": True}
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
    await quality_review_service.mark_shots_needs_human_review(sorted(last_reviews))
    return {**_abort("review_shot_videos", f"以下镜头视频未通过质量审核（已重试 {max_retries} 次），转人工审核: {_review_summary(last_reviews)}"), "needs_human_review": True}


async def _compose(state: AgentState) -> dict:
    if state.get("errors"):
        return {}
    project_id, store, base = _stage_context(state, StageName.EDIT_COMPOSITION)
    if store.stage_is_reusable(StageName.EDIT_COMPOSITION.value, base["input_fingerprint"]):
        return _restore_stage(state, store, StageName.EDIT_COMPOSITION)
    try:
        from services.quality_review_service import quality_review_service
        video_gate = quality_review_service.video_gate_status(project_id)
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
        return _failed_stage(state, store, StageName.EDIT_COMPOSITION, base, exc, critique=critique_compose(project_id, _db_shots(project_id), ""))


async def _final_review(state: AgentState) -> dict:
    critique = critique_final({**state, "shot_artifacts": _latest_artifacts(state, StageName.VIDEO_GENERATION)})
    return {"critiques": [critique.model_dump(mode="json")], "current_step": "final_review"}


async def _final_decision(state: AgentState) -> dict:
    return _decision_node(state, StageName.FINAL_REVIEW, next_target="completed", artifacts_stage=StageName.VIDEO_GENERATION)


async def _final_recovery(state: AgentState) -> dict:
    return _recovery_node(state, StageName.FINAL_REVIEW, default_target="compose", artifacts_stage=StageName.VIDEO_GENERATION)


async def _human_gate(state: AgentState) -> dict:
    project_id = state.get("project_id", "")
    _set_project_status(project_id, "needs_review")
    return {
        "run_status": RunStatus.WAITING_HUMAN.value,
        "needs_human_review": True,
        "human_reason": state.get("human_reason") or "Agent 已保留成功结果，等待人工确认后从检查点续跑",
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


def _route_quality_decision(state: AgentState) -> str:
    return _route_decision(state, StageName.QUALITY_REVIEW, "next", "recover")


def _route_video_decision(state: AgentState) -> str:
    return _route_decision(state, StageName.VIDEO_GENERATION, "next", "recover")


def _route_audio_decision(state: AgentState) -> str:
    return _route_decision(state, StageName.AUDIO_PRODUCTION, "next", "recover")


def _route_final_decision(state: AgentState) -> str:
    if state.get("needs_human_review"):
        return "human"
    critique = _latest_critique(state, StageName.FINAL_REVIEW)
    if critique and critique.get("passed"):
        return "done"
    return _route_decision(state, StageName.FINAL_REVIEW, "done", "recover")


def _route_director_recovery(state: AgentState) -> str:
    return "human" if state.get("needs_human_review") else "retry"


def _route_storyboard_recovery(state: AgentState) -> str:
    return "human" if state.get("needs_human_review") else "retry"


def _route_asset_recovery(state: AgentState) -> str:
    return "human" if state.get("needs_human_review") else "retry"


def _route_quality_recovery(state: AgentState) -> str:
    if state.get("needs_human_review"):
        return "human"
    return "storyboard" if state.get("pending_recovery_target") == StageName.STORYBOARD_DESIGN.value else "retry"


def _route_video_recovery(state: AgentState) -> str:
    if state.get("needs_human_review"):
        return "human"
    return "storyboard" if state.get("pending_recovery_target") == StageName.STORYBOARD_DESIGN.value else "retry"


def _route_audio_recovery(state: AgentState) -> str:
    if state.get("needs_human_review"):
        return "human"
    return "storyboard" if state.get("pending_recovery_target") == StageName.STORYBOARD_DESIGN.value else "retry"


def _route_final_recovery(state: AgentState) -> str:
    if state.get("needs_human_review"):
        return "human"
    target = str(state.get("pending_recovery_target") or "")
    if target == StageName.IMAGE_GENERATION.value:
        return "image"
    if target == StageName.VIDEO_GENERATION.value:
        return "video"
    if target == StageName.AUDIO_PRODUCTION.value:
        return "audio"
    return "compose"


def _route_decision(state: AgentState, stage: StageName | str, next_key: str, recover_key: str) -> str:
    if state.get("needs_human_review"):
        return "human"
    key = str(stage)
    status = str((state.get("stage_status") or {}).get(key, StageStatus.PENDING.value))
    critique = _latest_critique(state, stage)
    passed = bool(critique and critique.get("passed"))
    if status in {StageStatus.SUCCEEDED.value, StageStatus.DEGRADED.value} and passed:
        return next_key
    attempts = int((state.get("recovery_attempts") or {}).get(key, 0))
    max_attempts = default_quality_profile(state.get("quality_profile")).max_recovery_attempts
    return recover_key if attempts < max_attempts else "human"


# --- common helpers ---


def _stage_context(state: AgentState, stage: StageName) -> tuple[str, CheckpointStore, dict[str, Any]]:
    project_id = str(state.get("project_id") or "")
    if not project_id:
        raise RuntimeError("缺少 project_id")
    store = CheckpointStore.get(project_id, str(state.get("run_id") or "auto"))
    changes = store.detect_changes()
    initial_state = dict(state.get("initial_state") or {})
    input_fingerprint = fingerprint(
        {
            "project_id": project_id,
            "stage": stage.value,
            "initial_state": initial_state,
            "quality_profile": state.get("quality_profile") or QualityProfileName.STANDARD.value,
            "prompt_revisions": state.get("prompt_revisions") or [],
            "provider_switch": state.get("provider_switch") or {},
            "resolution": state.get("resolution") or "",
            "versions": store.data.get("version_snapshot", {}),
        }
    )
    return project_id, store, {"input_fingerprint": input_fingerprint, "initial_state": initial_state, "changes": changes}


def _save_stage(state: AgentState, store: CheckpointStore, stage: StageName, base: dict[str, Any], payload: dict[str, Any], *, critique: Any = None, failure: Any = None) -> dict:
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
    )
    strategy = default_quality_profile(state.get("quality_profile"))
    update = {
        "stage_status": {stage.value: status.value},
        "stage_outputs": {stage.value: row},
        "current_step": stage.value,
        "run_status": RunStatus.RUNNING.value,
        "quality_threshold": strategy.quality_threshold,
    }
    update.update(_payload_state_fields(stage, payload))
    if critique is not None:
        update["critiques"] = [critique.model_dump(mode="json")]
    return update


def _failed_stage(state: AgentState, store: CheckpointStore, stage: StageName, base: dict[str, Any], exc: Exception, *, critique: Any = None) -> dict:
    failure = FailureRecord(kind=FailureKind.LLM_INVALID_OUTPUT if "无法使用" in str(exc) or "JSON" in str(exc) else FailureKind.UNKNOWN, stage=stage, message=str(exc), retryable=True)
    row = store.save_stage(stage.value, status=StageStatus.FAILED.value, input_fingerprint=base["input_fingerprint"], failure=failure, critique=critique)
    trace = choose_recovery(failure, stage=stage, run_id=str(state.get("run_id") or "auto"), quality=state.get("quality_profile"), project_id=str(state.get("project_id") or ""), critique=critique, input_fingerprint=base["input_fingerprint"])
    store.add_decision(trace)
    return {
        "stage_status": {stage.value: StageStatus.FAILED.value},
        "stage_outputs": {stage.value: row},
        "critiques": [critique.model_dump(mode="json")] if critique else [],
        "decision_traces": [trace.model_dump(mode="json")],
        "current_step": stage.value,
        "run_status": RunStatus.RECOVERING.value,
    }


def _restore_stage(state: AgentState, store: CheckpointStore, stage: StageName) -> dict:
    row = store.stage(stage.value)
    payload = dict(row.get("payload") or {})
    update = {
        "stage_status": {stage.value: str(row.get("status") or StageStatus.SUCCEEDED.value)},
        "stage_outputs": {stage.value: row},
        "current_step": stage.value,
    }
    update.update(_payload_state_fields(stage, payload))
    return update


def _payload_state_fields(stage: StageName, payload: dict[str, Any]) -> dict[str, Any]:
    fields = {
        StageName.DIRECTOR_PLANNING: ("script_title", "genre", "style_suggestion", "characters", "raw_script", "script_scenes", "logic_issues", "rag_context", "requested_style", "effective_style", "style_source"),
        StageName.STORYBOARD_DESIGN: ("shots", "timing_plan"),
        StageName.ASSET_PREPARATION: ("characters", "script_scenes", "shots", "reference_supported", "consistency_report"),
        StageName.EDIT_COMPOSITION: ("output_path", "video_path"),
    }.get(stage, ())
    return {key: payload[key] for key in fields if key in payload}


def _decision_node(state: AgentState, stage: StageName, *, next_target: str, artifacts_stage: StageName | None = None) -> dict:
    critique = _latest_critique(state, stage)
    artifacts = _latest_artifacts(state, artifacts_stage or stage)
    failure = _failure_from_artifacts(stage, artifacts) if artifacts else None
    trace = choose_recovery(failure, stage=stage, run_id=str(state.get("run_id") or "auto"), quality=state.get("quality_profile"), project_id=str(state.get("project_id") or ""), critique=critique, input_fingerprint=fingerprint({"stage": stage.value, "state": state.get("stage_outputs", {})}))
    return {"decision_traces": [trace.model_dump(mode="json")], "recovery_candidates": [item.model_dump(mode="json") for item in trace.candidates], "current_step": f"{stage.value}_decision", "next_target": next_target}


def _recovery_node(state: AgentState, stage: StageName, *, default_target: str, artifacts_stage: StageName | None = None) -> dict:
    artifacts = _latest_artifacts(state, artifacts_stage or stage)
    failure = _failure_from_artifacts(stage, artifacts) or _failure_from_critique(stage, _latest_critique(state, stage))
    trace = choose_recovery(failure, stage=stage, run_id=str(state.get("run_id") or "auto"), quality=state.get("quality_profile"), project_id=str(state.get("project_id") or ""), critique=_latest_critique(state, stage), input_fingerprint=fingerprint({"stage": stage.value, "attempts": state.get("recovery_attempts", {})}))
    selected = trace.selected
    selected_strategy = selected.strategy if selected else RecoveryStrategy.HUMAN_REVIEW
    prompt_revisions = []
    provider_switch = dict(state.get("provider_switch") or {})
    pending_ids = list(state.get("pending_shot_ids") or [])
    target = default_target
    if selected_strategy is RecoveryStrategy.HUMAN_REVIEW:
        return {"needs_human_review": True, "human_reason": trace.reason, "decision_traces": [trace.model_dump(mode="json")], "current_step": f"{stage.value}_recovery"}
    if selected_strategy is RecoveryStrategy.REVISE_PROMPT:
        prompt_revisions = [selected.prompt_changes]
    elif selected_strategy is RecoveryStrategy.SWITCH_PROVIDER:
        provider_switch[stage.value] = selected.provider
    elif selected_strategy is RecoveryStrategy.LOWER_RESOLUTION:
        provider_switch[f"{stage.value}:resolution"] = "540p"
    elif selected_strategy in {RecoveryStrategy.SPLIT_SHOT, RecoveryStrategy.MERGE_SHOTS}:
        prompt_revisions = [selected.prompt_changes]
        target = StageName.STORYBOARD_DESIGN.value
    elif selected_strategy in {RecoveryStrategy.REGENERATE_FAILED_SHOTS, RecoveryStrategy.REPLACE_REFERENCE, RecoveryStrategy.RETRY, RecoveryStrategy.RESUME_CHECKPOINT}:
        pending_ids = [item.get("shot_id") for item in artifacts if item.get("status") == StageStatus.FAILED.value] or pending_ids
    attempts = dict(state.get("recovery_attempts") or {})
    attempts[stage.value] = int(attempts.get(stage.value, 0)) + 1
    return {
        "decision_traces": [trace.model_dump(mode="json")],
        "prompt_revisions": prompt_revisions,
        "provider_switch": provider_switch,
        "pending_shot_ids": pending_ids,
        "pending_recovery_target": target,
        "recovery_attempts": attempts,
        "stage_status": {stage.value: StageStatus.RECOVERING.value},
        "current_step": f"{stage.value}_recovery",
    }


def _failure_from_artifacts(stage: StageName, artifacts: list[dict[str, Any]]) -> FailureRecord | None:
    for item in reversed(artifacts):
        if item.get("failure"):
            data = dict(item["failure"])
            try:
                return FailureRecord(**data)
            except Exception:
                return FailureRecord(kind=FailureKind(data.get("kind", FailureKind.UNKNOWN.value)), stage=stage, shot_id=str(item.get("shot_id") or ""), message=str(data.get("message") or ""))
        if item.get("status") == StageStatus.FAILED.value:
            return FailureRecord(kind=FailureKind.IMAGE_FAILED if stage is StageName.IMAGE_GENERATION else FailureKind.VIDEO_FAILED, stage=stage, shot_id=str(item.get("shot_id") or ""), message="镜头产物失败")
    return None


def _failure_from_critique(stage: StageName, critique: dict[str, Any] | None) -> FailureRecord | None:
    for issue in (critique or {}).get("issues", []):
        if issue.get("severity") == "error":
            code = str(issue.get("code") or "")
            kind = {
                "dialogue_too_long": FailureKind.DIALOGUE_TOO_LONG,
                "provider_reference_unsupported": FailureKind.PROVIDER_REFERENCE_UNSUPPORTED,
                "llm_invalid_output": FailureKind.LLM_INVALID_OUTPUT,
                "image_invalid": FailureKind.IMAGE_FAILED,
                "video_invalid": FailureKind.VIDEO_FAILED,
            }.get(code, FailureKind.QUALITY_BELOW_THRESHOLD)
            return FailureRecord(kind=kind, stage=stage, shot_id=str(issue.get("shot_id") or ""), message=str(issue.get("message") or ""))
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
    if result.get("failures") and result.get("successes"):
        return StageStatus.DEGRADED
    if result.get("failures") and not result.get("successes"):
        return StageStatus.FAILED
    return StageStatus.SUCCEEDED


def _shot_versions(project_id: str, only_ids: list[str] | None = None, *, require_storyboard: bool = False) -> dict[str, int]:
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
                "video_path": row.video_path,
                "audio_path": row.audio_path,
                "status": row.status,
                "version": row.version,
            }
            for row in db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
        ]
    finally:
        db.close()


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

    ids = {str(item.get("shot_id")) for item in artifacts if item.get("status") == StageStatus.SUCCEEDED.value and item.get("path")}
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


def refresh_project_reference_state_for_graph(project_id: str) -> dict[str, Any]:
    from db import SessionLocal
    from services.reference_readiness_service import refresh_project_reference_state

    db = SessionLocal()
    try:
        return refresh_project_reference_state(db, project_id)
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
) -> dict[str, Any]:
    """Persist the real per-shot quality results in the resumable stage store."""

    _project_id, store, base = _stage_context(state, StageName.QUALITY_REVIEW)
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
        input_fingerprint=base["input_fingerprint"],
        output_fingerprint=fingerprint(payload),
        payload=payload,
        critique=critique,
    )


async def _run_storyboard_quality_gate(project_id: str, shot_ids: list[str] | None = None) -> dict[str, Any]:
    """Run the persisted storyboard quality service and fail closed."""
    from api.routes.shot import _run_storyboard_generation, prepare_storyboard_quality_retry
    from config import settings
    from services.quality_review_service import STAGE_STORYBOARD, quality_review_service

    shot_ids = list(shot_ids or _shot_ids(project_id))
    if not shot_ids:
        return {**_abort("quality_review", "无镜头可进行故事板质量审核"), "passed": False, "reviews": {}}
    capability = quality_review_service.storyboard_capability()
    if not capability.get("supported"):
        await quality_review_service.record_unsupported_reviews(project_id, shot_ids, STAGE_STORYBOARD, capability.get("reason") or "未配置")
        await quality_review_service.mark_shots_needs_human_review(shot_ids)
        return {**_abort("quality_review", f"质量审核能力未配置，自动模式不得批准故事板（{capability.get('reason') or 'unsupported'}）"), "passed": False, "needs_human_review": True, "reviews": {}}
    max_retries = max(0, int(settings.QUALITY_STORYBOARD_MAX_RETRIES))
    last_reviews: dict[str, Any] = {}
    for attempt in range(max_retries + 1):
        reviews = {shot_id: await quality_review_service.review_storyboard_shot(shot_id) for shot_id in shot_ids}
        last_reviews = reviews
        errored = {shot_id: review for shot_id, review in reviews.items() if review.verdict == "error"}
        if errored:
            return {**_abort("quality_review", f"质量审核 Provider 调用失败: {_review_summary(errored)}"), "passed": False, "reviews": reviews}
        unsupported = {shot_id: review for shot_id, review in reviews.items() if review.verdict == "unsupported"}
        if unsupported:
            await quality_review_service.mark_shots_needs_human_review(sorted(unsupported))
            return {**_abort("quality_review", f"质量审核存在未检测维度，转人工: {_review_summary(unsupported)}"), "passed": False, "needs_human_review": True, "reviews": reviews}
        failed = {shot_id: review for shot_id, review in reviews.items() if not review.passed}
        if not failed:
            return {"passed": True, "reviews": reviews, "attempt": attempt}
        if attempt >= max_retries:
            break
        retry_ids = [shot_id for shot_id in sorted(failed) if prepare_storyboard_quality_retry(shot_id, failed[shot_id])]
        if not retry_ids:
            break
        try:
            await _run_storyboard_generation_auto(project_id, retry_ids)
        except Exception as exc:
            return {**_abort("quality_review", exc), "passed": False, "reviews": reviews}
    await quality_review_service.mark_shots_needs_human_review(sorted(last_reviews))
    return {**_abort("quality_review", f"以下镜头未通过质量审核（已重试 {max_retries} 次），转人工审核: {_review_summary(last_reviews)}"), "passed": False, "needs_human_review": True, "reviews": last_reviews}


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
    reference_gate = _reference_gate(project_id)
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

    gate = await _run_storyboard_quality_gate(project_id)
    if gate.get("errors") or not gate.get("passed"):
        return gate

    db = SessionLocal()
    try:
        shots = db.query(Shot).filter(Shot.project_id == project_id).all()
        failed = _structural_failures(shots)
        if failed:
            return _abort("auto_approve_storyboard", f"以下镜头故事板未通过结构检查（已重试一次仍失败）: {', '.join(failed)}")
        for shot in shots:
            shot.confirmed = True
            shot.status = "storyboard_approved"
        db.commit()
    finally:
        db.close()
    return {"current_step": "auto_approve_storyboard"}


_graph = None


def get_graph() -> StateGraph:
    global _graph
    if _graph is None:
        _graph = build_graph().compile()
    return _graph


__all__ = ["GRAPH_NODE_META", "GRAPH_STAGE_ORDER", "build_graph", "get_graph"]
