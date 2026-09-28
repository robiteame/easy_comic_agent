"""自动模式 LangGraph 流水线。

本图是【自动模式】的真实执行器:一次 `ainvoke` 从剧本解析跑到成片导出,中途无人工卡点。
每个节点都是薄包装,在函数体内**惰性 import** 并复用 `api/routes` 里已验证的步骤函数 ——
与【手动模式】(前端逐步触发 `api/routes`)共用同一批业务函数,不重复实现业务逻辑。

`/api/graph/structure` 由 `build_graph()` + `GRAPH_NODE_META` 派生,保证可视化与真实流程一致。

注:各步骤函数沿用其手动模式的 WebSocket 进度百分比,自动模式下数值会跳变,属已知 cosmetic。
"""

import logging

from langgraph.graph import END, START, StateGraph

from services.error_reporter import ERROR_PIPELINE, log_failure, new_error_id, redact

from .state import AgentState

logger = logging.getLogger(__name__)

# 节点可视化元数据(供 /api/graph/structure 派生中文标签/类型/描述)
GRAPH_NODE_META: dict[str, dict] = {
    "parse_and_storyboard": {
        "label": "剧本解析+分镜",
        "type": "process",
        "description": "解析人物/场景/对白,生成分镜列表、角色三视图与场景基准图",
    },
    "generate_storyboard_images": {
        "label": "定稿故事板",
        "type": "process",
        "description": "逐镜头生成成品故事板参考图",
    },
    "auto_approve_storyboard": {
        "label": "结构检查+自动审核",
        "type": "process",
        "description": (
            "仅结构检查（可读性/尺寸/文件大小）通过的镜头自动确认；"
            "不合格镜头单独重生成一次，仍失败则中止，不批量放过"
        ),
    },
    "generate_shot_videos": {
        "label": "逐镜头视频",
        "type": "process",
        "description": "逐镜头按音频路由生成（TTS 配音合成或原生音视频）并出视频；失败镜头只重试自身",
    },
    "compose": {
        "label": "合成成片",
        "type": "output",
        "description": "FFmpeg 合成最终成片",
    },
}


def build_graph() -> StateGraph:
    """构建自动模式的端到端流水线图(线性串联,失败即短路至 END)。"""
    graph = StateGraph(AgentState)

    graph.add_node("parse_and_storyboard", _parse_and_storyboard)
    graph.add_node("generate_storyboard_images", _generate_storyboard_images)
    graph.add_node("auto_approve_storyboard", _auto_approve_storyboard)
    graph.add_node("generate_shot_videos", _generate_shot_videos)
    graph.add_node("compose", _compose)

    graph.add_edge(START, "parse_and_storyboard")
    graph.add_edge("parse_and_storyboard", "generate_storyboard_images")
    graph.add_edge("generate_storyboard_images", "auto_approve_storyboard")
    graph.add_edge("auto_approve_storyboard", "generate_shot_videos")
    graph.add_edge("generate_shot_videos", "compose")
    graph.add_edge("compose", END)
    return graph


# --- 节点实现:复用 route 层步骤函数 ---


async def _parse_and_storyboard(state: AgentState) -> dict:
    if state.get("errors"):
        return {}
    from api.routes.script import _run_storyboard_phase

    project_id = state["project_id"]
    try:
        await _run_storyboard_phase(project_id, dict(state.get("initial_state") or {}))
    except Exception as exc:
        return _abort("parse_and_storyboard", exc)
    if _project_failed(project_id):
        return _abort("parse_and_storyboard", "剧本解析或分镜生成失败")
    return {"current_step": "parse_and_storyboard"}


async def _generate_storyboard_images(state: AgentState) -> dict:
    if state.get("errors"):
        return {}
    from api.routes.shot import _run_storyboard_generation

    project_id = state["project_id"]
    shot_ids = _shot_ids(project_id)
    if not shot_ids:
        return _abort("generate_storyboard_images", "无分镜可生成定稿故事板")
    try:
        await _run_storyboard_generation(project_id, shot_ids)
    except Exception as exc:
        return _abort("generate_storyboard_images", exc)
    if _project_failed(project_id):
        return _abort("generate_storyboard_images", "定稿故事板生成失败")
    return {"current_step": "generate_storyboard_images"}


async def _auto_approve_storyboard(state: AgentState) -> dict:
    """结构检查门禁 + 单镜头重试，而不是「有图就全批」。

    1. 每个已出图镜头先过结构检查（文件存在、可解码、尺寸与字节数达标）；
    2. 不合格的镜头单独重新生成一次（不重跑整个项目），再复检；
    3. 复检仍不合格则中止自动流程并明确列出失败镜头——绝不自动批准
       结构不合格的故事板，也不把结构检查结果当成「质量通过」。
    """
    if state.get("errors"):
        return {}
    from db import SessionLocal
    from models import Shot
    from services.structural_validation import validate_image_file

    project_id = state["project_id"]

    def _structural_failures(shot_rows) -> list[str]:
        failed: list[str] = []
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
        # 与人工审核同口径：参数已修改、素材待重生成的镜头不得自动批准。
        stale_shots = [shot.id for shot in shots if shot.media_stale]
        if stale_shots:
            return _abort(
                "auto_approve_storyboard",
                f"以下镜头参数已修改、素材待重新生成: {', '.join(stale_shots)}",
            )
        failed_once = _structural_failures(shots)
    finally:
        db.close()

    if failed_once:
        logger.info("结构检查未通过的故事板镜头，单独重生成一次: %s", failed_once)
        try:
            from api.routes.shot import _run_storyboard_generation

            await _run_storyboard_generation(project_id, list(failed_once))
        except Exception as exc:
            return _abort("auto_approve_storyboard", exc)

    db = SessionLocal()
    try:
        shots = db.query(Shot).filter(Shot.project_id == project_id).all()
        failed = _structural_failures(shots)
        if failed:
            return _abort(
                "auto_approve_storyboard",
                f"以下镜头故事板未通过结构检查（已重试一次仍失败）: {', '.join(failed)}",
            )
        for shot in shots:
            shot.confirmed = True
            shot.status = "storyboard_approved"
        db.commit()
    finally:
        db.close()
    return {"current_step": "auto_approve_storyboard"}


async def _generate_shot_videos(state: AgentState) -> dict:
    if state.get("errors"):
        return {}
    from api.routes.shot import _run_single_shot_video

    project_id = state["project_id"]
    shot_ids = _shot_ids(project_id)
    failures: dict[str, str] = {}
    for shot_id in shot_ids:
        try:
            await _run_single_shot_video(shot_id, force=False)
        except Exception as exc:
            failures[shot_id] = str(exc)
    if failures:
        # 只重试失败镜头本身，不重复生成整个项目。
        logger.info("视频生成失败镜头，单独重试一次: %s", sorted(failures))
        retried = list(failures)
        for shot_id in retried:
            try:
                await _run_single_shot_video(shot_id, force=True)
                failures.pop(shot_id, None)
            except Exception as exc:
                failures[shot_id] = str(exc)
    if failures:
        summary = "; ".join(f"{shot_id}: {reason[:120]}" for shot_id, reason in sorted(failures.items()))
        return _abort("generate_shot_videos", f"以下镜头视频生成失败（已单独重试一次）: {summary}")
    if _has_unfinished_videos(project_id):
        return _abort("generate_shot_videos", "仍有镜头视频未生成")
    return {"current_step": "generate_shot_videos"}


async def _compose(state: AgentState) -> dict:
    if state.get("errors"):
        return {}
    from api.routes.render import _render_task

    project_id = state["project_id"]
    try:
        await _render_task(project_id, state.get("output_format") or "9:16", state.get("resolution") or "1080p")
    except Exception as exc:
        return _abort("compose", exc)
    # A route may report a non-throwing status for compatibility with older
    # callers. Treat anything except completed as a failed graph node.
    from api.routes.render import _render_status

    render_status = _render_status.get(project_id, {})
    if render_status.get("status") != "completed":
        return _abort("compose", render_status.get("message") or "成片导出未完成")
    return {"current_step": "compose"}


# --- 辅助 ---


def _abort(node: str, exc) -> dict:
    """记录完整异常，只把可读的一行摘要交给状态机。

    state 里的 errors 会经 WebSocket / 任务表回显给前端，因此这里不写入堆栈、
    本地路径或供应商原始响应；完整堆栈留在服务端日志里，用错误编号关联。
    """

    if isinstance(exc, BaseException):
        error_id = log_failure(exc, error_type=ERROR_PIPELINE, context={"node": node}, log=logger)
    else:
        error_id = new_error_id()
        logger.error("自动流程节点失败 [%s] node=%s reason=%s", error_id, node, redact(exc))
    return {"errors": [f"[{node}] 执行失败（错误编号 {error_id}）"], "current_step": "aborted"}


def _shot_ids(project_id: str) -> list[str]:
    from db import SessionLocal
    from models import Shot

    db = SessionLocal()
    try:
        return [s.id for s in db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()]
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


_graph = None


def get_graph() -> StateGraph:
    global _graph
    if _graph is None:
        _graph = build_graph().compile()
    return _graph
