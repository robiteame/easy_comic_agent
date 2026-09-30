"""自动模式 LangGraph 流水线。

本图是【自动模式】的真实执行器:一次 `ainvoke` 从剧本解析跑到成片导出,中途无人工卡点。
每个节点都是薄包装,在函数体内**惰性 import** 并复用 `api/routes` 里已验证的步骤函数 ——
与【手动模式】(前端逐步触发 `api/routes`)共用同一批业务函数,不重复实现业务逻辑。

`/api/graph/structure` 由 `build_graph()` + `GRAPH_NODE_META` 派生,保证可视化与真实流程一致。

质量闭环：``auto_approve_storyboard`` = StructuralCheck（文件可用性）+
QualityReviewService（真实质量门禁，未通过按修正建议重生成，重试耗尽转
needs_review）；``review_shot_videos`` 对视频做运动/衔接/口型/音画/清晰度
审核。故事板未过门禁不生成视频，视频未过门禁不导出成片。

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
        "label": "结构检查+质量审核门禁",
        "type": "decision",
        "description": (
            "先过结构检查（仅文件可用性，不代表质量通过），再经 QualityReviewService "
            "逐镜头评分（语义还原/角色身份/服装发型/构图/伪影）；不达标镜头按审核建议"
            "修正 prompt 后重生成，重试耗尽或审核能力未配置则标记 needs_review 转人工，"
            "绝不批量放过"
        ),
    },
    "generate_shot_videos": {
        "label": "逐镜头视频",
        "type": "process",
        "description": "逐镜头按音频路由生成（TTS 配音合成或原生音视频）并出视频；失败镜头只重试自身",
    },
    "review_shot_videos": {
        "label": "视频质量审核",
        "type": "decision",
        "description": (
            "对每个镜头视频审核运动连贯性、镜头间衔接、口型对白、音画同步与音频清晰度；"
            "不达标按建议修正后重生成，重试耗尽转 needs_review；未通过不得导出成片"
        ),
    },
    "compose": {
        "label": "合成成片",
        "type": "output",
        "description": "FFmpeg 合成最终成片（前置校验：全部镜头已通过视频质量门禁）",
    },
}


def build_graph() -> StateGraph:
    """构建自动模式的端到端流水线图(线性串联,失败即短路至 END)。"""
    graph = StateGraph(AgentState)

    graph.add_node("parse_and_storyboard", _parse_and_storyboard)
    graph.add_node("generate_storyboard_images", _generate_storyboard_images)
    graph.add_node("auto_approve_storyboard", _auto_approve_storyboard)
    graph.add_node("generate_shot_videos", _generate_shot_videos)
    graph.add_node("review_shot_videos", _review_shot_videos)
    graph.add_node("compose", _compose)

    graph.add_edge(START, "parse_and_storyboard")
    graph.add_edge("parse_and_storyboard", "generate_storyboard_images")
    graph.add_edge("generate_storyboard_images", "auto_approve_storyboard")
    graph.add_edge("auto_approve_storyboard", "generate_shot_videos")
    graph.add_edge("generate_shot_videos", "review_shot_videos")
    graph.add_edge("review_shot_videos", "compose")
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
    """结构检查门禁 + 质量审核闭环，而不是「有图就全批」。

    1. StructuralCheck：每个已出图镜头检查文件存在、可解码、尺寸与字节数
       达标——这只代表「文件可用」，绝不代表「质量通过」；
       不合格的镜头单独重新生成一次，复检仍不合格则中止；
    2. QualityReview：结构合格后由 QualityReviewService 逐镜头评分（场景/
       动作/景别/机位还原、角色身份、服装发型、构图、伪影）。不达标镜头按
       审核修正建议改写 prompt 后重生成，超过最大重试次数仍不达标则标记
       needs_review 转人工；
    3. 审核能力未配置（VLM 缺失等）时同样标记 needs_review 并中止——
       没有真实评分就绝不自动批准。
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
    finally:
        db.close()

    # --- StructuralCheck 全部通过 ≠ 质量通过；继续真实质量门禁 ---
    gate = await _run_storyboard_quality_gate(project_id)
    if gate.get("errors"):
        return gate

    db = SessionLocal()
    try:
        shots = db.query(Shot).filter(Shot.project_id == project_id).all()
        for shot in shots:
            shot.confirmed = True
            shot.status = "storyboard_approved"
        db.commit()
    finally:
        db.close()
    return {"current_step": "auto_approve_storyboard"}


async def _run_storyboard_quality_gate(project_id: str) -> dict:
    """故事板质量审核 + 修正重试闭环；返回带 errors 的 dict 表示门禁未过。"""
    from api.routes.shot import _run_storyboard_generation, prepare_storyboard_quality_retry
    from config import settings
    from services.quality_review_service import STAGE_STORYBOARD, quality_review_service

    shot_ids = _shot_ids(project_id)
    capability = quality_review_service.storyboard_capability()
    if not capability["supported"]:
        await quality_review_service.record_unsupported_reviews(
            project_id, shot_ids, STAGE_STORYBOARD, capability["reason"]
        )
        await quality_review_service.mark_shots_needs_human_review(shot_ids)
        return {
            **_abort(
                "auto_approve_storyboard",
                f"质量审核能力未配置，自动模式不得批准故事板（{capability['reason']}）；"
                f"{len(shot_ids)} 个镜头已标记 needs_review，请配置审核 Provider 或人工审核",
            ),
            "needs_human_review": True,
        }

    max_retries = max(0, int(settings.QUALITY_STORYBOARD_MAX_RETRIES))
    last_failed: dict[str, object] = {}
    for attempt in range(max_retries + 1):
        reviews = {}
        for shot_id in shot_ids:
            reviews[shot_id] = await quality_review_service.review_storyboard_shot(shot_id)

        errored = {shot_id: r for shot_id, r in reviews.items() if r.verdict == "error"}
        if errored:
            return _abort(
                "auto_approve_storyboard",
                f"质量审核 Provider 调用失败（不重生成素材，请检查配置后重跑）: {_review_summary(errored)}",
            )
        unsupported = {shot_id: r for shot_id, r in reviews.items() if r.verdict == "unsupported"}
        if unsupported:
            await quality_review_service.mark_shots_needs_human_review(sorted(unsupported))
            return {
                **_abort(
                    "auto_approve_storyboard",
                    f"以下镜头质量审核存在未检测维度，按当前策略不得自动批准，转人工: {_review_summary(unsupported)}",
                ),
                "needs_human_review": True,
            }
        failed = {shot_id: r for shot_id, r in reviews.items() if not r.passed}
        if not failed:
            last_failed = {}
            break
        if attempt >= max_retries:
            last_failed = failed
            break
        retry_ids = [
            shot_id
            for shot_id in sorted(failed)
            if prepare_storyboard_quality_retry(shot_id, failed[shot_id])
        ]
        if not retry_ids:
            last_failed = failed
            break
        logger.info(
            "质量审核不达标镜头（第 %d/%d 次重试），按审核建议修正 prompt 后重生成: %s",
            attempt + 1, max_retries, retry_ids,
        )
        try:
            await _run_storyboard_generation(project_id, retry_ids)
        except Exception as exc:
            return _abort("auto_approve_storyboard", exc)

    if last_failed:
        await quality_review_service.mark_shots_needs_human_review(sorted(last_failed))
        return {
            **_abort(
                "auto_approve_storyboard",
                f"以下镜头未通过质量审核（已重试 {max_retries} 次），转人工审核: {_review_summary(last_failed)}",
            ),
            "needs_human_review": True,
        }
    return {}


async def _generate_shot_videos(state: AgentState) -> dict:
    if state.get("errors"):
        return {}
    from api.routes.shot import _run_single_shot_video
    from services.quality_review_service import quality_review_service

    project_id = state["project_id"]
    # 质量门禁：故事板审核未全部通过（或未审核）的镜头禁止进入视频生成。
    gate = quality_review_service.storyboard_gate_status(project_id)
    if not gate["ok"]:
        return _abort(
            "generate_shot_videos",
            "故事板质量门禁未通过，自动模式禁止生成视频: " + _gate_failure_summary(gate),
        )
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


async def _review_shot_videos(state: AgentState) -> dict:
    """视频质量审核 + 修正重试；未通过则镜头转 needs_review 并阻断成片导出。"""
    if state.get("errors"):
        return {}
    from api.routes.shot import _run_single_shot_video, prepare_video_quality_retry
    from config import settings
    from services.quality_review_service import STAGE_VIDEO, quality_review_service

    project_id = state["project_id"]
    shot_ids = _shot_ids(project_id)
    capability = quality_review_service.video_capability()
    if not capability["supported"]:
        await quality_review_service.record_unsupported_reviews(
            project_id, shot_ids, STAGE_VIDEO, capability["reason"]
        )
        await quality_review_service.mark_shots_needs_human_review(shot_ids)
        return {
            **_abort(
                "review_shot_videos",
                f"视频质量审核能力未配置，自动模式不得导出成片（{capability['reason']}）；"
                f"{len(shot_ids)} 个镜头已标记 needs_review",
            ),
            "needs_human_review": True,
        }

    max_retries = max(0, int(settings.QUALITY_VIDEO_MAX_RETRIES))
    previous_frames = _previous_last_frames(project_id)
    last_failed: dict[str, object] = {}
    for attempt in range(max_retries + 1):
        reviews = {}
        for shot_id in shot_ids:
            reviews[shot_id] = await quality_review_service.review_video_shot(
                shot_id, previous_frame_path=previous_frames.get(shot_id, "")
            )
        errored = {shot_id: r for shot_id, r in reviews.items() if r.verdict == "error"}
        if errored:
            return _abort(
                "review_shot_videos",
                f"视频质量审核 Provider 调用失败: {_review_summary(errored)}",
            )
        unsupported = {shot_id: r for shot_id, r in reviews.items() if r.verdict == "unsupported"}
        if unsupported:
            await quality_review_service.mark_shots_needs_human_review(sorted(unsupported))
            return {
                **_abort(
                    "review_shot_videos",
                    f"以下镜头视频审核存在未检测维度，按当前策略不得放行，转人工: {_review_summary(unsupported)}",
                ),
                "needs_human_review": True,
            }
        failed = {shot_id: r for shot_id, r in reviews.items() if not r.passed}
        if not failed:
            last_failed = {}
            break
        if attempt >= max_retries:
            last_failed = failed
            break
        retry_ids = [
            shot_id
            for shot_id in sorted(failed)
            if prepare_video_quality_retry(shot_id, failed[shot_id])
        ]
        if not retry_ids:
            last_failed = failed
            break
        logger.info(
            "视频质量审核不达标镜头（第 %d/%d 次重试），修正 prompt 后重生成: %s",
            attempt + 1, max_retries, retry_ids,
        )
        for shot_id in retry_ids:
            try:
                await _run_single_shot_video(shot_id, force=True)
            except Exception as exc:
                return _abort("review_shot_videos", exc)

    if last_failed:
        await quality_review_service.mark_shots_needs_human_review(sorted(last_failed))
        return {
            **_abort(
                "review_shot_videos",
                f"以下镜头视频未通过质量审核（已重试 {max_retries} 次），转人工审核: {_review_summary(last_failed)}",
            ),
            "needs_human_review": True,
        }
    return {"current_step": "review_shot_videos"}


async def _compose(state: AgentState) -> dict:
    if state.get("errors"):
        return {}
    from api.routes.render import _render_task
    from services.quality_review_service import quality_review_service

    project_id = state["project_id"]
    # 质量门禁：视频审核未全部通过的镜头禁止进入成片导出。
    gate = quality_review_service.video_gate_status(project_id)
    if not gate["ok"]:
        return _abort(
            "compose",
            "视频质量门禁未通过，自动模式禁止导出成片: " + _gate_failure_summary(gate),
        )
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


def _review_summary(reviews: dict) -> str:
    """把一组失败审核压成一行可读摘要（进 errors/日志，注意不要堆栈）。"""
    parts = []
    for shot_id, review in sorted(reviews.items()):
        top_issue = review.issues[0] if review.issues else review.verdict
        parts.append(f"{shot_id}[{review.verdict} {review.overall_score:.2f}] {top_issue}")
    return "; ".join(parts)[:500]


def _gate_failure_summary(gate: dict) -> str:
    failed = gate.get("failed") or []
    if not failed:
        return str(gate.get("reason") or "门禁未通过")
    return "; ".join(f"{item['shot_id']}: {item['reason']}" for item in failed[:10])


def _shot_ids(project_id: str) -> list[str]:
    from db import SessionLocal
    from models import Shot

    db = SessionLocal()
    try:
        return [s.id for s in db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()]
    finally:
        db.close()


def _previous_last_frames(project_id: str) -> dict[str, str]:
    """{shot_id: 上一镜头（按 sequence）的尾帧路径}，供镜头衔接审核。"""
    from db import SessionLocal
    from models import Shot

    db = SessionLocal()
    try:
        shots = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
    finally:
        db.close()
    frames: dict[str, str] = {}
    previous = ""
    for shot in shots:
        if previous:
            frames[shot.id] = previous
        previous = shot.last_frame_path or shot.storyboard_path or shot.image_path or ""
    return frames


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
