from fastapi import APIRouter, HTTPException

from agent.checkpoints import CheckpointStore, summarize_trace
from agent.contracts import QUALITY_STRATEGIES, STAGE_CONTRACTS, STAGE_ORDER
from agent.graph import GRAPH_NODE_META, GRAPH_STAGE_NODE_NAMES, GRAPH_STAGE_ORDER, get_graph
from services.security import validate_identifier

router = APIRouter(prefix="/api/graph", tags=["graph"])

_SPECIAL_NODES = {
    "__start__": {"label": "开始", "type": "input", "description": "自动流水线入口"},
    "__end__": {"label": "完成", "type": "output", "description": "输出可播放成片或明确人工卡点"},
}


@router.get("/structure")
async def get_graph_structure():
    """真实图结构 + 阶段契约 + 质量档位，供前端可视化。"""
    drawable = get_graph().get_graph()

    nodes = []
    for node_id in drawable.nodes:
        meta = _SPECIAL_NODES.get(node_id) or GRAPH_NODE_META.get(
            node_id, {"label": node_id, "type": "process", "description": ""}
        )
        nodes.append({"id": node_id, **meta})

    edges = [{"source": edge.source, "target": edge.target, "label": str(edge.data or "")} for edge in drawable.edges]
    contracts = [
        {
            "stage": item.value,
            **STAGE_CONTRACTS[item].model_dump(mode="json", exclude={"input_model", "output_model"}),
            "input_model": STAGE_CONTRACTS[item].input_model.__name__,
            "output_model": STAGE_CONTRACTS[item].output_model.__name__,
        }
        for item in STAGE_ORDER
    ]
    quality_profiles = [item.model_dump(mode="json") for item in QUALITY_STRATEGIES.values()]
    return {
        "nodes": nodes,
        "edges": edges,
        "stage_order": list(GRAPH_STAGE_ORDER),
        "stage_node_roles": {stage: dict(roles) for stage, roles in GRAPH_STAGE_NODE_NAMES.items()},
        "stage_contracts": contracts,
        "quality_profiles": quality_profiles,
    }


@router.get("/runs/{project_id}")
async def get_agent_run(project_id: str, run_id: str = "auto"):
    """读取幂等检查点：阶段状态、逐镜头产物、失败和人工介入状态。"""
    store = _store(project_id, run_id)
    return store.snapshot()


@router.get("/runs/{project_id}/trace")
async def get_agent_trace(project_id: str, run_id: str = "auto"):
    """可解释追踪：决策树、候选、评分、修改原因、最终选择和逐镜头时间线。

    ``summary`` 是面向展示的稳定汇总（当前阶段、镜头状态、阶段质量分、Critic
    问题、恢复候选与最终决策、Prompt 修改、候选结果、Provider/模型、实际发送
    参考图、成本、预计/实际耗时、自动降级原因、检查点与恢复次数）；顶层保留
    原始 ``stages/decisions/shots/events`` 供旧消费者使用。
    """
    store = _store(project_id, run_id)
    snapshot = store.snapshot()
    stages = [
        {
            "stage": stage,
            "label": GRAPH_NODE_META.get(stage, {}).get("label", stage),
            **snapshot.get("stages", {}).get(stage, {"status": "pending"}),
        }
        for stage in GRAPH_STAGE_ORDER
        if stage in {item.value for item in STAGE_ORDER}
    ]
    summary = summarize_trace(snapshot)
    for row in summary.get("stages", []):
        row["label"] = GRAPH_NODE_META.get(str(row.get("stage")), {}).get("label", row.get("stage"))
    summary["shots"] = _attach_shot_reference_rows(summary.get("shots", []))
    return {
        "project_id": snapshot.get("project_id"),
        "run_id": snapshot.get("run_id"),
        "status": snapshot.get("status"),
        "status_reason": snapshot.get("status_reason", ""),
        "stages": stages,
        "decisions": snapshot.get("decisions", []),
        "shots": snapshot.get("shots", {}),
        "events": snapshot.get("events", []),
        "summary": summary,
        "mermaid": _mermaid(snapshot),
    }


def _attach_shot_reference_rows(shots: list[dict]) -> list[dict]:
    """把数据库里的镜头版本/状态和「实际发送参考图」清单并入追踪汇总。

    检查点文件只记录生成结果；参考图发送清单落在 Shot 行的
    storyboard/video reference manifest 上，这里按 shot_id 合并。数据库不可用
    时保持检查点原样，绝不因读取失败让整条追踪 404。
    """

    if not shots:
        return shots
    try:
        import json as _json

        from db import SessionLocal
        from models import Shot

        db = SessionLocal()
        try:
            rows = {
                row.id: row
                for row in db.query(Shot).filter(Shot.id.in_([str(item.get("shot_id")) for item in shots])).all()
            }
        finally:
            db.close()
    except Exception:
        return shots
    for item in shots:
        row = rows.get(str(item.get("shot_id")))
        if row is None:
            continue
        item["db_status"] = str(row.status or "")
        item["db_shot_version"] = int(row.version or 1)
        item["confirmed"] = bool(row.confirmed)

        def manifest(field: str) -> list[dict]:
            try:
                value = _json.loads(getattr(row, field, "") or "[]")
                return [entry for entry in value if isinstance(entry, dict)]
            except Exception:
                return []

        item["references_sent"] = {
            "image_generation": manifest("storyboard_reference_manifest"),
            "video_generation": manifest("video_reference_manifest"),
        }
    return shots


@router.get("/runs/{project_id}/decisions")
async def get_agent_decisions(project_id: str, run_id: str = "auto"):
    store = _store(project_id, run_id)
    return {"project_id": project_id, "run_id": run_id, "decisions": store.decisions()}


def _store(project_id: str, run_id: str) -> CheckpointStore:
    try:
        safe_project_id = validate_identifier(project_id, "项目 ID")
        safe_run_id = validate_identifier(run_id, "运行 ID")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CheckpointStore.get(safe_project_id, safe_run_id)


def _mermaid(snapshot: dict) -> str:
    lines = ["flowchart TD"]
    status = snapshot.get("status", "pending")
    lines.append(f"  run((运行 {status}))")
    previous = "run"
    for item in snapshot.get("stages", {}).values():
        stage = str(item.get("stage") or "")
        state = str(item.get("status") or "pending")
        label = GRAPH_NODE_META.get(stage, {}).get("label", stage)
        node_id = "".join(char if char.isalnum() else "_" for char in stage)
        lines.append(f"  {node_id}[\"{label}: {state}\"]")
        lines.append(f"  {previous} --> {node_id}")
        previous = node_id
    for trace in snapshot.get("decisions", []):
        trace_id = "".join(char if char.isalnum() else "_" for char in str(trace.get("trace_id") or "decision"))
        selected = trace.get("selected") or {}
        strategy = selected.get("strategy", "human_review")
        reason = str(trace.get("reason") or "").replace('"', "'")[:80]
        lines.append(f"  {trace_id}(\"选择 {strategy}<br/>{reason}\")")
        lines.append(f"  {previous} -.-> {trace_id}")
    return "\n".join(lines)
