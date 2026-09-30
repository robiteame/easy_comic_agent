from fastapi import APIRouter, HTTPException

from agent.checkpoints import CheckpointStore
from agent.contracts import QUALITY_STRATEGIES, STAGE_CONTRACTS, STAGE_ORDER
from agent.graph import GRAPH_NODE_META, GRAPH_STAGE_ORDER, get_graph
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

    edges = [{"source": edge.source, "target": edge.target, "label": ""} for edge in drawable.edges]
    contracts = [
        {
            "stage": item.value,
            **STAGE_CONTRACTS[item].model_dump(mode="json"),
        }
        for item in STAGE_ORDER
    ]
    quality_profiles = [item.model_dump(mode="json") for item in QUALITY_STRATEGIES.values()]
    return {
        "nodes": nodes,
        "edges": edges,
        "stage_order": list(GRAPH_STAGE_ORDER),
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
    """可解释追踪：决策树、候选、评分、修改原因、最终选择和逐镜头时间线。"""
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
    return {
        "project_id": snapshot.get("project_id"),
        "run_id": snapshot.get("run_id"),
        "status": snapshot.get("status"),
        "status_reason": snapshot.get("status_reason", ""),
        "stages": stages,
        "decisions": snapshot.get("decisions", []),
        "shots": snapshot.get("shots", {}),
        "events": snapshot.get("events", []),
        "mermaid": _mermaid(snapshot),
    }


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
