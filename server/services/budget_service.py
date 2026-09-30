"""预算配置、任务估算与预算检查（软预算提示 / 硬预算阻断）。

三条硬规则：

1. **估算与实际分开**：估算写 cost_estimates（本模块），实际用量写 usage_records
   （usage_service），互不覆盖；
2. **硬预算必须真的拦得住**：检查时比对「已用 + 已预留 + 本次估算」。只比对已用
   是不够的——两个并发任务同时启动时，第一个任务的花费要等结束才可见；因此每个
   任务在抢占成功前先按估算金额预留额度（budget_reservations，reservation_key 唯一，
   重复抢占不会重复预留），任务终结时释放；
3. **算不出来就不拦也不编**：估值未知（价格未配置）时不阻断，只在状态里标
   estimate_unknown，前端显示「成本未知」；已用金额本身已经超过硬预算时仍然阻断。

时长为第二维度，语义与金额一致，统计口径是「任务执行时长」（background_jobs 的
真实起止时间），而不是供应商计费秒数。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from db import SessionLocal
from models import BackgroundJob, BudgetConfig, BudgetReservation, Character, Project, SceneAsset, Shot
from models.budget import SCOPE_GLOBAL, SCOPE_PROJECT
from models.pricing import DEFAULT_CURRENCY
from services import pricing_service, usage_service
from services.job_types import (
    JOB_TYPE_ASSET_GENERATION,
    JOB_TYPE_RENDER,
    JOB_TYPE_SCRIPT_PIPELINE,
    JOB_TYPE_SHOT_AUDIO,
    JOB_TYPE_SHOT_IMAGE,
    JOB_TYPE_SHOT_VIDEO,
    JOB_TYPE_STORYBOARD,
    JOB_TYPE_UNKNOWN,
    TERMINAL_STATUSES,
    job_type_label,
)
from services.providers.endpoint import get_endpoint
from services.providers.usage import (
    CAPABILITY_FFMPEG,
    CAPABILITY_IMAGE,
    CAPABILITY_LLM,
    CAPABILITY_LABELS,
    CAPABILITY_TTS,
    CAPABILITY_VIDEO,
    CAPABILITY_BASE_UNITS,
)

logger = logging.getLogger(__name__)

# 预算状态等级。
LEVEL_UNLIMITED = "unlimited"
LEVEL_OK = "ok"
LEVEL_SOFT_EXCEEDED = "soft_exceeded"
LEVEL_HARD_EXCEEDED = "hard_exceeded"

# 稳定错误码：与 services/job_types.ERROR_CODE_BUDGET_EXCEEDED 保持一致。
CODE_BUDGET_EXCEEDED = "budget_exceeded"
CODE_BUDGET_SOFT_EXCEEDED = "budget_soft_exceeded"

# 预留状态。
RESERVATION_ACTIVE = "active"
RESERVATION_RELEASED = "released"

# 单位耗时模型（毫秒→秒）：仅在同类任务历史样本不足时使用，并在结果里标明来源。
_PER_CALL_SECONDS = {
    CAPABILITY_LLM: 20.0,
    CAPABILITY_IMAGE: 6.0,
    CAPABILITY_TTS: 3.0,
    CAPABILITY_VIDEO: 20.0,
    CAPABILITY_FFMPEG: 4.0,
}
_PER_UNIT_SECONDS = {
    CAPABILITY_LLM: 0.0,
    CAPABILITY_IMAGE: 4.0,
    CAPABILITY_TTS: 0.02,
    CAPABILITY_VIDEO: 20.0,
    CAPABILITY_FFMPEG: 1.0,
}

# LLM 估算：输出 token 按端点 max_tokens 上限估（保守上界），输入按剧本字符数 / 4。
_CHARS_PER_TOKEN = 4
_MIN_INPUT_TOKENS = 400


def _dialogue_chars(shot) -> int:
    """TTS 字符口径：结构化对白逐句求和（兼容旧版纯文本镜头）。"""

    from services.shot_dialogue import dialogue_total_chars, parse_shot_dialogue

    return dialogue_total_chars(parse_shot_dialogue(getattr(shot, "dialogue", "") or ""))


@dataclass(frozen=True)


class WorkloadComponent:
    """任务在某一能力上的预计工作量。"""

    capability: str
    provider: str = ""
    model: str = ""
    quantity: int = 0
    secondary_quantity: int = 0
    calls: int = 1
    resolution: str = ""
    label: str = ""

    def estimated_seconds(self) -> int:
        per_call = _PER_CALL_SECONDS.get(self.capability, 5.0) * max(1, int(self.calls or 1))
        per_unit = _PER_UNIT_SECONDS.get(self.capability, 0.0) * max(0, int(self.quantity or 0))
        return max(0, int(round(per_call + per_unit)))


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    level: str = LEVEL_UNLIMITED
    code: str = ""
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    estimate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": bool(self.allowed),
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "estimate": self.estimate,
        }


def _cents_label(amount_micro: int | None, currency: str) -> str:
    if amount_micro is None:
        return "未设置"
    return pricing_service.format_amount_micro(int(amount_micro), currency)


# ---------------------------------------------------------------------------
# 预算配置
# ---------------------------------------------------------------------------


def _budget_dto(row: BudgetConfig) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "scope_type": str(row.scope_type),
        "scope_id": str(row.scope_id or ""),
        "currency": str(row.currency or DEFAULT_CURRENCY),
        "soft_cost_micro": int(row.soft_cost_micro) if row.soft_cost_micro is not None else None,
        "hard_cost_micro": int(row.hard_cost_micro) if row.hard_cost_micro is not None else None,
        "soft_seconds": int(row.soft_seconds) if row.soft_seconds is not None else None,
        "hard_seconds": int(row.hard_seconds) if row.hard_seconds is not None else None,
        "enabled": bool(row.enabled),
        "note": str(row.note or ""),
        "updated_at": row.updated_at.isoformat() if isinstance(row.updated_at, datetime) else None,
    }


def _optional_int(value: Any, name: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数")
    if isinstance(value, float):
        raise ValueError(f"{name} 必须是整数（金额为最小货币单位的整数倍），不接受小数：{value}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if number < 0:
        raise ValueError(f"{name} 不能为负数")
    return number


def get_budget(db: Session, *, project_id: str = "") -> dict[str, Any]:
    """读取预算配置：全局 + 该项目（含其父系列）的行，并给出生效值。"""

    rows = db.query(BudgetConfig).all()
    global_row = next((row for row in rows if str(row.scope_type) == SCOPE_GLOBAL), None)
    project_rows = [row for row in rows if str(row.scope_type) == SCOPE_PROJECT]
    project_row = next((row for row in project_rows if str(row.scope_id) == project_id), None)
    series_id = _series_id_of(db, project_id) if project_id else ""
    series_row = next((row for row in project_rows if series_id and str(row.scope_id) == series_id), None)
    return {
        "project_id": project_id,
        "series_id": series_id,
        "global": _budget_dto(global_row) if global_row is not None else None,
        "project": _budget_dto(project_row) if project_row is not None else None,
        "series": _budget_dto(series_row) if series_row is not None else None,
        "effective": effective_limits(db, project_id=project_id),
    }


def save_budget(db: Session, payload: dict[str, Any]) -> dict[str, Any]:
    """保存一条预算配置（scope_type + scope_id 唯一）。"""

    scope_type = str((payload or {}).get("scope_type") or SCOPE_PROJECT).strip().lower()
    if scope_type not in (SCOPE_GLOBAL, SCOPE_PROJECT):
        raise ValueError(f"未知预算作用域: {scope_type}，可选值: global, project")
    scope_id = "" if scope_type == SCOPE_GLOBAL else str(payload.get("scope_id") or "").strip()
    if scope_type == SCOPE_PROJECT:
        if not scope_id:
            raise ValueError("项目预算必须指定 scope_id")
        if db.query(Project.id).filter(Project.id == scope_id).first() is None:
            raise ValueError("项目不存在，无法设置预算")

    soft_cost = _optional_int(payload.get("soft_cost_micro"), "soft_cost_micro")
    hard_cost = _optional_int(payload.get("hard_cost_micro"), "hard_cost_micro")
    soft_seconds = _optional_int(payload.get("soft_seconds"), "soft_seconds")
    hard_seconds = _optional_int(payload.get("hard_seconds"), "hard_seconds")
    if soft_cost is not None and hard_cost is not None and soft_cost > hard_cost:
        raise ValueError("软预算不能高于硬预算")
    if soft_seconds is not None and hard_seconds is not None and soft_seconds > hard_seconds:
        raise ValueError("软时长预算不能高于硬时长预算")

    row = (
        db.query(BudgetConfig)
        .filter(BudgetConfig.scope_type == scope_type, BudgetConfig.scope_id == scope_id)
        .first()
    )
    if row is None:
        row = BudgetConfig(id=uuid.uuid4().hex, scope_type=scope_type, scope_id=scope_id)
        db.add(row)
    row.currency = str(payload.get("currency") or row.currency or DEFAULT_CURRENCY)[:8]
    row.soft_cost_micro = soft_cost
    row.hard_cost_micro = hard_cost
    row.soft_seconds = soft_seconds
    row.hard_seconds = hard_seconds
    row.enabled = bool(payload.get("enabled", True))
    row.note = str(payload.get("note") or "")[:200]
    db.commit()
    return get_budget(db, project_id=scope_id if scope_type == SCOPE_PROJECT else "")


def _series_id_of(db: Session, project_id: str) -> str:
    if not project_id:
        return ""
    row = db.query(Project.project_type, Project.parent_project_id).filter(Project.id == project_id).first()
    if row is None:
        return project_id
    project_type, parent_id = str(row[0] or ""), str(row[1] or "")
    if project_type == "episode" and parent_id:
        return parent_id
    return project_id


def effective_limits(db: Session, *, project_id: str = "") -> dict[str, Any]:
    """生效预算：项目自身 > 父系列 > 全局。"""

    rows = db.query(BudgetConfig).filter(BudgetConfig.enabled.is_(True)).all()
    by_scope = {(str(row.scope_type), str(row.scope_id or "")): row for row in rows}
    series_id = _series_id_of(db, project_id) if project_id else ""
    candidates: list[tuple[str, BudgetConfig | None]] = []
    if project_id:
        candidates.append((SCOPE_PROJECT, by_scope.get((SCOPE_PROJECT, project_id))))
        if series_id and series_id != project_id:
            candidates.append((SCOPE_PROJECT, by_scope.get((SCOPE_PROJECT, series_id))))
    candidates.append((SCOPE_GLOBAL, by_scope.get((SCOPE_GLOBAL, ""))))
    for source, row in candidates:
        if row is None:
            continue
        dto = _budget_dto(row)
        dto["source"] = source
        dto["source_label"] = "项目预算" if source == SCOPE_PROJECT else "全局预算"
        return dto
    return {
        "scope_type": "",
        "scope_id": "",
        "source": "none",
        "source_label": "未设置预算",
        "currency": DEFAULT_CURRENCY,
        "soft_cost_micro": None,
        "hard_cost_micro": None,
        "soft_seconds": None,
        "hard_seconds": None,
        "enabled": False,
        "note": "",
    }


# ---------------------------------------------------------------------------
# 任务工作量与估算
# ---------------------------------------------------------------------------


def _endpoint_ref(capability: str) -> tuple[str, str]:
    """当前生效端点的 (protocol, model)，用于把估算落到具体价目。"""

    try:
        endpoint = get_endpoint(capability)
    except Exception:  # noqa: BLE001 - 端点配置异常不应让估算接口 500
        return "", ""
    return str(endpoint.protocol or ""), str(endpoint.model or "")


def _storyboard_pending_shots(db: Session, project_id: str, shot_ids: list[str] | None = None) -> list[Shot]:
    query = db.query(Shot).filter(Shot.project_id == project_id)
    if shot_ids:
        query = query.filter(Shot.id.in_(list(shot_ids)))
    else:
        query = query.filter(Shot.confirmed.is_(False))
    return query.all()


def job_workload(
    db: Session,
    job_type: str,
    *,
    project_id: str = "",
    shot_id: str = "",
    shot_ids: list[str] | None = None,
) -> list[WorkloadComponent]:
    """按任务类型给出「预计要消耗什么」的清单。

    数量来自项目当前真实状态（镜头数 / 台词字符数 / 镜头时长），未知模型参数（如
    首次解析会产出多少角色与场景）用服务端既有配额上限作为上界，并在 note 中说明。
    """

    job_type = str(job_type or JOB_TYPE_UNKNOWN)
    components: list[WorkloadComponent] = []
    llm_provider, llm_model = _endpoint_ref("script")
    image_provider, image_model = _endpoint_ref("image")
    video_provider, video_model = _endpoint_ref("video")
    voice_provider, voice_model = _endpoint_ref("voice")

    project = db.query(Project).filter(Project.id == project_id).first() if project_id else None
    shots: list[Shot] = []
    if project_id:
        query = db.query(Shot).filter(Shot.project_id == project_id)
        if shot_id:
            query = query.filter(Shot.id == shot_id)
        shots = query.order_by(Shot.sequence).all()
    elif shot_id:
        single = db.query(Shot).filter(Shot.id == shot_id).first()
        shots = [single] if single is not None else []
        project = db.query(Project).filter(Project.id == shots[0].project_id).first() if shots else project

    if job_type == JOB_TYPE_SCRIPT_PIPELINE:
        script_chars = len(str(getattr(project, "input_text", "") or ""))
        input_tokens = max(_MIN_INPUT_TOKENS, script_chars // _CHARS_PER_TOKEN)
        # 解析 + 分镜两次 LLM 调用；输出按端点 max_tokens 上界估算。
        max_tokens = 4096
        try:
            from config import settings as app_settings

            max_tokens = int(app_settings.LLM_MAX_TOKENS)
        except Exception:  # noqa: BLE001
            max_tokens = 4096
        components.append(
            WorkloadComponent(
                capability=CAPABILITY_LLM,
                provider=llm_provider,
                model=llm_model,
                quantity=input_tokens,
                secondary_quantity=max_tokens,
                calls=2,
                label="剧本解析与分镜拆解（2 次调用，输出按上限估算）",
            )
        )
        asset_count = 0
        if project_id:
            asset_count = int(db.query(Character).filter(Character.project_id == project_id).count()) + int(
                db.query(SceneAsset).filter(SceneAsset.project_id == project_id).count()
            )
        if not asset_count:
            try:
                from config import settings as app_settings

                asset_count = int(app_settings.LLM_MAX_CHARACTERS) + int(app_settings.LLM_MAX_SCENES)
            except Exception:  # noqa: BLE001
                asset_count = 14
        if asset_count:
            components.append(
                WorkloadComponent(
                    capability=CAPABILITY_IMAGE,
                    provider=image_provider,
                    model=image_model,
                    quantity=int(asset_count),
                    calls=int(asset_count),
                    label="角色三视图与场景基准图",
                )
            )

    elif job_type in (JOB_TYPE_STORYBOARD, JOB_TYPE_ASSET_GENERATION):
        pending = _storyboard_pending_shots(db, project_id, shot_ids) if project_id else []
        count = len(pending)
        if count:
            components.append(
                WorkloadComponent(
                    capability=CAPABILITY_IMAGE,
                    provider=image_provider,
                    model=image_model,
                    quantity=count,
                    calls=count,
                    resolution=str(getattr(project, "output_format", "") or ""),
                    label=f"{count} 个镜头的定稿故事板",
                )
            )

    elif job_type == JOB_TYPE_SHOT_IMAGE:
        count = len(shots) or (1 if shot_id else 0)
        if count:
            components.append(
                WorkloadComponent(
                    capability=CAPABILITY_IMAGE,
                    provider=image_provider,
                    model=image_model,
                    quantity=count,
                    calls=count,
                    label="镜头故事板",
                )
            )

    elif job_type == JOB_TYPE_SHOT_AUDIO:
        characters = sum(_dialogue_chars(shot) for shot in shots)
        if characters:
            components.append(
                WorkloadComponent(
                    capability=CAPABILITY_TTS,
                    provider=voice_provider,
                    model=voice_model,
                    quantity=characters,
                    calls=max(1, len(shots)),
                    label="镜头配音",
                )
            )

    elif job_type == JOB_TYPE_SHOT_VIDEO:
        seconds = sum(max(0.0, float(shot.duration or 0)) for shot in shots)
        characters = sum(_dialogue_chars(shot) for shot in shots)
        if seconds:
            components.append(
                WorkloadComponent(
                    capability=CAPABILITY_VIDEO,
                    provider=video_provider,
                    model=video_model,
                    quantity=int(round(seconds)),
                    calls=max(1, len(shots)),
                    resolution=str(getattr(project, "resolution", "") or ""),
                    label=f"{len(shots) or 1} 个镜头视频（共 {int(round(seconds))} 秒）",
                )
            )
        if characters:
            components.append(
                WorkloadComponent(
                    capability=CAPABILITY_TTS,
                    provider=voice_provider,
                    model=voice_model,
                    quantity=characters,
                    calls=max(1, len(shots)),
                    label="配套配音",
                )
            )

    elif job_type == JOB_TYPE_RENDER:
        seconds = sum(max(0.0, float(shot.duration or 0)) for shot in shots)
        if not seconds:
            seconds = 60.0
        components.append(
            WorkloadComponent(
                capability=CAPABILITY_FFMPEG,
                provider="local",
                model="ffmpeg",
                quantity=int(round(seconds)),
                calls=2,
                resolution=str(getattr(project, "resolution", "") or ""),
                label=f"成片合成（{int(round(seconds))} 秒编码）",
            )
        )

    return components


def _history_seconds(db: Session, job_type: str) -> int | None:
    from config import settings as app_settings

    rows = (
        db.query(BackgroundJob.started_at, BackgroundJob.finished_at)
        .filter(
            BackgroundJob.job_type == job_type,
            BackgroundJob.status == "completed",
            BackgroundJob.started_at.isnot(None),
            BackgroundJob.finished_at.isnot(None),
        )
        .order_by(BackgroundJob.finished_at.desc())
        .limit(20)
        .all()
    )
    samples: list[float] = []
    for started, finished in rows:
        try:
            seconds = float((finished - started).total_seconds())
        except TypeError:
            continue
        if seconds > 0:
            samples.append(seconds)
    if len(samples) < max(1, int(app_settings.JOB_ETA_MIN_SAMPLES)):
        return None
    samples.sort()
    middle = len(samples) // 2
    median = samples[middle] if len(samples) % 2 else (samples[middle - 1] + samples[middle]) / 2
    return max(1, int(round(median)))


def estimate_job(
    db: Session,
    *,
    job_type: str,
    project_id: str = "",
    shot_id: str = "",
    shot_ids: list[str] | None = None,
) -> dict[str, Any]:
    """估算一次任务的成本与耗时（不落库，供提交前确认与预算检查共用）。"""

    components = job_workload(db, job_type, project_id=project_id, shot_id=shot_id, shot_ids=shot_ids)
    currency = DEFAULT_CURRENCY
    total_cost = 0
    cost_known = bool(components)
    unknown: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    heuristic_seconds = 0
    for component in components:
        price = pricing_service.resolve_price(
            db,
            component.capability,
            component.provider,
            component.model,
            resolution=component.resolution,
        )
        currency = price.currency or currency
        cost = price.cost_micro(component.quantity, component.secondary_quantity) if price.priced else None
        entry = {
            "capability": component.capability,
            "label": CAPABILITY_LABELS.get(component.capability, component.capability),
            "component_label": component.label,
            "provider": component.provider,
            "model": component.model,
            "quantity": int(component.quantity),
            "secondary_quantity": int(component.secondary_quantity),
            "base_unit": CAPABILITY_BASE_UNITS.get(component.capability, ""),
            "resolution": component.resolution,
            "calls": int(component.calls),
            "cost_micro": int(cost) if cost is not None else None,
            "cost_known": cost is not None,
            "estimated_seconds": component.estimated_seconds(),
        }
        detail.append(entry)
        heuristic_seconds += entry["estimated_seconds"]
        if cost is None:
            cost_known = False
            unknown.append(
                {
                    "capability": component.capability,
                    "label": entry["label"],
                    "provider": component.provider,
                    "model": component.model,
                    "quantity": int(component.quantity),
                    "reason": "未配置该 provider / 模型的单价" if component.provider else "未配置该能力的单价",
                }
            )
        else:
            total_cost += int(cost)

    history = _history_seconds(db, str(job_type))
    if history is not None:
        seconds, duration_source = history, "history"
    elif heuristic_seconds:
        seconds, duration_source = heuristic_seconds, "heuristic"
    else:
        seconds, duration_source = None, "unknown"

    note = ""
    if not components:
        note = "当前项目状态下没有需要执行的工作量，无法估算"
    elif unknown:
        note = "部分能力的单价尚未配置，成本显示为未知"

    return {
        "job_type": str(job_type),
        "job_type_label": job_type_label(str(job_type)),
        "project_id": project_id,
        "shot_id": shot_id,
        "currency": currency,
        "estimated_cost_micro": int(total_cost) if cost_known else None,
        "cost_known": bool(cost_known and components),
        "estimated_seconds": int(seconds) if seconds is not None else None,
        "duration_source": duration_source,
        "components": detail,
        "unknown_components": unknown,
        "note": note,
    }


# ---------------------------------------------------------------------------
# 预算检查与预留
# ---------------------------------------------------------------------------


def project_elapsed_seconds(db: Session, *, project_id: str = "", series_id: str = "") -> int:
    """项目 / 剧集已消耗的任务执行时长（任务真实起止时间之和）。"""

    query = db.query(BackgroundJob)
    if project_id:
        query = query.filter(BackgroundJob.project_id == project_id)
    elif series_id:
        episode_ids = [
            str(row[0])
            for row in db.query(Project.id).filter(Project.parent_project_id == series_id).all()
        ]
        query = query.filter(BackgroundJob.project_id.in_([series_id, *episode_ids]))
    else:
        return 0
    now = datetime.utcnow()
    total = 0.0
    for job in query.all():
        started = job.started_at or job.created_at
        if started is None:
            continue
        end = job.finished_at if str(job.status) in TERMINAL_STATUSES and job.finished_at else now
        try:
            total += max(0.0, (end - started).total_seconds())
        except TypeError:
            continue
    return int(round(total))


def _active_reservations(db: Session, *, project_id: str = "", series_id: str = "") -> list[BudgetReservation]:
    query = db.query(BudgetReservation).filter(BudgetReservation.status == RESERVATION_ACTIVE)
    if project_id:
        query = query.filter(BudgetReservation.project_id == project_id)
    elif series_id:
        episode_ids = [
            str(row[0]) for row in db.query(Project.id).filter(Project.parent_project_id == series_id).all()
        ]
        query = query.filter(BudgetReservation.project_id.in_([series_id, *episode_ids]))
    return query.all()


def evaluate_limits(
    limits: dict[str, Any],
    *,
    used_cost_micro: int,
    used_seconds: int = 0,
    reserved_cost_micro: int = 0,
    reserved_seconds: int = 0,
    estimate_cost_micro: int | None = None,
    estimate_seconds: int | None = None,
    cost_known: bool = True,
) -> dict[str, Any]:
    """比对预算，给出 ok / soft_exceeded / hard_exceeded。"""

    currency = str(limits.get("currency") or DEFAULT_CURRENCY)
    hard_cost = limits.get("hard_cost_micro")
    soft_cost = limits.get("soft_cost_micro")
    hard_seconds = limits.get("hard_seconds")
    soft_seconds = limits.get("soft_seconds")
    unlimited = not any(value is not None for value in (hard_cost, soft_cost, hard_seconds, soft_seconds))

    committed_cost = int(used_cost_micro) + int(reserved_cost_micro)
    committed_seconds = int(used_seconds) + int(reserved_seconds)
    projected_cost = committed_cost + (int(estimate_cost_micro) if estimate_cost_micro else 0)
    projected_seconds = committed_seconds + (int(estimate_seconds) if estimate_seconds else 0)

    level = LEVEL_UNLIMITED if unlimited else LEVEL_OK
    code = ""
    message = ""
    reason = ""

    if hard_cost is not None:
        if committed_cost > int(hard_cost) or (estimate_cost_micro is not None and projected_cost > int(hard_cost)):
            level, code = LEVEL_HARD_EXCEEDED, CODE_BUDGET_EXCEEDED
            reason = "cost"
            message = (
                f"项目硬预算 {_cents_label(int(hard_cost), currency)} 已不足以启动该任务"
                f"（已用 {_cents_label(committed_cost, currency)}，本次预计 "
                f"{_cents_label(estimate_cost_micro, currency)}）"
            )
    if level != LEVEL_HARD_EXCEEDED and hard_seconds is not None:
        if committed_seconds > int(hard_seconds) or (
            estimate_seconds is not None and projected_seconds > int(hard_seconds)
        ):
            level, code = LEVEL_HARD_EXCEEDED, CODE_BUDGET_EXCEEDED
            reason = "seconds"
            message = (
                f"项目硬时长预算 {int(hard_seconds)} 秒已不足以启动该任务"
                f"（已用 {committed_seconds} 秒，本次预计 {estimate_seconds if estimate_seconds is not None else '未知'} 秒）"
            )
    if level not in (LEVEL_HARD_EXCEEDED,):
        if soft_cost is not None and (committed_cost >= int(soft_cost) or projected_cost > int(soft_cost)):
            level, code = LEVEL_SOFT_EXCEEDED, CODE_BUDGET_SOFT_EXCEEDED
            reason = "cost"
            if committed_cost >= int(soft_cost):
                # 已用金额本身已达/超出软预算：这是「实际超支」，不是预测。
                message = (
                    f"项目软预算 {_cents_label(int(soft_cost), currency)} 已超支"
                    f"（已用 {_cents_label(committed_cost, currency)}，本次预计 "
                    f"{_cents_label(estimate_cost_micro, currency)}），任务仍会继续执行"
                )
            else:
                # 已用尚未超支，只是「已用 + 本次预计」的投影会超：文案必须区分，
                # 并明确给出已用、本次预计与预算上限。
                message = (
                    f"项目软预算 {_cents_label(int(soft_cost), currency)} 预计将超支"
                    f"（已用 {_cents_label(committed_cost, currency)}，本次预计 "
                    f"{_cents_label(estimate_cost_micro, currency)}，预算上限 "
                    f"{_cents_label(int(soft_cost), currency)}），任务仍会继续执行"
                )
        elif soft_seconds is not None and (
            committed_seconds >= int(soft_seconds) or projected_seconds > int(soft_seconds)
        ):
            level, code = LEVEL_SOFT_EXCEEDED, CODE_BUDGET_SOFT_EXCEEDED
            reason = "seconds"
            if committed_seconds >= int(soft_seconds):
                message = (
                    f"项目软时长预算 {int(soft_seconds)} 秒已超支（已用 {committed_seconds} 秒），任务仍会继续执行"
                )
            else:
                message = (
                    f"项目软时长预算 {int(soft_seconds)} 秒预计将超支"
                    f"（已用 {committed_seconds} 秒，本次预计 {projected_seconds - committed_seconds} 秒，"
                    f"预算上限 {int(soft_seconds)} 秒），任务仍会继续执行"
                )

    if level in (LEVEL_OK, LEVEL_UNLIMITED) and not cost_known:
        level = LEVEL_OK
        message = "该任务的成本未知（未配置对应模型单价），已跳过金额校验"

    return {
        "level": level,
        "code": code,
        "message": message,
        "reason": reason,
        "unlimited": unlimited,
        "currency": currency,
        "cost_known": bool(cost_known),
        "soft_cost_micro": int(soft_cost) if soft_cost is not None else None,
        "hard_cost_micro": int(hard_cost) if hard_cost is not None else None,
        "soft_seconds": int(soft_seconds) if soft_seconds is not None else None,
        "hard_seconds": int(hard_seconds) if hard_seconds is not None else None,
        "used_cost_micro": int(used_cost_micro),
        "reserved_cost_micro": int(reserved_cost_micro),
        "committed_cost_micro": committed_cost,
        "projected_cost_micro": projected_cost,
        "used_seconds": int(used_seconds),
        "reserved_seconds": int(reserved_seconds),
        "projected_seconds": projected_seconds,
        "estimate_cost_micro": int(estimate_cost_micro) if estimate_cost_micro is not None else None,
        "estimate_seconds": int(estimate_seconds) if estimate_seconds is not None else None,
        "budget_source": str(limits.get("source") or "none"),
        "budget_source_label": str(limits.get("source_label") or "未设置预算"),
    }


def budget_state(
    db: Session,
    *,
    project_id: str = "",
    estimate_cost_micro: int | None = None,
    estimate_seconds: int | None = None,
    cost_known: bool = True,
    exclude_job_key: str = "",
) -> dict[str, Any]:
    """给定项目当前的预算状态（含本次估算）。"""

    limits = effective_limits(db, project_id=project_id)
    series_id = _series_id_of(db, project_id) if project_id else ""
    # 预算挂在项目上时按项目统计；挂在父系列上时按整个系列统计。
    if limits.get("source") == SCOPE_PROJECT and str(limits.get("scope_id")) == series_id and series_id != project_id:
        used = usage_service.summarize(db, series_id=series_id)
        used_seconds = project_elapsed_seconds(db, series_id=series_id)
        reservations = _active_reservations(db, series_id=series_id)
    else:
        used = usage_service.summarize(db, project_id=project_id)
        used_seconds = project_elapsed_seconds(db, project_id=project_id)
        reservations = _active_reservations(db, project_id=project_id)
    reserved_cost = sum(
        int(row.estimated_cost_micro or 0)
        for row in reservations
        if row.cost_known and row.reservation_key != exclude_job_key
    )
    reserved_seconds = sum(
        int(row.estimated_seconds or 0) for row in reservations if row.reservation_key != exclude_job_key
    )
    state = evaluate_limits(
        limits,
        used_cost_micro=int(used.get("cost_micro") or 0),
        used_seconds=int(used_seconds or 0),
        reserved_cost_micro=int(reserved_cost),
        reserved_seconds=int(reserved_seconds),
        estimate_cost_micro=estimate_cost_micro,
        estimate_seconds=estimate_seconds,
        cost_known=bool(cost_known),
    )
    state["usage"] = {
        "call_count": int(used.get("call_count") or 0),
        "unknown_call_count": int(used.get("unknown_call_count") or 0),
        "failed_call_count": int(used.get("failed_call_count") or 0),
        "cost_known": bool(used.get("cost_known", True)),
    }
    state["reservation_count"] = len([row for row in reservations if row.reservation_key != exclude_job_key])
    return state


def check_and_reserve(
    *,
    job_key: str,
    job_type: str,
    project_id: str = "",
    shot_id: str = "",
    job_id: str = "",
) -> BudgetDecision:
    """任务抢占前的预算检查：通过则预留额度，超硬预算则返回阻断结果。"""

    db = SessionLocal()
    try:
        estimate = estimate_job(db, job_type=job_type, project_id=project_id, shot_id=shot_id)
        state = budget_state(
            db,
            project_id=project_id,
            estimate_cost_micro=estimate.get("estimated_cost_micro"),
            estimate_seconds=estimate.get("estimated_seconds"),
            cost_known=bool(estimate.get("cost_known")),
            exclude_job_key=job_key,
        )
        estimate_id = f"job:{job_key}" if job_key else f"adhoc:{uuid.uuid4().hex}"
        usage_service.save_estimate(
            estimate_key=estimate_id,
            job_key=job_key,
            job_id=job_id,
            job_type=job_type,
            project_id=project_id,
            series_id=_series_id_of(db, project_id) if project_id else "",
            shot_id=shot_id,
            currency=str(estimate.get("currency") or DEFAULT_CURRENCY),
            estimated_cost_micro=estimate.get("estimated_cost_micro"),
            cost_known=bool(estimate.get("cost_known")),
            estimated_seconds=estimate.get("estimated_seconds"),
            duration_source=str(estimate.get("duration_source") or "unknown"),
            components=list(estimate.get("components") or []),
            unknown_components=list(estimate.get("unknown_components") or []),
            note=str(estimate.get("note") or ""),
        )
        blocked = state.get("level") == LEVEL_HARD_EXCEEDED
        if not blocked:
            _reserve(
                db,
                job_key=job_key,
                job_id=job_id,
                job_type=job_type,
                project_id=project_id,
                series_id=_series_id_of(db, project_id) if project_id else "",
                estimate=estimate,
            )
        return BudgetDecision(
            allowed=not blocked,
            level=str(state.get("level") or LEVEL_OK),
            code=str(state.get("code") or ""),
            message=str(state.get("message") or ""),
            details=state,
            estimate=estimate,
        )
    except Exception:  # noqa: BLE001 - 预算检查失败时放行（不能被预算模块的异常卡死生产）
        logger.warning("预算检查失败，已放行本次任务: job_key=%s", job_key, exc_info=True)
        return BudgetDecision(allowed=True, level=LEVEL_OK, message="预算检查异常，已放行本次任务")
    finally:
        db.close()


def _reserve(
    db: Session,
    *,
    job_key: str,
    job_id: str,
    job_type: str,
    project_id: str,
    series_id: str,
    estimate: dict[str, Any],
) -> None:
    if not job_key:
        return
    row = db.query(BudgetReservation).filter(BudgetReservation.reservation_key == job_key).first()
    if row is None:
        row = BudgetReservation(id=uuid.uuid4().hex, reservation_key=job_key)
        db.add(row)
    row.job_id = str(job_id or "")
    row.job_key = str(job_key)
    row.job_type = str(job_type or "")
    row.project_id = str(project_id or "")
    row.series_id = str(series_id or "")
    row.currency = str(estimate.get("currency") or DEFAULT_CURRENCY)
    row.estimated_cost_micro = (
        int(estimate["estimated_cost_micro"]) if estimate.get("estimated_cost_micro") is not None else None
    )
    row.cost_known = bool(estimate.get("cost_known"))
    row.estimated_seconds = (
        int(estimate["estimated_seconds"]) if estimate.get("estimated_seconds") is not None else None
    )
    row.status = RESERVATION_ACTIVE
    row.released_at = None
    db.commit()


def attach_job_id(job_key: str, job_id: str) -> int:
    """抢占成功后把 job_id 补进预留行，便于按任务反查预留。"""

    if not job_key or not job_id:
        return 0
    db = SessionLocal()
    try:
        updated = (
            db.query(BudgetReservation)
            .filter(BudgetReservation.reservation_key == job_key, BudgetReservation.job_id == "")
            .update({BudgetReservation.job_id: str(job_id)[:200]}, synchronize_session=False)
        )
        db.commit()
        return int(updated or 0)
    finally:
        db.close()


def release_reservation(job_key: str) -> bool:
    """任务终结时释放预留（幂等；没有预留时返回 False）。"""

    if not job_key:
        return False
    db = SessionLocal()
    try:
        row = (
            db.query(BudgetReservation)
            .filter(BudgetReservation.reservation_key == job_key, BudgetReservation.status == RESERVATION_ACTIVE)
            .first()
        )
        if row is None:
            return False
        row.status = RESERVATION_RELEASED
        row.released_at = datetime.utcnow()
        db.commit()
        return True
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 汇总（项目 / 剧集 / 镜头 / 任务维度）
# ---------------------------------------------------------------------------


def _remaining_workload(db: Session, project_id: str) -> list[WorkloadComponent]:
    """项目剩下的活：未出故事板的镜头 + 未出视频的镜头 + 成片合成。"""

    if not project_id:
        return []
    shots = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
    if not shots:
        return []
    project = db.query(Project).filter(Project.id == project_id).first()
    image_provider, image_model = _endpoint_ref("image")
    video_provider, video_model = _endpoint_ref("video")
    voice_provider, voice_model = _endpoint_ref("voice")

    pending_storyboard = [shot for shot in shots if not (shot.storyboard_path or shot.image_path)]
    pending_video = [shot for shot in shots if not shot.video_path]
    components: list[WorkloadComponent] = []
    if pending_storyboard:
        components.append(
            WorkloadComponent(
                capability=CAPABILITY_IMAGE,
                provider=image_provider,
                model=image_model,
                quantity=len(pending_storyboard),
                calls=len(pending_storyboard),
                label=f"待生成故事板 {len(pending_storyboard)} 个镜头",
            )
        )
    if pending_video:
        seconds = int(round(sum(max(0.0, float(shot.duration or 0)) for shot in pending_video)))
        characters = sum(_dialogue_chars(shot) for shot in pending_video)
        components.append(
            WorkloadComponent(
                capability=CAPABILITY_VIDEO,
                provider=video_provider,
                model=video_model,
                quantity=seconds,
                calls=len(pending_video),
                resolution=str(getattr(project, "resolution", "") or ""),
                label=f"待生成视频 {len(pending_video)} 个镜头（{seconds} 秒）",
            )
        )
        if characters:
            components.append(
                WorkloadComponent(
                    capability=CAPABILITY_TTS,
                    provider=voice_provider,
                    model=voice_model,
                    quantity=characters,
                    calls=len(pending_video),
                    label="待生成配音",
                )
            )
    if all(shot.video_path for shot in shots):
        components.append(
            WorkloadComponent(
                capability=CAPABILITY_FFMPEG,
                provider="local",
                model="ffmpeg",
                quantity=int(round(sum(max(0.0, float(shot.duration or 0)) for shot in shots))) or 60,
                calls=2,
                resolution=str(getattr(project, "resolution", "") or ""),
                label="成片合成",
            )
        )
    return components


def _cost_components(db: Session, components: list[WorkloadComponent]) -> dict[str, Any]:
    total = 0
    known = bool(components)
    unknown: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    seconds = 0
    currency = DEFAULT_CURRENCY
    for component in components:
        price = pricing_service.resolve_price(
            db, component.capability, component.provider, component.model, resolution=component.resolution
        )
        currency = price.currency or currency
        cost = price.cost_micro(component.quantity, component.secondary_quantity) if price.priced else None
        component_seconds = component.estimated_seconds()
        seconds += component_seconds
        detail.append(
            {
                "capability": component.capability,
                "label": CAPABILITY_LABELS.get(component.capability, component.capability),
                "component_label": component.label,
                "provider": component.provider,
                "model": component.model,
                "quantity": int(component.quantity),
                "secondary_quantity": int(component.secondary_quantity),
                "base_unit": CAPABILITY_BASE_UNITS.get(component.capability, ""),
                "cost_micro": int(cost) if cost is not None else None,
                "cost_known": cost is not None,
                "estimated_seconds": component_seconds,
            }
        )
        if cost is None:
            known = False
            unknown.append(
                {
                    "capability": component.capability,
                    "label": CAPABILITY_LABELS.get(component.capability, component.capability),
                    "component_label": component.label,
                    "provider": component.provider,
                    "model": component.model,
                    "reason": "未配置该 provider / 模型的单价" if component.provider else "未配置该能力的单价",
                }
            )
        else:
            total += int(cost)
    return {
        "currency": currency,
        "cost_micro": int(total) if known else None,
        "cost_known": bool(known and components),
        "partial_cost_micro": int(total),
        "seconds": int(seconds) if components else None,
        "components": detail,
        "unknown_components": unknown,
    }


def project_summary(db: Session, *, project_id: str = "", series_id: str = "") -> dict[str, Any]:
    """项目 / 剧集页所需的预算、已用成本、预计成本与预计耗时。"""

    if not project_id and not series_id:
        raise ValueError("必须提供 project_id 或 series_id")
    resolved_series = series_id or _series_id_of(db, project_id)
    is_episode = bool(project_id) and resolved_series != project_id

    used = usage_service.summarize(db, project_id=project_id) if project_id else usage_service.summarize(
        db, series_id=resolved_series
    )
    elapsed_seconds = project_elapsed_seconds(db, project_id=project_id) if project_id else project_elapsed_seconds(
        db, series_id=resolved_series
    )
    reservations = (
        _active_reservations(db, project_id=project_id)
        if project_id
        else _active_reservations(db, series_id=resolved_series)
    )
    reserved_cost = sum(int(row.estimated_cost_micro or 0) for row in reservations if row.cost_known)
    reserved_seconds = sum(int(row.estimated_seconds or 0) for row in reservations)

    remaining: dict[str, Any] = {
        "currency": DEFAULT_CURRENCY,
        "cost_micro": None,
        "cost_known": False,
        "seconds": None,
        "components": [],
        "unknown_components": [],
    }
    if project_id:
        remaining = _cost_components(db, _remaining_workload(db, project_id))

    limits = effective_limits(db, project_id=project_id or resolved_series)
    state = evaluate_limits(
        limits,
        used_cost_micro=int(used.get("cost_micro") or 0),
        used_seconds=int(elapsed_seconds),
        reserved_cost_micro=int(reserved_cost),
        reserved_seconds=int(reserved_seconds),
        estimate_cost_micro=remaining.get("cost_micro"),
        estimate_seconds=remaining.get("seconds"),
        cost_known=bool(remaining.get("cost_known")),
    )

    return {
        "project_id": project_id,
        "series_id": resolved_series,
        "is_episode": is_episode,
        "currency": str(limits.get("currency") or DEFAULT_CURRENCY),
        "used": {
            "cost_micro": int(used.get("cost_micro") or 0),
            "cost_known": bool(used.get("cost_known", True)),
            "unknown_call_count": int(used.get("unknown_call_count") or 0),
            "failed_call_count": int(used.get("failed_call_count") or 0),
            "call_count": int(used.get("call_count") or 0),
            "seconds": int(elapsed_seconds),
            "by_capability": used.get("by_capability") or [],
        },
        "reserved": {
            "cost_micro": int(reserved_cost),
            "seconds": int(reserved_seconds),
            "count": len(reservations),
        },
        "remaining": remaining,
        "budget": limits,
        "status": state,
        "by_job_type": usage_service.group_usage(
            db, "job_type", **({"project_id": project_id} if project_id else {"series_id": resolved_series})
        ),
        "by_shot": usage_service.group_usage(
            db, "shot", **({"project_id": project_id} if project_id else {"series_id": resolved_series})
        ),
        "by_capability": usage_service.group_usage(
            db, "capability", **({"project_id": project_id} if project_id else {"series_id": resolved_series})
        ),
    }


def series_episode_breakdown(db: Session, series_id: str) -> list[dict[str, Any]]:
    """系列下各剧集的成本汇总（剧集列表页用）。"""

    if not series_id:
        return []
    episodes = db.query(Project).filter(Project.parent_project_id == series_id).order_by(Project.episode_number).all()
    breakdown: list[dict[str, Any]] = []
    for episode in episodes:
        used = usage_service.summarize(db, project_id=str(episode.id))
        elapsed = project_elapsed_seconds(db, project_id=str(episode.id))
        limits = effective_limits(db, project_id=str(episode.id))
        state = evaluate_limits(
            limits,
            used_cost_micro=int(used.get("cost_micro") or 0),
            used_seconds=int(elapsed),
            cost_known=bool(used.get("cost_known", True)),
        )
        breakdown.append(
            {
                "project_id": str(episode.id),
                "title": str(episode.title or ""),
                "episode_number": int(episode.episode_number or 0),
                "status": str(episode.status or ""),
                "used": {
                    "cost_micro": int(used.get("cost_micro") or 0),
                    "cost_known": bool(used.get("cost_known", True)),
                    "unknown_call_count": int(used.get("unknown_call_count") or 0),
                    "seconds": int(elapsed),
                },
                "budget_status": state,
            }
        )
    return breakdown


__all__ = [
    "CODE_BUDGET_EXCEEDED",
    "CODE_BUDGET_SOFT_EXCEEDED",
    "LEVEL_HARD_EXCEEDED",
    "LEVEL_OK",
    "LEVEL_SOFT_EXCEEDED",
    "LEVEL_UNLIMITED",
    "RESERVATION_ACTIVE",
    "RESERVATION_RELEASED",
    "BudgetDecision",
    "WorkloadComponent",
    "attach_job_id",
    "budget_state",
    "check_and_reserve",
    "effective_limits",
    "estimate_job",
    "evaluate_limits",
    "get_budget",
    "job_workload",
    "project_elapsed_seconds",
    "project_summary",
    "release_reservation",
    "save_budget",
    "series_episode_breakdown",
]
