"""视频生成策略 A/B 离线评估与确定性报告生成。

本模块只读取已经生成好的同批镜头结果并汇总，不调用视频 Provider，也不写回
``settings``、默认 Provider、质量阈值或线上生成策略。四组策略必须覆盖同一批
镜头，且每个镜头在四组中的 ``execution_plan_hash`` 必须一致，避免拿不同计划
的结果做不公平比较。

输入/输出都以整数 micro 保存成本（1 单位 = 1_000_000 micro），耗时使用
``elapsed_ms``。报告中的比率同时给出分子、分母与覆盖率，不把未检测样本伪装成
通过；少量样本只形成观测报告，不产生任何自动策略变更。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from services.atomic_json import atomic_write_json, atomic_write_text

REPORT_SCHEMA_VERSION = 1
INPUT_SCHEMA_VERSION = 1

STRATEGY_ORDER = ("A", "B", "C", "D")
STRATEGY_DEFINITIONS: dict[str, dict[str, str]] = {
    "A": {
        "label": "当前首帧 I2V",
        "mode": "first_frame_i2v",
        "description": "使用已审核首帧作为唯一条件参考的当前 I2V 策略",
    },
    "B": {
        "label": "首帧 I2V + 条件连续性参考",
        "mode": "first_frame_i2v_plus_conditional_continuity",
        "description": "首帧 I2V，并按连续性条件附加上一镜尾帧等连续性参考",
    },
    "C": {
        "label": "多参考 R2V",
        "mode": "multi_reference_r2v",
        "description": "使用角色、场景与连续性等多参考素材的 R2V 策略",
    },
    "D": {
        "label": "首尾帧或视频续写",
        "mode": "first_last_frame_or_video_continuation",
        "description": "首尾帧约束生成，或在 Provider 支持时采用视频续写",
    },
}

SHOT_TYPE_ORDER = ("dialogue", "close_up", "ordinary_action", "complex_action", "cross_scene")
SHOT_TYPE_DEFINITIONS: dict[str, dict[str, str]] = {
    "dialogue": {"label": "对白", "description": "以人物对白、口型与反应为主的镜头"},
    "close_up": {"label": "近景", "description": "面部、手部或局部主体占主导的近景镜头"},
    "ordinary_action": {"label": "普通动作", "description": "低复杂度走位、转身、拿放等动作镜头"},
    "complex_action": {"label": "复杂动作", "description": "追逐、打斗、翻越等多拍复杂动作镜头"},
    "cross_scene": {"label": "跨场景", "description": "场景、空间或时间线切换的镜头"},
}

_STRATEGY_ALIASES = {
    "a": "A",
    "a_first_frame_i2v": "A",
    "first_frame_i2v": "A",
    "current_first_frame_i2v": "A",
    "b": "B",
    "b_first_frame_i2v_continuity": "B",
    "first_frame_i2v_plus_conditional_continuity": "B",
    "first_frame_i2v_continuity": "B",
    "c": "C",
    "c_multi_reference_r2v": "C",
    "multi_reference_r2v": "C",
    "r2v_multi_reference": "C",
    "d": "D",
    "d_first_last_frame_or_continuation": "D",
    "first_last_frame_or_video_continuation": "D",
    "first_last_frame": "D",
    "video_continuation": "D",
}

_SHOT_TYPE_ALIASES = {
    "dialogue": "dialogue",
    "对白": "dialogue",
    "对话": "dialogue",
    "talking": "dialogue",
    "speech": "dialogue",
    "close_up": "close_up",
    "closeup": "close_up",
    "close-up": "close_up",
    "近景": "close_up",
    "ordinary_action": "ordinary_action",
    "normal_action": "ordinary_action",
    "普通动作": "ordinary_action",
    "complex_action": "complex_action",
    "复杂动作": "complex_action",
    "cross_scene": "cross_scene",
    "scene_transition": "cross_scene",
    "跨场景": "cross_scene",
}

_MANUAL_STATUS_ALIASES = {
    "accepted": "accepted",
    "accept": "accepted",
    "approved": "accepted",
    "pass": "accepted",
    "passed": "accepted",
    "通过": "accepted",
    "接受": "accepted",
    "rejected": "rejected",
    "reject": "rejected",
    "declined": "rejected",
    "failed": "rejected",
    "不通过": "rejected",
    "拒绝": "rejected",
    "pending": "pending",
    "待审": "pending",
    "review": "pending",
    "unknown": "unknown",
    "未知": "unknown",
}


class VideoABEvaluationError(ValueError):
    """A/B 输入契约不满足；拒绝生成可能误导决策的报告。"""


def _canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VideoABEvaluationError(f"{label} 必须是对象")
    return value


def _required_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise VideoABEvaluationError(f"{label} 必须是数组")
    return value


def _string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if value is None:
        text = ""
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value).strip()
    else:
        raise VideoABEvaluationError(f"{label} 必须是字符串")
    if not allow_empty and not text:
        raise VideoABEvaluationError(f"{label} 不能为空")
    return text


def _optional_bool(value: Any, label: str) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "passed", "pass", "valid", "有效", "通过"}:
            return True
        if normalized in {"false", "0", "no", "failed", "fail", "invalid", "无效", "不通过"}:
            return False
    raise VideoABEvaluationError(f"{label} 必须是布尔值或空值")


def _optional_float(value: Any, label: str, *, minimum: float, maximum: float | None = None) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise VideoABEvaluationError(f"{label} 必须是数字")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VideoABEvaluationError(f"{label} 必须是数字") from exc
    if not math.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
        limit = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise VideoABEvaluationError(f"{label} 必须位于 {limit}")
    return number


def _required_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise VideoABEvaluationError(f"{label} 必须是整数")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise VideoABEvaluationError(f"{label} 必须是整数") from exc
    if isinstance(value, float) and not value.is_integer():
        raise VideoABEvaluationError(f"{label} 必须是整数")
    if number < minimum:
        raise VideoABEvaluationError(f"{label} 不能小于 {minimum}")
    return number


def _round_rate(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else numerator / denominator


def _normalize_strategy(value: Any, label: str) -> str:
    text = _string(value, label)
    if text in STRATEGY_DEFINITIONS:
        return text
    normalized = text.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in _STRATEGY_ALIASES:
        return _STRATEGY_ALIASES[normalized]
    raise VideoABEvaluationError(f"{label} 不是支持的 A/B/C/D 策略: {text}")


def _normalize_shot_type(value: Any, label: str) -> str:
    text = _string(value, label)
    if text in SHOT_TYPE_DEFINITIONS:
        return text
    normalized = text.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in _SHOT_TYPE_ALIASES:
        return _SHOT_TYPE_ALIASES[normalized]
    if text in _SHOT_TYPE_ALIASES:
        return _SHOT_TYPE_ALIASES[text]
    raise VideoABEvaluationError(f"{label} 不是支持的五类镜头类型: {text}")


def _normalize_manual_review(record: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    source = record.get("manual_review", record.get("manual_acceptance"))
    if source is None:
        source = {
            "status": metrics.get("manual_review_status", metrics.get("manual_acceptance_status", "")),
            "accepted": metrics.get("manual_accepted", metrics.get("manual_accept", None)),
            "note": metrics.get("manual_review_note", ""),
        }
    if isinstance(source, bool):
        source = {"accepted": source}
    source_map = _required_mapping(source, "manual_review")
    status_text = str(source_map.get("status") or "").strip().lower()
    status = _MANUAL_STATUS_ALIASES.get(status_text, "") if status_text else ""
    accepted = _optional_bool(source_map.get("accepted"), "manual_review.accepted")
    if status:
        normalized_status = status
    elif accepted is True:
        normalized_status = "accepted"
    elif accepted is False:
        normalized_status = "rejected"
    else:
        normalized_status = "unknown"
    return {
        "status": normalized_status,
        "accepted": True if normalized_status == "accepted" else False if normalized_status == "rejected" else None,
        "note": str(source_map.get("note") or ""),
    }


def _normalize_reference_manifest(value: Any, label: str) -> list[dict[str, Any]]:
    items = _required_list(value if value is not None else [], label)
    output: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        mapping = _required_mapping(item, f"{label}[{index}]")
        try:
            output.append(json.loads(json.dumps(mapping, ensure_ascii=False, sort_keys=True)))
        except (TypeError, ValueError) as exc:
            raise VideoABEvaluationError(f"{label}[{index}] 必须是可序列化对象") from exc
    return output


def _normalize_cost(record: Mapping[str, Any]) -> dict[str, Any]:
    source = record.get("cost")
    if source is None:
        source = {}
    cost_map = _required_mapping(source, "cost")
    raw_amount = cost_map.get("cost_micro", cost_map.get("amount_micro", record.get("cost_micro")))
    currency = str(cost_map.get("currency") or record.get("currency") or "CNY").strip() or "CNY"
    if raw_amount is None or raw_amount == "":
        return {"cost_known": False, "cost_micro": None, "currency": currency}
    amount = _required_int(raw_amount, "cost.cost_micro")
    return {"cost_known": True, "cost_micro": amount, "currency": currency}


def _execution_plan_hash(shot: Mapping[str, Any]) -> str:
    explicit = shot.get("execution_plan_hash", shot.get("plan_hash"))
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    plan = shot.get("execution_plan")
    if isinstance(plan, Mapping):
        recipe_hash = plan.get("recipe_hash")
        if recipe_hash:
            return str(recipe_hash).strip()
        return _canonical_json_hash(plan)
    profile = shot.get("continuity_profile")
    if isinstance(profile, Mapping):
        nested = profile.get("execution_plan")
        if isinstance(nested, Mapping):
            recipe_hash = nested.get("recipe_hash")
            if recipe_hash:
                return str(recipe_hash).strip()
            return _canonical_json_hash(nested)
    raise VideoABEvaluationError(f"镜头 {shot.get('shot_id') or '<unknown>'} 缺少 execution_plan_hash / execution_plan")


def _metric_value(metrics: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def _normalize_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _required_mapping(record.get("metrics") or {}, "metrics")
    validation = _required_mapping(record.get("validation") or {}, "validation")
    structural = _metric_value(metrics, "structural_passed", "structural_pass", "structure_passed")
    if structural is None:
        structural = validation.get("structural_passed", validation.get("passed"))
    first_frame = _metric_value(
        metrics,
        "first_frame_match_score",
        "first_frame_match",
        "first_frame_similarity",
    )
    tail = _metric_value(metrics, "tail_frame_valid", "tail_frame_ok")
    if tail is None:
        tail = validation.get("tail_frame_valid", validation.get("tail_frame_ok"))
    frozen = _metric_value(metrics, "frozen", "freeze_detected", "is_frozen")
    if frozen is None:
        frozen = validation.get("frozen", validation.get("freeze_detected"))
    freeze_seconds = _metric_value(metrics, "freeze_seconds", "frozen_seconds")
    if freeze_seconds is None:
        freeze_seconds = validation.get("freeze_seconds", validation.get("frozen_seconds"))
    output_duration_s = _metric_value(metrics, "output_duration_s", "video_duration_s")
    return {
        "structural_passed": _optional_bool(structural, "metrics.structural_passed"),
        "first_frame_match_score": _optional_float(
            first_frame,
            "metrics.first_frame_match_score",
            minimum=0.0,
            maximum=1.0,
        ),
        "tail_frame_valid": _optional_bool(tail, "metrics.tail_frame_valid"),
        "frozen": _optional_bool(frozen, "metrics.frozen"),
        "freeze_seconds": _optional_float(freeze_seconds, "metrics.freeze_seconds", minimum=0.0),
        "output_duration_s": _optional_float(output_duration_s, "metrics.output_duration_s", minimum=0.0),
    }


def _normalize_shot(item: Any, index: int) -> dict[str, Any]:
    shot = _required_mapping(item, f"shots[{index}]")
    shot_id = _string(shot.get("shot_id") or shot.get("id"), f"shots[{index}].shot_id")
    return {
        "shot_id": shot_id,
        "shot_type": _normalize_shot_type(shot.get("shot_type"), f"shots[{index}].shot_type"),
        "execution_plan_hash": _execution_plan_hash(shot),
    }


def _normalize_result(
    item: Any,
    index: int,
    *,
    default_batch_id: str,
    shots_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    record = _required_mapping(item, f"results[{index}]")
    shot_id = _string(record.get("shot_id"), f"results[{index}].shot_id")
    if shot_id not in shots_by_id:
        raise VideoABEvaluationError(f"results[{index}] 引用了未声明镜头: {shot_id}")
    strategy_id = _normalize_strategy(
        record.get("strategy_id", record.get("strategy")),
        f"results[{index}].strategy",
    )
    batch_id = _string(record.get("batch_id", default_batch_id), f"results[{index}].batch_id")
    if batch_id != default_batch_id:
        raise VideoABEvaluationError(
            f"results[{index}] 的 batch_id={batch_id} 与评估批次 {default_batch_id} 不一致"
        )
    plan_hash = _string(
        record.get("execution_plan_hash", record.get("plan_hash")),
        f"results[{index}].execution_plan_hash",
    )
    expected_plan_hash = str(shots_by_id[shot_id]["execution_plan_hash"])
    if plan_hash != expected_plan_hash:
        raise VideoABEvaluationError(
            f"镜头 {shot_id} 的策略 {strategy_id} 使用了不同执行计划: "
            f"{plan_hash} != {expected_plan_hash}"
        )
    metrics = _normalize_metrics(record)
    elapsed_ms = _required_int(
        record.get("elapsed_ms", record.get("generation_duration_ms", record.get("duration_ms"))),
        f"results[{index}].elapsed_ms",
    )
    manifest = _normalize_reference_manifest(
        record.get("reference_manifest", record.get("references")),
        f"results[{index}].reference_manifest",
    )
    return {
        "shot_id": shot_id,
        "shot_type": str(shots_by_id[shot_id]["shot_type"]),
        "strategy_id": strategy_id,
        "strategy_label": STRATEGY_DEFINITIONS[strategy_id]["label"],
        "batch_id": batch_id,
        "execution_plan_hash": plan_hash,
        "provider": _string(record.get("provider"), f"results[{index}].provider", allow_empty=True),
        "model": _string(record.get("model"), f"results[{index}].model", allow_empty=True),
        "reference_manifest": manifest,
        "cost": _normalize_cost(record),
        "elapsed_ms": elapsed_ms,
        "output_path": _string(
            record.get("output_path", record.get("video_path")),
            f"results[{index}].output_path",
            allow_empty=True,
        ),
        "metrics": metrics,
        "manual_review": _normalize_manual_review(record, metrics),
    }


def _metric_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sample_count = len(records)
    structural_values = [item["metrics"]["structural_passed"] for item in records]
    structural_known = [value for value in structural_values if value is not None]
    first_frame_values = [
        item["metrics"]["first_frame_match_score"]
        for item in records
        if item["metrics"]["first_frame_match_score"] is not None
    ]
    tail_values = [item["metrics"]["tail_frame_valid"] for item in records]
    tail_known = [value for value in tail_values if value is not None]
    frozen_values = [item["metrics"]["frozen"] for item in records]
    frozen_known = [value for value in frozen_values if value is not None]
    freeze_known = [
        item
        for item in records
        if item["metrics"]["freeze_seconds"] is not None and item["metrics"]["output_duration_s"] is not None
    ]
    freeze_seconds = sum(float(item["metrics"]["freeze_seconds"] or 0) for item in freeze_known)
    output_seconds = sum(float(item["metrics"]["output_duration_s"] or 0) for item in freeze_known)
    manual_counts = {
        "accepted": sum(item["manual_review"]["status"] == "accepted" for item in records),
        "rejected": sum(item["manual_review"]["status"] == "rejected" for item in records),
        "pending": sum(item["manual_review"]["status"] == "pending" for item in records),
        "unknown": sum(item["manual_review"]["status"] == "unknown" for item in records),
    }
    reviewed = manual_counts["accepted"] + manual_counts["rejected"]
    rates = {
        "structural_pass_rate": _round_rate(_rate(sum(value is True for value in structural_known), len(structural_known))),
        "first_frame_match_score": _round_rate(
            _rate(sum(first_frame_values), len(first_frame_values)) if first_frame_values else None
        ),
        "tail_frame_valid_rate": _round_rate(_rate(sum(value is True for value in tail_known), len(tail_known))),
        "freeze_rate": _round_rate(_rate(sum(value is True for value in frozen_known), len(frozen_known))),
        "manual_acceptance_rate": _round_rate(_rate(manual_counts["accepted"], reviewed)),
    }
    return {
        "sample_count": sample_count,
        "rates": rates,
        "structural_pass": {
            "passed_count": sum(value is True for value in structural_known),
            "failed_count": sum(value is False for value in structural_known),
            "unknown_count": sample_count - len(structural_known),
            "measured_count": len(structural_known),
            "rate": _round_rate(_rate(sum(value is True for value in structural_known), len(structural_known))),
            "coverage_rate": _round_rate(_rate(len(structural_known), sample_count)),
        },
        "first_frame_match": {
            "measured_count": len(first_frame_values),
            "unknown_count": sample_count - len(first_frame_values),
            "mean_score": _round_rate(
                _rate(sum(first_frame_values), len(first_frame_values)) if first_frame_values else None
            ),
            "min_score": _round_rate(min(first_frame_values)) if first_frame_values else None,
            "max_score": _round_rate(max(first_frame_values)) if first_frame_values else None,
            "coverage_rate": _round_rate(_rate(len(first_frame_values), sample_count)),
        },
        "tail_frame_valid": {
            "valid_count": sum(value is True for value in tail_known),
            "invalid_count": sum(value is False for value in tail_known),
            "unknown_count": sample_count - len(tail_known),
            "measured_count": len(tail_known),
            "rate": _round_rate(_rate(sum(value is True for value in tail_known), len(tail_known))),
            "coverage_rate": _round_rate(_rate(len(tail_known), sample_count)),
        },
        "freeze": {
            "frozen_count": sum(value is True for value in frozen_known),
            "not_frozen_count": sum(value is False for value in frozen_known),
            "unknown_count": sample_count - len(frozen_known),
            "measured_count": len(frozen_known),
            "frozen_sample_rate": _round_rate(_rate(sum(value is True for value in frozen_known), len(frozen_known))),
            "rate": _round_rate(_rate(sum(value is True for value in frozen_known), len(frozen_known))),
            "coverage_rate": _round_rate(_rate(len(frozen_known), sample_count)),
            "freeze_duration_ratio": _round_rate(_rate(freeze_seconds, output_seconds)) if freeze_known else None,
            "measured_output_seconds": round(output_seconds, 3),
            "freeze_seconds": round(freeze_seconds, 3),
        },
        "manual_acceptance": {
            **manual_counts,
            "reviewed_count": reviewed,
            "pending_count": manual_counts["pending"],
            "acceptance_rate": _round_rate(_rate(manual_counts["accepted"], reviewed)),
            "review_coverage_rate": _round_rate(_rate(reviewed, sample_count)),
            "pending_rate": _round_rate(_rate(manual_counts["pending"], sample_count)),
        },
    }


def _execution_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    costs: dict[str, dict[str, Any]] = {}
    known_records = [item for item in records if item["cost"]["cost_known"]]
    for item in known_records:
        currency = item["cost"]["currency"]
        bucket = costs.setdefault(currency, {"currency": currency, "cost_micro": 0, "known_count": 0})
        bucket["cost_micro"] += int(item["cost"]["cost_micro"] or 0)
        bucket["known_count"] += 1
    elapsed_values = [int(item["elapsed_ms"]) for item in records]
    return {
        "elapsed_ms_total": sum(elapsed_values),
        "elapsed_ms_mean": round(sum(elapsed_values) / len(elapsed_values), 3) if elapsed_values else None,
        "cost_known_count": len(known_records),
        "cost_unknown_count": len(records) - len(known_records),
        "cost_known": len(known_records) == len(records) and bool(records),
        "cost_by_currency": [costs[key] for key in sorted(costs)],
        "provider_model_pairs": [
            {"provider": provider, "model": model}
            for provider, model in sorted(
                {
                    (str(item["provider"]), str(item["model"]))
                    for item in records
                }
            )
        ],
    }


def _group_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "execution": _execution_summary(records),
        "metrics": _metric_summary(records),
    }


def load_evaluation_payload(path: str | Path) -> dict[str, Any]:
    """读取 A/B 评估输入 JSON；只读，不触发任何生成或配置写入。"""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VideoABEvaluationError(f"评估输入文件不存在: {source}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoABEvaluationError(f"评估输入 JSON 无法读取: {exc}") from exc
    return dict(_required_mapping(payload, "评估输入"))


def build_evaluation_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """校验四策略同批同计划覆盖，并生成可重复序列化的评估报告。"""

    root = _required_mapping(payload, "评估输入")
    schema_version = _required_int(root.get("schema_version", INPUT_SCHEMA_VERSION), "schema_version", minimum=1)
    if schema_version != INPUT_SCHEMA_VERSION:
        raise VideoABEvaluationError(f"不支持的评估输入 schema_version: {schema_version}")
    evaluation_id = _string(root.get("evaluation_id"), "evaluation_id")
    batch_id = _string(root.get("batch_id"), "batch_id")
    shots_raw = _required_list(root.get("shots"), "shots")
    if not shots_raw:
        raise VideoABEvaluationError("shots 不能为空")
    shots = [_normalize_shot(item, index) for index, item in enumerate(shots_raw)]
    shot_ids = [item["shot_id"] for item in shots]
    if len(set(shot_ids)) != len(shot_ids):
        raise VideoABEvaluationError("shots 中存在重复 shot_id")
    shots_by_id = {item["shot_id"]: item for item in shots}

    results_raw = _required_list(root.get("results"), "results")
    results = [
        _normalize_result(item, index, default_batch_id=batch_id, shots_by_id=shots_by_id)
        for index, item in enumerate(results_raw)
    ]
    seen: set[tuple[str, str]] = set()
    for item in results:
        key = (item["shot_id"], item["strategy_id"])
        if key in seen:
            raise VideoABEvaluationError(f"存在重复结果: shot={key[0]} strategy={key[1]}")
        seen.add(key)
    expected = {(shot_id, strategy_id) for shot_id in shot_ids for strategy_id in STRATEGY_ORDER}
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        details = []
        if missing:
            details.append("缺少 " + ", ".join(f"{shot}/{strategy}" for shot, strategy in missing))
        if extra:
            details.append("多出 " + ", ".join(f"{shot}/{strategy}" for shot, strategy in extra))
        raise VideoABEvaluationError("四策略必须覆盖同一批镜头；" + "；".join(details))

    strategy_order = {strategy_id: index for index, strategy_id in enumerate(STRATEGY_ORDER)}
    shot_order = {shot_id: index for index, shot_id in enumerate(shot_ids)}
    results.sort(key=lambda item: (shot_order[item["shot_id"]], strategy_order[item["strategy_id"]]))

    by_strategy: dict[str, dict[str, Any]] = {}
    for strategy_id in STRATEGY_ORDER:
        strategy_records = [item for item in results if item["strategy_id"] == strategy_id]
        by_strategy[strategy_id] = {
            **STRATEGY_DEFINITIONS[strategy_id],
            "strategy_id": strategy_id,
            "overall": _group_summary(strategy_records),
            "by_shot_type": {
                shot_type: _group_summary([item for item in strategy_records if item["shot_type"] == shot_type])
                for shot_type in SHOT_TYPE_ORDER
            },
        }

    plan_fingerprint = _canonical_json_hash(
        [
            {"shot_id": item["shot_id"], "shot_type": item["shot_type"], "execution_plan_hash": item["execution_plan_hash"]}
            for item in sorted(shots, key=lambda item: item["shot_id"])
        ]
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "batch_id": batch_id,
        "execution_plan_fingerprint": plan_fingerprint,
        "shot_count": len(shots),
        "strategy_count": len(STRATEGY_ORDER),
        "comparison_scope": {
            "shots": sorted(shots, key=lambda item: (shot_order[item["shot_id"]], item["shot_id"])),
            "strategy_order": list(STRATEGY_ORDER),
            "shot_type_order": list(SHOT_TYPE_ORDER),
            "complete_matrix": True,
        },
        "strategies": by_strategy,
        "results": results,
        "metric_definitions": {
            "structural_pass_rate": "结构通过数 / 已检测结构样本数；未检测项单列，不计为通过",
            "first_frame_match_score": "首帧匹配度输入分数（0..1）的均值；只对已评分样本求均值",
            "tail_frame_valid_rate": "尾帧有效数 / 已检测尾帧样本数",
            "freeze_rate": "冻结样本率 = 检测到冻结的样本 / 已检测冻结样本；freeze_duration_ratio 为冻结秒数 / 可测输出秒数",
            "manual_acceptance_rate": "人工接受数 /（人工接受数 + 人工拒绝数）；待审与未知单列",
        },
        "policy_safety": {
            "observation_only": True,
            "auto_policy_mutation_enabled": False,
            "default_provider_changed": False,
            "quality_threshold_changed": False,
            "applied_policy_changes": [],
            "note": "仅生成离线观测报告；不根据少量样本自动修改默认 Provider 或质量阈值。",
        },
    }
    return report


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """把 JSON 报告渲染成稳定排序、便于人工审阅的 Markdown。"""

    def percent(value: Any) -> str:
        return "N/A" if value is None else f"{float(value) * 100:.1f}%"

    def score(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.3f}"

    def number(value: Any) -> str:
        return "N/A" if value is None else str(value)

    lines: list[str] = [
        "# 视频工作流 A/B 评估报告",
        "",
        f"- 评估 ID：`{report['evaluation_id']}`",
        f"- 镜头批次：`{report['batch_id']}`",
        f"- 执行计划指纹：`{report['execution_plan_fingerprint']}`",
        f"- 覆盖矩阵：{report['shot_count']} 个镜头 × {report['strategy_count']} 个策略",
        "",
        "> 仅作离线观测。报告生成器不调用视频 Provider，也不根据少量样本自动修改默认 Provider 或质量阈值。",
        "",
        "## 指标口径",
        "",
    ]
    for key, value in report["metric_definitions"].items():
        lines.append(f"- `{key}`：{value}")
    lines.extend(
        [
            "",
            "## 总体对比",
            "",
            "| 策略 | 模式 | 样本 | 结构通过率 | 首帧匹配度 | 尾帧有效率 | 冻结率 | 冻结时长占比 | 人工接受率 | 已知总成本 | 平均耗时 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy_id in STRATEGY_ORDER:
        item = report["strategies"][strategy_id]
        metrics = item["overall"]["metrics"]
        execution = item["overall"]["execution"]
        cost_text = "; ".join(
            f"{entry['cost_micro']} micro {entry['currency']}"
            for entry in execution["cost_by_currency"]
        ) or ("成本未知" if execution["cost_unknown_count"] else "0")
        elapsed = execution["elapsed_ms_mean"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{strategy_id}. {item['label']}",
                    item["mode"],
                    number(metrics["sample_count"]),
                    percent(metrics["structural_pass"]["rate"]),
                    score(metrics["first_frame_match"]["mean_score"]),
                    percent(metrics["tail_frame_valid"]["rate"]),
                    percent(metrics["freeze"]["frozen_sample_rate"]),
                    percent(metrics["freeze"]["freeze_duration_ratio"]),
                    percent(metrics["manual_acceptance"]["acceptance_rate"]),
                    cost_text,
                    "N/A" if elapsed is None else f"{elapsed:.0f} ms",
                ]
            )
            + " |"
        )

    lines.extend(["", "## 按镜头类型分组", ""])
    for shot_type in SHOT_TYPE_ORDER:
        shot_meta = SHOT_TYPE_DEFINITIONS[shot_type]
        lines.extend([f"### {shot_meta['label']}（`{shot_type}`）", "", shot_meta["description"], ""])
        lines.extend(
            [
                "| 策略 | 样本 | 结构通过率 | 首帧匹配度 | 尾帧有效率 | 冻结率 | 人工接受率 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for strategy_id in STRATEGY_ORDER:
            group = report["strategies"][strategy_id]["by_shot_type"][shot_type]
            metrics = group["metrics"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{strategy_id}. {report['strategies'][strategy_id]['label']}",
                        number(metrics["sample_count"]),
                        percent(metrics["structural_pass"]["rate"]),
                        score(metrics["first_frame_match"]["mean_score"]),
                        percent(metrics["tail_frame_valid"]["rate"]),
                        percent(metrics["freeze"]["frozen_sample_rate"]),
                        percent(metrics["manual_acceptance"]["acceptance_rate"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## 执行审计",
            "",
            "以下表格逐条保留 Provider、模型、参考 manifest 数量、成本、耗时和输出路径；完整 manifest 与逐条指标见 JSON。",
            "",
            "| 镜头 | 类型 | 策略 | Provider | 模型 | 参考数 | 成本 | 耗时 | 输出路径 | 人工结果 |",
            "|---|---|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    shot_type_labels = {key: value["label"] for key, value in SHOT_TYPE_DEFINITIONS.items()}
    strategy_labels = {key: value["label"] for key, value in STRATEGY_DEFINITIONS.items()}
    for item in report["results"]:
        cost = item["cost"]
        cost_text = (
            f"{cost['cost_micro']} micro {cost['currency']}"
            if cost["cost_known"]
            else f"成本未知 ({cost['currency']})"
        )
        manual = item["manual_review"]["status"]
        output_path = str(item["output_path"] or "（空）").replace("|", "\\|")
        lines.append(
            "| "
            + " | ".join(
                [
                    item["shot_id"],
                    shot_type_labels[item["shot_type"]],
                    f"{item['strategy_id']}. {strategy_labels[item['strategy_id']]}",
                    item["provider"] or "（空）",
                    item["model"] or "（空）",
                    str(len(item["reference_manifest"])),
                    cost_text,
                    f"{item['elapsed_ms']} ms",
                    output_path,
                    manual,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 策略安全声明",
            "",
            "- `observation_only`: true",
            "- `auto_policy_mutation_enabled`: false",
            "- `default_provider_changed`: false",
            "- `quality_threshold_changed`: false",
            "- `applied_policy_changes`: []",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_reports(
    report: Mapping[str, Any],
    output_dir: str | Path,
    *,
    json_name: str = "video_ab_evaluation.json",
    markdown_name: str = "video_ab_evaluation.md",
) -> tuple[Path, Path]:
    """原子写出 JSON 与 Markdown；相同输入会得到逐字节相同的报告。"""

    target_dir = Path(output_dir)
    json_path = atomic_write_json(target_dir / json_name, report, indent=2, ensure_ascii=False)
    markdown_path = atomic_write_text(target_dir / markdown_name, render_markdown_report(report) + "\n")
    return json_path, markdown_path
