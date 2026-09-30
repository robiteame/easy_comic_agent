"""Agent 的策略选择器：从失败/反思结果到可执行恢复候选。

决策不只返回布尔值：每个候选都带目标阶段、Prompt 修改、Provider 能力、预计成本、
预计时长和剩余预算适配度，并将接受/拒绝原因写入 DecisionTrace。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .contracts import (
    DecisionTrace,
    FailureKind,
    FailureRecord,
    ProviderProfile,
    QualityProfileName,
    QualityStrategy,
    RecoveryCandidate,
    RecoveryStrategy,
    StageName,
    default_quality_profile,
    utc_now,
)


_FAILURE_PATTERNS: tuple[tuple[FailureKind, re.Pattern[str]], ...] = (
    (FailureKind.LLM_INVALID_OUTPUT, re.compile(r"json|schema|结构无法解析|非法输出|输出无法使用|invalid.*output", re.I)),
    (FailureKind.DIALOGUE_TOO_LONG, re.compile(r"dialogue|台词|对白|too long|过长", re.I)),
    (FailureKind.PROVIDER_REFERENCE_UNSUPPORTED, re.compile(r"reference.*(unsupported|not support)|不支持.*参考|参考图.*不支持|references_sent=0", re.I)),
    (FailureKind.BUDGET_EXCEEDED, re.compile(r"budget|预算|quota|额度", re.I)),
    (FailureKind.VERSION_CONFLICT, re.compile(r"version|版本|stale|过期|changed", re.I)),
    (FailureKind.VIDEO_FAILED, re.compile(r"video|视频|seedance|wanx", re.I)),
    (FailureKind.IMAGE_FAILED, re.compile(r"image|图片|故事板|storyboard|seedream|qwen", re.I)),
    (FailureKind.AUDIO_FAILED, re.compile(r"audio|tts|配音|voice", re.I)),
    (FailureKind.TIMEOUT, re.compile(r"timeout|timed out|超时", re.I)),
)


def classify_failure(
    *,
    stage: StageName | str,
    message: str = "",
    kind: FailureKind | str | None = None,
    shot_id: str = "",
    provider: str = "",
    details: dict[str, Any] | None = None,
) -> FailureRecord:
    selected: FailureKind | None = FailureKind(kind) if kind else None
    text = str(message or "")
    if selected is None:
        for candidate, pattern in _FAILURE_PATTERNS:
            if pattern.search(text):
                selected = candidate
                break
    if selected is None:
        selected = FailureKind.UNKNOWN
    retryable = selected not in {FailureKind.USER_CHANGED_INPUT, FailureKind.BUDGET_EXCEEDED}
    return FailureRecord(
        kind=selected,
        stage=stage,
        shot_id=shot_id,
        message=text[:500],
        provider=provider,
        retryable=retryable,
        details=dict(details or {}),
    )


def budget_snapshot(project_id: str = "") -> dict[str, Any]:
    """读取预算/剩余时长；任何异常都返回 unlimited，让生成不被预算模块卡死。"""

    try:
        from db import SessionLocal
        from services.budget_service import budget_state

        db = SessionLocal()
        try:
            state = budget_state(db, project_id=project_id)
        finally:
            db.close()
        hard_cost = state.get("hard_cost_micro")
        soft_cost = state.get("soft_cost_micro")
        committed_cost = int(state.get("committed_cost_micro") or 0)
        remaining_cost = None
        limit_cost = hard_cost if hard_cost is not None else soft_cost
        if limit_cost is not None:
            remaining_cost = max(0, int(limit_cost) - committed_cost)
        hard_seconds = state.get("hard_seconds")
        soft_seconds = state.get("soft_seconds")
        committed_seconds = int(state.get("committed_seconds") or 0)
        remaining_seconds = None
        limit_seconds = hard_seconds if hard_seconds is not None else soft_seconds
        if limit_seconds is not None:
            remaining_seconds = max(0, int(limit_seconds) - committed_seconds)
        return {
            **state,
            "remaining_cost_micro": remaining_cost,
            "remaining_seconds": remaining_seconds,
            "limit_cost_micro": limit_cost,
            "limit_seconds": limit_seconds,
        }
    except Exception:
        return {
            "level": "unlimited",
            "cost_known": False,
            "remaining_cost_micro": None,
            "remaining_seconds": None,
            "unlimited": True,
        }


def provider_profiles(capability: str, *, reference_required: bool = False) -> list[ProviderProfile]:
    """返回已注册 Provider 的能力/成本/时长画像，供切换和降级决策使用。"""

    capability = str(capability or "").lower()
    try:
        from services.providers.endpoint import get_endpoint, image_protocol_defaults, video_protocol_defaults
        from services.providers.registry import get_adapter, protocols_for
    except Exception:
        return []
    protocols = protocols_for(capability)
    current = None
    try:
        current = get_endpoint(capability)
    except Exception:
        current = None
    profiles: list[ProviderProfile] = []
    for protocol in protocols:
        endpoint = current if current and current.protocol == protocol else None
        if endpoint is None and capability == "image":
            try:
                endpoint = image_protocol_defaults(protocol)
            except Exception:
                endpoint = None
        if endpoint is None and capability == "video":
            try:
                endpoint = video_protocol_defaults(protocol)
            except Exception:
                endpoint = None
        if endpoint is None:
            continue
        adapter = get_adapter(capability, protocol)
        caps = getattr(adapter, "capabilities", None)
        api_key = str(getattr(endpoint, "api_key", "") or "")
        placeholder = protocol in {"placeholder", "local", "native-audio"}
        available = bool(placeholder or api_key)
        supports_ref = bool(getattr(caps, "reference_images", False) or getattr(caps, "reference_image", False))
        native_audio = bool(getattr(caps, "native_audio", False))
        reason = "" if available else "未配置 API Key"
        if reference_required and not supports_ref:
            reason = (reason + "；" if reason else "") + "不支持参考图"
        profiles.append(
            ProviderProfile(
                capability=capability,
                provider=protocol,
                model=str(getattr(endpoint, "model", "") or ""),
                available=available,
                supports_reference_images=bool(getattr(caps, "reference_images", False)),
                supports_reference_image=bool(getattr(caps, "reference_image", False)),
                native_audio=native_audio,
                max_resolution=_max_resolution_for(protocol),
                estimated_cost_micro=_base_cost_micro(capability, protocol),
                estimated_seconds=_base_seconds(capability, protocol),
                reliability=_reliability(protocol, available),
                reason=reason,
            )
        )
    profiles.sort(key=lambda item: (not item.available, item.provider != getattr(current, "protocol", ""), -item.reliability))
    return profiles


def _max_resolution_for(protocol: str) -> str:
    return {"placeholder": "540p", "local": "540p", "stability": "1080p", "qwen-image": "1080p", "ark-seedream": "4k"}.get(protocol, "1080p")


def _base_cost_micro(capability: str, protocol: str) -> int | None:
    if protocol in {"placeholder", "local"}:
        return 0
    values = {
        "script": {"openai-chat": 2_000},
        "image": {"stability": 20_000, "qwen-image": 12_000, "ark-seedream": 18_000},
        "video": {"ark-seedance": 120_000, "dashscope-wanx": 90_000, "native-audio": 70_000},
        "voice": {"mimo-tts": 2_000, "dashscope-tts": 1_500, "tencent-tts": 1_200},
    }
    return values.get(capability, {}).get(protocol)


def _base_seconds(capability: str, protocol: str) -> int:
    return {"script": 3, "image": 8, "video": 35, "voice": 3}.get(capability, 10) if protocol not in {"placeholder", "local"} else 1


def _reliability(protocol: str, available: bool) -> float:
    if not available:
        return 0.2
    return {"placeholder": 0.98, "local": 0.98, "ark-seedream": 0.9, "qwen-image": 0.86, "stability": 0.82, "ark-seedance": 0.88, "dashscope-wanx": 0.84}.get(protocol, 0.75)


_STRATEGY_PRIORITY: dict[RecoveryStrategy, float] = {
    RecoveryStrategy.RESUME_CHECKPOINT: 1.25,
    RecoveryStrategy.REGENERATE_FAILED_SHOTS: 1.15,
    RecoveryStrategy.REVISE_PROMPT: 1.10,
    RecoveryStrategy.SWITCH_PROVIDER: 1.05,
    RecoveryStrategy.REPLACE_REFERENCE: 1.0,
    RecoveryStrategy.LOWER_RESOLUTION: 0.95,
    RecoveryStrategy.SPLIT_SHOT: 0.92,
    RecoveryStrategy.MERGE_SHOTS: 0.82,
    RecoveryStrategy.RETRY: 0.8,
    RecoveryStrategy.HUMAN_REVIEW: 0.45,
}

_STRATEGY_GAIN: dict[RecoveryStrategy, float] = {
    RecoveryStrategy.RESUME_CHECKPOINT: 0.04,
    RecoveryStrategy.REGENERATE_FAILED_SHOTS: 0.25,
    RecoveryStrategy.REVISE_PROMPT: 0.32,
    RecoveryStrategy.SWITCH_PROVIDER: 0.30,
    RecoveryStrategy.REPLACE_REFERENCE: 0.27,
    RecoveryStrategy.LOWER_RESOLUTION: 0.12,
    RecoveryStrategy.SPLIT_SHOT: 0.22,
    RecoveryStrategy.MERGE_SHOTS: 0.18,
    RecoveryStrategy.RETRY: 0.08,
    RecoveryStrategy.HUMAN_REVIEW: 0.5,
}


def recovery_candidates(
    failure: FailureRecord | None,
    *,
    quality: QualityStrategy | QualityProfileName | str | None = None,
    project_id: str = "",
    provider_profiles_by_capability: dict[str, list[ProviderProfile]] | None = None,
    budget: dict[str, Any] | None = None,
) -> list[RecoveryCandidate]:
    strategy = _strategy_for_failure(failure.kind if failure else FailureKind.UNKNOWN)
    target_stage = _target_stage_for(failure.stage if failure else StageName.QUALITY_REVIEW, strategy)
    profiles = (provider_profiles_by_capability or {}).get(_capability_for_stage(failure.stage if failure else StageName.QUALITY_REVIEW), [])
    selected_provider = ""
    if profiles:
        selected_provider = next((item.provider for item in profiles if item.available), profiles[0].provider)
    remaining_cost = (budget or {}).get("remaining_cost_micro")
    remaining_seconds = (budget or {}).get("remaining_seconds")
    candidates: list[RecoveryCandidate] = []
    for item in strategy:
        provider = selected_provider if item is RecoveryStrategy.SWITCH_PROVIDER else ""
        profile = next((entry for entry in profiles if entry.provider == provider), None)
        capability_ok = True
        if item is RecoveryStrategy.SWITCH_PROVIDER:
            capability_ok = bool(profile and profile.available)
            if failure and failure.kind is FailureKind.PROVIDER_REFERENCE_UNSUPPORTED:
                capability_ok = bool(profile and (profile.supports_reference_images or profile.supports_reference_image))
        if item is RecoveryStrategy.REPLACE_REFERENCE and failure and failure.kind is FailureKind.PROVIDER_REFERENCE_UNSUPPORTED:
            capability_ok = True
        cost = _candidate_cost(item, profile, quality)
        seconds = _candidate_seconds(item, profile, quality)
        budget_fit = True
        if remaining_cost is not None and cost is not None:
            budget_fit = cost <= int(remaining_cost)
        if remaining_seconds is not None and seconds is not None:
            budget_fit = budget_fit and seconds <= int(remaining_seconds)
        candidate = RecoveryCandidate(
            strategy=item,
            provider=provider,
            target_stage=target_stage,
            shot_ids=[failure.shot_id] if failure and failure.shot_id else [],
            prompt_changes=_prompt_changes(item, failure),
            estimated_cost_micro=cost,
            estimated_seconds=seconds,
            quality_gain=_STRATEGY_GAIN[item],
            provider_capability_ok=capability_ok,
            budget_fit=budget_fit,
            rationale=_rationale(item, failure, profile),
        )
        candidate.score = _score_candidate(candidate, quality)
        candidates.append(candidate)
    candidates.sort(key=lambda item: item.score, reverse=True)
    preferred_order = {item: index for index, item in enumerate(strategy)}
    if preferred_order:
        candidates.sort(key=lambda item: (preferred_order.get(item.strategy, 99), -item.score))
    return candidates


def choose_recovery(
    failure: FailureRecord | None,
    *,
    stage: StageName | str,
    run_id: str = "",
    trace_id: str = "",
    quality: QualityStrategy | QualityProfileName | str | None = None,
    project_id: str = "",
    critique: Any = None,
    input_fingerprint: str = "",
    candidates: Iterable[RecoveryCandidate] | None = None,
) -> DecisionTrace:
    strategy = default_quality_profile(quality if isinstance(quality, (QualityProfileName, str)) else None)
    budget = budget_snapshot(project_id)
    profiles = {capability: provider_profiles(capability, reference_required=True) for capability in ("script", "image", "video", "voice")}
    options = list(candidates or recovery_candidates(failure, quality=strategy, project_id=project_id, provider_profiles_by_capability=profiles, budget=budget))
    selected = next((item for item in options if item.provider_capability_ok and item.budget_fit), None)
    if selected is None:
        selected = next((item for item in options if item.strategy is RecoveryStrategy.HUMAN_REVIEW), None)
    rejected = [
        {"strategy": item.strategy.value, "score": item.score, "reason": "能力不匹配" if not item.provider_capability_ok else "超出剩余预算"}
        for item in options
        if selected is not None and item is not selected and (not item.provider_capability_ok or not item.budget_fit)
    ]
    if failure is not None and not isinstance(failure, FailureRecord):
        if hasattr(failure, "model_dump"):
            failure_data = failure.model_dump(mode="json")
        else:
            failure_data = {
                "kind": getattr(failure, "kind", FailureKind.UNKNOWN),
                "stage": getattr(failure, "stage", stage),
                "shot_id": getattr(failure, "shot_id", ""),
                "message": getattr(failure, "message", ""),
                "provider": getattr(failure, "provider", ""),
                "retryable": getattr(failure, "retryable", True),
                "details": getattr(failure, "details", {}),
            }
        failure = FailureRecord.model_validate(failure_data)
    return DecisionTrace(
        trace_id=trace_id or f"trace:{run_id or 'run'}:{stage}:{utc_now()}",
        run_id=run_id,
        stage=stage,
        input_fingerprint=input_fingerprint,
        failure=failure,
        critique=critique,
        candidates=options,
        selected=selected,
        considered_rejected=rejected,
        budget_snapshot=budget,
        provider_profiles=[item for values in profiles.values() for item in values],
        reason=_decision_reason(selected, failure, strategy),
        created_at=utc_now(),
    )


def _strategy_for_failure(kind: FailureKind) -> list[RecoveryStrategy]:
    if kind is FailureKind.LLM_INVALID_OUTPUT:
        return [RecoveryStrategy.REVISE_PROMPT, RecoveryStrategy.SWITCH_PROVIDER, RecoveryStrategy.HUMAN_REVIEW]
    if kind is FailureKind.DIALOGUE_TOO_LONG:
        return [RecoveryStrategy.SPLIT_SHOT, RecoveryStrategy.REVISE_PROMPT, RecoveryStrategy.HUMAN_REVIEW]
    if kind is FailureKind.SHOT_TOO_COMPLEX:
        return [RecoveryStrategy.SPLIT_SHOT, RecoveryStrategy.MERGE_SHOTS, RecoveryStrategy.REVISE_PROMPT, RecoveryStrategy.HUMAN_REVIEW]
    if kind is FailureKind.PROVIDER_REFERENCE_UNSUPPORTED:
        return [RecoveryStrategy.SWITCH_PROVIDER, RecoveryStrategy.REPLACE_REFERENCE, RecoveryStrategy.REVISE_PROMPT, RecoveryStrategy.HUMAN_REVIEW]
    if kind in {FailureKind.IMAGE_FAILED, FailureKind.VIDEO_FAILED, FailureKind.AUDIO_FAILED}:
        return [RecoveryStrategy.REVISE_PROMPT, RecoveryStrategy.SWITCH_PROVIDER, RecoveryStrategy.REPLACE_REFERENCE, RecoveryStrategy.LOWER_RESOLUTION, RecoveryStrategy.REGENERATE_FAILED_SHOTS, RecoveryStrategy.HUMAN_REVIEW]
    if kind in {FailureKind.USER_CHANGED_INPUT, FailureKind.VERSION_CONFLICT}:
        return [RecoveryStrategy.RESUME_CHECKPOINT, RecoveryStrategy.REGENERATE_FAILED_SHOTS, RecoveryStrategy.HUMAN_REVIEW]
    if kind is FailureKind.BUDGET_EXCEEDED:
        return [RecoveryStrategy.LOWER_RESOLUTION, RecoveryStrategy.HUMAN_REVIEW]
    if kind is FailureKind.QUALITY_BELOW_THRESHOLD:
        return [RecoveryStrategy.REVISE_PROMPT, RecoveryStrategy.REPLACE_REFERENCE, RecoveryStrategy.SPLIT_SHOT, RecoveryStrategy.REGENERATE_FAILED_SHOTS, RecoveryStrategy.HUMAN_REVIEW]
    return [RecoveryStrategy.RETRY, RecoveryStrategy.SWITCH_PROVIDER, RecoveryStrategy.HUMAN_REVIEW]


def _target_stage_for(stage: StageName | str, strategy: RecoveryStrategy) -> StageName | str:
    if strategy is RecoveryStrategy.RESUME_CHECKPOINT:
        return StageName.QUALITY_REVIEW
    if strategy is RecoveryStrategy.SPLIT_SHOT:
        return StageName.STORYBOARD_DESIGN
    if strategy is RecoveryStrategy.MERGE_SHOTS:
        return StageName.STORYBOARD_DESIGN
    return stage


def _capability_for_stage(stage: StageName | str) -> str:
    mapping = {
        StageName.DIRECTOR_PLANNING: "script",
        StageName.STORYBOARD_DESIGN: "script",
        StageName.ASSET_PREPARATION: "image",
        StageName.IMAGE_GENERATION: "image",
        StageName.QUALITY_REVIEW: "image",
        StageName.VIDEO_GENERATION: "video",
        StageName.AUDIO_PRODUCTION: "voice",
        StageName.EDIT_COMPOSITION: "video",
        StageName.FINAL_REVIEW: "video",
    }
    return mapping.get(StageName(stage), "image")


def _candidate_cost(strategy: RecoveryStrategy, profile: ProviderProfile | None, quality: QualityStrategy | QualityProfileName | str | None) -> int | None:
    if strategy is RecoveryStrategy.HUMAN_REVIEW:
        return 0
    base = profile.estimated_cost_micro if profile else 20_000
    multiplier = default_quality_profile(quality if isinstance(quality, (QualityProfileName, str)) else None).cost_multiplier
    factor = {
        RecoveryStrategy.REGENERATE_FAILED_SHOTS: 1.0,
        RecoveryStrategy.REVISE_PROMPT: 0.65,
        RecoveryStrategy.SWITCH_PROVIDER: 1.0,
        RecoveryStrategy.REPLACE_REFERENCE: 0.7,
        RecoveryStrategy.LOWER_RESOLUTION: 0.55,
        RecoveryStrategy.SPLIT_SHOT: 1.1,
        RecoveryStrategy.MERGE_SHOTS: 0.6,
        RecoveryStrategy.RETRY: 0.9,
        RecoveryStrategy.RESUME_CHECKPOINT: 0.0,
    }.get(strategy, 1.0)
    return int((base or 0) * multiplier * factor)


def _candidate_seconds(strategy: RecoveryStrategy, profile: ProviderProfile | None, quality: QualityStrategy | QualityProfileName | str | None) -> int | None:
    if strategy in {RecoveryStrategy.HUMAN_REVIEW, RecoveryStrategy.RESUME_CHECKPOINT}:
        return 0
    base = profile.estimated_seconds if profile else 10
    multiplier = default_quality_profile(quality if isinstance(quality, (QualityProfileName, str)) else None).cost_multiplier
    factor = {RecoveryStrategy.LOWER_RESOLUTION: 0.7, RecoveryStrategy.REVISE_PROMPT: 0.75, RecoveryStrategy.SPLIT_SHOT: 1.2}.get(strategy, 1.0)
    return int((base or 0) * multiplier * factor)


def _prompt_changes(strategy: RecoveryStrategy, failure: FailureRecord | None) -> dict[str, Any]:
    shot_id = failure.shot_id if failure else ""
    if strategy is RecoveryStrategy.REVISE_PROMPT:
        return {"shot_id": shot_id, "instruction": "把主体、动作、镜头运动、光线和负面提示拆成可验证短句，移除互相冲突的描述。"}
    if strategy is RecoveryStrategy.SPLIT_SHOT:
        return {"shot_id": shot_id, "instruction": "按动作节拍拆成 2-5 秒短镜头，每个镜头只保留一个主体和一个动作。"}
    if strategy is RecoveryStrategy.MERGE_SHOTS:
        return {"shot_id": shot_id, "instruction": "合并连续短镜头为一个稳定镜头，保留原有叙事节拍并统一镜头运动。"}
    if strategy is RecoveryStrategy.REPLACE_REFERENCE:
        return {"shot_id": shot_id, "instruction": "替换为场景基准图/上一镜尾帧/无冲突角色参考，并记录参考权重。"}
    if strategy is RecoveryStrategy.LOWER_RESOLUTION:
        return {"shot_id": shot_id, "instruction": "将输出分辨率降到质量档位允许的最低值，先恢复可生成性再补拍。"}
    return {}


def _rationale(strategy: RecoveryStrategy, failure: FailureRecord | None, profile: ProviderProfile | None) -> str:
    kind = failure.kind.value if failure else "unknown"
    base = {
        RecoveryStrategy.REVISE_PROMPT: "结果或结构不合格，先用可验证的 Prompt 修改降低不确定性。",
        RecoveryStrategy.SWITCH_PROVIDER: "当前 Provider 能力/稳定性不匹配，切换到声明能力更合适的 Provider。",
        RecoveryStrategy.SPLIT_SHOT: "镜头包含过多动作或对白，拆分后可降低生成失败面。",
        RecoveryStrategy.MERGE_SHOTS: "短镜头连续且动作一致，合并可减少转场和连续性风险。",
        RecoveryStrategy.REPLACE_REFERENCE: "参考图可能冲突或未被当前 Provider 接受，改用可追溯参考。",
        RecoveryStrategy.LOWER_RESOLUTION: "优先保证生成闭环，降低分辨率后只重算失败镜头。",
        RecoveryStrategy.REGENERATE_FAILED_SHOTS: "保留已成功镜头，只对失败镜头补拍，避免全量失效。",
        RecoveryStrategy.RESUME_CHECKPOINT: "检测到版本变化或进程续跑，复用未变检查点并局部重算。",
        RecoveryStrategy.HUMAN_REVIEW: "连续自动修复仍无法满足质量或需要创意取舍，明确转人工。",
        RecoveryStrategy.RETRY: "错误可能是瞬时的，保留低成本重试候选。",
    }.get(strategy, "依据结果选择恢复动作。")
    if profile and not profile.available:
        base += f" 当前候选 Provider 不可用：{profile.reason or '未配置凭据'}。"
    return f"{base} 失败类别={kind}。"


def _score_candidate(candidate: RecoveryCandidate, quality: QualityStrategy | QualityProfileName | str | None) -> float:
    strategy = default_quality_profile(quality if isinstance(quality, (QualityProfileName, str)) else None)
    base = _STRATEGY_PRIORITY[candidate.strategy] / 1.4
    gain = candidate.quality_gain * 0.35
    cost_penalty = 0.0
    if candidate.estimated_cost_micro:
        cost_penalty = min(0.18, candidate.estimated_cost_micro / 500_000 * strategy.cost_multiplier)
    seconds_penalty = min(0.12, (candidate.estimated_seconds or 0) / 500 * strategy.cost_multiplier)
    capability_bonus = 0.12 if candidate.provider_capability_ok else -0.35
    budget_bonus = 0.10 if candidate.budget_fit else -0.45
    return round(max(0.0, min(1.0, base + gain - cost_penalty - seconds_penalty + capability_bonus + budget_bonus)), 4)


def _decision_reason(selected: RecoveryCandidate | None, failure: FailureRecord | None, quality: QualityStrategy) -> str:
    if selected is None:
        return "没有满足 Provider 能力和预算约束的候选，进入人工审核。"
    return (
        f"质量档位={quality.name.value}，优先策略={selected.strategy.value}，"
        f"预计成本={selected.estimated_cost_micro} micro，预计时长={selected.estimated_seconds}s，"
        f"失败类别={(failure.kind.value if failure else 'unknown')}。"
    )


__all__ = [
    "budget_snapshot",
    "choose_recovery",
    "classify_failure",
    "provider_profiles",
    "recovery_candidates",
]
