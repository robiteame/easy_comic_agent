"""Agent 的策略选择器：从失败/反思结果到可执行恢复候选。

决策不只返回布尔值：每个候选都带目标阶段、结构化 Prompt 补丁、Provider 能力、
预计成本、预计时长、剩余预算适配度和剩余重试次数，选择按「失败类别首选策略 +
综合评分」进行，并把每个候选的接受/淘汰原因完整写入 DecisionTrace。

硬性规则：
- 不做无条件「重试一次」：RETRY 只在失败证据表明是瞬时错误（超时、存储写入
  抖动等）时才会成为候选。
- 自动模式永远不选择 human_review；恢复无法继续时选择 degraded_publish（有
  可用的部分结果）或 terminal_failure（没有任何可用结果）。
- 只有 manual 模式才允许选择 human_review。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .contracts import (
    MANUAL_ONLY_STRATEGIES,
    NON_RECOVERABLE_FAILURES,
    TERMINAL_STRATEGIES,
    DecisionTrace,
    FailureKind,
    FailureRecord,
    PromptPatch,
    ProviderProfile,
    QualityProfileName,
    QualityStrategy,
    RecoveryCandidate,
    RecoveryStrategy,
    StageName,
    default_quality_profile,
    stage_contract,
    utc_now,
)


# 截断必须排在 LLM_INVALID_OUTPUT 之前判定：截断消息里常带 JSON 解析失败细节，
# 若先命中 llm_invalid_output 就会退化成「改提示词重试」，对长度上限无效。
_TRUNCATION_PATTERN = re.compile(
    r"finish_reason\s*=\s*length|输出超过最大长度|输出疑似达到上限|输出被截断|被截断|output\s+truncated|llm_output_truncated",
    re.I,
)

_FAILURE_PATTERNS: tuple[tuple[FailureKind, re.Pattern[str]], ...] = (
    (FailureKind.LLM_OUTPUT_TRUNCATED, _TRUNCATION_PATTERN),
    (FailureKind.LLM_INVALID_OUTPUT, re.compile(r"json|schema|结构无法解析|非法输出|输出无法使用|invalid.*output", re.I)),
    (FailureKind.TIMEOUT, re.compile(r"timeout|timed out|超时|connection (reset|refused|error)|network (error|unreachable)|502|503|temporarily unavailable", re.I)),
    (FailureKind.BUDGET_EXCEEDED, re.compile(r"budget|预算|quota|额度", re.I)),
    (FailureKind.VERSION_CONFLICT, re.compile(r"version|版本|stale|过期|changed", re.I)),
    (FailureKind.DIALOGUE_TOO_LONG, re.compile(r"dialogue|台词|对白|too long|过长", re.I)),
    (FailureKind.SHOT_TOO_COMPLEX, re.compile(r"too complex|过于复杂|动作节拍过多|complex.*(shot|镜头)|镜头.*复杂", re.I)),
    (FailureKind.PROVIDER_REFERENCE_UNSUPPORTED, re.compile(r"reference.*(unsupported|not support)|不支持.*参考|参考图.*不支持|references_sent=0", re.I)),
    (FailureKind.PROVIDER_CAPABILITY_MISMATCH, re.compile(r"capability.*(mismatch|unsupported)|能力不匹配|不支持.*(分辨率|时长|能力)|降级|not.*supported.*(resolution|duration)", re.I)),
    (FailureKind.STORAGE_FAILED, re.compile(r"storage|存储|写入失败|save.*fail|尾帧缺失|disk", re.I)),
    (FailureKind.QUALITY_BELOW_THRESHOLD, re.compile(r"quality|质量|阈值|threshold|低于.*分|score.*below", re.I)),
    (FailureKind.CANCELLED, re.compile(r"cancel|取消", re.I)),
    (FailureKind.VIDEO_FAILED, re.compile(r"video|视频|seedance|wanx", re.I)),
    (FailureKind.IMAGE_FAILED, re.compile(r"image|图片|故事板|storyboard|seedream|qwen", re.I)),
    (FailureKind.AUDIO_FAILED, re.compile(r"audio|tts|配音|voice", re.I)),
)

# 固定自动恢复阶梯：恢复执行器必须按此顺序消费可行候选，不以评分把后续
# Provider 切换提前到当前 Provider 的重试/参数修复之前。降级和终止是出口，不是人工门。
AUTOMATIC_RECOVERY_ORDER: tuple[RecoveryStrategy, ...] = (
    RecoveryStrategy.RETRY,
    RecoveryStrategy.CHANGE_SEED,
    RecoveryStrategy.REVISE_PROMPT,
    RecoveryStrategy.REPLACE_REFERENCE,
    RecoveryStrategy.SPLIT_SHOT,
    RecoveryStrategy.SWITCH_PROVIDER,
    RecoveryStrategy.LOWER_RESOLUTION,
    RecoveryStrategy.DEGRADED_PUBLISH,
    RecoveryStrategy.TERMINAL_FAILURE,
)

# 这些失败类别在证据上属于瞬时错误，才允许 RETRY 候选；其余一律要先改输入。
TRANSIENT_FAILURES: frozenset[FailureKind] = frozenset({FailureKind.TIMEOUT, FailureKind.STORAGE_FAILED})


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
    retryable = selected not in NON_RECOVERABLE_FAILURES
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
        # 模型级能力判断：一律用 effective_capabilities(model)，适配器默认能力
        # 会掩盖同协议下不同模型的差异（如 wan i2v vs wan r2v）。
        effective = getattr(adapter, "effective_capabilities", None)
        model = str(getattr(endpoint, "model", "") or "")
        caps = effective(model) if callable(effective) else getattr(adapter, "capabilities", None)
        api_key = str(getattr(endpoint, "api_key", "") or "")
        placeholder = protocol in {"placeholder", "local", "native-audio"}
        available = bool(placeholder or api_key)
        multiple_refs = bool(getattr(caps, "reference_images", False) or getattr(caps, "multiple_reference_images", False))
        single_ref = bool(getattr(caps, "reference_image", False))
        supports_ref = multiple_refs or single_ref
        native_audio = bool(getattr(caps, "native_audio", False))
        reason = "" if available else "未配置 API Key"
        if reference_required and not supports_ref:
            reason = (reason + "；" if reason else "") + "不支持参考图"
        profiles.append(
            ProviderProfile(
                capability=capability,
                provider=protocol,
                model=model,
                available=available,
                supports_reference_images=multiple_refs,
                supports_reference_image=single_ref,
                native_audio=native_audio,
                max_resolution=_max_resolution_for(protocol),
                estimated_cost_micro=_base_cost_micro(capability, protocol),
                estimated_seconds=_base_seconds(capability, protocol),
                reliability=_reliability(protocol, available),
                reason=reason,
                is_current=bool(current and current.protocol == protocol and current.api_key),
            )
        )
    if capability == "script":
        # LLM 的「换 Provider」实际可选项是备端点（script_fallback）：协议同为
        # openai-chat，但模型/服务可能支持更大输出上限。只有在配置了密钥且与
        # 主端点不是同一服务时才作为可切换候选，避免切换回自己。
        try:
            from services.providers.endpoint import endpoint_identity, get_endpoint

            fallback_endpoint = get_endpoint("script_fallback")
            primary_endpoint = get_endpoint("script")
            if fallback_endpoint.api_key and endpoint_identity(fallback_endpoint.base_url) != endpoint_identity(primary_endpoint.base_url):
                profiles.append(
                    ProviderProfile(
                        capability=capability,
                        provider="script_fallback",
                        model=str(fallback_endpoint.model or ""),
                        available=True,
                        estimated_cost_micro=_base_cost_micro("script", "openai-chat"),
                        estimated_seconds=_base_seconds("script", "openai-chat"),
                        reliability=_reliability("openai-chat", True),
                        reason="",
                    )
                )
        except Exception:  # noqa: BLE001 - 备端点不可读时只是少一个切换候选
            pass
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
    # 数值只用于同一恢复阶梯内的可行候选排序；跨步骤的顺序由
    # AUTOMATIC_RECOVERY_ORDER/_ordered_strategies 保证。
    RecoveryStrategy.RETRY: 1.0,
    RecoveryStrategy.CHANGE_SEED: 0.99,
    RecoveryStrategy.REVISE_PROMPT: 0.98,
    RecoveryStrategy.REPLACE_REFERENCE: 0.97,
    RecoveryStrategy.SPLIT_SHOT: 0.96,
    RecoveryStrategy.SWITCH_PROVIDER: 0.95,
    RecoveryStrategy.LOWER_RESOLUTION: 0.94,
    RecoveryStrategy.RESUME_CHECKPOINT: 0.9,
    RecoveryStrategy.REGENERATE_FAILED_SHOTS: 0.89,
    RecoveryStrategy.MERGE_SHOTS: 0.88,
    RecoveryStrategy.HUMAN_REVIEW: 0.1,
    RecoveryStrategy.DEGRADED_PUBLISH: 0.05,
    RecoveryStrategy.TERMINAL_FAILURE: 0.0,
}

_STRATEGY_GAIN: dict[RecoveryStrategy, float] = {
    RecoveryStrategy.RESUME_CHECKPOINT: 0.04,
    RecoveryStrategy.REGENERATE_FAILED_SHOTS: 0.25,
    RecoveryStrategy.REVISE_PROMPT: 0.32,
    RecoveryStrategy.SWITCH_PROVIDER: 0.30,
    RecoveryStrategy.REPLACE_REFERENCE: 0.27,
    RecoveryStrategy.LOWER_RESOLUTION: 0.12,
    RecoveryStrategy.SPLIT_SHOT: 0.22,
    RecoveryStrategy.CHANGE_SEED: 0.24,
    RecoveryStrategy.MERGE_SHOTS: 0.18,
    RecoveryStrategy.RETRY: 0.08,
    RecoveryStrategy.HUMAN_REVIEW: 0.5,
    RecoveryStrategy.DEGRADED_PUBLISH: 0.1,
    RecoveryStrategy.TERMINAL_FAILURE: 0.0,
}

# 失败类别 -> 首选策略：排序时优先于纯评分（与 Critic 的 recommended_strategy 对齐）。
_PRIMARY_STRATEGY: dict[FailureKind, RecoveryStrategy] = {
    FailureKind.LLM_INVALID_OUTPUT: RecoveryStrategy.REVISE_PROMPT,
    # 输出截断改提示词救不了：唯一有意义的动作是换 Provider（更大的输出上限），
    # 换不了就明确终止，绝不进入「改提示词 → 同配置重试」循环。
    FailureKind.LLM_OUTPUT_TRUNCATED: RecoveryStrategy.SWITCH_PROVIDER,
    FailureKind.DIALOGUE_TOO_LONG: RecoveryStrategy.SPLIT_SHOT,
    FailureKind.SHOT_TOO_COMPLEX: RecoveryStrategy.SPLIT_SHOT,
    FailureKind.PROVIDER_REFERENCE_UNSUPPORTED: RecoveryStrategy.SWITCH_PROVIDER,
    FailureKind.PROVIDER_CAPABILITY_MISMATCH: RecoveryStrategy.SWITCH_PROVIDER,
    FailureKind.PROVIDER_UNAVAILABLE: RecoveryStrategy.SWITCH_PROVIDER,
    FailureKind.IMAGE_FAILED: RecoveryStrategy.REGENERATE_FAILED_SHOTS,
    FailureKind.VIDEO_FAILED: RecoveryStrategy.REGENERATE_FAILED_SHOTS,
    FailureKind.AUDIO_FAILED: RecoveryStrategy.REGENERATE_FAILED_SHOTS,
    FailureKind.STORAGE_FAILED: RecoveryStrategy.REGENERATE_FAILED_SHOTS,
    FailureKind.QUALITY_BELOW_THRESHOLD: RecoveryStrategy.REVISE_PROMPT,
    FailureKind.VERSION_CONFLICT: RecoveryStrategy.RESUME_CHECKPOINT,
    FailureKind.USER_CHANGED_INPUT: RecoveryStrategy.RESUME_CHECKPOINT,
    FailureKind.DEPENDENCY_FAILED: RecoveryStrategy.RESUME_CHECKPOINT,
    FailureKind.TIMEOUT: RecoveryStrategy.RETRY,
    FailureKind.BUDGET_EXCEEDED: RecoveryStrategy.DEGRADED_PUBLISH,
    FailureKind.CANCELLED: RecoveryStrategy.TERMINAL_FAILURE,
    FailureKind.UNKNOWN: RecoveryStrategy.SWITCH_PROVIDER,
}

# 终止回退候选：任何失败都保留这两个出口，恢复无法继续时按结果可用性选择。
_FALLBACK_STRATEGIES: tuple[RecoveryStrategy, ...] = (
    RecoveryStrategy.DEGRADED_PUBLISH,
    RecoveryStrategy.TERMINAL_FAILURE,
    RecoveryStrategy.HUMAN_REVIEW,
)


def _failure_kind(failure: FailureRecord | Any | None) -> FailureKind:
    kind = getattr(failure, "kind", None) if failure is not None else None
    return FailureKind(kind) if kind else FailureKind.UNKNOWN


def _stage_value(value: Any) -> str:
    """str(StageName.X) 在新版 Python 会得到 'StageName.X'，必须显式取 .value。"""

    if isinstance(value, StageName):
        return value.value
    return str(value or "")


def primary_strategy_for(kind: FailureKind | None) -> RecoveryStrategy | None:
    return _PRIMARY_STRATEGY.get(kind) if kind else None


def recovery_candidates(
    failure: FailureRecord | None,
    *,
    quality: QualityStrategy | QualityProfileName | str | None = None,
    project_id: str = "",
    provider_profiles_by_capability: dict[str, list[ProviderProfile]] | None = None,
    budget: dict[str, Any] | None = None,
    mode: str = "auto",
    retries_remaining: int | None = None,
    quality_score: float | None = None,
    candidate_results: Iterable[dict[str, Any]] | None = None,
    attempted_strategies: Iterable[RecoveryStrategy | str] | None = None,
) -> list[RecoveryCandidate]:
    """按失败类型、质量分、候选结果、Provider 能力、预算和剩余重试次数生成恢复候选。

    生成阶段不做最终取舍：即使某个候选已被预算/能力/次数判定为不可行，也保留在
    返回值中并带上判定字段，由 ``choose_recovery`` 记录淘汰原因。
    """

    stage = StageName(_stage_value(getattr(failure, "stage", None)) or StageName.QUALITY_REVIEW.value)
    allowed = set(stage_contract(stage).allowed_recovery)
    kind = _failure_kind(failure)
    strategies = _ordered_strategies(
        [item for item in _strategy_for_failure(kind) if item in allowed],
        mode=mode,
    )
    for item in _FALLBACK_STRATEGIES:
        if item in allowed and item not in strategies and (mode == "manual" or item not in MANUAL_ONLY_STRATEGIES):
            strategies.append(item)
    profiles = (provider_profiles_by_capability or {}).get(_capability_for_stage(stage), [])
    failing_provider = str(getattr(failure, "provider", "") or "")
    switch_target = _switch_provider_target(profiles, failing_provider=failing_provider, failure_kind=kind)
    remaining_cost = (budget or {}).get("remaining_cost_micro")
    remaining_seconds = (budget or {}).get("remaining_seconds")
    if retries_remaining is None:
        retries_remaining = default_quality_profile(quality if isinstance(quality, (QualityProfileName, str)) else None).max_recovery_attempts
    retries_remaining = max(0, int(retries_remaining))
    candidate_evidence = _summarize_candidate_results(candidate_results)
    attempted = _normalize_attempted_strategies(attempted_strategies)
    candidates: list[RecoveryCandidate] = []
    for item in strategies:
        if item in attempted:
            continue
        provider = ""
        profile: ProviderProfile | None = None
        capability_ok = True
        if item is RecoveryStrategy.SWITCH_PROVIDER:
            provider = switch_target
            profile = next((entry for entry in profiles if entry.provider == provider), None)
            capability_ok = bool(profile and profile.available and not profile.is_current)
            if capability_ok and kind is FailureKind.PROVIDER_REFERENCE_UNSUPPORTED:
                capability_ok = bool(profile.supports_reference_images or profile.supports_reference_image)
        if item is RecoveryStrategy.REPLACE_REFERENCE and kind is FailureKind.PROVIDER_REFERENCE_UNSUPPORTED:
            capability_ok = True
        cost = _candidate_cost(item, profile, quality)
        seconds = _candidate_seconds(item, profile, quality)
        budget_fit = True
        if remaining_cost is not None and cost is not None:
            budget_fit = cost <= int(remaining_cost)
        if remaining_seconds is not None and seconds is not None:
            budget_fit = budget_fit and seconds <= int(remaining_seconds)
        gain = _quality_gain(item, kind, candidate_evidence)
        patches = _prompt_patches(item, failure, provider=provider, target_stage=stage.value)
        candidate = RecoveryCandidate(
            strategy=item,
            provider=provider,
            target_stage=_target_stage_for(stage, item),
            shot_ids=[str(getattr(failure, "shot_id", "") or "")] if getattr(failure, "shot_id", "") else [],
            prompt_changes={
                "shot_id": str(getattr(failure, "shot_id", "") or ""),
                "instruction": _instruction(item),
                "patches": [patch.model_dump(mode="json") for patch in patches],
            },
            prompt_patches=patches,
            estimated_cost_micro=cost,
            estimated_seconds=seconds,
            quality_gain=gain,
            provider_capability_ok=capability_ok,
            budget_fit=budget_fit,
            retries_remaining=retries_remaining,
            rationale=_rationale(item, failure, profile, mode=mode, retries_remaining=retries_remaining),
        )
        candidate.score = _score_candidate(candidate, quality, quality_score=quality_score, primary=primary_strategy_for(kind))
        candidates.append(candidate)
    candidates.sort(key=lambda item: _rank_key(item, primary_strategy_for(kind)), reverse=True)
    return candidates


def choose_recovery(
    failure: FailureRecord | None,
    *,
    stage: StageName | str,
    run_id: str = "",
    trace_id: str = "",
    quality: QualityStrategy | QualityProfileName | str | None = None,
    project_id: str = "",
    shot_version: int = 0,
    critique: Any = None,
    input_fingerprint: str = "",
    candidates: Iterable[RecoveryCandidate] | None = None,
    mode: str = "auto",
    retries_remaining: int | None = None,
    quality_score: float | None = None,
    candidate_results: Iterable[dict[str, Any]] | None = None,
    budget: dict[str, Any] | None = None,
    provider_profiles_by_capability: dict[str, list[ProviderProfile]] | None = None,
    attempted_strategies: Iterable[RecoveryStrategy | str] | None = None,
) -> DecisionTrace:
    """选择恢复策略并留下完整 DecisionTrace（候选、淘汰原因、最终选择）。"""

    mode = "manual" if str(mode).lower() == "manual" else "auto"
    strategy = default_quality_profile(quality if isinstance(quality, (QualityProfileName, str)) else None)
    if failure is not None and not isinstance(failure, FailureRecord):
        if hasattr(failure, "model_dump"):
            failure = FailureRecord.model_validate(failure.model_dump(mode="json"))
        else:
            failure = FailureRecord(
                kind=_failure_kind(failure),
                stage=_stage_value(getattr(failure, "stage", stage)) or _stage_value(stage),
                shot_id=str(getattr(failure, "shot_id", "") or ""),
                message=str(getattr(failure, "message", "") or ""),
                provider=str(getattr(failure, "provider", "") or ""),
                retryable=bool(getattr(failure, "retryable", True)),
                details=dict(getattr(failure, "details", {}) or {}),
            )
    if retries_remaining is None:
        retries_remaining = strategy.max_recovery_attempts
    retries_remaining = max(0, int(retries_remaining))
    if quality_score is None and critique is not None:
        score = getattr(critique, "score", None) if not isinstance(critique, dict) else critique.get("score")
        if score is not None:
            quality_score = float(score)
    budget = budget if budget is not None else budget_snapshot(project_id)
    profiles = (
        provider_profiles_by_capability
        if provider_profiles_by_capability is not None
        else {capability: provider_profiles(capability, reference_required=True) for capability in ("script", "image", "video", "voice")}
    )
    options = list(
        candidates
        if candidates is not None
        else recovery_candidates(
            failure,
            quality=strategy,
            project_id=project_id,
            provider_profiles_by_capability=profiles,
            budget=budget,
            mode=mode,
            retries_remaining=retries_remaining,
            quality_score=quality_score,
            candidate_results=candidate_results,
            attempted_strategies=attempted_strategies,
        )
    )
    primary = primary_strategy_for(_failure_kind(failure))
    attempted = _normalize_attempted_strategies(attempted_strategies)
    eliminations = {id(item): _elimination_reason(item, mode=mode, retries_remaining=retries_remaining) for item in options}
    selected = _select_candidate(options, mode=mode, eliminations=eliminations, critique=critique, candidate_results=candidate_results, primary=primary)
    rejected = _rejection_records(options, selected, eliminations, primary=primary)
    return DecisionTrace(
        trace_id=trace_id or f"trace:{run_id or 'run'}:{stage}:{utc_now()}",
        project_id=project_id,
        shot_version=int(shot_version or 0),
        run_id=run_id,
        stage=stage,
        input_fingerprint=input_fingerprint,
        mode=mode,
        failure=failure,
        critique=critique,
        candidates=options,
        selected=selected,
        considered_rejected=rejected,
        attempted_strategies=[item.value for item in sorted(attempted, key=lambda item: _strategy_rank(item))],
        budget_snapshot=budget,
        provider_profiles=[item for values in profiles.values() for item in values],
        retries_remaining=retries_remaining,
        quality_score=quality_score,
        reason=_decision_reason(selected, failure, strategy, mode=mode, retries_remaining=retries_remaining),
        created_at=utc_now(),
    )


def _normalize_attempted_strategies(values: Iterable[RecoveryStrategy | str] | None) -> set[RecoveryStrategy]:
    result: set[RecoveryStrategy] = set()
    for value in values or ():
        try:
            result.add(value if isinstance(value, RecoveryStrategy) else RecoveryStrategy(str(value)))
        except ValueError:
            continue
    return result


def _strategy_rank(strategy: RecoveryStrategy) -> int:
    return {item: index for index, item in enumerate(AUTOMATIC_RECOVERY_ORDER)}.get(strategy, len(AUTOMATIC_RECOVERY_ORDER))


def _ordered_strategies(strategies: Iterable[RecoveryStrategy], *, mode: str) -> list[RecoveryStrategy]:
    """按自动恢复阶梯稳定排序，并去掉自动模式人工候选。"""

    unique = list(dict.fromkeys(strategies))
    # 保留 human_review 候选用于审计/兼容 API，但自动模式会在
    # _elimination_reason 中明确淘汰，绝不会选择或进入 human_gate。
    rank = {item: index for index, item in enumerate(AUTOMATIC_RECOVERY_ORDER)}
    return sorted(unique, key=lambda item: rank.get(item, len(rank)))


def _strategy_for_failure(kind: FailureKind) -> list[RecoveryStrategy]:
    if kind is FailureKind.LLM_OUTPUT_TRUNCATED:
        # 截断是确定性的：改提示词/换 seed/原参数重试都注定得到同样的截断，
        # 只允许换 Provider；没有可换的端点时直接走终止出口。
        return [RecoveryStrategy.SWITCH_PROVIDER, RecoveryStrategy.HUMAN_REVIEW]
    if kind is FailureKind.LLM_INVALID_OUTPUT:
        return [RecoveryStrategy.REVISE_PROMPT, RecoveryStrategy.SWITCH_PROVIDER, RecoveryStrategy.HUMAN_REVIEW]
    if kind is FailureKind.DIALOGUE_TOO_LONG:
        return [RecoveryStrategy.SPLIT_SHOT, RecoveryStrategy.REVISE_PROMPT, RecoveryStrategy.HUMAN_REVIEW]
    if kind is FailureKind.SHOT_TOO_COMPLEX:
        return [RecoveryStrategy.SPLIT_SHOT, RecoveryStrategy.MERGE_SHOTS, RecoveryStrategy.REVISE_PROMPT, RecoveryStrategy.HUMAN_REVIEW]
    if kind is FailureKind.PROVIDER_REFERENCE_UNSUPPORTED:
        return [RecoveryStrategy.SWITCH_PROVIDER, RecoveryStrategy.REPLACE_REFERENCE, RecoveryStrategy.REVISE_PROMPT, RecoveryStrategy.HUMAN_REVIEW]
    if kind is FailureKind.PROVIDER_CAPABILITY_MISMATCH:
        return [RecoveryStrategy.SWITCH_PROVIDER, RecoveryStrategy.LOWER_RESOLUTION, RecoveryStrategy.REVISE_PROMPT, RecoveryStrategy.HUMAN_REVIEW]
    if kind in {FailureKind.IMAGE_FAILED, FailureKind.VIDEO_FAILED, FailureKind.AUDIO_FAILED}:
        return [
            RecoveryStrategy.RETRY,
            RecoveryStrategy.CHANGE_SEED,
            RecoveryStrategy.REVISE_PROMPT,
            RecoveryStrategy.REPLACE_REFERENCE,
            RecoveryStrategy.SPLIT_SHOT,
            RecoveryStrategy.SWITCH_PROVIDER,
            RecoveryStrategy.LOWER_RESOLUTION,
            RecoveryStrategy.REGENERATE_FAILED_SHOTS,
            RecoveryStrategy.HUMAN_REVIEW,
        ]
    if kind in {FailureKind.USER_CHANGED_INPUT, FailureKind.VERSION_CONFLICT}:
        return [RecoveryStrategy.RESUME_CHECKPOINT, RecoveryStrategy.REGENERATE_FAILED_SHOTS, RecoveryStrategy.HUMAN_REVIEW]
    if kind is FailureKind.BUDGET_EXCEEDED:
        # 预算耗尽后不允许再花一次生成成本，只能降级发布或明确失败。
        return [RecoveryStrategy.DEGRADED_PUBLISH, RecoveryStrategy.TERMINAL_FAILURE]
    if kind is FailureKind.QUALITY_BELOW_THRESHOLD:
        return [RecoveryStrategy.RETRY, RecoveryStrategy.CHANGE_SEED, RecoveryStrategy.REVISE_PROMPT, RecoveryStrategy.REPLACE_REFERENCE, RecoveryStrategy.SPLIT_SHOT, RecoveryStrategy.SWITCH_PROVIDER, RecoveryStrategy.LOWER_RESOLUTION, RecoveryStrategy.REGENERATE_FAILED_SHOTS, RecoveryStrategy.HUMAN_REVIEW]
    if kind is FailureKind.TIMEOUT:
        # 超时是瞬时错误证据，RETRY 才被允许成为候选。
        return [RecoveryStrategy.RETRY, RecoveryStrategy.SWITCH_PROVIDER, RecoveryStrategy.REVISE_PROMPT, RecoveryStrategy.HUMAN_REVIEW]
    if kind is FailureKind.STORAGE_FAILED:
        return [RecoveryStrategy.REGENERATE_FAILED_SHOTS, RecoveryStrategy.RESUME_CHECKPOINT, RecoveryStrategy.RETRY, RecoveryStrategy.HUMAN_REVIEW]
    if kind is FailureKind.CANCELLED:
        return [RecoveryStrategy.TERMINAL_FAILURE]
    # UNKNOWN：没有瞬时错误证据时禁止无条件 RETRY，先换 Provider / 收紧 Prompt。
    return [RecoveryStrategy.SWITCH_PROVIDER, RecoveryStrategy.REVISE_PROMPT, RecoveryStrategy.REGENERATE_FAILED_SHOTS, RecoveryStrategy.HUMAN_REVIEW]


def _target_stage_for(stage: StageName | str, strategy: RecoveryStrategy) -> StageName | str:
    if strategy in {RecoveryStrategy.SPLIT_SHOT, RecoveryStrategy.MERGE_SHOTS}:
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
        StageName.VIDEO_REVIEW: "video",
        StageName.AUDIO_PRODUCTION: "voice",
        StageName.EDIT_COMPOSITION: "video",
        StageName.FINAL_REVIEW: "video",
    }
    return mapping.get(StageName(stage), "image")


def _switch_provider_target(profiles: list[ProviderProfile], *, failing_provider: str, failure_kind: FailureKind) -> str:
    """选择切换目标：优先「可用且与失败 Provider 不同」的最高可靠性 Provider。

    当前在用的端点（is_current）即使可用也不是切换目标——切到自己等于原参数
    重跑，对截断这类确定性失败只会重复烧 token。
    """

    if not profiles:
        return ""
    need_reference = failure_kind is FailureKind.PROVIDER_REFERENCE_UNSUPPORTED
    pool = [item for item in profiles if item.available and not item.is_current and item.provider != failing_provider]
    if need_reference:
        with_ref = [item for item in pool if item.supports_reference_images or item.supports_reference_image]
        if with_ref:
            pool = with_ref
    if not pool:
        pool = [item for item in profiles if item.available and not item.is_current]
    if not pool:
        return ""
    return max(pool, key=lambda item: (item.reliability, item.provider)).provider


def _summarize_candidate_results(candidate_results: Iterable[dict[str, Any]] | None) -> dict[str, Any]:
    """聚合候选证据；成功状态本身不足以证明候选可降级发布。

    ``degraded_publish`` 只能消费明确结构完整的候选。旧调用方可能只提供 status，
    因此保留这些行用于 Provider/收益统计，但将其标记为 ``structurally_usable=False``，
    避免把“调用返回 succeeded”误当成媒体文件可播放。
    """

    rows: list[dict[str, Any]] = []
    for item in candidate_results or ():
        if not isinstance(item, dict):
            continue
        structural = item.get("structural_validity")
        if not isinstance(structural, dict):
            structural = item.get("structural_report")
        if not isinstance(structural, dict):
            structural = {}
        path = str(item.get("path") or item.get("video_path") or item.get("artifact_path") or "")
        structural_passed = item.get("structural_passed")
        if structural_passed is None and "passed" in structural:
            structural_passed = structural.get("passed")
        technical_passed = item.get("technical_passed")
        technical = item.get("technical_quality")
        if technical_passed is None and isinstance(technical, dict) and "passed" in technical:
            technical_passed = technical.get("passed")
        status = str(item.get("status") or "")
        # Explicit false always rejects. Missing evidence is intentionally unknown,
        # not an implicit pass; path is also required because a status is not a file.
        usable = bool(
            status == "succeeded"
            and path
            and structural_passed is True
            and (technical_passed is not False)
        )
        rows.append({
            "status": status,
            "provider": str(item.get("provider") or ""),
            "path": path,
            "structural_passed": structural_passed,
            "technical_passed": technical_passed,
            "structurally_usable": usable,
        })
    if not rows:
        return {
            "count": 0, "success": 0, "failed": 0, "structurally_usable": 0,
            "all_failed_same_provider": False, "success_ratio": None,
        }
    success = sum(1 for row in rows if row["status"] == "succeeded")
    failed = sum(1 for row in rows if row["status"] == "failed")
    usable = sum(1 for row in rows if row["structurally_usable"])
    failed_providers = {row["provider"] for row in rows if row["status"] == "failed" and row["provider"]}
    return {
        "count": len(rows),
        "success": success,
        "failed": failed,
        "structurally_usable": usable,
        "all_failed_same_provider": bool(failed_providers) and len(failed_providers) == 1 and success == 0,
        "success_ratio": round(success / len(rows), 3),
    }


def _quality_gain(strategy: RecoveryStrategy, kind: FailureKind, evidence: dict[str, Any]) -> float:
    gain = _STRATEGY_GAIN[strategy]
    if evidence.get("all_failed_same_provider") and strategy is RecoveryStrategy.SWITCH_PROVIDER:
        gain += 0.1  # 历史候选证明当前 Provider 持续失败，切换的预期收益更高。
    if evidence.get("success") and strategy is RecoveryStrategy.REGENERATE_FAILED_SHOTS:
        gain += 0.1  # 已有成功候选，局部补拍的边际收益更高。
    return round(min(1.0, gain), 3)


def _candidate_cost(strategy: RecoveryStrategy, profile: ProviderProfile | None, quality: QualityStrategy | QualityProfileName | str | None) -> int | None:
    if strategy in {RecoveryStrategy.HUMAN_REVIEW, RecoveryStrategy.DEGRADED_PUBLISH, RecoveryStrategy.TERMINAL_FAILURE}:
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
        RecoveryStrategy.CHANGE_SEED: 0.9,
        RecoveryStrategy.RESUME_CHECKPOINT: 0.0,
    }.get(strategy, 1.0)
    return int((base or 0) * multiplier * factor)


def _candidate_seconds(strategy: RecoveryStrategy, profile: ProviderProfile | None, quality: QualityStrategy | QualityProfileName | str | None) -> int | None:
    if strategy in {RecoveryStrategy.HUMAN_REVIEW, RecoveryStrategy.RESUME_CHECKPOINT, RecoveryStrategy.DEGRADED_PUBLISH, RecoveryStrategy.TERMINAL_FAILURE}:
        return 0
    base = profile.estimated_seconds if profile else 10
    multiplier = default_quality_profile(quality if isinstance(quality, (QualityProfileName, str)) else None).cost_multiplier
    factor = {RecoveryStrategy.LOWER_RESOLUTION: 0.7, RecoveryStrategy.REVISE_PROMPT: 0.75, RecoveryStrategy.SPLIT_SHOT: 1.2}.get(strategy, 1.0)
    return int((base or 0) * multiplier * factor)


def _instruction(strategy: RecoveryStrategy) -> str:
    return {
        RecoveryStrategy.REVISE_PROMPT: "把主体、动作、镜头运动、光线和负面提示拆成可验证短句，移除互相冲突的描述。",
        RecoveryStrategy.SPLIT_SHOT: "按动作节拍拆成 2-5 秒短镜头，每个镜头只保留一个主体和一个动作。",
        RecoveryStrategy.MERGE_SHOTS: "合并连续短镜头为一个稳定镜头，保留原有叙事节拍并统一镜头运动。",
        RecoveryStrategy.REPLACE_REFERENCE: "替换为场景基准图/上一镜尾帧/无冲突角色参考，并记录参考权重。",
        RecoveryStrategy.LOWER_RESOLUTION: "将输出分辨率降到质量档位允许的最低值，先恢复可生成性再补拍。",
    }.get(strategy, "")


def _prompt_patches(strategy: RecoveryStrategy, failure: FailureRecord | None, *, provider: str = "", target_stage: str = "") -> list[PromptPatch]:
    """结构化 Prompt/参数补丁：字段级 op/value，执行节点可直接应用。"""

    shot_id = str(getattr(failure, "shot_id", "") or "")
    kind = _failure_kind(failure)
    stage_value = _stage_value(target_stage) or _stage_value(getattr(failure, "stage", ""))

    def patch(field: str, op: str, value: Any, reason: str) -> PromptPatch:
        return PromptPatch(field=field, op=op, value=value, shot_id=shot_id, target_stage=stage_value, reason=reason)

    if strategy is RecoveryStrategy.CHANGE_SEED:
        return [patch("seed", "set", {"mode": "derive_next_attempt", "salt": shot_id or "run"}, "当前候选结构/技术失败，改变随机种子避免重复产物")]
    if strategy is RecoveryStrategy.RETRY:
        return [patch("retry", "set", {"same_provider": True, "preserve_inputs": True}, "瞬时错误先在当前 Provider 原参数重试")]
    if strategy is RecoveryStrategy.REVISE_PROMPT:
        if kind is FailureKind.LLM_INVALID_OUTPUT:
            return [
                patch("output_format", "set", "strict_json", "限制输出为可解析 JSON 对象"),
                patch("temperature", "set", 0.2, "降低随机性，减少结构解析失败"),
                patch("negative_prefix", "append", "不要输出 Markdown、注释或多余字段", "抑制非 JSON 内容"),
            ]
        return [
            patch("visual_prompt", "replace", {"rule": "主体、动作、镜头运动、光线各一句，删除互相冲突的描述"}, "把自然语言描述收敛为可验证短句"),
            patch("negative_prompt", "append", "模糊, 多主体, 文字水印", "减少常见生成缺陷"),
        ]
    if strategy is RecoveryStrategy.SPLIT_SHOT:
        if kind is FailureKind.DIALOGUE_TOO_LONG:
            return [patch("dialogue", "split", {"max_chars": 90, "append_pause_shot": True}, "对白超出单镜头安全长度，拆成短句并增加停顿镜头")]
        return [patch("character_action", "split", {"max_beats": 1, "max_seconds": 5.0}, "动作节拍过多，按节拍拆成短镜头")]
    if strategy is RecoveryStrategy.MERGE_SHOTS:
        return [patch("shots", "replace", {"strategy": "merge_adjacent", "max_duration_s": 5.0}, "连续短镜头合并，减少转场和连续性风险")]
    if strategy is RecoveryStrategy.REPLACE_REFERENCE:
        return [patch("reference_images", "replace", {"source": "scene_baseline_or_previous_tail_frame", "record_weight": True}, "当前参考可能冲突或未被 Provider 接受，改用可追溯参考")]
    if strategy is RecoveryStrategy.LOWER_RESOLUTION:
        return [patch("resolution", "set", "540p", "先恢复可生成性，再择机补拍高分辨率")]
    if strategy is RecoveryStrategy.SWITCH_PROVIDER:
        return [patch("provider", "set", provider or "auto_select", "当前 Provider 能力/稳定性不匹配，切换到声明能力更合适的 Provider")]
    if strategy is RecoveryStrategy.REGENERATE_FAILED_SHOTS:
        return [patch("shot_scope", "set", "failed_only", "保留已成功镜头，只补拍失败镜头")]
    return []


def _rationale(strategy: RecoveryStrategy, failure: FailureRecord | None, profile: ProviderProfile | None, *, mode: str = "auto", retries_remaining: int = 0) -> str:
    kind = _failure_kind(failure).value
    base = {
        RecoveryStrategy.REVISE_PROMPT: "结果或结构不合格，先用可验证的 Prompt 修改降低不确定性。",
        RecoveryStrategy.SWITCH_PROVIDER: "当前 Provider 能力/稳定性不匹配，切换到声明能力更合适的 Provider。",
        RecoveryStrategy.SPLIT_SHOT: "镜头包含过多动作或对白，拆分后可降低生成失败面。",
        RecoveryStrategy.MERGE_SHOTS: "短镜头连续且动作一致，合并可减少转场和连续性风险。",
        RecoveryStrategy.REPLACE_REFERENCE: "参考图可能冲突或未被当前 Provider 接受，改用可追溯参考。",
        RecoveryStrategy.LOWER_RESOLUTION: "优先保证生成闭环，降低分辨率后只重算失败镜头。",
        RecoveryStrategy.REGENERATE_FAILED_SHOTS: "保留已成功镜头，只对失败镜头补拍，避免全量失效。",
        RecoveryStrategy.RESUME_CHECKPOINT: "检测到版本变化或进程续跑，复用未变检查点并局部重算。",
        RecoveryStrategy.RETRY: "失败证据表明是瞬时错误（超时/存储抖动），保留一次低成本重试。",
        RecoveryStrategy.HUMAN_REVIEW: "连续自动修复仍无法满足质量或需要创意取舍，转人工处理。",
        RecoveryStrategy.DEGRADED_PUBLISH: "自动恢复无法继续，但存在可用的部分结果，按降级结果发布并保留失败清单。",
        RecoveryStrategy.TERMINAL_FAILURE: "自动恢复无法继续且没有任何可用结果，明确终止并保留检查点。",
    }.get(strategy, "依据结果选择恢复动作。")
    if profile is not None and not profile.available:
        base += f" 当前候选 Provider 不可用：{profile.reason or '未配置凭据'}。"
    if strategy in MANUAL_ONLY_STRATEGIES and mode != "manual":
        base += " 该策略只有 manual 模式才允许被选择。"
    if strategy not in TERMINAL_STRATEGIES and retries_remaining <= 0:
        base += " 剩余恢复次数为 0。"
    return f"{base} 失败类别={kind}。"


def _rank_key(candidate: RecoveryCandidate, primary: RecoveryStrategy | None) -> tuple[int, float, float]:
    """排序键：失败类别首选策略优先，其次综合评分，最后策略优先级。"""

    fit = 1 if primary is not None and candidate.strategy is primary else 0
    return (fit, candidate.score, _STRATEGY_PRIORITY.get(candidate.strategy, 0.0))


def _score_candidate(
    candidate: RecoveryCandidate,
    quality: QualityStrategy | QualityProfileName | str | None,
    *,
    quality_score: float | None = None,
    primary: RecoveryStrategy | None = None,
) -> float:
    strategy = default_quality_profile(quality if isinstance(quality, (QualityProfileName, str)) else None)
    base = _STRATEGY_PRIORITY[candidate.strategy] / 1.4
    gain = candidate.quality_gain * 0.35
    cost_penalty = 0.0
    if candidate.estimated_cost_micro:
        cost_penalty = min(0.18, candidate.estimated_cost_micro / 500_000 * strategy.cost_multiplier)
    seconds_penalty = min(0.12, (candidate.estimated_seconds or 0) / 500 * strategy.cost_multiplier)
    capability_bonus = 0.12 if candidate.provider_capability_ok else -0.35
    budget_bonus = 0.10 if candidate.budget_fit else -0.45
    sufficiency = 0.0
    if quality_score is not None:
        gap = max(0.0, strategy.quality_threshold - float(quality_score))
        sufficiency = 0.08 if candidate.quality_gain >= max(gap, 0.05) else -0.08
    return round(max(0.0, min(1.0, base + gain - cost_penalty - seconds_penalty + capability_bonus + budget_bonus + sufficiency)), 4)


def _elimination_reason(candidate: RecoveryCandidate, *, mode: str, retries_remaining: int) -> str:
    """不可行候选的淘汰原因；可行候选返回空字符串。"""

    if not candidate.provider_capability_ok:
        return "Provider 能力不匹配"
    if not candidate.budget_fit:
        return "超出剩余预算"
    if candidate.strategy in MANUAL_ONLY_STRATEGIES and mode != "manual":
        return "自动模式禁止选择 human_review"
    if candidate.strategy not in TERMINAL_STRATEGIES and retries_remaining <= 0:
        return "剩余恢复次数为 0"
    return ""


def _has_partial_results(critique: Any, candidate_results: Iterable[dict[str, Any]] | None) -> bool:
    """仅明确结构完整且可定位文件的候选允许降级发布。"""

    evidence = _summarize_candidate_results(candidate_results)
    return int(evidence.get("structurally_usable") or 0) > 0


def _select_candidate(
    options: list[RecoveryCandidate],
    *,
    mode: str,
    eliminations: dict[int, str],
    critique: Any,
    candidate_results: Iterable[dict[str, Any]] | None,
    primary: RecoveryStrategy | None,
) -> RecoveryCandidate | None:
    feasible = [item for item in options if not eliminations.get(id(item))]
    non_terminal = [item for item in feasible if item.strategy not in {RecoveryStrategy.DEGRADED_PUBLISH, RecoveryStrategy.TERMINAL_FAILURE}]
    if non_terminal:
        # 自动恢复是固定阶梯而非全局评分竞价：在同一阶梯中仍允许评分/能力排序，
        # 但绝不跳过更早的可行动作（例如 seed 尚未尝试时不能直接切 Provider）。
        earliest = min(_strategy_rank(item.strategy) for item in non_terminal)
        same_step = [item for item in non_terminal if _strategy_rank(item.strategy) == earliest]
        return max(same_step, key=lambda item: _rank_key(item, primary))
    terminal = [item for item in feasible if item.strategy is RecoveryStrategy.DEGRADED_PUBLISH]
    if terminal and _has_partial_results(critique, candidate_results):
        return terminal[0]
    explicit_failure = [item for item in feasible if item.strategy is RecoveryStrategy.TERMINAL_FAILURE]
    if explicit_failure:
        return explicit_failure[0]
    if mode == "manual":
        human = [item for item in options if item.strategy is RecoveryStrategy.HUMAN_REVIEW]
        if human:
            return human[0]
        return RecoveryCandidate(strategy=RecoveryStrategy.HUMAN_REVIEW, rationale="manual 模式下没有可行候选，转人工审核。")
    return RecoveryCandidate(strategy=RecoveryStrategy.TERMINAL_FAILURE, rationale="自动模式下没有可行候选且无可用部分结果，明确终止。")


def _rejection_records(
    options: list[RecoveryCandidate],
    selected: RecoveryCandidate | None,
    eliminations: dict[int, str],
    *,
    primary: RecoveryStrategy | None,
) -> list[dict[str, Any]]:
    """所有落选候选都带原因：硬约束淘汰或综合排序落后。"""

    records: list[dict[str, Any]] = []
    for item in options:
        if selected is not None and item is selected:
            continue
        reason = eliminations.get(id(item)) or ""
        if not reason:
            reason = (
                f"综合排序低于选中策略（选中={selected.strategy.value if selected else 'none'}，"
                f"rank={_rank_key(item, primary)}）"
            )
        records.append(
            {
                "strategy": item.strategy.value,
                "score": item.score,
                "rank": list(_rank_key(item, primary)),
                "reason": reason,
            }
        )
    return records


def _decision_reason(selected: RecoveryCandidate | None, failure: FailureRecord | None, quality: QualityStrategy, *, mode: str, retries_remaining: int) -> str:
    if selected is None:
        return "没有生成任何恢复候选。"
    head = f"模式={mode}，质量档位={quality.name.value}，剩余恢复次数={retries_remaining}"
    if selected.strategy in TERMINAL_STRATEGIES:
        return f"{head}；自动修复无法继续，选择 {selected.strategy.value}。失败类别={_failure_kind(failure).value}。"
    return (
        f"{head}，优先策略={selected.strategy.value}，"
        f"预计成本={selected.estimated_cost_micro} micro，预计时长={selected.estimated_seconds}s，"
        f"失败类别={_failure_kind(failure).value}。"
    )


__all__ = [
    "TRANSIENT_FAILURES",
    "budget_snapshot",
    "choose_recovery",
    "classify_failure",
    "primary_strategy_for",
    "provider_profiles",
    "recovery_candidates",
]
