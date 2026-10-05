"""Critic/Reviewer：把生成结果转成可执行的修改意见。

这里优先使用确定性指标（结构、时长、对白、版本和 Provider 能力），因此离线、
测试和供应商故障时仍能给出具体修改；LLM/视觉模型可作为增强，不影响主流程。

每份 CritiqueReport 都是完整反思结果：除 passed/score 外，还必须给出失败分类
（failure_kind）、是否可恢复（recoverable）、证据（evidence）、受影响镜头
（affected_shot_ids）和建议恢复策略（recommended_strategy），供决策节点生成
恢复候选使用，而不是只回答通过/不通过。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from services.error_reporter import redact

from .contracts import (
    NON_RECOVERABLE_FAILURES,
    CriticIssue,
    CritiqueReport,
    FailureKind,
    QualityMetric,
    RecoveryStrategy,
    StageName,
)

MAX_DIALOGUE_CHARS_PER_SHOT = 180
MAX_DIALOGUE_CHARS_PER_SECOND = 8.0
MIN_SHOT_SECONDS = 2.0
MAX_SHOT_SECONDS = 5.0
COMPLEX_ACTION_WORDS = ("追逐", "打斗", "翻滚", "连续", "然后", "接着", "同时", "突然", "爆炸", "奔跑", "转身", "跳")

# 视觉质量维度的兜底定义（与 services.structural_validation.VISUAL_DIMENSIONS
# 同源）。这里惰性导入，保证结构检查模块不可用时 critic 仍能给出 pending 结论。
_VISUAL_DIMENSION_FALLBACK: tuple[tuple[str, str], ...] = (
    ("first_frame_storyboard_similarity", "首帧与故事板相似度"),
    ("reference_match", "角色和场景参考匹配度"),
    ("motion_stability", "运动稳定度"),
    ("shot_continuity", "镜头连续性"),
    ("action_completion", "动作完成度"),
)


def _visual_dimension_specs() -> list[dict[str, str]]:
    """视觉维度定义；优先取结构检查模块的注册表，缺失时用本地兜底。"""

    try:
        from services.structural_validation import VISUAL_DIMENSIONS

        return [dict(item) for item in VISUAL_DIMENSIONS]
    except Exception:  # noqa: BLE001 - 注册表不可用时不得阻断检查
        return [{"key": key, "label": label, "reason": "未接入视觉模型"} for key, label in _VISUAL_DIMENSION_FALLBACK]


def _visual_pending_reason() -> str:
    try:
        from services.structural_validation import VISUAL_PENDING_REASON

        return VISUAL_PENDING_REASON
    except Exception:  # noqa: BLE001
        return "未接入视觉模型，视觉质量保持待审"


def _visual_pending_metrics(report_dimensions: Any = None, *, detail: str = "") -> list[QualityMetric]:
    """五项视觉维度各出一条 ``passed=None`` 指标：只声明待审，不给结论。

    维度键来自结构检查报告（存在时）以保证与产物检查一致；报告缺失时回退到
    注册表。任何情况下都不得产出 ``passed=True``。
    """

    keys_from_report = [
        (str(item.get("key") or ""), str(item.get("label") or ""), str(item.get("reason") or ""))
        for item in (report_dimensions or [])
        if isinstance(item, dict) and item.get("key")
    ]
    specs = keys_from_report or [
        (item["key"], item["label"], item.get("reason", "")) for item in _visual_dimension_specs()
    ]
    metrics: list[QualityMetric] = []
    for key, label, reason in specs:
        metrics.append(
            QualityMetric(
                name=key,
                passed=None,
                detail=f"{label}：{reason or detail or _visual_pending_reason()}（pending，未评估）",
            )
        )
    return metrics


# 稳定 issue code -> 失败分类；决策节点据此生成恢复候选。
_ISSUE_FAILURE_KINDS: dict[str, FailureKind] = {
    "llm_invalid_output": FailureKind.LLM_INVALID_OUTPUT,
    "llm_failed": FailureKind.LLM_INVALID_OUTPUT,
    "empty_storyboard": FailureKind.LLM_INVALID_OUTPUT,
    "missing_characters": FailureKind.LLM_INVALID_OUTPUT,
    "missing_scenes": FailureKind.LLM_INVALID_OUTPUT,
    "dialogue_too_long": FailureKind.DIALOGUE_TOO_LONG,
    "dialogue_duration_ratio": FailureKind.DIALOGUE_TOO_LONG,
    "shot_too_complex": FailureKind.SHOT_TOO_COMPLEX,
    "provider_reference_unsupported": FailureKind.PROVIDER_REFERENCE_UNSUPPORTED,
    "provider_capability_mismatch": FailureKind.PROVIDER_CAPABILITY_MISMATCH,
    "provider_unavailable": FailureKind.PROVIDER_UNAVAILABLE,
    "image_invalid": FailureKind.IMAGE_FAILED,
    "image_generation_failure": FailureKind.IMAGE_FAILED,
    "video_generation_failure": FailureKind.VIDEO_FAILED,
    "audio_missing": FailureKind.AUDIO_FAILED,
    "audio_generation_failure": FailureKind.AUDIO_FAILED,
    "render_missing": FailureKind.STORAGE_FAILED,
    "tail_frame_missing": FailureKind.STORAGE_FAILED,
    "storage_failed": FailureKind.STORAGE_FAILED,
    "video_check_error": FailureKind.VIDEO_FAILED,
    "quality_review_failed": FailureKind.QUALITY_BELOW_THRESHOLD,
    "video_review_failed": FailureKind.QUALITY_BELOW_THRESHOLD,
    "incomplete_timeline": FailureKind.DEPENDENCY_FAILED,
}

# 失败分类 -> Critic 层建议策略（决策节点仍会按预算/能力/重试次数重新评估）。
_RECOMMENDED_STRATEGY: dict[FailureKind, RecoveryStrategy] = {
    FailureKind.LLM_INVALID_OUTPUT: RecoveryStrategy.REVISE_PROMPT,
    FailureKind.DIALOGUE_TOO_LONG: RecoveryStrategy.SPLIT_SHOT,
    FailureKind.SHOT_TOO_COMPLEX: RecoveryStrategy.SPLIT_SHOT,
    FailureKind.PROVIDER_REFERENCE_UNSUPPORTED: RecoveryStrategy.SWITCH_PROVIDER,
    FailureKind.PROVIDER_CAPABILITY_MISMATCH: RecoveryStrategy.SWITCH_PROVIDER,
    FailureKind.PROVIDER_UNAVAILABLE: RecoveryStrategy.SWITCH_PROVIDER,
    FailureKind.IMAGE_FAILED: RecoveryStrategy.REGENERATE_FAILED_SHOTS,
    FailureKind.VIDEO_FAILED: RecoveryStrategy.REGENERATE_FAILED_SHOTS,
    FailureKind.AUDIO_FAILED: RecoveryStrategy.REGENERATE_FAILED_SHOTS,
    FailureKind.QUALITY_BELOW_THRESHOLD: RecoveryStrategy.REVISE_PROMPT,
    FailureKind.VERSION_CONFLICT: RecoveryStrategy.RESUME_CHECKPOINT,
    FailureKind.USER_CHANGED_INPUT: RecoveryStrategy.RESUME_CHECKPOINT,
    FailureKind.DEPENDENCY_FAILED: RecoveryStrategy.RESUME_CHECKPOINT,
    FailureKind.BUDGET_EXCEEDED: RecoveryStrategy.DEGRADED_PUBLISH,
    FailureKind.STORAGE_FAILED: RecoveryStrategy.REGENERATE_FAILED_SHOTS,
    FailureKind.TIMEOUT: RecoveryStrategy.RETRY,
    FailureKind.CANCELLED: RecoveryStrategy.TERMINAL_FAILURE,
    FailureKind.UNKNOWN: RecoveryStrategy.REVISE_PROMPT,
}

# 视频结构/技术检查可能返回大量具体失败码，统一按前缀归为 VIDEO_FAILED。
_VIDEO_CODE_PREFIX = "video_"


def classify_issue_code(code: str) -> FailureKind:
    """把稳定 issue code 映射为失败分类；未知 code 归为质量不达标。"""

    text = str(code or "")
    if text in _ISSUE_FAILURE_KINDS:
        return _ISSUE_FAILURE_KINDS[text]
    if text.startswith(_VIDEO_CODE_PREFIX):
        return FailureKind.VIDEO_FAILED
    return FailureKind.QUALITY_BELOW_THRESHOLD


def recommended_strategy_for(kind: FailureKind | None) -> RecoveryStrategy | None:
    return _RECOMMENDED_STRATEGY.get(kind) if kind else None


def critique_director(state: dict[str, Any]) -> CritiqueReport:
    """导演规划的结构/覆盖检查。"""
    characters = list(state.get("characters") or [])
    scenes = list(state.get("script_scenes") or [])
    logic_issues = list(state.get("logic_issues") or [])
    issues: list[CriticIssue] = []
    if not characters:
        issues.append(
            CriticIssue(
                code="missing_characters",
                severity="error",
                message="导演规划缺少角色",
                recommendation="要求模型输出至少一名可执行角色",
            )
        )
    if not scenes:
        issues.append(
            CriticIssue(
                code="missing_scenes",
                severity="error",
                message="导演规划缺少场景",
                recommendation="要求模型输出至少一个明确场景",
            )
        )
    for item in logic_issues[:10]:
        issues.append(
            CriticIssue(
                code="logic_issue",
                severity="warning",
                message=str(item),
                recommendation="在分镜中补足因果/时间线说明或转人工确认",
            )
        )
    metrics = [
        QualityMetric(name="schema_valid", passed=bool(characters and scenes)),
        QualityMetric(name="character_coverage", value=len(characters), threshold=1, passed=bool(characters)),
        QualityMetric(name="scene_coverage", value=len(scenes), threshold=1, passed=bool(scenes)),
        QualityMetric(name="logic_issue_count", value=len(logic_issues), threshold=0, passed=not logic_issues),
    ]
    score = max(0.0, 1.0 - 0.25 * sum(1 for item in issues if item.severity == "error") - 0.08 * len(logic_issues))
    return _finalize(
        StageName.DIRECTOR_PLANNING,
        passed=not any(item.severity == "error" for item in issues),
        score=score,
        metrics=metrics,
        issues=issues,
    )


def critique_storyboard(state: dict[str, Any]) -> CritiqueReport:
    shots = list(state.get("shots") or [])
    issues: list[CriticIssue] = []
    metrics: list[QualityMetric] = []
    shot_count = len(shots)
    metrics.append(QualityMetric(name="shot_count", value=shot_count, threshold=1, passed=shot_count > 0))
    if not shots:
        issues.append(
            CriticIssue(
                code="empty_storyboard",
                severity="error",
                message="没有可用镜头",
                recommendation="重新解析剧本并要求至少一个镜头",
            )
        )
    for shot in shots:
        shot_id = str(shot.get("shot_id") or shot.get("id") or "")
        dialogue = str(shot.get("dialogue") or "")
        duration = float(shot.get("duration") or 0.0)
        if len(dialogue) > MAX_DIALOGUE_CHARS_PER_SHOT:
            issues.append(
                CriticIssue(
                    code="dialogue_too_long",
                    severity="error",
                    message=f"镜头 {shot_id or '?'} 对白 {len(dialogue)} 字，超出单镜头安全长度",
                    shot_id=shot_id,
                    recommendation=f"拆成每段不超过 {MAX_DIALOGUE_CHARS_PER_SHOT // 2} 字的台词，并增加停顿镜头",
                    details={"length": len(dialogue), "max": MAX_DIALOGUE_CHARS_PER_SHOT},
                )
            )
        if duration > 0 and len(dialogue) / duration > MAX_DIALOGUE_CHARS_PER_SECOND:
            issues.append(
                CriticIssue(
                    code="dialogue_duration_ratio",
                    severity="warning",
                    message=f"镜头 {shot_id or '?'} 对白密度偏高，预计无法自然说完",
                    shot_id=shot_id,
                    recommendation="拆分镜头或减少台词，使语速不高于每秒 8 字",
                    details={"chars_per_second": round(len(dialogue) / duration, 2)},
                )
            )
        if duration and (duration < MIN_SHOT_SECONDS or duration > MAX_SHOT_SECONDS):
            issues.append(
                CriticIssue(
                    code="shot_duration",
                    severity="warning",
                    message=f"镜头 {shot_id or '?'} 时长 {duration:.1f}s 超出 2-5 秒生成安全区间",
                    shot_id=shot_id,
                    recommendation="拆分过长镜头，合并连续短镜头",
                )
            )
        action = str(shot.get("character_action") or shot.get("scene_description") or "")
        hit_words = [word for word in COMPLEX_ACTION_WORDS if word in action]
        if len(hit_words) >= 2:
            issues.append(
                CriticIssue(
                    code="shot_too_complex",
                    severity="warning",
                    message=f"镜头 {shot_id or '?'} 包含多个动作节拍: {', '.join(hit_words[:4])}",
                    shot_id=shot_id,
                    recommendation="按动作节拍拆成 2-5 秒短镜头",
                )
            )
    valid_count = sum(1 for issue in issues if issue.severity == "error")
    dialogue_ok = not any(issue.code in {"dialogue_too_long", "dialogue_duration_ratio"} for issue in issues)
    duration_ok = not any(issue.code == "shot_duration" for issue in issues)
    metrics.extend(
        [
            QualityMetric(name="dialogue_duration_ratio", passed=dialogue_ok, detail="检查对白长度和预计语速"),
            QualityMetric(name="shot_rule_compliance", passed=duration_ok, detail="检查镜头时长和动作复杂度"),
        ]
    )
    score = 1.0 if not issues else max(0.0, 1.0 - sum(0.2 if item.severity == "error" else 0.08 for item in issues))
    return _finalize(
        StageName.STORYBOARD_DESIGN,
        passed=valid_count == 0 and score >= 0.55,
        score=score,
        metrics=metrics,
        issues=issues,
    )


def critique_assets(state: dict[str, Any], *, reference_supported: bool = True) -> CritiqueReport:
    characters = list(state.get("characters") or [])
    scenes = list(state.get("script_scenes") or [])
    issues: list[CriticIssue] = []
    missing_chars = [
        str(item.get("name") or item.get("id") or "?") for item in characters if not item.get("reference_images")
    ]
    missing_scenes = [
        str(item.get("id") or item.get("name") or "?")
        for item in scenes
        if not (item.get("baseline_image_path") or item.get("reference_images"))
    ]
    if missing_chars:
        issues.append(
            CriticIssue(
                code="missing_character_reference",
                severity="warning",
                message=f"缺少角色参考: {', '.join(missing_chars[:8])}",
                recommendation="补生成角色三视图或替换为可靠参考",
            )
        )
    if missing_scenes:
        issues.append(
            CriticIssue(
                code="missing_scene_reference",
                severity="warning",
                message=f"缺少场景基准图: {', '.join(missing_scenes[:8])}",
                recommendation="补生成场景基准图或降低一致性要求",
            )
        )
    if not reference_supported and (characters or scenes):
        issues.append(
            CriticIssue(
                code="provider_reference_unsupported",
                severity="warning",
                message="当前图像 Provider 不支持参考图",
                recommendation="切换支持参考图的 Provider，或明确降级为纯文本生成",
            )
        )
    metrics = [
        QualityMetric(
            name="character_reference_coverage",
            value=round(1 - len(missing_chars) / max(1, len(characters)), 3),
            passed=not missing_chars,
        ),
        QualityMetric(
            name="scene_reference_coverage",
            value=round(1 - len(missing_scenes) / max(1, len(scenes)), 3),
            passed=not missing_scenes,
        ),
        QualityMetric(name="reference_compatibility", passed=reference_supported or not (characters or scenes)),
    ]
    score = sum(1.0 for item in metrics if item.passed) / max(1, len(metrics))
    return _finalize(
        StageName.ASSET_PREPARATION,
        passed=not issues or all(item.severity != "error" for item in issues),
        score=score,
        metrics=metrics,
        issues=issues,
    )


def critique_images(shot_artifacts: Iterable[dict[str, Any]]) -> CritiqueReport:
    """图片 Critic：结构合格 ≠ 画面合格。

    ``candidate_score`` 只统计**真实提供**的候选评分；没有任何评分证据时该维度
    保持 pending（``passed=None``），绝不因为「结构检查通过」就默认给满分——
    否则未接入视觉模型时会伪造出一次质量通过。
    """

    artifacts = list(shot_artifacts)
    issues: list[CriticIssue] = []
    valid = 0
    scores: list[float] = []
    for item in artifacts:
        shot_id = str(item.get("shot_id") or "")
        path = str(item.get("path") or item.get("storyboard_path") or item.get("image_path") or "")
        structural = _validate_image(path)
        if not structural.get("passed"):
            issues.append(
                CriticIssue(
                    code="image_invalid",
                    severity="error",
                    message=f"镜头 {shot_id or '?'} 图片结构不合格",
                    shot_id=shot_id,
                    recommendation="只重生成该镜头，必要时替换参考图或降分辨率",
                    details={**structural, "path": path},
                )
            )
        else:
            valid += 1
        raw_score = item.get("score")
        if raw_score is not None and raw_score != "":
            try:
                scores.append(max(0.0, min(1.0, float(raw_score))))
            except (TypeError, ValueError):
                pass
        if item.get("failure"):
            issues.append(
                CriticIssue(
                    code="image_generation_failure",
                    severity="error",
                    message=f"镜头 {shot_id or '?'} 生成失败",
                    shot_id=shot_id,
                    recommendation="保留成功镜头，只重算失败镜头",
                    details=dict(item.get("failure") or {}),
                )
            )
    has_scores = bool(scores)
    metric_score = round(sum(scores) / len(scores), 3) if has_scores else 0.0
    metrics = [
        QualityMetric(name="image_valid", value=valid, threshold=len(artifacts), passed=valid == len(artifacts)),
        # 无评分证据时保持 pending：结构通过不能替代画面质量结论。
        QualityMetric(
            name="candidate_score",
            value=metric_score if has_scores else None,
            threshold=0.72,
            passed=(metric_score >= 0.72) if has_scores else None,
            detail="候选评分来自审核模型" if has_scores else "无候选评分证据（未接入视觉模型），保持待审",
        ),
        # 视觉维度同样逐项 pending，与视频阶段保持一致。
        *_visual_pending_metrics(detail="故事板视觉质量未评估"),
    ]
    if not has_scores and artifacts:
        issues.append(
            CriticIssue(
                code="visual_quality_pending",
                severity="info",
                message=f"{len(artifacts)} 个故事板镜头的视觉质量待审：没有候选评分证据，本阶段不判断画面质量",
                recommendation="接入视觉审核模型后复检；结构合格不代表画面合格",
                details={"stage": StageName.IMAGE_GENERATION.value, "status": "pending"},
            )
        )
    return _finalize(
        StageName.IMAGE_GENERATION,
        # 只有 error 级问题才阻断；visual_quality_pending 是 info 级待审说明。
        passed=valid == len(artifacts) and not any(issue.severity == "error" for issue in issues),
        # 无评分证据时分数按结构合格率给（可追踪），但候选评分维度仍是 pending。
        score=metric_score if has_scores else (valid / max(1, len(artifacts))),
        metrics=metrics,
        issues=issues,
    )


def critique_videos(shot_artifacts: Iterable[dict[str, Any]]) -> CritiqueReport:
    """视频 Critic：结构 / 技术 / 视觉待审 三维度，不冒充视觉质量结论。

    结构与技术问题都带稳定 issue code 与恢复建议；视觉质量按五项固定维度
    （首帧与故事板相似度、角色和场景参考匹配度、运动稳定度、镜头连续性、
    动作完成度）逐项输出 ``passed=None`` 的 pending 结论，绝不据此宣称角色
    一致性或画面质量通过——那需要身份/场景/动作识别模型，当前尚未接入。
    """

    artifacts = list(shot_artifacts)
    issues: list[CriticIssue] = []
    structural_pass_count = 0
    technical_pass_count = 0
    valid = 0
    pending_reason = ""
    pending_shot_ids: list[str] = []
    report_dimensions: list[dict[str, Any]] = []
    for item in artifacts:
        shot_id = str(item.get("shot_id") or "")
        path = str(item.get("path") or item.get("video_path") or "")
        expected_duration = _optional_float(item.get("expected_duration_s"))
        expected_aspect = _optional_float(item.get("expected_aspect_ratio"))
        audio_duration = _probe_audio_duration(item.get("audio_path"))
        tail_frame = str(item.get("tail_frame_path") or "")
        report = _validate_video(
            path,
            expected_duration_s=expected_duration,
            expected_aspect_ratio=expected_aspect,
            audio_duration_s=audio_duration,
            tail_frame_path=tail_frame or None,
            first_frame_path=str(item.get("first_frame_path") or "") or None,
            expect_audio=item.get("expect_audio") if isinstance(item.get("expect_audio"), bool) else None,
        )
        categories = report.get("categories") or {}
        structural = categories.get("structural_validity") or {}
        technical = categories.get("technical_quality") or {}
        visual_category = categories.get("visual_quality_pending") or {}
        pending_reason = str(visual_category.get("reason") or pending_reason)
        if not report_dimensions and visual_category.get("dimensions"):
            report_dimensions = [dict(entry) for entry in visual_category["dimensions"] if isinstance(entry, dict)]
        item_failed = False
        if structural.get("passed"):
            structural_pass_count += 1
        else:
            item_failed = True
        if technical.get("passed") is True:
            technical_pass_count += 1
        else:
            item_failed = True
        for category, issue_items in (
            ("structural_validity", structural.get("issues") or []),
            ("technical_quality", technical.get("issues") or []),
        ):
            for entry in issue_items:
                item_failed = True
                issues.append(
                    CriticIssue(
                        code=str(entry.get("code") or f"{category}_failed"),
                        severity="error",
                        message=f"镜头 {shot_id or '?'} {entry.get('message') or '视频检查未通过'}",
                        shot_id=shot_id,
                        recommendation=str(entry.get("recommendation") or "按问题类型局部重生成该镜头"),
                        details={"category": category, "path": path, "stage": StageName.VIDEO_GENERATION.value},
                    )
                )
        if not item_failed:
            valid += 1
        # 视觉待审逐镜头登记：最终报告要能定位到具体镜头，而不是只给一句总体说明。
        pending_shot_ids.append(shot_id or path)
        for entry in visual_category.get("dimensions") or []:
            if isinstance(entry, dict) and entry.get("key"):
                issues.append(
                    CriticIssue(
                        code=f"visual_pending:{entry['key']}",
                        severity="info",
                        message=f"镜头 {shot_id or '?'} 视觉维度「{entry.get('label') or entry['key']}」待审：{entry.get('reason') or pending_reason}",
                        shot_id=shot_id,
                        recommendation="接入真实视觉模型后复检；pending 既不代表通过也不代表失败",
                        details={
                            "category": "visual_quality_pending",
                            "dimension": str(entry["key"]),
                            "stage": StageName.VIDEO_GENERATION.value,
                            "status": "pending",
                            "path": path,
                        },
                    )
                )
        for warning in report.get("warnings") or []:
            # 非阻断观察项（轻微比例偏差/短暂静止等）：只提示，不影响通过。
            issues.append(
                CriticIssue(
                    code=str(warning.get("code") or "video_observation"),
                    severity="warning",
                    message=f"镜头 {shot_id or '?'} {warning.get('message') or '存在非阻断观察项'}",
                    shot_id=shot_id,
                    recommendation=str(warning.get("recommendation") or "按需局部优化"),
                    details={"category": "technical_quality", "path": path, "blocking": False},
                )
            )
        if item.get("failure"):
            issues.append(
                CriticIssue(
                    code="video_generation_failure",
                    severity="error",
                    message=f"镜头 {shot_id or '?'} 视频生成失败",
                    shot_id=shot_id,
                    recommendation="仅重生成失败镜头，不回滚成功镜头",
                    details=dict(item.get("failure") or {}),
                )
            )
    if artifacts:
        issues.append(
            CriticIssue(
                code="visual_quality_pending",
                severity="info",
                message=f"{len(artifacts)} 个镜头的视觉质量待审：未接入身份识别、场景识别或动作识别模型，本阶段不评估角色一致性/构图/美学",
                recommendation="需要视觉结论时接入视觉审核模型；当前结构+技术通过不代表视觉质量通过",
                details={
                    "stage": StageName.VIDEO_GENERATION.value,
                    "status": "pending",
                    "dimensions": [item["key"] for item in (report_dimensions or _visual_dimension_specs())],
                    "shot_ids": sorted({shot_id for shot_id in pending_shot_ids if shot_id}),
                },
            )
        )
    score = valid / max(1, len(artifacts))  # valid = 结构+技术均通过；视觉质量始终单独待审。
    metrics = [
        QualityMetric(
            name="structural_validity",
            value=structural_pass_count,
            threshold=len(artifacts),
            passed=len(artifacts) == 0 or structural_pass_count == len(artifacts),
            detail="可播放性、视频轨与基本时长",
        ),
        QualityMetric(
            name="technical_quality",
            value=technical_pass_count,
            threshold=len(artifacts),
            passed=len(artifacts) == 0 or technical_pass_count == len(artifacts),
            detail="时长/分辨率/比例/黑帧/空帧/冻结/音画时长/尾帧/文件过小",
        ),
        QualityMetric(
            name="visual_quality_pending", passed=None, detail=pending_reason or "未接入视觉模型，视觉质量保持待审"
        ),
        # 五项视觉维度逐项 pending：显式暴露"未评估"，不静默缺失也不伪造通过。
        *_visual_pending_metrics(report_dimensions, detail=pending_reason),
        QualityMetric(
            name="video_valid",
            value=valid,
            threshold=len(artifacts),
            passed=len(artifacts) == 0 or valid == len(artifacts),
        ),
        QualityMetric(
            name="duration_match",
            passed=not any(issue.code == "video_duration_shorter_than_plan" for issue in issues),
            detail="实际时长与执行计划的偏差",
        ),
    ]
    return _finalize(
        StageName.VIDEO_GENERATION,
        passed=not any(issue.severity == "error" for issue in issues),
        score=score,
        metrics=metrics,
        issues=issues,
    )


def critique_audio(
    shots: Iterable[dict[str, Any]], audio_artifacts: Iterable[dict[str, Any]] | None = None
) -> CritiqueReport:
    shot_list = list(shots)
    artifacts = {str(item.get("shot_id") or ""): item for item in (audio_artifacts or [])}
    issues: list[CriticIssue] = []
    ready = 0
    for shot in shot_list:
        shot_id = str(shot.get("shot_id") or shot.get("id") or "")
        dialogue = str(shot.get("dialogue") or "")
        item = artifacts.get(shot_id, {})
        path = str(item.get("path") or shot.get("audio_path") or "")
        if len(dialogue) > MAX_DIALOGUE_CHARS_PER_SHOT:
            issues.append(
                CriticIssue(
                    code="dialogue_too_long",
                    severity="error",
                    message=f"镜头 {shot_id or '?'} 对白过长，无法稳定配音",
                    shot_id=shot_id,
                    recommendation="拆句、拆镜头或转人工改写台词",
                    details={"length": len(dialogue), "max": MAX_DIALOGUE_CHARS_PER_SHOT},
                )
            )
        native_audio = any(
            str(metric.get("name") or "") == "audio_source" and str(metric.get("value") or "") == "native"
            for metric in (item.get("metrics") or [])
            if isinstance(metric, dict)
        )
        if dialogue and not path and not native_audio:
            issues.append(
                CriticIssue(
                    code="audio_missing",
                    severity="error",
                    message=f"镜头 {shot_id or '?'} 有对白但没有配音",
                    shot_id=shot_id,
                    recommendation="只重生成该镜头音频，复用其它成功音频",
                )
            )
        elif path or not dialogue or native_audio:
            ready += 1
    score = ready / max(1, len(shot_list))
    metrics = [
        QualityMetric(name="dialogue_length", passed=not any(item.code == "dialogue_too_long" for item in issues)),
        QualityMetric(name="tts_valid", value=ready, threshold=len(shot_list), passed=ready == len(shot_list)),
    ]
    return _finalize(
        StageName.AUDIO_PRODUCTION,
        passed=not issues and ready == len(shot_list),
        score=score,
        metrics=metrics,
        issues=issues,
    )


def critique_compose(project_id: str, shots: Iterable[dict[str, Any]], output_path: str = "") -> CritiqueReport:
    shot_list = list(shots)
    missing = [str(item.get("shot_id") or item.get("id") or "?") for item in shot_list if not item.get("video_path")]
    issues: list[CriticIssue] = []
    if missing:
        issues.append(
            CriticIssue(
                code="incomplete_timeline",
                severity="warning",
                message=f"有 {len(missing)} 个镜头没有视频，无法无损合成",
                recommendation="明确跳过并降级，或只补拍缺失镜头",
                details={"shot_ids": missing[:20]},
            )
        )
    if not output_path:
        issues.append(
            CriticIssue(
                code="render_missing",
                severity="error",
                message="没有成片输出路径",
                recommendation="重新剪辑合成或转人工检查媒体清单",
            )
        )
    score = max(0.0, 1.0 - 0.2 * len(missing)) if output_path else 0.0
    metrics = [
        QualityMetric(
            name="shot_completeness", value=len(shot_list) - len(missing), threshold=len(shot_list), passed=not missing
        ),
        QualityMetric(name="render_valid", passed=bool(output_path)),
    ]
    return _finalize(
        StageName.EDIT_COMPOSITION,
        passed=bool(output_path) and not missing,
        score=score,
        metrics=metrics,
        issues=issues,
    )


def critique_final(state: dict[str, Any]) -> CritiqueReport:
    """生成成片复审报告，并保留每个恢复动作的可定位证据。

    ``CritiqueReport`` 是历史上稳定的跨节点契约，不能在这里返回一个平行的
    report 类型。因此最终报告以 ``evidence[kind=final_report]`` 的形式嵌入，
    其中的结构化对象可直接写入检查点/API；阶段、镜头和候选信息均来自已有
    ``critiques``、``decision_traces``、``recovery_history`` 与产物状态，不猜测
    未记录的视觉结论。
    """
    issues: list[CriticIssue] = []
    shots = _latest_current_artifacts(state.get("shot_artifacts") or state.get("shots") or [])
    failed = [item for item in shots if item.get("failure") or item.get("status") in {"failed", "needs_review"}]
    if failed:
        issues.append(
            CriticIssue(
                code="degraded_shots",
                severity="warning",
                message=f"成片包含 {len(failed)} 个降级或待审核镜头",
                recommendation="只补拍失败镜头或在人工确认后发布",
                details={"shot_ids": [str(item.get("shot_id") or "") for item in failed[:20]]},
            )
        )
        for item in failed:
            shot_id = str(item.get("shot_id") or "")
            source_stage = str(item.get("stage") or StageName.VIDEO_GENERATION.value)
            failure = item.get("failure") or {}
            kind = str(failure.get("kind") or "artifact_failed") if isinstance(failure, dict) else "artifact_failed"
            issues.append(
                CriticIssue(
                    code=kind,
                    severity="error",
                    message=f"镜头 {shot_id or '?'} 当前产物失败",
                    shot_id=shot_id,
                    recommendation="仅恢复该镜头对应阶段，保留其它已成功镜头",
                    details={
                        "source_stage": source_stage,
                        "artifact_status": str(item.get("status") or ""),
                        "failure_kind": kind,
                    },
                )
            )
    if state.get("human_feedback"):
        issues.append(
            CriticIssue(
                code="human_feedback",
                severity="info",
                message="收到人工反馈",
                recommendation="将反馈转为局部 Prompt/镜头修改，不重跑已成功部分",
            )
        )
    if state.get("run_status") == "waiting_human":
        issues.append(
            CriticIssue(
                code="human_gate", severity="info", message="流程停在人工卡点", recommendation="人工确认后从检查点续跑"
            )
        )

    # 视觉模型未接入时，只记录 pending 风险；不能把它转换成通过或失败。
    visual_pending = _visual_pending_entries(state.get("critiques") or [])
    if visual_pending:
        issues.append(
            CriticIssue(
                code="visual_quality_pending",
                severity="info",
                message="视觉质量仍待评估：未接入身份/场景/动作识别模型",
                recommendation="接入真实视觉审核模型后再给出视觉质量结论",
                details={"stages": sorted({str(item.get("stage") or "") for item in visual_pending})},
            )
        )

    score = max(0.0, 1.0 - 0.16 * len(failed))
    threshold = float(state.get("quality_threshold") or 0.72)
    metrics = [
        QualityMetric(name="overall_score", value=round(score, 3), threshold=threshold, passed=score >= threshold),
        QualityMetric(name="human_gate", passed=state.get("run_status") != "waiting_human"),
        QualityMetric(name="visual_quality_pending", passed=None, detail="未接入视觉模型，视觉质量保持待审")
        if visual_pending
        else QualityMetric(name="visual_quality_pending", passed=None, detail="本次没有可观测的视觉模型结论"),
    ]
    report = _finalize(
        StageName.FINAL_REVIEW,
        passed=score >= threshold and not failed,
        score=score,
        metrics=metrics,
        issues=issues,
    )
    final_report = build_final_report(state, final_critique=report)
    # _finalize 已经构造了指标/问题证据；把最终汇总放在首项，便于消费者按
    # ``kind`` 稳定查找，同时保留原始证据供旧客户端继续使用。
    return report.model_copy(
        update={
            "evidence": [{"kind": "final_report", "report": final_report}, *report.evidence],
        }
    )


def _latest_current_artifacts(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the latest artifact per shot so repaired historical failures stay resolved."""

    latest: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        shot_id = str(item.get("shot_id") or "")
        key = shot_id or f"__row_{len(latest)}"
        latest[key] = dict(item)
    return list(latest.values())


def build_final_report(
    state: dict[str, Any],
    *,
    final_critique: CritiqueReport | None = None,
) -> dict[str, Any]:
    """从已有追踪数据汇总可序列化的最终报告。

    报告不执行新的生成或质量推断：``automatic_repairs`` 只来自已落盘的
    DecisionTrace/recovery_history，``unresolved_risks`` 只列出仍可见的失败、
    warning、terminal 或 visual pending 证据。这样 final_review 既能定位到镜头
    和阶段，也不会把缺少视觉能力误报为通过。
    """
    traces = [item for item in (state.get("decision_traces") or []) if isinstance(item, dict)]
    history = [item for item in (state.get("recovery_history") or []) if isinstance(item, dict)]
    artifacts = [
        item for item in (state.get("shot_artifacts") or state.get("artifacts") or []) if isinstance(item, dict)
    ]
    critiques = [item for item in (state.get("critiques") or []) if isinstance(item, dict)]
    visual_pending = _visual_pending_entries(critiques)

    automatic_repairs: list[dict[str, Any]] = []
    seen_trace_ids: set[str] = set()
    for trace in traces:
        selected = trace.get("selected") or {}
        if not isinstance(selected, dict):
            continue
        strategy = str(selected.get("strategy") or "")
        if strategy in {
            "",
            RecoveryStrategy.DEGRADED_PUBLISH.value,
            RecoveryStrategy.TERMINAL_FAILURE.value,
            RecoveryStrategy.HUMAN_REVIEW.value,
        }:
            continue
        trace_id = str(trace.get("trace_id") or "")
        if trace_id and trace_id in seen_trace_ids:
            continue
        if trace_id:
            seen_trace_ids.add(trace_id)
        stage = str(trace.get("stage") or selected.get("target_stage") or "")
        failure = trace.get("failure") or {}
        critique = trace.get("critique") or {}
        shot_ids = _trace_shot_ids(trace, selected, critique)
        after = _recovery_after_state(artifacts, shot_ids, stage)
        automatic_repairs.append(
            {
                "trace_id": trace_id,
                "action": strategy,
                "stage": stage,
                "shot_ids": shot_ids,
                "attempt": _trace_attempt(trace, history, stage, strategy),
                "outcome": str(after.get("outcome") or "attempted"),
                "reason": _safe_report_text(trace.get("reason") or selected.get("rationale") or ""),
                "provider": _safe_report_text(selected.get("provider") or ""),
                "seed": _candidate_seed(selected),
                "prompt_changes": _safe_report_value(selected.get("prompt_changes") or {}),
                "before": {
                    "status": "failed" if failure else "quality_check",
                    "failure_kind": str(failure.get("kind") or ""),
                    "message": _safe_report_text(str(failure.get("message") or "")),
                    "score": trace.get("quality_score", critique.get("score")),
                },
                "after": after,
            }
        )

    # 旧检查点可能只有 recovery_history、没有完整 trace；补一条最小但仍可定位
    # 的记录，避免最终报告丢失实际执行过的动作。
    for row in history:
        strategy = str(row.get("strategy") or row.get("selected_strategy") or "")
        if strategy in {
            "",
            RecoveryStrategy.DEGRADED_PUBLISH.value,
            RecoveryStrategy.TERMINAL_FAILURE.value,
            RecoveryStrategy.HUMAN_REVIEW.value,
        }:
            continue
        trace_id = str(row.get("trace_id") or "")
        if trace_id and trace_id in seen_trace_ids:
            continue
        automatic_repairs.append(
            {
                "trace_id": trace_id,
                "action": strategy,
                "stage": str(row.get("stage") or ""),
                "shot_ids": [str(item) for item in (row.get("shot_ids") or []) if item],
                "attempt": row.get("attempt"),
                "outcome": str((row.get("after") or {}).get("outcome") or "attempted"),
                "reason": _safe_report_text(row.get("reason") or ""),
                "provider": _safe_report_text(row.get("provider") or ""),
                "seed": row.get("seed"),
                "prompt_changes": _safe_report_value(row.get("prompt_changes") or {}),
                "before": _safe_report_value(dict(row.get("before") or {})),
                "after": _safe_report_value(dict(row.get("after") or {})),
            }
        )

    degradations: list[dict[str, Any]] = []
    for item in artifacts:
        if str(item.get("status") or "") != "degraded":
            continue
        degradations.append(
            {
                "stage": str(item.get("stage") or ""),
                "shot_ids": [str(item.get("shot_id") or "")] if item.get("shot_id") else [],
                "reason": _safe_report_text(
                    item.get("degraded_reason") or (item.get("failure") or {}).get("message") or "产物被标记为 degraded"
                ),
                "candidate_id": str(item.get("selected_video_candidate_id") or ""),
                "path_available": _has_media_evidence(item),
            }
        )
    for trace in traces:
        selected = trace.get("selected") or {}
        if (
            not isinstance(selected, dict)
            or str(selected.get("strategy") or "") != RecoveryStrategy.DEGRADED_PUBLISH.value
        ):
            continue
        degradations.append(
            {
                "stage": str(trace.get("stage") or selected.get("target_stage") or ""),
                "shot_ids": _trace_shot_ids(trace, selected, trace.get("critique") or {}),
                "reason": _safe_report_text(
                    trace.get("reason") or selected.get("rationale") or state.get("degraded_reason") or ""
                ),
                "candidate_id": str(
                    trace.get("selected_video_candidate_id") or selected.get("selected_video_candidate_id") or ""
                ),
                "path_available": _trace_has_candidate_evidence(trace, selected, artifacts),
            }
        )
    if state.get("degraded_published") and not degradations:
        degradations.append(
            {
                "stage": str(state.get("current_stage") or ""),
                "shot_ids": [str(item) for item in (state.get("degraded_shot_ids") or []) if item],
                "reason": _safe_report_text(state.get("degraded_reason") or "自动恢复无法继续，按可用结果降级"),
                "candidate_id": "",
                "path_available": _has_media_evidence(state),
            }
        )

    unresolved_risks = _unresolved_risks(
        state,
        critiques=critiques,
        artifacts=artifacts,
        final_critique=final_critique,
        visual_pending=visual_pending,
    )
    selected_strategy = str(state.get("selected_strategy") or "")
    final_trace = next(
        (item for item in reversed(traces) if str(item.get("stage") or "") == StageName.FINAL_REVIEW.value), None
    )
    if not selected_strategy and final_trace:
        selected_strategy = str((final_trace.get("selected") or {}).get("strategy") or "")
    run_status = str(state.get("run_status") or ("completed" if final_critique and final_critique.passed else ""))
    if final_critique is not None and final_critique.passed and run_status in {"", "running", "recovering"}:
        run_status = "completed"
    terminal = selected_strategy == RecoveryStrategy.TERMINAL_FAILURE.value or run_status == "failed"
    pending_dimensions = sorted(
        {
            str(dimension)
            for item in unresolved_risks
            if item.get("code") == "visual_quality_pending"
            for dimension in (item.get("dimensions") or [])
            if dimension
        }
    )
    visual_pending_state = bool(pending_dimensions) or bool(state.get("visual_quality_pending"))
    return {
        "schema_version": 1,
        "automatic_repairs": automatic_repairs,
        "degradations": degradations,
        "unresolved_risks": unresolved_risks,
        "final_choice": {
            "status": run_status,
            "strategy": selected_strategy,
            "passed": bool(final_critique.passed) if final_critique is not None else None,
            "terminal_failure": terminal,
            # 视觉质量未验证时成片只能算"结构/技术可用"：如实标记降级，不冒充完整通过。
            "degraded": bool(state.get("degraded_published") or run_status == "degraded" or visual_pending_state),
            "visual_quality": {
                "status": "pending" if visual_pending_state else "not_evaluated",
                "dimensions": pending_dimensions,
                "reason": _safe_report_text(str(state.get("visual_pending_reason") or ""))
                or "未接入真实视觉模型，视觉质量未评估",
            },
            "output_path": ""
            if not final_critique or not final_critique.passed
            else _safe_report_text(state.get("output_path") or state.get("video_path") or ""),
            "selected_video_candidate_id": str(state.get("selected_video_candidate_id") or ""),
            "candidate_selection": state.get("candidate_selection") or None,
        },
    }


def extract_final_report(report: CritiqueReport | dict[str, Any]) -> dict[str, Any]:
    """读取 ``critique_final`` 嵌入的结构化报告；兼容普通 dict。"""
    evidence = report.get("evidence") if isinstance(report, dict) else report.evidence
    for item in evidence or []:
        if isinstance(item, dict) and item.get("kind") == "final_report" and isinstance(item.get("report"), dict):
            return dict(item["report"])
    return {}


def _visual_pending_entries(critiques: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """收集视觉待审证据：带阶段、维度和镜头，供最终报告定位。"""

    entries: list[dict[str, Any]] = []
    dimension_keys = {item["key"] for item in _visual_dimension_specs()}
    for critique in critiques:
        stage = str(critique.get("stage") or "")
        for metric in critique.get("metrics") or []:
            if not isinstance(metric, dict):
                continue
            name = str(metric.get("name") or "")
            if name == "visual_quality_pending" and metric.get("passed") is None:
                entries.append(
                    {
                        "stage": stage,
                        "shot_ids": [],
                        "dimensions": [],
                        "detail": str(metric.get("detail") or "视觉质量保持待审"),
                    }
                )
            elif name in dimension_keys and metric.get("passed") is None:
                # 逐维度 pending：既不通过也不失败，必须原样进入未解决风险。
                entries.append(
                    {
                        "stage": stage,
                        "shot_ids": [],
                        "dimensions": [name],
                        "detail": str(metric.get("detail") or f"{name} 未评估（pending）"),
                    }
                )
        for issue in critique.get("issues") or []:
            if not isinstance(issue, dict):
                continue
            code = str(issue.get("code") or "")
            details = issue.get("details") if isinstance(issue.get("details"), dict) else {}
            if code == "visual_quality_pending":
                shot_ids = [str(item) for item in (details.get("shot_ids") or []) if item]
                entries.append(
                    {
                        "stage": str(details.get("stage") or stage),
                        "shot_ids": shot_ids,
                        "dimensions": [str(item) for item in (details.get("dimensions") or []) if item],
                        "detail": str(issue.get("message") or "视觉质量保持待审"),
                    }
                )
            elif code.startswith("visual_pending:"):
                shot_id = str(issue.get("shot_id") or "")
                entries.append(
                    {
                        "stage": str(details.get("stage") or stage),
                        "shot_ids": [shot_id] if shot_id else [],
                        "dimensions": [str(details.get("dimension") or code.split(":", 1)[1])],
                        "detail": str(issue.get("message") or "视觉维度待审"),
                    }
                )
    return entries


def _trace_shot_ids(trace: dict[str, Any], selected: dict[str, Any], critique: dict[str, Any]) -> list[str]:
    values: list[Any] = list(selected.get("shot_ids") or [])
    failure = trace.get("failure") or {}
    values.extend(critique.get("affected_shot_ids") or [])
    if failure.get("shot_id"):
        values.append(failure.get("shot_id"))
    result: list[str] = []
    for value in values:
        shot_id = str(value or "")
        if shot_id and shot_id not in result:
            result.append(shot_id)
    return result


def _trace_attempt(trace: dict[str, Any], history: list[dict[str, Any]], stage: str, strategy: str) -> int | None:
    if trace.get("attempt") is not None:
        return trace.get("attempt")
    matching = [
        item
        for item in history
        if str(item.get("stage") or "") == stage
        and str(item.get("strategy") or item.get("selected_strategy") or "") == strategy
    ]
    return len(matching) or None


def _candidate_seed(candidate: dict[str, Any]) -> int | None:
    value = candidate.get("seed")
    if value is None:
        changes = candidate.get("prompt_changes") or {}
        patches = changes.get("patches") if isinstance(changes, dict) else []
        for patch in patches or []:
            if isinstance(patch, dict) and patch.get("field") == "seed":
                raw = patch.get("value")
                if isinstance(raw, dict):
                    raw = raw.get("seed")
                try:
                    return int(raw) if raw is not None else None
                except (TypeError, ValueError):
                    return None
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_report_text(value: Any) -> str:
    """Expose only redacted, bounded diagnostic text in a user-facing report."""

    return redact(value, limit=512)


def _safe_report_value(value: Any) -> Any:
    """Recursively redact user-facing report values without changing their shape."""

    if isinstance(value, dict):
        return {str(key): _safe_report_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_report_value(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_report_value(item) for item in value]
    if isinstance(value, str):
        return _safe_report_text(value)
    return value


def _has_media_evidence(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    path = str(item.get("path") or item.get("video_path") or "")
    if not path:
        return False
    structural_passed = item.get("structural_passed")
    if structural_passed is None:
        structural = item.get("structural_validity") or item.get("structural_report") or {}
        if isinstance(structural, dict):
            structural_passed = structural.get("passed")
    if structural_passed is not True:
        return False
    return True


def _trace_has_candidate_evidence(
    trace: dict[str, Any], selected: dict[str, Any], artifacts: list[dict[str, Any]]
) -> bool:
    candidate_id = str(trace.get("selected_video_candidate_id") or selected.get("selected_video_candidate_id") or "")
    if candidate_id:
        return any(
            candidate_id == str(candidate.get("candidate_id") or "") and _has_media_evidence(candidate)
            for item in artifacts
            for candidate in item.get("video_candidates") or []
            if isinstance(candidate, dict)
        )
    return any(
        _has_media_evidence(item) and str(item.get("status") or "") in {"succeeded", "degraded"} for item in artifacts
    )


def _recovery_after_state(artifacts: list[dict[str, Any]], shot_ids: list[str], stage: str) -> dict[str, Any]:
    relevant = [
        item
        for item in artifacts
        if (not shot_ids or str(item.get("shot_id") or "") in shot_ids)
        and (not stage or str(item.get("stage") or "") in {stage, ""})
    ]
    latest = relevant[-1] if relevant else None
    if latest is None:
        return {"status": "pending", "outcome": "attempted", "stage": stage, "shot_ids": shot_ids}
    status = str(latest.get("status") or "unknown")
    return {
        "status": status,
        "outcome": "succeeded"
        if status in {"succeeded", "degraded"}
        else ("failed" if status in {"failed", "needs_review"} else "attempted"),
        "stage": str(latest.get("stage") or stage),
        "shot_ids": shot_ids,
        "path_available": _has_media_evidence(latest),
        "output_fingerprint": str(latest.get("output_fingerprint") or ""),
    }


def _unresolved_risks(
    state: dict[str, Any],
    *,
    critiques: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    final_critique: CritiqueReport | None,
    visual_pending: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    # 视觉待审是"未验证"，不是"失败"：severity=pending，并逐镜头/逐维度列出，
    # 保证 final_review 能把这条风险定位到具体镜头与阶段。
    pending_fallback_shots = sorted(
        {
            str(item.get("shot_id") or "")
            for item in artifacts
            if item.get("shot_id")
            and str(item.get("stage") or "") in {StageName.VIDEO_GENERATION.value, StageName.IMAGE_GENERATION.value, ""}
        }
    )
    for item in visual_pending:
        shot_ids = [str(value) for value in (item.get("shot_ids") or []) if value] or pending_fallback_shots
        risks.append(
            {
                "code": "visual_quality_pending",
                "severity": "pending",
                "stage": item.get("stage", "") or StageName.VIDEO_REVIEW.value,
                "shot_ids": shot_ids,
                "dimensions": [str(value) for value in (item.get("dimensions") or []) if value],
                "message": item.get("detail", "视觉质量保持待审"),
            }
        )
    source_issues: list[dict[str, Any]] = []
    if final_critique is not None:
        source_issues.extend(item.model_dump(mode="json") for item in final_critique.issues)
    else:
        for critique in critiques:
            source_issues.extend(item for item in (critique.get("issues") or []) if isinstance(item, dict))
    seen: set[tuple[str, str, str]] = set()
    for issue in source_issues:
        code = str(issue.get("code") or "")
        # 视觉待审已在上面按维度展开，避免重复条目。
        if code == "visual_quality_pending" or code.startswith("visual_pending:"):
            continue
        key = (code, str(issue.get("shot_id") or ""), str(issue.get("message") or ""))
        if key in seen:
            continue
        seen.add(key)
        risks.append(
            {
                "code": code,
                "severity": str(issue.get("severity") or "warning"),
                "stage": _issue_stage(issue, critiques),
                "shot_ids": [str(issue.get("shot_id"))] if issue.get("shot_id") else [],
                "message": _safe_report_text(str(issue.get("message") or "")),
                "recommendation": _safe_report_text(str(issue.get("recommendation") or "")),
            }
        )
    for item in artifacts:
        if str(item.get("status") or "") not in {"failed", "needs_review"} and not item.get("failure"):
            continue
        failure = item.get("failure") or {}
        risks.append(
            {
                "code": str(failure.get("kind") or "artifact_failed"),
                "severity": "error",
                "stage": str(item.get("stage") or ""),
                "shot_ids": [str(item.get("shot_id") or "")] if item.get("shot_id") else [],
                "message": _safe_report_text(str(failure.get("message") or "镜头产物失败")),
                "recommendation": "按恢复追踪补拍；预算耗尽时保留降级或终止记录",
            }
        )
    run_status = str(state.get("run_status") or "")
    selected = str(state.get("selected_strategy") or "")
    if run_status == "failed" or selected == RecoveryStrategy.TERMINAL_FAILURE.value:
        errors = [str(item) for item in (state.get("errors") or []) if item]
        risks.append(
            {
                "code": "terminal_failure",
                "severity": "error",
                "stage": str(state.get("current_stage") or StageName.FINAL_REVIEW.value),
                "shot_ids": [],
                "message": _safe_report_text(
                    str(state.get("human_reason") or (errors[-1] if errors else "自动恢复终止"))
                ),
                "recommendation": "检查 DecisionTrace、预算和候选历史后重新运行或修复输入",
            }
        )
    return risks


def _issue_stage(issue: dict[str, Any], critiques: list[dict[str, Any]]) -> str:
    explicit = str(issue.get("stage") or "")
    if explicit:
        return explicit
    code = str(issue.get("code") or "")
    for critique in critiques:
        if any(
            str(item.get("code") or "") == code for item in (critique.get("issues") or []) if isinstance(item, dict)
        ):
            return str(critique.get("stage") or "")
    if code.startswith("image_"):
        return StageName.IMAGE_GENERATION.value
    if code.startswith("audio_"):
        return StageName.AUDIO_PRODUCTION.value
    if code.startswith("video_"):
        return StageName.VIDEO_REVIEW.value
    if code.startswith("render_") or code == "incomplete_timeline":
        return StageName.EDIT_COMPOSITION.value
    return StageName.FINAL_REVIEW.value


def critique_llm_failure(exc: BaseException | str, *, stage: StageName | str) -> CritiqueReport:
    message = str(exc)
    truncated = bool(
        re.search(
            r"finish_reason\s*=\s*length|输出超过最大长度|输出疑似达到上限|输出被截断|被截断|llm_output_truncated",
            message,
            re.I,
        )
    )
    invalid = not truncated and bool(re.search(r"json|schema|结构无法解析|输出无法使用|invalid", message, re.I))
    if truncated:
        kind = FailureKind.LLM_OUTPUT_TRUNCATED
        issues = [
            CriticIssue(
                code="llm_output_truncated",
                severity="error",
                message="模型输出超过最大长度并被截断",
                recommendation="提高输出额度或按场次分段解析；改提示词无法修复长度截断，仍失败则切换 Provider",
            )
        ]
    else:
        kind = FailureKind.LLM_INVALID_OUTPUT if invalid else FailureKind.UNKNOWN
        issues = [
            CriticIssue(
                code="llm_invalid_output" if invalid else "llm_failed",
                severity="error",
                message="LLM 输出无法安全使用" if invalid else "LLM 调用失败",
                recommendation="收紧 JSON schema、降低温度并重试；仍失败则切换 Provider 或转人工",
            )
        ]
    report = _finalize(
        stage,
        passed=False,
        score=0.0,
        metrics=[QualityMetric(name="llm_output_usable", passed=False)],
        issues=issues,
        source="exception",
    )
    # 异常路径的 failure_kind 必须与 issue 一致：截断按确定性失败处理（只允许换
    # Provider），非解析类异常按 UNKNOWN 分类，让决策节点基于 Provider/网络证据
    # 而不是盲目重试。
    return report.model_copy(update={"failure_kind": kind})


def split_dialogue(text: str, *, max_chars: int = MAX_DIALOGUE_CHARS_PER_SHOT // 2) -> list[str]:
    """按标点/长度拆对白，保留可配音的短句。"""
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = [part.strip() for part in re.split(r"(?<=[。！？!?；;])", raw) if part.strip()]
    result: list[str] = []
    for part in parts:
        while len(part) > max_chars:
            cut = part.rfind("，", 0, max_chars)
            if cut < max_chars // 2:
                cut = max_chars
            result.append(part[:cut].strip())
            part = part[cut:].lstrip("，, ")
        if part:
            result.append(part)
    return result or [raw]


def _finalize(
    stage: StageName | str,
    *,
    passed: bool,
    score: float,
    metrics: list[QualityMetric],
    issues: list[CriticIssue],
    source: str = "deterministic",
) -> CritiqueReport:
    """统一补全反思字段：失败分类、可恢复性、证据、受影响镜头和建议策略。"""

    kind = _primary_failure_kind(issues, passed=passed)
    return CritiqueReport(
        stage=stage,
        passed=passed,
        score=round(max(0.0, min(1.0, score)), 3),
        metrics=metrics,
        issues=issues,
        evidence=_build_evidence(metrics, issues),
        failure_kind=kind,
        recoverable=kind is None or kind not in NON_RECOVERABLE_FAILURES,
        proposed_changes=_proposals(issues),
        affected_shot_ids=_affected_shot_ids(issues),
        recommended_strategy=recommended_strategy_for(kind),
        source=source,
    )


def _primary_failure_kind(issues: list[CriticIssue], *, passed: bool) -> FailureKind | None:
    for issue in issues:
        if issue.severity == "error":
            return classify_issue_code(issue.code)
    # 没有硬错误但未通过：按质量不达标分类（分数低于阈值等）。
    if not passed:
        return FailureKind.QUALITY_BELOW_THRESHOLD
    # Provider 能力类 warning 即使未阻断流程也标记分类，供决策节点切换/降级参考。
    for issue in issues:
        if classify_issue_code(issue.code) in {
            FailureKind.PROVIDER_REFERENCE_UNSUPPORTED,
            FailureKind.PROVIDER_CAPABILITY_MISMATCH,
        }:
            return classify_issue_code(issue.code)
    return None


def _affected_shot_ids(issues: list[CriticIssue]) -> list[str]:
    affected: list[str] = []
    for issue in issues:
        shot_id = str(issue.shot_id or "")
        if shot_id and shot_id not in affected:
            affected.append(shot_id)
    return affected


def _build_evidence(metrics: list[QualityMetric], issues: list[CriticIssue]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for metric in metrics:
        entry: dict[str, Any] = {"kind": "metric", "name": metric.name}
        if metric.value is not None:
            entry["value"] = metric.value
        if metric.threshold is not None:
            entry["threshold"] = metric.threshold
        if metric.passed is not None:
            entry["passed"] = metric.passed
        if metric.detail:
            entry["detail"] = metric.detail
        evidence.append(entry)
    for issue in issues:
        if issue.severity == "error" or issue.details:
            entry = {"kind": "issue", "code": issue.code, "severity": issue.severity, "message": issue.message}
            if issue.shot_id:
                entry["shot_id"] = issue.shot_id
            if issue.details:
                entry["details"] = issue.details
            evidence.append(entry)
    return evidence


def _proposals(issues: Iterable[CriticIssue]) -> list[str]:
    return [f"{item.code}: {item.recommendation}" for item in issues if item.recommendation]


def _validate_image(path: str) -> dict[str, Any]:
    try:
        from services.structural_validation import validate_image_file

        return validate_image_file(str(path or ""))
    except Exception as exc:
        return {"passed": False, "issues": [str(exc)]}


def _validate_video(
    path: str,
    *,
    expected_duration_s: float | None = None,
    expected_aspect_ratio: float | None = None,
    audio_duration_s: float | None = None,
    tail_frame_path: str | None = None,
    first_frame_path: str | None = None,
    expect_audio: bool | None = None,
) -> dict[str, Any]:
    try:
        from services.structural_validation import validate_video_sync

        return validate_video_sync(
            str(path or ""),
            expected_duration_s=expected_duration_s,
            expected_aspect_ratio=expected_aspect_ratio,
            audio_duration_s=audio_duration_s,
            tail_frame_path=tail_frame_path,
            first_frame_path=first_frame_path,
            expect_audio=expect_audio,
        )
    except Exception as exc:
        categories = {
            "structural_validity": {
                "passed": False,
                "issues": [
                    {
                        "code": "video_check_error",
                        "message": f"检查器异常: {exc}",
                        "recommendation": "重跑检查；持续失败时转人工核实该镜头",
                    }
                ],
            },
            "technical_quality": {"passed": None, "issues": [], "skipped": ["all"]},
            "visual_quality_pending": {"status": "pending", "passed": None, "reason": "检查器异常，视觉质量未知"},
        }
        return {
            "kind": "video",
            "path": str(path or ""),
            "passed": False,
            "issues": [str(exc)],
            "categories": categories,
        }


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _probe_audio_duration(path: Any) -> float | None:
    """读取外部配音时长（秒）；缺失/不可读返回 None（该维度记 skipped）。"""

    if not path:
        return None
    try:
        from services.structural_validation import probe_media_duration_sync

        return probe_media_duration_sync(str(path))
    except Exception:
        return None


__all__ = [
    "MAX_DIALOGUE_CHARS_PER_SHOT",
    "classify_issue_code",
    "critique_assets",
    "critique_audio",
    "critique_compose",
    "critique_director",
    "critique_final",
    "critique_images",
    "critique_llm_failure",
    "critique_storyboard",
    "critique_videos",
    "build_final_report",
    "extract_final_report",
    "recommended_strategy_for",
    "split_dialogue",
]
