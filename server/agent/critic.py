"""Critic/Reviewer：把生成结果转成可执行的修改意见。

这里优先使用确定性指标（结构、时长、对白、版本和 Provider 能力），因此离线、
测试和供应商故障时仍能给出具体修改；LLM/视觉模型可作为增强，不影响主流程。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .contracts import CriticIssue, CritiqueReport, FailureKind, QualityMetric, StageName

MAX_DIALOGUE_CHARS_PER_SHOT = 180
MAX_DIALOGUE_CHARS_PER_SECOND = 8.0
MIN_SHOT_SECONDS = 2.0
MAX_SHOT_SECONDS = 5.0
COMPLEX_ACTION_WORDS = ("追逐", "打斗", "翻滚", "连续", "然后", "接着", "同时", "突然", "爆炸", "奔跑", "转身", "跳")


def critique_director(state: dict[str, Any]) -> CritiqueReport:
    """导演规划的结构/覆盖检查。"""
    characters = list(state.get("characters") or [])
    scenes = list(state.get("script_scenes") or [])
    logic_issues = list(state.get("logic_issues") or [])
    issues: list[CriticIssue] = []
    if not characters:
        issues.append(CriticIssue(code="missing_characters", severity="error", message="导演规划缺少角色", recommendation="要求模型输出至少一名可执行角色"))
    if not scenes:
        issues.append(CriticIssue(code="missing_scenes", severity="error", message="导演规划缺少场景", recommendation="要求模型输出至少一个明确场景"))
    for item in logic_issues[:10]:
        issues.append(CriticIssue(code="logic_issue", severity="warning", message=str(item), recommendation="在分镜中补足因果/时间线说明或转人工确认"))
    metrics = [
        QualityMetric(name="schema_valid", passed=bool(characters and scenes)),
        QualityMetric(name="character_coverage", value=len(characters), threshold=1, passed=bool(characters)),
        QualityMetric(name="scene_coverage", value=len(scenes), threshold=1, passed=bool(scenes)),
        QualityMetric(name="logic_issue_count", value=len(logic_issues), threshold=0, passed=not logic_issues),
    ]
    score = max(0.0, 1.0 - 0.25 * sum(1 for item in issues if item.severity == "error") - 0.08 * len(logic_issues))
    return CritiqueReport(
        stage=StageName.DIRECTOR_PLANNING,
        passed=not any(item.severity == "error" for item in issues),
        score=round(score, 3),
        metrics=metrics,
        issues=issues,
        proposed_changes=_proposals(issues),
    )


def critique_storyboard(state: dict[str, Any]) -> CritiqueReport:
    shots = list(state.get("shots") or [])
    issues: list[CriticIssue] = []
    metrics: list[QualityMetric] = []
    shot_count = len(shots)
    metrics.append(QualityMetric(name="shot_count", value=shot_count, threshold=1, passed=shot_count > 0))
    if not shots:
        issues.append(CriticIssue(code="empty_storyboard", severity="error", message="没有可用镜头", recommendation="重新解析剧本并要求至少一个镜头"))
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
    return CritiqueReport(
        stage=StageName.STORYBOARD_DESIGN,
        passed=valid_count == 0 and score >= 0.55,
        score=round(score, 3),
        metrics=metrics,
        issues=issues,
        proposed_changes=_proposals(issues),
    )


def critique_assets(state: dict[str, Any], *, reference_supported: bool = True) -> CritiqueReport:
    characters = list(state.get("characters") or [])
    scenes = list(state.get("script_scenes") or [])
    issues: list[CriticIssue] = []
    missing_chars = [str(item.get("name") or item.get("id") or "?") for item in characters if not item.get("reference_images")]
    missing_scenes = [str(item.get("id") or item.get("name") or "?") for item in scenes if not (item.get("baseline_image_path") or item.get("reference_images"))]
    if missing_chars:
        issues.append(CriticIssue(code="missing_character_reference", severity="warning", message=f"缺少角色参考: {', '.join(missing_chars[:8])}", recommendation="补生成角色三视图或替换为可靠参考"))
    if missing_scenes:
        issues.append(CriticIssue(code="missing_scene_reference", severity="warning", message=f"缺少场景基准图: {', '.join(missing_scenes[:8])}", recommendation="补生成场景基准图或降低一致性要求"))
    if not reference_supported and (characters or scenes):
        issues.append(CriticIssue(code="provider_reference_unsupported", severity="warning", message="当前图像 Provider 不支持参考图", recommendation="切换支持参考图的 Provider，或明确降级为纯文本生成"))
    metrics = [
        QualityMetric(name="character_reference_coverage", value=round(1 - len(missing_chars) / max(1, len(characters)), 3), passed=not missing_chars),
        QualityMetric(name="scene_reference_coverage", value=round(1 - len(missing_scenes) / max(1, len(scenes)), 3), passed=not missing_scenes),
        QualityMetric(name="reference_compatibility", passed=reference_supported or not (characters or scenes)),
    ]
    score = sum(1.0 for item in metrics if item.passed) / max(1, len(metrics))
    return CritiqueReport(stage=StageName.ASSET_PREPARATION, passed=not issues or all(item.severity != "error" for item in issues), score=round(score, 3), metrics=metrics, issues=issues, proposed_changes=_proposals(issues))


def critique_images(shot_artifacts: Iterable[dict[str, Any]]) -> CritiqueReport:
    artifacts = list(shot_artifacts)
    issues: list[CriticIssue] = []
    valid = 0
    scores: list[float] = []
    for item in artifacts:
        shot_id = str(item.get("shot_id") or "")
        path = str(item.get("path") or item.get("storyboard_path") or item.get("image_path") or "")
        structural = _validate_image(path)
        if not structural.get("passed"):
            issues.append(CriticIssue(code="image_invalid", severity="error", message=f"镜头 {shot_id or '?'} 图片结构不合格", shot_id=shot_id, recommendation="只重生成该镜头，必要时替换参考图或降分辨率", details=structural))
        else:
            valid += 1
        score = float(item.get("score") or (1.0 if structural.get("passed") else 0.0))
        scores.append(max(0.0, min(1.0, score)))
        if item.get("failure"):
            issues.append(CriticIssue(code="image_generation_failure", severity="error", message=f"镜头 {shot_id or '?'} 生成失败", shot_id=shot_id, recommendation="保留成功镜头，只重算失败镜头", details=dict(item.get("failure") or {})))
    metric_score = round(sum(scores) / max(1, len(scores)), 3) if scores else 0.0
    metrics = [QualityMetric(name="image_valid", value=valid, threshold=len(artifacts), passed=valid == len(artifacts)), QualityMetric(name="candidate_score", value=metric_score, threshold=0.72, passed=metric_score >= 0.72)]
    return CritiqueReport(stage=StageName.IMAGE_GENERATION, passed=valid == len(artifacts) and not issues, score=metric_score, metrics=metrics, issues=issues, proposed_changes=_proposals(issues))


def critique_videos(shot_artifacts: Iterable[dict[str, Any]]) -> CritiqueReport:
    artifacts = list(shot_artifacts)
    issues: list[CriticIssue] = []
    valid = 0
    for item in artifacts:
        shot_id = str(item.get("shot_id") or "")
        path = str(item.get("path") or item.get("video_path") or "")
        structural = _validate_video(path)
        if not structural.get("passed"):
            issues.append(CriticIssue(code="video_invalid", severity="error", message=f"镜头 {shot_id or '?'} 视频不可用", shot_id=shot_id, recommendation="只补拍失败镜头，可先降分辨率或切换 Provider", details=structural))
        else:
            valid += 1
        if item.get("failure"):
            issues.append(CriticIssue(code="video_generation_failure", severity="error", message=f"镜头 {shot_id or '?'} 视频生成失败", shot_id=shot_id, recommendation="仅重生成失败镜头，不回滚成功镜头", details=dict(item.get("failure") or {})))
    score = valid / max(1, len(artifacts))
    metrics = [QualityMetric(name="video_valid", value=valid, threshold=len(artifacts), passed=valid == len(artifacts)), QualityMetric(name="duration_match", passed=not issues, detail="结构检查包含视频轨和时长")]
    return CritiqueReport(stage=StageName.VIDEO_GENERATION, passed=not issues and valid == len(artifacts), score=round(score, 3), metrics=metrics, issues=issues, proposed_changes=_proposals(issues))


def critique_audio(shots: Iterable[dict[str, Any]], audio_artifacts: Iterable[dict[str, Any]] | None = None) -> CritiqueReport:
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
            issues.append(CriticIssue(code="dialogue_too_long", severity="error", message=f"镜头 {shot_id or '?'} 对白过长，无法稳定配音", shot_id=shot_id, recommendation="拆句、拆镜头或转人工改写台词", details={"length": len(dialogue), "max": MAX_DIALOGUE_CHARS_PER_SHOT}))
        if dialogue and not path:
            issues.append(CriticIssue(code="audio_missing", severity="error", message=f"镜头 {shot_id or '?'} 有对白但没有配音", shot_id=shot_id, recommendation="只重生成该镜头音频，复用其它成功音频"))
        elif path or not dialogue:
            ready += 1
    score = ready / max(1, len(shot_list))
    metrics = [QualityMetric(name="dialogue_length", passed=not any(item.code == "dialogue_too_long" for item in issues)), QualityMetric(name="tts_valid", value=ready, threshold=len(shot_list), passed=ready == len(shot_list))]
    return CritiqueReport(stage=StageName.AUDIO_PRODUCTION, passed=not issues and ready == len(shot_list), score=round(score, 3), metrics=metrics, issues=issues, proposed_changes=_proposals(issues))


def critique_compose(project_id: str, shots: Iterable[dict[str, Any]], output_path: str = "") -> CritiqueReport:
    shot_list = list(shots)
    missing = [str(item.get("shot_id") or item.get("id") or "?") for item in shot_list if not item.get("video_path")]
    issues: list[CriticIssue] = []
    if missing:
        issues.append(CriticIssue(code="incomplete_timeline", severity="warning", message=f"有 {len(missing)} 个镜头没有视频，无法无损合成", recommendation="明确跳过并降级，或只补拍缺失镜头", details={"shot_ids": missing[:20]}))
    if not output_path:
        issues.append(CriticIssue(code="render_missing", severity="error", message="没有成片输出路径", recommendation="重新剪辑合成或转人工检查媒体清单"))
    score = max(0.0, 1.0 - 0.2 * len(missing)) if output_path else 0.0
    metrics = [QualityMetric(name="shot_completeness", value=len(shot_list) - len(missing), threshold=len(shot_list), passed=not missing), QualityMetric(name="render_valid", passed=bool(output_path))]
    return CritiqueReport(stage=StageName.EDIT_COMPOSITION, passed=bool(output_path) and not missing, score=round(score, 3), metrics=metrics, issues=issues, proposed_changes=_proposals(issues))


def critique_final(state: dict[str, Any]) -> CritiqueReport:
    issues: list[CriticIssue] = []
    shots = list(state.get("shot_artifacts") or state.get("shots") or [])
    failed = [item for item in shots if item.get("failure") or item.get("status") in {"failed", "needs_review"}]
    if failed:
        issues.append(CriticIssue(code="degraded_shots", severity="warning", message=f"成片包含 {len(failed)} 个降级或待审核镜头", recommendation="只补拍失败镜头或在人工确认后发布"))
    if state.get("human_feedback"):
        issues.append(CriticIssue(code="human_feedback", severity="info", message="收到人工反馈", recommendation="将反馈转为局部 Prompt/镜头修改，不重跑已成功部分"))
    if state.get("run_status") == "waiting_human":
        issues.append(CriticIssue(code="human_gate", severity="info", message="流程停在人工卡点", recommendation="人工确认后从检查点续跑"))
    score = max(0.0, 1.0 - 0.16 * len(failed))
    metrics = [QualityMetric(name="overall_score", value=round(score, 3), threshold=float(state.get("quality_threshold") or 0.72), passed=score >= float(state.get("quality_threshold") or 0.72)), QualityMetric(name="human_gate", passed=state.get("run_status") != "waiting_human")]
    return CritiqueReport(stage=StageName.FINAL_REVIEW, passed=score >= float(state.get("quality_threshold") or 0.72) and not failed, score=round(score, 3), metrics=metrics, issues=issues, proposed_changes=_proposals(issues))


def critique_llm_failure(exc: BaseException | str, *, stage: StageName | str) -> CritiqueReport:
    message = str(exc)
    invalid = bool(re.search(r"json|schema|结构无法解析|输出无法使用|invalid", message, re.I))
    issues = [CriticIssue(code="llm_invalid_output" if invalid else "llm_failed", severity="error", message="LLM 输出无法安全使用" if invalid else "LLM 调用失败", recommendation="收紧 JSON schema、降低温度并重试；仍失败则切换 Provider 或转人工")]
    return CritiqueReport(stage=stage, passed=False, score=0.0, issues=issues, proposed_changes=["将输出限制为 JSON 对象并显式列出必需字段", "把温度降到 0.2 以下并要求不要输出 Markdown"])


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


def _proposals(issues: Iterable[CriticIssue]) -> list[str]:
    return [f"{item.code}: {item.recommendation}" for item in issues if item.recommendation]


def _validate_image(path: str) -> dict[str, Any]:
    try:
        from services.structural_validation import validate_image_file

        return validate_image_file(str(path or ""))
    except Exception as exc:
        return {"passed": False, "issues": [str(exc)]}


def _validate_video(path: str) -> dict[str, Any]:
    try:
        from services.structural_validation import validate_video_sync

        return validate_video_sync(str(path or ""))
    except Exception as exc:
        return {"passed": False, "issues": [str(exc)]}


__all__ = [
    "MAX_DIALOGUE_CHARS_PER_SHOT",
    "critique_assets",
    "critique_audio",
    "critique_compose",
    "critique_director",
    "critique_final",
    "critique_images",
    "critique_llm_failure",
    "critique_storyboard",
    "critique_videos",
    "split_dialogue",
]
