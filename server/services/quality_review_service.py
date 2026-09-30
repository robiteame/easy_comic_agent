"""QualityReviewService：自动模式的真实质量审核门禁。

与结构检查（``services.structural_validation``）的分工：
- StructuralCheck 只回答「产物文件是否可用」（存在、可解码、尺寸达标），
  永远不代表「质量通过」；
- 本服务回答「画面/视频是否真的符合镜头要求」，对每个镜头输出结构化
  评分与问题列表，覆盖语义还原、角色身份、服装发型、构图、伪影，
  以及视频镜头的运动连贯性、镜头衔接、口型、音画同步与音频清晰度。

诚实性铁律：
- 审核能力未配置（VLM 缺失 / embedding 缺失 / ffmpeg 缺失）时对应维度
  标记 ``unsupported``，整体裁决可以是 ``unsupported``，绝不返回 passed；
- Provider 调用失败标记 ``error``（fail-closed），绝不静默放行；
- 按降级策略（strict/lenient）放行的未检测项必须置 ``degraded`` 并在
  日志与界面如实展示。
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from config import settings
from db import SessionLocal
from models import Character, QualityReview, Shot
from services.quality_review_providers import (
    VLMJudge,
    analyze_audio_clarity,
    extract_video_frames,
    probe_media_streams,
)

logger = logging.getLogger(__name__)

STAGE_STORYBOARD = "storyboard"
STAGE_VIDEO = "video"

SHOT_TYPE_LABELS = {
    "wide": "远景",
    "medium": "中景",
    "close-up": "近景",
    "extreme_close": "特写",
}

QUALITY_FIX_TAG = "【质量修正】"


@dataclass(frozen=True)
class DimensionSpec:
    key: str
    label: str
    stage: str
    weight: float
    min_score: float  # 单维度下限：低于该值即使均分达标也不通过
    fix_template: str  # 修正指令模板（占位符在 build_fix_plan 填充）


DIMENSIONS: dict[str, DimensionSpec] = {
    spec.key: spec
    for spec in (
        # --- 故事板（图片）阶段 ---
        DimensionSpec("scene_match", "场景还原（scene_description）", STAGE_STORYBOARD, 1.2, 0.6,
                      "画面必须严格呈现场景：{scene_description}"),
        DimensionSpec("action_match", "动作还原（character_action）", STAGE_STORYBOARD, 1.0, 0.6,
                      "人物动作必须符合：{character_action}"),
        DimensionSpec("shot_type_match", "景别（shot_type）", STAGE_STORYBOARD, 0.8, 0.6,
                      "景别必须是「{shot_type_label}」，不得改变取景范围"),
        DimensionSpec("camera_angle_match", "机位角度（camera_angle）", STAGE_STORYBOARD, 0.8, 0.6,
                      "拍摄角度必须是「{camera_angle}」视角"),
        DimensionSpec("character_identity", "角色身份一致性", STAGE_STORYBOARD, 1.5, 0.7,
                      "角色{character_names}的脸型五官必须与角色参考图完全一致，严禁改变身份特征"),
        DimensionSpec("appearance_consistency", "服装发型一致性", STAGE_STORYBOARD, 1.2, 0.65,
                      "角色服装与发型必须与角色参考图设定完全一致，不得漂移"),
        DimensionSpec("composition", "构图与主体", STAGE_STORYBOARD, 1.0, 0.6,
                      "主体完整、构图清晰，关键角色与道具不得被裁切或移出画面"),
        DimensionSpec("artifacts", "伪影与画面缺陷", STAGE_STORYBOARD, 1.5, 0.7,
                      "避免肢体畸形、手部错误、文字乱码、重复元素、边缘截断等伪影"),
        # --- 视频阶段 ---
        DimensionSpec("motion_coherence", "运动连贯性", STAGE_VIDEO, 1.2, 0.6,
                      "动作连贯流畅，避免瞬移、抖动与画面闪烁"),
        DimensionSpec("shot_continuity", "镜头间衔接", STAGE_VIDEO, 1.0, 0.6,
                      "与上一镜头保持场景与角色状态连续，避免突兀跳变"),
        DimensionSpec("lip_sync", "对白与口型", STAGE_VIDEO, 0.8, 0.6,
                      "角色口型必须与台词「{dialogue}」匹配"),
        DimensionSpec("audio_video_sync", "音画同步", STAGE_VIDEO, 0.8, 0.6,
                      "配音必须与画面对齐，音画时长一致"),
        DimensionSpec("audio_clarity", "音频清晰度", STAGE_VIDEO, 1.0, 0.6,
                      "配音清晰可辨，避免长时间静音、音量过低或破音"),
    )
}

STORYBOARD_VLM_KEYS = [
    "scene_match",
    "action_match",
    "shot_type_match",
    "camera_angle_match",
    "character_identity",
    "appearance_consistency",
    "composition",
    "artifacts",
]
VIDEO_VLM_KEYS = ["motion_coherence", "shot_continuity", "lip_sync"]


@dataclass
class DimensionResult:
    key: str
    label: str
    status: str  # scored / unsupported / skipped / error
    score: float | None = None
    weight: float = 1.0
    issues: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    provider: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "score": None if self.score is None else round(self.score, 3),
            "weight": self.weight,
            "issues": self.issues,
            "evidence": self.evidence,
            "provider": self.provider,
        }


@dataclass
class ShotReview:
    shot_id: str
    project_id: str
    stage: str
    attempt: int = 1
    shot_version: int = 1
    target_path: str = ""
    verdict: str = "failed"  # passed / failed / unsupported / error
    passed: bool = False
    overall_score: float = 0.0
    dimensions: list[DimensionResult] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggestion: str = ""
    degraded: bool = False
    gate_policy: str = ""
    fix: dict = field(default_factory=dict)  # {directives, summary}
    row_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.row_id,
            "shot_id": self.shot_id,
            "project_id": self.project_id,
            "stage": self.stage,
            "attempt": self.attempt,
            "shot_version": self.shot_version,
            "target_path": self.target_path,
            "verdict": self.verdict,
            "passed": self.passed,
            "overall_score": round(self.overall_score, 3),
            "degraded": self.degraded,
            "dimensions": [dimension.to_dict() for dimension in self.dimensions],
            "issues": self.issues,
            "unsupported_dimensions": [
                dimension.label for dimension in self.dimensions if dimension.status == "unsupported"
            ],
            "suggestion": self.suggestion,
            "prompt_fix": self.fix,
            "gate_policy": self.gate_policy,
        }


_STORYBOARD_SYSTEM_PROMPT = """你是漫画分镜与短视频的质量审核员。只依据画面可见事实逐项评分，给出可核验的证据，不臆测、不美化。
输出严格 JSON（不要 markdown 围栏），形如：
{"dimensions": {"<维度key>": {"score": <0-10 整数>, "issues": ["具体问题"], "evidence": ["画面中可见的依据"]}}, "suggestion": "一句修正建议"}
维度 key 与含义（全部必填，缺一不可）：
- scene_match：画面是否呈现 scene_description 描述的场景与环境；
- action_match：人物动作/姿态是否符合 character_action；
- shot_type_match：景别是否符合 shot_type（远景/中景/近景/特写）；
- camera_angle_match：机位角度是否符合 camera_angle（正面/侧面/俯视/仰视）；
- character_identity：出镜角色的脸型五官是否与角色参考图一致（身份漂移必须打低分）；
- appearance_consistency：服装、发型是否与角色参考图一致（不得漂移）；
- composition：构图是否有效——主体完整、清晰、未被裁切，画面要素与镜头描述对应；
- artifacts：是否存在伪影——肢体畸形、手指错误、文字乱码、重复元素、边缘截断、闪烁残影。
评分标准：10=完全符合；7-9=基本符合有小瑕疵；4-6=明显不符；0-3=严重错误。"""

_VIDEO_SYSTEM_PROMPT = """你是短视频镜头的质量审核员。只依据给出的视频帧与客观数据评审，输出严格 JSON（不要 markdown 围栏），形如：
{"dimensions": {"<维度key>": {"score": <0-10 整数>, "issues": ["具体问题"], "evidence": ["可见依据"]}}, "suggestion": "一句修正建议"}
维度 key 与含义（全部必填）：
- motion_coherence：运动连贯性——动作是否流畅自然，有无瞬移、抖动、画面闪烁、主体变形；
- shot_continuity：与上一镜头（如提供衔接参考帧）是否连续——场景、光线、角色状态有无突兀跳变；
- lip_sync：口型与台词是否匹配——说话镜头口型应对应台词内容，非说话镜头应无张口说话痕迹。
评分标准：10=完全符合；7-9=基本符合有小瑕疵；4-6=明显不符；0-3=严重错误。"""


def _parse_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _json_list(raw: str | None) -> list:
    try:
        value = json.loads(raw or "[]")
        return value if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


class QualityReviewService:
    """镜头质量审核：评分、裁决、修正建议与历史落库。"""

    def __init__(self, vlm_judge: VLMJudge | None = None, identity_provider=None):
        self._vlm = vlm_judge or VLMJudge()
        self._identity = identity_provider  # 延迟创建：默认实例在首次使用时构造

    # --- 能力状态 --------------------------------------------------------

    def _identity_provider(self):
        if self._identity is None:
            from services.quality_review_providers import IdentityEmbeddingProvider

            self._identity = IdentityEmbeddingProvider()
        return self._identity

    def storyboard_capability(self) -> dict:
        vlm = self._vlm.capability()
        identity = self._identity_provider().capability()
        supported = vlm.supported
        return {
            "supported": supported,
            "reason": "" if supported else vlm.reason,
            "vlm": {"supported": vlm.supported, "reason": vlm.reason, "provider": vlm.provider},
            "identity_embedding": {
                "supported": identity.supported,
                "reason": identity.reason,
                "provider": identity.provider,
            },
        }

    def video_capability(self) -> dict:
        storyboard = self.storyboard_capability()
        return {
            **storyboard,
            "supported": storyboard["vlm"]["supported"],
            "reason": "" if storyboard["vlm"]["supported"] else storyboard["vlm"]["reason"],
        }

    def gate_snapshot(self) -> dict:
        return {
            "threshold": float(settings.QUALITY_REVIEW_PASS_SCORE),
            "policy": str(settings.QUALITY_DEGRADATION_POLICY),
            "storyboard_max_retries": int(settings.QUALITY_STORYBOARD_MAX_RETRIES),
            "video_max_retries": int(settings.QUALITY_VIDEO_MAX_RETRIES),
        }

    def capability_summary(self) -> dict:
        return {
            "storyboard": self.storyboard_capability(),
            "video": self.video_capability(),
            "gate": self.gate_snapshot(),
        }

    # --- 审核：故事板阶段 -------------------------------------------------

    async def review_storyboard_shot(self, shot_id: str) -> ShotReview:
        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if shot is None:
                raise ValueError(f"镜头不存在: {shot_id}")
            shot_fields = _shot_fields(shot)
            character_refs = _character_references(db, shot)
        finally:
            db.close()

        target = shot_fields["target_path"]
        review = ShotReview(
            shot_id=shot_id,
            project_id=shot_fields["project_id"],
            stage=STAGE_STORYBOARD,
            attempt=self._next_attempt(shot_id, STAGE_STORYBOARD),
            shot_version=shot_fields["version"],
            target_path=target,
        )

        vlm_capability = self._vlm.capability()
        has_characters = bool(shot_fields["character_names"])
        has_refs = bool(character_refs)
        vlm_result: dict | None = None
        vlm_error = ""
        if vlm_capability.supported:
            try:
                images = [target] + [ref["path"] for ref in character_refs]
                vlm_result = await self._vlm.judge(
                    _STORYBOARD_SYSTEM_PROMPT,
                    _storyboard_user_prompt(shot_fields, character_refs),
                    images,
                )
            except Exception as exc:
                vlm_error = str(exc)
                logger.warning("镜头 %s 故事板 VLM 评审失败: %s", shot_id, exc)

        dimensions: list[DimensionResult] = []
        for key in STORYBOARD_VLM_KEYS:
            spec = DIMENSIONS[key]
            if key in ("character_identity", "appearance_consistency"):
                if not has_characters:
                    dimensions.append(
                        DimensionResult(
                            key, spec.label, "skipped", weight=spec.weight,
                            issues=[], evidence={"reason": "无角色出镜，不适用"}, provider="-",
                        )
                    )
                    continue
                if not has_refs:
                    # 没有参考图时任何一方都无法核验身份/服装，如实标记未检测。
                    dimensions.append(
                        DimensionResult(
                            key, spec.label, "unsupported", weight=spec.weight,
                            issues=["缺少角色参考图，无法核验角色一致性（unsupported）"],
                            evidence={"reason": "角色参考图缺失"},
                            provider="-",
                        )
                    )
                    continue
            dimensions.append(
                self._vlm_dimension(key, spec, vlm_capability, vlm_result, vlm_error)
            )

        # 身份维度叠加 embedding 相似度证据（未配置时如实标注 unsupported 证据）。
        if has_characters and has_refs:
            identity_report = await self._identity_provider().similarity(target, character_refs)
            self._merge_identity_embedding_report(dimensions, identity_report)

        self._finalize(review, dimensions)
        await self._persist_and_notify(review)
        return review

    # --- 审核：视频阶段 ---------------------------------------------------

    async def review_video_shot(self, shot_id: str, previous_frame_path: str = "") -> ShotReview:
        db = SessionLocal()
        try:
            shot = db.query(Shot).filter(Shot.id == shot_id).first()
            if shot is None:
                raise ValueError(f"镜头不存在: {shot_id}")
            shot_fields = _shot_fields(shot)
        finally:
            db.close()

        target = shot_fields["video_path"]
        if not target:
            raise ValueError(f"镜头 {shot_id} 没有视频产物，无法审核")

        review = ShotReview(
            shot_id=shot_id,
            project_id=shot_fields["project_id"],
            stage=STAGE_VIDEO,
            attempt=self._next_attempt(shot_id, STAGE_VIDEO),
            shot_version=shot_fields["version"],
            target_path=target,
        )

        probe = await probe_media_streams(target)
        dimensions: list[DimensionResult] = []
        vlm_capability = self._vlm.capability()
        vlm_result: dict | None = None
        vlm_error = ""

        frames: list[str] = []
        if vlm_capability.supported:
            frames = await extract_video_frames(target, count=4)
            if not frames:
                vlm_error = "视频抽帧失败（ffmpeg 不可用或无视频轨）"
        if frames:
            images = frames + ([previous_frame_path] if previous_frame_path else [])
            try:
                vlm_result = await self._vlm.judge(
                    _VIDEO_SYSTEM_PROMPT,
                    _video_user_prompt(shot_fields, frames, previous_frame_path),
                    images,
                )
            except Exception as exc:
                vlm_error = str(exc)
                logger.warning("镜头 %s 视频 VLM 评审失败: %s", shot_id, exc)

        for key in VIDEO_VLM_KEYS:
            spec = DIMENSIONS[key]
            if key == "lip_sync" and not (shot_fields["dialogue"] or "").strip():
                dimensions.append(
                    DimensionResult(
                        spec.key, spec.label, "skipped", weight=spec.weight,
                        issues=[], evidence={"reason": "该镜头无台词，不适用"}, provider="-",
                    )
                )
                continue
            dimensions.append(self._vlm_dimension(key, spec, vlm_capability, vlm_result, vlm_error))

        dimensions.append(self._audio_sync_dimension(probe))
        dimensions.append(await self._audio_clarity_dimension(target, probe))

        self._finalize(review, dimensions)
        await self._persist_and_notify(review)
        return review

    # --- 维度组装 ---------------------------------------------------------

    def _vlm_dimension(
        self,
        key: str,
        spec: DimensionSpec,
        vlm_capability,
        vlm_result: dict | None,
        vlm_error: str,
    ) -> DimensionResult:
        if not vlm_capability.supported:
            return DimensionResult(
                key, spec.label, "unsupported", weight=spec.weight,
                issues=[f"{vlm_capability.reason}（unsupported）"],
                evidence={"reason": vlm_capability.reason}, provider="-",
            )
        if vlm_error or vlm_result is None:
            return DimensionResult(
                key, spec.label, "error", weight=spec.weight,
                issues=[f"VLM 评审调用失败: {vlm_error[:200]}"],
                evidence={"error": vlm_error[:500]}, provider=vlm_capability.provider,
            )
        entry = (vlm_result.get("dimensions") or {}).get(key)
        if not isinstance(entry, dict) or _parse_number(entry.get("score")) is None:
            return DimensionResult(
                key, spec.label, "error", weight=spec.weight,
                issues=["VLM 未返回该维度的有效评分（fail-closed）"],
                evidence={"returned": entry}, provider=vlm_capability.provider,
            )
        score = _clamp01(_parse_number(entry.get("score")) / 10.0)
        issues = [str(item) for item in (entry.get("issues") or []) if str(item).strip()]
        evidence = [str(item) for item in (entry.get("evidence") or []) if str(item).strip()]
        return DimensionResult(
            key, spec.label, "scored", score=score, weight=spec.weight,
            issues=issues, evidence={"vlm": evidence}, provider=vlm_capability.provider,
        )

    def _merge_identity_embedding_report(
        self, dimensions: list[DimensionResult], report
    ) -> None:
        identity = next((d for d in dimensions if d.key == "character_identity"), None)
        if identity is None:
            return
        if report.status == "unsupported":
            identity.evidence["identity_embedding"] = {"status": "unsupported", "reason": report.error}
            identity.issues.append(f"身份 embedding 未执行: {report.error}")
            return
        if report.status == "error":
            identity.evidence["identity_embedding"] = {"status": "error", "error": report.error}
            identity.issues.append(f"身份 embedding 调用失败: {report.error[:200]}")
            return
        threshold = float(settings.QUALITY_IDENTITY_SIMILARITY_THRESHOLD)
        identity.evidence["identity_embedding"] = {
            "status": "scored",
            "similarities": report.similarities,
            "threshold": threshold,
            "provider": report.provider,
        }
        if report.min_score is not None and report.min_score < threshold:
            identity.issues.append(
                f"与角色参考图的 embedding 相似度 {report.min_score:.2f} 低于阈值 {threshold}"
            )
        embedding_score = (
            _clamp01(report.min_score / threshold) if report.min_score is not None else None
        )
        if embedding_score is not None:
            identity.score = (
                min(identity.score, embedding_score) if identity.score is not None else embedding_score
            )
            providers = [p for p in (identity.provider, report.provider) if p and p != "-"]
            identity.provider = " + ".join(providers)

    def _audio_sync_dimension(self, probe) -> DimensionResult:
        spec = DIMENSIONS["audio_video_sync"]
        if probe.status == "unsupported":
            return DimensionResult(
                spec.key, spec.label, "unsupported", weight=spec.weight,
                issues=[probe.error], evidence={"reason": probe.error}, provider="-",
            )
        if probe.status == "error":
            return DimensionResult(
                spec.key, spec.label, "error", weight=spec.weight,
                issues=[probe.error], evidence={"error": probe.error}, provider="ffprobe",
            )
        if probe.status == "skipped":
            return DimensionResult(
                spec.key, spec.label, "skipped", weight=spec.weight,
                issues=probe.issues, evidence={"reason": probe.issues[0] if probe.issues else "不适用"},
                provider="-",
            )
        score = 1.0 if not probe.issues else 0.3
        return DimensionResult(
            spec.key, spec.label, "scored", score=score, weight=spec.weight,
            issues=probe.issues,
            evidence={
                "video_duration": probe.video_duration,
                "audio_duration": probe.audio_duration,
            },
            provider="ffprobe",
        )

    async def _audio_clarity_dimension(self, target: str, probe) -> DimensionResult:
        spec = DIMENSIONS["audio_clarity"]
        if probe.status in {"skipped", "unsupported"}:
            reason = probe.issues[0] if probe.issues else probe.error or "不适用"
            status = "skipped" if probe.status == "skipped" else "unsupported"
            return DimensionResult(
                spec.key, spec.label, status, weight=spec.weight,
                issues=[] if status == "skipped" else [reason],
                evidence={"reason": reason}, provider="-",
            )
        if probe.status == "error":
            return DimensionResult(
                spec.key, spec.label, "error", weight=spec.weight,
                issues=[probe.error], evidence={"error": probe.error}, provider="ffmpeg",
            )
        clarity = await analyze_audio_clarity(target)
        if clarity.status == "unsupported":
            return DimensionResult(
                spec.key, spec.label, "unsupported", weight=spec.weight,
                issues=[clarity.error], evidence={"reason": clarity.error}, provider="-",
            )
        if clarity.status == "error":
            return DimensionResult(
                spec.key, spec.label, "error", weight=spec.weight,
                issues=[clarity.error], evidence={"error": clarity.error}, provider="ffmpeg",
            )
        score = _clamp01(1.0 - 0.4 * len(clarity.issues))
        return DimensionResult(
            spec.key, spec.label, "scored", score=score, weight=spec.weight,
            issues=clarity.issues,
            evidence={
                "mean_volume_db": clarity.mean_volume_db,
                "max_volume_db": clarity.max_volume_db,
                "silence_ratio": clarity.silence_ratio,
            },
            provider="ffmpeg",
        )

    # --- 裁决 -------------------------------------------------------------

    def _finalize(self, review: ShotReview, dimensions: list[DimensionResult]) -> None:
        gate = self.gate_snapshot()
        threshold = gate["threshold"]
        policy = gate["policy"]
        review.dimensions = dimensions
        review.gate_policy = (
            f"threshold={threshold} policy={policy} "
            f"retries={gate['storyboard_max_retries'] if review.stage == STAGE_STORYBOARD else gate['video_max_retries']}"
        )

        scored = [d for d in dimensions if d.status == "scored" and d.score is not None]
        errored = [d for d in dimensions if d.status == "error"]
        unsupported = [d for d in dimensions if d.status == "unsupported"]
        all_issues: list[str] = []
        for dimension in dimensions:
            all_issues.extend(f"[{dimension.label}] {issue}" for issue in dimension.issues)
        review.issues = all_issues[:8]

        if errored:
            review.verdict = "error"
            review.passed = False
        elif not scored:
            review.verdict = "unsupported"
            review.passed = False
        else:
            total_weight = sum(d.weight for d in scored)
            review.overall_score = (
                sum(d.score * d.weight for d in scored) / total_weight if total_weight else 0.0
            )
            bar_ok = review.overall_score >= threshold
            floors_ok = all(
                d.score >= DIMENSIONS[d.key].min_score for d in scored
            )
            if unsupported and policy == "strict":
                review.verdict = "unsupported"
                review.passed = False
                review.issues.extend(
                    f"[门禁] 以下维度未检测，strict 策略不允许降级通过: "
                    + "、".join(d.label for d in unsupported)
                )
            else:
                if unsupported:
                    review.degraded = True
                review.passed = bar_ok and floors_ok
                review.verdict = "passed" if review.passed else "failed"

        review.suggestion = self._collect_suggestion(dimensions)
        review.fix = self._build_fix(review, dimensions)
        self._log_review(review, unsupported)

    def _collect_suggestion(self, dimensions: list[DimensionResult]) -> str:
        failed = [
            dimension
            for dimension in dimensions
            if dimension.status == "scored"
            and dimension.score is not None
            and dimension.score < DIMENSIONS[dimension.key].min_score
        ]
        if not failed:
            return ""
        return "；".join(
            f"{dimension.label}偏低（{(dimension.score or 0):.2f}）：{'；'.join(dimension.issues[:2])}"
            for dimension in failed[:4]
        )

    def _build_fix(self, review: ShotReview, dimensions: list[DimensionResult]) -> dict:
        """根据失败/未达标维度生成 prompt 修正指令（确定性模板）。"""
        if review.verdict == "unsupported":
            return {"directives": [], "summary": "审核能力未配置，无修正建议（需人工处理）"}
        directives: list[str] = []
        for dimension in dimensions:
            if dimension.status != "scored" or dimension.score is None:
                continue
            spec = DIMENSIONS[dimension.key]
            if dimension.score >= spec.min_score:
                continue
            directives.append(spec.fix_template)
        return {
            "directives": directives,
            "summary": "；".join(directives[:4]) if directives else "",
        }

    def _log_review(self, review: ShotReview, unsupported: list[DimensionResult]) -> None:
        missing = "、".join(d.label for d in unsupported) or "-"
        logger.info(
            "质量审核 shot=%s stage=%s attempt=%d verdict=%s passed=%s score=%.2f "
            "degraded=%s unsupported=[%s] issues=%s",
            review.shot_id,
            review.stage,
            review.attempt,
            review.verdict,
            review.passed,
            review.overall_score,
            review.degraded,
            missing,
            review.issues[:3],
        )
        if review.degraded:
            logger.warning(
                "质量审核存在降级放行: shot=%s stage=%s 未检测维度=[%s]（已在界面如实标注）",
                review.shot_id, review.stage, missing,
            )

    # --- 持久化与通知 ------------------------------------------------------

    def _next_attempt(self, shot_id: str, stage: str) -> int:
        db = SessionLocal()
        try:
            latest = (
                db.query(QualityReview)
                .filter(QualityReview.shot_id == shot_id, QualityReview.stage == stage)
                .order_by(QualityReview.created_at.desc(), QualityReview.id.desc())
                .first()
            )
            return (latest.attempt + 1) if latest else 1
        finally:
            db.close()

    async def _persist_and_notify(self, review: ShotReview) -> None:
        row = QualityReview(
            id=f"qr_{uuid.uuid4().hex[:16]}",
            project_id=review.project_id,
            shot_id=review.shot_id,
            stage=review.stage,
            attempt=review.attempt,
            shot_version=review.shot_version,
            target_path=review.target_path,
            verdict=review.verdict,
            passed=bool(review.passed),
            overall_score=review.overall_score,
            degraded=bool(review.degraded),
            dimensions=json.dumps([d.to_dict() for d in review.dimensions], ensure_ascii=False),
            issues=json.dumps(review.issues, ensure_ascii=False),
            unsupported_dimensions=json.dumps(
                [d.label for d in review.dimensions if d.status == "unsupported"], ensure_ascii=False
            ),
            suggestion=review.suggestion,
            prompt_fix=json.dumps(review.fix, ensure_ascii=False),
            gate_policy=review.gate_policy,
        )
        db = SessionLocal()
        try:
            db.add(row)
            db.commit()
            review.row_id = row.id
        finally:
            db.close()
        await _send_quality_review_event(review)

    async def record_unsupported_reviews(
        self, project_id: str, shot_ids: list[str], stage: str, reason: str
    ) -> None:
        """能力未配置时为每个镜头落一条 unsupported 审核，界面可见原因。"""
        db = SessionLocal()
        try:
            shots = db.query(Shot).filter(Shot.project_id == project_id, Shot.id.in_(shot_ids)).all()
            for shot in shots:
                review = ShotReview(
                    shot_id=shot.id,
                    project_id=project_id,
                    stage=stage,
                    attempt=self._next_attempt(shot.id, stage),
                    shot_version=shot.version or 1,
                    target_path=(shot.video_path if stage == STAGE_VIDEO else (shot.storyboard_path or shot.image_path)) or "",
                    verdict="unsupported",
                    gate_policy=f"threshold={settings.QUALITY_REVIEW_PASS_SCORE} policy={settings.QUALITY_DEGRADATION_POLICY}",
                )
                review.dimensions = [
                    DimensionResult(
                        spec.key, spec.label, "unsupported", weight=spec.weight,
                        issues=[reason], evidence={"reason": reason}, provider="-",
                    )
                    for spec in DIMENSIONS.values()
                    if spec.stage == stage
                ]
                review.issues = [reason]
                review.fix = {"directives": [], "summary": "审核能力未配置，需人工审核"}
                row = QualityReview(
                    id=f"qr_{uuid.uuid4().hex[:16]}",
                    project_id=project_id,
                    shot_id=shot.id,
                    stage=stage,
                    attempt=review.attempt,
                    shot_version=review.shot_version,
                    target_path=review.target_path,
                    verdict="unsupported",
                    passed=False,
                    dimensions=json.dumps([d.to_dict() for d in review.dimensions], ensure_ascii=False),
                    issues=json.dumps(review.issues, ensure_ascii=False),
                    unsupported_dimensions=json.dumps(
                        [d.label for d in review.dimensions], ensure_ascii=False
                    ),
                    suggestion="",
                    prompt_fix=json.dumps(review.fix, ensure_ascii=False),
                    gate_policy=review.gate_policy,
                )
                db.add(row)
                review.row_id = row.id
            db.commit()
        finally:
            db.close()
        logger.error("质量审核能力未配置，%s 阶段 %d 个镜头标记 unsupported: %s", stage, len(shot_ids), reason)

    async def mark_shots_needs_human_review(self, shot_ids: list[str]) -> None:
        """始终未通过的镜头转人工：状态 needs_review 并通知前端。"""
        if not shot_ids:
            return
        db = SessionLocal()
        try:
            shots = db.query(Shot).filter(Shot.id.in_(shot_ids)).all()
            for shot in shots:
                shot.status = "needs_review"
            db.commit()
            project_ids = {shot.project_id for shot in shots}
        finally:
            db.close()
        logger.warning("以下镜头质量审核始终未通过，转人工审核（needs_review）: %s", shot_ids)
        from api.websocket import ws_manager

        for shot_id in shot_ids:
            await ws_manager.send_to_project(
                _shot_project_id(shot_id), {"type": "shot_update", "shot_id": shot_id, "status": "needs_review"}
            )
        for project_id in project_ids:
            await ws_manager.send_to_project(
                project_id,
                {
                    "type": "quality_gate_needs_human",
                    "project_id": project_id,
                    "shot_ids": list(shot_ids),
                },
            )

    # --- 查询 -------------------------------------------------------------

    def rows_for_shot(self, shot_id: str) -> list[dict]:
        db = SessionLocal()
        try:
            rows = (
                db.query(QualityReview)
                .filter(QualityReview.shot_id == shot_id)
                .order_by(QualityReview.created_at.desc(), QualityReview.id.desc())
                .all()
            )
            return [row_to_dict(row) for row in rows]
        finally:
            db.close()

    def rows_for_project(self, project_id: str) -> list[dict]:
        db = SessionLocal()
        try:
            rows = (
                db.query(QualityReview)
                .filter(QualityReview.project_id == project_id)
                .order_by(QualityReview.created_at.desc(), QualityReview.id.desc())
                .all()
            )
            return [row_to_dict(row) for row in rows]
        finally:
            db.close()

    def latest_reviews(self, project_id: str, stage: str) -> dict[str, dict]:
        """每个镜头该阶段最新一条审核（按创建时间）。"""
        rows = self.rows_for_project(project_id)
        latest: dict[str, dict] = {}
        for row in rows:
            if row["stage"] != stage:
                continue
            latest.setdefault(row["shot_id"], row)
        return latest

    def shot_review_summary(self, shot_id: str) -> dict:
        """镜头序列化用的轻量摘要：两个阶段各自最新裁决。"""
        rows = self.rows_for_shot(shot_id)
        summary: dict[str, dict | None] = {"storyboard": None, "video": None}
        for stage in ("storyboard", "video"):
            row = next((item for item in rows if item["stage"] == stage), None)
            if row:
                summary[stage] = {
                    "verdict": row["verdict"],
                    "passed": row["passed"],
                    "overall_score": row["overall_score"],
                    "attempt": row["attempt"],
                    "degraded": row["degraded"],
                    "issues_count": len(row["issues"]),
                    "unsupported": row["unsupported_dimensions"],
                }
        return summary

    def storyboard_gate_status(self, project_id: str) -> dict:
        """全部镜头是否都通过了故事板质量门禁（且已确认）。"""
        return self._gate_status(project_id, STAGE_STORYBOARD)

    def video_gate_status(self, project_id: str) -> dict:
        return self._gate_status(project_id, STAGE_VIDEO)

    def _gate_status(self, project_id: str, stage: str) -> dict:
        db = SessionLocal()
        try:
            shots = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
        finally:
            db.close()
        if not shots:
            return {"ok": False, "reason": "无镜头"}
        latest = self.latest_reviews(project_id, stage)
        failed: list[dict] = []
        for shot in shots:
            row = latest.get(shot.id)
            if row is None:
                failed.append({"shot_id": shot.id, "reason": "尚无质量审核记录"})
            elif not row["passed"]:
                failed.append({"shot_id": shot.id, "reason": f"最新审核未通过（{row['verdict']}）"})
            elif stage == STAGE_STORYBOARD and not shot.confirmed:
                failed.append({"shot_id": shot.id, "reason": "镜头未确认"})
            elif stage == STAGE_VIDEO and not (shot.video_path or ""):
                failed.append({"shot_id": shot.id, "reason": "镜头没有视频产物"})
        return {"ok": not failed, "failed": failed}


# ---------------------------------------------------------------------------
# 模块级辅助
# ---------------------------------------------------------------------------


def _shot_fields(shot: Shot) -> dict:
    return {
        "project_id": shot.project_id,
        "sequence": shot.sequence,
        "version": shot.version or 1,
        "shot_type": shot.shot_type or "medium",
        "shot_type_label": SHOT_TYPE_LABELS.get(shot.shot_type or "", str(shot.shot_type)),
        "scene_description": shot.scene_description or "",
        "character_action": shot.character_action or "",
        "camera_angle": shot.camera_angle or "正面",
        "camera_movement": shot.camera_movement or "静止",
        "dialogue": shot.dialogue or "",
        "emotion": shot.emotion or "neutral",
        "character_names": [str(name) for name in _json_list(shot.characters_in_scene)],
        "target_path": shot.storyboard_path or shot.image_path or "",
        "video_path": shot.video_path or "",
    }


def _character_references(db, shot: Shot) -> list[dict]:
    """出镜角色的参考图（每角色取首图，总量截断），供身份/服装核验。"""
    names = [str(name) for name in _json_list(shot.characters_in_scene)]
    if not names:
        return []
    rows = (
        db.query(Character)
        .filter(Character.project_id == shot.project_id, Character.name.in_(names))
        .all()
    )
    references: list[dict] = []
    for row in rows:
        for path in _json_list(row.reference_images)[:1]:
            if path and Path(path).exists():
                references.append(
                    {
                        "label": row.name,
                        "path": str(path),
                        "appearance": row.appearance or "",
                        "outfit": row.default_outfit or "",
                    }
                )
                break
    return references[:4]


def _storyboard_user_prompt(shot_fields: dict, character_refs: list[dict]) -> str:
    lines = [
        "【待审画面】第 1 张图。",
    ]
    if character_refs:
        ref_names = "、".join(ref["label"] for ref in character_refs)
        lines.append(f"【角色参考图】其后 {len(character_refs)} 张为出镜角色（{ref_names}）的设定参考图，身份/服装维度必须对照它们评分。")
    lines += [
        "镜头参数：",
        f"- 景别: {shot_fields['shot_type_label']}（{shot_fields['shot_type']}）",
        f"- 机位角度: {shot_fields['camera_angle']}",
        f"- 场景描述: {shot_fields['scene_description'] or '（空）'}",
        f"- 人物动作: {shot_fields['character_action'] or '（空）'}",
        f"- 出镜角色: {'、'.join(shot_fields['character_names']) or '（无）'}",
        f"- 台词: {shot_fields['dialogue'] or '（无）'}",
        "请逐维度给出 score/issues/evidence，并给一句总体修正建议。",
    ]
    return "\n".join(lines)


def _video_user_prompt(shot_fields: dict, frames: list[str], previous_frame_path: str) -> str:
    lines = [
        f"【待审视频】第 1-{len(frames)} 张图为按时间顺序从该镜头视频抽取的帧。",
    ]
    if previous_frame_path:
        lines.append(f"【衔接参考】第 {len(frames) + 1} 张图为上一镜头尾帧，shot_continuity 维度对照它评分。")
    lines += [
        "镜头参数：",
        f"- 人物动作: {shot_fields['character_action'] or '（空）'}",
        f"- 运镜: {shot_fields['camera_movement']}",
        f"- 台词: {shot_fields['dialogue'] or '（无台词时 lip_sync 按 10 分并在 evidence 说明）'}",
        "请逐维度给出 score/issues/evidence，并给一句总体修正建议。",
    ]
    return "\n".join(lines)


async def _send_quality_review_event(review: ShotReview) -> None:
    try:
        from api.websocket import ws_manager

        await ws_manager.send_to_project(
            review.project_id,
            {"type": "quality_review", "project_id": review.project_id, "review": review.to_dict()},
        )
    except Exception as exc:  # 通知失败不影响审核落库
        logger.warning("质量审核事件推送失败: %s", exc)


def _shot_project_id(shot_id: str) -> str:
    db = SessionLocal()
    try:
        row = db.query(Shot.project_id).filter(Shot.id == shot_id).first()
        return row[0] if row else ""
    finally:
        db.close()


def row_to_dict(row: QualityReview) -> dict:
    return {
        "id": row.id,
        "shot_id": row.shot_id,
        "project_id": row.project_id,
        "stage": row.stage,
        "attempt": row.attempt,
        "shot_version": row.shot_version,
        "target_path": row.target_path,
        "verdict": row.verdict,
        "passed": bool(row.passed),
        "overall_score": row.overall_score,
        "degraded": bool(row.degraded),
        "dimensions": _json_list(row.dimensions),
        "issues": _json_list(row.issues),
        "unsupported_dimensions": _json_list(row.unsupported_dimensions),
        "suggestion": row.suggestion,
        "prompt_fix": json.loads(row.prompt_fix or "{}"),
        "gate_policy": row.gate_policy,
        "created_at": row.created_at.isoformat() if row.created_at else "",
    }


def merge_quality_fix_notes(current_notes: str, directives: list[str]) -> str:
    """把修正指令合入 visual_notes：替换旧的质量修正块，避免反复叠加。

    修正块以「【质量修正】」开头标记，每次重试整块替换；块之外的原始
    notes 保持不变，用户手写内容不会被吞掉。
    """
    base = str(current_notes or "").split(QUALITY_FIX_TAG)[0].rstrip().rstrip("，,;")
    if not directives:
        return base
    block = QUALITY_FIX_TAG + "；".join(directives)
    return (base + "\n" + block) if base else block


quality_review_service = QualityReviewService()
