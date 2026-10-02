"""生成 Agent 的稳定契约、状态机词汇与可解释决策数据结构。

本模块只描述数据和规则，不调用数据库或供应商。所有 LangGraph 节点、检查点、
API 追踪和测试都使用这里的枚举/模型，避免不同层各自发明一套状态含义。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StageName(str, Enum):
    DIRECTOR_PLANNING = "director_planning"
    STORYBOARD_DESIGN = "storyboard_design"
    ASSET_PREPARATION = "asset_preparation"
    IMAGE_GENERATION = "image_generation"
    QUALITY_REVIEW = "quality_review"
    AUDIO_PRODUCTION = "audio_production"
    VIDEO_GENERATION = "video_generation"
    VIDEO_REVIEW = "video_review"
    EDIT_COMPOSITION = "edit_composition"
    FINAL_REVIEW = "final_review"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RECOVERING = "recovering"
    SUCCEEDED = "succeeded"
    DEGRADED = "degraded"
    WAITING_HUMAN = "waiting_human"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RECOVERING = "recovering"
    WAITING_HUMAN = "waiting_human"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailureKind(str, Enum):
    LLM_INVALID_OUTPUT = "llm_invalid_output"
    # 模型输出达到 max_tokens 上限被截断（finish_reason=length 或输出顶满额度）。
    # 这是确定性失败：不允许改提示词或同配置重试，只能提高/分段输出或换 Provider。
    LLM_OUTPUT_TRUNCATED = "llm_output_truncated"
    IMAGE_FAILED = "image_failed"
    VIDEO_FAILED = "video_failed"
    AUDIO_FAILED = "audio_failed"
    DIALOGUE_TOO_LONG = "dialogue_too_long"
    SHOT_TOO_COMPLEX = "shot_too_complex"
    PROVIDER_REFERENCE_UNSUPPORTED = "provider_reference_unsupported"
    PROVIDER_CAPABILITY_MISMATCH = "provider_capability_mismatch"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    BUDGET_EXCEEDED = "budget_exceeded"
    VERSION_CONFLICT = "version_conflict"
    USER_CHANGED_INPUT = "user_changed_input"
    DEPENDENCY_FAILED = "dependency_failed"
    QUALITY_BELOW_THRESHOLD = "quality_below_threshold"
    STORAGE_FAILED = "storage_failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


# 这些失败类别无法通过再花一次生成成本解决，恢复决策只能降级发布或明确终止。
NON_RECOVERABLE_FAILURES: frozenset[FailureKind] = frozenset(
    {
        FailureKind.BUDGET_EXCEEDED,
        FailureKind.USER_CHANGED_INPUT,
        FailureKind.CANCELLED,
    }
)


class RecoveryStrategy(str, Enum):
    # 自动恢复顺序由 agent.decision.AUTOMATIC_RECOVERY_ORDER 统一定义；
    # 这些枚举值同时是检查点/API 中可追踪的动作名称。
    RETRY = "retry"
    CHANGE_SEED = "change_seed"
    REVISE_PROMPT = "revise_prompt"
    REPLACE_REFERENCE = "replace_reference"
    SPLIT_SHOT = "split_shot"
    SWITCH_PROVIDER = "switch_provider"
    LOWER_RESOLUTION = "lower_resolution"
    REGENERATE_FAILED_SHOTS = "regenerate_failed_shots"
    MERGE_SHOTS = "merge_shots"
    RESUME_CHECKPOINT = "resume_checkpoint"
    HUMAN_REVIEW = "human_review"
    DEGRADED_PUBLISH = "degraded_publish"
    TERMINAL_FAILURE = "terminal_failure"


# 自动模式下永远不可被选择的策略（只有 manual 模式允许进入人工节点）。
MANUAL_ONLY_STRATEGIES: frozenset[RecoveryStrategy] = frozenset({RecoveryStrategy.HUMAN_REVIEW})

# 不再消耗恢复次数的终止策略；只有当没有其它可行候选时才会被选择。
TERMINAL_STRATEGIES: frozenset[RecoveryStrategy] = frozenset(
    {RecoveryStrategy.DEGRADED_PUBLISH, RecoveryStrategy.TERMINAL_FAILURE, RecoveryStrategy.HUMAN_REVIEW}
)


class QualityProfileName(str, Enum):
    DRAFT = "draft"
    STANDARD = "standard"
    FINISHING = "finishing"


class HumanInterventionPolicy(str, Enum):
    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED = "required"


class PublishPolicy(str, Enum):
    AUTO_PUBLISH = "auto_publish"
    PUBLISH_IF_PASSED = "publish_if_passed"
    HOLD_FOR_REVIEW = "hold_for_review"


class QualityStrategy(BaseModel):
    """自动模式的质量档位：成本、候选数、恢复预算与人工介入边界。"""

    model_config = ConfigDict(frozen=True)

    name: QualityProfileName
    mode: Literal["manual", "auto"] = "auto"
    label: str
    description: str
    cost_multiplier: float
    candidate_count: int
    max_recovery_attempts: int
    quality_threshold: float
    expected_duration: int
    resolution: Literal["540p", "720p", "1080p", "4k"]
    publish_policy: PublishPolicy
    human_intervention: HumanInterventionPolicy
    human_intervention_note: str = ""
    auto_approve: bool

    @model_validator(mode="after")
    def validate_automatic_human_policy(self) -> "QualityStrategy":
        if self.mode == "auto" and self.human_intervention is not HumanInterventionPolicy.DISABLED:
            raise ValueError("自动模式的 human_intervention 必须为 disabled")
        return self


QUALITY_STRATEGIES: dict[QualityProfileName, QualityStrategy] = {
    QualityProfileName.DRAFT: QualityStrategy(
        name=QualityProfileName.DRAFT,
        label="草稿",
        description="快速验证叙事和镜头可行性，接受较低视觉一致性；成本最低。",
        cost_multiplier=0.6,
        candidate_count=1,
        max_recovery_attempts=1,
        quality_threshold=0.55,
        expected_duration=180,
        resolution="540p",
        publish_policy=PublishPolicy.AUTO_PUBLISH,
        human_intervention=HumanInterventionPolicy.DISABLED,
        human_intervention_note="自动模式不等待人工确认；硬失败或预算耗尽时明确记录失败。",
        auto_approve=True,
    ),
    QualityProfileName.STANDARD: QualityStrategy(
        name=QualityProfileName.STANDARD,
        label="标准",
        description="默认发布档位，保留候选并自动修复常见问题；成本和耗时居中。",
        cost_multiplier=1.0,
        candidate_count=2,
        max_recovery_attempts=2,
        quality_threshold=0.72,
        expected_duration=420,
        resolution="720p",
        publish_policy=PublishPolicy.PUBLISH_IF_PASSED,
        human_intervention=HumanInterventionPolicy.DISABLED,
        human_intervention_note="自动模式只发布通过质量门禁的结果，失败时保留检查点和恢复候选。",
        auto_approve=True,
    ),
    QualityProfileName.FINISHING: QualityStrategy(
        name=QualityProfileName.FINISHING,
        label="精修",
        description="多候选、强一致性与成片复审，优先质量；成本和耗时最高。",
        cost_multiplier=1.8,
        candidate_count=3,
        max_recovery_attempts=3,
        quality_threshold=0.86,
        expected_duration=900,
        resolution="1080p",
        publish_policy=PublishPolicy.PUBLISH_IF_PASSED,
        human_intervention=HumanInterventionPolicy.DISABLED,
        human_intervention_note="自动模式执行严格门禁，但不插入人工确认；未通过时不发布。",
        auto_approve=True,
    ),
}


class ProviderCapability(str, Enum):
    JSON_OUTPUT = "json_output"
    VISION = "vision"
    REFERENCE_IMAGES = "reference_images"
    REFERENCE_IMAGE = "reference_image"
    NATIVE_AUDIO = "native_audio"
    DIALOGUE_IN_PROMPT = "dialogue_in_prompt"
    FIXED_DURATION = "fixed_duration"
    RESOLUTION_CONTROL = "resolution_control"


class ProviderProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capability: str
    provider: str
    model: str = ""
    available: bool = True
    # 当前实际在用的端点（switch_provider 的语义是换到别的 Provider，
    # 切换目标等于当前端点时必须判定为不可行，而不是原参数重跑一遍）。
    is_current: bool = False
    supports_reference_images: bool = False
    supports_reference_image: bool = False
    native_audio: bool = False
    max_resolution: str = "1080p"
    estimated_cost_micro: int | None = None
    estimated_seconds: int | None = None
    reliability: float = 0.8
    reason: str = ""


class QualityMetric(BaseModel):
    name: str
    value: float | int | bool | str | None = None
    threshold: float | None = None
    passed: bool | None = None
    weight: float = 1.0
    detail: str = ""


class FailureRecord(BaseModel):
    kind: FailureKind
    stage: StageName | str
    shot_id: str = ""
    message: str = ""
    provider: str = ""
    retryable: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class CriticIssue(BaseModel):
    code: str
    severity: Literal["info", "warning", "error"]
    message: str
    shot_id: str = ""
    recommendation: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class CritiqueReport(BaseModel):
    """Critic 的完整反思报告：不只 passed/failed，还带失败分类、证据与恢复建议。"""

    stage: StageName | str
    passed: bool
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    metrics: list[QualityMetric] = Field(default_factory=list)
    issues: list[CriticIssue] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    failure_kind: FailureKind | None = None
    recoverable: bool = True
    proposed_changes: list[str] = Field(default_factory=list)
    affected_shot_ids: list[str] = Field(default_factory=list)
    recommended_strategy: RecoveryStrategy | None = None
    candidate_count: int = 1
    source: str = "deterministic"
    created_at: str = Field(default_factory=utc_now)


class PromptPatch(BaseModel):
    """结构化 Prompt 修改补丁：字段级 op/value，节点可直接应用。"""

    field: str
    op: Literal["replace", "append", "remove", "set", "split"] = "set"
    value: Any = None
    shot_id: str = ""
    target_stage: str = ""
    reason: str = ""


class RecoveryCandidate(BaseModel):
    strategy: RecoveryStrategy
    provider: str = ""
    target_stage: StageName | str = ""
    shot_ids: list[str] = Field(default_factory=list)
    prompt_changes: dict[str, Any] = Field(default_factory=dict)
    prompt_patches: list[PromptPatch] = Field(default_factory=list)
    estimated_cost_micro: int | None = None
    estimated_seconds: int | None = None
    quality_gain: float = 0.0
    provider_capability_ok: bool = True
    budget_fit: bool = True
    retries_remaining: int | None = None
    score: float = 0.0
    rationale: str = ""


class DecisionTrace(BaseModel):
    """每次 Agent 选择的完整可解释记录：候选、淘汰原因和最终选择。"""

    model_config = ConfigDict(extra="ignore")

    trace_id: str
    project_id: str = ""
    shot_version: int = 0
    run_id: str
    stage: StageName | str
    input_fingerprint: str = ""
    mode: Literal["manual", "auto"] = "auto"
    failure: FailureRecord | None = None
    critique: CritiqueReport | None = None
    candidates: list[RecoveryCandidate] = Field(default_factory=list)
    selected: RecoveryCandidate | None = None
    considered_rejected: list[dict[str, Any]] = Field(default_factory=list)
    attempted_strategies: list[str] = Field(default_factory=list)
    budget_snapshot: dict[str, Any] = Field(default_factory=dict)
    provider_profiles: list[ProviderProfile] = Field(default_factory=list)
    retries_remaining: int = 0
    quality_score: float | None = None
    reason: str = ""
    shot_id: str = ""
    selected_video_candidate_id: str = ""
    video_candidates: list[dict[str, Any]] = Field(default_factory=list)
    candidate_selection: dict[str, Any] | None = None
    created_at: str = Field(default_factory=utc_now)


class VideoCandidateStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INVALIDATED = "invalidated"


class VideoCandidateRecord(BaseModel):
    """一次独立视频候选的可恢复记录；失败与成功都只追加，不互相覆盖。

    ``path/last_frame_path/recipe_hash/metrics`` 是稳定的新契约字段；
    ``video_path/tail_frame_path/execution_plan_hash/structural_metrics`` 作为旧
    调用方兼容字段保留，并在验证时双向同步。候选媒体路径永远是临时候选路径，
    不是 ``Shot.video_path`` 正式发布路径。
    """

    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    shot_id: str
    shot_version: int
    batch_id: str = ""
    candidate_index: int = 1
    status: VideoCandidateStatus
    path: str = ""
    last_frame_path: str = ""
    provider: str = ""
    model: str = ""
    seed: int | None = None
    recipe_hash: str = ""
    reference_manifest: list[dict[str, Any]] = Field(default_factory=list)
    generation_duration_ms: int = 0
    score: float = 0.0
    metrics: dict[str, Any] = Field(default_factory=dict)
    failure: FailureRecord | None = None
    retry_of_candidate_id: str = ""
    selected: bool = False
    selection_reason: str = ""
    created_at: str = Field(default_factory=utc_now)
    # 兼容旧持久化/API 字段。
    video_path: str = ""
    tail_frame_path: str = ""
    execution_plan_hash: str = ""
    structural_passed: bool | None = None
    structural_metrics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if not data.get("failure"):
            data["failure"] = None
        data.setdefault("path", data.get("video_path") or "")
        data.setdefault("last_frame_path", data.get("tail_frame_path") or "")
        data.setdefault("recipe_hash", data.get("execution_plan_hash") or "")
        data.setdefault("metrics", data.get("structural_metrics") or {})
        data.setdefault("video_path", data.get("path") or "")
        data.setdefault("tail_frame_path", data.get("last_frame_path") or "")
        data.setdefault("execution_plan_hash", data.get("recipe_hash") or "")
        data.setdefault("structural_metrics", data.get("metrics") or {})
        return data

    @model_validator(mode="after")
    def _sync_aliases(self) -> "VideoCandidateRecord":
        self.path = str(self.path or self.video_path or "")
        self.video_path = self.path
        self.last_frame_path = str(self.last_frame_path or self.tail_frame_path or "")
        self.tail_frame_path = self.last_frame_path
        self.recipe_hash = str(self.recipe_hash or self.execution_plan_hash or "")
        self.execution_plan_hash = self.recipe_hash
        self.metrics = dict(self.metrics or self.structural_metrics or {})
        self.structural_metrics = dict(self.metrics)
        return self


class VideoCandidateSelection(BaseModel):
    candidate_id: str = ""
    shot_version: int = 0
    score: float = 0.0
    reason: str = ""
    considered: list[str] = Field(default_factory=list)
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    score_breakdown: list[dict[str, Any]] = Field(default_factory=list)


def select_video_candidate(
    candidates: list[VideoCandidateRecord | dict[str, Any]],
    *,
    require_structural: bool = True,
    allow_single_candidate_fallback: bool = False,
) -> VideoCandidateSelection:
    """本步骤只做确定性结构选择，不调用深度视觉质量模型。

    默认优先 ``structural_passed=True`` 且分数最高者；同分时生成耗时更短者优先，
    最后用 candidate_index 保证结果稳定。失败候选始终留在输入集合中但不参与选择。
    """

    rows: list[VideoCandidateRecord] = []
    for item in candidates:
        rows.append(item if isinstance(item, VideoCandidateRecord) else VideoCandidateRecord.model_validate(item))
    considered = [row.candidate_id for row in rows]
    succeeded = [row for row in rows if row.status is VideoCandidateStatus.SUCCEEDED]
    # ``False`` 表示结构检查明确失败，任何模式都不得选中；``None`` 只用于旧 Provider
    # 无法探测媒体时的兼容路径，严格模式仍要求 True。
    eligible = [row for row in succeeded if row.structural_passed is True] if require_structural else [
        row for row in succeeded if row.structural_passed is not False
    ]
    rejected: list[dict[str, Any]] = []
    for row in rows:
        if row.status is not VideoCandidateStatus.SUCCEEDED:
            rejected.append({"candidate_id": row.candidate_id, "reason": "candidate_failed"})
        elif row.structural_passed is False or (require_structural and row.structural_passed is not True):
            rejected.append({"candidate_id": row.candidate_id, "reason": "structural_check_failed"})
    if not eligible and allow_single_candidate_fallback and len(succeeded) == 1 and succeeded[0].structural_passed is not False:
        chosen = succeeded[0]
        return VideoCandidateSelection(
            candidate_id=chosen.candidate_id,
            shot_version=chosen.shot_version,
            score=chosen.score,
            reason="single_candidate_compatibility_fallback",
            considered=considered,
            rejected=rejected,
        )
    if not eligible:
        return VideoCandidateSelection(reason="no_structurally_valid_candidate", considered=considered, rejected=rejected)
    chosen = min(eligible, key=lambda row: (-row.score, row.generation_duration_ms, row.candidate_index, row.candidate_id))
    return VideoCandidateSelection(
        candidate_id=chosen.candidate_id,
        shot_version=chosen.shot_version,
        score=chosen.score,
        reason="structural_pass_highest_score",
        considered=considered,
        rejected=rejected,
        score_breakdown=[
            {
                "candidate_id": row.candidate_id,
                "score": row.score,
                "eligible": row in eligible,
                "status": row.status.value,
                "structural_passed": row.structural_passed,
                "rejection_reason": next((item["reason"] for item in rejected if item["candidate_id"] == row.candidate_id), ""),
            }
            for row in sorted(rows, key=lambda item: item.candidate_id)
        ],
    )


def score_video_candidate(candidate: VideoCandidateRecord | dict[str, Any]) -> float:
    """候选全部完成后调用的确定性 Agent 评分。

    当前只使用结构/技术检查的可观测指标，不伪装成深度视觉评分。明确的结构失败
    一律为 0 分；有已有质量分时保留，否则按结构、技术、告警三项归一化。
    """

    row = candidate if isinstance(candidate, VideoCandidateRecord) else VideoCandidateRecord.model_validate(candidate)
    if row.status is not VideoCandidateStatus.SUCCEEDED or row.structural_passed is False:
        return 0.0
    metrics = row.metrics or {}
    # 候选全部完成后统一评分；有结构报告时总是按可观测指标重算，
    # 只有旧记录没有报告时才保留历史 score，避免把生成器自填分数当质量结论。
    if not metrics and row.score > 0:
        return max(0.0, min(1.0, float(row.score)))
    categories = metrics.get("categories") or {}
    structural = categories.get("structural_validity") or {}
    technical = categories.get("technical_quality") or {}
    structural_score = 1.0 if row.structural_passed is True or structural.get("passed") is True else 0.5
    technical_score = 1.0 if technical.get("passed") is True else (0.5 if technical.get("passed") is None else 0.0)
    warning_count = len(metrics.get("warnings") or [])
    warning_score = max(0.0, 1.0 - min(1.0, warning_count * 0.1))
    return round(0.6 * structural_score + 0.3 * technical_score + 0.1 * warning_score, 3)


class ShotArtifact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    project_id: str = ""
    run_id: str = ""
    input_fingerprint: str = ""
    shot_id: str
    shot_version: int
    stage: StageName | str
    status: StageStatus
    path: str = ""
    score: float = 0.0
    provider: str = ""
    cost_micro: int | None = None
    duration_ms: int = 0
    failure: FailureRecord | None = None
    metrics: list[QualityMetric] = Field(default_factory=list)
    output_fingerprint: str = ""
    video_candidates: list[VideoCandidateRecord] = Field(default_factory=list)
    selected_video_candidate_id: str = ""
    candidate_selection: VideoCandidateSelection | None = None
    decision_trace: dict[str, Any] | None = None
    # 视频 Critic 的技术检查上下文（执行计划时长 / 目标画幅 / 配音 / 尾帧）。
    # 缺省时对应检查维度记为 skipped，不会伪装成通过。
    expected_duration_s: float | None = None
    expected_aspect_ratio: float | None = None
    audio_path: str = ""
    tail_frame_path: str = ""
    created_at: str = Field(default_factory=utc_now)


class StageInput(BaseModel):
    project_id: str
    shot_version: int = 0
    run_id: str
    stage: StageName | str
    input_fingerprint: str
    quality_profile: QualityProfileName = QualityProfileName.STANDARD
    payload: dict[str, Any] = Field(default_factory=dict)
    budget_snapshot: dict[str, Any] = Field(default_factory=dict)
    provider_profiles: list[ProviderProfile] = Field(default_factory=list)


class StageOutput(BaseModel):
    project_id: str
    shot_version: int = 0
    run_id: str
    stage: StageName | str
    status: StageStatus
    input_fingerprint: str
    output_fingerprint: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    quality: list[QualityMetric] = Field(default_factory=list)
    failure: FailureRecord | None = None
    critique: CritiqueReport | None = None
    shot_artifacts: list[ShotArtifact] = Field(default_factory=list)
    checkpoint_version: int = 1
    created_at: str = Field(default_factory=utc_now)


class CheckpointRecord(BaseModel):
    """检查点的统一持久化记录，覆盖阶段级和逐镜头级状态。"""

    model_config = ConfigDict(extra="ignore")

    key: str
    checkpoint_key: str
    kind: Literal["stage", "shot"]
    project_id: str
    shot_version: int = 0
    run_id: str
    stage: StageName | str
    shot_id: str = ""
    status: StageStatus | Literal["invalidated"]
    input_fingerprint: str
    output_fingerprint: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    quality: list[QualityMetric] = Field(default_factory=list)
    failure: FailureRecord | None = None
    critique: CritiqueReport | None = None
    shot_artifacts: list[ShotArtifact] = Field(default_factory=list)
    artifact: ShotArtifact | None = None
    checkpoint_version: int = 1
    valid: bool = True
    invalidated_reason: str = ""
    invalidated_at: str = ""
    created_at: str = Field(default_factory=utc_now)
    saved_at: str = Field(default_factory=utc_now)

    @property
    def reusable(self) -> bool:
        return self.valid and self.status in {
            StageStatus.SUCCEEDED,
            StageStatus.DEGRADED,
            StageStatus.SKIPPED,
        }

    def identity_matches(
        self,
        *,
        project_id: str,
        run_id: str,
        input_fingerprint: str | None = None,
        output_fingerprint: str | None = None,
        shot_version: int | None = None,
    ) -> bool:
        return (
            self.project_id == str(project_id)
            and self.run_id == str(run_id)
            and (input_fingerprint is None or self.input_fingerprint == str(input_fingerprint))
            and (output_fingerprint is None or self.output_fingerprint == str(output_fingerprint))
            and (shot_version is None or self.shot_version == int(shot_version))
        )


_DEFAULT_ALLOWED_RECOVERY: dict[StageName, tuple[RecoveryStrategy, ...]] = {
    StageName.DIRECTOR_PLANNING: (
        RecoveryStrategy.REVISE_PROMPT,
        RecoveryStrategy.SWITCH_PROVIDER,
        RecoveryStrategy.RETRY,
        RecoveryStrategy.RESUME_CHECKPOINT,
        RecoveryStrategy.HUMAN_REVIEW,
    ),
    StageName.STORYBOARD_DESIGN: (
        RecoveryStrategy.SPLIT_SHOT,
        RecoveryStrategy.MERGE_SHOTS,
        RecoveryStrategy.REVISE_PROMPT,
        RecoveryStrategy.RETRY,
        RecoveryStrategy.RESUME_CHECKPOINT,
        RecoveryStrategy.HUMAN_REVIEW,
    ),
    StageName.ASSET_PREPARATION: (
        RecoveryStrategy.REPLACE_REFERENCE,
        RecoveryStrategy.SWITCH_PROVIDER,
        RecoveryStrategy.RETRY,
        RecoveryStrategy.RESUME_CHECKPOINT,
        RecoveryStrategy.HUMAN_REVIEW,
    ),
    StageName.IMAGE_GENERATION: (
        RecoveryStrategy.RETRY,
        RecoveryStrategy.CHANGE_SEED,
        RecoveryStrategy.REVISE_PROMPT,
        RecoveryStrategy.REPLACE_REFERENCE,
        RecoveryStrategy.SPLIT_SHOT,
        RecoveryStrategy.SWITCH_PROVIDER,
        RecoveryStrategy.LOWER_RESOLUTION,
        RecoveryStrategy.REGENERATE_FAILED_SHOTS,
        RecoveryStrategy.RESUME_CHECKPOINT,
        RecoveryStrategy.HUMAN_REVIEW,
    ),
    StageName.QUALITY_REVIEW: (
        RecoveryStrategy.RETRY,
        RecoveryStrategy.CHANGE_SEED,
        RecoveryStrategy.REGENERATE_FAILED_SHOTS,
        RecoveryStrategy.REVISE_PROMPT,
        RecoveryStrategy.REPLACE_REFERENCE,
        RecoveryStrategy.SPLIT_SHOT,
        RecoveryStrategy.SWITCH_PROVIDER,
        RecoveryStrategy.LOWER_RESOLUTION,
        RecoveryStrategy.MERGE_SHOTS,
        RecoveryStrategy.RESUME_CHECKPOINT,
        RecoveryStrategy.HUMAN_REVIEW,
    ),
    StageName.AUDIO_PRODUCTION: (
        RecoveryStrategy.REGENERATE_FAILED_SHOTS,
        RecoveryStrategy.REVISE_PROMPT,
        RecoveryStrategy.SPLIT_SHOT,
        RecoveryStrategy.SWITCH_PROVIDER,
        RecoveryStrategy.RETRY,
        RecoveryStrategy.RESUME_CHECKPOINT,
        RecoveryStrategy.HUMAN_REVIEW,
    ),
    StageName.VIDEO_GENERATION: (
        RecoveryStrategy.RETRY,
        RecoveryStrategy.CHANGE_SEED,
        RecoveryStrategy.REVISE_PROMPT,
        RecoveryStrategy.REPLACE_REFERENCE,
        RecoveryStrategy.SPLIT_SHOT,
        RecoveryStrategy.SWITCH_PROVIDER,
        RecoveryStrategy.LOWER_RESOLUTION,
        RecoveryStrategy.REGENERATE_FAILED_SHOTS,
        RecoveryStrategy.RESUME_CHECKPOINT,
        RecoveryStrategy.HUMAN_REVIEW,
    ),
    StageName.VIDEO_REVIEW: (
        RecoveryStrategy.RETRY,
        RecoveryStrategy.CHANGE_SEED,
        RecoveryStrategy.REVISE_PROMPT,
        RecoveryStrategy.REPLACE_REFERENCE,
        RecoveryStrategy.SPLIT_SHOT,
        RecoveryStrategy.SWITCH_PROVIDER,
        RecoveryStrategy.LOWER_RESOLUTION,
        RecoveryStrategy.REGENERATE_FAILED_SHOTS,
        RecoveryStrategy.RESUME_CHECKPOINT,
        RecoveryStrategy.HUMAN_REVIEW,
    ),
    StageName.EDIT_COMPOSITION: (
        RecoveryStrategy.REGENERATE_FAILED_SHOTS,
        RecoveryStrategy.RETRY,
        RecoveryStrategy.RESUME_CHECKPOINT,
        RecoveryStrategy.HUMAN_REVIEW,
    ),
    StageName.FINAL_REVIEW: (
        RecoveryStrategy.REVISE_PROMPT,
        RecoveryStrategy.SPLIT_SHOT,
        RecoveryStrategy.MERGE_SHOTS,
        RecoveryStrategy.REGENERATE_FAILED_SHOTS,
        RecoveryStrategy.RESUME_CHECKPOINT,
        RecoveryStrategy.HUMAN_REVIEW,
    ),
}


class StageContract(BaseModel):
    """阶段契约的单一事实源。"""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    stage: StageName
    label: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    quality_metrics: list[str]
    failure_classes: list[FailureKind]
    checkpoint_key: str
    recoverable: bool = True
    allowed_recovery: list[RecoveryStrategy] = Field(default_factory=list)
    fan_out: bool = False
    description: str = ""

    @model_validator(mode="after")
    def validate_contract(self) -> "StageContract":
        allowed = list(_DEFAULT_ALLOWED_RECOVERY[self.stage])
        # 终止回退（降级发布/明确失败）在任何阶段都必须可选，否则恢复耗尽后无路可走。
        allowed.extend(item for item in (RecoveryStrategy.DEGRADED_PUBLISH, RecoveryStrategy.TERMINAL_FAILURE) if item not in allowed)
        if not self.allowed_recovery:
            object.__setattr__(self, "allowed_recovery", allowed)
        if self.recoverable and not self.allowed_recovery:
            raise ValueError(f"{self.stage.value} 可恢复阶段必须声明 allowed_recovery")
        if not self.recoverable and self.allowed_recovery:
            raise ValueError(f"{self.stage.value} 不可恢复阶段不得声明恢复策略")
        return self


STAGE_CONTRACTS: dict[StageName, StageContract] = {
    StageName.DIRECTOR_PLANNING: StageContract(
        stage=StageName.DIRECTOR_PLANNING,
        label="导演规划",
        input_model=StageInput,
        output_model=StageOutput,
        quality_metrics=["schema_valid", "character_coverage", "scene_coverage", "logic_issue_count", "duration_deviation"],
        failure_classes=[FailureKind.LLM_INVALID_OUTPUT, FailureKind.LLM_OUTPUT_TRUNCATED, FailureKind.DEPENDENCY_FAILED, FailureKind.TIMEOUT],
        checkpoint_key="director",
        description="解析剧本、人物、场景、叙事目标和资源预算，产出导演意图与候选拍摄计划。",
    ),
    StageName.STORYBOARD_DESIGN: StageContract(
        stage=StageName.STORYBOARD_DESIGN,
        label="分镜设计",
        input_model=StageInput,
        output_model=StageOutput,
        quality_metrics=["shot_count", "shot_rule_compliance", "dialogue_duration_ratio", "continuity_score", "complex_action_score"],
        failure_classes=[FailureKind.LLM_INVALID_OUTPUT, FailureKind.LLM_OUTPUT_TRUNCATED, FailureKind.DIALOGUE_TOO_LONG, FailureKind.SHOT_TOO_COMPLEX, FailureKind.QUALITY_BELOW_THRESHOLD],
        checkpoint_key="storyboard",
        description="将导演意图拆成可执行镜头，必要时自动拆分或合并镜头并保留决策记录。",
    ),
    StageName.ASSET_PREPARATION: StageContract(
        stage=StageName.ASSET_PREPARATION,
        label="素材准备",
        input_model=StageInput,
        output_model=StageOutput,
        quality_metrics=["character_reference_coverage", "scene_reference_coverage", "reference_compatibility", "asset_version_match"],
        failure_classes=[FailureKind.PROVIDER_REFERENCE_UNSUPPORTED, FailureKind.PROVIDER_CAPABILITY_MISMATCH, FailureKind.PROVIDER_UNAVAILABLE, FailureKind.STORAGE_FAILED, FailureKind.USER_CHANGED_INPUT],
        checkpoint_key="assets",
        description="生成/校验角色三视图、场景基准图和连续性参考，标记 Provider 能力限制。",
    ),
    StageName.IMAGE_GENERATION: StageContract(
        stage=StageName.IMAGE_GENERATION,
        label="图像生成",
        input_model=StageInput,
        output_model=StageOutput,
        quality_metrics=["image_valid", "prompt_alignment", "character_consistency", "composition", "candidate_score"],
        failure_classes=[FailureKind.IMAGE_FAILED, FailureKind.PROVIDER_REFERENCE_UNSUPPORTED, FailureKind.PROVIDER_CAPABILITY_MISMATCH, FailureKind.TIMEOUT, FailureKind.VERSION_CONFLICT, FailureKind.BUDGET_EXCEEDED, FailureKind.STORAGE_FAILED],
        checkpoint_key="image",
        fan_out=True,
        description="逐镜头独立生成故事板候选；单镜头失败不会回滚已成功镜头。",
    ),
    StageName.QUALITY_REVIEW: StageContract(
        stage=StageName.QUALITY_REVIEW,
        label="质量审核",
        input_model=StageInput,
        output_model=StageOutput,
        quality_metrics=["structural_validity", "continuity", "prompt_alignment", "review_score", "failure_recovery_rate"],
        failure_classes=[FailureKind.QUALITY_BELOW_THRESHOLD, FailureKind.DIALOGUE_TOO_LONG, FailureKind.DEPENDENCY_FAILED],
        checkpoint_key="quality",
        description="Critic/Reviewer 对候选结果给出评分、问题证据、具体修改和恢复建议。",
    ),
    StageName.VIDEO_GENERATION: StageContract(
        stage=StageName.VIDEO_GENERATION,
        label="视频生成",
        input_model=StageInput,
        output_model=StageOutput,
        quality_metrics=[
            "structural_validity",
            "technical_quality",
            "visual_quality_pending",
            "video_valid",
            "duration_match",
            "provider_capability",
        ],
        failure_classes=[FailureKind.VIDEO_FAILED, FailureKind.PROVIDER_UNAVAILABLE, FailureKind.PROVIDER_CAPABILITY_MISMATCH, FailureKind.TIMEOUT, FailureKind.STORAGE_FAILED, FailureKind.VERSION_CONFLICT, FailureKind.BUDGET_EXCEEDED],
        checkpoint_key="video",
        fan_out=True,
        description="逐镜头生成视频候选，支持失败镜头局部补拍、降分辨率和 Provider 切换；视觉质量在接入视觉模型前保持待审（visual_quality_pending）。",
    ),
    StageName.VIDEO_REVIEW: StageContract(
        stage=StageName.VIDEO_REVIEW,
        label="视频检查",
        input_model=StageInput,
        output_model=StageOutput,
        quality_metrics=["structural_validity", "technical_quality", "visual_quality_pending", "review_score", "failure_recovery_rate"],
        failure_classes=[FailureKind.QUALITY_BELOW_THRESHOLD, FailureKind.VIDEO_FAILED, FailureKind.PROVIDER_CAPABILITY_MISMATCH, FailureKind.DEPENDENCY_FAILED],
        checkpoint_key="video_review",
        description="检查已生成视频的结构、技术质量和待审视觉质量，并把失败镜头送入局部恢复。",
    ),
    StageName.AUDIO_PRODUCTION: StageContract(
        stage=StageName.AUDIO_PRODUCTION,
        label="音频制作",
        input_model=StageInput,
        output_model=StageOutput,
        quality_metrics=["dialogue_length", "tts_valid", "voice_consistency", "audio_duration", "mix_readiness"],
        failure_classes=[FailureKind.DIALOGUE_TOO_LONG, FailureKind.AUDIO_FAILED, FailureKind.PROVIDER_UNAVAILABLE, FailureKind.VERSION_CONFLICT],
        checkpoint_key="audio",
        fan_out=True,
        description="为对白镜头生成/复用配音，过长对白自动拆句或转人工。",
    ),
    StageName.EDIT_COMPOSITION: StageContract(
        stage=StageName.EDIT_COMPOSITION,
        label="剪辑合成",
        input_model=StageInput,
        output_model=StageOutput,
        quality_metrics=["shot_completeness", "av_sync", "timeline_duration", "render_valid", "degraded_shot_count"],
        failure_classes=[FailureKind.DEPENDENCY_FAILED, FailureKind.STORAGE_FAILED, FailureKind.TIMEOUT, FailureKind.BUDGET_EXCEEDED],
        checkpoint_key="compose",
        description="只消费已完成/明确降级的镜头，生成成片并保留跳过或人工介入状态。",
    ),
    StageName.FINAL_REVIEW: StageContract(
        stage=StageName.FINAL_REVIEW,
        label="成片复审",
        input_model=StageInput,
        output_model=StageOutput,
        quality_metrics=["story_coherence", "visual_consistency", "audio_quality", "pacing", "overall_score", "human_gate"],
        failure_classes=[FailureKind.QUALITY_BELOW_THRESHOLD, FailureKind.USER_CHANGED_INPUT, FailureKind.DEPENDENCY_FAILED],
        checkpoint_key="final",
        description="从成片层面反思节奏、连续性、对白和视觉质量，决定发布、局部重算或转人工。",
    ),
}

STAGE_ORDER: tuple[StageName, ...] = tuple(StageName)

STAGE_TRANSITIONS: dict[StageName, frozenset[StageName]] = {
    StageName.DIRECTOR_PLANNING: frozenset({StageName.DIRECTOR_PLANNING, StageName.STORYBOARD_DESIGN}),
    StageName.STORYBOARD_DESIGN: frozenset({StageName.STORYBOARD_DESIGN, StageName.ASSET_PREPARATION}),
    StageName.ASSET_PREPARATION: frozenset({StageName.ASSET_PREPARATION, StageName.IMAGE_GENERATION}),
    StageName.IMAGE_GENERATION: frozenset({StageName.IMAGE_GENERATION, StageName.QUALITY_REVIEW}),
    StageName.QUALITY_REVIEW: frozenset({StageName.QUALITY_REVIEW, StageName.AUDIO_PRODUCTION}),
    StageName.AUDIO_PRODUCTION: frozenset({StageName.AUDIO_PRODUCTION, StageName.VIDEO_GENERATION}),
    StageName.VIDEO_GENERATION: frozenset({StageName.VIDEO_GENERATION, StageName.VIDEO_REVIEW}),
    StageName.VIDEO_REVIEW: frozenset({StageName.VIDEO_REVIEW, StageName.EDIT_COMPOSITION}),
    StageName.EDIT_COMPOSITION: frozenset({StageName.EDIT_COMPOSITION, StageName.FINAL_REVIEW}),
    StageName.FINAL_REVIEW: frozenset({StageName.FINAL_REVIEW}),
}

STAGE_RECOVERY_TRANSITIONS: dict[StageName, frozenset[StageName]] = {
    StageName.DIRECTOR_PLANNING: frozenset({StageName.DIRECTOR_PLANNING}),
    StageName.STORYBOARD_DESIGN: frozenset({StageName.DIRECTOR_PLANNING, StageName.STORYBOARD_DESIGN}),
    StageName.ASSET_PREPARATION: frozenset({StageName.STORYBOARD_DESIGN, StageName.ASSET_PREPARATION}),
    StageName.IMAGE_GENERATION: frozenset(
        {StageName.STORYBOARD_DESIGN, StageName.ASSET_PREPARATION, StageName.IMAGE_GENERATION}
    ),
    StageName.QUALITY_REVIEW: frozenset(
        {StageName.STORYBOARD_DESIGN, StageName.IMAGE_GENERATION, StageName.QUALITY_REVIEW}
    ),
    StageName.AUDIO_PRODUCTION: frozenset(
        {StageName.STORYBOARD_DESIGN, StageName.IMAGE_GENERATION, StageName.AUDIO_PRODUCTION}
    ),
    StageName.VIDEO_GENERATION: frozenset(
        {
            StageName.STORYBOARD_DESIGN,
            StageName.IMAGE_GENERATION,
            StageName.AUDIO_PRODUCTION,
            StageName.VIDEO_GENERATION,
        }
    ),
    StageName.VIDEO_REVIEW: frozenset(
        {
            StageName.STORYBOARD_DESIGN,
            StageName.IMAGE_GENERATION,
            StageName.AUDIO_PRODUCTION,
            StageName.VIDEO_GENERATION,
            StageName.VIDEO_REVIEW,
        }
    ),
    StageName.EDIT_COMPOSITION: frozenset(
        {
            StageName.IMAGE_GENERATION,
            StageName.AUDIO_PRODUCTION,
            StageName.VIDEO_GENERATION,
            StageName.VIDEO_REVIEW,
            StageName.EDIT_COMPOSITION,
        }
    ),
    StageName.FINAL_REVIEW: frozenset(
        {
            StageName.STORYBOARD_DESIGN,
            StageName.IMAGE_GENERATION,
            StageName.AUDIO_PRODUCTION,
            StageName.VIDEO_GENERATION,
            StageName.VIDEO_REVIEW,
            StageName.EDIT_COMPOSITION,
            StageName.FINAL_REVIEW,
        }
    ),
}

STAGE_STATUS_TRANSITIONS: dict[StageStatus, frozenset[StageStatus]] = {
    StageStatus.PENDING: frozenset({StageStatus.PENDING, StageStatus.RUNNING, StageStatus.SKIPPED, StageStatus.FAILED}),
    StageStatus.RUNNING: frozenset(
        {
            StageStatus.RUNNING,
            StageStatus.RECOVERING,
            StageStatus.SUCCEEDED,
            StageStatus.DEGRADED,
            StageStatus.WAITING_HUMAN,
            StageStatus.FAILED,
        }
    ),
    StageStatus.RECOVERING: frozenset(
        {
            StageStatus.RECOVERING,
            StageStatus.RUNNING,
            StageStatus.SUCCEEDED,
            StageStatus.DEGRADED,
            StageStatus.WAITING_HUMAN,
            StageStatus.FAILED,
        }
    ),
    StageStatus.WAITING_HUMAN: frozenset(
        {
            StageStatus.WAITING_HUMAN,
            StageStatus.RUNNING,
            StageStatus.RECOVERING,
            StageStatus.SUCCEEDED,
            StageStatus.DEGRADED,
            StageStatus.FAILED,
        }
    ),
    StageStatus.SUCCEEDED: frozenset({StageStatus.SUCCEEDED}),
    StageStatus.DEGRADED: frozenset({StageStatus.DEGRADED, StageStatus.RUNNING, StageStatus.RECOVERING}),
    StageStatus.FAILED: frozenset({StageStatus.FAILED, StageStatus.RUNNING, StageStatus.RECOVERING}),
    StageStatus.SKIPPED: frozenset({StageStatus.SKIPPED}),
}


def stage_contract(stage: StageName | str) -> StageContract:
    return STAGE_CONTRACTS[StageName(stage)]


def stage_transition_allowed(
    current: StageName | str | None,
    target: StageName | str,
    *,
    recovery: bool = False,
) -> bool:
    """校验阶段图的合法前进或恢复跳转；禁止跨阶段静默跳跃。"""

    target_stage = StageName(target)
    if current is None or current == "":
        return True
    current_stage = StageName(current)
    transitions = STAGE_RECOVERY_TRANSITIONS if recovery else STAGE_TRANSITIONS
    return target_stage in transitions[current_stage]


def ensure_stage_transition(
    current: StageName | str | None,
    target: StageName | str,
    *,
    recovery: bool = False,
) -> StageName:
    target_stage = StageName(target)
    if not stage_transition_allowed(current, target_stage, recovery=recovery):
        mode = "恢复" if recovery else "运行"
        raise ValueError(f"非法阶段{mode}迁移: {current or 'start'} -> {target_stage.value}")
    return target_stage


def stage_status_transition_allowed(current: StageStatus | str | None, target: StageStatus | str) -> bool:
    current_status = StageStatus(current) if current else StageStatus.PENDING
    target_status = StageStatus(target)
    return target_status in STAGE_STATUS_TRANSITIONS[current_status]


def ensure_stage_status_transition(current: StageStatus | str | None, target: StageStatus | str) -> StageStatus:
    target_status = StageStatus(target)
    if not stage_status_transition_allowed(current, target_status):
        raise ValueError(f"非法阶段状态迁移: {current or 'pending'} -> {target_status.value}")
    return target_status


def transition_allowed(current: RunStatus | str | None, target: RunStatus | str) -> bool:
    """运行状态机：防止已完成任务被恢复逻辑静默改写为运行中。"""

    try:
        current_status = RunStatus(current) if current else RunStatus.PENDING
        target_status = RunStatus(target)
    except ValueError:
        if _is_stage_name(current) or _is_stage_name(target):
            return stage_transition_allowed(current, target)
        return stage_status_transition_allowed(current, target)
    allowed: dict[RunStatus, set[RunStatus]] = {
        RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED},
        RunStatus.RUNNING: {RunStatus.RECOVERING, RunStatus.WAITING_HUMAN, RunStatus.COMPLETED, RunStatus.DEGRADED, RunStatus.FAILED, RunStatus.CANCELLED},
        RunStatus.RECOVERING: {RunStatus.RUNNING, RunStatus.WAITING_HUMAN, RunStatus.COMPLETED, RunStatus.DEGRADED, RunStatus.FAILED, RunStatus.CANCELLED},
        RunStatus.WAITING_HUMAN: {RunStatus.RUNNING, RunStatus.RECOVERING, RunStatus.COMPLETED, RunStatus.DEGRADED, RunStatus.FAILED, RunStatus.CANCELLED},
        RunStatus.DEGRADED: {RunStatus.RUNNING, RunStatus.RECOVERING, RunStatus.WAITING_HUMAN, RunStatus.COMPLETED, RunStatus.FAILED},
        RunStatus.COMPLETED: set(),
        RunStatus.FAILED: {RunStatus.RUNNING, RunStatus.RECOVERING},
        RunStatus.CANCELLED: set(),
    }
    return target_status == current_status or target_status in allowed[current_status]


def ensure_transition(current: RunStatus | str | None, target: RunStatus | str) -> RunStatus | StageName | StageStatus:
    try:
        target_status = RunStatus(target)
    except ValueError:
        if _is_stage_name(current) or _is_stage_name(target):
            return ensure_stage_transition(current, target)
        return ensure_stage_status_transition(current, target)
    if not transition_allowed(current, target_status):
        raise ValueError(f"非法运行状态迁移: {current or 'pending'} -> {target}")
    return target_status


def _is_stage_name(value: StageName | str | None) -> bool:
    if isinstance(value, StageName):
        return True
    try:
        StageName(value)
    except ValueError:
        return False
    return True


def default_quality_profile(value: QualityProfileName | str | None) -> QualityStrategy:
    if not value:
        return QUALITY_STRATEGIES[QualityProfileName.STANDARD]
    return QUALITY_STRATEGIES[QualityProfileName(value)]


__all__ = [
    "CritiqueReport",
    "CriticIssue",
    "MANUAL_ONLY_STRATEGIES",
    "NON_RECOVERABLE_FAILURES",
    "PromptPatch",
    "TERMINAL_STRATEGIES",
    "CheckpointRecord",
    "DecisionTrace",
    "FailureKind",
    "FailureRecord",
    "HumanInterventionPolicy",
    "ProviderCapability",
    "ProviderProfile",
    "PublishPolicy",
    "QualityMetric",
    "QualityProfileName",
    "QualityStrategy",
    "QUALITY_STRATEGIES",
    "RecoveryCandidate",
    "RecoveryStrategy",
    "RunStatus",
    "ShotArtifact",
    "StageContract",
    "StageInput",
    "StageName",
    "StageOutput",
    "StageStatus",
    "STAGE_CONTRACTS",
    "STAGE_ORDER",
    "STAGE_RECOVERY_TRANSITIONS",
    "STAGE_STATUS_TRANSITIONS",
    "STAGE_TRANSITIONS",
    "default_quality_profile",
    "ensure_transition",
    "ensure_stage_status_transition",
    "ensure_stage_transition",
    "stage_contract",
    "stage_status_transition_allowed",
    "stage_transition_allowed",
    "transition_allowed",
    "utc_now",
]
