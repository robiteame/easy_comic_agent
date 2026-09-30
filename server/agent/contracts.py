"""生成 Agent 的稳定契约、状态机词汇与可解释决策数据结构。

本模块只描述数据和规则，不调用数据库或供应商。所有 LangGraph 节点、检查点、
API 追踪和测试都使用这里的枚举/模型，避免不同层各自发明一套状态含义。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StageName(str, Enum):
    DIRECTOR_PLANNING = "director_planning"
    STORYBOARD_DESIGN = "storyboard_design"
    ASSET_PREPARATION = "asset_preparation"
    IMAGE_GENERATION = "image_generation"
    QUALITY_REVIEW = "quality_review"
    VIDEO_GENERATION = "video_generation"
    AUDIO_PRODUCTION = "audio_production"
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
    IMAGE_FAILED = "image_failed"
    VIDEO_FAILED = "video_failed"
    AUDIO_FAILED = "audio_failed"
    DIALOGUE_TOO_LONG = "dialogue_too_long"
    SHOT_TOO_COMPLEX = "shot_too_complex"
    PROVIDER_REFERENCE_UNSUPPORTED = "provider_reference_unsupported"
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


class RecoveryStrategy(str, Enum):
    REVISE_PROMPT = "revise_prompt"
    SWITCH_PROVIDER = "switch_provider"
    SPLIT_SHOT = "split_shot"
    MERGE_SHOTS = "merge_shots"
    REPLACE_REFERENCE = "replace_reference"
    LOWER_RESOLUTION = "lower_resolution"
    REGENERATE_FAILED_SHOTS = "regenerate_failed_shots"
    RESUME_CHECKPOINT = "resume_checkpoint"
    HUMAN_REVIEW = "human_review"
    RETRY = "retry"


class QualityProfileName(str, Enum):
    DRAFT = "draft"
    STANDARD = "standard"
    FINISHING = "finishing"


class QualityStrategy(BaseModel):
    """自动模式的质量档位：成本、候选数、恢复预算与人工介入边界。"""

    model_config = ConfigDict(frozen=True)

    name: QualityProfileName
    label: str
    description: str
    cost_multiplier: float
    candidate_count: int
    max_recovery_attempts: int
    quality_threshold: float
    resolution: Literal["540p", "720p", "1080p", "4k"]
    human_intervention: str
    auto_approve: bool


QUALITY_STRATEGIES: dict[QualityProfileName, QualityStrategy] = {
    QualityProfileName.DRAFT: QualityStrategy(
        name=QualityProfileName.DRAFT,
        label="草稿",
        description="快速验证叙事和镜头可行性，接受较低视觉一致性；成本最低。",
        cost_multiplier=0.6,
        candidate_count=1,
        max_recovery_attempts=1,
        quality_threshold=0.55,
        resolution="540p",
        human_intervention="仅硬失败、预算耗尽或用户请求时介入；可带降级成片结束。",
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
        resolution="720p",
        human_intervention="自动修复失败镜头；连续两轮不合格或需要创意取舍时转人工。",
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
        resolution="1080p",
        human_intervention="关键镜头和最终成片建议人工确认；未确认时只输出 waiting_human。",
        auto_approve=False,
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
    stage: StageName | str
    passed: bool
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    metrics: list[QualityMetric] = Field(default_factory=list)
    issues: list[CriticIssue] = Field(default_factory=list)
    proposed_changes: list[str] = Field(default_factory=list)
    candidate_count: int = 1
    source: str = "deterministic"
    created_at: str = Field(default_factory=utc_now)


class RecoveryCandidate(BaseModel):
    strategy: RecoveryStrategy
    provider: str = ""
    target_stage: StageName | str = ""
    shot_ids: list[str] = Field(default_factory=list)
    prompt_changes: dict[str, Any] = Field(default_factory=dict)
    estimated_cost_micro: int | None = None
    estimated_seconds: int | None = None
    quality_gain: float = 0.0
    provider_capability_ok: bool = True
    budget_fit: bool = True
    score: float = 0.0
    rationale: str = ""


class DecisionTrace(BaseModel):
    """每次 Agent 选择的完整可解释记录。"""

    model_config = ConfigDict(extra="ignore")

    trace_id: str
    run_id: str
    stage: StageName | str
    input_fingerprint: str = ""
    failure: FailureRecord | None = None
    critique: CritiqueReport | None = None
    candidates: list[RecoveryCandidate] = Field(default_factory=list)
    selected: RecoveryCandidate | None = None
    considered_rejected: list[dict[str, Any]] = Field(default_factory=list)
    budget_snapshot: dict[str, Any] = Field(default_factory=dict)
    provider_profiles: list[ProviderProfile] = Field(default_factory=list)
    reason: str = ""
    created_at: str = Field(default_factory=utc_now)


class ShotArtifact(BaseModel):
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
    created_at: str = Field(default_factory=utc_now)


class StageInput(BaseModel):
    project_id: str
    run_id: str
    stage: StageName | str
    input_fingerprint: str = ""
    quality_profile: QualityProfileName = QualityProfileName.STANDARD
    payload: dict[str, Any] = Field(default_factory=dict)
    budget_snapshot: dict[str, Any] = Field(default_factory=dict)
    provider_profiles: list[ProviderProfile] = Field(default_factory=list)


class StageOutput(BaseModel):
    project_id: str
    run_id: str
    stage: StageName | str
    status: StageStatus
    input_fingerprint: str = ""
    output_fingerprint: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    quality: list[QualityMetric] = Field(default_factory=list)
    failure: FailureRecord | None = None
    critique: CritiqueReport | None = None
    shot_artifacts: list[ShotArtifact] = Field(default_factory=list)
    checkpoint_version: int = 1
    created_at: str = Field(default_factory=utc_now)


class StageContract(BaseModel):
    """阶段契约的单一事实源。"""

    stage: StageName
    label: str
    input_model: str
    output_model: str
    quality_metrics: list[str]
    failure_classes: list[FailureKind]
    checkpoint_key: str
    recoverable: bool = True
    fan_out: bool = False
    description: str = ""


STAGE_CONTRACTS: dict[StageName, StageContract] = {
    StageName.DIRECTOR_PLANNING: StageContract(
        stage=StageName.DIRECTOR_PLANNING,
        label="导演规划",
        input_model="StageInput",
        output_model="StageOutput",
        quality_metrics=["schema_valid", "character_coverage", "scene_coverage", "logic_issue_count", "duration_deviation"],
        failure_classes=[FailureKind.LLM_INVALID_OUTPUT, FailureKind.DEPENDENCY_FAILED, FailureKind.TIMEOUT],
        checkpoint_key="director",
        description="解析剧本、人物、场景、叙事目标和资源预算，产出导演意图与候选拍摄计划。",
    ),
    StageName.STORYBOARD_DESIGN: StageContract(
        stage=StageName.STORYBOARD_DESIGN,
        label="分镜设计",
        input_model="StageInput",
        output_model="StageOutput",
        quality_metrics=["shot_count", "shot_rule_compliance", "dialogue_duration_ratio", "continuity_score", "complex_action_score"],
        failure_classes=[FailureKind.LLM_INVALID_OUTPUT, FailureKind.DIALOGUE_TOO_LONG, FailureKind.SHOT_TOO_COMPLEX, FailureKind.QUALITY_BELOW_THRESHOLD],
        checkpoint_key="storyboard",
        description="将导演意图拆成可执行镜头，必要时自动拆分或合并镜头并保留决策记录。",
    ),
    StageName.ASSET_PREPARATION: StageContract(
        stage=StageName.ASSET_PREPARATION,
        label="素材准备",
        input_model="StageInput",
        output_model="StageOutput",
        quality_metrics=["character_reference_coverage", "scene_reference_coverage", "reference_compatibility", "asset_version_match"],
        failure_classes=[FailureKind.PROVIDER_REFERENCE_UNSUPPORTED, FailureKind.PROVIDER_UNAVAILABLE, FailureKind.STORAGE_FAILED, FailureKind.USER_CHANGED_INPUT],
        checkpoint_key="assets",
        description="生成/校验角色三视图、场景基准图和连续性参考，标记 Provider 能力限制。",
    ),
    StageName.IMAGE_GENERATION: StageContract(
        stage=StageName.IMAGE_GENERATION,
        label="图像生成",
        input_model="StageInput",
        output_model="StageOutput",
        quality_metrics=["image_valid", "prompt_alignment", "character_consistency", "composition", "candidate_score"],
        failure_classes=[FailureKind.IMAGE_FAILED, FailureKind.PROVIDER_REFERENCE_UNSUPPORTED, FailureKind.VERSION_CONFLICT, FailureKind.BUDGET_EXCEEDED],
        checkpoint_key="image",
        fan_out=True,
        description="逐镜头独立生成故事板候选；单镜头失败不会回滚已成功镜头。",
    ),
    StageName.QUALITY_REVIEW: StageContract(
        stage=StageName.QUALITY_REVIEW,
        label="质量审核",
        input_model="StageInput",
        output_model="StageOutput",
        quality_metrics=["structural_validity", "continuity", "prompt_alignment", "review_score", "failure_recovery_rate"],
        failure_classes=[FailureKind.QUALITY_BELOW_THRESHOLD, FailureKind.DIALOGUE_TOO_LONG, FailureKind.DEPENDENCY_FAILED],
        checkpoint_key="quality",
        description="Critic/Reviewer 对候选结果给出评分、问题证据、具体修改和恢复建议。",
    ),
    StageName.VIDEO_GENERATION: StageContract(
        stage=StageName.VIDEO_GENERATION,
        label="视频生成",
        input_model="StageInput",
        output_model="StageOutput",
        quality_metrics=["video_valid", "duration_match", "motion_quality", "continuity", "provider_capability"],
        failure_classes=[FailureKind.VIDEO_FAILED, FailureKind.PROVIDER_UNAVAILABLE, FailureKind.VERSION_CONFLICT, FailureKind.BUDGET_EXCEEDED],
        checkpoint_key="video",
        fan_out=True,
        description="逐镜头生成视频候选，支持失败镜头局部补拍、降分辨率和 Provider 切换。",
    ),
    StageName.AUDIO_PRODUCTION: StageContract(
        stage=StageName.AUDIO_PRODUCTION,
        label="音频制作",
        input_model="StageInput",
        output_model="StageOutput",
        quality_metrics=["dialogue_length", "tts_valid", "voice_consistency", "audio_duration", "mix_readiness"],
        failure_classes=[FailureKind.DIALOGUE_TOO_LONG, FailureKind.AUDIO_FAILED, FailureKind.PROVIDER_UNAVAILABLE, FailureKind.VERSION_CONFLICT],
        checkpoint_key="audio",
        fan_out=True,
        description="为对白镜头生成/复用配音，过长对白自动拆句或转人工。",
    ),
    StageName.EDIT_COMPOSITION: StageContract(
        stage=StageName.EDIT_COMPOSITION,
        label="剪辑合成",
        input_model="StageInput",
        output_model="StageOutput",
        quality_metrics=["shot_completeness", "av_sync", "timeline_duration", "render_valid", "degraded_shot_count"],
        failure_classes=[FailureKind.DEPENDENCY_FAILED, FailureKind.STORAGE_FAILED, FailureKind.TIMEOUT, FailureKind.BUDGET_EXCEEDED],
        checkpoint_key="compose",
        description="只消费已完成/明确降级的镜头，生成成片并保留跳过或人工介入状态。",
    ),
    StageName.FINAL_REVIEW: StageContract(
        stage=StageName.FINAL_REVIEW,
        label="成片复审",
        input_model="StageInput",
        output_model="StageOutput",
        quality_metrics=["story_coherence", "visual_consistency", "audio_quality", "pacing", "overall_score", "human_gate"],
        failure_classes=[FailureKind.QUALITY_BELOW_THRESHOLD, FailureKind.USER_CHANGED_INPUT, FailureKind.DEPENDENCY_FAILED],
        checkpoint_key="final",
        description="从成片层面反思节奏、连续性、对白和视觉质量，决定发布、局部重算或转人工。",
    ),
}

STAGE_ORDER: tuple[StageName, ...] = tuple(StageName)


def stage_contract(stage: StageName | str) -> StageContract:
    return STAGE_CONTRACTS[StageName(stage)]


def transition_allowed(current: RunStatus | str | None, target: RunStatus | str) -> bool:
    """运行状态机：防止已完成任务被恢复逻辑静默改写为运行中。"""

    current_status = RunStatus(current) if current else RunStatus.PENDING
    target_status = RunStatus(target)
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


def ensure_transition(current: RunStatus | str | None, target: RunStatus | str) -> RunStatus:
    if not transition_allowed(current, target):
        raise ValueError(f"非法运行状态迁移: {current or 'pending'} -> {target}")
    return RunStatus(target)


def default_quality_profile(value: QualityProfileName | str | None) -> QualityStrategy:
    if not value:
        return QUALITY_STRATEGIES[QualityProfileName.STANDARD]
    return QUALITY_STRATEGIES[QualityProfileName(value)]


__all__ = [
    "CritiqueReport",
    "CriticIssue",
    "DecisionTrace",
    "FailureKind",
    "FailureRecord",
    "ProviderCapability",
    "ProviderProfile",
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
    "default_quality_profile",
    "ensure_transition",
    "stage_contract",
    "transition_allowed",
    "utc_now",
]
