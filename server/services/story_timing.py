"""Story timing planning, provider-duration constraints and render-time validation.

This module owns all decisions that turn narrative timing into executable shots:
speech estimation, action beat segmentation, provider duration capability checks,
long-shot splitting, short-shot merging/extension and final timeline validation.
It intentionally has no FastAPI or SQLAlchemy dependencies so the planning rules
can be regression-tested as a deterministic service.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from config import settings
from services.consistency_service import normalize_continuity_mode

PROVIDER_TIMELINE_TOLERANCE_MS = 500
ACTION_LEAD_RESERVE_MS = 250
COMPLEX_ACTION_MAX_SECONDS = 5.0
DEFAULT_DURATION_STEP_SECONDS = 1.0
MAX_PLANNED_SHOTS = 200

# 执行计划存放在 shots.continuity_profile JSON 的这个键下（复用现有字段，
# 不引入数据库迁移）。
EXECUTION_PLAN_PROFILE_KEY = "execution_plan"
EXECUTION_PLAN_SCHEMA_VERSION = 1
EXECUTION_PLAN_AUDIO_MODES = ("tts", "native")

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
_SENTENCE_BREAK_RE = re.compile(r"[。！？!?；;\n]+")
_CLAUSE_BREAK_RE = re.compile(r"[，,。！？!?；;：:\n]+")
_PAUSE_RE = re.compile(r"[，,、：:；;]")
_SENTENCE_END_RE = re.compile(r"[。！？!?；;\n]")

_COMPLEX_ACTION_MARKERS = (
    "追逐",
    "追击",
    "打斗",
    "搏斗",
    "交手",
    "冲刺",
    "飞奔",
    "奔跑",
    "跑步",
    "跑",
    "翻滚",
    "闪避",
    "跳跃",
    "挥剑",
    "挥拳",
    "踢",
    "摔",
    "转身走位",
    "连续转身",
    "连续转体",
    "连续",
    "翻越",
    "攀爬",
    "躲闪",
    "chase",
    "chasing",
    "fight",
    "fighting",
    "combat",
    "run",
    "running",
    "sprint",
    "roll",
    "vault",
    "dodge",
)
_ACTION_CONJUNCTIONS = ("然后", "接着", "随后", "同时", "之后", "再", "and then", "then", "after that")
ACTION_BEAT_PHASES = ("preparation", "action", "reaction", "continuation")


@dataclass(frozen=True)
class ProviderDurationCapability:
    """Duration contract advertised by the active video provider."""

    protocol: str = ""
    fixed_duration: float | None = None
    min_duration: float = settings.MIN_SHOT_DURATION_SECONDS
    max_duration: float = settings.MAX_SHOT_DURATION_SECONDS
    duration_step: float = DEFAULT_DURATION_STEP_SECONDS
    # 规划侧的视频路由事实；首尾帧能力由视频服务按模型级 capability 裁决。
    reference_mode: str = ""

    def __post_init__(self) -> None:
        fixed = self.fixed_duration
        min_duration = float(fixed if fixed is not None else self.min_duration)
        max_duration = float(fixed if fixed is not None else self.max_duration)
        step = abs(float(fixed if fixed is not None else self.duration_step))
        if not math.isfinite(min_duration) or min_duration <= 0:
            raise ValueError("视频 Provider 最小时长必须大于 0")
        if not math.isfinite(max_duration) or max_duration < min_duration:
            raise ValueError("视频 Provider 最大时长不能小于最小时长")
        if not math.isfinite(step) or step <= 0:
            raise ValueError("视频 Provider 时长步长必须大于 0")
        if fixed is not None and (not math.isfinite(float(fixed)) or float(fixed) <= 0):
            raise ValueError("视频 Provider 固定时长必须大于 0")
        object.__setattr__(self, "fixed_duration", float(fixed) if fixed is not None else None)
        object.__setattr__(self, "min_duration", min_duration)
        object.__setattr__(self, "max_duration", max_duration)
        object.__setattr__(self, "duration_step", step)

    @property
    def is_fixed(self) -> bool:
        return self.fixed_duration is not None

    @property
    def first_frame_only(self) -> bool:
        return str(self.reference_mode or "").strip().lower() == "first_frame_only"

    @property
    def tolerance_s(self) -> float:
        return max(0.5, self.duration_step / 2 + 1e-6)

    def contains(self, duration_s: float) -> bool:
        value = float(duration_s)
        return math.isfinite(value) and value + 1e-6 >= self.min_duration and value - 1e-6 <= self.max_duration

    def is_aligned(self, duration_s: float) -> bool:
        value = float(duration_s)
        units = value / self.duration_step
        return math.isfinite(value) and abs(units - round(units)) <= 1e-6

    def validate(self, duration_s: float, *, shot_id: str = "") -> None:
        value = float(duration_s)
        label = f"镜头 {shot_id}" if shot_id else "镜头"
        if not math.isfinite(value):
            raise StoryTimingError(
                [
                    TimingIssue(
                        code="invalid_shot_duration",
                        message=f"{label} 时长不是有限数值",
                        shot_ids=(shot_id,) if shot_id else (),
                    )
                ]
            )
        if not self.contains(value):
            capability = self.describe()
            raise StoryTimingError(
                [
                    TimingIssue(
                        code="provider_duration_out_of_range",
                        message=(
                            f"{label} 时长 {value:g} 秒不在当前视频 Provider {self.protocol or '<unknown>'} "
                            f"能力范围内；仅允许 {capability}"
                        ),
                        shot_ids=(shot_id,) if shot_id else (),
                    )
                ]
            )
        if not self.is_aligned(value):
            raise StoryTimingError(
                [
                    TimingIssue(
                        code="provider_duration_step_mismatch",
                        message=(
                            f"{label} 时长 {value:g} 秒不符合当前视频 Provider 的 {self.duration_step:g} 秒生成步长"
                        ),
                        shot_ids=(shot_id,) if shot_id else (),
                    )
                ]
            )

    def describe(self) -> str:
        if self.is_fixed:
            return f"固定 {self.fixed_duration:g} 秒"
        return f"{self.min_duration:g} 到 {self.max_duration:g} 秒，步长 {self.duration_step:g} 秒"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def provider_duration_capability(
    protocol: str | None = None,
    *,
    capabilities: object | None = None,
    model: str = "",
) -> ProviderDurationCapability:
    """Read duration limits from a specific video adapter capability declaration."""

    from services.providers.endpoint import get_endpoint
    from services.providers.registry import get_adapter

    endpoint = get_endpoint("video")
    selected_protocol = protocol or endpoint.protocol
    if capabilities is None:
        adapter_cls = get_adapter("video", selected_protocol)
        effective = getattr(adapter_cls, "effective_capabilities", None)
        capabilities = effective(model or endpoint.model) if callable(effective) else adapter_cls.capabilities
    fixed = getattr(capabilities, "fixed_duration", None)
    raw_min = getattr(capabilities, "min_duration", None)
    raw_max = getattr(capabilities, "max_duration", None)
    raw_step = getattr(capabilities, "duration_step", None)
    default_min = float(fixed) if fixed is not None else settings.MIN_SHOT_DURATION_SECONDS
    default_max = float(fixed) if fixed is not None else settings.MAX_SHOT_DURATION_SECONDS
    default_step = float(fixed) if fixed is not None else DEFAULT_DURATION_STEP_SECONDS
    return ProviderDurationCapability(
        protocol=selected_protocol,
        fixed_duration=float(fixed) if fixed is not None else None,
        min_duration=float(raw_min) if raw_min is not None else default_min,
        max_duration=float(raw_max) if raw_max is not None else default_max,
        duration_step=float(raw_step) if raw_step not in (None, 0) else default_step,
        reference_mode=str(getattr(capabilities, "reference_mode", "") or ""),
    )


@dataclass(frozen=True)
class ActionBeat:
    text: str
    complex_motion: bool = False
    phase: str = "continuation"
    entry_state: str = ""
    exit_state: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "phase": self.phase if self.phase in ACTION_BEAT_PHASES else "continuation",
            "complex_motion": bool(self.complex_motion),
            "entry_state": self.entry_state,
            "exit_state": self.exit_state,
        }


@dataclass(frozen=True)
class TimingAdjustment:
    """拆镜/合镜/时长调整的可追溯记录：原因 + 前后结构快照。"""

    code: str
    message: str
    shot_ids: tuple[str, ...] = ()
    reason: str = ""
    # before/after 记录拆合前后的镜头结构（shot_id/duration/version/plan 摘要），
    # 供版本历史与下游失效决策追溯；普通时长调整可为空。
    before: tuple[Mapping[str, Any], ...] = ()
    after: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "shot_ids": list(self.shot_ids),
            "reason": self.reason,
            "before": [dict(item) for item in self.before],
            "after": [dict(item) for item in self.after],
        }


@dataclass(frozen=True)
class TimingIssue:
    code: str
    message: str
    shot_ids: tuple[str, ...] = ()
    track_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "shot_ids": list(self.shot_ids),
            "track_ids": list(self.track_ids),
        }


class StoryTimingError(RuntimeError):
    """A timing contract cannot be satisfied without cutting story content."""

    def __init__(self, issues: Sequence[TimingIssue]) -> None:
        self.issues = list(issues)
        super().__init__("；".join(issue.message for issue in self.issues) or "分镜时长校验失败")


def dialogue_text(value: Any) -> str:
    """Return all spoken text from legacy strings or structured dialogue payloads."""

    if value is None:
        return ""
    if hasattr(value, "line") or hasattr(value, "speaker"):
        return str(getattr(value, "line", "") or "").strip()
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                loaded = json.loads(text)
            except ValueError:
                loaded = None
            if isinstance(loaded, list):
                return dialogue_text(loaded)
        return text
    if isinstance(value, Mapping):
        return str(value.get("line") or value.get("text") or value.get("dialogue") or "").strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(part for part in (dialogue_text(item) for item in value) if part)
    return str(value).strip()


def dialogue_items(value: Any) -> list[dict[str, Any]]:
    """Normalize dialogue into a list of speaker/line dictionaries."""

    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                loaded = json.loads(text)
            except ValueError:
                loaded = None
            if isinstance(loaded, list):
                return dialogue_items(loaded)
        return [{"speaker": "", "line": text, "emotion": "", "action": "", "start_ms": None, "end_ms": None}]
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return [{"speaker": "", "line": str(value), "emotion": "", "action": "", "start_ms": None, "end_ms": None}]
    output: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            line = str(item.get("line") or item.get("text") or item.get("dialogue") or "").strip()
            if not line:
                continue
            output.append(
                {
                    "speaker": str(item.get("speaker") or item.get("character") or item.get("role") or ""),
                    "line": line,
                    "emotion": str(item.get("emotion") or ""),
                    "action": str(item.get("action") or ""),
                    "start_ms": item.get("start_ms"),
                    "end_ms": item.get("end_ms"),
                }
            )
        else:
            line = str(item).strip()
            if line:
                output.append(
                    {"speaker": "", "line": line, "emotion": "", "action": "", "start_ms": None, "end_ms": None}
                )
    return output


def estimate_speech_ms(text: Any) -> int:
    """Estimate TTS duration from CJK characters, Latin words and punctuation pauses."""

    value = dialogue_text(text) if isinstance(text, (Mapping, list, tuple)) else str(text or "").strip()
    if not value:
        return 0
    cjk_chars = len(_CJK_RE.findall(value))
    latin_words = len(_LATIN_WORD_RE.findall(value))
    commas = len(_PAUSE_RE.findall(value))
    sentence_ends = len(_SENTENCE_END_RE.findall(value))
    speech_seconds = cjk_chars / 5.0 + latin_words / 2.6
    speech_seconds += commas * 0.12 + sentence_ends * 0.22
    return max(300, int(round(speech_seconds * 1000)))


def _has_complex_motion(text: Any) -> bool:
    value = str(text or "").strip().lower()
    return any(marker.lower() in value for marker in _COMPLEX_ACTION_MARKERS)


def _coerce_action_beat(value: Any) -> ActionBeat | None:
    if isinstance(value, ActionBeat):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, str):
        text = value.strip()
        return ActionBeat(text=text, complex_motion=_has_complex_motion(text)) if text else None
    if isinstance(value, Mapping):
        text = str(value.get("text") or value.get("action") or value.get("description") or "").strip()
        if not text:
            return None
        phase = str(value.get("phase") or value.get("kind") or "continuation").strip().lower()
        aliases = {
            "prepare": "preparation",
            "setup": "preparation",
            "准备": "preparation",
            "动作": "action",
            "反应": "reaction",
            "续接": "continuation",
        }
        phase = aliases.get(phase, phase)
        return ActionBeat(
            text=text,
            complex_motion=bool(value.get("complex_motion")) or _has_complex_motion(text),
            phase=phase if phase in ACTION_BEAT_PHASES else "continuation",
            entry_state=str(value.get("entry_state") or ""),
            exit_state=str(value.get("exit_state") or ""),
        )
    return None


def normalize_action_beats(value: Any, *, fallback_text: Any = "") -> list[ActionBeat]:
    """Normalize explicit beats; fall back to deterministic action segmentation."""

    explicit: list[ActionBeat] = []
    if isinstance(value, Mapping):
        value = [value]
    if isinstance(value, (list, tuple)):
        explicit = [beat for beat in (_coerce_action_beat(item) for item in value) if beat]
    if explicit:
        return _expand_complex_beats(explicit, source_text=str(fallback_text or ""))
    return estimate_action_beats(fallback_text)


def _inherit_action_states(beats: Sequence[ActionBeat], shot: Mapping[str, Any]) -> list[ActionBeat]:
    entry = str(shot.get("action_entry_state") or "")
    exit_state = str(shot.get("action_exit_state") or "")
    output: list[ActionBeat] = []
    for beat in beats:
        # 镜头级边界状态是拆镜前后的连续性契约；只有节拍自带更细状态时才覆盖。
        beat_entry = beat.entry_state or entry
        beat_exit = beat.exit_state or exit_state
        output.append(
            ActionBeat(
                text=beat.text,
                complex_motion=beat.complex_motion,
                phase=beat.phase,
                entry_state=beat_entry,
                exit_state=beat_exit,
            )
        )
    return output


def _complex_phase_beats(source_text: str, *, entry_state: str = "", exit_state: str = "") -> list[ActionBeat]:
    """Prepare/action/reaction shots for chase, fight, run, roll and turn sequences."""

    source = " ".join(str(source_text or "").split())
    first = re.split(r"[，,。！？!?；;：:\n]+", source)[0].strip() or source
    last = re.split(r"[，,。！？!?；;：:\n]+", source)[-1].strip() or source
    return [
        ActionBeat(
            text=f"准备：调整重心、视线和身体朝向，进入「{first}」的起始状态",
            complex_motion=True,
            phase="preparation",
            entry_state=entry_state,
            exit_state=entry_state,
        ),
        ActionBeat(
            text=f"动作：{source}",
            complex_motion=True,
            phase="action",
            entry_state=entry_state,
            exit_state=exit_state,
        ),
        ActionBeat(
            text=f"反应：完成「{last}」后稳住动作，确认落点并进入退出状态",
            complex_motion=True,
            phase="reaction",
            entry_state=exit_state,
            exit_state=exit_state,
        ),
    ]


def _expand_complex_beats(beats: Sequence[ActionBeat], *, source_text: str) -> list[ActionBeat]:
    if not any(beat.complex_motion for beat in beats):
        return list(beats[:12])
    source = source_text or "；".join(beat.text for beat in beats)
    phases = {beat.phase for beat in beats}
    if {"preparation", "action", "reaction"}.issubset(phases):
        return list(beats[:12])
    return _complex_phase_beats(
        source,
        entry_state=next((beat.entry_state for beat in beats if beat.entry_state), ""),
        exit_state=next((beat.exit_state for beat in reversed(beats) if beat.exit_state), ""),
    )


def estimate_action_beats(text: Any) -> list[ActionBeat]:
    """Split an action description into one-narrative-beat segments."""

    value = str(text or "").strip()
    if not value:
        return []
    normalized = value
    for conjunction in _ACTION_CONJUNCTIONS:
        normalized = normalized.replace(conjunction, "，")
    pieces = [piece.strip() for piece in _CLAUSE_BREAK_RE.split(normalized) if piece.strip()]
    if not pieces:
        pieces = [value]
    beats = [
        ActionBeat(
            text=piece,
            complex_motion=_has_complex_motion(piece),
            phase="action",
        )
        for piece in pieces[:12]
    ]
    return _expand_complex_beats(beats, source_text=value)


def action_is_complex(text: Any) -> bool:
    beats = estimate_action_beats(text)
    return len(beats) >= 2 or any(beat.complex_motion for beat in beats)


def shot_speech_ms(shot: Mapping[str, Any]) -> int:
    """Return the deterministic speech estimate for the current dialogue text.

    Persisted ``estimated_speech_ms`` is metadata, not an authority: stale values
    must never hide a dialogue/timeline conflict after an edit.
    """

    return estimate_speech_ms(dialogue_text(shot.get("dialogue")))


def usable_speech_ms(shot: Mapping[str, Any], duration_s: float | None = None) -> int:
    duration_ms = int(round(float(duration_s if duration_s is not None else shot.get("duration") or 0) * 1000))
    reserve = ACTION_LEAD_RESERVE_MS if str(shot.get("character_action") or "").strip() else 0
    return max(0, duration_ms - reserve)


@dataclass(frozen=True)
class DialogueTiming:
    """单句对白的实际时间轴（相对镜头起点，毫秒）。"""

    speaker: str = ""
    text: str = ""
    start_ms: int = 0
    end_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }

    @classmethod
    def from_mapping(cls, item: Any) -> DialogueTiming | None:
        if not isinstance(item, Mapping):
            return None
        text = str(item.get("text") or item.get("line") or item.get("dialogue") or "").strip()
        if not text:
            return None
        try:
            start_ms = max(0, int(round(float(item.get("start_ms") or 0))))
            end_ms = max(0, int(round(float(item.get("end_ms") or 0))))
        except (TypeError, ValueError):
            return None
        return cls(speaker=str(item.get("speaker") or ""), text=text, start_ms=start_ms, end_ms=end_ms)


@dataclass(frozen=True)
class ShotExecutionPlan:
    """镜头统一执行计划：故事时间、Provider 生成时长与最终剪辑区间同源。

    视频 Prompt 的有效时长、TTS 实测对白时间轴和后期合成时间线都消费同一份
    计划，避免各阶段从 ``duration`` / ``estimated_speech_ms`` 等旧字段各自推导
    出不一致的结果。固定档 Provider（如固定 5 秒）允许生成满固定秒数，再在
    成片中按 ``trim_start_ms`` / ``trim_end_ms`` 裁剪为更短的故事时长；TTS 实测
    对白超出故事时间但在生成片段内时，剪辑区间延伸到实测对白结束。旧数据没有
    计划时由 :meth:`derive` 从旧字段推导，行为与历史路径一致。
    """

    shot_id: str
    narrative_duration_ms: int
    provider_generation_duration_s: float
    trim_start_ms: int = 0
    trim_end_ms: int = 0
    audio_mode: str = "tts"
    # tts_measured（本轮 TTS 实测）| stored（库内既有时间戳）| native_prompt
    # （编入视频 Prompt 的时间）| none（无对白时间轴）
    dialogue_timing_source: str = "none"
    dialogue_timing: tuple[DialogueTiming, ...] = ()
    continuity_mode: str = "independent"
    video_mode: str = "text_only"
    # 本镜头计划并行生成的候选数与允许的自动恢复次数（质量档位决定）。
    candidate_count: int = 1
    # 镜头执行所需的 Provider 能力（first_frame/character_identity/...）；
    # 视频路由按可加载参考素材给出权威清单后回写计划。
    required_capabilities: tuple[str, ...] = ()
    recovery_budget: int = 0
    provider: str = ""
    recipe_hash: str = ""
    warnings: tuple[str, ...] = ()
    schema_version: int = EXECUTION_PLAN_SCHEMA_VERSION

    @property
    def generation_duration_ms(self) -> int:
        return int(round(self.provider_generation_duration_s * 1000))

    @property
    def effective_duration_ms(self) -> int:
        """最终剪辑区间长度，即镜头在成片时间线上的实际时长。"""

        return max(0, self.trim_end_ms - max(0, self.trim_start_ms))

    @property
    def dialogue_end_ms(self) -> int:
        return max((item.end_ms for item in self.dialogue_timing), default=0)

    @property
    def has_measured_dialogue(self) -> bool:
        return self.dialogue_timing_source in {"tts_measured", "stored"} and bool(self.dialogue_timing)

    def to_dict(self) -> dict[str, Any]:
        dialogue_timing = [item.to_dict() for item in self.dialogue_timing]
        return {
            "schema_version": self.schema_version,
            "shot_id": self.shot_id,
            "narrative_duration_ms": self.narrative_duration_ms,
            "provider_generation_duration_s": self.provider_generation_duration_s,
            "trim_start_ms": self.trim_start_ms,
            "trim_end_ms": self.trim_end_ms,
            "audio_mode": self.audio_mode,
            "dialogue_timing_source": self.dialogue_timing_source,
            # actual_dialogue_timing 是对外契约名；dialogue_timing 旧键保留，
            # 两处始终同值，旧消费者不受影响。
            "actual_dialogue_timing": dialogue_timing,
            "dialogue_timing": dialogue_timing,
            "continuity_mode": self.continuity_mode,
            "video_mode": self.video_mode,
            "candidate_count": self.candidate_count,
            "required_capabilities": list(self.required_capabilities),
            "recovery_budget": self.recovery_budget,
            "provider": self.provider,
            "recipe_hash": self.recipe_hash,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_mapping(cls, payload: Any) -> ShotExecutionPlan | None:
        """解析持久化计划；无法识别/非法时返回 None（调用方走旧字段推导）。"""

        if not isinstance(payload, Mapping):
            return None
        try:
            narrative_ms = int(round(float(payload.get("narrative_duration_ms"))))
            generation_s = float(payload.get("provider_generation_duration_s"))
        except (TypeError, ValueError):
            return None
        if narrative_ms <= 0 or not math.isfinite(generation_s) or generation_s <= 0:
            return None

        def _int_ms(value: Any, default: int) -> int:
            try:
                return max(0, int(round(float(value))))
            except (TypeError, ValueError):
                return default

        trim_start = _int_ms(payload.get("trim_start_ms"), 0)
        trim_end = _int_ms(payload.get("trim_end_ms"), narrative_ms)
        raw_timing = payload.get("actual_dialogue_timing")
        if raw_timing is None:
            raw_timing = payload.get("dialogue_timing")
        timings = tuple(filter(None, (DialogueTiming.from_mapping(item) for item in (raw_timing or []))))
        audio_mode = str(payload.get("audio_mode") or "").strip().lower()
        if audio_mode not in EXECUTION_PLAN_AUDIO_MODES:
            audio_mode = "tts"
        try:
            schema_version = int(payload.get("schema_version") or 0)
        except (TypeError, ValueError):
            schema_version = 0
        try:
            candidate_count = max(1, int(payload.get("candidate_count") or 1))
        except (TypeError, ValueError):
            candidate_count = 1
        try:
            recovery_budget = max(0, int(payload.get("recovery_budget") or 0))
        except (TypeError, ValueError):
            recovery_budget = 0
        return cls(
            shot_id=str(payload.get("shot_id") or ""),
            narrative_duration_ms=narrative_ms,
            provider_generation_duration_s=generation_s,
            trim_start_ms=trim_start,
            trim_end_ms=max(trim_start, trim_end),
            audio_mode=audio_mode,
            dialogue_timing_source=str(payload.get("dialogue_timing_source") or ("stored" if timings else "none")),
            dialogue_timing=timings,
            continuity_mode=normalize_continuity_mode(payload.get("continuity_mode")),
            video_mode=str(payload.get("video_mode") or "text_only"),
            candidate_count=candidate_count,
            required_capabilities=tuple(
                dict.fromkeys(str(item) for item in (payload.get("required_capabilities") or []) if str(item).strip())
            ),
            recovery_budget=recovery_budget,
            provider=str(payload.get("provider") or ""),
            recipe_hash=str(payload.get("recipe_hash") or ""),
            warnings=tuple(str(item) for item in (payload.get("warnings") or [])),
            schema_version=schema_version or EXECUTION_PLAN_SCHEMA_VERSION,
        )

    @classmethod
    def derive(
        cls,
        shot: Mapping[str, Any],
        *,
        provider: ProviderDurationCapability | None = None,
        audio_mode: str = "",
        dialogue_timing: Sequence[Any] | None = None,
        dialogue_timing_source: str = "",
        shot_id: str = "",
        candidate_count: int = 0,
        required_capabilities: Sequence[str] | None = None,
        recovery_budget: int = -1,
    ) -> ShotExecutionPlan:
        """从镜头字段（或旧数据）推导执行计划。

        ``dialogue_timing`` 传入本轮 TTS 实测时间轴时（``tts_measured``），实测
        对白决定剪辑下限：故事时间在生成片段容量内延伸到实测结束，超出容量则
        封顶并记录 warning。其余情况从库内对白时间戳（``stored``）或空时间轴
        推导，行为与历史 duration 语义一致。``candidate_count`` /
        ``recovery_budget`` 来自质量档位；``required_capabilities`` 未显式给出
        时按镜头内容确定性推导（视频路由会按可加载素材覆写为权威清单）。
        """

        shot_id = shot_id or str(shot.get("shot_id") or shot.get("id") or "")
        try:
            narrative_ms = max(1, int(round(float(shot.get("duration") or 3.0) * 1000)))
        except (TypeError, ValueError):
            narrative_ms = 3000

        profile = shot.get("continuity_profile") if isinstance(shot.get("continuity_profile"), Mapping) else {}
        timing = shot.get("timing") if isinstance(shot.get("timing"), Mapping) else {}

        source = str(dialogue_timing_source or "").strip().lower()
        timings: list[DialogueTiming] = []
        if dialogue_timing is not None:
            timings = [item for item in (DialogueTiming.from_mapping(entry) for entry in dialogue_timing) if item]
            if not source:
                source = "tts_measured" if timings else "none"
        else:
            raw = shot.get("dialogue_timing")
            if raw is None:
                raw = shot.get("dialogue")
            timings = [
                item
                for item in (
                    DialogueTiming.from_mapping(entry)
                    if entry.get("start_ms") is not None and entry.get("end_ms") is not None
                    else None
                    for entry in dialogue_items(raw)
                )
                if item
            ]
            if not source:
                source = "stored" if timings else ""
        dialogue_end_ms = max((item.end_ms for item in timings), default=0)

        mode = str(audio_mode or "").strip().lower()
        if not mode:
            mode = (
                str(
                    shot.get("audio_mode")
                    or timing.get("audio_mode")
                    or timing.get("audio_source")
                    or profile.get("audio_mode")
                    or profile.get("audio_source")
                    or ""
                )
                .strip()
                .lower()
            )
        if mode not in EXECUTION_PLAN_AUDIO_MODES:
            mode = "tts"

        # continuity_mode 的五种规范值由一致性策略统一归一化；旧计划里的
        # previous_final_frame 映射为 continuous_action，其余旧控制源回落独立镜头。
        continuity_mode = normalize_continuity_mode(
            shot.get("continuity_mode") or profile.get("continuity_mode") or profile.get("control_source"),
            default="independent",
        )

        video_mode = str(shot.get("video_mode") or profile.get("reference_mode") or "").strip().lower()
        if not video_mode:
            video_mode = (
                "first_frame_reference" if (shot.get("storyboard_path") or shot.get("image_path")) else "text_only"
            )

        warnings: list[str] = []
        needed_ms = narrative_ms
        if mode == "tts" and source == "tts_measured" and dialogue_end_ms > needed_ms:
            needed_ms = dialogue_end_ms
            warnings.append("narrative_extended_for_dialogue")

        try:
            requested_generation_s = float(shot.get("generation_duration_s") or 0)
        except (TypeError, ValueError):
            requested_generation_s = 0.0
        if provider is not None and provider.is_fixed:
            generation_s = float(provider.fixed_duration or 0) or needed_ms / 1000
        elif provider is not None:
            step_ms = max(1, int(round(provider.duration_step * 1000)))
            snapped_ms = int(math.ceil(needed_ms / step_ms - 1e-6)) * step_ms
            generation_s = min(max(snapped_ms / 1000, provider.min_duration), provider.max_duration)
        else:
            generation_s = requested_generation_s if requested_generation_s > 0 else needed_ms / 1000
        generation_s = round(max(generation_s, 0.001), 3)
        generation_ms = int(round(generation_s * 1000))

        trim_start_ms = 0
        trim_end_ms = min(generation_ms, needed_ms)
        if provider is not None and not provider.is_fixed:
            # 非固定档的最终时长必须落在步长网格上，否则扩展后的 duration
            # 会在下一次生成入口被步长校验拒绝。
            step_ms = max(1, int(round(provider.duration_step * 1000)))
            trim_end_ms = min(generation_ms, int(math.ceil(trim_end_ms / step_ms - 1e-6)) * step_ms)
        if narrative_ms > generation_ms:
            warnings.append("narrative_exceeds_provider_clip")
        if mode == "tts" and source == "tts_measured" and dialogue_end_ms > generation_ms:
            warnings.append("dialogue_exceeds_provider_clip")

        recipe_payload = {
            "schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
            "shot_id": shot_id,
            "narrative_duration_ms": narrative_ms,
            "provider_generation_duration_s": generation_s,
            "audio_mode": mode,
            "continuity_mode": continuity_mode,
            "video_mode": video_mode,
            "provider": str(provider.protocol if provider is not None else ""),
            "dialogue": [
                {"speaker": str(item.get("speaker") or ""), "text": str(item.get("line") or "")}
                for item in dialogue_items(shot.get("dialogue"))
            ],
        }
        recipe_hash = hashlib.sha256(
            json.dumps(recipe_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        try:
            planned_candidates = max(1, int(candidate_count or shot.get("candidate_count") or 1))
        except (TypeError, ValueError):
            planned_candidates = 1
        try:
            planned_recovery = int(
                recovery_budget
                if recovery_budget >= 0
                else (shot.get("recovery_budget") if shot.get("recovery_budget") is not None else 0)
            )
        except (TypeError, ValueError):
            planned_recovery = 0
        if required_capabilities is None:
            capabilities = tuple(plan_required_capabilities(shot))
        else:
            capabilities = tuple(dict.fromkeys(str(item) for item in required_capabilities if str(item).strip()))
        return cls(
            shot_id=shot_id,
            # narrative 保持真实故事时间（对白超时时为延展后的需要值）：入口
            # 校验靠它拒绝超出 Provider 能力的镜头；生成容量上限只约束 trim。
            narrative_duration_ms=needed_ms,
            provider_generation_duration_s=generation_s,
            trim_start_ms=trim_start_ms,
            trim_end_ms=trim_end_ms,
            audio_mode=mode,
            dialogue_timing_source=source or "none",
            dialogue_timing=tuple(timings),
            continuity_mode=continuity_mode,
            video_mode=video_mode,
            candidate_count=planned_candidates,
            required_capabilities=capabilities,
            recovery_budget=max(0, planned_recovery),
            provider=str(provider.protocol if provider is not None else ""),
            recipe_hash=recipe_hash,
            warnings=tuple(dict.fromkeys(warnings)),
        )


def plan_required_capabilities(shot: Mapping[str, Any]) -> list[str]:
    """按镜头内容确定性推导执行所需的 Provider 能力清单。

    规划阶段（分镜/音频）尚无可加载素材，这里只根据镜头语义给出需求：
    已审核分镜首帧是视频链路硬性要求；场内角色需要角色身份参考；有场景组
    需要场景基准；continuous_action 需要上一镜尾帧作为连续性参考；任一非
    首帧参考都意味着多参考能力。视频生成时按实际可加载素材覆写为权威清单。
    """

    required: list[str] = ["first_frame"]
    profile = shot.get("continuity_profile") if isinstance(shot.get("continuity_profile"), Mapping) else {}
    characters = [str(item) for item in (shot.get("characters_in_scene") or []) if str(item).strip()]
    has_scene = bool(
        shot.get("scene_asset_id")
        or shot.get("scene_group_id")
        or shot.get("scene_number")
        or str(shot.get("scene_description") or "").strip()
    )
    continuity_mode = normalize_continuity_mode(
        shot.get("continuity_mode") or profile.get("continuity_mode"),
        default="independent",
    )
    if characters:
        required.append("character_identity")
    if has_scene:
        required.append("scene_reference")
    extra_references = len(required) > 1 or continuity_mode == "continuous_action"
    if extra_references:
        required.append("multiple_reference_images")
    return list(dict.fromkeys(required))


def load_shot_execution_plan(shot: Mapping[str, Any]) -> ShotExecutionPlan | None:
    """读取已持久化的执行计划；旧数据没有计划时返回 None。"""

    candidates: list[Any] = [shot.get(EXECUTION_PLAN_PROFILE_KEY)]
    profile = shot.get("continuity_profile")
    if isinstance(profile, Mapping):
        candidates.append(profile.get(EXECUTION_PLAN_PROFILE_KEY))
    timing = shot.get("timing")
    if isinstance(timing, Mapping):
        candidates.append(timing.get(EXECUTION_PLAN_PROFILE_KEY))
    for payload in candidates:
        plan = ShotExecutionPlan.from_mapping(payload)
        if plan is not None:
            return plan
    return None


def resolve_shot_execution_plan(
    shot: Mapping[str, Any],
    *,
    provider: ProviderDurationCapability | None = None,
    audio_mode: str = "",
    dialogue_timing: Sequence[Any] | None = None,
    dialogue_timing_source: str = "",
    shot_id: str = "",
    candidate_count: int = 0,
    required_capabilities: Sequence[str] | None = None,
    recovery_budget: int = -1,
) -> ShotExecutionPlan:
    """优先使用持久化执行计划（后期合成与校验的单一事实源）。

    本轮有新的实测对白时间轴（``dialogue_timing``）、显式音频路由或显式
    执行参数（候选数/能力/恢复预算）时必须重新推导；否则直接复用已落库
    计划，旧数据（无计划）从旧字段推导。
    """

    if (
        dialogue_timing is None
        and not audio_mode
        and not candidate_count
        and required_capabilities is None
        and recovery_budget < 0
    ):
        persisted = load_shot_execution_plan(shot)
        if persisted is not None:
            return persisted
    return ShotExecutionPlan.derive(
        shot,
        provider=provider,
        audio_mode=audio_mode,
        dialogue_timing=dialogue_timing,
        dialogue_timing_source=dialogue_timing_source,
        shot_id=shot_id,
        candidate_count=candidate_count,
        required_capabilities=required_capabilities,
        recovery_budget=recovery_budget,
    )


def _unique_id(existing: Iterable[str], base: str) -> str:
    used = set(existing)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _base_shot_id(shot_id: str) -> str:
    return re.sub(r"_part_\d+$", "", str(shot_id or ""))


def _split_text(text: str, parts: int) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return [""] * parts
    clauses = [item.strip() for item in _CLAUSE_BREAK_RE.split(value) if item.strip()]
    if len(clauses) >= parts:
        groups: list[list[str]] = [[] for _ in range(parts)]
        for index, clause in enumerate(clauses):
            groups[min(index * parts // len(clauses), parts - 1)].append(clause)
        return ["；".join(group) for group in groups]
    if parts <= 1:
        return [value]
    # Character-level partitioning preserves all dialogue/action content when the
    # model returned one long run-on sentence. It is preferable to dropping text.
    size = max(1, math.ceil(len(value) / parts))
    chunks = [value[index * size : (index + 1) * size] for index in range(parts)]
    while len(chunks) < parts:
        chunks.append("")
    return chunks


def _split_dialogue_by_speech(value: Any, parts: int) -> list[list[dict[str, Any]]]:
    items = dialogue_items(value)
    groups: list[list[dict[str, Any]]] = [[] for _ in range(parts)]
    if not items:
        return groups
    target_ms = sum(estimate_speech_ms(item.get("line")) for item in items) / parts
    cursor = 0
    for item in items:
        groups[cursor].append(dict(item))
        if sum(estimate_speech_ms(line.get("line")) for line in groups[cursor]) >= target_ms:
            cursor = min(parts - 1, cursor + 1)

    # A single utterance can still overflow one part. Split only that utterance at
    # character boundaries while retaining its speaker/emotion; never drop words.
    output: list[list[dict[str, Any]]] = [[] for _ in range(parts)]
    for index, group in enumerate(groups):
        if sum(estimate_speech_ms(line.get("line")) for line in group) <= target_ms * 1.25 or len(group) != 1:
            output[index].extend(group)
            continue
        item = group[0]
        chunks = _split_text(item.get("line") or "", parts)
        for part_index, chunk in enumerate(chunks):
            if not chunk:
                continue
            split_item = dict(item)
            split_item["line"] = chunk
            split_item["start_ms"] = None
            split_item["end_ms"] = None
            output[min(part_index, parts - 1)].append(split_item)
    return output


def _structure_snapshot(shot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "shot_id": str(shot.get("shot_id") or shot.get("id") or ""),
        "duration": float(shot.get("duration") or 0),
        "version": int(shot.get("version") or 1),
    }


def _split_timing_metadata(
    shot: Mapping[str, Any],
    part_number: int,
    part_count: int,
    beat: ActionBeat | None = None,
    *,
    reason: str = "",
    before: Mapping[str, Any] | None = None,
    part_shot_id: str = "",
) -> dict[str, Any]:
    timing = dict(shot.get("timing") or {})
    timing.update(
        {
            "split_from": str(shot.get("shot_id") or shot.get("id") or ""),
            "split_part": part_number,
            "split_total": part_count,
            "narrative_continuation": True,
            "action_phase": beat.phase if beat else "continuation",
            "action_entry_state": str(
                (beat.entry_state if beat else "")
                or shot.get("action_entry_state")
                or timing.get("action_entry_state")
                or ""
            ),
            "action_exit_state": str(
                (beat.exit_state if beat else "")
                or shot.get("action_exit_state")
                or timing.get("action_exit_state")
                or ""
            ),
            "gaze_direction": str(shot.get("gaze_direction") or timing.get("gaze_direction") or ""),
            "screen_axis": str(shot.get("screen_axis") or timing.get("screen_axis") or ""),
        }
    )
    if reason:
        # 拆镜审计：原因 + 拆镜前后结构（id/时长/版本），与 timing_plan 调整记录同源。
        fallback_id = f"{_base_shot_id(str(shot.get('shot_id') or shot.get('id') or 'shot'))}_part_{part_number:02d}"
        timing["structure_change"] = {
            "kind": "split",
            "reason": reason,
            "before": dict(before if before is not None else _structure_snapshot(shot)),
            "after": {
                "shot_id": str(part_shot_id or fallback_id),
                "part": part_number,
                "part_count": part_count,
                "version": int(shot.get("version") or 1),
            },
        }
    return timing


def split_shot(
    shot: Mapping[str, Any],
    parts: int,
    *,
    existing_ids: Iterable[str] = (),
    reason: str = "",
) -> list[dict[str, Any]]:
    """Split one shot into continuous one-beat parts without dropping continuity state.

    ``reason`` 记录拆镜原因（复杂动作/时长上限/对白容量/目标预算），连同拆镜
    前的结构快照写入每个子镜头的 ``timing.structure_change``，供版本历史与
    下游失效决策追溯。
    """

    requested_count = max(2, int(parts))
    base_id = _base_shot_id(str(shot.get("shot_id") or shot.get("id") or "shot"))
    before = _structure_snapshot(shot)
    beats = _inherit_action_states(
        normalize_action_beats(shot.get("action_beats"), fallback_text=shot.get("character_action")),
        shot,
    )
    if beats:
        if len(beats) == 1:
            text = beats[0].text
            beats = (
                [
                    ActionBeat(f"准备：进入「{text}」起始状态", True, "preparation"),
                    ActionBeat(f"动作：{text}", True, "action"),
                    ActionBeat(f"反应：完成「{text}」并进入退出状态", True, "reaction"),
                ]
                if beats[0].complex_motion
                else beats
            )
        count = max(requested_count, len(beats), 3 if any(item.complex_motion for item in beats) else 1)
    else:
        count = requested_count
    while len(beats) < count:
        beats.append(
            ActionBeat(
                text=f"动作延续：保持并推进「{shot.get('character_action') or shot.get('scene_description') or '当前动作'}」",
                complex_motion=any(item.complex_motion for item in beats),
                phase="continuation",
            )
        )
    beats = beats[:count]
    actions = [item.text for item in beats]
    dialogues = _split_dialogue_by_speech(shot.get("dialogue"), count)
    source_duration = float(shot.get("duration") or 0)
    part_duration = round(source_duration / count, 3) if source_duration > 0 else 0.0
    used = set(existing_ids)
    output: list[dict[str, Any]] = []
    for index in range(count):
        item = dict(shot)
        shot_id = f"{base_id}_part_{index + 1:02d}"
        shot_id = _unique_id(used, shot_id)
        used.add(shot_id)
        item["shot_id"] = shot_id
        # 场景、角色、视线、轴线及动作边界在所有拆分镜头中保持同一份语义，
        # 不把场景描述切成互不完整的碎片。
        item["scene_description"] = str(shot.get("scene_description") or "")
        item["characters_in_scene"] = list(shot.get("characters_in_scene") or [])
        item["gaze_direction"] = str(shot.get("gaze_direction") or "")
        item["screen_axis"] = str(shot.get("screen_axis") or "")
        item["action_entry_state"] = str(beats[index].entry_state or shot.get("action_entry_state") or "")
        item["action_exit_state"] = str(beats[index].exit_state or shot.get("action_exit_state") or "")
        item["character_action"] = actions[index]
        item["action_beats"] = [beats[index].to_dict()]
        item["duration"] = part_duration
        item["dialogue"] = dialogues[index]
        item["estimated_speech_ms"] = estimate_speech_ms(dialogue_text(dialogues[index]))
        item["timing"] = _split_timing_metadata(
            shot,
            index + 1,
            count,
            beats[index],
            reason=reason,
            before=before,
            part_shot_id=shot_id,
        )
        item["continuity_mode"] = "continuous_action"
        item["continuity_mode_source"] = (
            "complex_action_split" if any(b.complex_motion for b in beats) else "action_beat_split"
        )
        output.append(item)
    return output


def _same_story_location(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_scene = str(left.get("scene_asset_id") or left.get("scene_group_id") or left.get("scene_number") or "")
    right_scene = str(right.get("scene_asset_id") or right.get("scene_group_id") or right.get("scene_number") or "")
    return bool(left_scene and left_scene == right_scene)


def _mergeable(left: Mapping[str, Any], right: Mapping[str, Any], capability: ProviderDurationCapability) -> bool:
    if not _same_story_location(left, right):
        return False
    left_timing = left.get("timing") or {}
    right_timing = right.get("timing") or {}
    if left_timing.get("split_total") or right_timing.get("split_total"):
        return False
    left_action = str(left.get("character_action") or "").strip()
    right_action = str(right.get("character_action") or "").strip()
    if (left_action or right_action) and left_action != right_action:
        return False
    combined_action = left_action or right_action
    if action_is_complex(combined_action):
        return False
    combined_dialogue = dialogue_text([*dialogue_items(left.get("dialogue")), *dialogue_items(right.get("dialogue"))])
    combined_duration = float(left.get("duration") or 0) + float(right.get("duration") or 0)
    if estimate_speech_ms(combined_dialogue) > usable_speech_ms(
        {"character_action": combined_action}, combined_duration
    ):
        return False
    return combined_duration <= capability.max_duration + 1e-6


def _mergeable_for_budget(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    capability: ProviderDurationCapability,
) -> bool:
    """Budget-count fallback: merge only simple adjacent beats, never complex motion/dialogue."""

    left_timing = left.get("timing") or {}
    right_timing = right.get("timing") or {}
    if left_timing.get("split_total") or right_timing.get("split_total"):
        return False
    left_action = str(left.get("character_action") or "").strip()
    right_action = str(right.get("character_action") or "").strip()
    if (left_action or right_action) and left_action != right_action:
        return False
    combined_action = left_action or right_action
    combined_beats = estimate_action_beats(combined_action)
    if any(beat.complex_motion for beat in combined_beats) or len(combined_beats) > 1:
        return False
    combined_dialogue = dialogue_text([*dialogue_items(left.get("dialogue")), *dialogue_items(right.get("dialogue"))])
    if estimate_speech_ms(combined_dialogue) > capability.max_duration * 1000 - ACTION_LEAD_RESERVE_MS:
        return False
    # Cross-scene cuts are allowed only for dialogue-free simple beats when the
    # fixed-duration budget makes the shot count otherwise infeasible.
    return _same_story_location(left, right) or not combined_dialogue


def merge_shots(left: Mapping[str, Any], right: Mapping[str, Any], *, reason: str = "") -> dict[str, Any]:
    """Merge two adjacent story beats while retaining all source text.

    ``reason`` 记录合镜原因；合并前后结构写入 ``timing.structure_change``，
    与拆镜审计同源。
    """

    item = dict(left)
    right_id = str(right.get("shot_id") or right.get("id") or "")
    left_id = str(left.get("shot_id") or left.get("id") or "")
    item["shot_id"] = left_id
    item["scene_description"] = "；".join(
        dict.fromkeys(
            part
            for part in (str(left.get("scene_description") or ""), str(right.get("scene_description") or ""))
            if part
        )
    )
    left_action = str(left.get("character_action") or "").strip()
    right_action = str(right.get("character_action") or "").strip()
    item["character_action"] = left_action or right_action
    left_beats = normalize_action_beats(left.get("action_beats"), fallback_text=left_action)
    right_beats = normalize_action_beats(right.get("action_beats"), fallback_text=right_action)
    item["action_beats"] = [(left_beats or right_beats or [ActionBeat("", False, "continuation")])[0].to_dict()]
    item["dialogue"] = [*dialogue_items(left.get("dialogue")), *dialogue_items(right.get("dialogue"))]
    item["duration"] = round(float(left.get("duration") or 0) + float(right.get("duration") or 0), 3)
    item["transition"] = right.get("transition") or left.get("transition") or "cut"
    item["estimated_speech_ms"] = estimate_speech_ms(dialogue_text(item["dialogue"]))
    timing = dict(left.get("timing") or {})
    merged_from = [str(value) for value in timing.get("merged_from", []) if value]
    merged_from.extend([left_id, right_id])
    timing["merged_from"] = list(dict.fromkeys(merged_from))
    if reason:
        timing["structure_change"] = {
            "kind": "merge",
            "reason": reason,
            "before": [_structure_snapshot(left), _structure_snapshot(right)],
            "after": {
                "shot_id": left_id,
                "duration": float(item["duration"]),
                "version": int(left.get("version") or 1),
                "merged_from": [left_id, right_id],
            },
        }
    item["timing"] = timing
    return item


def _shot_weight(shot: Mapping[str, Any]) -> float:
    speech_weight = shot_speech_ms(shot) / 1000.0
    action_weight = len(estimate_action_beats(shot.get("character_action"))) * 0.45
    if action_is_complex(shot.get("character_action")):
        action_weight += 0.8
    return max(0.8, speech_weight + action_weight)


@dataclass
class StoryTimingPlan:
    """Narrative timing budget for one project storyboard."""

    target_duration_s: float
    provider: ProviderDurationCapability
    dialogue_duration_s: float = 0.0
    action_beats: list[ActionBeat] = field(default_factory=list)
    shot_count: int = 0
    planned_total_duration_s: float = 0.0
    adjustments: list[TimingAdjustment] = field(default_factory=list)

    @classmethod
    def from_shots(
        cls,
        target_duration_s: float,
        shots: Sequence[Mapping[str, Any]],
        provider: ProviderDurationCapability,
    ) -> StoryTimingPlan:
        plan = cls(target_duration_s=float(target_duration_s), provider=provider)
        plan.rebalance(shots)
        return plan

    @property
    def tolerance_s(self) -> float:
        return self.provider.tolerance_s

    @property
    def target_feasible(self) -> bool:
        return abs(self.planned_total_duration_s - self.target_duration_s) <= self.tolerance_s + 1e-6

    def _record_adjustment(
        self,
        code: str,
        message: str,
        shot_ids: Sequence[str] = (),
        *,
        reason: str = "",
        before: Sequence[Mapping[str, Any]] = (),
        after: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.adjustments.append(
            TimingAdjustment(
                code=code,
                message=message,
                shot_ids=tuple(str(item) for item in shot_ids if item),
                reason=reason or code,
                before=tuple(before),
                after=tuple(after),
            )
        )

    def _normalize_segments(self, shots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        normalized = [dict(item) for item in shots]
        output: list[dict[str, Any]] = []
        index = 0
        existing_ids = [str(item.get("shot_id") or item.get("id") or "") for item in normalized]
        while index < len(normalized):
            shot = normalized[index]
            shot_id = str(shot.get("shot_id") or shot.get("id") or f"shot_{index + 1:04d}")
            shot["shot_id"] = shot_id
            duration = float(shot.get("duration") or 0)
            beats = _inherit_action_states(
                normalize_action_beats(shot.get("action_beats"), fallback_text=shot.get("character_action")),
                shot,
            )
            complex_motion = any(beat.complex_motion for beat in beats)
            speech_ms = estimate_speech_ms(dialogue_text(shot.get("dialogue")))
            short_motion = complex_motion or self.provider.first_frame_only
            segment_limit = min(
                self.provider.max_duration,
                COMPLEX_ACTION_MAX_SECONDS if short_motion else self.provider.max_duration,
            )
            parts_for_duration = max(1, math.ceil(max(duration, 0.0) / max(segment_limit, 1e-6)))
            parts_for_speech = max(1, math.ceil(speech_ms / max(segment_limit * 1000 - ACTION_LEAD_RESERVE_MS, 1)))
            part_count = max(parts_for_duration, parts_for_speech, len(beats) if beats else 1)
            if part_count > 1:
                split_reasons = []
                if parts_for_duration > 1:
                    split_reasons.append(
                        "first_frame_duration_limit"
                        if short_motion and not complex_motion
                        else "duration_over_provider_limit"
                    )
                if parts_for_speech > 1:
                    split_reasons.append("dialogue_capacity")
                if len(beats) > 1:
                    split_reasons.append("action_beat_count")
                if complex_motion:
                    split_reasons.append("complex_motion")
                split_reason = "+".join(split_reasons) or "auto_split"
                split_parts = split_shot(shot, part_count, existing_ids=existing_ids, reason=split_reason)
                existing_ids.extend(item["shot_id"] for item in split_parts)
                output.extend(split_parts)
                self._record_adjustment(
                    "shot_split",
                    f"镜头 {shot_id} 因复杂动作、超长时长或对白容量拆分为 {part_count} 个连续镜头",
                    [item["shot_id"] for item in split_parts],
                    reason=split_reason,
                    before=[_structure_snapshot(shot)],
                    after=[_structure_snapshot(item) for item in split_parts],
                )
            else:
                shot["estimated_speech_ms"] = speech_ms
                shot["action_beats"] = [beat.to_dict() for beat in beats] or [
                    {"text": "", "phase": "continuation", "complex_motion": False, "entry_state": "", "exit_state": ""}
                ]
                shot.setdefault("timing", {})
                shot["timing"].update(
                    {
                        "action_entry_state": str(shot.get("action_entry_state") or ""),
                        "action_exit_state": str(shot.get("action_exit_state") or ""),
                        "gaze_direction": str(shot.get("gaze_direction") or ""),
                        "screen_axis": str(shot.get("screen_axis") or ""),
                    }
                )
                output.append(shot)
            index += 1

        # Merge genuinely short adjacent beats before extending standalone short shots.
        cursor = 0
        merged: list[dict[str, Any]] = []
        while cursor < len(output):
            current = output[cursor]
            while (
                cursor + 1 < len(output)
                and float(current.get("duration") or 0) < self.provider.min_duration
                and _mergeable(current, output[cursor + 1], self.provider)
            ):
                next_shot = output[cursor + 1]
                merged_id = str(current.get("shot_id") or "")
                next_id = str(next_shot.get("shot_id") or "")
                before = [_structure_snapshot(current), _structure_snapshot(next_shot)]
                current = merge_shots(current, next_shot, reason="short_adjacent_beats")
                self._record_adjustment(
                    "shots_merged",
                    f"过短镜头 {merged_id} 与 {next_id} 按同场景剧情合并",
                    [merged_id, next_id],
                    reason="short_adjacent_beats",
                    before=before,
                    after=[_structure_snapshot(current)],
                )
                cursor += 1
            merged.append(current)
            cursor += 1

        for shot in merged:
            duration = float(shot.get("duration") or 0)
            if self.provider.first_frame_only:
                timing = dict(shot.get("timing") or {})
                timing.update({"video_mode": "first_frame_i2v", "short_shot": True})
                shot["timing"] = timing
            if duration < self.provider.min_duration:
                old_id = str(shot.get("shot_id") or "")
                shot["duration"] = self.provider.min_duration
                self._record_adjustment(
                    "shot_extended",
                    f"过短镜头 {old_id} 为保留动作与对白延长到 Provider 最小时长 {self.provider.min_duration:g} 秒",
                    [old_id],
                )
        return merged

    def _adjust_count(self, shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not shots:
            raise StoryTimingError([TimingIssue(code="empty_storyboard", message="没有可用于时长规划的镜头")])
        target = self.target_duration_s
        if self.provider.is_fixed:
            desired_count = max(1, int(round(target / float(self.provider.fixed_duration or 1))))
        else:
            min_count = max(1, math.ceil(target / self.provider.max_duration - 1e-9))
            max_count = max(min_count, math.floor(target / self.provider.min_duration + 1e-9))
            desired_count = min(max(len(shots), min_count), max_count)
        desired_count = min(desired_count, MAX_PLANNED_SHOTS)

        while len(shots) > desired_count:
            pair_index = next(
                (
                    index
                    for index in range(len(shots) - 1)
                    if _mergeable_for_budget(shots[index], shots[index + 1], self.provider)
                ),
                None,
            )
            if pair_index is None:
                self._record_adjustment(
                    "shot_count_unreduced",
                    "无法在不合并复杂动作或超长对白的前提下继续减少镜头数量",
                    [str(item.get("shot_id") or "") for item in shots],
                )
                break
            left = shots[pair_index]
            right = shots[pair_index + 1]
            before = [_structure_snapshot(left), _structure_snapshot(right)]
            merged = merge_shots(left, right, reason="shot_count_budget")
            self._record_adjustment(
                "shots_merged_for_budget",
                f"为匹配目标总时长合并镜头 {left.get('shot_id')} 与 {right.get('shot_id')}",
                [str(left.get("shot_id") or ""), str(right.get("shot_id") or "")],
                reason="shot_count_budget",
                before=before,
                after=[_structure_snapshot(merged)],
            )
            shots[pair_index : pair_index + 2] = [merged]

        while len(shots) < desired_count:
            split_index = max(range(len(shots)), key=lambda index: _shot_weight(shots[index]))
            source = shots[split_index]
            parts = split_shot(
                source,
                2,
                existing_ids=(str(item.get("shot_id") or "") for item in shots),
                reason="shot_count_budget",
            )
            self._record_adjustment(
                "shot_added_for_budget",
                f"为匹配目标总时长并保持叙事节拍，将镜头 {source.get('shot_id')} 扩展为连续双镜头",
                [item["shot_id"] for item in parts],
                reason="shot_count_budget",
                before=[_structure_snapshot(source)],
                after=[_structure_snapshot(item) for item in parts],
            )
            shots[split_index : split_index + 1] = parts
        return shots

    def _allocate_durations(self, shots: list[dict[str, Any]]) -> None:
        if self.provider.is_fixed:
            fixed = float(self.provider.fixed_duration or 0)
            for shot in shots:
                shot["duration"] = fixed
            return

        step = self.provider.duration_step
        target_units = int(round(self.target_duration_s / step))
        min_units = max(1, int(round(self.provider.min_duration / step)))
        max_units = max(min_units, int(round(self.provider.max_duration / step)))
        units = [min_units for _ in shots]
        remaining = max(0, target_units - min_units * len(shots))
        capacities = [max(0, max_units - min_units) for _ in shots]
        weights = [_shot_weight(shot) for shot in shots]
        while remaining > 0 and sum(capacities) > 0:
            candidates = [index for index, capacity in enumerate(capacities) if capacity > 0]
            total_weight = sum(weights[index] for index in candidates) or float(len(candidates))
            allocations = {
                index: remaining * weights[index] / total_weight if total_weight else remaining / len(candidates)
                for index in candidates
            }
            progressed = False
            for index in candidates:
                if remaining <= 0:
                    break
                add = min(capacities[index], remaining, int(math.floor(allocations[index])) or 1)
                if add <= 0:
                    continue
                units[index] += add
                capacities[index] -= add
                remaining -= add
                progressed = True
            if not progressed:
                index = max(candidates, key=lambda item: capacities[item])
                units[index] += 1
                capacities[index] -= 1
                remaining -= 1

        for index, shot in enumerate(shots):
            shot["duration"] = round(units[index] * step, 3)

        # Transfer available units to dialogue-heavy shots so speech is never cropped.
        for index, shot in enumerate(shots):
            required_ms = shot_speech_ms(shot) + (
                ACTION_LEAD_RESERVE_MS if str(shot.get("character_action") or "").strip() else 0
            )
            required_units = min(max_units, math.ceil(required_ms / (step * 1000)))
            deficit = required_units - units[index]
            donor_cursor = 0
            while deficit > 0 and donor_cursor < len(shots):
                if donor_cursor == index:
                    donor_cursor += 1
                    continue
                donor = shots[donor_cursor]
                donor_required_ms = shot_speech_ms(donor) + (
                    ACTION_LEAD_RESERVE_MS if str(donor.get("character_action") or "").strip() else 0
                )
                donor_required_units = math.ceil(donor_required_ms / (step * 1000))
                spare = units[donor_cursor] - max(min_units, donor_required_units)
                transfer = min(deficit, max(0, spare))
                if transfer > 0:
                    units[donor_cursor] -= transfer
                    units[index] += transfer
                    deficit -= transfer
                donor_cursor += 1
            if deficit > 0:
                raise StoryTimingError(
                    [
                        TimingIssue(
                            code="dialogue_exceeds_shot_duration",
                            message=(
                                f"镜头 {shot.get('shot_id')} 的对白预计需要 "
                                f"{shot_speech_ms(shot)} 毫秒，超过当前镜头可容纳时长"
                            ),
                            shot_ids=(str(shot.get("shot_id") or ""),),
                        )
                    ]
                )

        for index, shot in enumerate(shots):
            shot["duration"] = round(units[index] * step, 3)

    def rebalance(self, shots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Split/merge/retime shots so the result is executable and close to target."""

        self.adjustments = []
        normalized = self._normalize_segments(shots)
        normalized = self._adjust_count(normalized)
        self._allocate_durations(normalized)
        for shot in normalized:
            shot["estimated_speech_ms"] = estimate_speech_ms(dialogue_text(shot.get("dialogue")))

        self.shot_count = len(normalized)
        self.planned_total_duration_s = round(sum(float(item.get("duration") or 0) for item in normalized), 3)
        self.dialogue_duration_s = round(sum(shot_speech_ms(item) for item in normalized) / 1000.0, 3)
        self.action_beats = [
            beat
            for shot in normalized
            for beat in normalize_action_beats(shot.get("action_beats"), fallback_text=shot.get("character_action"))
        ]
        if not self.target_feasible:
            self._record_adjustment(
                "target_quantized",
                (
                    f"目标 {self.target_duration_s:g} 秒受 Provider 时长能力量化为 "
                    f"{self.planned_total_duration_s:g} 秒，未静默忽略目标时长"
                ),
            )
        return normalized

    def validate_shots(
        self,
        shots: Sequence[Mapping[str, Any]],
        *,
        provider: ProviderDurationCapability | None = None,
        require_target: bool = True,
    ) -> list[TimingIssue]:
        active_provider = provider or self.provider
        issues: list[TimingIssue] = []
        total_ms = 0
        for shot in shots:
            shot_id = str(shot.get("shot_id") or shot.get("id") or "")
            duration = float(shot.get("duration") or 0)
            total_ms += int(round(duration * 1000))
            if not active_provider.contains(duration) or not active_provider.is_aligned(duration):
                issues.append(
                    TimingIssue(
                        code="provider_duration_invalid",
                        message=(
                            f"镜头 {shot_id} 时长 {duration:g} 秒不符合视频 Provider "
                            f"{active_provider.protocol or '<unknown>'} 的 {active_provider.describe()}"
                        ),
                        shot_ids=(shot_id,),
                    )
                )
            speech_ms = estimate_speech_ms(dialogue_text(shot.get("dialogue")))
            available_ms = usable_speech_ms(shot, duration)
            raw_estimate = shot.get("estimated_speech_ms")
            if raw_estimate is not None:
                try:
                    persisted_estimate = int(round(float(raw_estimate)))
                except (TypeError, ValueError):
                    persisted_estimate = -1
                if abs(persisted_estimate - speech_ms) > 100:
                    issues.append(
                        TimingIssue(
                            code="estimated_speech_ms_stale",
                            message=(
                                f"镜头 {shot_id} 的 estimated_speech_ms={persisted_estimate} 与当前对白估算 "
                                f"{speech_ms} 毫秒不一致"
                            ),
                            shot_ids=(shot_id,),
                        )
                    )
            if speech_ms > available_ms:
                issues.append(
                    TimingIssue(
                        code="dialogue_exceeds_shot_duration",
                        message=(f"镜头 {shot_id} 对白预计 {speech_ms} 毫秒，超过镜头可用时长 {available_ms} 毫秒"),
                        shot_ids=(shot_id,),
                    )
                )
            timing = shot.get("timing") if isinstance(shot.get("timing"), dict) else {}
            requested_ms = timing.get("requested_video_duration_ms")
            if requested_ms is not None:
                try:
                    requested_value = int(round(float(requested_ms)))
                except (TypeError, ValueError):
                    requested_value = -1
                if abs(requested_value - int(round(duration * 1000))) > PROVIDER_TIMELINE_TOLERANCE_MS:
                    issues.append(
                        TimingIssue(
                            code="generation_timeline_mismatch",
                            message=(
                                f"镜头 {shot_id} 的视频生成时长 {requested_value} 毫秒与分镜时间线 "
                                f"{int(round(duration * 1000))} 毫秒不一致，导出会裁掉动作或对白"
                            ),
                            shot_ids=(shot_id,),
                        )
                    )
        planned_total_s = total_ms / 1000.0
        if require_target and abs(planned_total_s - self.target_duration_s) > self.tolerance_s + 1e-6:
            issues.append(
                TimingIssue(
                    code="target_duration_mismatch",
                    message=(
                        f"镜头总时长 {planned_total_s:g} 秒与目标 {self.target_duration_s:g} 秒相差超过允许误差 "
                        f"{self.tolerance_s:g} 秒"
                    ),
                )
            )
        return issues

    def validate_timeline(
        self,
        shots: Sequence[Mapping[str, Any]],
        *,
        audio_tracks: Sequence[Mapping[str, Any]] = (),
        media_durations_ms: Mapping[str, int] | None = None,
        provider: ProviderDurationCapability | None = None,
        require_target: bool = True,
    ) -> list[TimingIssue]:
        active_provider = provider or self.provider
        issues = self.validate_shots(shots, provider=active_provider, require_target=require_target)
        durations = {str(path): int(value) for path, value in (media_durations_ms or {}).items() if path}
        spans: dict[str, tuple[int, int]] = {}
        cursor = 0
        total_ms = 0
        for shot in shots:
            shot_id = str(shot.get("shot_id") or shot.get("id") or "")
            duration_ms = int(round(float(shot.get("duration") or 0) * 1000))
            spans[shot_id] = (cursor, cursor + duration_ms)
            total_ms = cursor + duration_ms
            cursor = total_ms

            video_path = str(shot.get("video_path") or "")
            actual_video_ms = durations.get(video_path)
            if actual_video_ms is not None:
                # 执行计划在案的「生成满固定秒数、成片裁剪为故事时长」是预期行为
                # （例如固定 5 秒 Provider 生成 5 秒后裁成 4.5 秒），不算违规裁剪。
                execution_plan = load_shot_execution_plan(shot)
                planned_trim = bool(
                    execution_plan is not None
                    and abs(execution_plan.generation_duration_ms - actual_video_ms) <= PROVIDER_TIMELINE_TOLERANCE_MS
                    and abs(execution_plan.effective_duration_ms - duration_ms) <= PROVIDER_TIMELINE_TOLERANCE_MS
                )
                if actual_video_ms + PROVIDER_TIMELINE_TOLERANCE_MS < duration_ms:
                    issues.append(
                        TimingIssue(
                            code="video_shorter_than_timeline",
                            message=(
                                f"镜头 {shot_id} 视频实际 {actual_video_ms} 毫秒，短于时间线 "
                                f"{duration_ms} 毫秒，无法完整承载动作与对白"
                            ),
                            shot_ids=(shot_id,),
                        )
                    )
                elif actual_video_ms > duration_ms + PROVIDER_TIMELINE_TOLERANCE_MS and not planned_trim:
                    issues.append(
                        TimingIssue(
                            code="video_will_be_trimmed",
                            message=(
                                f"镜头 {shot_id} 视频实际 {actual_video_ms} 毫秒，长于时间线 "
                                f"{duration_ms} 毫秒；导出将裁掉尾部动作或对白"
                            ),
                            shot_ids=(shot_id,),
                        )
                    )

            dialogue = str(shot.get("dialogue") or "").strip()
            timing = shot.get("timing") if isinstance(shot.get("timing"), dict) else {}
            profile = shot.get("continuity_profile") if isinstance(shot.get("continuity_profile"), dict) else {}
            audio_source = (
                str(
                    timing.get("audio_source")
                    or timing.get("audio_mode")
                    or profile.get("audio_source")
                    or profile.get("audio_mode")
                    or ""
                )
                .strip()
                .lower()
            )
            if dialogue and audio_source != "native":
                audio_path = str(shot.get("audio_path") or "")
                if not audio_path:
                    issues.append(
                        TimingIssue(
                            code="dialogue_audio_missing",
                            message=f"镜头 {shot_id} 含对白但缺少配音文件，禁止导出无声或错位对白",
                            shot_ids=(shot_id,),
                        )
                    )
                else:
                    actual_audio_ms = durations.get(audio_path)
                    if actual_audio_ms is None:
                        issues.append(
                            TimingIssue(
                                code="dialogue_audio_duration_unknown",
                                message=f"镜头 {shot_id} 配音时长无法读取，无法确认对白是否超出镜头",
                                shot_ids=(shot_id,),
                            )
                        )
                    elif actual_audio_ms > duration_ms + PROVIDER_TIMELINE_TOLERANCE_MS:
                        issues.append(
                            TimingIssue(
                                code="dialogue_audio_exceeds_shot",
                                message=(
                                    f"镜头 {shot_id} 配音实际 {actual_audio_ms} 毫秒，超过镜头 "
                                    f"{duration_ms} 毫秒，导出会截断对白"
                                ),
                                shot_ids=(shot_id,),
                            )
                        )

        for track in audio_tracks:
            if bool(track.get("muted")):
                continue
            track_id = str(track.get("id") or "")
            source_path = str(track.get("resolved_source_path") or track.get("source_path") or "")
            measured_ms = durations.get(source_path)
            source_ms = int(measured_ms if measured_ms is not None else (track.get("source_duration_ms") or 0))
            start_ms = int(track.get("start_ms") or 0)
            delay_ms = int(track.get("delay_ms") or 0)
            trimmed_ms = max(0, source_ms - int(track.get("trim_start_ms") or 0) - int(track.get("trim_end_ms") or 0))
            effective_end = start_ms + delay_ms + trimmed_ms
            shot_id = str(track.get("shot_id") or "")
            if track.get("kind") == "dialogue":
                span = spans.get(shot_id)
                if not span:
                    issues.append(
                        TimingIssue(
                            code="dialogue_track_shot_missing",
                            message=f"音频轨道 {track_id} 关联的镜头 {shot_id or '<empty>'} 不在时间线上",
                            track_ids=(track_id,),
                            shot_ids=(shot_id,) if shot_id else (),
                        )
                    )
                elif effective_end > span[1] + PROVIDER_TIMELINE_TOLERANCE_MS:
                    issues.append(
                        TimingIssue(
                            code="dialogue_track_exceeds_span",
                            message=(
                                f"镜头 {shot_id} 的对白轨道 {track_id} 结束于 {effective_end} 毫秒，"
                                f"超出镜头区间 {span[0]}-{span[1]} 毫秒"
                            ),
                            shot_ids=(shot_id,),
                            track_ids=(track_id,),
                        )
                    )
            elif not track.get("loop") and effective_end > total_ms + PROVIDER_TIMELINE_TOLERANCE_MS:
                issues.append(
                    TimingIssue(
                        code="audio_track_exceeds_timeline",
                        message=(f"音频轨道 {track_id} 结束于 {effective_end} 毫秒，超出成片总时长 {total_ms} 毫秒"),
                        track_ids=(track_id,),
                    )
                )
        return issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_duration_s": self.target_duration_s,
            "planned_total_duration_s": self.planned_total_duration_s,
            "dialogue_duration_s": self.dialogue_duration_s,
            "action_beat_count": len(self.action_beats),
            "shot_count": self.shot_count,
            "provider_duration": self.provider.to_dict(),
            "tolerance_s": self.tolerance_s,
            "target_feasible": self.target_feasible,
            "adjustments": [item.to_dict() for item in self.adjustments],
        }


def ensure_timing_valid(
    shots: Sequence[Mapping[str, Any]],
    plan: StoryTimingPlan,
    *,
    audio_tracks: Sequence[Mapping[str, Any]] = (),
    media_durations_ms: Mapping[str, int] | None = None,
    provider: ProviderDurationCapability | None = None,
    require_target: bool = True,
) -> None:
    issues = plan.validate_timeline(
        shots,
        audio_tracks=audio_tracks,
        media_durations_ms=media_durations_ms,
        provider=provider,
        require_target=require_target,
    )
    if issues:
        raise StoryTimingError(issues)


__all__ = [
    "ActionBeat",
    "DialogueTiming",
    "ProviderDurationCapability",
    "ShotExecutionPlan",
    "StoryTimingError",
    "StoryTimingPlan",
    "TimingAdjustment",
    "TimingIssue",
    "action_is_complex",
    "ensure_timing_valid",
    "dialogue_items",
    "dialogue_text",
    "estimate_action_beats",
    "estimate_speech_ms",
    "load_shot_execution_plan",
    "normalize_action_beats",
    "plan_required_capabilities",
    "provider_duration_capability",
    "resolve_shot_execution_plan",
    "shot_speech_ms",
    "split_shot",
    "usable_speech_ms",
]
