"""Story timing planning, provider-duration constraints and render-time validation.

This module owns all decisions that turn narrative timing into executable shots:
speech estimation, action beat segmentation, provider duration capability checks,
long-shot splitting, short-shot merging/extension and final timeline validation.
It intentionally has no FastAPI or SQLAlchemy dependencies so the planning rules
can be regression-tested as a deterministic service.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from config import settings

PROVIDER_TIMELINE_TOLERANCE_MS = 500
ACTION_LEAD_RESERVE_MS = 250
COMPLEX_ACTION_MAX_SECONDS = 5.0
DEFAULT_DURATION_STEP_SECONDS = 1.0
MAX_PLANNED_SHOTS = 200

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
    "翻滚",
    "闪避",
    "跳跃",
    "挥剑",
    "挥拳",
    "踢",
    "摔",
    "转身走位",
    "连续",
    "翻越",
    "攀爬",
    "躲闪",
    "chase",
    "fight",
    "combat",
    "sprint",
    "roll",
    "vault",
    "dodge",
)
_ACTION_CONJUNCTIONS = ("然后", "接着", "随后", "同时", "之后", "再", "and then", "then", "after that")


@dataclass(frozen=True)
class ProviderDurationCapability:
    """Duration contract advertised by the active video provider."""

    protocol: str = ""
    fixed_duration: float | None = None
    min_duration: float = settings.MIN_SHOT_DURATION_SECONDS
    max_duration: float = settings.MAX_SHOT_DURATION_SECONDS
    duration_step: float = DEFAULT_DURATION_STEP_SECONDS

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
    def tolerance_s(self) -> float:
        return max(0.5, self.duration_step / 2 + 1e-6)

    def contains(self, duration_s: float) -> bool:
        value = float(duration_s)
        return (
            math.isfinite(value)
            and value + 1e-6 >= self.min_duration
            and value - 1e-6 <= self.max_duration
        )

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
                            f"{label} 时长 {value:g} 秒不符合当前视频 Provider 的 "
                            f"{self.duration_step:g} 秒生成步长"
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
    )


@dataclass(frozen=True)
class ActionBeat:
    text: str
    complex_motion: bool = False


@dataclass(frozen=True)
class TimingAdjustment:
    code: str
    message: str
    shot_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "shot_ids": list(self.shot_ids),
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
                output.append({"speaker": "", "line": line, "emotion": "", "action": "", "start_ms": None, "end_ms": None})
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


def estimate_action_beats(text: Any) -> list[ActionBeat]:
    """Split an action description into narrative beats and classify complex motion."""

    value = str(text or "").strip()
    if not value:
        return []
    normalized = value
    for conjunction in _ACTION_CONJUNCTIONS:
        normalized = normalized.replace(conjunction, "，")
    pieces = [piece.strip() for piece in _CLAUSE_BREAK_RE.split(normalized) if piece.strip()]
    if not pieces:
        pieces = [value]
    return [
        ActionBeat(
            text=piece,
            complex_motion=any(marker.lower() in piece.lower() for marker in _COMPLEX_ACTION_MARKERS),
        )
        for piece in pieces[:12]
    ]


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
    chunks = [value[index * size:(index + 1) * size] for index in range(parts)]
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


def _split_timing_metadata(shot: Mapping[str, Any], part_number: int, part_count: int) -> dict[str, Any]:
    timing = dict(shot.get("timing") or {})
    timing.update(
        {
            "split_from": str(shot.get("shot_id") or shot.get("id") or ""),
            "split_part": part_number,
            "split_total": part_count,
            "narrative_continuation": True,
        }
    )
    return timing


def split_shot(
    shot: Mapping[str, Any],
    parts: int,
    *,
    existing_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Split one shot into continuous parts without dropping action or dialogue."""

    count = max(2, int(parts))
    base_id = _base_shot_id(str(shot.get("shot_id") or shot.get("id") or "shot"))
    actions = _split_text(str(shot.get("character_action") or ""), count)
    dialogues = _split_dialogue_by_speech(shot.get("dialogue"), count)
    descriptions = _split_text(str(shot.get("scene_description") or ""), count)
    used = set(existing_ids)
    output: list[dict[str, Any]] = []
    for index in range(count):
        item = dict(shot)
        shot_id = f"{base_id}_part_{index + 1:02d}"
        shot_id = _unique_id(used, shot_id)
        used.add(shot_id)
        item["shot_id"] = shot_id
        item["scene_description"] = descriptions[index]
        item["character_action"] = actions[index]
        item["dialogue"] = dialogues[index]
        item["estimated_speech_ms"] = estimate_speech_ms(dialogue_text(dialogues[index]))
        item["timing"] = _split_timing_metadata(shot, index + 1, count)
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
    combined_action = "；".join(part for part in (str(left.get("character_action") or ""), str(right.get("character_action") or "")) if part)
    if action_is_complex(combined_action):
        return False
    combined_dialogue = dialogue_text([*dialogue_items(left.get("dialogue")), *dialogue_items(right.get("dialogue"))])
    combined_duration = float(left.get("duration") or 0) + float(right.get("duration") or 0)
    if estimate_speech_ms(combined_dialogue) > usable_speech_ms({"character_action": combined_action}, combined_duration):
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
    combined_action = "；".join(
        part for part in (str(left.get("character_action") or ""), str(right.get("character_action") or "")) if part
    )
    combined_beats = estimate_action_beats(combined_action)
    if any(beat.complex_motion for beat in combined_beats) or len(combined_beats) > 2:
        return False
    combined_dialogue = dialogue_text([*dialogue_items(left.get("dialogue")), *dialogue_items(right.get("dialogue"))])
    if estimate_speech_ms(combined_dialogue) > capability.max_duration * 1000 - ACTION_LEAD_RESERVE_MS:
        return False
    # Cross-scene cuts are allowed only for dialogue-free simple beats when the
    # fixed-duration budget makes the shot count otherwise infeasible.
    return _same_story_location(left, right) or not combined_dialogue


def merge_shots(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Merge two adjacent story beats while retaining all source text."""

    item = dict(left)
    right_id = str(right.get("shot_id") or right.get("id") or "")
    left_id = str(left.get("shot_id") or left.get("id") or "")
    item["shot_id"] = left_id
    item["scene_description"] = "；".join(
        dict.fromkeys(part for part in (str(left.get("scene_description") or ""), str(right.get("scene_description") or "")) if part)
    )
    item["character_action"] = "；然后".join(
        part for part in (str(left.get("character_action") or ""), str(right.get("character_action") or "")) if part
    )
    item["dialogue"] = [*dialogue_items(left.get("dialogue")), *dialogue_items(right.get("dialogue"))]
    item["duration"] = round(float(left.get("duration") or 0) + float(right.get("duration") or 0), 3)
    item["transition"] = right.get("transition") or left.get("transition") or "cut"
    item["estimated_speech_ms"] = estimate_speech_ms(dialogue_text(item["dialogue"]))
    timing = dict(left.get("timing") or {})
    merged_from = [str(value) for value in timing.get("merged_from", []) if value]
    merged_from.extend([left_id, right_id])
    timing["merged_from"] = list(dict.fromkeys(merged_from))
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
    ) -> "StoryTimingPlan":
        plan = cls(target_duration_s=float(target_duration_s), provider=provider)
        plan.rebalance(shots)
        return plan

    @property
    def tolerance_s(self) -> float:
        return self.provider.tolerance_s

    @property
    def target_feasible(self) -> bool:
        return abs(self.planned_total_duration_s - self.target_duration_s) <= self.tolerance_s + 1e-6

    def _record_adjustment(self, code: str, message: str, shot_ids: Sequence[str] = ()) -> None:
        self.adjustments.append(
            TimingAdjustment(code=code, message=message, shot_ids=tuple(str(item) for item in shot_ids if item))
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
            beats = estimate_action_beats(shot.get("character_action"))
            complex_motion = action_is_complex(shot.get("character_action"))
            speech_ms = estimate_speech_ms(dialogue_text(shot.get("dialogue")))
            segment_limit = min(self.provider.max_duration, COMPLEX_ACTION_MAX_SECONDS if complex_motion else self.provider.max_duration)
            parts_for_duration = max(1, math.ceil(max(duration, 0.0) / max(segment_limit, 1e-6)))
            parts_for_speech = max(1, math.ceil(speech_ms / max(segment_limit * 1000 - ACTION_LEAD_RESERVE_MS, 1)))
            part_count = max(parts_for_duration, parts_for_speech)
            if part_count > 1:
                split_parts = split_shot(shot, part_count, existing_ids=existing_ids)
                existing_ids.extend(item["shot_id"] for item in split_parts)
                output.extend(split_parts)
                self._record_adjustment(
                    "shot_split",
                    f"镜头 {shot_id} 因复杂动作、超长时长或对白容量拆分为 {part_count} 个连续镜头",
                    [item["shot_id"] for item in split_parts],
                )
            else:
                shot["estimated_speech_ms"] = speech_ms
                shot.setdefault("timing", {})
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
                current = merge_shots(current, next_shot)
                self._record_adjustment("shots_merged", f"过短镜头 {merged_id} 与 {next_id} 按同场景剧情合并", [merged_id, next_id])
                cursor += 1
            merged.append(current)
            cursor += 1

        for shot in merged:
            duration = float(shot.get("duration") or 0)
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
            raise StoryTimingError(
                [TimingIssue(code="empty_storyboard", message="没有可用于时长规划的镜头")]
            )
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
            merged = merge_shots(left, right)
            self._record_adjustment(
                "shots_merged_for_budget",
                f"为匹配目标总时长合并镜头 {left.get('shot_id')} 与 {right.get('shot_id')}",
                [str(left.get("shot_id") or ""), str(right.get("shot_id") or "")],
            )
            shots[pair_index:pair_index + 2] = [merged]

        while len(shots) < desired_count:
            split_index = max(range(len(shots)), key=lambda index: _shot_weight(shots[index]))
            source = shots[split_index]
            parts = split_shot(source, 2, existing_ids=(str(item.get("shot_id") or "") for item in shots))
            self._record_adjustment(
                "shot_added_for_budget",
                f"为匹配目标总时长并保持叙事节拍，将镜头 {source.get('shot_id')} 扩展为连续双镜头",
                [item["shot_id"] for item in parts],
            )
            shots[split_index:split_index + 1] = parts
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
            required_ms = shot_speech_ms(shot) + (ACTION_LEAD_RESERVE_MS if str(shot.get("character_action") or "").strip() else 0)
            required_units = min(max_units, math.ceil(required_ms / (step * 1000)))
            deficit = required_units - units[index]
            donor_cursor = 0
            while deficit > 0 and donor_cursor < len(shots):
                if donor_cursor == index:
                    donor_cursor += 1
                    continue
                donor = shots[donor_cursor]
                donor_required_ms = shot_speech_ms(donor) + (ACTION_LEAD_RESERVE_MS if str(donor.get("character_action") or "").strip() else 0)
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
            for beat in estimate_action_beats(shot.get("character_action"))
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
                        message=(
                            f"镜头 {shot_id} 对白预计 {speech_ms} 毫秒，超过镜头可用时长 "
                            f"{available_ms} 毫秒"
                        ),
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
                elif actual_video_ms > duration_ms + PROVIDER_TIMELINE_TOLERANCE_MS:
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
            audio_source = str(
                timing.get("audio_source")
                or timing.get("audio_mode")
                or profile.get("audio_source")
                or profile.get("audio_mode")
                or ""
            ).strip().lower()
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
                        message=(
                            f"音频轨道 {track_id} 结束于 {effective_end} 毫秒，超出成片总时长 "
                            f"{total_ms} 毫秒"
                        ),
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
    "ProviderDurationCapability",
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
    "provider_duration_capability",
    "shot_speech_ms",
    "split_shot",
    "usable_speech_ms",
]
