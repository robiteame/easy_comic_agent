"""可审查的成片后期时间线（PostProductionPlan）。

本模块把 Shot、字幕/音频配置和 Provider/FFmpeg 能力收敛成一份纯 JSON 计划：
每个镜头的入点、出点、持续时间、转场、对白、音频、字幕和特效都会在
FFmpeg 运行前确定。渲染器只消费该计划，避免 `scene_group_id` 等隐式规则
悄悄覆盖用户参数。

转场优先级（边界以左侧镜头的 ``transition`` 为用户自定义）：

1. 用户显式设置的 ``shot.transition``；
2. 跨场景规则（``continuity_profile.cross_scene_transition``，默认 white_flash）；
3. 同场景规则（``continuity_profile.same_scene_transition``，默认 cut）；
4. 最终兜底 cut。

不支持的转场不会静默忽略：计划中记录 requested/effective/fallback_reason，
并降级为 cut；渲染日志和时间线 JSON 都保留原因。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from services.shot_dialogue import parse_shot_dialogue
from services.story_timing import ShotExecutionPlan, load_shot_execution_plan

TRANSITION_DURATIONS_MS = {
    "cut": 0,
    "fade": 500,
    "dissolve": 500,
    "white_flash": 350,
    "push": 500,
    "wipe": 500,
}
SUPPORTED_TRANSITIONS = tuple(TRANSITION_DURATIONS_MS)
TRANSITION_ALIASES = {
    "hard_cut": "cut",
    "hard cut": "cut",
    "crossfade": "dissolve",
    "white-flash": "white_flash",
    "white flash": "white_flash",
    "push_left": "push",
    "wipe_left": "wipe",
}

# FFmpeg/xfade 真实策略。white_flash 由 dissolve + 白色覆盖完成，
# duration 仍由 TransitionSpec 精确控制，不使用每镜头 fade 猜测。
FFMPEG_TRANSITION_FILTERS = {
    "cut": "cut",
    "fade": "fade",
    "dissolve": "dissolve",
    "white_flash": "dissolve",
    "push": "slideleft",
    "wipe": "wipeleft",
}

CAMERA_MOVEMENT_PROMPTS = {
    "静止": "locked static tripod shot; camera remains stationary",
    "推": "slow cinematic dolly-in / push-in toward the subject",
    "缓慢推进": "very slow controlled push-in toward the subject",
    "拉": "slow pull-out / dolly-out revealing more of the scene",
    "摇": "smooth horizontal pan across the scene",
    "移": "lateral tracking move parallel to the subject",
    "跟": "stable tracking shot following the subject",
    "升降": "vertical crane move rising or falling smoothly",
    "环绕": "smooth orbiting camera move around the subject",
}


@dataclass
class TransitionSpec:
    """一条镜头边界转场的最终参数。"""

    boundary_id: str
    from_shot_id: str
    to_shot_id: str
    scene_relation: str
    requested: str
    effective: str
    duration_ms: int
    source: str
    supported: bool
    renderer: str
    fallback_reason: str = ""
    config_key: str = ""
    requested_duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TimelineAudio:
    id: str
    kind: str
    source_path: str
    start_ms: int
    end_ms: int
    shot_id: str = ""
    speaker: str = ""
    emotion: str = ""
    clip_start_ms: int = 0
    clip_end_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TimelineDialogue:
    shot_id: str
    speaker: str
    text: str
    emotion: str
    start_ms: int
    end_ms: int
    audio_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TimelineCue:
    track_id: str
    shot_id: str
    start_ms: int
    end_ms: int
    text: str
    character_name: str = ""
    clamped: bool = False
    clamp_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShotTimelineEntry:
    shot_id: str
    sequence: int
    source_in_ms: int
    source_out_ms: int
    timeline_start_ms: int
    timeline_end_ms: int
    duration_ms: int
    transition_in: dict[str, Any] | None
    transition_out: dict[str, Any] | None
    camera: dict[str, Any]
    emotion: str
    dialogue: list[TimelineDialogue] = field(default_factory=list)
    audio: list[TimelineAudio] = field(default_factory=list)
    subtitles: list[TimelineCue] = field(default_factory=list)
    effects: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dialogue"] = [item.to_dict() for item in self.dialogue]
        payload["audio"] = [item.to_dict() for item in self.audio]
        payload["subtitles"] = [item.to_dict() for item in self.subtitles]
        return payload


@dataclass
class PostProductionPlan:
    """一次渲染的完整后期计划。``to_dict`` 可直接写成 timeline.json。"""

    version: int
    project_id: str
    fps: int
    total_duration_ms: int
    shots: list[ShotTimelineEntry]
    transitions: list[TransitionSpec]
    audio_tracks: list[dict[str, Any]]
    subtitle_tracks: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)

    @property
    def shot_spans(self) -> dict[str, tuple[int, int]]:
        return {shot.shot_id: (shot.timeline_start_ms, shot.timeline_end_ms) for shot in self.shots}

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "project_id": self.project_id,
            "fps": self.fps,
            "total_duration_ms": self.total_duration_ms,
            "shots": [shot.to_dict() for shot in self.shots],
            "transitions": [transition.to_dict() for transition in self.transitions],
            "audio_tracks": self.audio_tracks,
            "subtitle_tracks": self.subtitle_tracks,
            "warnings": self.warnings,
            "capabilities": self.capabilities,
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target


def normalize_transition(value: Any) -> str:
    text = str(value or "").strip().lower()
    return TRANSITION_ALIASES.get(text, text)


def _profile_value(profile: Mapping[str, Any], key: str, default: Any) -> Any:
    value = profile.get(key) if isinstance(profile, Mapping) else None
    return default if value in (None, "") else value


def _duration_ms(value: Any, default: int) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0, int(round(number)))


def _seconds_to_ms(value: Any, default_ms: int) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default_ms
    # 配置明确写 seconds 时不能被当成毫秒；同时兼容历史毫秒字段。
    return max(0, int(round(number * 1000 if number < 100 else number)))


def _scene_relation(previous: Mapping[str, Any] | None, current: Mapping[str, Any]) -> str:
    if previous is None:
        return "start"
    previous_group = str(previous.get("scene_group_id") or previous.get("scene_asset_id") or "")
    current_group = str(current.get("scene_group_id") or current.get("scene_asset_id") or "")
    return "same_scene" if previous_group == current_group else "cross_scene"


def _resolve_boundary(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    boundary_index: int,
    warnings: list[str],
    supported_transitions: set[str] | None = None,
) -> TransitionSpec:
    relation = _scene_relation(previous, current)
    previous_profile = (previous or {}).get("continuity_profile") or {}
    current_profile = current.get("continuity_profile") or {}
    if relation == "start":
        return TransitionSpec(
            boundary_id=f"boundary-{boundary_index:04d}",
            from_shot_id="",
            to_shot_id=str(current.get("shot_id") or current.get("id") or ""),
            scene_relation=relation,
            requested="cut",
            effective="cut",
            duration_ms=0,
            source="start_of_timeline",
            supported=True,
            renderer=FFMPEG_TRANSITION_FILTERS["cut"],
        )

    requested_value = (previous or {}).get("transition")
    source = "user_shot_transition"
    config_key = "shot.transition"
    if not requested_value:
        # 自动规则只在用户没有提供值时生效，绝不覆盖用户选择。
        if relation == "cross_scene":
            requested_value = _profile_value(previous_profile, "cross_scene_transition", "white_flash")
            source = "cross_scene_rule"
            config_key = "continuity_profile.cross_scene_transition"
        else:
            requested_value = _profile_value(previous_profile, "same_scene_transition", "cut")
            source = "same_scene_rule"
            config_key = "continuity_profile.same_scene_transition"
    requested = normalize_transition(requested_value)
    allowed = supported_transitions if supported_transitions is not None else set(SUPPORTED_TRANSITIONS)
    supported = requested in allowed
    effective = requested if supported else "cut"
    requested_duration: int | None = None
    if supported and effective != "cut":
        if effective == "white_flash":
            requested_duration = _seconds_to_ms(
                _profile_value(previous_profile, "cross_scene_flash_seconds", 0.35),
                TRANSITION_DURATIONS_MS[effective],
            )
        else:
            explicit = _profile_value(previous_profile, "transition_duration_ms", None)
            requested_duration = _duration_ms(explicit, TRANSITION_DURATIONS_MS[effective])
        duration_ms = requested_duration
    else:
        duration_ms = 0

    fallback_reason = ""
    if not supported:
        fallback_reason = f"unsupported_transition:{requested or 'empty'}"
        warnings.append(
            f"边界 {boundary_index} 的转场 {requested or '空'} 不受当前 FFmpeg/Provider 支持，已降级为 cut"
        )
    return TransitionSpec(
        boundary_id=f"boundary-{boundary_index:04d}",
        from_shot_id=str((previous or {}).get("shot_id") or (previous or {}).get("id") or ""),
        to_shot_id=str(current.get("shot_id") or current.get("id") or ""),
        scene_relation=relation,
        requested=requested,
        effective=effective,
        duration_ms=duration_ms,
        source=source,
        supported=supported,
        renderer=FFMPEG_TRANSITION_FILTERS[effective],
        fallback_reason=fallback_reason,
        config_key=config_key,
        requested_duration_ms=requested_duration,
    )


def _normal_shot_id(shot: Mapping[str, Any]) -> str:
    return str(shot.get("shot_id") or shot.get("id") or "")


def _normal_duration_ms(shot: Mapping[str, Any]) -> int:
    # Shot.duration 使用秒；ShotDuration 在 API 层限制了最小值，这里再保护一次。
    try:
        seconds = float(shot.get("duration") or 3.0)
    except (TypeError, ValueError):
        seconds = 3.0
    return max(500, int(round(seconds * 1000)))


def _execution_window(shot: Mapping[str, Any]) -> tuple[ShotExecutionPlan, int, int, int] | None:
    """读取镜头统一执行计划的 (计划, 时长, 入点, 出点)；旧数据返回 None。

    后期合成必须消费生成阶段落库的同一份执行计划：固定档 Provider 生成的
    片段在这里按 trim 区间裁剪为故事时长，时间线不再从 duration 自行推导。
    """

    plan = load_shot_execution_plan(shot)
    if plan is None:
        return None
    duration_ms = plan.effective_duration_ms
    if duration_ms <= 0:
        return None
    return plan, duration_ms, max(0, plan.trim_start_ms), max(plan.trim_start_ms, plan.trim_end_ms)


def _speaker_for_shot(shot: Mapping[str, Any]) -> str:
    speakers = shot.get("characters_in_scene") or []
    if isinstance(speakers, str):
        try:
            speakers = json.loads(speakers)
        except (TypeError, ValueError):
            speakers = [speakers]
    return str(speakers[0] if speakers else "").strip()


def _clamp_subtitles(
    subtitle_tracks: Iterable[Mapping[str, Any]],
    shot_entries: list[ShotTimelineEntry],
    nominal_spans: dict[str, tuple[int, int]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """把所有字幕条目绑定到镜头并夹在其边界内。"""

    serialized: list[dict[str, Any]] = []
    for track in subtitle_tracks:
        track_id = str(track.get("id") or "")
        cues: list[dict[str, Any]] = []
        for cue in track.get("cues") or []:
            original_start = int(cue.get("start_ms") or 0)
            original_end = int(cue.get("end_ms") or 0)
            target = next(
                (
                    shot
                    for shot in shot_entries
                    if nominal_spans.get(shot.shot_id, (0, 0))[0] <= original_start < nominal_spans.get(shot.shot_id, (0, 0))[1]
                ),
                shot_entries[-1] if shot_entries else None,
            )
            if target is None:
                continue
            start = max(target.timeline_start_ms, min(original_start, target.timeline_end_ms))
            end = max(start, min(original_end, target.timeline_end_ms))
            clamped = start != original_start or end != original_end
            if clamped:
                warnings.append(
                    f"字幕轨 {track_id} 条目 {original_start}-{original_end} 越过镜头 {target.shot_id} 边界，已夹到 {start}-{end}"
                )
            cues.append(
                TimelineCue(
                    track_id=track_id,
                    shot_id=target.shot_id,
                    start_ms=start,
                    end_ms=end,
                    text=str(cue.get("text") or ""),
                    character_name=str(cue.get("character_name") or ""),
                    clamped=clamped,
                    clamp_reason="subtitle_outside_shot_boundary" if clamped else "",
                ).to_dict()
            )
        serialized.append({**dict(track), "cues": cues})
    return serialized


def build_post_production_plan(
    shots: list[Mapping[str, Any]],
    av_config: Mapping[str, Any] | None = None,
    *,
    project_id: str = "",
    fps: int = 24,
    capabilities: Mapping[str, Any] | None = None,
) -> PostProductionPlan:
    """从镜头与字幕/音频配置生成可审查时间线。"""

    av_config = av_config or {}
    warnings: list[str] = []
    normalized_shots = [dict(shot) for shot in shots]
    windows = [_execution_window(shot) for shot in normalized_shots]
    durations = [window[1] if window is not None else _normal_duration_ms(shot) for shot, window in zip(normalized_shots, windows)]
    nominal_spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for shot, duration in zip(normalized_shots, durations):
        nominal_spans[_normal_shot_id(shot)] = (cursor, cursor + duration)
        cursor += duration

    transitions: list[TransitionSpec] = []
    boundaries: list[TransitionSpec | None] = [None]
    ffmpeg_caps = (capabilities or {}).get("ffmpeg") or {}
    supported_transitions = set(ffmpeg_caps.get("supported_transitions") or SUPPORTED_TRANSITIONS)
    for index in range(1, len(normalized_shots)):
        boundary = _resolve_boundary(
            normalized_shots[index - 1],
            normalized_shots[index],
            index,
            warnings,
            supported_transitions,
        )
        boundaries.append(boundary)
        transitions.append(boundary)
    boundaries.append(None)

    # 输出时间线允许叠化/推拉/划像重叠；镜头源区间保持原始 duration。
    timeline_starts: list[int] = []
    timeline_ends: list[int] = []
    output_cursor = 0
    for index, duration in enumerate(durations):
        if index == 0:
            start = 0
        else:
            overlap = boundaries[index].duration_ms if boundaries[index] else 0
            start = max(0, timeline_ends[-1] - overlap)
        end = start + duration
        timeline_starts.append(start)
        timeline_ends.append(end)
        output_cursor = max(output_cursor, end)
    total_duration_ms = output_cursor

    audio_tracks = [dict(item) for item in av_config.get("audio_tracks") or []]
    # 字幕绑定需要 ShotTimelineEntry，先建主体再生成。
    entries: list[ShotTimelineEntry] = []
    for index, shot in enumerate(normalized_shots):
        shot_id = _normal_shot_id(shot)
        duration = durations[index]
        entry_audio: list[TimelineAudio] = []
        entry_dialogue: list[TimelineDialogue] = []
        speaker = _speaker_for_shot(shot)
        dialogue_lines = parse_shot_dialogue(
            shot.get("dialogue_timing") or shot.get("dialogue"),
            fallback_speaker=speaker,
            default_emotion=str(shot.get("emotion") or "neutral"),
            warn_key=f"timeline shot {shot_id}",
        )
        if dialogue_lines:
            audio_path = str(shot.get("audio_path") or "")
            plan_window = windows[index]
            if plan_window is not None and plan_window[0].dialogue_timing:
                # 执行计划里的对白时间轴即 TTS 实测结果，直接采用；不再按
                # 字符比例二次推导（避免时间线与生成阶段各说各话）。
                raw_lines = [
                    {
                        "speaker": item.speaker or speaker,
                        "text": item.text,
                        "emotion": str(shot.get("emotion") or "neutral"),
                        "start_ms": item.start_ms,
                        "end_ms": item.end_ms,
                    }
                    for item in plan_window[0].dialogue_timing
                ]
            else:
                raw_lines = [
                    {
                        "speaker": line.speaker or speaker,
                        "text": line.line,
                        "emotion": line.emotion or str(shot.get("emotion") or "neutral"),
                        "start_ms": line.start_ms,
                        "end_ms": line.end_ms,
                    }
                    for line in dialogue_lines
                ]
                missing_timing = [item for item in raw_lines if item["start_ms"] is None or item["end_ms"] is None]
                if missing_timing:
                    total_chars = sum(max(1, len(str(item["text"]))) for item in missing_timing)
                    cursor_rel = 0
                    for item in missing_timing:
                        share = max(1, int(round(duration * max(1, len(str(item["text"]))) / total_chars)))
                        item["start_ms"] = cursor_rel
                        item["end_ms"] = min(duration, cursor_rel + share)
                        cursor_rel = item["end_ms"]
            for line_index, line in enumerate(raw_lines):
                rel_start = max(0, min(duration, int(line["start_ms"] or 0)))
                rel_end = max(rel_start, min(duration, int(line["end_ms"] or rel_start)))
                if rel_end <= rel_start:
                    rel_end = min(duration, rel_start + 1)
                audio_id = f"dialogue:{shot_id}:{line_index + 1}"
                entry_dialogue.append(
                    TimelineDialogue(
                        shot_id=shot_id,
                        speaker=str(line["speaker"] or speaker),
                        text=str(line["text"] or ""),
                        emotion=str(line["emotion"] or shot.get("emotion") or "neutral"),
                        start_ms=timeline_starts[index] + rel_start,
                        end_ms=timeline_starts[index] + rel_end,
                        audio_id=audio_id,
                    )
                )
                entry_audio.append(
                    TimelineAudio(
                        id=audio_id,
                        kind="dialogue",
                        source_path=audio_path,
                        start_ms=timeline_starts[index] + rel_start,
                        end_ms=timeline_starts[index] + rel_end,
                        shot_id=shot_id,
                        speaker=str(line["speaker"] or speaker),
                        emotion=str(line["emotion"] or shot.get("emotion") or "neutral"),
                        clip_start_ms=rel_start,
                        clip_end_ms=rel_end,
                    )
                )
        for track in audio_tracks:
            if str(track.get("shot_id") or "") != shot_id:
                continue
            track_start = int(track.get("start_ms") or 0)
            track_duration = int(track.get("source_duration_ms") or 0)
            if track_duration <= 0:
                track_duration = duration
            start = max(timeline_starts[index], track_start)
            end = min(timeline_ends[index], track_start + track_duration)
            if end <= start:
                continue
            entry_audio.append(
                TimelineAudio(
                    id=str(track.get("id") or ""),
                    kind=str(track.get("kind") or "dialogue"),
                    source_path=str(track.get("resolved_source_path") or track.get("source_path") or ""),
                    start_ms=start,
                    end_ms=end,
                    shot_id=shot_id,
                    speaker=speaker,
                    emotion=str(shot.get("emotion") or "neutral"),
                    clip_start_ms=max(0, start - timeline_starts[index]),
                    clip_end_ms=max(0, end - timeline_starts[index]),
                )
            )
        camera = {
            "camera_movement": shot.get("camera_movement") or "静止",
            "camera_angle": shot.get("camera_angle") or "正面",
            "shot_type": shot.get("shot_type") or "medium",
            "prompt_strategy": CAMERA_MOVEMENT_PROMPTS.get(
                str(shot.get("camera_movement") or "静止"),
                f"camera movement: {shot.get('camera_movement') or '静止'}",
            ),
        }
        entry = ShotTimelineEntry(
            shot_id=shot_id,
            sequence=int(shot.get("sequence") or index),
            source_in_ms=windows[index][2] if windows[index] is not None else 0,
            source_out_ms=windows[index][3] if windows[index] is not None else duration,
            timeline_start_ms=timeline_starts[index],
            timeline_end_ms=timeline_ends[index],
            duration_ms=duration,
            transition_in=boundaries[index].to_dict() if boundaries[index] else None,
            transition_out=boundaries[index + 1].to_dict() if boundaries[index + 1] else None,
            camera=camera,
            emotion=str(shot.get("emotion") or "neutral"),
            dialogue=entry_dialogue,
            audio=entry_audio,
            effects=[
                {
                    "type": "camera_movement",
                    "value": camera["camera_movement"],
                    "renderer": "ffmpeg_zoompan_or_video_prompt",
                    "prompt_strategy": camera["prompt_strategy"],
                },
                {"type": "emotion", "value": str(shot.get("emotion") or "neutral")},
                {"type": "color_profile", "value": "scene_group_locked"},
            ],
        )
        entries.append(entry)

    # 将字幕绑定回填到每条镜头。
    subtitle_tracks = _clamp_subtitles(av_config.get("subtitle_tracks") or [], entries, nominal_spans, warnings)
    normalized_subtitles: list[dict[str, Any]] = []
    for track in subtitle_tracks:
        cues = track.get("cues") or []
        by_shot: dict[str, list[dict[str, Any]]] = {}
        for cue in cues:
            by_shot.setdefault(str(cue.get("shot_id") or ""), []).append(cue)
        normalized_subtitles.append({**track, "cues": cues})
        for entry in entries:
            entry.subtitles.extend(TimelineCue(**cue) for cue in by_shot.get(entry.shot_id, []))

    # 对白即使尚未在字幕工作台落 cue，也必须在计划里拥有同时间、同说话人的
    # 可审查字幕槽位；真正烧录仍只使用用户启用的字幕轨，避免擅自开启字幕。
    for entry in entries:
        existing = {(cue.start_ms, cue.end_ms, cue.text) for cue in entry.subtitles}
        for dialogue in entry.dialogue:
            key = (dialogue.start_ms, dialogue.end_ms, dialogue.text)
            if key in existing:
                continue
            entry.subtitles.append(
                TimelineCue(
                    track_id="derived-dialogue",
                    shot_id=entry.shot_id,
                    start_ms=dialogue.start_ms,
                    end_ms=dialogue.end_ms,
                    text=dialogue.text,
                    character_name=dialogue.speaker,
                    clamp_reason="derived_from_dialogue_timeline",
                )
            )

    return PostProductionPlan(
        version=1,
        project_id=project_id,
        fps=int(fps or 24),
        total_duration_ms=total_duration_ms,
        shots=entries,
        transitions=transitions,
        audio_tracks=audio_tracks,
        subtitle_tracks=normalized_subtitles,
        warnings=warnings,
        capabilities=dict(capabilities or {}),
    )


__all__ = [
    "CAMERA_MOVEMENT_PROMPTS",
    "FFMPEG_TRANSITION_FILTERS",
    "PostProductionPlan",
    "SUPPORTED_TRANSITIONS",
    "TransitionSpec",
    "build_post_production_plan",
    "normalize_transition",
]
