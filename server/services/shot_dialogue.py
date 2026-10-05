"""镜头对白的结构化表示与旧格式迁移。

历史上 ``shots.dialogue`` 是一个纯文本字符串，多角色对白的说话人在分镜阶段
就被压扁，配音只能拿 ``characters_in_scene[0]`` 冒充说话人。现在镜头对白统一
为结构化列表，每句对白都带自己的 ``speaker``：

- ``speaker``：说话人名字，必须能在项目角色列表里找到，否则配音阶段会给出
  可追踪警告（绝不静默回落到第一个角色）；
- ``line``：台词文本；
- ``emotion`` / ``action``：该句自带的情绪与动作（可为空）；
- ``start_ms`` / ``end_ms``：相对镜头起点的毫秒时间轴。TTS 逐句生成后会用
  实测时长回填；LLM 估算值或空值在字幕/时间线计算时按镜头时长兜底。

存储兼容：``shots.dialogue`` 仍是 TEXT 列，但内容是本模块序列化的 JSON 数组；
旧项目的纯文本字符串在读取时自动迁移成单条对白（说话人按调用方给的兜底值，
通常是 ``characters_in_scene[0]``，并记录 warning，绝不无声发生）。

本模块只做纯数据变换，不碰 IO / DB / TTS，方便在任何层安全引用。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

logger = logging.getLogger(__name__)

MAX_DIALOGUE_LINES_PER_SHOT = 24
MAX_SPEAKER_CHARS = 60
MAX_LINE_CHARS = 2000
MAX_TIMELINE_MS = 3_600_000

_SPEAKER_KEYS = ("speaker", "character", "role", "name")
_LINE_KEYS = ("line", "text", "dialogue")
_ACTION_KEYS = ("action", "act")
_EMOTION_KEYS = ("emotion", "tone")

# 旧格式迁移警告只按 shot 去重一次，避免高频序列化路径刷屏。
_legacy_warned: set[str] = set()


@dataclass
class DialogueLine:
    """单句结构化对白（镜头内相对时间轴，毫秒）。"""

    speaker: str = ""
    line: str = ""
    emotion: str = ""
    action: str = ""
    start_ms: int | None = None
    end_ms: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "line": self.line,
            "emotion": self.emotion,
            "action": self.action,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }


def _clamp_ms(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if number < 0 or number > MAX_TIMELINE_MS:
        return None
    return number


def _clean_text(value: Any, limit: int) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set, bool)):
        return ""
    return str(value).strip()[:limit]


def _pick(item: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def _line_from_item(item: Any, *, fallback_speaker: str, default_emotion: str) -> DialogueLine | None:
    """把任意形态的单条对白（dict / str / DialogueLine）归一为 DialogueLine。"""

    if isinstance(item, DialogueLine):
        return replace(item)
    if isinstance(item, str):
        text = item.strip()
        return (
            DialogueLine(speaker=fallback_speaker, line=text[:MAX_LINE_CHARS], emotion=default_emotion)
            if text
            else None
        )
    if not isinstance(item, dict):
        return None
    text = _clean_text(_pick(item, _LINE_KEYS), MAX_LINE_CHARS)
    if not text:
        return None
    speaker = _clean_text(_pick(item, _SPEAKER_KEYS), MAX_SPEAKER_CHARS) or fallback_speaker
    emotion = _clean_text(_pick(item, _EMOTION_KEYS), 40) or default_emotion
    action = _clean_text(_pick(item, _ACTION_KEYS), MAX_LINE_CHARS)
    return DialogueLine(
        speaker=speaker,
        line=text,
        emotion=emotion,
        action=action,
        start_ms=_clamp_ms(item.get("start_ms")),
        end_ms=_clamp_ms(item.get("end_ms")),
    )


def parse_shot_dialogue(
    raw: Any,
    *,
    fallback_speaker: str = "",
    default_emotion: str = "",
    warn_key: str = "",
) -> list[DialogueLine]:
    """把任意存储形态的镜头对白解析为结构化列表。

    接受：结构化列表（dict / DialogueLine 混合）、JSON 数组字符串、旧版纯文本
    字符串。旧字符串迁移为单条对白，说话人取 ``fallback_speaker``（通常为
    ``characters_in_scene[0]``，这是旧配音行为的如实记录），并记录可追踪
    warning（``warn_key`` 用于定位镜头，例如 ``shot xxx``）。
    """

    if raw in (None, "", []):
        return []
    fallback_speaker = str(fallback_speaker or "").strip()[:MAX_SPEAKER_CHARS]
    default_emotion = str(default_emotion or "").strip()

    if isinstance(raw, (list, tuple)):
        items: list[Any] = list(raw)[:MAX_DIALOGUE_LINES_PER_SHOT]
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                loaded = json.loads(text)
            except ValueError:
                loaded = None
            if isinstance(loaded, list):
                return parse_shot_dialogue(
                    loaded,
                    fallback_speaker=fallback_speaker,
                    default_emotion=default_emotion,
                    warn_key=warn_key,
                )
        # 旧版纯文本：迁移为单条对白，说话人如实标注兜底来源。
        if warn_key and warn_key not in _legacy_warned:
            _legacy_warned.add(warn_key)
            logger.warning(
                "镜头对白为旧版纯文本，已迁移为单条结构化对白: key=%s speaker=%r",
                warn_key,
                fallback_speaker,
            )
        line = DialogueLine(speaker=fallback_speaker, line=text[:MAX_LINE_CHARS], emotion=default_emotion)
        return [line]
    else:
        return parse_shot_dialogue(
            [raw], fallback_speaker=fallback_speaker, default_emotion=default_emotion, warn_key=warn_key
        )

    lines: list[DialogueLine] = []
    for item in items:
        parsed = _line_from_item(item, fallback_speaker=fallback_speaker, default_emotion=default_emotion)
        if parsed is not None and parsed.line:
            lines.append(parsed)
    return lines


def serialize_dialogue_lines(lines: Iterable[DialogueLine]) -> str:
    """结构化对白 → 数据库存储文本（JSON 数组；空对白存空串保持真值判断兼容）。"""

    payload = [line.as_dict() for line in lines if str(line.line).strip()]
    if not payload:
        return ""
    return json.dumps(payload, ensure_ascii=False)


def dialogue_lines_payload(lines: Iterable[DialogueLine]) -> list[dict[str, Any]]:
    """结构化对白 → API DTO 形态（字段齐全，空时间轴为 None）。"""

    return [line.as_dict() for line in lines if str(line.line).strip()]


def dialogue_plain_text(lines: Iterable[DialogueLine]) -> str:
    """全部台词拼接的纯文本（Prompt 拼接、成本估算用）。"""

    return "\n".join(str(line.line).strip() for line in lines if str(line.line).strip())


def dialogue_display_text(lines: Iterable[DialogueLine]) -> str:
    """带说话人前缀的展示文本（预览、音轨工作台提示用）。"""

    parts: list[str] = []
    for line in lines:
        text = str(line.line).strip()
        if not text:
            continue
        parts.append(f"{line.speaker}：{text}" if line.speaker else text)
    return "\n".join(parts)


def dialogue_total_chars(lines: Iterable[DialogueLine]) -> int:
    """TTS 计费口径的字符数：逐句求和，剔除空白。"""

    return sum(len(re.sub(r"\s", "", str(line.line))) for line in lines if str(line.line).strip())


def assign_line_timings(durations_ms: Sequence[int]) -> list[tuple[int, int]]:
    """按实测单句时长推导 (start_ms, end_ms)：顺序累加，严格单调不减。

    这是「同一镜头多句对白必须按时间顺序生成」的纯函数核心，供逐句 TTS
    拼接与回归测试共用。
    """

    timings: list[tuple[int, int]] = []
    cursor = 0
    for duration in durations_ms:
        safe = max(0, int(duration))
        timings.append((cursor, cursor + safe))
        cursor += safe
    return timings


def resolve_speaker_voice(
    speaker: str,
    characters: Sequence[dict[str, Any]],
    *,
    context: str = "",
) -> str:
    """按说话人名字解析角色音色。

    说话人为空或不在角色列表中时，记录带上下文的可追踪 warning 并返回空音色
    （TTS 端点默认音色）。绝不静默改用第一个角色的音色——那是本改造要修掉的
    错配来源。
    """

    name = str(speaker or "").strip()
    if name:
        for character in characters or []:
            if str(character.get("name") or "").strip() == name:
                return str(character.get("voice_id") or "").strip()
    logger.warning(
        "对白说话人不在角色列表中，使用端点默认音色: speaker=%r context=%s known=%s",
        name,
        context or "-",
        [str(item.get("name") or "") for item in (characters or [])],
    )
    return ""


def normalize_character_names(characters: Iterable[dict[str, Any]]) -> list[str]:
    return [str(item.get("name") or "").strip() for item in characters or [] if str(item.get("name") or "").strip()]


def warn_unknown_speakers(
    lines: Sequence[DialogueLine], characters: Iterable[dict[str, Any]], *, context: str
) -> list[str]:
    """说话人校验：返回不在角色列表中的说话人（去重），并记录可追踪 warning。"""

    known = set(normalize_character_names(characters))
    unknown: list[str] = []
    for line in lines:
        speaker = str(line.speaker or "").strip()
        if not speaker or speaker in known:
            continue
        if speaker not in unknown:
            unknown.append(speaker)
    if unknown:
        logger.warning(
            "镜头对白存在未登记说话人（不会静默改配第一个角色）: context=%s unknown=%s known=%s",
            context,
            unknown,
            sorted(known),
        )
    return unknown


__all__ = [
    "DialogueLine",
    "MAX_DIALOGUE_LINES_PER_SHOT",
    "assign_line_timings",
    "dialogue_display_text",
    "dialogue_lines_payload",
    "dialogue_plain_text",
    "dialogue_total_chars",
    "normalize_character_names",
    "parse_shot_dialogue",
    "resolve_speaker_voice",
    "serialize_dialogue_lines",
    "warn_unknown_speakers",
]
