"""字幕工作台的纯文本处理：SRT/VTT 解析与序列化、内容校验、ASS 烧录文档。

本模块不做任何 IO（文件 / 数据库 / ffprobe 一概不碰），路由层与 FFmpegService
共享这里的实现，保证「导入导出无损」与「预览样式 = 渲染样式」由同一段代码保证。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from services.shot_dialogue import parse_shot_dialogue

from config import settings


class SubtitleValidationError(ValueError):
    """字幕内容未通过长度 / 非法字符 / 时间校验。"""


@dataclass
class SubtitleCueData:
    """与 ORM 解耦的字幕条目：路由 DTO、解析器与 ASS 生成共用。"""

    start_ms: int
    end_ms: int
    text: str
    character_name: str = ""
    client_id: str = ""
    order_index: int = 0

    def clamp(self) -> "SubtitleCueData":
        self.start_ms = max(0, int(self.start_ms))
        self.end_ms = max(0, int(self.end_ms))
        return self


# --- 时间码 -----------------------------------------------------------------

_SRT_TIME = re.compile(
    r"^(?P<h>\d{1,3}):(?P<m>\d{1,2}):(?P<s>\d{1,2})[,.](?P<ms>\d{1,3})$"
)
_ARROW_SPLIT = re.compile(r"\s*-->\s*")


def _ms_from_match(match: re.Match) -> int:
    ms = match.group("ms").ljust(3, "0")
    return (
        int(match.group("h")) * 3_600_000
        + int(match.group("m")) * 60_000
        + int(match.group("s")) * 1_000
        + int(ms)
    )


def format_srt_time(total_ms: int) -> str:
    total_ms = max(0, int(round(total_ms)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def format_vtt_time(total_ms: int) -> str:
    total_ms = max(0, int(round(total_ms)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def parse_timecode(value: str) -> int | None:
    """SRT（逗号）或 VTT（点号）毫秒时间码；不合法返回 None。"""

    text = str(value or "").strip()
    match = _SRT_TIME.match(text)
    if not match:
        return None
    return _ms_from_match(match)


# --- 非法字符与长度校验 ------------------------------------------------------

# 允许 Tab 与换行；其余 C0 控制字符、DEL 与 Unicode 行/段分隔符一律拒绝。
_FORBIDDEN_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u2028\u2029]")


def validate_cue_text(text: str) -> str:
    """校验单条字幕文本：长度上限 + 非法字符。返回规范化后的文本。"""

    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    stripped = value.strip()
    if not stripped:
        raise SubtitleValidationError("字幕文本不能为空")
    if len(value) > settings.MAX_SUBTITLE_CUE_CHARS:
        raise SubtitleValidationError(
            f"字幕文本超过 {settings.MAX_SUBTITLE_CUE_CHARS} 字符上限"
        )
    if _FORBIDDEN_CHARS.search(value):
        raise SubtitleValidationError("字幕文本包含非法控制字符")
    return value


def validate_character_name(name: str) -> str:
    value = str(name or "").strip()
    if len(value) > settings.MAX_SUBTITLE_CHARACTER_CHARS:
        raise SubtitleValidationError("字幕角色名过长")
    if _FORBIDDEN_CHARS.search(value):
        raise SubtitleValidationError("字幕角色名包含非法字符")
    return value


def validate_cues(cues: Iterable[SubtitleCueData]) -> list[SubtitleCueData]:
    """整批校验：文本、时间顺序、条数上限；返回按 start_ms 排序的结果。"""

    items = [cue.clamp() for cue in cues]
    if len(items) > settings.MAX_SUBTITLE_CUES:
        raise SubtitleValidationError(f"字幕条数超过 {settings.MAX_SUBTITLE_CUES} 条上限")
    for index, cue in enumerate(items):
        cue.text = validate_cue_text(cue.text)
        cue.character_name = validate_character_name(cue.character_name)
        if cue.end_ms <= cue.start_ms:
            raise SubtitleValidationError(f"第 {index + 1} 条字幕的结束时间必须晚于开始时间")
        if cue.end_ms - cue.start_ms < 50:
            raise SubtitleValidationError(f"第 {index + 1} 条字幕时长不足 50 毫秒")
    items.sort(key=lambda cue: (cue.start_ms, cue.end_ms))
    for index, cue in enumerate(items):
        cue.order_index = index
    return items


# --- SRT ---------------------------------------------------------------------

def serialize_srt(cues: Sequence[SubtitleCueData]) -> str:
    blocks: list[str] = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n{format_srt_time(cue.start_ms)} --> {format_srt_time(cue.end_ms)}\n{cue.text}\n"
        )
    return "\n".join(blocks)


def parse_srt(content: str) -> list[SubtitleCueData]:
    text = str(content or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[SubtitleCueData] = []
    blocks = re.split(r"\n{2,}", text.strip("\n"))
    for block in blocks:
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        # 序号行可选：有些导出器省略。找到时间码行即认定为字幕块。
        time_index = next(
            (i for i, line in enumerate(lines) if _ARROW_SPLIT.search(line) and _SRT_TIME.match(_ARROW_SPLIT.split(line)[0].strip())),
            None,
        )
        if time_index is None:
            continue
        start_raw, end_raw = _ARROW_SPLIT.split(lines[time_index], 1)
        end_raw = end_raw.strip().split(" ")[0]  # 丢弃可能的坐标定位后缀
        start_ms = parse_timecode(start_raw)
        end_ms = parse_timecode(end_raw)
        if start_ms is None or end_ms is None:
            continue
        body = "\n".join(lines[time_index + 1 :]).strip()
        if not body:
            continue
        cues.append(SubtitleCueData(start_ms=start_ms, end_ms=end_ms, text=body))
    return cues


# --- VTT ---------------------------------------------------------------------

def serialize_vtt(cues: Sequence[SubtitleCueData]) -> str:
    lines = ["WEBVTT", ""]
    for index, cue in enumerate(cues, start=1):
        lines.append(str(index))
        lines.append(f"{format_vtt_time(cue.start_ms)} --> {format_vtt_time(cue.end_ms)}")
        lines.append(cue.text)
        lines.append("")
    return "\n".join(lines)


def parse_vtt(content: str) -> list[SubtitleCueData]:
    text = str(content or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if not text.lstrip().startswith("WEBVTT"):
        raise SubtitleValidationError("VTT 内容缺少 WEBVTT 头")
    cues: list[SubtitleCueData] = []
    for block in re.split(r"\n{2,}", text.strip("\n")):
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        header = lines[0].strip()
        if header.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            # NOTE/STYLE/REGION 元数据块直接忽略；WEBVTT 头块内不含时间码。
            if header.startswith("WEBVTT") and not any(_ARROW_SPLIT.search(line) for line in lines):
                continue
            if not any(_ARROW_SPLIT.search(line) for line in lines):
                continue
        time_index = next(
            (i for i, line in enumerate(lines) if _ARROW_SPLIT.search(line) and _SRT_TIME.match(_ARROW_SPLIT.split(line)[0].strip().replace(",", "."))),
            None,
        )
        if time_index is None:
            continue
        start_raw, end_raw = _ARROW_SPLIT.split(lines[time_index], 1)
        end_raw = end_raw.strip().split(" ")[0]
        start_ms = parse_timecode(start_raw.strip().replace(",", "."))
        end_ms = parse_timecode(end_raw.replace(",", "."))
        if start_ms is None or end_ms is None:
            continue
        body = "\n".join(lines[time_index + 1 :]).strip()
        if not body:
            continue
        # 去掉 VTT 行内_voice_ / &lt; 类标记属于导出器职责；这里保留纯文本。
        cues.append(SubtitleCueData(start_ms=start_ms, end_ms=end_ms, text=body))
    return cues


def serialize_cues(cues: Sequence[SubtitleCueData], fmt: str) -> str:
    normalized = str(fmt or "srt").strip().lower()
    if normalized in {"srt", "application/x-subrip"}:
        return serialize_srt(cues)
    if normalized in {"vtt", "text/vtt"}:
        return serialize_vtt(cues)
    raise SubtitleValidationError("不支持的字幕格式，仅支持 SRT / VTT")


def parse_cues(content: str, fmt: str) -> list[SubtitleCueData]:
    normalized = str(fmt or "srt").strip().lower()
    if len(content) > settings.MAX_SUBTITLE_IMPORT_CHARS:
        raise SubtitleValidationError("导入的字幕内容过长")
    if normalized in {"srt", "application/x-subrip"}:
        return parse_srt(content)
    if normalized in {"vtt", "text/vtt"}:
        return parse_vtt(content)
    raise SubtitleValidationError("不支持的字幕格式，仅支持 SRT / VTT")


# --- 从镜头对白生成字幕 -------------------------------------------------------

MIN_CUE_MS = 800


@dataclass
class DialogueLineInput:
    """镜头内单句对白（结构化）：时间轴为相对镜头起点的毫秒偏移。"""

    speaker: str = ""
    line: str = ""
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass
class ShotDialogueInput:
    """生成字幕所需的镜头信息（由路由层从 Shot ORM 构建）。

    ``lines`` 为结构化对白（新口径，逐句生成 cue）；为空时回落到旧口径——
    整镜一条 ``dialogue`` 文本 + 单一 ``character_name``。
    """

    shot_id: str
    sequence: int
    start_ms: int  # 该镜头在时间线上的起点（按镜头时长累积）
    duration_ms: int
    dialogue: Any
    character_name: str = ""
    tts_duration_ms: int = 0  # TTS 音频实际时长；0 表示未知
    lines: list[DialogueLineInput] = field(default_factory=list)


def _cue_text_safe(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    try:
        return validate_cue_text(text)
    except SubtitleValidationError:
        return text[: settings.MAX_SUBTITLE_CUE_CHARS].strip()


def _cues_from_lines(shot: ShotDialogueInput, lines: Sequence[DialogueLineInput]) -> list[SubtitleCueData]:
    """逐句字幕：优先使用结构化对白的实测时间轴，缺失时按句均分镜头预算。"""

    shot_start = max(0, int(shot.start_ms))
    shot_end = shot_start + max(0, int(shot.duration_ms))
    usable = [line for line in lines if str(line.line or "").strip()]
    if not usable:
        return []
    explicit = all(
        line.start_ms is not None and line.end_ms is not None and int(line.end_ms) > int(line.start_ms)
        for line in usable
    )
    cues: list[SubtitleCueData] = []
    if explicit:
        for line in usable:
            rel_start = max(0, int(line.start_ms or 0))
            rel_end = max(rel_start, int(line.end_ms or 0))
            start = min(shot_start + rel_start, shot_end)
            end = min(shot_start + rel_end, shot_end)
            if end <= start:
                end = min(start + MIN_CUE_MS, shot_end)
            text = _cue_text_safe(line.line)
            if text and end > start:
                cues.append(
                    SubtitleCueData(
                        start_ms=start,
                        end_ms=end,
                        text=text,
                        character_name=str(line.speaker or "").strip(),
                    )
                )
        return cues

    # 时间轴不完整（原生音视频镜头 / 旧数据）：TTS 总时长（不越出镜头）按句
    # 均分，保证多句对白严格按顺序错开。
    budget = max(0, int(shot.duration_ms))
    if shot.tts_duration_ms > 0:
        budget = max(min(int(shot.tts_duration_ms), budget), min(MIN_CUE_MS, budget))
    share = max(50, budget // max(1, len(usable)))
    for index, line in enumerate(usable):
        start = min(shot_start + index * share, shot_end)
        end = min(start + share, shot_end)
        text = _cue_text_safe(line.line)
        if text and end > start:
            cues.append(
                SubtitleCueData(
                    start_ms=start,
                    end_ms=end,
                    text=text,
                    character_name=str(line.speaker or "").strip(),
                )
            )
    return cues


def _shot_lines(shot: ShotDialogueInput) -> list[DialogueLineInput]:
    """镜头的逐句对白：优先取显式 ``lines``；否则尝试把 ``dialogue`` 解析为
    结构化列表（列表 / JSON 数组文本）。旧版纯文本返回空，走整镜单条口径。"""

    lines = list(getattr(shot, "lines", None) or [])
    if lines:
        return lines
    raw = shot.dialogue
    if isinstance(raw, (list, tuple)) or (isinstance(raw, str) and raw.strip().startswith("[")):
        from services.shot_dialogue import parse_shot_dialogue

        parsed = parse_shot_dialogue(raw, warn_key=f"subtitle shot {shot.shot_id}")
        return [
            DialogueLineInput(speaker=line.speaker, line=line.line, start_ms=line.start_ms, end_ms=line.end_ms)
            for line in parsed
        ]
    return []


def cues_from_shots(shots: Sequence[ShotDialogueInput]) -> list[SubtitleCueData]:
    """按镜头区间生成字幕：结构化对白逐句出 cue，旧口径整镜一条。

    时长优先取 TTS 实际时长，且不超出镜头边界；逐句 cue 严格按说话顺序排列。
    """

    cues: list[SubtitleCueData] = []
    for shot in shots:
        lines = _shot_lines(shot)
        if lines:
            cues.extend(_cues_from_lines(shot, lines))
            continue
        text = _cue_text_safe(shot.dialogue)
        if not text:
            continue
        start = max(0, int(shot.start_ms))
        available = max(0, int(shot.duration_ms))
        duration = available
        if shot.tts_duration_ms > 0:
            # 对白读完即可收字幕，但至少停留 MIN_CUE_MS 且不越出镜头。
            duration = max(min(int(shot.tts_duration_ms), available), min(MIN_CUE_MS, available))
        duration = max(50, duration)
        cues.append(
            SubtitleCueData(
                start_ms=start,
                end_ms=start + duration,
                text=text,
                character_name=str(shot.character_name or "").strip(),
            )
        )
    return cues


# --- ASS（烧录字幕文档） ------------------------------------------------------

_ASS_HEADER = (
    "[Script Info]\n"
    "; Generated by ComicAgent subtitle workbench\n"
    "ScriptType: v4.00+\n"
    "PlayResX: {width}\n"
    "PlayResY: {height}\n"
    "WrapStyle: 0\n"
    "ScaledBorderAndShadow: yes\n"
    "\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
    "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
    "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
)

_POSITION_ALIGNMENT = {"top": 8, "middle": 9, "bottom": 2}


def _ass_color(hex_color: str, alpha: str = "00") -> str:
    """#RRGGBB → ASS 的 &HAABBGGRR（不透明）。"""

    value = str(hex_color or "").strip().lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", value):
        value = "FFFFFF"
    r, g, b = value[0:2], value[2:4], value[4:6]
    return f"&H{alpha}{b}{g}{r}".upper()


def _escape_ass_text(text: str) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\\", "\\\\")
    value = value.replace("{", "\\{").replace("}", "\\}")
    value = value.replace("\n", "\\N")
    return value


def _ass_time(total_ms: int) -> str:
    total_ms = max(0, int(round(total_ms)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    centis = millis // 10
    return f"{hours:d}:{minutes:02d}:{seconds:02d}.{centis:02d}"


@dataclass
class SubtitleStyle:
    """SubtitleTrack 的样式投影：ASS 生成只依赖这些字段。"""

    font_family: str = "sans-serif"
    font_size: int = 54
    primary_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: int = 3
    bold: bool = False
    position: str = "bottom"
    safe_margin: int = 54

    def scaled(self, width: int, height: int) -> "SubtitleStyle":
        """font_size / safe_margin 以画面短边 1080 像素为基准，按实际分辨率等比缩放。

        竖版 1080p（1080×1920）与横版 1080p（1920×1080）的短边都是 1080，
        因此两种画幅下默认字号一致；短边更小的分辨率按比例缩小。
        """

        factor = max(0.05, min(int(width), int(height)) / 1080.0)
        return SubtitleStyle(
            font_family=self.font_family,
            font_size=max(8, int(round(self.font_size * factor))),
            primary_color=self.primary_color,
            outline_color=self.outline_color,
            outline_width=max(0, int(round(self.outline_width * factor))),
            bold=self.bold,
            position=self.position if self.position in _POSITION_ALIGNMENT else "bottom",
            safe_margin=max(0, int(round(self.safe_margin * factor))),
        )


def build_ass_document(style: SubtitleStyle, cues: Sequence[SubtitleCueData], width: int, height: int) -> str:
    """生成用于 subtitles 滤镜烧录的完整 ASS 文档（纯函数，可单测）。"""

    scaled = style.scaled(width, height)
    alignment = _POSITION_ALIGNMENT.get(scaled.position, 2)
    margin_h = scaled.safe_margin
    margin_v = scaled.safe_margin
    font_name = re.sub(r"[,:;\\]", " ", str(scaled.font_family or "sans-serif")).strip() or "sans-serif"
    header = _ASS_HEADER.format(width=int(width), height=int(height))
    style_line = (
        f"Style: Default,{font_name},{scaled.font_size},"
        f"{_ass_color(scaled.primary_color)},&H000000FF,{_ass_color(scaled.outline_color)},&H00000000,"
        f"{-1 if scaled.bold else 0},0,0,0,100,100,0,0,1,{max(0, scaled.outline_width)},0,"
        f"{alignment},{margin_h},{margin_h},{margin_v},1"
    )
    lines = [header, style_line, "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for cue in cues:
        name = _escape_ass_text(cue.character_name)
        lines.append(
            f"Dialogue: 0,{_ass_time(cue.start_ms)},{_ass_time(cue.end_ms)},Default,{name},0,0,0,,{_escape_ass_text(cue.text)}"
        )
    return "\n".join(lines) + "\n"


@dataclass
class SubtitleOverlapWarning:
    index_a: int
    index_b: int
    overlap_ms: int


def detect_cue_overlaps(cues: Sequence[SubtitleCueData]) -> list[SubtitleOverlapWarning]:
    """相邻字幕重叠检测（供工作台显示警告；不阻断保存）。"""

    warnings: list[SubtitleOverlapWarning] = []
    ordered = sorted(cues, key=lambda cue: (cue.start_ms, cue.end_ms))
    for index in range(1, len(ordered)):
        previous, current = ordered[index - 1], ordered[index]
        if current.start_ms < previous.end_ms:
            warnings.append(
                SubtitleOverlapWarning(
                    index_a=index - 1,
                    index_b=index,
                    overlap_ms=previous.end_ms - current.start_ms,
                )
            )
    return warnings


__all__ = [
    "MIN_CUE_MS",
    "DialogueLineInput",
    "ShotDialogueInput",
    "SubtitleCueData",
    "SubtitleOverlapWarning",
    "SubtitleStyle",
    "SubtitleValidationError",
    "build_ass_document",
    "cues_from_shots",
    "detect_cue_overlaps",
    "format_srt_time",
    "format_vtt_time",
    "parse_cues",
    "parse_srt",
    "parse_timecode",
    "parse_vtt",
    "serialize_cues",
    "serialize_srt",
    "serialize_vtt",
    "validate_character_name",
    "validate_cue_text",
    "validate_cues",
]
