"""LLM 返回值的强类型 schema。

模型返回的是不可信输入：字段可能缺失、类型错误、枚举非法、时长为负、数组超大
或文本超长。这里为剧本解析与分镜生成定义独立的 Pydantic 输出模型，把「可安全修
复」与「无法使用」分开处理：

- 可修复：枚举别名归一化、超长文本截断、数值钳制到合法区间、数组截断到上限；
- 无法修复：该条记录被丢弃，并记录字段路径与原因（不含模型原文）；
- 顶层结构错误：抛出 LLMOutputError，由节点转成可读的业务错误。

诊断信息只包含字段路径、长度与类型等元数据，不写入 API Key、完整 prompt 或本地
绝对路径。
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any

from pydantic import AliasChoices, BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, ValidationInfo

from config import settings
from services.consistency_service import normalize_continuity_mode
from services.shot_dialogue import MAX_DIALOGUE_LINES_PER_SHOT, MAX_TIMELINE_MS
from services.style_templates import STYLE_TEMPLATES

logger = logging.getLogger(__name__)

MAX_CHARACTER_NAME = 60
MAX_ID_CHARS = 128
MAX_FEATURE_ITEMS = 12
MAX_FEATURE_CHARS = 200
MAX_APPEARANCE_KEYS = 24
MAX_APPEARANCE_VALUE_CHARS = 400
MAX_CHARACTERS_IN_SCENE = 12
MAX_LOGIC_ISSUES = 20
MAX_SEED = 2_147_483_647
MAX_ACTION_BEATS_PER_SHOT = 12


class LLMOutputError(ValueError):
    """模型输出无法使用时抛出的业务错误（消息可安全展示给用户）。"""

    def __init__(self, message: str, *, issues: list[str] | None = None) -> None:
        super().__init__(message)
        self.issues = list(issues or [])


# --- 枚举归一化 ---

EMOTIONS = ("neutral", "happy", "shy", "sad", "angry", "surprised")
SHOT_TYPES = ("wide", "medium", "close-up", "extreme_close")
CAMERA_ANGLES = ("正面", "侧面", "俯视", "仰视")
CAMERA_MOVEMENTS = ("静止", "推", "拉", "摇", "移", "跟", "升降", "环绕", "缓慢推进")
TRANSITIONS = ("cut", "fade", "dissolve", "white_flash", "push", "wipe")
# CONTINUITY_MODES 由一致性策略统一维护，分镜输出与执行阶段共用同一枚举。
SHOT_STATUSES = (
    "pending",
    "storyboard_done",
    "storyboard_approved",
    "video_generating",
    "video_done",
    "failed",
    "needs_review",
)

_EMOTION_ALIASES = {
    "calm": "neutral",
    "平静": "neutral",
    "neutral": "neutral",
    "joy": "happy",
    "开心": "happy",
    "happy": "happy",
    "shy": "shy",
    "害羞": "shy",
    "sad": "sad",
    "悲伤": "sad",
    "难过": "sad",
    "angry": "angry",
    "愤怒": "angry",
    "tense": "angry",
    "紧张": "angry",
    "surprised": "surprised",
    "惊讶": "surprised",
}
_SHOT_TYPE_ALIASES = {
    "wide": "wide",
    "wide shot": "wide",
    "establishing": "wide",
    "long shot": "wide",
    "全景": "wide",
    "远景": "wide",
    "medium": "medium",
    "medium shot": "medium",
    "mid shot": "medium",
    "中景": "medium",
    "close-up": "close-up",
    "close up": "close-up",
    "closeup": "close-up",
    "特写": "close-up",
    "近景": "close-up",
    "extreme_close": "extreme_close",
    "extreme close-up": "extreme_close",
    "extreme close up": "extreme_close",
    "extreme closeup": "extreme_close",
    "大特写": "extreme_close",
}
_CAMERA_ANGLE_ALIASES = {
    "正面": "正面",
    "front": "正面",
    "front view": "正面",
    "eye level": "正面",
    "侧面": "侧面",
    "side": "侧面",
    "side view": "侧面",
    "3/4": "侧面",
    "俯视": "俯视",
    "high": "俯视",
    "high angle": "俯视",
    "overhead": "俯视",
    "高机位": "俯视",
    "仰视": "仰视",
    "low": "仰视",
    "low angle": "仰视",
    "低机位": "仰视",
}
_CAMERA_MOVEMENT_ALIASES = {
    "静止": "静止",
    "static": "静止",
    "fixed": "静止",
    "still": "静止",
    "none": "静止",
    "推": "推",
    "push": "推",
    "push in": "推",
    "dolly in": "推",
    "zoom in": "推",
    "拉": "拉",
    "pull": "拉",
    "pull out": "拉",
    "zoom out": "拉",
    "摇": "摇",
    "pan": "摇",
    "移": "移",
    "平移": "移",
    "truck": "移",
    "跟": "跟",
    "跟随": "跟",
    "follow": "跟",
    "tracking": "跟",
    "升降": "升降",
    "crane": "升降",
    "环绕": "环绕",
    "orbit": "环绕",
    "arc": "环绕",
    "缓慢推进": "缓慢推进",
    "slow push": "缓慢推进",
    "slow push in": "缓慢推进",
}
_TRANSITION_ALIASES = {
    "cut": "cut",
    "hard cut": "cut",
    "硬切": "cut",
    "fade": "fade",
    "淡入淡出": "fade",
    "dissolve": "dissolve",
    "叠化": "dissolve",
    "white_flash": "white_flash",
    "white flash": "white_flash",
    "白闪": "white_flash",
    "push": "push",
    "推拉": "push",
    "wipe": "wipe",
    "划像": "wipe",
}
_STATUS_ALIASES = {status.replace("_", " "): status for status in SHOT_STATUSES}


def _normalize_choice(value: Any, canonical: tuple[str, ...], aliases: dict[str, str], default: str) -> str:
    """把模型给出的枚举写法收敛到项目真实支持的取值集。"""

    text = "" if value is None else str(value).strip()
    if not text:
        return default
    if text in canonical:
        return text
    key = text.lower().replace("_", " ").strip()
    if key in aliases:
        return aliases[key]
    raw = text.lower().replace(" ", "_")
    if raw in canonical:
        return raw
    return default


def normalize_emotion(value: Any) -> str:
    return _normalize_choice(value, EMOTIONS, _EMOTION_ALIASES, "neutral")


def normalize_shot_type(value: Any) -> str:
    return _normalize_choice(value, SHOT_TYPES, _SHOT_TYPE_ALIASES, "medium")


def normalize_camera_angle(value: Any) -> str:
    return _normalize_choice(value, CAMERA_ANGLES, _CAMERA_ANGLE_ALIASES, "正面")


def normalize_camera_movement(value: Any) -> str:
    return _normalize_choice(value, CAMERA_MOVEMENTS, _CAMERA_MOVEMENT_ALIASES, "静止")


def normalize_transition(value: Any) -> str:
    return _normalize_choice(value, TRANSITIONS, _TRANSITION_ALIASES, "cut")


def normalize_shot_status(value: Any) -> str:
    return _normalize_choice(value, SHOT_STATUSES, _STATUS_ALIASES, "pending")


def normalize_style_suggestion(value: Any, fallback: str = "anime") -> str:
    """画风建议：只接受内置风格 key，其余回落到调用方给定的风格。"""

    text = "" if value is None else str(value).strip()
    if text in STYLE_TEMPLATES:
        return text
    return fallback if fallback in STYLE_TEMPLATES else "anime"


# --- 字段级归一化 ---


def _text(limit: int):
    """文本字段：容器类型按「无法作为文本」归一化为空串，超长按上限截断。

    单字段类型错误不丢弃整条记录：分镜/角色/场景的结构信息比某个描述字段更值钱，
    这里记一条带字段名的 warning，让调用方用默认值继续流程。
    """

    def validate(value: Any, info: ValidationInfo) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list, tuple, set)):
            logger.warning("LLM 输出字段类型不可用,已置空: field=%s type=%s", info.field_name, type(value).__name__)
            return ""
        text = str(value).strip()
        if len(text) > limit:
            logger.warning("LLM 输出字段超长已截断: field=%s limit=%d actual=%d", info.field_name, limit, len(text))
            text = text[:limit]
        return text

    return BeforeValidator(validate)


def _optional_emotion(value: Any, info: ValidationInfo) -> str:
    """情绪字段：空值保持为空（由镜头级情绪兜底），非空归一化到合法枚举。"""

    del info
    if value in (None, ""):
        return ""
    return normalize_emotion(value)


def _shot_dialogue_list(limit: int = MAX_DIALOGUE_LINES_PER_SHOT):
    """镜头对白：接受结构化数组、旧版字符串或数组字符串，统一为逐句对象列表。

    每句对白必须有机会携带 speaker（speaker/character/role 均可）；旧版字符串
    形态迁移为 speaker 为空的单句，由配音阶段给出可追踪警告而不是冒名顶替。
    """

    def validate(value: Any, info: ValidationInfo) -> list:
        del info
        if value in (None, ""):
            return []
        if isinstance(value, (str, dict)):
            value = [value]
        if not isinstance(value, (list, tuple)):
            logger.warning("LLM 输出对白字段类型不可用,已置空: field=dialogue type=%s", type(value).__name__)
            return []
        items = list(value)
        if len(items) > limit:
            logger.warning("LLM 输出对白句数超长已截断: path=dialogue limit=%d actual=%d", limit, len(items))
            items = items[:limit]
        parsed: list[ShotDialogueLineOutput] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                item = {"line": item}
            if not isinstance(item, dict):
                logger.warning("LLM 输出对白条目被丢弃: path=dialogue[%d] 原因=应为对象", index)
                continue
            try:
                parsed.append(ShotDialogueLineOutput.model_validate(item))
            except ValidationError as exc:
                logger.warning("LLM 输出对白条目被丢弃: path=dialogue[%d] 原因=%s", index, describe_error(exc))
        return [line for line in parsed if line.line.strip()]

    return BeforeValidator(validate)


def _choice(normalizer, default: str):
    """枚举字段：归一化到合法取值，空值使用给定默认值。"""

    def validate(value: Any, info: ValidationInfo) -> str:
        del info
        return normalizer(value) if value not in (None, "") else default

    return BeforeValidator(validate)


def _continuity_mode():
    """连续性模式：合法值/别名归一化，缺失或非法值留空交给确定性规则兜底。"""

    def validate(value: Any, info: ValidationInfo) -> str:
        del info
        return normalize_continuity_mode(value, default="")

    return BeforeValidator(validate)


def _model_list(model: type[BaseModel], limit: int, path: str):
    """对象数组字段：逐条校验，丢弃无法修复的条目并记录路径与原因。"""

    def validate(value: Any, info: ValidationInfo) -> list:
        del info
        if value in (None, ""):
            return []
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError("应为数组")
        items = list(value)
        if len(items) > limit:
            logger.warning("LLM 输出数组超长已截断: path=%s limit=%d actual=%d", path, limit, len(items))
            items = items[:limit]
        parsed: list[BaseModel] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                logger.warning("LLM 输出条目被丢弃: path=%s[%d] 原因=应为对象", path, index)
                continue
            try:
                parsed.append(model.model_validate(item))
            except ValidationError as exc:
                logger.warning("LLM 输出条目被丢弃: path=%s[%d] 原因=%s", path, index, describe_error(exc))
        return parsed

    return BeforeValidator(validate)


def describe_error(error: ValidationError) -> str:
    """把校验错误压成「字段路径: 原因」，不包含字段原文。"""

    parts: list[str] = []
    for item in error.errors()[:3]:
        location = ".".join(str(part) for part in item.get("loc", ()))
        message = str(item.get("msg", ""))[:120]
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts)


def _bounded_list(limit: int, item_limit: int | None = None):
    """数组字段：非数组报错，超长截断并记录字段名与长度。"""

    def validate(value: Any, info: ValidationInfo) -> list:
        if value in (None, ""):
            return []
        if isinstance(value, (str, int, float)):
            value = [value]
        if not isinstance(value, (list, tuple)):
            logger.warning("LLM 输出数组字段类型不可用,已置空: field=%s type=%s", info.field_name, type(value).__name__)
            return []
        items = list(value)
        if len(items) > limit:
            logger.warning("LLM 输出数组超长已截断: field=%s limit=%d actual=%d", info.field_name, limit, len(items))
            items = items[:limit]
        if item_limit is not None:
            items = [str(item).strip()[:item_limit] for item in items if isinstance(item, (str, int, float))]
        return items

    return BeforeValidator(validate)


def _clamped_number(minimum: float, maximum: float, default: float, *, cast=int, label: str = "数值"):
    """数值字段：拒绝 NaN/Infinity 与无法解析的值，越界钳制到合法区间。"""

    def validate(value: Any, info: ValidationInfo):
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return default
        try:
            number = float(value)
        except (TypeError, ValueError):
            logger.warning("LLM 输出%s无法解析,已使用默认值: field=%s", label, info.field_name)
            return default
        if not math.isfinite(number):
            logger.warning("LLM 输出%s非有限值,已使用默认值: field=%s", label, info.field_name)
            return default
        if number < minimum or number > maximum:
            logger.warning("LLM 输出%s超出范围,已钳制: field=%s", label, info.field_name)
        number = max(minimum, min(maximum, number))
        return cast(number)

    return BeforeValidator(validate)


def _mapping(limit: int, value_limit: int, *, fallback_key: str = "summary"):
    """对象字段：只保留标量键值，限制键数量与单值长度。"""

    def validate(value: Any, info: ValidationInfo) -> dict[str, str]:
        if value in (None, ""):
            return {}
        if isinstance(value, str):
            return {fallback_key: value.strip()[:value_limit]}
        if not isinstance(value, dict):
            logger.warning("LLM 输出对象字段类型不可用,已置空: field=%s type=%s", info.field_name, type(value).__name__)
            return {}
        bounded: dict[str, str] = {}
        for key, item in list(value.items())[:limit]:
            if isinstance(item, (dict, list, tuple, set)):
                continue
            bounded[str(key).strip()[:40]] = ("" if item is None else str(item).strip())[:value_limit]
        return bounded

    return BeforeValidator(validate)


# --- 输出模型 ---


class DialogueOutput(BaseModel):
    """单句对白。"""

    model_config = ConfigDict(extra="ignore")

    character: Annotated[
        str,
        _text(MAX_CHARACTER_NAME),
        Field(validation_alias=AliasChoices("character", "speaker", "role")),
    ] = ""
    line: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""
    emotion: Annotated[str, _choice(normalize_emotion, "neutral")] = "neutral"
    action: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""


class ShotDialogueLineOutput(BaseModel):
    """镜头内单句结构化对白：说话人是配音音色的唯一依据，必须显式保留。"""

    model_config = ConfigDict(extra="ignore")

    speaker: Annotated[
        str,
        _text(MAX_CHARACTER_NAME),
        Field(validation_alias=AliasChoices("speaker", "character", "role")),
    ] = ""
    line: Annotated[
        str,
        _text(settings.LLM_MAX_TEXT_CHARS),
        Field(validation_alias=AliasChoices("line", "text", "dialogue")),
    ] = ""
    emotion: Annotated[str, BeforeValidator(_optional_emotion)] = ""
    action: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""
    start_ms: Annotated[int | None, _clamped_number(0, MAX_TIMELINE_MS, None, label="对白开始时间")] = None
    end_ms: Annotated[int | None, _clamped_number(0, MAX_TIMELINE_MS, None, label="对白结束时间")] = None


class CharacterOutput(BaseModel):
    """角色卡片。"""

    model_config = ConfigDict(extra="ignore")

    name: Annotated[str, _text(MAX_CHARACTER_NAME)] = ""
    appearance: Annotated[dict[str, str], _mapping(MAX_APPEARANCE_KEYS, MAX_APPEARANCE_VALUE_CHARS)] = Field(
        default_factory=dict
    )
    personality: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""
    visual_prompt: Annotated[str, _text(settings.LLM_MAX_PROMPT_CHARS)] = ""
    negative_prompt: Annotated[str, _text(settings.LLM_MAX_PROMPT_CHARS)] = ""
    voice_type: Annotated[str, _text(40)] = ""
    key_features: Annotated[list[str], _bounded_list(MAX_FEATURE_ITEMS, MAX_FEATURE_CHARS)] = Field(
        default_factory=list
    )
    default_outfit: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""
    seed: Annotated[int | None, _clamped_number(0, MAX_SEED, None, label="seed")] = None


class SceneOutput(BaseModel):
    """剧本场景。"""

    model_config = ConfigDict(extra="ignore")

    scene_number: Annotated[int, _clamped_number(1, 9999, 1, label="场景序号")] = 1
    location: Annotated[str, _text(200)] = ""
    characters_in_scene: Annotated[list[str], _bounded_list(MAX_CHARACTERS_IN_SCENE, MAX_CHARACTER_NAME)] = Field(
        default_factory=list
    )
    actions: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""
    description: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""
    dialogue: Annotated[
        list[DialogueOutput], _model_list(DialogueOutput, settings.LLM_MAX_DIALOGUE_LINES, "dialogue")
    ] = Field(default_factory=list)
    emotion: Annotated[str, _choice(normalize_emotion, "neutral")] = "neutral"
    camera_suggestion: Annotated[str, _choice(normalize_shot_type, "medium")] = "medium"


class ActionBeatOutput(BaseModel):
    """一个镜头动作节拍；字符串输入会安全归一化为单个节拍。"""

    model_config = ConfigDict(extra="ignore")

    text: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""
    phase: Annotated[str, _text(24)] = "continuation"
    complex_motion: bool = False
    entry_state: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""
    exit_state: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""


def _action_beat_list():
    def validate(value: Any, info: ValidationInfo) -> list[ActionBeatOutput]:
        del info
        if value in (None, ""):
            return []
        if isinstance(value, (str, dict)):
            value = [value]
        if not isinstance(value, (list, tuple)):
            logger.warning("LLM 输出动作节拍类型不可用,已置空: field=action_beats type=%s", type(value).__name__)
            return []
        parsed: list[ActionBeatOutput] = []
        for item in list(value)[:MAX_ACTION_BEATS_PER_SHOT]:
            try:
                if isinstance(item, str):
                    item = {"text": item}
                parsed.append(ActionBeatOutput.model_validate(item))
            except ValidationError as exc:
                logger.warning("LLM 输出动作节拍被丢弃: path=action_beats 原因=%s", describe_error(exc))
        return [item for item in parsed if item.text.strip()]

    return BeforeValidator(validate)


class ShotOutput(BaseModel):
    """分镜镜头。"""

    model_config = ConfigDict(extra="ignore")

    shot_id: Annotated[str, _text(MAX_ID_CHARS)] = ""
    scene_number: Annotated[int, _clamped_number(1, 9999, 1, label="场景序号")] = 1
    source_scene_number: Annotated[int, _clamped_number(1, 9999, 1, label="场景序号")] = 1
    shot_type: Annotated[str, _choice(normalize_shot_type, "medium")] = "medium"
    scene_description: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""
    characters_in_scene: Annotated[list[str], _bounded_list(MAX_CHARACTERS_IN_SCENE, MAX_CHARACTER_NAME)] = Field(
        default_factory=list
    )
    character_action: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""
    action_beats: Annotated[list[ActionBeatOutput], _action_beat_list()] = Field(default_factory=list)
    gaze_direction: Annotated[str, _text(200)] = ""
    screen_axis: Annotated[str, _text(200)] = ""
    action_entry_state: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""
    action_exit_state: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""
    dialogue: Annotated[
        list[ShotDialogueLineOutput],
        _shot_dialogue_list(),
    ] = Field(default_factory=list)
    camera_angle: Annotated[str, _choice(normalize_camera_angle, "正面")] = "正面"
    camera_movement: Annotated[str, _choice(normalize_camera_movement, "静止")] = "静止"
    emotion: Annotated[str, _choice(normalize_emotion, "neutral")] = "neutral"
    duration: Annotated[
        float,
        _clamped_number(
            settings.MIN_SHOT_DURATION_SECONDS, settings.MAX_SHOT_DURATION_SECONDS, 3.0, cast=float, label="时长"
        ),
    ] = 3.0
    estimated_speech_ms: Annotated[int, _clamped_number(0, 600_000, 0, cast=int, label="对白预计时长")] = 0
    transition: Annotated[str, _choice(normalize_transition, "cut")] = "cut"
    continuity_mode: Annotated[str, _continuity_mode()] = ""
    continuity_mode_source: Annotated[str, _text(40)] = ""
    image_path: Annotated[str, _text(500)] = ""
    audio_path: Annotated[str, _text(500)] = ""
    status: Annotated[str, _choice(normalize_shot_status, "pending")] = "pending"
    confirmed: bool = False
    version: Annotated[int, _clamped_number(1, 10000, 1, label="版本号")] = 1
    seed: Annotated[int | None, _clamped_number(0, MAX_SEED, None, label="seed")] = None
    visual_notes: Annotated[str, _text(settings.LLM_MAX_TEXT_CHARS)] = ""


class ScriptParseOutput(BaseModel):
    """剧本解析的顶层输出。"""

    model_config = ConfigDict(extra="ignore")

    title: Annotated[str, _text(120)] = ""
    genre: Annotated[str, _text(60)] = ""
    style_suggestion: Annotated[str, _text(40)] = "anime"
    characters: Annotated[
        list[CharacterOutput], _model_list(CharacterOutput, settings.LLM_MAX_CHARACTERS, "characters")
    ] = Field(default_factory=list)
    script_scenes: Annotated[list[SceneOutput], _model_list(SceneOutput, settings.LLM_MAX_SCENES, "script_scenes")] = (
        Field(default_factory=list)
    )
    logic_issues: Annotated[list[str], _bounded_list(MAX_LOGIC_ISSUES, 300)] = Field(default_factory=list)


class StoryboardOutput(BaseModel):
    """分镜生成的顶层输出（shots 数组）。"""

    model_config = ConfigDict(extra="ignore")

    shots: Annotated[list[ShotOutput], _model_list(ShotOutput, settings.LLM_MAX_SHOTS, "shots")] = Field(
        default_factory=list
    )


# --- 顶层入口 ---


def parse_script_output(payload: Any, *, fallback_style: str = "anime") -> ScriptParseOutput:
    """校验并归一化剧本解析结果；结构不可用时抛出可读业务错误。"""

    if not isinstance(payload, dict):
        raise LLMOutputError("剧本解析结果必须是 JSON 对象")
    payload = _normalize_script_aliases(payload)
    scenes = payload.get("script_scenes")
    if scenes in (None, ""):
        scenes = payload.get("scenes")
    data = {
        "title": payload.get("title"),
        "genre": payload.get("genre"),
        "style_suggestion": payload.get("style_suggestion") or payload.get("style"),
        "characters": payload.get("characters"),
        "script_scenes": scenes,
        "logic_issues": payload.get("logic_issues"),
    }
    try:
        output = ScriptParseOutput.model_validate(data)
    except ValidationError as exc:
        raise LLMOutputError("剧本解析结果结构无法解析", issues=[describe_error(exc)]) from exc
    output.style_suggestion = normalize_style_suggestion(output.style_suggestion, fallback_style)
    return output


_CHARACTER_NAME_KEYS = ("name", "character_name", "character", "角色名", "姓名", "人物", "角色")


def _normalize_script_aliases(payload: dict[str, Any]) -> dict[str, Any]:
    """接受 Mimo 偶尔返回的中文字段和「姓名 -> 描述」映射。"""

    normalized = dict(payload)
    raw_characters = next(
        (
            payload.get(key)
            for key in ("characters", "character_list", "人物", "角色")
            if payload.get(key) not in (None, "")
        ),
        None,
    )
    if isinstance(raw_characters, dict):
        # 兼容 {"林夏": {"personality": ...}} 或单个角色对象。
        if any(key in raw_characters for key in _CHARACTER_NAME_KEYS):
            raw_characters = [raw_characters]
        else:
            mapped = []
            for name, details in list(raw_characters.items())[: settings.LLM_MAX_CHARACTERS]:
                if isinstance(details, dict):
                    item = dict(details)
                    item["name"] = str(name)
                else:
                    item = {"name": str(name), "appearance": str(details)}
                mapped.append(item)
            raw_characters = mapped
    if isinstance(raw_characters, (list, tuple)):
        converted = []
        for item in raw_characters:
            if isinstance(item, dict):
                item = dict(item)
                if not str(item.get("name") or "").strip():
                    for key in _CHARACTER_NAME_KEYS[1:]:
                        if str(item.get(key) or "").strip():
                            item["name"] = item[key]
                            break
                converted.append(item)
        normalized["characters"] = converted
    return normalized


def parse_storyboard_output(payload: Any) -> StoryboardOutput:
    """校验并归一化分镜生成结果；结构不可用时抛出可读业务错误。"""

    if isinstance(payload, list):
        data: dict[str, Any] = {"shots": payload}
    elif isinstance(payload, dict):
        shots = payload.get("shots")
        if shots in (None, ""):
            shots = payload.get("storyboard")
        data = {"shots": shots}
    else:
        raise LLMOutputError("分镜生成结果必须是 JSON 对象或数组")
    data["shots"] = _apply_shot_type_fallback(data["shots"])
    try:
        return StoryboardOutput.model_validate(data)
    except ValidationError as exc:
        raise LLMOutputError("分镜生成结果结构无法解析", issues=[describe_error(exc)]) from exc


def _apply_shot_type_fallback(shots: Any) -> Any:
    """场景级 camera_suggestion 可作为镜头类型的兜底（不修改调用方数据）。"""

    if isinstance(shots, dict):
        shots = [shots]
    if not isinstance(shots, (list, tuple)):
        return shots
    normalized: list[Any] = []
    for item in shots:
        if isinstance(item, dict) and not str(item.get("shot_type") or "").strip():
            suggestion = str(item.get("camera_suggestion") or "").strip()
            if suggestion:
                item = {**item, "shot_type": suggestion}
        normalized.append(item)
    return normalized
