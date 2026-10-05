"""API 请求 DTO 的共享校验规则。

前端的 min/max 与选项列表只是交互提示，不能当成安全边界：字符串长度、业务枚
举、数值范围、浮点有限性、ID 格式与数组规模都在这里统一约束，各路由 DTO 直
接复用。

枚举取值来自项目现有实现（前端选项、服务映射表、模板与既有测试），不新增业务
上并不支持的值。
"""

from __future__ import annotations

import json
import math
import re
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints

from config import settings
from services.security import validate_identifier

# --- 业务枚举（与前端选项 / 服务映射表保持一致） ---
ProjectType = Literal["series", "episode"]
InputType = Literal["text", "file"]
OutputFormat = Literal["9:16", "16:9", "1:1", "4:3", "3:4"]
Resolution = Literal["720p", "1080p", "2k", "4k"]
Platform = Literal["douyin", "kuaishou", "bilibili", "custom"]
PipelineMode = Literal["manual", "auto"]
AudioMode = Literal["tts", "native", "auto"]
# 镜头级音频覆盖允许空串表示「清除覆盖」。
AudioModeOverride = Literal["", "tts", "native", "auto"]
ShotType = Literal["wide", "medium", "close-up", "extreme_close"]
CameraAngle = Literal["正面", "侧面", "俯视", "仰视"]
CameraMovement = Literal["静止", "推", "拉", "摇", "移", "跟", "升降", "环绕", "缓慢推进"]
Emotion = Literal["neutral", "happy", "shy", "sad", "angry", "surprised"]
Transition = Literal["cut", "fade", "dissolve", "white_flash", "push", "wipe"]
ShotStatus = Literal[
    "pending",
    "storyboard_done",
    "storyboard_approved",
    "video_generating",
    "video_done",
    "failed",
    "needs_review",
    "structural_check_passed",
]
ScriptStatus = Literal["started", "already_running", "updated", "deleted"]

# --- 字幕与音频混音工作台 ---
SubtitlePosition = Literal["top", "middle", "bottom"]
AudioTrackKind = Literal["dialogue", "music", "ambient", "sfx"]
SubtitleFormat = Literal["srt", "vtt"]

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _hex_color(value: Any) -> str:
    text = str(value or "").strip()
    if not _HEX_COLOR.fullmatch(text):
        raise ValueError("颜色必须是 #RRGGBB 格式")
    return text.upper()


HexColor = Annotated[str, AfterValidator(_hex_color)]

_STYLE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _reject_blank(value: Any) -> Any:
    """拒绝「非空但只包含空白字符」的输入；空串由长度约束判定。"""

    if isinstance(value, str) and value and not value.strip():
        raise ValueError("不能只包含空白字符")
    return value


def _optional_identifier(value: Any) -> str:
    """可空 ID：空串表示未设置，非空时沿用统一 identifier 规则。"""

    text = str(value or "").strip()
    if not text:
        return ""
    return validate_identifier(text, "ID")


def _style_id(value: Any) -> str:
    """画风标识：内置风格 key 或运行期创建的自定义风格 key。"""

    text = str(value or "").strip()
    if not text:
        raise ValueError("画风不能为空")
    if len(text) > settings.MAX_PROJECT_STYLE_CHARS or not _STYLE_ID.fullmatch(text):
        raise ValueError("画风标识非法")
    return text


def _finite_float(value: Any) -> Any:
    """显式拒绝 NaN / Infinity（JSON 扩展字面量可达此处）。"""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("必须是有限数值")
    if isinstance(value, bool):
        raise ValueError("必须是数值")
    return value


Identifier = Annotated[str, AfterValidator(validate_identifier)]
OptionalIdentifier = Annotated[str, AfterValidator(_optional_identifier)]
StyleId = Annotated[str, AfterValidator(_style_id)]

ProjectTitle = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, min_length=1, max_length=settings.MAX_PROJECT_TITLE_CHARS),
]
OptionalTitle = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, min_length=1, max_length=settings.MAX_PROJECT_TITLE_CHARS),
]
EpisodeTitle = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, max_length=settings.MAX_PROJECT_TITLE_CHARS),
]
Genre = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, max_length=settings.MAX_PROJECT_GENRE_CHARS),
]
GenerationPrompt = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, min_length=1, max_length=settings.MAX_GENERATION_PROMPT_CHARS),
]
OptionalGenerationPrompt = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, max_length=settings.MAX_GENERATION_PROMPT_CHARS),
]
ScriptText = Annotated[
    str,
    StringConstraints(max_length=settings.MAX_SCRIPT_TEXT_CHARS),
]
CharactersHint = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, max_length=500),
]
ShotText = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, max_length=settings.MAX_SHOT_TEXT_CHARS),
]


class ShotDialogueLine(BaseModel):
    """镜头单句对白的输入 DTO：说话人是配音音色的唯一依据。"""

    model_config = ConfigDict(extra="ignore")

    speaker: Annotated[str, StringConstraints(strip_whitespace=True, max_length=60)] = ""
    line: Annotated[
        str,
        BeforeValidator(_reject_blank),
        StringConstraints(strip_whitespace=True, max_length=settings.MAX_SHOT_TEXT_CHARS),
    ]
    emotion: Literal["neutral", "happy", "shy", "sad", "angry", "surprised"] = "neutral"
    action: Annotated[str, StringConstraints(strip_whitespace=True, max_length=settings.MAX_SHOT_TEXT_CHARS)] = ""
    start_ms: int | None = Field(default=None, ge=0, le=3_600_000)
    end_ms: int | None = Field(default=None, ge=0, le=3_600_000)


def _coerce_shot_dialogue(value: Any) -> Any:
    """对白输入兼容层：旧客户端发字符串时迁移为单条对白（speaker 为空）。"""

    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [{"line": text}] if text else []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        return [{"line": item.strip()} if isinstance(item, str) else item for item in value]
    return value


ShotDialogueList = Annotated[list[ShotDialogueLine], BeforeValidator(_coerce_shot_dialogue)]
VisualNotes = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, max_length=settings.MAX_VISUAL_NOTES_CHARS),
]
ReasonText = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, max_length=settings.MAX_SHOT_TEXT_CHARS),
]

EpisodeNumber = Annotated[int, Field(ge=0, le=settings.MAX_EPISODE_NUMBER)]
# 关键镜头故事板候选数：至少 1 个，最多 2 个（每个候选各占一版版本历史）。
CandidateCount = Annotated[int, Field(ge=1, le=2, allow_inf_nan=False)]
ShotDuration = Annotated[
    float,
    BeforeValidator(_finite_float),
    Field(
        ge=settings.MIN_SHOT_DURATION_SECONDS,
        le=settings.MAX_SHOT_DURATION_SECONDS,
        allow_inf_nan=False,
    ),
]
TargetDuration = Annotated[
    int,
    Field(
        ge=settings.MIN_TARGET_DURATION_SECONDS,
        le=settings.MAX_TARGET_DURATION_SECONDS,
        allow_inf_nan=False,
    ),
]

CharacterName = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, min_length=1, max_length=60),
]


def _bounded_string_list(value: Any) -> Any:
    """限制特征类列表的元素数量与单条长度（也接受单个字符串形式）。"""

    if isinstance(value, list):
        if len(value) > 12:
            raise ValueError("列表元素过多")
        for item in value:
            if not isinstance(item, str):
                raise ValueError("列表元素必须是字符串")
            if len(item) > 200:
                raise ValueError("单个元素过长")
    elif isinstance(value, str) and len(value) > 500:
        raise ValueError("文本过长")
    return value


def _bounded_mapping(value: Any) -> Any:
    """限制字典类字段的键数量与整体序列化长度，避免超大 JSON 落库。"""

    if isinstance(value, dict):
        if len(value) > 24:
            raise ValueError("字典键过多")
        if len(json.dumps(value, ensure_ascii=False, default=str)) > 8000:
            raise ValueError("字典内容过长")
    elif isinstance(value, str) and len(value) > 8000:
        raise ValueError("文本过长")
    return value


ShortKey = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, max_length=64),
]
JsonText = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, max_length=8000),
]
StyleLabel = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, min_length=1, max_length=60),
]
StyleKeywords = Annotated[
    str,
    BeforeValidator(_reject_blank),
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]
KeyFeatureInput = Annotated[list[str] | str, BeforeValidator(_bounded_string_list)]
JsonFieldInput = Annotated[dict | str, BeforeValidator(_bounded_mapping)]

ShotIdList = Annotated[list[Identifier], Field(max_length=settings.MAX_BATCH_SHOT_IDS)]
CharacterAssetIdList = Annotated[list[Identifier], Field(max_length=settings.MAX_CHARACTER_ASSET_IDS)]
NameList = Annotated[list[str], Field(max_length=12)]

# --- 任务中心（取值与 services/job_types 的稳定词表一致） ---
JobStatus = Literal["queued", "running", "cancelling", "completed", "failed", "cancelled", "interrupted"]
JobType = Literal[
    "script_pipeline",
    "storyboard",
    "asset_generation",
    "shot_image",
    "shot_audio",
    "shot_video",
    "render",
    "av_preview",
]
JobScope = Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)]
JobSearch = Annotated[str, BeforeValidator(_reject_blank), StringConstraints(strip_whitespace=True, max_length=120)]
