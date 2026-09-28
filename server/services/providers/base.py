"""协议适配器基类、各能力的能力声明与请求/结果 dataclass。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from services.providers.endpoint import EndpointConfig
from services.providers.usage import UsageMetadata, unknown_usage


@dataclass(frozen=True)
class LLMCapabilities:
    json_mode: str = "auto"  # auto=试探 response_format 并在报错时自动降级；supported/unsupported
    vision: bool = True  # 是否支持图片输入（call_with_image）


@dataclass(frozen=True)
class ImageCapabilities:
    reference_images: bool = False  # 是否支持参考图输入（seedream 支持多图）
    requires_credentials: bool = True  # 是否需要 API Key（placeholder 免凭据）


@dataclass(frozen=True)
class VideoCapabilities:
    reference_image: bool = False  # 是否需要/支持首帧参考图（Seedance 必须）
    native_audio: bool = False  # 是否原生生成音频（含对白语音）
    dialogue_in_prompt: bool = False  # 对白是否通过 prompt 文本驱动（Veo 3 风格）
    voice_consistent: bool = False  # 能否指定/锁定音色
    fixed_duration: int | None = None  # 协议固定时长（秒）；None 表示按镜头时长
    reference_mode: str = "text_only"  # first_frame_only / text_only
    # 该协议参考图内联（base64 data URL）时的编码字节预算；0 表示沿用全局
    # settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES。按网关实际上限声明，
    # 避免所有协议都被压到同一个保守值。
    max_reference_inline_bytes: int = 0


@dataclass
class Dialogue:
    """一句台词：角色、文本、情绪。原生音频适配器会将其编入 prompt。"""

    role: str = ""
    text: str = ""
    emotion: str = "neutral"


@dataclass
class ImageRequest:
    prompt: str
    negative_prompt: str = ""
    seed: int = 42
    reference_images: list[str] = field(default_factory=list)  # data URL / http(s) URL
    size: str = ""  # 期望出图尺寸（如 1440x2560）；空表示适配器自定
    label: str = "PLACEHOLDER"  # 占位图标题（仅 placeholder 适配器使用）


@dataclass
class VideoRequest:
    prompt: str
    reference_image: str | None = None  # 已审核首帧参考图（data URL / http(s) URL）
    dialogues: list[Dialogue] | None = None  # 台词；native_audio 适配器将其编入 prompt
    duration: int = 5
    ratio: str = "9:16"
    resolution: str = "720p"
    project_id: str = ""  # 用于存储配额核算
    output_video_path: Path | None = None  # 产物写入路径（服务层已做安全校验与配额）
    output_frame_path: Path | None = None


@dataclass
class VideoResult:
    video_path: str
    frame_path: str = ""
    native_audio: bool = False  # 产物是否自带音轨（适配器如实声明）
    payload_mode: str = ""  # 参考图载荷模式（first_frame_reference / image_reference / text_only）
    task_id: str = ""


@dataclass
class TTSRequest:
    text: str
    voice_id: str = ""
    emotion: str = "neutral"


class BaseAdapter:
    """协议适配器基类：持有一个端点配置，由子类实现具体协议调用。

    协议骨架（厂商实现未落地）应把 ``production_ready`` 置为 False：路由层会把
    该协议对应的高级能力（如原生音频）视为暂不可用并安全回退。

    用量回报：每个适配器都用 ``usage_for_request`` / ``usage_from_response`` 宣告
    本次调用消耗了多少（token / 张 / 秒 / 字符）。没有声明用量的适配器一律返回
    「未知」，记账层据此显示「成本未知」，绝不按 0 或猜测值入账。
    """

    capabilities: object = None
    production_ready: bool = True

    def __init__(self, endpoint: EndpointConfig):
        self.endpoint = endpoint

    # --- 统一 usage metadata ------------------------------------------------

    def usage_for_request(
        self,
        capability: str,
        request: object | None = None,
        *,
        model: str = "",
    ) -> UsageMetadata:
        """按请求 / 端点推导用量；未声明协议的适配器返回「未知」。"""

        return unknown_usage(capability, self.endpoint.protocol, model or self.endpoint.model)

    def usage_from_response(
        self,
        capability: str,
        response: object | None = None,
        *,
        request: object | None = None,
        model: str = "",
        duration_ms: int = 0,
    ) -> UsageMetadata:
        """用供应商返回值细化用量（默认沿用请求侧推导）。"""

        return self.usage_for_request(capability, request, model=model)
