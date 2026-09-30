"""端点配置模型与统一读取入口。

每类能力对应一份完整端点：协议 + base_url + api_key + model + 鉴权方式 + 扩展参数。
生效优先级：持久化 JSON（``data/model_api_config.json``，由 model_config_service
管理）> .env（settings）> 代码默认值。``get_endpoint`` 每次调用实时读取，
保证保存配置后新任务立即生效。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

from config import settings

# 能力类别。script_fallback 是 LLM 的备用端点（主端点失败时回落）。
CAPABILITIES = ("script", "script_fallback", "image", "video", "voice")

# 支持的鉴权方式。bearer=标准 Authorization 头；api-key-header=附加 api-key 请求头。
AUTH_STYLES = ("bearer", "api-key-header")

# 持久化类别中的标记字段：换端点未换 key 时置位，读取时强制返回空密钥。
API_KEY_REQUIRED = "api_key_required"

# 各能力支持的显式协议。运行时代码路径一律由 protocol 决定；接入同协议新服务
# 只需改端点配置，接入全新协议需新增适配器并注册到 providers.registry。
KNOWN_PROTOCOLS: dict[str, tuple[str, ...]] = {
    "script": ("openai-chat",),
    "image": ("placeholder", "stability", "ark-seedream", "qwen-image"),
    "video": ("ark-seedance", "native-audio", "dashscope-wanx"),
    "voice": ("mimo-tts", "tencent-tts", "dashscope-tts"),
}

DEFAULT_PROTOCOLS = {capability: protocols[0] for capability, protocols in KNOWN_PROTOCOLS.items()}

# 旧版 provider 字符串 -> 显式协议。仅用于加载旧 JSON / 旧 .env 字段时的一次性
# 归一化，运行时代码不再做 provider 嗅探。
LEGACY_PROTOCOL_ALIASES: dict[str, dict[str, str]] = {
    "script": {
        "openai": "openai-chat",
        "deepseek": "openai-chat",
        "mimo": "openai-chat",
        "seeddance": "openai-chat",
    },
    "image": {
        "local": "placeholder",
        "placeholder": "placeholder",
        "stability": "stability",
        "seedream": "ark-seedream",
        "volcengine": "ark-seedream",
        "ark": "ark-seedream",
        "qwen": "qwen-image",
        "qwen-image": "qwen-image",
        "qwen-image-plus": "qwen-image",
        "dashscope-qwen-image": "qwen-image",
    },
    "video": {
        "seedance": "ark-seedance",
        "ark": "ark-seedance",
        "dashscope": "dashscope-wanx",
        "dashscope-wanx": "dashscope-wanx",
        "wanx": "dashscope-wanx",
        "wan": "dashscope-wanx",
        "bailian": "dashscope-wanx",
        "aliyun": "dashscope-wanx",
        "native": "native-audio",
        "native-audio": "native-audio",
        "veo3": "native-audio",
        "sora2": "native-audio",
    },
    "voice": {
        "mimo": "mimo-tts",
        "mimo-tts": "mimo-tts",
        "tencent": "tencent-tts",
        "tencent-tts": "tencent-tts",
        "dashscope": "dashscope-tts",
        "dashscope-tts": "dashscope-tts",
        "bailian": "dashscope-tts",
        "aliyun": "dashscope-tts",
        "cosyvoice": "dashscope-tts",
        "qwen-tts": "dashscope-tts",
    },
}

# 旧版平铺在类别顶层的参数字段，统一收敛进 params。
FLAT_PARAM_FIELDS = ("max_tokens", "vision_model", "image_size", "voice", "format", "audio_mode")


@dataclass
class EndpointConfig:
    """一个可调用服务端点的完整描述。"""

    protocol: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    auth_style: str = "bearer"
    params: dict = field(default_factory=dict)

    def param(self, key: str, default=None):
        return self.params.get(key, default)


def protocol_family(capability: str) -> str:
    return "script" if capability.startswith("script") else capability


def normalize_protocol(capability: str, value: str) -> str:
    """把 provider/protocol 原始字符串归一化为显式协议（小写）。

    未知值原样返回，由 registry.get_adapter 在运行时给出带可选值列表的明确报错，
    或由调用方执行安全回退（如图像占位图）。
    """
    capability = protocol_family(capability)
    text = (value or "").strip().lower()
    if not text:
        return DEFAULT_PROTOCOLS.get(capability, "")
    known = KNOWN_PROTOCOLS.get(capability, ())
    if text in known:
        return text
    alias = LEGACY_PROTOCOL_ALIASES.get(capability, {}).get(text)
    if alias:
        return alias
    # 旧版允许把完整模型名填进 provider（如 Doubao-Seedream-4.5 / Doubao-Seedance-1.5-pro）。
    if capability == "image" and "seedream" in text:
        return "ark-seedream"
    if capability == "video" and ("seedance" in text or "veo" in text or "sora" in text or "wan" in text):
        if "seedance" in text:
            return "ark-seedance"
        if "wan" in text:
            return "dashscope-wanx"
        return "native-audio"
    return text


def normalize_auth_style(value: str) -> str:
    text = (value or "").strip().lower()
    return text if text in AUTH_STYLES else "bearer"


def endpoint_identity(value: str | None) -> tuple[str, str, int | None] | None:
    """端点的 (scheme, host, effective_port) 标识，用于比较是否指向同一服务。"""

    text = str(value or "").strip()
    if not text:
        return None
    parsed = urlparse(text)
    try:
        port = parsed.port
    except ValueError:
        return None
    if not parsed.scheme or not parsed.hostname:
        return None
    host = parsed.hostname.rstrip(".").lower()
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return None
    effective_port = port if port is not None else {"http": 80, "https": 443}.get(parsed.scheme.lower())
    return parsed.scheme.lower(), host, effective_port


def settings_defaults(capability: str) -> dict:
    """从 settings（.env / 默认值）构造各类能力的默认端点。"""
    if capability == "script_fallback":
        return _script_defaults(fallback=True)
    capability = protocol_family(capability)
    if capability == "script":
        return _script_defaults(fallback=False)
    if capability == "image":
        return _image_defaults()
    if capability == "video":
        return _video_defaults()
    if capability == "voice":
        return _voice_defaults()
    raise ValueError(f"未知能力类别: {capability}，可选值: {', '.join(CAPABILITIES)}")


def _script_defaults(*, fallback: bool) -> dict:
    if fallback:
        return {
            "protocol": "openai-chat",
            "base_url": settings.OPENAI_BASE_URL or "https://api.deepseek.com",
            "api_key": settings.OPENAI_API_KEY,
            "model": settings.OPENAI_MODEL or "deepseek-chat",
            "auth_style": "bearer",
            "params": {},
        }
    provider = (settings.LLM_PROVIDER or "").strip().lower()
    if provider in {"mimo"}:
        return {
            "protocol": "openai-chat",
            "base_url": settings.MIMO_BASE_URL,
            "api_key": settings.MIMO_API_KEY,
            "model": settings.MIMO_MODEL,
            "auth_style": "api-key-header",
            "params": {
                "max_tokens": settings.LLM_MAX_TOKENS,
                "vision_model": settings.MIMO_MULTIMODAL_MODEL,
            },
        }
    return {
        "protocol": "openai-chat",
        "base_url": settings.OPENAI_BASE_URL,
        "api_key": settings.OPENAI_API_KEY,
        "model": settings.OPENAI_MODEL,
        "auth_style": "bearer",
        "params": {"max_tokens": settings.LLM_MAX_TOKENS},
    }


def _image_defaults(protocol: str = "") -> dict:
    """图像端点默认值。protocol 为空时按 .env 的 IMAGE_PROVIDER 推导；
    显式传入时按指定协议构造（用于探测「其它已配置的图像端点」，
    例如当前 Provider 不支持参考图时寻找支持参考图的替代端点）。"""
    if not protocol:
        provider = (settings.IMAGE_PROVIDER or "").strip().lower()
        protocol = normalize_protocol("image", provider)
    if protocol == "stability":
        return {
            "protocol": "stability",
            "base_url": settings.STABILITY_API_URL,
            "api_key": settings.STABILITY_API_KEY,
            "model": "",
            "auth_style": "bearer",
            "params": {},
        }
    if protocol == "placeholder":
        return {
            "protocol": "placeholder",
            "base_url": "",
            "api_key": "",
            "model": "",
            "auth_style": "bearer",
            "params": {},
        }
    if protocol == "qwen-image":
        return {
            "protocol": "qwen-image",
            "base_url": settings.QWEN_IMAGE_BASE_URL,
            "api_key": settings.QWEN_IMAGE_API_KEY or settings.DASHSCOPE_API_KEY,
            "model": settings.QWEN_IMAGE_MODEL,
            "auth_style": "bearer",
            "params": {"image_size": settings.QWEN_IMAGE_SIZE},
        }
    return {
        "protocol": "ark-seedream",
        "base_url": settings.SEEDDANCE_BASE_URL,
        "api_key": settings.ARK_API_KEY or settings.SEEDREAM_API_KEY,
        "model": settings.SEEDREAM_MODEL,
        "auth_style": "bearer",
        "params": {"image_size": settings.SEEDREAM_IMAGE_SIZE},
    }


def _video_defaults() -> dict:
    protocol = normalize_protocol("video", settings.VIDEO_PROVIDER)
    if protocol == "dashscope-wanx":
        return {
            "protocol": "dashscope-wanx",
            "base_url": settings.DASHSCOPE_BASE_URL,
            "api_key": settings.DASHSCOPE_API_KEY,
            "model": settings.DASHSCOPE_VIDEO_MODEL,
            "auth_style": "bearer",
            "params": {},
        }
    return {
        "protocol": protocol,
        "base_url": settings.SEEDDANCE_BASE_URL,
        "api_key": settings.SEEDDANCE_API_KEY or settings.ARK_API_KEY,
        "model": settings.SEEDDANCE_MODEL,
        "auth_style": "bearer",
        "params": {"audio_mode": "tts"},
    }


def _voice_defaults() -> dict:
    # 配置了腾讯云凭据时，语音端点默认切到腾讯云（tencent-tts 协议）。
    if settings.TENCENT_SECRET_ID and settings.TENCENT_SECRET_KEY:
        return {
            "protocol": "tencent-tts",
            "base_url": settings.TENCENT_TTS_URL,
            "api_key": f"{settings.TENCENT_SECRET_ID}:{settings.TENCENT_SECRET_KEY}",
            "model": "",
            "auth_style": "bearer",
            "params": {"voice": settings.TENCENT_TTS_VOICE, "format": settings.TENCENT_TTS_FORMAT},
        }
    # TTS_PROVIDER 显式填 dashscope / bailian / cosyvoice 时切到百炼语音合成。
    if normalize_protocol("voice", settings.TTS_PROVIDER) == "dashscope-tts":
        return {
            "protocol": "dashscope-tts",
            "base_url": settings.DASHSCOPE_TTS_BASE_URL,
            "api_key": settings.DASHSCOPE_TTS_API_KEY or settings.DASHSCOPE_API_KEY,
            "model": settings.DASHSCOPE_TTS_MODEL,
            "auth_style": "bearer",
            "params": {"voice": settings.DASHSCOPE_TTS_VOICE, "format": settings.DASHSCOPE_TTS_FORMAT},
        }
    return {
        "protocol": "mimo-tts",
        "base_url": settings.MIMO_BASE_URL,
        "api_key": settings.MIMO_API_KEY,
        "model": settings.MIMO_TTS_MODEL,
        "auth_style": "bearer",
        "params": {"voice": settings.MIMO_TTS_VOICE, "format": settings.MIMO_TTS_FORMAT},
    }


def endpoint_from_stored(capability: str, stored: dict | None) -> EndpointConfig:
    """按「JSON 覆盖 > settings 默认值」合并出 EndpointConfig。

    持久化字段留空表示沿用默认值；``api_key_required`` 标记强制返回空密钥，
    防止换端点后误用旧凭据。
    """
    merged = settings_defaults(capability)
    params = dict(merged.get("params") or {})
    for key, value in (stored or {}).items():
        if key == API_KEY_REQUIRED:
            continue
        if key == "params":
            if isinstance(value, dict):
                params.update({name: item for name, item in value.items() if item is not None})
            continue
        if key not in ("protocol", "base_url", "api_key", "model", "auth_style"):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value
    required = bool((stored or {}).get(API_KEY_REQUIRED))
    return EndpointConfig(
        protocol=normalize_protocol(capability, merged.get("protocol", "")),
        base_url=str(merged.get("base_url") or "").rstrip("/"),
        api_key="" if required else str(merged.get("api_key") or ""),
        model=str(merged.get("model") or ""),
        auth_style=normalize_auth_style(merged.get("auth_style", "")),
        params=params,
    )


def video_protocol_defaults(protocol: str) -> EndpointConfig:
    """按显式视频协议构造端点，供 Agent ``switch_provider`` 使用。"""

    normalized = normalize_protocol("video", protocol)
    if normalized == "dashscope-wanx":
        return _endpoint_from_merged(
            "video",
            {
                "protocol": normalized,
                "base_url": settings.DASHSCOPE_BASE_URL,
                "api_key": settings.DASHSCOPE_API_KEY,
                "model": settings.DASHSCOPE_VIDEO_MODEL,
                "auth_style": "bearer",
                "params": {},
            },
        )
    if normalized == "native-audio":
        return _endpoint_from_merged("video", {"protocol": normalized, "auth_style": "bearer", "params": {}})
    return _endpoint_from_merged("video", _video_defaults())


def image_protocol_defaults(protocol: str) -> EndpointConfig:
    """按显式协议构造图像端点（只用 settings / .env，不读持久化选择）。

    用于在多 Provider 环境下探测「账号已配置、且适配器声明支持参考图」
    的候选图像端点，作为当前 Provider 的能力补充。
    """
    return _endpoint_from_merged("image", _image_defaults(protocol))


def _endpoint_from_merged(capability: str, merged: dict) -> EndpointConfig:
    """把一份已合并的默认配置转为 EndpointConfig（不做持久化覆盖）。"""
    return EndpointConfig(
        protocol=normalize_protocol(capability, merged.get("protocol", "")),
        base_url=str(merged.get("base_url") or "").rstrip("/"),
        api_key=str(merged.get("api_key") or ""),
        model=str(merged.get("model") or ""),
        auth_style=normalize_auth_style(merged.get("auth_style", "")),
        params=dict(merged.get("params") or {}),
    )


def get_endpoint(capability: str) -> EndpointConfig:
    """单一读取入口：返回某能力的生效端点配置。

    每次调用实时读取持久化 JSON 与 settings，保存配置后新任务即用新配置。
    """
    if capability not in CAPABILITIES:
        raise ValueError(f"未知能力类别: {capability}，可选值: {', '.join(CAPABILITIES)}")
    from services import model_config_service  # 延迟导入，避免与持久化层循环依赖

    return endpoint_from_stored(capability, model_config_service.stored_category(capability))


__all__ = [
    "API_KEY_REQUIRED",
    "AUTH_STYLES",
    "CAPABILITIES",
    "EndpointConfig",
    "FLAT_PARAM_FIELDS",
    "KNOWN_PROTOCOLS",
    "endpoint_from_stored",
    "endpoint_identity",
    "get_endpoint",
    "image_protocol_defaults",
    "normalize_auth_style",
    "normalize_protocol",
    "protocol_family",
    "settings_defaults",
    "video_protocol_defaults",
]
