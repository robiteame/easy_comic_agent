"""音频路由：按适配器能力与配置在「TTS 配音合成」与「原生音视频」两条路径间选择。

audio_mode 优先级：镜头级覆盖（shot.audio_mode / continuity_profile.audio_mode）
> video 端点 params.audio_mode > 默认 tts。

- ``tts``：现有路径——无声视频 + 独立 TTS + ffmpeg 音轨合成。
- ``native``：要求视频适配器声明 ``native_audio`` 能力且已接入厂商实现
  （production_ready）；对白编入 prompt，跳过独立 TTS 与音轨合成。
- ``auto``：适配器具备原生音频能力时，角色特写且带台词的镜头用 native，其余 tts。

两条路径的输出契约一致：都产出带音轨的镜头 mp4；native 能力不可用时安全回退
tts 路径并记 warning，绝不产出无声成品。
"""

from __future__ import annotations

import logging

from services.providers.endpoint import EndpointConfig, get_endpoint
from services.providers.registry import UnknownProtocolError, get_adapter
from services.shot_dialogue import parse_shot_dialogue

logger = logging.getLogger(__name__)

AUDIO_MODES = ("tts", "native", "auto")
CLOSEUP_SHOT_TYPES = {"close-up", "extreme_close"}


def resolve_audio_mode(shot: dict, endpoint: EndpointConfig | None = None) -> str:
    """解析某镜头最终生效的音频路径（"tts" | "native"）。"""
    endpoint = endpoint or get_endpoint("video")
    requested = _shot_override(shot) or _clean_mode(endpoint.param("audio_mode")) or "tts"
    if requested not in AUDIO_MODES:
        logger.warning("未知 audio_mode=%r，回退 tts 路径", requested)
        requested = "tts"
    if requested == "tts":
        return "tts"

    if not native_audio_capable(endpoint):
        logger.warning(
            "audio_mode=%s 但视频协议 %s 不具备可用的原生音频能力，本镜头回退 tts 路径",
            requested,
            endpoint.protocol,
        )
        return "tts"
    if requested == "native":
        return "native"
    # auto：角色特写且带关键台词的镜头交给原生音频，其余用 tts。
    # dialogue 可能是结构化列表（新）或纯文本/JSON 字符串（旧），统一解析。
    has_dialogue = any(line.line.strip() for line in parse_shot_dialogue(shot.get("dialogue")))
    closeup = str(shot.get("shot_type") or "").strip().lower() in CLOSEUP_SHOT_TYPES
    return "native" if has_dialogue and closeup else "tts"


def native_audio_capable(endpoint: EndpointConfig | None = None) -> bool:
    """当前视频端点的适配器是否真正具备原生音频能力。"""
    endpoint = endpoint or get_endpoint("video")
    try:
        adapter_cls = get_adapter("video", endpoint.protocol)
    except UnknownProtocolError:
        return False
    # 模型级能力判断优先；测试替身/未声明 effective_capabilities 的适配器回落类级声明。
    effective = getattr(adapter_cls, "effective_capabilities", None)
    capabilities = effective(endpoint.model) if callable(effective) else getattr(adapter_cls, "capabilities", None)
    if capabilities is None:
        return False
    return bool(capabilities.native_audio and getattr(adapter_cls, "production_ready", True))


def _shot_override(shot: dict) -> str:
    value = _clean_mode((shot or {}).get("audio_mode"))
    if value:
        return value
    profile = (shot or {}).get("continuity_profile") or {}
    return _clean_mode(profile.get("audio_mode") if isinstance(profile, dict) else "")


def _clean_mode(value) -> str:
    return str(value or "").strip().lower()
