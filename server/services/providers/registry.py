"""协议适配器注册表。

``ADAPTERS`` 以 ``(capability, protocol)`` 为键登记各能力的协议适配器；
接入同协议新服务无需改这里，接入全新协议时新增适配器文件并登记一行即可。
"""

from __future__ import annotations

from services.providers.base import BaseAdapter
from services.providers.image_ark_seedream import ArkSeedreamImageAdapter
from services.providers.image_placeholder import PlaceholderImageAdapter
from services.providers.image_qwen import QwenImageAdapter
from services.providers.image_stability import StabilityImageAdapter
from services.providers.llm_openai_chat import OpenAIChatAdapter
from services.providers.tts_dashscope import DashscopeTTSAdapter
from services.providers.tts_mimo import MimoTTSAdapter
from services.providers.tts_tencent import TencentTTSAdapter
from services.providers.video_ark_seedance import ArkSeedanceVideoAdapter
from services.providers.video_dashscope_wanx import DashscopeWanxVideoAdapter
from services.providers.video_google_veo import GoogleVeoVideoAdapter
from services.providers.video_kling import KlingVideoAdapter
from services.providers.video_luma import LumaVideoAdapter
from services.providers.video_minimax import MiniMaxVideoAdapter
from services.providers.video_native_audio import NativeAudioVideoAdapter
from services.providers.video_openai_sora import OpenAISoraVideoAdapter
from services.providers.video_pixverse import PixVerseVideoAdapter
from services.providers.video_runway import RunwayVideoAdapter
from services.providers.video_vidu import ViduVideoAdapter

ADAPTERS: dict[tuple[str, str], type[BaseAdapter]] = {
    ("script", "openai-chat"): OpenAIChatAdapter,
    ("image", "placeholder"): PlaceholderImageAdapter,
    ("image", "stability"): StabilityImageAdapter,
    ("image", "ark-seedream"): ArkSeedreamImageAdapter,
    ("image", "qwen-image"): QwenImageAdapter,
    ("video", "ark-seedance"): ArkSeedanceVideoAdapter,
    ("video", "dashscope-wanx"): DashscopeWanxVideoAdapter,
    ("video", "native-audio"): NativeAudioVideoAdapter,
    ("video", "openai-sora"): OpenAISoraVideoAdapter,
    ("video", "google-veo"): GoogleVeoVideoAdapter,
    ("video", "runway"): RunwayVideoAdapter,
    ("video", "kling"): KlingVideoAdapter,
    ("video", "minimax"): MiniMaxVideoAdapter,
    ("video", "vidu"): ViduVideoAdapter,
    ("video", "pixverse"): PixVerseVideoAdapter,
    ("video", "luma"): LumaVideoAdapter,
    ("voice", "dashscope-tts"): DashscopeTTSAdapter,
    ("voice", "mimo-tts"): MimoTTSAdapter,
    ("voice", "tencent-tts"): TencentTTSAdapter,
}


class UnknownProtocolError(RuntimeError):
    """协议未注册适配器时抛出，消息中带该能力的可选协议列表。"""


def get_adapter(capability: str, protocol: str) -> type[BaseAdapter]:
    key = (str(capability).lower(), str(protocol or "").strip().lower())
    adapter = ADAPTERS.get(key)
    if adapter is None:
        options = protocols_for(key[0])
        raise UnknownProtocolError(
            f"未知的 {key[0]} 协议: {key[1] or '<empty>'}，可选值: {', '.join(options) if options else '（暂无）'}"
        )
    return adapter


def protocols_for(capability: str) -> list[str]:
    capability = str(capability).lower()
    return sorted({protocol for name, protocol in ADAPTERS if name == capability})


__all__ = ["ADAPTERS", "UnknownProtocolError", "get_adapter", "protocols_for"]
