"""原生音视频协议骨架（Veo 3 / Sora 2 风格：视频 + 语音对白一次生成）。

完整厂商实现可后续按协议补充；当前骨架已落地：
- 能力声明（native_audio / dialogue_in_prompt），供音频路由按能力自动选择路径；
- 对白 → prompt 的统一编码（``build_prompt``），各厂商实现应复用以保证行为一致。

``production_ready=False``：路由层将该协议视为暂不可用，自动安全回退 tts 路径，
不会产出无声成品；直接调用 ``generate`` 会得到明确的 NotImplementedError。
"""

from __future__ import annotations

from services.providers.base import BaseAdapter, VideoCapabilities, VideoRequest, VideoResult
from services.providers.usage import CAPABILITY_VIDEO, UsageMetadata

EMOTION_LABELS = {
    "happy": "轻快",
    "shy": "害羞",
    "sad": "低落",
    "angry": "急切",
    "surprised": "惊讶",
    "neutral": "平静",
}


class NativeAudioVideoAdapter(BaseAdapter):
    capabilities = VideoCapabilities(
        reference_image=False,
        native_audio=True,
        dialogue_in_prompt=True,
        voice_consistent=False,
        min_duration=2,
        max_duration=8,
        duration_step=1,
        camera_movement_prompt=True,
        timed_dialogue=True,
    )
    production_ready = False

    def usage_for_request(
        self,
        capability: str,
        request: VideoRequest | None = None,
        *,
        model: str = "",
    ) -> UsageMetadata:
        """原生音视频协议的用量由请求推导：秒数 + 分辨率（未接入厂商实现前不报价）。"""

        return UsageMetadata(
            capability=CAPABILITY_VIDEO,
            provider=self.endpoint.protocol,
            model=model or self.endpoint.model,
            seconds=float(getattr(request, "duration", 0) or 0),
            resolution=str(getattr(request, "resolution", "") or ""),
            known=True,
            billable=True,
            source="request",
            extra={"ratio": str(getattr(request, "ratio", "") or ""), "native_audio": True},
        )

    async def generate(self, request: VideoRequest) -> VideoResult:
        raise NotImplementedError("native-audio 协议适配器尚未接入具体厂商实现；音频路由已在调用前安全回退 tts 路径")

    def build_prompt(self, request: VideoRequest) -> str:
        """把对白台词编入 prompt（Veo 3 风格：台词由文本驱动）。"""
        parts = [request.prompt.strip()]
        for dialogue in request.dialogues or []:
            if not str(dialogue.text or "").strip():
                continue
            emotion = EMOTION_LABELS.get(str(dialogue.emotion or "neutral"), "平静")
            role = str(dialogue.role or "角色").strip() or "角色"
            timing = ""
            if dialogue.end_ms > dialogue.start_ms:
                timing = f"（时间线 {int(dialogue.start_ms)}-{int(dialogue.end_ms)}ms）"
            parts.append(f"{role}以{emotion}的语气开口说{timing}：「{str(dialogue.text).strip()}」")
        if request.dialogues:
            parts.append("对白必须由画面角色原生开口说出，口型与台词同步，音频随视频一次性生成")
        return "\n".join(part for part in parts if part)
