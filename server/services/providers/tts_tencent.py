"""腾讯云语音合成（基础语音合成 TextToVoice）适配器。

TC3-HMAC-SHA256 签名调用 ``tts.tencentcloudapi.com``；凭据以 ``SecretId:SecretKey``
组合存放在端点 ``api_key`` 字段（复用现有掩码/换端点清空机制）。单次请求上限
150 字节 UTF-8（约 50 个汉字），适配器按句切分分段合成后本地拼接：wav 用标准库
按参数严格拼接，其余格式按字节顺序拼接。角色音色档案存量是 Mimo 归一化音色名，
这里映射到腾讯云 VoiceType 数字 ID。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
import wave
from datetime import UTC, datetime
from io import BytesIO
from urllib.parse import urlparse

import httpx

from config import settings
from services.providers.base import BaseAdapter, TTSRequest
from services.providers.usage import CAPABILITY_TTS, UsageMetadata

_DEFAULT_HOST = "tts.tencentcloudapi.com"
_TTS_SERVICE = "tts"
_TTS_VERSION = "2019-08-23"
_DEFAULT_VOICE = "101001"  # 智瑜 · 情感女声

# TextToVoice 单次上限 150 字节 UTF-8；按字符切分时给中英混排留余量。
_SEGMENT_CHAR_LIMIT = 45
_SENTENCE_BREAKS = "。！？；，…,.!?;"

# Mimo 音色名（角色档案存量）与通用风格标签 → 腾讯云基础音色 VoiceType。
TENCENT_VOICE_ALIASES = {
    "mimo_default": "101001",
    "冰糖": "101001",  # 智瑜 · 情感女声
    "茉莉": "101027",  # 智梅 · 通用女声
    "苏打": "101016",  # 智甜 · 女童声
    "Milo": "101030",  # 智柯 · 通用男声
    "Dean": "101004",  # 智云 · 通用男声
    "白桦": "101013",  # 智辉 · 新闻男声
    "Mia": "101001",
    "Chloe": "101026",  # 智希 · 通用女声
    "少女": "101001",
    "女声": "101026",
    "甜美": "101001",
    "少年": "101030",
    "男声": "101030",
    "青年": "101030",
    "御姐": "101027",
    "成熟女声": "101027",
    "姐姐": "101027",
    "大叔": "101004",
    "成熟男声": "101004",
    "叔叔": "101004",
    "儿童": "101016",
    "孩子": "101016",
    "老人": "101013",
    "老年": "101013",
    "智瑜": "101001",
    "智甜": "101016",
    "智萌": "101015",
    "智燕": "101011",
    "智云": "101004",
    "智希": "101026",
    "智梅": "101027",
    "智友": "101054",
    "智柯": "101030",
    "智彤": "101019",
    "智辉": "101013",
    "智瑞": "101021",
    "智付": "101055",
    "WeJack": "101050",
}


def normalize_tencent_voice(voice_id: str = "", default_voice: str = "") -> str:
    """把 Mimo 音色名 / 通用标签 / 腾讯音色名归一化为腾讯云 VoiceType 数字 ID。"""

    voice = (voice_id or "").strip()
    if voice.isdigit():
        return voice
    if voice in TENCENT_VOICE_ALIASES:
        return TENCENT_VOICE_ALIASES[voice]
    default = (default_voice or "").strip()
    if default.isdigit():
        return default
    if default in TENCENT_VOICE_ALIASES:
        return TENCENT_VOICE_ALIASES[default]
    return _DEFAULT_VOICE


class TencentTTSAdapter(BaseAdapter):
    def usage_for_request(
        self,
        capability: str,
        request: TTSRequest | None = None,
        *,
        model: str = "",
    ) -> UsageMetadata:
        """腾讯云基础语音合成按字符计费；音色与音频格式记进明细便于核对。"""

        text = str(getattr(request, "text", "") or "")
        return UsageMetadata(
            capability=CAPABILITY_TTS,
            provider=self.endpoint.protocol,
            model=model or self.endpoint.model,
            characters=len(text.strip()),
            known=True,
            billable=True,
            source="request",
            extra={
                "voice": normalize_tencent_voice(
                    str(getattr(request, "voice_id", "") or ""), str(self.endpoint.param("voice") or "")
                ),
                "format": str(self.endpoint.param("format") or ""),
            },
        )

    async def synthesize(self, request: TTSRequest) -> bytes:
        secret_id, secret_key = self._credentials()
        text = str(request.text or "").strip()
        if not text:
            raise RuntimeError("配音文本为空，无法调用 TTS")
        if len(text) > settings.MAX_SCRIPT_TEXT_CHARS:
            raise RuntimeError("配音文本超过长度限制")

        voice = normalize_tencent_voice(request.voice_id, str(self.endpoint.param("voice") or ""))
        codec = str(self.endpoint.param("format") or "wav").strip().lower() or "wav"
        chunks: list[bytes] = []
        for segment in self._split_text(text):
            chunks.append(await self._synthesize_segment(segment, voice, codec, secret_id, secret_key))
        audio = chunks[0] if len(chunks) == 1 else self._concat_audio(chunks, codec)
        if len(audio) > settings.MAX_TTS_AUDIO_BYTES:
            raise RuntimeError("音频数据超过大小限制")
        return audio

    # --- 协议细节 ---

    def _credentials(self) -> tuple[str, str]:
        raw = (self.endpoint.api_key or "").strip()
        if not raw:
            raise RuntimeError("未配置语音端点 API Key（SecretId:SecretKey），无法调用腾讯云 TTS")
        if ":" not in raw:
            raise RuntimeError("腾讯云 TTS 凭据格式应为 SecretId:SecretKey")
        secret_id, secret_key = raw.split(":", 1)
        if not secret_id.strip() or not secret_key.strip():
            raise RuntimeError("腾讯云 TTS 凭据格式应为 SecretId:SecretKey")
        return secret_id.strip(), secret_key.strip()

    async def _synthesize_segment(self, text: str, voice: str, codec: str, secret_id: str, secret_key: str) -> bytes:
        # TC3 签名覆盖请求体字节，payload 只序列化一次并原样发送。
        payload = json.dumps(
            {
                "Text": text,
                "SessionId": uuid.uuid4().hex,
                "VoiceType": int(voice),
                "Codec": codec,
                "SampleRate": 16000,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        host = self._host()
        headers = self._signed_headers(host, "TextToVoice", payload, secret_id, secret_key)
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"https://{host}/", headers=headers, content=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"腾讯云 TTS 调用失败: {response.status_code} {response.text[:600]}")

        wrapped = (response.json() or {}).get("Response") or {}
        error = wrapped.get("Error")
        if error:
            raise RuntimeError(f"腾讯云 TTS 调用失败: {error.get('Code')} {error.get('Message')}")
        b64_data = wrapped.get("Audio")
        if not b64_data:
            raise RuntimeError(f"腾讯云 TTS 返回缺少 Audio: {wrapped}")

        limit = int(settings.MAX_TTS_AUDIO_BYTES)
        if len(b64_data) > int(limit * 4 / 3) + 16:
            raise RuntimeError("音频数据超过大小限制")
        try:
            decoded = base64.b64decode(b64_data, validate=True)
        except Exception as exc:
            raise RuntimeError("音频数据编码无效") from exc
        if len(decoded) > limit:
            raise RuntimeError("音频数据超过大小限制")
        return decoded

    def _split_text(self, text: str) -> list[str]:
        """按句切分长文本：优先在标点断句，避免把一句话切成两段语气断裂。"""

        segments: list[str] = []
        remaining = text
        while len(remaining) > _SEGMENT_CHAR_LIMIT:
            cut = _SEGMENT_CHAR_LIMIT
            for index in range(_SEGMENT_CHAR_LIMIT, max(_SEGMENT_CHAR_LIMIT // 2, 1), -1):
                if remaining[index - 1] in _SENTENCE_BREAKS:
                    cut = index
                    break
            segments.append(remaining[:cut])
            remaining = remaining[cut:]
        if remaining.strip():
            segments.append(remaining)
        return segments or [text]

    def _concat_audio(self, chunks: list[bytes], codec: str) -> bytes:
        if codec != "wav":
            return b"".join(chunks)
        output = BytesIO()
        with wave.open(output, "wb") as writer:
            for index, chunk in enumerate(chunks):
                with wave.open(BytesIO(chunk)) as reader:
                    if index == 0:
                        writer.setnchannels(reader.getnchannels())
                        writer.setsampwidth(reader.getsampwidth())
                        writer.setframerate(reader.getframerate())
                    writer.writeframes(reader.readframes(reader.getnframes()))
        return output.getvalue()

    def _signed_headers(self, host: str, action: str, payload: str, secret_id: str, secret_key: str) -> dict[str, str]:
        timestamp = int(time.time())
        date = datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%d")
        hashed_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        canonical_request = "\n".join(
            [
                "POST",
                "/",
                "",
                "content-type:application/json",
                f"host:{host}",
                f"x-tc-action:{action.lower()}",
                "",
                "content-type;host;x-tc-action",
                hashed_payload,
            ]
        )
        credential_scope = f"{date}/{_TTS_SERVICE}/tc3_request"
        string_to_sign = "\n".join(
            [
                "TC3-HMAC-SHA256",
                str(timestamp),
                credential_scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            ]
        )

        def _hmac_sha256(key: bytes, message: str) -> bytes:
            return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()

        secret_date = _hmac_sha256(("TC3" + secret_key).encode("utf-8"), date)
        secret_service = _hmac_sha256(secret_date, _TTS_SERVICE)
        secret_signing = _hmac_sha256(secret_service, "tc3_request")
        signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        authorization = (
            f"TC3-HMAC-SHA256 Credential={secret_id}/{credential_scope}, "
            f"SignedHeaders=content-type;host;x-tc-action, Signature={signature}"
        )
        return {
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Host": host,
            "X-TC-Action": action,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": _TTS_VERSION,
        }

    def _host(self) -> str:
        raw = (self.endpoint.base_url or "").strip()
        if raw:
            parsed = urlparse(raw if "//" in raw else f"https://{raw}")
            if parsed.hostname:
                return parsed.hostname.lower()
        return _DEFAULT_HOST
