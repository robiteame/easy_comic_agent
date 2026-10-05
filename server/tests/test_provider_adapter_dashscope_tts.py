"""阿里云百炼语音合成（dashscope-tts / CosyVoice）适配器的离线验收测试。

不发起真实网络请求：httpx.AsyncClient 用替身按序返回预置响应。
"""

from __future__ import annotations

import asyncio
import base64
import struct
import sys
import unittest
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from services.providers.endpoint import (  # noqa: E402
    EndpointConfig,
    normalize_protocol,
    settings_defaults,
)
from services.providers.registry import get_adapter  # noqa: E402
from services.providers.tts_dashscope import (  # noqa: E402
    DashscopeTTSAdapter,
    normalize_dashscope_voice,
)
from services.providers.usage import CAPABILITY_TTS  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402


class _FakeResponse:
    def __init__(self, payload=None, status_code=200, text="", content=b""):
        self._payload = payload or {}
        self.status_code = status_code
        self.text = text
        self.content = content

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """按序返回预置响应并记录请求的 httpx.AsyncClient 替身。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, **kwargs):
        self.requests.append({"method": "POST", "url": url, **kwargs})
        return self._responses.pop(0)

    async def get(self, url, **kwargs):
        self.requests.append({"method": "GET", "url": url, **kwargs})
        return self._responses.pop(0)


def _dashscope_endpoint(api_key: str = "sk-test") -> EndpointConfig:
    return EndpointConfig(
        protocol="dashscope-tts",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        api_key=api_key,
        model="cosyvoice-v2",
        params={"voice": "longwan_v2", "format": "wav"},
    )


def _make_wav(marker: int, frames: int = 160) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(22050)
        writer.writeframes(b"".join(struct.pack("<h", 1000 * marker) for _ in range(frames)))
    return buffer.getvalue()


# 官方非流式协议：POST 拿到 output.audio.url（24 小时有效），再 GET 下载音频。
_AUDIO_URL = "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/prod/cosyvoice-v2/test.wav"


def _tts_post_response() -> _FakeResponse:
    return _FakeResponse(
        {
            "output": {
                "audio": {"data": "", "id": "audio-test", "url": _AUDIO_URL},
                "finish_reason": "stop",
            },
            "usage": {"characters": 4},
            "request_id": "req-test",
        }
    )


class DashscopeTTSProtocolTests(unittest.TestCase):
    def test_protocol_registered(self) -> None:
        self.assertIs(get_adapter("voice", "dashscope-tts"), DashscopeTTSAdapter)

    def test_normalize_protocol_aliases(self) -> None:
        for alias in ("dashscope", "dashscope-tts", "bailian", "aliyun", "cosyvoice", "qwen-tts"):
            self.assertEqual(normalize_protocol("voice", alias), "dashscope-tts")
        # 既有别名不受影响
        self.assertEqual(normalize_protocol("voice", "mimo"), "mimo-tts")
        self.assertEqual(normalize_protocol("voice", "tencent"), "tencent-tts")

    def test_voice_defaults_follow_tts_provider(self) -> None:
        from services.providers import endpoint as endpoint_module

        settings_obj = endpoint_module.settings
        original = (
            settings_obj.TTS_PROVIDER,
            settings_obj.DASHSCOPE_TTS_API_KEY,
            settings_obj.DASHSCOPE_API_KEY,
        )
        try:
            settings_obj.TTS_PROVIDER = "dashscope"
            settings_obj.DASHSCOPE_TTS_API_KEY = ""
            settings_obj.DASHSCOPE_API_KEY = "sk-shared"
            defaults = settings_defaults("voice")
            self.assertEqual(defaults["protocol"], "dashscope-tts")
            self.assertEqual(defaults["api_key"], "sk-shared")  # 未单独配置 TTS Key 时复用百炼 Key
            self.assertEqual(defaults["model"], settings_obj.DASHSCOPE_TTS_MODEL)
            self.assertEqual(defaults["params"]["voice"], settings_obj.DASHSCOPE_TTS_VOICE)
            settings_obj.DASHSCOPE_TTS_API_KEY = "sk-tts"
            self.assertEqual(settings_defaults("voice")["api_key"], "sk-tts")
            # 未显式选择百炼时保持 Mimo 默认，不因共用百炼 Key 而被劫持
            settings_obj.TTS_PROVIDER = "mimo"
            self.assertEqual(settings_defaults("voice")["protocol"], "mimo-tts")
        finally:
            (
                settings_obj.TTS_PROVIDER,
                settings_obj.DASHSCOPE_TTS_API_KEY,
                settings_obj.DASHSCOPE_API_KEY,
            ) = original

    def test_voice_defaults_tencent_credentials_win(self) -> None:
        from services.providers import endpoint as endpoint_module

        settings_obj = endpoint_module.settings
        original = (
            settings_obj.TTS_PROVIDER,
            settings_obj.TENCENT_SECRET_ID,
            settings_obj.TENCENT_SECRET_KEY,
        )
        try:
            settings_obj.TTS_PROVIDER = "dashscope"
            settings_obj.TENCENT_SECRET_ID = "AKIDtest"
            settings_obj.TENCENT_SECRET_KEY = "secret"
            self.assertEqual(settings_defaults("voice")["protocol"], "tencent-tts")
        finally:
            (settings_obj.TTS_PROVIDER, settings_obj.TENCENT_SECRET_ID, settings_obj.TENCENT_SECRET_KEY) = original

    def test_pricing_defaults_include_dashscope_tts(self) -> None:
        from services.pricing_service import DEFAULT_PRICING_PROVIDERS

        self.assertIn("dashscope-tts", DEFAULT_PRICING_PROVIDERS[CAPABILITY_TTS])


class DashscopeTTSAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = DashscopeTTSAdapter(_dashscope_endpoint())

    def test_normalize_dashscope_voice(self) -> None:
        self.assertEqual(normalize_dashscope_voice("冰糖"), "longwan_v2")
        self.assertEqual(normalize_dashscope_voice("Milo"), "longshuo_v2")
        self.assertEqual(normalize_dashscope_voice("御姐"), "longyue_v2")
        self.assertEqual(normalize_dashscope_voice("苏打"), "longhuhu")  # 童声；旧目标 longxiaoxuan_v2 触发引擎 418
        self.assertEqual(normalize_dashscope_voice("Chloe"), "longxiaobai_v2")  # 旧目标 longtong_v2 无效
        self.assertEqual(normalize_dashscope_voice("longwan_v2"), "longwan_v2")  # 验证过的百炼音色 ID 透传
        self.assertEqual(normalize_dashscope_voice("loongstella_v2"), "loongstella_v2")
        self.assertEqual(normalize_dashscope_voice("未知音色", default_voice="longyue_v2"), "longyue_v2")
        self.assertEqual(normalize_dashscope_voice(""), "longwan_v2")

    def test_normalize_voice_v2_rejects_unverified_ids(self) -> None:
        # v2 模型下，无后缀 v1 名、_v3 名和拼错的 _v2 名都回退默认音色，不再透传触发引擎 418
        for voice in ("longwan", "longxiaochun", "longxiaoxuan_v2", "longtong_v2", "longfei_v3"):
            self.assertEqual(normalize_dashscope_voice(voice, model="cosyvoice-v2"), "longwan_v2", voice)
        # 端点默认音色同样要过验证
        self.assertEqual(normalize_dashscope_voice("", default_voice="longxiaoxuan_v2"), "longwan_v2")

    def test_normalize_voice_non_v2_models_pass_through(self) -> None:
        # v3/v3.5 等模型没有本地音色表，long*/loong* ID 原样透传
        self.assertEqual(normalize_dashscope_voice("longfei_v3", model="cosyvoice-v3-flash"), "longfei_v3")
        self.assertEqual(normalize_dashscope_voice("longanyang", model="cosyvoice-v3.5-plus"), "longanyang")
        # 复刻音色自带模型版本，任何模型下都透传
        self.assertEqual(normalize_dashscope_voice("cosyvoice-v2-clone-abc123"), "cosyvoice-v2-clone-abc123")

    def test_synthesize_single_segment(self) -> None:
        from services.providers.base import TTSRequest

        wav = _make_wav(1)
        client = _FakeAsyncClient([_tts_post_response(), _FakeResponse(content=wav)])
        with patch("services.providers.tts_dashscope.httpx.AsyncClient", return_value=client):
            audio = asyncio.run(self.adapter.synthesize(TTSRequest(text="你好", voice_id="冰糖")))
        self.assertEqual(audio, wav)
        post = client.requests[0]
        self.assertEqual(post["url"], "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer")
        self.assertEqual(post["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(post["json"]["model"], "cosyvoice-v2")
        self.assertEqual(post["json"]["input"]["text"], "你好")
        self.assertEqual(post["json"]["input"]["voice"], "longwan_v2")
        self.assertEqual(post["json"]["input"]["format"], "wav")
        self.assertNotIn("parameters", post["json"])  # 2026-08 协议参数全在 input 内
        download = client.requests[1]
        self.assertEqual(download["method"], "GET")
        self.assertEqual(download["url"], _AUDIO_URL)

    def test_synthesize_inline_base64_fallback(self) -> None:
        """流式式内联 base64 响应（output.audio.data）仍可直接解码。"""

        from services.providers.base import TTSRequest

        wav = _make_wav(3)
        client = _FakeAsyncClient(
            [_FakeResponse({"output": {"audio": {"data": base64.b64encode(wav).decode("ascii")}}})]
        )
        with patch("services.providers.tts_dashscope.httpx.AsyncClient", return_value=client):
            audio = asyncio.run(self.adapter.synthesize(TTSRequest(text="你好")))
        self.assertEqual(audio, wav)
        self.assertEqual(len(client.requests), 1)  # 无需二次下载

    def test_synthesize_rejects_non_aliyun_audio_url(self) -> None:
        from services.providers.base import TTSRequest

        client = _FakeAsyncClient([_FakeResponse({"output": {"audio": {"url": "https://evil.example.com/audio.wav"}}})])
        with patch("services.providers.tts_dashscope.httpx.AsyncClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "阿里云域名"):
                asyncio.run(self.adapter.synthesize(TTSRequest(text="你好")))

    def test_synthesize_audio_download_error_raises(self) -> None:
        from services.providers.base import TTSRequest

        client = _FakeAsyncClient([_tts_post_response(), _FakeResponse(status_code=403, text="denied")])
        with patch("services.providers.tts_dashscope.httpx.AsyncClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "下载失败"):
                asyncio.run(self.adapter.synthesize(TTSRequest(text="你好")))

    def test_api_base_appends_api_v1(self) -> None:
        adapter = DashscopeTTSAdapter(
            EndpointConfig(protocol="dashscope-tts", base_url="https://dashscope.aliyuncs.com", api_key="sk")
        )
        self.assertEqual(adapter._api_base(), "https://dashscope.aliyuncs.com/api/v1")

    def test_api_base_normalizes_compatible_mode_url(self) -> None:
        # 业务空间兼容模式地址（存量配置常见）归一到 /api/v1 根，不再拼出 compatible-mode/v1/api/v1
        adapter = DashscopeTTSAdapter(
            EndpointConfig(
                protocol="dashscope-tts",
                base_url="https://llm-demo.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                api_key="sk",
            )
        )
        self.assertEqual(adapter._api_base(), "https://llm-demo.cn-beijing.maas.aliyuncs.com/api/v1")

    def test_synthesize_http_error_raises(self) -> None:
        from services.providers.base import TTSRequest

        client = _FakeAsyncClient(
            [
                _FakeResponse(
                    {"code": "InvalidApiKey", "message": "Invalid API-key"},
                    status_code=401,
                    text='{"code":"InvalidApiKey","message":"Invalid API-key"}',
                )
            ]
        )
        with patch("services.providers.tts_dashscope.httpx.AsyncClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "InvalidApiKey"):
                asyncio.run(self.adapter.synthesize(TTSRequest(text="你好")))

    def test_synthesize_engine_418_hint(self) -> None:
        from services.providers.base import TTSRequest

        client = _FakeAsyncClient(
            [
                _FakeResponse(
                    {"code": "InvalidParameter", "message": "[cosyvoice:]Engine return error code: 418"},
                )
            ]
        )
        with patch("services.providers.tts_dashscope.httpx.AsyncClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "音色与模型版本不匹配"):
                asyncio.run(self.adapter.synthesize(TTSRequest(text="你好")))

    def test_synthesize_missing_audio_raises(self) -> None:
        from services.providers.base import TTSRequest

        client = _FakeAsyncClient([_FakeResponse({"output": {}})])
        with patch("services.providers.tts_dashscope.httpx.AsyncClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "缺少 audio"):
                asyncio.run(self.adapter.synthesize(TTSRequest(text="你好")))

    def test_synthesize_without_api_key_raises(self) -> None:
        from services.providers.base import TTSRequest

        adapter = DashscopeTTSAdapter(_dashscope_endpoint(api_key=""))
        with self.assertRaisesRegex(RuntimeError, "API Key"):
            asyncio.run(adapter.synthesize(TTSRequest(text="你好")))

    def test_synthesize_multi_segment_concatenates_wav(self) -> None:
        from services.providers.base import TTSRequest

        wavs = [_make_wav(marker) for marker in (1, 2)]
        # 每段一次 POST（拿 URL）+ 一次 GET（下载音频），交替返回。
        client = _FakeAsyncClient(
            [_tts_post_response(), _FakeResponse(content=wavs[0]), _tts_post_response(), _FakeResponse(content=wavs[1])]
        )
        text = "这是一句测试台词，" * 70  # 1260 字符 → 每段 ≤600 字符共 3 段，这里前 2 段打桩
        with patch("services.providers.tts_dashscope.httpx.AsyncClient", return_value=client):
            with patch.object(self.adapter, "_split_text", return_value=["第一段", "第二段"]):
                audio = asyncio.run(self.adapter.synthesize(TTSRequest(text=text, voice_id="Milo")))
        self.assertEqual(len(client.requests), 4)
        posts = [request for request in client.requests if request["method"] == "POST"]
        self.assertEqual(len(posts), 2)
        self.assertTrue(all(request["json"]["input"]["voice"] == "longshuo_v2" for request in posts))
        with wave.open(BytesIO(audio)) as reader:
            self.assertEqual(reader.getnchannels(), 1)
            self.assertEqual(reader.getframerate(), 22050)
            self.assertEqual(reader.getnframes(), 160 * 2)

    def test_split_text_keeps_segments_within_limit(self) -> None:
        text = "一句话。" * 500  # 2000 字符
        segments = self.adapter._split_text(text)
        self.assertGreater(len(segments), 1)
        for segment in segments:
            self.assertLessEqual(len(segment), 600)
        self.assertEqual("".join(segments), text)

    def test_split_text_short_text_single_segment(self) -> None:
        self.assertEqual(self.adapter._split_text("你好呀"), ["你好呀"])

    def test_usage_for_request_bills_by_characters(self) -> None:
        from services.providers.base import TTSRequest

        usage = self.adapter.usage_for_request(CAPABILITY_TTS, TTSRequest(text="  你好世界  ", voice_id="冰糖"))
        self.assertEqual(usage.characters, 4)
        self.assertTrue(usage.known)
        self.assertTrue(usage.billable)
        self.assertEqual(usage.extra.get("voice"), "longwan_v2")
        self.assertEqual(usage.extra.get("format"), "wav")


if __name__ == "__main__":
    unittest.main()
