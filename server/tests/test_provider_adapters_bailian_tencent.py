"""阿里云百炼（dashscope-wanx）与腾讯云（tencent-tts）适配器的离线验收测试。

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
from unittest.mock import AsyncMock, patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT  # noqa: F401,E402

from services.providers.endpoint import EndpointConfig, normalize_protocol, settings_defaults  # noqa: E402
from services.providers.registry import get_adapter  # noqa: E402
from services.providers.tts_tencent import TencentTTSAdapter, normalize_tencent_voice  # noqa: E402
from services.providers.usage import CAPABILITY_TTS, CAPABILITY_VIDEO  # noqa: E402
from services.providers.video_dashscope_wanx import DashscopeWanxVideoAdapter  # noqa: E402


class _FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload or {}
        self.status_code = status_code
        self.text = text

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

    async def request(self, method, url, **kwargs):
        # 适配 request_with_retry 所用的通用请求入口。
        self.requests.append({"method": str(method).upper(), "url": url, **kwargs})
        return self._responses.pop(0)


def _wanx_endpoint(api_key: str = "sk-test", model: str = "wan2.6-i2v") -> EndpointConfig:
    return EndpointConfig(
        protocol="dashscope-wanx",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        api_key=api_key,
        model=model,
    )


def _tencent_endpoint(api_key: str = "AKIDtest:secret-key") -> EndpointConfig:
    return EndpointConfig(
        protocol="tencent-tts",
        base_url="https://tts.tencentcloudapi.com",
        api_key=api_key,
        model="",
        params={"voice": "101001", "format": "wav"},
    )


def _make_wav(marker: int, frames: int = 160) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"".join(struct.pack("<h", 1000 * marker) for _ in range(frames)))
    return buffer.getvalue()


class DashscopeWanxProtocolTests(unittest.TestCase):
    def test_protocol_registered(self) -> None:
        self.assertIs(get_adapter("video", "dashscope-wanx"), DashscopeWanxVideoAdapter)

    def test_normalize_protocol_aliases(self) -> None:
        for alias in ("wanx", "dashscope", "bailian", "aliyun", "wan2.5-i2v-plus"):
            self.assertEqual(normalize_protocol("video", alias), "dashscope-wanx")
        # 既有别名不受影响
        self.assertEqual(normalize_protocol("video", "seedance"), "ark-seedance")
        self.assertEqual(normalize_protocol("video", "veo3"), "native-audio")

    def test_video_defaults_follow_wanx_provider(self) -> None:
        from services.providers import endpoint as endpoint_module

        settings_obj = endpoint_module.settings
        original = settings_obj.VIDEO_PROVIDER
        try:
            settings_obj.VIDEO_PROVIDER = "wanx"
            defaults = settings_defaults("video")
            self.assertEqual(defaults["protocol"], "dashscope-wanx")
            self.assertEqual(defaults["model"], settings_obj.DASHSCOPE_VIDEO_MODEL)
            self.assertEqual(defaults["base_url"], settings_obj.DASHSCOPE_BASE_URL)
        finally:
            settings_obj.VIDEO_PROVIDER = original
        self.assertEqual(settings_defaults("video")["protocol"], "ark-seedance")


class DashscopeWanxAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = DashscopeWanxVideoAdapter(_wanx_endpoint())

    def test_resolve_resolution(self) -> None:
        # 默认端点模型 wan2.6-i2v 仅提供 720P/1080P 档位，480P 自动降档。
        self.assertEqual(self.adapter._resolve_resolution("720p"), "720P")
        self.assertEqual(self.adapter._resolve_resolution("1080p"), "1080P")
        self.assertEqual(self.adapter._resolve_resolution("480p"), "720P")
        # 2K/4K/未知值统一回落到万相最高档 1080P。
        self.assertEqual(self.adapter._resolve_resolution("4k"), "1080P")
        self.assertEqual(self.adapter._resolve_resolution(""), "1080P")
        # Wan 2.5 / 3.0 支持全部档位，480P 原样通过。
        for model in ("wan2.5-i2v-preview", "wan3.0-video"):
            adapter = DashscopeWanxVideoAdapter(_wanx_endpoint(model=model))
            self.assertEqual(adapter._resolve_resolution("480p"), "480P")

    def test_create_task_payload_and_headers(self) -> None:
        client = _FakeAsyncClient(
            [_FakeResponse({"output": {"task_id": "tid-1", "task_status": "PENDING"}})]
        )
        from services.providers.base import VideoRequest

        request = VideoRequest(
            prompt="少女在雨中奔跑",
            reference_image="data:image/png;base64,AAAA",
            duration=5,
            ratio="9:16",
            resolution="720p",
        )
        with patch("services.providers.video_dashscope_wanx.httpx.AsyncClient", return_value=client):
            task_id = asyncio.run(self.adapter._create_task(request))
        self.assertEqual(task_id, "tid-1")
        sent = client.requests[0]
        self.assertEqual(
            sent["url"], "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        )
        self.assertEqual(sent["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(sent["headers"]["X-DashScope-Async"], "enable")
        # Wan 2.6 及更早：旧代接口，首帧放 input.img_url，分辨率用档位。
        self.assertEqual(sent["json"]["model"], "wan2.6-i2v")
        self.assertEqual(sent["json"]["input"]["img_url"], "data:image/png;base64,AAAA")
        self.assertNotIn("media", sent["json"]["input"])
        self.assertEqual(sent["json"]["parameters"]["resolution"], "720P")
        self.assertEqual(sent["json"]["parameters"]["duration"], 5)
        self.assertNotIn("audio", sent["json"]["parameters"])

    def test_create_task_wan3_uses_media_and_disables_audio(self) -> None:
        adapter = DashscopeWanxVideoAdapter(_wanx_endpoint(model="wan3.0-video"))
        client = _FakeAsyncClient(
            [_FakeResponse({"output": {"task_id": "tid-3", "task_status": "PENDING"}})]
        )
        from services.providers.base import VideoRequest

        request = VideoRequest(
            prompt="少女在雨中奔跑",
            reference_image="data:image/png;base64,AAAA",
            duration=5,
            ratio="9:16",
            resolution="1080p",
        )
        with patch("services.providers.video_dashscope_wanx.httpx.AsyncClient", return_value=client):
            task_id = asyncio.run(adapter._create_task(request))
        self.assertEqual(task_id, "tid-3")
        payload = client.requests[0]["json"]
        # Wan 3.0：新一代接口，首帧放 input.media（type=first_frame），
        # 默认有声视频，需显式 audio=false 维持「无声视频 + 独立 TTS」契约。
        self.assertNotIn("img_url", payload["input"])
        self.assertEqual(payload["input"]["media"], [{"type": "first_frame", "url": "data:image/png;base64,AAAA"}])
        self.assertEqual(payload["parameters"]["resolution"], "1080P")
        self.assertFalse(payload["parameters"]["audio"])

    def test_create_task_wan27_media_without_audio_toggle(self) -> None:
        adapter = DashscopeWanxVideoAdapter(_wanx_endpoint(model="wan2.7-i2v"))
        client = _FakeAsyncClient(
            [_FakeResponse({"output": {"task_id": "tid-7", "task_status": "PENDING"}})]
        )
        from services.providers.base import VideoRequest

        request = VideoRequest(prompt="空镜", reference_image="data:image/png;base64,AAAA", duration=5)
        with patch("services.providers.video_dashscope_wanx.httpx.AsyncClient", return_value=client):
            asyncio.run(adapter._create_task(request))
        payload = client.requests[0]["json"]
        # Wan 2.7 同为 media 输入，但无默认音频，不携带 audio 开关。
        self.assertNotIn("img_url", payload["input"])
        self.assertEqual(payload["input"]["media"][0]["type"], "first_frame")
        self.assertNotIn("audio", payload["parameters"])

    def test_create_task_r2v_pairs_reference_image_with_first_frame(self) -> None:
        # r2v 契约：参考图像/参考视频至少 1 个，仅传 first_frame 会被服务商判
        # InvalidParameter；已审核故事板首帧同时以主体参考身份重复传入。
        adapter = DashscopeWanxVideoAdapter(_wanx_endpoint(model="wan2.7-r2v"))
        client = _FakeAsyncClient(
            [_FakeResponse({"output": {"task_id": "tid-r", "task_status": "PENDING"}})]
        )
        from services.providers.base import VideoRequest

        request = VideoRequest(
            prompt="少女在雨中奔跑",
            reference_image="data:image/png;base64,AAAA",
            duration=5,
            resolution="480p",
        )
        with patch("services.providers.video_dashscope_wanx.httpx.AsyncClient", return_value=client):
            asyncio.run(adapter._create_task(request))
        payload = client.requests[0]["json"]
        self.assertNotIn("img_url", payload["input"])
        self.assertEqual(
            payload["input"]["media"],
            [
                {"type": "first_frame", "url": "data:image/png;base64,AAAA"},
                {"type": "reference_image", "url": "data:image/png;base64,AAAA"},
            ],
        )
        # r2v 仅提供 720P/1080P 档位，480P 请求自动降档。
        self.assertEqual(payload["parameters"]["resolution"], "720P")

    def test_r2v_detection_covers_dated_model_variants(self) -> None:
        adapter = DashscopeWanxVideoAdapter(_wanx_endpoint(model="wan2.7-r2v-2026-06-12"))
        self.assertTrue(adapter._is_reference_to_video_model())
        media = adapter._media_entries("data:image/png;base64,AAAA")
        self.assertEqual([item["type"] for item in media], ["first_frame", "reference_image"])
        # 非 r2v 的 media 输入模型（i2v / All-in-One）仅传首帧。
        for model in ("wan2.7-i2v", "wan3.0-video"):
            adapter = DashscopeWanxVideoAdapter(_wanx_endpoint(model=model))
            self.assertFalse(adapter._is_reference_to_video_model())
            self.assertEqual(
                adapter._media_entries("data:image/png;base64,AAAA"),
                [{"type": "first_frame", "url": "data:image/png;base64,AAAA"}],
            )

    def test_create_task_text_only_omits_reference_fields(self) -> None:
        client = _FakeAsyncClient(
            [_FakeResponse({"output": {"task_id": "tid-2", "task_status": "PENDING"}})]
        )
        from services.providers.base import VideoRequest

        with patch("services.providers.video_dashscope_wanx.httpx.AsyncClient", return_value=client):
            asyncio.run(self.adapter._create_task(VideoRequest(prompt="空镜", duration=5)))
        self.assertNotIn("img_url", client.requests[0]["json"]["input"])
        self.assertNotIn("media", client.requests[0]["json"]["input"])

    def test_create_task_missing_task_id_raises(self) -> None:
        client = _FakeAsyncClient([_FakeResponse({"output": {}})])
        from services.providers.base import VideoRequest

        with patch("services.providers.video_dashscope_wanx.httpx.AsyncClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "任务 ID"):
                asyncio.run(self.adapter._create_task(VideoRequest(prompt="p")))

    def test_wait_for_task_polls_until_succeeded(self) -> None:
        client = _FakeAsyncClient(
            [
                _FakeResponse({"output": {"task_id": "tid-1", "task_status": "RUNNING"}}),
                _FakeResponse(
                    {"output": {"task_id": "tid-1", "task_status": "SUCCEEDED", "video_url": "https://cdn.test/v.mp4"}}
                ),
            ]
        )
        with (
            patch("services.providers.video_dashscope_wanx.httpx.AsyncClient", return_value=client),
            patch("services.providers.video_dashscope_wanx.asyncio.sleep", new_callable=AsyncMock),
        ):
            data = asyncio.run(self.adapter._wait_for_task("tid-1"))
        self.assertEqual(data["output"]["video_url"], "https://cdn.test/v.mp4")
        self.assertTrue(client.requests[0]["url"].endswith("/tasks/tid-1"))

    def test_wait_for_task_failed_status_raises(self) -> None:
        client = _FakeAsyncClient(
            [_FakeResponse({"output": {"task_id": "tid-1", "task_status": "FAILED", "code": "InternalError"}})]
        )
        with (
            patch("services.providers.video_dashscope_wanx.httpx.AsyncClient", return_value=client),
            patch("services.providers.video_dashscope_wanx.asyncio.sleep", new_callable=AsyncMock),
        ):
            with self.assertRaisesRegex(RuntimeError, "百炼视频任务失败"):
                asyncio.run(self.adapter._wait_for_task("tid-1"))

    def test_usage_for_request_bills_by_seconds(self) -> None:
        from services.providers.base import VideoRequest

        usage = self.adapter.usage_for_request(
            CAPABILITY_VIDEO, VideoRequest(prompt="p", duration=5, ratio="9:16", resolution="720p")
        )
        self.assertEqual(usage.seconds, 5.0)
        self.assertTrue(usage.known)
        self.assertTrue(usage.billable)
        self.assertEqual(usage.quantity, 5)

    def test_capabilities_silent_video(self) -> None:
        capabilities = self.adapter.capabilities
        self.assertTrue(capabilities.reference_image)
        self.assertFalse(capabilities.native_audio)


class TencentTTSProtocolTests(unittest.TestCase):
    def test_protocol_registered(self) -> None:
        self.assertIs(get_adapter("voice", "tencent-tts"), TencentTTSAdapter)

    def test_normalize_protocol_aliases(self) -> None:
        self.assertEqual(normalize_protocol("voice", "tencent"), "tencent-tts")
        self.assertEqual(normalize_protocol("voice", "mimo"), "mimo-tts")

    def test_voice_defaults_switch_to_tencent_when_configured(self) -> None:
        from services.providers import endpoint as endpoint_module

        settings_obj = endpoint_module.settings
        original = (settings_obj.TENCENT_SECRET_ID, settings_obj.TENCENT_SECRET_KEY)
        try:
            settings_obj.TENCENT_SECRET_ID = ""
            settings_obj.TENCENT_SECRET_KEY = ""
            self.assertEqual(settings_defaults("voice")["protocol"], "mimo-tts")
            settings_obj.TENCENT_SECRET_ID = "AKIDtest"
            settings_obj.TENCENT_SECRET_KEY = "secret"
            defaults = settings_defaults("voice")
            self.assertEqual(defaults["protocol"], "tencent-tts")
            self.assertEqual(defaults["api_key"], "AKIDtest:secret")
            self.assertEqual(defaults["params"]["voice"], settings_obj.TENCENT_TTS_VOICE)
        finally:
            settings_obj.TENCENT_SECRET_ID, settings_obj.TENCENT_SECRET_KEY = original


class TencentTTSAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = TencentTTSAdapter(_tencent_endpoint())

    def test_normalize_tencent_voice(self) -> None:
        self.assertEqual(normalize_tencent_voice("冰糖"), "101001")
        self.assertEqual(normalize_tencent_voice("Milo"), "101030")
        self.assertEqual(normalize_tencent_voice("Dean"), "101004")
        self.assertEqual(normalize_tencent_voice("101030"), "101030")  # 数字 ID 透传
        self.assertEqual(normalize_tencent_voice("未知音色", default_voice="101027"), "101027")
        self.assertEqual(normalize_tencent_voice(""), "101001")

    def test_split_text_keeps_segments_within_limit(self) -> None:
        text = "一句话。" * 30  # 120 字符
        segments = self.adapter._split_text(text)
        self.assertGreater(len(segments), 1)
        for segment in segments:
            self.assertLessEqual(len(segment), 45)
            self.assertLessEqual(len(segment.encode("utf-8")), 150)
        self.assertEqual("".join(segments), text)

    def test_split_text_short_text_single_segment(self) -> None:
        self.assertEqual(self.adapter._split_text("你好呀"), ["你好呀"])

    def test_credentials_reject_invalid_format(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "SecretId:SecretKey"):
            TencentTTSAdapter(_tencent_endpoint(api_key="no-colon"))._credentials()
        with self.assertRaisesRegex(RuntimeError, "API Key"):
            TencentTTSAdapter(_tencent_endpoint(api_key=""))._credentials()

    def test_signed_headers_structure(self) -> None:
        headers = self.adapter._signed_headers(
            "tts.tencentcloudapi.com", "TextToVoice", "{}", "AKIDtest", "secret"
        )
        authorization = headers["Authorization"]
        self.assertTrue(authorization.startswith("TC3-HMAC-SHA256 Credential=AKIDtest/"))
        self.assertIn("SignedHeaders=content-type;host;x-tc-action", authorization)
        self.assertIn("Signature=", authorization)
        self.assertEqual(headers["X-TC-Action"], "TextToVoice")
        self.assertEqual(headers["X-TC-Version"], "2019-08-23")
        self.assertEqual(headers["Host"], "tts.tencentcloudapi.com")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_synthesize_multi_segment_concatenates_wav(self) -> None:
        from services.providers.base import TTSRequest

        wavs = [_make_wav(marker) for marker in (1, 2, 3, 4)]
        client = _FakeAsyncClient(
            [_FakeResponse({"Response": {"Audio": base64.b64encode(item).decode("ascii")}}) for item in wavs]
        )
        text = "这是一句测试台词，" * 20  # 180 字符 → 每段 45 字符共 4 段
        with patch("services.providers.tts_tencent.httpx.AsyncClient", return_value=client):
            audio = asyncio.run(self.adapter.synthesize(TTSRequest(text=text, voice_id="冰糖")))
        self.assertEqual(len(client.requests), 4)
        sent_texts = [request["content"] for request in client.requests]
        self.assertTrue(all('"VoiceType":101001' in body for body in sent_texts))
        with wave.open(BytesIO(audio)) as reader:
            self.assertEqual(reader.getnchannels(), 1)
            self.assertEqual(reader.getframerate(), 16000)
            self.assertEqual(reader.getnframes(), 160 * 4)

    def test_synthesize_single_segment_passthrough(self) -> None:
        from services.providers.base import TTSRequest

        wav = _make_wav(1)
        client = _FakeAsyncClient(
            [_FakeResponse({"Response": {"Audio": base64.b64encode(wav).decode("ascii")}})]
        )
        with patch("services.providers.tts_tencent.httpx.AsyncClient", return_value=client):
            audio = asyncio.run(self.adapter.synthesize(TTSRequest(text="你好")))
        self.assertEqual(audio, wav)
        self.assertEqual(client.requests[0]["url"], "https://tts.tencentcloudapi.com/")

    def test_synthesize_provider_error_raises(self) -> None:
        from services.providers.base import TTSRequest

        client = _FakeAsyncClient(
            [_FakeResponse({"Response": {"Error": {"Code": "AuthFailure", "Message": "签名错误"}}})]
        )
        with patch("services.providers.tts_tencent.httpx.AsyncClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "AuthFailure"):
                asyncio.run(self.adapter.synthesize(TTSRequest(text="你好")))

    def test_usage_for_request_bills_by_characters(self) -> None:
        from services.providers.base import TTSRequest

        usage = self.adapter.usage_for_request(CAPABILITY_TTS, TTSRequest(text="  你好世界  ", voice_id="冰糖"))
        self.assertEqual(usage.characters, 4)
        self.assertTrue(usage.known)
        self.assertEqual(usage.extra.get("voice"), "101001")
        self.assertEqual(usage.extra.get("format"), "wav")


if __name__ == "__main__":
    unittest.main()
