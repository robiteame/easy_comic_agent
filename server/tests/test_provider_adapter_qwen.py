"""Qwen-Image 协议适配器的离线验收测试。"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import ANY, AsyncMock, patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from services.providers.base import ImageRequest  # noqa: E402
from services.providers.endpoint import EndpointConfig, normalize_protocol, settings_defaults  # noqa: E402
from services.providers.image_qwen import QwenImageAdapter  # noqa: E402
from services.providers.registry import get_adapter  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or str(payload)

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def post(self, url, **kwargs):
        self.requests.append({"method": "POST", "url": url, **kwargs})
        return self.responses.pop(0)

    async def get(self, url, **kwargs):
        self.requests.append({"method": "GET", "url": url, **kwargs})
        return self.responses.pop(0)


def _endpoint() -> EndpointConfig:
    return EndpointConfig(
        protocol="qwen-image",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        api_key="sk-test",
        model="qwen-image-plus",
    )


class QwenImageProtocolTests(unittest.TestCase):
    def test_protocol_registered(self) -> None:
        self.assertIs(get_adapter("image", "qwen-image"), QwenImageAdapter)

    def test_normalize_protocol_aliases(self) -> None:
        for alias in ("qwen", "qwen-image", "qwen-image-plus", "dashscope-qwen-image"):
            self.assertEqual(normalize_protocol("image", alias), "qwen-image")

    def test_defaults_follow_qwen_image_provider(self) -> None:
        from services.providers import endpoint as endpoint_module

        settings_obj = endpoint_module.settings
        original = (
            settings_obj.IMAGE_PROVIDER,
            settings_obj.QWEN_IMAGE_API_KEY,
            settings_obj.QWEN_IMAGE_BASE_URL,
            settings_obj.QWEN_IMAGE_MODEL,
        )
        try:
            settings_obj.IMAGE_PROVIDER = "qwen-image"
            settings_obj.QWEN_IMAGE_API_KEY = "qwen-key"
            settings_obj.QWEN_IMAGE_BASE_URL = "https://qwen.example.test/api/v1"
            settings_obj.QWEN_IMAGE_MODEL = "qwen-image"
            defaults = settings_defaults("image")
            self.assertEqual(defaults["protocol"], "qwen-image")
            self.assertEqual(defaults["api_key"], "qwen-key")
            self.assertEqual(defaults["base_url"], "https://qwen.example.test/api/v1")
            self.assertEqual(defaults["model"], "qwen-image")
        finally:
            (
                settings_obj.IMAGE_PROVIDER,
                settings_obj.QWEN_IMAGE_API_KEY,
                settings_obj.QWEN_IMAGE_BASE_URL,
                settings_obj.QWEN_IMAGE_MODEL,
            ) = original


class QwenImageAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = QwenImageAdapter(_endpoint())

    def test_payload_uses_native_dashscope_shape(self) -> None:
        payload = self.adapter._payload(ImageRequest(prompt="一只猫", negative_prompt="文字", seed=7, size="1440x2560"))
        self.assertEqual(payload["model"], "qwen-image-plus")
        self.assertEqual(payload["input"], {"prompt": "一只猫"})
        self.assertEqual(payload["parameters"]["negative_prompt"], "文字")
        self.assertEqual(payload["parameters"]["size"], "1440*2560")
        self.assertEqual(payload["parameters"]["seed"], 7)

    def test_compatible_mode_uses_openai_images_endpoint_and_shape(self) -> None:
        adapter = QwenImageAdapter(
            EndpointConfig(
                protocol="qwen-image",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key="sk-test",
                model="qwen-image-3.0",
            )
        )
        payload = adapter._compatible_payload(ImageRequest(prompt="一只猫", size="1440*2560"))
        self.assertEqual(
            adapter._compatible_url(), "https://dashscope.aliyuncs.com/compatible-mode/v1/images/generations"
        )
        self.assertEqual(payload["model"], "qwen-image-3.0")
        self.assertEqual(payload["prompt"], "一只猫")
        self.assertEqual(payload["size"], "1440x2560")

    def test_qwen_2_0_uses_multimodal_api_even_with_compatible_base_url(self) -> None:
        adapter = QwenImageAdapter(
            EndpointConfig(
                protocol="qwen-image",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key="sk-test",
                model="qwen-image-2.0",
            )
        )
        with patch.object(adapter, "_generate_multimodal", new=AsyncMock(return_value=b"image")) as generate:
            data = asyncio.run(adapter.generate(ImageRequest(prompt="p")))
        self.assertEqual(data, b"image")
        generate.assert_awaited_once()

    def test_qwen_3_0_uses_compatible_api(self) -> None:
        adapter = QwenImageAdapter(
            EndpointConfig(
                protocol="qwen-image",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key="sk-test",
                model="qwen-image-3.0",
            )
        )
        with patch.object(adapter, "_generate_compatible", new=AsyncMock(return_value=b"image")) as generate:
            data = asyncio.run(adapter.generate(ImageRequest(prompt="p")))
        self.assertEqual(data, b"image")
        generate.assert_awaited_once()

    def test_multimodal_payload_and_response_url(self) -> None:
        adapter = QwenImageAdapter(
            EndpointConfig(
                protocol="qwen-image",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key="sk-test",
                model="qwen-image-2.0",
            )
        )
        client = _FakeAsyncClient(
            [
                _FakeResponse(
                    {
                        "output": {
                            "choices": [
                                {
                                    "message": {
                                        "content": [{"type": "image", "image": "https://cdn.example.test/image.png"}]
                                    }
                                }
                            ]
                        }
                    }
                )
            ]
        )
        with (
            patch("services.providers.image_qwen.httpx.AsyncClient", return_value=client),
            patch(
                "services.providers.image_qwen.download_remote_bytes",
                new=AsyncMock(return_value=b"image"),
            ),
        ):
            data = asyncio.run(adapter._generate_multimodal(ImageRequest(prompt="p", size="1024x1024")))
        self.assertEqual(data, b"image")
        request = client.requests[0]
        self.assertTrue(request["url"].endswith("/api/v1/services/aigc/multimodal-generation/generation"))
        self.assertEqual(request["json"]["input"]["messages"][0]["content"][0]["text"], "p")

    def test_create_task_payload_and_headers(self) -> None:
        client = _FakeAsyncClient([_FakeResponse({"output": {"task_id": "tid-1"}})])
        with patch("services.providers.image_qwen.httpx.AsyncClient", return_value=client):
            task_id = asyncio.run(self.adapter._create_task(ImageRequest(prompt="p", size="1024x1024")))
        self.assertEqual(task_id, "tid-1")
        request = client.requests[0]
        self.assertEqual(
            request["url"],
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis",
        )
        self.assertEqual(request["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(request["headers"]["X-DashScope-Async"], "enable")

    def test_wait_for_task_polls_until_succeeded(self) -> None:
        client = _FakeAsyncClient(
            [
                _FakeResponse({"output": {"task_status": "RUNNING"}}),
                _FakeResponse(
                    {
                        "output": {
                            "task_status": "SUCCEEDED",
                            "results": [{"url": "https://cdn.example.test/image.png"}],
                        }
                    }
                ),
            ]
        )
        with (
            patch("services.providers.image_qwen.httpx.AsyncClient", return_value=client),
            patch("services.providers.image_qwen.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = asyncio.run(self.adapter._wait_for_task("tid-1"))
        self.assertEqual(result["output"]["task_status"], "SUCCEEDED")
        self.assertTrue(client.requests[0]["url"].endswith("/tasks/tid-1"))

    def test_generate_downloads_result_url(self) -> None:
        client = _FakeAsyncClient(
            [
                _FakeResponse({"output": {"task_id": "tid-1"}}),
                _FakeResponse(
                    {
                        "output": {
                            "task_status": "SUCCEEDED",
                            "results": [{"url": "https://cdn.example.test/image.png"}],
                        }
                    }
                ),
            ]
        )
        with (
            patch("services.providers.image_qwen.httpx.AsyncClient", return_value=client),
            patch(
                "services.providers.image_qwen.download_remote_bytes",
                new=AsyncMock(return_value=b"png-bytes"),
            ) as download,
        ):
            data = asyncio.run(self.adapter.generate(ImageRequest(prompt="p")))
        self.assertEqual(data, b"png-bytes")
        download.assert_awaited_once_with(
            "https://cdn.example.test/image.png",
            max_bytes=ANY,
            timeout=180,
        )
