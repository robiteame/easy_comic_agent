"""阿里云百炼万相视频适配器的离线验收测试。

覆盖 2026-09-20 镜头视频故障引入的三项修复：URL 归一化（compatible-mode
杂交路径 404）、参考图请求体预算压缩（~200KB 网关上限掐连接）、瞬态传输
错误重试。全部用替身，不发起真实网络请求。
"""

from __future__ import annotations

import asyncio
import base64
import io
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx  # noqa: E402
from PIL import Image

from config import settings  # noqa: E402
from services.providers.base import VideoRequest  # noqa: E402
from services.providers.endpoint import EndpointConfig  # noqa: E402
from services.providers.http_retry import request_with_retry  # noqa: E402
from services.providers.registry import get_adapter  # noqa: E402
from services.providers.video_dashscope_wanx import DashscopeWanxVideoAdapter  # noqa: E402
from services.reference_asset_service import ReferenceAssetService  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


def _endpoint(base_url: str = "") -> EndpointConfig:
    return EndpointConfig(
        protocol="dashscope-wanx",
        base_url=base_url,
        api_key="sk-test",
        model="wan2.6-i2v-flash",
        params={},
    )


def setUpModule() -> None:
    """共享夹具：随机噪声大图（无法无损压缩，逼近最坏情况）与纯色小图。"""
    global _BIG_PNG, _SMALL_PNG
    _BIG_PNG = TEST_ROOT / "output" / "wanx_ref_big.png"
    _SMALL_PNG = TEST_ROOT / "output" / "wanx_ref_small.png"
    _BIG_PNG.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    image = Image.new("RGB", (720, 1280))
    image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(720 * 1280)])
    image.save(_BIG_PNG, format="PNG")
    Image.new("RGB", (64, 64), (200, 30, 30)).save(_SMALL_PNG, format="PNG")


_BIG_PNG = Path()
_SMALL_PNG = Path()


class _FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload or {}
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload

    async def aclose(self):
        return None


class _FakeAsyncClient:
    """支持 request_with_retry 所用 request() 接口的 httpx.AsyncClient 替身。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def request(self, method, url, **kwargs):
        self.requests.append({"method": method, "url": url, **kwargs})
        step = self._responses.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class ApiBaseNormalizationTests(unittest.TestCase):
    """base_url 无论粘贴何种形式，统一归一化到主机的 /api/v1。"""

    def test_registered(self) -> None:
        self.assertIs(get_adapter("video", "dashscope-wanx"), DashscopeWanxVideoAdapter)

    def test_compatible_mode_url_is_normalized(self) -> None:
        adapter = DashscopeWanxVideoAdapter(_endpoint("https://llm-x.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"))
        self.assertEqual(
            adapter._create_url(),
            "https://llm-x.cn-beijing.maas.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis",
        )

    def test_plain_api_v1_bare_host_and_empty(self) -> None:
        cases = {
            "https://dashscope.aliyuncs.com/api/v1": "https://dashscope.aliyuncs.com/api/v1",
            "https://dashscope.aliyuncs.com/api/v1/": "https://dashscope.aliyuncs.com/api/v1",
            "dashscope.aliyuncs.com": "https://dashscope.aliyuncs.com/api/v1",
            "": "https://dashscope.aliyuncs.com/api/v1",
        }
        for base_url, expected in cases.items():
            self.assertEqual(DashscopeWanxVideoAdapter(_endpoint(base_url))._api_base(), expected)

    def test_port_and_host_header_preserved(self) -> None:
        adapter = DashscopeWanxVideoAdapter(_endpoint("https://gw.example.test:8443/compatible-mode/v1"))
        self.assertEqual(adapter._api_base(), "https://gw.example.test:8443/api/v1")


class ReferenceBudgetTests(unittest.TestCase):
    """超预算参考图必须压进预算，小图保持原样。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.service = ReferenceAssetService()
        cls.big_png = _BIG_PNG
        cls.small_png = _SMALL_PNG

    def test_oversized_local_file_is_recompressed_within_budget(self) -> None:
        self.assertGreater(self.big_png.stat().st_size, settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES)
        url = self.service.to_image_url(str(self.big_png), max_bytes=settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES)
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        raw = base64.b64decode(url.partition(",")[2])
        self.assertLessEqual(len(raw), settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES)
        with Image.open(io.BytesIO(raw)) as check:
            check.verify()

    def test_oversized_data_url_is_recompressed(self) -> None:
        oversized = self.service.to_image_url(str(self.big_png))  # 无预算 → 原样内联
        self.assertGreater(len(oversized), settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES * 4 // 3)
        url = self.service.to_image_url(oversized, max_bytes=settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES)
        raw = base64.b64decode(url.partition(",")[2])
        self.assertLessEqual(len(raw), settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES)
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))

    def test_small_file_and_budget_free_behavior_unchanged(self) -> None:
        url = self.service.to_image_url(str(self.small_png), max_bytes=settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES)
        self.assertTrue(url.startswith("data:image/png;base64,"))
        oversized = self.service.to_image_url(str(self.big_png))
        self.assertTrue(oversized.startswith("data:image/png;base64,"))

    def test_http_url_passthrough(self) -> None:
        self.assertEqual(
            self.service.to_image_url("https://cdn.example.test/a.png", max_bytes=1024),
            "https://cdn.example.test/a.png",
        )


class HttpRetryTests(unittest.TestCase):
    def _run(self, client):
        return asyncio.run(request_with_retry(client, "POST", "https://x.test", attempts=3, backoff_seconds=0))

    def test_transport_errors_are_retried(self) -> None:
        client = _FakeAsyncClient(
            [httpx.ReadError("reset"), httpx.ConnectError("boom"), _FakeResponse(status_code=200)]
        )
        response = self._run(client)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(client.requests), 3)

    def test_transient_status_is_retried(self) -> None:
        client = _FakeAsyncClient([_FakeResponse(status_code=503), _FakeResponse(status_code=200)])
        response = self._run(client)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(client.requests), 2)

    def test_business_error_is_not_retried(self) -> None:
        client = _FakeAsyncClient([_FakeResponse(status_code=400, text="bad request")])
        response = self._run(client)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(client.requests), 1)

    def test_exhausted_attempts_raise_last_transport_error(self) -> None:
        client = _FakeAsyncClient([httpx.ReadError("a"), httpx.ReadError("b"), httpx.ReadError("c")])
        with self.assertRaises(httpx.ReadError):
            self._run(client)
        self.assertEqual(len(client.requests), 3)

    def test_last_transient_status_returned_when_retries_exhausted(self) -> None:
        client = _FakeAsyncClient(
            [_FakeResponse(status_code=503), _FakeResponse(status_code=503), _FakeResponse(status_code=503)]
        )
        response = self._run(client)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(len(client.requests), 3)


class CreateTaskBudgetGuardTests(unittest.TestCase):
    """创建任务时超预算参考图必须先压缩，压不进预算要给出明确报错。"""

    def _request(self, reference_image: str) -> VideoRequest:
        return VideoRequest(
            prompt="a quiet rooftop at dusk",
            reference_image=reference_image,
            dialogues=None,
            duration=5,
            ratio="9:16",
            resolution="720p",
            project_id="wanx-test-project",
            output_video_path=TEST_ROOT / "output" / "wanx-test.mp4",
            output_frame_path=TEST_ROOT / "output" / "wanx-test_frame.png",
        )

    def test_create_task_payload_compresses_oversized_reference(self) -> None:
        oversized = ReferenceAssetService().to_image_url(str(_BIG_PNG))
        self.assertGreater(len(oversized), settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES * 4 // 3)
        client = _FakeAsyncClient([_FakeResponse({"output": {"task_id": "task-1"}})])
        adapter = DashscopeWanxVideoAdapter(_endpoint("https://dashscope.aliyuncs.com/api/v1"))
        with patch("services.providers.video_dashscope_wanx.httpx.AsyncClient", return_value=client):
            task_id = asyncio.run(adapter._create_task(self._request(oversized)))
        self.assertEqual(task_id, "task-1")
        sent = client.requests[0]["json"]["input"]["img_url"]
        raw = base64.b64decode(sent.partition(",")[2])
        self.assertLessEqual(len(raw), settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES)
        self.assertEqual(
            client.requests[0]["url"],
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis",
        )
        self.assertEqual(client.requests[0]["headers"]["X-DashScope-Async"], "enable")

    def test_create_task_rejects_uncompressible_reference(self) -> None:
        adapter = DashscopeWanxVideoAdapter(_endpoint())
        # 超预算且内容不是有效图像（PIL 无法解码 → 压缩阶梯必然失败）
        garbage = "data:image/png;base64," + base64.b64encode(
            b"\x00" * (settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES + 1024)
        ).decode("ascii")
        request = self._request(garbage)
        with self.assertRaisesRegex(RuntimeError, "参考图"):
            asyncio.run(adapter._create_task(request))


if __name__ == "__main__":
    unittest.main()
