"""火山方舟 Seedance / Seedream 系列接入的关键协议行为验收。

依据官方文档（2026-09 版）：
- Seedance 1.5 Pro 与 2.x 系列支持 ``generate_audio``（音画同生，默认开启）；
  本链路契约是无声视频 + 独立 TTS 配音，需显式关闭；1.0 系列不支持该参数，
  强校验请求体下携带会报错，不应发送。
- Seedream 5.0 系列（pro/lite）支持 ``output_format=png``；4.5/4.0 仅输出
  jpeg 且不支持自定义该参数。
- Seedream 参考图上限 14 张；出图总像素范围 [2560x1440, 4096x4096]，
  5.0 pro 上限约 2048x2048×1.1（≈462 万像素）。
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT  # noqa: F401,E402  (必须先于服务导入，绑定隔离环境)

from services.providers.base import ImageRequest, VideoRequest  # noqa: E402
from services.providers.endpoint import EndpointConfig  # noqa: E402
from services.providers.image_ark_seedream import ArkSeedreamImageAdapter  # noqa: E402
from services.providers.video_ark_seedance import ArkSeedanceVideoAdapter  # noqa: E402
from services.image_service import ImageService  # noqa: E402


def _image_endpoint(model: str) -> EndpointConfig:
    return EndpointConfig(protocol="ark-seedream", model=model)


def _image_request(**overrides) -> ImageRequest:
    base = {"prompt": "雨夜天台", "negative_prompt": "低质", "seed": 7, "size": "1440x2560"}
    base.update(overrides)
    return ImageRequest(**base)


class SeedreamSeriesPayloadTests(unittest.TestCase):
    def test_seedream_5_series_requests_png(self) -> None:
        for model in (
            "doubao-seedream-5-0-pro-260628",
            "doubao-seedream-5-0-260128",
            "doubao-seedream-5.0-lite",
        ):
            with self.subTest(model=model):
                adapter = ArkSeedreamImageAdapter(_image_endpoint(model))
                payload = adapter._payload(adapter._model_candidates()[0], _image_request(), "1440x2560")
                self.assertEqual(payload["output_format"], "png")

    def test_seedream_4_series_omits_output_format(self) -> None:
        # 4.5 / 4.0 仅输出 jpeg 且不支持自定义 output_format，携带会被强校验拒绝。
        for model in ("doubao-seedream-4-5-251128", "doubao-seedream-4-0-250828"):
            with self.subTest(model=model):
                adapter = ArkSeedreamImageAdapter(_image_endpoint(model))
                payload = adapter._payload(model, _image_request(), "1440x2560")
                self.assertNotIn("output_format", payload)

    def test_reference_images_capped_at_14(self) -> None:
        adapter = ArkSeedreamImageAdapter(_image_endpoint("doubao-seedream-5.0-lite"))
        payload = adapter._payload(
            "doubao-seedream-5-0-lite",
            _image_request(reference_images=[f"data:image/png;base64,{i:04d}" for i in range(20)]),
            "1440x2560",
        )
        self.assertEqual(len(payload["image"]), 14)

    def test_payload_common_contract(self) -> None:
        adapter = ArkSeedreamImageAdapter(_image_endpoint("doubao-seedream-5.0-lite"))
        payload = adapter._payload("doubao-seedream-5-0-lite", _image_request(), "1440x2560")
        self.assertEqual(payload["response_format"], "b64_json")
        self.assertFalse(payload["watermark"])
        self.assertEqual(payload["sequential_image_generation"], "disabled")
        # guidance_scale 对 5.0/4.5/4.0 系列不支持，不应发送。
        self.assertNotIn("guidance_scale", payload)

    def test_model_candidates_cover_dotted_and_dashed_ids(self) -> None:
        adapter = ArkSeedreamImageAdapter(_image_endpoint("doubao-seedream-5.0-lite"))
        candidates = adapter._model_candidates()
        self.assertIn("doubao-seedream-5.0-lite", candidates)
        self.assertIn("doubao-seedream-5-0-lite", candidates)


class SeedanceSeriesPayloadTests(unittest.TestCase):
    def _captured_create_task_payload(self, model: str, resolution: str = "720p") -> dict:
        adapter = ArkSeedanceVideoAdapter(
            EndpointConfig(
                protocol="ark-seedance",
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key="ark-test",
                model=model,
            )
        )

        class _FakeResponse:
            status_code = 200
            text = ""

            @staticmethod
            def json():
                return {"id": "task-1"}

        captured: dict = {}

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def request(self, method, url, **kwargs):
                captured.update(kwargs)
                return _FakeResponse()

        with patch("services.providers.video_ark_seedance.httpx.AsyncClient", return_value=_FakeClient()):
            asyncio.run(adapter._create_task("雨夜天台", 5, "9:16", resolution, None))
        return captured["json"]

    def test_generate_audio_disabled_for_audio_capable_series(self) -> None:
        # 音画同生系列默认有声；本链路无声 + TTS，必须显式关闭。
        for model in (
            "doubao-seedance-2-5-260628",
            "doubao-seedance-2-0-260128",
            "doubao-seedance-2-0-fast-260128",
            "doubao-seedance-1-5-pro-251215",
        ):
            with self.subTest(model=model):
                payload = self._captured_create_task_payload(model)
                self.assertIs(payload["generate_audio"], False)
                self.assertTrue(payload["return_last_frame"])
                self.assertFalse(payload["watermark"])

    def test_generate_audio_omitted_for_1_0_series(self) -> None:
        payload = self._captured_create_task_payload("doubao-seedance-1-0-pro-250528")
        self.assertNotIn("generate_audio", payload)
        self.assertTrue(payload["return_last_frame"])

    def test_fast_variant_caps_resolution_at_720p(self) -> None:
        # 官方模型列表：2.0 Fast 仅提供 480p/720p；应用默认 1080p 必须自动降档。
        for requested in ("1080p", "1080", "2k", "4k"):
            with self.subTest(requested=requested):
                payload = self._captured_create_task_payload("doubao-seedance-2-0-fast-260128", requested)
                self.assertEqual(payload["resolution"], "720p")

    def test_standard_series_keeps_requested_resolution(self) -> None:
        for model in ("doubao-seedance-2-5-260628", "doubao-seedance-2-0-260128", "doubao-seedance-1-5-pro-251215"):
            with self.subTest(model=model):
                payload = self._captured_create_task_payload(model, "1080p")
                self.assertEqual(payload["resolution"], "1080p")


class SeedreamRatioSizeTests(unittest.TestCase):
    def test_ratio_sizes_meet_seedream_series_pixel_bounds(self) -> None:
        service = ImageService()
        expected_ratios = {"9:16": 9 / 16, "3:4": 3 / 4, "1:1": 1.0, "4:3": 4 / 3, "16:9": 16 / 9}
        for ratio, expected in expected_ratios.items():
            with self.subTest(ratio=ratio):
                width_text, height_text = service._size_for_ratio(ratio).split("x")
                width, height = int(width_text), int(height_text)
                total = width * height
                # 系列公共下限 2560x1440；5.0 pro 上限约 2048x2048×1.1。
                self.assertGreaterEqual(total, 2560 * 1440)
                self.assertLessEqual(total, int(2048 * 2048 * 1.1))
                self.assertAlmostEqual(width / height, expected, places=2)


if __name__ == "__main__":
    unittest.main()
