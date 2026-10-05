"""图像/语音适配器路由的验收测试：协议决定路径，异常场景安全回退。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from services.image_service import ImageService  # noqa: E402
from services.providers.endpoint import EndpointConfig  # noqa: E402
from services.providers.image_placeholder import PlaceholderImageAdapter  # noqa: E402
from services.providers.registry import UnknownProtocolError, get_adapter  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402


def _endpoint(protocol: str, api_key: str = "") -> EndpointConfig:
    return EndpointConfig(
        protocol=protocol,
        base_url="https://img.example.test",
        api_key=api_key,
        model="m",
        params={"image_size": "1024x1024"},
    )


class OfflineImageFlowTests(unittest.TestCase):
    """离线验收（IMAGE_PROVIDER=local 等价场景）：不配任何 key 跑通出图。"""

    def test_generate_shot_image_produces_placeholder_without_keys(self) -> None:
        import asyncio

        service = ImageService()
        shot = {
            "shot_id": "offline-shot",
            "version": 1,
            "scene_description": "a quiet classroom in the morning",
            "character_action": "a girl raises her hand",
            "shot_type": "medium",
            "camera_angle": "正面",
            "output_format": "9:16",
        }
        path = asyncio.run(
            service.generate_shot_image(shot, [], {"prompt_prefix": "", "style_label": "anime"}, "offline-project")
        )
        image_path = Path(path)
        self.assertTrue(image_path.exists())
        self.assertGreater(image_path.stat().st_size, 4096)
        from PIL import Image

        with Image.open(image_path) as image:
            image.verify()


class ImageRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ImageService()

    def test_missing_key_falls_back_to_placeholder(self) -> None:
        with patch("services.image_service.get_endpoint", return_value=_endpoint("ark-seedream", api_key="")):
            adapter, endpoint = self.service._resolve_route()
            self.assertIsInstance(adapter, PlaceholderImageAdapter)
            self.assertEqual(endpoint.protocol, "ark-seedream")

    def test_unknown_protocol_falls_back_to_placeholder(self) -> None:
        with (
            patch("services.image_service.get_endpoint", return_value=_endpoint("mystery")),
            patch("services.image_service.get_adapter", side_effect=UnknownProtocolError("boom")),
        ):
            with self.assertLogs("services.image_service", level="WARNING"):
                adapter, _ = self.service._resolve_route()
            self.assertIsInstance(adapter, PlaceholderImageAdapter)

    def test_known_protocol_with_key_routes_to_adapter(self) -> None:
        with (
            patch("services.image_service.get_endpoint", return_value=_endpoint("stability", api_key="sk-test")),
            patch("services.image_service.get_adapter", return_value=get_adapter("image", "stability")),
        ):
            adapter, _ = self.service._resolve_route()
            from services.providers.image_stability import StabilityImageAdapter

            self.assertIsInstance(adapter, StabilityImageAdapter)

    def test_placeholder_generation_offline(self) -> None:
        """离线全流程底线：无任何 key 也能产出可打开的占位图。"""
        import asyncio

        from services.providers.base import ImageRequest

        adapter = PlaceholderImageAdapter(EndpointConfig(protocol="placeholder"))
        data = asyncio.run(
            adapter.generate(ImageRequest(prompt="a lonely robot", label="SHOT PLACEHOLDER", size="1024x1024"))
        )
        self.assertGreater(len(data), 1024)

    def test_reference_images_gated_by_adapter_capability(self) -> None:
        import asyncio

        from services.providers.base import BaseAdapter, ImageCapabilities

        captured: dict = {}

        def _make_adapter(supports_refs: bool):
            class _RecordingAdapter(BaseAdapter):
                capabilities = ImageCapabilities(reference_images=supports_refs, requires_credentials=False)

                async def generate(self, request):
                    captured["reference_images"] = request.reference_images
                    placeholder = PlaceholderImageAdapter(EndpointConfig(protocol="placeholder"))
                    return placeholder._render("X", request.prompt, (64, 64))

            return _RecordingAdapter

        refs = ["data:image/png;base64,AAAA"]

        with (
            patch("services.image_service.get_endpoint", return_value=_endpoint("placeholder")),
            patch("services.image_service.get_adapter", return_value=_make_adapter(supports_refs=False)),
        ):
            with self.assertRaisesRegex(RuntimeError, "不支持参考图"):
                asyncio.run(
                    self.service._generate(
                        prompt="p", negative_prompt="", seed=1, reference_images=refs, preferred_size="64x64", label="X"
                    )
                )
        self.assertNotIn("reference_images", captured)

        with (
            patch("services.image_service.get_endpoint", return_value=_endpoint("ark-seedream")),
            patch("services.image_service.get_adapter", return_value=_make_adapter(supports_refs=True)),
        ):
            asyncio.run(
                self.service._generate(
                    prompt="p", negative_prompt="", seed=1, reference_images=refs, preferred_size="64x64", label="X"
                )
            )
        self.assertEqual(captured["reference_images"], refs)


if __name__ == "__main__":
    unittest.main()
