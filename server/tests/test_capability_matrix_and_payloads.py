"""Capability Matrix 与适配器真实请求载荷契约测试。"""

from __future__ import annotations

import asyncio
import base64
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT  # noqa: F401,E402

from services.consistency_metrics import payload_metrics  # noqa: E402
from services.image_service import ImageService  # noqa: E402
from services.providers.base import ImageCapabilities, ImageRequest, ReferenceAsset, VideoRequest  # noqa: E402
from services.providers.capability_matrix import capability_report, payload_control_types  # noqa: E402
from services.providers.endpoint import EndpointConfig  # noqa: E402
from services.providers.image_ark_seedream import ArkSeedreamImageAdapter  # noqa: E402
from services.providers.video_dashscope_wanx import DashscopeWanxVideoAdapter  # noqa: E402
from services.providers.video_ark_seedance import ArkSeedanceVideoAdapter  # noqa: E402
from services.video_service import VideoService  # noqa: E402

_PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 64).decode("ascii")


class CapabilityMatrixTests(unittest.TestCase):
    def test_real_adapters_do_not_fabricate_controls_or_weights(self) -> None:
        image = capability_report(
            "image",
            "ark-seedream",
            model="doubao-seedream-5-0-lite",
            adapter_cls=ArkSeedreamImageAdapter,
        )
        video = capability_report(
            "video",
            "dashscope-wanx",
            model="wan2.7-i2v",
            adapter_cls=DashscopeWanxVideoAdapter,
        )
        self.assertEqual(image["features"]["multiple_reference_images"]["status"], "supported")
        self.assertEqual(image["features"]["reference_weights"]["status"], "unsupported")
        self.assertEqual(image["reference_weight_policy"], "text_only_policy")
        for key in ("pose_control", "depth_control", "lora", "ip_adapter"):
            self.assertEqual(image["features"][key]["status"], "unsupported")
            self.assertEqual(video["features"][key]["status"], "unsupported")
        self.assertEqual(video["reference_mode"], "first_frame_only")
        self.assertEqual(video["features"]["multiple_reference_images"]["status"], "unsupported")

    def test_r2v_model_is_partial_or_supported_for_multi_reference(self) -> None:
        report = capability_report(
            "video",
            "dashscope-wanx",
            model="wan2.7-r2v-2026-06-12",
            adapter_cls=DashscopeWanxVideoAdapter,
        )
        self.assertEqual(report["reference_mode"], "multi_reference")
        self.assertIn(report["features"]["multiple_reference_images"]["status"], {"supported", "partial"})
        self.assertEqual(report["reference_role_parameter"], "input.media[].type")


class AdapterPayloadTests(unittest.TestCase):
    def test_seedream_payload_contains_all_reference_images_but_no_fake_weights(self) -> None:
        adapter = ArkSeedreamImageAdapter(
            EndpointConfig(protocol="ark-seedream", api_key="k", model="doubao-seedream-5-0-lite")
        )
        request = ImageRequest(
            prompt="p",
            reference_images=["https://cdn.test/character.png", "https://cdn.test/scene.png"],
            reference_assets=[
                ReferenceAsset(url="https://cdn.test/character.png", type="character_three_view", role="identity"),
                ReferenceAsset(url="https://cdn.test/scene.png", type="scene_baseline", role="environment"),
            ],
            size="1024x1024",
        )
        payload = adapter._payload(request.reference_images and "doubao-seedream-5-0-lite" or "", request, "1024x1024")
        self.assertEqual(payload["image"], request.reference_images)
        self.assertNotIn("image_weight", payload)
        self.assertNotIn("reference_weights", payload)

    def test_wanx_r2v_sends_first_frame_and_typed_reference_media(self) -> None:
        adapter = DashscopeWanxVideoAdapter(
            EndpointConfig(protocol="dashscope-wanx", api_key="k", model="wan2.7-r2v")
        )
        request = VideoRequest(
            prompt="p",
            reference_image=_PNG_DATA_URL,
            reference_assets=[
                ReferenceAsset(url=_PNG_DATA_URL, type="approved_storyboard_first_frame", provider_type="first_frame"),
                ReferenceAsset(url=_PNG_DATA_URL + "1", type="character_three_view", provider_type="reference_image"),
                ReferenceAsset(url=_PNG_DATA_URL + "2", type="scene_baseline", provider_type="reference_image"),
            ],
        )
        media = adapter._media_entries(_PNG_DATA_URL, request)
        self.assertEqual(media[0]["type"], "first_frame")
        self.assertEqual([item["type"] for item in media[1:]], ["reference_image", "reference_image"])
        self.assertNotIn("weight", media[0])

    def test_wanx_i2v_does_not_claim_multi_reference(self) -> None:
        adapter = DashscopeWanxVideoAdapter(
            EndpointConfig(protocol="dashscope-wanx", api_key="k", model="wan2.7-i2v")
        )
        request = VideoRequest(
            prompt="p",
            reference_image=_PNG_DATA_URL,
            reference_assets=[ReferenceAsset(url=_PNG_DATA_URL, type="character_three_view")],
        )
        media = adapter._media_entries(_PNG_DATA_URL, request)
        self.assertEqual(media, [{"type": "first_frame", "url": _PNG_DATA_URL}])


class GenerationMetadataTests(unittest.TestCase):
    def test_image_metadata_reports_actual_payload(self) -> None:
        captured = {}

        class Adapter:
            capabilities = ImageCapabilities(reference_images=True, max_reference_images=2)

            def __init__(self, endpoint):
                self.endpoint = endpoint

            async def generate(self, request):
                captured["request"] = request
                return b"\x89PNG\r\n\x1a\n" + b"0" * 2048

            def usage_for_request(self, *args, **kwargs):
                from services.providers.usage import unknown_usage

                return unknown_usage("image", "test", "m")

        service = ImageService()
        with (
            patch("services.image_service.get_endpoint", return_value=EndpointConfig(protocol="ark-seedream", api_key="k", model="m")),
            patch("services.image_service.get_adapter", return_value=Adapter),
        ):
            asyncio.run(
                service._generate(
                    prompt="p",
                    negative_prompt="n",
                    seed=1,
                    reference_images=[],
                    preferred_size="1024x1024",
                    label="T",
                    reference_assets=[
                        ReferenceAsset(url="https://cdn.test/a.png", type="character_three_view", role="identity"),
                        ReferenceAsset(url="https://cdn.test/b.png", type="scene_baseline", role="environment"),
                    ],
                )
            )
        metadata = service.last_generation_metadata
        self.assertEqual(metadata["references_validated"], 2)
        self.assertEqual(metadata["references_sent"], 2)
        self.assertEqual(metadata["control_types_sent"], ["reference_images"])
        self.assertEqual(metadata["reference_weight_policy"], "text_only_policy")
        self.assertEqual(len(captured["request"].reference_images), 2)

    def test_payload_metrics_measure_reference_and_control_coverage(self) -> None:
        report = payload_metrics(
            references_validated=3,
            references_sent=[
                {"type": "character_three_view", "role": "identity"},
                {"type": "scene_baseline", "role": "environment"},
            ],
            required_roles=["identity", "environment", "continuity"],
            control_types_sent=["reference_images"],
        )
        self.assertAlmostEqual(report["reference_coverage"]["ratio"], 2 / 3, places=4)
        self.assertAlmostEqual(report["role_coverage"]["ratio"], 2 / 3, places=4)
        self.assertEqual(report["control_types_sent"], ["reference_images"])


class ProviderSelectionTests(unittest.TestCase):
    def test_auto_route_prefers_configured_multi_reference_video_provider(self) -> None:
        service = VideoService()
        primary = EndpointConfig(protocol="ark-seedance", api_key="ark", model="doubao-seedance-2-0")
        alternate = EndpointConfig(protocol="dashscope-wanx", api_key="dash", model="wan2.7-r2v")
        with (
            patch("services.video_service.video_protocol_defaults", return_value=alternate),
            patch(
                "services.video_service.get_adapter",
                side_effect=lambda _capability, protocol: (
                    DashscopeWanxVideoAdapter if protocol == "dashscope-wanx" else ArkSeedanceVideoAdapter
                ),
            ),
        ):
            selected = service._preferred_video_endpoint(primary)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.protocol, "dashscope-wanx")
        self.assertEqual(selected.model, "wan2.7-r2v")

    def test_video_downgrade_requires_confirmation_even_in_manual_mode(self) -> None:
        from PIL import Image

        from services.providers.capability_matrix import CapabilityDowngradeRequiredError

        root = TEST_ROOT / "output" / "video_downgrade"
        root.mkdir(parents=True, exist_ok=True)
        first = root / "first.png"
        extra = root / "character.png"
        Image.new("RGB", (32, 32), "white").save(first)
        Image.new("RGB", (32, 32), "gray").save(extra)
        service = VideoService()
        endpoint = EndpointConfig(protocol="ark-seedance", api_key="k", model="doubao-seedance-2-0")
        shot = {
            "shot_id": "s1",
            "storyboard_path": str(first),
            "reference_assets": [{"type": "character_three_view", "role": "identity", "path": str(extra)}],
            "duration": 3,
        }
        with (
            patch("services.video_service.get_endpoint", return_value=endpoint),
            patch("services.video_service.get_adapter", return_value=ArkSeedanceVideoAdapter),
        ):
            with self.assertRaises(CapabilityDowngradeRequiredError):
                asyncio.run(service.generate_shot_video(shot, [], {}, "tests", capability_mode="manual"))


if __name__ == "__main__":
    unittest.main()
