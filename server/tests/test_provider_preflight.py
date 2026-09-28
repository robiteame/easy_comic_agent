"""图像/视频 Provider 预检与参考图能力声明的验收测试。

- 视频端点预检报告包含协议/模型/密钥/参考图能力/固定时长，且不含密钥值；
- 缺 model 或协议不支持首帧参考图时，任务启动被明确拒绝；
- 不支持参考图的图像 Provider 收到参考图时阻止生成（不静默丢弃）；
- references_validated 与 references_sent 分别记录「已校验」与「实际发送」。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT  # noqa: F401,E402

from services import provider_readiness  # noqa: E402
from services.image_service import ImageService  # noqa: E402
from services.providers.base import ImageCapabilities  # noqa: E402
from services.providers.endpoint import EndpointConfig  # noqa: E402


def _video_endpoint(*, api_key: str = "ark-key", model: str = "doubao-seedance-1-5-pro-251215", protocol: str = "ark-seedance") -> EndpointConfig:
    return EndpointConfig(
        protocol=protocol,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
        model=model,
    )


class VideoPreflightTests(unittest.TestCase):
    def test_preflight_report_shape_and_no_secret(self) -> None:
        endpoint = _video_endpoint()
        with patch.object(provider_readiness, "get_endpoint", return_value=endpoint):
            report = provider_readiness.video_provider_preflight()
        self.assertEqual(report["protocol"], "ark-seedance")
        self.assertEqual(report["model"], "doubao-seedance-1-5-pro-251215")
        self.assertTrue(report["api_key_configured"])
        self.assertTrue(report["adapter_registered"])
        self.assertTrue(report["reference_image"])
        self.assertEqual(report["reference_mode"], "first_frame_only")
        self.assertFalse(report["native_audio"])
        self.assertEqual(report["fixed_duration"], 5)
        self.assertEqual(report["issues"], [])
        self.assertNotIn("ark-key", str(report))

    def test_preflight_flags_missing_api_key_as_issue(self) -> None:
        endpoint = _video_endpoint(api_key="")
        with patch.object(provider_readiness, "get_endpoint", return_value=endpoint):
            report = provider_readiness.video_provider_preflight()
        self.assertFalse(report["api_key_configured"])
        self.assertTrue(any("API Key" in issue for issue in report["issues"]))

    def test_missing_model_blocks_task_start(self) -> None:
        endpoint = _video_endpoint(model="")
        with patch.object(provider_readiness, "get_endpoint", return_value=endpoint):
            missing = provider_readiness.missing_providers("shot_video", has_dialogue=False)
        self.assertTrue(any(item["capability"] == "video" for item in missing))
        self.assertIn("未配置模型名", "；".join(item.get("message", "") for item in missing))

    def test_missing_api_key_blocks_task_start(self) -> None:
        endpoint = _video_endpoint(api_key="")
        with patch.object(provider_readiness, "get_endpoint", return_value=endpoint):
            missing = provider_readiness.missing_providers("shot_video", has_dialogue=False)
        self.assertEqual([item["capability"] for item in missing], ["video"])

    def test_protocol_without_first_frame_reference_is_rejected(self) -> None:
        # native-audio 适配器声明 reference_image=False：预检必须拒绝。
        endpoint = _video_endpoint(protocol="native-audio", model="some-model")
        with patch.object(provider_readiness, "get_endpoint", return_value=endpoint):
            missing = provider_readiness.missing_providers("shot_video", has_dialogue=False)
        self.assertTrue(any("不支持首帧参考图" in item.get("message", "") for item in missing))


class _NoRefAdapter:
    capabilities = ImageCapabilities(reference_images=False, requires_credentials=True)

    async def generate(self, request):
        raise AssertionError("不应真的调用生成")

    def usage_for_request(self, *args, **kwargs):
        raise AssertionError("不应记账")


class _RefAdapter(_NoRefAdapter):
    capabilities = ImageCapabilities(reference_images=True, requires_credentials=True)


class ImageReferenceGuardTests(unittest.TestCase):
    def test_provider_without_reference_support_blocks_generation(self) -> None:
        service = ImageService()
        endpoint = EndpointConfig(protocol="qwen-image", base_url="https://x.test", api_key="sk", model="qwen-image-2.0")
        with (
            patch("services.image_service.get_endpoint", return_value=endpoint),
            patch("services.image_service.get_adapter", return_value=_NoRefAdapter),
        ):
            with self.assertRaisesRegex(RuntimeError, "不支持参考图"):
                import asyncio

                asyncio.run(
                    service._generate(
                        prompt="p",
                        negative_prompt="n",
                        seed=1,
                        reference_images=["data:image/png;base64,AAAA"],
                        preferred_size="1024x1024",
                        label="TEST",
                    )
                )

    def test_metadata_separates_validated_and_sent(self) -> None:
        import asyncio

        from services.providers.base import ImageRequest

        service = ImageService()
        endpoint = EndpointConfig(protocol="ark-seedream", base_url="https://x.test", api_key="sk", model="m")
        sent: list[ImageRequest] = []

        class _Adapter:
            capabilities = ImageCapabilities(reference_images=True, requires_credentials=True)

            async def generate(self, request):
                sent.append(request)
                return b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048

            def usage_for_request(self, *args, **kwargs):
                from services.providers.usage import unknown_usage

                return unknown_usage("image", "ark-seedream", "m")

        with (
            patch("services.image_service.get_endpoint", return_value=endpoint),
            patch("services.image_service.get_adapter", return_value=_Adapter),
            patch.object(service, "_validate_image", lambda data: None),
            patch.object(service, "_write_image", lambda *args, **kwargs: None),
        ):
            asyncio.run(
                service._generate(
                    prompt="p",
                    negative_prompt="n",
                    seed=1,
                    reference_images=["https://cdn.test/a.png", "https://cdn.test/b.png"],
                    preferred_size="1024x1024",
                    label="TEST",
                )
            )
        metadata = service.last_generation_metadata
        self.assertEqual(metadata["references_validated"], 2)
        self.assertEqual(metadata["references_sent"], 2)
        self.assertEqual(len(sent[0].reference_images), 2)
        self.assertEqual(metadata["reference_mode"], "multi_reference")


class _RoutingRefAdapter:
    """支持参考图的替身适配器：记录每次请求实际发送了几张参考图。"""

    capabilities = ImageCapabilities(reference_images=True, requires_credentials=True)

    def __init__(self, endpoint=None):
        self.endpoint = endpoint
        self.sent: list = []

    async def generate(self, request):
        self.sent.append(request)
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048

    def usage_for_request(self, *args, **kwargs):
        from services.providers.usage import unknown_usage

        return unknown_usage("image", "ark-seedream", "m")


class _RoutingNoRefAdapter(_RoutingRefAdapter):
    capabilities = ImageCapabilities(reference_images=False, requires_credentials=True)


class ReferenceProviderPreferenceTests(unittest.TestCase):
    """需要参考图时优先用「已配置且声明支持参考图」的图像 Provider。"""

    def _primary(self) -> EndpointConfig:
        return EndpointConfig(protocol="qwen-image", base_url="https://dashscope.test", api_key="sk", model="qwen-image-2.0")

    def _alternate(self) -> EndpointConfig:
        return EndpointConfig(protocol="ark-seedream", base_url="https://ark.test", api_key="ark-key", model="doubao-seedream-5.0-lite")

    def _shot(self) -> dict:
        return {
            "shot_id": "s1",
            "reference_assets": [],
            "scene_reference_images": ["https://cdn.test/a.png"],
            "scene_description": "room",
            "character_action": "turns",
            "output_format": "9:16",
        }

    def test_switches_to_configured_reference_capable_provider(self) -> None:
        import asyncio

        service = ImageService()

        def _adapter_for(_capability, protocol):
            return _RoutingRefAdapter if protocol == "ark-seedream" else _RoutingNoRefAdapter

        with (
            patch("services.image_service.get_endpoint", return_value=self._primary()),
            patch("services.image_service.get_adapter", side_effect=_adapter_for),
            patch("services.image_service.image_protocol_defaults", return_value=self._alternate()),
        ):
            asyncio.run(
                service._generate(
                    prompt="p",
                    negative_prompt="n",
                    seed=1,
                    reference_images=["https://cdn.test/a.png", "https://cdn.test/b.png"],
                    preferred_size="1024x1024",
                    label="TEST",
                )
            )
        metadata = service.last_generation_metadata
        self.assertEqual(metadata["provider"], "ark-seedream")
        self.assertEqual(metadata["provider_source"], "preferred_reference_provider")
        self.assertEqual(metadata["reference_mode"], "multi_reference")
        self.assertEqual(metadata["references_validated"], 2)
        self.assertEqual(metadata["references_sent"], 2)
        self.assertFalse(metadata["references_unsupported"])

    def test_prefer_policy_warns_and_never_pretends_references_were_sent(self) -> None:
        import asyncio

        service = ImageService()

        def _adapter_for(_capability, _protocol):
            return _RoutingNoRefAdapter

        with (
            patch("services.image_service.get_endpoint", return_value=self._primary()),
            patch("services.image_service.get_adapter", side_effect=_adapter_for),
            patch("services.image_service.image_protocol_defaults", return_value=self._alternate()),
            patch.object(ImageService, "_reference_enforcement", lambda _self: "prefer"),
            patch.object(service, "_validate_image", lambda data: None),
            patch.object(service, "_write_image", lambda *args, **kwargs: None),
        ):
            asyncio.run(
                service.generate_shot_image(
                    shot=self._shot(),
                    characters=[],
                    style_params={},
                    project_id="ref_pref_tests",
                )
            )
        metadata = service.last_generation_metadata
        self.assertEqual(metadata["references_validated"], 1)
        self.assertEqual(metadata["references_sent"], 0)
        self.assertEqual(metadata["reference_mode"], "text_only")
        self.assertTrue(metadata["references_unsupported"])
        self.assertIn("不支持参考图", metadata["reference_capability_warning"])

    def test_strict_policy_blocks_generation(self) -> None:
        import asyncio

        service = ImageService()

        def _adapter_for(_capability, _protocol):
            return _RoutingNoRefAdapter

        with (
            patch("services.image_service.get_endpoint", return_value=self._primary()),
            patch("services.image_service.get_adapter", side_effect=_adapter_for),
            patch("services.image_service.image_protocol_defaults", return_value=self._alternate()),
            patch.object(ImageService, "_reference_enforcement", lambda _self: "strict"),
        ):
            with self.assertRaisesRegex(RuntimeError, "不支持参考图"):
                asyncio.run(
                    service.generate_shot_image(
                        shot=self._shot(),
                        characters=[],
                        style_params={},
                        project_id="ref_pref_tests",
                    )
                )


if __name__ == "__main__":
    unittest.main()
