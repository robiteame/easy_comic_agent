"""Regression checks for native protocol selection and model duration planning."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from config import settings
from services.providers.base import ReferenceAsset, VideoRequest
from services.providers.endpoint import (
    VIDEO_PROTOCOL_PRESETS,
    EndpointConfig,
    endpoint_from_stored,
    normalize_protocol,
    video_protocol_defaults,
)
from services.providers.registry import get_adapter, protocols_for
from services.story_timing import (
    ShotExecutionPlan,
    StoryTimingError,
    StoryTimingPlan,
    provider_duration_capability,
)
from tests.support.test_environment import TEST_ROOT  # noqa: F401


class NativeProtocolDefaultsTests(unittest.TestCase):
    def test_new_native_adapters_are_registered(self):
        protocols = protocols_for("video")
        for protocol in VIDEO_PROTOCOL_PRESETS:
            with self.subTest(protocol=protocol):
                self.assertIn(protocol, protocols)
                self.assertTrue(get_adapter("video", protocol).production_ready)

    def test_protocol_switch_does_not_inherit_another_vendor_credentials(self):
        with (
            patch.object(settings, "VIDEO_PROVIDER", "ark-seedance"),
            patch.object(settings, "SEEDDANCE_API_KEY", "ark-only-test-key"),
            patch.object(settings, "ARK_API_KEY", "ark-only-fallback-key"),
            patch.object(settings, "SEEDDANCE_BASE_URL", "https://ark.test/api/v3"),
            patch.object(settings, "SEEDDANCE_MODEL", "doubao-seedance-1-0-pro"),
        ):
            for protocol, preset in VIDEO_PROTOCOL_PRESETS.items():
                with self.subTest(protocol=protocol):
                    endpoint = endpoint_from_stored("video", {"protocol": protocol})
                    self.assertEqual(endpoint.protocol, protocol)
                    self.assertEqual(endpoint.api_key, "")
                    self.assertEqual(endpoint.base_url, preset["base_url"])
                    self.assertEqual(endpoint.model, preset["model"])
                    self.assertNotIn("audio_mode", endpoint.params)

    def test_explicit_override_reuses_matching_configured_endpoint(self):
        configured = EndpointConfig(
            protocol="kling",
            base_url="https://gateway.test/v1",
            api_key="a:b",
            model="custom-kling-model",
            params={"mode": "pro"},
        )
        with patch("services.providers.endpoint.get_endpoint", return_value=configured):
            self.assertIs(video_protocol_defaults("kling"), configured)
            other = video_protocol_defaults("vidu")
        self.assertEqual(other.protocol, "vidu")
        self.assertEqual(other.api_key, "")
        self.assertEqual(other.model, "viduq3-pro")

    def test_legacy_model_names_select_native_contract(self):
        self.assertEqual(normalize_protocol("video", "sora2"), "openai-sora")
        self.assertEqual(normalize_protocol("video", "sora-2-pro"), "openai-sora")
        self.assertEqual(normalize_protocol("video", "veo3"), "google-veo")
        self.assertEqual(normalize_protocol("video", "veo-3.1-generate-001"), "google-veo")


class NativeDurationIntegrationTests(unittest.TestCase):
    def _capability(self, protocol: str, model: str):
        return provider_duration_capability(
            protocol, capabilities=get_adapter("video", protocol).effective_capabilities(model), model=model
        )

    def test_discrete_durations_reach_the_timing_contract(self):
        cases = [
            ("openai-sora", "sora-2", (4, 8, 12, 16, 20), 6),
            ("google-veo", "veo-3.1-generate-001", (4, 6, 8), 5),
            ("luma", "ray-2", (5, 9), 6),
            ("runway", "gen4_turbo", (5, 10), 6),
            ("kling", "kling-v2-6", (5, 10), 6),
        ]
        for protocol, model, expected, invalid in cases:
            with self.subTest(protocol=protocol):
                capability = self._capability(protocol, model)
                self.assertEqual(capability.duration_options, expected)
                for value in expected:
                    capability.validate(value)
                with self.assertRaises(StoryTimingError):
                    capability.validate(invalid)

    def test_execution_plan_rounds_up_to_next_model_option(self):
        cases = [("openai-sora", "sora-2", 5, 8), ("google-veo", "veo-3.1-generate-001", 5, 6), ("luma", "ray-2", 6, 9)]
        for protocol, model, narrative, generated in cases:
            with self.subTest(protocol=protocol):
                capability = self._capability(protocol, model)
                plan = ShotExecutionPlan.derive({"shot_id": "native-enum", "duration": narrative}, provider=capability)
                self.assertEqual(plan.provider_generation_duration_s, generated)
                self.assertEqual(plan.narrative_duration_ms, narrative * 1000)
                capability.validate(plan.provider_generation_duration_s)

    def test_story_budget_uses_legal_discrete_combinations(self):
        cases = [("openai-sora", "sora-2", 20), ("google-veo", "veo-3.1-generate-001", 14), ("luma", "ray-2", 14)]
        for protocol, model, target in cases:
            with self.subTest(protocol=protocol):
                capability = self._capability(protocol, model)
                plan = StoryTimingPlan(target_duration_s=target, provider=capability)
                shots = plan.rebalance(
                    [
                        {
                            "shot_id": "one",
                            "scene_number": 1,
                            "duration": target / 2,
                            "dialogue": [],
                            "character_action": "",
                        },
                        {
                            "shot_id": "two",
                            "scene_number": 2,
                            "duration": target / 2,
                            "dialogue": [],
                            "character_action": "",
                        },
                    ]
                )
                self.assertEqual(sum(shot["duration"] for shot in shots), target)
                for shot in shots:
                    capability.validate(shot["duration"])
                self.assertEqual(plan.validate_shots(shots), [])

    def test_model_specific_continuous_ranges_remain_continuous(self):
        cases = [
            ("runway", "gen4.5", 2, 10),
            ("vidu", "viduq3-pro", 1, 16),
            ("kling", "kling-v3", 3, 15),
            ("minimax", "MiniMax-H3", 4, 15),
            ("minimax", "MiniMax-H3-Max", 5, 15),
        ]
        for protocol, model, low, high in cases:
            with self.subTest(protocol=protocol, model=model):
                capability = self._capability(protocol, model)
                self.assertEqual(capability.duration_options, ())
                for value in range(low, high + 1):
                    capability.validate(value)

    def test_common_adapter_rejects_unsupported_assets_before_network(self):
        adapter = get_adapter("video", "runway")(EndpointConfig(protocol="runway", model="gen4.5"))
        with self.assertRaisesRegex(ValueError, "参考视频"):
            adapter.validate_request(
                VideoRequest(
                    prompt="p",
                    reference_assets=[ReferenceAsset(url="https://cdn.test/ref.mp4", type="reference_video")],
                )
            )
        with self.assertRaisesRegex(ValueError, "多参考图"):
            adapter.validate_request(
                VideoRequest(
                    prompt="p",
                    reference_assets=[ReferenceAsset(url="https://cdn.test/ref.png", type="character_three_view")],
                )
            )


class PublicMediaMappingTests(unittest.TestCase):
    def test_configured_output_mirror_maps_local_files_with_url_encoding(self):
        import tempfile
        from pathlib import Path

        from services.video_service import VideoService

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            image = output / "projects" / "p" / "first frame 中文.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            original = ReferenceAsset(
                url="data:image/png;base64,aW1hZ2U=", source_path=str(image), type="approved_storyboard_first_frame"
            )
            endpoint = EndpointConfig(
                protocol="minimax", params={"public_media_base_url": "https://media.test/output/"}
            )
            with patch.object(settings, "OUTPUT_DIR", output):
                first, assets = VideoService._public_video_references(endpoint, original.url, [original])
            self.assertEqual(first, "https://media.test/output/projects/p/first%20frame%20%E4%B8%AD%E6%96%87.png")
            self.assertEqual(assets[0].url, first)
            self.assertTrue(original.url.startswith("data:"))
            self.assertEqual(assets[0].source_path, str(image))

    def test_mapping_is_opt_in_and_restricts_paths_to_output(self):
        import tempfile
        from pathlib import Path

        from services.video_service import VideoService

        asset = ReferenceAsset(
            url="data:image/png;base64,aW1hZ2U=", source_path="/outside.png", type="approved_storyboard_first_frame"
        )
        first, untouched = VideoService._public_video_references(EndpointConfig(protocol="minimax"), asset.url, [asset])
        self.assertEqual(first, asset.url)
        self.assertIs(untouched[0], asset)
        with tempfile.TemporaryDirectory() as directory, patch.object(settings, "OUTPUT_DIR", Path(directory)):
            endpoint = EndpointConfig(protocol="luma", params={"public_media_base_url": "https://media.test/output"})
            with self.assertRaisesRegex(ValueError, "OUTPUT_DIR"):
                VideoService._public_video_references(endpoint, asset.url, [asset])

    def test_existing_public_url_is_preserved(self):
        from services.video_service import VideoService

        url = "https://cdn.test/first.png"
        endpoint = EndpointConfig(protocol="minimax", params={"public_media_base_url": "https://media.test/output"})
        first, assets = VideoService._public_video_references(endpoint, url, [])
        self.assertEqual(first, url)
        self.assertEqual(assets, [])


class OperationBaseNormalizationTests(unittest.TestCase):
    def test_full_runway_creation_url_can_resolve_polling_and_upload_urls(self):
        from services.providers.video_runway import RunwayVideoAdapter

        adapter = RunwayVideoAdapter(
            EndpointConfig(protocol="runway", model="gen4.5", base_url="https://gateway.test/v1/image_to_video")
        )
        self.assertEqual(
            adapter._api_url("https://api.dev.runwayml.com", "/v1/tasks/job"), "https://gateway.test/v1/tasks/job"
        )
        self.assertEqual(
            adapter._api_url("https://api.dev.runwayml.com", "/v1/uploads"), "https://gateway.test/v1/uploads"
        )

    def test_unsupported_task_types_are_rejected_before_billing(self):
        from services.providers.video_kling import KlingVideoAdapter
        from services.providers.video_minimax import MiniMaxVideoAdapter
        from services.providers.video_vidu import ViduVideoAdapter

        cases = [
            (KlingVideoAdapter, "kling", "kling-v3"),
            (ViduVideoAdapter, "vidu", "viduq3-pro"),
            (MiniMaxVideoAdapter, "minimax", "MiniMax-H3"),
        ]
        for cls, protocol, model in cases:
            with self.subTest(protocol=protocol):
                adapter = cls(EndpointConfig(protocol=protocol, model=model))
                with self.assertRaisesRegex(ValueError, "任务|续写"):
                    adapter.build_payload(VideoRequest(prompt="p", duration=5, task_type="extend"))


class NativeContentNormalizationTests(unittest.TestCase):
    def test_native_last_frame_and_multimodal_roles_are_preserved(self):
        from services.video_service import VideoService

        first, assets, _ = VideoService()._reference_from_content(
            [
                {
                    "type": "image_url",
                    "role": "reference_image",
                    "image_url": {"url": "https://cdn.test/character.png"},
                },
                {"type": "image_url", "role": "first_frame", "image_url": {"url": "https://cdn.test/first.png"}},
                {"type": "image_url", "role": "last_frame", "image_url": {"url": "https://cdn.test/last.png"}},
                {"type": "video_url", "role": "reference_video", "video_url": {"url": "https://cdn.test/clip.mp4"}},
                {"type": "audio_url", "role": "reference_audio", "audio_url": {"url": "https://cdn.test/voice.wav"}},
            ]
        )
        self.assertEqual(first, "https://cdn.test/first.png")
        self.assertEqual(
            [item.type for item in assets], ["reference_image", "end_frame", "reference_video", "reference_audio"]
        )
        self.assertEqual(assets[1].provider_type, "end_frame")
        self.assertEqual(assets[2].provider_type, "reference_video")
