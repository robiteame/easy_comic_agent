"""Offline protocol contracts from the November video API research note."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from services.providers.base import ReferenceAsset, VideoRequest
from services.providers.endpoint import EndpointConfig
from services.providers.video_ark_seedance import ArkSeedanceVideoAdapter
from services.providers.video_dashscope_wanx import DashscopeWanxVideoAdapter
from services.providers.video_luma import LumaVideoAdapter
from services.providers.video_pixverse import PixVerseVideoAdapter
from tests.support.test_environment import TEST_ROOT  # noqa: F401


class ArkDocumentTests(unittest.TestCase):
    def adapter(self, model="doubao-seedance-2-5-260628"):
        return ArkSeedanceVideoAdapter(EndpointConfig(protocol="ark-seedance", model=model))

    def test_modern_capabilities_preserve_legacy(self):
        self.assertEqual(self.adapter().effective_capabilities("doubao-seedance-2-5-260628").max_duration, 30)
        self.assertEqual(self.adapter().effective_capabilities("doubao-seedance-2-0-260128").max_reference_images, 9)
        self.assertEqual(self.adapter().effective_capabilities("doubao-seedance-1-5-pro").fixed_duration, 5)

    def test_first_last_frames_force_adaptive(self):
        payload = self.adapter().build_payload(
            VideoRequest(
                prompt="motion",
                reference_image="https://media.test/first.png",
                end_frame="https://media.test/last.png",
                duration=20,
            )
        )
        self.assertEqual(payload["ratio"], "adaptive")
        self.assertEqual([x.get("role") for x in payload["content"][1:]], ["first_frame", "last_frame"])
        self.assertFalse(payload["generate_audio"])

    def test_reference_media_and_edit(self):
        payload = self.adapter().build_payload(
            VideoRequest(
                prompt="replace the sky",
                reference_video="https://media.test/source.mp4",
                reference_audio="https://media.test/audio.mp3",
                task_type="edit",
                generate_audio=True,
            )
        )
        self.assertEqual(payload["duration"], -1)
        self.assertEqual(payload["ratio"], "adaptive")
        self.assertEqual(payload["omni_reference_task_type"], "edit")
        self.assertEqual([x["type"] for x in payload["content"]], ["text", "video_url", "audio_url"])
        self.assertTrue(payload["generate_audio"])

    def test_reference_limits_raise_before_submit(self):
        request = VideoRequest(
            prompt="motion", reference_assets=[ReferenceAsset(url=f"https://media.test/{i}.png") for i in range(31)]
        )
        with self.assertRaisesRegex(ValueError, "数量"):
            self.adapter().build_payload(request)

    def test_legacy_rejects_unsupported_tail(self):
        with self.assertRaises(ValueError):
            self.adapter("doubao-seedance-1-5-pro").build_payload(
                VideoRequest(
                    prompt="motion", reference_image="https://media.test/a.png", end_frame="https://media.test/b.png"
                )
            )


class WanDocumentTests(unittest.TestCase):
    def adapter(self):
        return DashscopeWanxVideoAdapter(EndpointConfig(protocol="dashscope-wanx", model="wan2.7-i2v-2026-04-25"))

    def test_media_combination_and_names(self):
        request = VideoRequest(
            prompt="motion",
            reference_image="https://media.test/a.png",
            end_frame="https://media.test/b.png",
            reference_audio="https://media.test/a.mp3",
            duration=13,
            seed=8,
            negative_prompt="blur",
        )
        payload = self.adapter().build_payload(request)
        self.assertEqual([m["type"] for m in payload["input"]["media"]], ["first_frame", "last_frame", "driving_audio"])
        self.assertEqual(payload["parameters"]["duration"], 13)
        self.assertEqual(payload["parameters"]["seed"], 8)
        self.assertEqual(payload["input"]["negative_prompt"], "blur")
        self.assertNotIn("ratio", payload["parameters"])

    def test_continuation_with_tail(self):
        payload = self.adapter().build_payload(
            VideoRequest(
                prompt="continue", reference_video="https://media.test/a.mp4", end_frame="https://media.test/b.png"
            )
        )
        self.assertEqual([m["type"] for m in payload["input"]["media"]], ["first_clip", "last_frame"])

    def test_invalid_combinations_fail_locally(self):
        for request in (
            VideoRequest(
                prompt="p", reference_image="https://media.test/a.png", reference_video="https://media.test/a.mp4"
            ),
            VideoRequest(
                prompt="p", reference_video="https://media.test/a.mp4", reference_audio="https://media.test/a.mp3"
            ),
            VideoRequest(prompt="p", end_frame="https://media.test/b.png"),
            VideoRequest(prompt="p", reference_image="https://media.test/a.png", duration=16),
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.adapter().build_payload(request)


class PixVerseDocumentTests(unittest.TestCase):
    def adapter(self, model="v6"):
        return PixVerseVideoAdapter(EndpointConfig(protocol="pixverse", model=model, api_key="test"))

    def test_upload_happens_before_image_generate(self):
        adapter = self.adapter()
        request = VideoRequest(prompt="p", reference_image="https://media.test/a.png", duration=7)
        mocked = AsyncMock(
            side_effect=[{"ErrCode": 0, "Resp": {"img_id": 123}}, {"ErrCode": 0, "Resp": {"video_id": 456}}]
        )
        with patch.object(adapter, "_request_json", mocked):
            task_id = asyncio.run(adapter._submit(request))
        self.assertEqual(task_id, "456")
        upload, create = mocked.call_args_list
        self.assertTrue(upload.args[1].endswith("/image/upload"))
        self.assertNotIn("Content-Type", upload.kwargs["headers"])
        self.assertEqual(create.kwargs["json"]["img_id"], 123)
        self.assertNotIn("reference_image", create.kwargs["json"])

    def test_transition_uses_verified_official_fields(self):
        payload = self.adapter().build_payload(
            VideoRequest(prompt="p", reference_image="https://media.test/a.png", end_frame="https://media.test/b.png"),
            [11, 12],
        )
        self.assertEqual(payload["first_frame_img"], 11)
        self.assertEqual(payload["last_frame_img"], 12)
        self.assertNotIn("aspect_ratio", payload)

    def test_fusion_uses_uploaded_reference_records(self):
        payload = self.adapter().build_payload(
            VideoRequest(prompt="p", reference_assets=[ReferenceAsset(url="https://media.test/a.png")]), [11]
        )
        self.assertEqual(payload["image_references"], [{"img_id": 11}])

    def test_business_errors_not_hidden_by_http200(self):
        with self.assertRaisesRegex(RuntimeError, "10004"):
            self.adapter()._response({"ErrCode": 10004, "ErrMsg": "apiKey is empty"})

    def test_version_specific_duration(self):
        with self.assertRaisesRegex(ValueError, "1080p"):
            self.adapter("v5.6").build_payload(VideoRequest(prompt="p", duration=10, resolution="1080p"))

    def test_poll_terminal_failure(self):
        adapter = self.adapter()
        with patch.object(adapter, "_request_json", AsyncMock(return_value={"ErrCode": 0, "Resp": {"status": 7}})):
            with self.assertRaisesRegex(RuntimeError, "7"):
                asyncio.run(adapter._poll_task("1"))


class LumaDocumentTests(unittest.TestCase):
    def adapter(self):
        return LumaVideoAdapter(EndpointConfig(protocol="luma", model="ray-2"))

    def test_duration_units_and_generation_keyframe(self):
        payload = self.adapter().build_payload(
            VideoRequest(
                prompt="p", duration=9, reference_video="generation:previous-id", end_frame="https://media.test/end.png"
            )
        )
        self.assertEqual(payload["duration"], "9s")
        self.assertEqual(payload["keyframes"]["frame0"], {"type": "generation", "id": "previous-id"})
        self.assertEqual(payload["keyframes"]["frame1"]["type"], "image")

    def test_duration_and_url_reference_rejected(self):
        for request in (
            VideoRequest(prompt="p", duration=6),
            VideoRequest(prompt="p", reference_video="https://media.test/a.mp4"),
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.adapter().build_payload(request)

    def test_poll_failed_state(self):
        adapter = self.adapter()
        with patch.object(
            adapter, "_request_json", AsyncMock(return_value={"state": "failed", "failure_reason": "moderated"})
        ):
            with self.assertRaisesRegex(RuntimeError, "moderated"):
                asyncio.run(adapter._poll_task("id"))


if __name__ == "__main__":
    unittest.main()
