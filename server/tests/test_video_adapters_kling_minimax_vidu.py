"""Offline native protocol checks for Kling, MiniMax H3 and Vidu Q3."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from services.providers.base import ReferenceAsset, VideoRequest
from services.providers.endpoint import EndpointConfig
from services.providers.video_kling import KlingVideoAdapter
from services.providers.video_minimax import MiniMaxVideoAdapter
from services.providers.video_vidu import ViduVideoAdapter
from tests.support.test_environment import TEST_ROOT  # noqa: F401


class NativeVideoPayloadTests(unittest.TestCase):
    def test_kling_jwt_is_signed_and_expiring(self):
        adapter = KlingVideoAdapter(EndpointConfig(protocol="kling", api_key="access:secret", model="kling-v3"))
        with patch("services.providers.video_kling.time.time", return_value=1000):
            token = adapter._headers()["Authorization"].removeprefix("Bearer ")
        header, claims, signature = token.split(".")
        payload = json.loads(base64.urlsafe_b64decode(claims + "=" * (-len(claims) % 4)))
        self.assertEqual(payload, {"iss": "access", "exp": 2800, "nbf": 995})
        expected = hmac.new(b"secret", f"{header}.{claims}".encode(), hashlib.sha256).digest()
        self.assertEqual(base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4)), expected)

    def test_kling_first_last_and_duration_type(self):
        adapter = KlingVideoAdapter(EndpointConfig(protocol="kling", api_key="a:b", model="kling-v3"))
        payload = adapter.build_payload(
            VideoRequest(
                prompt="a smile",
                reference_image="https://cdn.test/first.png",
                end_frame="data:image/png;base64,aW1hZ2U=",
                duration=9,
                generate_audio=True,
            )
        )
        self.assertEqual(payload["duration"], "9")
        self.assertEqual(payload["image_tail"], "aW1hZ2U=")
        self.assertEqual(payload["sound"], "on")
        self.assertNotIn("seed", payload)
        self.assertNotIn("ratio", payload)

    def test_kling_version_limits_and_tail_validation(self):
        adapter = KlingVideoAdapter(EndpointConfig(protocol="kling", model="kling-v3-turbo"))
        with self.assertRaisesRegex(ValueError, "尾帧"):
            adapter.build_payload(
                VideoRequest(
                    prompt="x", reference_image="https://cdn.test/x.png", end_frame="https://cdn.test/y.png", duration=5
                )
            )
        adapter = KlingVideoAdapter(EndpointConfig(protocol="kling", model="kling-v2-6"))
        with self.assertRaisesRegex(ValueError, "时长"):
            adapter.build_payload(VideoRequest(prompt="x", duration=6))
        with self.assertRaisesRegex(ValueError, "AccessKey"):
            adapter._headers()

    def test_minimax_h3_content_roles_and_adaptive_ratio(self):
        adapter = MiniMaxVideoAdapter(EndpointConfig(protocol="minimax", model="MiniMax-H3"))
        request = VideoRequest(
            prompt="grow older",
            reference_image="https://cdn.test/first.png",
            end_frame="https://cdn.test/last.png",
            duration=8,
            resolution="1080p",
            reference_video="https://cdn.test/ref.mp4",
            reference_audio="https://cdn.test/ref.mp3",
        )
        payload = adapter.build_payload(request)
        self.assertEqual(
            [item.get("role") for item in payload["content"]],
            [None, "first_frame", "last_frame", "reference_video", "reference_audio"],
        )
        self.assertEqual(payload["ratio"], "adaptive")
        self.assertEqual(payload["resolution"], "2K")
        self.assertTrue(adapter._native_audio(request))
        self.assertNotIn("generate_audio", payload)

    def test_minimax_caps_and_public_url_constraint(self):
        adapter = MiniMaxVideoAdapter(EndpointConfig(protocol="minimax", model="MiniMax-H3-Max"))
        with self.assertRaisesRegex(ValueError, "小于"):
            adapter.build_payload(VideoRequest(prompt="x", duration=4))
        with self.assertRaisesRegex(ValueError, "分辨率"):
            adapter.build_payload(VideoRequest(prompt="x", duration=5, resolution="2k"))
        with self.assertRaisesRegex(ValueError, "HTTP"):
            adapter.build_payload(VideoRequest(prompt="x", duration=5, reference_image="data:image/png;base64,aQ=="))
        refs = [ReferenceAsset(url=f"https://cdn.test/{i}.png") for i in range(10)]
        with self.assertRaisesRegex(ValueError, "最多"):
            MiniMaxVideoAdapter(EndpointConfig(protocol="minimax", model="MiniMax-H3")).build_payload(
                VideoRequest(prompt="x", reference_assets=refs)
            )

    def test_vidu_endpoint_switching_and_no_ratio(self):
        adapter = ViduVideoAdapter(EndpointConfig(protocol="vidu", api_key="test", model="viduq3-pro"))
        request = VideoRequest(prompt="wave", reference_image="https://cdn.test/first.png", duration=1)
        self.assertEqual(adapter._task_kind(request), "img2video")
        self.assertEqual(adapter._headers()["Authorization"], "Token test")
        request.end_frame = "https://cdn.test/last.png"
        self.assertEqual(adapter._task_kind(request), "start-end2video")
        payload = adapter.build_payload(request)
        self.assertEqual(len(payload["images"]), 2)
        self.assertNotIn("ratio", payload)
        self.assertNotIn("aspect_ratio", payload)
        request.end_frame = None
        request.reference_assets = [ReferenceAsset(url="https://cdn.test/ref.png", type="character")]
        self.assertEqual(adapter._task_kind(request), "reference2video")
        self.assertEqual(len(adapter.build_payload(request)["images"]), 2)

    def test_vidu_audio_off_peak_is_rejected(self):
        adapter = ViduVideoAdapter(EndpointConfig(protocol="vidu", model="viduq3-pro", params={"off_peak": True}))
        with self.assertRaisesRegex(ValueError, "off_peak"):
            adapter.build_payload(VideoRequest(prompt="x", generate_audio=True))


class NativeVideoTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_protocol_submission_and_completed_poll(self):
        cases = [
            (
                KlingVideoAdapter,
                "kling",
                "kling-v3",
                "key:secret",
                {"code": 0, "data": {"task_id": "job"}},
                {
                    "code": 0,
                    "data": {
                        "task_status": "succeed",
                        "task_result": {"videos": [{"url": "https://cdn.test/video.mp4"}]},
                    },
                },
                "/v1/videos/image2video",
                "/v1/videos/image2video/job",
            ),
            (
                MiniMaxVideoAdapter,
                "minimax",
                "MiniMax-H3",
                "key",
                {"task_id": "job", "base_resp": {"status_code": 0}},
                {"status": "Success", "content": {"url": "https://cdn.test/video.mp4"}},
                "/v2/video_generation",
                "/v2/query/video_generation/job",
            ),
            (
                ViduVideoAdapter,
                "vidu",
                "viduq3-pro",
                "key",
                {"task_id": "job"},
                {"state": "success", "creations": [{"url": "https://cdn.test/video.mp4"}]},
                "/ent/v2/img2video",
                "/ent/v2/tasks/job/creations",
            ),
        ]
        for cls, protocol, model, key, created, completed, submit_suffix, poll_suffix in cases:
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as directory:
                requests = []

                def handler(request, requests=requests, created=created, completed=completed):
                    requests.append(request)
                    return httpx.Response(200, json=created if request.method == "POST" else completed)

                client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
                adapter = cls(EndpointConfig(protocol=protocol, base_url="https://api.test", api_key=key, model=model))
                request = VideoRequest(
                    prompt="wave",
                    reference_image="https://cdn.test/first.png",
                    output_video_path=Path(directory) / "out.mp4",
                    output_frame_path=Path(directory) / "out.png",
                )

                async def save_video(project_id, url, destination, *, minimum_size):
                    self.assertEqual(url, "https://cdn.test/video.mp4")
                    destination.write_bytes(b"v" * 5000)

                async def extract(video, frame):
                    frame.write_bytes(b"f" * 2000)

                # _request_json opens one client per call; use a reusable wrapper
                # whose exit does not close the underlying MockTransport client.
                class ClientContext:
                    def __init__(self, client):
                        self.client = client

                    async def __aenter__(self):
                        return self.client

                    async def __aexit__(self, *args):
                        return False

                with (
                    patch("services.providers.video_http.httpx.AsyncClient", return_value=ClientContext(client)),
                    patch.object(adapter, "_download_url_to_path", side_effect=save_video),
                    patch.object(adapter, "_extract_last_frame", side_effect=extract),
                ):
                    result = await adapter.generate(request)
                await client.aclose()
                self.assertEqual(result.task_id, "job")
                self.assertEqual(requests[0].url.path, submit_suffix)
                self.assertEqual(requests[1].url.path, poll_suffix)
                self.assertEqual(result.native_audio, protocol == "minimax")
                if protocol == "vidu":
                    self.assertEqual(requests[0].headers["authorization"], "Token key")

    async def test_failed_tasks_and_application_errors_surface(self):
        cases = [
            (
                KlingVideoAdapter(EndpointConfig(protocol="kling", api_key="a:b", model="kling-v3")),
                {"code": 0, "data": {"task_status": "failed", "task_status_msg": "moderation"}},
            ),
            (MiniMaxVideoAdapter(EndpointConfig(protocol="minimax", model="MiniMax-H3")), {"status": "Fail"}),
            (
                ViduVideoAdapter(EndpointConfig(protocol="vidu", model="viduq3-pro")),
                {"state": "failed", "err_code": "invalid_input"},
            ),
        ]
        for adapter, payload in cases:
            with (
                self.subTest(provider=adapter.endpoint.protocol),
                patch.object(adapter, "_request_json", new=AsyncMock(return_value=payload)),
            ):
                with self.assertRaisesRegex(RuntimeError, "失败|错误"):
                    await adapter._poll_task("job")

    async def test_minimax_application_error_without_http_failure(self):
        adapter = MiniMaxVideoAdapter(EndpointConfig(protocol="minimax", model="MiniMax-H3"))
        with patch.object(
            adapter,
            "_request_json",
            new=AsyncMock(return_value={"base_resp": {"status_code": 1004, "status_msg": "invalid key"}}),
        ):
            with self.assertRaisesRegex(RuntimeError, "1004"):
                await adapter._submit(VideoRequest(prompt="x"))

    async def test_timeout_and_missing_task_id(self):
        adapter = ViduVideoAdapter(EndpointConfig(protocol="vidu", model="viduq3-pro", params={"max_poll_attempts": 1}))
        with patch.object(adapter, "_request_json", new=AsyncMock(return_value={"state": "processing"})):
            with self.assertRaises(TimeoutError):
                await adapter._poll_task("job")
        with patch.object(adapter, "_request_json", new=AsyncMock(return_value={})):
            with self.assertRaisesRegex(RuntimeError, "task_id"):
                await adapter._submit(VideoRequest(prompt="x"))
