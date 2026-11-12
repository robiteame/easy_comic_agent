"""Native request and asynchronous lifecycle contracts; no live API calls."""

from __future__ import annotations

from tests.support.test_environment import TEST_ROOT  # noqa: F401

# isort: split

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image

from services.providers.base import ReferenceAsset, VideoRequest
from services.providers.endpoint import EndpointConfig
from services.providers.video_google_veo import GoogleVeoVideoAdapter
from services.providers.video_openai_sora import OpenAISoraVideoAdapter
from services.providers.video_runway import RunwayVideoAdapter
from services.security import UploadLimitExceeded

_REAL_CLIENT = httpx.AsyncClient


def _image() -> str:
    output = io.BytesIO()
    Image.new("RGB", (80, 120), "red").save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _endpoint(protocol: str, model: str, **params) -> EndpointConfig:
    return EndpointConfig(protocol=protocol, model=model, api_key="test-only", params=params)


class NativeVideoPayloadTests(unittest.TestCase):
    def test_sora_string_duration_and_pixel_size(self):
        adapter = OpenAISoraVideoAdapter(_endpoint("openai-sora", "sora-2"))
        payload = adapter.build_payload(VideoRequest(prompt="test", duration=8, ratio="16:9"))
        self.assertEqual(payload, {"model": "sora-2", "prompt": "test", "seconds": "8", "size": "1280x720"})
        for duration in (5, 6, 21):
            with self.assertRaises(ValueError):
                adapter.build_payload(VideoRequest(prompt="test", duration=duration))
        with self.assertRaises(ValueError):
            adapter.build_payload(VideoRequest(prompt="test", duration=8, end_frame=_image()))

    def test_sora_reference_matches_output_dimensions(self):
        adapter = OpenAISoraVideoAdapter(_endpoint("openai-sora", "sora-2-pro"))
        raw = adapter._reference_file(_image(), "1024x1792")
        with Image.open(io.BytesIO(raw)) as image:
            self.assertEqual(image.size, (1024, 1792))
            self.assertEqual(image.mode, "RGB")

    def test_veo_nested_fields_and_reference_limit(self):
        adapter = GoogleVeoVideoAdapter(_endpoint("google-veo", "veo-3.1-generate-001", project_id="project"))
        payload = adapter.build_payload(
            VideoRequest(
                prompt="test",
                duration=6,
                reference_image=_image(),
                end_frame=_image(),
                negative_prompt="blur",
                generate_audio=True,
                reference_assets=[ReferenceAsset(url=_image(), role="style")],
            )
        )
        self.assertEqual(payload["parameters"]["durationSeconds"], 6)
        self.assertTrue(payload["parameters"]["generateAudio"])
        instance = payload["instances"][0]
        self.assertEqual(instance["negativePrompt"], "blur")
        self.assertEqual(instance["image"]["mimeType"], "image/png")
        self.assertIn("lastFrame", instance)
        self.assertEqual(instance["referenceImages"][0]["referenceType"], "style")
        with self.assertRaises(ValueError):
            adapter.build_payload(VideoRequest(prompt="test", duration=5))
        with self.assertRaises(ValueError):
            adapter.build_payload(
                VideoRequest(prompt="test", duration=8, reference_assets=[ReferenceAsset(url=_image())] * 4)
            )

    def test_veo_resource_url_and_unknown_google_line(self):
        adapter = GoogleVeoVideoAdapter(
            _endpoint("google-veo", "veo-3.1-generate-001", project_id="project", location="us-central1")
        )
        self.assertEqual(
            adapter._model_url(),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/project/locations/us-central1/publishers/google/models/veo-3.1-generate-001",
        )
        adapter.endpoint.model = "gemini-omni-1.1-flash"
        with self.assertRaisesRegex(ValueError, "generateContent"):
            adapter.build_payload(VideoRequest(prompt="test", duration=8))
        adapter.endpoint.model = "veo-3.1-generate-preview"
        with self.assertRaisesRegex(ValueError, "停用"):
            adapter.build_payload(VideoRequest(prompt="test", duration=8))

    def test_runway_pixels_and_turbo_require_image(self):
        adapter = RunwayVideoAdapter(_endpoint("runway", "gen4.5"))
        payload = adapter.build_payload(VideoRequest(prompt="move", duration=7, ratio="4:3", seed=42))
        self.assertEqual(payload["ratio"], "1104:832")
        self.assertEqual(payload["promptText"], "move")
        self.assertNotIn("prompt", payload)
        self.assertEqual(adapter._headers()["X-Runway-Version"], "2024-11-06")
        adapter.endpoint.model = "gen4_turbo"
        with self.assertRaisesRegex(ValueError, "必须提供首帧"):
            adapter.build_payload(VideoRequest(prompt="move", duration=5))
        with self.assertRaises(ValueError):
            adapter.build_payload(
                VideoRequest(prompt="move", duration=7, reference_image="https://example.com/image.png")
            )

    def test_api_url_accepts_bare_host_and_version_root(self):
        for base in ("https://gateway.example", "https://gateway.example/v1"):
            adapter = OpenAISoraVideoAdapter(EndpointConfig(base_url=base))
            self.assertEqual(
                adapter._api_url("https://api.openai.com", "v1/videos"), "https://gateway.example/v1/videos"
            )


class NativeVideoLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def _request(self, **changes) -> VideoRequest:
        return VideoRequest(
            prompt="move",
            duration=8,
            project_id="protocol-test",
            output_video_path=self.directory / "clip.mp4",
            output_frame_path=self.directory / "frame.png",
            **changes,
        )

    def _client(self, handler):
        transport = httpx.MockTransport(handler)
        return patch("httpx.AsyncClient", side_effect=lambda *a, **kw: _REAL_CLIENT(*a, transport=transport, **kw))

    async def _frame(self, video, frame):
        self.assertTrue(video.exists())
        frame.write_bytes(b"f" * 2048)

    async def test_sora_authenticated_content_lifecycle(self):
        calls = []

        def handler(request):
            calls.append(request)
            self.assertEqual(request.headers["authorization"], "Bearer test-only")
            if request.method == "POST":
                self.assertEqual(json.loads(request.content)["seconds"], "8")
                return httpx.Response(200, json={"id": "video-1"})
            if request.url.path.endswith("/content"):
                return httpx.Response(200, content=b"v" * 5000)
            return httpx.Response(200, json={"status": "completed"})

        adapter = OpenAISoraVideoAdapter(_endpoint("openai-sora", "sora-2"))
        with (
            self._client(handler),
            patch.object(adapter, "_extract_last_frame", self._frame),
            patch.object(
                adapter.storage,
                "ensure_project_capacity",
                return_value=10000,
            ),
        ):
            result = await adapter.generate(self._request())
        self.assertEqual(len(calls), 3)
        self.assertEqual(Path(result.video_path).read_bytes(), b"v" * 5000)
        self.assertFalse(result.native_audio)

    async def test_sora_inline_image_uses_multipart(self):
        def handler(request):
            self.assertIn("multipart/form-data", request.headers["content-type"])
            self.assertIn(b'name="input_reference"; filename="reference.png"', request.content)
            return httpx.Response(200, json={"id": "video-1"})

        adapter = OpenAISoraVideoAdapter(_endpoint("openai-sora", "sora-2"))
        with self._client(handler):
            self.assertEqual(await adapter._submit(self._request(reference_image=_image())), "video-1")

    async def test_sora_rejects_redirect_and_preserves_existing_output(self):
        adapter = OpenAISoraVideoAdapter(_endpoint("openai-sora", "sora-2"))
        request = self._request()
        request.output_video_path.write_bytes(b"existing")
        with (
            self._client(lambda r: httpx.Response(302, headers={"location": "https://evil.example/video"})),
            patch.object(
                adapter.storage,
                "ensure_project_capacity",
                return_value=10000,
            ),
            self.assertRaises(RuntimeError),
        ):
            await adapter._save_video(request, {}, "id")
        self.assertEqual(request.output_video_path.read_bytes(), b"existing")
        self.assertEqual(list(self.directory.glob(".*.download")), [])

    async def test_sora_content_limit_is_bounded(self):
        adapter = OpenAISoraVideoAdapter(_endpoint("openai-sora", "sora-2"))
        with (
            self._client(lambda r: httpx.Response(200, content=b"v" * 5000)),
            patch.object(
                adapter.storage,
                "ensure_project_capacity",
                return_value=1000,
            ),
            self.assertRaises(UploadLimitExceeded),
        ):
            await adapter._save_video(self._request(), {}, "id")
        self.assertFalse((self.directory / "clip.mp4").exists())

    async def test_veo_long_running_operation_and_inline_output(self):
        calls = []

        def handler(request):
            calls.append(request)
            if request.url.path.endswith(":predictLongRunning"):
                return httpx.Response(200, json={"name": "projects/p/operations/op-1"})
            self.assertTrue(request.url.path.endswith(":fetchPredictOperation"))
            self.assertEqual(json.loads(request.content), {"operationName": "projects/p/operations/op-1"})
            return httpx.Response(
                200,
                json={
                    "done": True,
                    "response": {"videos": [{"bytesBase64Encoded": base64.b64encode(b"v" * 5000).decode()}]},
                },
            )

        adapter = GoogleVeoVideoAdapter(_endpoint("google-veo", "veo-3.1-generate-001", project_id="p"))
        with (
            self._client(handler),
            patch.object(adapter, "_extract_last_frame", self._frame),
            patch.object(
                adapter.storage,
                "ensure_project_capacity",
                return_value=10000,
            ),
        ):
            result = await adapter.generate(self._request(generate_audio=True))
        self.assertEqual(len(calls), 2)
        self.assertTrue(result.native_audio)
        self.assertEqual(Path(result.video_path).stat().st_size, 5000)

    async def test_veo_gcs_output_uses_fixed_storage_host(self):
        calls = []

        def handler(request):
            calls.append(request)
            self.assertEqual(request.url.host, "storage.googleapis.com")
            self.assertEqual(request.headers["authorization"], "Bearer test-only")
            self.assertIn("folder%2Fvideo.mp4", str(request.url))
            return httpx.Response(200, content=b"v" * 5000)

        adapter = GoogleVeoVideoAdapter(_endpoint("google-veo", "veo-3.1-generate-001", project_id="p"))
        with self._client(handler), patch.object(adapter.storage, "ensure_project_capacity", return_value=10000):
            await adapter._save_video(
                self._request(), {"response": {"videos": [{"gcsUri": "gs://bucket/folder/video.mp4"}]}}, "op"
            )
        self.assertEqual(len(calls), 1)

    async def test_runway_upload_then_submit_and_poll(self):
        calls = []

        def handler(request):
            calls.append(request)
            if request.url.path == "/v1/uploads":
                self.assertEqual(json.loads(request.content)["type"], "ephemeral")
                return httpx.Response(
                    200,
                    json={
                        "uploadUrl": "https://uploads.example/image",
                        "fields": {"key": "signed"},
                        "runwayUri": "runway://uploaded-image",
                    },
                )
            if request.url.host == "uploads.example":
                self.assertNotIn("authorization", request.headers)
                self.assertIn("multipart/form-data", request.headers["content-type"])
                return httpx.Response(204)
            if request.url.path == "/v1/image_to_video":
                self.assertEqual(json.loads(request.content)["promptImage"], "runway://uploaded-image")
                self.assertEqual(request.headers["x-runway-version"], "2024-11-06")
                return httpx.Response(200, json={"id": "task-1"})
            return httpx.Response(200, json={"status": "SUCCEEDED", "output": ["https://cdn.example/video.mp4"]})

        async def download(project, url, destination, **kwargs):
            self.assertEqual(url, "https://cdn.example/video.mp4")
            destination.write_bytes(b"v" * 5000)

        adapter = RunwayVideoAdapter(_endpoint("runway", "gen4.5"))
        with (
            self._client(handler),
            patch("services.providers.video_runway.validate_remote_url", return_value=""),
            patch.object(
                adapter,
                "_extract_last_frame",
                self._frame,
            ),
            patch.object(adapter, "_download_url_to_path", download),
        ):
            result = await adapter.generate(self._request(reference_image=_image()))
        self.assertEqual(result.task_id, "task-1")
        self.assertEqual(len(calls), 4)

    async def test_asynchronous_failures_are_reported(self):
        cases = [
            (
                OpenAISoraVideoAdapter(_endpoint("openai-sora", "sora-2")),
                {"status": "failed", "error": {"code": "policy"}},
            ),
            (
                GoogleVeoVideoAdapter(_endpoint("google-veo", "veo-3.1-generate-001", project_id="p")),
                {"done": True, "error": {"code": 3}},
            ),
            (RunwayVideoAdapter(_endpoint("runway", "gen4.5")), {"status": "FAILED", "failureCode": "SAFETY"}),
        ]
        for adapter, data in cases:
            with (
                self.subTest(protocol=adapter.endpoint.protocol),
                patch.object(
                    adapter,
                    "_request_json",
                    AsyncMock(return_value=data),
                ),
                self.assertRaises(RuntimeError),
            ):
                await adapter._poll_task("task")

    async def test_polling_is_bounded(self):
        adapter = RunwayVideoAdapter(_endpoint("runway", "gen4.5", max_poll_attempts=2, poll_interval_seconds=0))
        response = AsyncMock(return_value={"status": "RUNNING"})
        with patch.object(adapter, "_request_json", response), self.assertRaises(TimeoutError):
            await adapter._poll_task("task")
        self.assertEqual(response.await_count, 2)
