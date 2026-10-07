"""Seedance 视频链路的关键行为验收测试。

- 固定时长约 5 秒：超出时长的镜头明确报错（要求拆分），不静默截短；
- 参考图模式为 first_frame_only，只发送已审核故事板首帧；
- 创建任务载荷保留 ``return_last_frame``，供同场景下一镜使用上一镜尾帧；
- ``native_audio=False``：对白继续走 TTS 路由；
- references_validated（已校验）与 references_sent（实际发送）分开记录；
- 缺少已审核首帧时阻止纯文本生成。
"""

from __future__ import annotations

import asyncio
import base64
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image  # noqa: E402

from services.providers.base import VideoCapabilities, VideoRequest  # noqa: E402
from services.providers.endpoint import EndpointConfig  # noqa: E402
from services.providers.video_ark_seedance import ArkSeedanceVideoAdapter  # noqa: E402
from services.video_service import VideoService  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


def _endpoint() -> EndpointConfig:
    return EndpointConfig(
        protocol="ark-seedance",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key="ark-test",
        model="doubao-seedance-1-5-pro-251215",
    )


def _storyboard_png() -> str:
    path = TEST_ROOT / "output" / "seedance_tests" / "storyboard.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (720, 1280), (60, 80, 120)).save(path)
    return str(path)


def _shot(**overrides) -> dict:
    base = {
        "shot_id": "proj_shot_0001_v1",
        "storyboard_path": _storyboard_png(),
        "image_path": "",
        "duration": 4.5,
        "output_format": "9:16",
        "resolution": "720p",
        "style": "realistic",
        "scene_description": "雨夜天台",
        "character_action": "她转身",
        "emotion": "sad",
        "camera_movement": "跟随",
        "camera_angle": "正面",
        "shot_type": "medium",
        "reference_assets": [{"type": "scene_baseline", "path": "/x.png", "role": "env"}],
        "continuity_profile": {"editing_logic": ["eye_line_continuity"]},
    }
    base.update(overrides)
    return base


class FixedDurationTests(unittest.TestCase):
    def test_capabilities_declare_fixed_duration_and_first_frame_only(self) -> None:
        capabilities = ArkSeedanceVideoAdapter.capabilities
        self.assertIsInstance(capabilities, VideoCapabilities)
        self.assertTrue(capabilities.reference_image)
        self.assertEqual(capabilities.reference_mode, "first_frame_only")
        self.assertFalse(capabilities.native_audio)
        self.assertEqual(capabilities.fixed_duration, 5)
        self.assertGreater(capabilities.max_reference_inline_bytes, 1024 * 1024)

    def test_overlong_shot_is_rejected_with_split_guidance(self) -> None:
        service = VideoService()
        endpoint = _endpoint()
        with (
            patch("services.video_service.get_endpoint", return_value=endpoint),
            patch("services.video_service.get_adapter", return_value=ArkSeedanceVideoAdapter),
        ):
            with self.assertRaisesRegex(RuntimeError, "拆分"):
                asyncio.run(service.generate_shot_video(_shot(duration=12.0), [], {}, project_id="seedance_tests"))

    def test_short_shot_uses_fixed_duration(self) -> None:
        adapter = ArkSeedanceVideoAdapter(_endpoint())
        self.assertEqual(
            adapter.usage_for_request(
                "video", VideoRequest(prompt="p", duration=3, ratio="9:16", resolution="720p")
            ).seconds,
            3.0,
        )


class FirstFrameOnlyTests(unittest.TestCase):
    def test_generate_shot_video_records_first_frame_only_metadata(self) -> None:
        service = VideoService()
        endpoint = _endpoint()
        shot = _shot()
        captured: dict = {}

        class _FakeResult:
            video_path = "/tmp/v.mp4"
            frame_path = "/tmp/f.png"
            native_audio = False
            payload_mode = "first_frame_reference"
            task_id = "task-1"

        async def fake_generate_single(prompt, **kwargs):
            captured.update(kwargs)
            captured["prompt"] = prompt
            return {
                "video_path": "/tmp/v.mp4",
                "frame_path": "/tmp/f.png",
                "task_id": "task-1",
                "reference_payload_mode": "first_frame_reference",
                "native_audio": False,
            }

        with (
            patch("services.video_service.get_endpoint", return_value=endpoint),
            patch("services.video_service.get_adapter", return_value=ArkSeedanceVideoAdapter),
            patch.object(VideoService, "generate_single_shot", side_effect=fake_generate_single),
        ):
            result = asyncio.run(service.generate_shot_video(shot, [], {}, "seedance_tests"))

        self.assertEqual(shot["reference_mode"], "first_frame_only")
        self.assertEqual(shot["references_validated"], 1)
        self.assertEqual(shot["references_sent"], ["approved_storyboard_first_frame"])
        self.assertTrue(captured["content"][1]["type"] == "image_url")
        self.assertEqual(captured["content"][1]["role"], "first_frame")
        self.assertIn("camera movement", captured["prompt"])
        self.assertIn("emotional tone", captured["prompt"])
        self.assertEqual(result["reference_payload_mode"], "first_frame_reference")

    def test_missing_storyboard_blocks_text_only_generation(self) -> None:
        service = VideoService()
        endpoint = _endpoint()
        with (
            patch("services.video_service.get_endpoint", return_value=endpoint),
            patch("services.video_service.get_adapter", return_value=ArkSeedanceVideoAdapter),
        ):
            with self.assertRaisesRegex(RuntimeError, "approved_storyboard_first_frame"):
                asyncio.run(
                    service.generate_shot_video(_shot(storyboard_path="", image_path=""), [], {}, "seedance_tests")
                )

    def test_no_silent_fallback_to_wanx(self) -> None:
        # Seedance 配置下的适配器就是 ark-seedance；不存在失败后换协议的路径。
        service = VideoService()
        endpoint = _endpoint()
        with (
            patch("services.video_service.get_endpoint", return_value=endpoint),
            patch("services.video_service.get_adapter", return_value=ArkSeedanceVideoAdapter) as get_adapter,
            patch.object(VideoService, "generate_single_shot", side_effect=RuntimeError("Seedance 创建任务失败: 401")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Seedance 创建任务失败"):
                asyncio.run(service.generate_shot_video(_shot(), [], {}, "seedance_tests"))
            self.assertIs(get_adapter.return_value, ArkSeedanceVideoAdapter)


class SeedancePayloadTests(unittest.TestCase):
    def test_create_task_payload_keeps_return_last_frame(self) -> None:
        adapter = ArkSeedanceVideoAdapter(_endpoint())
        storyboard = _storyboard_png()
        request = VideoRequest(
            prompt="雨夜天台，她转身",
            reference_image="data:image/png;base64," + base64.b64encode(Path(storyboard).read_bytes()).decode(),
            duration=5,
            ratio="9:16",
            resolution="720p",
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
                captured["url"] = url
                return _FakeResponse()

        with patch("services.providers.video_ark_seedance.httpx.AsyncClient", return_value=_FakeClient()):
            asyncio.run(adapter._create_task(request.prompt, 5, "9:16", "720p", None))

        payload = captured["json"]
        self.assertTrue(payload["return_last_frame"])
        self.assertEqual(payload["duration"], 5)
        self.assertEqual(payload["model"], "doubao-seedance-1-5-pro-251215")
        self.assertEqual(captured["url"], "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks")


class TtsRoutingTests(unittest.TestCase):
    def test_seedance_routes_dialogue_to_tts(self) -> None:
        from services.audio_routing import resolve_audio_mode

        endpoint = _endpoint()
        self.assertFalse(endpoint.param("native_audio_fallback"))
        mode = resolve_audio_mode({"dialogue": "我等这一天很久了"}, endpoint)
        self.assertEqual(mode, "tts")

    def test_generate_single_shot_metadata_counts_references(self) -> None:
        service = VideoService()
        endpoint = _endpoint()
        storyboard = _storyboard_png()
        data_url = "data:image/png;base64," + base64.b64encode(Path(storyboard).read_bytes()).decode("ascii")
        content = [
            {"type": "text", "text": "p"},
            {"type": "image_url", "image_url": {"url": data_url}, "role": "first_frame"},
        ]

        async def fake_adapter_generate(_self, request):
            return type(
                "R",
                (),
                {
                    "video_path": str(TEST_ROOT / "v.mp4"),
                    "frame_path": str(TEST_ROOT / "f.png"),
                    "native_audio": False,
                    "payload_mode": "first_frame_reference",
                    "task_id": "t",
                },
            )()

        adapter = ArkSeedanceVideoAdapter(_endpoint())
        with (
            patch("services.video_service.get_endpoint", return_value=endpoint),
            patch("services.video_service.get_adapter", return_value=ArkSeedanceVideoAdapter),
            patch.object(ArkSeedanceVideoAdapter, "generate", fake_adapter_generate),
        ):
            result = asyncio.run(
                service.generate_single_shot(prompt="p", project_id="seedance_tests", shot_id="s1", content=content)
            )
        self.assertEqual(result["reference_payload_mode"], "first_frame_reference")
        self.assertFalse(result["native_audio"])
        metadata = service.last_generation_metadata
        self.assertEqual(metadata["provider"], "ark-seedance")
        self.assertEqual(metadata["model"], "doubao-seedance-1-5-pro-251215")
        self.assertEqual(metadata["reference_mode"], "first_frame_only")
        self.assertEqual(metadata["references_validated"], 1)
        self.assertEqual(metadata["references_sent"], ["approved_storyboard_first_frame"])
        self.assertEqual(adapter.capabilities.native_audio, False)


if __name__ == "__main__":
    unittest.main()
