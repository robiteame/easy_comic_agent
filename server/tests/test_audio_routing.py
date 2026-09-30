"""音频路由与视频适配器能力的验收测试。

覆盖：
- audio_mode 解析：全局配置、镜头级覆盖优先级、auto 规则、未知值安全回退；
- native 适配器缺失/骨架未就绪时回退 tts 路径并记 warning；
- 流水线级：native 模式跳过 TTS、audio_path 清空、continuity_profile 标记音轨来源；
- 两条路径输出契约一致：ffmpeg 归一化后的镜头 mp4 均带音轨。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT  # noqa: F401,E402

from config import settings  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import Project, Shot  # noqa: E402
from api.routes import shot as shot_route  # noqa: E402
from services import audio_routing  # noqa: E402
from services.audio_routing import native_audio_capable, resolve_audio_mode  # noqa: E402
from services.ffmpeg_service import FFmpegService  # noqa: E402
from services.providers.base import VideoCapabilities, VideoResult  # noqa: E402
from services.providers.endpoint import EndpointConfig  # noqa: E402
from services.providers.video_native_audio import NativeAudioVideoAdapter  # noqa: E402

FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))


def _endpoint(protocol: str = "ark-seedance", audio_mode: str = "tts") -> EndpointConfig:
    return EndpointConfig(protocol=protocol, base_url="https://video.example.test", api_key="k", model="m", params={"audio_mode": audio_mode})


class _FakeNativeAdapter:
    capabilities = VideoCapabilities(reference_image=False, native_audio=True, dialogue_in_prompt=True)
    production_ready = True

    def __init__(self, endpoint):
        self.endpoint = endpoint


class _FakeSilentAdapter:
    capabilities = VideoCapabilities(reference_image=True, native_audio=False, dialogue_in_prompt=False)
    production_ready = True


class ResolveAudioModeTests(unittest.TestCase):
    def test_default_is_tts(self) -> None:
        shot = {"dialogue": "你好", "shot_type": "close-up"}
        with patch.object(audio_routing, "get_endpoint", return_value=_endpoint(audio_mode="tts")):
            self.assertEqual(resolve_audio_mode(shot), "tts")

    def test_global_native_with_capable_adapter_routes_native(self) -> None:
        shot = {"dialogue": "你好"}
        with (
            patch.object(audio_routing, "get_endpoint", return_value=_endpoint(audio_mode="native")),
            patch.object(audio_routing, "get_adapter", return_value=_FakeNativeAdapter),
        ):
            self.assertEqual(resolve_audio_mode(shot), "native")

    def test_native_falls_back_to_tts_when_adapter_lacks_capability(self) -> None:
        shot = {"dialogue": "你好"}
        with (
            patch.object(audio_routing, "get_endpoint", return_value=_endpoint(protocol="ark-seedance", audio_mode="native")),
            patch.object(audio_routing, "get_adapter", return_value=_FakeSilentAdapter),
        ):
            with self.assertLogs(audio_routing.logger, level="WARNING"):
                self.assertEqual(resolve_audio_mode(shot), "tts")

    def test_skeleton_protocol_is_not_native_capable(self) -> None:
        # native-audio 协议骨架（production_ready=False）必须被视为不可用。
        with patch.object(audio_routing, "get_endpoint", return_value=_endpoint(protocol="native-audio")):
            self.assertFalse(native_audio_capable())

    def test_unknown_mode_value_falls_back_to_tts(self) -> None:
        with (
            patch.object(audio_routing, "get_endpoint", return_value=_endpoint(audio_mode="whatever")),
            patch.object(audio_routing, "get_adapter", return_value=_FakeNativeAdapter),
        ):
            with self.assertLogs(audio_routing.logger, level="WARNING"):
                self.assertEqual(resolve_audio_mode({"dialogue": "hi"}), "tts")

    def test_shot_override_beats_global_config(self) -> None:
        with (
            patch.object(audio_routing, "get_endpoint", return_value=_endpoint(audio_mode="native")),
            patch.object(audio_routing, "get_adapter", return_value=_FakeNativeAdapter),
        ):
            self.assertEqual(resolve_audio_mode({"dialogue": "hi", "audio_mode": "tts"}), "tts")
            self.assertEqual(
                resolve_audio_mode({"dialogue": "hi", "continuity_profile": {"audio_mode": "tts"}}),
                "tts",
            )
        with (
            patch.object(audio_routing, "get_endpoint", return_value=_endpoint(audio_mode="tts")),
            patch.object(audio_routing, "get_adapter", return_value=_FakeNativeAdapter),
        ):
            self.assertEqual(resolve_audio_mode({"dialogue": "hi", "audio_mode": "native"}), "native")
            self.assertEqual(
                resolve_audio_mode({"dialogue": "hi", "continuity_profile": {"audio_mode": "native"}}),
                "native",
            )

    def test_auto_uses_native_only_for_dialogue_closeups(self) -> None:
        with (
            patch.object(audio_routing, "get_endpoint", return_value=_endpoint(audio_mode="auto")),
            patch.object(audio_routing, "get_adapter", return_value=_FakeNativeAdapter),
        ):
            self.assertEqual(resolve_audio_mode({"dialogue": "关键台词", "shot_type": "close-up"}), "native")
            self.assertEqual(resolve_audio_mode({"dialogue": "关键台词", "shot_type": "extreme_close"}), "native")
            self.assertEqual(resolve_audio_mode({"dialogue": "关键台词", "shot_type": "wide"}), "tts")
            self.assertEqual(resolve_audio_mode({"shot_type": "close-up"}), "tts")

    def test_unknown_video_protocol_disables_native(self) -> None:
        from services.providers.registry import UnknownProtocolError

        def _raise(capability, protocol):
            raise UnknownProtocolError("boom")

        with (
            patch.object(audio_routing, "get_endpoint", return_value=_endpoint(protocol="mystery", audio_mode="native")),
            patch.object(audio_routing, "get_adapter", side_effect=_raise),
        ):
            self.assertFalse(native_audio_capable())
            self.assertEqual(resolve_audio_mode({"dialogue": "hi"}), "tts")


class SkeletonAdapterTests(unittest.TestCase):
    def test_skeleton_builds_dialogue_prompt_and_refuses_generation(self) -> None:
        from services.providers.base import Dialogue, VideoRequest

        adapter = NativeAudioVideoAdapter(_endpoint(protocol="native-audio"))
        request = VideoRequest(
            prompt="A girl waves.",
            dialogues=[Dialogue(role="小雨", text="你好呀！", emotion="happy")],
        )
        prompt = adapter.build_prompt(request)
        self.assertIn("A girl waves.", prompt)
        self.assertIn("小雨", prompt)
        self.assertIn("你好呀！", prompt)
        self.assertIn("原生开口", prompt)

        with self.assertRaises(NotImplementedError):
            asyncio.run(adapter.generate(request))


class _PipelineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    def setUp(self) -> None:
        self.db = SessionLocal()
        self.db.query(Shot).delete()
        self.db.query(Project).delete()
        self.db.commit()
        self.project = Project(id="audio-route-project", title="AudioRoute", status="assets_ready")
        self.shot = Shot(
            id="audio-route-shot",
            project_id=self.project.id,
            sequence=1,
            version=1,
            confirmed=True,
            dialogue="我看见故事成形了",
            emotion="happy",
            characters_in_scene=json.dumps(["小雨"]),
            storyboard_path="story.png",
            image_path="story.png",
            status="storyboard_approved",
        )
        self.db.add_all([self.project, self.shot])
        self.db.commit()

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()

    def _reload_shot(self) -> Shot:
        self.db.expire_all()
        return self.db.get(Shot, self.shot.id)


class NativeModeSkipsVoiceTests(_PipelineTestCase):
    def test_native_mode_skips_tts_and_marks_audio_source(self) -> None:
        tts_calls: list[dict] = []
        requests: list = []

        from PIL import Image

        class _RecordingNativeAdapter(_FakeNativeAdapter):
            """真实写盘的 mock 适配器：产出带音轨的 mp4，走完整服务链路。"""

            async def generate(self, request):
                requests.append(request)
                video_path = request.output_video_path
                frame_path = request.output_frame_path
                video_path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-f", "lavfi", "-i", "color=c=green:s=256x256:d=1",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-shortest", str(video_path),
                    ],
                    check=True,
                    capture_output=True,
                )
                Image.new("RGB", (256, 256), (40, 160, 90)).save(frame_path)
                return VideoResult(
                    video_path=str(video_path),
                    frame_path=str(frame_path),
                    native_audio=True,
                    payload_mode="text_only",
                )

        async def fake_tts(**kwargs):
            tts_calls.append(kwargs)
            return str(settings.OUTPUT_DIR / "should-not-exist.wav")

        # 路由层：全局 audio_mode=native，适配器具备原生音频能力。
        # 服务层：只替换适配器解析，让台词经 VideoRequest 真实传入适配器并走契约校验。
        with (
            patch.object(audio_routing, "get_endpoint", return_value=_endpoint(audio_mode="native")),
            patch.object(audio_routing, "get_adapter", return_value=_RecordingNativeAdapter),
            patch("services.video_service.get_adapter", return_value=_RecordingNativeAdapter),
            patch.object(shot_route.tts_service, "generate_dialogue", side_effect=fake_tts),
        ):
            asyncio.run(shot_route._run_single_shot_video(self.shot.id, force=True, confirm_capability_downgrade=True))

        self.assertEqual(tts_calls, [], "native 模式必须跳过 TTS 配音节点")
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertIsNotNone(request.dialogues, "native 模式必须把台词传给视频适配器")
        self.assertEqual(request.dialogues[0].text, "我看见故事成形了")
        self.assertEqual(request.dialogues[0].role, "小雨")
        self.assertTrue(Path(request.output_video_path).exists())
        current = self._reload_shot()
        self.assertEqual(current.audio_path, "", "native 模式必须清空旧的 TTS 音频引用")
        self.assertTrue(current.video_path.endswith(".mp4"))
        profile = json.loads(current.continuity_profile or "{}")
        self.assertEqual(profile.get("audio_source"), "native")

    def test_native_video_without_audio_track_is_rejected(self) -> None:
        # 适配器宣称 native 但没带音轨：不允许无声成品进入成片流程。
        async def fake_video(shot_data, *_args, **_kwargs):
            return {
                "video_path": str(settings.OUTPUT_DIR / "silent.mp4"),
                "frame_path": str(settings.OUTPUT_DIR / "silent.png"),
                "native_audio": False,
            }

        with (
            patch.object(audio_routing, "get_endpoint", return_value=_endpoint(audio_mode="native")),
            patch.object(audio_routing, "get_adapter", return_value=_FakeNativeAdapter),
            patch.object(shot_route.seedance_service, "generate_shot_video", side_effect=fake_video),
        ):
            with self.assertRaises(RuntimeError):
                asyncio.run(shot_route._run_single_shot_video(self.shot.id, force=True))

    def test_shot_level_tts_override_beats_global_native(self) -> None:
        self.shot.continuity_profile = json.dumps({"audio_mode": "tts"})
        self.db.commit()

        tts_calls: list[dict] = []

        async def fake_tts(**kwargs):
            tts_calls.append(kwargs)
            return str(settings.OUTPUT_DIR / "tts.wav")

        async def fake_video(shot_data, *_args, **_kwargs):
            return {
                "video_path": str(settings.OUTPUT_DIR / "tts.mp4"),
                "frame_path": str(settings.OUTPUT_DIR / "tts.png"),
                "native_audio": False,
            }

        with (
            patch.object(audio_routing, "get_endpoint", return_value=_endpoint(audio_mode="native")),
            patch.object(audio_routing, "get_adapter", return_value=_FakeNativeAdapter),
            patch.object(shot_route.tts_service, "generate_dialogue", side_effect=fake_tts),
            patch.object(shot_route.seedance_service, "generate_shot_video", side_effect=fake_video),
        ):
            asyncio.run(shot_route._run_single_shot_video(self.shot.id, force=True))

        self.assertEqual(len(tts_calls), 1, "镜头级 audio_mode=tts 必须优先于全局 native")
        current = self._reload_shot()
        self.assertEqual(current.audio_path, str(settings.OUTPUT_DIR / "tts.wav"))
        profile = json.loads(current.continuity_profile or "{}")
        self.assertIsNone(profile.get("audio_source"))


class TtsModeUnchangedTests(_PipelineTestCase):
    def test_tts_mode_behaves_like_before(self) -> None:
        tts_calls: list[dict] = []

        async def fake_tts(**kwargs):
            tts_calls.append(kwargs)
            return str(settings.OUTPUT_DIR / "tts-default.wav")

        async def fake_video(shot_data, *_args, **_kwargs):
            return {
                "video_path": str(settings.OUTPUT_DIR / "tts-default.mp4"),
                "frame_path": str(settings.OUTPUT_DIR / "tts-default.png"),
                "native_audio": False,
                "reference_payload_mode": "first_frame_reference",
            }

        # 全局默认（真实端点默认 tts），不打任何补丁于路由配置。
        with (
            patch.object(shot_route.tts_service, "generate_dialogue", side_effect=fake_tts),
            patch.object(shot_route.seedance_service, "generate_shot_video", side_effect=fake_video),
        ):
            asyncio.run(shot_route._run_single_shot_video(self.shot.id, force=True))

        self.assertEqual(len(tts_calls), 1)
        current = self._reload_shot()
        self.assertEqual(current.audio_path, str(settings.OUTPUT_DIR / "tts-default.wav"))
        profile = json.loads(current.continuity_profile or "{}")
        self.assertNotEqual(profile.get("audio_source"), "native")


@unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg/ffprobe unavailable")
class OutputContractTests(unittest.TestCase):
    """两条音频路径的输出契约：归一化后的镜头 mp4 均带音轨。"""

    def _probe_streams(self, path: Path) -> set[str]:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return {stream.get("codec_type") for stream in json.loads(result.stdout).get("streams", [])}

    def _make_video_with_audio(self, path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=256x256:d=1",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-shortest", str(path),
            ],
            check=True,
            capture_output=True,
        )

    def _make_silent_video(self, path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "testsrc=size=512x512:duration=2",
                "-frames:v", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(path),
            ],
            check=True,
            capture_output=True,
        )

    def _make_wav(self, path: Path) -> None:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=660:duration=1", str(path)],
            check=True,
            capture_output=True,
        )

    def test_native_clip_keeps_source_audio_track(self) -> None:
        root = Path(settings.OUTPUT_DIR) / "contract-native"
        root.mkdir(parents=True, exist_ok=True)
        source = root / "native_source.mp4"
        self._make_video_with_audio(source)

        workdir = root / "work"
        workdir.mkdir(exist_ok=True)
        shot = {"video_path": str(source), "native_audio": True, "duration": 1.0}
        clip = asyncio.run(FFmpegService()._normalize_video_clip(shot, 256, 256, 0, workdir))

        streams = self._probe_streams(clip)
        self.assertIn("video", streams)
        self.assertIn("audio", streams, "native 路径产物必须自带音轨")

    def test_tts_clip_still_gets_mixed_audio_track(self) -> None:
        root = Path(settings.OUTPUT_DIR) / "contract-tts"
        root.mkdir(parents=True, exist_ok=True)
        source = root / "silent_source.mp4"
        audio = root / "tts.wav"
        self._make_silent_video(source)
        self._make_wav(audio)

        workdir = root / "work"
        workdir.mkdir(exist_ok=True)
        shot = {"video_path": str(source), "audio_path": str(audio), "duration": 1.0}
        clip = asyncio.run(FFmpegService()._normalize_video_clip(shot, 256, 256, 0, workdir))

        streams = self._probe_streams(clip)
        self.assertIn("video", streams)
        self.assertIn("audio", streams, "tts 路径产物必须带合成音轨")

    def test_native_flag_from_continuity_profile(self) -> None:
        self.assertTrue(
            FFmpegService._shot_has_native_audio({"continuity_profile": {"audio_source": "native"}})
        )
        self.assertFalse(FFmpegService._shot_has_native_audio({"continuity_profile": {}}))
        self.assertFalse(FFmpegService._shot_has_native_audio({}))


if __name__ == "__main__":
    unittest.main()
