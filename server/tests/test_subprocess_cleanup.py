from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.ffmpeg_service import FFmpegService
from services.providers.endpoint import EndpointConfig
from services.providers.video_ark_seedance import ArkSeedanceVideoAdapter
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


class _BlockingProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False
        self.started = asyncio.Event()
        self.communicate_calls = 0

    async def communicate(self):
        self.communicate_calls += 1
        if self.communicate_calls == 1:
            self.started.set()
            await asyncio.Event().wait()
        return b"", b""

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class SubprocessCleanupTests(unittest.TestCase):
    def test_render_cancellation_kills_and_reaps_ffmpeg(self) -> None:
        async def scenario() -> None:
            process = _BlockingProcess()

            async def create_process(*_args, **_kwargs):
                return process

            with patch("asyncio.create_subprocess_exec", new=create_process):
                task = asyncio.create_task(FFmpegService()._run(["ffmpeg", "-version"]))
                await process.started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            self.assertTrue(process.killed)
            self.assertEqual(process.communicate_calls, 2)

        asyncio.run(scenario())

    def test_frame_extraction_cancellation_kills_and_reaps_ffmpeg(self) -> None:
        async def scenario() -> None:
            process = _BlockingProcess()

            async def create_process(*_args, **_kwargs):
                return process

            with tempfile.TemporaryDirectory(prefix="comic-agent-frame-") as root:
                with patch("asyncio.create_subprocess_exec", new=create_process):
                    task = asyncio.create_task(
                        ArkSeedanceVideoAdapter(EndpointConfig(protocol="ark-seedance"))._extract_last_frame(
                            Path(root) / "video.mp4",
                            Path(root) / "frame.png",
                        )
                    )
                    await process.started.wait()
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task

            self.assertTrue(process.killed)
            self.assertEqual(process.communicate_calls, 2)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
