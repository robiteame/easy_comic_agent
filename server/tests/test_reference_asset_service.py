"""ReferenceAssetService 压缩、尺寸记录与失败路径的验收测试。

- 超预算参考图按 JPEG 阶梯压进预算，且优先保留高分辨率档位；
- ``last_transform_metadata`` 记录原图/发送的尺寸与字节；
- 无法解码/压不进预算的内容返回空串，由上层明确报错，不发送空参考图；
- 声明了大预算（如 Seedance 8MB）时不得无条件降到最低分辨率。
"""

from __future__ import annotations

import base64
import io
import random
import sys
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from PIL import Image  # noqa: E402

from services.providers.video_ark_seedance import ArkSeedanceVideoAdapter  # noqa: E402
from services.reference_asset_service import ReferenceAssetService  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402


def _png_bytes(width: int, height: int, *, noise: bool = False) -> bytes:
    image = Image.new("RGB", (width, height))
    if noise:
        rng = random.Random(11)
        image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(width * height)])
    else:
        image.putdata([(200, 120, 40)] * (width * height))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_png(path: Path, width: int, height: int, *, noise: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes(width, height, noise=noise))
    return path


class CompressionBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = ReferenceAssetService()
        cls.noise_png = _write_png(TEST_ROOT / "output" / "ref_assets" / "noise_720x1280.png", 720, 1280, noise=True)

    def test_oversized_file_recompressed_within_budget_with_metadata(self) -> None:
        budget = 128 * 1024
        url = self.service.to_image_url(str(self.noise_png), max_bytes=budget)
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        raw = base64.b64decode(url.partition(",")[2])
        self.assertLessEqual(len(raw), budget)
        meta = self.service.last_transform_metadata
        self.assertEqual(meta.get("original_bytes"), self.noise_png.stat().st_size)
        self.assertEqual(meta.get("sent_bytes"), len(raw))
        self.assertEqual(tuple(meta.get("original_size") or ()), (720, 1280))
        sent_size = tuple(meta.get("sent_size") or ())
        self.assertTrue(sent_size and max(sent_size) >= 512, f"压缩后分辨率过低: {sent_size}")

    def test_large_budget_keeps_high_resolution(self) -> None:
        # Seedance 声明 8MB 预算：噪声图 720x1280 应保持原始分辨率（无损内联），
        # 不允许无条件降采样到最低档。
        big_budget = ArkSeedanceVideoAdapter.capabilities.max_reference_inline_bytes
        self.assertGreater(big_budget, 1024 * 1024)
        url = self.service.to_image_url(str(self.noise_png), max_bytes=big_budget)
        self.assertTrue(url.startswith("data:image/png;base64,"))
        raw = base64.b64decode(url.partition(",")[2])
        with Image.open(io.BytesIO(raw)) as image:
            self.assertEqual(image.size, (720, 1280))

    def test_small_file_within_budget_untouched(self) -> None:
        small = _write_png(TEST_ROOT / "output" / "ref_assets" / "small.png", 64, 64)
        url = self.service.to_image_url(str(small), max_bytes=128 * 1024)
        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_budget_free_behavior_unchanged(self) -> None:
        url = self.service.to_image_url(str(self.noise_png))
        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_http_url_passthrough(self) -> None:
        self.assertEqual(
            self.service.to_image_url("https://cdn.example.test/a.png", max_bytes=1024),
            "https://cdn.example.test/a.png",
        )


class FailurePathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = ReferenceAssetService()

    def test_undecodable_data_url_returns_empty(self) -> None:
        garbage = "data:image/png;base64," + base64.b64encode(b"\x00" * (256 * 1024)).decode("ascii")
        self.assertEqual(self.service.to_image_url(garbage, max_bytes=128 * 1024), "")

    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(self.service.to_image_url(str(TEST_ROOT / "nope" / "missing.png"), max_bytes=1024), "")

    def test_video_build_content_raises_on_unencodable_first_frame(self) -> None:
        from services.video_service import VideoService

        garbage = "data:image/png;base64," + base64.b64encode(b"\x00" * (512 * 1024)).decode("ascii")
        service = VideoService()
        with self.assertRaisesRegex(RuntimeError, "首帧参考图"):
            service._build_content("prompt", {"storyboard_path": garbage, "image_path": garbage}, None)

    def test_incompressible_image_fails_closed(self) -> None:
        # 极小预算 + 噪声图：阶梯全部失败 → 返回空串（上层必须报错）。
        noise = _write_png(TEST_ROOT / "output" / "ref_assets" / "noise_small.png", 256, 256, noise=True)
        self.assertEqual(self.service.to_image_url(str(noise), max_bytes=256), "")


if __name__ == "__main__":
    unittest.main()
