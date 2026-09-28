import base64
import logging
import mimetypes
from io import BytesIO
from pathlib import Path

from PIL import Image

from config import settings
from services.security import existing_file

logger = logging.getLogger(__name__)


class ReferenceAssetService:
    """Prepare persisted visual references for model payloads and continuity controls."""

    # JPEG 压缩阶梯：(最长边像素, 质量)。参考图超出目标预算时从原尺寸
    # 逐级降采样+降质量，直到编码结果压进预算；档位从高到低尝试，保证在
    # 预算允许时尽量保留脸部、服装与材质细节，不会无条件落到最低分辨率。
    _BUDGET_LADDER: tuple[tuple[int, int], ...] = (
        (1536, 92), (1440, 88), (1280, 84), (1024, 82), (1024, 72),
        (768, 74), (768, 62), (512, 68), (512, 55), (384, 50),
    )

    def __init__(self):
        self.output_dir = settings.OUTPUT_DIR / "projects"
        self.last_transform_metadata: dict[str, object] = {}

    def to_image_url(self, path_or_url: str, *, max_bytes: int | None = None) -> str:
        """把参考图转成可内联进 API payload 的 URL。

        ``max_bytes`` 为编码后字节预算（如百炼视频网关的请求体上限）：
        超预算的图片会走 JPEG 压缩阶梯重编码；未指定时保持原样内联。
        """

        value = str(path_or_url or "").strip()
        if not value:
            return ""
        if value.startswith(("http://", "https://")):
            return value
        if value.startswith("data:image/"):
            if not max_bytes:
                return value
            payload = value.partition(",")[2]
            if len(payload) * 3 // 4 <= max_bytes:
                self.last_transform_metadata = {"original_bytes": len(payload) * 3 // 4, "sent_bytes": len(payload) * 3 // 4}
                return value
            # 压不进预算（含无法解码的 data URL）视同缺失，让上层给出
            # 「参考图不可用」的明确报错，而不是把超限请求发出去吃连接重置。
            return self._encode_jpeg_within_budget(
                self._decode_data_url(value) or b"", max_bytes
            )
        path = existing_file(
            value,
            minimum_size=1,
            allowed_roots=(settings.OUTPUT_DIR, settings.ASSETS_DIR, settings.DATA_DIR),
        )
        if path is None:
            return ""
        raw = path.read_bytes()
        self.last_transform_metadata = {"original_bytes": len(raw), "original_size": None, "sent_bytes": len(raw), "sent_size": None}
        # These references must be embedded in an API payload, so keep the
        # unavoidable in-memory base64 conversion tightly bounded.
        if len(raw) > settings.MAX_INLINE_REFERENCE_BYTES:
            return ""
        if max_bytes and len(raw) > max_bytes:
            # 压不进预算视同缺失，让上层给出「参考图不可用」的明确报错，
            # 而不是把超限请求发出去吃一个连接重置。
            return self._encode_jpeg_within_budget(raw, max_bytes)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{data}"

    @staticmethod
    def _decode_data_url(value: str) -> bytes | None:
        header, _, payload = value.partition(",")
        if not payload or ";base64" not in header:
            return None
        try:
            return base64.b64decode(payload)
        except ValueError:
            return None

    def _encode_jpeg_within_budget(self, raw: bytes, max_bytes: int) -> str:
        if not raw:
            return ""
        try:
            with Image.open(BytesIO(raw)) as image:
                image.load()
                original_size = tuple(image.size)
                flattened = self._flatten_alpha(image)
        except Exception:
            return ""
        for max_side, quality in self._BUDGET_LADDER:
            candidate = flattened.copy()
            if max(candidate.size) > max_side:
                candidate.thumbnail((max_side, max_side))
            buffer = BytesIO()
            candidate.save(buffer, format="JPEG", quality=quality, optimize=True)
            data = buffer.getvalue()
            if len(data) <= max_bytes:
                self.last_transform_metadata = {
                    "original_bytes": len(raw),
                    "original_size": original_size,
                    "sent_bytes": len(data),
                    "sent_size": tuple(candidate.size),
                }
                return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
        return ""

    @staticmethod
    def _flatten_alpha(image: Image.Image) -> Image.Image:
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            return background
        return image.convert("RGB")

    def materialize_continuity_controls(
        self,
        project_id: str,
        shot_id: str,
        source_path: str,
        enabled: bool,
    ) -> dict[str, str]:
        """已停用：不产出伪造的姿态/深度控制图。

        历史实现用 ``PIL.FIND_EDGES`` 生成边缘图并命名为 ``*_openpose_ref``、
        用灰度高斯模糊命名为 ``*_depth_ref``——它们既不是 OpenPose 也不是
        Depth，作为参考图发给模型反而会污染画面。在接入真实的姿态/深度
        估计模型之前，本方法一律返回空 dict，画像层统一标记 unsupported。
        """
        if enabled and source_path:
            logger.info(
                "连续性控制图物化已停用（无真实 OpenPose/Depth 模型）: project=%s shot=%s",
                project_id,
                shot_id,
            )
        return {}
