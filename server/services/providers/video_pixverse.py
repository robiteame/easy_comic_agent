"""PixVerse native upload → generation → query protocol.

The current official schema uses first_frame_img/last_frame_img for transition
and image_references for fusion, rather than the shorthand in the research note.
"""

from __future__ import annotations

import asyncio
import base64
import io
import uuid
from dataclasses import replace
from urllib.parse import quote

from PIL import Image

from services.providers.base import VideoCapabilities, VideoRequest
from services.providers.video_http import VideoHTTPAdapter

_BASE = "https://app-api.pixverse.ai"
_ROOT = "/openapi/v2"
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


class PixVerseVideoAdapter(VideoHTTPAdapter):
    capabilities = VideoCapabilities(
        reference_image=True,
        multiple_reference_images=True,
        max_reference_images=10,
        reference_parameter="img_id",
        reference_role_parameter="image_references[].type",
        last_frame_input=True,
        first_last_frame_interpolation=True,
        reference_mode="multi_reference",
        min_duration=1,
        max_duration=15,
        native_audio=True,
        dialogue_in_prompt=True,
        camera_movement_prompt=True,
    )

    @classmethod
    def effective_capabilities(cls, model: str = ""):
        name = (model or "v6").lower()
        if name in {"v6", "c1"}:
            return cls.capabilities
        audio = name in {"v5.5", "v5.6"}
        return replace(
            cls.capabilities,
            min_duration=5,
            max_duration=10 if audio else 8,
            duration_options=(5, 8, 10) if audio else (5, 8),
            native_audio=audio,
            dialogue_in_prompt=audio,
            last_frame_input=name in {"v5", "v5.6"},
            first_last_frame_interpolation=name in {"v5", "v5.6"},
            multiple_reference_images=name not in {"v3.5", "v4"},
        )

    def _headers(self) -> dict[str, str]:
        return {"API-KEY": self.endpoint.api_key, "Ai-trace-id": str(uuid.uuid4()), "Content-Type": "application/json"}

    @staticmethod
    def _response(data: dict) -> dict:
        if data.get("ErrCode", 0) != 0:
            raise RuntimeError(f"PixVerse API 错误 {data.get('ErrCode')}: {data.get('ErrMsg') or 'unknown'}")
        result = data.get("Resp")
        if not isinstance(result, dict):
            raise RuntimeError("PixVerse API 返回缺少 Resp")
        return result

    def _references(self, request: VideoRequest) -> list:
        return [a for a in request.reference_assets if a.type not in {"approved_storyboard_first_frame", "end_frame"}]

    def _validate_payload_request(self, request: VideoRequest) -> None:
        self.validate_request(request)
        if request.task_type != "auto":
            raise ValueError("当前 PixVerse 适配器支持视频生成，暂未适配视频编辑或延长端点")
        caps = self.effective_capabilities(self.endpoint.model)
        refs = self._references(request)
        if refs and not caps.multiple_reference_images:
            raise ValueError("当前 PixVerse 模型不支持多图参考")
        if request.end_frame and not request.reference_image:
            raise ValueError("PixVerse 首尾帧生成需要首帧")
        if request.end_frame and refs:
            raise ValueError("PixVerse 首尾帧与主体参考使用不同端点，不可混用")
        if len(refs) + bool(request.reference_image) > caps.max_reference_images:
            raise ValueError("PixVerse 参考图片不能超过 10 张")
        if len(request.prompt) > 5000:
            raise ValueError("PixVerse 提示词不能超过 5000 字符")
        model = (self.endpoint.model or "v6").lower()
        ratios = {"16:9", "4:3", "1:1", "3:4", "9:16"}
        if model in {"v6", "c1"}:
            ratios |= {"2:3", "3:2", "21:9"}
        if request.ratio not in ratios:
            raise ValueError("当前 PixVerse 模型不支持此画幅比例")
        if request.resolution.lower() not in {"360p", "540p", "720p", "1080p"}:
            raise ValueError("PixVerse 不支持此分辨率")
        if model in {"v5.5", "v5.6"} and request.duration == 10 and request.resolution.lower() == "1080p":
            raise ValueError("PixVerse v5.5/v5.6 的 1080p 不支持 10 秒")
        if request.seed is not None and not 0 <= request.seed <= 2147483647:
            raise ValueError("PixVerse seed 必须在 0–2147483647 范围内")

    def build_payload(self, request: VideoRequest, image_ids: list[int] | None = None) -> dict:
        self._validate_payload_request(request)
        refs = self._references(request)
        image_ids = image_ids or []
        payload = {
            "model": self.endpoint.model or "v6",
            "prompt": request.prompt,
            "duration": request.duration,
            "quality": request.resolution.lower(),
        }
        if not request.end_frame:
            payload["aspect_ratio"] = request.ratio
        if request.seed is not None:
            payload["seed"] = request.seed
        if self.effective_capabilities(self.endpoint.model).native_audio:
            payload["generate_audio_switch"] = bool(request.generate_audio)
        if request.end_frame:
            if len(image_ids) != 2:
                raise ValueError("PixVerse 首尾帧需要两张已上传图片的 ID")
            payload.update(first_frame_img=image_ids[0], last_frame_img=image_ids[1])
        elif refs:
            if len(image_ids) != len(refs) + bool(request.reference_image):
                raise ValueError("PixVerse 主体参考需要已上传图片的 ID")
            payload["image_references"] = [{"img_id": image_id} for image_id in image_ids]
        elif request.reference_image:
            if len(image_ids) != 1:
                raise ValueError("PixVerse 图生视频需要已上传图片的 ID")
            payload["img_id"] = image_ids[0]
        for key in ("generate_multi_clip_switch", "template_id", "motion_mode", "camera_movement", "webhook_id"):
            if self.endpoint.param(key) is not None:
                payload[key] = self.endpoint.param(key)
        return payload

    async def _upload_image(self, value: str) -> int:
        headers = self._headers()
        headers.pop("Content-Type", None)  # httpx must generate the multipart boundary.
        if value.startswith(("http://", "https://")):
            files = {"image_url": (None, value)}
        else:
            image_url = self.reference_assets.to_image_url(value, max_bytes=_MAX_IMAGE_BYTES - 1)
            if not image_url.startswith("data:"):
                raise ValueError("PixVerse 图片必须为公网 URL、有效图片文件或 data URL")
            try:
                raw = base64.b64decode(image_url.partition(",")[2], validate=True)
                if len(raw) >= _MAX_IMAGE_BYTES:
                    raise ValueError("PixVerse 图片须小于 20MB")
                with Image.open(io.BytesIO(raw)) as image:
                    if max(image.size) > 10000 or image.format not in {"JPEG", "PNG", "WEBP"}:
                        raise ValueError("PixVerse 图片须为 JPEG/PNG/WebP，最长边不能超过 10000 像素")
                    extension = "jpg" if image.format == "JPEG" else image.format.lower()
                    image.verify()
            except (OSError, ValueError) as exc:
                raise ValueError(f"PixVerse 图片无效: {exc}") from exc
            mime = "image/jpeg" if extension == "jpg" else f"image/{extension}"
            files = {"image": (f"reference.{extension}", raw, mime)}
        data = await self._request_json(
            "POST", self._api_url(_BASE, f"{_ROOT}/image/upload"), headers=headers, files=files
        )
        image_id = self._response(data).get("img_id")
        if not isinstance(image_id, int) or image_id <= 0:
            raise RuntimeError("PixVerse 上传图片返回缺少 img_id")
        return image_id

    async def _submit(self, request: VideoRequest) -> str:
        self._validate_payload_request(request)
        refs = self._references(request)
        sources = [request.reference_image] if request.reference_image else []
        if request.end_frame:
            sources.append(request.end_frame)
        else:
            sources.extend(a.url or a.source_path for a in refs)
        image_ids = [await self._upload_image(source) for source in sources]
        mode = "transition" if request.end_frame else "fusion" if refs else "img" if request.reference_image else "text"
        data = await self._request_json(
            "POST",
            self._api_url(_BASE, f"{_ROOT}/video/{mode}/generate"),
            json=self.build_payload(request, image_ids),
        )
        task_id = self._response(data).get("video_id")
        if not task_id:
            raise RuntimeError("PixVerse 创建任务返回缺少 video_id")
        return str(task_id)

    async def _poll_task(self, task_id: str) -> dict:
        path = str(self.endpoint.param("query_path", f"{_ROOT}/video/result/{{task_id}}"))
        path = path.replace("{task_id}", quote(task_id, safe=""))
        for _ in range(max(1, int(self.endpoint.param("max_poll_attempts", 120)))):
            data = await self._request_json("GET", self._api_url(_BASE, path))
            result = self._response(data)
            status = result.get("status")
            if status == 1:
                return data
            if status in {7, 8}:
                raise RuntimeError(f"PixVerse 任务失败，状态码 {status}")
            await asyncio.sleep(max(0.1, float(self.endpoint.param("poll_interval_seconds", 15))))
        raise TimeoutError("PixVerse 视频生成超时")

    def _video_url(self, data: dict) -> str:
        return str(self._response(data).get("url") or "")
