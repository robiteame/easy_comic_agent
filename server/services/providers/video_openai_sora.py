"""OpenAI Sora native videos API, including authenticated binary content."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import uuid
from urllib.parse import quote

import httpx
from PIL import Image, ImageOps

from config import settings
from services.providers.base import VideoCapabilities, VideoRequest
from services.providers.video_http import VideoHTTPAdapter
from services.security import UploadLimitExceeded


class OpenAISoraVideoAdapter(VideoHTTPAdapter):
    capabilities = VideoCapabilities(
        reference_image=True,
        reference_parameter="input_reference",
        reference_mode="first_frame_only",
        min_duration=4,
        max_duration=20,
        duration_options=(4, 8, 12, 16, 20),
        duration_step=4,
        camera_movement_prompt=True,
        max_reference_inline_bytes=20 * 1024 * 1024,
    )

    def _size(self, request: VideoRequest) -> str:
        pro = "pro" in self.endpoint.model.lower()
        configured = str(self.endpoint.param("size", ""))
        allowed = {"1280x720", "720x1280"}
        if pro:
            allowed.update({"1024x1792", "1792x1024"})
        if configured:
            if configured not in allowed:
                raise ValueError(f"Sora {self.endpoint.model} 不支持 size={configured}")
            return configured
        if request.ratio not in {"16:9", "9:16"}:
            raise ValueError("Sora 仅支持 16:9 / 9:16；尺寸可通过 params.size 显式配置")
        if request.resolution.lower() in {"720p", "720", ""}:
            return "1280x720" if request.ratio == "16:9" else "720x1280"
        if pro and request.resolution.lower() in {"1080p", "1080", "1024p"}:
            return "1792x1024" if request.ratio == "16:9" else "1024x1792"
        raise ValueError("Sora 当前模型不支持该分辨率，请使用 720p 或配置 Pro 的 params.size")

    def build_payload(self, request: VideoRequest) -> dict:
        self.validate_request(request)
        if request.task_type != "auto":
            raise ValueError("Sora 此适配器仅支持视频创建，不支持 remix / extensions / Azure 专有参数")
        payload = {
            "model": self.endpoint.model or "sora-2",
            "prompt": request.prompt,
            "seconds": str(int(request.duration)),
            "size": self._size(request),
        }
        if request.reference_image:
            payload["input_reference"] = request.reference_image
        return payload

    def _reference_file(self, value: str, size: str) -> bytes:
        url = self.reference_assets.to_image_url(value, max_bytes=self.capabilities.max_reference_inline_bytes)
        if not url.startswith("data:image/"):
            raise ValueError("Sora 本地首帧无法读取或编码")
        try:
            raw = base64.b64decode(url.partition(",")[2], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                width, height = (int(part) for part in size.split("x"))
                image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (width, height))
                output = io.BytesIO()
                image.save(output, format="PNG")
                return output.getvalue()
        except (ValueError, OSError) as exc:
            raise ValueError("Sora 首帧图片编码无效") from exc

    async def _submit(self, request: VideoRequest) -> str:
        payload = self.build_payload(request)
        url = self._api_url("https://api.openai.com", "v1/videos")
        image = str(payload.get("input_reference") or "")
        if image and not image.startswith(("https://", "http://")):
            payload.pop("input_reference")
            headers = {key: value for key, value in self._headers().items() if key.lower() != "content-type"}
            data = await self._request_json(
                "POST",
                url,
                headers=headers,
                data=payload,
                files={"input_reference": ("reference.png", self._reference_file(image, payload["size"]), "image/png")},
            )
        else:
            data = await self._request_json("POST", url, json=payload)
        task_id = str(data.get("id") or "")
        if not task_id:
            raise RuntimeError(f"Sora 创建任务返回缺少 id: {data}")
        return task_id

    async def _poll_task(self, task_id: str) -> dict:
        url = self._api_url("https://api.openai.com", f"v1/videos/{quote(task_id, safe='')}")
        for _ in range(max(1, int(self.endpoint.param("max_poll_attempts", 120)))):
            data = await self._request_json("GET", url)
            status = str(data.get("status") or "").lower()
            if status == "completed":
                return data
            if status in {"failed", "cancelled", "canceled"}:
                raise RuntimeError(f"Sora 视频任务失败: {data.get('error') or data}")
            await asyncio.sleep(max(0, float(self.endpoint.param("poll_interval_seconds", 5))))
        raise TimeoutError(f"Sora 视频任务超时: {task_id}")

    async def _save_video(self, request: VideoRequest, data: dict, task_id: str) -> None:
        """Stream authenticated content atomically; never forward the token to redirects."""
        destination = request.output_video_path
        if destination is None:
            raise ValueError("Sora 缺少视频保存路径")
        available = self.storage.ensure_project_capacity(request.project_id, replacing=destination)
        maximum = min(settings.MAX_REMOTE_MEDIA_BYTES, available)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.download")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with httpx.AsyncClient(
                timeout=180,
                follow_redirects=False,
                trust_env=settings.PROVIDER_HTTP_TRUST_ENV,
            ) as client:
                url = self._api_url("https://api.openai.com", f"v1/videos/{quote(task_id, safe='')}/content")
                async with client.stream("GET", url, headers=self._headers()) as response:
                    if response.status_code >= 300:
                        raise RuntimeError(f"Sora 下载视频失败: HTTP {response.status_code}")
                    length = response.headers.get("content-length")
                    if length and int(length) > maximum:
                        raise UploadLimitExceeded("Sora 视频超过媒体大小或存储配额")
                    total = 0
                    with temporary.open("wb") as output:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            total += len(chunk)
                            if total > maximum:
                                raise UploadLimitExceeded("Sora 视频超过媒体大小或存储配额")
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    if total <= 4096:
                        raise RuntimeError("Sora 返回的视频为空或过小")
                    os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
