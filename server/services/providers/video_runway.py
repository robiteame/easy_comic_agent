"""Runway Gen-4.5/Gen-4 Turbo native task API (not an OpenAI wrapper)."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from dataclasses import replace
from urllib.parse import quote

import httpx

from config import settings
from services.providers.base import VideoCapabilities, VideoRequest
from services.providers.video_http import VideoHTTPAdapter
from services.security import validate_remote_url

_RATIOS = {
    "16:9": "1280:720",
    "9:16": "720:1280",
    "4:3": "1104:832",
    "3:4": "832:1104",
    "1:1": "960:960",
    "21:9": "1584:672",
}


class RunwayVideoAdapter(VideoHTTPAdapter):
    capabilities = VideoCapabilities(
        reference_image=True,
        reference_parameter="promptImage",
        reference_mode="first_frame_only",
        min_duration=2,
        max_duration=10,
        camera_movement_prompt=True,
        max_reference_inline_bytes=20 * 1024 * 1024,
    )

    @classmethod
    def effective_capabilities(cls, model: str = ""):
        if "turbo" in model.lower() or model.lower() == "gen4":
            return replace(cls.capabilities, min_duration=5, duration_options=(5, 10), duration_step=5)
        return cls.capabilities

    def _headers(self) -> dict[str, str]:
        return {
            **super()._headers(),
            "X-Runway-Version": str(self.endpoint.param("runway_version", "2024-11-06")),
        }

    def build_payload(self, request: VideoRequest) -> dict:
        self.validate_request(request)
        model = self.endpoint.model or "gen4.5"
        if "aleph" in model.lower() or request.reference_video or request.task_type != "auto":
            raise ValueError("Runway Aleph 视频编辑的 REST 请求字段尚未确认，请使用 Gen-4.5 文生/图生视频")
        if ("turbo" in model.lower() or model.lower() == "gen4") and not request.reference_image:
            raise ValueError(f"Runway {model} 必须提供首帧图片")
        ratio = _RATIOS.get(request.ratio, request.ratio)
        if ratio not in _RATIOS.values():
            raise ValueError(f"Runway 不支持比例 {request.ratio}，可用比例：{', '.join(_RATIOS)}")
        payload = {"model": model, "promptText": request.prompt, "ratio": ratio, "duration": int(request.duration)}
        if request.reference_image:
            image = str(request.reference_image)
            if not image.startswith(("https://", "http://", "runway://")):
                raise ValueError("Runway 首帧需要公网 URL 或已上传的 runway:// URI；请先上传本地/内联图片")
            payload["promptImage"] = image
        if request.seed is not None:
            if not 0 <= request.seed <= 4294967295:
                raise ValueError("Runway seed 必须在 0–4294967295 之间")
            payload["seed"] = request.seed
        for key in ("watermark", "contentModeration"):
            value = self.endpoint.param(key)
            if value is not None:
                payload[key] = bool(value)
        return payload

    async def _submit(self, request: VideoRequest) -> str:
        if request.reference_image and not request.reference_image.startswith(("https://", "http://", "runway://")):
            # Validate all non-image options before initiating the upload.
            self.build_payload(replace(request, reference_image="https://example.com/reference.png"))
            reference = await self._upload_image(request.reference_image)
            request = replace(request, reference_image=reference)
        payload = self.build_payload(request)
        path = "image_to_video" if request.reference_image else "text_to_video"
        data = await self._request_json(
            "POST", self._api_url("https://api.dev.runwayml.com", f"v1/{path}"), json=payload
        )
        task_id = str(data.get("id") or "")
        if not task_id:
            raise RuntimeError(f"Runway 创建任务返回缺少 id: {data}")
        return task_id

    async def _upload_image(self, value: str) -> str:
        inline = self.reference_assets.to_image_url(value, max_bytes=self.capabilities.max_reference_inline_bytes)
        if not inline.startswith("data:image/"):
            raise ValueError("Runway 本地首帧无法读取或编码")
        header, _, encoded = inline.partition(",")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError("Runway 首帧 Base64 无效") from exc
        mime = header[5:].split(";", 1)[0]
        filename = "reference" + (mimetypes.guess_extension(mime) or ".png")
        data = await self._request_json(
            "POST",
            self._api_url("https://api.dev.runwayml.com", "v1/uploads"),
            json={"filename": filename, "type": "ephemeral"},
        )
        upload_url = str(data.get("uploadUrl") or "")
        uri = str(data.get("runwayUri") or "")
        fields = data.get("fields")
        if not upload_url or not uri.startswith("runway://") or not isinstance(fields, dict):
            raise RuntimeError("Runway 上传接口返回缺少 uploadUrl / fields / runwayUri")
        validate_remote_url(upload_url)
        async with httpx.AsyncClient(
            timeout=120, follow_redirects=False, trust_env=settings.PROVIDER_HTTP_TRUST_ENV
        ) as client:
            # Upload hosts receive only their signed form fields, never the API key.
            response = await client.post(upload_url, data=fields, files={"file": (filename, raw, mime)})
        if response.status_code >= 300:
            raise RuntimeError(f"Runway 图片上传失败: HTTP {response.status_code}")
        return uri

    async def _poll_task(self, task_id: str) -> dict:
        url = self._api_url("https://api.dev.runwayml.com", f"v1/tasks/{quote(task_id, safe='')}")
        for _ in range(max(1, int(self.endpoint.param("max_poll_attempts", 120)))):
            data = await self._request_json("GET", url)
            status = str(data.get("status") or "").upper()
            if status == "SUCCEEDED":
                return data
            if status in {"FAILED", "CANCELLED", "CANCELED", "DELETED"}:
                raise RuntimeError(f"Runway 视频任务失败: {data.get('failure') or data.get('failureCode') or data}")
            await asyncio.sleep(max(0, float(self.endpoint.param("poll_interval_seconds", 5))))
        raise TimeoutError(f"Runway 视频任务超时: {task_id}")

    def _video_url(self, data: dict) -> str:
        output = data.get("output") or []
        return str(output[0]) if isinstance(output, list) and output else ""
