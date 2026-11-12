"""Vidu Q3 官方接口：Token 鉴权及四种生成端点。"""

from __future__ import annotations

import asyncio
import base64
from urllib.parse import quote

from services.providers.base import VideoCapabilities, VideoRequest
from services.providers.video_http import VideoHTTPAdapter

_DEFAULT_BASE = "https://api.vidu.com"


class ViduVideoAdapter(VideoHTTPAdapter):
    capabilities = VideoCapabilities(
        reference_image=True,
        multiple_reference_images=True,
        # The supplied document does not confirm the reference2video count cap.
        max_reference_images=0,
        reference_parameter="images",
        reference_mode="multi_reference",
        last_frame_input=True,
        first_last_frame_interpolation=True,
        native_audio=True,
        dialogue_in_prompt=True,
        camera_movement_prompt=True,
        min_duration=1,
        max_duration=16,
    )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.endpoint.api_key}", "Content-Type": "application/json"}

    def _image(self, value: str) -> str:
        image = self.reference_assets.to_image_url(value, max_bytes=10 * 1024 * 1024)
        if not image:
            raise ValueError("Vidu 参考图片无法读取")
        if image.startswith("data:"):
            try:
                raw = base64.b64decode(image.partition(",")[2], validate=True)
            except ValueError as exc:
                raise ValueError("Vidu 图片 Base64 编码无效") from exc
            if len(raw) > 10 * 1024 * 1024:
                raise ValueError("Vidu Base64 图片解码后不可超过 10MB")
        return image

    def _reference_images(self, request: VideoRequest) -> list[str]:
        values = [
            asset.url or asset.source_path
            for asset in request.reference_assets
            if asset.type not in {"approved_storyboard_first_frame", "end_frame"}
        ]
        return list(dict.fromkeys(value for value in values if value))

    def _task_kind(self, request: VideoRequest) -> str:
        if request.end_frame:
            return "start-end2video"
        if self._reference_images(request):
            return "reference2video"
        return "img2video" if request.reference_image else "txt2video"

    def build_payload(self, request: VideoRequest) -> dict:
        self.validate_request(request)
        if request.task_type != "auto":
            raise ValueError("Vidu 当前适配器仅支持视频生成，不支持编辑/续写任务")
        if request.end_frame and not request.reference_image:
            raise ValueError("Vidu 首尾帧任务必须同时提供首帧")
        if any(
            asset.type in {"reference_video", "reference_audio"}
            or asset.provider_type in {"reference_video", "reference_audio"}
            for asset in request.reference_assets
        ):
            raise ValueError("Vidu reference2video 的视频/音频参考字段尚未确认，当前仅支持图片参考")
        if len(request.prompt) > 20000:
            raise ValueError("Vidu prompt 超过 20000 字符")
        refs = self._reference_images(request)
        if request.end_frame and refs:
            raise ValueError("Vidu 首尾帧与多参考图属于不同端点，请拆分任务")
        model = self.endpoint.model or "viduq3-pro"
        resolution = request.resolution.lower()
        allowed = {"720p", "1080p"} if "pro-fast" in model else {"540p", "720p", "1080p"}
        if resolution not in allowed:
            raise ValueError(f"{model} 不支持分辨率 {resolution}")
        off_peak = bool(self.endpoint.param("off_peak", False))
        if request.generate_audio and off_peak:
            raise ValueError("Vidu 开启音频时不能同时使用 off_peak")
        payload = {
            "model": model,
            "prompt": request.prompt,
            "duration": int(request.duration),
            "resolution": resolution,
            "audio": bool(request.generate_audio),
        }
        kind = self._task_kind(request)
        if kind == "start-end2video":
            payload["images"] = [self._image(request.reference_image), self._image(request.end_frame)]
        elif kind == "reference2video":
            images = list(dict.fromkeys(([request.reference_image] if request.reference_image else []) + refs))
            payload["images"] = [self._image(image) for image in images]
        elif kind == "img2video":
            payload["images"] = [self._image(request.reference_image)]
        if request.seed is not None:
            payload["seed"] = int(request.seed)
        for name in ("is_rec", "voice_id", "callback_url", "payload"):
            value = self.endpoint.param(name)
            if value is not None:
                payload[name] = value
        if off_peak:
            payload["off_peak"] = True
        # Deliberately no ratio/aspect_ratio: the provider follows its input.
        return payload

    @staticmethod
    def _check_response(data: dict) -> None:
        if data.get("err_code") not in (None, 0, "0"):
            raise RuntimeError(f"Vidu API 错误 {data.get('err_code')}: {data.get('message', '任务失败')}")

    async def _submit(self, request: VideoRequest) -> str:
        data = await self._request_json(
            "POST",
            self._api_url(_DEFAULT_BASE, f"/ent/v2/{self._task_kind(request)}"),
            json=self.build_payload(request),
        )
        self._check_response(data)
        task_id = str(data.get("task_id") or "")
        if not task_id:
            raise RuntimeError("Vidu 创建任务未返回 task_id")
        return task_id

    async def _poll_task(self, task_id: str) -> dict:
        attempts = max(1, int(self.endpoint.param("max_poll_attempts", 180)))
        interval = max(0.1, float(self.endpoint.param("poll_interval_seconds", 5)))
        for index in range(attempts):
            data = await self._request_json(
                "GET", self._api_url(_DEFAULT_BASE, f"/ent/v2/tasks/{quote(task_id, safe='')}/creations")
            )
            self._check_response(data)
            state = str(data.get("state") or data.get("status") or "").lower()
            if state in {"success", "succeeded", "completed"}:
                return data
            if state in {"failed", "canceled", "cancelled", "error"}:
                raise RuntimeError(f"Vidu 视频任务失败: {data.get('err_code') or data.get('message') or state}")
            if index + 1 < attempts:
                await asyncio.sleep(interval)
        raise TimeoutError(f"Vidu 视频任务超时: {task_id}")

    def _video_url(self, data: dict) -> str:
        creations = data.get("creations") or []
        return str(creations[0].get("url") or "") if creations else ""
