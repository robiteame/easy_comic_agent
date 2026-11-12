"""Luma Dream Machine native video and generation-keyframe protocol."""

from __future__ import annotations

import asyncio
from urllib.parse import quote

from services.providers.base import VideoCapabilities, VideoRequest
from services.providers.video_http import VideoHTTPAdapter

_BASE = "https://api.lumalabs.ai"
_ROOT = "/dream-machine/v1"


class LumaVideoAdapter(VideoHTTPAdapter):
    capabilities = VideoCapabilities(
        reference_image=True,
        last_frame_input=True,
        first_last_frame_interpolation=True,
        reference_parameter="keyframes",
        reference_role_parameter="keyframes.frame0/frame1",
        reference_mode="first_last_frame",
        reference_video=True,
        max_reference_videos=1,
        min_duration=5,
        max_duration=9,
        duration_options=(5, 9),
        camera_movement_prompt=True,
    )

    @staticmethod
    def _keyframe(value: str) -> dict:
        if value.startswith(("http://", "https://")):
            return {"type": "image", "url": value}
        if value.startswith("generation:") and value.removeprefix("generation:"):
            return {"type": "generation", "id": value.removeprefix("generation:")}
        raise ValueError("Luma 关键帧需要公网图片 URL 或 generation:<任务ID>")

    def build_payload(self, request: VideoRequest) -> dict:
        self.validate_request(request)
        if request.task_type not in {"auto", "extend"}:
            raise ValueError("当前 Luma 适配器支持生成或 generation ID 续写，不支持视频 modify 编辑")
        if request.task_type == "extend" and not request.reference_video:
            raise ValueError("Luma 视频续写需要历史 generation ID")
        if request.ratio not in {"1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"}:
            raise ValueError("Luma 不支持此画幅比例")
        if request.resolution.lower() not in {"540p", "720p", "1080p", "4k"}:
            raise ValueError("Luma 不支持此分辨率")
        if any(a.type not in {"approved_storyboard_first_frame", "end_frame"} for a in request.reference_assets):
            raise ValueError("Luma 视频接口未声明主体多图参考参数")
        frames = {}
        if request.reference_image:
            frames["frame0"] = self._keyframe(request.reference_image)
        if request.reference_video:
            if request.reference_image:
                raise ValueError("Luma 首帧与续写 generation ID 不可同时设置")
            # Luma does not accept arbitrary reference-video URLs. A previous
            # generation ID is the documented continuation/interpolation input.
            if request.reference_video.startswith(("http:", "https:", "data:")):
                raise ValueError("Luma 视频续写需要历史 generation ID，不接受参考视频 URL")
            generation_id = request.reference_video.removeprefix("generation:")
            if not generation_id:
                raise ValueError("Luma generation ID 不能为空")
            frames["frame0"] = {"type": "generation", "id": generation_id}
        if request.end_frame:
            frames["frame1"] = self._keyframe(request.end_frame)
        payload = {
            "model": self.endpoint.model or "ray-2",
            "prompt": request.prompt,
            "aspect_ratio": request.ratio,
            "resolution": request.resolution.lower(),
            "duration": f"{request.duration}s",
            "loop": bool(self.endpoint.param("loop", False)),
        }
        if frames:
            payload["keyframes"] = frames
        for name in ("concepts", "callback_url"):
            if self.endpoint.param(name) is not None:
                payload[name] = self.endpoint.param(name)
        return payload

    async def _submit(self, request: VideoRequest) -> str:
        data = await self._request_json(
            "POST", self._api_url(_BASE, f"{_ROOT}/generations/video"), json=self.build_payload(request)
        )
        task_id = str(data.get("id") or "")
        if not task_id:
            raise RuntimeError("Luma 创建任务返回缺少 ID")
        return task_id

    async def _poll_task(self, task_id: str) -> dict:
        for _ in range(max(1, int(self.endpoint.param("max_poll_attempts", 120)))):
            data = await self._request_json(
                "GET", self._api_url(_BASE, f"{_ROOT}/generations/{quote(task_id, safe='')}")
            )
            state = str(data.get("state") or "").lower()
            if state == "completed":
                return data
            if state in {"failed", "cancelled", "canceled"}:
                raise RuntimeError(f"Luma 任务失败: {data.get('failure_reason') or state}")
            await asyncio.sleep(max(0.1, float(self.endpoint.param("poll_interval_seconds", 10))))
        raise TimeoutError("Luma 视频生成超时")

    def _video_url(self, data: dict) -> str:
        return str((data.get("assets") or {}).get("video") or "")
