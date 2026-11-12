"""MiniMax H3 官方 content/role 协议。旧 Hailuo 请求形状不混用。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from urllib.parse import quote, urlparse

from services.providers.base import VideoCapabilities, VideoRequest
from services.providers.video_http import VideoHTTPAdapter

_DEFAULT_BASE = "https://api.minimax.cn"


class MiniMaxVideoAdapter(VideoHTTPAdapter):
    capabilities = VideoCapabilities(
        reference_image=True,
        multiple_reference_images=True,
        max_reference_images=9,
        reference_parameter="content",
        reference_role_parameter="content[].role",
        reference_mode="multi_reference",
        last_frame_input=True,
        first_last_frame_interpolation=True,
        reference_video=True,
        reference_audio=True,
        max_reference_videos=3,
        max_reference_audios=3,
        native_audio=True,
        dialogue_in_prompt=True,
        camera_movement_prompt=True,
        min_duration=4,
        max_duration=15,
    )

    @classmethod
    def effective_capabilities(cls, model: str = ""):
        if str(model).lower().endswith("-max"):
            return replace(cls.capabilities, min_duration=5)
        return cls.capabilities

    @staticmethod
    def _public_url(value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("MiniMax 素材须为可公开访问的 HTTP(S) URL，请先上传本地文件")
        return value

    def build_payload(self, request: VideoRequest) -> dict:
        self.validate_request(request)
        if request.task_type not in {"auto", "edit"}:
            raise ValueError("MiniMax 当前适配器仅支持 auto/edit 任务")
        if request.task_type == "edit" and not request.reference_video:
            raise ValueError("MiniMax edit 任务须提供 reference_video 作为 base_video")
        model = self.endpoint.model or "MiniMax-H3"
        if "hailuo" in model.lower():
            raise ValueError("旧版 Hailuo 使用另一套 first_frame_image 协议；当前适配器请选择 MiniMax-H3/H3-Max")
        if len(request.prompt) > 7000:
            raise ValueError("MiniMax prompt 超过 7000 字符")
        content = [{"type": "text", "text": request.prompt}]
        counts = {"image": 0, "video": 0, "audio": 0}
        seen: set[tuple[str, str]] = set()

        def append(kind: str, role: str, value: str) -> None:
            if not value or (role, value) in seen:
                return
            seen.add((role, value))
            url = self._public_url(value)
            content.append({"type": f"{kind}_url", f"{kind}_url": {"url": url}, "role": role})
            counts[kind] += 1

        if request.reference_image:
            append("image", "first_frame", request.reference_image)
        if request.end_frame:
            append("image", "last_frame", request.end_frame)
        for asset in request.reference_assets:
            if asset.type in {"approved_storyboard_first_frame", "end_frame"}:
                continue
            role = asset.provider_type or asset.role or asset.type
            kind = (
                "video"
                if role in {"reference_video", "base_video"}
                else "audio"
                if role == "reference_audio"
                else "image"
            )
            if kind == "image":
                role = "reference_image"
            append(kind, role, asset.url or asset.source_path)
        if request.reference_video:
            append("video", "base_video" if request.task_type == "edit" else "reference_video", request.reference_video)
        if request.reference_audio:
            append("audio", "reference_audio", request.reference_audio)
        if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3 or sum(counts.values()) > 12:
            raise ValueError("MiniMax 最多 9 张图、3 段视频、3 段音频，混合素材总计最多 12 个")
        resolution = str(self.endpoint.param("resolution", request.resolution)).upper()
        # The app's common 720p and 1080p choices map to the model's named tiers.
        resolution = {"720P": "768P", "1080P": "2K"}.get(resolution, resolution)
        valid_resolutions = {"480P", "768P"} if model.lower().endswith("-max") else {"768P", "2K"}
        if resolution not in valid_resolutions:
            raise ValueError(f"{model} 不支持分辨率 {resolution}；可选 {', '.join(sorted(valid_resolutions))}")
        ratio = "adaptive" if counts["image"] or counts["video"] else request.ratio
        if ratio == "adaptive" and not sum(counts.values()):
            raise ValueError("MiniMax 纯文生视频必须指定固定比例")
        return {
            "model": model,
            "content": content,
            "duration": int(request.duration),
            "resolution": resolution,
            "ratio": ratio,
        }

    @staticmethod
    def _check_response(data: dict) -> None:
        error = data.get("base_resp") or {}
        if error.get("status_code") not in (None, 0, "0"):
            raise RuntimeError(f"MiniMax API 错误 {error.get('status_code')}: {error.get('status_msg', '任务失败')}")

    async def _submit(self, request: VideoRequest) -> str:
        data = await self._request_json(
            "POST", self._api_url(_DEFAULT_BASE, "/v2/video_generation"), json=self.build_payload(request)
        )
        self._check_response(data)
        task_id = str(data.get("task_id") or (data.get("data") or {}).get("task_id") or "")
        if not task_id:
            raise RuntimeError("MiniMax 创建任务未返回 task_id")
        return task_id

    async def _poll_task(self, task_id: str) -> dict:
        attempts = max(1, int(self.endpoint.param("max_poll_attempts", 180)))
        interval = max(0.1, float(self.endpoint.param("poll_interval_seconds", 10)))
        for index in range(attempts):
            data = await self._request_json(
                "GET", self._api_url(_DEFAULT_BASE, f"/v2/query/video_generation/{quote(task_id, safe='')}")
            )
            self._check_response(data)
            state = str(data.get("status") or (data.get("data") or {}).get("status") or "").lower()
            if state in {"success", "succeeded", "completed"}:
                return data
            if state in {"fail", "failed", "error", "canceled", "cancelled"}:
                raise RuntimeError(f"MiniMax 视频任务失败: {data.get('message') or state}")
            if index + 1 < attempts:
                await asyncio.sleep(interval)
        raise TimeoutError(f"MiniMax 视频任务超时: {task_id}")

    def _native_audio(self, request: VideoRequest) -> bool:
        # H3 has no documented audio disable flag: report its actual output.
        return True

    def _video_url(self, data: dict) -> str:
        content = data.get("content") or (data.get("data") or {}).get("content") or {}
        if isinstance(content, dict):
            return str(content.get("url") or "")
        if isinstance(content, list):
            for item in content:
                if item.get("type") in {"video", "video_url"}:
                    value = item.get("url") or item.get("video_url")
                    return str(value.get("url") or "") if isinstance(value, dict) else str(value or "")
        return ""
