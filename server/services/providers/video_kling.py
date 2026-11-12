"""可灵官方视频接口：AK/SK JWT、首尾帧及异步任务状态。"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from dataclasses import replace
from urllib.parse import quote

from services.providers.base import VideoCapabilities, VideoRequest
from services.providers.video_http import VideoHTTPAdapter

_DEFAULT_BASE = "https://api-beijing.klingai.com"


class KlingVideoAdapter(VideoHTTPAdapter):
    capabilities = VideoCapabilities(
        reference_image=True,
        last_frame_input=True,
        first_last_frame_interpolation=True,
        reference_parameter="image",
        reference_role_parameter="image_tail",
        reference_mode="first_frame_only",
        camera_movement_prompt=True,
        min_duration=5,
        max_duration=10,
        duration_options=(5, 10),
    )

    @classmethod
    def effective_capabilities(cls, model: str = ""):
        normalized = str(model or "").lower().replace(".", "-")
        if "v4" in normalized or "4-0" in normalized:
            flash = "flash" in normalized
            return replace(
                cls.capabilities,
                min_duration=3,
                max_duration=20 if flash else 30,
                duration_options=(),
                native_audio=True,
                dialogue_in_prompt=True,
                last_frame_input=not flash,
                first_last_frame_interpolation=not flash,
            )
        if "v3" in normalized or "3-0" in normalized:
            turbo = "turbo" in normalized
            return replace(
                cls.capabilities,
                min_duration=3,
                max_duration=15,
                duration_options=(),
                native_audio=True,
                dialogue_in_prompt=True,
                last_frame_input=not turbo,
                first_last_frame_interpolation=not turbo,
            )
        return cls.capabilities

    def _headers(self) -> dict[str, str]:
        # A short-lived JWT is regenerated for every request, including polls.
        access_key, separator, secret_key = self.endpoint.api_key.partition(":")
        if not separator or not access_key or not secret_key:
            raise ValueError("Kling API Key 须按 AccessKey:SecretKey 配置")
        now = int(time.time())

        def encode(value: dict) -> str:
            raw = json.dumps(value, separators=(",", ":")).encode()
            return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

        header = encode({"alg": "HS256", "typ": "JWT"})
        claims = encode({"iss": access_key, "exp": now + 1800, "nbf": now - 5})
        signing_input = f"{header}.{claims}"
        signature = hmac.new(secret_key.encode(), signing_input.encode(), hashlib.sha256).digest()
        token = signing_input + "." + base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _image(self, value: str) -> str:
        image = self.reference_assets.to_image_url(value, max_bytes=10 * 1024 * 1024)
        if not image:
            raise ValueError("Kling 参考图片无法读取")
        if image.startswith("data:"):
            return image.partition(",")[2]
        return image

    def _task_kind(self, request: VideoRequest) -> str:
        return "image2video" if request.reference_image else "text2video"

    def build_payload(self, request: VideoRequest) -> dict:
        self.validate_request(request)
        if request.task_type != "auto":
            raise ValueError("Kling 当前适配器仅支持生成任务，未确认编辑/续写接口")
        if request.end_frame and not request.reference_image:
            raise ValueError("Kling 首尾帧任务必须同时提供首帧")
        if not self.endpoint.model:
            raise ValueError("请配置 Kling model_name；4.0 的模型 ID 应以控制台实际值为准")
        if len(request.prompt) > 2000:
            raise ValueError("Kling prompt 超过 2000 字符")
        if any(
            asset.type not in {"approved_storyboard_first_frame", "end_frame"} for asset in request.reference_assets
        ):
            raise ValueError("Kling 多模态/多参考任务字段尚未确认，请使用已支持的首帧或首尾帧接口")
        mode = str(
            self.endpoint.param(
                "mode",
                "4k"
                if request.resolution.lower() == "4k"
                else "pro"
                if request.resolution.lower() == "1080p"
                else "std",
            )
        )
        if mode not in {"std", "pro", "4k"}:
            raise ValueError("Kling mode 只能为 std/pro/4k")
        if request.end_frame and "v2-6" in self.endpoint.model.lower() and mode != "pro":
            raise ValueError("Kling 2.6 首尾帧仅支持 1080P 无声模式（mode=pro）")
        payload = {
            "model_name": self.endpoint.model,
            "prompt": request.prompt,
            "duration": str(request.duration),
            "mode": mode,
        }
        if request.reference_image:
            payload["image"] = self._image(request.reference_image)
        elif request.ratio:
            payload["aspect_ratio"] = request.ratio
        if request.end_frame:
            payload["image_tail"] = self._image(request.end_frame)
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        if self.effective_capabilities(self.endpoint.model).native_audio:
            payload["sound"] = "on" if request.generate_audio else "off"
        cfg_scale = self.endpoint.param("cfg_scale")
        if cfg_scale is not None:
            if not 0 <= float(cfg_scale) <= 1:
                raise ValueError("Kling cfg_scale 必须在 0–1 之间")
            payload["cfg_scale"] = float(cfg_scale)
        for name in ("camera_control", "multi_shot", "element_list", "voice_list", "callback_url", "external_task_id"):
            value = self.endpoint.param(name)
            if value is not None:
                payload[name] = value
        return payload

    @staticmethod
    def _check_response(data: dict) -> None:
        if data.get("code") not in (None, 0, "0"):
            raise RuntimeError(f"Kling API 错误 {data.get('code')}: {data.get('message', '任务失败')}")

    async def _submit(self, request: VideoRequest) -> str:
        payload = self.build_payload(request)
        kind = self._task_kind(request)
        data = await self._request_json("POST", self._api_url(_DEFAULT_BASE, f"/v1/videos/{kind}"), json=payload)
        self._check_response(data)
        task_id = str((data.get("data") or {}).get("task_id") or "")
        if not task_id:
            raise RuntimeError("Kling 创建任务未返回 task_id")
        # task IDs do not encode the endpoint. Persisting this on the adapter is
        # safe because each generation creates a new adapter instance.
        self._submitted_kind = kind
        return task_id

    async def _poll_task(self, task_id: str) -> dict:
        kind = getattr(self, "_submitted_kind", "image2video")
        attempts = max(1, int(self.endpoint.param("max_poll_attempts", 180)))
        interval = max(0.1, float(self.endpoint.param("poll_interval_seconds", 5)))
        for index in range(attempts):
            data = await self._request_json(
                "GET", self._api_url(_DEFAULT_BASE, f"/v1/videos/{kind}/{quote(task_id, safe='')}")
            )
            self._check_response(data)
            task = data.get("data") or {}
            state = str(task.get("task_status") or "").lower()
            if state in {"succeed", "succeeded", "success", "completed"}:
                return data
            if state in {"failed", "canceled", "cancelled"}:
                raise RuntimeError(f"Kling 视频任务失败: {task.get('task_status_msg') or state}")
            if index + 1 < attempts:
                await asyncio.sleep(interval)
        raise TimeoutError(f"Kling 视频任务超时: {task_id}")

    def _video_url(self, data: dict) -> str:
        videos = ((data.get("data") or {}).get("task_result") or {}).get("videos") or []
        return str(videos[0].get("url") or "") if videos else ""
