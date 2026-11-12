"""方舟 Seedance：保留旧版首帧协议，并适配 2.x 多模态参考与首尾帧。"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

import httpx

from config import settings
from services.providers.base import BaseAdapter, VideoCapabilities, VideoRequest, VideoResult
from services.providers.http_retry import request_with_retry
from services.providers.usage import CAPABILITY_VIDEO, UsageMetadata
from services.reference_asset_service import ReferenceAssetService
from services.security import (
    UploadLimitExceeded,
    atomic_write_bytes,
    download_remote_file,
)
from services.storage_service import StorageQuotaExceeded, StorageService


class ArkSeedanceVideoAdapter(BaseAdapter):
    capabilities = VideoCapabilities(
        reference_image=True,
        multiple_reference_images=False,
        reference_parameter="content",
        reference_weight_policy="text_only_policy",
        reference_role_parameter="content[].role",
        native_audio=False,
        dialogue_in_prompt=False,
        voice_consistent=False,
        camera_movement_prompt=True,
        fixed_duration=5,
        min_duration=5,
        max_duration=5,
        duration_step=5,
        reference_mode="first_frame_only",
        # 方舟内容生成接口允许 MB 级 base64 图片内联；参考图按此预算压缩，
        # 尽量保留首帧里的人脸、服装与材质细节。
        max_reference_inline_bytes=8 * 1024 * 1024,
    )

    def __init__(self, endpoint):
        super().__init__(endpoint)
        self.storage = StorageService()
        self.reference_assets = ReferenceAssetService()

    @classmethod
    def effective_capabilities(cls, model: str = ""):
        normalized = model.replace(".", "-").lower()
        if "seedance-2" not in normalized:
            return cls.capabilities
        version_25 = "seedance-2-5" in normalized
        return replace(
            cls.capabilities,
            fixed_duration=None,
            min_duration=4,
            max_duration=30 if version_25 else 15,
            duration_step=1,
            multiple_reference_images=True,
            max_reference_images=30 if version_25 else 9,
            last_frame_input=True,
            first_last_frame_interpolation=True,
            native_audio=True,
            dialogue_in_prompt=True,
            reference_mode="multi_reference",
            reference_video=True,
            reference_audio=True,
            max_reference_videos=10 if version_25 else 3,
            max_reference_audios=10 if version_25 else 3,
        )

    def _content(self, request: VideoRequest) -> list[dict]:
        caps = self.effective_capabilities(self.endpoint.model)
        content: list[dict] = [{"type": "text", "text": request.prompt}]
        image_count = video_count = audio_count = 0

        def append_media(value: str | None, media_type: str, role: str) -> None:
            nonlocal image_count, video_count, audio_count
            if not value:
                return
            if media_type == "image_url":
                if not value.startswith("asset://"):
                    value = self.reference_assets.to_image_url(value, max_bytes=caps.max_reference_inline_bytes)
                if not value:
                    raise RuntimeError("视频参考图缺失，或无法压缩到方舟请求体预算内")
                image_count += 1
            elif media_type == "video_url":
                video_count += 1
            else:
                audio_count += 1
            content.append({"type": media_type, media_type: {"url": value}, "role": role})

        append_media(request.reference_image, "image_url", "first_frame")
        if request.end_frame:
            if not caps.last_frame_input or not request.reference_image:
                raise ValueError("当前 Seedance 模型不支持此首尾帧请求，或缺少首帧")
            append_media(request.end_frame, "image_url", "last_frame")
        if caps.multiple_reference_images:
            seen = {request.reference_image, request.end_frame}
            for asset in request.reference_assets:
                value = asset.url or asset.source_path
                if asset.type in {"approved_storyboard_first_frame", "end_frame"} or value in seen:
                    continue
                role = asset.provider_type or asset.role or asset.type
                if role in {"reference_video", "reference_audio"}:
                    append_media(value, role.replace("reference_", "") + "_url", role)
                else:
                    append_media(value, "image_url", "reference_image")
                seen.add(value)
        elif any(a.type not in {"approved_storyboard_first_frame", "end_frame"} for a in request.reference_assets):
            raise ValueError("当前 Seedance 模型仅支持首帧参考")
        if request.reference_video or request.reference_audio:
            if not caps.reference_video:
                raise ValueError("当前 Seedance 模型不支持视频或音频参考")
            append_media(request.reference_video, "video_url", "reference_video")
            append_media(request.reference_audio, "audio_url", "reference_audio")
        modern = "seedance-2" in self.endpoint.model.replace(".", "-").lower()
        if modern:
            version_25 = "seedance-2-5" in self.endpoint.model.replace(".", "-").lower()
            if (
                image_count > caps.max_reference_images
                or video_count > (10 if version_25 else 3)
                or audio_count > (10 if version_25 else 3)
            ):
                raise ValueError("Seedance 参考素材数量超过当前模型上限")
            if not version_25 and audio_count and not (image_count or video_count):
                raise ValueError("Seedance 2.0 音频参考必须搭配图片或视频")
        return content

    def build_payload(self, request: VideoRequest, *, content: list[dict] | None = None) -> dict:
        caps = self.effective_capabilities(self.endpoint.model)
        if request.generate_audio and not caps.native_audio:
            raise ValueError("当前 Seedance 适配模式不支持原生音频，请使用独立 TTS")
        request_content = self._content(request) if content is None else content
        modern = "seedance-2" in self.endpoint.model.replace(".", "-").lower()
        task_type = request.task_type or "auto"
        if task_type not in {"auto", "edit", "extend"}:
            raise ValueError("Seedance task_type 仅支持 auto / edit / extend")
        has_video = any(item.get("type") == "video_url" for item in request_content)
        if task_type != "auto" and (not modern or not has_video):
            raise ValueError("Seedance 编辑或延长任务需要 2.x 模型和参考视频")
        duration = -1 if task_type == "edit" else request.duration
        if duration != -1 and modern and not (caps.min_duration <= duration <= caps.max_duration):
            raise ValueError(f"Seedance 时长必须为 {caps.min_duration:g}–{caps.max_duration:g} 秒")
        adaptive = modern and (
            task_type in {"edit", "extend"}
            or any(item.get("role") in {"first_frame", "last_frame"} for item in request_content)
        )
        ratio = "adaptive" if adaptive else request.ratio
        if modern and ratio not in {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "adaptive"}:
            raise ValueError("Seedance 不支持此画幅比例")
        resolution = self._normalized_resolution(request.resolution)
        allowed_resolutions = {"480p", "720p", "1080p"}
        if "seedance-2-0" in self.endpoint.model.replace(".", "-").lower():
            allowed_resolutions.add("4k")
        if modern and resolution not in allowed_resolutions:
            raise ValueError("当前 Seedance 模型不支持此分辨率")
        payload = {
            "model": self.endpoint.model,
            "content": request_content,
            "duration": duration,
            "ratio": ratio,
            "resolution": resolution,
            "return_last_frame": bool(self.endpoint.param("return_last_frame", True)),
            "watermark": bool(self.endpoint.param("watermark", False)),
        }
        if self._supports_generate_audio(self.endpoint.model):
            payload["generate_audio"] = bool(request.generate_audio)
        if modern:
            payload["omni_reference_task_type"] = task_type
            if request.seed is not None:
                payload["seed"] = request.seed
        return payload

    def usage_for_request(
        self,
        capability: str,
        request: VideoRequest | None = None,
        *,
        model: str = "",
    ) -> UsageMetadata:
        """Seedance 按秒计费：时长取请求里的有效时长（已按协议固定时长归一化）。"""

        auto_duration = bool(request and (request.duration == -1 or request.task_type == "edit"))
        return UsageMetadata(
            capability=CAPABILITY_VIDEO,
            provider=self.endpoint.protocol,
            model=model or self.endpoint.model,
            seconds=0 if auto_duration else float(getattr(request, "duration", 0) or 0),
            resolution=str(getattr(request, "resolution", "") or ""),
            known=not auto_duration,
            billable=True,
            source="unknown" if auto_duration else "request",
            extra={"ratio": str(getattr(request, "ratio", "") or "")},
        )

    async def generate(self, request: VideoRequest) -> VideoResult:
        content = self._content(request)
        payload_mode = self._reference_payload_mode(content)
        task = await self._create_task(
            request.prompt, request.duration, request.ratio, request.resolution, content, request=request
        )
        task_id = self._extract_task_id(task)
        data = await self._wait_for_task(task_id)

        video_path = request.output_video_path
        frame_path = request.output_frame_path
        has_frame = await self._download_outputs_to_files(data, video_path, frame_path, request.project_id)
        if not has_frame:
            await self._extract_last_frame(video_path, frame_path)

        if video_path.stat().st_size <= 4096:
            raise RuntimeError("Seedance 返回视频为空或过小")
        if not frame_path.exists() or frame_path.stat().st_size <= 1024:
            raise RuntimeError("Seedance 单帧画面保存失败")
        return VideoResult(
            video_path=str(video_path),
            frame_path=str(frame_path),
            native_audio=bool(request.generate_audio and self.effective_capabilities(self.endpoint.model).native_audio),
            payload_mode=payload_mode,
            task_id=task_id,
        )

    # --- 异步任务协议 ---

    async def _create_task(
        self,
        prompt: str,
        duration: int,
        ratio: str,
        resolution: str,
        content: list[dict] | None = None,
        *,
        request: VideoRequest | None = None,
    ) -> dict:
        request = request or VideoRequest(prompt=prompt, duration=duration, ratio=ratio, resolution=resolution)
        payload = self.build_payload(request, content=content)
        async with httpx.AsyncClient(timeout=60, trust_env=settings.PROVIDER_HTTP_TRUST_ENV) as client:
            response = await request_with_retry(
                client,
                "POST",
                self._tasks_url(),
                headers=self._headers(),
                json=payload,
                name="ark-seedance:create",
                attempts=1,
            )
        if response.status_code >= 400:
            raise RuntimeError(f"Seedance 创建任务失败: {response.status_code} {response.text[:1000]}")
        return response.json()

    async def _wait_for_task(self, task_id: str) -> dict:
        async with httpx.AsyncClient(timeout=60, trust_env=settings.PROVIDER_HTTP_TRUST_ENV) as client:
            for _ in range(max(1, int(self.endpoint.param("max_poll_attempts", 90)))):
                response = await request_with_retry(
                    client,
                    "GET",
                    f"{self._tasks_url()}/{quote(task_id, safe='')}",
                    headers=self._headers(),
                    name="ark-seedance:poll",
                )
                if response.status_code >= 400:
                    raise RuntimeError(f"Seedance 查询任务失败: {response.status_code} {response.text[:1000]}")
                data = response.json()
                status = str(
                    data.get("status") or data.get("task_status") or data.get("data", {}).get("status") or ""
                ).lower()
                if status in {"succeeded", "success", "completed", "done"}:
                    return data
                if status in {"failed", "cancelled", "canceled", "error"}:
                    raise RuntimeError(f"Seedance 任务失败: {data}")
                await asyncio.sleep(max(0.1, float(self.endpoint.param("poll_interval_seconds", 30))))
        raise TimeoutError(f"Seedance 任务超时: {task_id}")

    @staticmethod
    def _supports_generate_audio(model: str) -> bool:
        """Seedance 2.x 系列支持 generate_audio；1.0 系列不支持。

        1.5 Pro 已从模型列表下线，保留匹配仅为兼容历史端点配置。
        """

        normalized = model.replace(".", "-").lower()
        return "seedance-1-5" in normalized or "seedance-2" in normalized

    def _normalized_resolution(self, resolution: str) -> str:
        """按模型档位归一化分辨率。

        官方模型列表：Seedance 2.0 Fast / 2.0 Mini 仅提供 480p/720p 档位
        （2.5 与 2.0 标准版到 1080p，2.0 另有 4k），而项目默认分辨率为 1080p，
        直传会被方舟拒绝，统一降 720p。
        """

        value = str(resolution or "").strip().lower()
        normalized_model = self.endpoint.model.replace(".", "-").lower()
        limited = "seedance-2-0-fast" in normalized_model or "seedance-2-0-mini" in normalized_model
        if limited and value in {"1080p", "1080", "2k", "4k"}:
            return "720p"
        return value

    def _reference_payload_mode(self, content: list[dict]) -> str:
        roles = {str(item.get("role") or "") for item in content if item.get("type") != "text"}
        if {"first_frame", "last_frame"} <= roles:
            return "first_last_frame"
        if "first_frame" in roles:
            return "first_frame_reference"
        if roles:
            return "image_reference"
        return "text_only"

    def _extract_task_id(self, data: dict) -> str:
        for key in ("id", "task_id"):
            value = data.get(key) or data.get("data", {}).get(key)
            if value:
                return str(value)
        raise RuntimeError(f"Seedance 创建任务返回缺少任务 ID: {data}")

    # --- 产物下载与落盘 ---

    async def _download_outputs_to_files(self, data: dict, video_path: Path, frame_path: Path, project_id: str) -> bool:
        video_url = self._find_url(data, {"video_url", "video", "url"})
        frame_url = self._find_url(data, {"last_frame_url", "frame_url", "image_url", "cover_url"})
        video_b64 = self._find_b64(data, {"video_base64", "video_b64", "b64_json"})
        frame_b64 = self._find_b64(data, {"last_frame_base64", "frame_base64", "image_base64"})

        if video_b64:
            video_bytes = self._decode_b64(video_b64, "视频")
            self._write_media(project_id, video_path, video_bytes, minimum_size=4096)
        elif video_url:
            await self._download_url_to_path(project_id, video_url, video_path, minimum_size=4096)
        else:
            raise RuntimeError(f"Seedance 返回缺少视频 URL/base64: {data}")

        if frame_b64:
            self._write_media(project_id, frame_path, self._decode_b64(frame_b64, "单帧"), minimum_size=1024)
            return True
        elif frame_url:
            await self._download_url_to_path(project_id, frame_url, frame_path, minimum_size=1024)
            return True
        return False

    def _decode_b64(self, value: str, label: str) -> bytes:
        raw = str(value or "")
        # Base64 expands data by roughly 4/3. Reject oversized payloads before
        # allocating a potentially unbounded decoded buffer.
        if len(raw) > int(settings.MAX_REMOTE_MEDIA_BYTES * 4 / 3) + 16:
            raise RuntimeError(f"{label}数据超过大小限制")
        try:
            decoded = base64.b64decode(raw, validate=True)
        except Exception as exc:
            raise RuntimeError(f"{label}数据编码无效") from exc
        if len(decoded) > settings.MAX_REMOTE_MEDIA_BYTES:
            raise RuntimeError(f"{label}数据超过大小限制")
        return decoded

    async def _download_url_to_path(self, project_id: str, url: str, destination: Path, *, minimum_size: int) -> None:
        try:
            available = self.storage.ensure_project_capacity(project_id, replacing=destination)
            size = await download_remote_file(
                url,
                destination,
                max_bytes=min(settings.MAX_REMOTE_MEDIA_BYTES, available),
                timeout=180,
            )
        except (StorageQuotaExceeded, UploadLimitExceeded, ValueError, httpx.HTTPError) as exc:
            raise RuntimeError(f"下载远程媒体失败: {exc}") from exc
        if size < minimum_size:
            destination.unlink(missing_ok=True)
            raise RuntimeError("下载的远程媒体为空或过小")

    def _write_media(self, project_id: str, destination: Path, content: bytes, *, minimum_size: int) -> None:
        try:
            self.storage.ensure_project_capacity(project_id, len(content), replacing=destination)
        except StorageQuotaExceeded as exc:
            raise RuntimeError("项目媒体存储空间不足") from exc
        atomic_write_bytes(destination, content, minimum_size=minimum_size)

    async def _extract_last_frame(self, video_path: Path, frame_path: Path) -> None:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-sseof",
            "-0.1",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(frame_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=max(30, int(settings.FFMPEG_TIMEOUT_SECONDS)),
            )
        except asyncio.CancelledError:
            if proc.returncode is None:
                proc.kill()
            await proc.communicate()
            raise
        except TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise TimeoutError("提取 Seedance 单帧超时") from exc
        if proc.returncode != 0:
            raise RuntimeError(f"提取 Seedance 单帧失败: {stderr.decode('utf-8', errors='ignore')[-1000:]}")

    # --- 响应解析 ---

    def _find_url(self, value, keys: set[str]) -> str:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in keys and isinstance(item, str) and item.startswith(("http://", "https://")):
                    return item
                nested = self._find_url(item, keys)
                if nested:
                    return nested
        if isinstance(value, list):
            for item in value:
                nested = self._find_url(item, keys)
                if nested:
                    return nested
        return ""

    def _find_b64(self, value, keys: set[str]) -> str:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in keys and isinstance(item, str) and not item.startswith(("http://", "https://")):
                    return item.split(",", 1)[-1] if item.startswith("data:") else item
                nested = self._find_b64(item, keys)
                if nested:
                    return nested
        if isinstance(value, list):
            for item in value:
                nested = self._find_b64(item, keys)
                if nested:
                    return nested
        return ""

    def _headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.endpoint.api_key}", "Content-Type": "application/json"}
        if self.endpoint.auth_style == "api-key-header":
            headers["api-key"] = self.endpoint.api_key
        return headers

    def _tasks_url(self) -> str:
        base = (self.endpoint.base_url or "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
        base = base.replace("/api/plan/v3", "/api/v3")
        if base.endswith("/contents/generations/tasks"):
            return base
        return f"{base}/contents/generations/tasks"
