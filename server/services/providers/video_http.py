"""Native video protocols share transport and safe, immediate output persistence."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from config import settings
from services.providers.base import BaseAdapter, VideoCapabilities, VideoRequest, VideoResult
from services.providers.http_retry import request_with_retry
from services.providers.usage import CAPABILITY_VIDEO, UsageMetadata
from services.reference_asset_service import ReferenceAssetService
from services.security import UploadLimitExceeded, download_remote_file
from services.storage_service import StorageQuotaExceeded, StorageService


class VideoHTTPAdapter(BaseAdapter):
    capabilities = VideoCapabilities()

    def __init__(self, endpoint):
        super().__init__(endpoint)
        self.storage = StorageService()
        self.reference_assets = ReferenceAssetService()

    def validate_request(self, request: VideoRequest) -> None:
        caps = self.effective_capabilities(self.endpoint.model)
        duration = float(request.duration)
        if not math.isfinite(duration) or not duration.is_integer():
            raise ValueError("视频时长必须为整数秒")
        if caps.duration_options and duration not in caps.duration_options:
            raise ValueError(f"{self.endpoint.protocol} 时长仅支持 {caps.duration_options} 秒")
        if caps.min_duration is not None and duration < caps.min_duration:
            raise ValueError(f"视频时长不能小于 {caps.min_duration:g} 秒")
        if caps.max_duration is not None and duration > caps.max_duration:
            raise ValueError(f"视频时长不能大于 {caps.max_duration:g} 秒")
        if request.end_frame and not caps.last_frame_input:
            raise ValueError(f"{self.endpoint.protocol} 当前模型不支持尾帧")
        if request.reference_image and not caps.reference_image:
            raise ValueError(f"{self.endpoint.protocol} 当前模型不支持首帧")
        image_assets = []
        video_assets = []
        audio_assets = []
        for asset in request.reference_assets:
            role = asset.provider_type or asset.role or asset.type
            if role in {"reference_video", "video_url", "first_clip"}:
                video_assets.append(asset)
            elif role in {"reference_audio", "audio_url", "driving_audio"}:
                audio_assets.append(asset)
            elif asset.type not in {"approved_storyboard_first_frame", "end_frame"}:
                image_assets.append(asset)
        if image_assets and not caps.multiple_reference_images:
            raise ValueError(f"{self.endpoint.protocol} 当前适配器不支持多参考图")
        if caps.max_reference_images and len(image_assets) > caps.max_reference_images:
            raise ValueError(f"参考图最多 {caps.max_reference_images} 张")
        if video_assets and not caps.reference_video:
            raise ValueError(f"{self.endpoint.protocol} 当前适配器不支持参考视频")
        if audio_assets and not caps.reference_audio:
            raise ValueError(f"{self.endpoint.protocol} 当前适配器不支持参考音频")
        if request.reference_video and not caps.reference_video:
            raise ValueError(f"{self.endpoint.protocol} 当前适配器不支持参考视频")
        if request.reference_audio and not caps.reference_audio:
            raise ValueError(f"{self.endpoint.protocol} 当前适配器不支持参考音频")
        if request.generate_audio and not caps.native_audio:
            raise ValueError(f"{self.endpoint.protocol} 当前模型不支持原生音频")

    async def generate(self, request: VideoRequest) -> VideoResult:
        self.validate_request(request)
        if request.output_video_path is None or request.output_frame_path is None:
            raise ValueError("视频与尾帧输出路径不能为空")
        task_id = await self._submit(request)
        data = await self._poll_task(task_id)
        await self._save_video(request, data, task_id)
        await self._extract_last_frame(request.output_video_path, request.output_frame_path)
        if not request.output_frame_path.exists() or request.output_frame_path.stat().st_size <= 1024:
            raise RuntimeError("视频尾帧保存失败")
        mode = (
            "first_last_frame"
            if request.end_frame
            else "first_frame_reference"
            if request.reference_image
            else "text_only"
        )
        if any(a.type not in {"approved_storyboard_first_frame", "end_frame"} for a in request.reference_assets):
            mode = "multi_reference"
        return VideoResult(
            video_path=str(request.output_video_path),
            frame_path=str(request.output_frame_path),
            native_audio=self._native_audio(request),
            payload_mode=mode,
            task_id=task_id,
        )

    async def _submit(self, request: VideoRequest) -> str:
        raise NotImplementedError

    async def _poll_task(self, task_id: str) -> dict:
        raise NotImplementedError

    def _video_url(self, data: dict) -> str:
        raise NotImplementedError

    def _native_audio(self, request: VideoRequest) -> bool:
        return bool(request.generate_audio)

    async def _save_video(self, request: VideoRequest, data: dict, task_id: str) -> None:
        url = self._video_url(data)
        if not url.startswith(("http://", "https://")):
            raise RuntimeError(f"{self.endpoint.protocol} 返回缺少可下载的视频 URL")
        await self._download_url_to_path(request.project_id, url, request.output_video_path, minimum_size=4096)

    async def _request_json(self, method: str, url: str, **kwargs) -> dict:
        kwargs.setdefault("headers", self._headers())
        # Creation POSTs are billable and usually not idempotent. An ambiguous
        # transport failure must not create another paid generation silently.
        async with httpx.AsyncClient(timeout=60, trust_env=settings.PROVIDER_HTTP_TRUST_ENV) as client:
            response = await request_with_retry(
                client,
                method,
                url,
                attempts=3 if method.upper() == "GET" else 1,
                name=self.endpoint.protocol,
                **kwargs,
            )
        if response.status_code >= 400:
            raise RuntimeError(f"{self.endpoint.protocol} API 返回 HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"{self.endpoint.protocol} API 返回无效 JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"{self.endpoint.protocol} API 返回结构无效")
        return data

    def _headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.endpoint.api_key}", "Content-Type": "application/json"}
        if self.endpoint.auth_style == "api-key-header":
            headers = {"api-key": self.endpoint.api_key, "Content-Type": "application/json"}
        return headers

    def _api_url(self, default_base: str, path: str) -> str:
        base = (self.endpoint.base_url or default_base).rstrip("/")
        suffix = path.strip("/")
        parsed = urlsplit(base)
        base_path = parsed.path.strip("/").split("/") if parsed.path.strip("/") else []
        target = suffix.split("/")
        if target[0] in {"v1", "v2", "ent", "openapi", "dream-machine"} and target[0] in base_path:
            # A pasted create URL must not remain in polling/upload URLs.
            prefix = base_path[: base_path.index(target[0])]
            return urlunsplit((parsed.scheme, parsed.netloc, "/" + "/".join(prefix + target), "", ""))
        # Accept host, version root, or complete endpoint as Base URL.
        base_parts = base.split("/")
        path_parts = suffix.split("/")
        for count in range(len(path_parts), 0, -1):
            if base_parts[-count:] == path_parts[:count]:
                return base + ("/" + "/".join(path_parts[count:]) if path_parts[count:] else "")
        return f"{base}/{suffix}"

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
            raise RuntimeError(f"下载视频失败: {exc}") from exc
        if size < minimum_size:
            destination.unlink(missing_ok=True)
            raise RuntimeError("下载的视频为空或过小")

    async def _extract_last_frame(self, video_path: Path, frame_path: Path) -> None:
        frame_path.parent.mkdir(parents=True, exist_ok=True)
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
                proc.communicate(), timeout=max(30, int(settings.FFMPEG_TIMEOUT_SECONDS))
            )
        except (asyncio.CancelledError, TimeoutError):
            if proc.returncode is None:
                proc.kill()
            await proc.communicate()
            raise
        if proc.returncode != 0:
            raise RuntimeError(f"提取视频尾帧失败: {stderr.decode('utf-8', errors='ignore')[-1000:]}")

    def usage_for_request(self, capability: str, request=None, *, model: str = "") -> UsageMetadata:
        return UsageMetadata(
            capability=CAPABILITY_VIDEO,
            provider=self.endpoint.protocol,
            model=model or self.endpoint.model,
            seconds=float(getattr(request, "duration", 0) or 0),
            resolution=str(getattr(request, "resolution", "") or ""),
            known=True,
            billable=True,
            source="request",
        )
