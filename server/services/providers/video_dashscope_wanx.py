"""阿里云百炼（DashScope）通义万相视频适配器。

异步任务式：``X-DashScope-Async`` 创建任务 → 轮询 ``/tasks/{id}`` → 下载视频
+ FFmpeg 提取尾帧。产物为无声视频（对白由独立的 TTS 路径配音），
capabilities.native_audio=False 供音频路由识别。图生视频模型（wan*-i2v-*）以
首帧参考图驱动，文生视频模型（wan*-t2v-*）纯文本驱动。

按模型代次走两种请求结构（端点相同，均按 2026-09 官方文档）：
- Wan 2.7 / 3.0（新一代 All-in-One）：``input.media`` 数组携带首帧
  （type=first_frame）；``parameters.resolution`` 用档位（480P/720P/1080P）。
  Wan 3.0 默认生成有声视频，显式 ``audio=false`` 维持无声契约；
- Wan 2.6 及更早（旧代）：``input.img_url`` 携带首帧；``parameters.resolution``
  同为档位取值（像素级 size 已从文档移除）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from config import settings
from services.providers.base import BaseAdapter, VideoCapabilities, VideoRequest, VideoResult
from services.providers.http_retry import request_with_retry
from services.providers.usage import CAPABILITY_VIDEO, UsageMetadata
from services.reference_asset_service import ReferenceAssetService
from services.security import (
    UploadLimitExceeded,
    download_remote_file,
)
from services.storage_service import StorageQuotaExceeded, StorageService

_DEFAULT_BASE = "https://dashscope.aliyuncs.com/api/v1"

# 视频分辨率档位：万相新旧代接口均已统一为 resolution 枚举，不再传像素级 size。
_RESOLUTION_TIERS = ("480P", "720P", "1080P")


class DashscopeWanxVideoAdapter(BaseAdapter):
    capabilities = VideoCapabilities(
        reference_image=True,
        native_audio=False,
        dialogue_in_prompt=False,
        voice_consistent=False,
        fixed_duration=None,
        reference_mode="first_frame_only",
    )

    def __init__(self, endpoint):
        super().__init__(endpoint)
        self.storage = StorageService()
        self.reference_assets = ReferenceAssetService()

    def usage_for_request(
        self,
        capability: str,
        request: VideoRequest | None = None,
        *,
        model: str = "",
    ) -> UsageMetadata:
        """万相按秒计费：时长取请求里的有效时长（服务层已按模型能力归一化）。"""

        return UsageMetadata(
            capability=CAPABILITY_VIDEO,
            provider=self.endpoint.protocol,
            model=model or self.endpoint.model,
            seconds=float(getattr(request, "duration", 0) or 0),
            resolution=str(getattr(request, "resolution", "") or ""),
            known=True,
            billable=True,
            source="request",
            extra={"ratio": str(getattr(request, "ratio", "") or "")},
        )

    async def generate(self, request: VideoRequest) -> VideoResult:
        payload_mode = "first_frame_reference" if request.reference_image else "text_only"

        task_id = await self._create_task(request)
        data = await self._wait_for_task(task_id)
        video_url = str((data.get("output") or {}).get("video_url") or "")
        if not video_url.startswith(("http://", "https://")):
            raise RuntimeError(f"百炼返回缺少视频 URL: {data}")

        video_path = request.output_video_path
        frame_path = request.output_frame_path
        await self._download_url_to_path(request.project_id, video_url, video_path, minimum_size=4096)
        await self._extract_last_frame(video_path, frame_path)

        if video_path.stat().st_size <= 4096:
            raise RuntimeError("百炼返回视频为空或过小")
        if not frame_path.exists() or frame_path.stat().st_size <= 1024:
            raise RuntimeError("百炼单帧画面保存失败")
        return VideoResult(
            video_path=str(video_path),
            frame_path=str(frame_path),
            native_audio=False,
            payload_mode=payload_mode,
            task_id=task_id,
        )

    # --- 异步任务协议 ---

    async def _create_task(self, request: VideoRequest) -> str:
        img_url = ""
        if request.reference_image:
            # 百炼网关对超过 ~200KB 的请求体直接重置连接，参考图必须先压进
            # 预算再内联；服务层已在构造 content 时压缩，这里兜底覆盖直传
            # 大 data URL 的调用路径。
            img_url = self.reference_assets.to_image_url(
                request.reference_image, max_bytes=settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES
            )
            if not img_url:
                raise RuntimeError("视频首帧参考图缺失，或无法压缩到请求体预算内")

        task_input: dict[str, Any] = {"prompt": request.prompt}
        if self._uses_media_input():
            # Wan 2.7 / 3.0：首帧放 input.media 数组。
            if img_url:
                task_input["media"] = [{"type": "first_frame", "url": img_url}]
        elif img_url:
            # Wan 2.6 及更早：首帧放 input.img_url。
            task_input["img_url"] = img_url

        parameters: dict[str, Any] = {
            "resolution": self._resolve_resolution(request.resolution),
            "duration": max(1, int(request.duration or 5)),
            "prompt_extend": True,
        }
        # Wan 3.0 默认输出有声视频；本链路契约为无声视频 + 独立 TTS 配音。
        if self._generates_audio_by_default():
            parameters["audio"] = False

        payload = {
            "model": self.endpoint.model,
            "input": task_input,
            "parameters": parameters,
        }
        headers = self._headers()
        headers["X-DashScope-Async"] = "enable"
        async with httpx.AsyncClient(timeout=60, trust_env=settings.PROVIDER_HTTP_TRUST_ENV) as client:
            response = await request_with_retry(
                client,
                "POST",
                self._create_url(),
                headers=headers,
                json=payload,
                name="dashscope-wanx:create",
            )
        if response.status_code >= 400:
            raise RuntimeError(f"百炼创建视频任务失败: {response.status_code} {response.text[:1000]}")
        data = response.json()
        task_id = str((data.get("output") or {}).get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"百炼创建任务返回缺少任务 ID: {data}")
        return task_id

    async def _wait_for_task(self, task_id: str) -> dict:
        async with httpx.AsyncClient(timeout=60, trust_env=settings.PROVIDER_HTTP_TRUST_ENV) as client:
            for _ in range(90):
                response = await request_with_retry(
                    client,
                    "GET",
                    f"{self._api_base()}/tasks/{task_id}",
                    headers=self._headers(),
                    name="dashscope-wanx:poll",
                )
                if response.status_code >= 400:
                    raise RuntimeError(f"百炼查询视频任务失败: {response.status_code} {response.text[:1000]}")
                data = response.json()
                status = str((data.get("output") or {}).get("task_status") or "").upper()
                if status == "SUCCEEDED":
                    return data
                if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                    raise RuntimeError(f"百炼视频任务失败: {data}")
                await asyncio.sleep(5)
        raise TimeoutError(f"百炼视频任务超时: {task_id}")

    def _resolve_resolution(self, resolution: str) -> str:
        """把请求分辨率归一化为万相档位（480P/720P/1080P）。"""

        value = str(resolution or "").strip().upper()
        if value in _RESOLUTION_TIERS:
            return value
        # 2K/4K 等更高档位或未知值统一回落到 1080P（万相最高档）。
        return "1080P"

    def _uses_media_input(self) -> bool:
        """Wan 2.7 / 3.0 新一代接口以 input.media 数组承载参考素材。"""

        model = (self.endpoint.model or "").strip().lower()
        return model.startswith("wan2.7") or model.startswith("wan3")

    def _generates_audio_by_default(self) -> bool:
        """Wan 3.0 默认输出有声视频（audio=true），需显式关闭。"""

        return (self.endpoint.model or "").strip().lower().startswith("wan3")

    # --- 产物下载与落盘 ---

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
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise TimeoutError("提取百炼单帧超时") from exc
        if proc.returncode != 0:
            raise RuntimeError(f"提取百炼单帧失败: {stderr.decode('utf-8', errors='ignore')[-1000:]}")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.endpoint.api_key}", "Content-Type": "application/json"}

    def _api_base(self) -> str:
        # 只保留 scheme://netloc：配置里可能粘贴的是 OpenAI 兼容地址
        # （…/compatible-mode/v1）或已带 /api/v1 的地址，万相异步任务
        # 固定挂在主机的 /api/v1 下，统一归一化避免拼出杂交路径。
        base = (self.endpoint.base_url or _DEFAULT_BASE).strip().rstrip("/")
        parsed = urlparse(base)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}/api/v1"
        return f"https://{base}/api/v1"

    def _create_url(self) -> str:
        return f"{self._api_base()}/services/aigc/video-generation/video-synthesis"
