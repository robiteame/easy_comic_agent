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

r2v（参考生视频，如 wan2.7-r2v）的 media 契约：参考图像/参考视频至少传入
1 个，仅传首帧会被服务商判 InvalidParameter（任务创建成功但调度即失败）。
已审核故事板首帧会同时以 first_frame + reference_image 重复传入——官方推荐
「首帧已含主体时搭配主体参考强化一致性」的用法，首帧驱动契约保持不变。
wan2.7 默认生成有声视频且无 audio 参数可关，视频自带音轨由成片混音阶段
按「无声视频 + 独立 TTS」契约丢弃（对白轨优先，无对白补静音）。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
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
        multiple_reference_images=False,
        max_reference_images=4,
        reference_parameter="input.media",
        reference_weight_policy="text_only_policy",
        reference_role_parameter="input.media[].type",
        native_audio=False,
        dialogue_in_prompt=False,
        voice_consistent=False,
        camera_movement_prompt=True,
        fixed_duration=None,
        min_duration=5,
        max_duration=10,
        duration_step=5,
        reference_mode="model_conditional",
    )

    def __init__(self, endpoint):
        super().__init__(endpoint)
        self.storage = StorageService()
        self.reference_assets = ReferenceAssetService()

    @classmethod
    def effective_capabilities(cls, model: str = ""):
        """r2v 模型支持多参考图；普通 i2v/t2v 仍只能驱动首帧。"""

        caps = cls.capabilities
        if "-r2v" in str(model or "").strip().lower():
            return replace(
                caps,
                multiple_reference_images=True,
                reference_mode="multi_reference",
            )
        return replace(caps, reference_mode="first_frame_only")

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
                task_input["media"] = self._media_entries(img_url, request)
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
        """把请求分辨率归一化为万相档位。

        全系覆盖 480P/720P/1080P，但 Wan 2.6 / 2.7 仅提供 720P/1080P 档位
        （r2v / i2v 官方文档一致），480P 直传会被判 InvalidParameter，
        统一降 720P；2K/4K 等更高档位或未知值回落到 1080P（万相最高档）。
        """

        value = str(resolution or "").strip().upper()
        if value not in _RESOLUTION_TIERS:
            return "1080P"
        if value == "480P" and self._limited_resolution_model():
            return "720P"
        return value

    def _limited_resolution_model(self) -> bool:
        """Wan 2.6 / 2.7 系列仅提供 720P/1080P 档位（Wan 2.5/3.0 支持 480P）。"""

        model = (self.endpoint.model or "").strip().lower()
        return model.startswith("wan2.6") or model.startswith("wan2.7")

    def _media_entries(self, img_url: str, request: VideoRequest | None = None) -> list[dict[str, str]]:
        """构造 input.media 素材数组（首帧驱动 + r2v 多参考素材契约）。"""

        media: list[dict[str, str]] = [{"type": "first_frame", "url": img_url}]
        if not self._is_reference_to_video_model():
            return media
        seen: set[str] = set()
        for asset in getattr(request, "reference_assets", None) or []:
            if getattr(asset, "type", "") == "approved_storyboard_first_frame":
                continue
            url = self.reference_assets.to_image_url(
                getattr(asset, "url", "") or getattr(asset, "source_path", ""),
                max_bytes=settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES,
            )
            if not url or url == img_url or url in seen:
                continue
            seen.add(url)
            media.append({"type": getattr(asset, "provider_type", "") or "reference_image", "url": url})
        if len(media) == 1:
            # r2v 至少需要一个 reference_image；没有额外素材时，首帧只以
            # generic reference_image 重复一次，manifest 会如实记录重复项。
            media.append({"type": "reference_image", "url": img_url})
        return media[: 1 + max(0, int(self.capabilities.max_reference_images or 4))]

    def _is_reference_to_video_model(self) -> bool:
        """r2v（参考生视频）模型，含带日期后缀的变体（wan2.7-r2v-2026-06-12）。"""

        return "-r2v" in (self.endpoint.model or "").strip().lower()

    def _uses_media_input(self) -> bool:
        """Wan 2.7 / 3.0 新一代接口以 input.media 数组承载参考素材。"""

        model = (self.endpoint.model or "").strip().lower()
        return model.startswith("wan2.7") or model.startswith("wan3")

    def _generates_audio_by_default(self) -> bool:
        """Wan 3.0 默认输出有声视频（audio=true），需显式关闭。

        wan2.7 同样默认有声，但已移除 audio 参数无法关闭；其自带音轨由成片
        混音按「无声视频 + 独立 TTS」契约丢弃，无需在此处理。
        """

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
