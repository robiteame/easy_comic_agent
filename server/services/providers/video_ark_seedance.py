"""火山方舟 Seedance 视频适配器（原 SeedanceVideoService 异步任务逻辑迁入）。

异步任务式：创建任务 → 轮询 → 下载视频 + 尾帧。产物为无声视频（对白由独立的
TTS 路径配音），capabilities.native_audio=False 供音频路由识别。
参考图模式为 first_frame_only：只接受一张已审核故事板首帧。
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import httpx

from config import settings
from services.providers.base import BaseAdapter, VideoCapabilities, VideoRequest, VideoResult
from services.providers.http_retry import request_with_retry
from services.providers.usage import CAPABILITY_VIDEO, UsageMetadata
from services.reference_asset_service import ReferenceAssetService
from services.security import (
    UploadLimitExceeded,
    atomic_write_bytes,
    download_remote_bytes,
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
        fixed_duration=5,
        reference_mode="first_frame_only",
        # 方舟内容生成接口允许 MB 级 base64 图片内联；参考图按此预算压缩，
        # 尽量保留首帧里的人脸、服装与材质细节。
        max_reference_inline_bytes=8 * 1024 * 1024,
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
        """Seedance 按秒计费：时长取请求里的有效时长（已按协议固定时长归一化）。"""

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
        content: list[dict] = [{"type": "text", "text": request.prompt}]
        if request.reference_image:
            # 服务层已按能力预算压缩；这里兜底覆盖直传大 data URL 的调用路径。
            reference_url = self.reference_assets.to_image_url(
                request.reference_image,
                max_bytes=self.capabilities.max_reference_inline_bytes or settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES,
            )
            if not reference_url:
                raise RuntimeError("视频首帧参考图缺失，或无法压缩到方舟请求体预算内")
            content.append(
                {"type": "image_url", "image_url": {"url": reference_url}, "role": "first_frame"}
            )
        payload_mode = self._reference_payload_mode(content)

        task = await self._create_task(request.prompt, request.duration, request.ratio, request.resolution, content)
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
            native_audio=False,
            payload_mode=payload_mode,
            task_id=task_id,
        )

    # --- 异步任务协议 ---

    async def _create_task(
        self, prompt: str, duration: int, ratio: str, resolution: str, content: list[dict] | None = None
    ) -> dict:
        request_content = content or [{"type": "text", "text": prompt}]
        payload = {
            "model": self.endpoint.model,
            "content": request_content,
            "duration": duration,
            "ratio": ratio,
            "resolution": self._normalized_resolution(resolution),
            # 保留尾帧返回：同场景下一镜的连续性参考依赖上一镜尾帧。
            "return_last_frame": True,
            "watermark": False,
        }
        # Seedance 1.5 Pro 及 2.x 系列「音画同生」默认开启（generate_audio=true）；
        # 本链路契约为无声视频 + 独立 TTS 配音，需显式关闭。1.0 系列不支持该参数，
        # 强校验下携带会报错，故仅在受支持的模型上发送。
        if self._supports_generate_audio(self.endpoint.model):
            payload["generate_audio"] = False
        async with httpx.AsyncClient(timeout=60, trust_env=settings.PROVIDER_HTTP_TRUST_ENV) as client:
            response = await request_with_retry(
                client,
                "POST",
                self._tasks_url(),
                headers=self._headers(),
                json=payload,
                name="ark-seedance:create",
            )
        if response.status_code >= 400:
            raise RuntimeError(f"Seedance 创建任务失败: {response.status_code} {response.text[:1000]}")
        return response.json()

    async def _wait_for_task(self, task_id: str) -> dict:
        async with httpx.AsyncClient(timeout=60, trust_env=settings.PROVIDER_HTTP_TRUST_ENV) as client:
            for _ in range(90):
                response = await request_with_retry(
                    client,
                    "GET",
                    f"{self._tasks_url()}/{task_id}",
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
                await asyncio.sleep(5)
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
        roles = {str(item.get("role") or "") for item in content if item.get("type") == "image_url"}
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
        except asyncio.TimeoutError as exc:
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
        return f"{base}/contents/generations/tasks"
