"""Google Vertex Veo predictLongRunning/fetchPredictOperation API.

Omni's generateContent schema is intentionally not treated as Veo's schema.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import uuid
from dataclasses import replace
from urllib.parse import quote, urlparse

import httpx

from config import settings
from services.providers.base import ReferenceAsset, VideoCapabilities, VideoRequest
from services.providers.video_http import VideoHTTPAdapter
from services.security import UploadLimitExceeded, atomic_write_bytes, download_remote_bytes


class GoogleVeoVideoAdapter(VideoHTTPAdapter):
    capabilities = VideoCapabilities(
        reference_image=True,
        multiple_reference_images=True,
        max_reference_images=3,
        reference_parameter="instances[0].referenceImages",
        reference_role_parameter="instances[0].referenceImages[].referenceType",
        last_frame_input=True,
        first_last_frame_interpolation=True,
        native_audio=True,
        dialogue_in_prompt=True,
        reference_mode="multi_reference",
        min_duration=4,
        max_duration=8,
        duration_options=(4, 6, 8),
        duration_step=2,
        camera_movement_prompt=True,
        max_reference_inline_bytes=20 * 1024 * 1024,
    )

    @classmethod
    def effective_capabilities(cls, model: str = ""):
        if model and "3.1" not in model and "3-1" not in model:
            return replace(
                cls.capabilities,
                multiple_reference_images=False,
                max_reference_images=0,
                reference_mode="first_frame_only",
            )
        return cls.capabilities

    def _model_url(self) -> str:
        model = self.endpoint.model or "veo-3.1-generate-001"
        if "omni" in model.lower():
            raise ValueError("Gemini Omni 使用 generateContent，文档未确认其 REST 字段，不能套用 Vertex Veo 协议")
        if "preview" in model.lower():
            raise ValueError("Veo 预览模型已停用，请配置 Vertex GA 模型 veo-3.1-generate-001")
        location = str(self.endpoint.param("location", "us-central1"))
        base = (self.endpoint.base_url or f"https://{location}-aiplatform.googleapis.com").rstrip("/")
        if "/models/" in base:
            return base.split(":predictLongRunning", 1)[0].split(":fetchPredictOperation", 1)[0]
        project = str(self.endpoint.param("project_id", "") or self.endpoint.param("project", ""))
        if not project:
            raise ValueError("Vertex Veo 需要 params.project_id 和 params.location，或包含完整模型资源路径的 base_url")
        suffix = (
            f"v1/projects/{quote(project, safe='')}/locations/{quote(location, safe='')}/"
            f"publishers/google/models/{quote(model, safe='')}"
        )
        return self._api_url(f"https://{location}-aiplatform.googleapis.com", suffix)

    def _image(self, value: str) -> dict:
        if value.startswith("gs://"):
            return {"gcsUri": value, "mimeType": mimetypes.guess_type(value)[0] or "image/png"}
        url = self.reference_assets.to_image_url(value, max_bytes=self.capabilities.max_reference_inline_bytes)
        if not url.startswith("data:image/"):
            raise ValueError("Veo 图片需要内联 Base64 或 gs:// URI；公网 URL 会在提交前安全下载并转换")
        header, separator, encoded = url.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("Veo 图片必须为 Base64 data URL")
        try:
            base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError("Veo 图片 Base64 无效") from exc
        return {"bytesBase64Encoded": encoded, "mimeType": header[5:].split(";", 1)[0]}

    def build_payload(self, request: VideoRequest) -> dict:
        self.validate_request(request)
        # Fail unsupported Google model lines even when inspecting a payload.
        model = self.endpoint.model or "veo-3.1-generate-001"
        if "omni" in model.lower() or "preview" in model.lower():
            self._model_url()
        if request.task_type != "auto":
            raise ValueError("Veo 此适配器仅支持生成，延长视频的请求结构尚未在文档中确认")
        if request.ratio not in {"16:9", "9:16"}:
            raise ValueError("Veo 仅支持 16:9 / 9:16")
        resolution = request.resolution.lower()
        if resolution not in {"720p", "1080p"}:
            raise ValueError("Veo 分辨率仅支持 720p / 1080p")
        if request.end_frame and not request.reference_image:
            raise ValueError("Veo 尾帧必须与首帧一起提供")
        instance: dict = {"prompt": request.prompt}
        if request.negative_prompt:
            instance["negativePrompt"] = request.negative_prompt
        if request.reference_image:
            instance["image"] = self._image(request.reference_image)
        if request.end_frame:
            instance["lastFrame"] = self._image(request.end_frame)
        images = [
            asset
            for asset in request.reference_assets
            if asset.type not in {"approved_storyboard_first_frame", "end_frame"}
        ]
        if images:
            if not self.effective_capabilities(model).multiple_reference_images:
                raise ValueError("只有 Veo 3.1 支持多张参考图片")
            if len(images) > 3:
                raise ValueError("Veo 3.1 最多支持 3 张参考图片")
            instance["referenceImages"] = [
                {
                    "image": self._image(asset.url or asset.source_path),
                    "referenceType": "style" if asset.role == "style" else "asset",
                }
                for asset in images
            ]
        # A shot maps to one output file. Keep sampleCount at one to avoid
        # billing for extra generations that the current workflow cannot use.
        if int(self.endpoint.param("sample_count", 1)) != 1:
            raise ValueError("当前单镜头工作流仅接收 1 条视频，请设置 params.sample_count=1")
        parameters: dict = {
            "durationSeconds": int(request.duration),
            "aspectRatio": request.ratio,
            "resolution": resolution,
            "sampleCount": 1,
            "generateAudio": bool(request.generate_audio),
        }
        for key in ("personGeneration", "enhancePrompt"):
            value = self.endpoint.param(key)
            if value is not None:
                parameters[key] = value
        if self.endpoint.param("storage_uri"):
            parameters["storageUri"] = str(self.endpoint.param("storage_uri"))
        return {"instances": [instance], "parameters": parameters}

    async def _inline_image(self, value: str | None) -> str | None:
        if not value or not value.startswith(("https://", "http://")):
            return value
        raw = await download_remote_bytes(value, max_bytes=self.capabilities.max_reference_inline_bytes, timeout=60)
        mime = mimetypes.guess_type(urlparse(value).path)[0] or "image/png"
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    async def _submit(self, request: VideoRequest) -> str:
        model_url = self._model_url()
        # Resolve only supported API fields; arbitrary reference URLs never
        # become authenticated HTTP destinations.
        assets: list[ReferenceAsset] = []
        for asset in request.reference_assets:
            value = await self._inline_image(asset.url or asset.source_path)
            assets.append(replace(asset, url=value or ""))
        prepared = replace(
            request,
            reference_image=await self._inline_image(request.reference_image),
            end_frame=await self._inline_image(request.end_frame),
            reference_assets=assets,
        )
        data = await self._request_json("POST", f"{model_url}:predictLongRunning", json=self.build_payload(prepared))
        task_id = str(data.get("name") or "")
        if not task_id:
            raise RuntimeError(f"Veo 创建操作返回缺少 name: {data}")
        return task_id

    async def _poll_task(self, task_id: str) -> dict:
        for _ in range(max(1, int(self.endpoint.param("max_poll_attempts", 120)))):
            data = await self._request_json(
                "POST",
                f"{self._model_url()}:fetchPredictOperation",
                json={"operationName": task_id},
            )
            if data.get("error"):
                raise RuntimeError(f"Veo 视频任务失败: {data['error']}")
            if data.get("done"):
                response = data.get("response") or {}
                if response.get("raiMediaFilteredCount") and not response.get("videos"):
                    raise RuntimeError(
                        f"Veo 视频未通过内容审核: {response.get('raiMediaFilteredReasons') or 'filtered'}"
                    )
                return data
            await asyncio.sleep(max(0, float(self.endpoint.param("poll_interval_seconds", 5))))
        raise TimeoutError(f"Veo 视频任务超时: {task_id}")

    def _video_url(self, data: dict) -> str:
        videos = (data.get("response") or {}).get("videos") or []
        if videos:
            return str(videos[0].get("gcsUri") or videos[0].get("uri") or "")
        return ""

    async def _save_video(self, request: VideoRequest, data: dict, task_id: str) -> None:
        videos = (data.get("response") or {}).get("videos") or []
        if not videos:
            raise RuntimeError("Veo 返回缺少 videos；视频可能未通过审核")
        encoded = videos[0].get("bytesBase64Encoded")
        if encoded:
            available = self.storage.ensure_project_capacity(request.project_id, replacing=request.output_video_path)
            maximum = min(settings.MAX_REMOTE_MEDIA_BYTES, available)
            if len(encoded) > ((maximum + 2) // 3) * 4:
                raise ValueError("Veo 视频超过媒体大小或存储配额")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise RuntimeError("Veo 返回无效视频 Base64") from exc
            if len(raw) > maximum:
                raise ValueError("Veo 视频超过媒体大小或存储配额")
            atomic_write_bytes(request.output_video_path, raw, minimum_size=4096)
            return
        uri = self._video_url(data)
        if uri.startswith("gs://"):
            await self._save_gcs_video(request, uri)
            return
        await super()._save_video(request, data, task_id)

    async def _save_gcs_video(self, request: VideoRequest, uri: str) -> None:
        """Use the same OAuth bearer to download a private GCS object."""
        parsed = urlparse(uri)
        if not parsed.netloc or not parsed.path.lstrip("/"):
            raise RuntimeError("Veo 返回无效 GCS 视频 URI")
        url = (
            f"https://storage.googleapis.com/download/storage/v1/b/{quote(parsed.netloc, safe='')}/"
            f"o/{quote(parsed.path.lstrip('/'), safe='')}?alt=media"
        )
        destination = request.output_video_path
        if destination is None:
            raise ValueError("Veo 缺少视频保存路径")
        available = self.storage.ensure_project_capacity(request.project_id, replacing=destination)
        maximum = min(settings.MAX_REMOTE_MEDIA_BYTES, available)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.download")
        try:
            async with httpx.AsyncClient(
                timeout=180,
                follow_redirects=False,
                trust_env=settings.PROVIDER_HTTP_TRUST_ENV,
            ) as client:
                async with client.stream("GET", url, headers=self._headers()) as response:
                    if response.status_code >= 300:
                        raise RuntimeError(
                            f"Veo 下载 GCS 视频失败: HTTP {response.status_code}，请检查令牌的 Storage 读取权限"
                        )
                    length = response.headers.get("content-length")
                    if length and int(length) > maximum:
                        raise UploadLimitExceeded("Veo 视频超过媒体大小或存储配额")
                    total = 0
                    with temporary.open("wb") as output:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            total += len(chunk)
                            if total > maximum:
                                raise UploadLimitExceeded("Veo 视频超过媒体大小或存储配额")
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    if total <= 4096:
                        raise RuntimeError("Veo 返回的视频为空或过小")
                    os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
