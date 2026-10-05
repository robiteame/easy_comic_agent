"""阿里云百炼 Qwen-Image 图像适配器。

Qwen-Image 的 DashScope 原生接口是异步任务协议：创建任务后轮询
``/tasks/{task_id}``，再下载 ``output.results[0].url`` 返回的临时图片。
"""

from __future__ import annotations

import asyncio

import httpx

from config import settings
from services.providers.base import BaseAdapter, ImageCapabilities, ImageRequest
from services.providers.image_common import extract_image_bytes, read_bounded_response
from services.providers.usage import CAPABILITY_IMAGE, UsageMetadata
from services.security import download_remote_bytes

_DEFAULT_BASE = "https://dashscope.aliyuncs.com/api/v1"
_DEFAULT_MODEL = "qwen-image-plus"


class QwenImageAdapter(BaseAdapter):
    """调用 DashScope 原生 Qwen-Image 文生图接口。"""

    # Qwen-Image 文生图接口不接受 Seedream 风格的参考图字段；图像编辑
    # 模型是另一套能力，避免把参考图误传给普通 qwen-image 模型。
    capabilities = ImageCapabilities(reference_images=False, requires_credentials=True)

    def usage_for_request(
        self,
        capability: str,
        request: ImageRequest | None = None,
        *,
        model: str = "",
    ) -> UsageMetadata:
        return UsageMetadata(
            capability=CAPABILITY_IMAGE,
            provider=self.endpoint.protocol,
            model=model or self.endpoint.model,
            images=1,
            resolution=str(getattr(request, "size", "") or ""),
            known=True,
            billable=True,
            source="request",
        )

    async def generate(self, request: ImageRequest) -> bytes:
        if self._uses_multimodal_api():
            return await self._generate_multimodal(request)
        if self._is_compatible_mode():
            return await self._generate_compatible(request)
        task_id = await self._create_task(request)
        result = await self._wait_for_task(task_id)
        output = result.get("output") or {}
        results = output.get("results") or []
        first_result = results[0] if results else {}
        image_url = str(first_result.get("url") or "") if isinstance(first_result, dict) else ""
        return await self._download_image_url(image_url, result)

    async def _generate_compatible(self, request: ImageRequest) -> bytes:
        """调用 DashScope OpenAI 兼容模式的同步 /images/generations。"""

        async with httpx.AsyncClient(timeout=180) as client:
            request_message = client.build_request(
                "POST",
                self._compatible_url(),
                headers=self._headers(),
                json=self._compatible_payload(request),
            )
            response = await client.send(request_message, stream=True)
            response_bytes = await read_bounded_response(response)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"百炼 Qwen-Image 兼容接口调用失败: {response.status_code} "
                    f"{response_bytes.decode('utf-8', errors='replace')[:1000]}"
                )
            return await extract_image_bytes(client, response, response_bytes)

    async def _generate_multimodal(self, request: ImageRequest) -> bytes:
        """调用 qwen-image-2.0/3.0 的 DashScope 原生同步接口。"""

        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                self._multimodal_url(),
                headers=self._headers(),
                json=self._multimodal_payload(request),
            )
        if response.status_code >= 400:
            raise RuntimeError(f"百炼 Qwen-Image 多模态接口调用失败: {response.status_code} {response.text[:1000]}")
        data = response.json()
        output = data.get("output") or {}
        choices = output.get("choices") or []
        content = ((choices[0].get("message") or {}).get("content") or []) if choices else []
        image_url = ""
        if content and isinstance(content[0], dict):
            image_url = str(content[0].get("image") or content[0].get("url") or "")
        if not image_url:
            results = output.get("results") or []
            if results and isinstance(results[0], dict):
                image_url = str(results[0].get("url") or "")
        return await self._download_image_url(image_url, data)

    async def _download_image_url(self, image_url: str, response_data: dict) -> bytes:
        if not image_url.startswith(("http://", "https://")):
            raise RuntimeError(f"百炼 Qwen-Image 返回缺少图片 URL: {response_data}")
        try:
            return await download_remote_bytes(
                image_url,
                max_bytes=settings.MAX_IMAGE_GENERATION_BYTES,
                timeout=180,
            )
        except Exception as exc:
            raise RuntimeError(f"百炼 Qwen-Image 图片下载失败: {exc}") from exc

    async def _create_task(self, request: ImageRequest) -> str:
        payload = self._payload(request)
        headers = self._headers()
        headers["X-DashScope-Async"] = "enable"
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(self._generation_url(), headers=headers, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"百炼 Qwen-Image 创建任务失败: {response.status_code} {response.text[:1000]}")
        data = response.json()
        task_id = str((data.get("output") or {}).get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"百炼 Qwen-Image 创建任务返回缺少任务 ID: {data}")
        return task_id

    async def _wait_for_task(self, task_id: str) -> dict:
        async with httpx.AsyncClient(timeout=60) as client:
            for _ in range(90):
                response = await client.get(f"{self._api_base()}/tasks/{task_id}", headers=self._headers())
                if response.status_code >= 400:
                    raise RuntimeError(f"百炼 Qwen-Image 查询任务失败: {response.status_code} {response.text[:1000]}")
                data = response.json()
                output = data.get("output") or {}
                status = str(output.get("task_status") or "").upper()
                if status == "SUCCEEDED":
                    return data
                if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                    raise RuntimeError(f"百炼 Qwen-Image 任务失败: {data}")
                await asyncio.sleep(5)
        raise TimeoutError(f"百炼 Qwen-Image 任务超时: {task_id}")

    def _payload(self, request: ImageRequest) -> dict:
        parameters: dict[str, object] = {
            "negative_prompt": request.negative_prompt,
            "prompt_extend": True,
            "watermark": False,
            "size": self._resolve_size(request.size),
            "n": 1,
            "seed": request.seed,
        }
        return {
            "model": self.endpoint.model or _DEFAULT_MODEL,
            "input": {"prompt": request.prompt},
            "parameters": parameters,
        }

    def _compatible_payload(self, request: ImageRequest) -> dict:
        return {
            "model": self.endpoint.model or _DEFAULT_MODEL,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "size": self._resolve_openai_size(request.size),
            "n": 1,
            "seed": request.seed,
            "prompt_extend": True,
            "watermark": False,
        }

    def _multimodal_payload(self, request: ImageRequest) -> dict:
        return {
            "model": self.endpoint.model or _DEFAULT_MODEL,
            "input": {
                "messages": [{"role": "user", "content": [{"text": request.prompt}]}],
            },
            "parameters": {
                "negative_prompt": request.negative_prompt,
                "prompt_extend": True,
                "watermark": False,
                "size": self._resolve_size(request.size),
                "n": 1,
                "seed": request.seed,
            },
        }

    @staticmethod
    def _resolve_size(size: str) -> str:
        value = str(size or "").strip()
        if not value:
            return "1440*2560"
        return value.replace("x", "*").replace("X", "*")

    @staticmethod
    def _resolve_openai_size(size: str) -> str:
        value = str(size or "").strip()
        if not value:
            return "1440x2560"
        return value.replace("*", "x")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.endpoint.api_key}", "Content-Type": "application/json"}

    def _api_base(self) -> str:
        base = (self.endpoint.base_url or _DEFAULT_BASE).rstrip("/")
        if base.endswith("/compatible-mode/v1"):
            return f"{base[: -len('/compatible-mode/v1')]}/api/v1"
        if not base.endswith("/api/v1"):
            base = f"{base}/api/v1"
        return base

    def _is_compatible_mode(self) -> bool:
        return (self.endpoint.base_url or "").rstrip("/").endswith("/compatible-mode/v1")

    def _uses_multimodal_api(self) -> bool:
        model = (self.endpoint.model or "").strip().lower()
        return model.startswith("qwen-image-2.0") or (
            model.startswith("qwen-image-3.0") and not self._is_compatible_mode()
        )

    def _compatible_url(self) -> str:
        return f"{self.endpoint.base_url.rstrip('/')}/images/generations"

    def _multimodal_url(self) -> str:
        return f"{self._api_base()}/services/aigc/multimodal-generation/generation"

    def _generation_url(self) -> str:
        return f"{self._api_base()}/services/aigc/text2image/image-synthesis"
