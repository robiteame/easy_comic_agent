"""火山方舟 Seedream 图像适配器（原 ImageService._call_seedream 迁入）。

支持多参考图输入（capabilities.reference_images=True）；model 只来自端点配置，
不再把 provider 字符串当模型名兜底。按官方图片生成 API（/images/generations）：
- Seedream 5.0 系列（pro/lite）支持 ``output_format=png``；4.5/4.0 仅输出 jpeg
  且不支持自定义该参数，强校验下显式传 png 会报错，故按模型族选择性携带；
- 参考图最多 14 张（5.0/4.5/4.0）；
- ``guidance_scale`` 对 5.0/4.5/4.0 不支持，不发送。
"""

from __future__ import annotations

import httpx

from services.providers.base import BaseAdapter, ImageCapabilities, ImageRequest
from services.providers.image_common import extract_image_bytes, read_bounded_response
from services.providers.usage import CAPABILITY_IMAGE, UsageMetadata


class ArkSeedreamImageAdapter(BaseAdapter):
    capabilities = ImageCapabilities(reference_images=True, requires_credentials=True)

    def usage_for_request(
        self,
        capability: str,
        request: ImageRequest | None = None,
        *,
        model: str = "",
    ) -> UsageMetadata:
        """方舟 Seedream 按张计费：一次请求出一张，分辨率取请求尺寸。"""

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
        errors: list[str] = []
        models = self._model_candidates()
        sizes = [
            request.size,
            self.endpoint.param("image_size"),
            "2048x2048",
            "1440x2560",
            "2K",
        ]
        base_url = self.endpoint.base_url.rstrip("/")

        async with httpx.AsyncClient(timeout=180) as client:
            for model in models:
                for size in list(dict.fromkeys(size for size in sizes if size)):
                    payload = self._payload(model, request, size)
                    try:
                        request_message = client.build_request(
                            "POST",
                            f"{base_url}/images/generations",
                            headers={
                                "Authorization": f"Bearer {self.endpoint.api_key}",
                                "Content-Type": "application/json",
                            },
                            json=payload,
                        )
                        response = await client.send(request_message, stream=True)
                        response_bytes = await read_bounded_response(response)
                        if response.status_code in {401, 403}:
                            raise PermissionError(response_bytes.decode("utf-8", errors="replace")[:600])
                        if response.status_code >= 400:
                            errors.append(
                                f"{model} {size}: {response.status_code} {response_bytes.decode('utf-8', errors='replace')[:400]}"
                            )
                            continue
                        return await extract_image_bytes(client, response, response_bytes)
                    except PermissionError as exc:
                        raise RuntimeError(f"火山方舟鉴权失败，请确认 API Key 和模型权限: {exc}") from exc
                    except Exception as exc:
                        errors.append(f"{model} {size}: {exc}")

        raise RuntimeError("Seedream 图像生成失败: " + " | ".join(errors[-4:]))

    def _payload(self, model: str, request: ImageRequest, size: str) -> dict:
        payload = {
            "model": model,
            "prompt": f"{request.prompt}\nNegative prompt: {request.negative_prompt}",
            "response_format": "b64_json",
            "size": size,
            "n": 1,
            "seed": request.seed,
            "watermark": False,
            "sequential_image_generation": "disabled",
        }
        if self._supports_png(model):
            payload["output_format"] = "png"
        if request.reference_images:
            # 官方上限：5.0/4.5/4.0 支持最多 14 张参考图。
            payload["image"] = request.reference_images[:14]
        return payload

    @staticmethod
    def _supports_png(model: str) -> bool:
        """仅 Seedream 5.0 系列支持自定义 output_format（png）；其余保持默认 jpeg。"""

        return "seedream-5" in model.replace(".", "-").lower()

    def _model_candidates(self) -> list[str]:
        raw = self.endpoint.model
        candidates: list[str] = []
        if raw:
            candidates.extend(
                [
                    raw,
                    raw.replace(".", "-"),
                    raw.replace(".0", "-0"),
                ]
            )
        # 历史 API 兼容垫片：部分方舟网点仅识别带日期后缀的模型名。
        candidates.extend(
            [
                "doubao-seedream-5-0-lite",
                "doubao-seedream-5-0-lite-260128",
            ]
        )
        return list(dict.fromkeys(candidates))
