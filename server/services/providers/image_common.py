"""图像适配器共享的 HTTP 响应处理工具。"""

from __future__ import annotations

import base64
import json

import httpx

from config import settings
from services.security import download_remote_bytes


async def read_bounded_response(response: httpx.Response) -> bytes:
    limit = max(1024, int(settings.MAX_IMAGE_GENERATION_BYTES * 4 / 3) + 16)
    length = response.headers.get("content-length")
    if length and int(length) > limit:
        await response.aclose()
        raise RuntimeError("图像接口响应超过大小限制")
    body = bytearray()
    try:
        async for chunk in response.aiter_bytes(1024 * 1024):
            body.extend(chunk)
            if len(body) > limit:
                raise RuntimeError("图像接口响应超过大小限制")
        return bytes(body)
    finally:
        await response.aclose()


async def extract_image_bytes(client: httpx.AsyncClient, response: httpx.Response, response_bytes: bytes) -> bytes:
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("image/"):
        if len(response_bytes) > settings.MAX_IMAGE_GENERATION_BYTES:
            raise RuntimeError("图像数据超过大小限制")
        return response_bytes

    try:
        data = json.loads(response_bytes)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("图像接口返回不是有效 JSON") from exc
    items = data.get("data") if isinstance(data, dict) else None
    if not items:
        raise RuntimeError(f"图像接口返回缺少 data: {data}")

    item = items[0]
    b64_data = item.get("b64_json") or item.get("image") or item.get("base64")
    if b64_data:
        if "," in b64_data and b64_data.startswith("data:"):
            b64_data = b64_data.split(",", 1)[1]
        if len(b64_data) > int(settings.MAX_IMAGE_GENERATION_BYTES * 4 / 3) + 16:
            raise RuntimeError("图像数据超过大小限制")
        decoded = base64.b64decode(b64_data, validate=True)
        if len(decoded) > settings.MAX_IMAGE_GENERATION_BYTES:
            raise RuntimeError("图像数据超过大小限制")
        return decoded

    image_url = item.get("url") or item.get("image_url")
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    if image_url:
        try:
            return await download_remote_bytes(
                image_url,
                max_bytes=settings.MAX_IMAGE_GENERATION_BYTES,
                timeout=180,
            )
        except Exception as exc:
            raise RuntimeError(f"图像 URL 下载失败: {exc}") from exc

    raise RuntimeError(f"无法解析图像接口返回: {data}")
