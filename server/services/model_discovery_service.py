"""Discover models from a configured provider endpoint.

The discovery request is deliberately stateless: credentials are accepted for
one outbound request only and are never written to the model config store.
Providers that implement the OpenAI-compatible ``GET /models`` contract work
out of the box; the response is normalized for the settings UI.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from services.model_config_service import _validate_base_url


class ModelDiscoveryError(ValueError):
    """A safe, user-facing error raised while discovering models."""


def _models_url(base_url: str) -> str:
    try:
        endpoint = _validate_base_url(base_url)
    except ValueError as exc:
        raise ModelDiscoveryError(str(exc)) from exc
    if not endpoint:
        raise ModelDiscoveryError("请先填写 Base URL")
    parsed = urlparse(endpoint)
    path = parsed.path.rstrip("/")
    if path.endswith("/models"):
        return endpoint
    return f"{endpoint}/models"


def _audio_modes(protocol: str, item: dict[str, Any]) -> list[str]:
    """Return the audio modes a video model is known to support.

    Explicit provider metadata wins. Existing native adapters are otherwise
    authoritative: Seedance/Wanx produce silent video and the native adapter
    produces audio-bearing video.
    """

    metadata = item.get("capabilities")
    if isinstance(metadata, dict):
        explicit = metadata.get("audio_modes") or metadata.get("audioModes")
        if isinstance(explicit, list):
            modes = [
                str(value).strip().lower() for value in explicit if str(value).strip().lower() in {"native", "silent"}
            ]
            if modes:
                return list(dict.fromkeys(modes))
        if metadata.get("native_audio") is True or metadata.get("audio") is True:
            return ["native"]
        if metadata.get("native_audio") is False or metadata.get("audio") is False:
            return ["silent"]

    modalities = item.get("modalities")
    if isinstance(modalities, list) and any(str(value).lower() in {"audio", "audio_output"} for value in modalities):
        return ["native"]

    protocol = str(protocol or "").strip().lower()
    if protocol == "native-audio":
        return ["native"]
    if protocol in {"ark-seedance", "dashscope-wanx"}:
        return ["silent"]
    return []


def _normalize_model(item: Any, *, category: str, protocol: str) -> dict[str, Any] | None:
    if isinstance(item, str):
        model_id = item.strip()
        raw: dict[str, Any] = {}
    elif isinstance(item, dict):
        raw = item
        model_id = str(raw.get("id") or raw.get("name") or raw.get("model") or "").strip()
    else:
        return None
    if not model_id:
        return None

    result: dict[str, Any] = {
        "id": model_id,
        "label": str(raw.get("display_name") or raw.get("displayName") or model_id),
    }
    owner = raw.get("owned_by") or raw.get("ownedBy")
    if owner:
        result["owned_by"] = str(owner)
    if category == "video":
        modes = _audio_modes(protocol, raw)
        result["capabilities"] = {"audio_modes": modes}
    return result


def _extract_models(payload: Any, *, category: str, protocol: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        values = payload.get("data")
        if values is None:
            values = payload.get("models")
    else:
        values = payload
    if not isinstance(values, list):
        raise ModelDiscoveryError("服务返回的数据中没有可用模型列表")

    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        normalized = _normalize_model(item, category=category, protocol=protocol)
        if normalized is None or normalized["id"] in seen:
            continue
        seen.add(normalized["id"])
        models.append(normalized)
    return models


async def discover_models(
    *,
    category: str,
    base_url: str,
    api_key: str = "",
    protocol: str = "",
    auth_style: str = "bearer",
) -> dict[str, Any]:
    """Fetch and normalize a provider's model list without persisting secrets."""

    category = str(category or "").strip().lower()
    if category not in {"script", "image", "video", "voice"}:
        raise ModelDiscoveryError("不支持的模型类别")
    url = _models_url(base_url)
    key = str(api_key or "").strip()
    if key == "********":
        key = ""
    headers = {"Accept": "application/json"}
    if key:
        if str(auth_style or "").strip().lower() == "api-key-header":
            headers["api-key"] = key
        else:
            headers["Authorization"] = f"Bearer {key}"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=5.0), follow_redirects=False) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            if len(response.content) > 2 * 1024 * 1024:
                raise ModelDiscoveryError("模型列表响应过大")
            payload = response.json()
    except ModelDiscoveryError:
        raise
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in {401, 403}:
            raise ModelDiscoveryError("API Key 无效或没有读取模型列表的权限") from exc
        raise ModelDiscoveryError(f"服务返回 HTTP {status}，暂时无法获取模型") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise ModelDiscoveryError("无法连接到该 Base URL，请检查地址、网络和证书") from exc

    models = _extract_models(payload, category=category, protocol=protocol)
    return {"models": models, "count": len(models), "endpoint": url}


__all__ = ["ModelDiscoveryError", "discover_models"]
