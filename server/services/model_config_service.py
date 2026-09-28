"""模型与 API 自定义配置服务（端点式存储）。

四类能力（script/image/video/voice，另含可选的 script_fallback 备用 LLM 端点）
各持久化一份完整端点配置：protocol/base_url/api_key/model/auth_style/params，
以全局 JSON 形式保存在 ``data/model_api_config.json``（原子写 + 0600）。

各生成服务通过 ``services.providers.endpoint.get_endpoint(capability)`` 实时读取
端点配置，因此保存后新任务立即生效、无需重启；已生成的存量产物不受影响。

安全与兼容：
- base_url SSRF 校验、api_key 掩码回显（``********``）、换端点未换 key 时置
  ``api_key_required`` 并清空相关密钥的逻辑全部保留。
- 旧版「provider 字符串 + 平铺字段」JSON 由 ``migrate_store`` 在首次启动时自动
  迁移为新端点结构；旧 .env 字段继续作为默认值来源（JSON 覆盖 > .env > 默认值）。
- GET 额外回显 ``provider``（等于 protocol）及平铺参数字段，保证旧前端在迁移期
  可正常回填与保存。
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import uuid
from typing import Any
from urllib.parse import urlparse

from config import settings
from services.providers.endpoint import (
    API_KEY_REQUIRED,
    CAPABILITIES,
    FLAT_PARAM_FIELDS,
    KNOWN_PROTOCOLS,
    endpoint_from_stored,
    normalize_protocol,
    protocol_family,
)

MASKED_SECRET = "********"


def _store_path():
    return settings.DATA_DIR / "model_api_config.json"


def _read_raw() -> dict[str, Any]:
    path = _store_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_raw(data: dict[str, Any]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        # The file contains API credentials. Set restrictive permissions before
        # publication so there is no window where another local user can read it.
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_category(capability: str, payload: dict) -> dict:
    """把任意版本的类别数据归一化为新端点结构（幂等）。"""
    data = payload if isinstance(payload, dict) else {}
    legacy_provider = str(data.get("provider") or "").strip().lower()
    protocol = str(data.get("protocol") or "").strip().lower() or legacy_provider
    normalized: dict[str, Any] = {"protocol": normalize_protocol(capability, protocol)}

    if capability.startswith("script"):
        # 旧版 mimo provider 需要 api-key 请求头鉴权。
        auth_style = str(data.get("auth_style") or "").strip().lower()
        if not auth_style and legacy_provider in {"mimo"}:
            auth_style = "api-key-header"
        if auth_style:
            normalized["auth_style"] = auth_style

    for field in ("base_url", "api_key", "model"):
        if data.get(field) is not None:
            normalized[field] = data[field]

    params = dict(data.get("params") or {})
    for name in FLAT_PARAM_FIELDS:
        if name in data and name not in params and data[name] not in (None, ""):
            params[name] = data[name]
    if params:
        normalized["params"] = params

    if data.get(API_KEY_REQUIRED):
        normalized[API_KEY_REQUIRED] = True
    return normalized


def _normalize_store(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        capability: normalize_category(capability, payload)
        for capability, payload in raw.items()
        if isinstance(payload, dict)
    }


def _load_raw() -> dict[str, Any]:
    """读取持久化配置（归一化为新端点结构后返回）。"""
    return _normalize_store(_read_raw())


def migrate_store() -> bool:
    """把旧格式存储原地迁移为新端点结构。返回是否发生了改写。"""
    raw = _read_raw()
    normalized = _normalize_store(raw)
    if normalized == raw:
        return False
    _save_raw(normalized)
    return True


def stored_category(capability: str) -> dict[str, Any]:
    """返回某能力归一化后的持久化端点数据（可能为空 dict）。"""
    if capability not in CAPABILITIES:
        raise ValueError(f"未知能力类别: {capability}，可选值: {', '.join(CAPABILITIES)}")
    return _load_raw().get(capability) or {}


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _validate_base_url(value: Any) -> str:
    """Reject malformed/private endpoints that could be used for SSRF."""

    text = _clean(value)
    if not text:
        return text
    if len(text) > 2048:
        raise ValueError("Base URL 过长")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Base URL 必须是 http(s) 地址")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Base URL 端口非法") from exc
    host = parsed.hostname.rstrip(".").lower()
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None

    if ip is None:
        # inet_aton accepts ambiguous IPv4 forms such as 2130706433 and 127.1;
        # HTTP stacks may interpret those as loopback even though ip_address does
        # not. Reject them rather than treating them as ordinary DNS names.
        try:
            socket.inet_aton(host)
        except OSError:
            pass
        else:
            raise ValueError("Base URL 不允许使用非标准 IP 地址")
        try:
            ascii_host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError("Base URL 主机名非法") from exc
        if len(ascii_host) > 253 or not all(
            label and len(label) <= 63 and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
            for label in ascii_host.split(".")
        ):
            raise ValueError("Base URL 主机名非法")
        try:
            addresses = {
                result[4][0]
                for result in socket.getaddrinfo(ascii_host, port, type=socket.SOCK_STREAM)
            }
        except OSError as exc:
            raise ValueError("Base URL 主机名无法解析") from exc
        if not addresses:
            raise ValueError("Base URL 主机名无法解析")
        try:
            resolved = [ipaddress.ip_address(address.split("%", 1)[0]) for address in addresses]
        except ValueError as exc:
            raise ValueError("Base URL 解析结果非法") from exc
    else:
        resolved = [ip]

    if any(
        not address.is_global
        or address.is_loopback
        or address.is_link_local
        or address.is_private
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
        for address in resolved
    ):
        raise ValueError("禁止使用本机/内网 Base URL")
    return text.rstrip("/")


def _endpoint_identity(value: Any) -> tuple[str, str, int | None] | None:
    from services.providers.endpoint import endpoint_identity

    return endpoint_identity(value)


# ---------------------------------------------------------------------------
# 生效值读取（持久化覆盖 > 当前 settings）
# ---------------------------------------------------------------------------


def _mask_secret(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    return MASKED_SECRET


def _capability_summary(capability: str, protocol: str) -> dict[str, Any]:
    """协议适配器的能力摘要（供设置页如实展示，不含密钥）。"""
    from services.providers.registry import UnknownProtocolError, get_adapter

    summary: dict[str, Any] = {"protocol": protocol}
    try:
        adapter_cls = get_adapter(protocol_family(capability), protocol)
    except UnknownProtocolError:
        summary["adapter_registered"] = False
        return summary
    summary["adapter_registered"] = True
    capabilities = getattr(adapter_cls, "capabilities", None)
    if capabilities is None:
        return summary
    for field in (
        "reference_images",
        "requires_credentials",
        "reference_image",
        "reference_mode",
        "native_audio",
        "dialogue_in_prompt",
        "fixed_duration",
        "max_reference_inline_bytes",
    ):
        value = getattr(capabilities, field, None)
        if value is not None:
            summary[field] = value
    return summary


def _effective(*, mask_secrets: bool = True) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for capability in CAPABILITIES:
        endpoint = endpoint_from_stored(capability, _load_raw().get(capability) or {})
        data: dict[str, Any] = {
            "protocol": endpoint.protocol,
            "auth_style": endpoint.auth_style,
            "base_url": endpoint.base_url,
            "model": endpoint.model,
            "api_key": _mask_secret(endpoint.api_key) if mask_secrets else endpoint.api_key,
            "params": dict(endpoint.params),
            "capabilities": _capability_summary(capability, endpoint.protocol),
            # 旧版前端兼容镜像：provider 等价于 protocol，参数字段平铺到顶层。
            "provider": endpoint.protocol,
        }
        for name, value in endpoint.params.items():
            if name in FLAT_PARAM_FIELDS:
                data[name] = value
        result[capability] = data
    return result


def get_model_config() -> dict[str, Any]:
    """返回各能力端点的生效配置，供前端回填。"""
    return {"categories": _effective(mask_secrets=True)}


# ---------------------------------------------------------------------------
# 保存与应用
# ---------------------------------------------------------------------------


def _validate_protocol(capability: str, protocol: str) -> str:
    known = KNOWN_PROTOCOLS.get(protocol_family(capability), ())
    if protocol not in known:
        raise ValueError(
            f"未知的 {capability} 协议: {protocol or '<empty>'}，可选值: {', '.join(known) if known else '（暂无）'}"
        )
    return protocol


def save_model_config(data: dict[str, Any]) -> dict[str, Any]:
    """合并保存各能力端点配置并立即应用到运行时。"""
    stored = _load_raw()
    previous_effective = _effective(mask_secrets=False)
    incoming = data.get("categories") if isinstance(data.get("categories"), dict) else data
    for capability in CAPABILITIES:
        if capability in (incoming or {}):
            payload = incoming[capability] or {}
            if isinstance(payload, dict):
                # A masked/empty value from the UI means "leave the existing
                # secret untouched"; only a genuinely new key replaces it.
                sanitized_payload = normalize_category(capability, payload)
                _validate_protocol(capability, sanitized_payload.get("protocol", ""))
                explicit_api_key = bool(
                    "api_key" in sanitized_payload
                    and _clean(sanitized_payload["api_key"])
                    and _clean(sanitized_payload["api_key"]) != MASKED_SECRET
                )
                if "base_url" in sanitized_payload and sanitized_payload["base_url"] not in (None, ""):
                    sanitized_payload["base_url"] = _validate_base_url(sanitized_payload["base_url"])
                if "api_key" in sanitized_payload and (
                    not _clean(sanitized_payload["api_key"])
                    or _clean(sanitized_payload["api_key"]) == MASKED_SECRET
                ):
                    sanitized_payload.pop("api_key", None)
                merged = {**(stored.get(capability) or {}), **sanitized_payload}
                old_endpoint = _endpoint_identity(previous_effective.get(capability, {}).get("base_url"))
                new_endpoint = _endpoint_identity(merged.get("base_url"))
                if _clean(sanitized_payload.get("base_url")) and old_endpoint != new_endpoint and not explicit_api_key:
                    merged.pop("api_key", None)
                    merged[API_KEY_REQUIRED] = True
                elif explicit_api_key:
                    merged.pop(API_KEY_REQUIRED, None)
                stored[capability] = {k: v for k, v in merged.items() if k is not None}
    _save_raw(stored)
    apply_model_config_to_settings()
    return get_model_config()


def _set(field: str, value: Any) -> None:
    """仅在值非空时覆盖 settings，留空沿用 .env / 默认值。"""
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    setattr(settings, field, value)


def apply_model_config_to_settings(config: dict[str, Any] | None = None) -> None:
    """配置生效钩子（启动 / 保存后调用）。

    自适配器化改造起，LLM/图像/视频/语音服务均通过
    ``services.providers.endpoint.get_endpoint(capability)`` 实时读取端点配置，
    保存即生效、无需重启，因此不再需要把配置覆盖到 settings。本函数保留用于：
    - 启动时触发旧格式存储到端点式新格式的一次性迁移；
    - 兼容既有调用点（main.py lifespan / save_model_config / 旧测试）。
    """
    migrate_store()


__all__ = [
    "get_model_config",
    "save_model_config",
    "apply_model_config_to_settings",
    "migrate_store",
    "stored_category",
]
