"""设置页「测试连接」服务：对四类模型能力做廉价的真实探活。

原则：
- 只做廉价真实调用，不做「非空即通过」的伪校验：
  - LLM：一次 max_tokens=1 的补全请求；
  - TTS：合成一个字符的测试音频（按字符计费，成本可忽略）；
  - 图像 / 视频：优先读 OpenAI 兼容模型列表（火山方舟 ``/api/v3/models``、
    百炼 ``compatible-mode/v1/models`` 均为免费接口）；Stability 用免费的
    账户信息接口。某 Provider 没有免费探活手段时如实返回
    ``unsupported_check``，绝不伪造成功，也绝不为验证而发起计费生成。
  - ``image`` 协议为 ``placeholder``（本地占位图）时直接视为有效。
- 单次探活有硬性超时（个位数秒），外部服务无响应也不阻塞请求方。
- 安全：用户传入的 Base URL 走与保存配置相同的 SSRF 校验；密钥只在请求头
  中出现一次，任何响应消息经 error_reporter 脱敏后才返回。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from services import error_reporter
from services.model_config_service import MASKED_SECRET, _validate_base_url
from services.providers.base import TTSRequest
from services.providers.endpoint import (
    EndpointConfig,
    endpoint_from_stored,
    endpoint_identity,
    get_endpoint,
)
from services.providers.registry import get_adapter

# 用户可见能力名 -> 内部能力类别（端点存储键）。
CAPABILITY_ALIASES = {"llm": "script", "image": "image", "video": "video", "tts": "voice"}

STATUS_OK = "ok"
STATUS_FAIL = "fail"
STATUS_UNSUPPORTED = "unsupported_check"

# 单次探活的硬上限（个位数秒）：无论外部服务挂起多久都必须在此返回。
PROBE_TIMEOUT_SECONDS = 8.0
_HTTP_TIMEOUT = httpx.Timeout(PROBE_TIMEOUT_SECONDS, connect=4.0)

# 模型列表响应的大小上限，超过视为「该地址不是模型列表接口」。
_MAX_LISTING_BYTES = 2 * 1024 * 1024
_SNIPPET_CHARS = 120
_TTS_PROBE_TEXT = "好"

# 各协议探活前必须具备的字段。腾讯云 / 百炼 TTS 的 base_url 留空时适配器会
# 回落到官方域名，因此不强制；其余协议都需要显式地址。
_NEEDS_BASE_URL = {
    "openai-chat",
    "mimo-tts",
    "ark-seedream",
    "qwen-image",
    "stability",
    "ark-seedance",
    "dashscope-wanx",
    "native-audio",
}
_NEEDS_MODEL = {"openai-chat", "mimo-tts"}

# 百炼系协议：OpenAI 兼容模型列表在 compatible-mode 根下，而非 api/v1。
_DASHSCOPE_PROTOCOLS = {"qwen-image", "dashscope-wanx"}


@dataclass
class ConnectionTestResult:
    status: str
    provider: str
    model: str
    latency_ms: int
    message: str

    def as_dict(self) -> dict:
        # 出参再过一次脱敏：响应体片段、适配器异常文本都可能夹带密钥样式串。
        return {
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "message": error_reporter.redact(self.message),
        }


async def test_connection(capability: str, config: dict | None = None) -> dict:
    """测试一类模型能力的连通性。

    ``config`` 为设置页表单中未保存的待测配置；不传时按现行优先级
    （已保存 JSON > .env > 默认值）取生效端点。返回统一结构：
    ``{status, provider, model, latency_ms, message}``，绝不包含密钥。
    """

    family = CAPABILITY_ALIASES.get(str(capability or "").strip().lower())
    if family is None:
        raise ValueError("不支持的能力类别，可选值: llm / image / video / tts")

    endpoint, credentials_withheld = _resolve_endpoint(family, config)
    started = time.monotonic()
    if credentials_withheld:
        status, message = (
            STATUS_FAIL,
            "接口地址已变更但未填写新的 API Key：为避免把旧密钥发往新地址，请先填写密钥再测试。",
        )
    else:
        status, message = await _run_probe(family, endpoint)
    latency_ms = int((time.monotonic() - started) * 1000)
    return ConnectionTestResult(
        status=status,
        provider=endpoint.protocol,
        model=endpoint.model,
        latency_ms=max(0, latency_ms),
        message=message,
    ).as_dict()


# ---------------------------------------------------------------------------
# 端点解析
# ---------------------------------------------------------------------------


def _resolve_endpoint(family: str, config: dict | None) -> tuple[EndpointConfig, bool]:
    """解析待测端点；第二项表示「密钥被刻意扣留」（见下），此时不应发起探活。"""

    if config is None:
        return get_endpoint(family), False

    supplied = {key: value for key, value in dict(config or {}).items() if value is not None}
    key = str(supplied.get("api_key") or "").strip()
    if key and key != MASKED_SECRET:
        supplied["api_key"] = key
        return endpoint_from_stored(family, supplied), False

    # 表单密钥为空或掩码时，仅当仍指向同一端点才复用已保存密钥（与模型
    # 发现路由同一策略）。换地址且确有旧密钥时扣留凭据并直接判失败：不能
    # 把旧凭据发往新主机，也不能依赖 endpoint_from_stored 的「空值沿用默认」
    # 语义——那样 .env 里的密钥会悄悄混进来。本来就无密钥可继承时走正常
    # 解析，由前置检查按「未配置」报错。
    effective = get_endpoint(family)
    form_base = str(supplied.get("base_url") or "").strip()
    same_target = not form_base or endpoint_identity(form_base) == endpoint_identity(effective.base_url)
    if same_target or not effective.api_key.strip():
        supplied["api_key"] = effective.api_key
        return endpoint_from_stored(family, supplied), False
    supplied.pop("api_key", None)
    return endpoint_from_stored(family, supplied), True


# ---------------------------------------------------------------------------
# 探活调度
# ---------------------------------------------------------------------------


async def _run_probe(family: str, endpoint: EndpointConfig) -> tuple[str, str]:
    if endpoint.protocol == "placeholder":
        return STATUS_OK, "本地占位图模式：不访问外部接口，配置即生效。"

    if endpoint.protocol in _NEEDS_BASE_URL and not endpoint.base_url.strip():
        return STATUS_FAIL, "未填写 Base URL，无法测试。"
    if endpoint.protocol in _NEEDS_MODEL and not endpoint.model.strip():
        return STATUS_FAIL, "未填写模型名称，无法测试。"
    if not endpoint.api_key.strip():
        if endpoint.protocol == "tencent-tts":
            return STATUS_FAIL, "未配置语音凭据（SecretId:SecretKey），无法测试。"
        return STATUS_FAIL, "未配置 API Key，无法测试：请填写密钥后重试。"

    if endpoint.base_url.strip():
        try:
            _validate_base_url(endpoint.base_url)
        except ValueError as exc:
            return STATUS_FAIL, f"Base URL 不合法：{exc}"

    try:
        return await asyncio.wait_for(_dispatch_probe(family, endpoint), timeout=PROBE_TIMEOUT_SECONDS)
    except TimeoutError:
        return STATUS_FAIL, f"连接超时（{PROBE_TIMEOUT_SECONDS:.0f} 秒），端点无响应。"
    except Exception as exc:  # noqa: BLE001 - 探活失败必须转成可展示原因，不能 500
        return STATUS_FAIL, error_reporter.summarize(exc)


async def _dispatch_probe(family: str, endpoint: EndpointConfig) -> tuple[str, str]:
    if family == "script":
        return await _probe_openai_chat(endpoint)
    if family == "voice":
        return await _probe_tts_synthesis(endpoint)
    if endpoint.protocol == "stability":
        return await _probe_stability_account(endpoint)
    return await _probe_model_listing(endpoint)


def _build_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=False)


def _auth_headers(endpoint: EndpointConfig, *, json_body: bool = False) -> dict[str, str]:
    headers = {"Accept": "application/json", "Authorization": f"Bearer {endpoint.api_key}"}
    if endpoint.auth_style == "api-key-header":
        # 与 OpenAIChatAdapter 一致：api-key 头与 Bearer 同时携带。
        headers["api-key"] = endpoint.api_key
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def _snippet(response: httpx.Response) -> str:
    text = " ".join(response.text.split())
    if len(text) > _SNIPPET_CHARS:
        text = text[:_SNIPPET_CHARS] + "…"
    return text


# ---------------------------------------------------------------------------
# LLM：一次极小补全
# ---------------------------------------------------------------------------


async def _probe_openai_chat(endpoint: EndpointConfig) -> tuple[str, str]:
    payload = {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }
    async with _build_client() as client:
        response = await client.post(
            f"{endpoint.base_url.rstrip('/')}/chat/completions",
            headers=_auth_headers(endpoint, json_body=True),
            json=payload,
        )
    if response.status_code == 200:
        return STATUS_OK, "补全请求成功，端点、密钥与模型可用。"
    if response.status_code in {401, 403}:
        return STATUS_FAIL, f"鉴权失败（HTTP {response.status_code}）：API Key 无效或没有权限。{_snippet(response)}"
    if response.status_code in {404, 405}:
        return STATUS_FAIL, f"接口不存在（HTTP {response.status_code}）：请检查 Base URL 是否为 OpenAI 兼容根地址。"
    return STATUS_FAIL, f"补全请求失败：HTTP {response.status_code}。{_snippet(response)}"


# ---------------------------------------------------------------------------
# 图像 / 视频：模型列表探活
# ---------------------------------------------------------------------------


def _model_list_urls(endpoint: EndpointConfig) -> list[str]:
    base = endpoint.base_url.rstrip("/")
    candidates: list[str] = []
    if endpoint.protocol in _DASHSCOPE_PROTOCOLS:
        parsed = urlparse(base)
        if parsed.scheme and parsed.hostname:
            candidates.append(f"{parsed.scheme}://{parsed.netloc}/compatible-mode/v1/models")
    # Base 已指向 /models 时原样使用（与模型发现路由一致）。
    candidates.append(base if base.endswith("/models") else f"{base}/models")
    return list(dict.fromkeys(candidates))


async def _probe_model_listing(endpoint: EndpointConfig) -> tuple[str, str]:
    candidates = _model_list_urls(endpoint)
    network_errors = 0
    async with _build_client() as client:
        for url in candidates:
            try:
                response = await client.get(url, headers=_auth_headers(endpoint))
            except httpx.HTTPError:
                network_errors += 1
                continue
            if response.status_code in {401, 403}:
                return (
                    STATUS_FAIL,
                    f"鉴权失败（HTTP {response.status_code}）：API Key 无效或没有权限。{_snippet(response)}",
                )
            if response.status_code == 200 and _looks_like_model_list(response):
                return STATUS_OK, "已成功读取模型列表，端点与密钥可用。"
    if network_errors == len(candidates):
        return STATUS_FAIL, "无法连接到该 Base URL，请检查地址、网络与证书。"
    return (
        STATUS_UNSUPPORTED,
        "该服务未提供模型列表接口，也没有其它免费探活方式；为避免产生费用，不做真实生成验证。",
    )


def _looks_like_model_list(response: httpx.Response) -> bool:
    if len(response.content) > _MAX_LISTING_BYTES:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    if isinstance(payload, list):
        return True
    if isinstance(payload, dict):
        values = payload.get("data")
        if values is None:
            values = payload.get("models")
        return isinstance(values, list)
    return False


# ---------------------------------------------------------------------------
# Stability：免费账户信息接口
# ---------------------------------------------------------------------------


async def _probe_stability_account(endpoint: EndpointConfig) -> tuple[str, str]:
    parsed = urlparse(endpoint.base_url)
    if not parsed.scheme or not parsed.hostname:
        return STATUS_FAIL, "Stability 端点未配置 Base URL。"
    url = f"{parsed.scheme}://{parsed.netloc}/v1/user/account"
    async with _build_client() as client:
        response = await client.get(url, headers=_auth_headers(endpoint))
    if response.status_code == 200:
        return STATUS_OK, "已成功读取账户信息，密钥可用。"
    if response.status_code in {401, 403}:
        return STATUS_FAIL, f"鉴权失败（HTTP {response.status_code}）：API Key 无效或没有权限。{_snippet(response)}"
    return (
        STATUS_UNSUPPORTED,
        "该 Stability 端点未提供免费探活接口；为避免产生费用，不做真实生成验证。",
    )


# ---------------------------------------------------------------------------
# TTS：一个字符的最小合成
# ---------------------------------------------------------------------------


async def _probe_tts_synthesis(endpoint: EndpointConfig) -> tuple[str, str]:
    adapter_cls = get_adapter("voice", endpoint.protocol)
    adapter = adapter_cls(endpoint)
    audio = await adapter.synthesize(TTSRequest(text=_TTS_PROBE_TEXT))
    if not audio:
        return STATUS_FAIL, "接口返回了空音频，请检查模型与音色配置。"
    return STATUS_OK, f"已合成 {_TTS_PROBE_TEXT} 的测试音频，密钥、模型与音色可用。"


__all__ = ["CAPABILITY_ALIASES", "ConnectionTestResult", "test_connection"]
