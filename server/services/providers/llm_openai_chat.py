"""OpenAI Chat Completions 协议适配器（唯一的 LLM 协议适配器）。

所有 OpenAI 兼容服务（OpenAI / DeepSeek / MiMo / 各类聚合网关）共用本适配器：
接入新的兼容服务只需在端点配置里填 base_url/api_key/model，零代码。
- auth_style 决定鉴权头：bearer=SDK 标准 Authorization 头；
  api-key-header=额外附加 ``api-key`` 请求头（MiMo 等服务要求）。
- json_mode="auto" 时保留 response_format 试探降级：服务端报错含
  ``response_format`` 时自动去掉该参数重试。
"""

from __future__ import annotations

from openai import AsyncOpenAI

from services.providers.base import BaseAdapter, LLMCapabilities
from services.providers.usage import UsageMetadata, usage_from_chat_response


class OpenAIChatAdapter(BaseAdapter):
    capabilities = LLMCapabilities()

    def __init__(self, endpoint):
        super().__init__(endpoint)
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            endpoint = self.endpoint
            default_headers = None
            if endpoint.auth_style == "api-key-header":
                default_headers = {"api-key": endpoint.api_key}
            self._client = AsyncOpenAI(
                api_key=endpoint.api_key,
                base_url=endpoint.base_url or None,
                timeout=60,
                default_headers=default_headers,
            )
        return self._client

    def usage_from_response(
        self,
        capability: str,
        response: object | None = None,
        *,
        request: object | None = None,
        model: str = "",
        duration_ms: int = 0,
    ) -> UsageMetadata:
        """Chat Completions 的 usage 映射：prompt_tokens / completion_tokens。

        供应商没有返回 usage 时标记为「未知」，由记账层显示「成本未知」。
        """

        return usage_from_chat_response(
            response,
            provider=self.endpoint.protocol,
            model=model or self.endpoint.model,
            duration_ms=duration_ms,
        )

    async def complete(
        self,
        *,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ):
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def complete_json(
        self,
        *,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ):
        if self.capabilities.json_mode == "unsupported":
            return await self.complete(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
        try:
            return await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if "response_format" not in str(exc):
                raise
            # 服务端不支持 response_format 时降级为纯文本补全重试。
            return await self.complete(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
