"""基于 openai-chat 协议适配器的 LLM 客户端。

通过 ``get_endpoint("script")`` 实时读取主端点、``get_endpoint("script_fallback")``
读取备端点，保存模型/API 配置后新任务即生效。主端点调用失败且备端点可用
（配置了 api_key 且端点标识不同）时自动回落。解析层保留 reasoning_content
与 markdown 围栏 JSON 容错。
"""

import asyncio
import json
import re
import time

from config import settings
from services.providers.base import BaseAdapter
from services.providers.endpoint import EndpointConfig, endpoint_identity, get_endpoint
from services.providers.registry import get_adapter
from services.providers.usage import (
    CAPABILITY_LLM,
    ERROR_CODE_PROVIDER_CALL_FAILED,
    adapter_usage_for_request,
    adapter_usage_from_response,
)
from services import usage_service
from services.job_debug import record_api_request, record_api_result


class LLMOutputTruncatedError(RuntimeError):
    """模型输出达到 max_tokens 上限被截断（确定性失败，不允许同配置重试）。

    触发条件（满足其一）：
    - Provider 返回 ``finish_reason == "length"``（明确截断）；
    - ``output_tokens >= max_tokens``（保守判定：即使 finish_reason=stop，
      输出恰好顶满上限也应视为可能被截断，而不是当成普通 JSON 解析失败重试）。

    诊断字段（finish_reason / token 数 / max_tokens / provider / model）会保留在
    异常消息与 ``details`` 中，供任务中心与恢复决策展示。
    """

    def __init__(
        self,
        *,
        finish_reason: str = "",
        output_tokens: int | None = None,
        prompt_tokens: int | None = None,
        max_tokens: int = 0,
        provider: str = "",
        model: str = "",
        suspected: bool = False,
    ) -> None:
        self.finish_reason = str(finish_reason or "")
        self.output_tokens = output_tokens
        self.prompt_tokens = prompt_tokens
        self.max_tokens = int(max_tokens or 0)
        self.provider = str(provider or "")
        self.model = str(model or "")
        self.suspected = bool(suspected)
        head = "模型输出疑似达到上限被截断" if suspected else "模型输出超过最大长度并被截断"
        self.diagnostics = (
            f"finish_reason={self.finish_reason or 'unknown'}"
            f"，输出 {output_tokens if output_tokens is not None else '未知'}/{self.max_tokens} tokens"
            f"，输入 {prompt_tokens if prompt_tokens is not None else '未知'} tokens"
            f"，provider={self.provider or 'unknown'}，model={self.model or 'unknown'}"
        )
        super().__init__(
            f"{head}（{self.diagnostics}）；请增加输出额度或按场次分段解析，"
            "不要用相同配置重试"
        )

    def detail(self) -> dict:
        return {
            "finish_reason": self.finish_reason,
            "output_tokens": self.output_tokens,
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "provider": self.provider,
            "model": self.model,
            "suspected": self.suspected,
        }


def large_json_max_tokens() -> int:
    """剧本解析/分镜等大 JSON 调用的输出额度。

    目标值取 ``settings.LLM_LARGE_JSON_MAX_TOKENS``（默认 16384），但不得超过
    端点声明的实际上限：端点 params 里显式配置了 ``max_output_tokens`` 时按它
    收紧（模型/网关不支持更大输出时如实用端点值，绝不发出超限请求）。
    """

    target = max(1, int(settings.LLM_LARGE_JSON_MAX_TOKENS))
    try:
        endpoint = get_endpoint("script")
        declared = endpoint.param("max_output_tokens")
        try:
            declared = int(declared) if declared not in (None, "") else None
        except (TypeError, ValueError):
            declared = None
    except Exception:  # noqa: BLE001 - 端点配置不可读时沿用目标值
        declared = None
    if declared and declared > 0:
        return max(1, min(target, declared))
    return target


class LLMService:
    """OpenAI 兼容协议 LLM 客户端：主/备端点自动回落 + JSON 清洗。"""

    def __init__(self):
        self._adapters: dict[tuple, BaseAdapter] = {}
        self._sync_config()

    # --- 用量记账 --------------------------------------------------------

    def _record(self, adapter: BaseAdapter, response, model: str, started: float) -> None:
        """把一次成功调用的 token 用量入账（供应商未回报 usage 时记为「成本未知」）。"""

        duration_ms = int((time.monotonic() - started) * 1000)
        metadata = adapter_usage_from_response(
            adapter, CAPABILITY_LLM, response, model=model, duration_ms=duration_ms
        )
        usage_service.record_metadata(
            metadata,
            duration_ms=duration_ms,
            scope=usage_service.current_scope(),
        )

    def _record_failure(self, adapter: BaseAdapter, model: str, started: float) -> None:
        duration_ms = int((time.monotonic() - started) * 1000)
        metadata = adapter_usage_for_request(adapter, CAPABILITY_LLM, None, model=model)
        usage_service.record_failure(
            metadata,
            error_code=ERROR_CODE_PROVIDER_CALL_FAILED,
            duration_ms=duration_ms,
            scope=usage_service.current_scope(),
        )

    def _record_cancelled(self, adapter: BaseAdapter, model: str, started: float) -> None:
        """任务在 LLM 调用途中被取消：留痕但不虚增金额。"""

        duration_ms = int((time.monotonic() - started) * 1000)
        metadata = adapter_usage_for_request(adapter, CAPABILITY_LLM, None, model=model)
        usage_service.record_cancelled(
            metadata,
            duration_ms=duration_ms,
            scope=usage_service.current_scope(),
        )

    @staticmethod
    def _connection_key(endpoint: EndpointConfig | None) -> tuple | None:
        if endpoint is None:
            return None
        return (endpoint.protocol, endpoint.base_url, endpoint.api_key, endpoint.auth_style)

    def _adapter_for(self, endpoint: EndpointConfig) -> BaseAdapter:
        key = self._connection_key(endpoint)
        adapter = self._adapters.get(key)
        if adapter is None:
            adapter = get_adapter("script", endpoint.protocol)(endpoint)
            self._adapters[key] = adapter
        return adapter

    def _sync_config(self) -> None:
        """实时从端点配置读取，保证保存模型/API 配置后新任务即生效。"""
        endpoint = get_endpoint("script")
        fallback = get_endpoint("script_fallback")
        # 备端点必须配置了密钥、且与主端点不是同一个服务地址时才参与回落。
        if not fallback.api_key or endpoint_identity(fallback.base_url) == endpoint_identity(endpoint.base_url):
            fallback = None

        self._endpoint = endpoint
        self._fallback_endpoint = fallback
        self.model = endpoint.model or "gpt-4o-mini"
        self.vision_model = endpoint.param("vision_model") or self.model
        self.max_tokens = self._int_param(endpoint.param("max_tokens"), settings.LLM_MAX_TOKENS)
        self.last_provider_used = self._endpoint_label(endpoint)

    @staticmethod
    def _int_param(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _endpoint_label(endpoint: EndpointConfig) -> str:
        return f"{endpoint.protocol}:{endpoint.model or 'default'}"

    @property
    def available(self) -> bool:
        self._sync_config()
        return bool(self._endpoint.api_key or self._fallback_endpoint)

    @property
    def client(self):
        self._sync_config()
        if not self.available:
            raise RuntimeError("未配置可用的 LLM API Key")
        return self._adapter_for(self._endpoint).client

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        allow_fallback: bool = True,
        model_override: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        effective_max_tokens = int(max_tokens) if max_tokens else self.max_tokens
        debug_request_id = record_api_request(
            api="LLM Chat",
            provider=self._endpoint.protocol,
            model=model_override or self.model,
            params={
                "temperature": temperature,
                "max_tokens": effective_max_tokens,
                "allow_fallback": allow_fallback,
                "response_format": "text",
            },
            prompt={"system": system_prompt, "user": user_prompt},
        )
        try:
            response = await self._completion_with_fallback(
                lambda adapter, model: adapter.complete(
                    messages=messages,
                    model=model_override or model,
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                ),
                allow_fallback=allow_fallback,
            )
            self._raise_if_truncated(response, effective_max_tokens)
            text = self._message_text(response.choices[0].message)
        except LLMOutputTruncatedError as exc:
            record_api_result(
                debug_request_id,
                api="LLM Chat",
                status="error",
                message=f"LLM 输出被截断：{exc}",
                detail={"truncation": exc.detail()},
            )
            raise
        except Exception as exc:
            record_api_result(debug_request_id, api="LLM Chat", status="error", message=f"LLM 调用失败：{exc}")
            raise
        record_api_result(
            debug_request_id,
            api="LLM Chat",
            status="success",
            message="LLM 调用成功",
            detail={"output_chars": len(text), "provider_used": self.last_provider_used},
        )
        return text

    async def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_retries: int = 2,
        allow_fallback: bool = True,
        model_override: str | None = None,
        max_tokens: int | None = None,
        prefer_fallback: bool = False,
    ) -> dict:
        """JSON 调用 + 清洗。输出截断是确定性失败：立即抛出，绝不同配置重试。

        ``max_tokens`` 允许调用方（剧本解析/分镜）申请比端点默认更大的输出额度；
        ``prefer_fallback`` 供恢复决策切换 Provider 时直接走备端点。
        """
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = await self._create_json_completion(
                    system_prompt,
                    user_prompt,
                    temperature,
                    allow_fallback,
                    model_override,
                    max_tokens,
                    prefer_fallback,
                )
                content = self._message_text(response.choices[0].message)
                self._raise_if_truncated(response, int(max_tokens) if max_tokens else self.max_tokens)
                return self._loads_json(content or "{}")
            except LLMOutputTruncatedError:
                # 截断不会被“再试一次”修复：同样的输入 + 同样的 max_tokens 只会
                # 得到同样的截断。交给调用方提高额度或分段，或由恢复决策换 Provider。
                raise
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    continue

        raise ValueError(f"LLM JSON 解析失败，已重试 {max_retries} 次: {last_error}")

    async def _create_json_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        allow_fallback: bool = True,
        model_override: str | None = None,
        max_tokens: int | None = None,
        prefer_fallback: bool = False,
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        effective_max_tokens = int(max_tokens) if max_tokens else self.max_tokens
        debug_request_id = record_api_request(
            api="LLM JSON",
            provider=self._endpoint.protocol,
            model=model_override or self.model,
            params={
                "temperature": temperature,
                "max_tokens": effective_max_tokens,
                "allow_fallback": allow_fallback,
                "response_format": "json_object",
            },
            prompt={"system": system_prompt, "user": user_prompt},
        )
        try:
            response = await self._completion_with_fallback(
                lambda adapter, model: adapter.complete_json(
                    messages=messages,
                    model=model_override or model,
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                ),
                allow_fallback=allow_fallback,
                prefer_fallback=prefer_fallback,
            )
        except Exception as exc:
            record_api_result(debug_request_id, api="LLM JSON", status="error", message=f"LLM JSON 调用失败：{exc}")
            raise
        record_api_result(
            debug_request_id,
            api="LLM JSON",
            status="success",
            message="LLM JSON 调用成功",
            detail={"provider_used": self.last_provider_used},
        )
        return response

    async def _completion_with_fallback(self, create_completion, allow_fallback: bool = True, prefer_fallback: bool = False):
        self._sync_config()
        primary_adapter = self._adapter_for(self._endpoint)
        fallback = self._fallback_endpoint
        # 恢复决策选中 switch_provider(script_fallback) 时：备端点作为首选，
        # 主端点退居其 fallback，两个端点都保持原有记账与取消语义。
        if prefer_fallback and fallback is not None:
            fallback_adapter = self._adapter_for(fallback)
            fallback_model = fallback.model or self.model
            fallback_started = time.monotonic()
            try:
                response = await create_completion(fallback_adapter, fallback_model)
            except asyncio.CancelledError:
                self._record_cancelled(fallback_adapter, fallback_model, fallback_started)
                raise
            except Exception:
                self._record_failure(fallback_adapter, fallback_model, fallback_started)
                if not allow_fallback:
                    raise
            else:
                self.last_provider_used = self._endpoint_label(fallback)
                self._record(fallback_adapter, response, fallback_model, fallback_started)
                return response
        started = time.monotonic()
        try:
            response = await create_completion(primary_adapter, self.model)
            self.last_provider_used = self._endpoint_label(self._endpoint)
            self._record(primary_adapter, response, self.model, started)
            return response
        except asyncio.CancelledError:
            self._record_cancelled(primary_adapter, self.model, started)
            raise
        except Exception as primary_error:
            self._record_failure(primary_adapter, self.model, started)
            if not allow_fallback or fallback is None:
                raise
            fallback_adapter = self._adapter_for(fallback)
            fallback_model = fallback.model or self.model
            fallback_started = time.monotonic()
            try:
                response = await create_completion(fallback_adapter, fallback_model)
            except asyncio.CancelledError:
                self._record_cancelled(fallback_adapter, fallback_model, fallback_started)
                raise
            except Exception as fallback_error:
                self._record_failure(fallback_adapter, fallback_model, fallback_started)
                # 备端点也失败时优先暴露主端点错误（更具诊断价值）。
                raise primary_error from fallback_error
            self.last_provider_used = self._endpoint_label(fallback)
            self._record(fallback_adapter, response, fallback_model, fallback_started)
            return response

    # --- 输出截断识别 ------------------------------------------------------

    @staticmethod
    def _usage_tokens(response) -> tuple[int | None, int | None]:
        """从 response.usage 读 (prompt_tokens, output_tokens)；缺失返回 (None, None)。"""

        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None

        def field(*names: str) -> int | None:
            for name in names:
                value = getattr(usage, name, None)
                if isinstance(value, int):
                    return value
                try:
                    if value not in (None, ""):
                        return int(value)
                except (TypeError, ValueError):
                    continue
            return None

        prompt = field("prompt_tokens", "input_tokens")
        output = field("completion_tokens", "output_tokens")
        return prompt, output

    def _raise_if_truncated(self, response, max_tokens: int) -> None:
        """finish_reason=length 或输出顶满 max_tokens 时判定为截断并抛出。

        该检查发生在 JSON 解析之前：截断内容必然解析失败，把它当普通解析
        错误重试只会以相同配置反复烧 token。
        """

        if response is None or not getattr(response, "choices", None):
            return
        finish_reason = str(getattr(response.choices[0], "finish_reason", "") or "")
        prompt_tokens, output_tokens = self._usage_tokens(response)
        provider = self.last_provider_used or self._endpoint.protocol
        model = str(self._endpoint.model or self.model or "")
        if finish_reason == "length":
            raise LLMOutputTruncatedError(
                finish_reason=finish_reason,
                output_tokens=output_tokens,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                provider=provider,
                model=model,
            )
        if output_tokens is not None and max_tokens > 0 and output_tokens >= max_tokens:
            raise LLMOutputTruncatedError(
                finish_reason=finish_reason or "unknown",
                output_tokens=output_tokens,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                provider=provider,
                model=model,
                suspected=True,
            )

    def _loads_json(self, content: str) -> dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.S)
            if fenced:
                return json.loads(fenced.group(1))
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                return json.loads(content[start : end + 1])
            raise

    def _message_text(self, message) -> str:
        content = getattr(message, "content", None)
        if content:
            return content
        return getattr(message, "reasoning_content", None) or ""

    @property
    def vision_available(self) -> bool:
        """VLM 评分能力是否可用：script 端点已配置且适配器声明支持图片输入。

        质量审核门禁据此判断能力，未配置时调用方必须把审核标记为
        unsupported，而不是假装通过。
        """
        self._sync_config()
        if not self.available:
            return False
        try:
            adapter = self._adapter_for(self._endpoint)
        except Exception:
            return False
        return bool(getattr(adapter.capabilities, "vision", False))

    @property
    def vision_provider_label(self) -> str:
        self._sync_config()
        model = self.vision_model or self.model
        return f"vlm:{self._endpoint.protocol}:{model}"

    async def call_with_images(self, prompt: str, image_paths: list[str]) -> str:
        """多图视觉调用：第 1 张起依次作为附图（用于 VLM 质量审核）。"""
        import base64
        import mimetypes

        parts: list[dict] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
            parts.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
            )

        adapter = self._adapter_for(self._endpoint)
        debug_request_id = record_api_request(
            api="LLM Vision",
            provider=self._endpoint.protocol,
            model=self.vision_model,
            params={"temperature": 0.3, "max_tokens": self.max_tokens, "input_images": 1},
            prompt=prompt,
        )
        started = time.monotonic()
        try:
            response = await self.client.chat.completions.create(
                model=self.vision_model,
                messages=[{"role": "user", "content": parts}],
                temperature=0.3,
                max_tokens=self.max_tokens,
            )
        except asyncio.CancelledError:
            self._record_cancelled(adapter, self.vision_model, started)
            raise
        except Exception as exc:
            self._record_failure(adapter, self.vision_model, started)
            record_api_result(debug_request_id, api="LLM Vision", status="error", message=f"视觉模型调用失败：{exc}")
            raise
        self._record(adapter, response, self.vision_model, started)
        text = self._message_text(response.choices[0].message)
        record_api_result(debug_request_id, api="LLM Vision", status="success", message="视觉模型调用成功")
        return text

    async def call_with_image(self, prompt: str, image_path: str) -> str:
        return await self.call_with_images(prompt, [image_path])

    async def call_json_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
        temperature: float = 0.2,
        max_retries: int = 2,
    ) -> dict:
        """多图视觉调用 + JSON 清洗（质量审核的 VLM 评分入口）。"""
        prompt = f"{system_prompt}\n\n{user_prompt}"
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                content = await self.call_with_images(prompt, image_paths)
                return self._loads_json(content or "{}")
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    continue
        raise ValueError(f"VLM JSON 解析失败，已重试 {max_retries} 次: {last_error}")
