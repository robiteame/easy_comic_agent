"""Provider HTTP 调用的瞬态错误重试。

只重试「重试可能成功」的失败：连接类/读类传输错误（``httpx.TransportError``，
含超时），以及网关类状态码 429/502/503/504。4xx 业务错误（鉴权、参数、配额）
是确定性失败，原样透传给调用方报错。

注意：对「创建计费任务」的 POST 做传输错误重试时，存在极小概率重复建任务
（请求已被服务端受理但响应丢失）。attempts 保持小值、退避从 1 秒起，
把概率与代价都压到最低；轮询类幂等 GET 则可以放心重试。
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

TRANSIENT_STATUS_CODES = frozenset({429, 502, 503, 504})


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    attempts: int = 3,
    backoff_seconds: float = 1.0,
    name: str = "provider",
    **kwargs,
) -> httpx.Response:
    """带指数退避的请求封装：瞬态传输错误与网关状态码自动重试，其余立即返回。"""

    last_error: Exception | None = None
    for index in range(max(1, attempts)):
        if index:
            await asyncio.sleep(backoff_seconds * (2 ** (index - 1)))
        try:
            response = await client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            last_error = exc
            logger.warning(
                "[%s] %s %s 传输错误（第 %d/%d 次尝试）: %r",
                name, method, url, index + 1, attempts, exc,
            )
            continue
        if response.status_code in TRANSIENT_STATUS_CODES and index < attempts - 1:
            await response.aclose()
            logger.warning(
                "[%s] %s %s 网关瞬态状态 %d（第 %d/%d 次尝试）",
                name, method, url, response.status_code, index + 1, attempts,
            )
            continue
        return response
    assert last_error is not None
    raise last_error
