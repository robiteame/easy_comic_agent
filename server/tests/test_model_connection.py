"""POST /api/settings/test-connection 的行为与安全测试。

覆盖：ok / fail / unsupported_check / 未配置 Key / 消息脱敏 / 掩码密钥复用 /
换地址扣留旧密钥 / SSRF 拒绝 / 超时。全部外呼都由 MockTransport 截获，
不产生真实网络请求，也不会消费任何供应商额度。
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient  # noqa: E402

from config import settings  # noqa: E402
from main import app  # noqa: E402
from services import connection_test_service, model_config_service  # noqa: E402
from services.providers.tts_mimo import MimoTTSAdapter  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402

SECRET_KEY = "sk-test-abcdefghijklmnop1234"
SECRET_PATH = "/Users/someone/private/keys.txt"
# 公网 IP 字面量：通过 SSRF 校验且无需 DNS 解析，测试环境稳定。
PUBLIC_BASE = "https://93.184.216.34/v1"
ARK_BASE = "https://93.184.216.34/api/v3"
DASHSCOPE_BASE = "https://93.184.216.34/api/v1"


def _client_factory(transport: httpx.MockTransport):
    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(10.0), follow_redirects=False)

    return factory


class ConnectionTestApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="comic-agent-conn-test-")
        self._data_dir = patch.object(settings, "DATA_DIR", Path(self._tmp.name))
        self._data_dir.start()
        self.requests: list[httpx.Request] = []

    def tearDown(self) -> None:
        self._data_dir.stop()
        self._tmp.cleanup()

    def _post(self, payload: dict):
        return self.client.post("/api/settings/test-connection", json=payload)

    def _patch_transport(self, handler):
        transport = httpx.MockTransport(handler)
        return patch.object(connection_test_service, "_build_client", _client_factory(transport))

    @staticmethod
    def _assert_no_secrets(test: unittest.TestCase, response) -> None:
        rendered = json.dumps(response.json(), ensure_ascii=False)
        test.assertNotIn(SECRET_KEY, rendered)
        test.assertNotIn(SECRET_PATH, rendered)
        test.assertNotIn("/Users/", rendered)

    # --- llm ----------------------------------------------------------------

    def test_llm_ok_uses_minimal_completion(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "p"}}]})

        with self._patch_transport(handler):
            response = self._post(
                {
                    "capability": "llm",
                    "config": {
                        "protocol": "openai-chat",
                        "base_url": PUBLIC_BASE,
                        "api_key": "key-one",
                        "model": "test-model",
                    },
                }
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["provider"], "openai-chat")
        self.assertEqual(body["model"], "test-model")
        self.assertIsInstance(body["latency_ms"], int)
        self.assertGreaterEqual(body["latency_ms"], 0)
        self.assertIn("成功", body["message"])
        self.assertEqual(len(self.requests), 1)
        sent = json.loads(self.requests[0].content)
        self.assertEqual(self.requests[0].url.path, "/v1/chat/completions")
        self.assertEqual(sent["max_tokens"], 1)
        self.assertEqual(self.requests[0].headers["Authorization"], "Bearer key-one")

    def test_llm_invalid_key_message_is_sanitized(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"error": {"message": f"invalid api_key {SECRET_KEY} loaded from {SECRET_PATH}"}},
            )

        with self._patch_transport(handler):
            response = self._post(
                {
                    "capability": "llm",
                    "config": {
                        "protocol": "openai-chat",
                        "base_url": PUBLIC_BASE,
                        "api_key": SECRET_KEY,
                        "model": "test-model",
                    },
                }
            )
        body = response.json()
        self.assertEqual(body["status"], "fail")
        self.assertIn("401", body["message"])
        self.assertIn("API Key 无效", body["message"])
        self._assert_no_secrets(self, response)

    # --- image ---------------------------------------------------------------

    def test_image_placeholder_is_ok_without_network(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("本地占位模式不应发起任何外部请求")

        with self._patch_transport(handler):
            response = self._post({"capability": "image", "config": {"provider": "local", "protocol": "placeholder"}})
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["provider"], "placeholder")
        self.assertIn("本地占位", body["message"])

    def test_image_ark_without_models_listing_is_unsupported(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(404, text="not found")

        with self._patch_transport(handler):
            response = self._post(
                {
                    "capability": "image",
                    "config": {
                        "protocol": "ark-seedream",
                        "base_url": ARK_BASE,
                        "api_key": "key-two",
                        "model": "doubao-seedream-5-0",
                    },
                }
            )
        body = response.json()
        self.assertEqual(body["status"], "unsupported_check")
        self.assertEqual(body["provider"], "ark-seedream")
        self.assertIn("未提供模型列表", body["message"])
        self.assertEqual(self.requests[0].url.path, "/api/v3/models")
        self.assertEqual(self.requests[0].headers["Authorization"], "Bearer key-two")

    def test_image_qwen_probes_compatible_mode_first(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.url.path == "/compatible-mode/v1/models":
                return httpx.Response(200, json={"data": [{"id": "qwen-image"}]})
            return httpx.Response(404)

        with self._patch_transport(handler):
            response = self._post(
                {
                    "capability": "image",
                    "config": {
                        "protocol": "qwen-image",
                        "base_url": DASHSCOPE_BASE,
                        "api_key": "key-three",
                        "model": "qwen-image",
                    },
                }
            )
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(self.requests[0].url.path, "/compatible-mode/v1/models")

    def test_image_stability_account_probe(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.url.path == "/v1/user/account":
                return httpx.Response(200, json={"id": "acc", "credits": 1})
            return httpx.Response(404)

        with self._patch_transport(handler):
            response = self._post(
                {
                    "capability": "image",
                    "config": {
                        "protocol": "stability",
                        "base_url": "https://93.184.216.34/v2beta",
                        "api_key": "key-four",
                    },
                }
            )
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["provider"], "stability")
        self.assertEqual(self.requests[0].url.path, "/v1/user/account")

    # --- video ----------------------------------------------------------------

    def test_video_ark_seedance_model_listing_ok(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json={"data": []})

        with self._patch_transport(handler):
            response = self._post(
                {
                    "capability": "video",
                    "config": {
                        "protocol": "ark-seedance",
                        "base_url": ARK_BASE,
                        "api_key": "key-five",
                        "model": "doubao-seedance-2-0",
                    },
                }
            )
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["provider"], "ark-seedance")
        self.assertEqual(self.requests[0].url.path, "/api/v3/models")

    # --- tts ------------------------------------------------------------------

    def test_tts_mimo_synthesis_ok(self) -> None:
        synthesize = AsyncMock(return_value=b"RIFF-audio-bytes")
        with patch.object(MimoTTSAdapter, "synthesize", synthesize):
            response = self._post(
                {
                    "capability": "tts",
                    "config": {
                        "protocol": "mimo-tts",
                        "base_url": PUBLIC_BASE,
                        "api_key": "key-six",
                        "model": "mimo-tts-x",
                    },
                }
            )
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["provider"], "mimo-tts")
        self.assertIn("测试音频", body["message"])
        request = synthesize.await_args.args[0]
        self.assertEqual(request.text, "好")

    def test_tts_tencent_bad_credentials_format_fails(self) -> None:
        response = self._post(
            {
                "capability": "tts",
                "config": {
                    "protocol": "tencent-tts",
                    "base_url": "https://93.184.216.34",
                    "api_key": "missing-colon",
                },
            }
        )
        body = response.json()
        self.assertEqual(body["status"], "fail")
        self.assertIn("SecretId:SecretKey", body["message"])

    def test_tts_failure_message_is_sanitized(self) -> None:
        synthesize = AsyncMock(side_effect=RuntimeError(f"Mimo TTS 调用失败: 401 {SECRET_KEY} {SECRET_PATH}"))
        with patch.object(MimoTTSAdapter, "synthesize", synthesize):
            response = self._post(
                {
                    "capability": "tts",
                    "config": {
                        "protocol": "mimo-tts",
                        "base_url": PUBLIC_BASE,
                        "api_key": SECRET_KEY,
                        "model": "mimo-tts-x",
                    },
                }
            )
        self.assertEqual(response.json()["status"], "fail")
        self._assert_no_secrets(self, response)
        self.assertIn("调用失败", response.json()["message"])

    # --- 配置解析与安全 ---------------------------------------------------------

    def test_missing_api_key_fails_without_outbound_call(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("未配置密钥时不应发起外部请求")

        env_patches = (
            patch.object(settings, "LLM_PROVIDER", ""),
            patch.object(settings, "OPENAI_API_KEY", ""),
            patch.object(settings, "MIMO_API_KEY", ""),
        )
        with self._patch_transport(handler), env_patches[0], env_patches[1], env_patches[2]:
            response = self._post(
                {
                    "capability": "llm",
                    "config": {"protocol": "openai-chat", "base_url": PUBLIC_BASE, "model": "test-model"},
                }
            )
        body = response.json()
        self.assertEqual(body["status"], "fail")
        self.assertIn("未配置 API Key", body["message"])

    def test_no_config_uses_saved_endpoint_and_key(self) -> None:
        model_config_service._save_raw(
            {
                "script": {
                    "protocol": "openai-chat",
                    "base_url": PUBLIC_BASE,
                    "api_key": "stored-secret-key",
                    "model": "stored-model",
                }
            }
        )

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "p"}}]})

        with self._patch_transport(handler):
            response = self._post({"capability": "llm"})
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["model"], "stored-model")
        self.assertEqual(self.requests[0].headers["Authorization"], "Bearer stored-secret-key")
        self.assertNotIn("stored-secret-key", json.dumps(body, ensure_ascii=False))

    def test_masked_key_reuses_saved_secret_for_same_endpoint_only(self) -> None:
        model_config_service._save_raw(
            {
                "script": {
                    "protocol": "openai-chat",
                    "base_url": PUBLIC_BASE,
                    "api_key": "stored-secret-key",
                    "model": "stored-model",
                }
            }
        )

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "p"}}]})

        with self._patch_transport(handler):
            same_endpoint = self._post(
                {
                    "capability": "llm",
                    "config": {
                        "protocol": "openai-chat",
                        "base_url": PUBLIC_BASE,
                        "api_key": "********",
                        "model": "stored-model",
                    },
                }
            )
        self.assertEqual(same_endpoint.json()["status"], "ok")
        self.assertEqual(self.requests[0].headers["Authorization"], "Bearer stored-secret-key")
        self.assertNotIn("stored-secret-key", same_endpoint.text)

        # 换地址 + 掩码密钥：必须扣留旧密钥、直接失败，绝不把密钥发往新主机。
        self.requests.clear()

        def failing_handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("换地址未填新密钥时不应发起外部请求")

        with self._patch_transport(failing_handler):
            moved = self._post(
                {
                    "capability": "llm",
                    "config": {
                        "protocol": "openai-chat",
                        "base_url": "https://93.184.217.35/v1",
                        "api_key": "********",
                        "model": "stored-model",
                    },
                }
            )
        body = moved.json()
        self.assertEqual(body["status"], "fail")
        self.assertIn("旧密钥", body["message"])

    def test_private_base_url_is_rejected_without_outbound_call(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("内网地址不应发起外部请求")

        with self._patch_transport(handler):
            response = self._post(
                {
                    "capability": "llm",
                    "config": {
                        "protocol": "openai-chat",
                        "base_url": "http://127.0.0.1:8000/v1",
                        "api_key": "key",
                        "model": "m",
                    },
                }
            )
        body = response.json()
        self.assertEqual(body["status"], "fail")
        self.assertIn("Base URL", body["message"])

    def test_probe_timeout_reports_failure(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(2)
            return httpx.Response(200, json={"choices": []})

        transport = httpx.MockTransport(handler)

        def factory() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(10.0))

        with (
            patch.object(connection_test_service, "_build_client", factory),
            patch.object(connection_test_service, "PROBE_TIMEOUT_SECONDS", 0.2),
        ):
            response = self._post(
                {
                    "capability": "llm",
                    "config": {
                        "protocol": "openai-chat",
                        "base_url": PUBLIC_BASE,
                        "api_key": "key",
                        "model": "m",
                    },
                }
            )
        body = response.json()
        self.assertEqual(body["status"], "fail")
        self.assertIn("超时", body["message"])

    def test_unknown_capability_is_rejected(self) -> None:
        response = self._post({"capability": "audio"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
