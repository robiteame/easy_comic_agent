"""端点配置（EndpointConfig）、协议注册表与旧配置迁移的单元测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from services import model_config_service
from services.llm_service import LLMService
from services.providers import UnknownProtocolError, get_adapter
from services.providers.endpoint import EndpointConfig, get_endpoint
from test_environment import TEST_ROOT  # noqa: F401,E402


def _isolated_store():
    """隔离 DATA_DIR 的上下文管理器，避免污染真实 data 目录。"""
    return tempfile.TemporaryDirectory(prefix="comic-agent-endpoint-")


class EndpointConfigTests(unittest.TestCase):
    def test_script_endpoint_defaults_follow_legacy_llm_provider(self) -> None:
        settings_obj = model_config_service.settings
        original = (
            settings_obj.LLM_PROVIDER,
            settings_obj.MIMO_API_KEY,
            settings_obj.MIMO_BASE_URL,
            settings_obj.MIMO_MODEL,
        )
        try:
            settings_obj.LLM_PROVIDER = "mimo"
            settings_obj.MIMO_API_KEY = "mimo-key"
            settings_obj.MIMO_BASE_URL = "https://mimo.example.test/v1"
            settings_obj.MIMO_MODEL = "mimo-v2.5"
            endpoint = get_endpoint("script")
            self.assertEqual(endpoint.protocol, "openai-chat")
            self.assertEqual(endpoint.auth_style, "api-key-header")
            self.assertEqual(endpoint.api_key, "mimo-key")
            self.assertEqual(endpoint.model, "mimo-v2.5")
            self.assertTrue(endpoint.param("max_tokens"))
        finally:
            (
                settings_obj.LLM_PROVIDER,
                settings_obj.MIMO_API_KEY,
                settings_obj.MIMO_BASE_URL,
                settings_obj.MIMO_MODEL,
            ) = original

    def test_stored_json_overrides_env_defaults(self) -> None:
        public_dns = [(2, 1, 6, "", ("93.184.216.34", 443))]
        with _isolated_store() as root:
            with (
                patch.object(model_config_service.settings, "DATA_DIR", Path(root)),
                patch.object(model_config_service.socket, "getaddrinfo", return_value=public_dns),
            ):
                model_config_service._save_raw(
                    {
                        "script": {
                            "protocol": "openai-chat",
                            "base_url": "https://script.example.test/v1",
                            "api_key": "script-key",
                            "model": "script-model",
                            "params": {"max_tokens": 1234},
                        }
                    }
                )
                endpoint = get_endpoint("script")
                self.assertEqual(endpoint.base_url, "https://script.example.test/v1")
                self.assertEqual(endpoint.api_key, "script-key")
                self.assertEqual(endpoint.model, "script-model")
                self.assertEqual(endpoint.param("max_tokens"), 1234)
                self.assertEqual(endpoint.auth_style, "bearer")

    def test_api_key_required_marker_forces_empty_secret(self) -> None:
        with _isolated_store() as root:
            with patch.object(model_config_service.settings, "DATA_DIR", Path(root)):
                model_config_service._save_raw(
                    {
                        "script": {
                            "protocol": "openai-chat",
                            "base_url": "https://script.example.test/v1",
                            "api_key": "old-key",
                            model_config_service.API_KEY_REQUIRED: True,
                        }
                    }
                )
                self.assertEqual(get_endpoint("script").api_key, "")

    def test_unknown_protocol_raises_with_options(self) -> None:
        with self.assertRaises(UnknownProtocolError) as ctx:
            get_adapter("script", "wat-protocol")
        self.assertIn("openai-chat", str(ctx.exception))

    def test_image_defaults_follow_legacy_image_provider(self) -> None:
        settings_obj = model_config_service.settings
        original = (
            settings_obj.IMAGE_PROVIDER,
            settings_obj.ARK_API_KEY,
            settings_obj.SEEDDANCE_BASE_URL,
            settings_obj.SEEDREAM_MODEL,
        )
        try:
            settings_obj.IMAGE_PROVIDER = "Doubao-Seedream-5.0-lite"
            settings_obj.ARK_API_KEY = "ark-key"
            endpoint = get_endpoint("image")
            self.assertEqual(endpoint.protocol, "ark-seedream")
            self.assertEqual(endpoint.api_key, "ark-key")
            settings_obj.IMAGE_PROVIDER = "local"
            self.assertEqual(get_endpoint("image").protocol, "placeholder")
            settings_obj.IMAGE_PROVIDER = "stability"
            self.assertEqual(get_endpoint("image").protocol, "stability")
        finally:
            (
                settings_obj.IMAGE_PROVIDER,
                settings_obj.ARK_API_KEY,
                settings_obj.SEEDDANCE_BASE_URL,
                settings_obj.SEEDREAM_MODEL,
            ) = original


class LegacyMigrationTests(unittest.TestCase):
    def test_legacy_store_is_migrated_in_place(self) -> None:
        import json

        legacy = {
            "script": {
                "provider": "mimo",
                "api_key": "mimo-secret",
                "base_url": "https://mimo.example.test/v1",
                "model": "mimo-v2.5",
                "max_tokens": 2048,
            },
            "image": {
                "provider": "Doubao-Seedream-4.5",
                "api_key": "img-key",
                "base_url": "https://ark.example.test/api/v3",
                "model": "Doubao-Seedream-4.5",
                "image_size": "1440x2560",
            },
            "video": {
                "provider": "Doubao-Seedance-1.5-pro",
                "base_url": "https://ark.example.test/api/v3",
                "model": "doubao-seedance-1-5-pro-251215",
            },
            "voice": {
                "api_key": "tts-key",
                "base_url": "https://mimo.example.test/v1",
                "model": "mimo-v2.5-tts",
                "voice": "冰糖",
                "format": "wav",
            },
        }
        with _isolated_store() as root:
            with patch.object(model_config_service.settings, "DATA_DIR", Path(root)):
                model_config_service._save_raw(legacy)
                self.assertTrue(model_config_service.migrate_store())
                # 二次迁移应为幂等 no-op
                self.assertFalse(model_config_service.migrate_store())

                migrated = json.loads((Path(root) / "model_api_config.json").read_text(encoding="utf-8"))
                self.assertEqual(migrated["script"]["protocol"], "openai-chat")
                self.assertEqual(migrated["script"]["auth_style"], "api-key-header")
                self.assertEqual(migrated["script"]["api_key"], "mimo-secret")
                self.assertEqual(migrated["script"]["params"]["max_tokens"], 2048)
                self.assertNotIn("provider", migrated["script"])
                self.assertEqual(migrated["image"]["protocol"], "ark-seedream")
                self.assertEqual(migrated["image"]["params"]["image_size"], "1440x2560")
                self.assertEqual(migrated["video"]["protocol"], "ark-seedance")
                self.assertEqual(migrated["voice"]["protocol"], "mimo-tts")
                self.assertEqual(migrated["voice"]["params"]["voice"], "冰糖")

                endpoint = get_endpoint("script")
                self.assertEqual(endpoint.auth_style, "api-key-header")
                self.assertEqual(endpoint.param("max_tokens"), 2048)
                image_endpoint = get_endpoint("image")
                self.assertEqual(image_endpoint.protocol, "ark-seedream")
                self.assertEqual(image_endpoint.param("image_size"), "1440x2560")

    def test_get_returns_effective_values_matching_legacy_semantics(self) -> None:
        public_dns = [(2, 1, 6, "", ("93.184.216.34", 443))]
        with _isolated_store() as root:
            with (
                patch.object(model_config_service.settings, "DATA_DIR", Path(root)),
                patch.object(model_config_service.socket, "getaddrinfo", return_value=public_dns),
            ):
                model_config_service._save_raw(
                    {
                        "script": {
                            "provider": "openai",
                            "api_key": "k",
                            "base_url": "https://s.example.test/v1",
                            "model": "m",
                        }
                    }
                )
                categories = model_config_service.get_model_config()["categories"]
                script = categories["script"]
                self.assertEqual(script["api_key"], "********")
                self.assertEqual(script["base_url"], "https://s.example.test/v1")
                self.assertEqual(script["model"], "m")
                # 旧前端兼容镜像
                self.assertEqual(script["provider"], "openai-chat")
                self.assertIn("max_tokens", script)


class ScriptFallbackTests(unittest.TestCase):
    def test_fallback_suppressed_for_same_endpoint_identity(self) -> None:
        settings_obj = model_config_service.settings
        original = (settings_obj.OPENAI_API_KEY, settings_obj.OPENAI_BASE_URL, settings_obj.LLM_PROVIDER)
        try:
            settings_obj.OPENAI_API_KEY = "same-key"
            settings_obj.OPENAI_BASE_URL = ""
            settings_obj.LLM_PROVIDER = "mimo"  # 主端点来自 MIMO_*，备端点来自 OPENAI_*
            service = LLMService()
            service._sync_config()
            self.assertIsNotNone(service._fallback_endpoint)

            settings_obj.OPENAI_BASE_URL = service._endpoint.base_url  # 与主端点同一地址
            service._sync_config()
            self.assertIsNone(service._fallback_endpoint)
        finally:
            settings_obj.OPENAI_API_KEY, settings_obj.OPENAI_BASE_URL, settings_obj.LLM_PROVIDER = original

    async def _exercise_fallback(self) -> tuple[str, str]:
        primary = EndpointConfig(
            protocol="openai-chat", base_url="https://primary.example.test/v1", api_key="k1", model="m1"
        )
        backup = EndpointConfig(
            protocol="openai-chat", base_url="https://backup.example.test/v1", api_key="k2", model="m2"
        )

        class FakeAdapter:
            def __init__(self, endpoint):
                self.endpoint = endpoint

            @property
            def client(self):
                return None

            async def complete(self, **kwargs):
                if self.endpoint.api_key == "k1":
                    raise RuntimeError("primary down")
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

            async def complete_json(self, **kwargs):
                return await self.complete(**kwargs)

        service = LLMService.__new__(LLMService)
        with (
            patch("services.llm_service.get_endpoint", side_effect=lambda cap: primary if cap == "script" else backup),
            patch("services.llm_service.get_adapter", side_effect=lambda cap, proto: FakeAdapter),
        ):
            service._adapters = {}
            service._sync_config()
            text = await service.call("sys", "user")
        return text, service.last_provider_used

    def test_primary_failure_falls_back_to_backup(self) -> None:
        import asyncio

        text, label = asyncio.run(self._exercise_fallback())
        self.assertEqual(text, "ok")
        self.assertIn("m2", label)


if __name__ == "__main__":
    unittest.main()
