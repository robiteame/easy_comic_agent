from __future__ import annotations

import json
import os
import socket
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import model_config_service
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


class ModelConfigSecurityTests(unittest.TestCase):
    def test_credentials_are_atomically_stored_with_private_permissions_and_masked(self) -> None:
        payload = {
            "script": {
                "provider": "openai",
                "api_key": "test-secret-value",
                "base_url": "https://api.example.test/v1",
                "model": "example-model",
            }
        }

        public_dns = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        with tempfile.TemporaryDirectory(prefix="comic-agent-config-") as root:
            with (
                patch.object(model_config_service.settings, "DATA_DIR", Path(root)),
                patch.object(model_config_service.socket, "getaddrinfo", return_value=public_dns),
            ):
                model_config_service._save_raw(payload)
                config_path = Path(root) / "model_api_config.json"

                self.assertEqual(json.loads(config_path.read_text(encoding="utf-8")), payload)
                # POSIX permission bits are not enforceable via os.chmod on
                # Windows; the private-mode contract is a POSIX-only check.
                if os.name != "nt":
                    self.assertEqual(stat.S_IMODE(config_path.stat().st_mode) & 0o077, 0)
                self.assertEqual(model_config_service.get_model_config()["categories"]["script"]["api_key"], "********")
                self.assertEqual(list(Path(root).glob(".model_api_config.json.*.tmp")), [])

    def test_base_url_rejects_ambiguous_ip_and_non_public_dns(self) -> None:
        with self.assertRaisesRegex(ValueError, "非标准 IP"):
            model_config_service._validate_base_url("http://2130706433/v1")

        private_dns = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
        with patch.object(model_config_service.socket, "getaddrinfo", return_value=private_dns):
            with self.assertRaisesRegex(ValueError, "本机/内网"):
                model_config_service._validate_base_url("https://api.example.test/v1")

        reserved_dns = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.10", 443))]
        with patch.object(model_config_service.socket, "getaddrinfo", return_value=reserved_dns):
            with self.assertRaisesRegex(ValueError, "本机/内网"):
                model_config_service._validate_base_url("https://api.example.test/v1")

    def test_endpoint_identity_tracks_only_scheme_host_and_effective_port(self) -> None:
        identity = model_config_service._endpoint_identity
        self.assertEqual(identity("http://API.EXAMPLE.test/v1"), identity("http://api.example.test:80/v2"))
        self.assertNotEqual(identity("http://api.example.test"), identity("https://api.example.test"))
        self.assertNotEqual(identity("https://api.example.test"), identity("https://api.example.test:8443"))

    def test_image_endpoint_marker_clears_effective_key(self) -> None:
        from services.providers.endpoint import get_endpoint

        with tempfile.TemporaryDirectory(prefix="comic-agent-config-") as root:
            with patch.object(model_config_service.settings, "DATA_DIR", Path(root)):
                model_config_service._save_raw(
                    {
                        "image": {
                            "protocol": "stability",
                            "api_key": "old-key",
                            model_config_service.API_KEY_REQUIRED: True,
                        }
                    }
                )
                # 换端点未换 key 的标记必须让生效端点返回空密钥。
                self.assertEqual(get_endpoint("image").api_key, "")
                self.assertEqual(model_config_service.get_model_config()["categories"]["image"]["api_key"], "")

    def test_endpoint_change_requires_explicit_new_api_key(self) -> None:
        public_dns = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        original_key = model_config_service.settings.OPENAI_API_KEY
        original_base_url = model_config_service.settings.OPENAI_BASE_URL
        original_provider = model_config_service.settings.LLM_PROVIDER
        try:
            with tempfile.TemporaryDirectory(prefix="comic-agent-config-") as root:
                with (
                    patch.object(model_config_service.settings, "DATA_DIR", Path(root)),
                    patch.object(model_config_service.socket, "getaddrinfo", return_value=public_dns),
                ):
                    model_config_service._save_raw(
                        {
                            "script": {
                                "protocol": "openai-chat",
                                "api_key": "old-key",
                                "base_url": "https://old.example.test/v1",
                            }
                        }
                    )
                    model_config_service.settings.OPENAI_API_KEY = "old-key"

                    model_config_service.save_model_config(
                        {"script": {"base_url": "https://new.example.test/v1", "api_key": "********"}}
                    )
                    stored = model_config_service._load_raw()["script"]
                    self.assertNotIn("api_key", stored)
                    self.assertTrue(stored[model_config_service.API_KEY_REQUIRED])
                    # script 端点改地址未换 key 后，生效端点必须返回空密钥。
                    from services.providers.endpoint import get_endpoint

                    self.assertEqual(get_endpoint("script").api_key, "")
                    self.assertEqual(model_config_service.get_model_config()["categories"]["script"]["api_key"], "")

                    model_config_service.settings.OPENAI_API_KEY = "environment-old-key"
                    model_config_service.apply_model_config_to_settings()
                    self.assertEqual(get_endpoint("script").api_key, "")

                    model_config_service.save_model_config(
                        {"script": {"base_url": "https://third.example.test/v1", "api_key": "new-key"}}
                    )
                    stored = model_config_service._load_raw()["script"]
                    self.assertNotIn(model_config_service.API_KEY_REQUIRED, stored)
                    self.assertEqual(get_endpoint("script").api_key, "new-key")
        finally:
            model_config_service.settings.OPENAI_API_KEY = original_key
            model_config_service.settings.OPENAI_BASE_URL = original_base_url
            model_config_service.settings.LLM_PROVIDER = original_provider


if __name__ == "__main__":
    unittest.main()
