from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from services import model_discovery_service
from services.providers.endpoint import EndpointConfig


class _Response:
    status_code = 200
    content = b'{"data": []}'

    def __init__(self, payload):
        self.payload = payload
        self.content = b"x" * 10

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Client:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class ModelDiscoveryTests(unittest.TestCase):
    def test_masked_key_reuses_saved_secret_for_same_endpoint(self):
        from api.routes.settings import ModelDiscoveryRequest, discover_model_configs

        discover = AsyncMock(return_value={"models": [], "count": 0, "endpoint": "https://provider.example/v1/models"})
        with (
            patch(
                "api.routes.settings.get_endpoint",
                return_value=EndpointConfig(base_url="https://provider.example/v1", api_key="saved-secret"),
            ),
            patch("api.routes.settings.discover_models", discover),
        ):
            asyncio.run(
                discover_model_configs(
                    ModelDiscoveryRequest(
                        category="script",
                        base_url="https://provider.example/v1",
                        api_key="********",
                    )
                )
            )

        self.assertEqual(discover.await_args.kwargs["api_key"], "saved-secret")

    def test_normalizes_openai_response_and_auth_header(self):
        response = _Response(
            {
                "data": [
                    {"id": "video-native", "owned_by": "demo", "capabilities": {"audio_modes": ["native", "silent"]}},
                    {"id": "video-native"},
                ]
            }
        )
        client = _Client(response)
        with (
            patch.object(model_discovery_service, "_validate_base_url", return_value="https://provider.example/v1"),
            patch.object(model_discovery_service.httpx, "AsyncClient", return_value=client),
        ):
            result = asyncio.run(
                model_discovery_service.discover_models(
                    category="video",
                    base_url="https://provider.example/v1",
                    api_key="secret",
                    protocol="custom",
                )
            )

        self.assertEqual(result["endpoint"], "https://provider.example/v1/models")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["models"][0]["capabilities"]["audio_modes"], ["native", "silent"])
        self.assertEqual(client.calls[0][1]["headers"]["Authorization"], "Bearer secret")

    def test_uses_api_key_header_for_mimo_style_auth(self):
        client = _Client(_Response({"models": ["mimo-v2"]}))
        with (
            patch.object(model_discovery_service, "_validate_base_url", return_value="https://provider.example/v1"),
            patch.object(model_discovery_service.httpx, "AsyncClient", return_value=client),
        ):
            asyncio.run(
                model_discovery_service.discover_models(
                    category="script",
                    base_url="https://provider.example/v1",
                    api_key="secret",
                    auth_style="api-key-header",
                )
            )

        self.assertEqual(client.calls[0][1]["headers"]["api-key"], "secret")
        self.assertNotIn("Authorization", client.calls[0][1]["headers"])
