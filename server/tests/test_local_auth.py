from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from services.local_auth import (
    TOKEN_ENV,
    configured_token,
    is_allowed_websocket_origin,
    is_public_path,
    is_token_valid,
    request_token,
)
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


class LocalAuthTests(unittest.TestCase):
    def test_missing_configuration_keeps_development_compatible(self) -> None:
        self.assertEqual(configured_token({}), "")
        self.assertTrue(is_token_valid("", None))

    def test_token_is_exact_and_constant_time_compared(self) -> None:
        expected = "a" * 64
        self.assertTrue(is_token_valid(expected, expected))
        self.assertFalse(is_token_valid(expected, expected.upper()))
        self.assertFalse(is_token_valid(expected, expected[:-1]))

    def test_header_precedes_query_and_health_is_public(self) -> None:
        self.assertEqual(request_token({"x-comic-agent-token": "header"}, "query"), "header")
        self.assertEqual(request_token({}, "query"), "query")
        self.assertTrue(is_public_path("/health"))
        self.assertTrue(is_public_path("/readyz"))
        self.assertFalse(is_public_path("/api/project"))
        self.assertEqual(configured_token({TOKEN_ENV: " token "}), "token")

    def test_fastapi_middleware_protects_api_but_not_health(self) -> None:
        from fastapi.testclient import TestClient

        from main import app

        with patch.dict(os.environ, {TOKEN_ENV: "integration-token"}):
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.get("/api/project").status_code, 401)
                self.assertEqual(
                    client.get("/api/project", headers={"X-Comic-Agent-Token": "integration-token"}).status_code,
                    200,
                )

    def test_websocket_origin_allowlist(self) -> None:
        self.assertTrue(is_allowed_websocket_origin(None))
        self.assertTrue(is_allowed_websocket_origin("null"))
        self.assertTrue(is_allowed_websocket_origin("file://"))
        self.assertTrue(is_allowed_websocket_origin("http://localhost:5173"))
        self.assertFalse(is_allowed_websocket_origin("https://evil.example"))

    def test_websocket_requires_local_origin_and_token(self) -> None:
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        from main import app

        with patch.dict(os.environ, {TOKEN_ENV: "integration-token"}):
            with TestClient(app) as client:
                with self.assertRaises(WebSocketDisconnect) as raised:
                    with client.websocket_connect(
                        "/ws/test-project",
                        headers={
                            "Origin": "https://evil.example",
                            "X-Comic-Agent-Token": "integration-token",
                        },
                    ):
                        pass
                self.assertEqual(raised.exception.code, 1008)

                with client.websocket_connect(
                    "/ws/test-project",
                    headers={
                        "Origin": "http://localhost:5173",
                        "X-Comic-Agent-Token": "integration-token",
                    },
                ) as socket:
                    socket.send_text("ping")
                    self.assertEqual(socket.receive_json(), {"type": "pong"})


if __name__ == "__main__":
    unittest.main()
