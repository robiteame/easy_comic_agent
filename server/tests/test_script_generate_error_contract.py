"""剧本生成的上游失败必须返回稳定、可展示的结构化错误。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from api.routes import script as script_route
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


class _FailingLLM:
    available = True

    async def call(self, *_args, **_kwargs):
        raise RuntimeError("Error code: 401 - Incorrect API key provided")


class _EmptyLLM:
    available = True

    async def call(self, *_args, **_kwargs):
        return "   "


class ScriptGenerateErrorContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_authentication_failure_is_structured_and_sanitized(self) -> None:
        data = script_route.ScriptGenerateRequest(prompt="测试创作方向")
        with (
            patch.object(script_route, "llm_service", _FailingLLM()),
            self.assertRaises(HTTPException) as raised,
        ):
            await script_route._generate_script_text(data)

        detail = raised.exception.detail
        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(detail["status"], "provider_error")
        self.assertEqual(detail["error_code"], "provider_config_error")
        self.assertIn("API Key", detail["message"])
        self.assertNotIn("Incorrect API key", detail["message"])

    async def test_empty_provider_output_is_not_a_server_crash(self) -> None:
        data = script_route.ScriptGenerateRequest(prompt="测试创作方向")
        with (
            patch.object(script_route, "llm_service", _EmptyLLM()),
            self.assertRaises(HTTPException) as raised,
        ):
            await script_route._generate_script_text(data)

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.detail["error_code"], "provider_error")


if __name__ == "__main__":
    unittest.main()
