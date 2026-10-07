"""P2: 后台任务失败信息的脱敏测试。

确认 WebSocket / 任务表 / 落库备注里都不再出现完整 traceback、本地绝对路径、
API Key 或供应商原始响应，同时保留项目状态更新与任务失败逻辑。
"""

from __future__ import annotations

import asyncio
import logging
import unittest
from unittest.mock import patch

from agent import graph as graph_module  # noqa: E402
from api.routes import script as script_route  # noqa: E402
from api.routes import shot as shot_route  # noqa: E402
from api.websocket import ws_manager  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import BackgroundJob, Character, Project, SceneAsset, Shot  # noqa: E402
from services import (
    error_reporter,  # noqa: E402
    task_registry,  # noqa: E402
)
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402

SECRET_KEY = "sk-abcdefghijklmnop0123456789"
SECRET_PATH = "/Users/someone/private/comic-agent/keys.txt"
SENSITIVE_ERROR = f"provider rejected request with {SECRET_KEY} at {SECRET_PATH}"


def _assert_sanitized(test: unittest.TestCase, text: str) -> None:
    test.assertNotIn("Traceback (most recent call last)", text)
    test.assertNotIn(SECRET_KEY, text)
    test.assertNotIn(SECRET_PATH, text)
    test.assertNotIn("/Users/", text)
    test.assertNotIn('File "', text)


class ErrorReporterUnitTests(unittest.TestCase):
    def test_error_ids_are_short_and_unique(self) -> None:
        identifiers = {error_reporter.new_error_id() for _ in range(200)}
        self.assertEqual(len(identifiers), 200)
        for identifier in identifiers:
            self.assertEqual(len(identifier), 8)
            self.assertTrue(identifier.isalnum())

    def test_redact_removes_secrets_paths_and_truncates(self) -> None:
        sanitized = error_reporter.redact(SENSITIVE_ERROR)
        self.assertNotIn(SECRET_KEY, sanitized)
        self.assertNotIn(SECRET_PATH, sanitized)
        self.assertIn("已脱敏", sanitized)
        self.assertIn("本地路径", sanitized)
        long_text = error_reporter.redact("x" * 5000)
        self.assertLessEqual(len(long_text), error_reporter._MAX_LOG_CHARS + 1)

    def test_error_payload_only_carries_stable_fields(self) -> None:
        payload = error_reporter.error_payload(
            error_type=error_reporter.ERROR_PIPELINE,
            message=SENSITIVE_ERROR,
            error_id="abcd1234",
        )
        self.assertEqual(set(payload), {"type", "error_type", "message", "error_id"})
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error_type"], "pipeline_error")
        self.assertEqual(payload["error_id"], "abcd1234")
        _assert_sanitized(self, payload["message"])

    def test_report_failure_logs_the_traceback_server_side_only(self) -> None:
        try:
            raise RuntimeError(SENSITIVE_ERROR)
        except RuntimeError as exc:
            with self.assertLogs("services.error_reporter", level=logging.ERROR) as captured:
                payload = error_reporter.report_failure(
                    exc,
                    error_type=error_reporter.ERROR_RENDER,
                    message="导出失败",
                    context={"project_id": "p1", "api_key": SECRET_KEY},
                )
        logs = "\n".join(captured.output)
        # 服务端日志保留完整堆栈（含本地路径，便于定位），但密钥一律脱敏。
        self.assertIn("Traceback (most recent call last)", logs)
        self.assertIn("RuntimeError", logs)
        self.assertNotIn(SECRET_KEY, logs)
        self.assertIn(payload["error_id"], logs)
        _assert_sanitized(self, payload["message"])

    def test_failure_note_keeps_only_the_error_id(self) -> None:
        with self.assertLogs("services.error_reporter", level=logging.ERROR):
            note = error_reporter.failure_note(
                RuntimeError(SENSITIVE_ERROR),
                prefix="重新生成故事板失败",
                error_type=error_reporter.ERROR_STORYBOARD,
            )
        _assert_sanitized(self, note)
        self.assertTrue(note.startswith("重新生成故事板失败"))
        self.assertIn("错误编号", note)


class GraphAbortTests(unittest.TestCase):
    def test_graph_abort_keeps_only_the_error_id(self) -> None:
        try:
            raise RuntimeError(SENSITIVE_ERROR)
        except RuntimeError as exc:
            with self.assertLogs("agent.graph", level=logging.ERROR) as captured:
                state = graph_module._abort("compose", exc)
        self.assertEqual(state["current_step"], "aborted")
        self.assertEqual(len(state["errors"]), 1)
        _assert_sanitized(self, state["errors"][0])
        self.assertIn("[compose]", state["errors"][0])
        self.assertIn("Traceback (most recent call last)", "\n".join(captured.output))


class PipelineErrorReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    def setUp(self) -> None:
        self.db = SessionLocal()
        self.db.query(BackgroundJob).delete()
        self.db.query(Shot).delete()
        self.db.query(Character).delete()
        self.db.query(SceneAsset).delete()
        self.db.query(Project).delete()
        self.project = Project(id="sanitize-project", title="脱敏项目", status="draft")
        self.db.add(self.project)
        self.db.commit()
        self.messages: list[tuple[str, dict]] = []

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()

    async def _capture(self, project_id: str, message: dict) -> None:
        self.messages.append((project_id, message))

    def test_storyboard_phase_error_payload_is_sanitized(self) -> None:
        async def failing_run(state):
            raise RuntimeError(SENSITIVE_ERROR)

        with (
            patch.object(ws_manager, "send_to_project", self._capture),
            patch.object(script_route.script_parser, "run", failing_run),
        ):
            with self.assertLogs("services.error_reporter", level=logging.ERROR):
                with self.assertRaises(RuntimeError):
                    asyncio.run(
                        script_route._run_storyboard_phase(
                            self.project.id,
                            {"project_id": self.project.id, "user_input": "剧本"},
                        )
                    )

        self.assertTrue(self.messages)
        _, payload = self.messages[-1]
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error_type"], error_reporter.ERROR_PIPELINE)
        self.assertIn("error_id", payload)
        _assert_sanitized(self, str(payload))
        # 项目状态更新逻辑保持不变。
        self.db.expire_all()
        self.assertEqual(self.db.get(Project, self.project.id).status, "error")

    def test_regenerate_failure_note_and_payload_are_sanitized(self) -> None:
        shot = Shot(id="sanitize-shot", project_id=self.project.id, sequence=1, version=1)
        self.db.add(shot)
        self.db.commit()

        async def failing_baselines(*args, **kwargs):
            raise RuntimeError(SENSITIVE_ERROR)

        with (
            patch.object(ws_manager, "send_to_project", self._capture),
            patch.object(shot_route, "_ensure_scene_baselines", failing_baselines),
        ):
            with self.assertLogs("services.error_reporter", level=logging.ERROR):
                with self.assertRaises(RuntimeError):
                    asyncio.run(shot_route._regenerate_single_shot(shot.id, "重试", expected_version=1))

        self.assertTrue(self.messages)
        _, payload = self.messages[-1]
        self.assertEqual(payload["error_type"], error_reporter.ERROR_STORYBOARD)
        _assert_sanitized(self, str(payload))

        self.db.expire_all()
        stored = self.db.get(Shot, shot.id)
        self.assertEqual(stored.status, "failed")
        self.assertEqual(stored.storyboard_status, "failed")
        _assert_sanitized(self, stored.visual_notes)
        self.assertIn(payload["error_id"], stored.visual_notes)

    def test_task_registry_redacts_the_durable_error(self) -> None:
        key = f"project:{self.project.id}:pipeline:auto"
        self.assertTrue(task_registry.claim(key, f"project:{self.project.id}"))

        async def failing_worker():
            raise RuntimeError(SENSITIVE_ERROR)

        async def scenario() -> None:
            task = task_registry.start(key, failing_worker())
            with self.assertRaises(RuntimeError):
                await task

        with self.assertLogs("services.task_registry", level=logging.ERROR) as captured:
            asyncio.run(scenario())
        logs = "\n".join(captured.output)
        self.assertIn("Traceback (most recent call last)", logs)
        self.assertNotIn(SECRET_KEY, logs)

        job = self.db.query(BackgroundJob).filter_by(idempotency_key=key).one()
        self.assertEqual(job.status, "failed")
        _assert_sanitized(self, job.error)


if __name__ == "__main__":
    unittest.main()
