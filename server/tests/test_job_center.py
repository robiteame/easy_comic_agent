"""后台任务中心的回归测试。

覆盖范围：

- 任务列表分页 / 筛选 / 排序 / 统计；
- 详情不泄露 run_token、不泄露堆栈与本地路径；
- 状态迁移规则（终态是吸收态，不能被改回 running）；
- 取消 queued / running / completed / failed，以及重复取消幂等；
- failed / cancelled / interrupted 的重试与 attempt 历史保留；
- 活动任务重复重试被拒绝、作用域互斥；
- 续跑跳过已经完成的阶段与镜头；
- 服务重启后任务变为 interrupted 且可重试；
- 全局任务 WebSocket 的初始快照、重连快照与事件内容。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from config import settings  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from main import app, jobs_snapshot  # noqa: E402
from models import BackgroundJob, Character, Project, SceneAsset, Shot  # noqa: E402
from services import job_actions, job_dispatch, task_registry  # noqa: E402
from services.job_actions import RESUME_MODE, RETRY_MODE  # noqa: E402
from services.job_dispatch import DispatchResult  # noqa: E402
from services.job_types import can_transition, parse_job_key  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402


def _write_media(relative: str, size: int = 2048) -> str:
    path = settings.OUTPUT_DIR / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return str(path)


class JobCenterTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()
        cls.client = TestClient(app)

    def setUp(self) -> None:
        self.db = SessionLocal()
        self.db.query(BackgroundJob).delete()
        self.db.query(Shot).delete()
        self.db.query(Character).delete()
        self.db.query(SceneAsset).delete()
        self.db.query(Project).delete()
        self.db.commit()

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()

    # --- 工具 ---

    def make_project(self, project_id: str = "job-project", **overrides) -> Project:
        project = Project(id=project_id, title=overrides.pop("title", "任务项目"), **overrides)
        self.db.add(project)
        self.db.commit()
        return project

    def make_shot(self, shot_id: str = "job-shot", project_id: str = "job-project", **overrides) -> Shot:
        shot = Shot(id=shot_id, project_id=project_id, sequence=overrides.pop("sequence", 1), **overrides)
        self.db.add(shot)
        self.db.commit()
        return shot

    def make_job(
        self,
        key: str,
        scope: str,
        *,
        status: str = "running",
        job_type: str = "",
        project_id: str = "",
        error: str = "",
        progress: int = 0,
        attempt: int = 1,
        retry_of: str | None = None,
        updated_at: datetime | None = None,
    ) -> BackgroundJob:
        identity = parse_job_key(key)
        scope_project = scope.split(":", 1)[1] if scope.startswith("project:") else ""
        job = BackgroundJob(
            id=f"job-{key}-{status}-{attempt}",
            idempotency_key=key,
            scope=scope,
            status=status,
            progress=progress,
            error=error,
            error_code="",
            error_message="",
            run_token="secret-run-token",
            job_type=job_type or identity.job_type,
            project_id=project_id or scope_project,
            display_name="测试任务",
            attempt=attempt,
            retry_of=retry_of,
            created_at=updated_at or datetime.utcnow(),
            started_at=updated_at,
            finished_at=updated_at if status in {"completed", "failed", "cancelled", "interrupted"} else None,
            updated_at=updated_at or datetime.utcnow(),
        )
        self.db.add(job)
        self.db.commit()
        return job

    def get_job(self, job_id: str) -> BackgroundJob | None:
        self.db.expire_all()
        return self.db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()


class JobListApiTests(JobCenterTestCase):
    def test_list_supports_pagination_filters_and_default_sort(self) -> None:
        base = datetime.utcnow()
        self.make_project("p-list")
        self.make_job(
            "project:p-list:render", "project:p-list", status="completed", updated_at=base - timedelta(minutes=5)
        )
        self.make_job(
            "project:p-list:storyboard", "project:p-list", status="failed", updated_at=base - timedelta(minutes=4)
        )
        self.make_job("shot:s-list:video", "shot:s-list", status="running", updated_at=base - timedelta(minutes=3))
        self.make_job(
            "project:p-other:render", "project:p-other", status="completed", updated_at=base - timedelta(minutes=2)
        )
        self.make_job(
            "project:p-other:pipeline:auto", "project:p-other", status="running", updated_at=base - timedelta(minutes=1)
        )

        page = self.client.get("/api/jobs", params={"page": 1, "page_size": 2}).json()
        self.assertEqual(page["total"], 5)
        self.assertEqual(page["pages"], 3)
        self.assertEqual(len(page["items"]), 2)
        # 默认按 updated_at DESC
        self.assertEqual(page["items"][0]["project_id"], "p-other")
        self.assertEqual(page["items"][0]["job_type"], "script_pipeline")
        # DTO 时刻必须带 UTC 时区（历史 naive 行按 UTC 解释，见 test_job_dto_timezone）。
        self.assertEqual(
            page["items"][1]["updated_at"],
            (base - timedelta(minutes=2)).replace(tzinfo=UTC).isoformat(),
        )

        second = self.client.get("/api/jobs", params={"page": 2, "page_size": 2}).json()
        self.assertEqual(len(second["items"]), 2)
        self.assertEqual({item["id"] for item in page["items"]} & {item["id"] for item in second["items"]}, set())

        by_project = self.client.get("/api/jobs", params={"project_id": "p-list"}).json()
        self.assertEqual(by_project["total"], 2)
        self.assertTrue(all(item["project_id"] == "p-list" for item in by_project["items"]))

        by_status = self.client.get("/api/jobs", params={"status": "failed"}).json()
        self.assertEqual(by_status["total"], 1)
        self.assertEqual(by_status["items"][0]["status"], "failed")

        by_type = self.client.get("/api/jobs", params={"job_type": "render"}).json()
        self.assertEqual(by_type["total"], 2)
        self.assertTrue(all(item["job_type"] == "render" for item in by_type["items"]))

        active = self.client.get("/api/jobs", params={"active_only": "true"}).json()
        self.assertEqual(active["total"], 2)
        self.assertTrue(all(item["is_active"] for item in active["items"]))

        scoped = self.client.get("/api/jobs", params={"scope": "shot:s-list"}).json()
        self.assertEqual(scoped["total"], 1)

        searched = self.client.get("/api/jobs", params={"q": "p-other"}).json()
        self.assertEqual(searched["total"], 2)

    def test_list_returns_totals_active_count_and_status_statistics(self) -> None:
        self.make_job("project:p-stats:render", "project:p-stats", status="running")
        self.make_job("project:p-stats:storyboard", "project:p-stats", status="failed")
        self.make_job("project:p-stats:pipeline:manual", "project:p-stats", status="completed")

        body = self.client.get("/api/jobs").json()
        self.assertEqual(body["total"], 3)
        self.assertEqual(body["active_count"], 1)
        self.assertEqual(body["status_counts"]["failed"], 1)
        self.assertEqual(body["status_counts"]["completed"], 1)
        self.assertEqual(body["job_type_counts"]["render"], 1)

    def test_scope_block_rows_are_not_user_jobs(self) -> None:
        self.make_job("scope-block:abc123", "project:p-block", status="cancelling")
        self.make_job("project:p-block:render", "project:p-block", status="running")

        body = self.client.get("/api/jobs").json()
        self.assertEqual(body["total"], 1)
        self.assertEqual(body["items"][0]["job_type"], "render")

    def test_stats_endpoint_reports_entry_badge_numbers(self) -> None:
        self.make_job("project:p-badge:render", "project:p-badge", status="running")
        self.make_job("project:p-badge:storyboard", "project:p-badge", status="failed")
        self.make_job("project:p-other:render", "project:p-other", status="running")

        body = self.client.get("/api/jobs/stats").json()
        self.assertEqual(body["active_count"], 2)
        self.assertEqual(body["failed_count"], 1)
        self.assertEqual(body["total"], 3)
        self.assertIsNotNone(body["latest_job"])

        scoped = self.client.get("/api/jobs/stats", params={"project_id": "p-badge"}).json()
        self.assertEqual(scoped["active_count"], 1)
        self.assertEqual(scoped["total"], 2)

    def test_invalid_filters_are_rejected_without_touching_data(self) -> None:
        self.make_job("project:p-filter:render", "project:p-filter", status="running")
        self.assertEqual(self.client.get("/api/jobs", params={"status": "not-a-status"}).status_code, 422)
        self.assertEqual(self.client.get("/api/jobs", params={"job_type": "nope"}).status_code, 422)
        self.assertEqual(self.client.get("/api/jobs", params={"page": 0}).status_code, 422)
        self.assertEqual(self.client.get("/api/jobs", params={"page_size": 500}).status_code, 422)
        self.assertEqual(self.db.query(BackgroundJob).count(), 1)


class JobDetailApiTests(JobCenterTestCase):
    def test_detail_never_returns_run_token_or_internal_fields(self) -> None:
        job = self.make_job(
            "project:p-detail:render",
            "project:p-detail",
            status="failed",
            error="boom",
            progress=42,
        )
        response = self.client.get(f"/api/jobs/{job.id}")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotIn("run_token", body)
        self.assertNotIn("secret-run-token", response.text)
        self.assertNotIn("idempotency_key", body)
        self.assertEqual(body["status"], "failed")
        self.assertIn("duration_seconds", body)
        self.assertIn("eta_seconds", body)
        self.assertEqual(body["attempt"], 1)
        self.assertTrue(body["can_retry"])
        self.assertFalse(body["can_cancel"])
        self.assertTrue(body["can_delete"])
        self.assertEqual(body["error_code"], "job_failed")

    def test_missing_job_returns_404_without_leaking_internals(self) -> None:
        response = self.client.get("/api/jobs/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error_code"], "job_not_found")
        self.assertNotIn("Traceback", response.text)

    def test_detail_includes_attempt_history_after_retry(self) -> None:
        self.make_project("p-history")
        self.make_shot("s-history", "p-history", storyboard_status="failed", status="failed")
        old = self.make_job(
            "shot:s-history:storyboard",
            "shot:s-history",
            status="failed",
            error="provider exploded",
            project_id="p-history",
        )

        with patch.object(job_actions, "redispatch", _fake_redispatch):
            first = self.client.post(f"/api/jobs/{old.id}/retry")
        self.assertEqual(first.status_code, 200, first.text)
        started = first.json()["job"]
        self.assertEqual(started["attempt"], 2)
        self.assertEqual(started["retry_of"], old.id)

        detail = self.client.get(f"/api/jobs/{started['id']}").json()
        self.assertEqual(len(detail["attempts"]), 2)
        attempts = [item["attempt"] for item in detail["attempts"]]
        self.assertEqual(attempts, [1, 2])
        self.assertEqual(detail["attempts"][0]["status"], "failed")
        self.assertEqual(detail["attempts"][0]["error_message"], "provider exploded")
        self.assertEqual(detail["retry_relationship"]["retry_of_attempt"], 1)
        self.assertEqual(detail["latest_attempt_job_id"], started["id"])

        archived = self.client.get(f"/api/jobs/{old.id}")
        self.assertEqual(archived.status_code, 200, archived.text)
        self.assertEqual(archived.json()["status"], "failed")
        self.assertEqual(archived.json()["error_message"], "provider exploded")


async def _fake_redispatch(job, mode):
    """模拟一次成功的重新派发：复用真实的 claim（含归档 + 新 attempt 语义）。"""

    key = parse_job_key(job.idempotency_key).canonical
    claimed = task_registry.claim(
        key,
        job.scope,
        version=job.version,
        current_step="retry",
        message="已重新派发",
    )
    if not claimed:
        return DispatchResult(status="deduplicated", message="该任务已有正在执行的尝试，本次请求已合并")
    return DispatchResult(status="started", message="已重新派发", job_key=key)


class JobCancelApiTests(JobCenterTestCase):
    def test_cancel_running_job_without_local_task_settles_to_cancelled(self) -> None:
        self.make_project("p-cancel")
        job = self.make_job("project:p-cancel:render", "project:p-cancel", status="running")
        response = self.client.post(f"/api/jobs/{job.id}/cancel")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "cancelled")
        self.assertEqual(self.get_job(job.id).status, "cancelled")

    def test_cancel_queued_job_is_supported(self) -> None:
        job = self.make_job("project:p-queued:render", "project:p-queued", status="queued")
        response = self.client.post(f"/api/jobs/{job.id}/cancel")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "cancelled")

    def test_cancel_completed_and_failed_jobs_is_idempotent(self) -> None:
        completed = self.make_job("project:p-done:render", "project:p-done", status="completed")
        first = self.client.post(f"/api/jobs/{completed.id}/cancel")
        second = self.client.post(f"/api/jobs/{completed.id}/cancel")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertTrue(second.json()["idempotent"])
        self.assertEqual(second.json()["status"], "completed")

        failed = self.make_job("project:p-failed:render", "project:p-failed", status="failed")
        repeated = self.client.post(f"/api/jobs/{failed.id}/cancel")
        self.assertEqual(repeated.status_code, 200)
        self.assertEqual(repeated.json()["status"], "failed")

    def test_cancel_twice_on_running_job_is_idempotent(self) -> None:
        job = self.make_job("project:p-twice:render", "project:p-twice", status="running")
        self.assertEqual(self.client.post(f"/api/jobs/{job.id}/cancel").status_code, 200)
        again = self.client.post(f"/api/jobs/{job.id}/cancel")
        self.assertEqual(again.status_code, 200, again.text)
        self.assertTrue(again.json()["idempotent"])

    def test_cancel_waits_for_the_coroutine_to_unwind(self) -> None:
        key = "project:p-unwind:render"
        self.make_project("p-unwind")
        self.assertTrue(task_registry.claim(key, "project:p-unwind"))

        async def scenario():
            started = asyncio.Event()
            unwinding = asyncio.Event()

            async def worker():
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    unwinding.set()
                    await asyncio.sleep(0.05)
                    raise

            task = task_registry.start(key, worker())
            await started.wait()
            self.assertTrue(task_registry.cancel(key))
            during = task_registry.snapshot(key)
            self.assertEqual(during["status"], "cancelling")
            try:
                await task
            except asyncio.CancelledError:
                pass
            await asyncio.sleep(0)
            return task_registry.snapshot(key), unwinding.is_set()

        after, unwound = asyncio.run(scenario())
        self.assertTrue(unwound)
        self.assertEqual(after["status"], "cancelled")
        self.assertEqual(after["finished_at"] is not None, True)


class JobRetryApiTests(JobCenterTestCase):
    def test_retry_allowed_for_failed_cancelled_and_interrupted(self) -> None:
        for index, status in enumerate(("failed", "cancelled", "interrupted")):
            project_id = f"p-retry-{index}"
            self.make_project(project_id)
            job = self.make_job(
                f"project:{project_id}:render",
                f"project:{project_id}",
                status=status,
                project_id=project_id,
            )
            with patch.object(job_actions, "redispatch", _fake_redispatch):
                response = self.client.post(f"/api/jobs/{job.id}/retry")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["status"], "started")
            self.assertEqual(response.json()["job"]["attempt"], 2)

    def test_retry_rejected_while_another_attempt_is_active(self) -> None:
        job = self.make_job("project:p-active:render", "project:p-active", status="running")
        response = self.client.post(f"/api/jobs/{job.id}/retry")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error_code"], "job_not_retryable")

    def test_repeated_retry_creates_only_one_new_attempt(self) -> None:
        self.make_project("p-dedupe")
        job = self.make_job("project:p-dedupe:render", "project:p-dedupe", status="failed", project_id="p-dedupe")
        with patch.object(job_actions, "redispatch", _fake_redispatch):
            first = self.client.post(f"/api/jobs/{job.id}/retry")
            second = self.client.post(f"/api/jobs/{job.id}/retry")
            third = self.client.post(f"/api/jobs/{job.id}/retry")
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertTrue(second.json()["idempotent"])
        self.assertEqual(third.status_code, 200)
        self.db.expire_all()
        self.assertEqual(self.db.query(BackgroundJob).count(), 2)
        canonical = self.db.query(BackgroundJob).filter_by(idempotency_key="project:p-dedupe:render").one()
        self.assertEqual(canonical.status, "running")
        self.assertEqual(canonical.attempt, 2)

    def test_retry_keeps_the_original_failure_record(self) -> None:
        self.make_project("p-keep")
        job = self.make_job(
            "project:p-keep:render",
            "project:p-keep",
            status="failed",
            error="first attempt failed",
            project_id="p-keep",
        )
        original_updated_at = job.updated_at
        with patch.object(job_actions, "redispatch", _fake_redispatch):
            self.client.post(f"/api/jobs/{job.id}/retry")

        archived = self.get_job(job.id)
        self.assertEqual(archived.status, "failed")
        self.assertEqual(archived.error, "first attempt failed")
        self.assertEqual(archived.attempt, 1)
        self.assertEqual(archived.idempotency_key, "project:p-keep:render#attempt-1")
        self.assertEqual(archived.updated_at, original_updated_at, "归档不得刷新历史行的时间戳")
        current = self.db.query(BackgroundJob).filter_by(idempotency_key="project:p-keep:render").one()
        self.assertEqual(current.retry_of, job.id)
        self.assertEqual(current.attempt, 2)
        self.assertNotEqual(current.run_token, archived.run_token)

    def test_old_run_token_cannot_update_the_new_attempt(self) -> None:
        key = "project:p-token:render"
        self.assertTrue(task_registry.claim(key, "project:p-token"))
        stale_token = task_registry.snapshot(key)["run_token"]
        task_registry.finish(key, "failed", "first attempt", run_token=stale_token)

        with patch.object(job_actions, "redispatch", _fake_redispatch):
            job = self.db.query(BackgroundJob).filter_by(idempotency_key=key).one()
            self.client.post(f"/api/jobs/{job.id}/retry")

        fresh = task_registry.snapshot(key)
        self.assertNotEqual(fresh["run_token"], stale_token)
        self.assertFalse(task_registry.update_progress(key, 77, run_token=stale_token))
        self.assertFalse(task_registry.finish(key, "completed", run_token=stale_token))
        self.assertEqual(task_registry.snapshot(key)["status"], "running")
        self.assertTrue(task_registry.finish(key, "completed", run_token=fresh["run_token"]))

    def test_retry_is_blocked_while_a_mutex_scope_is_owned(self) -> None:
        self.make_project("p-scope")
        self.assertTrue(task_registry.claim("project:p-scope:render", "project:p-scope"))
        job = self.make_job("project:p-scope:storyboard", "project:p-scope", status="failed", project_id="p-scope")
        with patch.object(job_actions, "redispatch", _fake_redispatch):
            response = self.client.post(f"/api/jobs/{job.id}/retry")
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error_code"], "scope_conflict")
        task_registry.finish("project:p-scope:render", "completed")

    def test_unsupported_job_type_cannot_be_retried(self) -> None:
        job = self.make_job("shot:s-unknown:unknown-op", "shot:s-unknown", status="failed")
        response = self.client.post(f"/api/jobs/{job.id}/retry")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error_code"], "job_type_unsupported")


class JobDeliveryApiTests(JobCenterTestCase):
    def test_delete_only_allows_terminal_jobs(self) -> None:
        running = self.make_job("project:p-del:render", "project:p-del", status="running")
        blocked = self.client.delete(f"/api/jobs/{running.id}")
        self.assertEqual(blocked.status_code, 409)

        self.client.post(f"/api/jobs/{running.id}/cancel")
        removed = self.client.delete(f"/api/jobs/{running.id}")
        self.assertEqual(removed.status_code, 200, removed.text)
        self.assertIsNone(self.get_job(running.id))

    def test_cleanup_removes_only_terminal_jobs(self) -> None:
        self.make_job("project:p-clean:render", "project:p-clean", status="completed")
        self.make_job("project:p-clean:storyboard", "project:p-clean", status="cancelled")
        self.make_job("project:p-clean:pipeline:manual", "project:p-clean", status="running")

        response = self.client.delete("/api/jobs", params={"project_id": "p-clean"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted"], 2)
        self.db.expire_all()
        remaining = self.db.query(BackgroundJob).all()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].status, "running")


class JobErrorSanitizationTests(JobCenterTestCase):
    def test_durable_errors_are_truncated_and_redacted(self) -> None:
        key = "project:p-secret:render"
        self.assertTrue(task_registry.claim(key, "project:p-secret"))
        leaky = (
            "Traceback (most recent call last):\n"
            '  File "/Users/zhangshuai/workspace/ComicAgent/server/main.py", line 1\n'
            "RuntimeError: provider rejected api_key=sk-live-abcdef1234567890 "
            "for prompt 一只猫在月光下的完整分镜描述\n"
        ) * 40
        task_registry.finish(key, "failed", leaky)

        job = self.db.query(BackgroundJob).filter_by(idempotency_key=key).one()
        self.assertNotIn("sk-live", job.error)
        self.assertNotIn("/Users/", job.error)
        self.assertLessEqual(len(job.error), 2000)

        response = self.client.get(f"/api/jobs/{job.id}")
        body = response.json()
        self.assertNotIn("sk-live", response.text)
        self.assertNotIn("/Users/", response.text)
        self.assertNotIn("Traceback", response.text)
        self.assertLessEqual(len(body["error_message"]), 240)
        self.assertTrue(body["error_code"])

    def test_failure_responses_have_no_traceback_or_paths(self) -> None:
        job = self.make_job("project:p-shape:render", "project:p-shape", status="running")
        response = self.client.post(f"/api/jobs/{job.id}/retry")
        self.assertEqual(response.status_code, 409)
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn("/Users/", response.text)
        self.assertIn("error_code", response.json())


class JobStatusTransitionTests(JobCenterTestCase):
    def test_terminal_states_are_absorbing(self) -> None:
        for terminal in ("completed", "failed", "cancelled", "interrupted"):
            self.assertFalse(can_transition(terminal, "running"))
            self.assertFalse(can_transition(terminal, "queued"))
            self.assertFalse(can_transition(terminal, "completed"))
        self.assertTrue(can_transition("queued", "running"))
        self.assertTrue(can_transition("running", "cancelling"))
        self.assertTrue(can_transition("cancelling", "cancelled"))
        self.assertFalse(can_transition("queued", "completed"))

    def test_finish_refuses_to_resurrect_a_terminal_job(self) -> None:
        key = "project:p-absorb:render"
        self.assertTrue(task_registry.claim(key, "project:p-absorb"))
        token = task_registry.snapshot(key)["run_token"]
        self.assertTrue(task_registry.finish(key, "completed", run_token=token))
        self.assertFalse(task_registry.finish(key, "failed", "late failure", run_token=token))
        self.assertFalse(task_registry.update_progress(key, 50, run_token=token))
        self.assertEqual(task_registry.snapshot(key)["status"], "completed")

    def test_reclaim_creates_a_new_attempt_instead_of_resurrecting(self) -> None:
        key = "project:p-reclaim:render"
        self.assertTrue(task_registry.claim(key, "project:p-reclaim"))
        first_token = task_registry.snapshot(key)["run_token"]
        task_registry.finish(key, "failed", "boom", run_token=first_token)

        self.assertTrue(task_registry.claim(key, "project:p-reclaim"))
        rows = self.db.query(BackgroundJob).all()
        self.assertEqual(len(rows), 2)
        current = task_registry.snapshot(key)
        self.assertEqual(current["status"], "running")
        self.assertNotEqual(current["run_token"], first_token)
        archived = [row for row in rows if row.idempotency_key.endswith("#attempt-1")]
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].status, "failed")
        self.assertEqual(archived[0].attempt, 1)
        task_registry.finish(key, "completed")

    def test_restart_marks_jobs_interrupted_and_retryable(self) -> None:
        self.make_project("p-restart")
        self.assertTrue(task_registry.claim("project:p-restart:render", "project:p-restart"))
        self.assertEqual(task_registry.recover_interrupted(), 1)

        job = self.db.query(BackgroundJob).filter_by(idempotency_key="project:p-restart:render").one()
        self.assertEqual(job.status, "interrupted")
        self.assertEqual(job.error_code, "server_restart")

        detail = self.client.get(f"/api/jobs/{job.id}").json()
        self.assertEqual(detail["status"], "interrupted")
        self.assertEqual(detail["status_label"], "已中断")
        self.assertTrue(detail["can_retry"])
        self.assertTrue(detail["can_resume"])
        self.assertTrue(detail["can_delete"])

    def test_progress_updates_carry_step_and_message(self) -> None:
        key = "project:p-step:storyboard"
        self.assertTrue(task_registry.claim(key, "project:p-step"))
        token = task_registry.snapshot(key)["run_token"]
        self.assertTrue(
            task_registry.update_progress(
                key, 48, run_token=token, current_step="generate_storyboard", message="正在生成分镜"
            )
        )
        job = self.db.query(BackgroundJob).filter_by(idempotency_key=key).one()
        self.assertEqual(job.current_step, "generate_storyboard")
        self.assertEqual(job.message, "正在生成分镜")
        self.assertEqual(job.progress, 48)
        task_registry.finish(key, "completed")


class JobDispatchTests(JobCenterTestCase):
    def test_resume_skips_completed_shots_in_a_batch(self) -> None:
        self.make_project("p-batch")
        done = _write_media("projects/p-batch/storyboard/shot-done.png")
        self.make_shot(
            "shot-done", "p-batch", sequence=1, storyboard_status="done", storyboard_path=done, image_path=done
        )
        self.make_shot("shot-missing", "p-batch", sequence=2, storyboard_status="failed", status="failed")
        job = self.make_job("project:p-batch:storyboard", "project:p-batch", status="failed", project_id="p-batch")

        calls: list[list[str]] = []

        async def fake_generate_storyboard(project_id, data, db):
            calls.append(list(data.shot_ids))
            task_registry.claim(f"project:{project_id}:storyboard", f"project:{project_id}")
            return {"status": "storyboard_started", "project_id": project_id, "shots": len(data.shot_ids)}

        with patch.object(job_actions, "redispatch", _fake_redispatch):
            pass  # 真实派发路径在下面直接调用，确保覆盖 job_dispatch 自身逻辑

        from api.routes import shot as shot_route

        with patch.object(shot_route, "generate_storyboard_images", fake_generate_storyboard):
            result = asyncio.run(job_dispatch.redispatch(job, RESUME_MODE))
        self.assertEqual(result.status, "started", result.message)
        self.assertEqual(calls, [["shot-missing"]])
        task_registry.finish("project:p-batch:storyboard", "completed")

    def test_retry_of_a_storyboard_batch_repeats_unconfirmed_shots(self) -> None:
        self.make_project("p-batch2")
        done = _write_media("projects/p-batch2/storyboard/shot-a.png")
        self.make_shot(
            "shot-a", "p-batch2", sequence=1, storyboard_status="done", storyboard_path=done, image_path=done
        )
        self.make_shot(
            "shot-b", "p-batch2", sequence=2, storyboard_status="done", storyboard_path=done, image_path=done
        )
        job = self.make_job("project:p-batch2:storyboard", "project:p-batch2", status="failed", project_id="p-batch2")

        calls: list[list[str]] = []

        async def fake_generate_storyboard(project_id, data, db):
            calls.append(sorted(data.shot_ids))
            task_registry.claim(f"project:{project_id}:storyboard", f"project:{project_id}")
            return {"status": "storyboard_started", "project_id": project_id, "shots": len(data.shot_ids)}

        from api.routes import shot as shot_route

        with patch.object(shot_route, "generate_storyboard_images", fake_generate_storyboard):
            result = asyncio.run(job_dispatch.redispatch(job, RETRY_MODE))
        self.assertEqual(result.status, "started", result.message)
        self.assertEqual(calls, [["shot-a", "shot-b"]])
        task_registry.finish("project:p-batch2:storyboard", "completed")

    def test_resume_refuses_when_the_storyboard_already_exists(self) -> None:
        self.make_project("p-skip")
        media = _write_media("projects/p-skip/storyboard/shot-ok.png")
        self.make_shot("shot-ok", "p-skip", storyboard_status="done", storyboard_path=media, image_path=media)
        job = self.make_job("shot:shot-ok:storyboard", "shot:shot-ok", status="failed", project_id="p-skip")

        result = asyncio.run(job_dispatch.redispatch(job, RESUME_MODE))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.error_code, "job_not_resumable")
        self.assertIn("已存在", result.message)

    def test_resume_reuses_existing_audio_and_only_regenerates_video(self) -> None:
        self.make_project("p-video")
        storyboard = _write_media("projects/p-video/storyboard/shot-v.png")
        self.make_shot(
            "shot-v",
            "p-video",
            sequence=3,
            confirmed=True,
            status="failed",
            storyboard_status="done",
            storyboard_path=storyboard,
            image_path=storyboard,
        )
        job = self.make_job("shot:shot-v:video", "shot:shot-v", status="failed", project_id="p-video")

        captured: dict = {}

        async def fake_generate_video(shot_id, data, db):
            captured["shot_id"] = shot_id
            captured["force"] = data.force
            captured["reuse_audio"] = data.reuse_audio
            task_registry.claim(f"shot:{shot_id}:video", f"shot:{shot_id}", version=1)
            return {"id": shot_id, "status": "video_generating"}

        from api.routes import shot as shot_route

        with patch.object(shot_route, "generate_shot_video", fake_generate_video):
            result = asyncio.run(job_dispatch.redispatch(job, RESUME_MODE))
        self.assertEqual(result.status, "started", result.message)
        self.assertEqual(captured["shot_id"], "shot-v")
        self.assertFalse(captured["force"])
        self.assertTrue(captured["reuse_audio"])
        task_registry.finish("shot:shot-v:video", "completed")

    def test_resume_refuses_when_the_video_already_exists(self) -> None:
        self.make_project("p-video2")
        storyboard = _write_media("projects/p-video2/storyboard/shot-w.png")
        video = _write_media("projects/p-video2/video/shot-w.mp4", size=8192)
        self.make_shot(
            "shot-w",
            "p-video2",
            confirmed=True,
            status="video_done",
            storyboard_status="done",
            storyboard_path=storyboard,
            image_path=storyboard,
            video_path=video,
        )
        job = self.make_job("shot:shot-w:video", "shot:shot-w", status="cancelled", project_id="p-video2")
        result = asyncio.run(job_dispatch.redispatch(job, RESUME_MODE))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.error_code, "job_not_resumable")

    def test_render_retry_reports_missing_assets_instead_of_failing_late(self) -> None:
        self.make_project("p-render")
        storyboard = _write_media("projects/p-render/storyboard/shot-r.png")
        self.make_shot("shot-r", "p-render", confirmed=False, storyboard_path=storyboard, image_path=storyboard)
        job = self.make_job("project:p-render:render", "project:p-render", status="failed", project_id="p-render")

        result = asyncio.run(job_dispatch.redispatch(job, RETRY_MODE))
        self.assertEqual(result.status, "rejected")
        self.assertIn("审核", result.message)

    def test_resume_refuses_when_there_is_no_checkpoint(self) -> None:
        self.make_project("p-empty", input_text="", status="draft")
        job = self.make_job("project:p-empty:pipeline:manual", "project:p-empty", status="failed", project_id="p-empty")
        result = asyncio.run(job_dispatch.redispatch(job, RESUME_MODE))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.error_code, "job_not_resumable")
        self.assertIn("续跑", result.message)

    def test_retry_without_saved_script_returns_explicit_error(self) -> None:
        self.make_project("p-notext", input_text="")
        job = self.make_job(
            "project:p-notext:pipeline:manual", "project:p-notext", status="failed", project_id="p-notext"
        )
        result = asyncio.run(job_dispatch.redispatch(job, RETRY_MODE))
        self.assertEqual(result.status, "rejected")
        self.assertIn("原始剧本", result.message)


class JobEtaTests(JobCenterTestCase):
    def test_eta_is_only_reported_with_real_history(self) -> None:
        base = datetime.utcnow() - timedelta(hours=2)
        for index in range(3):
            job = self.make_job(
                f"project:p-eta-{index}:render",
                f"project:p-eta-{index}",
                status="completed",
                updated_at=base + timedelta(minutes=index),
            )
            job.job_type = "render"
            job.started_at = job.updated_at
            job.finished_at = job.updated_at + timedelta(seconds=100)
            self.db.commit()

        active = self.make_job("project:p-eta-live:render", "project:p-eta-live", status="running", progress=25)
        detail = self.client.get(f"/api/jobs/{active.id}").json()
        self.assertIsNotNone(detail["eta_seconds"])
        self.assertGreater(detail["eta_seconds"], 0)

        fresh = self.make_job("shot:s-eta:storyboard", "shot:s-eta", status="running", progress=25)
        fresh_detail = self.client.get(f"/api/jobs/{fresh.id}").json()
        self.assertIsNone(fresh_detail["eta_seconds"])

    def test_completed_jobs_never_report_eta(self) -> None:
        job = self.make_job("project:p-eta-done:render", "project:p-eta-done", status="completed")
        self.assertIsNone(self.client.get(f"/api/jobs/{job.id}").json()["eta_seconds"])


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def accept(self) -> None:
        return None

    async def send_text(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


class JobEventTests(JobCenterTestCase):
    def test_events_are_published_without_run_token(self) -> None:
        from api.websocket import jobs_manager

        async def scenario():
            socket = _FakeSocket()
            await jobs_manager.connect(socket)
            try:
                task_registry.claim(
                    "project:p-events:render", "project:p-events", current_step="rendering", message="开始导出"
                )
                await asyncio.sleep(0.05)
                token = task_registry.snapshot("project:p-events:render")["run_token"]
                task_registry.update_progress(
                    "project:p-events:render", 40, run_token=token, current_step="rendering", message="正在合成"
                )
                await asyncio.sleep(0.05)
                task_registry.finish("project:p-events:render", "failed", "boom", run_token=token)
                await asyncio.sleep(0.05)
            finally:
                jobs_manager.disconnect(socket)
            return socket.sent

        sent = asyncio.run(scenario())
        types = [item["type"] for item in sent]
        self.assertIn("job.created", types)
        self.assertIn("job.progress", types)
        self.assertIn("job.failed", types)
        rendered = json.dumps(sent, ensure_ascii=False)
        self.assertNotIn("run_token", rendered)
        self.assertNotIn("secret", rendered)
        progress = next(item for item in sent if item["type"] == "job.progress")
        self.assertEqual(progress["job"]["current_step"], "rendering")
        self.assertEqual(progress["job"]["message"], "正在合成")
        self.assertEqual(progress["project_id"], "p-events")

    def test_retry_publishes_retry_started(self) -> None:
        from api.websocket import jobs_manager

        async def scenario():
            socket = _FakeSocket()
            await jobs_manager.connect(socket)
            try:
                key = "project:p-retry-event:render"
                task_registry.claim(key, "project:p-retry-event")
                token = task_registry.snapshot(key)["run_token"]
                task_registry.finish(key, "failed", "boom", run_token=token)
                await asyncio.sleep(0.05)
                socket.sent.clear()
                task_registry.claim(key, "project:p-retry-event")
                await asyncio.sleep(0.05)
            finally:
                jobs_manager.disconnect(socket)
            return socket.sent

        sent = asyncio.run(scenario())
        self.assertEqual([item["type"] for item in sent], ["job.debug", "job.retry_started"])
        retry_event = next(item for item in sent if item["type"] == "job.retry_started")
        self.assertEqual(retry_event["job"]["attempt"], 2)

    def test_snapshot_payload_has_no_run_token(self) -> None:
        self.make_job("project:p-snap:render", "project:p-snap", status="running", progress=30)
        snapshot = jobs_snapshot()
        self.assertEqual(snapshot["type"], "job_snapshot")
        self.assertEqual(len(snapshot["jobs"]), 1)
        self.assertNotIn("run_token", json.dumps(snapshot, ensure_ascii=False))
        self.assertIn("active_count", snapshot)
        self.assertIn("status_counts", snapshot)


class JobDebugLogTests(JobCenterTestCase):
    def test_debug_log_tracks_progress_and_redacts_secrets(self) -> None:
        from services.job_debug import make_event

        key = "project:p-debug:render"
        self.assertTrue(task_registry.claim(key, "project:p-debug", current_step="rendering", message="开始"))
        token = task_registry.snapshot(key)["run_token"]
        self.assertTrue(
            task_registry.update_progress(
                key,
                35,
                run_token=token,
                current_step="rendering",
                message="正在调用图像 API",
            )
        )
        db = SessionLocal()
        try:
            job_id = str(db.query(BackgroundJob.id).filter(BackgroundJob.idempotency_key == key).scalar())
        finally:
            db.close()
        body = self.client.get(f"/api/jobs/{job_id}/debug").json()
        self.assertEqual(body["current_step"], "rendering")
        self.assertEqual(body["progress"], 35)
        self.assertGreaterEqual(body["debug_revision"], 2)
        self.assertTrue(any(item["kind"] == "progress" for item in body["events"]))

        event = make_event(
            "api_request",
            "发起请求",
            api="Image Generate",
            provider="qwen",
            model="test-model",
            params={"api_key": "sk-secret-value", "reference_images": ["data:image/png;base64,AAAA"]},
            prompt={"system": "system prompt", "user": "user prompt"},
        )
        self.assertEqual(event["params"]["api_key"], "[已脱敏]")
        self.assertIn("不记录内容", event["params"]["reference_images"])
        self.assertEqual(event["prompt"]["user"], "user prompt")


class JobWebSocketTests(JobCenterTestCase):
    def test_websocket_sends_initial_snapshot(self) -> None:
        self.make_job("project:p-ws:render", "project:p-ws", status="running", progress=12)
        with self.client.websocket_connect("/ws/jobs") as websocket:
            message = websocket.receive_json()
        self.assertEqual(message["type"], "job_snapshot")
        self.assertEqual(len(message["jobs"]), 1)
        self.assertEqual(message["jobs"][0]["progress"], 12)
        self.assertEqual(message["active_count"], 1)
        self.assertNotIn("run_token", json.dumps(message, ensure_ascii=False))
        self.assertEqual(websocket_payload_keys(message["jobs"][0]), None)

    def test_websocket_reconnect_gets_a_fresh_snapshot(self) -> None:
        with self.client.websocket_connect("/ws/jobs") as websocket:
            first = websocket.receive_json()
            self.assertEqual(len(first["jobs"]), 0)

        self.make_job("project:p-ws2:render", "project:p-ws2", status="running")

        with self.client.websocket_connect("/ws/jobs") as websocket:
            second = websocket.receive_json()
            self.assertEqual(len(second["jobs"]), 1)
            websocket.send_text("sync")
            refreshed = websocket.receive_json()
        self.assertEqual(refreshed["type"], "job_snapshot")
        self.assertEqual(len(refreshed["jobs"]), 1)

    def test_websocket_ping_is_answered(self) -> None:
        with self.client.websocket_connect("/ws/jobs") as websocket:
            websocket.receive_json()
            websocket.send_text("ping")
            self.assertEqual(websocket.receive_json(), {"type": "pong"})

    def test_job_socket_does_not_shadow_project_socket(self) -> None:
        paths = {getattr(route, "path", "") for route in app.routes}
        self.assertIn("/ws/jobs", paths)
        self.assertIn("/ws/{project_id}", paths)
        index = [getattr(route, "path", "") for route in app.routes].index("/ws/jobs")
        self.assertLess(index, [getattr(route, "path", "") for route in app.routes].index("/ws/{project_id}"))

    def test_project_socket_still_works(self) -> None:
        with self.client.websocket_connect("/ws/project-a") as websocket:
            websocket.send_text("ping")
            self.assertEqual(websocket.receive_json(), {"type": "pong"})


def websocket_payload_keys(job: dict) -> None:
    """断言任务 DTO 只包含稳定的、可公开的字段。"""

    allowed = {
        "id",
        "scope",
        "project_id",
        "job_type",
        "job_type_label",
        "display_name",
        "status",
        "status_label",
        "progress",
        "current_step",
        "message",
        "error_code",
        "error_code_label",
        "error_message",
        "error_detail",
        "attempt",
        "retry_of",
        "version",
        "created_at",
        "started_at",
        "updated_at",
        "finished_at",
        "cancel_requested_at",
        "duration_seconds",
        "eta_seconds",
        "is_active",
        "is_terminal",
        "has_active_successor",
        "can_cancel",
        "can_retry",
        "can_resume",
        "can_delete",
        "retry_blocked_reason",
        "resume_blocked_reason",
        # 成本快照（实际金额 + 启动前估算）：只含归一化后的金额与数量，
        # 不含密钥、供应商原始响应或本地路径，因此属于可公开字段。
        "cost",
        # 调试日志版本号：前端据此识别是否需要补齐日志，不含日志正文。
        "debug_revision",
    }
    unexpected = set(job) - allowed
    assert not unexpected, f"unexpected job fields: {sorted(unexpected)}"
    return None


class JobCenterPerformanceTests(JobCenterTestCase):
    def test_listing_many_jobs_stays_fast_and_does_not_block(self) -> None:
        for index in range(300):
            self.make_job(
                f"project:p-perf-{index}:render",
                f"project:p-perf-{index}",
                status="completed" if index % 2 else "failed",
                updated_at=datetime.utcnow() - timedelta(seconds=index),
            )
        started = time.monotonic()
        body = self.client.get("/api/jobs", params={"page": 1, "page_size": 50}).json()
        elapsed = time.monotonic() - started
        self.assertEqual(body["total"], 300)
        self.assertEqual(len(body["items"]), 50)
        self.assertLess(elapsed, 3.0)


if __name__ == "__main__":
    unittest.main()
