"""失败原因自动归因的回归测试。

覆盖范围：

- 规则分类：额度/限流/参数/依赖/通用供应商细分码，以及优先级（预算 > 供应商）；
- LLM 增强：泛化码升级、确定性码不被覆盖、跳过清单不调 LLM、幂等、指纹去重、
  LLM 输出不合法或调用失败时静默保留规则结果；
- 触发链路：task_registry.finish 失败终态调度分析；队列占位行落库错误码；
- DTO 暴露：error_code_label / error_detail 白名单解析，attempt 历史带标签。
"""

from __future__ import annotations

import asyncio
import json
import unittest
import uuid
from datetime import datetime
from unittest.mock import patch

from db import SessionLocal, init_db  # noqa: E402
from models import BackgroundJob  # noqa: E402
from services import error_analysis_service, job_center, task_registry  # noqa: E402
from services.error_analysis_service import analyze_job_failure  # noqa: E402
from services.job_dto import job_dto  # noqa: E402
from services.job_types import (  # noqa: E402
    ERROR_CODE_BUDGET_EXCEEDED,
    ERROR_CODE_DEPENDENCY_FAILED,
    ERROR_CODE_INVALID_REQUEST,
    ERROR_CODE_PROVIDER,
    ERROR_CODE_QUOTA_EXCEEDED,
    ERROR_CODE_RATE_LIMITED,
    classify_error_code,
    error_code_label,
)
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


class _FakeLLM:
    """替身 LLM：记录调用次数，返回预设结果或抛错。"""

    def __init__(self, result=None, error: Exception | None = None, available: bool = True):
        self.available = available
        self.last_provider_used = "fake:llm"
        self.calls = 0
        self._result = result
        self._error = error

    async def call_json(self, system_prompt: str, user_prompt: str, **kwargs):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._result


class ClassifyRulesTests(unittest.TestCase):
    def test_provider_side细分码(self) -> None:
        cases = [
            ("Seedance 创建任务失败: 429 TooManyRequests", ERROR_CODE_RATE_LIMITED),
            ("火山方舟: AccountHasArrears，余额不足", ERROR_CODE_QUOTA_EXCEEDED),
            ("AllocationQuota exceeded for this model", ERROR_CODE_QUOTA_EXCEEDED),
            ("百炼创建视频任务失败: 400 InvalidParameter", ERROR_CODE_INVALID_REQUEST),
            ("参数错误：size 不合法", ERROR_CODE_INVALID_REQUEST),
            ("前置阶段失败，当前镜头未执行", ERROR_CODE_DEPENDENCY_FAILED),
            ("Seedream 图像生成失败: 500 Internal Server Error", ERROR_CODE_PROVIDER),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(classify_error_code(text), expected)

    def test_业务语义优先于供应商细分(self) -> None:
        self.assertEqual(classify_error_code("超出项目硬预算 budget_exceeded"), ERROR_CODE_BUDGET_EXCEEDED)
        self.assertEqual(classify_error_code("budget_exceeded with provider 429"), ERROR_CODE_BUDGET_EXCEEDED)

    def test_标签覆盖全部错误码(self) -> None:
        self.assertEqual(error_code_label(ERROR_CODE_QUOTA_EXCEEDED), "额度不足")
        self.assertEqual(error_code_label(ERROR_CODE_RATE_LIMITED), "触发限流")
        self.assertEqual(error_code_label(ERROR_CODE_INVALID_REQUEST), "API 参数错误")
        self.assertEqual(error_code_label(ERROR_CODE_DEPENDENCY_FAILED), "前置阶段失败")
        self.assertEqual(error_code_label("provider_error"), "API 调用失败")
        self.assertEqual(error_code_label(""), "")
        self.assertEqual(error_code_label("never_defined_code"), "任务失败")


class ErrorAnalysisServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    def setUp(self) -> None:
        self.db = SessionLocal()
        self.db.query(BackgroundJob).delete()
        self.db.commit()
        error_analysis_service._analysis_cache.clear()
        error_analysis_service._in_flight.clear()
        error_analysis_service._pending_fingerprints.clear()
        # 测试环境默认禁用 LLM 分析（见 test_environment），本组用例显式开启。
        self.llm_toggle = patch.object(error_analysis_service.settings, "ERROR_ANALYSIS_LLM_ENABLED", True)
        self.llm_toggle.start()

    def tearDown(self) -> None:
        self.llm_toggle.stop()
        self.db.rollback()
        self.db.close()

    # --- 工具 ---

    def make_failed_job(
        self,
        *,
        error_code: str = "",
        error: str = "Seedance 创建任务失败: 500 Internal Server Error",
    ) -> BackgroundJob:
        suffix = uuid.uuid4().hex[:8]
        job = BackgroundJob(
            id=f"job-{suffix}",
            idempotency_key=f"project:p-{suffix}:render",
            scope=f"project:p-{suffix}",
            status="failed",
            progress=30,
            error=error,
            error_code=error_code,
            error_message=error[:240],
            run_token="tok",
            job_type="render",
            project_id=f"p-{suffix}",
            display_name="镜头渲染测试",
            attempt=1,
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
            finished_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self.db.add(job)
        self.db.commit()
        return job

    def reload(self, job_id: str) -> BackgroundJob:
        self.db.expire_all()
        return self.db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()

    def run_analysis(self, job_id: str) -> None:
        asyncio.run(analyze_job_failure(job_id))

    # --- LLM 增强 ---

    def test_泛化码允许LLM升级并写入摘要(self) -> None:
        job = self.make_failed_job(error_code="provider_error")
        fake = _FakeLLM(
            result={
                "category": "provider_quota_exceeded",
                "confidence": 0.9,
                "summary": "火山方舟账户欠费导致调用被拒",
                "suggestion": "前往控制台充值后重试该任务",
            }
        )
        with (
            patch.object(error_analysis_service, "llm_service", fake),
            patch("services.error_analysis_service.publish_job_event") as publish,
        ):
            self.run_analysis(job.id)
            publish.assert_called_once()
            event_type, payload = publish.call_args[0]
            self.assertEqual(event_type, "job.updated")
        refreshed = self.reload(job.id)
        self.assertEqual(refreshed.error_code, "provider_quota_exceeded")
        detail = json.loads(refreshed.error_detail)
        self.assertEqual(detail["source"], "llm")
        self.assertEqual(detail["summary"], "火山方舟账户欠费导致调用被拒")
        self.assertEqual(detail["suggestion"], "前往控制台充值后重试该任务")
        self.assertEqual(detail["model"], "fake:llm")
        # 事件负载携带新字段
        self.assertEqual(payload["error_code_label"], "额度不足")
        self.assertEqual(payload["error_detail"]["summary"], "火山方舟账户欠费导致调用被拒")

    def test_确定性规则码不被LLM覆盖(self) -> None:
        job = self.make_failed_job(
            error_code="provider_rate_limited",
            error="Seedance 创建任务失败: 429 TooManyRequests",
        )
        fake = _FakeLLM(
            result={
                "category": "provider_quota_exceeded",
                "confidence": 0.8,
                "summary": "疑似欠费",
                "suggestion": "充值",
            }
        )
        with patch.object(error_analysis_service, "llm_service", fake):
            self.run_analysis(job.id)
        refreshed = self.reload(job.id)
        self.assertEqual(refreshed.error_code, "provider_rate_limited", "规则命中的确定性分类优先")
        self.assertTrue(refreshed.error_detail, "LLM 的摘要与建议仍应保留")

    def test_跳过清单不调LLM但补齐规则码(self) -> None:
        job = self.make_failed_job(error_code="", error="超出项目硬预算 budget_exceeded")
        fake = _FakeLLM()
        with patch.object(error_analysis_service, "llm_service", fake):
            self.run_analysis(job.id)
        self.assertEqual(fake.calls, 0, "预算类失败信息已明确，不应调用 LLM")
        refreshed = self.reload(job.id)
        self.assertEqual(refreshed.error_code, "budget_exceeded", "空错误码应补齐规则分类")

    def test_幂等_已分析任务不重复处理(self) -> None:
        job = self.make_failed_job()
        job.error_detail = json.dumps({"summary": "s", "suggestion": "g", "source": "llm", "model": "m"})
        self.db.commit()
        fake = _FakeLLM(result={"category": "provider_error", "summary": "x", "suggestion": "y"})
        with patch.object(error_analysis_service, "llm_service", fake):
            self.run_analysis(job.id)
        self.assertEqual(fake.calls, 0, "error_detail 非空即已分析")

    def test_LLM输出类别不合法时只保留规则结果(self) -> None:
        job = self.make_failed_job(error_code="provider_error")
        fake = _FakeLLM(result={"category": "definitely_not_a_code", "summary": "x", "suggestion": "y"})
        with patch.object(error_analysis_service, "llm_service", fake):
            self.run_analysis(job.id)
        refreshed = self.reload(job.id)
        self.assertEqual(refreshed.error_code, "provider_error")
        self.assertEqual(refreshed.error_detail, "", "非法输出不得写入分析结果")

    def test_LLM调用失败时静默保留规则结果(self) -> None:
        job = self.make_failed_job(error_code="job_failed")
        fake = _FakeLLM(error=RuntimeError("LLM endpoint down"))
        with patch.object(error_analysis_service, "llm_service", fake):
            self.run_analysis(job.id)  # 不应抛出
        refreshed = self.reload(job.id)
        self.assertEqual(refreshed.status, "failed")
        self.assertEqual(refreshed.error_code, "job_failed")

    def test_LLM未配置时直接跳过(self) -> None:
        job = self.make_failed_job()
        fake = _FakeLLM(available=False)
        with patch.object(error_analysis_service, "llm_service", fake):
            self.run_analysis(job.id)
        self.assertEqual(fake.calls, 0)

    def test_相同错误指纹只调用一次LLM(self) -> None:
        first = self.make_failed_job()
        second = self.make_failed_job(error=first.error)
        fake = _FakeLLM(
            result={"category": "provider_quota_exceeded", "summary": "账户欠费", "suggestion": "充值后重试"}
        )
        with patch.object(error_analysis_service, "llm_service", fake):
            self.run_analysis(first.id)
            self.run_analysis(second.id)
        self.assertEqual(fake.calls, 1, "相同错误文本走指纹缓存")
        self.assertTrue(json.loads(self.reload(second.id).error_detail)["summary"])


class TriggerWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    def setUp(self) -> None:
        self.db = SessionLocal()
        self.db.query(BackgroundJob).delete()
        self.db.commit()

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()

    def test_finish失败终态调度分析(self) -> None:
        key = "project:p-wire-1:render"
        self.assertTrue(task_registry.claim(key, "project:p-wire-1"))
        token = task_registry.snapshot(key)["run_token"]
        with patch("services.error_analysis_service.schedule_failure_analysis") as schedule:
            self.assertTrue(
                task_registry.finish(key, "failed", "Seedance 创建任务失败: 429 TooManyRequests", run_token=token)
            )
            schedule.assert_called_once()
        job = self.db.query(BackgroundJob).filter(BackgroundJob.idempotency_key == key).first()
        self.assertEqual(job.error_code, "provider_rate_limited", "规则细分码在落库时即生效")

    def test_finish完成终态不调度分析(self) -> None:
        key = "project:p-wire-2:render"
        self.assertTrue(task_registry.claim(key, "project:p-wire-2"))
        token = task_registry.snapshot(key)["run_token"]
        with patch("services.error_analysis_service.schedule_failure_analysis") as schedule:
            self.assertTrue(task_registry.finish(key, "completed", run_token=token))
            schedule.assert_not_called()

    def test_队列占位行落库错误码并调度分析(self) -> None:
        from services import regeneration_queue

        job = BackgroundJob(
            id="job-queue-wire",
            idempotency_key="shot:s-wire:video",
            scope="shot:s-wire",
            status="running",
            error_code="",
            error_message="",
            run_token="",
            job_type="shot_video",
            display_name="镜头 1 · video",
            queue_batch_id="b1",
            queue_stage="video",
        )
        self.db.add(job)
        self.db.commit()
        with patch("services.error_analysis_service.schedule_failure_analysis") as schedule:
            regeneration_queue._mark_queue_job("job-queue-wire", "failed", "百炼创建视频任务失败: 400 InvalidParameter")
            schedule.assert_called_once_with("job-queue-wire")
        self.db.expire_all()
        refreshed = self.db.query(BackgroundJob).filter(BackgroundJob.id == "job-queue-wire").first()
        self.assertEqual(refreshed.error_code, "provider_invalid_request")


class DtoExposureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    def setUp(self) -> None:
        self.db = SessionLocal()
        self.db.query(BackgroundJob).delete()
        self.db.commit()

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()

    def _job(self, **overrides) -> BackgroundJob:
        now = datetime.utcnow()
        suffix = uuid.uuid4().hex[:8]
        fields = dict(
            id=f"job-dto-{suffix}",
            idempotency_key=f"project:p-{suffix}:render",
            scope=f"project:p-{suffix}",
            status="failed",
            progress=10,
            error="raw",
            error_code="provider_quota_exceeded",
            error_message="AccountHasArrears",
            run_token="secret-token",
            job_type="render",
            project_id="p-dto",
            display_name="任务",
            attempt=1,
            created_at=now,
            updated_at=now,
        )
        fields.update(overrides)
        job = BackgroundJob(**fields)
        self.db.add(job)
        self.db.commit()
        return job

    def test_dto暴露标签与分析结果(self) -> None:
        detail = json.dumps(
            {
                "summary": "账户欠费",
                "suggestion": "充值后重试",
                "source": "llm",
                "model": "m",
                "analyzed_at": "2026-01-01T00:00:00",
            }
        )
        job = self._job(error_detail=detail)
        dto = job_dto(job)
        self.assertEqual(dto["error_code_label"], "额度不足")
        self.assertEqual(dto["error_detail"]["summary"], "账户欠费")
        self.assertEqual(dto["error_detail"]["source"], "llm")
        self.assertNotIn("run_token", dto)

    def test_dto非法分析结果返回None(self) -> None:
        job = self._job(error_detail="not-json")
        self.assertIsNone(job_dto(job)["error_detail"])
        job = self._job(error_detail=json.dumps({"summary": "", "suggestion": ""}))
        self.assertIsNone(job_dto(job)["error_detail"])

    def test_attempt历史带错误码标签(self) -> None:
        job = self._job()
        history = job_center.attempt_history(self.db, job)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["error_code_label"], "额度不足")


if __name__ == "__main__":
    unittest.main()
