"""端到端：任务执行过程中的用量归属（contextvar 作用域 + 服务层记账）。

这些用例走的是真实路径：task_registry.claim_job -> start -> 服务层（图像 / LLM）
-> usage_records。它们保证「任务执行过程中记录实际 token / 图片 / 回放用量」不是
只存在于单元测试里的约定，而是真的挂在任务上下文上。
"""

from __future__ import annotations

import asyncio
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from db import SessionLocal, init_db  # noqa: E402
from models import BudgetReservation, CostEstimate, Project, Shot, UsageRecord  # noqa: E402
from services import pricing_service, task_registry, usage_service  # noqa: E402
from services.image_service import ImageService  # noqa: E402
from services.llm_service import LLMService  # noqa: E402
from services.providers.endpoint import EndpointConfig  # noqa: E402
from services.providers.image_stability import StabilityImageAdapter  # noqa: E402
from services.providers.llm_openai_chat import OpenAIChatAdapter  # noqa: E402
from services.providers.usage import UsageMetadata  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402

init_db()


class _Message:
    content = "{}"


class _Choice:
    message = _Message()


class _Usage:
    prompt_tokens = 1500
    completion_tokens = 400


class _Response:
    usage = _Usage()
    choices = [_Choice()]


def _project() -> str:
    db = SessionLocal()
    try:
        project_id = f"proj-{uuid.uuid4().hex[:10]}"
        db.add(Project(id=project_id, title="集成测试项目"))
        db.commit()
        return project_id
    finally:
        db.close()


def _shot(project_id: str) -> str:
    db = SessionLocal()
    try:
        shot_id = f"shot-{uuid.uuid4().hex[:10]}"
        db.add(
            Shot(
                id=shot_id,
                project_id=project_id,
                sequence=1,
                duration=4.0,
                scene_description="a quiet classroom in the morning",
                character_action="a girl raises her hand",
                dialogue="你好",
            )
        )
        db.commit()
        return shot_id
    finally:
        db.close()


def _shot_payload(shot_id: str) -> dict:
    return {
        "shot_id": shot_id,
        "version": 1,
        "scene_description": "a quiet classroom in the morning",
        "character_action": "a girl raises her hand",
        "shot_type": "medium",
        "camera_angle": "正面",
        "output_format": "9:16",
    }


def _rows(job_key: str) -> list[UsageRecord]:
    db = SessionLocal()
    try:
        return db.query(UsageRecord).filter(UsageRecord.job_key == job_key).all()
    finally:
        db.close()


class JobUsageScopeTests(unittest.TestCase):
    def test_image_generation_is_attributed_to_job_and_shot(self) -> None:
        project_id = _project()
        shot_id = _shot(project_id)
        key = f"shot:{shot_id}:storyboard"

        async def scenario() -> str:
            claim = task_registry.claim_job(key, f"shot:{shot_id}")
            self.assertTrue(claim.claimed)

            async def work() -> str:
                service = ImageService()
                return await service.generate_shot_image(
                    _shot_payload(shot_id), [], {"prompt_prefix": "", "style_label": "anime"}, project_id
                )

            task = task_registry.start(key, work())
            path = await task
            await asyncio.sleep(0.05)  # 让完成回调写终态
            return path

        path = asyncio.run(scenario())
        self.assertTrue(Path(path).exists())

        rows = _rows(key)
        self.assertEqual(len(rows), 1)
        record = rows[0]
        self.assertEqual(record.capability, "image")
        self.assertEqual(record.provider, "placeholder")
        self.assertEqual(int(record.quantity), 1)
        self.assertEqual(record.project_id, project_id)
        self.assertEqual(record.shot_id, shot_id)
        self.assertEqual(record.job_type, "shot_image")
        # 占位图是本地能力：用量可信、费用为 0（而不是「未知」）。出厂模板把
        # placeholder 记为已配置的 0 元单价，因此来源可能是 pricing；用户清空价目
        # 后则回落为 local。两者都表示「确定不花钱」。
        self.assertTrue(record.cost_known)
        self.assertEqual(int(record.cost_micro or 0), 0)
        self.assertIn(record.cost_source, {"local", "pricing"})
        self.assertEqual(record.job_status, "completed")
        self.assertTrue(record.job_id)

        # 估算与实际各自落表；预留随任务结束释放。
        db = SessionLocal()
        try:
            estimate = db.query(CostEstimate).filter(CostEstimate.estimate_key == f"job:{key}").first()
            active = (
                db.query(BudgetReservation)
                .filter(BudgetReservation.reservation_key == key, BudgetReservation.status == "active")
                .count()
            )
        finally:
            db.close()
        self.assertIsNotNone(estimate)
        self.assertEqual(active, 0)

    def test_failed_provider_call_records_failed_usage(self) -> None:
        project_id = _project()
        shot_id = _shot(project_id)
        key = f"shot:{shot_id}:storyboard"
        # 用真实云端协议（有单价）验证失败口径：失败时不能按请求数量伪造金额。
        endpoint = EndpointConfig(
            protocol="stability",
            base_url="https://api.stability.test",
            api_key="sk-test",
            model="core",
        )
        db = SessionLocal()
        try:
            pricing_service.save_pricing(
                db,
                {
                    "items": [
                        {
                            "capability": "image",
                            "provider": "stability",
                            "model": "",
                            "unit_price_micro": 50_000,
                            "configured": True,
                        }
                    ]
                },
            )
        finally:
            db.close()

        async def scenario() -> None:
            claim = task_registry.claim_job(key, f"shot:{shot_id}")
            self.assertTrue(claim.claimed)

            async def work() -> str:
                service = ImageService()
                return await service.generate_shot_image(
                    _shot_payload(shot_id), [], {"prompt_prefix": "", "style_label": "anime"}, project_id
                )

            task = task_registry.start(key, work())
            with self.assertRaises(RuntimeError):
                await task
            await asyncio.sleep(0.05)

        with (
            patch.object(ImageService, "_resolve_route", return_value=(StabilityImageAdapter(endpoint), endpoint)),
            patch.object(StabilityImageAdapter, "generate", side_effect=RuntimeError("provider down")),
        ):
            asyncio.run(scenario())

        rows = _rows(key)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "failed")
        self.assertEqual(rows[0].provider, "stability")
        # 调用失败且没有供应商回报用量：金额保持 NULL（未知），不是 0。
        self.assertFalse(rows[0].cost_known)
        self.assertIsNone(rows[0].cost_micro)

    def test_llm_tokens_are_recorded_with_cost(self) -> None:
        project_id = _project()
        db = SessionLocal()
        try:
            pricing_service.save_pricing(
                db,
                {
                    "items": [
                        {
                            "capability": "llm",
                            "provider": "openai-chat",
                            "model": "",
                            "unit_price_micro": 2_000_000,
                            "unit_price_secondary_micro": 8_000_000,
                            "configured": True,
                        }
                    ]
                },
            )
        finally:
            db.close()

        key = f"project:{project_id}:pipeline:manual"

        async def complete(self, **_kwargs):  # noqa: ANN001 - 替换适配器方法，接收 adapter 自身
            return _Response()

        async def scenario() -> None:
            claim = task_registry.claim_job(key, f"project:{project_id}")
            self.assertTrue(claim.claimed)

            async def work() -> None:
                service = LLMService()
                await service.call("system", "user")

            task = task_registry.start(key, work())
            await task
            await asyncio.sleep(0.05)

        with patch.object(OpenAIChatAdapter, "complete", new=complete):
            asyncio.run(scenario())

        rows = _rows(key)
        self.assertEqual(len(rows), 1)
        record = rows[0]
        self.assertEqual(record.capability, "llm")
        self.assertEqual(int(record.quantity), 1500)
        self.assertEqual(int(record.secondary_quantity), 400)
        self.assertEqual(record.project_id, project_id)
        self.assertEqual(record.job_type, "script_pipeline")
        self.assertTrue(record.cost_known)
        # 1500 token x 2 元/1M + 400 token x 8 元/1M = 3000 + 3200 micro
        self.assertEqual(int(record.cost_micro), 3000 + 3200)

    def test_cancelled_provider_call_is_recorded(self) -> None:
        """任务在供应商调用途中被取消：调用留痕（金额未知），任务记为已取消。"""

        project_id = _project()
        shot_id = _shot(project_id)
        key = f"shot:{shot_id}:storyboard"

        async def scenario() -> None:
            claim = task_registry.claim_job(key, f"shot:{shot_id}")
            self.assertTrue(claim.claimed)

            async def work() -> str:
                service = ImageService()
                return await service.generate_shot_image(
                    _shot_payload(shot_id), [], {"prompt_prefix": "", "style_label": "anime"}, project_id
                )

            task = task_registry.start(key, work())
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.05)

        async def cancel_soon(*_args, **_kwargs):
            raise asyncio.CancelledError()

        # 用真实云端协议（有单价但用量未知）验证：取消的调用不能被算成 0 元。
        endpoint = EndpointConfig(
            protocol="stability",
            base_url="https://api.stability.test",
            api_key="sk-test",
            model="core",
        )
        with (
            patch.object(ImageService, "_resolve_route", return_value=(StabilityImageAdapter(endpoint), endpoint)),
            patch.object(StabilityImageAdapter, "generate", new=cancel_soon),
        ):
            asyncio.run(scenario())

        rows = _rows(key)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "cancelled")
        self.assertFalse(rows[0].cost_known)
        self.assertIsNone(rows[0].cost_micro)
        self.assertEqual(rows[0].job_status, "cancelled")

    def test_manual_call_without_job_scope_is_still_recorded(self) -> None:
        """没有任务上下文的调用（例如诊断接口）也要落库，只是不带任务归属。"""

        project_id = _project()
        db = SessionLocal()
        try:
            pricing_service.save_pricing(
                db,
                {
                    "items": [
                        {
                            "capability": "image",
                            "provider": "ark-seedream",
                            "model": "",
                            "unit_price_micro": 10_000,
                            "configured": True,
                        }
                    ]
                },
            )
        finally:
            db.close()
        record = usage_service.record_metadata(
            UsageMetadata(capability="image", provider="ark-seedream", images=1),
            scope=usage_service.UsageScope(project_id=project_id),
        )
        self.assertEqual(record["job_key"], "")
        self.assertEqual(record["project_id"], project_id)


if __name__ == "__main__":
    unittest.main()
