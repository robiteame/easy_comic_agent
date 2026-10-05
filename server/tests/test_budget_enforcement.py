"""预算拦截的验收测试：软预算提示、硬预算阻止任务启动、预留与并发。

关键断言：

- 硬预算不足时任务根本不启动（不写 background_jobs 行），错误码稳定为
  budget_exceeded；
- 软预算只提示、不阻断；
- 并发启动时「已用 + 已预留」一起比对，第二个任务会被拦下（不会重复计费/超支）；
- 任务终结后预留释放，额度回到可用状态；
- 时长为第二维度，语义与金额一致。
"""

from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from db import SessionLocal, init_db  # noqa: E402
from models import BackgroundJob, BudgetReservation, Project, Shot  # noqa: E402
from services import budget_service, job_types, pricing_service, task_registry, usage_service  # noqa: E402
from services.providers.endpoint import get_endpoint  # noqa: E402
from services.providers.usage import CAPABILITY_IMAGE, CAPABILITY_VIDEO, UsageMetadata  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402

init_db()

MICRO = 1_000_000


def _project(title: str = "预算测试项目") -> str:
    db = SessionLocal()
    try:
        project_id = f"proj-{uuid.uuid4().hex[:10]}"
        db.add(Project(id=project_id, title=title))
        db.commit()
        return project_id
    finally:
        db.close()


def _shot(project_id: str, sequence: int = 1, duration: float = 5.0, dialogue: str = "") -> str:
    db = SessionLocal()
    try:
        shot_id = f"shot-{uuid.uuid4().hex[:10]}"
        db.add(Shot(id=shot_id, project_id=project_id, sequence=sequence, duration=duration, dialogue=dialogue))
        db.commit()
        return shot_id
    finally:
        db.close()


def _set_budget(
    project_id: str,
    *,
    soft: int | None = None,
    hard: int | None = None,
    hard_seconds: int | None = None,
    soft_seconds: int | None = None,
) -> None:
    db = SessionLocal()
    try:
        budget_service.save_budget(
            db,
            {
                "scope_type": "project",
                "scope_id": project_id,
                "soft_cost_micro": soft,
                "hard_cost_micro": hard,
                "soft_seconds": soft_seconds,
                "hard_seconds": hard_seconds,
            },
        )
    finally:
        db.close()


def _set_price(capability: str, provider: str, price: int, *, model: str = "") -> None:
    db = SessionLocal()
    try:
        pricing_service.save_pricing(
            db,
            {
                "items": [
                    {
                        "capability": capability,
                        "provider": provider,
                        "model": model,
                        "unit_price_micro": price,
                        "configured": True,
                    }
                ]
            },
        )
    finally:
        db.close()


def _image_provider() -> str:
    return get_endpoint("image").protocol or "placeholder"


def _active_jobs(key: str) -> int:
    db = SessionLocal()
    try:
        return db.query(BackgroundJob).filter(BackgroundJob.idempotency_key == key).count()
    finally:
        db.close()


class BudgetGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        # 图像按 provider 定价，保证估算「金额已知」，从而真的能触发硬预算。
        _set_price(CAPABILITY_IMAGE, _image_provider(), 100_000)  # 0.1 元/张

    def test_hard_budget_blocks_task_start(self) -> None:
        project_id = _project()
        shot_id = _shot(project_id)
        _set_budget(project_id, hard=50_000)  # 0.05 元 < 单张 0.1 元

        key = f"shot:{shot_id}:storyboard"
        result = task_registry.claim_job(key, f"shot:{shot_id}")

        self.assertFalse(result.claimed)
        self.assertTrue(result.blocked_by_budget)
        self.assertEqual(result.error_code, job_types.ERROR_CODE_BUDGET_EXCEEDED)
        # 任务根本没有被创建：硬预算不只是提示。
        self.assertEqual(_active_jobs(key), 0)
        self.assertIn("硬预算", result.message)
        # 估算被记录下来，便于用户看到「为什么被拦」。
        self.assertIn("estimate", result.budget)

    def test_soft_budget_warns_but_allows(self) -> None:
        project_id = _project()
        shot_id = _shot(project_id)
        _set_budget(project_id, soft=10_000)  # 软预算 0.01 元，低于单张成本

        key = f"shot:{shot_id}:storyboard"
        result = task_registry.claim_job(key, f"shot:{shot_id}")
        try:
            self.assertTrue(result.claimed)
            self.assertEqual(result.budget["level"], budget_service.LEVEL_SOFT_EXCEEDED)
            self.assertEqual(result.budget["code"], budget_service.CODE_BUDGET_SOFT_EXCEEDED)
        finally:
            task_registry.finish(key, "completed")

    def test_hard_budget_blocks_when_already_spent(self) -> None:
        project_id = _project()
        shot_id = _shot(project_id)
        _set_budget(project_id, hard=200_000)
        # 先真实发生一笔 0.3 元的调用，已超出硬预算。
        usage_service.record_metadata(
            UsageMetadata(capability=CAPABILITY_IMAGE, provider=_image_provider(), images=3),
            scope=usage_service.UsageScope(project_id=project_id, shot_id=shot_id),
        )
        result = task_registry.claim_job(f"shot:{shot_id}:storyboard", f"shot:{shot_id}")
        self.assertFalse(result.claimed)
        self.assertEqual(result.error_code, job_types.ERROR_CODE_BUDGET_EXCEEDED)

    def test_concurrent_tasks_cannot_double_spend(self) -> None:
        project_id = _project()
        first = _shot(project_id, sequence=1)
        second = _shot(project_id, sequence=2)
        # 硬预算刚好够一个镜头：单张 0.1 元。
        _set_budget(project_id, hard=100_000)

        key_a = f"shot:{first}:storyboard"
        key_b = f"shot:{second}:storyboard"
        claim_a = task_registry.claim_job(key_a, f"shot:{first}")
        claim_b = task_registry.claim_job(key_b, f"shot:{second}")
        try:
            self.assertTrue(claim_a.claimed)
            # A 的实际花费要等结束才可见；若没有 A 的预留，B 会误判为额度充足。
            self.assertFalse(claim_b.claimed)
            self.assertEqual(claim_b.error_code, job_types.ERROR_CODE_BUDGET_EXCEEDED)
        finally:
            task_registry.finish(key_a, "completed")
            task_registry.finish(key_b, "cancelled")

        # A 结束后预留释放（实际花费为 0，因为这条路径没有真正调供应商），额度恢复。
        claim_c = task_registry.claim_job(key_b, f"shot:{second}")
        try:
            self.assertTrue(claim_c.claimed)
        finally:
            task_registry.finish(key_b, "completed")

    def test_reservation_is_unique_per_job_and_released_on_finish(self) -> None:
        project_id = _project()
        shot_id = _shot(project_id)
        _set_budget(project_id, hard=10 * MICRO)
        key = f"shot:{shot_id}:storyboard"
        self.assertTrue(task_registry.claim_job(key, f"shot:{shot_id}").claimed)
        # 同一任务重复抢占（例如重复点击）不会重复预留。
        self.assertFalse(task_registry.claim_job(key, f"shot:{shot_id}").claimed)

        db = SessionLocal()
        try:
            rows = (
                db.query(BudgetReservation)
                .filter(BudgetReservation.reservation_key == key, BudgetReservation.status == "active")
                .all()
            )
        finally:
            db.close()
        self.assertEqual(len(rows), 1)

        task_registry.finish(key, "failed", "boom")
        db = SessionLocal()
        try:
            row = db.query(BudgetReservation).filter(BudgetReservation.reservation_key == key).first()
            status = str(row.status)
        finally:
            db.close()
        self.assertEqual(status, budget_service.RESERVATION_RELEASED)

    def test_failed_task_keeps_its_cost_and_releases_budget(self) -> None:
        project_id = _project()
        shot_id = _shot(project_id)
        _set_budget(project_id, hard=10 * MICRO)
        key = f"shot:{shot_id}:storyboard"
        self.assertTrue(task_registry.claim_job(key, f"shot:{shot_id}").claimed)

        # 任务执行中真实发生了一次付费调用，然后任务失败。
        usage_service.record_metadata(
            UsageMetadata(capability=CAPABILITY_IMAGE, provider=_image_provider(), images=1),
            scope=usage_service.UsageScope(project_id=project_id, shot_id=shot_id, job_key=key, job_type="shot_image"),
        )
        task_registry.finish(key, "failed", "provider exploded")

        db = SessionLocal()
        try:
            summary = usage_service.summarize(db, job_key=key)
            reservations = (
                db.query(BudgetReservation)
                .filter(BudgetReservation.reservation_key == key, BudgetReservation.status == "active")
                .count()
            )
        finally:
            db.close()
        # 失败任务的花费仍可查询，且不再占用预算额度。
        self.assertEqual(summary["cost_micro"], 100_000)
        self.assertEqual(reservations, 0)

    def test_duration_budget_blocks(self) -> None:
        project_id = _project()
        shot_id = _shot(project_id)
        # 硬时长预算设为 0 秒：任何需要时间的任务都必须被拦下。
        # 这里不用「1 秒」是为了不让同类任务的历史耗时样本影响结论——预计耗时优先取
        # 历史中位数（可能恰好是 1 秒），那样断言就会随测试执行顺序变化。
        _set_budget(project_id, hard_seconds=0)
        result = task_registry.claim_job(f"shot:{shot_id}:storyboard", f"shot:{shot_id}")
        self.assertFalse(result.claimed)
        self.assertEqual(result.error_code, job_types.ERROR_CODE_BUDGET_EXCEEDED)
        self.assertEqual(result.budget["details"]["reason"], "seconds")

    def test_unknown_estimate_does_not_block_but_is_reported(self) -> None:
        project_id = _project()
        shot_id = _shot(project_id)
        # 视频 provider 没有配置价目 -> 估算金额未知
        _set_budget(project_id, hard=1 * MICRO)
        db = SessionLocal()
        try:
            pricing_service.save_pricing(
                db,
                {
                    "items": [
                        {
                            "capability": CAPABILITY_VIDEO,
                            "provider": get_endpoint("video").protocol,
                            "model": "",
                            "unit_price_micro": None,
                            "configured": False,
                        }
                    ]
                },
            )
        finally:
            db.close()
        estimate = budget_service.estimate_job(
            db=SessionLocal(), job_type=job_types.JOB_TYPE_SHOT_VIDEO, project_id=project_id, shot_id=shot_id
        )
        self.assertFalse(estimate["cost_known"])
        self.assertIsNone(estimate["estimated_cost_micro"])
        self.assertTrue(estimate["unknown_components"])

    def test_evaluate_limits_levels(self) -> None:
        limits = {
            "currency": "CNY",
            "soft_cost_micro": 1 * MICRO,
            "hard_cost_micro": 2 * MICRO,
            "source": "project",
            "source_label": "项目预算",
        }
        ok = budget_service.evaluate_limits(
            limits, used_cost_micro=0, reserved_cost_micro=0, estimate_cost_micro=100_000
        )
        self.assertEqual(ok["level"], budget_service.LEVEL_OK)

        soft = budget_service.evaluate_limits(
            limits, used_cost_micro=900_000, reserved_cost_micro=0, estimate_cost_micro=300_000
        )
        self.assertEqual(soft["level"], budget_service.LEVEL_SOFT_EXCEEDED)

        hard = budget_service.evaluate_limits(
            limits, used_cost_micro=1_900_000, reserved_cost_micro=0, estimate_cost_micro=300_000
        )
        self.assertEqual(hard["level"], budget_service.LEVEL_HARD_EXCEEDED)
        self.assertEqual(hard["code"], budget_service.CODE_BUDGET_EXCEEDED)

        unlimited = budget_service.evaluate_limits(
            {"currency": "CNY", "soft_cost_micro": None, "hard_cost_micro": None},
            used_cost_micro=0,
            estimate_cost_micro=10 * MICRO,
        )
        self.assertEqual(unlimited["level"], budget_service.LEVEL_UNLIMITED)


class SoftBudgetSemanticsTests(unittest.TestCase):
    """软预算必须区分「实际已超支」与「预计将超支」（金额与时长同语义）。

    边界：实际 < 软预算但投影 > 软预算、实际 == 软预算、实际 > 软预算、
    投影 == 软预算、成本未知但时长超限。
    """

    MONEY_LIMITS = {
        "currency": "CNY",
        "soft_cost_micro": 1 * MICRO,
        "source": "project",
        "source_label": "项目预算",
    }
    SECONDS_LIMITS = {
        "currency": "CNY",
        "soft_seconds": 100,
        "source": "project",
        "source_label": "项目预算",
    }

    def test_projection_over_soft_warns_pending_not_exceeded(self) -> None:
        # 实际 0.4 元 < 软预算 1 元，但「已用 + 本次预计 0.8 元」的投影会超。
        state = budget_service.evaluate_limits(self.MONEY_LIMITS, used_cost_micro=400_000, estimate_cost_micro=800_000)
        self.assertEqual(state["level"], budget_service.LEVEL_SOFT_EXCEEDED)
        self.assertEqual(state["reason"], "cost")
        self.assertIn("预计将超支", state["message"])
        self.assertNotIn("已超支", state["message"])
        # 预计超支文案必须交代：已用、本次预计、预算上限。
        self.assertIn("预算上限", state["message"])
        self.assertIn("本次预计", state["message"])

    def test_actual_equal_to_soft_counts_as_exceeded(self) -> None:
        state = budget_service.evaluate_limits(self.MONEY_LIMITS, used_cost_micro=1 * MICRO, estimate_cost_micro=None)
        self.assertEqual(state["level"], budget_service.LEVEL_SOFT_EXCEEDED)
        self.assertIn("已超支", state["message"])
        self.assertNotIn("预计将超支", state["message"])

    def test_actual_over_soft_counts_as_exceeded(self) -> None:
        state = budget_service.evaluate_limits(
            self.MONEY_LIMITS, used_cost_micro=1_500_000, estimate_cost_micro=200_000
        )
        self.assertEqual(state["level"], budget_service.LEVEL_SOFT_EXCEEDED)
        self.assertIn("已超支", state["message"])
        self.assertNotIn("预计将超支", state["message"])

    def test_projection_equal_to_soft_stays_ok(self) -> None:
        # 投影恰好等于软预算不算超支（严格大于才触发提示）。
        state = budget_service.evaluate_limits(self.MONEY_LIMITS, used_cost_micro=400_000, estimate_cost_micro=600_000)
        self.assertEqual(state["level"], budget_service.LEVEL_OK)
        self.assertEqual(state["message"], "")

    def test_seconds_projection_over_soft_warns_pending(self) -> None:
        state = budget_service.evaluate_limits(
            self.SECONDS_LIMITS, used_cost_micro=0, used_seconds=40, estimate_seconds=90
        )
        self.assertEqual(state["level"], budget_service.LEVEL_SOFT_EXCEEDED)
        self.assertEqual(state["reason"], "seconds")
        self.assertIn("预计将超支", state["message"])
        self.assertNotIn("已超支", state["message"])
        self.assertIn("已用 40 秒", state["message"])
        self.assertIn("本次预计 90 秒", state["message"])
        self.assertIn("预算上限 100 秒", state["message"])

    def test_unknown_cost_with_seconds_over_soft_is_reported(self) -> None:
        # 成本未知不掩盖时长软预算提示；文案也不能声称金额超支。
        limits = dict(self.MONEY_LIMITS, soft_seconds=100)
        state = budget_service.evaluate_limits(
            limits, used_cost_micro=0, used_seconds=150, estimate_seconds=None, cost_known=False
        )
        self.assertEqual(state["level"], budget_service.LEVEL_SOFT_EXCEEDED)
        self.assertEqual(state["reason"], "seconds")
        self.assertIn("已超支", state["message"])
        self.assertIn("已用 150 秒", state["message"])


if __name__ == "__main__":
    unittest.main()
