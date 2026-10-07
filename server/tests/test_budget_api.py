"""成本 / 预算 API 的验收测试（HTTP 层）。

覆盖：价格配置读写与浮点拒绝、预算配置校验、提交前估算、项目/剧集/镜头/任务
统计接口、任务中心 DTO 的成本字段、硬预算通过真实路由返回 409、以及「不泄漏
API Key / 供应商原始响应」的安全约束。
"""

from __future__ import annotations

import unittest
import uuid

from fastapi.testclient import TestClient  # noqa: E402

from db import SessionLocal, init_db  # noqa: E402
from main import app  # noqa: E402
from models import BackgroundJob, Project, Shot  # noqa: E402
from services import pricing_service, task_registry, usage_service  # noqa: E402
from services.providers.endpoint import get_endpoint  # noqa: E402
from services.providers.usage import CAPABILITY_IMAGE, UsageMetadata  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402

init_db()

MICRO = 1_000_000


def _project(title: str = "API 成本测试项目", parent: str = "") -> str:
    db = SessionLocal()
    try:
        project_id = f"proj-{uuid.uuid4().hex[:10]}"
        db.add(
            Project(
                id=project_id,
                title=title,
                project_type="episode" if parent else "series",
                parent_project_id=parent,
                episode_number=1 if parent else 0,
            )
        )
        db.commit()
        return project_id
    finally:
        db.close()


def _shot(project_id: str, sequence: int = 1, duration: float = 5.0) -> str:
    db = SessionLocal()
    try:
        shot_id = f"shot-{uuid.uuid4().hex[:10]}"
        db.add(Shot(id=shot_id, project_id=project_id, sequence=sequence, duration=duration))
        db.commit()
        return shot_id
    finally:
        db.close()


def _set_price(capability: str, provider: str, price: int | None) -> None:
    db = SessionLocal()
    try:
        pricing_service.save_pricing(
            db,
            {
                "items": [
                    {
                        "capability": capability,
                        "provider": provider,
                        "model": "",
                        "unit_price_micro": price,
                        "configured": price is not None,
                    }
                ]
            },
        )
    finally:
        db.close()


class BudgetApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_pricing_endpoint_lists_capabilities_and_units(self) -> None:
        response = self.client.get("/api/budget/pricing")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        capabilities = {item["capability"] for item in payload["capabilities"]}
        self.assertEqual(capabilities, {"llm", "image", "video", "tts", "ffmpeg"})
        self.assertEqual(payload["micro_per_unit"], MICRO)
        self.assertIn("向上取整", payload["rounding"])

    def test_pricing_rejects_float_amount(self) -> None:
        response = self.client.put(
            "/api/budget/pricing",
            json={"items": [{"capability": "image", "provider": "ark-seedream", "unit_price_micro": 0.15}]},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("整数", response.json()["detail"])

    def test_pricing_roundtrip(self) -> None:
        response = self.client.put(
            "/api/budget/pricing",
            json={
                "items": [
                    {
                        "capability": "image",
                        "provider": "ark-seedream",
                        "model": "doubao-seedream-5.0-lite",
                        "unit_price_micro": 25_000,
                        "resolution_multipliers": {"2048x2048": 2_000_000},
                        "configured": True,
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 200)
        items = [
            item
            for group in response.json()["capabilities"]
            if group["capability"] == "image"
            for item in group["items"]
        ]
        saved = next(item for item in items if item["model"] == "doubao-seedream-5.0-lite")
        self.assertEqual(saved["unit_price_micro"], 25_000)
        self.assertEqual(saved["resolution_multipliers"], {"2048x2048": 2_000_000})
        self.assertTrue(saved["configured"])

    def test_budget_config_validation_and_effective_limits(self) -> None:
        project_id = _project()
        bad = self.client.put(
            "/api/budget/config",
            json={
                "scope_type": "project",
                "scope_id": project_id,
                "soft_cost_micro": 5 * MICRO,
                "hard_cost_micro": 1 * MICRO,
            },
        )
        self.assertEqual(bad.status_code, 400)
        self.assertIn("软预算", bad.json()["detail"])

        ok = self.client.put(
            "/api/budget/config",
            json={
                "scope_type": "project",
                "scope_id": project_id,
                "soft_cost_micro": 1 * MICRO,
                "hard_cost_micro": 3 * MICRO,
                "hard_seconds": 3600,
            },
        )
        self.assertEqual(ok.status_code, 200)
        effective = ok.json()["effective"]
        self.assertEqual(effective["source"], "project")
        self.assertEqual(effective["hard_cost_micro"], 3 * MICRO)
        self.assertEqual(effective["hard_seconds"], 3600)

        fetched = self.client.get("/api/budget/config", params={"project_id": project_id})
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json()["project"]["soft_cost_micro"], 1 * MICRO)

    def test_estimate_endpoint_reports_cost_duration_and_block(self) -> None:
        project_id = _project()
        shot_id = _shot(project_id)
        _set_price("image", get_endpoint("image").protocol or "placeholder", 100_000)

        preview = self.client.post(
            "/api/budget/estimate",
            json={"job_type": "shot_image", "project_id": project_id, "shot_id": shot_id},
        )
        self.assertEqual(preview.status_code, 200)
        body = preview.json()
        self.assertTrue(body["estimate"]["cost_known"])
        self.assertEqual(body["estimate"]["estimated_cost_micro"], 100_000)
        self.assertGreater(body["estimate"]["estimated_seconds"], 0)
        self.assertFalse(body["blocked"])

        self.client.put(
            "/api/budget/config",
            json={"scope_type": "project", "scope_id": project_id, "hard_cost_micro": 50_000},
        )
        blocked = self.client.post(
            "/api/budget/estimate",
            json={"job_type": "shot_image", "project_id": project_id, "shot_id": shot_id},
        ).json()
        self.assertTrue(blocked["blocked"])
        self.assertEqual(blocked["budget"]["code"], "budget_exceeded")

    def test_estimate_unknown_cost_is_explicit(self) -> None:
        project_id = _project()
        shot_id = _shot(project_id)
        # 视频 provider 明确「未配置价格」。
        _set_price("video", get_endpoint("video").protocol, None)
        body = self.client.post(
            "/api/budget/estimate",
            json={"job_type": "shot_video", "project_id": project_id, "shot_id": shot_id},
        ).json()
        self.assertFalse(body["estimate"]["cost_known"])
        self.assertIsNone(body["estimate"]["estimated_cost_micro"])
        self.assertTrue(body["estimate"]["unknown_components"])
        self.assertIn("未知", body["estimate"]["note"])

    def test_hard_budget_blocks_real_route(self) -> None:
        project_id = _project()
        _shot(project_id)
        _set_price("image", get_endpoint("image").protocol or "placeholder", 200_000)
        self.client.put(
            "/api/budget/config",
            json={"scope_type": "project", "scope_id": project_id, "hard_cost_micro": 100_000},
        )
        response = self.client.post(f"/api/shot/{project_id}/generate-storyboard", json={})
        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertEqual(detail["error_code"], "budget_exceeded")
        self.assertIn("硬预算", detail["message"])
        self.assertIn("budget", detail)

    def test_summary_and_usage_endpoints(self) -> None:
        series_id = _project("系列剧")
        episode_id = _project("第 1 集", parent=series_id)
        shot_id = _shot(episode_id)
        _set_price("image", get_endpoint("image").protocol or "placeholder", 100_000)
        _set_price("video", get_endpoint("video").protocol, 500_000)

        usage_service.record_metadata(
            UsageMetadata(capability=CAPABILITY_IMAGE, provider=get_endpoint("image").protocol, images=2),
            scope=usage_service.UsageScope(project_id=episode_id, shot_id=shot_id, job_type="shot_image"),
        )
        self.client.put(
            "/api/budget/config",
            json={
                "scope_type": "project",
                "scope_id": episode_id,
                "soft_cost_micro": 1 * MICRO,
                "hard_cost_micro": 5 * MICRO,
            },
        )

        summary = self.client.get("/api/budget/summary", params={"project_id": episode_id}).json()
        self.assertEqual(summary["used"]["cost_micro"], 200_000)
        self.assertTrue(summary["used"]["cost_known"])
        # 预计成本 = 剩余故事板 1 张（0.1 元）+ 剩余视频 5 秒（2.5 元）
        self.assertTrue(summary["remaining"]["cost_known"])
        self.assertEqual(summary["remaining"]["cost_micro"], 100_000 + 2_500_000)
        self.assertGreater(summary["remaining"]["seconds"], 0)
        self.assertEqual(summary["budget"]["hard_cost_micro"], 5 * MICRO)
        self.assertIn(summary["status"]["level"], {"ok", "soft_exceeded", "hard_exceeded"})

        series_summary = self.client.get("/api/budget/summary", params={"series_id": series_id}).json()
        self.assertEqual(series_summary["used"]["cost_micro"], 200_000)
        self.assertTrue(any(item["project_id"] == episode_id for item in series_summary["episodes"]))

        usage = self.client.get(
            "/api/budget/usage",
            params={"project_id": episode_id, "group_by": ["job_type", "shot", "capability"]},
        ).json()
        self.assertEqual(usage["summary"]["call_count"], 1)
        self.assertEqual(usage["summary"]["cost_micro"], 200_000)
        self.assertEqual(usage["records"]["total"], 1)
        self.assertTrue(usage["groups"]["job_type"])
        self.assertTrue(usage["groups"]["shot"])

    def test_usage_endpoint_marks_unknown_cost_explicitly(self) -> None:
        project_id = _project()
        usage_service.record_metadata(
            UsageMetadata(capability=CAPABILITY_IMAGE, provider="mystery-protocol", model="mystery", images=1),
            scope=usage_service.UsageScope(project_id=project_id),
        )
        usage = self.client.get("/api/budget/usage", params={"project_id": project_id}).json()
        self.assertFalse(usage["summary"]["cost_known"])
        self.assertEqual(usage["summary"]["unknown_call_count"], 1)
        record = usage["records"]["items"][0]
        self.assertIsNone(record["cost_micro"])
        self.assertFalse(record["cost_known"])
        self.assertEqual(record["cost_source"], "unknown")

    def test_task_center_dto_exposes_cost_and_estimate(self) -> None:
        project_id = _project()
        shot_id = _shot(project_id)
        _set_price("image", get_endpoint("image").protocol or "placeholder", 100_000)
        key = f"shot:{shot_id}:storyboard"
        claim = task_registry.claim_job(key, f"shot:{shot_id}")
        self.assertTrue(claim.claimed)
        job_id = claim.budget["details"].get("estimate", {}).get("job_type") and None
        db = SessionLocal()
        try:
            job = db.query(BackgroundJob).filter(BackgroundJob.idempotency_key == key).first()
            job_id = str(job.id)
        finally:
            db.close()

        usage_service.record_metadata(
            UsageMetadata(capability=CAPABILITY_IMAGE, provider=get_endpoint("image").protocol, images=1),
            scope=usage_service.UsageScope(project_id=project_id, shot_id=shot_id, job_key=key, job_id=job_id),
        )
        task_registry.finish(key, "failed", "provider exploded")

        listing = self.client.get("/api/jobs", params={"project_id": project_id}).json()
        item = next(entry for entry in listing["items"] if entry["id"] == job_id)
        self.assertEqual(item["cost"]["cost_micro"], 100_000)
        self.assertTrue(item["cost"]["cost_known"])
        self.assertEqual(item["cost"]["estimated_cost_micro"], 100_000)
        self.assertGreaterEqual(item["duration_seconds"], 0)

        detail = self.client.get(f"/api/budget/jobs/{job_id}").json()
        self.assertEqual(detail["summary"]["cost_micro"], 100_000)
        self.assertIsNotNone(detail["estimate"])
        self.assertEqual(detail["records"]["total"], 1)

        missing = self.client.get(f"/api/budget/jobs/{uuid.uuid4().hex}")
        self.assertEqual(missing.status_code, 404)

    def test_responses_never_leak_secrets(self) -> None:
        project_id = _project()
        self.client.put(
            "/api/budget/pricing",
            json={
                "items": [
                    {
                        "capability": "llm",
                        "provider": "openai-chat",
                        "model": "gpt-x",
                        "unit_price_micro": 1_000_000,
                        "configured": True,
                    }
                ]
            },
        )
        for url in ("/api/budget/pricing", f"/api/budget/config?project_id={project_id}", "/api/budget/usage"):
            body = self.client.get(url).text
            self.assertNotIn("api_key", body)
            self.assertNotIn("Authorization", body)
            self.assertNotIn("choices", body)
            self.assertNotIn("sk-", body)

    def test_budget_estimate_requires_valid_job_type(self) -> None:
        project_id = _project()
        response = self.client.post("/api/budget/estimate", json={"job_type": "not_a_job", "project_id": project_id})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
