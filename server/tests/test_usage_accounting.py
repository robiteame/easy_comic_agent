"""用量记账与价格配置的验收测试。

覆盖：统一 usage metadata、未知 provider 不伪造金额、整数金额运算、
估算与实际分表、失败与取消留痕、重复调用不重复计费、并发不重复计费。
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
import uuid
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from db import SessionLocal, init_db  # noqa: E402
from models import CostEstimate, PricingConfig, Project, Shot, UsageRecord  # noqa: E402
from services import pricing_service, usage_service  # noqa: E402
from services.providers.base import ImageRequest, TTSRequest, VideoRequest  # noqa: E402
from services.providers.endpoint import EndpointConfig  # noqa: E402
from services.providers.image_placeholder import PlaceholderImageAdapter  # noqa: E402
from services.providers.tts_mimo import MimoTTSAdapter  # noqa: E402
from services.providers.usage import (  # noqa: E402
    CAPABILITY_IMAGE,
    CAPABILITY_LLM,
    CAPABILITY_TTS,
    CAPABILITY_VIDEO,
    UsageMetadata,
    unknown_usage,
    usage_from_chat_response,
)
from services.providers.video_ark_seedance import ArkSeedanceVideoAdapter  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402

init_db()


class ChatUsageStub:
    """最小 OpenAI 兼容响应替身（只保留 usage 字段）。"""

    def __init__(self, prompt_tokens=None, completion_tokens=None):
        if prompt_tokens is None and completion_tokens is None:
            self.usage = None
        else:
            self.usage = type(
                "Usage",
                (),
                {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            )()


def _endpoint(protocol: str, model: str = "test-model") -> EndpointConfig:
    return EndpointConfig(protocol=protocol, base_url="https://example.test", api_key="k", model=model)


def _project(title: str = "成本测试项目") -> str:
    """建一个项目并返回纯字符串 ID（不返回 ORM 对象，避免 detached 访问）。"""

    db = SessionLocal()
    try:
        project_id = f"proj-{uuid.uuid4().hex[:10]}"
        db.add(Project(id=project_id, title=title))
        db.commit()
        return project_id
    finally:
        db.close()


def _shot(project_id: str, sequence: int = 1, duration: float = 3.0, dialogue: str = "你好呀") -> str:
    """建一个镜头并返回纯字符串 ID（不返回 ORM 对象，避免 detached 访问）。"""

    db = SessionLocal()
    try:
        shot_id = f"shot-{uuid.uuid4().hex[:10]}"
        db.add(
            Shot(
                id=shot_id,
                project_id=project_id,
                sequence=sequence,
                duration=duration,
                dialogue=dialogue,
            )
        )
        db.commit()
        return shot_id
    finally:
        db.close()


def _set_price(
    capability: str,
    provider: str,
    price: int | None,
    *,
    model: str = "",
    secondary: int | None = None,
    multipliers: dict | None = None,
    configured: bool | None = None,
) -> None:
    item: dict = {
        "capability": capability,
        "provider": provider,
        "model": model,
        "unit_price_micro": price,
        "unit_price_secondary_micro": secondary,
        "resolution_multipliers": multipliers or {},
    }
    if configured is not None:
        item["configured"] = configured
    db = SessionLocal()
    try:
        pricing_service.save_pricing(db, {"items": [item]})
    finally:
        db.close()


def _clear_price(capability: str, provider: str, model: str = "") -> None:
    db = SessionLocal()
    try:
        db.query(PricingConfig).filter(
            PricingConfig.capability == capability,
            PricingConfig.provider == provider,
            PricingConfig.model == model,
        ).delete()
        db.commit()
    finally:
        db.close()


class ProviderUsageMetadataTests(unittest.TestCase):
    """Provider 适配器统一返回 usage metadata。"""

    def test_chat_response_maps_tokens(self) -> None:
        metadata = usage_from_chat_response(
            ChatUsageStub(1200, 340), provider="openai-chat", model="m1", duration_ms=88
        )
        self.assertEqual(metadata.capability, CAPABILITY_LLM)
        self.assertEqual(metadata.quantity, 1200)
        self.assertEqual(metadata.secondary_quantity, 340)
        self.assertTrue(metadata.known)
        self.assertEqual(metadata.duration_ms, 88)

    def test_chat_response_without_usage_is_unknown(self) -> None:
        metadata = usage_from_chat_response(ChatUsageStub(), provider="openai-chat", model="m1")
        self.assertFalse(metadata.known)
        self.assertFalse(metadata.has_usage)

    def test_image_video_tts_adapters_declare_usage(self) -> None:
        image = PlaceholderImageAdapter(_endpoint("placeholder")).usage_for_request(
            CAPABILITY_IMAGE, ImageRequest(prompt="p", size="1440x2560")
        )
        self.assertEqual(image.capability, CAPABILITY_IMAGE)
        self.assertEqual(image.quantity, 1)
        self.assertEqual(image.resolution, "1440x2560")
        # 本地占位图不产生外部费用，但用量可信。
        self.assertTrue(image.known)
        self.assertFalse(image.billable)

        video = ArkSeedanceVideoAdapter(_endpoint("ark-seedance", "doubao-seedance")).usage_for_request(
            CAPABILITY_VIDEO, VideoRequest(prompt="p", duration=5, resolution="1080p")
        )
        self.assertEqual(video.quantity, 5)
        self.assertEqual(video.resolution, "1080p")
        self.assertTrue(video.billable)

        tts = MimoTTSAdapter(_endpoint("mimo-tts")).usage_for_request(
            CAPABILITY_TTS, TTSRequest(text="你好呀，世界", voice_id="冰糖")
        )
        self.assertEqual(tts.quantity, 6)

    def test_seconds_round_up_never_zero(self) -> None:
        metadata = UsageMetadata(capability=CAPABILITY_VIDEO, seconds=0.4)
        self.assertEqual(metadata.quantity, 1)


class MoneyMathTests(unittest.TestCase):
    """金额一律整数（最小货币单位），禁止浮点。"""

    def test_cost_uses_integer_micro_and_rounds_up(self) -> None:
        # 2 元 / 100 万 token，1000 token -> 2000 micro
        self.assertEqual(
            pricing_service.compute_cost_micro(1000, 0, unit_price_micro=2_000_000, unit_scale=1_000_000),
            2000,
        )
        # 极小额度向上取整到 1 micro，不会被舍成 0
        self.assertEqual(
            pricing_service.compute_cost_micro(1, 0, unit_price_micro=100, unit_scale=1_000_000),
            1,
        )

    def test_secondary_price_adds_output_cost(self) -> None:
        cost = pricing_service.compute_cost_micro(
            1000,
            2000,
            unit_price_micro=1_000_000,
            unit_price_secondary_micro=3_000_000,
            unit_scale=1_000_000,
        )
        self.assertEqual(cost, 1000 + 6000)

    def test_resolution_multiplier_is_integer(self) -> None:
        self.assertEqual(pricing_service.apply_multiplier(1_000_000, 1_500_000), 1_500_000)
        with self.assertRaises(ValueError):
            pricing_service.parse_multipliers({"1080p": 1.5})

    def test_save_pricing_rejects_float_amount(self) -> None:
        db = SessionLocal()
        try:
            with self.assertRaises(ValueError):
                pricing_service.save_pricing(
                    db,
                    {"items": [{"capability": CAPABILITY_IMAGE, "provider": "ark-seedream", "unit_price_micro": 1.5}]},
                )
        finally:
            db.close()


class UsageRecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        # 只保留纯字符串 ID：多线程用例里再碰 ORM 对象会拿到 detached instance。
        self.project_id = _project()

    def test_unknown_provider_reports_unknown_cost_not_zero(self) -> None:
        _clear_price(CAPABILITY_IMAGE, "mystery-protocol")
        job_key = f"project:{self.project_id}:storyboard"
        metadata = unknown_usage(CAPABILITY_IMAGE, "mystery-protocol", "unknown-model")
        record = usage_service.record_metadata(
            metadata,
            duration_ms=12,
            scope=usage_service.UsageScope(project_id=self.project_id, job_key=job_key, job_type="storyboard"),
        )
        self.assertIsNotNone(record)
        # 未知 provider 必须显示「成本未知」：cost_micro 为 NULL，而不是 0。
        self.assertFalse(record["cost_known"])
        self.assertIsNone(record["cost_micro"])
        self.assertEqual(record["cost_source"], "unknown")

        db = SessionLocal()
        try:
            summary = usage_service.summarize(db, project_id=self.project_id)
        finally:
            db.close()
        self.assertEqual(summary["unknown_call_count"], 1)
        self.assertEqual(summary["cost_micro"], 0)
        self.assertFalse(summary["cost_known"])

    def test_priced_provider_records_exact_cost(self) -> None:
        _set_price(CAPABILITY_IMAGE, "ark-seedream", 30_000)  # 0.03 元/张
        record = usage_service.record_metadata(
            UsageMetadata(capability=CAPABILITY_IMAGE, provider="ark-seedream", model="doubao", images=4),
            duration_ms=2500,
        )
        self.assertTrue(record["cost_known"])
        self.assertEqual(record["cost_micro"], 120_000)
        self.assertEqual(record["cost_source"], "pricing")

    def test_local_placeholder_is_zero_cost_not_unknown(self) -> None:
        _clear_price(CAPABILITY_IMAGE, "placeholder")
        record = usage_service.record_metadata(
            UsageMetadata(
                capability=CAPABILITY_IMAGE,
                provider="placeholder",
                images=1,
                billable=False,
                source="local",
            )
        )
        self.assertTrue(record["cost_known"])
        self.assertEqual(record["cost_micro"], 0)
        self.assertEqual(record["cost_source"], "local")

    def test_failed_call_is_recorded_without_fabricated_amount(self) -> None:
        _set_price(CAPABILITY_VIDEO, "ark-seedance", 500_000)  # 0.5 元/秒
        failed = usage_service.record_failure(
            # 适配器在调用前按请求推导数量（source=request）；调用失败时这 5 秒并不代表
            # 供应商已经计费，因此不能算进成本。
            UsageMetadata(
                capability=CAPABILITY_VIDEO,
                provider="ark-seedance",
                seconds=5,
                source="request",
            ),
            error_code="provider_call_failed",
            duration_ms=900,
        )
        self.assertEqual(failed["status"], "failed")
        # 调用失败且没有供应商回报用量 -> 成本未知，而不是把 5 秒当成已计费
        self.assertFalse(failed["cost_known"])
        self.assertIsNone(failed["cost_micro"])

    def test_provider_reported_usage_survives_failure(self) -> None:
        _set_price(CAPABILITY_LLM, "openai-chat", 2_000_000, secondary=8_000_000)
        metadata = usage_from_chat_response(ChatUsageStub(1000, 500), provider="openai-chat", model="m")
        record = usage_service.record_metadata(metadata, status="failed", error_code="downstream_error")
        self.assertTrue(record["cost_known"])
        self.assertEqual(record["cost_micro"], 2000 + 4000)

    def test_duplicate_usage_key_is_not_billed_twice(self) -> None:
        _set_price(CAPABILITY_IMAGE, "ark-seedream", 30_000)
        metadata = UsageMetadata(capability=CAPABILITY_IMAGE, provider="ark-seedream", images=1)
        first = usage_service.record_metadata(metadata, dedupe_key="dup-key-1")
        second = usage_service.record_metadata(metadata, dedupe_key="dup-key-1")
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["id"], second["id"])
        db = SessionLocal()
        try:
            rows = db.query(UsageRecord).filter(UsageRecord.usage_key == "dup-key-1").count()
        finally:
            db.close()
        self.assertEqual(rows, 1)

    def test_concurrent_recording_counts_each_call_once(self) -> None:
        _set_price(CAPABILITY_IMAGE, "ark-seedream", 10_000)
        project_id = self.project_id
        errors: list[Exception] = []

        def worker(index: int) -> None:
            try:
                for _ in range(5):
                    usage_service.record_metadata(
                        UsageMetadata(
                            capability=CAPABILITY_IMAGE,
                            provider="ark-seedream",
                            model="seedream",
                            images=1,
                        ),
                        scope=usage_service.UsageScope(project_id=project_id, shot_id=f"shot-{index}"),
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        db = SessionLocal()
        try:
            summary = usage_service.summarize(db, project_id=project_id)
            rows = db.query(UsageRecord).filter(UsageRecord.project_id == project_id).count()
        finally:
            db.close()
        self.assertEqual(rows, 40)
        self.assertEqual(summary["call_count"], 40)
        self.assertEqual(summary["cost_micro"], 40 * 10_000)

    def test_estimate_and_actual_are_stored_separately(self) -> None:
        project_id = self.project_id
        estimate = usage_service.save_estimate(
            estimate_key=f"job:project:{project_id}:render",
            job_key=f"project:{project_id}:render",
            job_type="render",
            project_id=project_id,
            estimated_cost_micro=5_000_000,
            cost_known=True,
            estimated_seconds=120,
            duration_source="heuristic",
            components=[{"capability": "ffmpeg", "quantity": 60}],
        )
        self.assertEqual(estimate["estimated_cost_micro"], 5_000_000)

        _set_price(CAPABILITY_IMAGE, "ark-seedream", 20_000)
        actual = usage_service.record_metadata(
            UsageMetadata(capability=CAPABILITY_IMAGE, provider="ark-seedream", images=2),
            scope=usage_service.UsageScope(project_id=project_id, job_key=f"project:{project_id}:render"),
        )
        self.assertNotEqual(actual["id"], estimate["estimate_key"])

        db = SessionLocal()
        try:
            estimate_row = db.query(CostEstimate).filter(CostEstimate.estimate_key == estimate["estimate_key"]).first()
            usage_row = db.query(UsageRecord).filter(UsageRecord.usage_key == actual["usage_key"]).first()
            self.assertIsNotNone(estimate_row)
            self.assertIsNotNone(usage_row)
            # 两张表：估算保持原值，实际用量单独落库
            self.assertEqual(int(estimate_row.estimated_cost_micro), 5_000_000)
            self.assertEqual(int(usage_row.cost_micro), 40_000)
        finally:
            db.close()

    def test_finalize_job_marks_failed_usage_queryable(self) -> None:
        project_id = self.project_id
        job_key = f"project:{project_id}:storyboard"
        _set_price(CAPABILITY_IMAGE, "ark-seedream", 25_000)
        usage_service.record_metadata(
            UsageMetadata(capability=CAPABILITY_IMAGE, provider="ark-seedream", images=3),
            scope=usage_service.UsageScope(project_id=project_id, job_key=job_key, job_type="storyboard"),
        )
        updated = usage_service.finalize_job(job_key, "failed", job_id="job-123")
        self.assertEqual(updated, 1)

        db = SessionLocal()
        try:
            summary = usage_service.summarize(db, job_key=job_key)
            rows = usage_service.list_records(db, job_key=job_key)["items"]
        finally:
            db.close()
        self.assertEqual(summary["cost_micro"], 75_000)
        self.assertEqual(rows[0]["job_status"], "failed")
        self.assertEqual(rows[0]["job_id"], "job-123")

    def test_group_usage_by_shot_and_job_type(self) -> None:
        project_id = self.project_id
        shot_id = _shot(project_id)
        _set_price(CAPABILITY_VIDEO, "ark-seedance", 100_000)
        usage_service.record_metadata(
            UsageMetadata(capability=CAPABILITY_VIDEO, provider="ark-seedance", seconds=4),
            scope=usage_service.UsageScope(project_id=project_id, shot_id=shot_id, job_type="shot_video"),
        )
        db = SessionLocal()
        try:
            by_shot = usage_service.group_usage(db, "shot", project_id=project_id)
            by_type = usage_service.group_usage(db, "job_type", project_id=project_id)
        finally:
            db.close()
        self.assertTrue(any(item["key"] == shot_id and item["cost_micro"] == 400_000 for item in by_shot))
        self.assertTrue(any(item["key"] == "shot_video" for item in by_type))

    def test_usage_record_payload_has_no_secrets_or_raw_response(self) -> None:
        record = usage_service.record_metadata(
            UsageMetadata(capability=CAPABILITY_IMAGE, provider="ark-seedream", images=1),
            dedupe_key=f"payload-{uuid.uuid4().hex}",
        )
        serialized = json.dumps(record, ensure_ascii=False)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("choices", serialized)
        self.assertNotIn("Authorization", serialized)
        self.assertNotIn("/Users/", serialized)


if __name__ == "__main__":
    unittest.main()
