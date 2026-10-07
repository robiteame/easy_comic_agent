"""任务启动 Provider 预检的验收测试。

覆盖：

- script 端点：主端点缺 Key 拦截、备端点（不同服务地址）可兜底、同址备端点不算可用；
- video 端点：缺 Key 拦截（视频无本地回退）；
- voice 端点：默认必配；视频模型具备原生对白语音能力（native_audio 且已接入厂商
  实现）时不强制要求 TTS；镜头无台词 / 镜头级显式 native 时不强制；
  镜头级显式 tts / 纯配音任务（shot_audio）仍必须配置语音端点；
- script_pipeline 手动模式只查 LLM，auto 模式查全链路；
- 路由层：缺失配置转成 HTTP 409（error_code=provider_not_configured），任务不启动。
"""

from __future__ import annotations

import asyncio
import unittest
import uuid
from unittest.mock import patch

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.provider_guard import ensure_providers_ready  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from main import app  # noqa: E402
from models import Project, Shot  # noqa: E402
from services import audio_routing, provider_readiness  # noqa: E402
from services.job_types import parse_job_key  # noqa: E402
from services.providers.endpoint import EndpointConfig  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


def _endpoint(capability: str, *, api_key: str = "", protocol: str = "") -> EndpointConfig:
    defaults = {
        "script": "openai-chat",
        "script_fallback": "openai-chat",
        "image": "placeholder",
        "video": "ark-seedance",
        "voice": "mimo-tts",
    }
    return EndpointConfig(
        protocol=protocol or defaults.get(capability, ""),
        base_url=f"https://{capability.replace('_', '-')}.example.test",
        api_key=api_key,
        model="model-x",
    )


def _native_adapter_class():
    from services.providers.base import VideoCapabilities

    return type(
        "_ReadyNativeAdapter",
        (),
        {
            "capabilities": VideoCapabilities(reference_image=False, native_audio=True, dialogue_in_prompt=True),
            "production_ready": True,
        },
    )


def _patch_endpoints(**keys: str):
    """按能力名指定 api_key 的端点配置补丁（未列出的能力视为未配置密钥）。"""

    def fake_get_endpoint(capability: str) -> EndpointConfig:
        return _endpoint(capability, api_key=keys.get(capability, ""))

    return patch.object(provider_readiness, "get_endpoint", side_effect=fake_get_endpoint)


class ScriptPipelineReadinessTests(unittest.TestCase):
    def test_manual_mode_requires_llm_key(self) -> None:
        with _patch_endpoints():
            missing = provider_readiness.missing_providers("script_pipeline", mode="manual")
        self.assertEqual([item["capability"] for item in missing], ["script"])

    def test_primary_key_passes_manual_mode(self) -> None:
        with _patch_endpoints(script="key-a"):
            self.assertEqual(provider_readiness.missing_providers("script_pipeline", mode="manual"), [])

    def test_fallback_key_on_different_host_passes(self) -> None:
        with _patch_endpoints(script_fallback="key-b"):
            self.assertEqual(provider_readiness.missing_providers("script_pipeline", mode="manual"), [])

    def test_fallback_key_on_same_host_is_ignored(self) -> None:
        def fake_get_endpoint(capability: str) -> EndpointConfig:
            endpoint = _endpoint(capability, api_key="key-b" if capability == "script_fallback" else "")
            if capability == "script_fallback":
                endpoint.base_url = _endpoint("script").base_url
            return endpoint

        with patch.object(provider_readiness, "get_endpoint", side_effect=fake_get_endpoint):
            missing = provider_readiness.missing_providers("script_pipeline", mode="manual")
        self.assertEqual([item["capability"] for item in missing], ["script"])

    def test_auto_mode_also_requires_video_and_voice(self) -> None:
        with _patch_endpoints(script="key-a"):
            missing = provider_readiness.missing_providers("script_pipeline", mode="auto")
        self.assertEqual([item["capability"] for item in missing], ["video", "voice"])

    def test_auto_mode_with_all_keys_passes(self) -> None:
        with _patch_endpoints(script="a", video="b", voice="c"):
            self.assertEqual(provider_readiness.missing_providers("script_pipeline", mode="auto"), [])


class ShotVideoReadinessTests(unittest.TestCase):
    def test_video_key_required(self) -> None:
        with _patch_endpoints(video="", voice="c"):
            missing = provider_readiness.missing_providers("shot_video", has_dialogue=True)
        self.assertEqual([item["capability"] for item in missing], ["video"])

    def test_voice_required_for_silent_video_with_dialogue(self) -> None:
        with _patch_endpoints(video="b", voice=""):
            missing = provider_readiness.missing_providers("shot_video", has_dialogue=True)
        self.assertEqual([item["capability"] for item in missing], ["voice"])

    def test_voice_not_required_without_dialogue(self) -> None:
        with _patch_endpoints(video="b", voice=""):
            self.assertEqual(
                provider_readiness.missing_providers("shot_video", has_dialogue=False),
                [],
            )

    def test_voice_not_required_when_video_supports_native_audio(self) -> None:
        """视频模型支持原生对白语音时不强制要求 TTS。"""

        with (
            _patch_endpoints(video="b", voice=""),
            patch.object(audio_routing, "get_adapter", return_value=_native_adapter_class()),
        ):
            self.assertTrue(provider_readiness.native_video_audio_ready())
            self.assertEqual(
                provider_readiness.missing_providers("shot_video", has_dialogue=True),
                [],
            )

    def test_shot_level_native_override_skips_voice(self) -> None:
        with _patch_endpoints(video="b", voice=""):
            self.assertEqual(
                provider_readiness.missing_providers("shot_video", has_dialogue=True, audio_mode_override="native"),
                [],
            )

    def test_shot_level_explicit_tts_still_requires_voice(self) -> None:
        with (
            _patch_endpoints(video="b", voice=""),
            patch.object(audio_routing, "get_adapter", return_value=_native_adapter_class()),
        ):
            missing = provider_readiness.missing_providers("shot_video", has_dialogue=True, audio_mode_override="tts")
        self.assertEqual([item["capability"] for item in missing], ["voice"])

    def test_voice_configured_passes(self) -> None:
        with _patch_endpoints(video="b", voice="c"):
            self.assertEqual(
                provider_readiness.missing_providers("shot_video", has_dialogue=True),
                [],
            )


class ShotAudioReadinessTests(unittest.TestCase):
    def test_voice_required_even_when_video_supports_native_audio(self) -> None:
        """纯配音任务本身就是 TTS 调用，原生音频能力替代不了它。"""

        with (
            _patch_endpoints(voice=""),
            patch.object(audio_routing, "get_adapter", return_value=_native_adapter_class()),
        ):
            missing = provider_readiness.missing_providers("shot_audio")
        self.assertEqual([item["capability"] for item in missing], ["voice"])

    def test_voice_configured_passes(self) -> None:
        with _patch_endpoints(voice="c"):
            self.assertEqual(provider_readiness.missing_providers("shot_audio"), [])


class GuardContractTests(unittest.TestCase):
    def test_unknown_job_type_rejected(self) -> None:
        with self.assertRaises(ValueError):
            provider_readiness.missing_providers("render")

    def test_ensure_raises_with_message_and_missing_list(self) -> None:
        with _patch_endpoints():
            with self.assertRaises(provider_readiness.ProviderNotConfiguredError) as ctx:
                provider_readiness.ensure_task_providers_ready("script_pipeline", mode="manual")
        self.assertIn("剧本解析（LLM）", ctx.exception.message)
        self.assertEqual(ctx.exception.missing[0]["capability"], "script")

    def test_route_guard_raises_structured_409(self) -> None:
        with _patch_endpoints(video="", voice=""):
            with self.assertRaises(HTTPException) as ctx:
                ensure_providers_ready("shot_video", has_dialogue=True)
        self.assertEqual(ctx.exception.status_code, 409)
        detail = ctx.exception.detail
        self.assertEqual(detail["error_code"], "provider_not_configured")
        self.assertEqual(detail["status"], "provider_not_configured")
        self.assertEqual(
            [item["capability"] for item in detail["missing"]],
            ["video", "voice"],
        )


class _RouteTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()
        cls.client = TestClient(app)

    def setUp(self) -> None:
        db = SessionLocal()
        try:
            self.project_id = f"proj-{uuid.uuid4().hex[:10]}"
            self.shot_id = f"shot-{uuid.uuid4().hex[:10]}"
            db.add(Project(id=self.project_id, title="预检测试项目"))
            db.add(
                Shot(
                    id=self.shot_id,
                    project_id=self.project_id,
                    sequence=1,
                    dialogue="这是一句台词",
                    status="storyboard_approved",
                    storyboard_path="story.png",
                )
            )
            db.commit()
        finally:
            db.close()

    def tearDown(self) -> None:
        db = SessionLocal()
        try:
            db.query(Shot).filter(Shot.id == self.shot_id).delete()
            db.query(Project).filter(Project.id == self.project_id).delete()
            db.commit()
        finally:
            db.close()


class ScriptParseRouteTests(_RouteTestCase):
    def test_parse_rejects_with_409_when_llm_unconfigured(self) -> None:
        with _patch_endpoints():
            response = self.client.post(
                "/api/script/parse",
                json={
                    "project_id": self.project_id,
                    "user_input": "第一场：小雨在教室里发现了一本旧日记。",
                },
            )
        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertEqual(detail["error_code"], "provider_not_configured")
        self.assertIn("剧本解析（LLM）", detail["message"])

        db = SessionLocal()
        try:
            from models import BackgroundJob

            jobs = (
                db.query(BackgroundJob)
                .filter(BackgroundJob.idempotency_key == f"project:{self.project_id}:pipeline:manual")
                .count()
            )
            self.assertEqual(jobs, 0, "预检失败时不得创建后台任务")
        finally:
            db.close()


class ShotAudioRouteTests(_RouteTestCase):
    def test_generate_audio_rejects_with_409_when_voice_unconfigured(self) -> None:
        with _patch_endpoints(voice=""):
            response = self.client.post(
                f"/api/shot/{self.shot_id}/generate-audio",
                json={"project_id": self.project_id, "force": True},
            )
        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertEqual(detail["error_code"], "provider_not_configured")
        self.assertIn("配音", detail["message"])


class JobRetryReadinessTests(unittest.TestCase):
    """任务中心重试 / 续跑同样走 Provider 预检：缺配置时拒绝重新派发。"""

    def setUp(self) -> None:
        init_db()
        self.db = SessionLocal()

    def tearDown(self) -> None:
        self.db.close()

    def _job(self, key: str, scope: str, status: str = "failed"):
        from datetime import datetime

        from models import BackgroundJob

        identity = parse_job_key(key)
        job = BackgroundJob(
            id=f"job-{uuid.uuid4().hex[:8]}",
            idempotency_key=key,
            scope=scope,
            status=status,
            job_type=identity.job_type,
            display_name="预检重试任务",
            attempt=1,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self.db.add(job)
        self.db.commit()
        return job

    def test_retry_shot_video_blocked_without_voice_config(self) -> None:
        from services import job_actions

        project_id = f"proj-{uuid.uuid4().hex[:10]}"
        shot_id = f"shot-{uuid.uuid4().hex[:10]}"
        self.db.add(Project(id=project_id, title="重试预检项目"))
        self.db.add(Shot(id=shot_id, project_id=project_id, sequence=1, dialogue="台词"))
        self.db.commit()
        job = self._job(f"shot:{shot_id}:video", f"shot:{shot_id}")

        with _patch_endpoints(video="k", voice=""):
            outcome = asyncio.run(job_actions.retry_job(self.db, job))

        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error_code, "provider_not_configured")
        self.assertIn("配音", outcome.message)

    def test_retry_pipeline_blocked_without_llm_config(self) -> None:
        from services import job_actions

        project_id = f"proj-{uuid.uuid4().hex[:10]}"
        self.db.add(Project(id=project_id, title="重试预检项目"))
        self.db.commit()
        job = self._job(f"project:{project_id}:pipeline:manual", f"project:{project_id}")

        with _patch_endpoints():
            outcome = asyncio.run(job_actions.retry_job(self.db, job))

        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error_code, "provider_not_configured")
        self.assertIn("剧本解析", outcome.message)


if __name__ == "__main__":
    unittest.main()
