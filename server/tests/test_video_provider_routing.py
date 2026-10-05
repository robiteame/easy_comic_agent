"""视频 Provider 模型级能力判断与逐镜头路由的验收测试。

覆盖三种模型配置的路由差异：
- Seedance（ark-seedance）：first_frame_only，只发送已审核故事板首帧，
  绝不声称角色图/场景图/尾帧已发送；
- Wan i2v（wan*-i2v-*）：同样 first_frame_only，路由到 first_frame_i2v；
- Wan r2v（wan*-r2v-*）：multi_reference，路由到 multi_reference_r2v，
  按 首帧 → 角色身份 → 场景基准 → 连续性 的优先级发送并遵守数量上限。

同时验收：
- 所有路由判断以 effective_capabilities(model) 为准（同协议不同模型结论不同）；
- 镜头级 required_capabilities / video_mode 生成；
- 显式 provider_override 优先于自动选择；
- reference_manifest 区分 candidate / validated / sent / not_sent_reason。
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from PIL import Image  # noqa: E402

from agent.decision import provider_profiles  # noqa: E402
from services.providers.base import ReferenceAsset, VideoRequest  # noqa: E402
from services.providers.capability_matrix import (  # noqa: E402
    VIDEO_MODE_FIRST_FRAME_I2V,
    VIDEO_MODE_FIRST_LAST_FRAME,
    VIDEO_MODE_MULTI_REFERENCE_R2V,
    select_video_mode,
)
from services.providers.endpoint import EndpointConfig  # noqa: E402
from services.providers.video_ark_seedance import ArkSeedanceVideoAdapter  # noqa: E402
from services.providers.video_dashscope_wanx import DashscopeWanxVideoAdapter  # noqa: E402
from services.video_service import VideoService  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402

SEEDANCE_ENDPOINT = EndpointConfig(
    protocol="ark-seedance",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="ark-test",
    model="doubao-seedance-1-5-pro-251215",
)
WAN_I2V_ENDPOINT = EndpointConfig(
    protocol="dashscope-wanx",
    base_url="https://dashscope.aliyuncs.com/api/v1",
    api_key="sk-test",
    model="wan2.6-i2v-flash",
)
WAN_R2V_ENDPOINT = EndpointConfig(
    protocol="dashscope-wanx",
    base_url="https://dashscope.aliyuncs.com/api/v1",
    api_key="sk-test",
    model="wan2.7-r2v-2026-06-12",
)


def _png(name: str, color: tuple[int, int, int] = (60, 80, 120)) -> str:
    path = TEST_ROOT / "output" / "video_routing_tests" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 96), color).save(path)
    return str(path)


def _shot(**overrides) -> dict:
    base = {
        "shot_id": "routing_0001",
        "storyboard_path": _png("storyboard"),
        "image_path": "",
        "duration": 4.5,
        "output_format": "9:16",
        "resolution": "720p",
        "style": "realistic",
        "scene_description": "雨夜天台",
        "character_action": "她转身",
        "emotion": "sad",
        "camera_movement": "跟随",
        "camera_angle": "正面",
        "shot_type": "medium",
        "reference_assets": [],
        "continuity_profile": {},
    }
    base.update(overrides)
    return base


def _continuity_shot(**overrides) -> dict:
    last_frame = _png("prev_last_frame", (10, 10, 10))
    return _shot(
        continuity_reference_path=last_frame,
        continuity_profile={
            "continuity_mode": "continuous_action",
            "continuity_reference_used": True,
            "continuity_reference_path": last_frame,
        },
        **overrides,
    )


class _FakeVideoResult(SimpleNamespace):
    pass


def _fake_adapter_generate(_self, request):  # noqa: ANN001
    async def run():
        return _FakeVideoResult(
            video_path=str(TEST_ROOT / "v.mp4"),
            frame_path=str(TEST_ROOT / "f.png"),
            native_audio=False,
            payload_mode="first_frame_reference",
            task_id="task-routing",
        )

    return run()


class ModelLevelCapabilityTests(unittest.TestCase):
    """路由判断必须用 effective_capabilities(model)，同协议不同模型结论不同。"""

    def test_wan_i2v_model_is_first_frame_only(self) -> None:
        caps = DashscopeWanxVideoAdapter.effective_capabilities(WAN_I2V_ENDPOINT.model)
        self.assertEqual(caps.reference_mode, "first_frame_only")
        self.assertFalse(caps.multiple_reference_images)
        self.assertTrue(caps.reference_image)

    def test_wan_r2v_model_is_multi_reference(self) -> None:
        caps = DashscopeWanxVideoAdapter.effective_capabilities(WAN_R2V_ENDPOINT.model)
        self.assertEqual(caps.reference_mode, "multi_reference")
        self.assertTrue(caps.multiple_reference_images)
        self.assertEqual(caps.max_reference_images, 4)

    def test_seedance_is_first_frame_only_for_any_model(self) -> None:
        caps = ArkSeedanceVideoAdapter.effective_capabilities(SEEDANCE_ENDPOINT.model)
        self.assertEqual(caps.reference_mode, "first_frame_only")
        self.assertFalse(caps.multiple_reference_images)


class ProviderProfilesModelCapabilityTests(unittest.TestCase):
    """provider_profiles 不得用适配器默认能力抹平模型差异。"""

    def _profiles(self, endpoint: EndpointConfig) -> dict[str, object]:
        with (
            patch("services.providers.endpoint.get_endpoint", return_value=endpoint),
            patch("services.providers.endpoint.video_protocol_defaults", return_value=endpoint),
        ):
            return {item.provider: item for item in provider_profiles("video", reference_required=True)}

    def test_r2v_model_reports_multi_reference_support(self) -> None:
        profile = self._profiles(WAN_R2V_ENDPOINT)["dashscope-wanx"]
        self.assertEqual(profile.model, WAN_R2V_ENDPOINT.model)
        self.assertTrue(profile.available)
        self.assertTrue(profile.supports_reference_images)
        self.assertTrue(profile.supports_reference_image)

    def test_i2v_model_reports_single_first_frame_only(self) -> None:
        profile = self._profiles(WAN_I2V_ENDPOINT)["dashscope-wanx"]
        self.assertEqual(profile.model, WAN_I2V_ENDPOINT.model)
        self.assertFalse(profile.supports_reference_images)
        self.assertTrue(profile.supports_reference_image)


class VideoModeSelectionTests(unittest.TestCase):
    """select_video_mode 按镜头要求 + 模型生效能力选择路由模式。"""

    def test_first_last_frame_requires_provider_declaration(self) -> None:
        caps = SimpleNamespace(
            last_frame_input=True,
            first_last_frame_interpolation=True,
            multiple_reference_images=False,
        )
        mode = select_video_mode(["first_frame", "first_last_frame_interpolation"], caps)
        self.assertEqual(mode, VIDEO_MODE_FIRST_LAST_FRAME)

    def test_first_last_frame_falls_back_when_model_lacks_input(self) -> None:
        caps = SimpleNamespace(
            last_frame_input=False,
            first_last_frame_interpolation=False,
            multiple_reference_images=True,
        )
        mode = select_video_mode(["first_frame", "multiple_reference_images", "first_last_frame_interpolation"], caps)
        self.assertEqual(mode, VIDEO_MODE_MULTI_REFERENCE_R2V)

    def test_default_is_first_frame_i2v(self) -> None:
        caps = SimpleNamespace(
            last_frame_input=False,
            first_last_frame_interpolation=False,
            multiple_reference_images=False,
        )
        self.assertEqual(select_video_mode(["first_frame"], caps), VIDEO_MODE_FIRST_FRAME_I2V)


class _RoutingTestCase(unittest.TestCase):
    def _run_shot(
        self,
        endpoint: EndpointConfig,
        adapter_cls,  # noqa: ANN001
        shot: dict,
        *,
        capture: dict | None = None,
        real_single_shot: bool = False,
        **kwargs,  # noqa: ANN003
    ) -> dict:
        service = VideoService()
        with (
            patch("services.video_service.get_endpoint", return_value=endpoint),
            patch("services.video_service.get_adapter", return_value=adapter_cls),
            patch("services.video_service.video_protocol_defaults", return_value=endpoint),
        ):
            if not real_single_shot:

                async def fake_single(prompt, **single_kwargs):  # noqa: ANN001
                    if capture is not None:
                        capture.update(single_kwargs)
                    return {
                        "video_path": "/tmp/v.mp4",
                        "frame_path": "/tmp/f.png",
                        "task_id": "task-routing",
                        "reference_payload_mode": "first_frame_reference",
                        "native_audio": False,
                    }

                with patch.object(VideoService, "generate_single_shot", side_effect=fake_single):
                    return asyncio.run(service.generate_shot_video(shot, [], {}, "routing_tests", **kwargs))
            with patch.object(adapter_cls, "generate", _fake_adapter_generate):
                return asyncio.run(service.generate_shot_video(shot, [], {}, "routing_tests", **kwargs))


class FirstFrameOnlyRoutingTests(_RoutingTestCase):
    """Seedance / Wan i2v：只发送首帧，manifest 不虚报其余参考。"""

    def test_seedance_auto_mode_blocks_silent_downgrade(self) -> None:
        shot = _continuity_shot(
            character_reference_images=[_png("char_seedance", (200, 30, 30))],
            scene_reference_images=[_png("scene_seedance", (30, 200, 30))],
        )
        from services.providers.capability_matrix import CapabilityDowngradeRequiredError

        with self.assertRaises(CapabilityDowngradeRequiredError):
            self._run_shot(SEEDANCE_ENDPOINT, ArkSeedanceVideoAdapter, shot, capability_mode="auto")

    def test_seedance_confirmed_downgrade_sends_first_frame_only(self) -> None:
        shot = _continuity_shot(
            character_reference_images=[_png("char_seedance", (200, 30, 30))],
            scene_reference_images=[_png("scene_seedance", (30, 200, 30))],
        )
        capture: dict = {}
        result = self._run_shot(
            SEEDANCE_ENDPOINT,
            ArkSeedanceVideoAdapter,
            shot,
            confirm_capability_downgrade=True,
            capture=capture,
        )

        self.assertEqual(shot["video_mode"], VIDEO_MODE_FIRST_FRAME_I2V)
        self.assertEqual(shot["provider_source"], "configured")
        self.assertEqual(
            shot["required_capabilities"],
            ["first_frame", "character_identity", "scene_reference", "multiple_reference_images"],
        )
        # 只有首帧进入请求：角色图、场景图、上一镜尾帧都不得发送。
        sent_types = [item.type for item in capture["reference_assets"]]
        self.assertEqual(sent_types, ["approved_storyboard_first_frame"])
        self.assertEqual(shot["references_sent"], ["approved_storyboard_first_frame"])
        manifest = {item["type"]: item for item in result["reference_manifest"]}
        self.assertTrue(manifest["approved_storyboard_first_frame"]["sent"])
        self.assertEqual(manifest["approved_storyboard_first_frame"]["not_sent_reason"], "")
        for kind in ("character_three_view", "scene_baseline"):
            self.assertFalse(manifest[kind]["sent"])
            self.assertTrue(manifest[kind]["validated"])
            self.assertTrue(manifest[kind]["candidate"])
            self.assertEqual(manifest[kind]["not_sent_reason"], "provider_first_frame_only")
        continuity = manifest["continuity_reference"]
        self.assertFalse(continuity["sent"])
        self.assertEqual(continuity["not_sent_reason"], "provider_first_frame_only")

    def test_wan_i2v_model_routes_first_frame_i2v(self) -> None:
        shot = _shot()  # 无额外参考，不需要降级确认
        capture: dict = {}
        self._run_shot(WAN_I2V_ENDPOINT, DashscopeWanxVideoAdapter, shot, capture=capture)

        self.assertEqual(shot["video_mode"], VIDEO_MODE_FIRST_FRAME_I2V)
        self.assertEqual(shot["required_capabilities"], ["first_frame"])
        self.assertEqual(
            [item.type for item in capture["reference_assets"]],
            ["approved_storyboard_first_frame"],
        )

    def test_seedance_full_run_claims_no_unsent_control_types(self) -> None:
        """真实 generate_single_shot 元数据：control_types 只含 first_frame。"""

        shot = _continuity_shot(
            character_reference_images=[_png("char_claim", (200, 30, 30))],
        )
        self._run_shot(
            SEEDANCE_ENDPOINT,
            ArkSeedanceVideoAdapter,
            shot,
            confirm_capability_downgrade=True,
            real_single_shot=True,
        )

        report = shot["consistency_metrics"].get("provider_capabilities", {})
        self.assertEqual(shot["control_types_sent"], ["first_frame"])
        self.assertNotIn("multiple_reference_images", report.get("supported", []))


class MultiReferenceRoutingTests(_RoutingTestCase):
    """Wan r2v：multi_reference_r2v 路由 + 优先级发送 + 数量上限。"""

    def test_r2v_routes_multi_reference_and_sends_by_priority(self) -> None:
        scene = _png("r2v_scene", (30, 200, 30))
        continuity = _png("r2v_continuity", (10, 10, 10))
        character = _png("r2v_char", (200, 30, 30))
        # 发现顺序（scene/continuity 在 shot.reference_assets，character 在
        # character_reference_images）与发送优先级不同，用于验证排序。
        shot = _shot(
            reference_assets=[
                {"type": "scene_baseline", "path": scene, "role": "env"},
                {"type": "continuity_frame", "path": continuity, "role": "motion"},
            ],
            character_reference_images=[character],
        )
        capture: dict = {}
        result = self._run_shot(WAN_R2V_ENDPOINT, DashscopeWanxVideoAdapter, shot, capture=capture)

        self.assertEqual(shot["video_mode"], VIDEO_MODE_MULTI_REFERENCE_R2V)
        self.assertEqual(shot["reference_mode"], "multi_reference")
        self.assertIn("multiple_reference_images", shot["required_capabilities"])
        self.assertEqual(
            [item.type for item in capture["reference_assets"]],
            ["approved_storyboard_first_frame", "character_three_view", "scene_baseline", "continuity_frame"],
        )
        manifest = {item["type"]: item for item in result["reference_manifest"]}
        for kind in ("approved_storyboard_first_frame", "character_three_view", "scene_baseline", "continuity_frame"):
            self.assertTrue(manifest[kind]["sent"], kind)
            self.assertEqual(manifest[kind]["not_sent_reason"], "")
        self.assertEqual(
            shot["references_sent"],
            ["approved_storyboard_first_frame", "character_three_view", "scene_baseline", "continuity_frame"],
        )

    def test_r2v_respects_reference_count_limit(self) -> None:
        characters = [_png(f"r2v_char_{index}", (200, 30 * index % 256, 30)) for index in range(5)]
        scene = _png("r2v_scene_over", (30, 200, 30))
        shot = _shot(
            character_reference_images=characters,
            scene_reference_images=[scene],
        )
        capture: dict = {}
        result = self._run_shot(WAN_R2V_ENDPOINT, DashscopeWanxVideoAdapter, shot, capture=capture)

        # 上限 4：首帧 + 4 张角色图；第 5 张角色图与场景图记 reference_count_limit。
        sent = [item.type for item in capture["reference_assets"]]
        self.assertEqual(sent.count("character_three_view"), 4)
        self.assertEqual(len(sent), 5)
        manifest = [item for item in result["reference_manifest"] if item["type"] != "continuity_reference"]
        dropped = [item for item in manifest if not item["sent"]]
        self.assertEqual(len(dropped), 2)
        self.assertTrue(all(item["not_sent_reason"] == "reference_count_limit" for item in dropped))
        self.assertTrue(all(item["validated"] and item["candidate"] for item in dropped))
        self.assertEqual(shot["references_validated"], 7)

    def test_r2v_full_run_reports_reference_control_types(self) -> None:
        shot = _shot(character_reference_images=[_png("r2v_char_full", (200, 30, 30))])
        self._run_shot(WAN_R2V_ENDPOINT, DashscopeWanxVideoAdapter, shot, real_single_shot=True)

        self.assertEqual(shot["video_mode"], VIDEO_MODE_MULTI_REFERENCE_R2V)
        self.assertIn("first_frame", shot["control_types_sent"])
        self.assertIn("reference_images", shot["control_types_sent"])


class ProviderOverridePriorityTests(_RoutingTestCase):
    """显式 provider_override 优先于自动选择。"""

    def test_override_selects_override_endpoint_even_when_global_differs(self) -> None:
        # 全局端点是 Seedance（first_frame_only），显式 override 到 Wan r2v：
        # 路由必须按 override 端点的模型能力裁决，而不是全局配置。
        shot = _shot(character_reference_images=[_png("override_char", (200, 30, 30))])
        adapter_calls: list[str] = []

        def fake_get_adapter(capability, protocol):  # noqa: ANN001
            adapter_calls.append(protocol)
            return DashscopeWanxVideoAdapter

        service = VideoService()

        async def fake_single(prompt, **kwargs):  # noqa: ANN001
            return {
                "video_path": "/tmp/v.mp4",
                "frame_path": "/tmp/f.png",
                "task_id": "task-override",
                "reference_payload_mode": "first_frame_reference",
                "native_audio": False,
            }

        with (
            patch("services.video_service.get_endpoint", return_value=SEEDANCE_ENDPOINT),
            patch("services.video_service.get_adapter", side_effect=fake_get_adapter),
            patch("services.video_service.video_protocol_defaults", return_value=WAN_R2V_ENDPOINT),
            patch.object(VideoService, "generate_single_shot", side_effect=fake_single),
        ):
            asyncio.run(service.generate_shot_video(shot, [], {}, "routing_tests", provider_override="dashscope-wanx"))

        self.assertEqual(adapter_calls, ["dashscope-wanx"])
        self.assertEqual(shot["video_mode"], VIDEO_MODE_MULTI_REFERENCE_R2V)
        self.assertEqual(shot["provider_source"], "agent_selected")

    def test_override_never_silently_switches_back_to_capable_provider(self) -> None:
        # 显式 override 到 first_frame_only Provider 且镜头要求多参考：自动模式下
        # 必须报能力降级错误，不允许悄悄换回支持多参考的 Provider。
        from services.providers.capability_matrix import CapabilityDowngradeRequiredError

        shot = _shot(character_reference_images=[_png("override_char2", (200, 30, 30))])

        with self.assertRaises(CapabilityDowngradeRequiredError):
            self._run_shot(
                WAN_I2V_ENDPOINT,
                DashscopeWanxVideoAdapter,
                shot,
                capability_mode="auto",
                provider_override="dashscope-wanx",
            )


class ManifestValidationTests(_RoutingTestCase):
    """reference_manifest 区分 candidate / validated / sent / not_sent_reason。"""

    def test_unreadable_candidate_is_recorded_not_dropped(self) -> None:
        shot = _shot(
            scene_reference_images=[str(TEST_ROOT / "output" / "video_routing_tests" / "missing.png")],
        )
        result = self._run_shot(SEEDANCE_ENDPOINT, ArkSeedanceVideoAdapter, shot)

        manifest = {item["type"]: item for item in result["reference_manifest"]}
        scene = manifest["scene_baseline"]
        self.assertTrue(scene["candidate"])
        self.assertFalse(scene["validated"])
        self.assertFalse(scene["sent"])
        self.assertEqual(scene["not_sent_reason"], "reference_unreadable")
        # 未校验素材不计入 references_validated。
        self.assertEqual(shot["references_validated"], 1)

    def test_continuity_not_used_records_decision_without_sent_claim(self) -> None:
        shot = _shot()  # 无连续性要求
        result = self._run_shot(SEEDANCE_ENDPOINT, ArkSeedanceVideoAdapter, shot)

        continuity = next(item for item in result["reference_manifest"] if item["type"] == "continuity_reference")
        self.assertFalse(continuity["candidate"])
        self.assertFalse(continuity["sent"])
        self.assertNotEqual(continuity["not_sent_reason"], "provider_first_frame_only")


class ExecutionPlanRoutingTests(_RoutingTestCase):
    """视频路由把权威能力清单、video_mode 与候选/恢复预算写进统一执行计划。"""

    def test_shot_plan_records_authoritative_capabilities_and_mode(self) -> None:
        shot = _continuity_shot(
            candidate_count=3,
            recovery_budget=2,
            character_reference_images=[_png("plan_character", (90, 40, 160))],
            scene_reference_images=[_png("plan_scene", (40, 160, 90))],
        )
        self._run_shot(WAN_R2V_ENDPOINT, DashscopeWanxVideoAdapter, shot)

        payload = shot["execution_plan"]
        self.assertEqual(payload["shot_id"], "routing_0001")
        self.assertEqual(payload["video_mode"], VIDEO_MODE_MULTI_REFERENCE_R2V)
        self.assertEqual(payload["candidate_count"], 3)
        self.assertEqual(payload["recovery_budget"], 2)
        # 权威能力清单按可加载素材覆写规划期推导值。
        for capability in ("character_identity", "scene_reference", "multiple_reference_images"):
            self.assertIn(capability, payload["required_capabilities"])
        # 时长字段与固定档 Provider 一致：生成 5 秒，按计划裁剪。
        self.assertEqual(payload["provider_generation_duration_s"], 5.0)
        self.assertGreaterEqual(payload["trim_end_ms"], 4500)

    def test_first_frame_only_plan_reports_only_first_frame_capability(self) -> None:
        shot = _shot()
        self._run_shot(WAN_I2V_ENDPOINT, DashscopeWanxVideoAdapter, shot)

        payload = shot["execution_plan"]
        self.assertEqual(payload["video_mode"], VIDEO_MODE_FIRST_FRAME_I2V)
        self.assertEqual(payload["required_capabilities"], ["first_frame"])
        # 未指定候选数/恢复预算时保持可执行的默认值。
        self.assertEqual(payload["candidate_count"], 1)
        self.assertGreaterEqual(payload["recovery_budget"], 0)


class EndFrameCapabilityRoutingTests(unittest.TestCase):
    """end_frame 只有在模型明确支持首尾帧插值时才进入请求载荷。"""

    def _capabilities(self, *, supported: bool) -> SimpleNamespace:
        return SimpleNamespace(
            reference_image=True,
            multiple_reference_images=False,
            max_reference_images=0,
            reference_mode="first_frame_only",
            last_frame_input=supported,
            first_last_frame_interpolation=supported,
            fixed_duration=5,
            min_duration=5,
            max_duration=5,
            duration_step=5,
            max_reference_inline_bytes=8 * 1024 * 1024,
        )

    def _run(self, *, supported: bool) -> tuple[dict, VideoRequest]:
        captured: list[VideoRequest] = []

        def fake_generate(_self, request):  # noqa: ANN001
            captured.append(request)
            return _fake_adapter_generate(_self, request)

        caps = self._capabilities(supported=supported)
        adapter = DashscopeWanxVideoAdapter(WAN_I2V_ENDPOINT)
        first_url = adapter.reference_assets.to_image_url(_png("end_route_first", (12, 34, 56)))
        end_url = adapter.reference_assets.to_image_url(_png("end_route_last", (78, 90, 12)))
        content = [
            {"type": "text", "text": "p"},
            {"type": "image_url", "image_url": {"url": first_url}, "role": "first_frame"},
            {"type": "image_url", "image_url": {"url": end_url}, "role": "end_frame"},
        ]
        with (
            patch("services.video_service.get_endpoint", return_value=WAN_I2V_ENDPOINT),
            patch("services.video_service.get_adapter", return_value=DashscopeWanxVideoAdapter),
            patch.object(DashscopeWanxVideoAdapter, "effective_capabilities", return_value=caps),
            patch.object(DashscopeWanxVideoAdapter, "generate", fake_generate),
        ):
            result = asyncio.run(
                VideoService().generate_single_shot(
                    prompt="p",
                    project_id="routing_tests",
                    shot_id=f"end_route_{supported}",
                    content=content,
                )
            )
        return result, captured[0]

    def test_first_last_frame_routes_and_sends_end_frame(self) -> None:
        result, request = self._run(supported=True)

        self.assertEqual(result["generation_report"]["video_mode"], VIDEO_MODE_FIRST_LAST_FRAME)
        self.assertIn("end_frame", result["generation_report"]["references_sent"])
        self.assertTrue(request.end_frame)
        self.assertIn("first_last_frame_interpolation", result["generation_report"]["control_types_sent"])

    def test_unsupported_provider_never_sends_end_frame_or_claims_interpolation(self) -> None:
        result, request = self._run(supported=False)

        self.assertEqual(result["generation_report"]["video_mode"], VIDEO_MODE_FIRST_FRAME_I2V)
        self.assertNotIn("end_frame", result["generation_report"]["references_sent"])
        self.assertFalse(request.end_frame)
        self.assertNotIn("first_last_frame_interpolation", result["generation_report"]["control_types_sent"])


class WanxMediaEntryPriorityTests(unittest.TestCase):
    """适配器层面同样保证优先级与数量上限（覆盖直连调用路径）。"""

    def _request(self, assets: list[ReferenceAsset], reference_image: str) -> VideoRequest:
        return VideoRequest(
            prompt="p",
            reference_image=reference_image,
            reference_assets=assets,
            duration=5,
            ratio="9:16",
            resolution="720p",
        )

    def test_r2v_orders_media_by_priority(self) -> None:
        adapter = DashscopeWanxVideoAdapter(WAN_R2V_ENDPOINT)
        first_url = adapter.reference_assets.to_image_url(_png("media_first", (1, 2, 3)))
        # 各素材用不同颜色，避免生成相同 data URL 被适配器按重复项去重。
        urls = {
            kind: adapter.reference_assets.to_image_url(_png(f"media_{kind}", (40 + index, 50, 60)))
            for index, kind in enumerate(("continuity_frame", "scene_baseline", "character_three_view"))
        }
        request = self._request(
            [
                ReferenceAsset(url=urls["continuity_frame"], type="continuity_frame", source_path="c"),
                ReferenceAsset(url=urls["scene_baseline"], type="scene_baseline", source_path="s"),
                ReferenceAsset(url=urls["character_three_view"], type="character_three_view", source_path="h"),
            ],
            first_url,
        )

        media = adapter._media_entries(first_url, request)

        self.assertEqual(
            [entry["type"] for entry in media], ["first_frame", "reference_image", "reference_image", "reference_image"]
        )
        self.assertEqual(
            [entry["url"] for entry in media],
            [first_url, urls["character_three_view"], urls["scene_baseline"], urls["continuity_frame"]],
        )

    def test_r2v_caps_media_at_max_reference_images(self) -> None:
        adapter = DashscopeWanxVideoAdapter(WAN_R2V_ENDPOINT)
        first_url = adapter.reference_assets.to_image_url(_png("media_first_cap", (1, 2, 3)))
        assets = [
            ReferenceAsset(
                url=adapter.reference_assets.to_image_url(_png(f"media_cap_{index}", (70, 80 + index, 90))),
                type="character_three_view",
                source_path=f"h{index}",
            )
            for index in range(6)
        ]

        media = adapter._media_entries(first_url, self._request(assets, first_url))

        self.assertEqual(len(media), 5)  # 首帧 + 4 张参考，上限生效

    def test_i2v_model_only_carries_first_frame(self) -> None:
        adapter = DashscopeWanxVideoAdapter(WAN_I2V_ENDPOINT)
        first_url = adapter.reference_assets.to_image_url(_png("media_first_i2v", (1, 2, 3)))
        assets = [
            ReferenceAsset(
                url=adapter.reference_assets.to_image_url(_png("media_i2v_char", (7, 8, 9))),
                type="character_three_view",
                source_path="h",
            ),
        ]

        media = adapter._media_entries(first_url, self._request(assets, first_url))

        self.assertEqual([entry["type"] for entry in media], ["first_frame"])


class SingleShotMetadataTests(unittest.TestCase):
    """generate_single_shot：provider_source / video_mode / 模型级能力元数据。"""

    def test_override_metadata_uses_override_model_capabilities(self) -> None:
        service = VideoService()
        import base64

        storyboard = _png("single_storyboard", (60, 80, 120))
        data_url = "data:image/png;base64," + base64.b64encode(Path(storyboard).read_bytes()).decode("ascii")
        character = ReferenceAsset(
            url="data:image/png;base64,aGVsbG8=",
            type="character_three_view",
            role="identity",
            source_path="char.png",
        )
        with (
            patch("services.video_service.get_endpoint", return_value=SEEDANCE_ENDPOINT),
            patch("services.video_service.get_adapter", return_value=DashscopeWanxVideoAdapter),
            patch("services.video_service.video_protocol_defaults", return_value=WAN_R2V_ENDPOINT),
            patch.object(DashscopeWanxVideoAdapter, "generate", _fake_adapter_generate),
        ):
            result = asyncio.run(
                service.generate_single_shot(
                    prompt="p",
                    project_id="routing_tests",
                    shot_id="s1",
                    content=[
                        {"type": "text", "text": "p"},
                        {"type": "image_url", "image_url": {"url": data_url}, "role": "first_frame"},
                    ],
                    reference_assets=[character],
                    provider_override="dashscope-wanx",
                )
            )

        metadata = service.last_generation_metadata
        self.assertEqual(metadata["provider"], "dashscope-wanx")
        self.assertEqual(metadata["model"], WAN_R2V_ENDPOINT.model)
        self.assertEqual(metadata["provider_source"], "agent_selected")
        self.assertEqual(metadata["video_mode"], VIDEO_MODE_MULTI_REFERENCE_R2V)
        self.assertIn("character_three_view", metadata["references_sent"])
        self.assertIn("reference_images", metadata["control_types_sent"])
        self.assertTrue(result["native_audio"] is False)


if __name__ == "__main__":
    unittest.main()
