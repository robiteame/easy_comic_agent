"""视频生成服务：prompt/参考图/一致性策略 + 协议适配器路由。

端点来自 ``get_endpoint("video")``，协议调用委托给
``services.providers.registry.get_adapter("video", protocol)`` 注册的适配器：
- ``ark-seedance``：异步任务式无声视频（对白走独立 TTS 路径），首帧参考
  （first_frame_only），固定时长约 5 秒；
- ``native-audio``：原生音视频骨架（对白编入 prompt，音频随视频直出）。
"""

import asyncio
import logging
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from config import settings
from services import usage_service
from services.job_debug import record_api_request, record_api_result
from services.consistency_service import ConsistencyService, normalize_continuity_mode
from services.providers.base import Dialogue, ReferenceAsset, VideoRequest
from services.providers.endpoint import get_endpoint, video_protocol_defaults, video_protocol_defaults
from services.providers.registry import get_adapter
from services.providers.capability_matrix import (
    CapabilityDowngradeRequiredError,
    VIDEO_MODE_FIRST_FRAME_I2V,
    VIDEO_MODE_FIRST_LAST_FRAME,
    VIDEO_MODE_MULTI_REFERENCE_R2V,
    capability_report,
    payload_control_types,
    reference_send_priority,
    select_video_mode,
    supports_first_last_frame,
)
from services.providers.usage import (
    CAPABILITY_VIDEO,
    ERROR_CODE_PROVIDER_CALL_FAILED,
    adapter_usage_for_request,
)
from services.post_production_plan import CAMERA_MOVEMENT_PROMPTS
from services.prompt_budget import assemble_prompt, ensure_critical_fields
from services.reference_asset_service import ReferenceAssetService
from services.consistency_metrics import combine_report, payload_metrics
from services.security import safe_path, validate_identifier
from services.story_timing import (
    StoryTimingError,
    dialogue_text,
    estimate_speech_ms,
    load_shot_execution_plan,
    provider_duration_capability,
    resolve_shot_execution_plan,
)
from services.style_templates import style_prompt_params

logger = logging.getLogger(__name__)


def effective_video_capabilities(adapter_cls, model: str = ""):
    """模型级生效能力：一律经 ``effective_capabilities(model)`` 裁决。"""

    factory = getattr(adapter_cls, "effective_capabilities", None)
    if callable(factory):
        return factory(model)
    return getattr(adapter_cls, "capabilities", None)


def get_video_generation_duration_s(requested_duration_s: float) -> float:
    """Resolve the exact clip duration the active provider will generate."""

    capability = provider_duration_capability()
    return float(capability.fixed_duration or requested_duration_s)


class VideoService:
    """按端点协议路由的视频生成服务（保留原 SeedanceVideoService 的策略逻辑）。"""

    def __init__(self):
        self.output_dir = settings.OUTPUT_DIR / "projects"
        self.consistency = ConsistencyService()
        self.reference_assets = ReferenceAssetService()
        self.last_prompt_trimmed_fields: list[str] = []
        self.last_prompt_readded_fields: list[str] = []
        self.last_generation_metadata: dict[str, object] = {}

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------

    async def generate_single_shot(
        self,
        prompt: str,
        project_id: str = "api_diagnostics",
        shot_id: str = "seedance_check",
        duration: int = 5,
        ratio: str = "9:16",
        resolution: str = "720p",
        content: list[dict] | None = None,
        dialogues: list[Dialogue] | None = None,
        reference_assets: list[ReferenceAsset] | None = None,
        provider_override: str = "",
        seed: int | None = None,
    ) -> dict[str, str]:
        endpoint = video_protocol_defaults(provider_override) if provider_override else get_endpoint("video")
        adapter_cls = get_adapter("video", endpoint.protocol)
        adapter = adapter_cls(endpoint)
        # 显式 provider_override 优先于任何自动选择：端点一旦由调用方指定，
        # 能力判断、路由与降级门禁都只针对该端点的模型生效能力裁决。
        provider_source = "agent_selected" if provider_override else "configured"
        if not prompt.strip():
            raise RuntimeError("视频生成提示词为空")

        if content is None:
            content = [{"type": "text", "text": prompt}]
        reference_image, content_reference_assets, content_payload_mode = self._reference_from_content(content)
        request_reference_assets = list(reference_assets or content_reference_assets)

        try:
            safe_project_id = validate_identifier(project_id, "项目 ID")
            safe_shot_id = validate_identifier(shot_id, "镜头 ID")
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        output_dir = safe_path(self.output_dir, safe_project_id, "seedance", create_parent=True)
        video_path = output_dir / f"{safe_shot_id}.mp4"
        frame_path = output_dir / f"{safe_shot_id}_frame.png"

        effective_caps = effective_video_capabilities(adapter_cls, endpoint.model)
        supports_end_frame = supports_first_last_frame(effective_caps)
        request_reference_assets = [
            item for item in request_reference_assets
            if item.type != "end_frame" or supports_end_frame
        ]
        if any(item.type == "end_frame" for item in request_reference_assets) and not reference_image:
            raise RuntimeError("首尾帧视频请求缺少已审核分镜首帧，已阻止只发送 end_frame")
        duration_capability = provider_duration_capability(
            endpoint.protocol,
            capabilities=effective_caps,
            model=endpoint.model,
        )
        requested_duration = float(duration or duration_capability.max_duration)
        try:
            duration_capability.validate(requested_duration, shot_id=shot_id)
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc
        generation_duration = float(duration_capability.fixed_duration or requested_duration)
        if "actual generated clip duration:" not in prompt:
            prompt = (
                f"{prompt.rstrip()}\n"
                f"actual generated clip duration: {generation_duration:g} seconds; "
                "complete the full action and all dialogue within this exact duration"
            )
        end_frame_asset = next((item for item in request_reference_assets if item.type == "end_frame"), None)
        request = VideoRequest(
            prompt=prompt,
            reference_image=reference_image,
            end_frame=(end_frame_asset.url or end_frame_asset.source_path) if end_frame_asset else None,
            reference_assets=request_reference_assets,
            dialogues=dialogues,
            duration=int(round(generation_duration)),
            ratio=ratio,
            resolution=resolution,
            project_id=safe_project_id,
            output_video_path=video_path,
            output_frame_path=frame_path,
            seed=int(seed) if seed is not None else None,
        )
        report = capability_report("video", endpoint.protocol, model=endpoint.model, adapter_cls=adapter_cls)
        sent_items: list[dict] = []
        if reference_image:
            sent_items.append({"type": "approved_storyboard_first_frame", "role": "first_frame", "parameter": "content[].role"})
        if getattr(effective_caps, "multiple_reference_images", False):
            sent_items.extend(
                {
                    "type": item.type,
                    "role": item.role or item.type,
                    "parameter": getattr(effective_caps, "reference_role_parameter", "") or "content[].role",
                    "weight_policy": report.get("reference_weight_policy", "text_only_policy"),
                }
                for item in request_reference_assets
                if item.type != "approved_storyboard_first_frame"
            )
        elif supports_end_frame:
            sent_items.extend(
                {
                    "type": item.type,
                    "role": item.role or item.type,
                    "parameter": "input.media[].type",
                }
                for item in request_reference_assets
                if item.type == "end_frame"
            )
        has_end_frame = any(item.get("type") == "end_frame" for item in sent_items)
        has_pose_control = any(item.type in {"pose_control", "openpose_control"} for item in request_reference_assets)
        has_depth_control = any(item.type in {"depth_control", "depth_map"} for item in request_reference_assets)
        control_types = payload_control_types(
            "video",
            report,
            has_first_frame=bool(reference_image),
            has_end_frame=has_end_frame,
            has_pose_control=has_pose_control,
            has_depth_control=has_depth_control,
        )
        validated_count = len(request_reference_assets)
        if reference_image and not any(item.type == "approved_storyboard_first_frame" for item in request_reference_assets):
            validated_count += 1
        if has_end_frame:
            payload_video_mode = VIDEO_MODE_FIRST_LAST_FRAME
        elif any(item.get("type") != "approved_storyboard_first_frame" for item in sent_items):
            payload_video_mode = VIDEO_MODE_MULTI_REFERENCE_R2V
        else:
            payload_video_mode = VIDEO_MODE_FIRST_FRAME_I2V if reference_image else "text_only"
        generation_metadata = {
            "provider": endpoint.protocol,
            "model": endpoint.model,
            "seed": request.seed,
            "provider_source": provider_source,
            "generation_duration_s": generation_duration,
            "estimated_speech_ms": estimate_speech_ms("\n".join(str(item.text or "") for item in (request.dialogues or []))),
            "reference_mode": "multi_reference" if len(sent_items) > 1 else ("first_frame_reference" if reference_image else "text_only"),
            "video_mode": payload_video_mode,
            "references_validated": validated_count,
            "references_sent": [item["type"] for item in sent_items],
            "references_sent_detail": sent_items,
            "control_types_sent": control_types,
            "provider_capabilities": report,
            "reference_weight_policy": report.get("reference_weight_policy", "text_only_policy"),
        }
        if reference_image and str(getattr(effective_caps, "reference_mode", "") or "") == "first_frame_only":
            generation_metadata["reference_mode"] = "first_frame_only"
        generation_metadata["consistency_metrics"] = combine_report(
            payload_metrics(
                references_validated=validated_count,
                references_sent=sent_items,
                required_roles=[item.get("role") or item.get("type") for item in sent_items],
                control_types_sent=control_types,
                provider_capabilities=report,
            ),
            {"status": "not_run"},
        )
        logger.info(
            "视频生成请求: provider=%s model=%s reference_mode=%s duration=%s shot=%s "
            "references_validated=%s references_sent=%s control_types_sent=%s reference_weight_policy=%s",
            endpoint.protocol,
            endpoint.model or "default",
            generation_metadata["reference_mode"],
            request.duration,
            safe_shot_id,
            generation_metadata["references_validated"],
            generation_metadata["references_sent"],
            control_types,
            generation_metadata["reference_weight_policy"],
        )
        metadata = adapter_usage_for_request(adapter, CAPABILITY_VIDEO, request)
        debug_request_id = record_api_request(
            api="Video Generate",
            provider=endpoint.protocol,
            model=endpoint.model or "default",
            params={
                "duration": request.duration,
                "ratio": request.ratio,
                "resolution": request.resolution,
                "dialogues": [
                    {"role": item.role, "text": item.text, "emotion": item.emotion}
                    for item in (request.dialogues or [])
                ],
                "reference_count": validated_count,
                "reference_mode": self.last_generation_metadata.get("reference_mode", "text_only"),
                "control_types_sent": control_types,
            },
            prompt=request.prompt,
        )
        scope = usage_service.current_scope().merged(project_id=safe_project_id, shot_id=safe_shot_id)
        started = time.monotonic()
        try:
            result = await adapter.generate(request)
        except asyncio.CancelledError:
            usage_service.record_cancelled(
                metadata,
                duration_ms=int((time.monotonic() - started) * 1000),
                scope=scope,
            )
            raise
        except Exception as exc:
            usage_service.record_failure(
                metadata,
                error_code=ERROR_CODE_PROVIDER_CALL_FAILED,
                duration_ms=int((time.monotonic() - started) * 1000),
                scope=scope,
            )
            record_api_result(
                debug_request_id,
                api="Video Generate",
                status="error",
                message=f"视频 API 调用失败：{exc}",
            )
            raise
        usage_service.record_metadata(
            metadata,
            duration_ms=int((time.monotonic() - started) * 1000),
            extra_units={"task_id": str(result.task_id or "")},
            scope=scope,
        )
        record_api_result(
            debug_request_id,
            api="Video Generate",
            status="success",
            message="视频 API 返回成功",
            detail={
                "task_id": str(result.task_id or ""),
                "payload_mode": str(result.payload_mode or ""),
                "native_audio": bool(result.native_audio),
            },
        )
        if result.native_audio:
            await self._assert_stream_has_audio(result.video_path)
        self.last_generation_metadata = dict(generation_metadata)
        return {
            "video_path": result.video_path,
            "frame_path": result.frame_path,
            "task_id": result.task_id,
            "reference_payload_mode": result.payload_mode or content_payload_mode or generation_metadata.get("reference_mode", "text_only"),
            "native_audio": bool(result.native_audio),
            "generation_report": dict(generation_metadata),
        }

    async def generate_shot_video(
        self,
        shot: dict,
        characters: list[dict],
        scenes: dict[str, dict],
        project_id: str,
        capability_mode: str = "manual",
        confirm_capability_downgrade: bool = False,
        provider_override: str = "",
        resolution_override: str = "",
    ) -> dict[str, str]:
        endpoint = video_protocol_defaults(provider_override) if provider_override else get_endpoint("video")
        adapter_cls = get_adapter("video", endpoint.protocol)
        # 显式 provider_override 优先于自动选择：端点由调用方指定后，能力判断
        # 与路由只针对该端点的模型生效能力裁决，绝不悄悄换 Provider/模型。
        provider_source = "agent_selected" if provider_override else "configured"
        capabilities = effective_video_capabilities(adapter_cls, endpoint.model)
        reference_assets, invalid_candidates = self._video_reference_assets(shot, capabilities)
        required_capabilities = self._shot_required_capabilities(shot, reference_assets)
        multi_required = any(
            item.type not in {"approved_storyboard_first_frame", "end_frame"}
            for item in reference_assets
        )
        if multi_required and not getattr(capabilities, "multiple_reference_images", False):
            warning = (
                f"视频 Provider {endpoint.protocol}/{endpoint.model or 'default'} 为 first_frame_only，"
                "角色三视图、场景基准图和连续性参考不会发送；自动模式禁止静默降级，手动模式需显式确认。"
            )
            if str(capability_mode or "manual").lower() == "auto" or not confirm_capability_downgrade:
                raise CapabilityDowngradeRequiredError(warning)
            logger.warning("视频参考能力降级已获人工确认: %s", warning)

        # 逐镜头路由：镜头要求 + 模型生效能力 → video_mode；确认降级后
        # first_frame_only Provider 走 first_frame_i2v，只发送首帧。
        video_mode = select_video_mode(required_capabilities, capabilities)
        shot["required_capabilities"] = required_capabilities
        shot["video_mode"] = video_mode
        shot["provider_source"] = provider_source
        selected_assets, drop_reasons = self._select_references_for_send(reference_assets, capabilities)

        reference_manifest = [
            self._manifest_item(
                asset,
                sent=(asset.type, asset.source_path) not in drop_reasons,
                not_sent_reason=drop_reasons.get((asset.type, asset.source_path), ""),
            )
            # manifest 与发送集同序：按 首帧 → 角色身份 → 场景基准 → 连续性 排列。
            for asset in sorted(reference_assets, key=lambda item: reference_send_priority(item.type))
        ]
        reference_manifest.extend(self._invalid_candidate_manifest_item(item) for item in invalid_candidates)
        continuity_profile = self.consistency._profile(shot.get("continuity_profile"))
        continuity_item = self.consistency.continuity_manifest_item(continuity_profile)
        continuity_drop_reason = ""
        if continuity_item.get("used"):
            if not getattr(capabilities, "multiple_reference_images", False):
                continuity_drop_reason = "provider_first_frame_only"
            else:
                continuity_drop_reason = drop_reasons.get(
                    ("continuity_frame", str(continuity_item.get("path") or "")), ""
                )
        continuity_item["candidate"] = bool(continuity_item.get("used"))
        continuity_item["validated"] = any(
            item.type == "continuity_frame" and item.source_path == str(continuity_item.get("path") or "")
            for item in reference_assets
        )
        continuity_item["sent"] = bool(continuity_item.get("used") and not continuity_drop_reason)
        continuity_item["not_sent_reason"] = continuity_drop_reason
        reference_manifest.append(continuity_item)
        requires_first_frame = bool(getattr(capabilities, "reference_image", True))
        if requires_first_frame and not any(item.get("type") == "approved_storyboard_first_frame" and item.get("loaded") for item in reference_manifest):
            raise RuntimeError("视频生成缺少 approved_storyboard_first_frame（已审核分镜首帧参考图），已阻止纯文本生成")
        shot["seedance_reference_manifest"] = reference_manifest

        duration_capability = provider_duration_capability(endpoint.protocol, capabilities=capabilities, model=endpoint.model)
        # 统一执行计划是有效时长的唯一来源：上游（TTS 实测后）已落计划时直接
        # 复用（保留 tts_measured 标记）；旧调用方没有计划则按当前 Provider
        # 能力推导。候选数/恢复预算来自调用方（质量档位），本函数按可加载素材
        # 给出权威能力清单与最终 video_mode，都通过 replace 叠加进计划后落库。
        try:
            planned_candidates = max(0, int(shot.get("candidate_count") or 0))
        except (TypeError, ValueError):
            planned_candidates = 0
        try:
            planned_recovery = int(shot.get("recovery_budget") if shot.get("recovery_budget") is not None else -1)
        except (TypeError, ValueError):
            planned_recovery = -1
        execution_plan = resolve_shot_execution_plan(
            shot,
            provider=duration_capability,
            audio_mode=str(shot.get("audio_mode") or ""),
        )
        continuity_mode = normalize_continuity_mode(
            shot.get("continuity_mode") or continuity_profile.get("continuity_mode")
        )
        execution_plan = replace(
            execution_plan,
            continuity_mode=continuity_mode,
            video_mode=video_mode,
            required_capabilities=tuple(required_capabilities),
            **({"candidate_count": planned_candidates} if planned_candidates > 0 else {}),
            **({"recovery_budget": max(0, planned_recovery)} if planned_recovery >= 0 else {}),
        )
        requested_duration = execution_plan.narrative_duration_ms / 1000.0
        # 固定档 Provider 可把 4.5s 叙事时间线用 5s 生成后剪辑；只有明显超出
        # 固定档/上限的镜头才要求先拆分，避免一刀切拒绝合法短镜头。
        if requested_duration <= 0 or requested_duration > float(duration_capability.max_duration or 0) + duration_capability.tolerance_s:
            try:
                duration_capability.validate(requested_duration, shot_id=str(shot.get("shot_id") or shot.get("id") or ""))
            except StoryTimingError as exc:
                raise RuntimeError(f"{exc}；请拆分镜头到 Provider 允许的时长后逐镜生成") from exc
        generation_duration = float(execution_plan.provider_generation_duration_s)
        shot["generation_duration_s"] = generation_duration
        shot["execution_plan"] = execution_plan.to_dict()
        shot["estimated_speech_ms"] = estimate_speech_ms(dialogue_text(shot.get("dialogue")))

        prompt = self._build_prompt(shot, characters, scenes)
        content = self._build_content(prompt, shot, capabilities, selected_assets)
        result = await self.generate_single_shot(
            prompt=prompt,
            project_id=project_id,
            shot_id=shot.get("shot_id", "seedance_shot"),
            duration=int(round(generation_duration)),
            ratio=shot.get("output_format", "9:16"),
            resolution=self._resolution(resolution_override or shot.get("resolution")),
            content=content,
            dialogues=self._dialogues_from_shot(shot),
            reference_assets=selected_assets,
            provider_override=provider_override,
            seed=int(shot.get("seed")) if shot.get("seed") is not None else None,
        )
        report = dict(result.get("generation_report") or self.last_generation_metadata or {})
        shot["reference_mode"] = str(getattr(capabilities, "reference_mode", "") or report.get("reference_mode") or ("multi_reference" if multi_required else "first_frame_only"))
        shot["references_validated"] = len(reference_assets)
        # 回退值只取 manifest 里真实标记为 sent 的项：first_frame_only Provider
        # 不得把已校验的角色图/场景图/尾帧虚报成已发送。
        shot["references_sent"] = list(report.get("references_sent") or [item.get("type") for item in reference_manifest if item.get("sent")])
        shot["references_sent_detail"] = list(report.get("references_sent_detail") or [])
        shot["control_types_sent"] = list(report.get("control_types_sent") or [])
        shot["provider_capabilities"] = dict(report.get("provider_capabilities") or {})
        shot["reference_weight_policy"] = str(report.get("reference_weight_policy") or "text_only_policy")
        shot["consistency_metrics"] = dict(report.get("consistency_metrics") or {})
        logger.info(
            "镜头视频参考: provider=%s model=%s reference_mode=%s continuity_mode=%s references_validated=%s references_sent=%s "
            "control_types_sent=%s reference_weight_policy=%s",
            report.get("provider", endpoint.protocol),
            report.get("model", endpoint.model or "default"),
            shot["reference_mode"],
            continuity_mode,
            shot["references_validated"],
            shot["references_sent"],
            shot["control_types_sent"],
            shot["reference_weight_policy"],
        )
        result["reference_manifest"] = reference_manifest
        return result

    # ------------------------------------------------------------------
    # 台词与参考图策略
    # ------------------------------------------------------------------

    @staticmethod
    def _dialogues_from_shot(shot: dict) -> list[Dialogue] | None:
        dialogues = shot.get("dialogues")
        if not dialogues:
            return None
        normalized: list[Dialogue] = []
        for item in dialogues:
            if isinstance(item, Dialogue):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(
                    Dialogue(
                        role=str(item.get("role") or item.get("speaker") or ""),
                        text=str(item.get("text") or item.get("line") or ""),
                        emotion=str(item.get("emotion") or "neutral"),
                        start_ms=int(item.get("start_ms") or 0),
                        end_ms=int(item.get("end_ms") or 0),
                    )
                )
        return normalized or None

    def _reference_from_content(self, content: list[dict]) -> tuple[str | None, list[ReferenceAsset], str]:
        """从内容列表中取首帧参考图和结构化参考素材；payload 模式由适配器按内容判定。"""
        first_frame = ""
        assets: list[ReferenceAsset] = []
        for item in content or []:
            if item.get("type") != "image_url":
                continue
            url = (item.get("image_url") or {}).get("url") if isinstance(item.get("image_url"), dict) else None
            if not url:
                continue
            role = str(item.get("role") or "")
            if role == "first_frame":
                first_frame = first_frame or str(url)
            elif role == "end_frame":
                assets.append(ReferenceAsset(url=str(url), type="end_frame", role="end_frame", provider_type="end_frame"))
            elif not first_frame:
                # 兼容旧 content：第一张未标 role 的图按历史语义作为首帧。
                first_frame = str(url)
            else:
                assets.append(ReferenceAsset(url=str(url), type=role or "reference_image", role=role or "reference_image"))
        return (first_frame or None), assets, ("first_frame_reference" if first_frame else "text_only")

    def _resolution(self, resolution: str | None) -> str:
        # 项目分辨率可为 720p/1080p/2k/4k；Seedance 1.5 pro 仅支持到 1080p，
        # 因此 2k/4k 统一降级到 1080p，保证 API 不因不支持的档位报错。
        value = str(resolution or "").strip().lower()
        mapping = {
            "480p": "480p",
            "720p": "720p",
            "1080p": "1080p",
            "1080": "1080p",
            "2k": "1080p",
            "4k": "1080p",
        }
        return mapping.get(value, "720p")

    def _build_prompt(self, shot: dict, characters: list[dict], scenes: dict[str, dict]) -> str:
        """按预算优先级组装视频 Prompt。

        字段顺序即裁剪优先级（从高到低）：
        1. 风格与负向约束；2. 人物身份/年龄/脸型/发型/服装/关键特征；
        3. 场景、构图与首帧参考说明；4. character_action/emotion/
        camera_movement/camera_angle/shot_type；5. 连续性规则与低优先级 SOP。
        超预算时从末尾整字段丢弃；人物身份、动作、情绪、运镜四类关键字段
        由 ``ensure_critical_fields`` 兜底，任何情况下都必须留在 Prompt 里。
        """
        scene = scenes.get(shot.get("scene_asset_id", "")) or {}
        selected_characters = self._select_character_cards(shot, characters)
        style_params = style_prompt_params(shot.get("style") or shot.get("style_id"))
        provider_label = get_endpoint("video").protocol or "video provider"

        # 时长与对白预算优先取统一执行计划：计划中的 Provider 生成时长、实测
        # 对白时间轴是有效值；没有计划（旧调用方）才回退旧字段推导。
        execution_plan = load_shot_execution_plan(shot)
        if execution_plan is not None:
            generation_duration = float(execution_plan.provider_generation_duration_s)
            if execution_plan.dialogue_timing:
                estimated_speech_ms = int(execution_plan.dialogue_end_ms or 0)
            else:
                estimated_speech_ms = int(shot.get("estimated_speech_ms") or estimate_speech_ms(dialogue_text(shot.get("dialogue"))))
            effective_duration_s = execution_plan.effective_duration_ms / 1000.0
        else:
            generation_duration = float(
                shot.get("generation_duration_s")
                or get_video_generation_duration_s(float(shot.get("duration") or 0))
            )
            estimated_speech_ms = int(shot.get("estimated_speech_ms") or estimate_speech_ms(dialogue_text(shot.get("dialogue"))))
            effective_duration_s = float(shot.get("duration") or 3.0)
        fields: list[tuple[str, str]] = [
            (
                "duration_policy",
                f"actual generated clip duration: {generation_duration:g} seconds; "
                "stage the complete action and dialogue within this exact duration",
            ),
            (
                "speech_timing",
                f"estimated speech duration: {estimated_speech_ms} ms; dialogue must finish inside the shot without rushing or truncation",
            ),
            ("effective_style", style_params.get("video_prompt", "")),
            ("style_label", f"locked visual style preset: {style_params.get('style_label', '')}"),
            ("identity_policy", "NON-NEGOTIABLE identity and style consistency policy"),
        ]
        # --- 2. 人物身份 ---
        for char in selected_characters:
            appearance = char.get("appearance") or {}
            appearance_parts = [str(value) for value in appearance.values()] if isinstance(appearance, dict) else []
            reference_lock = "preserve approved three-view character sheet identity" if char.get("reference_images") else ""
            fields.extend([
                ("character_identity", char.get("visual_prompt", "")),
                ("character_features", ", ".join(char.get("key_features", []))),
                ("character_appearance", ", ".join(appearance_parts)),
                ("identity_lock", reference_lock),
                ("wardrobe", char.get("wardrobe_lock", "")),
            ])
        # --- 3. 场景、构图与首帧参考说明 ---
        fields.extend([
            (
                "scene",
                ", ".join(str(v) for v in (scene.get("visual_prompt", ""), scene.get("description", ""), scene.get("prop_lock", "")) if v),
            ),
            ("scene_description", shot.get("scene_description", "")),
            ("approved_storyboard", shot.get("storyboard_prompt", "")),
            ("first_frame_policy", "match the approved storyboard first frame for identity, costume, scene palette and composition"),
            ("visual_notes", shot.get("visual_notes", "")),
        ])
        if self._approved_storyboard_first_frame(shot):
            fields.append(
                (
                    "reference_mode",
                    f"{provider_label} first_frame_only: attached image 1 is the approved storyboard first frame",
                )
            )
        # --- 4. 动作、情绪与运镜 ---
        fields.extend([
            ("character_action", shot.get("character_action", "")),
            (
                "action_beat",
                "; ".join(
                    f"{str(item.get('phase') or 'continuation')}:{str(item.get('text') or '')}"
                    for item in (shot.get("action_beats") or [])
                    if isinstance(item, dict)
                ),
            ),
            (
                "action_boundaries",
                f"entry={shot.get('action_entry_state') or (shot.get('timing') or {}).get('action_entry_state') or ''}; "
                f"exit={shot.get('action_exit_state') or (shot.get('timing') or {}).get('action_exit_state') or ''}",
            ),
            (
                "gaze_and_axis",
                f"gaze={shot.get('gaze_direction') or ''}; screen_axis={shot.get('screen_axis') or ''}",
            ),
            ("emotion", f"emotional tone: {shot.get('emotion', 'neutral')}"),
            ("camera_movement", f"camera movement: {shot.get('camera_movement', '静止')}"),
            ("camera_angle", f"camera angle: {shot.get('camera_angle', '正面')}"),
            ("shot_type", f"shot size: {shot.get('shot_type', 'medium')}"),
            (
                "camera_strategy",
                CAMERA_MOVEMENT_PROMPTS.get(
                    str(shot.get("camera_movement") or "静止"),
                    f"camera movement: {shot.get('camera_movement') or '静止'}",
                ),
            ),
            (
                "shot_timing",
                (
                    f"shot sequence {int(shot.get('sequence') or 0)}; "
                    f"requested source duration {effective_duration_s:.3f}s; "
                    f"estimated speech {estimated_speech_ms}ms; "
                    f"dialogue and action must remain inside this shot; "
                    f"timeline {int(shot.get('timeline_start_ms') or 0)}-{int(shot.get('timeline_end_ms') or 0)}ms; "
                    f"timing metadata {shot.get('timing') or {}}"
                ),
            ),
            ("motion_policy", "cinematic short drama video, coherent motion, no subtitles, no watermark"),
        ])
        if execution_plan is not None and execution_plan.dialogue_timing:
            dialogue_timing = [item.to_dict() for item in execution_plan.dialogue_timing]
        else:
            dialogue_timing = shot.get("dialogue_timing") or shot.get("dialogues") or shot.get("dialogue")
        if dialogue_timing:
            timed_parts = []
            for item in dialogue_timing if isinstance(dialogue_timing, list) else [dialogue_timing]:
                if not isinstance(item, dict):
                    continue
                timed_parts.append(
                    f"{item.get('speaker') or item.get('role') or '角色'} "
                    f"[{int(item.get('start_ms') or 0)}-{int(item.get('end_ms') or 0)}ms] "
                    f"{item.get('emotion') or shot.get('emotion') or 'neutral'}: "
                    f"{item.get('text') or item.get('line') or ''}"
                )
            if timed_parts:
                fields.append(("dialogue_timing", "; ".join(timed_parts)))
        # --- 5. 连续性规则与低优先级 SOP ---
        continuity_profile = self.consistency._profile(shot.get("continuity_profile"))
        continuity_mode = normalize_continuity_mode(continuity_profile.get("continuity_mode"))
        fields.append(("continuity_mode", f"shot continuity mode: {continuity_mode}"))
        if continuity_profile.get("continuity_reference_used"):
            fields.append(("continuity", "previous shot last frame is used only for continuous action"))
        elif continuity_profile.get("continuity_reference_reason"):
            fields.append(("continuity_reference", f"no previous-shot image used: {continuity_profile['continuity_reference_reason']}"))
        if shot.get("reference_weights"):
            weights = shot.get("reference_weights") or {}
            fields.append(("reference_weights",
                f"locked reference weights: environment/style {float(weights.get('environment') or 0.45):.2f}, character/action {float(weights.get('action') or 0.30):.2f}"
            ))
        if shot.get("reference_assets"):
            roles = ", ".join(str(item.get("role", "")) for item in shot.get("reference_assets", []) if isinstance(item, dict))
            fields.append(("reference_roles", f"validated persisted asset roles (not sent to {provider_label}): {roles}"))
        if shot.get("seedance_reference_manifest"):
            manifest = shot.get("seedance_reference_manifest") or []
            loaded = ", ".join(str(item.get("type", "")) for item in manifest if isinstance(item, dict))
            fields.append(("validated_assets", f"validated manifest only; {provider_label} receives first frame only, not {loaded}"))
        if shot.get("continuity_profile"):
            profile = shot.get("continuity_profile") or {}
            fields.append(("continuity_profile", f"text continuity rules: {', '.join(profile.get('editing_logic', []))}; no OpenPose/Depth control is sent"))
            blocking = profile.get("character_blocking") or {}
            if blocking:
                order = blocking.get("character_order_left_to_right") or []
                fields.append(("blocking",
                    "locked character blocking: "
                    f"left-to-right order {', '.join(order) if order else 'single subject'}; "
                    f"{blocking.get('axis_line', '180-degree axis locked')}; "
                    f"eye-line {blocking.get('eye_line_target', 'locked')}; "
                    f"{blocking.get('camera_movement_limit', '')}; "
                    f"{blocking.get('skin_light_integration', '')}"
                ))
        if shot.get("consistency_context"):
            fields.append(("consistency_context", shot["consistency_context"]))
        if shot.get("skill_prompt_append"):
            fields.append(("skill_sop", shot["skill_prompt_append"]))

        normalized_fields = [
            (name, self._clean_prompt_part(value)) for name, value in fields
        ]
        prompt, dropped = assemble_prompt(normalized_fields)
        critical = self._critical_fields(selected_characters, shot)
        prompt, readded = ensure_critical_fields(prompt, critical)
        self.last_prompt_trimmed_fields = dropped
        self.last_prompt_readded_fields = readded
        if dropped or readded:
            logger.info(
                "视频 Prompt 预算裁剪: dropped_fields=%s readded_critical_fields=%s",
                dropped,
                readded,
            )
        return prompt

    @staticmethod
    def _critical_fields(selected_characters: list[dict], shot: dict) -> list[tuple[str, str]]:
        """人物身份、动作、情绪、运镜、时长：裁剪后必须仍然存在的关键字段。"""
        critical: list[tuple[str, str]] = []
        execution_plan = load_shot_execution_plan(shot)
        if execution_plan is not None:
            generation_duration = float(execution_plan.provider_generation_duration_s)
            speech_ms = int(execution_plan.dialogue_end_ms or shot.get("estimated_speech_ms") or 0)
            effective_duration_s = execution_plan.effective_duration_ms / 1000.0
        else:
            generation_duration = float(
                shot.get("generation_duration_s")
                or get_video_generation_duration_s(float(shot.get("duration") or 0))
            )
            speech_ms = int(shot.get("estimated_speech_ms") or 0)
            effective_duration_s = float(shot.get("duration") or 3.0)
        critical.append((
            "duration_policy",
            f"actual generated clip duration: {generation_duration:g} seconds",
        ))
        critical.append((
            "speech_timing",
            f"estimated speech duration: {speech_ms} ms",
        ))
        for char in selected_characters[:2]:
            critical.append(("character_identity", str(char.get("visual_prompt", "") or char.get("name", ""))))
        critical.extend([
            ("character_action", str(shot.get("character_action", "") or "")),
            ("emotion", f"emotional tone: {shot.get('emotion', 'neutral')}"),
            ("camera_movement", CAMERA_MOVEMENT_PROMPTS.get(str(shot.get('camera_movement') or '静止'), f"camera movement: {shot.get('camera_movement', '静止')}")),
            ("camera_angle", f"camera angle: {shot.get('camera_angle', '正面')}"),
            ("shot_timing", f"requested duration {effective_duration_s:.3f}s"),
        ])
        return critical

    def _select_character_cards(self, shot: dict, characters: list[dict]) -> list[dict]:
        selected_ids = {str(item) for item in shot.get("character_asset_ids", []) if item}
        selected_names = {str(item) for item in shot.get("characters_in_scene", []) if item}
        selected: list[dict] = []
        seen: set[str] = set()

        for char in characters:
            char_id = str(char.get("id") or "")
            char_name = str(char.get("name") or "")
            if selected_ids and char_id not in selected_ids:
                continue
            key = char_id or char_name
            if key and key not in seen:
                selected.append(char)
                seen.add(key)

        if selected:
            return selected

        for char in characters:
            char_name = str(char.get("name") or "")
            if selected_names and char_name not in selected_names:
                continue
            key = str(char.get("id") or char_name)
            if key and key not in seen:
                selected.append(char)
                seen.add(key)
        return selected

    def _clean_prompt_part(self, value) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if "生成失败" in text or "Traceback" in text:
            return ""
        if "Seedance" in text and ("失败" in text or "failed" in text.lower() or "'id'" in text or '"id"' in text):
            return ""
        text = re.sub(r"[A-Za-z]:[\\/][^,，\\n]+", "", text)
        text = re.sub(r"(?:[A-Za-z]:)?[\\/][^,，\\n]*(?:output|projects|characters|shots|seedance)[^,，\\n]*", "", text)
        text = re.sub(r"output[\\/][^,，\\n]+", "", text)
        text = text.replace("{", "").replace("}", "")
        return " ".join(text.split())

    def _build_content(self, prompt: str, shot: dict, capabilities=None, reference_assets: list[ReferenceAsset] | None = None) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": prompt}]

        if capabilities is not None and not getattr(capabilities, "reference_image", False):
            return content

        first_frame = self._approved_storyboard_first_frame(shot)
        # 参考图按协议能力预算压缩后内联：优先保留高分辨率档位，只有超预算时
        # 才逐级降采样；压缩结果（原图/发送尺寸与字节）记录进日志。
        budget = getattr(capabilities, "max_reference_inline_bytes", 0) or settings.VIDEO_REFERENCE_INLINE_BUDGET_BYTES
        first_frame_url = self.reference_assets.to_image_url(first_frame, max_bytes=budget)
        transform = dict(self.reference_assets.last_transform_metadata or {})
        if first_frame and not first_frame_url:
            # 压缩/编码失败必须明确报错，不允许悄悄退化成纯文本生成。
            raise RuntimeError(
                "已审核故事板首帧参考图无法编码进视频请求体预算"
                f"（预算 {budget} 字节，原图 {transform.get('original_bytes', 0)} 字节）；"
                "请检查参考图文件是否损坏后重试"
            )
        if first_frame_url:
            logger.info(
                "视频首帧参考传输: provider_budget=%s original_bytes=%s sent_bytes=%s "
                "original_size=%s sent_size=%s",
                budget,
                transform.get("original_bytes"),
                transform.get("sent_bytes"),
                transform.get("original_size"),
                transform.get("sent_size"),
            )
            content.append({"type": "image_url", "image_url": {"url": first_frame_url}, "role": "first_frame"})
        return content

    def _has_image_content(self, content: list[dict]) -> bool:
        return any(item.get("type") == "image_url" for item in content)

    def _reference_payload_mode(self, content: list[dict]) -> str:
        roles = {str(item.get("role") or "") for item in content if item.get("type") == "image_url"}
        if "first_frame" in roles:
            return "first_frame_reference"
        if roles:
            return "image_reference"
        return "text_only"

    def _video_reference_assets(self, shot: dict, capabilities=None) -> tuple[list[ReferenceAsset], list[dict]]:
        """收集视频请求候选参考，返回 (可加载素材, 无法校验的候选)。

        首帧永远单独建模；无法读取/编码的候选不进入发送集，但以
        validated=False 记入 manifest，供发送前审计，不做静默丢弃。
        """

        assets: list[ReferenceAsset] = []
        invalid: list[dict] = []
        first = self._approved_storyboard_first_frame(shot)
        first_url = self.reference_assets.to_image_url(first) if first else ""
        if first_url:
            assets.append(ReferenceAsset(url=first_url, type="approved_storyboard_first_frame", role="first_frame", provider_type="first_frame", source_path=first))
        elif first:
            invalid.append({"type": "approved_storyboard_first_frame", "role": "first_frame", "path": first, "reason": "reference_unreadable"})
        supports_end_frame = supports_first_last_frame(capabilities)
        end_frame = str(shot.get("end_frame_path") or shot.get("last_frame_input_path") or "")
        candidates: list[tuple[str, str, str]] = []
        if end_frame and supports_end_frame:
            candidates.append(("end_frame", "end_frame", end_frame))
        for item in shot.get("reference_assets") or []:
            if isinstance(item, dict) and item.get("path"):
                kind = str(item.get("type") or "reference_image")
                if kind == "end_frame" and not supports_end_frame:
                    continue
                if kind in {"openpose_source_frame", "pose_control", "openpose_control"} and not getattr(capabilities, "openpose", False):
                    continue
                if kind in {"depth_source_frame", "depth_control", "depth_map"} and not getattr(capabilities, "depth", False):
                    continue
                candidates.append((kind, str(item.get("role") or ""), str(item["path"])))
        for path in shot.get("scene_reference_images") or []:
            if path:
                candidates.append(("scene_baseline", "environment_props_lighting_perspective", str(path)))
        for path in shot.get("character_reference_images") or []:
            if path:
                candidates.append(("character_three_view", "identity_outfit_face_body_hair", str(path)))
        continuity_profile = self.consistency._profile(shot.get("continuity_profile"))
        if (
            normalize_continuity_mode(continuity_profile.get("continuity_mode"), default="")
            == "continuous_action"
            and continuity_profile.get("continuity_reference_used")
            and shot.get("continuity_reference_path")
        ):
            candidates.append(("continuity_frame", "eye_line_axis_motion", str(shot["continuity_reference_path"])))
        seen = {(item.type, item.source_path) for item in assets}
        for kind, role, path in candidates:
            key = (kind, path)
            if key in seen:
                continue
            seen.add(key)
            url = self.reference_assets.to_image_url(path)
            if not url:
                invalid.append({"type": kind, "role": role or kind, "path": path, "reason": "reference_unreadable"})
                continue
            assets.append(ReferenceAsset(url=url, type=kind, role=role or kind, provider_type="reference_image", source_path=path))
        return assets, invalid

    @staticmethod
    def _approved_storyboard_first_frame(shot: Mapping[str, Any]) -> str:
        """只使用当前审核分镜首帧；绝不回退到视频尾帧或动作控制帧。"""

        return str(
            shot.get("approved_storyboard_first_frame_path")
            or shot.get("storyboard_path")
            or shot.get("image_path")
            or ""
        )

    @staticmethod
    def _shot_required_capabilities(shot: dict, reference_assets: list[ReferenceAsset]) -> list[str]:
        """镜头实际需要的模型能力清单，能力名与能力矩阵 features 对齐。"""

        types = {item.type for item in reference_assets}
        required: list[str] = []
        if "approved_storyboard_first_frame" in types:
            required.append("first_frame")
        if "character_three_view" in types:
            required.append("character_identity")
        if "scene_baseline" in types:
            required.append("scene_reference")
        if "end_frame" in types:
            required.append("first_last_frame_interpolation")
        if types & {"pose_control", "openpose_control", "openpose_source_frame"}:
            required.append("pose_control")
        if types & {"depth_control", "depth_map", "depth_source_frame"}:
            required.append("depth_control")
        if types - {"approved_storyboard_first_frame", "end_frame"}:
            required.append("multiple_reference_images")
        return required

    @staticmethod
    def _select_references_for_send(
        reference_assets: list[ReferenceAsset], capabilities=None
    ) -> tuple[list[ReferenceAsset], dict[tuple[str, str], str]]:
        """按优先级选出实际进入 Provider 载荷的参考集，返回 (发送集, 未发送原因表)。

        first_frame_only Provider 只保留首帧，角色/场景/连续性参考记
        ``provider_first_frame_only``；multi_reference（r2v）模型在
        ``max_reference_images`` 上限内按 首帧 → 角色身份 → 场景基准 →
        连续性 的优先级发送，超限项记 ``reference_count_limit``。
        """

        first = [item for item in reference_assets if item.type == "approved_storyboard_first_frame"]
        end_frames = [item for item in reference_assets if item.type == "end_frame"]
        others = sorted(
            (
                item for item in reference_assets
                if item.type not in {"approved_storyboard_first_frame", "end_frame"}
            ),
            key=lambda item: reference_send_priority(item.type),
        )
        multi = bool(getattr(capabilities, "multiple_reference_images", False))
        end_supported = supports_first_last_frame(capabilities)
        max_refs = int(getattr(capabilities, "max_reference_images", 0) or 0)
        selected = list(first)
        if end_supported:
            selected.extend(end_frames)
        drop_reasons: dict[tuple[str, str], str] = {}
        for asset in end_frames:
            if not end_supported:
                drop_reasons[(asset.type, asset.source_path)] = "provider_end_frame_unsupported"
        reference_count = 0
        for asset in others:
            if not multi:
                drop_reasons[(asset.type, asset.source_path)] = "provider_first_frame_only"
            elif max_refs > 0 and reference_count >= max_refs:
                drop_reasons[(asset.type, asset.source_path)] = "reference_count_limit"
            else:
                selected.append(asset)
                reference_count += 1
        return selected, drop_reasons

    @staticmethod
    def _manifest_item(asset: ReferenceAsset, *, sent: bool, not_sent_reason: str = "") -> dict:
        """manifest 项：candidate=进入发送候选，validated=文件可读且可编码，
        sent=实际进入 Provider 载荷；未发送的原因写进 not_sent_reason。"""

        return {
            "type": asset.type,
            "role": asset.role or asset.type,
            "path": asset.source_path,
            "candidate": True,
            "validated": True,
            "loaded": True,
            "sent": bool(sent),
            "not_sent_reason": "" if sent else (not_sent_reason or "not_selected"),
        }

    @staticmethod
    def _invalid_candidate_manifest_item(candidate: dict) -> dict:
        """候选存在但无法校验（文件缺失/无法编码）：candidate 而 validated=False。"""

        return {
            "type": str(candidate.get("type") or ""),
            "role": str(candidate.get("role") or ""),
            "path": str(candidate.get("path") or ""),
            "candidate": True,
            "validated": False,
            "loaded": False,
            "sent": False,
            "not_sent_reason": str(candidate.get("reason") or "reference_not_validated"),
        }

    def _validate_video_references(self, shot: dict) -> list[dict]:
        missing: list[str] = []
        manifest: list[dict] = []
        seen: set[tuple[str, str]] = set()

        def add_reference(kind: str, path: str, required: bool = True) -> None:
            value = str(path or "").strip()
            key = (kind, value)
            if key in seen:
                return
            seen.add(key)
            image_url = self.reference_assets.to_image_url(value)
            if image_url:
                manifest.append({"type": kind, "path": value, "loaded": True, "validated": True, "sent": kind == "approved_storyboard_first_frame"})
            elif required:
                missing.append(f"{kind}:{value or '<empty>'}")

        add_reference("approved_storyboard_first_frame", self._approved_storyboard_first_frame(shot))
        # 当前视频链路只发送一张图：已审核故事板首帧（first_frame_only）。
        # 场景基准图、角色三视图、连续性帧、OpenPose 与 Depth 素材仅作为
        # manifest 校验记录，不会进入视频请求载荷。

        if missing:
            raise RuntimeError("视频生成缺少必需一致性参考素材: " + " | ".join(missing))
        return manifest

    async def _assert_stream_has_audio(self, video_path: str) -> None:
        """ffprobe 校验原生音频视频确有音轨（两条路径输出契约一致性）。"""
        proc = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(video_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"校验视频音轨失败: {stderr.decode('utf-8', errors='ignore')[-500:]}")
        if '"audio"' not in stdout.decode("utf-8", errors="ignore"):
            raise RuntimeError("原生音频视频未包含音轨，已阻止无声成品进入成片流程")


# 兼容旧导入名（api_diagnostics / sop 脚本 / 测试）。
SeedanceVideoService = VideoService
