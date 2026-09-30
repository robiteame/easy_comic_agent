"""视频生成服务：prompt/参考图/一致性策略 + 协议适配器路由。

端点来自 ``get_endpoint("video")``，协议调用委托给
``services.providers.registry.get_adapter("video", protocol)`` 注册的适配器：
- ``ark-seedance``：异步任务式无声视频（对白走独立 TTS 路径），首帧参考
  （first_frame_only）；时长约束按模型能力校验；
- ``native-audio``：原生音视频骨架（对白编入 prompt，音频随视频直出）。
"""

import asyncio
import logging
import re
import time
from pathlib import Path

from config import settings
from services import usage_service
from services.consistency_service import ConsistencyService
from services.providers.base import Dialogue, ReferenceAsset, VideoRequest
from services.providers.endpoint import KNOWN_PROTOCOLS, EndpointConfig, get_endpoint, video_protocol_defaults
from services.providers.registry import get_adapter
from services.providers.capability_matrix import (
    CapabilityDowngradeRequiredError,
    capability_report,
    payload_control_types,
)
from services.providers.usage import (
    CAPABILITY_VIDEO,
    ERROR_CODE_PROVIDER_CALL_FAILED,
    adapter_usage_for_request,
)
from services.prompt_budget import assemble_prompt, ensure_critical_fields
from services.reference_asset_service import ReferenceAssetService
from services.consistency_metrics import combine_report, payload_metrics, validate_visual_consistency
from services.security import safe_path, validate_identifier
from services.style_templates import style_prompt_params

logger = logging.getLogger(__name__)


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

    def _preferred_video_endpoint(self, primary: EndpointConfig) -> EndpointConfig | None:
        """优先选择已配置且真正支持多参考图的视频 Provider。"""

        for protocol in KNOWN_PROTOCOLS.get("video", ()):
            if protocol == primary.protocol:
                continue
            endpoint = video_protocol_defaults(protocol)
            if not str(endpoint.api_key or "").strip():
                continue
            try:
                adapter_cls = get_adapter("video", protocol)
            except Exception:
                continue
            effective = getattr(adapter_cls, "effective_capabilities", None)
            caps = effective(endpoint.model) if callable(effective) else getattr(adapter_cls, "capabilities", None)
            if getattr(caps, "multiple_reference_images", False) and getattr(caps, "reference_image", False):
                return endpoint
        return None

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
        endpoint_override: EndpointConfig | None = None,
        provider_source: str = "configured",
    ) -> dict[str, str]:
        endpoint = endpoint_override or get_endpoint("video")
        adapter_cls = get_adapter("video", endpoint.protocol)
        adapter = adapter_cls(endpoint)
        effective_method = getattr(adapter_cls, "effective_capabilities", None)
        effective_caps = effective_method(endpoint.model) if callable(effective_method) else adapter.capabilities
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

        fixed_duration = getattr(adapter.capabilities, "fixed_duration", None)
        requested_duration = int(duration or 5)
        if fixed_duration and requested_duration > int(fixed_duration):
            raise RuntimeError(
                f"镜头时长 {requested_duration} 秒超过视频 Provider {endpoint.protocol} "
                f"的固定时长 {int(fixed_duration)} 秒；请把该镜头拆分为多个短镜头，"
                "或把时长调整到固定时长以内"
            )
        request = VideoRequest(
            prompt=prompt,
            reference_image=reference_image,
            reference_assets=request_reference_assets,
            dialogues=dialogues,
            duration=int(fixed_duration or requested_duration),
            ratio=ratio,
            resolution=resolution,
            project_id=safe_project_id,
            output_video_path=video_path,
            output_frame_path=frame_path,
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
        control_types = payload_control_types("video", report, has_first_frame=bool(reference_image))
        validated_count = len(request_reference_assets)
        if reference_image and not any(item.type == "approved_storyboard_first_frame" for item in request_reference_assets):
            validated_count += 1
        self.last_generation_metadata = {
            "provider": endpoint.protocol,
            "model": endpoint.model,
            "provider_source": provider_source,
            "reference_mode": str(getattr(effective_caps, "reference_mode", "text_only") if reference_image else "text_only"),
            "reference_payload_mode": "multi_reference" if len(sent_items) > 1 else ("first_frame_reference" if reference_image else "text_only"),
            "references_validated": validated_count,
            "references_sent": [item["type"] for item in sent_items],
            "references_sent_detail": sent_items,
            "control_types_sent": control_types,
            "provider_capabilities": report,
            "capability_states": report.get("states", {}),
            "reference_weight_policy": report.get("reference_weight_policy", "text_only_policy"),
        }
        self.last_generation_metadata["consistency_metrics"] = combine_report(
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
            "references_validated=%s references_sent=%s control_types_sent=%s capability_states=%s reference_weight_policy=%s",
            endpoint.protocol,
            endpoint.model or "default",
            self.last_generation_metadata["reference_mode"],
            request.duration,
            safe_shot_id,
            self.last_generation_metadata["references_validated"],
            self.last_generation_metadata["references_sent"],
            control_types,
            self.last_generation_metadata["capability_states"],
            self.last_generation_metadata["reference_weight_policy"],
        )
        metadata = adapter_usage_for_request(adapter, CAPABILITY_VIDEO, request)
        scope = usage_service.current_scope().merged(shot_id=safe_shot_id)
        started = time.monotonic()
        try:
            result = await adapter.generate(request)
        except asyncio.CancelledError:
            usage_service.record_cancelled(metadata, duration_ms=int((time.monotonic() - started) * 1000), scope=scope)
            raise
        except Exception:
            usage_service.record_failure(
                metadata,
                error_code=ERROR_CODE_PROVIDER_CALL_FAILED,
                duration_ms=int((time.monotonic() - started) * 1000),
                scope=scope,
            )
            raise
        usage_service.record_metadata(
            metadata,
            duration_ms=int((time.monotonic() - started) * 1000),
            extra_units={"task_id": str(result.task_id or "")},
            scope=scope,
        )
        if result.native_audio:
            await self._assert_stream_has_audio(result.video_path)
        return {
            "video_path": result.video_path,
            "frame_path": result.frame_path,
            "task_id": result.task_id,
            "reference_payload_mode": result.payload_mode or content_payload_mode or self.last_generation_metadata.get("reference_mode", "text_only"),
            "native_audio": bool(result.native_audio),
            "generation_report": dict(self.last_generation_metadata),
        }

    async def generate_shot_video(
        self,
        shot: dict,
        characters: list[dict],
        scenes: dict[str, dict],
        project_id: str,
        capability_mode: str = "manual",
        confirm_capability_downgrade: bool = False,
    ) -> dict[str, str]:
        endpoint = get_endpoint("video")
        provider_source = "configured"
        adapter_cls = get_adapter("video", endpoint.protocol)
        effective_method = getattr(adapter_cls, "effective_capabilities", None)
        capabilities = effective_method(endpoint.model) if callable(effective_method) else adapter_cls.capabilities
        reference_assets = self._video_reference_assets(shot)
        multi_required = any(item.type != "approved_storyboard_first_frame" for item in reference_assets)
        if multi_required and not getattr(capabilities, "multiple_reference_images", False):
            alternate = self._preferred_video_endpoint(endpoint)
            if alternate is not None:
                endpoint = alternate
                provider_source = "preferred_consistency_provider"
                adapter_cls = get_adapter("video", endpoint.protocol)
                capabilities = adapter_cls.effective_capabilities(endpoint.model)
                logger.warning(
                    "视频 Provider 自动优选: 改用支持多参考图的 %s/%s",
                    endpoint.protocol,
                    endpoint.model or "default",
                )
        if multi_required and not getattr(capabilities, "multiple_reference_images", False):
            warning = (
                f"视频 Provider {endpoint.protocol}/{endpoint.model or 'default'} 为 first_frame_only，"
                "角色三视图、场景基准图和连续性参考不会发送；自动模式禁止静默降级，手动模式需显式确认。"
            )
            if str(capability_mode or "manual").lower() == "auto" or not confirm_capability_downgrade:
                raise CapabilityDowngradeRequiredError(warning)
            logger.warning("视频参考能力降级已获人工确认: %s", warning)
            reference_assets = [item for item in reference_assets if item.type == "approved_storyboard_first_frame"]

        reference_manifest = [self._manifest_item(item, capabilities) for item in reference_assets]
        if getattr(capabilities, "reference_image", False) and not any(
            item.get("type") == "approved_storyboard_first_frame" and item.get("loaded") for item in reference_manifest
        ):
            raise RuntimeError("视频生成缺少 approved_storyboard_first_frame（已审核分镜首帧），已阻止纯文本生成")
        shot["seedance_reference_manifest"] = reference_manifest
        prompt = self._build_prompt(shot, characters, scenes)
        content = self._build_content(prompt, shot, capabilities)
        result = await self.generate_single_shot(
            prompt=prompt,
            project_id=project_id,
            shot_id=shot.get("shot_id", "seedance_shot"),
            duration=int(shot.get("duration") or 5),
            ratio=shot.get("output_format", "9:16"),
            resolution=self._resolution(shot.get("resolution")),
            content=content,
            dialogues=self._dialogues_from_shot(shot),
            reference_assets=reference_assets,
            endpoint_override=endpoint,
            provider_source=provider_source,
        )
        report = dict(result.get("generation_report") or self.last_generation_metadata or {})
        if not report:
            matrix = capability_report("video", endpoint.protocol, model=endpoint.model, adapter_cls=adapter_cls)
            sent_detail = []
            if any(item.type == "approved_storyboard_first_frame" for item in reference_assets):
                sent_detail.append({"type": "approved_storyboard_first_frame", "role": "first_frame", "parameter": "content[].role"})
            if getattr(capabilities, "multiple_reference_images", False):
                sent_detail.extend({
                    "type": item.type,
                    "role": item.role or item.type,
                    "parameter": getattr(capabilities, "reference_role_parameter", "") or "content[].role",
                    "weight_policy": matrix.get("reference_weight_policy", "text_only_policy"),
                } for item in reference_assets if item.type != "approved_storyboard_first_frame")
            controls = payload_control_types("video", matrix, has_first_frame=bool(sent_detail))
            report = {
                "provider": endpoint.protocol,
                "model": endpoint.model,
                "provider_source": provider_source,
                "reference_mode": str(getattr(capabilities, "reference_mode", "text_only")),
                "reference_payload_mode": result.get("reference_payload_mode", ""),
                "references_validated": len(reference_manifest),
                "references_sent": [item["type"] for item in sent_detail],
                "references_sent_detail": sent_detail,
                "control_types_sent": controls,
                "provider_capabilities": matrix,
                "capability_states": matrix.get("states", {}),
                "reference_weight_policy": matrix.get("reference_weight_policy", "text_only_policy"),
                "consistency_metrics": payload_metrics(
                    references_validated=len(reference_manifest),
                    references_sent=sent_detail,
                    required_roles=[item.get("role") or item.get("type") for item in sent_detail],
                    control_types_sent=controls,
                    provider_capabilities=matrix,
                ),
            }
        visual = await validate_visual_consistency(
            generated_path=str(result.get("frame_path") or ""),
            reference_paths=[item.source_path for item in reference_assets if item.source_path],
            metric_keys=("character_identity", "shot_continuity"),
        )
        metrics = dict(report.get("consistency_metrics") or {})
        metrics["visual_validation"] = visual
        metrics["claim_scope"] = "request_payload_and_vlm" if visual.get("status") == "passed" else "request_payload_only"
        report["consistency_metrics"] = metrics
        shot["reference_mode"] = str(report.get("reference_mode") or ("multi_reference" if multi_required else "first_frame_only"))
        shot["references_validated"] = len(reference_manifest)
        shot["references_sent"] = list(report.get("references_sent") or [])
        shot["references_sent_detail"] = list(report.get("references_sent_detail") or [])
        shot["control_types_sent"] = list(report.get("control_types_sent") or [])
        shot["provider_capabilities"] = dict(report.get("provider_capabilities") or {})
        shot["reference_weight_policy"] = str(report.get("reference_weight_policy") or "text_only_policy")
        shot["consistency_metrics"] = dict(report.get("consistency_metrics") or {})
        logger.info(
            "镜头视频参考: provider=%s model=%s reference_mode=%s references_validated=%s references_sent=%s "
            "control_types_sent=%s capability_states=%s reference_weight_policy=%s",
            report.get("provider", endpoint.protocol),
            report.get("model", endpoint.model or "default"),
            shot["reference_mode"],
            shot["references_validated"],
            shot["references_sent"],
            shot["control_types_sent"],
            dict(report.get("capability_states") or {}),
            shot["reference_weight_policy"],
        )
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
                        role=str(item.get("role") or ""),
                        text=str(item.get("text") or ""),
                        emotion=str(item.get("emotion") or "neutral"),
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
            if role == "first_frame" or not first_frame:
                first_frame = first_frame or str(url)
            else:
                assets.append(ReferenceAsset(url=str(url), type=role or "reference_image", role=role or "reference_image"))
        return (first_frame or None), assets, ("first_frame_reference" if first_frame else "text_only")

    def _resolution(self, resolution: str | None) -> str:
        # 项目分辨率可为 720p/1080p/2k/4k；部分 Seedance 型号仅支持到 1080p，
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

        fields: list[tuple[str, str]] = [
            ("effective_style", style_params.get("video_prompt", "")),
            ("style_label", f"visual style prompt preference: {style_params.get('style_label', '')}"),
            ("identity_policy", "identity and style prompt preferences only, not model hard constraints"),
        ]
        # --- 2. 人物身份 ---
        for char in selected_characters:
            appearance = char.get("appearance") or {}
            appearance_parts = [str(value) for value in appearance.values()] if isinstance(appearance, dict) else []
            reference_lock = "prefer consistency with the approved three-view character sheet" if char.get("reference_images") else ""
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
            ("first_frame_policy", "prompt preference: stay visually close to the approved storyboard first frame when the Provider supports it"),
            ("visual_notes", shot.get("visual_notes", "")),
        ])
        if shot.get("storyboard_path") or shot.get("image_path"):
            fields.append(
                (
                    "reference_mode",
                    f"{provider_label} reference delivery is recorded in provider_capabilities and references_sent",
                )
            )
        # --- 4. 动作、情绪与运镜 ---
        fields.extend([
            ("character_action", shot.get("character_action", "")),
            ("emotion", f"emotional tone: {shot.get('emotion', 'neutral')}"),
            ("camera_movement", f"camera movement: {shot.get('camera_movement', '静止')}"),
            ("camera_angle", f"camera angle: {shot.get('camera_angle', '正面')}"),
            ("shot_type", f"shot size: {shot.get('shot_type', 'medium')}"),
            ("motion_policy", "cinematic short drama video, coherent motion, no subtitles, no watermark"),
        ])
        # --- 5. 连续性规则与低优先级 SOP ---
        if shot.get("continuity_reference_path"):
            fields.append(("continuity", "previous shot final frame may inform eye-line and action continuity in text"))
        if shot.get("reference_weights"):
            fields.append(("reference_weights", "reference weight policy: text_only_policy; no numeric model weight is implied"))
        if shot.get("reference_assets"):
            roles = ", ".join(str(item.get("role", "")) for item in shot.get("reference_assets", []) if isinstance(item, dict))
            fields.append(("reference_roles", f"validated reference candidate roles: {roles}; actual sending follows {provider_label} Capability Matrix"))
        if shot.get("seedance_reference_manifest"):
            manifest = shot.get("seedance_reference_manifest") or []
            loaded = ", ".join(str(item.get("type", "")) for item in manifest if isinstance(item, dict))
            fields.append(("validated_assets", f"validated reference candidates: {loaded}; actual sending is recorded in provider_capabilities and references_sent"))
        if shot.get("continuity_profile"):
            profile = shot.get("continuity_profile") or {}
            fields.append(("continuity_profile", f"continuity prompt preferences: {', '.join(profile.get('editing_logic', []))}; no OpenPose/Depth control is sent"))
            blocking = profile.get("character_blocking") or {}
            if blocking:
                order = blocking.get("character_order_left_to_right") or []
                fields.append(("blocking",
                    "character blocking prompt preference: "
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
        """人物身份、动作、情绪、运镜：裁剪后必须仍然存在的关键字段。"""
        critical: list[tuple[str, str]] = []
        for char in selected_characters[:2]:
            critical.append(("character_identity", str(char.get("visual_prompt", "") or char.get("name", ""))))
        critical.extend([
            ("character_action", str(shot.get("character_action", "") or "")),
            ("emotion", f"emotional tone: {shot.get('emotion', 'neutral')}"),
            ("camera_movement", f"camera movement: {shot.get('camera_movement', '静止')}"),
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

    def _build_content(self, prompt: str, shot: dict, capabilities=None) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": prompt}]

        if capabilities is not None and not getattr(capabilities, "reference_image", False):
            return content

        first_frame = shot.get("storyboard_path") or shot.get("image_path") or ""
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

    def _video_reference_assets(self, shot: dict) -> list[ReferenceAsset]:
        """收集视频请求候选参考，并只保留可读文件。首帧永远单独建模。"""

        assets: list[ReferenceAsset] = []
        first = str(shot.get("storyboard_path") or shot.get("image_path") or "")
        first_url = self.reference_assets.to_image_url(first) if first else ""
        if first_url:
            assets.append(ReferenceAsset(url=first_url, type="approved_storyboard_first_frame", role="first_frame", provider_type="first_frame", source_path=first))
        candidates: list[tuple[str, str, str]] = []
        for item in shot.get("reference_assets") or []:
            if isinstance(item, dict) and item.get("path"):
                candidates.append((str(item.get("type") or "reference_image"), str(item.get("role") or ""), str(item["path"])))
        for path in shot.get("scene_reference_images") or []:
            if path:
                candidates.append(("scene_baseline", "environment_props_lighting_perspective", str(path)))
        for path in shot.get("character_reference_images") or []:
            if path:
                candidates.append(("character_three_view", "identity_outfit_face_body_hair", str(path)))
        if shot.get("continuity_reference_path"):
            candidates.append(("continuity_frame", "eye_line_axis_motion", str(shot["continuity_reference_path"])))
        seen = {(item.type, item.source_path) for item in assets}
        for kind, role, path in candidates:
            key = (kind, path)
            if key in seen:
                continue
            seen.add(key)
            url = self.reference_assets.to_image_url(path)
            if not url:
                continue
            assets.append(ReferenceAsset(url=url, type=kind, role=role or kind, provider_type="reference_image", source_path=path))
        return assets

    @staticmethod
    def _manifest_item(asset: ReferenceAsset, capabilities=None) -> dict:
        multi = bool(getattr(capabilities, "multiple_reference_images", False))
        sent = (asset.type == "approved_storyboard_first_frame" and bool(getattr(capabilities, "reference_image", False))) or multi
        return {
            "type": asset.type,
            "role": asset.role or asset.type,
            "path": asset.source_path,
            "loaded": True,
            "validated": True,
            "sent": sent,
            "not_sent_reason": "" if sent else "provider_first_frame_only",
        }

    def _validate_video_references(self, shot: dict) -> list[dict]:
        """返回候选参考清单；只有首帧在 first_frame_only 协议下标记为已发送。"""

        assets = self._video_reference_assets(shot)
        if not any(item.type == "approved_storyboard_first_frame" for item in assets):
            raise RuntimeError("视频生成缺少必需一致性参考素材: approved_storyboard_first_frame:<empty>")
        manifest = []
        for asset in assets:
            sent = asset.type == "approved_storyboard_first_frame"
            manifest.append(
                {
                    "type": asset.type,
                    "role": asset.role or asset.type,
                    "path": asset.source_path,
                    "loaded": True,
                    "validated": True,
                    "sent": sent,
                    "not_sent_reason": "" if sent else "provider_first_frame_only",
                }
            )
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
