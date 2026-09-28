import asyncio
import base64  # noqa: F401  (kept for reference-asset data URLs)
import logging
import re
import time

from PIL import Image
from io import BytesIO

from config import settings
from services import usage_service
from services.consistency_service import ConsistencyService
from services.providers.base import ImageRequest
from services.providers.endpoint import (
    KNOWN_PROTOCOLS,
    EndpointConfig,
    get_endpoint,
    image_protocol_defaults,
)
from services.providers.image_placeholder import PlaceholderImageAdapter
from services.providers.registry import UnknownProtocolError, get_adapter
from services.providers.usage import (
    CAPABILITY_IMAGE,
    ERROR_CODE_PROVIDER_CALL_FAILED,
    adapter_usage_for_request,
)
from services.reference_asset_service import ReferenceAssetService
from services.prompt_budget import assemble_prompt, dedupe_terms, ensure_critical_fields, remove_conflicting_terms
from services.security import atomic_write_bytes, safe_path, validate_identifier
from services.storage_service import StorageQuotaExceeded, StorageService
from services.style_templates import style_prompt_params

logger = logging.getLogger(__name__)


class ImageService:
    """Image generation service routing to protocol adapters via endpoint config."""

    def __init__(self):
        self.output_dir = settings.OUTPUT_DIR / "projects"
        self.consistency = ConsistencyService()
        self.reference_assets = ReferenceAssetService()
        self.storage = StorageService()
        self.last_prompt_trimmed_fields: list[str] = []
        self.last_prompt_readded_fields: list[str] = []
        self.last_prompt_conflicts: list[str] = []
        self.last_generation_metadata: dict[str, object] = {}

    # ------------------------------------------------------------------
    # 适配器路由：protocol 显式决定代码路径；协议非法或缺 key 回退占位图。
    # ------------------------------------------------------------------

    def _reference_enforcement(self) -> str:
        """参考图能力策略：prefer（默认，明确告警后继续）/ strict（阻止生成）。"""

        value = str(getattr(settings, "IMAGE_REFERENCE_ENFORCEMENT", "prefer") or "prefer").strip().lower()
        return value if value in {"prefer", "strict"} else "prefer"

    def _adapter_capabilities(self, protocol: str):
        try:
            adapter_cls = get_adapter("image", protocol)
        except UnknownProtocolError:
            return None
        return getattr(adapter_cls, "capabilities", None)

    def _preferred_reference_endpoint(self, primary: EndpointConfig) -> EndpointConfig | None:
        """找「账号已配置 + 适配器声明支持参考图」的替代图像端点。

        当前 Provider 不支持参考图时，优先改用这样的端点，避免角色/场景/故事板
        阶段的参考资产被丢弃。只使用 settings/.env 里确实配了密钥的协议，
        不会去调用未配置的厂商。
        """
        for protocol in KNOWN_PROTOCOLS.get("image", ()):
            if protocol == primary.protocol:
                continue
            capabilities = self._adapter_capabilities(protocol)
            if capabilities is None or not capabilities.reference_images:
                continue
            endpoint = image_protocol_defaults(protocol)
            if not str(endpoint.api_key or '').strip():
                continue
            return endpoint
        return None

    def _resolve_route(self) -> tuple[object, EndpointConfig]:
        """按配置的 image 端点解析适配器（协议非法或缺 key 回退占位图）。"""

        endpoint = get_endpoint("image")
        try:
            adapter_cls = get_adapter("image", endpoint.protocol)
        except UnknownProtocolError as exc:
            logger.warning("图像协议 %r 未注册适配器，回退占位图: %s", endpoint.protocol, exc)
            return PlaceholderImageAdapter(EndpointConfig(protocol="placeholder")), endpoint
        capabilities = adapter_cls.capabilities
        if capabilities.requires_credentials and not endpoint.api_key:
            logger.warning(
                "图像协议 %s 未配置 API Key，本次生成回退占位图（配置密钥后自动启用云端出图）",
                endpoint.protocol,
            )
            return PlaceholderImageAdapter(EndpointConfig(protocol="placeholder")), endpoint
        return self._build_adapter(adapter_cls, endpoint), endpoint

    @staticmethod
    def _build_adapter(adapter_cls, endpoint: EndpointConfig):
        try:
            return adapter_cls(endpoint)
        except TypeError:
            # Lightweight plugin/test adapters may not require endpoint state.
            return adapter_cls()

    def _upgrade_to_reference_provider(
        self, adapter, endpoint: EndpointConfig
    ) -> tuple[object, EndpointConfig, str]:
        """当前 Provider 不支持参考图时，优先切换到已配置的参考图 Provider。

        返回 (adapter, endpoint, provider_source)。找不到可用替代时原样返回，
        由调用方按 IMAGE_REFERENCE_ENFORCEMENT 决定告警后继续还是阻止生成。
        """
        if getattr(adapter.capabilities, "reference_images", False):
            return adapter, endpoint, "configured"
        alternate = self._preferred_reference_endpoint(endpoint)
        if alternate is None:
            return adapter, endpoint, "configured"
        logger.warning(
            "图像 Provider %s（model=%s）不支持参考图，本次改用已配置且支持参考图的 %s（model=%s）",
            endpoint.protocol,
            endpoint.model or "default",
            alternate.protocol,
            alternate.model or "default",
        )
        return (
            self._build_adapter(get_adapter("image", alternate.protocol), alternate),
            alternate,
            "preferred_reference_provider",
        )

    async def _generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
        reference_images: list[str],
        preferred_size: str,
        label: str,
        shot_id: str = "",
        allow_text_only_references: bool = False,
    ) -> bytes:
        """生成一张图。

        ``allow_text_only_references`` 由调用方按产品策略传入：默认 False 时，
        参考图不可用会直接报错（低层原语的严格契约）；服务入口在 prefer 策略下
        传 True，此时会明确告警、如实记录 references_sent=0 后继续生成，
        但绝不假装参考图已生效。
        """
        requires_references = bool(reference_images)
        adapter, endpoint = self._resolve_route()
        provider_source = "configured"
        if requires_references:
            adapter, endpoint, provider_source = self._upgrade_to_reference_provider(adapter, endpoint)
        references_unsupported = requires_references and not adapter.capabilities.reference_images
        reference_warning = ''
        if references_unsupported:
            reference_warning = (
                f"图像 Provider {endpoint.protocol}/{endpoint.model or 'default'} 声明不支持参考图，"
                f"本次 {len(reference_images)} 张参考图（角色三视图/场景基准图/续帧）不会发送给模型；"
                "请在「系统设置 → 模型服务」切换到支持参考图的图像 Provider（如 ark-seedream）。"
            )
            if not allow_text_only_references:
                logger.warning(reference_warning)
                raise RuntimeError(reference_warning)
            logger.warning("参考图未生效（已如实降级为纯文本生成）: %s", reference_warning)
        size = preferred_size or str(endpoint.param("image_size") or "")
        request = ImageRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            # 适配器声明支持参考图时才传入，不再硬编码假设。
            reference_images=list(reference_images) if adapter.capabilities.reference_images else [],
            size=size,
            label=label,
        )
        self.last_generation_metadata = {
            "provider": endpoint.protocol,
            "model": endpoint.model,
            "provider_source": provider_source,
            "reference_mode": ("multi_reference" if adapter.capabilities.reference_images else "text_only") if reference_images else "text_only",
            "references_validated": len(reference_images),
            "references_sent": len(request.reference_images),
            "references_unsupported": bool(references_unsupported),
            "reference_capability_warning": reference_warning,
        }
        logger.info(
            "图像生成请求: provider=%s model=%s provider_source=%s reference_mode=%s "
            "references_validated=%s references_sent=%s references_unsupported=%s label=%s",
            endpoint.protocol,
            endpoint.model or "default",
            provider_source,
            self.last_generation_metadata["reference_mode"],
            self.last_generation_metadata["references_validated"],
            self.last_generation_metadata["references_sent"],
            self.last_generation_metadata["references_unsupported"],
            label,
        )
        # 用量按「实际调用的适配器」记账：回退到占位图时 provider 记为 placeholder，
        # 不会被误记成已配置但未真正调用的云端 provider。
        metadata = adapter_usage_for_request(adapter, CAPABILITY_IMAGE, request)
        scope = usage_service.current_scope().merged(shot_id=shot_id)
        started = time.monotonic()
        try:
            image_data = await adapter.generate(request)
        except asyncio.CancelledError:
            # 任务被取消：调用已经发出，同样要留痕（金额未知，不虚增）。
            usage_service.record_cancelled(
                metadata,
                duration_ms=int((time.monotonic() - started) * 1000),
                scope=scope,
            )
            raise
        except Exception:
            usage_service.record_failure(
                metadata,
                error_code=ERROR_CODE_PROVIDER_CALL_FAILED,
                duration_ms=int((time.monotonic() - started) * 1000),
                scope=scope,
            )
            raise
        if not image_data:
            usage_service.record_failure(
                metadata,
                error_code=ERROR_CODE_PROVIDER_CALL_FAILED,
                duration_ms=int((time.monotonic() - started) * 1000),
                scope=scope,
            )
            raise RuntimeError("图像生成接口未返回图片数据")
        usage_service.record_metadata(
            metadata,
            duration_ms=int((time.monotonic() - started) * 1000),
            scope=scope,
        )
        return image_data

    async def generate_shot_image(
        self,
        shot: dict,
        characters: list,
        style_params: dict,
        project_id: str,
        seed: int = 42,
    ) -> str:
        prompt, negative_prompt = self._build_prompt(shot, characters, style_params)
        reference_images = self._reference_images_for_request(shot)

        try:
            safe_project_id = validate_identifier(project_id, "项目 ID")
            safe_shot_id = validate_identifier(str(shot["shot_id"]), "镜头 ID")
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        shot_dir = safe_path(self.output_dir, safe_project_id, "shots", create_parent=True)
        image_path = shot_dir / f"{safe_shot_id}_v{int(shot.get('version', 1) or 1)}.png"

        preferred_size = self._size_for_ratio(shot.get("output_format"))
        image_data = await self._generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            reference_images=reference_images,
            preferred_size=preferred_size,
            label="SHOT PLACEHOLDER",
            shot_id=safe_shot_id,
            allow_text_only_references=self._reference_enforcement() != "strict",
        )

        self._validate_image(image_data)
        self._write_image(project_id, image_path, image_data)
        return str(image_path)

    async def generate_scene_baseline_reference(
        self,
        scene: dict,
        style: str,
        project_id: str,
        seed: int = 1200,
    ) -> str:
        prompt, negative_prompt = self.consistency.scene_baseline_prompt(scene, style)
        style_params = style_prompt_params(style)
        prompt = ", ".join(part for part in [style_params.get("scene_baseline_prompt", ""), prompt] if part)
        negative_prompt = ", ".join(part for part in [negative_prompt, style_params.get("negative_prompt", "")] if part)

        try:
            safe_project_id = validate_identifier(project_id, "项目 ID")
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        ref_dir = safe_path(self.output_dir, safe_project_id, "scenes", create_parent=True)
        safe_key = scene.get("id") or scene.get("scene_group_key") or scene.get("location") or scene.get("name", "scene")
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(safe_key)).strip("_") or "scene"
        scene_dir = ref_dir / safe_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        image_path = scene_dir / "baseline_original.png"
        preferred_size = str(get_endpoint("image").param("image_size") or "1440x2560")

        image_data = await self._generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            reference_images=[],
            preferred_size=preferred_size,
            label="SCENE BASELINE",
            allow_text_only_references=self._reference_enforcement() != "strict",
        )

        self._validate_image(image_data)
        self._write_image(project_id, image_path, image_data)
        return str(image_path)

    async def generate_character_reference(
        self,
        character: dict,
        style: str,
        project_id: str,
        seed: int = 42,
    ) -> str:
        prompt, negative_prompt = self._build_character_reference_prompt(character, style)

        try:
            safe_project_id = validate_identifier(project_id, "项目 ID")
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        ref_dir = safe_path(self.output_dir, safe_project_id, "characters", create_parent=True)
        identity_key = character.get("id") or character.get("asset_id") or character.get("name", "character")
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(identity_key)).strip("_") or "character"
        character_dir = ref_dir / safe_name
        character_dir.mkdir(parents=True, exist_ok=True)
        image_path = character_dir / "three_view_original.png"
        preferred_size = "2048x2048"

        image_data = await self._generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            reference_images=[],
            preferred_size=preferred_size,
            label="CHARACTER REF",
            allow_text_only_references=self._reference_enforcement() != "strict",
        )

        self._validate_image(image_data)
        self._write_image(project_id, image_path, image_data)
        return str(image_path)

    def _seedream_payload(self, model: str, prompt: str, negative_prompt: str, seed: int, size: str, reference_images: list[str] | None = None) -> dict:
        """兼容保留：sop 校验脚本使用。实际载荷由 ark-seedream 适配器构造。"""
        from services.providers.image_ark_seedream import ArkSeedreamImageAdapter

        request = ImageRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            reference_images=list(reference_images or []),
            size=size,
        )
        return ArkSeedreamImageAdapter(EndpointConfig(protocol="ark-seedream"))._payload(model, request, size)

    def _size_for_ratio(self, output_format: str | None) -> str:
        # 画面比例 -> Seedream 出图尺寸。未识别的比例回退到配置的默认尺寸，
        # 保证用户在前端切换 9:16 / 16:9 / 1:1 等比例后，定稿故事板真实按比例出图。
        # 尺寸需满足 Seedream 系列官方总像素范围 [2560x1440, 4096x4096]，
        # 且不能超过 5.0 pro 的上限（约 2048x2048×1.1），故按比例就近放大。
        ratio = str(output_format or "").strip()
        ratio_size_map = {
            "9:16": "1440x2560",
            "3:4": "1728x2304",
            "1:1": "2048x2048",
            "4:3": "2304x1728",
            "16:9": "2560x1440",
        }
        return ratio_size_map.get(ratio, str(get_endpoint("image").param("image_size") or "1440x2560"))

    def build_shot_prompt(self, shot: dict, characters: list, style_params: dict) -> tuple[str, str]:
        return self._build_prompt(shot, characters, style_params)

    def _write_image(self, project_id: str, image_path, image_data: bytes) -> None:
        try:
            self.storage.ensure_project_capacity(project_id, len(image_data), replacing=image_path)
        except StorageQuotaExceeded as exc:
            raise RuntimeError("项目媒体存储空间不足") from exc
        atomic_write_bytes(image_path, image_data, minimum_size=1024)

    def _validate_image(self, image_data: bytes) -> None:
        """结构检查（非质量认证）：图片可打开且尺寸达到可用下限。"""
        try:
            with Image.open(BytesIO(image_data)) as image:
                image.verify()
        except Exception as exc:
            raise RuntimeError(f"图像数据无法打开: {exc}") from exc
        try:
            with Image.open(BytesIO(image_data)) as image:
                width, height = image.size
        except Exception as exc:
            raise RuntimeError(f"图像尺寸无法读取: {exc}") from exc
        if min(width, height) < 64:
            raise RuntimeError(f"图像尺寸过小（{width}x{height}），判定为生成失败")

    def _build_prompt(self, shot: dict, characters: list, style_params: dict) -> tuple[str, str]:
        """按预算优先级组装图像 Prompt。

        字段顺序即裁剪优先级（从高到低）：
        1. 风格与负向约束；2. 人物身份/脸型/发型/服装/关键特征；
        3. 场景、构图与首帧参考说明；4. 动作/情绪/景别/机位；
        5. 连续性规则与低优先级 SOP。超预算时从末尾整字段丢弃，
        绝不从字段中间硬截断。
        """
        fields: list[tuple[str, str]] = []
        negative_parts: list[str] = []

        # --- 1. 风格与负向约束 ---
        if style_params.get("prompt_prefix"):
            fields.append(("effective_style", style_params["prompt_prefix"]))
        if style_params.get("style_label"):
            fields.append(("style_label", f"locked visual style preset: {style_params['style_label']}"))
        if style_params.get("negative_prompt"):
            negative_parts.append(style_params["negative_prompt"])

        # --- 2. 人物身份、年龄、脸型、发型、服装和关键特征 ---
        selected_cards = self._select_character_cards(shot, characters)
        for char_card in selected_cards:
            fields.append(("character_identity", char_card.get("visual_prompt", "")))
            fields.append(("character_features", ", ".join(char_card.get("key_features", []))))
            appearance = char_card.get("appearance") or {}
            if isinstance(appearance, dict):
                fields.append(("character_appearance", ", ".join(str(value) for value in appearance.values() if value)))
            if char_card.get("reference_images"):
                fields.append(("character_identity_lock", "preserve identity from the approved character three-view reference sheet"))
            if char_card.get("wardrobe_lock"):
                fields.append(("wardrobe", char_card["wardrobe_lock"]))
            emotion = shot.get("emotion", "neutral")
            if char_card.get("emotion_variants", {}).get(emotion):
                fields.append(("emotion", char_card["emotion_variants"][emotion]))
            if char_card.get("negative_prompt"):
                negative_parts.append(char_card["negative_prompt"])

        # --- 3. 场景、构图和首帧参考说明 ---
        if shot.get("scene_reference_images"):
            fields.append(("scene_reference", "scene baseline/reference assets are loaded for environment, props, lighting and perspective"))
        if shot.get("character_reference_images"):
            fields.append(("character_reference", "character three-view reference assets are loaded for identity, outfit, face and hairstyle"))
        if shot.get("continuity_reference_path"):
            fields.append(("continuity_reference", "previous shot final frame reference anchors eye-line, pose, axis and depth continuity"))
        fields.extend([
            ("scene", shot.get("scene_description", "")),
            ("approved_storyboard", shot.get("storyboard_prompt", "")),
        ])

        # --- 4. 动作、情绪与镜头语言 ---
        fields.extend([
            ("character_action", shot.get("character_action", "")),
            ("shot_type", self._camera_prompt(shot.get("shot_type", "medium"))),
            ("camera_angle", self._angle_prompt(shot.get("camera_angle", "正面"))),
            ("camera_movement", f"camera movement: {shot.get('camera_movement', '静止')}"),
            ("visual_notes", shot.get("visual_notes", "")),
            ("finish", "finished production keyframe, expressive human acting, clean composition, high detail"),
        ])

        # --- 5. 连续性规则和低优先级 SOP ---
        fields.append(("identity_policy", "NON-NEGOTIABLE identity and style consistency policy"))
        if shot.get("reference_weights"):
            weights = shot.get("reference_weights") or {}
            fields.append(("reference_weights",
                f"apply locked reference weights: environment/style {float(weights.get('environment') or 0.45):.2f}, character/action {float(weights.get('action') or 0.30):.2f}"
            ))
        if shot.get("reference_assets"):
            roles = ", ".join(str(item.get("role", "")) for item in shot.get("reference_assets", []) if isinstance(item, dict))
            fields.append(("reference_roles", f"mandatory persisted reference assets drive these roles: {roles}"))
        if shot.get("continuity_profile"):
            profile = shot.get("continuity_profile") or {}
            fields.append(("continuity_rules",
                "locked continuity controls: "
                f"{', '.join(profile.get('editing_logic', []))}; "
                f"OpenPose {profile.get('openpose_lock', 'unsupported')}; "
                f"Depth {profile.get('depth_lock', 'unsupported')}; "
                f"LUT {profile.get('lut', 'project_scene_lut_locked')}; "
                f"{profile.get('ambient_audio_policy', '')}"
            ))
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

        negative_parts.extend(["low quality", "blurry", "watermark", "text artifacts", "bad anatomy"])
        cleaned = [(name, self._clean_prompt_part(value)) for name, value in fields]
        prompt, dropped = assemble_prompt(cleaned)
        self.last_prompt_trimmed_fields = dropped
        critical = [
            ("character_identity", str(card.get("visual_prompt", "") or card.get("name", "")))
            for card in selected_cards[:2]
        ]
        critical.append(("character_action", str(shot.get("character_action", "") or "")))
        prompt, readded = ensure_critical_fields(prompt, critical)
        self.last_prompt_readded_fields = readded
        if dropped or readded:
            logger.info(
                "图像 Prompt 预算裁剪: dropped_fields=%s readded_critical_fields=%s",
                dropped,
                readded,
            )
        prompt, conflicts = remove_conflicting_terms(prompt, negative_parts)
        self.last_prompt_conflicts = conflicts
        if conflicts:
            logger.info("图像 Prompt 正负向冲突词已移除: %s", conflicts)
        return prompt, ", ".join(dedupe_terms(negative_parts))

    def _select_character_cards(self, shot: dict, characters: list) -> list[dict]:
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

    def _reference_images_for_request(self, shot: dict) -> list[str]:
        refs: list[str] = []
        for asset in shot.get("reference_assets") or []:
            if not isinstance(asset, dict):
                continue
            path = asset.get("path")
            if path:
                refs.append(path)
        for key in ("scene_reference_images", "character_reference_images"):
            values = shot.get(key) or []
            if isinstance(values, str):
                values = [values]
            refs.extend(value for value in values if value)
        if shot.get("continuity_reference_path"):
            refs.append(shot["continuity_reference_path"])
        # pose/depth paths from older projects were diagnostic PIL images, not
        # real OpenPose/Depth controls. Never send them to an image Provider.

        image_urls: list[str] = []
        seen: set[str] = set()
        for ref in refs:
            if ref in seen:
                continue
            seen.add(ref)
            image_url = self.reference_assets.to_image_url(ref)
            if image_url:
                image_urls.append(image_url)
            if len(image_urls) >= 8:
                break
        return image_urls

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

    def _build_character_reference_prompt(self, character: dict, style: str) -> tuple[str, str]:
        appearance = character.get("appearance") if isinstance(character.get("appearance"), dict) else {}
        features = character.get("key_features") or []
        style_params = style_prompt_params(style)
        style_prompt = style_params.get("character_reference_prompt", "")
        prompt_parts = [
            style_prompt,
            "high-resolution original character asset, no thumbnail, production reference quality",
            "standard three-view reference sheet, front view, side view, back view",
            "same character identity across all views, neutral pose, full body, plain background",
            character.get("id", "") and f"unique character asset id: {character.get('id')}",
            character.get("name", ""),
            character.get("visual_prompt", ""),
            character.get("personality", ""),
            *[str(value) for value in appearance.values() if value],
            *[str(value) for value in features if value],
        ]
        negative_parts = [
            character.get("negative_prompt", ""),
            style_params.get("negative_prompt", ""),
            "different outfits between views, inconsistent face, extra characters, watermark, text artifacts, low quality",
        ]
        prompt, dropped = assemble_prompt([(f"character_{index}", part) for index, part in enumerate(prompt_parts)])
        self.last_prompt_trimmed_fields = dropped
        prompt, _ = remove_conflicting_terms(prompt, negative_parts)
        return prompt, ", ".join(dedupe_terms(negative_parts))

    def _camera_prompt(self, shot_type: str) -> str:
        return {
            "wide": "wide establishing shot",
            "medium": "medium shot",
            "close-up": "close-up shot",
            "extreme_close": "extreme close-up",
        }.get(shot_type, "medium shot")

    def _angle_prompt(self, camera_angle: str) -> str:
        return {
            "正面": "front view",
            "侧面": "side view",
            "俯视": "high angle",
            "仰视": "low angle",
        }.get(camera_angle, "front view")
