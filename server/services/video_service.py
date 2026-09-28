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
from pathlib import Path

from config import settings
from services import usage_service
from services.consistency_service import ConsistencyService
from services.providers.base import Dialogue, VideoRequest
from services.providers.endpoint import get_endpoint
from services.providers.registry import get_adapter
from services.providers.usage import (
    CAPABILITY_VIDEO,
    ERROR_CODE_PROVIDER_CALL_FAILED,
    adapter_usage_for_request,
)
from services.prompt_budget import assemble_prompt, ensure_critical_fields
from services.reference_asset_service import ReferenceAssetService
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
    ) -> dict[str, str]:
        endpoint = get_endpoint("video")
        adapter = get_adapter("video", endpoint.protocol)(endpoint)
        if not prompt.strip():
            raise RuntimeError("视频生成提示词为空")

        if content is None:
            content = [{"type": "text", "text": prompt}]
        reference_image, content_payload_mode = self._reference_from_content(content)

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
            # 固定时长协议不接受更长镜头：明确报错要求拆分，而不是静默截短。
            raise RuntimeError(
                f"镜头时长 {requested_duration} 秒超过视频 Provider {endpoint.protocol} "
                f"的固定时长 {int(fixed_duration)} 秒；请把该镜头拆分为多个短镜头，"
                "或把时长调整到固定时长以内"
            )
        request = VideoRequest(
            prompt=prompt,
            reference_image=reference_image,
            dialogues=dialogues,
            duration=int(fixed_duration or requested_duration),
            ratio=ratio,
            resolution=resolution,
            project_id=safe_project_id,
            output_video_path=video_path,
            output_frame_path=frame_path,
        )
        self.last_generation_metadata = {
            "provider": endpoint.protocol,
            "model": endpoint.model,
            "reference_mode": adapter.capabilities.reference_mode if reference_image else "text_only",
            "references_validated": 1 if reference_image else 0,
            "references_sent": ["approved_storyboard_first_frame"] if reference_image else [],
        }
        logger.info(
            "视频生成请求: provider=%s model=%s reference_mode=%s duration=%s shot=%s",
            endpoint.protocol,
            endpoint.model or "default",
            self.last_generation_metadata["reference_mode"],
            request.duration,
            safe_shot_id,
        )
        # 视频按「秒 x 分辨率」记账；轮询式协议耗时很长，调用耗时单独记录。
        metadata = adapter_usage_for_request(adapter, CAPABILITY_VIDEO, request)
        scope = usage_service.current_scope().merged(project_id=safe_project_id, shot_id=safe_shot_id)
        started = time.monotonic()
        try:
            result = await adapter.generate(request)
        except asyncio.CancelledError:
            # 轮询式视频任务被取消：调用确实发生过，留痕但不虚增金额。
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
        usage_service.record_metadata(
            metadata,
            duration_ms=int((time.monotonic() - started) * 1000),
            extra_units={"task_id": str(result.task_id or "")},
            scope=scope,
        )

        if result.native_audio:
            # 原生音频契约校验：不允许产出无声成品。
            await self._assert_stream_has_audio(result.video_path)

        return {
            "video_path": result.video_path,
            "frame_path": result.frame_path,
            "task_id": result.task_id,
            "reference_payload_mode": result.payload_mode or content_payload_mode,
            "native_audio": bool(result.native_audio),
        }

    async def generate_shot_video(
        self,
        shot: dict,
        characters: list[dict],
        scenes: dict[str, dict],
        project_id: str,
    ) -> dict[str, str]:
        endpoint = get_endpoint("video")
        adapter_cls = get_adapter("video", endpoint.protocol)
        capabilities = adapter_cls.capabilities

        reference_manifest: list[dict] = []
        if capabilities.reference_image:
            reference_manifest = self._validate_video_references(shot)
            shot["seedance_reference_manifest"] = reference_manifest
        prompt = self._build_prompt(shot, characters, scenes)
        content = self._build_content(prompt, shot, capabilities)
        shot["reference_mode"] = capabilities.reference_mode if capabilities.reference_image else "text_only"
        shot["references_validated"] = len(reference_manifest)
        shot["references_sent"] = ["approved_storyboard_first_frame"] if self._has_image_content(content) else []
        if capabilities.reference_image and not self._has_image_content(content):
            raise RuntimeError("视频生成缺少已审核分镜首帧参考图，已阻止纯文本生成")
        logger.info(
            "镜头视频参考: provider=%s model=%s reference_mode=%s "
            "references_validated=%s references_sent=%s（场景基准图/角色三视图/OpenPose/Depth 均不发送）",
            endpoint.protocol,
            endpoint.model or "default",
            shot["reference_mode"],
            shot["references_validated"],
            shot["references_sent"],
        )
        return await self.generate_single_shot(
            prompt=prompt,
            project_id=project_id,
            shot_id=shot.get("shot_id", "seedance_shot"),
            duration=int(shot.get("duration") or 5),
            ratio=shot.get("output_format", "9:16"),
            resolution=self._resolution(shot.get("resolution")),
            content=content,
            dialogues=self._dialogues_from_shot(shot),
        )

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

    def _reference_from_content(self, content: list[dict]) -> tuple[str | None, str]:
        """从内容列表中取首帧参考图；payload 模式由适配器按内容判定。"""
        for item in content or []:
            if item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url") if isinstance(item.get("image_url"), dict) else None
                return (url or None), ""
        return None, "text_only"

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

        fields: list[tuple[str, str]] = [
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
        if shot.get("storyboard_path") or shot.get("image_path"):
            fields.append(
                (
                    "reference_mode",
                    f"{provider_label} first_frame_only: attached image 1 is the approved storyboard first frame",
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

        add_reference("approved_storyboard_first_frame", shot.get("storyboard_path") or shot.get("image_path") or "")
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
