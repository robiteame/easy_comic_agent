import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from services.shot_dialogue import dialogue_plain_text, parse_shot_dialogue
from services.style_templates import style_template

CONTINUITY_MODES = (
    "independent",
    "same_scene",
    "continuous_action",
    "reverse_shot",
    "scene_transition",
)

_CONTINUITY_MODE_ALIASES = {
    "independent": "independent",
    "standalone": "independent",
    "separate shot": "independent",
    "独立": "independent",
    "独立镜头": "independent",
    "same scene": "same_scene",
    "same-scene": "same_scene",
    "同场景": "same_scene",
    "同一场景": "same_scene",
    "场景连续": "same_scene",
    "continuous action": "continuous_action",
    "continuous-action": "continuous_action",
    "match on action": "continuous_action",
    "match-on-action": "continuous_action",
    "previous final frame": "continuous_action",
    "previous-final-frame": "continuous_action",
    "previous_final_frame": "continuous_action",
    "previous last frame": "continuous_action",
    "current scene baseline": "same_scene",
    "current_scene_baseline": "same_scene",
    "none": "independent",
    "action continuity": "continuous_action",
    "连续动作": "continuous_action",
    "动作连续": "continuous_action",
    "动作衔接": "continuous_action",
    "reverse shot": "reverse_shot",
    "reverse-shot": "reverse_shot",
    "reverse": "reverse_shot",
    "正反打": "reverse_shot",
    "反打": "reverse_shot",
    "反打镜头": "reverse_shot",
    "对切": "reverse_shot",
    "scene transition": "scene_transition",
    "scene-transition": "scene_transition",
    "scene change": "scene_transition",
    "cross scene": "scene_transition",
    "cross-scene": "scene_transition",
    "转场": "scene_transition",
    "场景转场": "scene_transition",
    "跨场景": "scene_transition",
    "换场": "scene_transition",
}

_CONTINUITY_REFERENCE_ONLY_MODE = "continuous_action"


def normalize_continuity_mode(value: Any, default: str = "independent") -> str:
    """把分镜/旧数据中的连续性写法收敛到五种规范模式。"""

    text = "" if value is None else str(value).strip()
    if not text:
        return default
    canonical = text.lower().replace("_", " ").replace("-", " ").strip()
    if canonical in CONTINUITY_MODES:
        return canonical
    if canonical in _CONTINUITY_MODE_ALIASES:
        return _CONTINUITY_MODE_ALIASES[canonical]
    raw = text.lower().replace("-", "_").replace(" ", "_")
    return raw if raw in CONTINUITY_MODES else default


class ConsistencyService:
    """Agent-level visual consistency SOP shared by image and video generation."""

    SCENE_SOP = (
        "Prompt preference only, not a model hard constraint: keep the scene group visually coherent in color temperature, "
        "light direction, light intensity, weather, ambient mood, spatial perspective, props, LUT, saturation and sharpness. "
        "Prefer no unrequested set-dressing changes and keep day/night scene groups visually separate."
    )
    CHARACTER_SOP = (
        "Prompt preference only, not a model hard constraint: keep character face, body, hairstyle, skin tone, base outfit, "
        "makeup and accessories coherent with the supplied character references unless the script requests a costume change. "
        "Use scene lighting for natural character shading."
    )
    CONTINUITY_SOP = (
        "Prompt preference only, not a model hard constraint: favor eye-line continuity, reverse-shot eyelines, scene "
        "transitions, match-on-action cuts, and a stable 180-degree axis. A previous shot last frame may be used only for "
        "continuous_action. same_scene inherits scene and character identity but not the previous concrete image. No external "
        "pose or depth control model is integrated; such controls remain unsupported unless the Capability Matrix reports otherwise."
    )
    POST_SOP = (
        "Editing preference only, not a model hard constraint: favor a hard cut or 0.2s fade inside a scene, and a 0.3-0.5s "
        "transition between scenes. Keep ambient room tone continuous when possible."
    )

    def enrich_character(self, character: dict[str, Any], index: int = 0) -> dict[str, Any]:
        item = deepcopy(character)
        name = item.get("name") or f"character_{index + 1}"
        appearance = item.get("appearance") if isinstance(item.get("appearance"), dict) else {}
        default_outfit = (
            item.get("default_outfit")
            or appearance.get("default_outfit")
            or appearance.get("outfit")
            or "locked default outfit"
        )
        item["default_outfit"] = default_outfit
        item["wardrobe_lock"] = item.get("wardrobe_lock") or (
            f"Prompt preference for {name}: keep wardrobe near {default_outfit}; this is not a model hard constraint."
        )
        # LoRA / IP-Adapter 均未接入：不生成看似已绑定的档案名，避免任何
        # 「已启用」的虚假声明。字段保留为空串以兼容存储结构。
        item["lora_profile"] = ""
        item["ip_adapter_profile"] = ""
        item.setdefault("reference_images", item.get("reference_images") or [])
        return item

    def enrich_scene(self, scene: dict[str, Any], index: int = 0) -> dict[str, Any]:
        item = deepcopy(scene)
        location = item.get("location") or item.get("name") or f"scene_{index + 1}"
        time_of_day = item.get("time_of_day") or self._guess_time_of_day(
            " ".join(str(item.get(key, "")) for key in ("location", "actions", "description"))
        )
        scene_type = self._guess_scene_type(location)
        group_key = item.get("scene_group_key") or f"{self._slug(location)}-{self._slug(time_of_day)}"
        profile = self._build_scene_profile(item, group_key, time_of_day, scene_type, index)
        prop_lock = item.get("prop_lock") or self._build_prop_lock(item)

        item["scene_group_key"] = group_key
        item["time_of_day"] = time_of_day
        item["consistency_profile"] = profile
        item["prop_lock"] = prop_lock
        item.setdefault("reference_images", item.get("reference_images") or [])
        item["visual_prompt"] = item.get("visual_prompt") or self._scene_visual_prompt(item, profile)
        return item

    def scene_baseline_prompt(self, scene: dict[str, Any], style: str = "anime") -> tuple[str, str]:
        profile = self._profile(scene.get("consistency_profile"))
        template = style_template(style)
        prompt_parts = [
            template.get("scene_baseline_prompt", "production background key art, clean vertical composition"),
            "empty scene baseline reference image, no characters, no subtitles, no watermark",
            scene.get("location") or scene.get("name") or "",
            scene.get("visual_prompt", ""),
            scene.get("actions") or scene.get("description") or "",
            self._scene_profile_sentence(profile),
            scene.get("prop_lock", ""),
        ]
        negative = (
            "characters, people, changing props, extra furniture, inconsistent perspective, text, subtitles, watermark, "
            "low quality, blurry, distorted architecture"
        )
        return ", ".join(part for part in prompt_parts if part), negative

    def build_generation_context(
        self,
        shot: dict[str, Any],
        characters: list[dict[str, Any]],
        scenes: dict[str, dict[str, Any]],
        previous_reference_path: str = "",
        for_video: bool = False,
        previous_shot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        scene = scenes.get(shot.get("scene_asset_id") or "") or {}
        selected_characters = self.select_characters(shot, characters)
        weights = self.reference_weights(shot.get("shot_type", "medium"))
        scene_profile = self._profile(scene.get("consistency_profile"))
        scene_refs = self._scene_reference_images(scene)
        char_refs = [ref for char in selected_characters for ref in char.get("reference_images", []) if ref]
        character_blocking = self._character_blocking_profile(shot, scene, selected_characters, scene_profile)
        continuity_profile = self._continuity_profile(
            shot=shot,
            scene=scene,
            scene_profile=scene_profile,
            previous_reference_path=previous_reference_path,
            for_video=for_video,
            character_blocking=character_blocking,
            previous_shot=previous_shot,
        )
        previous_reference_path = str(continuity_profile.get("continuity_reference_path") or "")
        reference_assets = self._reference_assets(
            scene_refs, char_refs, previous_reference_path, weights, continuity_profile
        )

        parts = [
            self.SCENE_SOP,
            self.CHARACTER_SOP,
            self.CONTINUITY_SOP,
            self.POST_SOP,
            f"Scene group: {scene.get('scene_group_key') or shot.get('scene_group_id') or 'locked-current-scene'}; time: {scene.get('time_of_day') or 'locked'}; baseline: {scene.get('name') or 'scene baseline'}.",
            self._scene_profile_sentence(scene_profile),
            scene.get("prop_lock", ""),
            "Reference weight policy: text_only_policy. Numeric reference weights are not sent unless the Provider Capability Matrix declares a real provider parameter.",
            self._continuity_sentence(continuity_profile),
        ]
        if scene_refs:
            parts.append(
                "Scene baseline/reference assets are available for Provider input when the Capability Matrix reports multi-reference support; otherwise the request report will mark them not sent."
            )
        if char_refs:
            parts.append(
                "Character three-view references are available for Provider input when the Capability Matrix reports multi-reference support; otherwise the request report will mark them not sent."
            )
        if continuity_profile.get("continuity_reference_used"):
            parts.append(
                "Previous shot last frame is used only because continuity_mode is continuous_action; no pose/depth control is implied."
            )
        elif continuity_profile.get("continuity_reference_reason"):
            parts.append(
                f"No previous-shot image continuity is used ({continuity_profile['continuity_reference_reason']})."
            )
        if for_video:
            parts.append(
                "Video prompt preference: begin from the approved storyboard frame when supported, then keep scene and character cues visually coherent."
            )

        for char in selected_characters:
            wardrobe = char.get("wardrobe_lock", "")
            if wardrobe:
                parts.append(wardrobe)

        prompt = " ".join(self._clean_text(part) for part in parts if part)
        return {
            "consistency_context": prompt,
            "continuity_mode": continuity_profile.get("continuity_mode", "independent"),
            "continuity_mode_source": continuity_profile.get("continuity_mode_source", ""),
            "scene_group_id": scene.get("scene_group_key") or shot.get("scene_group_id", ""),
            "scene_reference_images": scene_refs,
            "character_reference_images": char_refs,
            "reference_weights": {"policy": "text_only_policy", "preferences": weights},
            "reference_assets": reference_assets,
            "continuity_profile": continuity_profile,
            "continuity_reference_path": previous_reference_path,
            "pose_reference_path": continuity_profile.get("pose_reference_path", ""),
            "depth_reference_path": continuity_profile.get("depth_reference_path", ""),
            "continuity_manifest_item": self.continuity_manifest_item(continuity_profile),
        }

    def resolve_continuity(
        self,
        shot: dict[str, Any],
        previous_shot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """解析规范连续性模式，并决定是否允许使用上一镜末帧。

        分镜显式输出优先；缺失时按「相邻镜头、场景变化、反打提示、动作连续」
        的确定性顺序兜底。无论模式如何，只有 ``continuous_action`` 可以读取
        ``previous_shot.last_frame_path``，且绝不回退到上一镜故事板/首帧。
        """

        profile = self._profile(shot.get("continuity_profile"))
        raw_mode = shot.get("continuity_mode") or profile.get("continuity_mode")
        mode = normalize_continuity_mode(raw_mode, default="")
        if mode:
            source = str(
                shot.get("continuity_mode_source")
                or profile.get("continuity_mode_source")
                or ("legacy_profile" if not shot.get("continuity_mode") else "storyboard")
            )
        else:
            legacy = normalize_continuity_mode(profile.get("control_source"), default="")
            if legacy == "continuous_action":
                mode = "continuous_action"
                source = "legacy_profile"
            else:
                mode = self._fallback_continuity_mode(shot, previous_shot)
                source = "deterministic_rule"

        previous_id = str((previous_shot or {}).get("shot_id") or (previous_shot or {}).get("id") or "")
        candidate = str((previous_shot or {}).get("last_frame_path") or "")
        reference_path = ""
        reason = ""
        if mode != _CONTINUITY_REFERENCE_ONLY_MODE:
            reason = f"not_required_by_continuity_mode:{mode}"
        elif not previous_shot:
            reason = "previous_shot_missing"
        elif not candidate:
            reason = "previous_last_frame_missing"
        elif not self._valid_continuity_reference(candidate):
            reason = "previous_last_frame_invalid"
        else:
            reference_path = candidate

        return {
            "continuity_mode": mode,
            "continuity_mode_source": source,
            "previous_shot_id": previous_id,
            "continuity_reference_type": "previous_last_frame" if reference_path else "",
            "continuity_reference_path": reference_path,
            "continuity_reference_used": bool(reference_path),
            "continuity_reference_reason": reason,
            "inherits_scene_identity": mode in {"same_scene", "continuous_action", "reverse_shot"},
            "inherits_character_identity": True,
            "inherits_previous_image": bool(reference_path),
        }

    @staticmethod
    def continuity_manifest_item(profile: dict[str, Any]) -> dict[str, Any]:
        """生成始终存在的连续性决策项；无有效参考时把原因写进 manifest。"""

        used = bool(profile.get("continuity_reference_used"))
        reason = str(profile.get("continuity_reference_reason") or "")
        return {
            "type": "continuity_reference",
            "asset_id": "continuity",
            "name": "previous-shot-last-frame" if used else "continuity-reference-decision",
            "path": str(profile.get("continuity_reference_path") or ""),
            "status": "ready" if used else "unavailable",
            "usage": "continuity_reference" if used else "not_used",
            "used": used,
            "sent": False,
            "reason": "" if used else reason,
            "not_used_reason": "" if used else reason,
            "continuity_mode": normalize_continuity_mode(profile.get("continuity_mode")),
            "continuity_mode_source": str(profile.get("continuity_mode_source") or ""),
            "previous_shot_id": str(profile.get("previous_shot_id") or ""),
            "inherits_scene_identity": bool(profile.get("inherits_scene_identity")),
            "inherits_character_identity": bool(profile.get("inherits_character_identity")),
            "inherits_previous_image": bool(profile.get("inherits_previous_image")),
        }

    def _fallback_continuity_mode(
        self,
        shot: dict[str, Any],
        previous_shot: dict[str, Any] | None,
    ) -> str:
        if not previous_shot:
            return "independent"
        if not self._same_scene(shot, previous_shot):
            return "scene_transition"
        if self._has_reverse_shot_cue(shot, previous_shot):
            return "reverse_shot"
        if self._has_continuous_action_cue(shot, previous_shot):
            return "continuous_action"
        return "same_scene"

    @staticmethod
    def _same_scene(shot: dict[str, Any], previous_shot: dict[str, Any]) -> bool:
        for key in ("scene_asset_id", "scene_group_id", "scene_number", "source_scene_number"):
            current = str(shot.get(key) or "")
            previous = str(previous_shot.get(key) or "")
            if current and previous:
                return current == previous
        return False

    @staticmethod
    def _shot_continuity_text(shot: dict[str, Any]) -> str:
        return " ".join(
            str(shot.get(key) or "")
            for key in ("character_action", "scene_description", "visual_notes", "camera_angle", "shot_type")
        ).lower()

    def _has_reverse_shot_cue(self, shot: dict[str, Any], previous_shot: dict[str, Any]) -> bool:
        text = f"{self._shot_continuity_text(previous_shot)} {self._shot_continuity_text(shot)}"
        return bool(
            re.search(r"reverse[_ -]?shot|over[- ]?the[- ]?shoulder|正反打|反打|对切|视线切换|互相凝视", text, re.I)
        )

    def _has_continuous_action_cue(self, shot: dict[str, Any], previous_shot: dict[str, Any]) -> bool:
        text = f"{self._shot_continuity_text(previous_shot)} {self._shot_continuity_text(shot)}"
        if re.search(
            r"continuous[_ -]?action|match[- ]?on[- ]?action|连续动作|动作连续|动作衔接|同一动作|接续动作", text, re.I
        ):
            return True
        previous_action = self._shot_continuity_text(previous_shot)
        current_action = self._shot_continuity_text(shot)
        motion_words = (
            "跑",
            "追",
            "冲",
            "跳",
            "摔",
            "落",
            "转身",
            "走",
            "打",
            "挥",
            "拥抱",
            "推",
            "拉",
            "run",
            "chase",
            "jump",
            "fall",
            "turn",
            "walk",
            "fight",
            "hug",
            "push",
            "pull",
        )
        return any(word in previous_action and word in current_action for word in motion_words)

    @staticmethod
    def _valid_continuity_reference(path: str) -> bool:
        try:
            candidate = Path(path)
            return candidate.is_file() and candidate.stat().st_size > 0
        except (OSError, ValueError):
            return False

    def select_characters(self, shot: dict[str, Any], characters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected_ids = {str(item) for item in shot.get("character_asset_ids", []) if item}
        selected_names = {str(item) for item in shot.get("characters_in_scene", []) if item}
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for char in characters:
            char_id = str(char.get("id") or "")
            char_name = str(char.get("name") or "")
            if selected_ids and char_id not in selected_ids:
                continue
            if selected_names and not selected_ids and char_name not in selected_names:
                continue
            key = char_id or char_name
            if key and key not in seen:
                selected.append(char)
                seen.add(key)
        return selected or characters[:1]

    def reference_weights(self, shot_type: str) -> dict[str, float]:
        shot_type = (shot_type or "medium").lower()
        if shot_type in {"wide", "establishing"}:
            return {"environment": 0.50, "action": 0.25}
        if shot_type in {"close-up", "closeup", "extreme_close", "extreme close-up"}:
            return {"environment": 0.40, "action": 0.35}
        return {"environment": 0.45, "action": 0.30}

    def project_config(self) -> dict[str, Any]:
        return {
            "agent_sop": "enabled",
            "scene_anchor": True,
            "character_identity_lock": True,
            "continuity_frame_lock": "continuous_action_only",
            # 没有接入真实的姿态/深度控制模型，画像统一声明 unsupported，
            # 不允许任何「已启用」的虚假能力标记。
            "pose_lock_for_complex_motion": False,
            "depth_lock_for_complex_motion": False,
            "pose_control_model": "unsupported",
            "depth_control_model": "unsupported",
            "reference_weight_policy": "text_only_policy",
            "prompt_rules_are_preferences": True,
            "manual_storyboard_approval_required_before_video": True,
        }

    def _build_scene_profile(
        self, scene: dict[str, Any], group_key: str, time_of_day: str, scene_type: str, index: int
    ) -> dict[str, Any]:
        indoor = scene_type == "indoor"
        return {
            "scene_group_key": group_key,
            "time_of_day": time_of_day,
            "scene_type": scene_type,
            "color_temperature": self._color_temperature(time_of_day, indoor),
            "light_source_direction": "camera-left 35 degrees, slightly above eye level"
            if indoor
            else "sun direction fixed from upper camera-left",
            "light_intensity": "soft medium"
            if indoor
            else ("low blue night ambience" if time_of_day == "night" else "bright soft daylight"),
            "weather": "locked clear weather unless script explicitly changes weather",
            "atmosphere": scene.get("emotion") or "neutral narrative atmosphere",
            "spatial_perspective": f"locked {scene.get('camera_suggestion') or 'medium'} perspective grid, axis line stable",
            "axis_rule": "180-degree axis locked; keep character standing order and facing direction unless script marks reposition",
            "transition_same_scene": "hard cut or 0.2s fade only",
            "transition_cross_scene": "0.3-0.5s white flash or push-pull",
            "lut": f"project_scene_lut_{index + 1:02d}_{self._slug(time_of_day)}",
        }

    def _scene_visual_prompt(self, scene: dict[str, Any], profile: dict[str, Any]) -> str:
        return ", ".join(
            part
            for part in [
                scene.get("location") or scene.get("name"),
                scene.get("actions") or scene.get("description"),
                self._scene_profile_sentence(profile),
                "stable set dressing, locked props, consistent depth and perspective",
            ]
            if part
        )

    def _build_prop_lock(self, scene: dict[str, Any]) -> str:
        description = (
            scene.get("actions") or scene.get("description") or scene.get("visual_prompt") or "baseline set dressing"
        )
        return (
            "Prop lock: preserve all visible set dressing from the baseline image, including position, scale, count and orientation. "
            f"Script-described baseline props: {self._clean_text(description)[:260]}."
        )

    def _scene_profile_sentence(self, profile: dict[str, Any]) -> str:
        if not profile:
            return ""
        return (
            f"Preferred scene lighting: {profile.get('color_temperature')}; source {profile.get('light_source_direction')}; "
            f"intensity {profile.get('light_intensity')}; weather {profile.get('weather')}; "
            f"perspective {profile.get('spatial_perspective')}; LUT {profile.get('lut')}; {profile.get('axis_rule')}."
        )

    def _continuity_profile(
        self,
        shot: dict[str, Any],
        scene: dict[str, Any],
        scene_profile: dict[str, Any],
        previous_reference_path: str,
        for_video: bool,
        character_blocking: dict[str, Any],
        previous_shot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        complex_motion = self._is_complex_motion(shot)
        same_scene_transition = scene_profile.get("transition_same_scene") or "hard cut or 0.2s fade only"
        cross_scene_transition = scene_profile.get("transition_cross_scene") or "0.3-0.5s white flash or push-pull"
        decision = self.resolve_continuity(shot, previous_shot)
        mode = decision["continuity_mode"]
        if mode == "continuous_action":
            editing_logic = ["eye_line_continuity", "match_on_action", "180_degree_axis_lock"]
        elif mode == "reverse_shot":
            editing_logic = ["reverse_shot_eyeline", "180_degree_axis_lock"]
        elif mode == "scene_transition":
            editing_logic = ["scene_identity_reset", "cross_scene_transition"]
        elif mode == "same_scene":
            editing_logic = ["scene_identity", "character_identity", "180_degree_axis_lock"]
        else:
            editing_logic = ["independent_composition"]
        return {
            **decision,
            "editing_logic": editing_logic,
            "same_scene_transition": same_scene_transition,
            "cross_scene_transition": cross_scene_transition,
            "lut": scene_profile.get("lut", "project_scene_lut_locked"),
            "saturation": "locked per scene group",
            "sharpness": "locked per scene group",
            "ambient_audio_policy": "continuous room tone; do not cut background ambience at shot boundary",
            "previous_reference_path": decision.get("continuity_reference_path", ""),
            "complex_motion": complex_motion,
            # 本流水线没有接入 OpenPose / 深度估计模型；即便镜头动作复杂，
            # 也只能如实标记 unsupported，由首帧参考 + 文本规则约束运动。
            "openpose_lock": "unsupported",
            "depth_lock": "unsupported",
            "pose_control_model": "unsupported",
            "depth_control_model": "unsupported",
            "pose_reference_path": "",
            "depth_reference_path": "",
            "control_source": (
                "previous_last_frame"
                if decision.get("continuity_reference_used")
                else f"no_previous_image:{decision.get('continuity_reference_reason') or mode}"
            ),
            "video_generation_gate": "approved_storyboard_required" if for_video else "storyboard_reference_generation",
            "axis_rule": scene_profile.get("axis_rule") or "180-degree axis locked",
            "scene_group_key": scene.get("scene_group_key") or shot.get("scene_group_id", ""),
            "character_blocking": character_blocking,
        }

    def _continuity_sentence(self, profile: dict[str, Any]) -> str:
        if not profile:
            return ""
        blocking = profile.get("character_blocking") or {}
        parts = [
            "Continuity prompt preference (not a model hard constraint):",
            f"mode {profile.get('continuity_mode', 'independent')}",
            ", ".join(profile.get("editing_logic", [])),
            f"same-scene transition {profile.get('same_scene_transition')}",
            f"cross-scene transition {profile.get('cross_scene_transition')}",
            f"LUT {profile.get('lut')}",
            "OpenPose control unsupported (no pose model integrated)",
            "Depth control unsupported (no depth model integrated)",
            self._blocking_sentence(blocking),
            profile.get("ambient_audio_policy", ""),
        ]
        return "; ".join(part for part in parts if part)

    def _character_blocking_profile(
        self,
        shot: dict[str, Any],
        scene: dict[str, Any],
        selected_characters: list[dict[str, Any]],
        scene_profile: dict[str, Any],
    ) -> dict[str, Any]:
        names = [str(char.get("name") or char.get("id") or "").strip() for char in selected_characters]
        names = [name for name in names if name]
        if not names:
            names = [str(item) for item in shot.get("characters_in_scene", []) if item]
        character_order = list(dict.fromkeys(names))
        allow_reposition = self._allows_reposition(shot)
        facing_lock = {
            name: "keep baseline facing direction; no mirror flip or side swap unless explicit reposition/cross-axis cue"
            for name in character_order
        }
        return {
            "scene_group_key": scene.get("scene_group_key") or shot.get("scene_group_id", ""),
            "axis_line": scene_profile.get("axis_rule") or "180-degree axis locked",
            "character_order_left_to_right": character_order,
            "facing_direction_lock": facing_lock,
            "eye_line_target": self._eye_line_target(shot, character_order),
            "match_on_action_policy": "cut during the same action beat; preserve limb direction and motion vector between adjacent shots",
            "camera_movement_limit": "same-scene camera may only zoom or make small position changes; no axis crossing",
            "skin_light_integration": (
                f"shade skin and character shadows with scene light {scene_profile.get('light_source_direction', 'locked source')} "
                f"and color temperature {scene_profile.get('color_temperature', 'locked palette')}"
            ),
            "reposition_override_allowed": allow_reposition,
        }

    def _blocking_sentence(self, blocking: dict[str, Any]) -> str:
        if not blocking:
            return ""
        order = blocking.get("character_order_left_to_right") or []
        return (
            "Character blocking lock: "
            f"left-to-right order {', '.join(order) if order else 'single subject'}; "
            f"{blocking.get('axis_line', '180-degree axis locked')}; "
            f"eye-line target {blocking.get('eye_line_target', 'next core subject')}; "
            f"{blocking.get('camera_movement_limit', '')}; "
            f"{blocking.get('skin_light_integration', '')}."
        )

    def _eye_line_target(self, shot: dict[str, Any], character_order: list[str]) -> str:
        dialogue_text = dialogue_plain_text(parse_shot_dialogue(shot.get("dialogue")))
        text = " ".join([dialogue_text, str(shot.get("character_action", "")), str(shot.get("scene_description", ""))])
        for name in character_order[1:] + character_order[:1]:
            if name and re.search(
                rf"\b(?:toward|to|looks at|faces|watching|gazes at)\s+{re.escape(name)}\b", text, re.I
            ):
                return f"maintain gaze toward {name} when they are the spoken-to or acted-on subject"
        for name in character_order[1:] + character_order[:1]:
            if name and name in text:
                return f"maintain gaze toward {name} when they are the spoken-to or acted-on subject"
        if len(character_order) >= 2:
            return (
                f"{character_order[0]} gaze anchors toward {character_order[1]} unless the script names another subject"
            )
        return "maintain gaze toward the next shot core subject or the established off-screen point"

    def _allows_reposition(self, shot: dict[str, Any]) -> bool:
        text = " ".join(
            str(shot.get(key, "")) for key in ("character_action", "scene_description", "visual_notes")
        ).lower()
        return bool(
            re.search(r"reposition|switch places|cross axis|crosses the line|turns around|walks past|exit|enter", text)
        )

    def _reference_assets(
        self,
        scene_refs: list[str],
        char_refs: list[str],
        previous_reference_path: str,
        weights: dict[str, float],
        continuity_profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        for path in scene_refs:
            assets.append(
                {
                    "type": "scene_baseline",
                    "path": path,
                    "role": "environment_props_lighting_perspective",
                    "weight_policy": "text_only_policy",
                    "required": True,
                }
            )
        for path in char_refs:
            assets.append(
                {
                    "type": "character_three_view",
                    "path": path,
                    "role": "identity_outfit_face_body_hair",
                    "weight_policy": "text_only_policy",
                    "required": True,
                }
            )
        if (
            previous_reference_path
            and continuity_profile.get("continuity_reference_used")
            and continuity_profile.get("continuity_mode") == "continuous_action"
        ):
            assets.append(
                {
                    "type": "continuity_frame",
                    "path": previous_reference_path,
                    "role": "eye_line_axis_motion",
                    "weight_policy": "text_only_policy",
                    "required": False,
                }
            )
        # 没有姿态/深度控制模型，不产出 openpose/depth 类参考资产；
        # 伪造这两类图只会误导下游与展示层。
        return assets

    def _is_complex_motion(self, shot: dict[str, Any]) -> bool:
        text = " ".join(
            str(shot.get(key, ""))
            for key in ("character_action", "camera_movement", "scene_description", "visual_notes")
        ).lower()
        return bool(
            re.search(
                r"run|running|jump|fight|fall|dance|turn|spin|walk|chase|push|pull|hug|挥手|奔跑|跑|追|跳|摔|跌|打|推|拉|转身|旋转|走|冲|拥抱|拉扯|挥",
                text,
            )
        )

    def _scene_reference_images(self, scene: dict[str, Any]) -> list[str]:
        refs = []
        if scene.get("baseline_image_path"):
            refs.append(scene["baseline_image_path"])
        raw_refs = scene.get("reference_images") or []
        if isinstance(raw_refs, str):
            try:
                raw_refs = json.loads(raw_refs)
            except Exception:
                raw_refs = []
        refs.extend(ref for ref in raw_refs if ref)
        return list(dict.fromkeys(refs))

    def _profile(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                data = json.loads(value)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def _guess_time_of_day(self, text: str) -> str:
        text = text.lower()
        if re.search(r"night|深夜|夜晚|晚上|夜色|moon", text):
            return "night"
        if re.search(r"dusk|sunset|黄昏|傍晚|夕阳", text):
            return "dusk"
        if re.search(r"morning|清晨|早晨|上午", text):
            return "morning"
        return "day"

    def _guess_scene_type(self, location: str) -> str:
        if re.search(r"室内|房间|教室|办公室|家|屋|indoor|room|office|classroom", location, re.I):
            return "indoor"
        return "outdoor"

    def _color_temperature(self, time_of_day: str, indoor: bool) -> str:
        if time_of_day == "night":
            return "cool blue 4200K night palette" if not indoor else "warm practical 3000K indoor night palette"
        if time_of_day == "dusk":
            return "warm amber 3600K dusk palette"
        if time_of_day == "morning":
            return "soft neutral 4800K morning palette"
        return "neutral daylight 5200K palette" if not indoor else "soft indoor daylight 4300K palette"

    def _slug(self, value: str) -> str:
        text = re.sub(r"\s+", "-", str(value or "").strip().lower())
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff_-]+", "", text)
        return text[:36] or "locked"

    def _digest(self, value: str) -> str:
        return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()

    def _clean_text(self, value: Any) -> str:
        text = str(value or "").strip()
        text = text.replace("{", "").replace("}", "")
        return " ".join(text.split())
