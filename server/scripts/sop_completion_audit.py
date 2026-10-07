"""Requirement-oriented offline audit for the Agent visual consistency SOP."""

from __future__ import annotations

import atexit
import inspect
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from api.routes import shot as shot_route
from api.routes.render import _apply_post_profiles
from config import settings
from services.consistency_service import ConsistencyService
from services.ffmpeg_service import FFmpegService
from services.image_service import ImageService
from services.providers.video_ark_seedance import ArkSeedanceVideoAdapter
from services.video_service import SeedanceVideoService

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _assert(condition: bool, name: str, evidence: str) -> dict:
    if not condition:
        raise AssertionError(f"{name}: {evidence}")
    return {"name": name, "evidence": evidence}


def _shot(**overrides):
    data = {
        "id": "shot1",
        "project_id": "project1",
        "sequence": 1,
        "scene_group_id": "classroom-morning",
        "scene_asset_id": "scene1",
        "confirmed": True,
        "status": "video_done",
        "storyboard_path": "story.png",
        "image_path": "image.png",
        "audio_path": "audio.wav",
        "video_path": "video.mp4",
        "last_frame_path": "last.png",
        "continuity_reference_path": "prev.png",
        "pose_reference_path": "pose.png",
        "depth_reference_path": "depth.png",
        "continuity_profile": "{}",
        "reference_weights": "{}",
        "consistency_context": "old",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


class _FakeQuery:
    def __init__(self, items):
        self.items = items

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def all(self):
        return self.items

    def first(self):
        return self.items[0] if self.items else None


class _FakeDB:
    def __init__(self, items):
        self.items = items

    def query(self, model):
        return _FakeQuery(self.items)


def main() -> None:
    checks: list[dict] = []
    # The reuse gate now verifies the file is present and large enough. Keep a
    # disposable fixture under an approved output root instead of using the
    # historical placeholder path (``video.mp4``).
    fixture_root = Path(tempfile.mkdtemp(prefix=".sop-completion-", dir=settings.OUTPUT_DIR))
    fixture_video = fixture_root / "video.mp4"
    fixture_video.write_bytes(b"\x00" * 4096)
    atexit.register(shutil.rmtree, fixture_root, ignore_errors=True)
    consistency = ConsistencyService()

    morning_scene = consistency.enrich_scene(
        {"location": "classroom", "time_of_day": "morning", "actions": "desks by window"}, 0
    )
    same_morning_scene = consistency.enrich_scene(
        {"location": "classroom", "time_of_day": "morning", "actions": "desks by window"}, 1
    )
    night_scene = consistency.enrich_scene(
        {"location": "classroom", "time_of_day": "night", "actions": "desks by moonlight"}, 2
    )
    checks.append(
        _assert(
            morning_scene["scene_group_key"] == same_morning_scene["scene_group_key"],
            "same_location_time_scene_group_reuse",
            morning_scene["scene_group_key"],
        )
    )
    checks.append(
        _assert(
            morning_scene["scene_group_key"] != night_scene["scene_group_key"],
            "day_night_scene_group_isolation",
            f"{morning_scene['scene_group_key']} != {night_scene['scene_group_key']}",
        )
    )
    scene_profile = morning_scene["consistency_profile"]
    locked_scene_fields = {
        "color_temperature",
        "light_source_direction",
        "light_intensity",
        "weather",
        "atmosphere",
        "spatial_perspective",
        "axis_rule",
        "lut",
    }
    checks.append(
        _assert(
            locked_scene_fields.issubset(scene_profile),
            "scene_locked_global_parameters",
            ",".join(sorted(locked_scene_fields)),
        )
    )
    checks.append(
        _assert(
            "Prop lock:" in morning_scene["prop_lock"]
            and "position, scale, count and orientation" in morning_scene["prop_lock"],
            "scene_baseline_prop_lock",
            morning_scene["prop_lock"],
        )
    )
    baseline_prompt, baseline_negative = consistency.scene_baseline_prompt(morning_scene, "anime")
    checks.append(
        _assert(
            "empty scene baseline reference image" in baseline_prompt and "characters" in baseline_negative,
            "scene_baseline_generation_prompt",
            baseline_prompt[:160],
        )
    )

    weights = {shot_type: consistency.reference_weights(shot_type) for shot_type in ("wide", "medium", "close-up")}
    checks.append(
        _assert(
            all(0.4 <= item["environment"] <= 0.5 and 0.25 <= item["action"] <= 0.35 for item in weights.values()),
            "reference_weight_ranges",
            json.dumps(weights, ensure_ascii=False),
        )
    )
    config = consistency.project_config()
    checks.append(
        _assert(
            config["rules_override_single_shot_customization"]
            and config["manual_storyboard_approval_required_before_video"],
            "project_sop_config_gates",
            json.dumps(config, ensure_ascii=False),
        )
    )

    character_a = consistency.enrich_character(
        {"id": "char1", "name": "Xia", "appearance": {"default_outfit": "uniform"}, "reference_images": ["char_a.png"]},
        0,
    )
    character_b = consistency.enrich_character(
        {"id": "char2", "name": "Bo", "appearance": {"default_outfit": "hoodie"}, "reference_images": ["char_b.png"]}, 1
    )
    checks.append(
        _assert(
            not character_a.get("lora_profile")
            and not character_a.get("ip_adapter_profile")
            and bool(character_a.get("wardrobe_lock")),
            "character_unsupported_lora_ip_and_wardrobe_lock",
            "LoRA/IP-Adapter remain empty; wardrobe text lock remains available",
        )
    )

    morning_scene.update(
        {"id": "scene1", "baseline_image_path": "scene_base.png", "reference_images": ["scene_base.png"]}
    )
    generation_shot = {
        "shot_id": "shot1",
        "shot_type": "medium",
        "scene_asset_id": "scene1",
        "scene_group_id": morning_scene["scene_group_key"],
        "characters_in_scene": ["Xia", "Bo"],
        "character_asset_ids": ["char1", "char2"],
        "character_action": "Xia runs toward Bo",
        "camera_movement": "following",
        "storyboard_path": "story.png",
        "image_path": "story.png",
    }
    generation_context = consistency.build_generation_context(
        generation_shot,
        [character_a, character_b],
        {"scene1": morning_scene},
        previous_reference_path="previous_last.png",
        for_video=True,
    )
    blocking = generation_context["continuity_profile"]["character_blocking"]
    checks.append(
        _assert(
            blocking["character_order_left_to_right"] == ["Xia", "Bo"] and "Bo" in blocking["eye_line_target"],
            "character_blocking_eye_line_axis_lock",
            json.dumps(blocking, ensure_ascii=False),
        )
    )
    checks.append(
        _assert(
            generation_context["continuity_profile"]["openpose_lock"] == "unsupported"
            and generation_context["continuity_profile"]["depth_lock"] == "unsupported"
            and not generation_context.get("pose_reference_path")
            and not generation_context.get("depth_reference_path"),
            "complex_motion_openpose_depth_unsupported",
            json.dumps(generation_context["continuity_profile"], ensure_ascii=False)[:240],
        )
    )
    checks.append(
        _assert(
            any(
                asset["type"] == "scene_baseline" and asset["required"]
                for asset in generation_context["reference_assets"]
            )
            and any(
                asset["type"] == "character_three_view" and asset["required"]
                for asset in generation_context["reference_assets"]
            )
            and any(
                asset["type"] == "continuity_frame" and asset["required"]
                for asset in generation_context["reference_assets"]
            ),
            "persisted_reference_assets_required",
            ",".join(asset["type"] for asset in generation_context["reference_assets"]),
        )
    )
    checks.append(
        _assert(
            "NON-NEGOTIABLE AGENT CONSISTENCY SOP" in generation_context["consistency_context"]
            and "override" in generation_context["consistency_context"].lower(),
            "sop_prompt_overrides_single_shot",
            generation_context["consistency_context"][:180],
        )
    )

    image_service = ImageService()
    image_prompt, _ = image_service._build_prompt(
        {**generation_shot, **generation_context}, [character_a, character_b], {}
    )
    checks.append(
        _assert(
            "locked character blocking" in image_prompt
            and "scene baseline/reference assets are loaded" in image_prompt,
            "image_prompt_contains_sop_context",
            image_prompt[:220],
        )
    )
    video_service = SeedanceVideoService()
    generation_context["seedance_reference_manifest"] = [
        {"type": "approved_storyboard_first_frame", "path": "story.png", "loaded": True},
        {"type": "scene_baseline", "path": "scene_base.png", "loaded": True},
        {"type": "character_three_view", "path": "char_a.png", "loaded": True},
        {"type": "continuity_frame", "path": "previous_last.png", "loaded": True},
    ]
    video_prompt = video_service._build_prompt(
        {**generation_shot, **generation_context}, [character_a, character_b], {"scene1": morning_scene}
    )
    checks.append(
        _assert(
            "locked character blocking" in video_prompt
            and "previous shot final frame" in video_prompt
            and "first_frame_only" in video_prompt
            and "receives first frame only" in video_prompt,
            "seedance_prompt_contains_honest_first_frame_context",
            video_prompt[:220],
        )
    )
    video_service_source = inspect.getsource(SeedanceVideoService)
    seedance_adapter_source = inspect.getsource(ArkSeedanceVideoAdapter)
    checks.append(
        _assert(
            '"role": "first_frame"' in seedance_adapter_source
            and "_validate_video_references" in video_service_source
            and "text_locked_reference_fallback" not in video_service_source,
            "seedance_first_frame_reference_without_text_fallback",
            "Seedance 1.5 payload uses first_frame and hard-fails missing/rejected references",
        )
    )
    checks.append(
        _assert(
            ArkSeedanceVideoAdapter.capabilities.fixed_duration == 5,
            "seedance_verified_duration_gate",
            "duration=5",
        )
    )

    source = inspect.getsource(shot_route.generate_shot_video)
    checks.append(
        _assert(
            "if not shot.confirmed" in source and "if not (shot.storyboard_path or shot.image_path)" in source,
            "manual_storyboard_approval_before_video",
            "generate_shot_video gate checks",
        )
    )
    reusable = _shot(status="video_done", video_path=str(fixture_video))
    not_reusable = _shot(status="storyboard_approved", video_path="video.mp4")
    checks.append(
        _assert(
            shot_route._can_reuse_existing_video(reusable, False)
            and not shot_route._can_reuse_existing_video(not_reusable, False),
            "stale_video_reuse_gate",
            "status must be video_done",
        )
    )
    stale = _shot()
    shot_route._invalidate_storyboard_outputs(stale)
    checks.append(
        _assert(
            not stale.confirmed
            and stale.media_stale
            and stale.video_path == "video.mp4"
            and stale.last_frame_path == "last.png"
            and stale.storyboard_path == "story.png",
            "shot_edit_marks_stale_media",
            "confirmed revoked, media paths preserved with stale marker",
        )
    )
    previous = _shot(
        sequence=1,
        scene_group_id="classroom-morning",
        scene_asset_id="sceneA",
        storyboard_path="prev_story.png",
        last_frame_path="prev_last.png",
    )
    current = _shot(sequence=2, scene_group_id="classroom-morning", scene_asset_id="sceneB")
    other = _shot(sequence=2, scene_group_id="street-night", scene_asset_id="sceneC")
    checks.append(
        _assert(
            shot_route._previous_reference_for_shot(_FakeDB([previous]), current, True) == "prev_last.png"
            and shot_route._previous_reference_for_shot(_FakeDB([previous]), other, True) == "",
            "scene_group_previous_frame_reference",
            "same scene group uses last frame; different group blocked",
        )
    )

    render_shots = [
        {"scene_group_id": "classroom-morning", "continuity_profile": generation_context["continuity_profile"]},
        {"scene_group_id": "classroom-morning", "continuity_profile": generation_context["continuity_profile"]},
        {
            "scene_group_id": "street-night",
            "continuity_profile": {**generation_context["continuity_profile"], "lut": "night_lut"},
        },
    ]
    _apply_post_profiles(render_shots)
    ffmpeg = FFmpegService()
    checks.append(
        _assert(
            render_shots[1]["post_profile"]["cross_scene_out"]
            and render_shots[2]["post_profile"]["cross_scene_in"]
            and ffmpeg._post_filter(render_shots[0]) == ffmpeg._post_filter(render_shots[1])
            and ffmpeg._post_filter(render_shots[0]) != ffmpeg._post_filter(render_shots[2])
            and "fade=t=out" in ffmpeg._clip_filter("scale=720:1280", render_shots[1], 1.2),
            "render_post_transition_lut_ambient_rules",
            "same-scene post chain stable; cross-scene flash applied",
        )
    )

    right_sidebar = (PROJECT_ROOT / "client/src/renderer/components/RightSidebar.tsx").read_text(encoding="utf-8")
    main_workspace = (PROJECT_ROOT / "client/src/renderer/components/MainWorkspace.tsx").read_text(encoding="utf-8")
    left_sidebar = (PROJECT_ROOT / "client/src/renderer/components/LeftSidebar.tsx").read_text(encoding="utf-8")
    api_ts = (PROJECT_ROOT / "client/src/renderer/services/api.ts").read_text(encoding="utf-8")
    asset_route = (PROJECT_ROOT / "server/api/routes/asset.py").read_text(encoding="utf-8")
    character_model = (PROJECT_ROOT / "server/models/character.py").read_text(encoding="utf-8")
    scene_model = (PROJECT_ROOT / "server/models/scene_asset.py").read_text(encoding="utf-8")
    checks.append(
        _assert(
            "一致性规划" in right_sidebar
            and "blockingRows" in right_sidebar
            and "character_blocking" in right_sidebar
            and "consistency-blocking-list" in right_sidebar
            and "bound-asset-list" in right_sidebar,
            "frontend_consistency_preview_panel_present",
            "RightSidebar consistency/blocking/reference panel present",
        )
    )
    checks.append(
        _assert(
            "WORKSPACE_TABS" in main_workspace
            and "workspace-tabbar" in main_workspace
            and "script" in main_workspace
            and "assets" in main_workspace
            and "storyboard" in main_workspace
            and "review-command-bar" in main_workspace
            and "review-approve-main" in main_workspace
            and "approval-pass-btn" in main_workspace
            and "shotApi.generateVideo" in main_workspace
            and "generateVideo" in api_ts
            and "approve-storyboard" in api_ts
            and "LeftOutlined" in left_sidebar
            and "RightOutlined" in left_sidebar
            and "project-tree-node" in left_sidebar
            and "project-expand-btn" in left_sidebar
            and "collapsed-project-btn" in left_sidebar,
            "frontend_tabs_approval_icons_project_tree_preserved",
            "Workspace tabs, per-shot approval/video controls, sidebar icons and project tree remain present",
        )
    )
    checks.append(
        _assert(
            "reference_images" in asset_route
            and "baseline_image_path" in asset_route
            and "scene_group_key" in asset_route
            and "lora_profile" in asset_route
            and "ip_adapter_profile" in asset_route
            and "wardrobe_lock" in asset_route
            and "reference_images" in character_model
            and "lora_profile" in character_model
            and "ip_adapter_profile" in character_model
            and "wardrobe_lock" in character_model
            and "reference_images" in scene_model
            and "baseline_image_path" in scene_model
            and "scene_group_key" in scene_model,
            "asset_persistence_contract_preserved",
            "Character and scene asset serializers/models retain reference and consistency fields",
        )
    )

    print(json.dumps({"checks": checks, "check_count": len(checks)}, ensure_ascii=False, indent=2))
    print("SOP_COMPLETION_AUDIT_OK")


if __name__ == "__main__":
    main()
