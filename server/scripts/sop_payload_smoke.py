"""Offline smoke test for Agent consistency SOP payload assembly."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.routes import shot as shot_route
from api.routes.render import _apply_post_profiles
from api.routes.shot import (
    _can_reuse_existing_video,
    _invalidate_downstream_media,
    _invalidate_storyboard_outputs,
    _invalidate_video_outputs,
    _previous_reference_for_shot,
)
from config import settings
from services.consistency_service import ConsistencyService
from services.ffmpeg_service import FFmpegService
from services.image_service import ImageService
from services.video_service import SeedanceVideoService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TMP_ROOT = PROJECT_ROOT / "tmp_sop_payload_smoke"


def _png(path: Path, label: str, color: tuple[int, int, int]) -> Path:
    image = Image.new("RGB", (160, 240), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 40, 130, 210), outline=(255, 255, 255), width=4)
    draw.text((46, 52), label, fill=(255, 255, 255))
    image.save(path)
    return path


def _stale_shot(**overrides):
    data = {
        "id": "stale_shot",
        "project_id": "project1",
        "sequence": 1,
        "scene_group_id": "classroom-morning",
        "scene_asset_id": "scene1",
        "confirmed": True,
        "status": "video_done",
        "storyboard_status": "done",
        "storyboard_path": "story.png",
        "image_path": "image.png",
        "audio_path": "audio.wav",
        "video_path": "video.mp4",
        "last_frame_path": "last.png",
        "continuity_reference_path": "prev.png",
        "pose_reference_path": "pose.png",
        "depth_reference_path": "depth.png",
        "continuity_profile": "{\"previous_reference_path\":\"prev.png\"}",
        "reference_weights": "{\"policy\":\"text_only_policy\",\"preferences\":{\"environment\":0.45}}",
        "consistency_context": "old context",
        "visual_notes": "",
        "version": 1,
        "media_stale": False,
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
        # 失效流程会在变更前做版本快照（create_version）：fake 只需让
        # 「查最新版本」返回空，快照行由 add/flush 空实现吞掉。
        return None


class _FakeDB:
    def __init__(self, items):
        self.items = items

    def query(self, model):
        return _FakeQuery(self.items)

    def add(self, obj):
        pass

    def flush(self):
        pass


def main() -> None:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    TMP_ROOT.mkdir(parents=True)
    original_output_dir = settings.OUTPUT_DIR
    try:
        # Reuse checks now validate that the media exists under an approved
        # root. Point the smoke fixture at its isolated temporary root instead
        # of using the old placeholder path.
        settings.OUTPUT_DIR = TMP_ROOT
        shot_route.image_service.reference_assets.output_dir = TMP_ROOT
        shot_route.seedance_service.reference_assets.output_dir = TMP_ROOT
        frame = _png(TMP_ROOT / "prev_frame.png", "prev", (90, 120, 180))
        storyboard = _png(TMP_ROOT / "storyboard.png", "story", (180, 120, 90))
        scene_ref = _png(TMP_ROOT / "scene_ref.png", "scene", (80, 160, 110))
        char_ref = _png(TMP_ROOT / "char_ref.png", "char", (180, 80, 160))
        char_ref_b = _png(TMP_ROOT / "char_ref_b.png", "charB", (80, 90, 190))

        consistency = ConsistencyService()
        scene = consistency.enrich_scene({"location": "classroom", "time_of_day": "morning", "actions": "morning desk"}, 0)
        scene.update({"id": "scene1", "name": "classroom-morning", "baseline_image_path": str(scene_ref), "reference_images": [str(scene_ref)]})
        character = consistency.enrich_character(
            {"name": "Xia", "appearance": {"default_outfit": "uniform"}, "reference_images": [str(char_ref)]},
            0,
        )
        character["id"] = "char1"
        character_b = consistency.enrich_character(
            {"name": "Bo", "appearance": {"default_outfit": "hoodie"}, "reference_images": [str(char_ref_b)]},
            1,
        )
        character_b["id"] = "char2"
        shot = {
            "shot_id": "shot_payload_smoke",
            "shot_type": "medium",
            "scene_asset_id": "scene1",
            "scene_group_id": scene["scene_group_key"],
            "characters_in_scene": ["Xia", "Bo"],
            "character_asset_ids": ["char1", "char2"],
            "character_action": "Xia runs and turns quickly toward Bo",
            "camera_movement": "following",
            "storyboard_path": str(storyboard),
            "image_path": str(storyboard),
        }
        shot.update(
            consistency.build_generation_context(
                shot,
                [character, character_b],
                {"scene1": scene},
                previous_reference_path=str(frame),
                for_video=True,
            )
        )
        blocking = shot["continuity_profile"]["character_blocking"]
        assert blocking["character_order_left_to_right"] == ["Xia", "Bo"]
        assert "180-degree axis preference" in blocking["axis_line"]
        assert "Bo" in blocking["eye_line_target"]
        assert "scene light" in blocking["skin_light_integration"]
        assert "Character blocking prompt preference" in shot["consistency_context"]
        shot_route._materialize_control_references("_sop_payload_smoke", shot)

        pose_path = Path(shot["pose_reference_path"])
        depth_path = Path(shot["depth_reference_path"])
        assert not shot["pose_reference_path"] and not shot["depth_reference_path"]

        image_service = ImageService()
        image_prompt, _ = image_service._build_prompt(shot, [character, character_b], {})
        assert "character blocking prompt preference" in image_prompt
        assert "left-to-right order Xia, Bo" in image_prompt
        image_refs = image_service._reference_images_for_request(shot)
        seedream_payload = image_service._seedream_payload("doubao-seedream-5-0-lite", "prompt", "negative", 42, "2K", image_refs)
        assert image_refs and all(item.startswith("data:image/") for item in image_refs), len(image_refs)
        assert seedream_payload.get("image") == image_refs[:8]

        video_service = SeedanceVideoService()
        reference_manifest = video_service._validate_video_references(shot)
        shot["seedance_reference_manifest"] = reference_manifest
        video_prompt = video_service._build_prompt(shot, [character, character_b], {"scene1": scene})
        assert "character blocking prompt preference" in video_prompt
        assert "left-to-right order Xia, Bo" in video_prompt
        assert "provider_capabilities and references_sent" in video_prompt
        video_content = video_service._build_content("prompt text", shot)
        assert video_content[0]["type"] == "text"
        image_content = [item for item in video_content if item.get("type") == "image_url"]
        assert len(image_content) == 1
        assert image_content[0]["role"] == "first_frame"
        assert video_service._reference_payload_mode(video_content) == "first_frame_reference"
        assert all("asset_type" not in item and "weight" not in item for item in image_content)
        loaded_types = {item["type"] for item in reference_manifest}
        assert {"approved_storyboard_first_frame", "scene_baseline", "character_three_view", "continuity_frame"}.issubset(loaded_types)

        post_shots = [
            {"scene_group_id": "classroom-morning", "continuity_profile": shot["continuity_profile"]},
            {"scene_group_id": "classroom-morning", "continuity_profile": shot["continuity_profile"]},
            {"scene_group_id": "street-night", "continuity_profile": {**shot["continuity_profile"], "lut": "project_scene_lut_02_night"}},
        ]
        _apply_post_profiles(post_shots)
        assert post_shots[0]["post_profile"]["transition_out"] == "hard cut or 0.2s fade only"
        assert post_shots[0]["post_profile"]["same_scene_fade_seconds"] == 0.2
        assert post_shots[1]["post_profile"]["cross_scene_out"] is True
        assert post_shots[1]["post_profile"]["cross_scene_flash_seconds"] == 0.35
        assert post_shots[2]["post_profile"]["cross_scene_in"] is True
        ffmpeg = FFmpegService()
        assert ffmpeg._post_filter(post_shots[0]) == ffmpeg._post_filter(post_shots[1])
        assert ffmpeg._post_filter(post_shots[0]) != ffmpeg._post_filter(post_shots[2])
        clip_filter = ffmpeg._clip_filter("scale=720:1280", post_shots[1], 1.2)
        assert "eq=saturation=" in clip_filter
        assert "unsharp=5:5:" in clip_filter
        assert "fade=t=out" in clip_filter
        assert "d=0.35" in clip_filter

        stale = _stale_shot()
        _invalidate_storyboard_outputs(stale)
        assert stale.confirmed is False
        assert stale.media_stale is True
        # 旧素材保留（仅标记过期），路径与用户设置不再被清空。
        assert stale.storyboard_path == "story.png"
        assert stale.image_path == "image.png"
        assert stale.video_path == "video.mp4"
        assert stale.audio_path == "audio.wav"
        assert stale.last_frame_path == "last.png"
        assert stale.pose_reference_path == "pose.png"
        assert stale.depth_reference_path == "depth.png"
        assert stale.continuity_profile == "{\"previous_reference_path\":\"prev.png\"}"
        assert stale.reference_weights == "{\"policy\":\"text_only_policy\",\"preferences\":{\"environment\":0.45}}"
        assert stale.consistency_context == "old context"

        valid_video = TMP_ROOT / "video.mp4"
        valid_video.write_bytes(b"\x00" * 4096)
        video_stale = _stale_shot(sequence=2, video_path=str(valid_video))
        assert _can_reuse_existing_video(video_stale, force=False)
        assert not _can_reuse_existing_video(video_stale, force=True)
        video_stale.status = "storyboard_approved"
        assert not _can_reuse_existing_video(video_stale, force=False)
        video_stale.status = "video_done"
        # stale 标记置位后，复用门必须拒绝旧视频。
        video_stale.media_stale = True
        assert not _can_reuse_existing_video(video_stale, force=False)
        video_stale.media_stale = False
        _invalidate_video_outputs(video_stale)
        assert video_stale.confirmed is True
        assert video_stale.storyboard_path == "story.png"
        assert video_stale.video_path == str(valid_video)
        assert video_stale.media_stale is True
        assert video_stale.status == "storyboard_approved"

        same_scene_next = _stale_shot(sequence=3, scene_group_id="classroom-morning")
        other_scene_next = _stale_shot(sequence=4, scene_group_id="street-night")
        same_scene_next.version = 2
        _invalidate_downstream_media(_FakeDB([same_scene_next, other_scene_next]), _stale_shot(sequence=1))
        assert same_scene_next.video_path == "video.mp4"
        assert same_scene_next.last_frame_path == "last.png"
        assert same_scene_next.media_stale is True
        assert same_scene_next.version == 3
        assert other_scene_next.video_path == "video.mp4"
        assert not other_scene_next.media_stale

        base_regen = _stale_shot(sequence=1, scene_group_id="classroom-morning", scene_asset_id="scene1")
        base_previous_key = base_regen.scene_group_id
        _invalidate_storyboard_outputs(base_regen)
        later_same_group = _stale_shot(sequence=2, scene_group_id="classroom-morning", scene_asset_id="scene1")
        later_same_group.version = 5
        _invalidate_downstream_media(_FakeDB([later_same_group]), base_regen, {base_previous_key, base_regen.scene_asset_id})
        assert later_same_group.video_path == "video.mp4"
        assert later_same_group.media_stale is True
        assert later_same_group.version == 6

        previous_same_group = _stale_shot(
            sequence=1,
            scene_group_id="classroom-morning",
            scene_asset_id="scene_asset_a",
            storyboard_path="previous_story.png",
            image_path="previous_image.png",
            last_frame_path="previous_last.png",
        )
        current_same_group = _stale_shot(sequence=2, scene_group_id="classroom-morning", scene_asset_id="scene_asset_b")
        assert _previous_reference_for_shot(_FakeDB([previous_same_group]), current_same_group) == "previous_story.png"
        assert _previous_reference_for_shot(_FakeDB([previous_same_group]), current_same_group, prefer_last_frame=True) == "previous_last.png"
        current_other_group = _stale_shot(sequence=2, scene_group_id="street-night", scene_asset_id="scene_asset_b")
        assert _previous_reference_for_shot(_FakeDB([previous_same_group]), current_other_group) == ""

        print(
            json.dumps(
                {
                    "image_reference_count": len(image_refs),
                    "seedream_image_param": len(seedream_payload.get("image", [])),
                    "seedance_content_items": len(video_content),
                    "seedance_image_items": len(image_content),
                    "seedance_payload_mode": video_service._reference_payload_mode(video_content),
                    "seedance_loaded_reference_types": sorted(loaded_types),
                    "post_profile_checked": True,
                    "stale_media_invalidation_checked": True,
                    "scene_group_previous_reference_checked": True,
                    "character_blocking_checked": True,
                    "pose_ref_exists": bool(shot["pose_reference_path"]),
                    "depth_ref_exists": bool(shot["depth_reference_path"]),
                },
                ensure_ascii=False,
            )
        )
        print("SOP_PAYLOAD_SMOKE_OK")
    finally:
        settings.OUTPUT_DIR = original_output_dir
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT)


if __name__ == "__main__":
    main()
