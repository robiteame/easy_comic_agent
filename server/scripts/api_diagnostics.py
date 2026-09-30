"""Compact API connectivity check.

This script uses existing .env configuration and prints no secrets:
- Mimo generates a short script and one storyboard shot.
- Mimo built-in TTS writes the matching local audio file.
- Doubao Seedance generates one shot video and a saved single frame.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from services.llm_service import LLMService
from services.providers.endpoint import get_endpoint
from services.tts_service import TTSService
from services.video_service import SeedanceVideoService


RUN_ID = uuid.uuid4().hex[:8]
DIAGNOSTIC_PROJECT_ID = f"api_diagnostics_{RUN_ID}"
DIAGNOSTIC_SHOT_ID = f"diagnostic_single_shot_{RUN_ID}"


def _print_config() -> None:
    script = get_endpoint("script")
    fallback = get_endpoint("script_fallback")
    video = get_endpoint("video")
    voice = get_endpoint("voice")
    print("CONFIG")
    print(
        f"  SCRIPT protocol={script.protocol} base_url={script.base_url} model={script.model} "
        f"auth={script.auth_style} key_present={bool(script.api_key)}"
    )
    print(
        f"  SCRIPT_FALLBACK base_url={fallback.base_url} model={fallback.model} key_present={bool(fallback.api_key)}"
    )
    print(
        f"  VIDEO protocol={video.protocol} base_url={video.base_url} model={video.model} "
        f"key_present={bool(video.api_key)}"
    )
    print(
        f"  VOICE protocol={voice.protocol} base_url={voice.base_url} model={voice.model} "
        f"key_present={bool(voice.api_key)}"
    )
    print(f"  LEGACY_TTS_PROVIDER={settings.TTS_PROVIDER}")


async def _check_mimo_storyboard() -> dict:
    script_endpoint = get_endpoint("script")
    # 多模态分镜优先用端点配置的 vision 模型，缺省回落到端点主模型 / 旧 MIMO 字段。
    vision_model = (
        script_endpoint.param("vision_model")
        or script_endpoint.model
        or settings.MIMO_MULTIMODAL_MODEL
    )
    result = await LLMService().call_json(
        "你是漫剧编导，只输出 JSON，不要 Markdown。",
        """
请生成一个极短中文漫剧测试剧本和单组分镜，用于接口连通性测试。
JSON 结构：
{
  "title": "标题",
  "script": "80字以内短剧本",
  "shot": {
    "shot_id": "mimo_seedance_check",
    "shot_type": "medium",
    "scene_description": "英文画面提示词，适合文生视频",
    "character_action": "英文动作提示词",
    "dialogue": "一句中文台词",
    "emotion": "happy",
    "duration": 5
  }
}
""",
        temperature=0,
        max_retries=0,
        allow_fallback=False,
        model_override=vision_model,
    )
    shot = result.get("shot") or {}
    if not result.get("script") or not shot.get("scene_description") or not shot.get("dialogue"):
        raise RuntimeError(f"unexpected Mimo storyboard result: {result}")
    shot["shot_id"] = DIAGNOSTIC_SHOT_ID
    # 对白为结构化列表（或旧字符串）：诊断只取第一句文本做 TTS 探测。
    dialogue = shot.get("dialogue")
    if isinstance(dialogue, list):
        first = dialogue[0] if dialogue else {}
        shot["dialogue"] = str((first or {}).get("line") or "") if isinstance(first, dict) else str(first or "")
    print(f"MIMO_OK title={result.get('title', '')} dialogue={shot.get('dialogue', '')}")
    return result


async def _check_tts(story: dict) -> str:
    shot = story["shot"]
    path = await TTSService().generate_dialogue(
        text=shot["dialogue"],
        emotion=shot.get("emotion", "neutral"),
        project_id=DIAGNOSTIC_PROJECT_ID,
        shot_id=shot.get("shot_id", DIAGNOSTIC_SHOT_ID),
    )
    audio = Path(path)
    if not audio.exists() or audio.stat().st_size <= 1024:
        raise RuntimeError(f"TTS output is missing or too small: {audio}")
    print(f"MIMO_TTS_OK {audio} {audio.stat().st_size} bytes")
    return str(audio)


async def _check_seedance(story: dict) -> dict[str, str]:
    shot = story["shot"]
    prompt = ", ".join(
        part
        for part in [
            shot.get("scene_description", ""),
            shot.get("character_action", ""),
            "vertical cinematic comic shot, gentle camera movement, clean composition, no subtitles, no watermark",
        ]
        if part
    )
    service = SeedanceVideoService()
    reference_shot = _seedance_reference_shot(story)
    reference_manifest = service._validate_video_references(reference_shot)
    reference_shot["seedance_reference_manifest"] = reference_manifest
    content = service._build_content(prompt, reference_shot)
    image_items = [item for item in content if item.get("type") == "image_url"]
    if not image_items:
        raise RuntimeError("Seedance diagnostic did not build reference image content")
    result = await service.generate_single_shot(
        prompt=prompt,
        project_id=DIAGNOSTIC_PROJECT_ID,
        shot_id=shot.get("shot_id", DIAGNOSTIC_SHOT_ID),
        duration=int(shot.get("duration") or 5),
        content=content,
    )
    frame_path = Path(result["frame_path"])
    with Image.open(frame_path) as image:
        image.verify()
    video_path = Path(result["video_path"])
    _assert_video(video_path)
    print(f"SEEDDANCE_OK video={video_path} {video_path.stat().st_size} bytes")
    print(f"SEEDDANCE_FRAME_OK {frame_path} {frame_path.stat().st_size} bytes")
    print(f"SEEDDANCE_REFERENCE_CONTENT_OK images={len(image_items)}")
    print(f"SEEDDANCE_REFERENCE_MANIFEST_OK loaded={len(reference_manifest)}")
    print(f"SEEDDANCE_REFERENCE_PAYLOAD_MODE {result.get('reference_payload_mode', '')}")
    if result.get("reference_payload_mode") != "first_frame_reference":
        raise RuntimeError(f"Seedance did not use first_frame reference payload: {result.get('reference_payload_mode')}")
    return result


def _seedance_reference_shot(story: dict) -> dict:
    shot = story["shot"]
    ref_dir = settings.OUTPUT_DIR / "projects" / DIAGNOSTIC_PROJECT_ID / "diagnostic_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    storyboard_path = _reference_png(ref_dir / "storyboard_keyframe.png", "story", (184, 122, 86))
    scene_path = _reference_png(ref_dir / "scene_baseline.png", "scene", (82, 148, 126))
    character_path = _reference_png(ref_dir / "character_three_view.png", "char", (146, 92, 172))
    previous_frame_path = _reference_png(ref_dir / "previous_last_frame.png", "prev", (92, 118, 184))
    return {
        "shot_id": shot.get("shot_id", DIAGNOSTIC_SHOT_ID),
        "storyboard_path": str(storyboard_path),
        "image_path": str(storyboard_path),
        "continuity_reference_path": str(previous_frame_path),
        "reference_weights": {"environment": 0.45, "action": 0.30},
        "reference_assets": [
            {"type": "scene_baseline", "path": str(scene_path), "weight": 0.45, "required": True},
            {"type": "character_three_view", "path": str(character_path), "weight": 0.30, "required": True},
            {"type": "continuity_frame", "path": str(previous_frame_path), "weight": 0.30, "required": True},
        ],
    }


def _reference_png(path: Path, label: str, color: tuple[int, int, int]) -> Path:
    image = Image.new("RGB", (384, 640), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((68, 92, 316, 548), outline=(250, 250, 250), width=8)
    draw.text((92, 118), label, fill=(250, 250, 250))
    image.save(path)
    return path


def _fallback_story() -> dict:
    return {
        "title": "diagnostic fallback",
        "script": "固定诊断镜头，仅在 Mimo 失败时用于继续探测 TTS 和 Seedance。",
        "shot": {
            "shot_id": DIAGNOSTIC_SHOT_ID,
            "shot_type": "medium",
            "scene_description": "A friendly white service robot waves beside a sunlit desk in a vertical cinematic shot",
            "character_action": "the robot gently raises one hand to greet the viewer",
            "dialogue": "主人，早上好！",
            "emotion": "happy",
            "duration": 5,
        },
    }


def _assert_video(video_path: Path) -> None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(video_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    if "video" not in result.stdout:
        raise RuntimeError(f"Seedance output has no video stream: {video_path}")


async def main() -> int:
    _print_config()
    failures: list[str] = []
    story: dict | None = None
    audio_path = ""
    seedance_result: dict[str, str] = {}

    try:
        story = await _check_mimo_storyboard()
    except Exception as exc:
        failures.append(f"MIMO: {exc}")
        print(f"MIMO_FAILED {exc}")

    probe_story = story or _fallback_story()
    if story is None:
        print("USING_STATIC_STORY_FOR_DOWNSTREAM_PROBES")

    try:
        audio_path = await _check_tts(probe_story)
    except Exception as exc:
        failures.append(f"MIMO_TTS: {exc}")
        print(f"MIMO_TTS_FAILED {exc}")

    try:
        seedance_result = await _check_seedance(probe_story)
    except Exception as exc:
        failures.append(f"SEEDDANCE: {exc}")
        print(f"SEEDDANCE_FAILED {exc}")

    if failures:
        print("DIAGNOSTICS_FAILED")
        return 1

    print(f"OUTPUT_AUDIO {audio_path}")
    print(f"OUTPUT_FRAME {seedance_result['frame_path']}")
    print(f"OUTPUT_VIDEO {seedance_result['video_path']}")
    print("DIAGNOSTICS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
