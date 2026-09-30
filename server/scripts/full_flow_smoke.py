"""End-to-end smoke test for the comic Agent pipeline.

This script calls the running FastAPI server, then verifies the current
manual-review pipeline:
1. Agent script generation.
2. Script parsing creates project assets and shot list.
3. Storyboard reference images are generated from the asset board.
4. Each shot is approved and generated individually.
5. Final render composes existing per-shot videos.

Usage:
    python scripts/full_flow_smoke.py
    API_BASE=http://127.0.0.1:8011 python scripts/full_flow_smoke.py
"""

from __future__ import annotations

import os
import json
import subprocess
import time
from pathlib import Path

from PIL import Image
import requests


API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8011").rstrip("/")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output" / "projects"


def post(path: str, payload: dict | None = None) -> dict:
    response = requests.post(f"{API_BASE}{path}", json=payload or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def get(path: str) -> dict | list:
    response = requests.get(f"{API_BASE}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def put(path: str, payload: dict) -> dict:
    response = requests.put(f"{API_BASE}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def wait_for(predicate, timeout: int, label: str):
    deadline = time.time() + timeout
    last_value = None
    while time.time() < deadline:
        last_value = predicate()
        if last_value:
            return last_value
        time.sleep(1)
    raise TimeoutError(f"{label} timed out; last value: {last_value!r}")


def output_file(url_or_path: str) -> Path:
    if not url_or_path:
        raise AssertionError("empty output path")
    normalized = url_or_path.replace("\\", "/")
    if "/output/" in normalized:
        return PROJECT_ROOT / "output" / normalized.split("/output/", 1)[1]
    return Path(url_or_path)


def assert_real_image(path: Path) -> None:
    assert path.exists() and path.stat().st_size > 4096, f"missing or tiny image: {path}"
    with Image.open(path) as image:
        image.verify()


def assert_media_streams(video_path: Path) -> None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,duration",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    stream_types = {stream.get("codec_type") for stream in streams}
    assert "video" in stream_types, f"video stream missing: {stream_types}"
    assert "audio" in stream_types, f"audio stream missing: {stream_types}"
    video_duration = max(float(stream.get("duration") or 0) for stream in streams if stream.get("codec_type") == "video")
    audio_duration = max(float(stream.get("duration") or 0) for stream in streams if stream.get("codec_type") == "audio")
    assert audio_duration >= video_duration - 0.5, f"audio is shorter than video: audio={audio_duration}, video={video_duration}"


def main() -> None:
    health = get("/health")
    assert health.get("status") == "ok", health

    project = post(
        "/api/project",
        {
            "title": "全链路冒烟测试",
            "style": "anime",
            "genre": "测试",
            "resolution": "720p",
            "output_format": "9:16",
            "platform": "douyin",
        },
    )
    project_id = project["id"]
    print(f"PROJECT {project_id}")

    generated = post(
        "/api/script/generate",
        {
            "project_id": project_id,
            "prompt": "一个年轻创作者把灵感做成漫剧成片的温暖故事",
            "style": "anime",
            "genre": "原创短剧",
            "target_duration": 20,
        },
    )
    script = generated["script"]
    assert len(script.strip()) > 80, "script is too short"
    print(f"SCRIPT {len(script)} chars")

    parse_result = post(
        "/api/script/parse",
        {
            "project_id": project_id,
            "user_input": script,
            "style": "anime",
            "output_format": "9:16",
            "resolution": "720p",
            "platform": "douyin",
            "target_duration": 20,
        },
    )
    assert parse_result["status"] == "started", parse_result

    def assets_ready():
        status = get(f"/api/project/{project_id}").get("status")
        if status == "error":
            raise RuntimeError("project entered error state during asset generation")
        shots = get(f"/api/shot/{project_id}/shots")
        board = get(f"/api/asset/{project_id}/board")
        refs_ready = all(item.get("reference_images") for item in board.get("characters", []))
        if shots and board.get("characters") and refs_ready:
            return shots
        return None

    shots = wait_for(assets_ready, timeout=300, label="asset generation")
    assert len(shots) >= 1, "no shots generated"
    print(f"ASSETS {len(shots)} shots")

    first_shot = shots[0]
    # 对白为结构化列表；编辑时保持原说话人，仅追加文本。
    dialogue_lines = first_shot.get("dialogue") or []
    first_line = dialogue_lines[0] if dialogue_lines else {"speaker": "", "line": "我看见故事成形了"}
    edit_result = put(
        f"/api/shot/{first_shot['id']}",
        {
            "duration": 1.0,
            "dialogue": [
                {
                    "speaker": first_line.get("speaker", ""),
                    "line": (first_line.get("line") or "我看见故事成形了") + "，这句来自冒烟测试。",
                }
            ],
        },
    )
    assert edit_result["needs_render"] is True, edit_result
    print("EDIT ok")

    storyboard_result = post(f"/api/shot/{project_id}/generate-storyboard")
    assert storyboard_result["status"] == "storyboard_started", storyboard_result

    def storyboard_ready():
        status = get(f"/api/project/{project_id}").get("status")
        if status == "error":
            raise RuntimeError("project entered error state during storyboard generation")
        next_shots = get(f"/api/shot/{project_id}/shots")
        if next_shots and all(shot.get("storyboard_path") or shot.get("image_path") for shot in next_shots):
            return next_shots
        return None

    shots = wait_for(storyboard_ready, timeout=300, label="storyboard generation")
    for shot in shots:
        assert_real_image(output_file(shot.get("storyboard_path") or shot["image_path"]))
    print(f"STORYBOARD {len(shots)} shots")

    for shot in shots:
        approve_result = post(f"/api/shot/{shot['id']}/approve-storyboard", {"approved": True})
        assert approve_result["approved"] is True, approve_result

    confirm_result = post(f"/api/shot/{project_id}/confirm-storyboard")
    assert confirm_result["status"] == "storyboard_approved", confirm_result
    print("CONFIRM ok")

    for shot in shots:
        video_result = post(f"/api/shot/{shot['id']}/generate-video", {"force": False})
        assert video_result["status"] == "video_generating", video_result

        def shot_video_ready():
            next_shots = get(f"/api/shot/{project_id}/shots")
            current = next((item for item in next_shots if item["id"] == shot["id"]), None)
            if current and current.get("status") == "failed":
                raise RuntimeError(f"shot video generation failed: {current}")
            if current and current.get("video_path"):
                return current
            return None

        generated_shot = wait_for(shot_video_ready, timeout=360, label=f"shot video {shot['id']}")
        video_file = output_file(generated_shot["video_path"])
        assert video_file.exists() and video_file.stat().st_size > 4096, f"missing or tiny shot video: {video_file}"
        if generated_shot.get("dialogue"):
            audio_path = output_file(generated_shot["audio_path"])
            assert audio_path.exists() and audio_path.stat().st_size > 1024, f"missing or tiny audio: {audio_path}"
        print(f"SHOT_VIDEO {shot['id']} ok")

    render_result = post(
        "/api/render",
        {
            "project_id": project_id,
            "output_format": "9:16",
            "resolution": "720p",
        },
    )
    assert render_result["status"] == "rendering", render_result

    final_video = OUTPUT_ROOT / project_id / "output" / "final.mp4"

    def final_video_ready():
        render_status = get(f"/api/render/{project_id}/status")
        if render_status.get("status") == "error":
            raise RuntimeError(f"render entered error state: {render_status}")
        shots_with_audio = get(f"/api/shot/{project_id}/shots")
        has_audio = any(shot.get("audio_path") for shot in shots_with_audio)
        if (
            final_video.exists()
            and final_video.stat().st_size > 4096
            and render_status.get("status") == "completed"
            and has_audio
        ):
            return {"status": render_status.get("status"), "size": final_video.stat().st_size}
        return None

    video_state = wait_for(final_video_ready, timeout=240, label="final video render")
    final_shots = get(f"/api/shot/{project_id}/shots")
    assert any(shot.get("audio_path") for shot in final_shots), "no audio files generated"
    for shot in final_shots:
        if shot.get("audio_path"):
            audio_path = output_file(shot["audio_path"])
            assert audio_path.exists() and audio_path.stat().st_size > 1024, f"missing or tiny audio: {audio_path}"
    assert_media_streams(final_video)
    print(f"VIDEO {final_video} {video_state['size']} bytes")
    print("FULL_FLOW_OK")


if __name__ == "__main__":
    main()
