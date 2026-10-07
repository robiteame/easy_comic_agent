"""Static audit for the focused project/episode workflow iteration."""

from __future__ import annotations

import json
from pathlib import Path

from services.style_templates import STYLE_TEMPLATES, style_prompt_params

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _assert(condition: bool, name: str, evidence: str) -> dict:
    if not condition:
        raise AssertionError(f"{name}: {evidence}")
    return {"name": name, "evidence": evidence}


def _read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def main() -> None:
    checks: list[dict] = []
    main_workspace = _read("client/src/renderer/components/MainWorkspace.tsx")
    left_sidebar = _read("client/src/renderer/components/LeftSidebar.tsx")
    api_ts = _read("client/src/renderer/services/api.ts")
    style_ts = _read("client/src/renderer/constants/styleTemplates.ts")
    project_route = _read("server/api/routes/project.py")
    script_route = _read("server/api/routes/script.py")
    shot_route = _read("server/api/routes/shot.py")
    image_service = _read("server/services/image_service.py")
    video_service = _read("server/services/video_service.py")

    checks.append(
        _assert(
            len(STYLE_TEMPLATES) >= 8
            and all(style_prompt_params(key).get("prompt_prefix") for key in STYLE_TEMPLATES)
            and all(key in style_ts for key in STYLE_TEMPLATES)
            and "STYLE_OPTIONS" in main_workspace
            and "STYLE_DESCRIPTIONS" in main_workspace
            and "style_prompt_params" in image_service
            and "style_prompt_params" in video_service
            and "_storyboard_style_params" in shot_route,
            "style_options_and_prompt_templates_wired",
            f"style_count={len(STYLE_TEMPLATES)}",
        )
    )

    checks.append(
        _assert(
            "projectApi.get(projectId)" in main_workspace
            and "setScript(projectDetail.input_text || '')" in main_workspace
            and "parentProjectTitle" in main_workspace
            and "episode-title-input" in main_workspace
            and "commitEpisodeTitle" in main_workspace,
            "script_tab_scoped_to_selected_episode",
            "MainWorkspace reloads selected project input_text and exposes parent/episode title",
        )
    )

    checks.append(
        _assert(
            "id: 'create'" in left_sidebar
            and "id: 'episode'" in left_sidebar
            and "id: 'import-video'" in left_sidebar
            and "导入剧本" not in left_sidebar
            and "生成分镜" not in left_sidebar
            and "合成成片" not in left_sidebar
            and "projectApi.importVideo" in left_sidebar
            and "video/mp4" in left_sidebar,
            "left_sidebar_three_entries_only",
            "Sidebar keeps New Project, New Episode and Import Final Video only",
        )
    )

    checks.append(
        _assert(
            "first_episode_title" in project_route
            and '"first_episode"' in project_route
            and 'project_type="episode"' in project_route
            and "first_episode_title" in api_ts
            and "project.first_episode || project" in main_workspace,
            "series_create_auto_first_episode",
            "Backend creates first episode and frontend selects it as active project",
        )
    )

    checks.append(
        _assert(
            '"/{project_id}/import-video"' in project_route
            and "importVideo" in api_ts
            and "final.mp4" in project_route
            and "setVideoPath(result.video_path" in left_sidebar,
            "import_final_video_flow_present",
            "Uploaded final video is persisted to project output and shown in UI",
        )
    )

    checks.append(
        _assert(
            all(
                label in main_workspace
                for label in ["新建剧集", "上传剧本", "AI解析", "资产板", "批量分镜", "逐镜审核", "生成视频"]
            )
            and "scriptApi.upload" in main_workspace
            and "scriptApi.parse" in main_workspace
            and "asset-board-panel" in main_workspace
            and "shotApi.generateStoryboard" in main_workspace
            and "shotApi.approveStoryboard" in main_workspace
            and "shotApi.generateVideo" in main_workspace,
            "visual_full_chain_controls_present",
            "UI exposes episode creation, upload, parse, assets, storyboard, approval and per-shot video actions",
        )
    )

    checks.append(
        _assert(
            "style_prompt_params(style)" in script_route
            and "_ensure_character_reference_images" in script_route
            and "_ensure_scene_baseline_images" in script_route
            and "generate_character_reference" in script_route
            and "generate_scene_baseline_reference" in script_route
            and "reference_images" in script_route
            and "baseline_image_path" in script_route,
            "asset_board_generation_and_persistence_preserved",
            "Parse flow still generates and persists character three-view refs and scene baselines",
        )
    )

    print(json.dumps({"checks": checks, "check_count": len(checks)}, ensure_ascii=False, indent=2))
    print("ITERATION_FLOW_AUDIT_OK")


if __name__ == "__main__":
    main()
