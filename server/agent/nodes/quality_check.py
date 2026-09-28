"""结构检查节点（structural validation）——不是质量认证。

历史上本节点叫 ``quality_check``，但只做「字段是否齐全」的检查，容易让
界面误读为「质量通过」。现在明确为结构检查语义：
- 图片可读性与尺寸检查（存在、可解码、达到最小边长）；
- 空画面/过小文件/失败状态检查；
- 结构结果与人工审核状态分开：``structural_validation`` 只描述结构，
  ``needs_human_review`` 代表人需要介入，``quality_state`` 保持
  ``needs_review`` 直到人工确认。

没有人脸一致性、美学评分或闪烁检测——这些能力未接入，结果里如实标注。
"""

from agent.state import AgentState
from services.structural_validation import validate_image_file

NOTE = "仅结构检查：不包含人脸一致性、美学或闪烁检测（未接入，unsupported）"


async def run(state: AgentState) -> dict:
    """Mark shots that still need attention after generation."""
    updated_shots = []
    issues_found = False

    for shot in state.get("shots", []):
        issues = []
        if shot.get("status") == "failed":
            issues.append("生成失败")
        if not shot.get("image_path"):
            issues.append("缺少画面")
        else:
            image_check = validate_image_file(str(shot.get("image_path")))
            if not image_check["passed"]:
                issues.append("画面未通过结构检查: " + "；".join(image_check["issues"]))
        if shot.get("dialogue") and not shot.get("audio_path"):
            issues.append("缺少配音")
        if not shot.get("scene_description"):
            issues.append("缺少场景描述")

        shot["structural_validation"] = {
            "passed": not issues,
            "issues": issues,
            "note": NOTE,
        }
        if issues:
            shot["status"] = "needs_review"
            shot["visual_notes"] = "，".join(issues)
            issues_found = True
        elif shot.get("status") not in {"done", "video_done"}:
            # Only structural checks are available here. Do not claim face
            # consistency, aesthetics, flicker, or motion quality passed.
            shot["status"] = "structural_check_passed"
            shot["quality_state"] = "needs_review"

        updated_shots.append(shot)

    return {
        "shots": updated_shots,
        "needs_human_review": issues_found,
        "structural_validation_note": NOTE,
        "current_step": "quality_check",
    }
