from __future__ import annotations

import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from agent.critic import build_final_report, critique_final


def test_final_critique_locates_current_failed_artifact_by_shot_and_stage() -> None:
    report = critique_final(
        {
            "run_status": "running",
            "shot_artifacts": [
                {
                    "shot_id": "shot-7",
                    "stage": "video_generation",
                    "status": "failed",
                    "failure": {"kind": "video_failed", "message": "provider failed"},
                }
            ],
        }
    )
    issue = next(item for item in report.issues if item.code == "video_failed")
    assert issue.shot_id == "shot-7"
    assert issue.details["source_stage"] == "video_generation"
    assert report.passed is False


def test_visual_pending_is_not_promoted_to_failed_issue() -> None:
    report = critique_final(
        {
            "run_status": "running",
            "critiques": [
                {
                    "stage": "video_review",
                    "passed": True,
                    "metrics": [{"name": "visual_quality_pending", "passed": None}],
                }
            ],
        }
    )
    assert report.passed is True
    assert any(item.code == "visual_quality_pending" and item.severity == "info" for item in report.issues)
    assert not any(item.code == "visual_quality_pending" and item.severity == "error" for item in report.issues)


def test_final_report_marks_trace_only_recovery_as_attempted_and_redacts_path() -> None:
    state = {
        "run_status": "running",
        "output_path": "/Users/alice/private/output.mp4",
        "decision_traces": [
            {
                "trace_id": "t1",
                "stage": "video_generation",
                "selected": {"strategy": "change_seed", "shot_ids": ["s1"]},
                "reason": "api_key=sk-secret123456",
            }
        ],
        "shot_artifacts": [],
        "critiques": [],
    }
    report = build_final_report(state)
    repair = report["automatic_repairs"][0]
    assert repair["outcome"] == "attempted"
    assert "/Users/" not in str(report)
    assert "sk-secret123456" not in str(report)
    assert report["final_choice"]["output_path"] == ""


def test_passed_final_critique_completes_running_state() -> None:
    report = critique_final({"run_status": "running", "quality_threshold": 0.5})
    from agent.critic import extract_final_report

    final = extract_final_report(report)
    assert report.passed is True
    assert final["final_choice"]["status"] == "completed"


def test_trace_only_degraded_has_no_false_path_available() -> None:
    report = build_final_report(
        {
            "decision_traces": [
                {
                    "trace_id": "d1",
                    "stage": "video_generation",
                    "selected": {"strategy": "degraded_publish", "shot_ids": ["s1"]},
                    "reason": "no candidate",
                }
            ],
            "shot_artifacts": [],
        }
    )
    assert report["degradations"][0]["path_available"] is False


def test_degraded_path_requires_structural_pass() -> None:
    no_check = build_final_report(
        {
            "degraded_published": True,
            "shot_artifacts": [
                {"shot_id": "s1", "status": "degraded", "path": "/tmp/video.mp4", "output_fingerprint": "fp"}
            ],
        }
    )
    assert no_check["degradations"][0]["path_available"] is False
    checked = build_final_report(
        {
            "degraded_published": True,
            "shot_artifacts": [
                {"shot_id": "s1", "status": "degraded", "path": "/tmp/video.mp4", "structural_passed": True}
            ],
        }
    )
    assert checked["degradations"][0]["path_available"] is True


def test_report_redacts_nested_prompt_changes_and_risks() -> None:
    report = build_final_report(
        {
            "decision_traces": [
                {
                    "trace_id": "t2",
                    "stage": "image_generation",
                    "selected": {
                        "strategy": "revise_prompt",
                        "prompt_changes": {"nested": {"text": "api_key=sk-secret123456"}},
                    },
                }
            ],
            "shot_artifacts": [
                {
                    "shot_id": "s1",
                    "status": "failed",
                    "stage": "video_generation",
                    "failure": {"message": "/Users/a/private.mp4"},
                }
            ],
        }
    )
    text = str(report)
    assert "sk-secret123456" not in text
    assert "/Users/a" not in text
