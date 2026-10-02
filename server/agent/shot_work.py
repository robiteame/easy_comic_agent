"""逐镜头 fan-out/fan-in 与现有 route 步骤的幂等适配器。

每个 worker 只负责一个 shot_id，成功/失败独立落检查点；fan-in 聚合时保留成功
结果，单个镜头异常绝不会取消或丢弃其它镜头的结果。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from collections.abc import Iterable
from typing import Any

from .checkpoints import CheckpointStore, fingerprint
from .contracts import (
    FailureKind,
    ShotArtifact,
    StageName,
    StageStatus,
    VideoCandidateRecord,
    VideoCandidateSelection,
    default_quality_profile,
)
from .decision import classify_failure


async def run_shot_fanout(
    *,
    project_id: str,
    shot_versions: dict[str, int],
    stage: StageName | str,
    worker: Callable[[str, int], Awaitable[dict[str, Any] | ShotArtifact]],
    checkpoint: CheckpointStore | None = None,
    concurrency: int = 3,
    reuse: bool = True,
    run_id: str = "auto",
    input_fingerprint: str = "",
) -> dict[str, Any]:
    """逐镜头并行执行；返回 successes/failures/degraded 三组，不互相覆盖。"""

    items = sorted((str(shot_id), int(version or 1)) for shot_id, version in shot_versions.items())
    if not items:
        return {"successes": [], "failures": [], "degraded": [], "skipped": [], "pending": [], "artifacts": []}

    semaphore = asyncio.Semaphore(max(1, min(8, int(concurrency))))
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    degraded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []

    async def run_one(shot_id: str, version: int) -> None:
        if checkpoint and reuse:
            reusable = checkpoint.reusable_shot_artifact(
                shot_id,
                _stage_key(stage),
                shot_version=version,
                input_fingerprint=input_fingerprint or None,
            )
            if reusable and not _artifact_file_ok(reusable, stage):
                reusable = None
            if reusable:
                skipped.append(reusable)
                artifacts.append(reusable)
                if reusable.get("status") == "degraded":
                    degraded.append(reusable)
                else:
                    successes.append(reusable)
                return
        started = time.monotonic()
        try:
            async with semaphore:
                raw = await worker(shot_id, version)
            artifact = _artifact_from(
                raw,
                shot_id=shot_id,
                version=version,
                stage=stage,
                started=started,
                project_id=project_id,
                run_id=run_id,
                input_fingerprint=input_fingerprint,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = classify_failure(stage=stage, message=str(exc), shot_id=shot_id, kind=_kind_from_exception(exc, stage))
            artifact = ShotArtifact(
                project_id=project_id,
                run_id=run_id,
                input_fingerprint=input_fingerprint,
                shot_id=shot_id,
                shot_version=version,
                stage=stage,
                status=StageStatus.FAILED,
                failure=failure,
                duration_ms=int((time.monotonic() - started) * 1000),
                output_fingerprint=fingerprint({"shot_id": shot_id, "version": version, "stage": str(stage), "error": str(exc)}),
            )
        row = artifact.model_dump(mode="json")
        artifacts.append(row)
        if artifact.status is StageStatus.SUCCEEDED:
            successes.append(row)
        elif artifact.status is StageStatus.DEGRADED:
            degraded.append(row)
        elif artifact.status is StageStatus.SKIPPED:
            skipped.append(row)
        else:
            failures.append(row)
        if checkpoint:
            checkpoint.save_shot_artifact(
                shot_id,
                _stage_key(stage),
                shot_version=version,
                status=artifact.status.value,
                path=artifact.path,
                score=artifact.score,
                provider=artifact.provider,
                cost_micro=artifact.cost_micro,
                duration_ms=artifact.duration_ms,
                failure=artifact.failure,
                metrics=artifact.metrics,
                output_fingerprint=artifact.output_fingerprint,
                input_fingerprint=artifact.input_fingerprint,
                extra={
                    "video_candidates": artifact.video_candidates,
                    "selected_video_candidate_id": artifact.selected_video_candidate_id,
                    "candidate_selection": artifact.candidate_selection,
                    "decision_trace": artifact.decision_trace,
                    "expected_duration_s": artifact.expected_duration_s,
                    "expected_aspect_ratio": artifact.expected_aspect_ratio,
                    "audio_path": artifact.audio_path,
                    "tail_frame_path": artifact.tail_frame_path,
                },
            )

    # 每个 worker 都由 run_one 捕获普通异常；单个镜头失败只进入 failures，
    # 不会取消同一 gather 中其它镜头。外层任务取消仍保持原生传播。
    await asyncio.gather(*(run_one(shot_id, version) for shot_id, version in items))
    return {
        "successes": successes,
        "failures": failures,
        "degraded": degraded,
        "skipped": skipped,
        "pending": [],
        "artifacts": artifacts,
        "success_count": len(successes),
        "failure_count": len(failures),
        "degraded_count": len(degraded),
        "skipped_count": len(skipped),
        "pending_count": 0,
    }


def fan_in_shot_results(artifacts: Iterable[dict[str, Any] | ShotArtifact]) -> dict[str, Any]:
    """把逐镜头产物聚合成稳定 fan-in 契约。

    ``pending`` 保留 running/pending 等尚未收敛的镜头；正常 fan-in 完成后为空，
    但仍始终存在该字段，调用方不需要判断键是否存在。
    """

    rows: list[dict[str, Any]] = []
    for item in artifacts:
        row = item.model_dump(mode="json") if isinstance(item, ShotArtifact) else dict(item or {})
        rows.append(row)
    groups: dict[str, list[dict[str, Any]]] = {
        "successes": [],
        "failures": [],
        "degraded": [],
        "skipped": [],
        "pending": [],
    }
    for row in rows:
        status = str(row.get("status") or "")
        if status == StageStatus.SUCCEEDED.value:
            groups["successes"].append(row)
        elif status == StageStatus.DEGRADED.value:
            groups["degraded"].append(row)
        elif status == StageStatus.SKIPPED.value:
            groups["skipped"].append(row)
        elif status in {StageStatus.PENDING.value, StageStatus.RUNNING.value, StageStatus.RECOVERING.value}:
            groups["pending"].append(row)
        else:
            groups["failures"].append(row)
    return {
        **groups,
        "artifacts": rows,
        "success_count": len(groups["successes"]),
        "failure_count": len(groups["failures"]),
        "degraded_count": len(groups["degraded"]),
        "skipped_count": len(groups["skipped"]),
        "pending_count": len(groups["pending"]),
    }


async def generate_storyboard_shot(
    shot_id: str,
    expected_version: int,
    *,
    project_id: str,
    provider_override: str = "",
    preferred_size: str = "",
    capability_mode: str = "auto",
    confirm_capability_downgrade: bool = False,
    seed_override: int | None = None,
    recovery_revisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """单镜头故事板适配器：复用 route 层实现，但不把单镜头结果写成项目级失败。"""

    from api.routes.shot import _run_storyboard_generation_impl

    await _run_storyboard_generation_impl(
        project_id,
        [str(shot_id)],
        {str(shot_id): int(expected_version)},
        emit_project_result=False,
        provider_override=provider_override,
        preferred_size=preferred_size,
        capability_mode=capability_mode,
        confirm_capability_downgrade=confirm_capability_downgrade,
        seed_override=seed_override,
        recovery_revisions=recovery_revisions,
    )
    return _db_artifact(shot_id, stage="image_generation", path_fields=("storyboard_path", "image_path"), expected_version=expected_version)


async def generate_video_shot(
    shot_id: str,
    expected_version: int,
    *,
    project_id: str,
    provider_override: str = "",
    resolution_override: str = "",
    capability_mode: str = "auto",
    confirm_capability_downgrade: bool = False,
    quality_profile: str = "standard",
    retry_of_candidate_id: str = "",
    seed_override: int | None = None,
    recovery_revisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from api.routes.shot import _run_single_shot_video

    candidate_count = max(1, int(default_quality_profile(quality_profile).candidate_count))
    generation = await _run_single_shot_video(
        str(shot_id),
        force=True,
        expected_version=int(expected_version),
        provider_override=provider_override,
        resolution_override=resolution_override,
        capability_mode=capability_mode,
        confirm_capability_downgrade=confirm_capability_downgrade,
        candidate_count=candidate_count,
        recovery_budget=max(0, int(default_quality_profile(quality_profile).max_recovery_attempts)),
        strict_structural_selection=True,
        retry_of_candidate_id=str(retry_of_candidate_id or ""),
        seed_override=seed_override,
        recovery_revisions=recovery_revisions,
    )
    artifact = _db_artifact(shot_id, stage="video_generation", path_fields=("video_path",), expected_version=expected_version)
    candidates = [VideoCandidateRecord.model_validate(item) for item in generation.get("video_candidates", [])]
    selection_raw = generation.get("candidate_selection") or {}
    artifact.update({
        "video_candidates": [item.model_dump(mode="json") for item in candidates],
        "selected_video_candidate_id": str(generation.get("selected_video_candidate_id") or ""),
        "candidate_selection": VideoCandidateSelection.model_validate(selection_raw).model_dump(mode="json") if selection_raw else None,
        "decision_trace": generation.get("decision_trace") or None,
    })
    return artifact


async def generate_audio_shot(shot_id: str, expected_version: int, *, project_id: str) -> dict[str, Any]:
    """外部 TTS 适配器；native audio 镜头不调用 TTS route/service。"""

    shot = _shot_audio_context(shot_id)
    if _resolved_audio_mode(shot) == "native":
        return {
            "shot_id": str(shot_id),
            "shot_version": int(expected_version),
            "stage": StageName.AUDIO_PRODUCTION.value,
            "status": StageStatus.SKIPPED.value,
            "path": "",
            "score": 1.0,
            "metrics": [{"name": "audio_source", "value": "native"}, {"name": "external_tts", "value": False}],
        }

    from api.routes.shot import _run_single_shot_audio

    await _run_single_shot_audio(str(shot_id), int(expected_version))
    return _db_artifact(shot_id, stage="audio_production", path_fields=("audio_path",), expected_version=expected_version)


def _shot_audio_context(shot_id: str) -> dict[str, Any]:
    from db import SessionLocal
    from models import Shot

    db = SessionLocal()
    try:
        shot = db.query(Shot).filter(Shot.id == str(shot_id)).first()
        if shot is None:
            raise RuntimeError("镜头不存在")
        return {
            "dialogue": shot.dialogue,
            "shot_type": shot.shot_type,
            "audio_mode": getattr(shot, "audio_mode", ""),
            "continuity_profile": json.loads(shot.continuity_profile or "{}"),
        }
    finally:
        db.close()


def _resolved_audio_mode(shot: dict[str, Any]) -> str:
    from services.audio_routing import resolve_audio_mode

    return resolve_audio_mode(shot)


def _stage_key(stage: StageName | str) -> str:
    return StageName(stage).value


def _artifact_file_ok(row: dict[str, Any], stage: StageName | str) -> bool:
    path = str(row.get("path") or "")
    if not path:
        # 无对白、native audio、检查点恢复等允许空产物；其它阶段必须有文件。
        return StageName(stage) is StageName.AUDIO_PRODUCTION
    try:
        from pathlib import Path

        target = Path(path)
        if not target.is_file():
            return False
        if StageName(stage) is StageName.IMAGE_GENERATION:
            from services.structural_validation import validate_image_file

            return bool(validate_image_file(path).get("passed"))
        minimum = 4096 if StageName(stage) is StageName.VIDEO_GENERATION else 1024
        return target.stat().st_size >= minimum
    except Exception:
        return False


def _artifact_from(
    raw: Any,
    *,
    shot_id: str,
    version: int,
    stage: StageName | str,
    started: float,
    project_id: str = "",
    run_id: str = "auto",
    input_fingerprint: str = "",
) -> ShotArtifact:
    duration_ms = int((time.monotonic() - started) * 1000)
    if isinstance(raw, ShotArtifact):
        return raw.model_copy(
            update={
                "project_id": raw.project_id or project_id,
                "run_id": raw.run_id or run_id,
                "input_fingerprint": raw.input_fingerprint or input_fingerprint,
            }
        )
    if not isinstance(raw, dict):
        return ShotArtifact(
            project_id=project_id,
            run_id=run_id,
            input_fingerprint=input_fingerprint,
            shot_id=shot_id,
            shot_version=version,
            stage=stage,
            status=StageStatus.SUCCEEDED,
            path=str(raw or ""),
            duration_ms=duration_ms,
        )
    status_text = str(raw.get("status") or StageStatus.SUCCEEDED.value)
    try:
        status = StageStatus(status_text)
    except ValueError:
        status = StageStatus.SUCCEEDED if raw.get("path") else StageStatus.FAILED
    failure = raw.get("failure")
    return ShotArtifact(
        project_id=str(raw.get("project_id") or project_id),
        run_id=str(raw.get("run_id") or run_id),
        input_fingerprint=str(raw.get("input_fingerprint") or input_fingerprint),
        shot_id=str(raw.get("shot_id") or shot_id),
        shot_version=int(raw.get("shot_version") or version),
        stage=raw.get("stage") or stage,
        status=status,
        path=str(raw.get("path") or ""),
        score=float(raw.get("score") or 0.0),
        provider=str(raw.get("provider") or ""),
        cost_micro=raw.get("cost_micro"),
        duration_ms=int(raw.get("duration_ms") or duration_ms),
        failure=failure,
        metrics=list(raw.get("metrics") or []),
        output_fingerprint=str(raw.get("output_fingerprint") or fingerprint(raw)),
        video_candidates=[VideoCandidateRecord.model_validate(item) for item in raw.get("video_candidates") or []],
        selected_video_candidate_id=str(raw.get("selected_video_candidate_id") or ""),
        candidate_selection=VideoCandidateSelection.model_validate(raw["candidate_selection"]) if raw.get("candidate_selection") else None,
        decision_trace=dict(raw.get("decision_trace") or {}) or None,
        expected_duration_s=_optional_positive_float(raw.get("expected_duration_s")),
        expected_aspect_ratio=_optional_positive_float(raw.get("expected_aspect_ratio")),
        audio_path=str(raw.get("audio_path") or ""),
        tail_frame_path=str(raw.get("tail_frame_path") or ""),
    )


def _optional_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _db_artifact(shot_id: str, *, stage: str, path_fields: tuple[str, ...], expected_version: int) -> dict[str, Any]:
    from db import SessionLocal
    from models import Project, Shot

    db = SessionLocal()
    try:
        shot = db.query(Shot).filter(Shot.id == str(shot_id)).first()
        if shot is None:
            raise RuntimeError("镜头不存在")
        if int(shot.version or 1) != int(expected_version):
            return {
                "shot_id": str(shot_id),
                "shot_version": int(expected_version),
                "stage": stage,
                "status": StageStatus.FAILED.value,
                "failure": classify_failure(stage=stage, kind=FailureKind.VERSION_CONFLICT, shot_id=str(shot_id), message="shot version changed").model_dump(mode="json"),
            }
        path = next((str(getattr(shot, field) or "") for field in path_fields if getattr(shot, field, "")), "")
        # 音频阶段对无对白 / native 音频镜头允许空路径并视为成功；
        # 这类镜头由视频模型负责音频或根本无需配音。
        audio_skipped = False
        if stage == "audio_production" and not path:
            try:
                from services.audio_routing import resolve_audio_mode
                from services.shot_dialogue import parse_shot_dialogue

                audio_skipped = not parse_shot_dialogue(shot.dialogue) or resolve_audio_mode({
                    "dialogue": shot.dialogue,
                    "shot_type": shot.shot_type,
                    "continuity_profile": json.loads(shot.continuity_profile or "{}"),
                }) == "native"
            except Exception:
                audio_skipped = False
        artifact: dict[str, Any] = {
            "shot_id": str(shot_id),
            "shot_version": int(shot.version or 1),
            "stage": stage,
            "status": StageStatus.SUCCEEDED.value if path or audio_skipped else StageStatus.FAILED.value,
            "path": path,
            "score": 1.0 if path else 0.0,
            "metrics": [{"name": "audio_source", "value": "native"}] if audio_skipped and path == "" and shot.dialogue else [],
        }
        if stage == "video_generation":
            # 视频 Critic 的技术检查需要执行计划上下文；从存档计划与镜头
            # 字段读取，缺什么就跳过什么维度（不猜测、不冒充通过）。
            artifact.update(_video_check_context(db, shot))
        return artifact
    finally:
        db.close()


def _video_check_context(db: Any, shot: Any) -> dict[str, Any]:
    """为视频质量检查补齐上下文：计划时长、目标画幅、配音与尾帧路径。"""

    context: dict[str, Any] = {}
    try:
        plan = json.loads(shot.continuity_profile or "{}").get("execution_plan") or {}
        generation_s = float(plan.get("provider_generation_duration_s") or 0.0)
        if generation_s > 0:
            context["expected_duration_s"] = round(generation_s, 3)
    except Exception:
        pass
    if shot.audio_path:
        context["audio_path"] = str(shot.audio_path)
    if shot.last_frame_path:
        context["tail_frame_path"] = str(shot.last_frame_path)
    try:
        project = db.query(Project).filter(Project.id == shot.project_id).first()
        output_format = str(getattr(project, "output_format", "") or "")
        width, _, height = output_format.partition(":")
        ratio = float(width) / float(height)
        if ratio > 0:
            context["expected_aspect_ratio"] = round(ratio, 4)
    except Exception:
        pass
    return context


def _kind_from_exception(exc: BaseException, stage: StageName | str) -> FailureKind | None:
    text = str(exc)
    lowered = text.lower()
    if "version" in lowered or "版本" in text or "stale" in lowered:
        return FailureKind.VERSION_CONFLICT
    if "reference" in lowered or "参考图" in text:
        return FailureKind.PROVIDER_REFERENCE_UNSUPPORTED
    if "dialogue" in lowered or "对白" in text or "台词" in text:
        return FailureKind.DIALOGUE_TOO_LONG
    if StageName(stage) is StageName.VIDEO_GENERATION:
        return FailureKind.VIDEO_FAILED
    if StageName(stage) is StageName.AUDIO_PRODUCTION:
        return FailureKind.AUDIO_FAILED
    if StageName(stage) is StageName.IMAGE_GENERATION:
        return FailureKind.IMAGE_FAILED
    return None


__all__ = [
    "fan_in_shot_results",
    "generate_audio_shot",
    "generate_storyboard_shot",
    "generate_video_shot",
    "run_shot_fanout",
]
