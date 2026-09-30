"""逐镜头 fan-out/fan-in 与现有 route 步骤的幂等适配器。

每个 worker 只负责一个 shot_id，成功/失败独立落检查点；fan-in 聚合时保留成功
结果，单个镜头异常绝不会取消或丢弃其它镜头的结果。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from .checkpoints import CheckpointStore, fingerprint
from .contracts import FailureKind, ShotArtifact, StageName, StageStatus, default_quality_profile
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
) -> dict[str, Any]:
    """逐镜头并行执行；返回 successes/failures/degraded 三组，不互相覆盖。"""

    items = sorted((str(shot_id), int(version or 1)) for shot_id, version in shot_versions.items())
    if not items:
        return {"successes": [], "failures": [], "degraded": [], "artifacts": [], "skipped": []}

    semaphore = asyncio.Semaphore(max(1, min(8, int(concurrency))))
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    degraded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []

    async def run_one(shot_id: str, version: int) -> None:
        if checkpoint and reuse:
            reusable = checkpoint.reusable_shot_artifact(shot_id, _stage_key(stage), shot_version=version)
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
            artifact = _artifact_from(raw, shot_id=shot_id, version=version, stage=stage, started=started)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = classify_failure(stage=stage, message=str(exc), shot_id=shot_id, kind=_kind_from_exception(exc, stage))
            artifact = ShotArtifact(
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
            )

    await asyncio.gather(*(run_one(shot_id, version) for shot_id, version in items))
    return {
        "successes": successes,
        "failures": failures,
        "degraded": degraded,
        "skipped": skipped,
        "artifacts": artifacts,
        "success_count": len(successes),
        "failure_count": len(failures),
        "degraded_count": len(degraded),
    }


async def generate_storyboard_shot(
    shot_id: str,
    expected_version: int,
    *,
    project_id: str,
    provider_override: str = "",
    preferred_size: str = "",
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
    )
    return _db_artifact(shot_id, stage="image_generation", path_fields=("storyboard_path", "image_path"), expected_version=expected_version)


async def generate_video_shot(
    shot_id: str,
    expected_version: int,
    *,
    project_id: str,
    provider_override: str = "",
    resolution_override: str = "",
) -> dict[str, Any]:
    from api.routes.shot import _run_single_shot_video

    await _run_single_shot_video(
        str(shot_id),
        force=True,
        expected_version=int(expected_version),
        provider_override=provider_override,
        resolution_override=resolution_override,
    )
    return _db_artifact(shot_id, stage="video_generation", path_fields=("video_path",), expected_version=expected_version)


async def generate_audio_shot(shot_id: str, expected_version: int, *, project_id: str) -> dict[str, Any]:
    from api.routes.shot import _run_single_shot_audio

    await _run_single_shot_audio(str(shot_id), int(expected_version))
    return _db_artifact(shot_id, stage="audio_production", path_fields=("audio_path",), expected_version=expected_version)


def _stage_key(stage: StageName | str) -> str:
    return StageName(stage).value


def _artifact_file_ok(row: dict[str, Any], stage: StageName | str) -> bool:
    path = str(row.get("path") or "")
    if not path:
        # 无对白音频、检查点恢复等允许空产物；其它阶段必须有文件。
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


def _artifact_from(raw: Any, *, shot_id: str, version: int, stage: StageName | str, started: float) -> ShotArtifact:
    duration_ms = int((time.monotonic() - started) * 1000)
    if isinstance(raw, ShotArtifact):
        return raw
    if not isinstance(raw, dict):
        return ShotArtifact(shot_id=shot_id, shot_version=version, stage=stage, status=StageStatus.SUCCEEDED, path=str(raw or ""), duration_ms=duration_ms)
    status_text = str(raw.get("status") or StageStatus.SUCCEEDED.value)
    try:
        status = StageStatus(status_text)
    except ValueError:
        status = StageStatus.SUCCEEDED if raw.get("path") else StageStatus.FAILED
    failure = raw.get("failure")
    return ShotArtifact(
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
    )


def _db_artifact(shot_id: str, *, stage: str, path_fields: tuple[str, ...], expected_version: int) -> dict[str, Any]:
    from db import SessionLocal
    from models import Shot

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
        return {
            "shot_id": str(shot_id),
            "shot_version": int(shot.version or 1),
            "stage": stage,
            "status": StageStatus.SUCCEEDED.value if path else StageStatus.FAILED.value,
            "path": path,
            "score": 1.0 if path else 0.0,
        }
    finally:
        db.close()


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
    "generate_audio_shot",
    "generate_storyboard_shot",
    "generate_video_shot",
    "run_shot_fanout",
]
