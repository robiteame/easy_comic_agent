"""生成 Agent 的幂等检查点与版本围栏。

检查点写入 JSON 文件而不是内存：进程重启、任务重试和局部重算都能读到同一次
运行的阶段输出、逐镜头产物、决策树和候选结果。写入复用 atomic_json，读者不会
看到半个 JSON；每个阶段/镜头都带 input/output fingerprint 与 Shot.version，只有
两者同时匹配才允许复用。
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterable
from datetime import UTC
from pathlib import Path
from typing import Any

from config import settings
from services.atomic_json import atomic_write_json, read_json_file

from .contracts import (
    STAGE_ORDER,
    CheckpointRecord,
    ShotArtifact,
    VideoCandidateRecord,
    VideoCandidateStatus,
    stage_contract,
)

_SCHEMA_VERSION = 1
_STORE_LOCK = threading.RLock()
_STORES: dict[tuple[str, str], CheckpointStore] = {}
_STAGE_ORDER = tuple(stage.value for stage in STAGE_ORDER)
_REUSABLE_STATUSES = {"succeeded", "degraded", "skipped"}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(
            (_jsonable(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False)
        )
    if isinstance(value, Path):
        return str(value)
    return value


def fingerprint(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _checkpoint_key(stage: str) -> str:
    return stage_contract(stage).checkpoint_key


def _stage_record_key(stage: str) -> str:
    return _checkpoint_key(stage)


def _shot_record_key(stage: str, shot_id: str, shot_version: int) -> str:
    return f"{_checkpoint_key(stage)}:{shot_id}:v{int(shot_version)}"


class CheckpointStore:
    """单项目、单运行的检查点文件。所有写操作都幂等且原子。"""

    def __init__(self, project_id: str, run_id: str = "auto", *, root: Path | None = None):
        self.project_id = str(project_id)
        self.run_id = str(run_id or "auto")
        self.root = Path(root or settings.CHECKPOINT_PATH)
        self.path = self.root / _safe_name(self.project_id) / f"{_safe_name(self.run_id)}.json"
        self._local = threading.RLock()
        self.data: dict[str, Any] = self._read()

    @classmethod
    def get(cls, project_id: str, run_id: str = "auto", *, root: Path | None = None) -> CheckpointStore:
        key = (str(project_id), str(run_id or "auto"))
        with _STORE_LOCK:
            store = _STORES.get(key)
            if store is None or (root is not None and store.root != Path(root)):
                store = cls(project_id, run_id, root=root)
                _STORES[key] = store
            return store

    def _read(self) -> dict[str, Any]:
        loaded = read_json_file(self.path, default=None)
        if not isinstance(loaded, dict):
            loaded = {}
        loaded.setdefault("schema_version", _SCHEMA_VERSION)
        loaded.setdefault("project_id", self.project_id)
        loaded.setdefault("run_id", self.run_id)
        loaded.setdefault("stages", {})
        loaded.setdefault("shots", {})
        loaded.setdefault("decisions", [])
        loaded.setdefault("events", [])
        loaded.setdefault("status", "pending")
        return loaded

    def save(self) -> None:
        with self._local:
            self.data["schema_version"] = _SCHEMA_VERSION
            self.data["project_id"] = self.project_id
            self.data["run_id"] = self.run_id
            self.data["updated_at"] = _now()
            atomic_write_json(self.path, _jsonable(self.data))

    def snapshot(self) -> dict[str, Any]:
        return _jsonable(self.data)

    def trace_summary(self) -> dict[str, Any]:
        """面向展示的可解释追踪汇总（阶段/镜头/决策/降级/计数）。"""

        return summarize_trace(self.snapshot())

    # --- run/status ---

    def set_status(self, status: str, *, reason: str = "", extra: dict[str, Any] | None = None) -> None:
        with self._local:
            self.data["status"] = str(status)
            self.data["status_reason"] = str(reason or "")
            if extra:
                self.data.setdefault("status_detail", {}).update(_jsonable(extra))
            self.save()

    def set_input_fingerprint(self, value: str, metadata: dict[str, Any] | None = None) -> None:
        with self._local:
            previous = str(self.data.get("input_fingerprint") or "")
            changed = bool(previous and previous != str(value))
            self.data["input_fingerprint"] = str(value)
            if metadata is not None:
                self.data["input_metadata"] = _jsonable(metadata)
            if changed:
                self.invalidate_from(_STAGE_ORDER[0], reason="input_fingerprint_changed")
            self.save()

    # --- stages ---

    def save_record(self, record: CheckpointRecord | dict[str, Any]) -> dict[str, Any]:
        raw = _jsonable(record)
        validated = CheckpointRecord.model_validate(raw)
        row = {**raw, **validated.model_dump(mode="json")}
        with self._local:
            if row["kind"] == "stage":
                self.data.setdefault("stages", {})[str(row["stage"])] = row
            else:
                shot_rows = self.data.setdefault("shots", {}).setdefault(str(row["shot_id"]), {})
                shot_rows[str(row["stage"])] = row
            self.save()
        return row

    def read_record(self, key: str) -> CheckpointRecord | None:
        wanted = str(key)
        for row in self._all_records():
            record = self._coerce_record(row)
            if record and record.key == wanted:
                return record
        return None

    def reuse_record(
        self,
        key: str,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
        input_fingerprint: str | None = None,
        output_fingerprint: str | None = None,
        shot_version: int | None = None,
    ) -> CheckpointRecord | None:
        record = self.read_record(key)
        if record is None or not record.reusable:
            return None
        if not record.identity_matches(
            project_id=str(project_id if project_id is not None else self.project_id),
            run_id=str(run_id if run_id is not None else self.run_id),
            input_fingerprint=input_fingerprint,
            output_fingerprint=output_fingerprint,
            shot_version=shot_version,
        ):
            return None
        return record

    def invalidate_record(self, key: str, *, reason: str = "recompute") -> bool:
        wanted = str(key)
        changed = False
        with self._local:
            for row in self._all_records():
                if str(row.get("key")) == wanted and row.get("valid", True):
                    self._mark_invalid(row, reason)
                    changed = True
            if changed:
                self.save()
        return changed

    def stage(self, stage: str) -> dict[str, Any]:
        record = self.stage_record(stage)
        return record.model_dump(mode="json") if record else {}

    def stage_record(self, stage: str) -> CheckpointRecord | None:
        row = self.data.get("stages", {}).get(str(stage))
        return self._coerce_record(row, kind="stage", stage=stage) if isinstance(row, dict) and row else None

    def stage_is_reusable(
        self,
        stage: str,
        input_fingerprint: str,
        *,
        require_success: bool = True,
        output_fingerprint: str | None = None,
    ) -> bool:
        record = self.reuse_record(
            _stage_record_key(stage),
            input_fingerprint=str(input_fingerprint),
            output_fingerprint=output_fingerprint,
        )
        status = str(record.status.value if record and hasattr(record.status, "value") else "")
        return bool(record and (not require_success or status in _REUSABLE_STATUSES))

    def save_stage(
        self,
        stage: str,
        *,
        status: str,
        input_fingerprint: str,
        output_fingerprint: str = "",
        payload: dict[str, Any] | None = None,
        quality: Any = None,
        failure: Any = None,
        critique: Any = None,
        shot_artifacts: Any = None,
        checkpoint_version: int = 1,
        shot_version: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = self.stage_record(stage)
        effective_output_fingerprint = str(output_fingerprint or "") or fingerprint(
            {"payload": payload or {}, "failure": failure, "status": status}
        )
        if existing and existing.input_fingerprint != str(input_fingerprint):
            # 输入指纹变化只失效下游阶段：本阶段的逐镜头产物若已按新指纹重新
            # 生成，它们是最新事实，不能被这里整体判失效（否则用户改一个镜头
            # 会导致全部镜头在下一轮重跑，违背「只重算变化镜头」）。
            try:
                downstream = _STAGE_ORDER[_STAGE_ORDER.index(str(stage)) + 1 :]
            except ValueError:
                downstream = []
            if downstream:
                self.invalidate_from(downstream[0], reason="input_fingerprint_changed")
        elif existing and existing.output_fingerprint != effective_output_fingerprint:
            try:
                downstream = _STAGE_ORDER[_STAGE_ORDER.index(str(stage)) + 1 :]
            except ValueError:
                downstream = []
            if downstream:
                self.invalidate_from(downstream[0], reason="output_fingerprint_changed")
        row = {
            "key": _stage_record_key(stage),
            "checkpoint_key": _checkpoint_key(stage),
            "kind": "stage",
            "project_id": self.project_id,
            "shot_version": int(shot_version or 0),
            "run_id": self.run_id,
            "stage": str(stage),
            "shot_id": "",
            "status": str(status),
            "input_fingerprint": str(input_fingerprint),
            "output_fingerprint": effective_output_fingerprint,
            "payload": _jsonable(payload or {}),
            "quality": _jsonable(quality or []),
            "failure": _jsonable(failure),
            "critique": _jsonable(critique),
            "shot_artifacts": _jsonable(shot_artifacts or []),
            "checkpoint_version": int(checkpoint_version),
            "valid": str(status) != "invalidated",
            "invalidated_reason": "",
            "invalidated_at": "",
            "created_at": existing.created_at if existing else _now(),
            "saved_at": _now(),
        }
        if extra:
            row.update(_jsonable(extra))
        return self.save_record(row)

    def invalidate_stage(self, stage: str, *, reason: str = "recompute") -> None:
        with self._local:
            row = self.data.setdefault("stages", {}).get(str(stage))
            if row is not None:
                self._mark_invalid(row, reason)
            self.save()

    def invalidate_from(self, stage: str, *, reason: str = "upstream_changed") -> list[str]:
        """使指定阶段及其下游阶段失效，返回受影响阶段。"""

        try:
            start = _STAGE_ORDER.index(str(stage))
        except ValueError:
            return []
        affected = list(_STAGE_ORDER[start:])
        with self._local:
            for name in affected:
                row = self.data.setdefault("stages", {}).get(name)
                if row is not None and row.get("valid", True):
                    self._mark_invalid(row, reason)
                for shot_rows in self.data.setdefault("shots", {}).values():
                    shot_row = shot_rows.get(name) if isinstance(shot_rows, dict) else None
                    if isinstance(shot_row, dict) and shot_row.get("valid", True):
                        self._mark_invalid(shot_row, reason)
            self.save()
        return list(affected)

    # --- shot artifacts ---

    def shot_artifact(self, shot_id: str, stage: str) -> dict[str, Any]:
        return dict(self.data.get("shots", {}).get(str(shot_id), {}).get(str(stage), {}))

    def shot_artifact_record(self, shot_id: str, stage: str) -> CheckpointRecord | None:
        row = self.data.get("shots", {}).get(str(shot_id), {}).get(str(stage))
        return (
            self._coerce_record(row, kind="shot", stage=stage, shot_id=shot_id)
            if isinstance(row, dict) and row
            else None
        )

    def save_shot_artifact(
        self,
        shot_id: str,
        stage: str,
        *,
        shot_version: int,
        status: str,
        path: str = "",
        score: float = 0.0,
        provider: str = "",
        cost_micro: int | None = None,
        duration_ms: int = 0,
        failure: Any = None,
        metrics: Any = None,
        output_fingerprint: str = "",
        input_fingerprint: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = self.shot_artifact_record(shot_id, stage)
        effective_output_fingerprint = str(output_fingerprint or "") or fingerprint(
            {"shot_id": shot_id, "stage": stage, "shot_version": shot_version, "path": path, "status": status}
        )
        if existing and existing.shot_version != int(shot_version):
            self.invalidate_shots([shot_id], from_stage=stage, reason="shot_version_changed")
            self._invalidate_stage_from(stage, reason="shot_version_changed")
        elif existing and input_fingerprint and existing.input_fingerprint != str(input_fingerprint):
            self.invalidate_shots([shot_id], from_stage=stage, reason="input_fingerprint_changed")
            self._invalidate_stage_from(stage, reason="input_fingerprint_changed")
        elif existing and existing.output_fingerprint != effective_output_fingerprint:
            try:
                downstream = _STAGE_ORDER[_STAGE_ORDER.index(str(stage)) + 1 :]
            except ValueError:
                downstream = []
            if downstream:
                self.invalidate_shots([shot_id], from_stage=downstream[0], reason="output_fingerprint_changed")
                self._invalidate_stage_from(downstream[0], reason="output_fingerprint_changed")
        artifact_fields = {
            "project_id": self.project_id,
            "run_id": self.run_id,
            "input_fingerprint": str(input_fingerprint or (existing.input_fingerprint if existing else "")),
            "shot_id": str(shot_id),
            "shot_version": int(shot_version),
            "stage": str(stage),
            "status": str(status),
            "path": str(path or ""),
            "score": float(score),
            "provider": str(provider or ""),
            "cost_micro": int(cost_micro) if cost_micro is not None else None,
            "duration_ms": int(duration_ms),
            "failure": _jsonable(failure),
            "metrics": _jsonable(metrics or []),
            "output_fingerprint": effective_output_fingerprint,
        }
        if extra:
            artifact_fields.update(_jsonable(extra))
        artifact = ShotArtifact.model_validate(artifact_fields)
        row = {
            "key": _shot_record_key(stage, shot_id, shot_version),
            "checkpoint_key": _checkpoint_key(stage),
            "kind": "shot",
            "project_id": self.project_id,
            "run_id": self.run_id,
            "shot_id": str(shot_id),
            "stage": str(stage),
            "shot_version": int(shot_version),
            "status": str(status),
            "path": str(path or ""),
            "score": float(score),
            "provider": str(provider or ""),
            "cost_micro": int(cost_micro) if cost_micro is not None else None,
            "duration_ms": int(duration_ms),
            "failure": _jsonable(failure),
            "metrics": _jsonable(metrics or []),
            "input_fingerprint": artifact.input_fingerprint,
            "output_fingerprint": effective_output_fingerprint,
            "artifact": artifact.model_dump(mode="json"),
            "valid": str(status) != "invalidated",
            "invalidated_reason": "",
            "invalidated_at": "",
            "created_at": existing.created_at if existing else _now(),
            "saved_at": _now(),
        }
        if extra:
            row.update(_jsonable(extra))
        return self.save_record(row)

    # --- video candidates ---

    def save_video_candidate(self, shot_id: str, candidate: Any) -> dict[str, Any]:
        # 统一经稳定契约验证/补齐：path、last_frame_path、seed、recipe_hash、
        # reference_manifest、score、metrics、failure 不能因不同调用方而缺失。
        row = _jsonable(VideoCandidateRecord.model_validate(candidate))
        if not row.get("candidate_id"):
            raise ValueError("video candidate requires candidate_id")
        row["shot_id"] = str(shot_id)
        row.setdefault("selected", False)
        row.setdefault("selection_reason", "")
        row["saved_at"] = _now()
        with self._local:
            rows = self.data.setdefault("shots", {}).setdefault(str(shot_id), {}).setdefault("video_candidates", [])
            for index, existing in enumerate(rows):
                if str(existing.get("candidate_id")) == str(row["candidate_id"]):
                    rows[index] = row
                    break
            else:
                rows.append(row)
            self.save()
        return row

    def video_candidates(
        self,
        shot_id: str,
        *,
        shot_version: int | None = None,
        statuses: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        rows = [
            dict(item)
            for item in self.data.get("shots", {}).get(str(shot_id), {}).get("video_candidates", [])
            if isinstance(item, dict)
        ]
        if shot_version is not None:
            rows = [item for item in rows if int(item.get("shot_version") or 0) == int(shot_version)]
        if statuses is not None:
            allowed = {str(item) for item in statuses}
            rows = [item for item in rows if str(item.get("status")) in allowed]
        return rows

    def failed_video_candidates(self, shot_id: str, *, shot_version: int | None = None) -> list[dict[str, Any]]:
        return self.video_candidates(shot_id, shot_version=shot_version, statuses={"failed"})

    def mark_video_candidate_selected(
        self,
        shot_id: str,
        candidate_id: str,
        *,
        shot_version: int,
        reason: str = "",
    ) -> dict[str, Any]:
        with self._local:
            rows = self.data.setdefault("shots", {}).setdefault(str(shot_id), {}).setdefault("video_candidates", [])
            matched: dict[str, Any] | None = None
            for row in rows:
                if str(row.get("candidate_id")) == str(candidate_id):
                    if int(row.get("shot_version") or 0) != int(shot_version):
                        raise RuntimeError("video candidate version conflict")
                    if row.get("status") != VideoCandidateStatus.SUCCEEDED.value:
                        raise RuntimeError("只能选择成功的视频候选")
                    if row.get("structural_passed") is False:
                        raise RuntimeError("结构检查失败的候选不能被选择")
                    matched = row
                    break
            if matched is None:
                raise RuntimeError("video candidate not found")
            for row in rows:
                selected = row is matched
                row["selected"] = selected
                row["selection_reason"] = str(reason) if selected else ""
            self.save()
        return dict(matched)

    def reusable_shot_artifact(
        self,
        shot_id: str,
        stage: str,
        *,
        shot_version: int | None = None,
        input_fingerprint: str | None = None,
        output_fingerprint: str | None = None,
    ) -> dict[str, Any] | None:
        row = self.shot_artifact(shot_id, stage)
        if not row or not row.get("valid", True) or row.get("status") not in _REUSABLE_STATUSES:
            return None
        if shot_version is not None and int(row.get("shot_version") or 0) != int(shot_version):
            return None
        if input_fingerprint is not None and str(row.get("input_fingerprint") or "") != str(input_fingerprint):
            return None
        if output_fingerprint is not None and str(row.get("output_fingerprint") or "") != str(output_fingerprint):
            return None
        if str(row.get("project_id") or self.project_id) != self.project_id:
            return None
        if str(row.get("run_id") or self.run_id) != self.run_id:
            return None
        return row

    def assert_shot_version(self, shot_id: str, stage: str, shot_version: int) -> dict[str, Any]:
        row = self.shot_artifact(shot_id, stage)
        if not row:
            raise RuntimeError("checkpoint not found")
        actual = int(row.get("shot_version") or 0)
        if actual != int(shot_version):
            raise RuntimeError(f"checkpoint version conflict: expected {shot_version}, got {actual}")
        return row

    def invalidate_shots(
        self, shot_ids: Iterable[str], *, from_stage: str = "image_generation", reason: str = "user_changed"
    ) -> list[str]:
        try:
            start = _STAGE_ORDER.index(str(from_stage))
        except ValueError:
            return []
        stages = list(_STAGE_ORDER[start:])
        affected: list[str] = []
        with self._local:
            for shot_id in shot_ids:
                shot_id = str(shot_id)
                for stage in stages:
                    row = self.data.get("shots", {}).get(shot_id, {}).get(stage)
                    if row is not None and row.get("valid", True):
                        self._mark_invalid(row, reason)
                        affected.append(f"{shot_id}:{stage}")
            self.save()
        return affected

    def _invalidate_stage_from(self, stage: str, *, reason: str) -> list[str]:
        try:
            start = _STAGE_ORDER.index(str(stage))
        except ValueError:
            return []
        affected: list[str] = []
        with self._local:
            for name in _STAGE_ORDER[start:]:
                row = self.data.get("stages", {}).get(name)
                if isinstance(row, dict) and row.get("valid", True):
                    self._mark_invalid(row, reason)
                    affected.append(name)
            self.save()
        return affected

    def _all_records(self) -> list[dict[str, Any]]:
        rows = [row for row in self.data.get("stages", {}).values() if isinstance(row, dict)]
        for shot_rows in self.data.get("shots", {}).values():
            if not isinstance(shot_rows, dict):
                continue
            for row in shot_rows.values():
                if isinstance(row, dict) and row.get("kind") in {"stage", "shot"}:
                    rows.append(row)
        return rows

    def _coerce_record(
        self,
        row: dict[str, Any],
        *,
        kind: str | None = None,
        stage: str | None = None,
        shot_id: str | None = None,
    ) -> CheckpointRecord | None:
        if not isinstance(row, dict) or not row:
            return None
        record_kind = str(kind or row.get("kind") or ("shot" if shot_id or row.get("shot_id") else "stage"))
        record_stage = str(stage or row.get("stage") or "")
        record_shot_id = str(shot_id or row.get("shot_id") or "")
        shot_version = int(row.get("shot_version") or 0)
        defaults = {
            "key": row.get("key")
            or (
                _stage_record_key(record_stage)
                if record_kind == "stage"
                else _shot_record_key(record_stage, record_shot_id, shot_version)
            ),
            "checkpoint_key": row.get("checkpoint_key") or _checkpoint_key(record_stage),
            "kind": record_kind,
            "project_id": row.get("project_id") or self.project_id,
            "run_id": row.get("run_id") or self.run_id,
            "stage": record_stage,
            "shot_id": record_shot_id,
            "shot_version": shot_version,
            "input_fingerprint": row.get("input_fingerprint") or "",
            "valid": bool(row.get("valid", row.get("status") != "invalidated")),
            "created_at": row.get("created_at") or row.get("saved_at") or _now(),
            "saved_at": row.get("saved_at") or _now(),
        }
        return CheckpointRecord.model_validate({**row, **defaults})

    @staticmethod
    def _mark_invalid(row: dict[str, Any], reason: str) -> None:
        row["status"] = "invalidated"
        row["valid"] = False
        row["invalidated_reason"] = str(reason)
        row["invalidated_at"] = _now()

    # --- decisions/events ---

    def add_decision(self, trace: Any) -> dict[str, Any]:
        row = _jsonable(trace)
        with self._local:
            rows = self.data.setdefault("decisions", [])
            rows.append(row)
            if len(rows) > 500:
                del rows[:-500]
            self.save()
        return row

    def decisions(self) -> list[dict[str, Any]]:
        return list(self.data.get("decisions", []))

    def add_event(self, event: str, **fields: Any) -> None:
        with self._local:
            rows = self.data.setdefault("events", [])
            rows.append({"event": str(event), "at": _now(), **_jsonable(fields)})
            if len(rows) > 1000:
                del rows[:-1000]
            self.save()

    # --- version/change detection ---

    def version_snapshot(self) -> dict[str, int]:
        """读取项目下镜头版本和 AV 配置版本，供断点续跑/用户中途修改检测。"""

        try:
            from db import SessionLocal
            from models import Project, Shot

            db = SessionLocal()
            try:
                shot_versions = {
                    str(shot_id): int(version or 1)
                    for shot_id, version in db.query(Shot.id, Shot.version)
                    .filter(Shot.project_id == self.project_id)
                    .all()
                }
                av_version = db.query(Project.av_config_version).filter(Project.id == self.project_id).scalar()
                snapshot = {"__av_config_version__": int(av_version or 0)}
                snapshot.update(shot_versions)
                return snapshot
            finally:
                db.close()
        except Exception:
            return dict(self.data.get("version_snapshot", {}))

    def detect_changes(self, current: dict[str, int] | None = None) -> dict[str, Any]:
        current_snapshot = dict(current if current is not None else self.version_snapshot())
        previous = dict(self.data.get("version_snapshot", {}))
        changed_shots = sorted(
            key
            for key in set(previous) | set(current_snapshot)
            if key != "__av_config_version__" and previous.get(key) != current_snapshot.get(key)
        )
        av_changed = previous.get("__av_config_version__") != current_snapshot.get("__av_config_version__")
        if changed_shots:
            self.invalidate_shots(changed_shots, from_stage="image_generation", reason="user_changed")
            self._invalidate_stage_from("image_generation", reason="shot_version_changed")
        if av_changed and previous:
            self.invalidate_from("edit_composition", reason="av_config_changed")
        with self._local:
            self.data["version_snapshot"] = current_snapshot
            self.save()
        return {"changed_shot_ids": changed_shots, "av_config_changed": bool(av_changed)}


def _now() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


def _safe_name(value: str) -> str:
    value = str(value or "unknown")
    return "".join(char if char.isalnum() or char in "-_.:" else "_" for char in value)[:120]


def _duration_ms(started_at: Any, finished_at: Any) -> int:
    """用检查点行内的 UTC ISO 时间估算实际耗时；无法解析返回 0。"""

    from datetime import datetime

    def parse(value: Any):
        try:
            return datetime.fromisoformat(str(value or ""))
        except ValueError:
            return None

    start, end = parse(started_at), parse(finished_at)
    if start is None or end is None:
        return 0
    return max(0, int((end - start).total_seconds() * 1000))


def _merge_video_candidates(per_stage: dict[str, Any], stage_entries: dict[str, Any]) -> list[dict[str, Any]]:
    """合并两处候选历史：顶层 ``video_candidates``（save_video_candidate）与
    各阶段检查点行 extra 内嵌的候选列表；按 candidate_id 去重，顶层优先。"""

    merged: dict[str, dict[str, Any]] = {}
    for row in per_stage.get("video_candidates") or []:
        if isinstance(row, dict) and row.get("candidate_id"):
            merged[str(row["candidate_id"])] = row
    for entry in stage_entries.values():
        for row in (entry.get("video_candidates") or []) if isinstance(entry, dict) else []:
            if isinstance(row, dict) and row.get("candidate_id") and str(row["candidate_id"]) not in merged:
                merged[str(row["candidate_id"])] = row
    return [merged[key] for key in sorted(merged)]


def _trim_candidate(row: Any) -> dict[str, Any]:
    item = row if isinstance(row, dict) else {}
    return {
        "strategy": str(item.get("strategy") or ""),
        "provider": str(item.get("provider") or ""),
        "target_stage": str(item.get("target_stage") or ""),
        "shot_ids": [str(value) for value in item.get("shot_ids") or []],
        "estimated_cost_micro": item.get("estimated_cost_micro"),
        "estimated_seconds": item.get("estimated_seconds"),
        "quality_gain": item.get("quality_gain"),
        "provider_capability_ok": bool(item.get("provider_capability_ok", True)),
        "budget_fit": bool(item.get("budget_fit", True)),
        "score": item.get("score"),
        "rationale": str(item.get("rationale") or ""),
        "prompt_patches": [dict(patch) for patch in item.get("prompt_patches") or []],
    }


def _trim_video_candidate(row: Any) -> dict[str, Any]:
    item = row if isinstance(row, dict) else {}
    return {
        "candidate_id": str(item.get("candidate_id") or ""),
        "status": str(item.get("status") or ""),
        "provider": str(item.get("provider") or ""),
        "model": str(item.get("model") or ""),
        "score": item.get("score"),
        "seed": item.get("seed"),
        "path": str(item.get("path") or ""),
        "generation_duration_ms": int(item.get("generation_duration_ms") or 0),
        "structural_passed": item.get("structural_passed"),
        "selected": bool(item.get("selected")),
        "selection_reason": str(item.get("selection_reason") or ""),
        "failure_kind": str((item.get("failure") or {}).get("kind") or "")
        if isinstance(item.get("failure"), dict)
        else "",
        # 实际发送给 Provider 的参考图清单（角色/场景/上一镜尾帧）。
        "reference_manifest": [
            dict(entry) for entry in item.get("reference_manifest") or [] if isinstance(entry, dict)
        ],
    }


def _trim_critique(row: Any) -> dict[str, Any]:
    item = row if isinstance(row, dict) else {}
    return {
        "passed": item.get("passed"),
        "score": item.get("score"),
        "issues": [dict(issue) for issue in item.get("issues") or [] if isinstance(issue, dict)],
        "proposed_changes": [str(value) for value in item.get("proposed_changes") or []],
        "failure_kind": item.get("failure_kind"),
        "recommended_strategy": item.get("recommended_strategy"),
        "affected_shot_ids": [str(value) for value in item.get("affected_shot_ids") or []],
    }


def summarize_trace(snapshot: dict[str, Any]) -> dict[str, Any]:
    """把检查点快照整理成面向展示/审计的可解释追踪汇总。

    只读取快照中已有的事实（阶段行、逐镜头产物、DecisionTrace、事件），不猜测
    未记录的结论；数据库侧的镜头版本和参考图清单由 API 路由另行合并。
    """

    data = snapshot if isinstance(snapshot, dict) else {}
    events = [dict(item) for item in data.get("events") or [] if isinstance(item, dict)]
    stage_rows = {str(key): dict(value) for key, value in (data.get("stages") or {}).items() if isinstance(value, dict)}

    def latest_event_stage() -> str:
        for item in reversed(events):
            stage = str(item.get("stage") or "")
            if stage:
                return stage
        return ""

    def latest_saved_stage() -> str:
        def saved_at(row: dict[str, Any]) -> str:
            return str(row.get("saved_at") or "")

        return max(stage_rows, key=saved_at) if stage_rows else ""

    current_stage = latest_event_stage() or latest_saved_stage()

    stages: list[dict[str, Any]] = []
    for stage, row in sorted(stage_rows.items()):
        stages.append(
            {
                "stage": stage,
                "status": str(row.get("status") or "pending"),
                "valid": bool(row.get("valid", True)),
                "invalidated_reason": str(row.get("invalidated_reason") or ""),
                "quality": _trim_critique(row.get("critique")),
                "failure_kind": str((row.get("failure") or {}).get("kind") or "")
                if isinstance(row.get("failure"), dict)
                else "",
                "failure_message": str((row.get("failure") or {}).get("message") or "")
                if isinstance(row.get("failure"), dict)
                else "",
                "actual_duration_ms": _duration_ms(row.get("created_at"), row.get("saved_at")),
                "created_at": str(row.get("created_at") or ""),
                "saved_at": str(row.get("saved_at") or ""),
            }
        )

    shots: list[dict[str, Any]] = []
    total_cost_micro = 0
    for shot_id, per_stage in sorted((data.get("shots") or {}).items()):
        if not isinstance(per_stage, dict):
            continue
        stage_entries: dict[str, Any] = {}
        shot_cost = 0
        shot_duration = 0
        for stage, row in per_stage.items():
            if stage == "video_candidates" or not isinstance(row, dict):
                continue
            cost = row.get("cost_micro")
            cost_value = int(cost) if isinstance(cost, int) else None
            duration = int(row.get("duration_ms") or 0)
            if cost_value is not None:
                shot_cost += cost_value
            shot_duration += duration
            stage_entries[str(stage)] = {
                "status": str(row.get("status") or ""),
                "provider": str(row.get("provider") or ""),
                "model": str(row.get("model") or ""),
                "score": row.get("score"),
                "cost_micro": cost_value,
                "duration_ms": duration,
                "path": str(row.get("path") or ""),
                "failure_kind": str((row.get("failure") or {}).get("kind") or "")
                if isinstance(row.get("failure"), dict)
                else "",
                "failure_message": str((row.get("failure") or {}).get("message") or "")
                if isinstance(row.get("failure"), dict)
                else "",
                "shot_version": int(row.get("shot_version") or 0),
                "selected_video_candidate_id": str(row.get("selected_video_candidate_id") or ""),
                "candidate_selection": row.get("candidate_selection") or None,
                "video_candidates": [item for item in row.get("video_candidates") or [] if isinstance(item, dict)],
            }
        video_candidates = _merge_video_candidates(per_stage, stage_entries)
        selected_video_candidate_id = ""
        candidate_selection = None
        for stage_key in ("video_generation", "video_review", "image_generation"):
            entry = stage_entries.get(stage_key) or {}
            if not selected_video_candidate_id and entry.get("selected_video_candidate_id"):
                selected_video_candidate_id = str(entry["selected_video_candidate_id"])
            if candidate_selection is None and entry.get("candidate_selection"):
                candidate_selection = entry.get("candidate_selection")
        shots.append(
            {
                "shot_id": str(shot_id),
                "stages": stage_entries,
                "cost_micro": shot_cost,
                "duration_ms": shot_duration,
                "video_candidates": [_trim_video_candidate(row) for row in video_candidates],
                "selected_video_candidate_id": selected_video_candidate_id,
                "candidate_selection": candidate_selection,
            }
        )
        total_cost_micro += shot_cost

    decisions: list[dict[str, Any]] = []
    prompt_changes: list[dict[str, Any]] = []
    for row in data.get("decisions") or []:
        if not isinstance(row, dict):
            continue
        selected = row.get("selected") if isinstance(row.get("selected"), dict) else {}
        trimmed = {
            "trace_id": str(row.get("trace_id") or ""),
            "stage": str(row.get("stage") or ""),
            "mode": str(row.get("mode") or ""),
            "shot_id": str(row.get("shot_id") or ""),
            "failure_kind": str((row.get("failure") or {}).get("kind") or "")
            if isinstance(row.get("failure"), dict)
            else "",
            "failure_message": str((row.get("failure") or {}).get("message") or "")
            if isinstance(row.get("failure"), dict)
            else "",
            "quality_score": row.get("quality_score"),
            "retries_remaining": row.get("retries_remaining"),
            "reason": str(row.get("reason") or ""),
            "selected": _trim_candidate(selected) if selected else None,
            "candidates": [_trim_candidate(item) for item in row.get("candidates") or [] if isinstance(item, dict)],
            "considered_rejected": [
                dict(item) for item in row.get("considered_rejected") or [] if isinstance(item, dict)
            ],
            "budget": {
                "level": (row.get("budget_snapshot") or {}).get("level")
                if isinstance(row.get("budget_snapshot"), dict)
                else None,
                "remaining_cost_micro": (row.get("budget_snapshot") or {}).get("remaining_cost_micro")
                if isinstance(row.get("budget_snapshot"), dict)
                else None,
                "remaining_seconds": (row.get("budget_snapshot") or {}).get("remaining_seconds")
                if isinstance(row.get("budget_snapshot"), dict)
                else None,
            },
            "provider_profiles": [
                {
                    "capability": str(item.get("capability") or ""),
                    "provider": str(item.get("provider") or ""),
                    "model": str(item.get("model") or ""),
                    "available": bool(item.get("available", False)),
                    "supports_reference_images": bool(item.get("supports_reference_images")),
                }
                for item in (row.get("provider_profiles") or [])
                if isinstance(item, dict)
            ],
            "selected_video_candidate_id": str(row.get("selected_video_candidate_id") or ""),
            "candidate_selection": row.get("candidate_selection") or None,
            "created_at": str(row.get("created_at") or ""),
        }
        decisions.append(trimmed)
        if selected:
            selected_shot_id = str(row.get("shot_id") or "")
            for patch in selected.get("prompt_patches") or []:
                if isinstance(patch, dict):
                    prompt_changes.append(
                        {
                            "trace_id": str(row.get("trace_id") or ""),
                            "stage": str(row.get("stage") or ""),
                            "shot_id": str(patch.get("shot_id") or selected_shot_id),
                            **{key: patch.get(key) for key in ("field", "op", "value", "target_stage", "reason")},
                        }
                    )

    recoveries = [item for item in events if str(item.get("event") or "").startswith("recovery")]
    degradations: list[dict[str, Any]] = [
        {
            "at": str(item.get("at") or ""),
            "stage": str(item.get("stage") or ""),
            "reason": str(item.get("reason") or item.get("strategy") or ""),
            "shot_ids": [str(value) for value in item.get("shot_ids") or []],
        }
        for item in events
        if str(item.get("event") or "") in {"degraded_publish", "auto_abort", "human_gate", "recovery_apply_failed"}
        or str(item.get("strategy") or "") in {"degraded_publish", "lower_resolution"}
    ]
    if str(data.get("status") or "") in {"degraded", "failed"} and str(data.get("status_reason") or ""):
        degradations.insert(
            0,
            {
                "at": str(data.get("updated_at") or ""),
                "stage": current_stage,
                "reason": f"运行{'降级' if data.get('status') == 'degraded' else '终止'}: {data.get('status_reason')}",
                "shot_ids": [],
            },
        )

    checkpoint_records = sum(
        1
        for per_stage in (data.get("shots") or {}).values()
        if isinstance(per_stage, dict)
        for key, row in per_stage.items()
        if key != "video_candidates" and isinstance(row, dict) and row.get("kind") in {"stage", "shot"}
    ) + sum(1 for row in stage_rows.values())

    return {
        "run": {
            "project_id": str(data.get("project_id") or ""),
            "run_id": str(data.get("run_id") or ""),
            "status": str(data.get("status") or "pending"),
            "status_reason": str(data.get("status_reason") or ""),
            "current_stage": current_stage,
            "input_fingerprint": str(data.get("input_fingerprint") or ""),
            "updated_at": str(data.get("updated_at") or ""),
        },
        "counters": {
            "checkpoint_records": checkpoint_records,
            "stage_checkpoints": len(stage_rows),
            "decisions": len(decisions),
            "recoveries": len(recoveries),
            "invalidated_checkpoints": sum(1 for row in stage_rows.values() if not row.get("valid", True))
            + sum(
                1
                for per_stage in (data.get("shots") or {}).values()
                if isinstance(per_stage, dict)
                for key, row in per_stage.items()
                if key != "video_candidates" and isinstance(row, dict) and not row.get("valid", True)
            ),
        },
        "totals": {
            "cost_micro": total_cost_micro,
            "video_candidates": sum(len(shot.get("video_candidates") or []) for shot in shots),
            "selected_video_candidates": sum(1 for shot in shots if shot.get("selected_video_candidate_id")),
        },
        "stages": stages,
        "shots": shots,
        "decisions": decisions,
        "prompt_changes": prompt_changes,
        "degradations": degradations,
        "events": events,
    }


__all__ = ["CheckpointRecord", "CheckpointStore", "fingerprint", "summarize_trace"]
