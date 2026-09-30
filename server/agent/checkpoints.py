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
from pathlib import Path
from typing import Any, Iterable

from config import settings
from services.atomic_json import atomic_write_json, read_json_file

_SCHEMA_VERSION = 1
_STORE_LOCK = threading.RLock()
_STORES: dict[tuple[str, str], "CheckpointStore"] = {}
_STAGE_ORDER = (
    "director_planning",
    "storyboard_design",
    "asset_preparation",
    "image_generation",
    "quality_review",
    "video_generation",
    "audio_production",
    "edit_composition",
    "final_review",
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def fingerprint(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


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
    def get(cls, project_id: str, run_id: str = "auto", *, root: Path | None = None) -> "CheckpointStore":
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
            self.data["input_fingerprint"] = str(value)
            if metadata is not None:
                self.data["input_metadata"] = _jsonable(metadata)
            self.save()

    # --- stages ---

    def stage(self, stage: str) -> dict[str, Any]:
        return dict(self.data.get("stages", {}).get(str(stage), {}))

    def stage_is_reusable(self, stage: str, input_fingerprint: str, *, require_success: bool = True) -> bool:
        row = self.stage(stage)
        if require_success and row.get("status") not in {"succeeded", "degraded", "skipped"}:
            return False
        return bool(row.get("input_fingerprint")) and row.get("input_fingerprint") == str(input_fingerprint)

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
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "stage": str(stage),
            "status": str(status),
            "input_fingerprint": str(input_fingerprint),
            "output_fingerprint": str(output_fingerprint or ""),
            "payload": _jsonable(payload or {}),
            "quality": _jsonable(quality or []),
            "failure": _jsonable(failure),
            "critique": _jsonable(critique),
            "shot_artifacts": _jsonable(shot_artifacts or []),
            "checkpoint_version": int(checkpoint_version),
            "saved_at": _now(),
        }
        if extra:
            row.update(_jsonable(extra))
        with self._local:
            self.data.setdefault("stages", {})[str(stage)] = row
            self.save()
        return row

    def invalidate_stage(self, stage: str, *, reason: str = "recompute") -> None:
        with self._local:
            row = self.data.setdefault("stages", {}).get(str(stage))
            if row is not None:
                row["status"] = "invalidated"
                row["invalidated_reason"] = str(reason)
                row["invalidated_at"] = _now()
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
                if row is not None:
                    row["status"] = "invalidated"
                    row["invalidated_reason"] = str(reason)
                    row["invalidated_at"] = _now()
            self.save()
        return affected

    # --- shot artifacts ---

    def shot_artifact(self, shot_id: str, stage: str) -> dict[str, Any]:
        return dict(self.data.get("shots", {}).get(str(shot_id), {}).get(str(stage), {}))

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
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
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
            "output_fingerprint": str(output_fingerprint or ""),
            "saved_at": _now(),
        }
        if extra:
            row.update(_jsonable(extra))
        with self._local:
            self.data.setdefault("shots", {}).setdefault(str(shot_id), {})[str(stage)] = row
            self.save()
        return row

    def reusable_shot_artifact(self, shot_id: str, stage: str, *, shot_version: int | None = None) -> dict[str, Any] | None:
        row = self.shot_artifact(shot_id, stage)
        if not row or row.get("status") not in {"succeeded", "degraded"}:
            return None
        if shot_version is not None and int(row.get("shot_version") or 0) != int(shot_version):
            return None
        return row

    def invalidate_shots(self, shot_ids: Iterable[str], *, from_stage: str = "image_generation", reason: str = "user_changed") -> list[str]:
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
                    if row is not None:
                        row["status"] = "invalidated"
                        row["invalidated_reason"] = str(reason)
                        row["invalidated_at"] = _now()
                        affected.append(f"{shot_id}:{stage}")
            self.save()
        return affected

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
                    for shot_id, version in db.query(Shot.id, Shot.version).filter(Shot.project_id == self.project_id).all()
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
            key for key in set(previous) | set(current_snapshot)
            if key != "__av_config_version__" and previous.get(key) != current_snapshot.get(key)
        )
        av_changed = previous.get("__av_config_version__") != current_snapshot.get("__av_config_version__")
        if changed_shots:
            self.invalidate_shots(changed_shots, from_stage="image_generation", reason="user_changed")
        if av_changed and previous:
            self.invalidate_from("edit_composition", reason="av_config_changed")
        with self._local:
            self.data["version_snapshot"] = current_snapshot
            self.save()
        return {"changed_shot_ids": changed_shots, "av_config_changed": bool(av_changed)}


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    value = str(value or "unknown")
    return "".join(char if char.isalnum() or char in "-_.:" else "_" for char in value)[:120]


__all__ = ["CheckpointStore", "fingerprint"]
