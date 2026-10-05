import json
import os
import re
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from config import settings
from db import SessionLocal
from models import AudioTrack, Character, SceneAsset, Shot, ShotVersion, ShotVideoCandidate
from services.security import atomic_write_bytes, safe_path, validate_identifier
from services.shot_version_service import iter_snapshot_media_paths


class StorageQuotaExceeded(ValueError):
    """Raised when a project would exceed its configured local media quota."""


class StorageService:
    """文件存储服务"""

    _SHOT_PATH_FIELDS = (
        "image_path",
        "video_path",
        "audio_path",
        "storyboard_path",
        "continuity_reference_path",
        "pose_reference_path",
        "depth_reference_path",
        "last_frame_path",
    )
    _SHOT_JSON_REFERENCE_FIELDS = ("consistency_context", "continuity_profile")
    _CHARACTER_JSON_REFERENCE_FIELDS = ("reference_images",)
    _SCENE_PATH_FIELDS = ("baseline_image_path",)
    _SCENE_JSON_REFERENCE_FIELDS = ("reference_images", "consistency_profile")

    def __init__(self):
        self.base_dir = settings.OUTPUT_DIR

    def get_project_dir(self, project_id: str) -> Path:
        return safe_path(self.base_dir / "projects", project_id, create_parent=True)

    def get_shots_dir(self, project_id: str) -> Path:
        shots_dir = self.get_project_dir(project_id) / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)
        return shots_dir

    def get_audio_dir(self, project_id: str) -> Path:
        audio_dir = self.get_project_dir(project_id) / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        return audio_dir

    def get_output_dir(self, project_id: str) -> Path:
        output_dir = self.get_project_dir(project_id) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def project_usage_bytes(self, project_id: str) -> int:
        """Return regular-file usage below one project without following links."""

        project_dir = self.get_project_dir(project_id)
        total = 0
        for path in project_dir.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
        return total

    def cleanup_project(self, project_id: str) -> int:
        """Remove abandoned renderer work dirs and old numbered media versions.

        Current image paths include ``_vN``; retaining the newest configured
        versions preserves a short rollback window while keeping repeated
        regeneration from filling the project directory indefinitely.
        """

        project_dir = self.get_project_dir(project_id)
        removed = 0
        now = time.time()
        ttl = max(0, int(settings.PROJECT_TEMP_FILE_TTL_SECONDS))
        for path in list(project_dir.rglob("*")):
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue
            is_work_dir = path.is_dir() and path.name.startswith(".render-")
            is_temp_file = path.is_file() and (
                path.name.endswith(".tmp")
                or path.name.endswith(".upload")
                or path.name.endswith(".download")
                or path.name.endswith(".candidate")
            )
            if ttl and age < ttl:
                continue
            if is_work_dir:
                removed += self._directory_size(path)
                shutil.rmtree(path, ignore_errors=True)
            elif is_temp_file:
                removed += path.stat().st_size
                path.unlink(missing_ok=True)

        versions: dict[tuple[Path, str, str], list[tuple[int, Path]]] = {}
        for path in project_dir.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            match = re.fullmatch(r"(.+)_v(\d+)(\.[^.]+)", path.name)
            if match:
                versions.setdefault((path.parent, match.group(1), match.group(3)), []).append(
                    (int(match.group(2)), path)
                )
        retain = max(1, int(settings.PROJECT_VERSION_RETENTION_COUNT))
        protected_paths = self._protected_project_paths(project_dir)
        if protected_paths is None:
            # A missing/unavailable database must never turn a cleanup pass into
            # destructive guessing. Temporary artifacts above remain safe to
            # reclaim, but versioned media is retained until references can be
            # read successfully.
            return removed
        for items in versions.values():
            for _, path in sorted(items, reverse=True, key=lambda item: item[0])[retain:]:
                try:
                    if path.resolve() in protected_paths:
                        continue
                    removed += path.stat().st_size
                    path.unlink()
                except OSError:
                    continue
        return removed

    def ensure_project_capacity(
        self, project_id: str, incoming_bytes: int = 0, *, replacing: str | Path | None = None
    ) -> int:
        """Reclaim safe garbage, then reserve capacity for an imminent write."""

        if incoming_bytes < 0:
            raise ValueError("写入大小不能为负数")
        self.cleanup_project(project_id)
        usage = self.project_usage_bytes(project_id)
        replaced = 0
        if replacing is not None:
            try:
                target = Path(replacing).resolve()
                root = self.get_project_dir(project_id).resolve()
                if target.is_relative_to(root) and target.is_file():
                    replaced = target.stat().st_size
            except OSError:
                pass
        available = max(0, int(settings.PROJECT_STORAGE_QUOTA_BYTES) - usage + replaced)
        if incoming_bytes > available:
            raise StorageQuotaExceeded("项目媒体存储空间不足")
        return available

    @staticmethod
    def _directory_size(directory: Path) -> int:
        total = 0
        for child in directory.rglob("*"):
            try:
                if child.is_file() and not child.is_symlink():
                    total += child.stat().st_size
            except OSError:
                continue
        return total

    def _protected_project_paths(self, project_dir: Path) -> set[Path] | None:
        """Return files below ``project_dir`` referenced by current DB rows."""

        session = SessionLocal()
        try:
            protected: set[Path] = set()
            shot_path_columns = [getattr(Shot, field) for field in self._SHOT_PATH_FIELDS]
            for row in session.query(*shot_path_columns).all():
                for value in row:
                    self._add_protected_path(protected, project_dir, value)

            shot_json_columns = [getattr(Shot, field) for field in self._SHOT_JSON_REFERENCE_FIELDS]
            for row in session.query(*shot_json_columns).all():
                for value in row:
                    self._add_json_references(protected, project_dir, value)

            character_columns = [getattr(Character, field) for field in self._CHARACTER_JSON_REFERENCE_FIELDS]
            for row in session.query(*character_columns).all():
                for value in row:
                    self._add_json_references(protected, project_dir, value)

            scene_path_columns = [getattr(SceneAsset, field) for field in self._SCENE_PATH_FIELDS]
            for row in session.query(*scene_path_columns).all():
                for value in row:
                    self._add_protected_path(protected, project_dir, value)

            scene_json_columns = [getattr(SceneAsset, field) for field in self._SCENE_JSON_REFERENCE_FIELDS]
            for row in session.query(*scene_json_columns).all():
                for value in row:
                    self._add_json_references(protected, project_dir, value)

            # 混音工作台上传的素材受保护：轨道行还在引用就不能被清理误删。
            for (track_source,) in session.query(AudioTrack.source_path).all():
                self._add_protected_path(protected, project_dir, track_source)

            # 未选用的视频候选也必须保留：失败候选用于重试审计，成功候选用于后续对比。
            for video_path, tail_frame_path, raw_manifest in session.query(
                ShotVideoCandidate.video_path,
                ShotVideoCandidate.tail_frame_path,
                ShotVideoCandidate.reference_manifest,
            ).all():
                self._add_protected_path(protected, project_dir, video_path)
                self._add_protected_path(protected, project_dir, tail_frame_path)
                self._add_json_references(protected, project_dir, raw_manifest)

            # 版本历史快照引用的媒体同样受保护：它们是「恢复历史版本」的唯一依据。
            for (raw_snapshot,) in session.query(ShotVersion.snapshot).all():
                for path in iter_snapshot_media_paths(raw_snapshot):
                    self._add_protected_path(protected, project_dir, path)
            return protected
        except Exception:
            return None
        finally:
            session.close()

    def _add_json_references(self, protected: set[Path], project_dir: Path, raw: Any) -> None:
        if raw in (None, ""):
            return
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            self._add_protected_path(protected, project_dir, raw)
            return
        for candidate in self._iter_strings(value):
            self._add_protected_path(protected, project_dir, candidate)

    @classmethod
    def _iter_strings(cls, value: Any) -> Iterator[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from cls._iter_strings(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from cls._iter_strings(item)

    def _add_protected_path(self, protected: set[Path], project_dir: Path, raw: Any) -> None:
        value = str(raw or "").strip()
        if not value:
            return
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https", "data"}:
            return
        if parsed.scheme == "file":
            value = unquote(parsed.path)

        unified = value.replace("\\", "/")
        # URL-style public paths (get_relative_url emits "/output/...") must
        # resolve identically on every OS: a drive-less "/output/..." is not
        # is_absolute() on Windows and would otherwise miss the mapping below.
        if unified.startswith("/output/"):
            candidates = [self.base_dir / unified[len("/output/") :]]
        else:
            normalized = value.replace("\\", os.sep)
            path = Path(normalized)
            if path.is_absolute():
                output_prefix = f"{os.sep}output{os.sep}"
                if normalized.startswith(output_prefix):
                    candidates = [self.base_dir / normalized[len(output_prefix) :]]
                else:
                    candidates = [path]
            else:
                parts = path.parts
                if parts and parts[0] == "output":
                    candidates = [self.base_dir.joinpath(*parts[1:])]
                elif parts and parts[0] == "projects":
                    candidates = [self.base_dir / path]
                else:
                    candidates = [project_dir / path, Path.cwd() / path]

        root = project_dir.resolve()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                resolved.relative_to(root)
            except (OSError, RuntimeError, ValueError):
                continue
            protected.add(resolved)

    def save_file(self, project_id: str, sub_dir: str, filename: str, content: bytes) -> str:
        try:
            safe_sub_dir = validate_identifier(sub_dir, "目录")
            safe_filename = validate_identifier(Path(filename).name, "文件名")
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        dir_path = safe_path(self.get_project_dir(project_id), safe_sub_dir, create_parent=True)
        file_path = dir_path / safe_filename
        self.ensure_project_capacity(project_id, len(content), replacing=file_path)
        atomic_write_bytes(file_path, content)
        return str(file_path)

    def delete_file(self, file_path: str) -> bool:
        try:
            Path(file_path).unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def get_relative_url(self, file_path: str) -> str:
        """将文件路径转换为可访问的 URL 路径"""
        try:
            rel = Path(file_path).relative_to(self.base_dir)
            return f"/output/{rel.as_posix()}"
        except ValueError:
            return file_path
