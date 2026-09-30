"""Regression coverage for the P1 reliability and isolation fixes.

Run from the repository root with::

    python3 -m unittest discover -s server/tests -p 'test_*.py'

The test module points the server settings at a temporary SQLite database and
output directory before importing application modules.  This keeps the suite
isolated from a developer's local runtime data.
"""

from __future__ import annotations

import asyncio
import gc
import json
import sys
import types
import unittest
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT as _TEST_ROOT  # noqa: E402

from fastapi import HTTPException  # noqa: E402
from sqlalchemy import event, text  # noqa: E402

from api.routes import asset as asset_route  # noqa: E402
from api.routes import character as character_route  # noqa: E402
from api.routes import project as project_route  # noqa: E402
from api.routes import render as render_route  # noqa: E402
from api.routes import script as script_route  # noqa: E402
from services import dialogue_audio  # noqa: E402
from api.routes import shot as shot_route  # noqa: E402
from api.routes.shot import _can_reuse_existing_video  # noqa: E402
from config import settings  # noqa: E402
from db import SessionLocal, engine, init_db  # noqa: E402
from models import BackgroundJob, Character, Project, SceneAsset, Shot  # noqa: E402
from rag.rag_service import RAGService  # noqa: E402
from services import task_registry  # noqa: E402
from services.task_registry import claim, finish, recover_interrupted  # noqa: E402
from main import app  # noqa: E402
from services.security import validate_script_upload, validate_video_upload  # noqa: E402
from services.storage_service import StorageQuotaExceeded, StorageService  # noqa: E402


class DatabaseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    def setUp(self) -> None:
        self.db = SessionLocal()
        # Explicit deletes keep each test independent even when SQLite foreign
        # key enforcement is enabled by the application.
        self.db.query(BackgroundJob).delete()
        self.db.query(Shot).delete()
        self.db.query(Character).delete()
        self.db.query(SceneAsset).delete()
        self.db.query(Project).delete()
        self.db.commit()

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()


class RagRegressionTests(unittest.TestCase):
    def test_scene_markers_do_not_create_none_parts(self) -> None:
        service = RAGService()
        chunks = service._chunk_script(
            "[场景 1]\n小明说你好。\n\n[场景 2]\n小红说再见。"
        )

        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(isinstance(chunk, dict) for chunk in chunks))
        self.assertTrue(all(isinstance(chunk.get("text"), str) for chunk in chunks))
        self.assertTrue(all(chunk.get("text", "").strip() for chunk in chunks))


class ApplicationSmokeTests(unittest.TestCase):
    def test_main_app_registers_health_and_output_routes(self) -> None:
        paths = {getattr(route, "path", "") for route in app.routes}
        self.assertIn("/health", paths)
        self.assertIn("/livez", paths)
        self.assertIn("/readyz", paths)
        self.assertIn("/output", paths)

    def test_liveness_identity_is_stable(self) -> None:
        result = asyncio.run(__import__("main").health())
        self.assertEqual(result["service"], "comic-agent")
        self.assertEqual(result["version"], app.version)


class StorageReliabilityTests(unittest.TestCase):
    def test_cleanup_retains_recent_versions_and_removes_abandoned_work(self) -> None:
        service = StorageService()
        project_dir = service.get_project_dir("gc-project")
        shots_dir = project_dir / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)
        for version in range(1, 5):
            (shots_dir / f"shot_v{version}.png").write_bytes(b"image")
        work_dir = project_dir / "output" / ".render-abandoned"
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "clip_0001.mp4").write_bytes(b"clip")
        (shots_dir / ".partial.upload").write_bytes(b"partial")

        old_ttl = settings.PROJECT_TEMP_FILE_TTL_SECONDS
        old_retention = settings.PROJECT_VERSION_RETENTION_COUNT
        settings.PROJECT_TEMP_FILE_TTL_SECONDS = 0
        settings.PROJECT_VERSION_RETENTION_COUNT = 2
        try:
            service.cleanup_project("gc-project")
        finally:
            settings.PROJECT_TEMP_FILE_TTL_SECONDS = old_ttl
            settings.PROJECT_VERSION_RETENTION_COUNT = old_retention

        self.assertFalse((shots_dir / "shot_v1.png").exists())
        self.assertFalse((shots_dir / "shot_v2.png").exists())
        self.assertTrue((shots_dir / "shot_v3.png").exists())
        self.assertTrue((shots_dir / "shot_v4.png").exists())
        self.assertFalse(work_dir.exists())
        self.assertFalse((shots_dir / ".partial.upload").exists())

    def test_project_quota_accounts_for_replaced_file(self) -> None:
        service = StorageService()
        project_dir = service.get_project_dir("quota-project")
        existing = project_dir / "output" / "final.mp4"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"12345678")
        old_quota = settings.PROJECT_STORAGE_QUOTA_BYTES
        settings.PROJECT_STORAGE_QUOTA_BYTES = 10
        try:
            self.assertEqual(service.ensure_project_capacity("quota-project", 10, replacing=existing), 10)
            with self.assertRaises(StorageQuotaExceeded):
                service.ensure_project_capacity("quota-project", 11, replacing=existing)
        finally:
            settings.PROJECT_STORAGE_QUOTA_BYTES = old_quota


class UploadValidationTests(unittest.TestCase):
    def test_upload_signatures_and_octet_stream_fallback(self) -> None:
        text_path = _TEST_ROOT / "plain-script.txt"
        text_path.write_text("hello", encoding="utf-8")
        validate_script_upload(text_path, ".txt", "application/octet-stream")

        bad_docx = _TEST_ROOT / "not-a-docx.docx"
        bad_docx.write_bytes(b"not a zip")
        with self.assertRaises(ValueError):
            validate_script_upload(bad_docx, ".docx", "application/octet-stream")

        video_path = _TEST_ROOT / "movie.mp4"
        video_path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"0" * 1200)
        validate_video_upload(video_path, "application/octet-stream")
        with self.assertRaises(ValueError):
            validate_video_upload(video_path, "text/plain")


class DatabaseReliabilityTests(DatabaseTestCase):
    def test_sqlite_pragmas_and_project_indexes_are_active(self) -> None:
        self.assertEqual(self.db.execute(text("PRAGMA foreign_keys")).scalar(), 1)
        self.assertEqual(str(self.db.execute(text("PRAGMA journal_mode")).scalar()).lower(), "wal")
        indexes = {row[1] for row in self.db.execute(text("PRAGMA index_list(projects)"))}
        self.assertIn("ix_projects_parent_type", indexes)
        self.assertIn("ix_projects_updated_at", indexes)


class AssetIsolationRegressionTests(DatabaseTestCase):
    def test_shot_cannot_bind_assets_from_another_project(self) -> None:
        project_a = Project(id="project-a", title="A")
        project_b = Project(id="project-b", title="B")
        shot = Shot(id="shot-a", project_id=project_a.id, sequence=1)
        foreign_scene = SceneAsset(id="scene-b", project_id=project_b.id, name="Foreign scene")
        foreign_character = Character(id="character-b", project_id=project_b.id, name="Foreign character")
        self.db.add_all([project_a, project_b, shot, foreign_scene, foreign_character])
        self.db.commit()

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(
                asset_route.update_shot_assets(
                    shot.id,
                    asset_route.ShotAssetUpdate(
                        project_id=project_a.id,
                        scene_asset_id=foreign_scene.id,
                        character_asset_ids=[foreign_character.id],
                    ),
                    self.db,
                )
            )

        self.assertEqual(raised.exception.status_code, 400)
        self.db.refresh(shot)
        self.assertEqual(shot.scene_asset_id, "")
        self.assertEqual(shot.character_asset_ids, "[]")

    def test_character_update_invalidates_referencing_episode(self) -> None:
        series = Project(id="asset-series", title="Series", project_type="series")
        episode = Project(
            id="asset-episode",
            title="Episode",
            project_type="episode",
            parent_project_id=series.id,
            status="completed",
        )
        character = Character(id="asset-character", project_id=series.id, name="Before")
        shot = Shot(
            id="asset-shot",
            project_id=episode.id,
            sequence=1,
            version=4,
            confirmed=True,
            character_asset_ids=json.dumps([character.id]),
            storyboard_path="story.png",
            image_path="story.png",
            video_path="video.mp4",
            status="video_done",
            storyboard_status="done",
        )
        self.db.add_all([series, episode, character, shot])
        self.db.commit()
        character_id = character.id
        episode_id = episode.id
        shot_id = shot.id

        result = asyncio.run(
            asset_route.update_character_asset(
                character_id,
                asset_route.CharacterAssetUpdate(project_id=episode_id, name="After"),
                self.db,
            )
        )

        self.assertEqual(result["name"], "After")
        self.db.expire_all()
        current = self.db.get(Shot, shot_id)
        self.assertEqual(current.version, 5)
        self.assertFalse(current.confirmed)
        # 素材只标记过期：旧故事板 / 视频保留引用，供预览与回滚。
        self.assertTrue(current.media_stale)
        self.assertEqual(current.storyboard_path, "story.png")
        self.assertEqual(current.video_path, "video.mp4")
        self.assertEqual(self.db.get(Project, episode_id).status, "assets_ready")

    def test_compat_character_update_invalidates_referencing_episode(self) -> None:
        series = Project(id="compat-series", title="Series", project_type="series")
        episode = Project(
            id="compat-episode",
            title="Episode",
            project_type="episode",
            parent_project_id=series.id,
            status="completed",
        )
        character = Character(id="compat-character", project_id=series.id, name="Before")
        shot = Shot(
            id="compat-shot",
            project_id=episode.id,
            sequence=1,
            version=3,
            confirmed=True,
            character_asset_ids=json.dumps([character.id]),
            storyboard_path="story.png",
            image_path="story.png",
            video_path="video.mp4",
            status="video_done",
            storyboard_status="done",
        )
        self.db.add_all([series, episode, character, shot])
        self.db.commit()

        result = asyncio.run(
            character_route.update_character(
                character.id,
                character_route.CharacterUpdate(project_id=episode.id, name="After"),
                self.db,
            )
        )

        self.assertEqual(result["status"], "updated")
        self.db.expire_all()
        current = self.db.get(Shot, shot.id)
        self.assertEqual(current.version, 4)
        self.assertFalse(current.confirmed)
        self.assertTrue(current.media_stale)
        self.assertEqual(current.storyboard_path, "story.png")
        self.assertEqual(current.video_path, "video.mp4")
        self.assertEqual(self.db.get(Project, episode.id).status, "assets_ready")

    def test_character_regeneration_does_not_overwrite_concurrent_edit(self) -> None:
        project = Project(id="regen-project", title="Regenerate")
        character = Character(
            id="regen-character",
            project_id=project.id,
            name="Before",
            reference_images=json.dumps(["old.png"]),
        )
        self.db.add_all([project, character])
        self.db.commit()
        project_id = project.id
        character_id = character.id

        async def generate_then_edit(**_kwargs):
            other = SessionLocal()
            try:
                current = other.get(Character, character_id)
                current.name = "Concurrent edit"
                current.reference_images = json.dumps(["manual.png"])
                current.updated_at = datetime(2035, 1, 1)
                other.commit()
            finally:
                other.close()
            return str(_TEST_ROOT / "generated.png")

        with patch.object(
            asset_route.image_service,
            "generate_character_reference",
            side_effect=generate_then_edit,
        ):
            result = asyncio.run(
                asset_route.update_character_asset(
                    character_id,
                    asset_route.CharacterAssetUpdate(project_id=project_id, regenerate=True),
                    self.db,
                )
            )

        self.assertEqual(result["name"], "Concurrent edit")
        self.assertEqual(result["reference_images"], ["manual.png"])
        self.db.expire_all()
        current = self.db.get(Character, character_id)
        self.assertEqual(current.name, "Concurrent edit")
        self.assertEqual(json.loads(current.reference_images), ["manual.png"])


class _ChunkedUpload:
    """Minimal UploadFile-compatible object for direct route tests."""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content
        self._offset = 0
        self.read_sizes: list[int | None] = []

    async def read(self, size: int | None = -1) -> bytes:
        self.read_sizes.append(size)
        if size is None or size < 0:
            size = len(self._content) - self._offset
        chunk = self._content[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class ScriptUploadRegressionTests(DatabaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.db.add(Project(id="upload-project", title="Upload test"))
        self.db.commit()

    def test_upload_rejects_traversal_before_writing(self) -> None:
        upload = _ChunkedUpload("script.txt", b"safe text")
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(
                script_route.upload_script(
                    project_id="../outside",
                    file=upload,
                )
            )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(upload.read_sizes, [])
        self.assertFalse((_TEST_ROOT / "outside.txt").exists())

    def test_upload_is_streamed_and_bounded(self) -> None:
        original_limit = settings.MAX_SCRIPT_UPLOAD_BYTES
        settings.MAX_SCRIPT_UPLOAD_BYTES = 8
        try:
            upload = _ChunkedUpload("script.txt", b"0123456789")
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(
                    script_route.upload_script(
                        project_id="upload-project",
                        file=upload,
                    )
                )

            self.assertEqual(raised.exception.status_code, 413)
            self.assertTrue(upload.read_sizes)
            self.assertTrue(all(size is not None and size > 0 for size in upload.read_sizes))
            upload_dir = Path(settings.DATA_DIR) / "uploads" / "upload-project"
            remaining = list(upload_dir.glob("*")) if upload_dir.exists() else []
            self.assertFalse(remaining)
        finally:
            settings.MAX_SCRIPT_UPLOAD_BYTES = original_limit


class MediaAndProjectRegressionTests(DatabaseTestCase):
    def test_invalid_video_replacement_preserves_current_final(self) -> None:
        project = Project(id="import-project", title="Import test")
        self.db.add(project)
        self.db.commit()
        final_path = settings.OUTPUT_DIR / "projects" / project.id / "output" / "final.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(b"known-good-final")

        upload = _ChunkedUpload("bad.mp4", b"not an mp4" + b"x" * 1200)
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(project_route.import_final_video(project.id, upload, self.db))

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(final_path.read_bytes(), b"known-good-final")

    def test_import_video_cancels_render_before_publishing(self) -> None:
        project = Project(id="import-cancel-render", title="Import cancel", status="rendering")
        self.db.add(project)
        self.db.commit()
        key = f"project:{project.id}:render"
        self.assertTrue(claim(key, f"project:{project.id}"))
        upload_bytes = b"\x00\x00\x00\x18ftypisom" + b"x" * 1600
        upload = _ChunkedUpload("replacement.mp4", upload_bytes)

        async def scenario() -> dict:
            started = asyncio.Event()

            async def worker() -> None:
                started.set()
                await asyncio.Event().wait()

            task = task_registry.start(key, worker())
            await started.wait()
            result = await project_route.import_final_video(project.id, upload, self.db)
            self.assertTrue(task.done())
            await asyncio.sleep(0)
            return result

        result = asyncio.run(scenario())
        self.assertTrue(result["imported"])
        final_path = settings.OUTPUT_DIR / "projects" / project.id / "output" / "final.mp4"
        self.assertEqual(final_path.read_bytes(), upload_bytes)
        self.db.expire_all()
        self.assertEqual(self.db.get(Project, project.id).status, "completed")
        self.assertEqual(self.db.query(BackgroundJob).filter_by(idempotency_key=key).one().status, "cancelled")

    def test_import_video_rechecks_quota_after_render_lock(self) -> None:
        project = Project(id="import-quota-race", title="Import quota")
        self.db.add(project)
        self.db.commit()
        final_path = settings.OUTPUT_DIR / "projects" / project.id / "output" / "final.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(b"old-final" * 200)
        upload_bytes = b"\x00\x00\x00\x18ftypisom" + b"x" * 1600
        upload = _ChunkedUpload("replacement.mp4", upload_bytes)

        with (
            patch.object(project_route.storage_service, "ensure_project_capacity", return_value=10_000),
            patch.object(
                project_route.storage_service,
                "project_usage_bytes",
                return_value=int(settings.PROJECT_STORAGE_QUOTA_BYTES) + final_path.stat().st_size + 1,
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(project_route.import_final_video(project.id, upload, self.db))

        self.assertEqual(raised.exception.status_code, 413)
        self.assertEqual(final_path.read_bytes(), b"old-final" * 200)
        self.assertFalse(list(final_path.parent.glob("*.candidate")))

    def test_project_list_batches_parent_title_lookup(self) -> None:
        series = Project(id="series-list", title="Series", project_type="series")
        episodes = [
            Project(id=f"episode-list-{index}", title=f"Episode {index}", parent_project_id=series.id, project_type="episode", episode_number=index)
            for index in range(1, 4)
        ]
        self.db.add_all([series, *episodes])
        self.db.commit()
        statements: list[str] = []

        def record(_conn, _cursor, statement, _parameters, _context, _executemany):
            if "FROM projects" in statement:
                statements.append(statement)

        event.listen(engine, "before_cursor_execute", record)
        try:
            result = asyncio.run(project_route.list_projects(self.db))
        finally:
            event.remove(engine, "before_cursor_execute", record)

        self.assertEqual(len(result), 4)
        self.assertLessEqual(len(statements), 2)
        self.assertEqual(next(item for item in result if item["id"] == episodes[0].id)["parent_project_title"], "Series")

    def test_missing_or_tiny_video_is_not_reused(self) -> None:
        missing = SimpleNamespace(video_path=str(settings.OUTPUT_DIR / "missing.mp4"), status="video_done")
        self.assertFalse(_can_reuse_existing_video(missing))

        media_root = Path(settings.OUTPUT_DIR)
        media_root.mkdir(parents=True, exist_ok=True)

        tiny_path = media_root / "tiny.mp4"
        tiny_path.write_bytes(b"0" * 4095)
        tiny = SimpleNamespace(video_path=str(tiny_path), status="video_done")
        self.assertFalse(_can_reuse_existing_video(tiny))

        valid_path = media_root / "valid.mp4"
        valid_path.write_bytes(b"0" * 4096)
        valid = SimpleNamespace(video_path=str(valid_path), status="video_done")
        self.assertTrue(_can_reuse_existing_video(valid))

    def test_project_delete_removes_all_nested_descendants(self) -> None:
        series = Project(id="series", title="Series", project_type="series")
        episode = Project(
            id="episode",
            title="Episode",
            project_type="episode",
            parent_project_id=series.id,
            episode_number=1,
        )
        nested = Project(
            id="nested",
            title="Nested",
            project_type="episode",
            parent_project_id=episode.id,
            episode_number=2,
        )
        nested_shot = Shot(id="nested-shot", project_id=nested.id, sequence=1)
        nested_character = Character(id="nested-character", project_id=nested.id, name="Nested")
        nested_scene = SceneAsset(id="nested-scene", project_id=nested.id, name="Nested scene")
        nested_shot_id = nested_shot.id
        nested_character_id = nested_character.id
        nested_scene_id = nested_scene.id
        self.db.add_all([series, episode, nested, nested_shot, nested_character, nested_scene])
        self.db.commit()

        project_ids = {series.id, episode.id, nested.id}
        result = asyncio.run(project_route.delete_project(series.id, self.db))

        self.assertEqual(result["status"], "deleted")
        self.assertEqual(
            set(result["deleted_project_ids"]),
            project_ids,
        )
        self.assertIsNone(self.db.get(Project, series.id))
        self.assertIsNone(self.db.get(Project, episode.id))
        self.assertIsNone(self.db.get(Project, nested.id))
        self.assertIsNone(self.db.get(Shot, nested_shot_id))
        self.assertIsNone(self.db.get(Character, nested_character_id))
        self.assertIsNone(self.db.get(SceneAsset, nested_scene_id))

    def test_project_delete_restores_trash_when_database_commit_fails(self) -> None:
        project = Project(id="delete-rollback", title="Rollback", status="assets_ready")
        self.db.add(project)
        self.db.commit()
        output_dir = settings.OUTPUT_DIR / "projects" / project.id
        upload_dir = settings.DATA_DIR / "uploads" / project.id
        output_dir.mkdir(parents=True, exist_ok=True)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sentinel.mp4").write_bytes(b"output")
        (upload_dir / "script.txt").write_text("script", encoding="utf-8")

        real_commit = self.db.commit
        commit_calls = 0

        def commit_then_fail() -> None:
            nonlocal commit_calls
            commit_calls += 1
            if commit_calls == 2:
                raise RuntimeError("database commit failed")
            real_commit()

        with patch.object(self.db, "commit", side_effect=commit_then_fail):
            with self.assertRaisesRegex(RuntimeError, "database commit failed"):
                asyncio.run(project_route.delete_project(project.id, self.db))

        self.db.expire_all()
        restored = self.db.get(Project, project.id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.status, "assets_ready")
        self.assertEqual((output_dir / "sentinel.mp4").read_bytes(), b"output")
        self.assertEqual((upload_dir / "script.txt").read_text(encoding="utf-8"), "script")
        self.assertFalse((settings.OUTPUT_DIR / "projects" / ".trash").exists())
        self.assertFalse((settings.DATA_DIR / "uploads" / ".trash").exists())

    def test_startup_recovery_restores_interrupted_project_delete(self) -> None:
        project = Project(id="delete-recover", title="Recover", status="deleting")
        self.db.add(project)
        self.db.commit()
        output_dir = settings.OUTPUT_DIR / "projects" / project.id
        output_dir.mkdir(parents=True, exist_ok=True)
        sentinel = output_dir / "sentinel.mp4"
        sentinel.write_bytes(b"output")
        token = "recover-token"
        staged = project_route._stage_project_paths(
            [project.id],
            token,
            {project.id: "assets_ready"},
        )
        self.assertTrue(staged)
        self.assertFalse(output_dir.exists())

        self.assertEqual(project_route.recover_staged_project_deletions(), 1)
        self.assertTrue(sentinel.exists())
        self.db.expire_all()
        self.assertEqual(self.db.get(Project, project.id).status, "assets_ready")
        self.assertFalse((settings.OUTPUT_DIR / "projects" / ".trash").exists())

    def test_edit_hides_stale_final_video(self) -> None:
        project = Project(id="stale-final", title="Stale final", status="completed")
        shot = Shot(id="stale-final-shot", project_id=project.id, sequence=1, scene_description="before")
        self.db.add_all([project, shot])
        self.db.commit()
        final_path = settings.OUTPUT_DIR / "projects" / project.id / "output" / "final.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(b"x" * 2048)
        render_key = f"project:{project.id}:render"
        self.assertTrue(claim(render_key, f"project:{project.id}"))
        finish(render_key, "completed")
        render_route._render_status[project.id] = {
            "status": "completed",
            "progress": 100,
            "video_path": str(final_path),
        }
        self.assertTrue(asyncio.run(project_route.get_project(project.id, self.db))["video_path"])

        asyncio.run(
            shot_route.update_shot(
                shot.id,
                shot_route.ShotUpdate(scene_description="after"),
                self.db,
            )
        )

        result = asyncio.run(project_route.get_project(project.id, self.db))
        self.assertEqual(result["status"], "assets_ready")
        self.assertEqual(result["video_path"], "")
        self.assertTrue(final_path.exists())
        self.assertEqual(asyncio.run(render_route.get_render_status(project.id))["status"], "idle")

    def test_generation_setting_change_invalidates_project_media(self) -> None:
        project = Project(id="config-stale", title="Config stale", status="completed", resolution="1080p")
        shot = Shot(
            id="config-stale-shot",
            project_id=project.id,
            sequence=1,
            version=5,
            confirmed=True,
            storyboard_path="story.png",
            image_path="story.png",
            video_path="video.mp4",
            status="video_done",
            storyboard_status="done",
        )
        self.db.add_all([project, shot])
        self.db.commit()

        asyncio.run(
            project_route.update_project(
                project.id,
                project_route.ProjectUpdate(resolution="720p"),
                self.db,
            )
        )

        self.db.expire_all()
        self.assertEqual(self.db.get(Project, project.id).status, "assets_ready")
        current = self.db.get(Shot, shot.id)
        self.assertEqual(current.version, 6)
        self.assertFalse(current.confirmed)
        # 生成配置变化：全部镜头标记待重新生成，媒体路径保留不清空。
        self.assertTrue(current.media_stale)
        self.assertEqual(current.storyboard_path, "story.png")
        self.assertEqual(current.video_path, "video.mp4")

    def test_project_delete_waits_for_background_scope(self) -> None:
        project_id = "delete-active"
        shot_id = "delete-active-shot"
        self.db.add_all(
            [
                Project(id=project_id, title="Delete active"),
                Shot(id=shot_id, project_id=project_id, sequence=1),
            ]
        )
        self.db.commit()
        key = f"project:{project_id}:pipeline:auto"
        self.assertTrue(claim(key, f"project:{project_id}"))
        worker_finished: list[bool] = []

        async def scenario() -> dict:
            started = asyncio.Event()

            async def worker() -> None:
                try:
                    started.set()
                    await asyncio.Event().wait()
                finally:
                    other = SessionLocal()
                    try:
                        current = other.get(Project, project_id)
                        if current:
                            current.status = "worker-unwound"
                            other.commit()
                    finally:
                        other.close()
                    worker_finished.append(True)

            task_registry.start(key, worker())
            await started.wait()
            result = await project_route.delete_project(project_id, self.db)
            self.assertEqual(worker_finished, [True])
            return result

        result = asyncio.run(scenario())
        self.assertEqual(result["status"], "deleted")
        self.assertIsNone(self.db.get(Project, project_id))
        self.assertIsNone(self.db.get(Shot, shot_id))
        self.assertEqual(self.db.query(BackgroundJob).filter_by(idempotency_key=key).one().status, "cancelled")


class BackgroundJobRegressionTests(DatabaseTestCase):
    def test_atomic_claim_blocks_duplicate_key_and_scope(self) -> None:
        self.assertTrue(claim("shot:s1:storyboard", "shot:s1", version=4))
        self.assertFalse(claim("shot:s1:storyboard", "shot:s1", version=4))
        self.assertFalse(claim("shot:s1:video", "shot:s1", version=4))

        job = self.db.query(BackgroundJob).filter_by(idempotency_key="shot:s1:storyboard").one()
        self.assertEqual(job.status, "running")
        self.assertEqual(job.version, 4)

        finish("shot:s1:storyboard", "completed")

    def test_old_run_token_cannot_update_reclaimed_attempt(self) -> None:
        key = "project:token-project:render"
        self.assertTrue(claim(key, "project:token-project"))
        first_token = task_registry.snapshot(key)["run_token"]
        finish(key, "failed", "first attempt", run_token=first_token)

        self.assertTrue(claim(key, "project:token-project"))
        second = task_registry.snapshot(key)
        second_token = second["run_token"]
        self.assertNotEqual(first_token, second_token)
        self.assertFalse(task_registry.update_progress(key, 91, run_token=first_token))
        self.assertFalse(task_registry.finish(key, "completed", run_token=first_token))
        current = task_registry.snapshot(key)
        self.assertEqual(current["status"], "running")
        self.assertEqual(current["progress"], 0)
        self.assertEqual(current["run_token"], second_token)
        self.assertTrue(task_registry.finish(key, "completed", run_token=second_token))

    def test_unregistered_async_worker_cannot_use_latest_token(self) -> None:
        key = "project:unregistered-worker:render"
        self.assertTrue(claim(key, "project:unregistered-worker"))
        first_token = task_registry.snapshot(key)["run_token"]
        finish(key, "failed", "first attempt", run_token=first_token)
        self.assertTrue(claim(key, "project:unregistered-worker"))
        second_token = task_registry.snapshot(key)["run_token"]

        async def stale_worker() -> None:
            self.assertFalse(task_registry.update_progress(key, 88))
            self.assertFalse(task_registry.finish(key, "completed"))

        asyncio.run(stale_worker())
        current = task_registry.snapshot(key)
        self.assertEqual(current["run_token"], second_token)
        self.assertEqual(current["status"], "running")
        finish(key, "cancelled", run_token=second_token)

    def test_duplicate_start_does_not_fail_existing_attempt(self) -> None:
        key = "project:duplicate-start:render"
        self.assertTrue(claim(key, "project:duplicate-start"))

        async def scenario() -> None:
            started = asyncio.Event()

            async def worker() -> None:
                started.set()
                await asyncio.Event().wait()

            task = task_registry.start(key, worker())
            await started.wait()
            with self.assertRaisesRegex(RuntimeError, "注册失败"):
                task_registry.start(key, asyncio.sleep(0))
            self.assertEqual(task_registry.snapshot(key)["status"], "running")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(scenario())
        self.assertEqual(task_registry.snapshot(key)["status"], "cancelled")
        self.assertTrue(claim("shot:s1:storyboard", "shot:s1", version=5))
        self.db.expire_all()
        job = self.db.query(BackgroundJob).filter_by(idempotency_key="shot:s1:storyboard").one()
        self.assertEqual(job.version, 5)
        finish("shot:s1:storyboard", "completed")

    def test_project_and_child_shot_scopes_are_mutually_exclusive(self) -> None:
        project = Project(id="hierarchy-project", title="Hierarchy")
        shot = Shot(id="hierarchy-shot", project_id=project.id, sequence=1)
        self.db.add_all([project, shot])
        self.db.commit()
        project_key = f"project:{project.id}:storyboard"
        shot_key = f"shot:{shot.id}:video"

        self.assertTrue(claim(project_key, f"project:{project.id}"))
        self.assertFalse(claim(shot_key, f"shot:{shot.id}"))
        finish(project_key, "completed")
        self.assertTrue(claim(shot_key, f"shot:{shot.id}"))
        self.assertFalse(claim(project_key, f"project:{project.id}"))
        finish(shot_key, "completed")

    def test_restart_marks_live_jobs_interrupted(self) -> None:
        project = Project(id="p1", title="Interrupted render", status="rendering")
        self.db.add(project)
        self.db.commit()
        self.assertTrue(claim("project:p1:render", "project:p1"))
        self.assertEqual(recover_interrupted(), 1)
        self.db.expire_all()
        job = self.db.query(BackgroundJob).filter_by(idempotency_key="project:p1:render").one()
        self.assertEqual(job.status, "interrupted")
        self.assertIn("restarted", job.error)
        self.assertEqual(self.db.get(Project, project.id).status, "error")

    def test_restart_reconciles_queued_storyboards(self) -> None:
        project = Project(id="restart-storyboard", title="Restart", status="storyboard_generating")
        shot = Shot(
            id="restart-shot",
            project_id=project.id,
            sequence=1,
            version=3,
            status="pending",
            storyboard_status="queued",
        )
        self.db.add_all([project, shot])
        self.db.commit()
        self.assertTrue(claim(f"project:{project.id}:storyboard", f"project:{project.id}"))

        self.assertEqual(recover_interrupted(), 1)

        self.db.expire_all()
        self.assertEqual(self.db.get(Project, project.id).status, "error")
        current = self.db.get(Shot, shot.id)
        self.assertEqual(current.status, "failed")
        self.assertEqual(current.storyboard_status, "failed")

    def test_render_status_falls_back_to_durable_job(self) -> None:
        project_id = "durable-render"
        key = f"project:{project_id}:render"
        self.assertTrue(claim(key, f"project:{project_id}"))
        task_registry.update_progress(key, 37)
        render_route._render_status.pop(project_id, None)

        running = asyncio.run(render_route.get_render_status(project_id))
        self.assertEqual(running["status"], "rendering")
        self.assertEqual(running["progress"], 37)

        finish(key, "failed", "render process stopped")
        failed = asyncio.run(render_route.get_render_status(project_id))
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["message"], "render process stopped")

    def test_task_creation_failure_releases_claim_and_transient_state(self) -> None:
        project = Project(id="schedule-project", title="Schedule")
        shot = Shot(
            id="schedule-shot",
            project_id=project.id,
            sequence=1,
            version=2,
            status="pending",
            storyboard_status="queued",
        )
        self.db.add_all([project, shot])
        self.db.commit()
        key = f"shot:{shot.id}:storyboard"
        self.assertTrue(claim(key, f"shot:{shot.id}", version=shot.version))

        async def fail_to_start() -> None:
            coroutine = asyncio.sleep(0)
            with patch.object(task_registry.asyncio, "create_task", side_effect=RuntimeError("loop stopped")):
                with self.assertRaisesRegex(RuntimeError, "loop stopped"):
                    task_registry.start(key, coroutine)

        asyncio.run(fail_to_start())
        self.db.expire_all()
        job = self.db.query(BackgroundJob).filter_by(idempotency_key=key).one()
        self.assertEqual(job.status, "failed")
        current = self.db.get(Shot, shot.id)
        self.assertEqual(current.status, "failed")
        self.assertEqual(current.storyboard_status, "failed")
        self.assertTrue(claim(key, f"shot:{shot.id}", version=shot.version))
        finish(key, "completed")

    def test_create_task_failure_closes_wrapper_without_coroutine_leak(self) -> None:
        """create_task 抛错时包装协程也必须被关闭：不留 never-awaited 警告与泄漏。"""
        project = Project(id="wrapper-leak-project", title="Wrapper leak")
        self.db.add(project)
        self.db.commit()
        key = f"project:{project.id}:render"
        self.assertTrue(claim(key, f"project:{project.id}"))
        executed: list[bool] = []

        async def scenario() -> list[warnings.WarningMessage]:
            async def worker() -> None:
                executed.append(True)

            coroutine = worker()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with patch.object(task_registry.asyncio, "create_task", side_effect=RuntimeError("loop stopped")):
                    try:
                        task_registry.start(key, coroutine)
                    except RuntimeError:
                        pass
                # start() 抛出的异常在 except 块结束时已释放；此处强制回收，
                # 任何未关闭的协程都会在这时发出 "never awaited" RuntimeWarning。
                gc.collect()
                self.assertIsNone(coroutine.cr_frame, "原始协程应被恰好关闭一次")
            return caught

        caught = asyncio.run(scenario())
        never_awaited = [
            warning
            for warning in caught
            if issubclass(warning.category, RuntimeWarning) and "never awaited" in str(warning.message)
        ]
        self.assertEqual(never_awaited, [], "create_task 失败时包装协程不得泄漏 never-awaited 警告")
        leaked = [
            obj
            for obj in gc.get_objects()
            if isinstance(obj, types.CoroutineType)
            and obj.cr_code.co_name == "_with_usage_scope"
            and obj.cr_frame is not None
        ]
        self.assertEqual(leaked, [], "create_task 失败后不得残留未完成的包装协程")
        self.assertEqual(executed, [], "关闭路径不得执行原始协程")
        # 既有语义不变：失败状态落库、claim 可回收、run token 防旧任务回写。
        self.db.expire_all()
        job = self.db.query(BackgroundJob).filter_by(idempotency_key=key).one()
        self.assertEqual(job.status, "failed")
        failed_token = task_registry.snapshot(key)["run_token"]
        self.assertTrue(claim(key, f"project:{project.id}"))
        new_token = task_registry.snapshot(key)["run_token"]
        self.assertNotEqual(failed_token, new_token)
        self.assertFalse(task_registry.update_progress(key, 50, run_token=failed_token))
        self.assertTrue(task_registry.finish(key, "completed", run_token=new_token))

    def test_scope_cancel_blocks_reclaim_and_resets_business_state(self) -> None:
        project = Project(id="cancel-project", title="Cancel", status="storyboard_generating")
        shot = Shot(
            id="cancel-shot",
            project_id=project.id,
            sequence=1,
            status="pending",
            storyboard_status="queued",
        )
        self.db.add_all([project, shot])
        self.db.commit()
        scope = f"project:{project.id}"
        key = f"project:{project.id}:storyboard"
        self.assertTrue(claim(key, scope))
        reclaim_results: list[bool] = []

        async def scenario() -> None:
            started = asyncio.Event()

            async def worker() -> None:
                try:
                    started.set()
                    await asyncio.Event().wait()
                finally:
                    reclaim_results.append(claim(f"project:{project.id}:render", scope))

            task = task_registry.start(key, worker())
            await started.wait()
            cancelled = await task_registry.cancel_scopes({scope}, "user cancelled")
            self.assertEqual(cancelled, 1)
            self.assertTrue(task.done())

        asyncio.run(scenario())
        self.assertEqual(reclaim_results, [False])
        self.db.expire_all()
        self.assertEqual(self.db.query(BackgroundJob).filter_by(idempotency_key=key).one().status, "cancelled")
        self.assertEqual(self.db.get(Project, project.id).status, "assets_ready")
        current = self.db.get(Shot, shot.id)
        self.assertEqual(current.status, "pending")
        self.assertEqual(current.storyboard_status, "pending")
        self.assertTrue(claim(f"project:{project.id}:render", scope))
        finish(f"project:{project.id}:render", "completed")

    def test_project_scope_cancel_finalizes_child_shot_job(self) -> None:
        project = Project(id="cancel-child-project", title="Cancel child")
        shot = Shot(
            id="cancel-child-shot",
            project_id=project.id,
            sequence=1,
            version=2,
            confirmed=True,
            status="video_generating",
        )
        self.db.add_all([project, shot])
        self.db.commit()
        key = f"shot:{shot.id}:video"
        scope = f"shot:{shot.id}"
        self.assertTrue(claim(key, scope, version=shot.version))

        async def scenario() -> None:
            started = asyncio.Event()

            async def worker() -> None:
                started.set()
                await asyncio.Event().wait()

            task_registry.start(key, worker())
            await started.wait()
            await task_registry.cancel_scopes({f"project:{project.id}"}, "project changed")

        asyncio.run(scenario())
        self.db.expire_all()
        self.assertEqual(self.db.query(BackgroundJob).filter_by(idempotency_key=key).one().status, "cancelled")
        self.assertTrue(claim(key, scope, version=shot.version))
        finish(key, "completed")

    def test_busy_generation_lock_fails_job_instead_of_completing(self) -> None:
        project = Project(id="lock-project", title="Lock")
        shot = Shot(id="lock-shot", project_id=project.id, sequence=1, version=1)
        self.db.add_all([project, shot])
        self.db.commit()
        key = f"shot:{shot.id}:storyboard"
        self.assertTrue(claim(key, f"shot:{shot.id}", version=shot.version))

        async def scenario() -> None:
            lock = shot_route._shot_generation_locks.setdefault(shot.id, asyncio.Lock())
            await lock.acquire()
            try:
                task = task_registry.start(key, shot_route._regenerate_single_shot(shot.id, expected_version=shot.version))
                with self.assertRaisesRegex(RuntimeError, "已在运行"):
                    await task
                await asyncio.sleep(0)
            finally:
                lock.release()

        asyncio.run(scenario())
        self.db.expire_all()
        self.assertEqual(self.db.query(BackgroundJob).filter_by(idempotency_key=key).one().status, "failed")

    def test_upstream_edit_fences_and_cancels_downstream_video(self) -> None:
        project = Project(id="downstream-project", title="Downstream", status="completed")
        first = Shot(
            id="downstream-first",
            project_id=project.id,
            sequence=1,
            version=1,
            scene_group_id="room",
            scene_description="before",
        )
        second = Shot(
            id="downstream-second",
            project_id=project.id,
            sequence=2,
            version=1,
            scene_group_id="room",
            confirmed=True,
            storyboard_path="story.png",
            image_path="story.png",
            status="video_generating",
        )
        self.db.add_all([project, first, second])
        self.db.commit()
        key = f"shot:{second.id}:video"
        self.assertTrue(claim(key, f"shot:{second.id}", version=second.version))

        async def scenario() -> None:
            started = asyncio.Event()

            async def worker() -> None:
                started.set()
                await asyncio.Event().wait()

            task_registry.start(key, worker())
            await started.wait()
            await shot_route.update_shot(
                first.id,
                shot_route.ShotUpdate(scene_description="after"),
                self.db,
            )

        asyncio.run(scenario())
        self.db.expire_all()
        current = self.db.get(Shot, second.id)
        self.assertEqual(current.version, 2)
        self.assertEqual(current.video_path, "")
        self.assertEqual(self.db.query(BackgroundJob).filter_by(idempotency_key=key).one().status, "cancelled")

    def test_video_and_audio_outputs_use_versioned_media_id(self) -> None:
        project = Project(id="media-version-project", title="Media version")
        shot = Shot(
            id="media-version-shot",
            project_id=project.id,
            sequence=1,
            version=7,
            confirmed=True,
            dialogue="hello",
            storyboard_path="story.png",
            image_path="story.png",
            status="video_generating",
        )
        self.db.add_all([project, shot])
        self.db.commit()
        captured: dict[str, str] = {}

        async def fake_tts(**kwargs):
            captured["audio_id"] = kwargs["shot_id"]
            return str(settings.OUTPUT_DIR / "versioned.wav")

        async def fake_video(shot_data, *_args):
            captured["video_id"] = shot_data["shot_id"]
            return {
                "video_path": str(settings.OUTPUT_DIR / "versioned.mp4"),
                "frame_path": str(settings.OUTPUT_DIR / "versioned.png"),
            }

        with (
            patch.object(dialogue_audio.tts_service, "generate_dialogue", side_effect=fake_tts),
            patch.object(shot_route.seedance_service, "generate_shot_video", side_effect=fake_video),
        ):
            asyncio.run(shot_route._run_single_shot_video(shot.id, expected_version=7))

        self.assertEqual(captured["audio_id"], f"{shot.id}_v7")
        self.assertEqual(captured["video_id"], f"{shot.id}_v7")

    def test_stale_shot_result_does_not_overwrite_newer_version(self) -> None:
        project = Project(id="stale-project", title="Stale")
        shot = Shot(id="stale-shot", project_id=project.id, sequence=1, version=1, status="pending")
        self.db.add_all([project, shot])
        self.db.commit()

        async def generate_then_bump_version(**_kwargs):
            other = SessionLocal()
            try:
                current = other.get(Shot, "stale-shot")
                current.version = 2
                current.status = "edited"
                other.commit()
            finally:
                other.close()
            return str(_TEST_ROOT / "stale-result.png")

        with patch.object(shot_route.image_service, "generate_shot_image", side_effect=generate_then_bump_version):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(shot_route._regenerate_single_shot("stale-shot", expected_version=1))

        self.db.expire_all()
        current = self.db.get(Shot, "stale-shot")
        self.assertEqual(current.version, 2)
        self.assertEqual(current.status, "edited")
        self.assertEqual(current.storyboard_path, "")

    def test_automatic_storyboard_snapshots_versions(self) -> None:
        project = Project(id="auto-stale-project", title="Auto stale", status="storyboard_generating")
        shot = Shot(
            id="auto-stale-shot",
            project_id=project.id,
            sequence=1,
            version=1,
            status="pending",
            storyboard_status="queued",
        )
        self.db.add_all([project, shot])
        self.db.commit()

        async def generate_then_edit(**_kwargs):
            other = SessionLocal()
            try:
                current = other.get(Shot, shot.id)
                current.version = 2
                current.status = "edited"
                current.storyboard_status = "pending"
                other.commit()
            finally:
                other.close()
            return str(_TEST_ROOT / "auto-stale-result.png")

        with patch.object(shot_route.image_service, "generate_shot_image", side_effect=generate_then_edit):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(shot_route._run_storyboard_generation(project.id, [shot.id]))

        self.db.expire_all()
        current = self.db.get(Shot, shot.id)
        self.assertEqual(current.version, 2)
        self.assertEqual(current.status, "edited")
        self.assertEqual(current.storyboard_status, "pending")
        self.assertEqual(current.storyboard_path, "")
        self.assertEqual(self.db.get(Project, project.id).status, "assets_ready")

    def test_render_exception_is_raised_and_status_is_error(self) -> None:
        project = Project(id="render-error", title="Render error")
        video_path = settings.OUTPUT_DIR / "render-error-source.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"x" * 4096)
        shot = Shot(
            id="render-error-shot",
            project_id=project.id,
            sequence=1,
            confirmed=True,
            storyboard_path=str(video_path),
            video_path=str(video_path),
            status="video_done",
        )
        self.db.add_all([project, shot])
        self.db.commit()

        async def compose_failure(**_kwargs):
            raise RuntimeError("ffmpeg exploded")

        with patch.object(render_route.ffmpeg_service, "compose_video", side_effect=compose_failure):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg exploded"):
                asyncio.run(render_route._render_task(project.id, "9:16", "1080p"))

        self.assertEqual(render_route._render_status[project.id]["status"], "error")
        self.db.expire_all()
        self.assertEqual(self.db.get(Project, project.id).status, "error")

    def test_render_manifest_change_does_not_publish_staged_video(self) -> None:
        project = Project(id="render-stale", title="Render stale")
        source = settings.OUTPUT_DIR / "render-stale-source.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"s" * 4096)
        shot = Shot(
            id="render-stale-shot",
            project_id=project.id,
            sequence=1,
            version=1,
            confirmed=True,
            storyboard_path=str(source),
            video_path=str(source),
            status="video_done",
        )
        self.db.add_all([project, shot])
        self.db.commit()
        final_path = settings.OUTPUT_DIR / "projects" / project.id / "output" / "final.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(b"old-final" * 200)
        old_final = final_path.read_bytes()

        async def compose_then_edit(**_kwargs):
            candidate = final_path.parent / ".candidate.mp4"
            candidate.write_bytes(b"new-final" * 200)
            other = SessionLocal()
            try:
                current = other.get(Shot, shot.id)
                current.version = 2
                current.confirmed = False
                other.commit()
            finally:
                other.close()
            return str(candidate)

        with patch.object(render_route.ffmpeg_service, "compose_video", side_effect=compose_then_edit):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(render_route._render_task(project.id, "9:16", "1080p"))

        self.assertEqual(final_path.read_bytes(), old_final)
        self.assertFalse((final_path.parent / ".candidate.mp4").exists())
        self.db.expire_all()
        self.assertNotEqual(self.db.get(Project, project.id).status, "completed")

    def test_render_publish_restores_previous_final_when_commit_fails(self) -> None:
        project = Project(id="render-commit-fail", title="Render commit fail", status="rendering")
        source = settings.OUTPUT_DIR / "render-commit-fail-source.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"s" * 4096)
        shot = Shot(
            id="render-commit-fail-shot",
            project_id=project.id,
            sequence=1,
            version=1,
            confirmed=True,
            video_path=str(source),
            status="video_done",
        )
        self.db.add_all([project, shot])
        self.db.commit()
        final_path = settings.OUTPUT_DIR / "projects" / project.id / "output" / "final.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(b"old-final" * 200)
        staged = final_path.parent / ".render-candidate.mp4"
        staged.write_bytes(b"new-final" * 200)
        manifest = {
            shot.id: (shot.version or 1, bool(shot.confirmed), shot.video_path or "", shot.audio_path or "")
        }
        project_manifest = (project.style, project.output_format, project.resolution)
        publish_db = SessionLocal()
        try:
            with patch.object(publish_db, "commit", side_effect=RuntimeError("database commit failed")), patch.object(
                render_route, "SessionLocal", return_value=publish_db
            ):
                with self.assertRaisesRegex(RuntimeError, "database commit failed"):
                    render_route._publish_render(project.id, staged, manifest, project_manifest)
        finally:
            publish_db.close()

        self.assertEqual(final_path.read_bytes(), b"old-final" * 200)
        self.assertFalse(staged.exists())
        self.db.expire_all()
        self.assertEqual(self.db.get(Project, project.id).status, "rendering")

    def test_render_config_change_does_not_publish_staged_video(self) -> None:
        project = Project(
            id="render-config-stale",
            title="Render config stale",
            output_format="9:16",
            resolution="1080p",
        )
        source = settings.OUTPUT_DIR / "render-config-stale-source.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"s" * 4096)
        shot = Shot(
            id="render-config-stale-shot",
            project_id=project.id,
            sequence=1,
            version=1,
            confirmed=True,
            storyboard_path=str(source),
            video_path=str(source),
            status="video_done",
        )
        self.db.add_all([project, shot])
        self.db.commit()
        project_id = project.id
        final_path = settings.OUTPUT_DIR / "projects" / project_id / "output" / "final.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(b"old-final" * 200)
        old_final = final_path.read_bytes()

        async def compose_then_reconfigure(**_kwargs):
            candidate = final_path.parent / ".candidate.mp4"
            candidate.write_bytes(b"new-final" * 200)
            other = SessionLocal()
            try:
                current = other.get(Project, project_id)
                current.resolution = "720p"
                other.commit()
            finally:
                other.close()
            return str(candidate)

        with patch.object(render_route.ffmpeg_service, "compose_video", side_effect=compose_then_reconfigure):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(render_route._render_task(project_id, "9:16", "1080p"))

        self.assertEqual(final_path.read_bytes(), old_final)
        self.assertFalse((final_path.parent / ".candidate.mp4").exists())
        self.db.expire_all()
        current = self.db.get(Project, project_id)
        self.assertEqual(current.resolution, "720p")
        self.assertNotEqual(current.status, "completed")

    def test_completed_project_restores_render_status_without_memory_job(self) -> None:
        project = Project(id="auto-render-complete", title="Auto complete", status="completed")
        self.db.add(project)
        self.db.commit()
        final_path = settings.OUTPUT_DIR / "projects" / project.id / "output" / "final.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(b"f" * 2048)
        render_route._render_status.pop(project.id, None)

        result = asyncio.run(render_route.get_render_status(project.id))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["progress"], 100)
        self.assertEqual(result["video_path"], str(final_path))


if __name__ == "__main__":
    unittest.main()
