from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from models import Character, Project, SceneAsset, Shot
from models.base import Base
from services import security, storage_service
from test_environment import TEST_ROOT  # noqa: F401,E402


class StorageReferenceGcTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="comic-agent-storage-")
        self.root = Path(self.temporary.name)
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.settings_patch = patch.multiple(
            storage_service.settings,
            OUTPUT_DIR=self.root / "output",
            PROJECT_TEMP_FILE_TTL_SECONDS=24 * 60 * 60,
            PROJECT_VERSION_RETENTION_COUNT=1,
            PROJECT_STORAGE_QUOTA_BYTES=1024 * 1024,
        )
        self.session_patch = patch.object(storage_service, "SessionLocal", self.session_factory)
        self.settings_patch.start()
        self.session_patch.start()
        self.service = storage_service.StorageService()

    def tearDown(self) -> None:
        self.session_patch.stop()
        self.settings_patch.stop()
        self.engine.dispose()
        self.temporary.cleanup()

    def test_gc_preserves_older_versions_referenced_by_models(self) -> None:
        project_id = "referenced-project"
        project_dir = self.service.get_project_dir(project_id)
        media_dir = project_dir / "shots"
        media_dir.mkdir(parents=True)

        paths = {
            name: media_dir / name
            for name in (
                "shot_v1.png",
                "shot_v2.png",
                "character_v1.png",
                "character_v2.png",
                "scene_v1.png",
                "scene_v2.png",
                "nested_v1.png",
                "nested_v2.png",
                "orphan_v1.png",
                "orphan_v2.png",
            )
        }
        for path in paths.values():
            path.write_bytes(b"media")

        session = self.session_factory()
        try:
            project = Project(id=project_id, title="Referenced")
            shot = Shot(
                id="referenced-shot",
                project_id=project_id,
                sequence=1,
                image_path=str(paths["shot_v1.png"]),
                consistency_context=json.dumps({"references": [f"/output/projects/{project_id}/shots/nested_v1.png"]}),
            )
            character = Character(
                id="referenced-character",
                project_id=project_id,
                name="Character",
                reference_images=json.dumps([f"projects/{project_id}/shots/character_v1.png"]),
            )
            scene = SceneAsset(
                id="referenced-scene",
                project_id=project_id,
                name="Scene",
                baseline_image_path=str(paths["scene_v1.png"]),
            )
            session.add_all([project, shot, character, scene])
            session.commit()
        finally:
            session.close()

        self.service.cleanup_project(project_id)

        for name in ("shot_v1.png", "character_v1.png", "scene_v1.png", "nested_v1.png"):
            self.assertTrue(paths[name].exists(), f"referenced version was deleted: {name}")
        self.assertFalse(paths["orphan_v1.png"].exists())
        for name in ("shot_v2.png", "character_v2.png", "scene_v2.png", "nested_v2.png", "orphan_v2.png"):
            self.assertTrue(paths[name].exists(), f"newest retained version was deleted: {name}")

    def test_save_file_uses_atomic_replace_and_preserves_previous_file_on_failure(self) -> None:
        path = Path(self.service.save_file("atomic-project", "shots", "result.bin", b"old-content"))
        real_atomic_write = storage_service.atomic_write_bytes
        with patch.object(storage_service, "atomic_write_bytes", wraps=real_atomic_write) as atomic_write:
            self.service.save_file("atomic-project", "shots", "result.bin", b"new-content")
            atomic_write.assert_called_once_with(path, b"new-content")
        self.assertEqual(path.read_bytes(), b"new-content")

        with patch.object(security.os, "replace", side_effect=OSError("publish failed")):
            with self.assertRaises(OSError):
                self.service.save_file("atomic-project", "shots", "result.bin", b"failed-content")
        self.assertEqual(path.read_bytes(), b"new-content")
        self.assertEqual(list(path.parent.glob(".result.bin.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
