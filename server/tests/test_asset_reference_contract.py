"""素材板 DTO 必须完整暴露参考素材生命周期状态。"""

from __future__ import annotations

import json
import unittest

from api.routes.asset import _serialize_character, _serialize_scene
from models import Character, SceneAsset
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


class AssetReferenceSerializationTests(unittest.TestCase):
    def test_character_serialization_preserves_reference_contract(self) -> None:
        item = Character(
            id="char-1",
            project_id="project-1",
            name="林晓",
            reference_images=json.dumps(["/tmp/character.png"]),
            reference_status="ready",
            reference_version=3,
            reference_retry_count=2,
            reference_failure_reason="",
            reference_error_id="",
            reference_skip_reason="",
            reference_capability_warning="参考图能力受限",
            reference_impact=json.dumps({"shot_ids": ["shot-1"], "shot_range": "镜头 1"}),
        )

        data = _serialize_character(item)

        self.assertEqual(data["reference_status"], "ready")
        self.assertEqual(data["reference_version"], 3)
        self.assertEqual(data["reference_retry_count"], 2)
        self.assertEqual(data["reference_capability_warning"], "参考图能力受限")
        self.assertEqual(data["reference_impact"]["shot_ids"], ["shot-1"])

    def test_scene_serialization_preserves_failure_fields(self) -> None:
        item = SceneAsset(
            id="scene-1",
            project_id="project-1",
            name="天台",
            reference_images=json.dumps(["/tmp/scene.png"]),
            reference_status="stale",
            reference_version=2,
            reference_retry_count=4,
            reference_failure_reason="画风已切换",
            reference_error_id="error-7",
            reference_skip_reason="",
            reference_capability_warning="",
            reference_impact=json.dumps({"shot_ids": ["shot-2"], "shot_range": "镜头 2"}),
        )

        data = _serialize_scene(item)

        self.assertEqual(data["reference_status"], "stale")
        self.assertEqual(data["reference_failure_reason"], "画风已切换")
        self.assertEqual(data["reference_error_id"], "error-7")
        self.assertEqual(data["reference_impact"]["shot_range"], "镜头 2")


if __name__ == "__main__":
    unittest.main()
