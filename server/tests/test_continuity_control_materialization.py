"""连续性控制能力声明的回归测试。

历史实现用 ``PIL.FIND_EDGES`` 生成边缘图冒充 OpenPose、用灰度高斯模糊冒充
Depth，并把画像标记为 enabled——这些是虚假能力声明。新契约：

- 没有接入真实姿态/深度模型时，画像统一标记 ``unsupported``；
- 不产出 openpose/depth 类参考资产，也不把它们当作必需素材；
- 视频参考预检只要求已审核分镜首帧（first_frame_only）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT  # noqa: F401,E402

from PIL import Image  # noqa: E402

from api.routes import shot as shot_route  # noqa: E402
from config import settings  # noqa: E402
from services.reference_asset_service import ReferenceAssetService  # noqa: E402
from services.video_service import VideoService  # noqa: E402


def _write_png(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (200, 120, 40)).save(path)
    return str(path)


def _shot_data(storyboard_path: str = "") -> dict:
    return {
        "shot_id": "regenfix_shot_0001",
        "storyboard_path": storyboard_path,
        "image_path": storyboard_path,
        "continuity_reference_path": "",
        "pose_reference_path": "/old/openpose_ref.png",
        "depth_reference_path": "/old/depth_ref.png",
        "reference_weights": {"action": 0.3, "environment": 0.45},
        "reference_assets": [
            {"type": "openpose_source_frame", "path": "/old/openpose_ref.png", "required": True},
            {"type": "depth_source_frame", "path": "/old/depth_ref.png", "required": True},
            {"type": "scene_baseline", "path": storyboard_path, "required": True},
        ],
        "continuity_profile": {
            "complex_motion": True,
            "openpose_lock": "enabled",  # 历史脏数据：必须被归一化为 unsupported
            "depth_lock": "enabled",
            "previous_reference_path": "",
        },
    }


class MaterializeControlReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_id = "regenfix_project"
        self.storyboard = _write_png(settings.OUTPUT_DIR / "regenfix" / "proj_shot_0001.png")

    def test_controls_marked_unsupported_and_purged(self) -> None:
        shot_data = _shot_data(self.storyboard)
        shot_route._materialize_control_references(self.project_id, shot_data, None)

        profile = shot_data["continuity_profile"]
        self.assertEqual(profile["openpose_lock"], "unsupported")
        self.assertEqual(profile["depth_lock"], "unsupported")
        self.assertEqual(profile.get("pose_control_model"), "unsupported")
        self.assertEqual(profile.get("depth_control_model"), "unsupported")
        self.assertEqual(shot_data["pose_reference_path"], "")
        self.assertEqual(shot_data["depth_reference_path"], "")
        types = {asset["type"] for asset in shot_data["reference_assets"]}
        self.assertNotIn("openpose_source_frame", types)
        self.assertNotIn("depth_source_frame", types)

    def test_materializer_never_produces_fake_control_images(self) -> None:
        service = ReferenceAssetService()
        controls = service.materialize_continuity_controls(
            project_id=self.project_id,
            shot_id="shot_x",
            source_path=self.storyboard,
            enabled=True,
        )
        self.assertEqual(controls, {})
        controls_dir = settings.OUTPUT_DIR / "projects" / self.project_id / "controls"
        self.assertFalse(controls_dir.exists())

    def test_consistency_profile_always_unsupported(self) -> None:
        from services.consistency_service import ConsistencyService

        service = ConsistencyService()
        context = service.build_generation_context(
            shot={"shot_id": "s1", "character_action": "转身跑向门口", "camera_movement": "跟随"},
            characters=[],
            scenes={},
            previous_reference_path="",
            for_video=True,
        )
        profile = context["continuity_profile"]
        self.assertEqual(profile["openpose_lock"], "unsupported")
        self.assertEqual(profile["depth_lock"], "unsupported")
        self.assertEqual(profile["pose_reference_path"], "")
        self.assertEqual(profile["depth_reference_path"], "")
        types = {asset["type"] for asset in context["reference_assets"]}
        self.assertNotIn("openpose_source_frame", types)
        self.assertNotIn("depth_source_frame", types)
        sentence = service._continuity_sentence(profile)
        self.assertIn("OpenPose control unsupported", sentence)
        self.assertIn("Depth control unsupported", sentence)

    def test_no_lora_or_ip_adapter_profiles_fabricated(self) -> None:
        from services.consistency_service import ConsistencyService

        service = ConsistencyService()
        enriched = service.enrich_character({"name": "林晚", "appearance": {"hair": "黑长直"}}, 0)
        self.assertEqual(enriched["lora_profile"], "")
        self.assertEqual(enriched["ip_adapter_profile"], "")


class VideoReferenceValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = VideoService()
        self.project_id = "regenfix_project"
        storyboard = _write_png(settings.OUTPUT_DIR / "regenfix" / "val_shot_0001.png")
        shot_data = _shot_data(storyboard)
        shot_route._materialize_control_references(self.project_id, shot_data, None)
        self.shot = shot_data

    def test_manifest_contains_only_first_frame_as_sent(self) -> None:
        manifest = self.service._validate_video_references(self.shot)
        kinds = {item["type"] for item in manifest}
        self.assertIn("approved_storyboard_first_frame", kinds)
        self.assertNotIn("openpose_control", kinds)
        self.assertNotIn("depth_control", kinds)
        for item in manifest:
            expected_sent = item["type"] == "approved_storyboard_first_frame"
            self.assertEqual(item["sent"], expected_sent)

    def test_missing_storyboard_fails_precheck(self) -> None:
        lying = _shot_data("")
        with self.assertRaisesRegex(RuntimeError, "approved_storyboard_first_frame"):
            self.service._validate_video_references(lying)


if __name__ == "__main__":
    unittest.main()
