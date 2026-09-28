"""素材保留（media_stale 标记）语义的验收测试。

核心原则：参数 / 配置变更只把素材标记为「待重新生成」，绝不清空
``image_path / storyboard_path / video_path / audio_path / last_frame_path``；
旧素材必须保留到新素材生成成功后的原子替换。

覆盖场景：
- 编辑镜头 Prompt：旧故事板保留 + 标记过期 + 版本历史记录被替换状态；
- 重生成排队 / 失败 / 成功：旧素材分别保留、保留、原子替换并入版本历史；
- 项目画风 / 配置变化：全部镜头标记待重生成但媒体路径不清空；
- 剧集切画风：不清空父项目共享的角色 / 场景资产；
- 版本恢复：四类媒体路径与快照一致、文件缺失返回 409 且当前镜头不变；
- 审核门禁：过期素材禁止通过审核；shot_update 下发版本号与过期标记。
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT  # noqa: F401,E402

from fastapi.testclient import TestClient  # noqa: E402

from api.routes import project as project_route  # noqa: E402
from api.routes import shot as shot_route  # noqa: E402
from agent import graph  # noqa: E402
from config import settings  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from main import app  # noqa: E402
from models import Character, Project, SceneAsset, Shot, ShotVersion  # noqa: E402
from services.shot_version_service import parse_snapshot  # noqa: E402


def _media_file(name: str, size: int = 2048) -> str:
    path = Path(settings.OUTPUT_DIR) / "media_stale_tests" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return str(path)


class MediaStaleTestCase(unittest.TestCase):
    prefix = "media_stale"

    @classmethod
    def setUpClass(cls) -> None:
        init_db()
        cls.client = TestClient(app)

    def setUp(self) -> None:
        self.db = SessionLocal()

    def tearDown(self) -> None:
        self.db.rollback()
        for model in (ShotVersion, Shot, Character, SceneAsset, Project):
            self.db.query(model).filter(
                getattr(model, "id", "").like(f"{self.prefix}%")
                if hasattr(model, "id") and model is not Project
                else model.id.like(f"{self.prefix}%")
            ).delete(synchronize_session=False)
        self.db.commit()
        self.db.close()

    def _project(self, project_id: str, **overrides) -> Project:
        project = Project(id=project_id, title="素材保留测试", **overrides)
        self.db.add(project)
        return project

    def _shot(self, shot_id: str, project_id: str, sequence: int, **overrides) -> Shot:
        defaults = dict(
            id=shot_id,
            project_id=project_id,
            sequence=sequence,
            version=1,
            storyboard_path="",
            image_path="",
        )
        defaults.update(overrides)
        shot = Shot(**defaults)
        self.db.add(shot)
        return shot

    def _versions(self, shot_id: str) -> list[ShotVersion]:
        return (
            self.db.query(ShotVersion)
            .filter(ShotVersion.shot_id == shot_id)
            .order_by(ShotVersion.number)
            .all()
        )


class EditPreservesMediaTests(MediaStaleTestCase):
    def test_edit_prompt_preserves_storyboard_and_marks_stale(self) -> None:
        """场景 1：修改镜头 Prompt 后旧 storyboard_path/image_path 仍保留。"""
        project_id = f"{self.prefix}_edit_project"
        shot_id = f"{self.prefix}_edit_shot"
        storyboard = _media_file("edit_old_storyboard.png")
        self._project(project_id)
        self._shot(
            shot_id,
            project_id,
            1,
            storyboard_path=storyboard,
            image_path=storyboard,
            status="storyboard_done",
            storyboard_status="done",
        )
        self.db.commit()

        asyncio.run(
            shot_route.update_shot(shot_id, shot_route.ShotUpdate(visual_notes="新的画面重点"), self.db)
        )

        self.db.expire_all()
        shot = self.db.get(Shot, shot_id)
        self.assertEqual(shot.storyboard_path, storyboard, "旧故事板路径不得被编辑清空")
        self.assertEqual(shot.image_path, storyboard, "旧图片路径不得被编辑清空")
        self.assertTrue(shot.media_stale, "编辑后素材必须标记待重新生成")
        self.assertFalse(shot.confirmed)
        # 被替换的编辑前状态进入版本历史（含旧素材路径，可回滚）。
        rows = self._versions(shot_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].source, "manual_edit")
        self.assertEqual(parse_snapshot(rows[0])["storyboard_path"], storyboard)

    def test_storyboard_and_ws_payload_carry_version_and_stale_flag(self) -> None:
        """序列化口径：列表接口与 shot_update 都带 version / media_stale。"""
        project_id = f"{self.prefix}_payload_project"
        shot_id = f"{self.prefix}_payload_shot"
        self._project(project_id)
        shot = self._shot(shot_id, project_id, 1, version=7, media_stale=True)
        self.db.commit()

        payload = shot_route._shot_update_payload(shot)
        self.assertEqual(payload["version"], 7)
        self.assertTrue(payload["media_stale"])

        response = self.client.get(f"/api/shot/{project_id}/shots")
        self.assertEqual(response.status_code, 200, response.text)
        serialized = next(item for item in response.json() if item["id"] == shot_id)
        self.assertTrue(serialized["media_stale"])
        self.assertEqual(serialized["version"], 7)

    def test_approve_blocked_while_media_stale(self) -> None:
        """过期素材（参数已修改）不得通过审核。"""
        project_id = f"{self.prefix}_approve_project"
        shot_id = f"{self.prefix}_approve_shot"
        storyboard = _media_file("approve_storyboard.png")
        self._project(project_id)
        self._shot(
            shot_id,
            project_id,
            1,
            storyboard_path=storyboard,
            image_path=storyboard,
            media_stale=True,
        )
        self.db.commit()

        response = self.client.post(f"/api/shot/{shot_id}/approve-storyboard", json={"approved": True})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("待重新生成", response.json()["detail"])


class RegenerationAtomicReplaceTests(MediaStaleTestCase):
    def _prepare(self, name: str) -> tuple[str, str, str]:
        project_id = f"{self.prefix}_{name}_project"
        shot_id = f"{self.prefix}_{name}_shot"
        storyboard = _media_file(f"{name}_old_storyboard.png")
        self._project(project_id)
        self._shot(
            shot_id,
            project_id,
            1,
            storyboard_path=storyboard,
            image_path=storyboard,
            status="storyboard_done",
            storyboard_status="done",
        )
        self.db.commit()
        return project_id, shot_id, storyboard

    def test_queue_and_failure_keep_old_storyboard_previewable(self) -> None:
        """场景 3：重生成排队与失败后，旧故事板仍保留、仍可预览。"""
        project_id, shot_id, storyboard = self._prepare("regen_fail")

        # 排队：_prepare_storyboard_candidate 只标记，不清路径。
        shot = self.db.get(Shot, shot_id)
        expected_version, _ = shot_route._prepare_storyboard_candidate(
            shot, self.db, shot_route.RegenerateRequest(reason="retry")
        )
        self.db.expire_all()
        shot = self.db.get(Shot, shot_id)
        self.assertEqual(shot.storyboard_path, storyboard, "排队重生成不得清空旧故事板")
        self.assertEqual(shot.storyboard_status, "queued")
        self.assertTrue(shot.media_stale)

        async def failing_generate(**kwargs):
            raise RuntimeError("图像服务不可用")

        with (
            patch.object(shot_route, "_ensure_scene_baselines", AsyncMock()),
            patch.object(shot_route.image_service, "generate_shot_image", side_effect=failing_generate),
        ):
            with self.assertRaises(RuntimeError):
                asyncio.run(shot_route._regenerate_single_shot(shot_id, "retry", expected_version))

        self.db.expire_all()
        shot = self.db.get(Shot, shot_id)
        self.assertEqual(shot.status, "failed")
        self.assertEqual(shot.storyboard_status, "failed")
        self.assertEqual(shot.storyboard_path, storyboard, "生成失败后旧故事板路径必须继续有效")
        self.assertEqual(shot.image_path, storyboard)

    def test_success_atomically_replaces_paths_and_versions(self) -> None:
        """场景 4：新故事板成功后旧路径被原子替换并进入版本历史。"""
        project_id, shot_id, storyboard = self._prepare("regen_ok")
        new_storyboard = _media_file("regen_ok_new_storyboard.png")

        shot = self.db.get(Shot, shot_id)
        expected_version, _ = shot_route._prepare_storyboard_candidate(
            shot, self.db, shot_route.RegenerateRequest(reason="retry")
        )
        self.db.commit()

        async def fake_generate(**kwargs):
            return new_storyboard

        with (
            patch.object(shot_route, "_ensure_scene_baselines", AsyncMock()),
            patch.object(shot_route.image_service, "generate_shot_image", side_effect=fake_generate),
        ):
            asyncio.run(shot_route._regenerate_single_shot(shot_id, "retry", expected_version))

        self.db.expire_all()
        shot = self.db.get(Shot, shot_id)
        self.assertEqual(shot.storyboard_path, new_storyboard, "成功后必须替换为新路径")
        self.assertEqual(shot.image_path, new_storyboard)
        self.assertEqual(shot.storyboard_status, "done")
        self.assertFalse(shot.media_stale, "无下游媒体时，成功重生成应清除过期标记")

        rows = self._versions(shot_id)
        snapshots = [parse_snapshot(row) for row in rows]
        self.assertTrue(any(s["storyboard_path"] == storyboard for s in snapshots), "旧素材必须留在版本历史")
        self.assertEqual(snapshots[-1]["storyboard_path"], new_storyboard, "生成结果必须入版本历史")

    def test_success_with_legacy_video_keeps_stale_until_video_rebuilt(self) -> None:
        """新故事板成功但旧视频仍引用旧故事板：保持待重生成提示。"""
        project_id, shot_id, storyboard = self._prepare("regen_video")
        video = _media_file("regen_video_old.mp4")
        shot = self.db.get(Shot, shot_id)
        shot.video_path = video
        self.db.commit()
        new_storyboard = _media_file("regen_video_new_storyboard.png")

        expected_version, _ = shot_route._prepare_storyboard_candidate(
            shot, self.db, shot_route.RegenerateRequest(reason="retry")
        )
        self.db.commit()

        async def fake_generate(**kwargs):
            return new_storyboard

        with (
            patch.object(shot_route, "_ensure_scene_baselines", AsyncMock()),
            patch.object(shot_route.image_service, "generate_shot_image", side_effect=fake_generate),
        ):
            asyncio.run(shot_route._regenerate_single_shot(shot_id, "retry", expected_version))

        self.db.expire_all()
        shot = self.db.get(Shot, shot_id)
        self.assertEqual(shot.storyboard_path, new_storyboard)
        self.assertEqual(shot.video_path, video, "旧视频保留预览")
        self.assertTrue(shot.media_stale, "旧视频仍待重生成，过期标记必须保留")


class ProjectConfigChangeTests(MediaStaleTestCase):
    def test_style_change_marks_eight_shots_stale_without_clearing(self) -> None:
        """场景 5：画风改变后 8 个镜头全部待重生成，但媒体路径不清空。"""
        project_id = f"{self.prefix}_style_project"
        self._project(project_id, style="anime")
        for index in range(1, 9):
            storyboard = _media_file(f"style_shot_{index}.png")
            self._shot(
                f"{self.prefix}_style_shot_{index}",
                project_id,
                index,
                storyboard_path=storyboard,
                image_path=storyboard,
                video_path=_media_file(f"style_shot_{index}.mp4"),
                status="video_done",
                storyboard_status="done",
            )
        self.db.commit()

        asyncio.run(
            project_route.update_project(project_id, project_route.ProjectUpdate(style="realistic"), self.db)
        )

        self.db.expire_all()
        shots = self.db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
        self.assertEqual(len(shots), 8)
        for shot in shots:
            self.assertTrue(shot.media_stale, "画风变化后必须标记待重新生成")
            self.assertTrue(bool(shot.storyboard_path), "故事板路径不得被清空")
            self.assertTrue(bool(shot.video_path), "视频路径不得被清空")
            self.assertFalse(shot.confirmed)

    def test_episode_style_change_keeps_parent_shared_assets(self) -> None:
        """场景 10：剧集风格改变不清空父项目共享角色和场景资源。"""
        series_id = f"{self.prefix}_ep_series"
        episode_id = f"{self.prefix}_ep_episode"
        self._project(series_id, project_type="series", style="anime")
        self._project(episode_id, project_type="episode", parent_project_id=series_id, style="anime")
        character = Character(
            id=f"{self.prefix}_ep_character",
            project_id=series_id,
            name="林晚",
            asset_status="active",
            reference_images='["/shared/character_ref.png"]',
        )
        scene = SceneAsset(
            id=f"{self.prefix}_ep_scene",
            project_id=series_id,
            name="天台",
            asset_status="active",
            baseline_image_path="/shared/scene_baseline.png",
            reference_images='["/shared/scene_baseline.png"]',
        )
        self.db.add_all([character, scene])
        storyboard = _media_file("ep_shot.png")
        self._shot(
            f"{self.prefix}_ep_shot",
            episode_id,
            1,
            storyboard_path=storyboard,
            image_path=storyboard,
            status="storyboard_done",
        )
        self.db.commit()

        asyncio.run(
            project_route.update_project(episode_id, project_route.ProjectUpdate(style="realistic"), self.db)
        )

        self.db.expire_all()
        self.assertEqual(character.reference_images, '["/shared/character_ref.png"]', "父项目共享角色资产不得被清空")
        self.assertEqual(character.asset_status, "active")
        self.assertEqual(scene.baseline_image_path, "/shared/scene_baseline.png", "父项目共享场景资产不得被清空")
        self.assertEqual(scene.asset_status, "active")
        shot = self.db.get(Shot, f"{self.prefix}_ep_shot")
        self.assertTrue(shot.media_stale, "本集镜头仍要标记待重新生成")
        self.assertEqual(shot.storyboard_path, storyboard)

    def test_auto_approve_aborts_when_any_shot_media_stale(self) -> None:
        """自动模式不得批准过期素材：与人工审核同口径。"""
        from PIL import Image

        project_id = f"{self.prefix}_auto_project"
        self._project(project_id)
        for index, stale in ((1, False), (2, True)):
            storyboard = _media_file(f"auto_shot_{index}.png")
            shot = self._shot(
                f"{self.prefix}_auto_shot_{index}",
                project_id,
                index,
                storyboard_path=storyboard,
                image_path=storyboard,
                media_stale=stale,
            )
            self.db.add(shot)
        self.db.commit()

        # 结构检查需要真实可解析图片（足够大的纯色图即可）。
        for shot in self.db.query(Shot).filter(Shot.project_id == project_id).all():
            Image.new("RGB", (640, 960), (200, 180, 160)).save(shot.storyboard_path)

        import logging

        with self.assertLogs("agent.graph", level=logging.ERROR) as captured:
            result = asyncio.run(graph._auto_approve_storyboard({"project_id": project_id}))

        self.assertIn("errors", result, "存在过期素材时自动批准必须中止")
        self.assertIn("素材待重新生成", "\n".join(captured.output))
        self.db.expire_all()
        for shot in self.db.query(Shot).filter(Shot.project_id == project_id).all():
            self.assertFalse(shot.confirmed, "过期镜头不得被自动批准")


class RestoreRecoveryTests(MediaStaleTestCase):
    def test_restore_reinstates_all_media_paths_from_snapshot(self) -> None:
        """场景 8：版本恢复后图片/视频/音频/尾帧路径与历史快照一致。"""
        project_id = f"{self.prefix}_restore_project"
        shot_id = f"{self.prefix}_restore_shot"
        image = _media_file("restore_image.png")
        video = _media_file("restore_video.mp4")
        audio = _media_file("restore_audio.wav")
        frame = _media_file("restore_frame.png")
        self._project(project_id)
        shot = self._shot(
            shot_id,
            project_id,
            1,
            storyboard_path=image,
            image_path=image,
            video_path=video,
            audio_path=audio,
            last_frame_path=frame,
            status="video_done",
        )
        row = shot_route.create_version(self.db, shot, "manual_edit")
        # 模拟历史缺陷：媒体路径被清空。
        shot.image_path = ""
        shot.storyboard_path = ""
        shot.video_path = ""
        shot.audio_path = ""
        shot.last_frame_path = ""
        shot.media_stale = True
        self.db.commit()

        response = self.client.post(f"/api/shot/{shot_id}/versions/{row.id}/restore")
        self.assertEqual(response.status_code, 200, response.text)

        self.db.expire_all()
        current = self.db.get(Shot, shot_id)
        self.assertEqual(current.image_path, image)
        self.assertEqual(current.storyboard_path, image)
        self.assertEqual(current.video_path, video)
        self.assertEqual(current.audio_path, audio)
        self.assertEqual(current.last_frame_path, frame)
        self.assertFalse(current.media_stale, "恢复后的素材与参数一致，应清除过期标记")

        sources = [item.source for item in self._versions(shot_id)]
        self.assertEqual(sources, ["manual_edit", "restore", "restore"], "恢复只追加，不改写历史")

    def test_restore_with_missing_file_returns_409_and_keeps_current(self) -> None:
        """场景 9：文件缺失时恢复返回 409，不修改当前镜头。"""
        project_id = f"{self.prefix}_409_project"
        shot_id = f"{self.prefix}_409_shot"
        media = _media_file("missing_media.png")
        self._project(project_id)
        shot = self._shot(shot_id, project_id, 1, storyboard_path=media, image_path=media)
        row = shot_route.create_version(self.db, shot, "manual_edit")
        shot.media_stale = True
        self.db.commit()
        Path(media).unlink()

        response = self.client.post(f"/api/shot/{shot_id}/versions/{row.id}/restore")
        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("媒体文件已缺失", response.json()["detail"])

        self.db.expire_all()
        current = self.db.get(Shot, shot_id)
        self.assertEqual(current.version, 1, "409 恢复不得递增版本号")
        self.assertEqual(current.storyboard_path, media, "409 恢复不得修改当前镜头状态")
        self.assertTrue(current.media_stale)
        self.assertEqual(len(self._versions(shot_id)), 1, "409 恢复不得追加版本记录")


if __name__ == "__main__":
    unittest.main()
