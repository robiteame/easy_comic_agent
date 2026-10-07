"""字幕/音频工作台 API 与渲染失效集成测试。

覆盖：轨道 CRUD 与版本推进、SRT/VTT 导入导出无损、从镜头对白自动生成、
素材路径安全、以及渲染期间字幕/音频配置变化时成片不得发布（旧 final 完整保留）。
"""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from api.routes import audio_track as audio_track_route  # noqa: E402
from api.routes import render as render_route  # noqa: E402
from api.routes import subtitle as subtitle_route  # noqa: E402
from config import settings  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import AudioTrack, BackgroundJob, Project, Shot, SubtitleCue, SubtitleTrack  # noqa: E402
from services.subtitle_service import format_srt_time  # noqa: E402
from tests.support.test_environment import TEST_ROOT as _TEST_ROOT  # noqa: E402,F401


class AvWorkbenchTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    def setUp(self) -> None:
        self.db = SessionLocal()
        self.db.query(BackgroundJob).delete()
        self.db.query(SubtitleCue).delete()
        self.db.query(SubtitleTrack).delete()
        self.db.query(AudioTrack).delete()
        self.db.query(Shot).delete()
        self.db.query(Project).delete()
        self.db.commit()
        self.project = Project(id="av-project", title="AV 项目")
        self.db.add(self.project)
        self.db.commit()

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()


class SubtitleRouteTests(AvWorkbenchTestCase):
    def test_track_crud_bumps_av_config_version(self) -> None:
        base_version = self.db.get(Project, self.project.id).av_config_version or 0
        track = asyncio.run(
            subtitle_route.create_subtitle_track(self.project.id, subtitle_route.SubtitleTrackCreate(), self.db)
        )
        self.db.expire_all()
        self.assertEqual(self.db.get(Project, self.project.id).av_config_version, base_version + 1)

        updated = asyncio.run(
            subtitle_route.update_subtitle_track(
                track["id"],
                subtitle_route.SubtitleTrackUpdate(project_id=self.project.id, font_size=72, position="top"),
                self.db,
            )
        )
        self.assertEqual(updated["font_size"], 72)
        self.assertEqual(updated["position"], "top")
        self.db.expire_all()
        self.assertEqual(self.db.get(Project, self.project.id).av_config_version, base_version + 2)

        listing = asyncio.run(subtitle_route.list_subtitle_tracks(self.project.id, self.db))
        self.assertEqual(len(listing["tracks"]), 1)
        self.assertEqual(listing["av_config_version"], base_version + 2)

        result = asyncio.run(subtitle_route.delete_subtitle_track(track["id"], self.project.id, self.db))
        self.assertTrue(result["ok"])
        self.assertEqual(asyncio.run(subtitle_route.list_subtitle_tracks(self.project.id, self.db))["tracks"], [])

    def test_cue_validation_rejects_bad_payload(self) -> None:
        from fastapi import HTTPException

        track = asyncio.run(
            subtitle_route.create_subtitle_track(self.project.id, subtitle_route.SubtitleTrackCreate(), self.db)
        )
        bad = subtitle_route.SubtitleCueReplace(
            project_id=self.project.id,
            cues=[subtitle_route.SubtitleCueInput(start_ms=1000, end_ms=500, text="倒序")],
        )
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(subtitle_route.replace_subtitle_cues(track["id"], bad, self.db))
        self.assertEqual(ctx.exception.status_code, 400)

        control = subtitle_route.SubtitleCueReplace(
            project_id=self.project.id,
            cues=[subtitle_route.SubtitleCueInput(start_ms=0, end_ms=900, text="合法", character_name="小明")],
        )
        saved = asyncio.run(subtitle_route.replace_subtitle_cues(track["id"], control, self.db))
        self.assertEqual(len(saved["cues"]), 1)
        self.assertEqual(saved["cues"][0]["text"], "合法")

    def test_import_export_round_trip_is_lossless(self) -> None:
        track = asyncio.run(
            subtitle_route.create_subtitle_track(self.project.id, subtitle_route.SubtitleTrackCreate(), self.db)
        )
        raw = "1\n00:00:01,000 --> 00:00:02,500\n第一句\n多行\n\n2\n00:00:03,000 --> 00:00:04,000\n第二句\n"
        imported = asyncio.run(
            subtitle_route.import_subtitle(
                track["id"],
                subtitle_route.SubtitleImportRequest(project_id=self.project.id, format="srt", content=raw),
                self.db,
            )
        )
        self.assertEqual(len(imported["cues"]), 2)

        exported = asyncio.run(subtitle_route.export_subtitle(track["id"], self.project.id, "srt", self.db))
        text = exported.body.decode("utf-8")
        self.assertIn(format_srt_time(1000), text)
        self.assertIn("第一句\n多行", text)

        reimported = asyncio.run(
            subtitle_route.import_subtitle(
                track["id"],
                subtitle_route.SubtitleImportRequest(project_id=self.project.id, format="srt", content=text),
                self.db,
            )
        )
        self.assertEqual(
            [(c["start_ms"], c["end_ms"], c["text"]) for c in reimported["cues"]],
            [(c["start_ms"], c["end_ms"], c["text"]) for c in imported["cues"]],
        )

    def test_generate_from_shots_uses_tts_duration(self) -> None:
        shot = Shot(
            id="av-shot-1", project_id=self.project.id, sequence=1, duration=4.0, dialogue="自动字幕", confirmed=True
        )
        shot.audio_path = "/nonexistent/tts.wav"
        self.db.add(shot)
        self.db.commit()
        track = asyncio.run(
            subtitle_route.create_subtitle_track(self.project.id, subtitle_route.SubtitleTrackCreate(), self.db)
        )

        async def fake_probe(path):
            return 2_500

        with patch.object(subtitle_route.ffmpeg_service, "probe_duration_ms", side_effect=fake_probe):
            generated = asyncio.run(
                subtitle_route.generate_subtitle_from_shots(
                    track["id"], subtitle_route.SubtitleGenerateRequest(project_id=self.project.id), self.db
                )
            )
        self.assertEqual(len(generated["cues"]), 1)
        # TTS 实际时长 2.5s：字幕在镜头内 2.5s 处结束。
        self.assertEqual((generated["cues"][0]["start_ms"], generated["cues"][0]["end_ms"]), (0, 2500))


class AudioTrackRouteTests(AvWorkbenchTestCase):
    def _media_file(self, name: str = "bgm.mp3", size: int = 2048) -> Path:
        target = settings.OUTPUT_DIR / "projects" / self.project.id / "audio_tracks" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"a" * size)
        return target

    def test_create_requires_validated_source(self) -> None:
        from fastapi import HTTPException

        outside = Path("/tmp/definitely-outside.mp3")
        outside.write_bytes(b"x" * 2048)
        try:
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(
                    audio_track_route.create_audio_track(
                        self.project.id,
                        audio_track_route.AudioTrackCreate(kind="music", source_path=str(outside)),
                        self.db,
                    )
                )
            self.assertEqual(ctx.exception.status_code, 400)
        finally:
            outside.unlink(missing_ok=True)

        media = self._media_file()

        async def fake_probe(path):
            return 12_000

        with patch.object(audio_track_route.ffmpeg_service, "probe_duration_ms", side_effect=fake_probe):
            track = asyncio.run(
                audio_track_route.create_audio_track(
                    self.project.id,
                    audio_track_route.AudioTrackCreate(
                        kind="music", source_path=str(media), duck_amount_db=-12.0, loop=True
                    ),
                    self.db,
                )
            )
        self.assertEqual(track["source_duration_ms"], 12_000)
        self.assertEqual(track["duck_amount_db"], -12.0)

        updated = asyncio.run(
            audio_track_route.update_audio_track(
                track["id"],
                audio_track_route.AudioTrackUpdate(project_id=self.project.id, volume=0.5, pan=-0.5),
                self.db,
            )
        )
        self.assertEqual(updated["volume"], 0.5)
        self.db.expire_all()
        self.assertGreaterEqual(self.db.get(Project, self.project.id).av_config_version, 2)

    def test_dialogue_track_requires_project_shot(self) -> None:
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                audio_track_route.create_audio_track(
                    self.project.id,
                    audio_track_route.AudioTrackCreate(kind="dialogue"),
                    self.db,
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)

        shot = Shot(id="av-dialogue-shot", project_id=self.project.id, sequence=1, duration=2.5, dialogue="对白")
        self.db.add(shot)
        self.db.commit()
        track = asyncio.run(
            audio_track_route.create_audio_track(
                self.project.id,
                audio_track_route.AudioTrackCreate(kind="dialogue", shot_id=shot.id, delay_ms=100),
                self.db,
            )
        )
        # 对白轨起点来自镜头区间，而非 start_ms 字段。
        self.assertEqual(track["shot_span"]["start_ms"], 0)
        self.assertEqual(track["shot_span"]["end_ms"], 2500)

    def test_analyze_reports_structural_warnings(self) -> None:
        shot = Shot(
            id="av-analyze-shot", project_id=self.project.id, sequence=1, duration=3.0, dialogue="对白", audio_path=""
        )
        self.db.add(shot)
        self.db.add(
            AudioTrack(
                id="av-analyze-1",
                project_id=self.project.id,
                kind="music",
                name="BGM",
                source_path=str(self._media_file("bgm2.mp3")),
                source_duration_ms=60_000,
                start_ms=10_000_000,  # 远超全片
            )
        )
        self.db.add(
            AudioTrack(
                id="av-analyze-2",
                project_id=self.project.id,
                kind="dialogue",
                name="对白",
                shot_id=shot.id,  # 镜头没有 TTS 配音 → 应产出警告
            )
        )
        self.db.commit()

        async def fake_detect(path):
            return {"max_volume_db": -0.2, "mean_volume_db": -20.0}

        with patch.object(audio_track_route.ffmpeg_service, "detect_volume", side_effect=fake_detect):
            result = asyncio.run(audio_track_route.analyze_audio_setup(self.project.id, self.db))
        codes = {item["code"] for item in result["warnings"]}
        self.assertIn("out_of_range", codes)
        self.assertIn("hot_source", codes)
        self.assertIn("dialogue_no_tts", codes)  # 对白轨绑定的镜头没有配音


class RenderInvalidationTests(AvWorkbenchTestCase):
    def _seed_renderable_project(self, project_id: str) -> tuple[Project, Shot]:
        project = Project(id=project_id, title="失效用例")
        source = settings.OUTPUT_DIR / f"{project_id}-source.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"s" * 4096)
        shot = Shot(
            id=f"{project_id}-shot",
            project_id=project_id,
            sequence=1,
            version=1,
            confirmed=True,
            storyboard_path=str(source),
            video_path=str(source),
            audio_path="",
            duration=3.0,
            status="video_done",
        )
        self.db.add_all([project, shot])
        self.db.commit()
        return project, shot

    def test_av_config_change_during_render_does_not_publish(self) -> None:
        project, shot = self._seed_renderable_project("av-render-stale")
        final_path = settings.OUTPUT_DIR / "projects" / project.id / "output" / "final.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(b"old-final" * 200)
        old_final = final_path.read_bytes()
        project_id = project.id

        captured: dict = {}

        async def compose_then_edit_av(**kwargs):
            captured.update(kwargs)
            candidate = final_path.parent / ".candidate.mp4"
            candidate.write_bytes(b"new-final" * 200)
            other = SessionLocal()
            try:
                # 渲染进行中：用户在工作台改了字幕轨道 → av_config_version 推进。
                other.add(
                    SubtitleTrack(
                        id="mid-render-track",
                        project_id=project_id,
                        name="中途创建",
                    )
                )
                current = other.get(Project, project_id)
                current.av_config_version = 5
                other.commit()
            finally:
                other.close()
            return str(candidate)

        with patch.object(render_route.ffmpeg_service, "compose_video", side_effect=compose_then_edit_av):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(render_route._render_task(project_id, "9:16", "1080p"))

        # compose_video 收到了工作台配置载荷。
        self.assertIn("av_config", captured)
        self.assertIn("audio_tracks", captured["av_config"])
        # 旧成片未被覆盖，暂存成片被丢弃。
        self.assertEqual(final_path.read_bytes(), old_final)
        self.assertFalse((final_path.parent / ".candidate.mp4").exists())
        self.db.expire_all()
        self.assertNotEqual(self.db.get(Project, project_id).status, "completed")

    def test_publish_av_manifest_mismatch_blocks_promotion(self) -> None:
        project, shot = self._seed_renderable_project("av-publish-stale")
        final_path = settings.OUTPUT_DIR / "projects" / project.id / "output" / "final.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(b"keep-me" * 200)
        staged = final_path.parent / ".staged.mp4"
        staged.write_bytes(b"new-video" * 200)

        manifest = {shot.id: (shot.version or 1, True, shot.video_path or "", "")}
        project_manifest = render_route._project_manifest_tuple(self.db.get(Project, project.id))

        # 渲染时没有任何轨道；发布前新增一条音轨（manifest 不一致）。
        self.db.add(
            AudioTrack(
                id="late-track",
                project_id=project.id,
                kind="music",
                name="后加的 BGM",
                source_path=str(settings.OUTPUT_DIR / "projects" / project.id / "output" / "final.mp4"),
                source_duration_ms=1000,
            )
        )
        self.db.commit()

        with self.assertRaises(asyncio.CancelledError):
            render_route._publish_render(project.id, staged, manifest, project_manifest, av_manifest=[])

        self.assertEqual(final_path.read_bytes(), b"keep-me" * 200)
        # staged 文件由 _render_task 的 finally 负责清理；发布层只保证不碰旧成片。


if __name__ == "__main__":
    unittest.main()
