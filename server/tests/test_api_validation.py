"""P2: API DTO 服务端边界校验的回归测试。

覆盖范围：
- 业务枚举（project_type / input_type / output_format / resolution / platform / mode /
  audio_mode 等）在服务端生效，非法值返回 422；
- 字符串长度上限与纯空白输入被拒绝；
- episode_number / target_duration / duration 的数值范围；
- NaN / Infinity 显式拒绝；
- shot_ids / character_asset_ids 的数量上限与 ID 格式；
- 非法输入不写数据库、不启动后台任务。
"""

from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient  # noqa: E402

from config import settings  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from main import app  # noqa: E402
from models import BackgroundJob, Character, Project, SceneAsset, Shot  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


def _post_json(client: TestClient, url: str, body: str):
    """发送原始 JSON 文本，便于构造 NaN / Infinity 这类非标准字面量。"""

    return client.post(url, content=body, headers={"content-type": "application/json"})


def _put_json(client: TestClient, url: str, body: str):
    return client.put(url, content=body, headers={"content-type": "application/json"})


class ValidationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()
        cls.client = TestClient(app)

    def setUp(self) -> None:
        self.db = SessionLocal()
        self.db.query(BackgroundJob).delete()
        self.db.query(Shot).delete()
        self.db.query(Character).delete()
        self.db.query(SceneAsset).delete()
        self.db.query(Project).delete()
        self.db.commit()

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()

    def counts(self) -> tuple[int, int, int]:
        return (
            self.db.query(Project).count(),
            self.db.query(Shot).count(),
            self.db.query(BackgroundJob).count(),
        )

    def create_project(self, **overrides) -> str:
        payload = {"title": "校验项目"}
        payload.update(overrides)
        response = self.client.post("/api/project", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["id"]


class ProjectDtoValidationTests(ValidationTestCase):
    def test_unknown_project_type_is_rejected(self) -> None:
        before = self.counts()
        response = self.client.post("/api/project", json={"title": "x", "project_type": "season"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.counts(), before)

    def test_unknown_input_type_is_rejected(self) -> None:
        response = self.client.post("/api/project", json={"title": "x", "input_type": "pdf"})
        self.assertEqual(response.status_code, 422)

    def test_unknown_output_format_and_resolution_are_rejected(self) -> None:
        for field, value in (
            ("output_format", "21:9"),
            ("output_format", "9-16"),
            ("resolution", "8k"),
            ("resolution", "1080"),
        ):
            with self.subTest(field=field, value=value):
                response = self.client.post("/api/project", json={"title": "x", field: value})
                self.assertEqual(response.status_code, 422)

    def test_supported_output_formats_and_resolutions_are_accepted(self) -> None:
        for value in ("9:16", "16:9", "1:1", "4:3", "3:4"):
            with self.subTest(output_format=value):
                response = self.client.post("/api/project", json={"title": "x", "output_format": value})
                self.assertEqual(response.status_code, 200, response.text)
        for value in ("720p", "1080p", "2k", "4k"):
            with self.subTest(resolution=value):
                response = self.client.post("/api/project", json={"title": "x", "resolution": value})
                self.assertEqual(response.status_code, 200, response.text)

    def test_unknown_platform_is_rejected(self) -> None:
        response = self.client.post("/api/project", json={"title": "x", "platform": "tiktok"})
        self.assertEqual(response.status_code, 422)

    def test_blank_and_overlong_title_are_rejected(self) -> None:
        before = self.counts()
        self.assertEqual(self.client.post("/api/project", json={"title": ""}).status_code, 422)
        self.assertEqual(self.client.post("/api/project", json={"title": "   "}).status_code, 422)
        long_title = "标" * (settings.MAX_PROJECT_TITLE_CHARS + 1)
        self.assertEqual(self.client.post("/api/project", json={"title": long_title}).status_code, 422)
        self.assertEqual(self.counts(), before)

    def test_overlong_genre_is_rejected(self) -> None:
        genre = "类" * (settings.MAX_PROJECT_GENRE_CHARS + 1)
        self.assertEqual(self.client.post("/api/project", json={"title": "x", "genre": genre}).status_code, 422)

    def test_negative_and_huge_episode_number_are_rejected(self) -> None:
        for value in (-1, -100, settings.MAX_EPISODE_NUMBER + 1):
            with self.subTest(episode_number=value):
                response = self.client.post("/api/project", json={"title": "x", "episode_number": value})
                self.assertEqual(response.status_code, 422)

    def test_style_and_parent_identifier_formats_are_validated(self) -> None:
        self.assertEqual(
            self.client.post("/api/project", json={"title": "x", "style": "../etc/passwd"}).status_code, 422
        )
        self.assertEqual(self.client.post("/api/project", json={"title": "x", "style": ""}).status_code, 422)
        self.assertEqual(
            self.client.post("/api/project", json={"title": "x", "parent_project_id": "../outside"}).status_code, 422
        )

    def test_update_rejects_invalid_values_without_writing(self) -> None:
        project_id = self.create_project()
        for payload in (
            {"output_format": "21:9"},
            {"resolution": "8k"},
            {"platform": "tiktok"},
            {"project_type": "season"},
            {"episode_number": -1},
            {"title": "   "},
        ):
            with self.subTest(payload=payload):
                response = self.client.put(f"/api/project/{project_id}", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
        self.db.expire_all()
        project = self.db.get(Project, project_id)
        self.assertEqual(project.output_format, "9:16")
        self.assertEqual(project.resolution, "1080p")
        self.assertEqual(project.platform, "douyin")
        self.assertEqual(project.project_type, "series")

    def test_valid_project_creation_still_works(self) -> None:
        project_id = self.create_project(title=" 正常项目 ", genre="甜宠", style="anime")
        self.db.expire_all()
        project = self.db.get(Project, project_id)
        self.assertEqual(project.title, "正常项目")
        self.assertEqual(project.project_type, "series")
        episodes = self.db.query(Project).filter(Project.parent_project_id == project_id).all()
        self.assertEqual(len(episodes), 1)


class ScriptDtoValidationTests(ValidationTestCase):
    def test_unknown_mode_is_rejected(self) -> None:
        project_id = self.create_project()
        before = self.counts()
        response = self.client.post(
            "/api/script/parse",
            json={"project_id": project_id, "user_input": "剧本内容", "mode": "turbo"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.counts(), before)

    def test_out_of_range_target_duration_is_rejected(self) -> None:
        project_id = self.create_project()
        for value in (0, -30, settings.MAX_TARGET_DURATION_SECONDS + 1):
            with self.subTest(target_duration=value):
                response = self.client.post(
                    "/api/script/parse",
                    json={"project_id": project_id, "user_input": "剧本内容", "target_duration": value},
                )
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.counts()[2], 0)

    def test_infinite_target_duration_is_rejected(self) -> None:
        project_id = self.create_project()
        body = json.dumps({"project_id": project_id, "user_input": "剧本", "target_duration": 45}).replace(
            '"target_duration": 45', '"target_duration": Infinity'
        )
        response = _post_json(self.client, "/api/script/parse", body)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.counts()[2], 0)

    def test_oversized_script_is_rejected_with_413(self) -> None:
        project_id = self.create_project()
        original = settings.MAX_SCRIPT_TEXT_CHARS
        settings.MAX_SCRIPT_TEXT_CHARS = 32
        try:
            response = self.client.post(
                "/api/script/parse",
                json={"project_id": project_id, "user_input": "剧" * 64},
            )
            self.assertEqual(response.status_code, 413)
        finally:
            settings.MAX_SCRIPT_TEXT_CHARS = original
        self.assertEqual(self.counts()[2], 0)

    def test_invalid_project_id_is_rejected(self) -> None:
        response = self.client.post(
            "/api/script/parse",
            json={"project_id": "../../etc", "user_input": "剧本内容"},
        )
        self.assertEqual(response.status_code, 422)

    def test_generate_request_validates_prompt_and_duration(self) -> None:
        too_long = "p" * (settings.MAX_GENERATION_PROMPT_CHARS + 1)
        self.assertEqual(self.client.post("/api/script/generate", json={"prompt": too_long}).status_code, 422)
        self.assertEqual(self.client.post("/api/script/generate", json={"prompt": "   "}).status_code, 422)
        self.assertEqual(
            self.client.post("/api/script/generate", json={"prompt": "ok", "target_duration": 0}).status_code, 422
        )


class ShotDtoValidationTests(ValidationTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.project_id = self.create_project()
        self.shot = Shot(id="validation-shot", project_id=self.project_id, sequence=1, version=1)
        self.db.add(self.shot)
        self.db.commit()

    def test_duration_range_is_enforced(self) -> None:
        for value in (0.0, 0.1, -3, settings.MAX_SHOT_DURATION_SECONDS + 1):
            with self.subTest(duration=value):
                response = self.client.put(f"/api/shot/{self.shot.id}", json={"duration": value})
                self.assertEqual(response.status_code, 422, response.text)
        self.db.expire_all()
        self.assertEqual(self.db.get(Shot, self.shot.id).duration, 3.0)

    def test_infinite_and_nan_duration_are_rejected(self) -> None:
        for literal in ("Infinity", "-Infinity", "NaN"):
            with self.subTest(literal=literal):
                response = _put_json(self.client, f"/api/shot/{self.shot.id}", '{"duration": ' + literal + "}")
                self.assertEqual(response.status_code, 422, response.text)
        self.db.expire_all()
        self.assertEqual(self.db.get(Shot, self.shot.id).duration, 3.0)

    def test_invalid_enum_values_are_rejected(self) -> None:
        for payload in (
            {"emotion": "furious"},
            {"shot_type": "panorama"},
            {"camera_angle": "背后"},
            {"camera_movement": "瞬移"},
            {"transition": "explode"},
            {"audio_mode": "loud"},
        ):
            with self.subTest(payload=payload):
                response = self.client.put(f"/api/shot/{self.shot.id}", json=payload)
                self.assertEqual(response.status_code, 422, response.text)

    def test_supported_enum_values_are_accepted(self) -> None:
        for payload in (
            {"emotion": "angry"},
            {"shot_type": "extreme_close"},
            {"camera_angle": "俯视"},
            {"camera_movement": "环绕"},
            {"transition": "white_flash"},
            {"audio_mode": ""},
            {"audio_mode": "native"},
        ):
            with self.subTest(payload=payload):
                response = self.client.put(f"/api/shot/{self.shot.id}", json=payload)
                self.assertEqual(response.status_code, 200, response.text)

    def test_overlong_shot_text_is_rejected(self) -> None:
        response = self.client.put(
            f"/api/shot/{self.shot.id}",
            json={"scene_description": "景" * (settings.MAX_SHOT_TEXT_CHARS + 1)},
        )
        self.assertEqual(response.status_code, 422)
        response = self.client.put(
            f"/api/shot/{self.shot.id}",
            json={"visual_notes": "注" * (settings.MAX_VISUAL_NOTES_CHARS + 1)},
        )
        self.assertEqual(response.status_code, 422)

    def test_asset_id_list_is_bounded_and_validated(self) -> None:
        too_many = [f"char-{index}" for index in range(settings.MAX_CHARACTER_ASSET_IDS + 1)]
        response = self.client.put(f"/api/shot/{self.shot.id}", json={"character_asset_ids": too_many})
        self.assertEqual(response.status_code, 422)
        response = self.client.put(f"/api/shot/{self.shot.id}", json={"character_asset_ids": ["../etc/passwd"]})
        self.assertEqual(response.status_code, 422)
        response = self.client.put(f"/api/shot/{self.shot.id}", json={"scene_asset_id": "../outside"})
        self.assertEqual(response.status_code, 422)

    def test_batch_regenerate_bounds_shot_ids(self) -> None:
        too_many = [f"shot-{index}" for index in range(settings.MAX_BATCH_SHOT_IDS + 1)]
        response = self.client.post("/api/shot/batch-regenerate", json=too_many)
        self.assertEqual(response.status_code, 422)
        response = self.client.post("/api/shot/batch-regenerate", json=["../escape"])
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.counts()[2], 0)

    def test_generate_storyboard_validates_shot_ids(self) -> None:
        response = self.client.post(
            f"/api/shot/{self.project_id}/generate-storyboard",
            json={"shot_ids": ["../escape"]},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.counts()[2], 0)


class RenderDtoValidationTests(ValidationTestCase):
    def test_render_rejects_unknown_format_and_resolution(self) -> None:
        project_id = self.create_project()
        before = self.counts()
        self.assertEqual(
            self.client.post("/api/render", json={"project_id": project_id, "output_format": "21:9"}).status_code,
            422,
        )
        self.assertEqual(
            self.client.post("/api/render", json={"project_id": project_id, "resolution": "16k"}).status_code,
            422,
        )
        self.assertEqual(self.counts(), before)

    def test_render_rejects_invalid_project_id_with_400(self) -> None:
        response = self.client.post("/api/render", json={"project_id": "../../etc"})
        self.assertEqual(response.status_code, 400)


class AssetDtoValidationTests(ValidationTestCase):
    """资产 ID 之外的字段不做 identifier 校验，避免误伤中文音色/场景键。"""

    def setUp(self) -> None:
        super().setUp()
        self.project_id = self.create_project()
        self.character = Character(id="validation-character", project_id=self.project_id, name="林夏")
        self.scene = SceneAsset(id="validation-scene", project_id=self.project_id, name="教室")
        self.db.add_all([self.character, self.scene])
        self.db.commit()

    def test_character_asset_accepts_chinese_voice_and_bounded_text(self) -> None:
        response = self.client.put(
            f"/api/asset/character/{self.character.id}",
            json={
                "project_id": self.project_id,
                "voice_id": "冰糖",
                "lora_profile": "lora-a",
                "ip_adapter_profile": "ip-a",
                "seed": "42",
                "default_outfit": "校服",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.db.expire_all()
        stored = self.db.get(Character, self.character.id)
        self.assertEqual(stored.voice_id, "冰糖")
        self.assertEqual(stored.lora_profile, "lora-a")
        self.assertEqual(stored.seed, "42")

    def test_character_asset_rejects_blank_name_and_overlong_prompt(self) -> None:
        self.assertEqual(
            self.client.put(
                f"/api/asset/character/{self.character.id}",
                json={"project_id": self.project_id, "name": "   "},
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.put(
                f"/api/asset/character/{self.character.id}",
                json={"project_id": self.project_id, "visual_prompt": "p" * (settings.MAX_SHOT_TEXT_CHARS + 1)},
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.put(
                f"/api/asset/character/{self.character.id}",
                json={"project_id": self.project_id, "key_features": ["f"] * 50},
            ).status_code,
            422,
        )

    def test_scene_asset_accepts_chinese_group_key_and_time_of_day(self) -> None:
        response = self.client.put(
            f"/api/asset/scene/{self.scene.id}",
            json={
                "project_id": self.project_id,
                "scene_group_key": "教室-morning",
                "time_of_day": "清晨",
                "prop_lock": "窗边课桌",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.db.expire_all()
        stored = self.db.get(SceneAsset, self.scene.id)
        self.assertEqual(stored.scene_group_key, "教室-morning")
        self.assertEqual(stored.time_of_day, "清晨")

    def test_scene_and_character_ids_still_use_identifier_rules(self) -> None:
        response = self.client.put(
            f"/api/asset/character/{self.character.id}",
            json={"project_id": "../escape", "name": "坏"},
        )
        self.assertEqual(response.status_code, 422)
        response = self.client.put(
            "/api/asset/shot/../escape",
            json={"project_id": self.project_id},
        )
        self.assertNotEqual(response.status_code, 200)


class BatchRegenerateDtoTests(ValidationTestCase):
    def test_valid_body_is_parsed_as_a_shot_id_list(self) -> None:
        # 合法数组能进入业务逻辑（镜头不存在 -> 404），证明 body 解析未被破坏。
        response = self.client.post("/api/shot/batch-regenerate", json=["missing-shot"])
        self.assertEqual(response.status_code, 404, response.text)

    def test_reason_query_parameter_is_still_accepted(self) -> None:
        response = self.client.post("/api/shot/batch-regenerate?reason=retry", json=["missing-shot"])
        self.assertEqual(response.status_code, 404, response.text)


if __name__ == "__main__":
    unittest.main()
