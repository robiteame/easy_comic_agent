"""镜头结构化对白（speaker 保留）的回归测试。

验收标准对应的回归点：
- 双角色轮流对话必须使用各自声音：逐句 TTS 的 voice_id 由该句 speaker 决定，
  说话人缺失/未登记时给出可追踪警告并使用端点默认音色，绝不静默回落到
  第一个角色；
- 同一镜头多句对白必须按时间顺序生成：逐句音频按顺序拼接，时间轴单调递增，
  字幕逐句按时间顺序错开；
- 旧项目的纯文本 dialogue 读取时自动迁移为单条对白；
- 版本快照 / 恢复完整保留说话人与逐句时间轴。
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from test_environment import TEST_ROOT  # noqa: F401,E402

from agent.output_schemas import parse_storyboard_output  # noqa: E402
from models.shot import Shot  # noqa: E402
from services import dialogue_audio  # noqa: E402
from services.shot_dialogue import (  # noqa: E402
    DialogueLine,
    assign_line_timings,
    dialogue_display_text,
    dialogue_total_chars,
    parse_shot_dialogue,
    resolve_speaker_voice,
    serialize_dialogue_lines,
    warn_unknown_speakers,
)
from services.shot_version_service import apply_snapshot_to_shot, capture_snapshot  # noqa: E402
from services.subtitle_service import DialogueLineInput, ShotDialogueInput, cues_from_shots  # noqa: E402

CHARACTERS = [
    {"name": "林夏", "voice_id": "voice_linxia"},
    {"name": "顾言", "voice_id": "voice_guyan"},
]


class ParseAndMigrateTests(unittest.TestCase):
    def test_legacy_string_migrates_to_single_line_with_warning(self) -> None:
        with self.assertLogs("services.shot_dialogue", level="WARNING") as logs:
            lines = parse_shot_dialogue("我看见故事成形了", fallback_speaker="林夏", warn_key="shot s1")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].speaker, "林夏")
        self.assertEqual(lines[0].line, "我看见故事成形了")
        self.assertTrue(any("旧版纯文本" in message for message in logs.output), logs.output)

    def test_structured_json_roundtrip_preserves_speaker(self) -> None:
        lines = [
            DialogueLine(speaker="林夏", line="早上好", emotion="happy", action="挥手", start_ms=0, end_ms=1200),
            DialogueLine(speaker="顾言", line="你也好", emotion="neutral", start_ms=1200, end_ms=2000),
        ]
        stored = serialize_dialogue_lines(lines)
        # 数据库里是 JSON 数组文本，说话人与时间轴完整可还原。
        restored = parse_shot_dialogue(stored)
        self.assertEqual([line.speaker for line in restored], ["林夏", "顾言"])
        self.assertEqual([line.line for line in restored], ["早上好", "你也好"])
        self.assertEqual([(line.start_ms, line.end_ms) for line in restored], [(0, 1200), (1200, 2000)])
        self.assertEqual(restored[0].action, "挥手")

    def test_speaker_aliases_and_empty_dialogue(self) -> None:
        lines = parse_shot_dialogue(
            [
                {"character": "林夏", "text": "你好"},
                {"role": "顾言", "line": "你好呀"},
                {"line": "   "},
            ]
        )
        self.assertEqual([line.speaker for line in lines], ["林夏", "顾言"])
        self.assertEqual(parse_shot_dialogue(""), [])
        self.assertEqual(parse_shot_dialogue([]), [])
        self.assertEqual(serialize_dialogue_lines([]), "")

    def test_display_and_cost_helpers(self) -> None:
        lines = parse_shot_dialogue(
            serialize_dialogue_lines(
                [DialogueLine(speaker="林夏", line="早上好"), DialogueLine(speaker="顾言", line="你也好")]
            )
        )
        self.assertEqual(dialogue_display_text(lines), "林夏：早上好\n顾言：你也好")
        self.assertEqual(dialogue_total_chars(lines), len("早上好") + len("你也好"))


class SpeakerVoiceResolutionTests(unittest.TestCase):
    def test_each_speaker_gets_own_voice(self) -> None:
        self.assertEqual(resolve_speaker_voice("林夏", CHARACTERS), "voice_linxia")
        self.assertEqual(resolve_speaker_voice("顾言", CHARACTERS), "voice_guyan")

    def test_unknown_speaker_warns_and_never_uses_first_character(self) -> None:
        with self.assertLogs("services.shot_dialogue", level="WARNING") as logs:
            self.assertEqual(resolve_speaker_voice("神秘人", CHARACTERS, context="shot s1"), "")
            self.assertEqual(resolve_speaker_voice("", CHARACTERS), "")
        self.assertTrue(any("神秘人" in message for message in logs.output), logs.output)
        # 第二个角色的台词绝不能因为 speaker 缺失而落到第一个角色头上。
        self.assertNotIn("voice_linxia", "\n".join(logs.output))

    def test_warn_unknown_speakers_reports_missing_names(self) -> None:
        lines = [DialogueLine(speaker="林夏", line="a"), DialogueLine(speaker="旁白", line="b")]
        with self.assertLogs("services.shot_dialogue", level="WARNING") as logs:
            unknown = warn_unknown_speakers(lines, CHARACTERS, context="storyboard shot=1")
        self.assertEqual(unknown, ["旁白"])
        self.assertTrue(any("旁白" in message for message in logs.output))


class LineTimingTests(unittest.TestCase):
    def test_multiple_lines_are_timed_in_speaking_order(self) -> None:
        timings = assign_line_timings([1200, 800, 2000])
        self.assertEqual(timings, [(0, 1200), (1200, 2000), (2000, 4000)])
        starts = [start for start, _ in timings]
        self.assertEqual(starts, sorted(starts), "多句对白必须按时间顺序单调排列")


class _FakeTTS:
    """记录逐句调用参数并落盘假音频，模拟真实 TTSService 行为。"""

    def __init__(self, root: Path):
        self.root = root
        self.calls: list[dict] = []
        self.output_dir = root  # 与 TTSService.output_dir 同构（拼接目标目录）

    async def generate_dialogue(self, *, text: str, voice_id: str = "", emotion: str = "neutral", project_id: str = "", shot_id: str = "") -> str:
        self.calls.append({"text": text, "voice_id": voice_id, "emotion": emotion, "shot_id": shot_id})
        path = self.root / f"{shot_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF" + b"\x00" * 2048)
        return str(path)


class _FakeFFmpeg:
    def __init__(self, root: Path):
        self.root = root
        self.concat_orders: list[list[str]] = []

    async def probe_duration_ms(self, path) -> int:
        # 按文件名尾部区分句序，返回递增时长。
        name = Path(path).stem
        if name.endswith("_d1"):
            return 1200
        if name.endswith("_d2"):
            return 800
        return 3000

    async def concat_audio_clips(self, clip_paths, output_path) -> None:
        self.concat_orders.append([str(path) for path in clip_paths])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF" + b"\x00" * 4096)


class _FakeStorage:
    def ensure_project_capacity(self, **_kwargs) -> None:
        return None


class DialogueTrackRegressionTests(unittest.TestCase):
    """验收回归 1：双角色轮流对话必须使用各自声音；
    验收回归 2：同一镜头多句对白必须按时间顺序生成。"""

    def setUp(self) -> None:
        self.root = Path(TEST_ROOT) / "dialogue-track"
        self.root.mkdir(parents=True, exist_ok=True)
        self.fake_tts = _FakeTTS(self.root)
        self.fake_ffmpeg = _FakeFFmpeg(self.root)

    def _run(self, lines):
        with (
            patch.object(dialogue_audio, "tts_service", self.fake_tts),
            patch.object(dialogue_audio, "ffmpeg_service", self.fake_ffmpeg),
            patch.object(dialogue_audio, "storage_service", _FakeStorage()),
        ):
            return asyncio.run(
                dialogue_audio.generate_dialogue_track(
                    lines,
                    characters=CHARACTERS,
                    project_id="proj_1",
                    media_id="shot_0001_v2",
                    default_emotion="neutral",
                )
            )

    def test_two_speakers_alternating_use_their_own_voices_in_order(self) -> None:
        lines = [
            DialogueLine(speaker="林夏", line="你来了。", emotion="happy"),
            DialogueLine(speaker="顾言", line="嗯，久等了。", emotion="shy"),
            DialogueLine(speaker="林夏", line="我们走吧。", emotion="happy"),
        ]
        audio_path, timed = self._run(lines)
        # 每句对白按各自 speaker 选择音色，顺序与台词顺序一致。
        self.assertEqual(
            [(call["voice_id"], call["text"]) for call in self.fake_tts.calls],
            [
                ("voice_linxia", "你来了。"),
                ("voice_guyan", "嗯，久等了。"),
                ("voice_linxia", "我们走吧。"),
            ],
        )
        # 第二角色的台词绝不会被交给第一个角色的音色。
        guyan_line = next(call for call in self.fake_tts.calls if call["text"] == "嗯，久等了。")
        self.assertEqual(guyan_line["voice_id"], "voice_guyan")
        self.assertNotEqual(guyan_line["voice_id"], "voice_linxia")
        # 逐句 emotion 一并下发。
        self.assertEqual([call["emotion"] for call in self.fake_tts.calls], ["happy", "shy", "happy"])
        # 多句音频按时间顺序拼接。
        self.assertEqual(len(self.fake_ffmpeg.concat_orders), 1)
        self.assertEqual(
            [Path(path).stem for path in self.fake_ffmpeg.concat_orders[0]],
            ["shot_0001_v2_d1", "shot_0001_v2_d2", "shot_0001_v2_d3"],
        )
        self.assertTrue(audio_path.endswith("shot_0001_v2.wav"))
        # 时间轴按实测时长顺序回填（第三句 fallback 3000ms）。
        self.assertEqual(
            [(line.start_ms, line.end_ms) for line in timed],
            [(0, 1200), (1200, 2000), (2000, 5000)],
        )

    def test_unknown_speaker_uses_default_voice_with_trackable_warning(self) -> None:
        lines = [DialogueLine(speaker="神秘人", line="是谁？")]
        with self.assertLogs("services.shot_dialogue", level="WARNING") as logs:
            self._run(lines)
        self.assertEqual(self.fake_tts.calls[0]["voice_id"], "", "未登记说话人使用端点默认音色")
        self.assertTrue(any("神秘人" in message and "voice" not in message.split("speaker=")[0] for message in logs.output), logs.output)

    def test_single_line_keeps_direct_output(self) -> None:
        lines = [DialogueLine(speaker="林夏", line="只有一句。")]
        audio_path, timed = self._run(lines)
        self.assertEqual(len(self.fake_tts.calls), 1)
        self.assertEqual(self.fake_ffmpeg.concat_orders, [], "单句无需拼接")
        self.assertTrue(audio_path.endswith("shot_0001_v2.wav"))
        self.assertEqual((timed[0].start_ms, timed[0].end_ms), (0, 3000))


class SubtitleFromLinesTests(unittest.TestCase):
    def test_lines_with_measured_timeline_produce_ordered_cues(self) -> None:
        shot = ShotDialogueInput(
            shot_id="s1",
            sequence=1,
            start_ms=4000,
            duration_ms=4000,
            dialogue="",
            tts_duration_ms=2000,
            lines=[
                DialogueLineInput(speaker="林夏", line="你来了。", start_ms=0, end_ms=1200),
                DialogueLineInput(speaker="顾言", line="嗯，久等了。", start_ms=1200, end_ms=2000),
            ],
        )
        cues = cues_from_shots([shot])
        self.assertEqual([(cue.start_ms, cue.end_ms) for cue in cues], [(4000, 5200), (5200, 6000)])
        self.assertEqual([cue.character_name for cue in cues], ["林夏", "顾言"])
        self.assertEqual([cue.text for cue in cues], ["你来了。", "嗯，久等了。"])

    def test_lines_without_timeline_share_shot_budget_in_order(self) -> None:
        shot = ShotDialogueInput(
            shot_id="s2",
            sequence=1,
            start_ms=0,
            duration_ms=3000,
            dialogue="",
            lines=[
                DialogueLineInput(speaker="林夏", line="第一句"),
                DialogueLineInput(speaker="顾言", line="第二句"),
            ],
        )
        cues = cues_from_shots([shot])
        self.assertEqual([(cue.start_ms, cue.end_ms) for cue in cues], [(0, 1500), (1500, 3000)])
        self.assertEqual([cue.character_name for cue in cues], ["林夏", "顾言"])

    def test_legacy_single_dialogue_input_still_works(self) -> None:
        shot = ShotDialogueInput("s3", 1, 0, 3000, "旧口径台词", "小明", tts_duration_ms=1800)
        cues = cues_from_shots([shot])
        self.assertEqual(len(cues), 1)
        self.assertEqual((cues[0].start_ms, cues[0].end_ms), (0, 1800))
        self.assertEqual(cues[0].character_name, "小明")


class StoryboardSchemaTests(unittest.TestCase):
    def test_storyboard_dialogue_keeps_speaker(self) -> None:
        payload = {
            "shots": [
                {
                    "scene_number": 1,
                    "scene_description": "教室",
                    "characters_in_scene": ["林夏", "顾言"],
                    "dialogue": [
                        {"speaker": "林夏", "line": "早上好", "emotion": "happy", "action": "挥手", "start_ms": 0, "end_ms": 1200},
                        {"character": "顾言", "line": "你也好", "emotion": "平静"},
                    ],
                }
            ]
        }
        parsed = parse_storyboard_output(payload)
        lines = parsed.shots[0].dialogue
        self.assertEqual([line.speaker for line in lines], ["林夏", "顾言"])
        self.assertEqual(lines[0].action, "挥手")
        self.assertEqual((lines[0].start_ms, lines[0].end_ms), (0, 1200))
        self.assertEqual(lines[1].emotion, "neutral")

    def test_storyboard_legacy_string_dialogue_becomes_single_line(self) -> None:
        payload = {"shots": [{"dialogue": ["旧格式台词"], "scene_description": "x"}]}
        parsed = parse_storyboard_output(payload)
        self.assertEqual(len(parsed.shots[0].dialogue), 1)
        self.assertEqual(parsed.shots[0].dialogue[0].line, "旧格式台词")
        self.assertEqual(parsed.shots[0].dialogue[0].speaker, "")


class SnapshotRoundTripTests(unittest.TestCase):
    def _shot(self) -> Shot:
        return Shot(
            id="snap-shot",
            project_id="proj_1",
            sequence=1,
            version=1,
            emotion="happy",
            characters_in_scene=json.dumps(["林夏", "顾言"], ensure_ascii=False),
        )

    def test_snapshot_and_restore_preserve_speaker_and_timeline(self) -> None:
        shot = self._shot()
        shot.dialogue = serialize_dialogue_lines(
            [
                DialogueLine(speaker="顾言", line="你先说。", emotion="shy", start_ms=0, end_ms=900),
                DialogueLine(speaker="林夏", line="好。", start_ms=900, end_ms=1400),
            ]
        )
        snapshot = capture_snapshot(shot)
        # 快照里对白是结构化列表：说话人与时间轴完整。
        self.assertEqual([item["speaker"] for item in snapshot["dialogue"]], ["顾言", "林夏"])
        self.assertEqual(snapshot["dialogue"][0]["end_ms"], 900)

        restored = self._shot()
        restored.dialogue = "别人的旧台词"
        apply_snapshot_to_shot(restored, snapshot)
        lines = parse_shot_dialogue(restored.dialogue)
        self.assertEqual([line.speaker for line in lines], ["顾言", "林夏"])
        self.assertEqual([(line.start_ms, line.end_ms) for line in lines], [(0, 900), (900, 1400)])

    def test_legacy_snapshot_string_restores_as_single_line(self) -> None:
        snapshot = capture_snapshot(self._shot())
        snapshot["dialogue"] = "旧版纯文本台词"  # 旧版本快照的原始形态
        restored = self._shot()
        apply_snapshot_to_shot(restored, snapshot)
        lines = parse_shot_dialogue(restored.dialogue)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].line, "旧版纯文本台词")

    def test_shot_model_serializes_list_assignment(self) -> None:
        shot = self._shot()
        shot.dialogue = [DialogueLine(speaker="林夏", line="列表直接赋值")]
        self.assertEqual(parse_shot_dialogue(shot.dialogue)[0].speaker, "林夏")
        # 字符串赋值原样透传（旧数据兼容）。
        shot.dialogue = "直接字符串"
        self.assertEqual(shot.dialogue, "直接字符串")


if __name__ == "__main__":
    unittest.main()


class ShotApiDialogueTests(unittest.TestCase):
    """API 层回归：保存 / 编辑 / 读取链路说话人不丢失，旧字符串自动迁移。"""

    @classmethod
    def setUpClass(cls) -> None:
        from db import init_db
        from main import app
        from fastapi.testclient import TestClient

        init_db()
        cls.client = TestClient(app)

    def setUp(self) -> None:
        from db import SessionLocal
        from models import Project
        from services.shot_dialogue import _legacy_warned

        _legacy_warned.clear()
        self.db = SessionLocal()
        self.db.query(Shot).delete()
        self.db.query(Project).delete()
        self.db.commit()
        self.project = Project(id="dialogue-api-project", title="DialogueApi", status="assets_ready")
        # duration 取 5 秒：与当前视频 Provider 的固定时长档一致，避免与
        # 时长能力校验（另一条独立规则）相互干扰。
        self.shot = Shot(
            id="dialogue-api-shot",
            project_id=self.project.id,
            sequence=1,
            version=1,
            duration=5.0,
            characters_in_scene=json.dumps(["林夏", "顾言"], ensure_ascii=False),
        )
        self.db.add_all([self.project, self.shot])
        self.db.commit()

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()

    def _get_shot(self) -> dict:
        response = self.client.get(f"/api/shot/{self.project.id}/shots")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()[0]

    def test_save_and_read_structured_dialogue_preserves_speaker(self) -> None:
        payload = {
            "dialogue": [
                {"speaker": "顾言", "line": "你先说。", "emotion": "shy"},
                {"speaker": "林夏", "line": "好。", "emotion": "happy"},
            ]
        }
        response = self.client.put("/api/shot/dialogue-api-shot", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        data = self._get_shot()
        self.assertEqual([item["speaker"] for item in data["dialogue"]], ["顾言", "林夏"])
        self.assertEqual([item["line"] for item in data["dialogue"]], ["你先说。", "好。"])
        self.assertEqual(data["dialogue"][0]["emotion"], "shy")

    def test_legacy_string_dialogue_is_migrated_on_read(self) -> None:
        self.shot.dialogue = "一句旧版台词"
        self.db.commit()
        data = self._get_shot()
        self.assertEqual(len(data["dialogue"]), 1)
        self.assertEqual(data["dialogue"][0]["line"], "一句旧版台词")
        # 旧数据迁移的说话人如实标注为场内第一个角色（与旧配音行为一致）。
        self.assertEqual(data["dialogue"][0]["speaker"], "林夏")

    def test_legacy_string_payload_is_accepted_and_normalized(self) -> None:
        response = self.client.put("/api/shot/dialogue-api-shot", json={"dialogue": "旧客户端字符串"})
        self.assertEqual(response.status_code, 200, response.text)
        data = self._get_shot()
        self.assertEqual(data["dialogue"][0]["line"], "旧客户端字符串")


if __name__ == "__main__":
    unittest.main()
