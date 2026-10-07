"""根据剧本自动为项目命名的验收测试。

覆盖：

- `_persist_phase1` 优先使用 LLM 解析出的剧名（state["script_title"]）回填项目标题，
  并把最终标题返回给调用方（随 WebSocket complete 事件下发给前端）；
- LLM 未给出剧名时回退到剧本文本中的「标题：」行 / 首行；
- `_script_title` 的截断与空文本兜底。
"""

from __future__ import annotations

import unittest
import uuid

from api.routes.script import _persist_phase1, _script_title  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import Project  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


def _state(project_id: str, *, script_title: str = "", user_input: str = "第一场：小雨捡到一本旧日记。") -> dict:
    return {
        "project_id": project_id,
        "user_input": user_input,
        "input_type": "text",
        "script_title": script_title,
        "genre": "悬疑",
        "style": "anime",
        "style_params": {},
        "characters": [],
        "script_scenes": [],
        "shots": [],
        "output_format": "9:16",
        "resolution": "1080p",
        "platform": "douyin",
    }


class AutoProjectTitleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    def setUp(self) -> None:
        self.db = SessionLocal()
        self.project_id = f"proj-{uuid.uuid4().hex[:10]}"
        self.db.add(Project(id=self.project_id, title="未命名项目"))
        self.db.commit()

    def tearDown(self) -> None:
        try:
            self.db.query(Project).filter(Project.id == self.project_id).delete()
            self.db.commit()
        finally:
            self.db.close()

    def test_llm_title_renames_project_and_is_returned(self) -> None:
        title = _persist_phase1(self.db, self.project_id, _state(self.project_id, script_title="雨夜电车"))
        self.assertEqual(title, "雨夜电车")
        project = self.db.get(Project, self.project_id)
        self.assertEqual(project.title, "雨夜电车", "剧本解析出的剧名必须回填到项目标题")

    def test_falls_back_to_script_text_title_without_llm_title(self) -> None:
        user_input = "标题：日记里的秘密\n第一场：小雨翻开日记。"
        title = _persist_phase1(self.db, self.project_id, _state(self.project_id, user_input=user_input))
        self.assertEqual(title, "日记里的秘密")
        self.assertEqual(self.db.get(Project, self.project_id).title, "日记里的秘密")

    def test_script_title_parses_title_line_and_falls_back_to_first_line(self) -> None:
        self.assertEqual(_script_title("标题：深海来信\n场景一"), "深海来信")
        self.assertEqual(_script_title("场景一：开场"), "场景一：开场")
        self.assertEqual(_script_title(""), "未命名项目")


if __name__ == "__main__":
    unittest.main()
