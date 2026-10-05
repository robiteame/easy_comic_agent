import json
from pathlib import Path

from config import settings


class ProjectMemory:
    """项目记忆管理 - 持久化项目级别的上下文"""

    def __init__(self):
        self.base_dir = settings.OUTPUT_DIR / "projects"

    def _get_project_dir(self, project_id: str) -> Path:
        return self.base_dir / project_id

    def _get_memory_dir(self, project_id: str) -> Path:
        memory_dir = self._get_project_dir(project_id) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        return memory_dir

    def save_characters(self, project_id: str, characters: list[dict]):
        """保存角色卡片"""
        memory_dir = self._get_memory_dir(project_id)
        with open(memory_dir / "characters.json", "w", encoding="utf-8") as f:
            json.dump(characters, f, ensure_ascii=False, indent=2)

    def load_characters(self, project_id: str) -> list[dict]:
        """加载角色卡片"""
        path = self._get_memory_dir(project_id) / "characters.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_narrative_context(self, project_id: str, context: dict):
        """保存剧情上下文"""
        memory_dir = self._get_memory_dir(project_id)
        with open(memory_dir / "narrative_context.json", "w", encoding="utf-8") as f:
            json.dump(context, f, ensure_ascii=False, indent=2)

    def load_narrative_context(self, project_id: str) -> dict:
        """加载剧情上下文"""
        path = self._get_memory_dir(project_id) / "narrative_context.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_style_params(self, project_id: str, style_params: dict):
        """保存风格参数锁定"""
        memory_dir = self._get_memory_dir(project_id)
        with open(memory_dir / "style_params.json", "w", encoding="utf-8") as f:
            json.dump(style_params, f, ensure_ascii=False, indent=2)

    def load_style_params(self, project_id: str) -> dict:
        """加载风格参数"""
        path = self._get_memory_dir(project_id) / "style_params.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_generation_log(self, project_id: str, log_entry: dict):
        """保存生成日志（用于学习用户偏好）"""
        memory_dir = self._get_memory_dir(project_id)
        log_path = memory_dir / "generation_log.json"

        logs = []
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)

        logs.append(log_entry)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

    def load_generation_log(self, project_id: str) -> list[dict]:
        """加载生成日志"""
        path = self._get_memory_dir(project_id) / "generation_log.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
