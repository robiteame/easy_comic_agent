import json

from config import settings


class UserMemory:
    """用户偏好记忆 - 跨项目的长期记忆"""

    def __init__(self):
        self.memory_path = settings.DATA_DIR / "user_preferences.json"
        self._ensure_file()

    def _ensure_file(self):
        if not self.memory_path.exists():
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load(self) -> dict:
        with open(self.memory_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict):
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_preferences(self, user_id: str = "default") -> dict:
        """获取用户偏好"""
        data = self._load()
        return data.get(
            user_id,
            {
                "preferred_style": "anime",
                "preferred_shot_types": {"close-up": 0.3, "medium": 0.5, "wide": 0.2},
                "preferred_transition": "fade",
                "emotion_intensity": "moderate",
                "preferred_resolution": "1080p",
                "preferred_format": "9:16",
                "common_corrections": [],
            },
        )

    def update_preferences(self, user_id: str = "default", updates: dict = None):
        """更新用户偏好"""
        data = self._load()
        if user_id not in data:
            data[user_id] = self.get_preferences(user_id)
        if updates:
            data[user_id].update(updates)
        self._save(data)

    def record_correction(self, user_id: str, correction: dict):
        """记录用户修正（用于学习偏好）"""
        data = self._load()
        if user_id not in data:
            data[user_id] = self.get_preferences(user_id)

        corrections = data[user_id].get("common_corrections", [])
        corrections.append(correction)
        # 只保留最近 50 条
        data[user_id]["common_corrections"] = corrections[-50:]
        self._save(data)
