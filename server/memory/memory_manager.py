from .project_memory import ProjectMemory
from .user_memory import UserMemory


class MemoryManager:
    """记忆系统统一管理器"""

    def __init__(self):
        self.project = ProjectMemory()
        self.user = UserMemory()

    def get_generation_context(self, project_id: str, user_id: str = "default") -> dict:
        """获取完整的生成上下文（合并项目记忆 + 用户偏好）"""
        characters = self.project.load_characters(project_id)
        narrative = self.project.load_narrative_context(project_id)
        style_params = self.project.load_style_params(project_id)
        user_prefs = self.user.get_preferences(user_id)

        return {
            "characters": characters,
            "narrative_context": narrative,
            "style_params": style_params,
            "user_preferences": user_prefs,
        }

    def learn_from_regeneration(self, project_id: str, shot_id: str, reason: str, user_id: str = "default"):
        """从用户重生成操作中学习偏好"""
        # 记录到项目日志
        self.project.save_generation_log(
            project_id,
            {
                "shot_id": shot_id,
                "action": "regenerate",
                "reason": reason,
            },
        )

        # 更新用户偏好
        self.user.record_correction(
            user_id,
            {
                "project_id": project_id,
                "shot_id": shot_id,
                "reason": reason,
            },
        )
