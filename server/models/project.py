from datetime import datetime

from sqlalchemy import CheckConstraint, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import relationship

from .base import Base


class Project(Base):
    """项目 / 剧集。

    ``project_type`` 只有两种取值：``series``（大项目，无父级）与 ``episode``
    （剧集，必须挂在 series 下）。这两个不变量同时由下面的 CHECK 约束、
    ``db/database.py`` 中的 SQLite 触发器以及 API 层校验保证，避免出现孤儿剧集
    或 series 挂在 episode 之下。

    说明：这里没有给 ``parent_project_id`` 加自引用外键。SQLite 无法在既有库上
    增加外键而不重建整张表，而重建会牵动 shots/characters/scene_assets 的外键
    定义；因此对存量库改用等价的触发器约束（见 database.py）。
    """

    __tablename__ = "projects"
    __table_args__ = (
        CheckConstraint("project_type IN ('series', 'episode')", name="ck_projects_type"),
        CheckConstraint("episode_number IS NULL OR episode_number >= 0", name="ck_projects_episode_number"),
        CheckConstraint(
            "(project_type = 'episode' AND parent_project_id IS NOT NULL AND parent_project_id <> '')"
            " OR (project_type = 'series' AND (parent_project_id IS NULL OR parent_project_id = ''))",
            name="ck_projects_parent_shape",
        ),
        CheckConstraint(
            "parent_project_id IS NULL OR parent_project_id = '' OR parent_project_id <> id",
            name="ck_projects_no_self_parent",
        ),
    )

    id = Column(String, primary_key=True)
    parent_project_id = Column(String, default="", index=True)
    project_type = Column(String, default="series")
    episode_number = Column(Integer, default=0)
    title = Column(String, nullable=False, default="未命名项目")
    genre = Column(String, default="")
    style = Column(String, default="anime")
    status = Column(String, default="draft")
    input_text = Column(Text, default="")
    input_type = Column(String, default="text")
    output_format = Column(String, default="9:16")
    resolution = Column(String, default="1080p")
    platform = Column(String, default="douyin")
    # 0 表示历史项目未显式设置；新项目由创建/解析请求写入，渲染时不得静默猜回默认值。
    target_duration = Column(Integer, default=0)
    timing_plan = Column(Text, default="{}")
    consistency_config = Column(Text, default="{}")
    # 当前项目一致性参考汇总：含每项状态、失败原因、错误编号和影响镜头范围。
    consistency_report = Column(Text, default="{}")
    # 字幕/音频配置版本：工作台每次修改轨道或字幕条目时 +1；渲染任务记录
    # 渲染时的值，发布前比对不一致即丢弃成片，避免旧配置覆盖新修改。
    av_config_version = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    shots = relationship("Shot", back_populates="project", cascade="all, delete-orphan")
    characters = relationship("Character", back_populates="project", cascade="all, delete-orphan")
    scenes = relationship("SceneAsset", back_populates="project", cascade="all, delete-orphan")
