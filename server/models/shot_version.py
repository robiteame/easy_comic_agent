from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, String, Text

from .base import Base


class ShotVersion(Base):
    """镜头历史的不可变快照。

    版本记录只追加：任何写路径都只 INSERT 新行，历史行不允许原地修改（SQLite
    触发器兜底，见 ``db/database.py::_ensure_shot_version_append_only``）。行的
    元数据描述「这份状态为什么被记录」，``snapshot`` 列保存当时镜头的完整字段
    快照（含媒体路径与生成 Prompt），恢复时按快照重建镜头并追加新行。
    """

    __tablename__ = "shot_versions"
    __table_args__ = (
        Index("ix_shot_versions_shot_number", "shot_id", "number"),
        Index("ix_shot_versions_project", "project_id"),
    )

    id = Column(String, primary_key=True)
    shot_id = Column(String, nullable=False)
    project_id = Column(String, nullable=False, default="")
    # 镜头时间线内的稠密序号（1,2,3...），供界面展示 v1/v2/v3。
    number = Column(Integer, nullable=False, default=1)
    # 快照时刻的 shot.version，用于与任务防覆盖机制对齐。
    version = Column(Integer, nullable=False, default=1)
    # manual_edit / regenerate / restore / import
    source = Column(String, nullable=False, default="manual_edit")
    # 触发变更的后台任务幂等键（手动编辑为空串）。
    task_id = Column(String, nullable=False, default="")
    # 链式父版本：追加该行之前的最新版本 ID，首条为空串。
    parent_version_id = Column(String, nullable=False, default="")
    # 快照内容的规范化哈希，用于追加去重与「当前是否等于历史版本」判断。
    content_hash = Column(String, nullable=False, default="")
    # 镜头字段快照 JSON（_serialize_shot 的字段集 + prompt/negative_prompt）。
    snapshot = Column(Text, nullable=False, default="{}")
    # 候选选择属于版本事实：选择结果和可解释 trace 同时写入不可变历史。
    candidate_selection = Column(Text, nullable=False, default="{}")
    decision_trace = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class ShotVideoCandidate(Base):
    """视频候选的独立持久化记录。

    候选生成不改写 ``Shot.video_path``：每个候选只保存自己的媒体路径和生成元数据，
    全部候选完成后才由选择器把选中项发布为正式视频。失败尝试与重试均追加新行，
    旧失败不会被成功结果覆盖。
    """

    __tablename__ = "shot_video_candidates"
    __table_args__ = (
        Index("ix_shot_video_candidates_shot_version", "shot_id", "shot_version"),
        Index("ix_shot_video_candidates_batch", "batch_id"),
    )

    candidate_id = Column(String, primary_key=True)
    shot_id = Column(String, nullable=False)
    project_id = Column(String, nullable=False, default="")
    shot_version = Column(Integer, nullable=False, default=1)
    batch_id = Column(String, nullable=False, default="", index=True)
    candidate_index = Column(Integer, nullable=False, default=1)
    # pending / running / succeeded / failed / invalidated
    status = Column(String, nullable=False, default="pending")
    # 稳定候选契约：候选产物永不等于 Shot.video_path 正式发布路径。
    path = Column(String, nullable=False, default="")
    last_frame_path = Column(String, nullable=False, default="")
    provider = Column(String, nullable=False, default="")
    model = Column(String, nullable=False, default="")
    seed = Column(Integer, nullable=True)
    recipe_hash = Column(String, nullable=False, default="")
    reference_manifest = Column(Text, nullable=False, default="[]")
    generation_duration_ms = Column(Integer, nullable=False, default=0)
    # 本步骤只有确定性结构/技术分数，不接入深度视觉质量模型。
    score = Column(Float, nullable=False, default=0.0)
    metrics = Column(Text, nullable=False, default="{}")
    failure = Column(Text, nullable=False, default="{}")
    # 旧字段继续保留，兼容已有查询/数据迁移。
    video_path = Column(String, nullable=False, default="")
    tail_frame_path = Column(String, nullable=False, default="")
    execution_plan_hash = Column(String, nullable=False, default="")
    structural_passed = Column(Boolean, nullable=True)
    structural_metrics = Column(Text, nullable=False, default="{}")
    failure_kind = Column(String, nullable=False, default="")
    failure_message = Column(Text, nullable=False, default="")
    retry_of_candidate_id = Column(String, nullable=False, default="")
    selected = Column(Boolean, nullable=False, default=False)
    selection_reason = Column(String, nullable=False, default="")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    selected_at = Column(DateTime, nullable=True)
