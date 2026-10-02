from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship, validates

from .base import Base


class Shot(Base):
    __tablename__ = "shots"
    __table_args__ = (Index("ix_shots_project_sequence", "project_id", "sequence"),)

    id = Column(String, primary_key=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False, index=True)
    sequence = Column(Integer, default=0)
    shot_type = Column(String, default="medium")
    scene_description = Column(Text, default="")
    character_action = Column(Text, default="")
    # 结构化对白的 JSON 数组文本（speaker/line/emotion/action/start_ms/end_ms）。
    # 旧项目的纯文本字符串允许直接落库，读取时由 shot_dialogue.parse 迁移。
    dialogue = Column(Text, default="")
    camera_angle = Column(String, default="正面")
    camera_movement = Column(String, default="静止")
    duration = Column(Float, default=3.0)
    estimated_speech_ms = Column(Integer, default=0)
    emotion = Column(String, default="neutral")
    transition = Column(String, default="cut")
    image_path = Column(String, default="")
    video_path = Column(String, default="")
    audio_path = Column(String, default="")
    status = Column(String, default="pending")
    version = Column(Integer, default=1)
    confirmed = Column(Boolean, default=False)
    parent_version_id = Column(String, default="")
    characters_in_scene = Column(Text, default="[]")
    scene_asset_id = Column(String, default="")
    character_asset_ids = Column(Text, default="[]")
    storyboard_path = Column(String, default="")
    storyboard_status = Column(String, default="pending")
    visual_notes = Column(Text, default="")
    scene_group_id = Column(String, default="")
    consistency_context = Column(Text, default="")
    reference_weights = Column(Text, default="{}")
    # continuity_profile 除一致性档案外，还承载镜头统一执行计划
    # （execution_plan 键：narrative_duration_ms / provider_generation_duration_s /
    # trim_start_ms / trim_end_ms / audio_mode / dialogue_timing /
    # continuity_mode / video_mode / recipe_hash），复用现有 JSON 字段落库，
    # 不为执行计划单独做数据库迁移；旧数据无该键时由 story_timing 推导。
    continuity_profile = Column(Text, default="{}")
    continuity_reference_path = Column(String, default="")
    pose_reference_path = Column(String, default="")
    depth_reference_path = Column(String, default="")
    last_frame_path = Column(String, default="")
    style_fingerprint = Column(String, default="")
    # 参数/配置已变更但旧素材仍保留：True 表示当前媒体与最新参数不一致，
    # 需要重新生成；旧路径在新素材成功生成前不得清空。
    media_stale = Column(Boolean, default=False)
    # 镜头级一致性状态与实际使用的参考素材版本。
    consistency_status = Column(String, default="pending")
    consistency_report = Column(Text, default="{}")
    storyboard_reference_manifest = Column(Text, default="[]")
    video_reference_manifest = Column(Text, default="[]")
    reference_capability_warning = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="shots")

    @validates("dialogue")
    def _serialize_dialogue_on_write(self, _key: str, value):
        # 允许调用方直接赋结构化列表（dict 或 DialogueLine 混合）；统一序列化
        # 为 JSON 文本入库。字符串原样透传（可能是 JSON 文本或待读取时迁移的
        # 旧版纯文本）。
        if isinstance(value, (list, tuple)):
            from services.shot_dialogue import parse_shot_dialogue, serialize_dialogue_lines

            return serialize_dialogue_lines(parse_shot_dialogue(value))
        return value
