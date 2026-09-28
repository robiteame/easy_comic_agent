from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship

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
    dialogue = Column(Text, default="")
    camera_angle = Column(String, default="正面")
    camera_movement = Column(String, default="静止")
    duration = Column(Float, default=3.0)
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
    continuity_profile = Column(Text, default="{}")
    continuity_reference_path = Column(String, default="")
    pose_reference_path = Column(String, default="")
    depth_reference_path = Column(String, default="")
    last_frame_path = Column(String, default="")
    style_fingerprint = Column(String, default="")
    # 参数/配置已变更但旧素材仍保留：True 表示当前媒体与最新参数不一致，
    # 需要重新生成；旧路径在新素材成功生成前不得清空。
    media_stale = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="shots")
