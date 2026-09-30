from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .base import Base


class Character(Base):
    __tablename__ = "characters"

    id = Column(String, primary_key=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    appearance = Column(Text, default="")
    personality = Column(Text, default="")
    visual_prompt = Column(Text, default="")
    negative_prompt = Column(Text, default="")
    voice_id = Column(String, default="")
    emotion_variants = Column(Text, default="{}")
    key_features = Column(Text, default="[]")
    default_outfit = Column(Text, default="")
    reference_images = Column(Text, default="[]")
    lora_profile = Column(Text, default="")
    ip_adapter_profile = Column(Text, default="")
    wardrobe_lock = Column(Text, default="")
    seed = Column(String, default="42")
    style_fingerprint = Column(String, default="")
    asset_status = Column(String, default="active")
    # 一致性参考素材生命周期状态：ready / failed / degraded / unsupported / stale。
    reference_status = Column(String, default="stale")
    reference_version = Column(Integer, default=1)
    reference_retry_count = Column(Integer, default=0)
    reference_failure_reason = Column(Text, default="")
    reference_error_id = Column(String, default="")
    reference_skip_reason = Column(Text, default="")
    reference_capability_warning = Column(Text, default="")
    # 影响镜头范围与稳定 ID，JSON：{"shot_ids":[...],"shot_range":"镜头 1-3"}。
    reference_impact = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="characters")
