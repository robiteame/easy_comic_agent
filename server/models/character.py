from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="characters")
