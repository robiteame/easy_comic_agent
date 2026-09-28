from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .base import Base


class SceneAsset(Base):
    __tablename__ = "scene_assets"

    id = Column(String, primary_key=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    visual_prompt = Column(Text, default="")
    negative_prompt = Column(Text, default="")
    reference_images = Column(Text, default="[]")
    key_features = Column(Text, default="[]")
    scene_group_key = Column(String, default="")
    time_of_day = Column(String, default="")
    baseline_image_path = Column(String, default="")
    consistency_profile = Column(Text, default="{}")
    prop_lock = Column(Text, default="")
    seed = Column(Integer, default=1200)
    style_fingerprint = Column(String, default="")
    asset_status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="scenes")
