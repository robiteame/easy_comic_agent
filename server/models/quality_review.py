"""镜头质量审核记录（quality review）。

自动模式质量闭环的落库凭证：每轮候选（含重试）各追加一行，记录结构化
评分、问题、证据、未检测维度与修正建议。它与结构检查（structural
validation）严格分开——结构检查只代表「文件可用」，本表才承载
「质量是否通过」的判定，且未检测项一律如实标注，绝不冒充通过。

字段约定：
- ``stage``: 审核对象阶段，storyboard（定稿故事板图）或 video（镜头视频）；
- ``verdict``: passed / failed / unsupported / error（能力未配置 =
  unsupported，绝不下发 passed）；
- ``dimensions``: JSON 数组，每项 {key,label,status,provider,score,weight,
  issues,evidence}，status ∈ scored / unsupported / skipped / error；
- ``prompt_fix``: 本轮依据审核结果应用的 prompt 修正指令（JSON）。
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, String, Text

from .base import Base


class QualityReview(Base):
    __tablename__ = "quality_reviews"
    __table_args__ = (
        Index("ix_quality_reviews_shot_stage_created", "shot_id", "stage", "created_at"),
        Index("ix_quality_reviews_project_stage", "project_id", "stage"),
    )

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    shot_id = Column(String, nullable=False)
    stage = Column(String, default="storyboard")
    # 该镜头在第几轮候选上被审核（1 = 首轮，>1 = 修正重生成后的复检）。
    attempt = Column(Integer, default=1)
    shot_version = Column(Integer, default=1)
    target_path = Column(String, default="")
    verdict = Column(String, default="failed")
    passed = Column(Boolean, default=False)
    overall_score = Column(Float, default=0.0)
    # 有维度未检测（Provider 未配置）但按降级策略放行时置位，界面必须提示。
    degraded = Column(Boolean, default=False)
    dimensions = Column(Text, default="[]")
    issues = Column(Text, default="[]")
    unsupported_dimensions = Column(Text, default="[]")
    suggestion = Column(Text, default="")
    prompt_fix = Column(Text, default="")
    # 判定时的门禁快照，如 "threshold=0.75 policy=strict max_retries=2"。
    gate_policy = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
