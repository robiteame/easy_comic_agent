from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Index, Integer, String, Text

from .base import Base


class BackgroundJob(Base):
    """Durable ownership record for a fire-and-forget server job.

    任务中心在原有字段之上追加展示、重试与续跑所需的元数据；原有字段（含
    ``run_token``、``error``、``version``）一个都没有删除或改名，历史数据可
    直接沿用。``run_token`` 是服务端内部的抢占令牌，任何 API/WebSocket 响应
    都不得序列化它。
    """

    __tablename__ = "background_jobs"
    __table_args__ = (
        Index("ix_background_jobs_scope_status", "scope", "status"),
        Index("ix_background_jobs_status_updated", "status", "updated_at"),
        Index("ix_background_jobs_project_updated", "project_id", "updated_at"),
        Index("ix_background_jobs_type_updated", "job_type", "updated_at"),
        Index("ix_background_jobs_updated_at", "updated_at"),
        Index("ix_background_jobs_queue_batch", "queue_batch_id", "queue_position"),
        Index("ix_background_jobs_queue_stage", "queue_stage", "status"),
    )

    id = Column(String, primary_key=True)
    idempotency_key = Column(String, nullable=False, unique=True, index=True)
    scope = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default="queued", index=True)
    progress = Column(Integer, nullable=False, default=0)
    error = Column(Text, nullable=False, default="")
    version = Column(Integer, nullable=False, default=0)
    run_token = Column(String, nullable=False, default="")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # --- 任务中心扩展字段（SQLite 走增量 ALTER TABLE ADD COLUMN） ---
    # 所属项目：任务列表按项目筛选、前端「跳转到对应项目」都依赖它。
    project_id = Column(String, nullable=False, default="", index=True)
    # 稳定任务类型（script_pipeline / storyboard / shot_image / ...），前端只按它分支。
    job_type = Column(String, nullable=False, default="unknown")
    # 面向用户的短名称，由服务端在抢占时按业务上下文生成，不含密钥或完整 prompt。
    display_name = Column(String, nullable=False, default="")
    current_step = Column(String, nullable=False, default="")
    message = Column(String, nullable=False, default="")
    # 稳定错误码 + 脱敏截断后的短错误消息；完整堆栈只写服务端日志。
    error_code = Column(String, nullable=False, default="")
    error_message = Column(String, nullable=False, default="")
    # 失败原因分析结果（JSON 字符串）：{"summary","suggestion","source","model","analyzed_at"}。
    # source 为 "rule" 或 "llm"；类别本身不在这里重复存，始终以 error_code 为准。
    error_detail = Column(Text, nullable=False, default="")
    # 任务完成后可复核的结果报告（如一致性降级影响范围），JSON 字符串。
    report = Column(Text, nullable=False, default="{}")
    # 调试日志（JSON 数组）：进度事件、API 请求参数、提示词与请求结果。
    # 内容写入前统一脱敏/限长；debug_revision 用于 WebSocket 增量事件去重。
    debug_events = Column(Text, nullable=False, default="[]")
    debug_revision = Column(Integer, nullable=False, default=0)
    # 第几次尝试；重试会新建一行 attempt+1 并保留原行，绝不覆盖历史。
    attempt = Column(Integer, nullable=False, default=1)
    retry_of = Column(String, nullable=True)
    cancel_requested_at = Column(DateTime, nullable=True)

    # 选择性重生成队列元数据。它们与任务状态共存于同一行，调度器不另建一套
    # 运行令牌或作用域锁；历史任务缺少这些列时由 init_db 做增量迁移。
    queue_batch_id = Column(String, nullable=False, default="", index=True)
    queue_position = Column(Integer, nullable=False, default=0)
    queue_priority = Column(Integer, nullable=False, default=0)
    queue_order = Column(Integer, nullable=False, default=0)
    queue_stage = Column(String, nullable=False, default="")
    queue_shot_id = Column(String, nullable=False, default="", index=True)
    queue_dependency_ids = Column(Text, nullable=False, default="[]")
    queue_blocked_reason = Column(String, nullable=False, default="")
    queue_concurrency = Column(Integer, nullable=False, default=1)
    queue_paused = Column(Boolean, nullable=False, default=False)
    queue_resume_missing = Column(Boolean, nullable=False, default=False)
    queue_reuse_audio = Column(Boolean, nullable=False, default=False)
    queue_force_confirmed = Column(Boolean, nullable=False, default=False)
    queue_requested_version = Column(Integer, nullable=False, default=0)
