import logging
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker

from config import settings
from models.base import Base

logger = logging.getLogger(__name__)

db_path = Path(settings.DATABASE_URL.replace("sqlite:///", ""))
db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _configure_sqlite(dbapi_connection, _connection_record) -> None:
    """Enable durability/concurrency pragmas for the local SQLite store."""

    if engine.url.get_backend_name() != "sqlite":
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
    finally:
        cursor.close()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    # 导入即注册：新增的计价 / 用量 / 预算表由 create_all 建表，旧库不受影响。
    from models import (  # noqa: F401
        AudioTrack,
        BackgroundJob,
        BudgetConfig,
        BudgetReservation,
        Character,
        CostEstimate,
        PricingConfig,
        Project,
        QualityReview,
        SceneAsset,
        Shot,
        ShotVersion,
        SubtitleCue,
        SubtitleTrack,
        UsageRecord,
    )

    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_columns()
    _ensure_sqlite_indexes()
    _seed_pricing_defaults()
    _ensure_shot_version_append_only()
    # 先修复历史数据，再建立约束：触发器一旦存在，非法存量行会连状态更新都被拒绝。
    repaired = _repair_project_tree()
    if repaired:
        print(f"已修复 {repaired} 条非法项目树记录（孤儿剧集 / 未知类型 / 负数集号）")
    _ensure_project_tree_constraints()


def _seed_pricing_defaults() -> None:
    """写入出厂价目模板（幂等）。只声明可配置项，绝不预置任何金额。"""

    from services.pricing_service import seed_pricing_defaults

    db = SessionLocal()
    try:
        created = seed_pricing_defaults(db)
    except Exception as exc:  # noqa: BLE001 - 计价模板失败不应阻止服务启动
        logger.warning("模型价格模板初始化失败: %s", exc)
        return
    finally:
        db.close()
    if created:
        logger.info("已初始化 %d 条模型价格模板（价格未配置，成本显示为未知）", created)


def _ensure_sqlite_columns() -> None:
    """Small additive migration for existing local SQLite databases."""
    inspector = inspect(engine)
    if "shots" in inspector.get_table_names():
        _add_missing_columns(
            "shots",
            {
                "confirmed": "BOOLEAN DEFAULT 0",
                "parent_version_id": "VARCHAR DEFAULT ''",
                "scene_asset_id": "VARCHAR DEFAULT ''",
                "character_asset_ids": "TEXT DEFAULT '[]'",
                "storyboard_path": "VARCHAR DEFAULT ''",
                "storyboard_status": "VARCHAR DEFAULT 'pending'",
                "video_path": "VARCHAR DEFAULT ''",
                "visual_notes": "TEXT DEFAULT ''",
                "scene_group_id": "VARCHAR DEFAULT ''",
                "consistency_context": "TEXT DEFAULT ''",
                "reference_weights": "TEXT DEFAULT '{}'",
                "continuity_profile": "TEXT DEFAULT '{}'",
                "continuity_reference_path": "VARCHAR DEFAULT ''",
                "pose_reference_path": "VARCHAR DEFAULT ''",
                "depth_reference_path": "VARCHAR DEFAULT ''",
                "last_frame_path": "VARCHAR DEFAULT ''",
                "style_fingerprint": "VARCHAR DEFAULT ''",
                "media_stale": "BOOLEAN DEFAULT 0",
            },
        )
    if "projects" in inspector.get_table_names():
        _add_missing_columns(
            "projects",
            {
                "output_format": "VARCHAR DEFAULT '9:16'",
                "resolution": "VARCHAR DEFAULT '1080p'",
                "platform": "VARCHAR DEFAULT 'douyin'",
                "parent_project_id": "VARCHAR DEFAULT ''",
                "project_type": "VARCHAR DEFAULT 'series'",
                "episode_number": "INTEGER DEFAULT 0",
                "consistency_config": "TEXT DEFAULT '{}'",
                # 字幕/音频工作台：轨道或字幕条目每次修改都会 +1，渲染任务
                # 记录渲染时的版本，发布前不一致则丢弃成片（配置已过期）。
                "av_config_version": "INTEGER DEFAULT 0",
            },
        )
    if "characters" in inspector.get_table_names():
        _add_missing_columns(
            "characters",
            {
                "reference_images": "TEXT DEFAULT '[]'",
                "default_outfit": "TEXT DEFAULT ''",
                "lora_profile": "TEXT DEFAULT ''",
                "ip_adapter_profile": "TEXT DEFAULT ''",
                "wardrobe_lock": "TEXT DEFAULT ''",
                "style_fingerprint": "VARCHAR DEFAULT ''",
                "asset_status": "VARCHAR DEFAULT 'active'",
            },
        )
    if "scene_assets" in inspector.get_table_names():
        _add_missing_columns(
            "scene_assets",
            {
                "scene_group_key": "VARCHAR DEFAULT ''",
                "time_of_day": "VARCHAR DEFAULT ''",
                "baseline_image_path": "VARCHAR DEFAULT ''",
                "consistency_profile": "TEXT DEFAULT '{}'",
                "prop_lock": "TEXT DEFAULT ''",
                "style_fingerprint": "VARCHAR DEFAULT ''",
                "asset_status": "VARCHAR DEFAULT 'active'",
            },
        )
    if "background_jobs" in inspector.get_table_names():
        _add_missing_columns(
            "background_jobs",
            {
                "run_token": "VARCHAR DEFAULT '' NOT NULL",
                # 任务中心扩展列：全部带默认值，存量 background_jobs 数据不受影响。
                "project_id": "VARCHAR DEFAULT '' NOT NULL",
                "job_type": "VARCHAR DEFAULT 'unknown' NOT NULL",
                "display_name": "VARCHAR DEFAULT '' NOT NULL",
                "current_step": "VARCHAR DEFAULT '' NOT NULL",
                "message": "VARCHAR DEFAULT '' NOT NULL",
                "error_code": "VARCHAR DEFAULT '' NOT NULL",
                "error_message": "VARCHAR DEFAULT '' NOT NULL",
                "error_detail": "TEXT DEFAULT '' NOT NULL",
                "attempt": "INTEGER DEFAULT 1 NOT NULL",
                "retry_of": "VARCHAR",
                "cancel_requested_at": "DATETIME",
                "queue_batch_id": "VARCHAR DEFAULT '' NOT NULL",
                "queue_position": "INTEGER DEFAULT 0 NOT NULL",
                "queue_priority": "INTEGER DEFAULT 0 NOT NULL",
                "queue_order": "INTEGER DEFAULT 0 NOT NULL",
                "queue_stage": "VARCHAR DEFAULT '' NOT NULL",
                "queue_shot_id": "VARCHAR DEFAULT '' NOT NULL",
                "queue_dependency_ids": "TEXT DEFAULT '[]' NOT NULL",
                "queue_blocked_reason": "VARCHAR DEFAULT '' NOT NULL",
                "queue_concurrency": "INTEGER DEFAULT 1 NOT NULL",
                "queue_paused": "BOOLEAN DEFAULT 0 NOT NULL",
                "queue_resume_missing": "BOOLEAN DEFAULT 0 NOT NULL",
                "queue_reuse_audio": "BOOLEAN DEFAULT 0 NOT NULL",
                "queue_force_confirmed": "BOOLEAN DEFAULT 0 NOT NULL",
                "queue_requested_version": "INTEGER DEFAULT 0 NOT NULL",
            },
        )
        _backfill_background_job_metadata()


def _add_missing_columns(table: str, columns: dict[str, str]) -> None:
    inspector = inspect(engine)
    existing = {column["name"] for column in inspector.get_columns(table)}
    with engine.begin() as conn:
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def _backfill_background_job_metadata() -> None:
    """为存量 background_jobs 补齐任务中心所需的展示字段（幂等、只补空缺）。

    只做纯派生：任务类型/项目/操作从幂等键解析，``attempt`` 从历史后缀解析。
    不覆盖任何已有的业务列，也不改写状态、错误或时间戳。
    """

    from services.job_types import JOB_TYPE_UNKNOWN, parse_job_key

    inspector = inspect(engine)
    if "background_jobs" not in inspector.get_table_names():
        return
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, idempotency_key, scope, job_type, project_id, attempt FROM background_jobs")
        ).mappings().all()
        for row in rows:
            key = parse_job_key(str(row["idempotency_key"] or ""))
            scope_parts = str(row["scope"] or "").split(":")
            scope_project = scope_parts[1] if len(scope_parts) > 1 and scope_parts[0] == "project" else ""
            project_id = str(row["project_id"] or "") or scope_project
            job_type = str(row["job_type"] or "")
            if job_type in ("", JOB_TYPE_UNKNOWN):
                job_type = key.job_type
            attempt = int(row["attempt"] or 0) or 1
            if key.archived:
                attempt = max(attempt, int(key.raw.rsplit("#attempt-", 1)[1]))
            if (
                project_id == str(row["project_id"] or "")
                and job_type == str(row["job_type"] or "")
                and attempt == int(row["attempt"] or 0)
            ):
                continue
            conn.execute(
                text(
                    "UPDATE background_jobs SET project_id = :project_id, job_type = :job_type, attempt = :attempt "
                    "WHERE id = :id"
                ),
                {"project_id": project_id, "job_type": job_type, "attempt": attempt, "id": row["id"]},
            )


def _ensure_sqlite_indexes() -> None:
    """Add indexes missing from databases created by older app versions."""

    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_shots_project_sequence ON shots (project_id, sequence)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_projects_parent_type ON projects (parent_project_id, project_type)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_projects_updated_at ON projects (updated_at DESC)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_background_jobs_scope_status ON background_jobs (scope, status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_background_jobs_status_updated ON background_jobs (status, updated_at)"))
        # 任务中心默认按 updated_at DESC 排序，并按项目 / 类型筛选。
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_background_jobs_project_updated ON background_jobs (project_id, updated_at)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_background_jobs_type_updated ON background_jobs (job_type, updated_at)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_background_jobs_updated_at ON background_jobs (updated_at)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_background_jobs_queue_batch ON background_jobs (queue_batch_id, queue_position)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_background_jobs_queue_stage ON background_jobs (queue_stage, status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_background_jobs_queue_shot ON background_jobs (queue_shot_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_background_jobs_scope ON background_jobs (scope)"))
        # 成本与预算：任务中心按 job_id / job_key 反查成本，统计页按项目 + 时间聚合。
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_usage_records_project_created ON usage_records (project_id, created_at)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_usage_records_job_key ON usage_records (job_key)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_usage_records_series_created ON usage_records (series_id, created_at)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_budget_reservations_status ON budget_reservations (status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_shot_versions_shot_number ON shot_versions (shot_id, number)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_shot_versions_project ON shot_versions (project_id)"))
        # 质量审核：按镜头 + 阶段取最新一轮，按项目聚合门禁状态。
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_quality_reviews_shot_stage_created ON quality_reviews (shot_id, stage, created_at)")
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_quality_reviews_project_stage ON quality_reviews (project_id, stage)"))
        # 字幕 / 音频工作台：列表按项目读取，字幕条目按轨道 + 顺序读取。
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_subtitle_tracks_project ON subtitle_tracks (project_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_subtitle_cues_track_order ON subtitle_cues (track_id, order_index)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_subtitle_cues_project ON subtitle_cues (project_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_audio_tracks_project_kind ON audio_tracks (project_id, kind)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_audio_tracks_project ON audio_tracks (project_id)"))


def _ensure_shot_version_append_only() -> None:
    """版本历史只允许追加：任何 UPDATE 都直接拒绝（应用层恢复走 INSERT 新行）。

    DELETE 不受限制：项目删除 / 重新解析需要按项目清理版本记录，与「不可变」
    的含义（历史内容不被改写）不冲突。
    """

    if "shot_versions" not in inspect(engine).get_table_names():
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TRIGGER IF NOT EXISTS trg_shot_versions_append_only "
                "BEFORE UPDATE ON shot_versions FOR EACH ROW "
                "BEGIN SELECT RAISE(ABORT, 'shot versions are append-only'); END"
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_background_jobs_status ON background_jobs (status)"))


# --- 项目树完整性（SQLite 触发器，兼容存量库） ---

# 形状约束：类型合法、集号非负、剧集必须有父级、大项目不能有父级、禁止自引用。
_PROJECT_TREE_SHAPE = """
    NEW.project_type NOT IN ('series', 'episode')
    OR (NEW.episode_number IS NOT NULL AND NEW.episode_number < 0)
    OR NEW.parent_project_id = NEW.id
    OR (NEW.project_type = 'episode' AND (NEW.parent_project_id IS NULL OR NEW.parent_project_id = ''))
    OR (NEW.project_type = 'series' AND NEW.parent_project_id IS NOT NULL AND NEW.parent_project_id <> '')
"""

# 父级约束：剧集的父级必须真实存在且是 series。
#
# 生效时机做了两点收敛，避免误伤合法写入，同时保证「谁改父子关系谁被校验」：
# 1. INSERT 只校验形状。调用方把多行 add_all 一次性落库时不保证父行先插入，
#    在 INSERT 阶段校验父级存在性会把合法写入判成违规；父级解析仍由 API 层在
#    写入前完成（写错会得到 400/404 而不是静默写入）。
# 2. UPDATE 只在本次真的改动父子/类型字段时校验父级，这样历史遗留行的状态更新
#    （例如删除流程写 status）不会被卡住，启动时的修复流程负责把它们修正。
_PROJECT_TREE_PARENT = """
    NEW.project_type = 'episode'
    AND (SELECT project_type FROM projects WHERE id = NEW.parent_project_id) IS NOT 'series'
"""

_PROJECT_TREE_PARENT_CHANGED = """
    (NEW.project_type IS NOT OLD.project_type OR NEW.parent_project_id IS NOT OLD.parent_project_id)
    AND (NEW.project_type = 'episode')
"""

_PROJECT_TREE_ABORT = "RAISE(ABORT, 'invalid project tree: episode requires an existing series parent')"


def _ensure_project_tree_constraints() -> None:
    """为存量 SQLite 库补上等价于 CHECK 约束的项目树触发器。"""

    if "projects" not in inspect(engine).get_table_names():
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TRIGGER IF NOT EXISTS trg_projects_insert_tree "
                "BEFORE INSERT ON projects FOR EACH ROW "
                f"WHEN ({_PROJECT_TREE_SHAPE}) BEGIN SELECT {_PROJECT_TREE_ABORT}; END"
            )
        )
        conn.execute(
            text(
                "CREATE TRIGGER IF NOT EXISTS trg_projects_update_tree "
                "BEFORE UPDATE ON projects FOR EACH ROW "
                f"WHEN ({_PROJECT_TREE_SHAPE} OR (({_PROJECT_TREE_PARENT_CHANGED}) AND ({_PROJECT_TREE_PARENT}))) "
                f"BEGIN SELECT {_PROJECT_TREE_ABORT}; END"
            )
        )


def _repair_project_tree() -> int:
    """把历史数据修成合法项目树；幂等，返回修复的行数。

    规则保守且只保留可恢复信息：孤儿剧集（父级缺失/父级不是 series）提升为
    series，series 的父级被清空，未知 project_type 归为 series，负数集号归零。
    """

    if "projects" not in inspect(engine).get_table_names():
        return 0
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, parent_project_id, project_type, episode_number FROM projects")
        ).mappings().all()
        if not rows:
            return 0
        types = {
            str(row["id"]): ("episode" if str(row["project_type"] or "").strip() == "episode" else "series")
            for row in rows
        }
        fixes: list[tuple[str, str, str, int]] = []
        for row in rows:
            project_id = str(row["id"])
            raw_type = str(row["project_type"] or "").strip()
            current_type = types[project_id]
            current_parent = str(row["parent_project_id"] or "").strip()
            raw_number = row["episode_number"]
            target_type, target_parent = current_type, current_parent
            if current_type == "episode":
                if not current_parent or current_parent == project_id or types.get(current_parent) != "series":
                    target_type, target_parent = "series", ""
            else:
                target_parent = ""
            target_number = int(raw_number) if isinstance(raw_number, int) and raw_number >= 0 else 0
            # 与原始列值比较（不能用归一化后的类型），否则未知类型会被当成已合法而漏修。
            if target_type != raw_type or target_parent != current_parent or target_number != raw_number:
                fixes.append((project_id, target_type, target_parent, target_number))
        for project_id, project_type, parent_id, number in fixes:
            conn.execute(
                text(
                    "UPDATE projects SET project_type = :project_type, parent_project_id = :parent_id, "
                    "episode_number = :number WHERE id = :project_id"
                ),
                {"project_type": project_type, "parent_id": parent_id, "number": number, "project_id": project_id},
            )
    if fixes:
        logger.warning("修复了 %d 条非法项目树记录（孤儿剧集 / 未知类型 / 负数集号）", len(fixes))
    return len(fixes)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
