"""任务中心时间戳的时区语义回归测试。

背景：SQLite 里的历史时间列都是 naive UTC。此前 DTO 直接 ``isoformat()``，输出
不带时区的字符串，前端 ``new Date`` 按本地时间解释，Asia/Shanghai 环境下刚创建
的任务显示成「8 小时前」。

约束：

- DTO / 详情 / 统计接口对外时刻必须带 UTC 时区（``+00:00``）；
- 历史 naive 行按 UTC 解释（向后兼容，不迁移数据）；
- 已带时区的输入统一换算到 UTC；
- 时长计算不得因 aware/naive 混用而归零；
- 同一时刻在 UTC 与 Asia/Shanghai 两个环境下的换算必须一致。
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from db import SessionLocal, init_db  # noqa: E402
from models import BackgroundJob, Project  # noqa: E402
from services.job_center import attempt_history, job_stats  # noqa: E402
from services.job_dto import as_utc, job_dto, job_duration_seconds  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402

init_db()

SHANGHAI = ZoneInfo("Asia/Shanghai")
UTC = UTC


def _naive_utc_job(**overrides) -> BackgroundJob:
    """模拟数据库读出的历史行：全部 naive UTC 列。"""
    fields = dict(
        id="tz-job",
        idempotency_key="project:tz-project:render",
        scope="project:tz-project",
        status="running",
        progress=10,
        version=1,
        project_id="tz-project",
        job_type="render",
        created_at=datetime(2025, 1, 1, 2, 0, 0),
        started_at=datetime(2025, 1, 1, 2, 0, 0),
        updated_at=datetime(2025, 1, 1, 2, 1, 0),
    )
    fields.update(overrides)
    return BackgroundJob(**fields)


class AsUtcTests(unittest.TestCase):
    def test_naive_is_interpreted_as_utc(self) -> None:
        moment = as_utc(datetime(2025, 1, 1, 2, 0, 0))
        self.assertEqual(moment, datetime(2025, 1, 1, 2, 0, 0, tzinfo=UTC))

    def test_aware_is_normalized_to_utc(self) -> None:
        moment = as_utc(datetime(2025, 1, 1, 10, 0, 0, tzinfo=SHANGHAI))
        self.assertEqual(moment, datetime(2025, 1, 1, 2, 0, 0, tzinfo=UTC))

    def test_none_stays_none(self) -> None:
        self.assertIsNone(as_utc(None))


class JobDtoTimezoneTests(unittest.TestCase):
    def test_naive_rows_serialize_with_utc_offset(self) -> None:
        dto = job_dto(_naive_utc_job())
        self.assertEqual(dto["created_at"], "2025-01-01T02:00:00+00:00")
        self.assertEqual(dto["updated_at"], "2025-01-01T02:01:00+00:00")
        self.assertIsNone(dto["finished_at"])

    def test_serialized_instant_is_stable_across_timezones(self) -> None:
        dto = job_dto(_naive_utc_job())
        instant = datetime.fromisoformat(dto["created_at"])
        self.assertEqual(instant.utcoffset(), UTC.utcoffset(instant))
        self.assertEqual(instant.astimezone(SHANGHAI).hour, 10, "02:00 UTC 在上海是 10 点")
        self.assertEqual(instant.astimezone(UTC).hour, 2)

    def test_aware_input_is_converted_to_utc(self) -> None:
        job = _naive_utc_job(created_at=datetime(2025, 1, 1, 10, 0, 0, tzinfo=SHANGHAI))
        self.assertEqual(job_dto(job)["created_at"], "2025-01-01T02:00:00+00:00")

    def test_duration_mixed_aware_and_naive(self) -> None:
        job = _naive_utc_job()
        naive_now = datetime(2025, 1, 1, 2, 30, 0)
        aware_now = datetime(2025, 1, 1, 2, 30, 0, tzinfo=UTC)
        shanghai_now = datetime(2025, 1, 1, 10, 30, 0, tzinfo=SHANGHAI)
        self.assertEqual(job_duration_seconds(job, now=naive_now), 1800)
        self.assertEqual(job_duration_seconds(job, now=aware_now), 1800)
        self.assertEqual(job_duration_seconds(job, now=shanghai_now), 1800)

    def test_terminal_job_uses_finished_at(self) -> None:
        job = _naive_utc_job(
            status="completed",
            finished_at=datetime(2025, 1, 1, 2, 20, 0),
        )
        dto = job_dto(job, now=datetime(2025, 1, 1, 9, 0, 0, tzinfo=UTC))
        self.assertEqual(dto["finished_at"], "2025-01-01T02:20:00+00:00")
        self.assertEqual(dto["duration_seconds"], 1200)


class JobCenterTimezoneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = SessionLocal()
        self.db.query(BackgroundJob).delete()
        self.db.query(Project).delete()
        self.db.commit()

    def tearDown(self) -> None:
        self.db.rollback()
        self.db.close()

    def test_attempt_history_timestamps_carry_utc_offset(self) -> None:
        project = Project(id="tz-project", title="TZ")
        job = BackgroundJob(
            id="tz-history-job",
            idempotency_key="project:tz-project:render",
            scope="project:tz-project",
            status="completed",
            progress=100,
            version=1,
            project_id="tz-project",
            job_type="render",
            created_at=datetime(2025, 1, 1, 2, 0, 0),
            started_at=datetime(2025, 1, 1, 2, 0, 0),
            finished_at=datetime(2025, 1, 1, 2, 5, 0),
            updated_at=datetime(2025, 1, 1, 2, 5, 0),
        )
        self.db.add_all([project, job])
        self.db.commit()

        history = attempt_history(self.db, job)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["started_at"], "2025-01-01T02:00:00+00:00")
        self.assertEqual(history[0]["finished_at"], "2025-01-01T02:05:00+00:00")
        self.assertEqual(history[0]["duration_seconds"], 300)
        instant = datetime.fromisoformat(history[0]["started_at"])
        self.assertEqual(instant.astimezone(SHANGHAI).hour, 10)

    def test_stats_generated_at_carry_utc_offset(self) -> None:
        payload = job_stats(self.db)
        self.assertTrue(str(payload["generated_at"]).endswith("+00:00"))


if __name__ == "__main__":
    unittest.main()
