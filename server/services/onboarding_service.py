"""首次启动向导完成标志的持久化。

标志保存在 ``DATA_DIR/onboarding_status.json``（原子写）。桌面打包模式下
``DATA_DIR`` 位于 Electron ``userData`` 内，随安装迁移；开发模式落在
``server/data``。前端通过 ``GET/PUT /api/settings/onboarding`` 读写，
不在渲染进程自行存储（localStorage 不随安装迁移，不采用）。
"""

from __future__ import annotations

from datetime import datetime

from config import settings
from services.atomic_json import atomic_write_json, read_json_file

_STATUS_PATH = settings.DATA_DIR / "onboarding_status.json"


def _normalize(raw: object) -> dict:
    data = raw if isinstance(raw, dict) else {}
    completed_at = str(data.get("completed_at") or "")
    return {
        "completed": bool(data.get("completed")),
        "completed_at": completed_at if data.get("completed") else "",
    }


def get_onboarding_status() -> dict:
    """读取向导完成状态；文件缺失/损坏时视为未完成（会重新出现向导）。"""

    return _normalize(read_json_file(_STATUS_PATH, default={}))


def save_onboarding_status(completed: bool) -> dict:
    """写入向导完成状态，返回写入后的完整状态。"""

    payload = {
        "completed": bool(completed),
        "completed_at": datetime.utcnow().isoformat() if completed else "",
    }
    atomic_write_json(_STATUS_PATH, payload)
    return payload
