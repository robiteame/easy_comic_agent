"""测试套件的环境引导：必须先于任何 services/config 导入执行。

pytest 收集阶段会按文件顺序导入测试模块；只要有一个模块在导入
``test_environment`` 之前传递导入了 ``config``，``settings`` 就会绑定到
真实仓库目录，后续所有依赖 OUTPUT_DIR/DATA_DIR allowed_roots 校验的模块
（参考图加载、媒体探测）都会在合跑时失败。conftest 在任何测试模块之前
加载，在这里统一初始化测试沙箱环境，消除模块顺序依赖。
"""

from __future__ import annotations

import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

import test_environment  # noqa: F401,E402
