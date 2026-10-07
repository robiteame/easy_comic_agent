"""Regression tests for the ComicAgent server.

The suite intentionally uses the standard library's unittest runner so a
fresh checkout can execute the safety checks before installing pytest.

unittest 加载本包的前提是 discovery 以 `-t` 指定 server 为顶层目录（此时测试
模块以 tests.test_xxx 导入，本 __init__ 先于所有测试模块执行）。与 pytest 的
tests/conftest.py 相同，这里必须在任何测试模块导入 services/config 之前完成
沙箱初始化，否则 settings 会绑定到真实仓库目录，测试间产生状态污染，并可能
改写仓库内的 server/data 文件。
"""

from __future__ import annotations

from tests.support import test_environment  # noqa: F401
