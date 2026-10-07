"""根目录 pytest 收集边界。"""

# 发行目录包含另一套 Python 运行时及其依赖自带测试，不能参与仓库测试收集。
collect_ignore_glob = ["client/**"]
