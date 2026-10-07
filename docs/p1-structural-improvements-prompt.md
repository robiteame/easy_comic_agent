# P1 结构性改进 — 执行提示词

> 用法：把本文件全文粘贴给一个新会话。四个任务相互独立，可整体执行，也可以只粘贴单个任务段落执行。
> 前置依赖：P0（lint 工具链）已完成并提交。若 `git status` 仍有大量未提交改动，先停下向用户确认。

---

## 角色与背景

你在 ComicAgent 仓库工作（macOS，pnpm workspace）：`client/` 是 Electron 44 + React 18 + Vite 5 + antd 5 + zustand 桌面客户端；`server/` 是 Python 3.11 + FastAPI + LangGraph + SQLAlchemy(SQLite) 服务端，约 7 万行。桌面打包通过 electron-builder 把整个 `server/` 目录塞进 extraResources，由主进程 spawn `python main.py` 随机 loopback 端口启动。

P0 已完成（本提示词的既成事实，不要重做）：
- ruff 已接入服务端（配置 `server/pyproject.toml`，目前只有 `[tool.ruff]` 段，无 `[project]`），`ruff==0.16.10` 在 `server/requirements-dev.lock`；biome 已接入客户端（配置根 `biome.json`，包是 `@biomejs/biome`——注意 npm 上的 `biome` 是抢注的无关旧包）。两者均为 CI 门禁（`.github/workflows/ci.yml`）+ lefthook pre-commit 自动修复。
- `client/dist-electron/` 已移出 git 跟踪。
- 当前基线：ruff check/format 全绿（214 文件）、biome 全绿（113 文件）、tsc 通过、客户端 152 个测试、服务端 918（unittest discover）/925（pytest）个测试全部通过。

本次 P1 目标：**覆盖率度量 → Python 打包化 → 客户端测试设施标准化 → 巨型文件渐进拆分**。这是纯结构性改造，用户可见行为必须零变化。

## 开始前必做

1. 确认 P0 已提交：`git log --oneline -5` 应有 lint 工具链接入类提交；`git status` 干净（或只有少量与本任务无关的用户改动）。
2. 通读 `server/CLAUDE.md`、`client/CLAUDE.md`（架构速览）。
3. 跑一遍「验证门禁」（见文末），确认起点全绿。
4. 本文所有行数是 2026-10 快照，执行时以 `wc -l` 实测为准，偏差很大时在汇报中说明。

---

## 任务 1：覆盖率度量（先建安全网，后续重构靠它兜底）

**事实**：两端目前零覆盖率工具。服务端测试跑法是 `python -m unittest discover -s tests -t . -p 'test_*.py'`（根目录 `pytest.ini` 定义了 `testpaths=server/tests`、`pythonpath=server`）；客户端用 node 内置 test runner（详见任务 3）。

**做什么**：
- 服务端：接入 coverage.py（跑法 `coverage run -m unittest discover -s tests -t . -p 'test_*.py'`，或改用 pytest-cov——二选一，与 CI 现有步骤融合即可）。`coverage` 加进 `server/requirements-dev.txt` 并按 P0 的方式重新生成哈希锁：`server/.venv/bin/uv pip compile --universal --python-version 3.11 --generate-hashes server/requirements-dev.txt --output-file server/requirements-dev.lock`（先确认 venv 里有 uv，没有就 `pip install uv`）。
- CI（ci.yml server job）加覆盖率步骤：**非阻塞**（不设阈值门槛），报告写入 `$GITHUB_STEP_SUMMARY`。
- 客户端：本次不单独接（node --test 的 `--experimental-test-coverage` 只是临时可选），正式方案随任务 3 的 vitest 自带 `--coverage` 落地。
- 在汇报中给出三个关键路径的基线数字：`services/budget_service.py`（预算扣费）、`services/task_registry.py`（任务状态机）、`models/` + `db/database.py`（shot 版本触发器/schema 迁移）。

**验收**：本地一条命令能产出服务端覆盖率报告；CI 有该步骤且不阻塞；拿到了三个关键路径的基线数字。

---

## 任务 2：服务端 pyproject 打包化（消 sys.path hack）

**事实**：`server/pyproject.toml` 无 `[project]` 段，服务端不是可安装包；`sys.path.insert` 引导散布在 `server/main.py:15`、`server/tests/__init__.py`、`server/tests/conftest.py` 以及根目录游离脚本里。`pytest.ini` 在仓库根。

**做什么**：
1. `server/pyproject.toml` 增加 `[project]`（name/version/requires-python>=3.11/dependencies 直接引用 requirements.txt 的钉版）+ setuptools 包发现。注意 `api/`、`agent/`、`services/`、`models/`、`db/`、`rag/`、`memory/` 应有 `__init__.py`（先验证），`scripts/` 没有、也不应被打进包。
2. `server/.venv/bin/pip install -e server` 后，逐一删除 sys.path hack（每删一处跑一次测试）。
3. pytest 配置迁入 pyproject 的 `[tool.pytest.ini_options]`，删除根目录 `pytest.ini`。CI 里的 `python -m pytest -q` 必须仍然通过（它承担收集校验职责）。
4. 清理根目录游离脚本（先 `grep -rn` 确认引用再动手）：
   - `test_environment.py`：**不是测试**，是 conftest 依赖的沙箱环境设置——移到 `tests/support/` 并同步修改 `tests/conftest.py`、`tests/__init__.py` 的导入，绝不能删。
   - `start_test.py`：文件名会中 pytest 的 `*_test.py` 收集规则，移到 `scripts/` 并改名（如 `startup_selfcheck.py`）。
   - `simple_test.py`、`test_import.py`、`test_start.py`：内容与现有测试重复或无引用，确认后删除（git 历史可找回）。

**红线**：桌面打包场景（`client/electron-builder.yml` 的 extraResources + `client/src/main/main.ts` spawn `python main.py`）里**没有 pip install 机会**——产物是把 server/ 目录原样拷贝、以脚本方式运行的。因此：如果 `main.py` 的 sys.path 引导实际服务于打包启动（脚本能靠"脚本所在目录自动入 path"就不需要它），删除前必须想清楚；改完本地验证 `cd server && .venv/bin/python main.py` 能启动且 `curl 127.0.0.1:8011/health` 返回 200，并在汇报中明确说明对打包路径的影响判断。拿不准就保留该处引导并注释原因。

**验收**：venv 内 editable install 后无 sys.path hack 也能跑全部测试；CI 全绿；`main.py` 本地可启动；游离脚本清理完毕。

---

## 任务 3：客户端测试设施迁移 vitest（删自研 loader）

**事实**：当前 `client/package.json` 的 `test` 脚本是 `node scripts/build-test-cache.mjs && node --experimental-strip-types --import ./scripts/test-loader.mjs --test <目录列表硬编码>`。自研设施包括：`scripts/test-loader-hooks.mjs`（145 行手写 ESM resolve/load：esbuild 转换 .tsx、扩展名解析、antd ESM 重写、zustand SSR 绑定替换）、`scripts/build-test-cache.mjs`（预转换缓存，为绕开 esbuild 多进程竞态导致的静默漏测）、`scripts/test-cache-paths.mjs`。45 个 `*.test.mts` 用 node:test 的 `test()` + node:assert，组件测试走 `react-dom/server` 的 `renderToString`（`src/renderer/test-support/ssrTestHelper.mts`、`fixtures.mts`）。

**做什么**：
1. 安装与 vite 5 兼容的 vitest，写 `client/vitest.config.ts`（复用现有 `@` → `src/renderer` 别名与 esbuild 目标；环境用 node，保持 renderToString 方案——**本次不引入 jsdom/Testing Library**）。
2. 批量迁移测试头：`import { test } from 'node:test'` → vitest 的导入方式（node:assert 的用法可以原样保留，vitest 环境下完全可用）。45 个文件是机械替换，可写一次性脚本或 sed，但替换后必须逐文件 spot-check。
3. 改 `package.json` test 脚本为 `vitest run`；删除 `test-loader.mjs`、`test-loader-hooks.mjs`、`build-test-cache.mjs`、`test-cache-paths.mjs` 及对它们的引用；根 `package.json` 的 `test`/`test:all` 链路同步确认。
4. 接入 `vitest run --coverage`（coverage.provider 默认即可），CI client job 加非阻塞覆盖率输出——与任务 1 呼应，客户端覆盖率在此落地。

**红线**：不改变任何断言语义；用例总数不少于 152；typecheck 与 biome 必须过（注意 biome 对新配置文件的格式要求）。

**验收**：`pnpm --dir client run test` 走 vitest 全绿；自研 loader 三件套 + build-test-cache 已删除且无残留引用；覆盖率可产出。

---

## 任务 4：巨型文件渐进拆分（行为保持，一次一个文件）

**事实**（快照行数，先实测校准）：
- server：`api/routes/shot.py` 3098（16 端点）、`agent/graph.py` 2745、`services/story_timing.py` 1885、`agent/critic.py` 1249、`services/budget_service.py` 1218、`services/task_registry.py` 1137、`agent/contracts.py` 1123、`services/video_service.py` 1067、`services/quality_review_service.py` 1066、`services/ffmpeg_service.py` 1036、`api/routes/project.py` 1000
- client：`components/MainWorkspace.tsx` 2355、`styles/global.css` 7335、`components/SystemSettingsPage.tsx` 1219、`components/RightSidebar.tsx` 1173、`services/costModel.ts` 1046、`components/TaskCenter.tsx` 1013

**策略（按序）**：

a. **先立规矩冻结增量**：写 `scripts/check-file-sizes.mjs`（client/ 和 server/ 一起管）——文件超过 1200 行即失败；当前超标的存量进白名单（写在脚本里，附当前行数），白名单文件行数**只减不增**（增长即失败）。接入 CI。这一步先落地，哪怕一个文件都不拆。

b. **拆分顺序：纯逻辑优先、入口文件靠后**。推荐顺序：`costModel.ts`（纯函数最易拆）→ `story_timing.py` → `global.css` → `SystemSettingsPage.tsx` / `RightSidebar.tsx` / `TaskCenter.tsx` → `shot.py` → `graph.py` → `MainWorkspace.tsx`（最后）。每个文件独立成一个提交粒度的工作单元，拆完一个跑全量验证门禁再拆下一个。

c. **client 拆法**：沿用仓库既有的 `*Model.ts` 视图模型模式（范例：`components/taskCenterModel.ts` 667 行 + 薄组件）。`MainWorkspace.tsx` 的派生状态、事件处理抽成 Model 文件 + 自定义 hooks（`useXxx`），UI 壳只留布局与组合；`App.tsx`（138 行组合根）不要动。每抽一块就为它补一个 Model 层单测（跟随现有 `*.test.mts` 模式）。

d. **server 拆法**：路由瘦身——`shot.py` 端点里的业务逻辑下沉到 services（参照 `budget_service.py` 的既有模式），route 只留参数校验/编排/DTO 组装；`graph.py` 按节点族（阶段节点/决策边/检查点）拆子模块，注意它被 `api/routes/graph.py` 引用了 4 个导出常量（`GRAPH_NODE_META` 等），公共导出签名不能变。

e. **CSS 拆法**：`global.css` 按区块（五栏布局/弹窗/动效/滚动条等）拆成 partials 由入口聚合。**必须保持选择器出现顺序完全不变**（级联行为对顺序敏感），最稳妥是纯切割不重排，拆完在浏览器里目检一遍关键页面。

**红线**：
- 一次只拆一个文件；禁止顺手重构、改命名、"改进"逻辑。
- 拆分提交里除移动外不应有 diff；用 `git diff --stat` 和通读自查。
- 任何拿不准是否等价的变换（如 CSS 选择器重排、graph.py 导出调整）——停下来在汇报里说明，不要自行决定。
- 本任务允许工作量大，做不完是正常的：按推荐顺序能拆几个是几个，白名单行数只减不增即为有效进展。

---

## 全局红线

- **不 commit、不 push**，除非用户明确要求；每完成一个任务输出一段小结（改了什么/为什么/验证结果），任务间是天然检查点。
- **行为保持**：这是结构改造，任何用户可见行为变化都是缺陷；不许改弱任何测试断言。
- **门禁不回退**：下述验证门禁在每个任务结束时必须全绿。
- **不要"顺手修"**：`server/pyproject.toml` 里 UP042（`str,Enum`→StrEnum）是被有意忽略的（f-string/str() 取值行为会变，存量契约按 `.value` 序列化）；biome 里 a11y 组、`noExplicitAny`、`useExhaustiveDependencies`、`useTemplate`、`noArrayIndexKey`、`noNonNullAssertion` 处于关闭状态，本次不开。
- 中文注释风格保持；新增注释说明"约束"而不是"变更过程"。

## 验证门禁（每任务结束必跑，全部通过才算完成）

```bash
# 客户端测试
cd client && pnpm run test
# 服务端测试（两种跑法都要）
cd server && .venv/bin/python -m unittest discover -s tests -t . -p 'test_*.py'
cd server && .venv/bin/python -m pytest -q
# lint + 类型
cd 仓库根 && pnpm lint
cd 仓库根 && server/.venv/bin/ruff check server && server/.venv/bin/ruff format --check server
cd 仓库根 && pnpm --dir client run typecheck
```

## 交付物（最终汇报格式）

1. 每任务小结：改动清单 + 理由 + 验证结果。
2. 覆盖率对比表：任务 1 基线 → 全部完成后的关键路径数字。
3. 大文件 before/after 行数表（含白名单剩余项）。
4. 未完成/跳过项及原因（尤其是任务 4 的剩余拆分清单）。
5. 对打包路径（electron-builder → server 目录）的影响评估。
