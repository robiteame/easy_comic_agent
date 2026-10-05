# AI漫剧Agent · 后端技术文档

## 项目概述

AI漫剧Agent 是一套"全流程自动化+轻量化人工干预"的漫剧生产系统。后端基于 **Python FastAPI** 构建，负责脚本解析、分镜生成、故事板出图、配音合成、逐镜头视频生成、成片渲染等核心 AI 流水线。

**核心定位**：输入脚本 → Agent 智能拆解 → 逐镜头生成 → 成片输出

**代码规范**：Python 代码由 Ruff 统一 lint + format（配置在 `server/pyproject.toml`，行宽 120、双引号）。提交前 lefthook 会自动修复暂存文件；CI 中 `ruff check` + `ruff format --check` 为门禁。改动服务端代码后请保持 `ruff check server && ruff format --check server` 通过（本地用 `server/.venv/bin/ruff`）。

**两种运行模式**：
- **手动模式（默认）**：由 `api/routes` 逐步触发（解析→分镜→定稿故事板→逐镜头审核→逐镜头视频→合成），每步之间有人工卡点，进度经 WebSocket 实时推送。
- **自动模式**：由 `agent/graph.py` 的 LangGraph 图一次 `ainvoke` 端到端跑到成片，无人工卡点。图节点**复用手动模式同一批 route 步骤函数**，不重复实现业务逻辑。

> ⚠️ 真实编排在 `api/routes`（手动）与 `agent/graph.py`（自动）两处，二者共用步骤函数。`agent/nodes/` 下的 `script_parser`/`storyboard_gen` 仍被复用，其余早期节点（voice_gen/video_compose/quality_check）为历史遗留，当前流程不直接经过它们。

---

## 生成 Agent（当前自动模式）

`agent/graph.py` 当前是十阶段生成 Agent，旧五节点线性路径只保留兼容入口并委托新图：导演规划 → 分镜设计 → 素材准备 → 图像生成 → 质量审核 → 音频制作 → 视频生成 → 视频检查 → 剪辑合成 → 成片复审。每个阶段都有 process、critic/reviewer、decision、recovery；外部 TTS 严格按“分镜确认 → 音频制作 → 音频检查 → 视频生成 → 视频检查 → 剪辑合成 → 成片复审”执行，native audio 可跳过外部 TTS。支持逐镜头 fan-out/fan-in、幂等检查点、版本校验、局部补拍、Prompt 修改、Provider 切换、降分辨率和显式人工卡点。完整契约、质量档位、恢复策略和追踪 API 见 `docs/GENERATIVE_AGENT.md`。

## 技术栈

| 组件 | 选型 | 用途 |
|------|------|------|
| Web 框架 | FastAPI | REST API + WebSocket |
| Agent 框架 | LangGraph | 自动模式状态图编排（手动模式不经过图） |
| LLM | Mimo（小米 MiMo，OpenAI 兼容；可切 OpenAI/DeepSeek 兜底） | 脚本生成、解析、分镜决策 |
| 图像生成 | Seedream（火山方舟）/ Qwen-Image（阿里云百炼）/ Stability AI / **local 占位 stub** | 角色三视图、场景基准图、定稿故事板 |
| 视频生成 | SeedDance（火山方舟，首帧图驱动） | 逐镜头视频生成 |
| TTS | Mimo 内置 TTS | 角色配音 |
| 视频渲染 | FFmpeg (ffmpeg-python) | 成片合成、转场、字幕、Ken Burns、音频混流 |
| ORM | SQLAlchemy 2.0 | 数据库模型 |
| 数据库 | SQLite | Demo 阶段零运维 |
| 向量检索 | ChromaDB + sentence-transformers | RAG 检索（**当前 requirements 中已注释，暂停用**） |

> 图像生成支持自动匹配：配置了真实 provider 且有对应 API Key 则走云端服务，否则自动回退到 PIL 占位图（`IMAGE_PROVIDER=local`），使全流程在无密钥环境也能端到端跑通。

---

## 目录结构

```
server/
├── main.py                    # FastAPI 入口，注册路由/中间件/WebSocket
├── config.py                  # Pydantic Settings 配置管理
├── requirements.txt           # Python 依赖
├── .env.example               # 环境变量模板
│
├── agent/                     # LangGraph Agent (自动模式执行器)
│   ├── state.py               # AgentState 状态定义 (TypedDict)
│   ├── graph.py               # 自动模式状态图: 节点复用 route 步骤函数
│   ├── nodes/                 # 图节点实现 (仅 script_parser/storyboard_gen 在用)
│   │   ├── script_parser.py   # 脚本解析 (LLM)
│   │   ├── storyboard_gen.py  # 分镜生成 (LLM)
│   │   ├── voice_gen.py       # (历史遗留, 当前流程未直接调用)
│   │   ├── video_compose.py   # (历史遗留)
│   │   └── quality_check.py   # (历史遗留)
│   └── edges/
│       └── conditions.py      # 条件路由逻辑
│
├── services/                  # 外部服务封装
│   ├── llm_service.py         # Mimo/OpenAI 兼容 LLM 调用 (含兜底链)
│   ├── image_service.py       # 图像生成 (Seedream/Qwen-Image/Stability/占位 stub, 角色卡片注入)
│   ├── video_service.py       # SeedDance 逐镜头视频生成 (任务轮询)
│   ├── tts_service.py         # Mimo 内置 TTS 配音
│   ├── ffmpeg_service.py      # FFmpeg 成片合成
│   ├── consistency_service.py # 角色/场景一致性 SOP 注入引擎
│   ├── reference_asset_service.py # 参考图/连续帧/OpenPose/深度图物料化
│   ├── style_templates.py     # 风格模板 (8 套)
│   └── storage_service.py     # 文件存储管理
│
├── memory/                    # 三层记忆系统
│   ├── project_memory.py      # 项目级记忆 (JSON 文件)
│   ├── user_memory.py         # 用户偏好记忆 (JSON)
│   └── memory_manager.py      # 统一管理器
│
├── rag/                       # RAG 检索系统
│   └── rag_service.py         # ChromaDB 文档导入与检索
│
├── api/                       # API 路由
│   ├── websocket.py           # WebSocket 连接管理
│   └── routes/
│       ├── project.py         # 项目 CRUD / 剧集
│       ├── script.py          # 脚本生成/解析 (触发流水线, mode=manual|auto)
│       ├── shot.py            # 镜头管理/故事板出图/审核/逐镜头视频
│       ├── render.py          # 成片渲染导出
│       ├── asset.py           # 素材板 (角色/场景资产绑定)
│       ├── character.py       # 角色资产 CRUD
│       ├── graph.py           # 流程图结构 (由 build_graph 派生)
│       └── chat.py            # 自然语言交互
│
├── models/                    # SQLAlchemy 数据模型
│   ├── base.py                # Base 声明
│   ├── project.py             # 项目模型
│   ├── shot.py                # 镜头模型
│   ├── scene_asset.py         # 场景资产模型 (场景基准图/一致性)
│   └── character.py           # 角色模型
│
├── db/                        # 数据库
│   └── database.py            # SQLite 连接与初始化
│
│   （原 prompts/styles/*.json 风格参数模板已删除：运行时不加载，属于死配置。
│     风格参数的唯一来源是 services/style_templates.py 的内置模板 +
│     data/custom_style_templates.json 的自定义模板）
│
└── data/                      # 运行时数据 (gitignore)
    ├── chromadb/              # ChromaDB 持久化
    ├── checkpoints/           # LangGraph 检查点
    └── comic_agent.db         # SQLite 数据库文件
```

---

## 流水线架构

### 手动模式（默认，route 层逐步触发）

```
POST /api/script/parse (mode=manual) ──▶ [阶段1] 解析剧本 → 生成分镜 → 角色三视图 + 场景基准图 → 落库
                                              │  (WebSocket: complete, asset_board_ready)
                                              ▼  人工确认素材
POST /api/shot/{pid}/generate-storyboard ──▶ 逐镜头生成定稿故事板参考图
                                              │  人工逐镜头审核
POST /api/shot/{sid}/approve-storyboard ───▶ 标记 confirmed
POST /api/shot/{sid}/generate-video ───────▶ 单镜头配音(Mimo TTS) + SeedDance 视频
                                              │  全部镜头完成后
POST /api/render ──────────────────────────▶ FFmpeg 合成成片 (render_complete)
```

每步异步执行（`asyncio.create_task`），进度经 `ws://.../ws/{project_id}` 推送，步骤之间由前端人工卡点。

### 自动模式（LangGraph 端到端）

```
POST /api/script/parse (mode=auto) ──▶ get_graph().ainvoke(state)

START → parse_and_storyboard → generate_storyboard_images
      → auto_approve_storyboard（结构检查 + 质量审核门禁）
      → generate_shot_videos → review_shot_videos（视频质量审核）→ compose → END
```

定义于 `agent/graph.py`，线性串联，任一节点失败即短路至 END（错误经 WebSocket 上报）。**每个图节点都是薄包装，惰性 import 并复用手动模式同一批 route 步骤函数**：

| 自动节点 | 复用的 route 步骤函数 |
|----------|----------------------|
| parse_and_storyboard | `api/routes/script.py::_run_storyboard_phase` |
| generate_storyboard_images | `api/routes/shot.py::_run_storyboard_generation` |
| auto_approve_storyboard | 结构检查（`structural_validation`）+ `services/quality_review_service`（评分/修正重试，全通过才置 confirmed） |
| generate_shot_videos | `api/routes/shot.py::_run_single_shot_video`（逐镜头循环；前置校验故事板质量门禁） |
| review_shot_videos | `services/quality_review_service`（运动/衔接/口型/音画/清晰度，未过不得导出） |
| compose | `api/routes/render.py::_render_task`（前置校验视频质量门禁） |

**质量闭环语义**：结构检查只代表「文件可用」，绝不代表「质量通过」。质量审核由
`QualityReviewService` 驱动（VLM 评分 + 可选身份 embedding + ffprobe/ffmpeg 客观探测）：
不达标镜头按审核修正指令改写 `visual_notes` 后重生成（`quality_retry` 版本来源），
超过 `QUALITY_STORYBOARD_MAX_RETRIES` / `QUALITY_VIDEO_MAX_RETRIES` 仍不达标则置
`needs_review` 转人工；审核能力未配置时裁决为 `unsupported`，自动模式拒绝批准。
每轮审核落库 `quality_reviews`（维度评分/问题/证据/未检测维度/建议），REST 见
`/api/quality-review/*`，降级与未检测必须在界面与日志如实展示。

`/api/graph/structure` 由 `build_graph()` + `GRAPH_NODE_META` 派生，保证可视化与真实自动流程一致。

### AgentState 核心字段（自动模式）

```python
class AgentState(TypedDict):
    project_id: str
    mode: Literal["manual", "auto"]
    initial_state: dict  # 传给阶段1 (_run_storyboard_phase) 的初始 state
    output_format: Literal["9:16", "16:9", "1:1"]
    resolution: str
    current_step: str
    errors: Annotated[list[str], add]  # reducer: 追加; 非空即触发后续节点短路
    # ... 另含手动流程节点 (script_parser/storyboard_gen) 操作的工作字段
    #     (characters / script_scenes / shots / style_params / rag_context 等)
```

---

## 核心模块说明

### 1. 角色一致性保障 (Character Card)

每次图像生成时，从角色卡片强制注入以下参数：

- `base_prompt`: 角色基础英文描述 (发型/发色/瞳色/身材)
- `key_features`: 关键视觉特征列表
- `emotion_variants`: 6 种情绪对应的 prompt 片段 (neutral/happy/shy/sad/angry/surprised)
- `negative_prompt`: 排除不符合特征的描述
- `seed`: 固定随机种子，保证生成一致性

### 2. 记忆系统 (三层架构)

| 层级 | 载体 | 生命周期 | 用途 |
|------|------|---------|------|
| 工作记忆 | LangGraph State | 单次流水线 | 节点间数据传递 |
| 项目记忆 | JSON 文件 | 项目生命周期 | 角色卡片、风格参数、剧情上下文 |
| 用户记忆 | JSON 文件 | 永久 | 用户偏好、常用修正模式 |

### 3. RAG 系统（当前暂停用，依赖已注释）

- **向量库**: ChromaDB (本地持久化)
- **嵌入模型**: all-MiniLM-L6-v2 (sentence-transformers)
- **分块策略**: 剧本按场景切分，小说按滑动窗口 (3段重叠1段)
- **chunk 类型**: scene / dialogue / narrative
- **检索支持**: 按 chunk_type 过滤，按角色名查询相关上下文

### 4. FFmpeg 视频渲染

- **Ken Burns 动效**: 全景缓慢推进、特写轻微抖动、中景静止
- **字幕**: drawtext 滤镜，白色描边，底部居中
- **拼接**: concat demuxer
- **音频混流**: 配音音轨与视频合并
- **输出**: libx264 + aac，支持 720p/1080p/4K

---

## API 接口

### REST API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/project` | 创建项目 |
| GET | `/api/project/{id}` | 获取项目详情 |
| PUT | `/api/project/{id}` | 更新项目 |
| GET | `/api/project` | 项目列表 |
| GET | `/api/project/{id}/episodes` | 剧集列表（系列项目） |
| POST | `/api/script/generate` | LLM 生成完整剧本 |
| POST | `/api/script/parse` | 提交脚本，触发流水线（`mode=manual\|auto`） |
| POST | `/api/script/upload` | 上传脚本文件（支持 `mode` 表单字段） |
| GET | `/api/shot/{project_id}/shots` | 获取项目镜头列表 |
| PUT | `/api/shot/{shot_id}` | 更新镜头（改动失效下游产物） |
| POST | `/api/shot/{shot_id}/regenerate` | 重新生成单个镜头（`candidates=2` 一次生成 2 个候选，各占一版版本历史） |
| POST | `/api/shot/batch-regenerate` | 批量重生成 |
| POST | `/api/shot/{project_id}/generate-storyboard` | 逐镜头生成定稿故事板图 |
| POST | `/api/shot/{project_id}/confirm-storyboard` | 批量确认故事板 |
| POST | `/api/shot/{shot_id}/approve-storyboard` | 单镜头审核通过 |
| POST | `/api/shot/{shot_id}/generate-video` | 单镜头配音 + SeedDance 视频 |
| POST | `/api/render` | 触发成片合成 |
| GET | `/api/render/{project_id}/status` | 渲染进度（内存态，重启丢失） |
| GET | `/api/asset/{project_id}/board` | 素材板（角色 + 场景资产） |
| PUT | `/api/asset/shot/{shot_id}` | 重新绑定镜头的场景/角色资产 |
| GET | `/api/character/{project_id}/characters` | 角色资产列表 |
| PUT | `/api/character/{character_id}` | 更新角色资产 |
| POST | `/api/project/{id}/assets/rebuild` | 画风切换后按当前生效画风重建 stale 的角色三视图 / 场景基准图 |
| GET | `/api/graph/structure` | 自动模式流程图结构（由 build_graph 派生） |
| POST | `/api/chat` | 自然语言交互 |
| GET/PUT | `/api/budget/pricing` | 模型价格配置读取 / 保存 |
| GET/PUT | `/api/budget/config` | 预算配置（项目 / 全局，软 + 硬 + 时长） |
| GET | `/api/budget/status` | 项目预算状态（已用 / 已预留 / 限额 / 等级） |
| POST | `/api/budget/estimate` | 提交前估算（成本 + 预计耗时 + 是否被硬预算拦下） |
| GET | `/api/budget/summary` | 项目 / 剧集统计（已用、剩余预计、各剧集成本） |
| GET | `/api/budget/usage` | 实际用量明细 + 多维分组统计 |
| GET | `/api/budget/jobs/{job_id}` | 单次任务成本明细（含失败 / 取消任务） |

### WebSocket

```
ws://localhost:8011/ws/{project_id}
```

**服务端推送事件：**

```json
{"type": "progress", "step": "parse_script", "progress": 20, "message": "..."}
{"type": "complete", "project_id": "...", "title": "根据剧本自动命名的项目标题", "shots": [...], "asset_board_ready": true}
{"type": "shot_update", "shot_id": "...", "status": "video_done", "image_path": "...", "storyboard_path": "...", "video_path": "..."}
{"type": "storyboard_ready", "project_id": "..."}
{"type": "render_complete", "video_url": "...", "duration": 30}
{"type": "error", "error_type": "pipeline_error|storyboard_error|shot_video_error|render_error", "message": "简短中文提示", "error_id": "8位十六进制"}
```

> 错误事件不携带堆栈、本地路径或供应商原始响应：服务端用 `services/error_reporter.py` 记录完整异常（含脱敏后的堆栈）并生成短错误编号，前端只需展示 `message` 与 `error_id`。

进度 `step` 取值与流程对应：`parse_script` / `generate_storyboard` / `wait_asset_confirm` / `generate_storyboard_images` / `wait_storyboard_approval` / `generate_voice` / `generate_seedance_video` / `rendering`。

**客户端消息：**

```json
"ping"  → 服务端回复 {"type": "pong"}
```

### 任务启动 Provider 预检

所有会消耗模型能力的任务入口在抢占任务槽位前先做配置预检（`api/provider_guard.py`
调 `services/provider_readiness.py`），缺配置时任务根本不启动：HTTP 409 +
`detail.error_code=provider_not_configured`，`detail.missing` 逐项列出缺失端点。

| 任务入口 | 预检的 job_type | 必需端点 |
|----------|----------------|----------|
| `POST /api/script/parse`、`/upload`（手动模式） | script_pipeline | script（LLM） |
| 同上（auto 模式） | script_pipeline | script + video + voice |
| `POST /api/shot/{id}/generate-video` | shot_video | video + voice（见下方豁免） |
| `POST /api/shot/{id}/generate-audio` | shot_audio | voice（纯 TTS 任务） |
| 任务中心「重试 / 续跑」（`services/job_actions.py`） | 按任务类型映射 | 同上对应规则 |

- **script**：主/备端点至少一个配置了 API Key（备端点需与主端点不同址）；
- **video**：必须配置 API Key（视频没有本地回退）；
- **image**：永不拦截 —— 缺 Key 自动回退占位图（既有约定）；
- **video 预检明细**：除 API Key 外还校验协议已注册适配器、`model` 非空、适配器声明
  `reference_image=True`（本流水线是首帧图生视频）。任一不满足即 409；启动日志由
  `video_provider_preflight()` 如实打印协议 / 模型 / 参考图模式 / 固定时长 / 原生音频，不含密钥；
- **voice 豁免**：视频适配器具备真正可用的原生音频能力
  （`native_audio` + `production_ready`，如接入 Veo 3 / Sora 2 厂商实现后）时
  不强制要求 TTS；镜头无台词、或镜头级 `audio_mode=native` 时同样豁免；
  镜头级显式 `audio_mode=tts` 仍要求配置语音端点。

前端用 `notifyProviderBlocked` 解析同一 409 结构并引导去「系统设置 → 模型服务」。

---

## 成本与时长预算

### 记账口径（硬约束）

1. **金额只允许整数**：统一用「货币最小单位的整数倍」表示，本项目管理到 10^-6 个货币单位
   （`MICRO_PER_UNIT = 1_000_000`，即 1 CNY = 1_000_000 micro）。所有换算走
   `services/pricing_service.py` 的整数运算（`ceil_div`，向上取整到 1 micro），
   **任何一步都不得引入浮点金额**；API 收到浮点金额一律 400。
2. **未知就是未知**：没有可用价目（未配置 `configured=true` 的价目行、或 provider 未注册）
   时 `cost_known=false` 且 `cost_micro=NULL`，前端显示「成本未知」；绝不按 0 元或猜测
   价格入账。本地零成本能力（占位图 `placeholder`、本地 `ffmpeg`）单独标为
   `cost_source=local`，是「确定不花钱」而不是「不知道」。
3. **估算与实际分表**：启动前估算写 `cost_estimates`（`services/budget_service.estimate_job`），
   每次真实调用写 `usage_records`（`services/usage_service`），互不覆盖。
4. **一次调用只入账一次**：`usage_records.usage_key` 唯一 + 事务内先查后插，
   并发/重复回调不会重复计费。
5. **失败与取消同样留痕**：调用失败记 `status=failed`；只有供应商真的回报过用量
   （`source=provider`）或本地实测（`source=local`）才保留数量，按请求推导的数量
   一律丢弃（未知），保证「任务失败后仍能查询已发生的调用成本」且不虚增金额。
6. **不落敏感信息**：用量与估算表不保存 API Key、供应商原始响应或本地绝对路径。

### 用量归属（contextvar）

`task_registry.start()` 会把任务的用量上下文（`project_id` / `series_id` / `shot_id` /
`job_key` / `job_id` / `job_type`）绑定到协程上下文，服务层（LLM / 图像 / 视频 / TTS /
FFmpeg）通过 `usage_service.current_scope()` 读取，无需层层透传参数。任务进入终态时
`task_registry.finish()`（含取消、服务重启中断路径）会调用 `usage_service.finalize_job`
标记 `job_status` 并释放预算预留。

### 预算执行

- 预算作用域：项目 > 父系列 > 全局（`budget_service.effective_limits`）；
- **软预算**：超出只返回提示（`budget_soft_exceeded` + 响应里的 `budget_warning`），任务照常执行；
- **硬预算**：`task_registry.claim_job` 在抢占前比对「已用 + 已预留 + 本次估算」，
  超限直接拒绝并返回 `error_code=budget_exceeded`（HTTP 409），任务不会被创建；
- **并发安全**：抢占前按估算金额写一条 `budget_reservations`（`reservation_key` 唯一），
  任务终结时释放，避免两个并发任务各自以为额度充足；
- **估算未知不拦**：没有单价时金额未知，只跳过金额校验并在状态里标注，已用金额本身
  超过硬预算时仍然拦截；
- 时长为第二维度，口径是「任务执行时长」（`background_jobs` 真实起止时间之和），
  与供应商计费秒数（`usage_records.quantity`）分开统计。

### 关键文件

| 文件 | 职责 |
|------|------|
| `models/pricing.py` / `models/usage.py` / `models/budget.py` | 价目、用量、估算、预算与预留表 |
| `services/providers/usage.py` | 统一 `UsageMetadata` 与适配器用量钩子（未实现钩子的适配器安全回退为「未知」） |
| `services/pricing_service.py` | 整数金额运算、价目匹配与读写、出厂模板（全部未配置金额） |
| `services/usage_service.py` | 用量上下文、记账、统计、任务终结标记 |
| `services/budget_service.py` | 预算配置、任务估算、预算检查与预留、项目/剧集汇总 |
| `api/routes/budget.py` | 成本 / 预算 / 统计 API |
| `api/claim_guard.py` | 路由层抢占守卫：预算拦截转 409、软预算提示透传 |

### 测试

```bash
cd server
python -m pytest tests/test_usage_accounting.py tests/test_budget_enforcement.py tests/test_budget_api.py tests/test_usage_integration.py -q
```

覆盖：适配器统一用量元数据、未知 provider 不伪造金额、整数金额运算、估算与实际分表、
失败/取消留痕、重复与并发不重复计费、硬预算阻止任务启动（真实路由 409）、预留释放、
项目/剧集/镜头/任务级统计、以及不泄漏密钥与供应商原始响应。

---

## 环境变量

复制 `.env.example` 为 `.env` 并填写：

```bash
# LLM —— 默认 Mimo (小米 MiMo, OpenAI 兼容)
LLM_PROVIDER=mimo            # mimo / openai / deepseek
MIMO_API_KEY=                # 留空 = 未配置（不要填占位值, 否则会被当成已配置）
# OPENAI_API_KEY=           # 作为兜底/可选（留空 = 未配置）
# OPENAI_BASE_URL=

# 图像生成 —— 默认 local 占位 stub (无需 key 即可跑通)
IMAGE_PROVIDER=local         # local(占位图) / stability / ark-seedream
# 火山方舟 (Seedream 图像 + SeedDance 视频, 任一 ARK/SEEDDANCE/SEEDREAM key 均可)
ARK_API_KEY=                 # 留空 = 未配置
# STABILITY_API_KEY=         # 若用 Stability

# TTS —— 固定使用 Mimo 内置 TTS (TTS_PROVIDER 为历史字段, 不再切换)
```

> 缺少对应 provider 的 API Key 时，图像生成会自动回退到占位图，不会中断流程。

---

## 启动方式

```bash
cd server

# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 4. 启动服务 (默认端口 8011, 可用 PORT 环境变量覆盖)
python main.py
# 或
uvicorn main:app --host 0.0.0.0 --port 8011 --reload
```

服务启动后：
- API 文档: http://localhost:8011/docs
- 健康检查: http://localhost:8011/health
- 输出文件: http://localhost:8011/output/{path}

---

## 数据库模型

### Project

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| title | str | 项目名称 |
| genre | str | 类型 (甜宠/悬疑/古风/都市) |
| style | str | 风格 (见风格模板，8 套之一) |
| status | str | draft/assets_ready/storyboard_ready/storyboard_approved/rendering/completed/error |
| input_text / input_type | str | 原始输入及类型 |
| output_format | str | 9:16 / 16:9 / 1:1 |
| resolution | str | 720p / 1080p / 4k |
| platform | str | douyin / kuaishou / bilibili / custom |
| consistency_config | JSON | 项目级一致性配置 |
| project_type / parent_project_id / episode_number | str/int | 系列剧集支持（剧集挂在父项目下，资产复用父项目） |

> 项目树不变量：系列（`series`）是根节点（无父级），剧集（`episode`）必须挂在真实存在的系列下，禁止自引用与负数集号。约束由 CHECK（新库）+ SQLite 触发器（存量库，见 `db/database.py`）+ API 层校验（`api/routes/project.py::_resolve_project_tree`）三层保证；启动时 `init_db()` 会先修复历史脏数据再补约束。

### Shot

镜头模型字段较多（45+），除基础字段外含大量一致性/连续性追踪字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| id / project_id / sequence | | 主键 / 所属项目 / 顺序号 |
| shot_type | str | wide/medium/close-up/extreme_close |
| scene_description / character_action / dialogue | str | 画面/动作/台词 |
| camera_angle / camera_movement | str | 机位 / 运镜 |
| duration / emotion / transition | | 时长 / 情绪 / 转场 |
| image_path / storyboard_path / audio_path / video_path / last_frame_path | str | 各阶段产物路径 |
| status | str | pending/storyboard_done/storyboard_approved/video_generating/video_done/failed/needs_review |
| storyboard_status / confirmed / version | | 故事板状态 / 审核确认 / 版本号 |
| scene_asset_id / scene_group_id / character_asset_ids | | 关联场景/角色资产 |
| consistency_context / reference_weights / continuity_profile | str/JSON | 一致性 SOP / 参考权重 / 连续性配置 |
| continuity_reference_path / pose_reference_path / depth_reference_path | str | 连续帧 / OpenPose / 深度图 |

### Character

| 字段 | 类型 | 说明 |
|------|------|------|
| id / project_id / name | | 主键 / 所属项目 / 角色名 |
| visual_prompt / negative_prompt | str | 英文绘画 prompt / 排除描述 |
| appearance / personality / default_outfit | | 外观 / 性格 / 默认服装 |
| emotion_variants / key_features | JSON | 情绪→prompt 映射 / 关键视觉特征 |
| reference_images | JSON | 已确认的角色三视图参考 |
| lora_profile / ip_adapter_profile / wardrobe_lock | str | 一致性锁定标识 |
| voice_id | str | Mimo 音色 |
| seed | str | 固定随机种子 |

### SceneAsset（场景资产）

| 字段 | 类型 | 说明 |
|------|------|------|
| id / project_id / name | | 主键 / 所属项目 / 场景名 |
| description / visual_prompt / negative_prompt | str | 场景描述与 prompt |
| scene_group_key / time_of_day | str | 场景分组键（地点-时段，锁定光照）/ 时段 |
| baseline_image_path / reference_images | str/JSON | 场景基准图 |
| consistency_profile / prop_lock | JSON/str | 一致性配置 / 道具锁 |
| key_features | JSON | 关键特征 |
| seed | int | 固定随机种子（注意：与 Character.seed 为 str 不同） |

---

## 风格模板

`services/style_templates.py` 内置 8 套预置风格（`STYLE_TEMPLATES`）：

| key | label |
|------|------|
| anime | 日系写实漫 |
| chinese | 国漫厚涂 |
| chibi | 简约条漫 |
| realistic | 电影写实 |
| watercolor | 水彩绘本 |
| ink | 新国风水墨 |
| noir | 悬疑电影感 |
| clay | 定格黏土 |

每套含 prompt_prefix、negative_prompt、style_label、scene_baseline_prompt、character_reference_prompt 等字段，由 `style_prompt_params()` 注入生成流程。

---

## 依赖说明

### 系统依赖

- **Python** >= 3.11
- **FFmpeg** (需要在 PATH 中)
  - Windows: 从 https://ffmpeg.org/download.html 下载，添加到系统 PATH
  - 验证: `ffmpeg -version`

### Python 依赖

1. **Web 框架**: fastapi, uvicorn, pydantic, pydantic-settings
2. **AI/Agent**: langgraph, langchain, langchain-openai, openai
3. **媒体处理**: ffmpeg-python, Pillow, edge-tts
4. **工具库**: httpx, aiohttp, aiofiles, python-docx, python-multipart

> **RAG 现状**：`chromadb` 与 `sentence-transformers` 在当前 `requirements.txt` 中被注释（跳过 grpcio 编译），RAG 检索功能暂停用；`rag/rag_service.py` 代码保留，启用时需取消注释并安装。

---

## 开发注意事项

1. **sys.path**: `main.py` 开头会将 server 目录加入 sys.path，确保模块导入正确
2. **运行模式**: `mode=manual`(默认) 由 route 逐步触发；`mode=auto` 经 `agent/graph.py` 端到端跑。二者**复用同一批 route 步骤函数**，新增/修改流程逻辑只需改一处
3. **异步执行**: 流水线通过 `asyncio.create_task()` 异步执行，不阻塞 API 响应
4. **WebSocket 进度**: 通过 `ws_manager` 向项目的所有连接推送进度
5. **图像 provider 回退**: 无对应 API Key 时自动用占位图，不中断流程（`IMAGE_PROVIDER=local`）
6. **重复解析保护**: 项目已有已确认/已出片镜头时再次 `/parse` 会被拒绝，避免误删既有成果
7. **角色一致性**: 图像生成时必须注入角色卡片 (Character Card)，不可省略 key_features
8. **FFmpeg 中文字幕**: Windows 需指定 `font=Microsoft YaHei`，Linux 需安装中文字体
9. **统一画风来源**：项目 / 本次请求的 `style` 是唯一权威来源；Skill 只有显式
   `style_override_enabled=true` 才能覆盖。全链路（script→storyboard→image→video）
   统一经 `skill_config_service.resolve_effective_style()` 解析，接口 / WebSocket / 日志
   返回 `requested_style` / `effective_style` / `style_source`；`_persist_phase1` 不得把
   anime 写回项目。
10. **参考图能力如实上报**：`references_validated`（文件存在并已读取）与
    `references_sent`（实际进入请求载荷）必须分开记录；没有真实姿态/深度模型时
    `openpose_lock`/`depth_lock` 一律标 `unsupported`，不产出伪造控制图；
    LoRA / IP-Adapter 未接入，字段保持空串，界面显示「未接入」。
11. **结构检查 ≠ 质量认证**：`services/structural_validation.py` 只做图片/视频可读性、
    尺寸、时长、首尾帧与空文件检查，不含人脸一致性 / 美学 / 闪烁检测；界面与日志
    统一显示「结构检查（仅结构，非质量认证）」。质量结论只能来自
    `services/quality_review_service.py`（StructuralCheck 之上真实质量门禁），
    能力未配置时其维度状态为 `unsupported`，绝不计为通过。
    视频检查固定分三类：`structural_validity`（可播放/轨道/时长 sane）、
    `technical_quality`（时长-计划/分辨率/比例/黑帧/**空帧**/冻结/音画时长/尾帧/文件过小）、
    `visual_quality_pending`（首帧与故事板相似度、参考匹配、运动稳定、镜头连续、
    动作完成——五项**恒为 pending**，无视觉模型时不得伪造成 passed）。
    显式 `mode=auto` 在视觉能力缺失时由 `QUALITY_VISUAL_PENDING_POLICY` 收口：
    `continue`（默认）改按结构 + 技术门禁自动继续、成片标 `degraded` 并把
    `visual_quality_pending` 写进最终报告的未解决风险；`block` 保持 fail-closed 明确终止。
    两条路径都不写 `needs_human_review`、都不进 `human_gate`/`waiting_human`；
    结构或技术不合格时一律拦截。manual 与未携带 `mode` 的兼容入口不受该开关影响。
