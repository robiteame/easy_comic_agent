# ComicAgent 生成 Agent 架构

`server/agent/graph.py` 已从五节点线性流水线升级为“分析结果 → Critic 反思 → 决策 → 局部恢复 → 续跑”的生成 Agent。自动流程可以局部成功、局部重算和明确降级；自动模式永不路由人工节点；只有 `mode=manual` 且显式设置 `human_gate_policy=manual` 才能进入 `human_gate`。

## 1. 阶段与契约

所有阶段契约位于 `server/agent/contracts.py` 的 `STAGE_CONTRACTS`。每个阶段都有结构化 `StageInput` / `StageOutput`、质量指标、失败分类、检查点键和恢复能力。

| 阶段 | 主要输入 | 主要输出 | 质量指标 | 可恢复检查点 |
|---|---|---|---|---|
| 导演规划 `director_planning` | 剧本、目标时长、风格、预算 | 角色、场景、逻辑问题、导演意图 | schema、人物/场景覆盖、逻辑问题 | `director` |
| 分镜设计 `storyboard_design` | 导演计划、Provider 时长能力 | 镜头列表、时长计划 | 镜头规则、对白密度、连续性 | `storyboard` |
| 素材准备 `asset_preparation` | 角色、场景、参考能力 | 三视图、场景基准图、参考清单 | 参考覆盖、兼容性、版本一致 | `assets` |
| 图像生成 `image_generation` | 逐镜头版本和 Prompt | 逐镜头故事板候选 | 结构、Prompt 对齐、一致性、候选分 | `image:<shot_id>` |
| 质量审核 `quality_review` | 图片候选和生成元数据 | Critic 报告、修改项 | 结构、连续性、对齐、总分 | `quality` |
| 音频制作 `audio_production` | 分镜确认、逐句对白、音色、情绪 | 外部 TTS 配音或 native audio 跳过记录 | 对白长度、TTS、音色、混音准备 | `audio:<shot_id>` |
| 视频生成 `video_generation` | 已审核首帧、已准备音频/native audio 路由 | 逐镜头视频候选 | 视频有效、时长、运动、连续性 | `video:<shot_id>` |
| 视频检查 `video_review` | 逐镜头视频候选与执行计划 | 结构/技术/视觉待审报告 | 结构、技术质量、视觉待审 | `video_review` |
| 剪辑合成 `edit_composition` | 镜头媒体清单、AV 配置版本 | 成片与时间线 | 完整度、AV 同步、总时长、降级数 | `compose` |
| 成片复审 `final_review` | 成片、人工反馈、全部候选 | 发布/局部重算/人工卡点 | 叙事、视觉、音频、节奏、总分 | `final` |

结构定义使用 Pydantic；运行状态使用 `StageStatus` 和 `RunStatus`。非法状态迁移会抛错，例如 `completed -> running` 不允许静默重启。

## 2. Critic/Reviewer

Critic 不只返回成功/失败。`server/agent/critic.py` 的每份 `CritiqueReport` 都输出：

- `stage` 与 0–1 的 `score`；
- 带阈值和证据的 `QualityMetric`；
- 带镜头 ID、严重级别和建议的 `CriticIssue`；
- `evidence`：结构化证据（指标值/阈值/通过位，问题细节与产物路径）；
- `failure_kind`：稳定失败分类（LLM_INVALID_OUTPUT、IMAGE_FAILED、VIDEO_FAILED、AUDIO_FAILED、DIALOGUE_TOO_LONG、SHOT_TOO_COMPLEX、PROVIDER_REFERENCE_UNSUPPORTED、PROVIDER_CAPABILITY_MISMATCH、QUALITY_BELOW_THRESHOLD、VERSION_CONFLICT、BUDGET_EXCEEDED、STORAGE_FAILED、TIMEOUT 等）；
- `recoverable`：该分类是否能靠再花一次生成成本解决（预算耗尽/用户改稿/取消为不可恢复）；
- `affected_shot_ids`：受影响镜头清单；
- `recommended_strategy`：Critic 层建议策略（决策节点仍会按预算/能力/剩余次数重新评估）；
- 可执行的 `proposed_changes`，例如“把动作拆成 2–5 秒短镜头”“替换为上一镜尾帧”“只重生成镜头 s003”。

LLM 非法 JSON、Schema 错误和字段越界不会被当成普通布尔失败。`agent/output_schemas.py` 先归一化/丢弃不可用条目；仍无法使用时，Critic 生成 Prompt 收紧建议并进入恢复决策。

## 3. 恢复策略

`server/agent/decision.py` 根据失败类别生成候选，而不是固定重试一次：

| 失败类别 | 默认候选顺序 |
|---|---|
| LLM 非法输出 | 修改 Prompt → 切换 Provider → 人工 |
| 对白过长 | 拆分镜头/拆句 → 修改 Prompt → 人工 |
| 镜头动作过复杂 | 拆分镜头 → 合并连续短镜 → 修改 Prompt → 人工 |
| 图片失败 | 修改 Prompt → 切换 Provider → 替换参考 → 降分辨率 → 只重生成失败镜头 → 人工 |
| 视频失败 | 修改 Prompt → 切换 Provider → 替换参考 → 降分辨率 → 只补拍失败镜头 → 人工 |
| Provider 不支持参考图 | 切换支持参考图的 Provider → 替换参考 → 修改 Prompt → 人工 |
| Provider 能力不匹配 | 切换 Provider → 降分辨率 → 修改 Prompt → 人工 |
| 用户中途修改 / 版本冲突 | 恢复检查点 → 只重算变化镜头 → 人工 |
| 预算不足 | 降级发布 / 明确终止（不再花生成成本） |
| 成片质量不足 | 修改 Prompt → 只重生成失败镜头 → 替换参考 → 拆分镜头 → 人工 |
| 超时 / 存储抖动 | 重试（唯一允许 RETRY 的瞬时错误）→ 切换 Provider → 修改 Prompt |

每个候选都包含目标阶段、结构化 Prompt 补丁（`PromptPatch`：字段级 `field/op/value/reason`，例如 `dialogue` `split` `{"max_chars": 90}`）、Provider 能力校验、预计成本、预计时长、质量增益、剩余预算适配度、剩余重试次数和最终得分。选择按「失败类别首选策略 + 综合评分」取最高且预算允许者。

硬性规则：

- 禁止无条件“重试一次”：`RETRY` 只在失败证据表明瞬时错误（超时、存储抖动、网络瞬断）时成为候选；UNKNOWN 失败先换 Provider / 收紧 Prompt。
- 自动模式永不选择 `human_review`；只有 `mode=manual` 才允许。
- 自动模式恢复无法继续时：存在可用部分结果（成功候选或质量分 > 0）→ `degraded_publish`（图节点 `degraded_publish`，按降级结果结束并保留失败清单）；否则 → `terminal_failure`（`auto_abort` 明确失败）。

被拒绝的候选全部记录原因（能力不匹配 / 超出剩余预算 / 自动模式禁止人工 / 剩余恢复次数为 0 / 综合排序落后），连同最终选择写入 `DecisionTrace`（含模式、质量分、剩余次数、预算快照与 Provider 画像）。

## 4. 逐镜头 fan-out/fan-in

`server/agent/shot_work.py` 为图片、视频和音频提供逐镜头 worker：

1. 读取每个镜头的 `Shot.version`；
2. 检查同版本、同阶段检查点；产物存在且结构有效才复用；
3. 逐镜头独立执行；
4. 单镜头异常只写该镜头的失败分类；
5. fan-in 同时保留 `successes`、`failures`、`degraded` 和 `skipped`。

因此镜头 A 失败不会回滚镜头 B 的成功产物。恢复队列默认只包含失败或低分镜头。

## 5. 幂等、版本与续跑

检查点位于 `settings.CHECKPOINT_PATH/<project_id>/<run_id>.json`，由 `CheckpointStore` 原子写入。检查点包含：

- 输入/输出指纹；
- 阶段状态和可恢复 payload；
- 每个镜头的 `shot_version`、产物路径、Provider、成本、耗时和评分；
- 每次决策树、候选、修改原因和最终选择；
- 人工介入、降级和用户变更事件。

恢复规则：

- 阶段输入指纹一致且状态为 `succeeded/degraded/skipped` 时跳过重算；
- 镜头产物必须同时匹配 `Shot.version` 和文件结构检查；
- 用户修改会推进 `Shot.version` 或 `av_config_version`，只失效受影响镜头及下游阶段；
- 断点续跑只处理失效/缺失部分；
- 渲染发布前再次校验媒体清单和 AV 配置版本。

## 6. 自动质量档位

`quality_profile` 可通过 `/api/script/parse` 或上传剧本选择：

| 档位 | 成本倍率 | 候选数 | 自动恢复轮数 | 输出分辨率 | 人工介入 |
|---|---:|---:|---:|---|---|
| `draft` 草稿 | 0.6× | 1 | 1 | 540p | 自动模式禁用人工路由 |
| `standard` 标准 | 1.0× | 2 | 2 | 720p | 自动模式禁用人工路由 |
| `finishing` 精修 | 1.8× | 3 | 3 | 1080p | 自动模式禁用人工路由 |

成本、预计时长、Provider 能力和剩余预算同时参与候选评分。预算未知时明确标记，不按 0 成本伪造结果。

### 6.1 三个质量分类与视觉待审收口

`services/structural_validation.py` 把每个视频产物拆成三个分类，`critique_videos` 原样转成
Critic 证据，任何界面/日志都不得把结构或技术通过说成视觉质量通过：

| 分类 | 检测项 | 缺失能力时的状态 |
|---|---|---|
| `structural_validity` | 文件存在/可解码播放、视频轨、期望时的音频轨、时长 sane | 确定性可测，不依赖外部能力 |
| `technical_quality` | 时长-计划偏差、分辨率、画幅比例、黑帧、**空帧**、长时间冻结、音画时长不一致、尾帧提取、输出文件过小 | 确定性可测（ffprobe/ffmpeg） |
| `visual_quality_pending` | 首帧与故事板相似度、角色和场景参考匹配度、运动稳定度、镜头连续性、动作完成度 | **恒为 pending**，永不 passed |

- 五项视觉维度逐项输出 `passed=None` 的 `QualityMetric`，并作为 `visual_pending:<key>` 逐镜头
  登记到 Critic issue；`_finalize_video_result` 会强制把视觉分类归位为 `pending`，调用方无法改写。
- **空帧**（纯黑/纯白/纯灰/纯色底）由 `signalstats` 的逐帧亮度动态范围（`YMAX-YMIN <= 12`）
  独立判定，与 `blackdetect` 的黑帧去重，不会把纯白/纯灰占位画面误判为正常。
- 没有真实视觉模型时，视觉质量既不通过也不失败：如实记 pending，并进入最终报告的
  `unresolved_risks`（带 `shot_ids` 与 `dimensions`），成片的 `final_choice.degraded` 为 true。

### 6.2 无视觉模型时的自动收口

`QUALITY_VISUAL_PENDING_POLICY` 决定显式 `mode=auto` 在视觉能力缺失时的收口方式：

- `continue`（默认）：改用**可测量**的结构 + 技术门禁收口，自动继续跑完成片并按 `degraded`
  发布。成片带 `visual_quality_pending` 未解决风险，绝不冒充「视觉质量通过」。
- `block`：保持旧的 fail-closed 语义，能力缺失即明确终止（仍然不进入人工卡点）。

约束与边界：

- 只有显式 `mode=auto` 受该开关影响；manual 与未携带 `mode` 的兼容入口一律保持原有语义。
- 结构或技术不合格时**一律拦截**（含 auto 模式）：自动继续不是无条件放行。
- 自动模式全程不写 `needs_human_review`、不进入 `human_gate`/`waiting_human`；参考素材阻塞时
  项目状态只写 `degraded`（`refresh_project_reference_state(..., allow_needs_review=False)`），
  不写 `needs_review`。
- 门禁读的是审核记录：`quality_review_service._gate_status` 仅当调用方显式传
  `allow_visual_pending=True` 时，才把 `verdict=unsupported` 的镜头按结构门禁降级放行，
  并在返回的 `degraded` 里逐镜头列出，绝不写成 `passed`。

## 7. 可视化追踪 API

- `GET /api/graph/structure`：真实图结构、阶段契约和质量档位；
- `GET /api/graph/runs/{project_id}`：检查点快照、阶段状态和逐镜头产物；
- `GET /api/graph/runs/{project_id}/trace`：决策树、候选、评分、修改原因、最终选择、逐镜头时间线和 Mermaid；响应中的 `summary` 是面向展示的稳定汇总（见下）；
- `GET /api/graph/runs/{project_id}/decisions`：仅返回决策记录。

Mermaid 输出可直接交给前端绘制；决策记录不含 API Key、供应商原始响应或本地秘密。

`trace.summary`（`agent.checkpoints.summarize_trace`）集中提供前端与审计需要的全部字段：

- 当前阶段（`run.current_stage`，来自持久化 `stage_entered` 事件，进程重启后仍可还原）与运行状态/终止原因（`auto_abort`、`degraded_publish`、`human_gate` 终态节点写入 `set_status` 与事件，自动流程完成时由入口写入 `completed`）；
- 镜头状态（逐镜头各阶段 status/Provider/模型/成本/实际耗时/失败分类，合并 DB 镜头版本与 `storyboard/video reference manifest` 即「实际发送参考图」）；
- 阶段质量分与 Critic 问题（`stages[].quality`）、恢复候选与最终决策（`decisions[].candidates/selected/considered_rejected`，含预计成本与预计时长）、Prompt 修改（`prompt_changes`，字段级补丁）、候选结果（`shots[].video_candidates`，含自动选择原因与结构检查结论）；
- 自动降级原因（`degradations`）与检查点/恢复次数（`counters`）、成本合计（`totals.cost_micro`，未知记 null 不伪造 0）。

客户端消费方：`FlowGraph`（十阶段进度条 + 可展开追踪）与任务中心详情的「Agent 追踪」（`AgentTracePanel` + `agentTraceModel` 纯函数模型）。

## 8. 测试覆盖

`server/tests/test_generative_agent.py` 覆盖：

- 十阶段结构契约、每阶段 process/critic/decision/recovery 图结构和运行状态机；
- 外部 TTS 与 native audio 执行依赖；
- 自动模式不进入人工节点，manual 模式显式 `human_gate_policy=manual`；
- 成片反馈路由回图像、音频、视频或剪辑；
- LLM 非法输出后的修改 Prompt/切换 Provider/人工决策；
- 图片失败与视频失败的逐镜头局部成功；
- 对白过长、镜头拆句建议；
- Provider 不支持参考图的能力降级决策；
- 用户中途修改、AV 配置变化和版本失效；
- 同版本检查点续跑不重复生成；
- 图结构包含 Critic、Reviewer、恢复和人工卡点节点。

`server/tests/test_agent_e2e_integration.py` 以 Mock Provider 覆盖十三个端到端场景：LLM 非法 JSON、图片/视频 Provider 失败、对白超时长（阶梯修正→拆镜）、Provider 不支持参考图（修正→替换参考→切换 Provider 全链路追踪）、单镜头失败其余成功、用户修改后的版本冲突与 `resume_checkpoint`、进程退出后断点续跑、候选结构自动选择、自动模式完成成片、自动模式失败终止于 `auto_abort` 绝不进 `human_gate`、预算耗尽明确终止/降级发布、成片复审反馈只局部重算受影响镜头。

`server/tests/test_agent_trace_summary.py` 锁定 `trace.summary` 的展示字段契约（当前阶段、镜头状态、质量分、Critic 问题、决策/候选/淘汰原因、Prompt 修改、参考图、成本、耗时、降级原因、检查点与恢复计数）。
