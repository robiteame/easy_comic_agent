# ComicAgent 生成 Agent 架构

`server/agent/graph.py` 已从五节点线性流水线升级为“分析结果 → Critic 反思 → 决策 → 局部恢复 → 续跑”的生成 Agent。自动流程可以局部成功、局部重算和明确降级；任何人工介入都会进入 `waiting_human`，不会伪装成完成。

## 1. 阶段与契约

所有阶段契约位于 `server/agent/contracts.py` 的 `STAGE_CONTRACTS`。每个阶段都有结构化 `StageInput` / `StageOutput`、质量指标、失败分类、检查点键和恢复能力。

| 阶段 | 主要输入 | 主要输出 | 质量指标 | 可恢复检查点 |
|---|---|---|---|---|
| 导演规划 `director_planning` | 剧本、目标时长、风格、预算 | 角色、场景、逻辑问题、导演意图 | schema、人物/场景覆盖、逻辑问题 | `director` |
| 分镜设计 `storyboard_design` | 导演计划、Provider 时长能力 | 镜头列表、时长计划 | 镜头规则、对白密度、连续性 | `storyboard` |
| 素材准备 `asset_preparation` | 角色、场景、参考能力 | 三视图、场景基准图、参考清单 | 参考覆盖、兼容性、版本一致 | `assets` |
| 图像生成 `image_generation` | 逐镜头版本和 Prompt | 逐镜头故事板候选 | 结构、Prompt 对齐、一致性、候选分 | `image:<shot_id>` |
| 质量审核 `quality_review` | 图片候选和生成元数据 | Critic 报告、修改项 | 结构、连续性、对齐、总分 | `quality` |
| 视频生成 `video_generation` | 已审核首帧、音频路由 | 逐镜头视频候选 | 视频有效、时长、运动、连续性 | `video:<shot_id>` |
| 音频制作 `audio_production` | 逐句对白、音色、情绪 | 逐镜头配音/原生音轨 | 对白长度、TTS、音色、混音准备 | `audio:<shot_id>` |
| 剪辑合成 `edit_composition` | 镜头媒体清单、AV 配置版本 | 成片与时间线 | 完整度、AV 同步、总时长、降级数 | `compose` |
| 成片复审 `final_review` | 成片、人工反馈、全部候选 | 发布/局部重算/人工卡点 | 叙事、视觉、音频、节奏、总分 | `final` |

结构定义使用 Pydantic；运行状态使用 `StageStatus` 和 `RunStatus`。非法状态迁移会抛错，例如 `completed -> running` 不允许静默重启。

## 2. Critic/Reviewer

Critic 不只返回成功/失败。`server/agent/critic.py` 会输出：

- 0–1 的评分；
- 带阈值和证据的 `QualityMetric`；
- 带镜头 ID、严重级别和建议的 `CriticIssue`；
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
| 用户中途修改 / 版本冲突 | 恢复检查点 → 只重算变化镜头 → 人工 |
| 预算不足 | 降分辨率/降低成本候选 → 人工 |
| 成片质量不足 | 成片级反馈 → 图片/视频/音频局部重算 → 重新剪辑 → 人工 |

每个候选都包含目标阶段、Prompt 修改、Provider 能力校验、预计成本、预计时长、质量增益、剩余预算适配度和最终得分。被拒绝的候选也会记录原因。

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
| `draft` 草稿 | 0.6× | 1 | 1 | 540p | 仅硬失败、预算耗尽或用户要求 |
| `standard` 标准 | 1.0× | 2 | 2 | 720p | 两轮失败、创意取舍或质量门禁未过 |
| `finishing` 精修 | 1.8× | 3 | 3 | 1080p | 关键镜头和最终成片建议人工确认 |

成本、预计时长、Provider 能力和剩余预算同时参与候选评分。预算未知时明确标记，不按 0 成本伪造结果。

## 7. 可视化追踪 API

- `GET /api/graph/structure`：真实图结构、阶段契约和质量档位；
- `GET /api/graph/runs/{project_id}`：检查点快照、阶段状态和逐镜头产物；
- `GET /api/graph/runs/{project_id}/trace`：决策树、候选、评分、修改原因、最终选择和 Mermaid 时间线；
- `GET /api/graph/runs/{project_id}/decisions`：仅返回决策记录。

Mermaid 输出可直接交给前端绘制；决策记录不含 API Key、供应商原始响应或本地秘密。

## 8. 测试覆盖

`server/tests/test_generative_agent.py` 覆盖：

- 九阶段结构契约和运行状态机；
- LLM 非法输出后的修改 Prompt/切换 Provider/人工决策；
- 图片失败与视频失败的逐镜头局部成功；
- 对白过长、镜头拆句建议；
- Provider 不支持参考图的能力降级决策；
- 用户中途修改、AV 配置变化和版本失效；
- 同版本检查点续跑不重复生成；
- 图结构包含 Critic、Reviewer、恢复和人工卡点节点。
