# 视频工作流 A/B 评估工具

本工具只做视频策略的离线对比报告，不参与线上生成编排。它不会调用视频
Provider，也不会根据少量样本自动修改默认 Provider、质量阈值或其它生成策略。

## 比较范围

固定比较四个策略：

| ID | 策略 | 模式 key |
|---|---|---|
| A | 当前首帧 I2V | `first_frame_i2v` |
| B | 首帧 I2V + 条件连续性参考 | `first_frame_i2v_plus_conditional_continuity` |
| C | 多参考 R2V | `multi_reference_r2v` |
| D | 首尾帧或视频续写 | `first_last_frame_or_video_continuation` |

结果固定按以下五类镜头分组：对白、近景、普通动作、复杂动作、跨场景。

## 输入契约

输入是一个 JSON 文件。顶层必须声明同一批镜头；每个策略结果必须完整覆盖
这批镜头，且同一镜头的四个策略必须使用相同的 `execution_plan_hash`。

```json
{
  "schema_version": 1,
  "evaluation_id": "video-ab-001",
  "batch_id": "batch-001",
  "shots": [
    {
      "shot_id": "shot_001",
      "shot_type": "dialogue",
      "execution_plan_hash": "plan-shot_001-v1"
    }
  ],
  "results": [
    {
      "shot_id": "shot_001",
      "strategy_id": "A",
      "batch_id": "batch-001",
      "execution_plan_hash": "plan-shot_001-v1",
      "provider": "example-provider",
      "model": "example-video-model",
      "reference_manifest": [
        {
          "type": "approved_storyboard_first_frame",
          "role": "first_frame",
          "path": "path/to/first.png",
          "used": true
        }
      ],
      "cost": {
        "cost_known": true,
        "cost_micro": 120000,
        "currency": "CNY"
      },
      "elapsed_ms": 4200,
      "output_path": "path/to/shot_001_A.mp4",
      "metrics": {
        "structural_passed": true,
        "first_frame_match_score": 0.82,
        "tail_frame_valid": true,
        "frozen": false,
        "freeze_seconds": 0.0,
        "output_duration_s": 4.8
      },
      "manual_review": {
        "status": "accepted",
        "accepted": true,
        "note": ""
      }
    }
  ]
}
```

约束：

- `shot_type` 只接受 `dialogue` / `close_up` / `ordinary_action` /
  `complex_action` / `cross_scene`，同时兼容文档中的中文名称。
- `results` 必须恰好包含 `shots × A/B/C/D` 的完整矩阵，不允许缺失、重复或额外项。
- `cost.cost_micro` 使用整数 micro；成本未知时设置 `cost_known: false` 并把
  `cost_micro` 留空，工具不会用 0 冒充未知成本。
- `first_frame_match_score` 是 0..1 分数。缺少的指标保持为空并单列覆盖率，
  不会被计为通过。
- `manual_review.status` 接受 `accepted` / `rejected` / `pending` / `unknown`。
- `execution_plan_hash` 可由 `execution_plan.recipe_hash` 提供，也可以对完整
  `execution_plan` 对象做稳定 SHA-256 指纹。

## 运行

```bash
python server/scripts/video_ab_evaluate.py \
  --input server/tests/fixtures/video_ab_evaluation/fixture.json \
  --output-dir output/video-ab-evaluation
```

输出：

- `video_ab_evaluation.json`：完整逐条执行元数据、五项指标、策略总体汇总和五类镜头分组。
- `video_ab_evaluation.md`：人工可读的总体对比、分组对比、执行审计和策略安全声明。

相同输入重复运行时，两份报告逐字节一致；报告中没有生成时间等易变字段。

## 指标口径

- 结构通过率：结构通过数 / 已检测结构样本数。
- 首帧匹配度：已评分样本的 0..1 分数均值，并保留最小值、最大值和覆盖率。
- 尾帧有效率：尾帧有效数 / 已检测尾帧样本数。
- 冻结率：检测到冻结的样本 / 已检测冻结样本；同时输出冻结秒数占可测输出秒数的比例。
- 人工接受结果：接受率 = 接受数 /（接受数 + 拒绝数）；待审、未知和审核覆盖率单列。

## 策略安全边界

工具的输出包含固定策略安全声明：

- `observation_only: true`
- `auto_policy_mutation_enabled: false`
- `default_provider_changed: false`
- `quality_threshold_changed: false`
- `applied_policy_changes: []`

任何样本量下的报告都只用于人工决策，不写回 Provider 配置或质量阈值。

## 本地 fixture 测试

```bash
python -m unittest server.tests.test_video_ab_evaluation -v
```

fixture 流程会通过 CLI 连续生成两次报告，并验证 JSON 与 Markdown 逐字节一致；
另测同计划约束、完整策略矩阵约束和默认 Provider / 质量阈值不变。
