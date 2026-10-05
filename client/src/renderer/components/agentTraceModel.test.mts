import assert from 'node:assert/strict'
import test from 'node:test'

import {
  AGENT_STAGE_ORDER,
  candidateResultRows,
  decisionRows,
  durationComparison,
  promptChangeRows,
  referenceRows,
  resolveStageStates,
  runStatusLabel,
  shotStatusRows,
  stageQualityRows,
  strategyLabel,
  traceCostText,
  traceDurationText,
  traceHasContent,
  type AgentTraceSummary,
} from './agentTraceModel.ts'

function makeSummary(overrides: Partial<AgentTraceSummary> = {}): AgentTraceSummary {
  return {
    run: {
      project_id: 'p1',
      run_id: 'auto',
      status: 'recovering',
      status_reason: '',
      current_stage: 'video_generation',
      input_fingerprint: 'fp',
      updated_at: '2026-10-01T00:00:00Z',
    },
    counters: { checkpoint_records: 6, stage_checkpoints: 3, decisions: 2, recoveries: 2, invalidated_checkpoints: 1 },
    totals: { cost_micro: 1300, video_candidates: 2, selected_video_candidates: 1 },
    stages: [
      {
        stage: 'director_planning',
        status: 'succeeded',
        valid: true,
        invalidated_reason: '',
        quality: { passed: true, score: 0.9, issues: [] },
        failure_kind: '',
        failure_message: '',
        actual_duration_ms: 1500,
        created_at: '',
        saved_at: '',
      },
      {
        stage: 'image_generation',
        status: 'failed',
        valid: true,
        invalidated_reason: '',
        quality: {
          passed: false,
          score: 0.3,
          issues: [
            { code: 'image_invalid', severity: 'error', message: '镜头 s2 图片损坏', shot_id: 's2' },
            { code: 'visual_note', severity: 'info', message: '提示' },
          ],
        },
        failure_kind: 'image_failed',
        failure_message: 'provider failed',
        actual_duration_ms: 4200,
        created_at: '',
        saved_at: '',
      },
      {
        stage: 'video_generation',
        status: 'running',
        valid: true,
        invalidated_reason: '',
        quality: null,
        failure_kind: '',
        failure_message: '',
        actual_duration_ms: 0,
        created_at: '',
        saved_at: '',
      },
    ],
    shots: [
      {
        shot_id: 's1',
        stages: {
          image_generation: {
            status: 'succeeded',
            provider: 'qwen-image',
            model: 'qwen-image-edit',
            score: 0.8,
            cost_micro: 100,
            duration_ms: 1200,
            path: 'a.png',
            failure_kind: '',
            failure_message: '',
            shot_version: 1,
          },
          video_generation: {
            status: 'succeeded',
            provider: 'ark-seedance',
            model: 'seedance-1',
            score: 0.9,
            cost_micro: 1200,
            duration_ms: 8000,
            path: 'a.mp4',
            failure_kind: '',
            failure_message: '',
            shot_version: 1,
          },
        },
        cost_micro: 1300,
        duration_ms: 9200,
        video_candidates: [
          {
            candidate_id: 'c1',
            status: 'succeeded',
            provider: 'ark-seedance',
            model: 'seedance-1',
            score: 0.9,
            seed: 7,
            path: 'a.mp4',
            generation_duration_ms: 8000,
            structural_passed: true,
            selected: true,
            selection_reason: 'structural_pass_highest_score',
            failure_kind: '',
            reference_manifest: [{ kind: 'character', name: '主角三视图' }],
          },
          {
            candidate_id: 'c2',
            status: 'failed',
            provider: 'ark-seedance',
            model: 'seedance-1',
            score: 0,
            seed: 8,
            path: '',
            generation_duration_ms: 3000,
            structural_passed: false,
            selected: false,
            selection_reason: '',
            failure_kind: 'video_failed',
            reference_manifest: [],
          },
        ],
        selected_video_candidate_id: 'c1',
        candidate_selection: { candidate_id: 'c1', reason: 'structural_pass_highest_score' },
        db_status: 'video_ready',
        db_shot_version: 1,
        confirmed: true,
        references_sent: {
          image_generation: [{ name: '主角三视图' }],
          video_generation: [{ name: '场景基准图' }, { name: '上一镜尾帧' }],
        },
      },
    ],
    decisions: [
      {
        trace_id: 't1',
        stage: 'image_generation',
        mode: 'auto',
        shot_id: 's2',
        failure_kind: 'image_failed',
        failure_message: 'provider failed',
        quality_score: 0.3,
        retries_remaining: 1,
        reason: '自动恢复',
        selected: {
          strategy: 'switch_provider',
          provider: 'qwen-image',
          target_stage: 'image_generation',
          shot_ids: ['s2'],
          estimated_cost_micro: 12000,
          estimated_seconds: 8,
          quality_gain: 0.3,
          provider_capability_ok: true,
          budget_fit: true,
          score: 0.88,
          rationale: '切换 Provider',
          prompt_patches: [
            {
              field: 'provider',
              op: 'set',
              value: 'qwen-image',
              shot_id: 's2',
              target_stage: 'image_generation',
              reason: '当前 Provider 失败',
            },
          ],
        },
        candidates: [],
        considered_rejected: [{ strategy: 'human_review', score: 0.1, reason: '自动模式禁止选择 human_review' }],
        budget: { level: 'project', remaining_cost_micro: 50000, remaining_seconds: 600 },
        provider_profiles: [
          {
            capability: 'image',
            provider: 'qwen-image',
            model: 'qwen-image-edit',
            available: true,
            supports_reference_images: true,
          },
        ],
        selected_video_candidate_id: '',
        candidate_selection: null,
        created_at: '',
      },
    ],
    prompt_changes: [
      {
        trace_id: 't1',
        stage: 'image_generation',
        shot_id: 's2',
        field: 'provider',
        op: 'set',
        value: 'qwen-image',
        target_stage: 'image_generation',
        reason: '当前 Provider 失败',
      },
    ],
    degradations: [{ at: '', stage: 'video_generation', reason: '自动恢复无法继续，按降级结果发布', shot_ids: ['s2'] }],
    events: [],
    ...overrides,
  }
}

test('resolveStageStates 标记当前阶段且不提前点亮后续阶段', () => {
  const states = resolveStageStates(makeSummary())
  assert.equal(states.director_planning, 'done')
  assert.equal(states.image_generation, 'failed')
  assert.equal(states.video_generation, 'running')
  assert.equal(states.video_review, 'pending')
  assert.equal(states.final_review, 'pending')
})

test('resolveStageStates 空追踪返回空映射', () => {
  assert.deepEqual(resolveStageStates(null), {})
})

test('十阶段顺序与后端 GRAPH_STAGE_ORDER 一致', () => {
  assert.equal(AGENT_STAGE_ORDER.length, 10)
  assert.equal(AGENT_STAGE_ORDER[0], 'director_planning')
  assert.equal(AGENT_STAGE_ORDER[9], 'final_review')
})

test('成本与耗时的诚实展示：未知不是 0', () => {
  assert.equal(traceCostText(null), '未知')
  assert.equal(traceCostText(12000), '¥0.0120')
  assert.equal(traceCostText(2_500_000), '¥2.50')
  assert.equal(traceDurationText(0), '—')
  assert.equal(traceDurationText(1500), '1.5s')
  assert.equal(traceDurationText(125_000), '2分05秒')
})

test('镜头状态行取最新阶段产物并统计参考图', () => {
  const rows = shotStatusRows(makeSummary())
  assert.equal(rows.length, 1)
  assert.equal(rows[0].shotId, 's1')
  assert.equal(rows[0].status, 'succeeded')
  assert.equal(rows[0].provider, 'ark-seedance')
  assert.equal(rows[0].model, 'seedance-1')
  assert.equal(rows[0].references, 3)
})

test('阶段质量行保留 error/warning 的 Critic 问题', () => {
  const rows = stageQualityRows(makeSummary())
  const image = rows.find((row) => row.stage === 'image_generation')
  assert.ok(image)
  assert.equal(image.scoreText, '0.30')
  assert.equal(image.issues.length, 1)
  assert.equal(image.issues[0].code, 'image_invalid')
})

test('决策行包含最终选择与被淘汰原因', () => {
  const rows = decisionRows(makeSummary())
  assert.equal(rows.length, 1)
  assert.equal(rows[0].selectedStrategy, 'switch_provider')
  assert.equal(rows[0].selectedStrategyLabel, '切换 Provider')
  assert.equal(rows[0].rejected[0].reason, '自动模式禁止选择 human_review')
})

test('Prompt 修改行输出字段级补丁', () => {
  const rows = promptChangeRows(makeSummary())
  assert.equal(rows.length, 1)
  assert.equal(rows[0].field, 'provider')
  assert.equal(rows[0].valueText, 'qwen-image')
})

test('候选结果行标记自动选中的候选', () => {
  const rows = candidateResultRows(makeSummary())
  assert.equal(rows.length, 2)
  const selected = rows.find((row) => row.selected)
  assert.equal(selected?.candidateId, 'c1')
  assert.equal(selected?.selectionReason, 'structural_pass_highest_score')
  assert.equal(rows.find((row) => row.candidateId === 'c2')?.structuralPassed, '失败')
})

test('参考图行按阶段汇总实际发送清单', () => {
  const rows = referenceRows(makeSummary())
  assert.equal(rows.length, 2)
  assert.equal(rows[0].stageLabel, '图像生成')
  assert.equal(rows[1].count, 2)
})

test('预计/实际耗时对比来自决策候选与镜头产物', () => {
  const comparison = durationComparison(makeSummary())
  assert.equal(comparison.estimatedText, '8s')
  assert.equal(comparison.actualText, '9.2s')
})

test('空汇总判定与运行状态标签', () => {
  assert.equal(traceHasContent(null), false)
  assert.equal(
    traceHasContent(
      makeSummary({
        stages: [],
        shots: [],
        decisions: [],
        degradations: [],
        counters: {
          checkpoint_records: 0,
          stage_checkpoints: 0,
          decisions: 0,
          recoveries: 0,
          invalidated_checkpoints: 0,
        },
      }),
    ),
    false,
  )
  assert.equal(runStatusLabel('degraded'), '降级发布')
  assert.equal(strategyLabel('split_shot'), '拆分镜头')
})
