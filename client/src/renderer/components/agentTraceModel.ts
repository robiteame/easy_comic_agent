/**
 * 生成 Agent 可解释追踪的展示模型。
 *
 * 全部是纯函数/纯类型：组件（AgentTracePanel、FlowGraph、TaskCenter）只负责
 * 渲染，字段归一、标签映射、耗时/成本文案都在这里，便于单测覆盖。
 */

export const AGENT_STAGE_ORDER = [
  'director_planning',
  'storyboard_design',
  'asset_preparation',
  'image_generation',
  'quality_review',
  'audio_production',
  'video_generation',
  'video_review',
  'edit_composition',
  'final_review',
] as const

export type AgentStageName = (typeof AGENT_STAGE_ORDER)[number]

export const AGENT_STAGE_LABELS: Record<string, string> = {
  director_planning: '导演规划',
  storyboard_design: '分镜设计',
  asset_preparation: '素材准备',
  image_generation: '图像生成',
  quality_review: '质量审核',
  audio_production: '音频制作',
  video_generation: '视频生成',
  video_review: '视频检查',
  edit_composition: '剪辑合成',
  final_review: '成片复审',
}

export const RUN_STATUS_LABELS: Record<string, string> = {
  pending: '待运行',
  running: '运行中',
  recovering: '自动恢复中',
  waiting_human: '等待人工',
  completed: '已完成',
  degraded: '降级发布',
  failed: '已终止',
  cancelled: '已取消',
}

export const STAGE_STATUS_LABELS: Record<string, string> = {
  pending: '待执行',
  running: '执行中',
  recovering: '恢复中',
  succeeded: '通过',
  degraded: '降级',
  waiting_human: '等待人工',
  failed: '失败',
  skipped: '跳过',
  invalidated: '已失效',
}

export const STRATEGY_LABELS: Record<string, string> = {
  retry: '原参数重试',
  change_seed: '更换种子',
  revise_prompt: '修改 Prompt',
  replace_reference: '替换参考图',
  split_shot: '拆分镜头',
  switch_provider: '切换 Provider',
  lower_resolution: '降低分辨率',
  regenerate_failed_shots: '只补拍失败镜头',
  merge_shots: '合并镜头',
  resume_checkpoint: '恢复检查点',
  human_review: '人工审核',
  degraded_publish: '降级发布',
  terminal_failure: '明确终止',
}

export interface AgentTraceMetric {
  name?: string
  value?: number | string | boolean | null
  threshold?: number | null
  passed?: boolean | null
}

export interface AgentTraceIssue {
  code: string
  severity: 'info' | 'warning' | 'error' | string
  message: string
  shot_id?: string
  recommendation?: string
}

export interface AgentTraceCritique {
  passed?: boolean | null
  score?: number | null
  issues?: AgentTraceIssue[]
  proposed_changes?: string[]
  failure_kind?: string | null
  recommended_strategy?: string | null
  affected_shot_ids?: string[]
}

export interface AgentTracePromptPatch {
  field: string
  op: string
  value?: unknown
  shot_id: string
  target_stage: string
  reason: string
}

export interface AgentTraceCandidate {
  strategy: string
  provider: string
  target_stage: string
  shot_ids: string[]
  estimated_cost_micro?: number | null
  estimated_seconds?: number | null
  quality_gain?: number | null
  provider_capability_ok: boolean
  budget_fit: boolean
  score?: number | null
  rationale: string
  prompt_patches: AgentTracePromptPatch[]
}

export interface AgentTraceRejection {
  strategy: string
  score?: number | null
  reason: string
}

export interface AgentTraceDecision {
  trace_id: string
  stage: string
  mode: string
  shot_id: string
  failure_kind: string
  failure_message: string
  quality_score?: number | null
  retries_remaining?: number | null
  reason: string
  selected: AgentTraceCandidate | null
  candidates: AgentTraceCandidate[]
  considered_rejected: AgentTraceRejection[]
  budget: {
    level?: string | null
    remaining_cost_micro?: number | null
    remaining_seconds?: number | null
  }
  provider_profiles: Array<{
    capability: string
    provider: string
    model: string
    available: boolean
    supports_reference_images: boolean
  }>
  selected_video_candidate_id: string
  candidate_selection: {
    candidate_id?: string
    reason?: string
    considered?: string[]
    rejected?: Array<{ candidate_id: string; reason: string }>
  } | null
  created_at: string
}

export interface AgentTraceVideoCandidate {
  candidate_id: string
  status: string
  provider: string
  model: string
  score?: number | null
  seed?: number | null
  path: string
  generation_duration_ms: number
  structural_passed?: boolean | null
  selected: boolean
  selection_reason: string
  failure_kind: string
  reference_manifest: Array<Record<string, unknown>>
}

export interface AgentTraceShot {
  shot_id: string
  stages: Record<
    string,
    {
      status: string
      provider: string
      model: string
      score?: number | null
      cost_micro?: number | null
      duration_ms: number
      path: string
      failure_kind: string
      failure_message: string
      shot_version: number
    }
  >
  cost_micro: number
  duration_ms: number
  video_candidates: AgentTraceVideoCandidate[]
  selected_video_candidate_id: string
  candidate_selection: AgentTraceDecision['candidate_selection']
  db_status?: string
  db_shot_version?: number
  confirmed?: boolean
  references_sent?: {
    image_generation?: Array<Record<string, unknown>>
    video_generation?: Array<Record<string, unknown>>
  }
}

export interface AgentTraceStage {
  stage: string
  label?: string
  status: string
  valid: boolean
  invalidated_reason: string
  quality: AgentTraceCritique | null
  failure_kind: string
  failure_message: string
  actual_duration_ms: number
  created_at: string
  saved_at: string
}

export interface AgentTraceSummary {
  run: {
    project_id: string
    run_id: string
    status: string
    status_reason: string
    current_stage: string
    input_fingerprint: string
    updated_at: string
  }
  counters: {
    checkpoint_records: number
    stage_checkpoints: number
    decisions: number
    recoveries: number
    invalidated_checkpoints: number
  }
  totals: {
    cost_micro: number
    video_candidates: number
    selected_video_candidates: number
  }
  stages: AgentTraceStage[]
  shots: AgentTraceShot[]
  decisions: AgentTraceDecision[]
  prompt_changes: Array<AgentTracePromptPatch & { trace_id: string; stage: string }>
  degradations: Array<{ at: string; stage: string; reason: string; shot_ids: string[] }>
  events: Array<Record<string, unknown>>
}

export interface AgentTraceResponse {
  project_id?: string
  run_id?: string
  status?: string
  status_reason?: string
  summary?: AgentTraceSummary
  mermaid?: string
}

export type TraceStageState = 'done' | 'running' | 'degraded' | 'failed' | 'pending'

/** 阶段进度状态：由 summary.run.current_stage 与各阶段 checkpoint 状态推导。 */
export function resolveStageStates(summary: AgentTraceSummary | null | undefined): Record<string, TraceStageState> {
  const result: Record<string, TraceStageState> = {}
  if (!summary) return result
  const rows = new Map(summary.stages.map((row) => [row.stage, row]))
  const current = String(summary.run?.current_stage || '')
  const currentIndex = AGENT_STAGE_ORDER.indexOf(current as AgentStageName)
  for (const stage of AGENT_STAGE_ORDER) {
    const row = rows.get(stage)
    if (!row) {
      result[stage] =
        currentIndex >= 0 && AGENT_STAGE_ORDER.indexOf(stage as AgentStageName) < currentIndex ? 'done' : 'pending'
      continue
    }
    if (!row.valid || row.status === 'invalidated') {
      result[stage] = 'pending'
      continue
    }
    if (row.status === 'succeeded' || row.status === 'skipped') result[stage] = 'done'
    else if (row.status === 'running' || row.status === 'recovering') result[stage] = 'running'
    else if (row.status === 'degraded') result[stage] = 'degraded'
    else if (row.status === 'failed') result[stage] = stage === current ? 'running' : 'failed'
    else if (row.status === 'waiting_human') result[stage] = 'degraded'
    else result[stage] = stage === current ? 'running' : 'pending'
  }
  // current_stage 之后的阶段还没执行，不能因为读到脏 checkpoint 显示成通过。
  if (currentIndex >= 0) {
    for (const stage of AGENT_STAGE_ORDER.slice(currentIndex + 1)) {
      if (result[stage] === 'done') result[stage] = 'pending'
    }
  }
  return result
}

export function stageLabel(stage: string): string {
  return AGENT_STAGE_LABELS[stage] || stage
}

export function runStatusLabel(status: string | null | undefined): string {
  const key = String(status || 'pending')
  return RUN_STATUS_LABELS[key] || key
}

export function strategyLabel(strategy: string | null | undefined): string {
  const key = String(strategy || '')
  return STRATEGY_LABELS[key] || key || '—'
}

/** 成本文案：micro 整数转「¥0.0123」；null/undefined 显示「未知」而不是 0。 */
export function traceCostText(costMicro: number | null | undefined): string {
  if (costMicro === null || costMicro === undefined) return '未知'
  const yuan = costMicro / 1_000_000
  return `¥${yuan.toFixed(yuan >= 1 ? 2 : 4)}`
}

/** 耗时文案：毫秒转「1.2s / 3分05秒」；0 视为未记录。 */
export function traceDurationText(ms: number | null | undefined): string {
  const value = Number(ms || 0)
  if (!value) return '—'
  if (value < 60_000) return `${(value / 1000).toFixed(1)}s`
  const minutes = Math.floor(value / 60_000)
  const seconds = Math.round((value % 60_000) / 1000)
  return `${minutes}分${String(seconds).padStart(2, '0')}秒`
}

export function secondsText(seconds: number | null | undefined): string {
  const value = Number(seconds || 0)
  if (!value) return '—'
  return `${value}s`
}

/** 每个镜头取最新一次产物状态，作为「镜头状态」列表行。 */
export function shotStatusRows(summary: AgentTraceSummary | null | undefined): Array<{
  shotId: string
  status: string
  provider: string
  model: string
  costText: string
  durationText: string
  failureKind: string
  references: number
}> {
  if (!summary) return []
  return summary.shots.map((shot) => {
    const orderedStages = [...Object.keys(shot.stages)].sort(
      (a, b) => AGENT_STAGE_ORDER.indexOf(a as AgentStageName) - AGENT_STAGE_ORDER.indexOf(b as AgentStageName),
    )
    let latest = { stage: '', status: '', provider: '', model: '', failureKind: '' }
    for (const stage of orderedStages) {
      const row = shot.stages[stage]
      if (!row?.status) continue
      if (stage === latest.stage) continue
      latest = {
        stage,
        status: row.status,
        provider: row.provider || (shot.video_candidates.find((item) => item.selected)?.provider ?? ''),
        model: row.model || (shot.video_candidates.find((item) => item.selected)?.model ?? ''),
        failureKind: row.failure_kind || '',
      }
    }
    const references =
      (shot.references_sent?.image_generation?.length || 0) + (shot.references_sent?.video_generation?.length || 0)
    return {
      shotId: shot.shot_id,
      status: latest.status || shot.db_status || 'pending',
      provider: latest.provider,
      model: latest.model,
      costText: traceCostText(shot.cost_micro || null),
      durationText: traceDurationText(shot.duration_ms),
      failureKind: latest.failureKind,
      references,
    }
  })
}

/** 阶段质量分行：阶段 + 质量分 + Critic 问题（只保留 error/warning）。 */
export function stageQualityRows(summary: AgentTraceSummary | null | undefined): Array<{
  stage: string
  label: string
  status: string
  scoreText: string
  issues: AgentTraceIssue[]
}> {
  if (!summary) return []
  return summary.stages
    .filter((row) => row.stage !== 'pending' || row.quality)
    .map((row) => ({
      stage: row.stage,
      label: row.label || stageLabel(row.stage),
      status: row.status,
      scoreText: row.quality?.score !== null && row.quality?.score !== undefined ? row.quality.score.toFixed(2) : '—',
      issues: (row.quality?.issues || []).filter((issue) => issue.severity !== 'info'),
    }))
}

export interface DecisionRow {
  traceId: string
  stage: string
  stageLabel: string
  failureKind: string
  failureMessage: string
  selectedStrategy: string
  selectedStrategyLabel: string
  provider: string
  estimatedCostText: string
  estimatedSecondsText: string
  reason: string
  rejected: Array<{ label: string; reason: string }>
  candidateCount: number
}

export function decisionRows(summary: AgentTraceSummary | null | undefined): DecisionRow[] {
  if (!summary) return []
  return summary.decisions.map((decision) => ({
    traceId: decision.trace_id,
    stage: decision.stage,
    stageLabel: stageLabel(decision.stage),
    failureKind: decision.failure_kind || '—',
    failureMessage: decision.failure_message || '',
    selectedStrategy: decision.selected?.strategy || decision.candidate_selection?.reason || 'none',
    selectedStrategyLabel: decision.selected
      ? strategyLabel(decision.selected.strategy)
      : decision.candidate_selection?.reason || '未选择恢复动作',
    provider: decision.selected?.provider || '',
    estimatedCostText: traceCostText(decision.selected?.estimated_cost_micro ?? null),
    estimatedSecondsText: secondsText(decision.selected?.estimated_seconds ?? null),
    reason: decision.reason,
    rejected: decision.considered_rejected.map((item) => ({
      label: strategyLabel(item.strategy),
      reason: item.reason,
    })),
    candidateCount: decision.candidates.length,
  }))
}

export interface PromptChangeRow {
  key: string
  shotId: string
  stageLabel: string
  field: string
  op: string
  valueText: string
  reason: string
}

export function promptChangeRows(summary: AgentTraceSummary | null | undefined): PromptChangeRow[] {
  if (!summary) return []
  return summary.prompt_changes.map((patch, index) => ({
    key: `${patch.trace_id}:${index}:${patch.field}`,
    shotId: patch.shot_id || '全部镜头',
    stageLabel: stageLabel(patch.target_stage || patch.stage),
    field: patch.field,
    op: patch.op,
    valueText: formatPatchValue(patch.value),
    reason: patch.reason,
  }))
}

function formatPatchValue(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

export interface CandidateResultRow {
  key: string
  shotId: string
  candidateId: string
  status: string
  providerText: string
  scoreText: string
  durationText: string
  selected: boolean
  selectionReason: string
  structuralPassed: string
}

export function candidateResultRows(summary: AgentTraceSummary | null | undefined): CandidateResultRow[] {
  if (!summary) return []
  const rows: CandidateResultRow[] = []
  for (const shot of summary.shots) {
    for (const candidate of shot.video_candidates) {
      rows.push({
        key: `${shot.shot_id}:${candidate.candidate_id}`,
        shotId: shot.shot_id,
        candidateId: candidate.candidate_id || '—',
        status: candidate.status,
        providerText: [candidate.provider, candidate.model].filter(Boolean).join(' / ') || '—',
        scoreText: candidate.score !== null && candidate.score !== undefined ? candidate.score.toFixed(2) : '—',
        durationText: traceDurationText(candidate.generation_duration_ms),
        selected: candidate.selected || shot.selected_video_candidate_id === candidate.candidate_id,
        selectionReason: candidate.selection_reason,
        structuralPassed:
          candidate.structural_passed === true ? '通过' : candidate.structural_passed === false ? '失败' : '未检',
      })
    }
  }
  return rows
}

export interface ReferenceRow {
  key: string
  shotId: string
  stage: string
  stageLabel: string
  count: number
  names: string
}

/** 实际发送参考图：来自 DB 镜头 manifest 与视频候选 manifest 的合并视图。 */
export function referenceRows(summary: AgentTraceSummary | null | undefined): ReferenceRow[] {
  if (!summary) return []
  const rows: ReferenceRow[] = []
  for (const shot of summary.shots) {
    const groups: Array<[string, Array<Record<string, unknown>>]> = [
      ['image_generation', shot.references_sent?.image_generation || []],
      ['video_generation', shot.references_sent?.video_generation || []],
    ]
    for (const [stage, manifest] of groups) {
      if (!manifest.length) continue
      rows.push({
        key: `${shot.shot_id}:${stage}`,
        shotId: shot.shot_id,
        stage,
        stageLabel: stageLabel(stage),
        count: manifest.length,
        names:
          manifest
            .map((entry) => String(entry.name || entry.asset_id || entry.kind || entry.path || '参考图'))
            .slice(0, 4)
            .join('、') + (manifest.length > 4 ? ` 等 ${manifest.length} 张` : ''),
      })
    }
  }
  return rows
}

/** 追踪是否有任何可展示内容（决定面板空态文案）。 */
export function traceHasContent(summary: AgentTraceSummary | null | undefined): boolean {
  if (!summary) return false
  return Boolean(
    summary.stages.length ||
      summary.shots.length ||
      summary.decisions.length ||
      summary.degradations.length ||
      summary.counters.checkpoint_records,
  )
}

/** 预计 vs 实际耗时对比：预计取最近一次选中恢复候选/Provider 画像，实际取阶段产物。 */
export function durationComparison(summary: AgentTraceSummary | null | undefined): {
  estimatedText: string
  actualText: string
} {
  if (!summary) {
    return { estimatedText: '—', actualText: '—' }
  }
  const lastWithEstimate = [...summary.decisions].reverse().find((decision) => decision.selected?.estimated_seconds)
  const estimated = lastWithEstimate?.selected?.estimated_seconds ?? null
  const actualMs = summary.shots.reduce((total, shot) => total + (shot.duration_ms || 0), 0)
  return {
    estimatedText: secondsText(estimated),
    actualText: traceDurationText(actualMs || null),
  }
}
