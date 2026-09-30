/**
 * 质量审核（quality review）的前端纯模型：类型、状态中文映射与派生视图。
 *
 * 与后端 `services/quality_review_service.py` 的序列化字段一一对应。
 * 状态语义（与后端诚实性铁律一致）：
 * - scored：真实检测并评分；
 * - unsupported：能力未配置（VLM / embedding / ffmpeg 缺失），界面必须如实
 *   展示为「未检测」，绝不显示为通过；
 * - skipped：该镜头不适用（如无台词的口型维度）；
 * - error：Provider 调用失败，fail-closed。
 */

export type ReviewVerdict = 'passed' | 'failed' | 'unsupported' | 'error'
export type DimensionStatus = 'scored' | 'unsupported' | 'skipped' | 'error'
export type ReviewStage = 'storyboard' | 'video'

export interface QualityDimension {
  key: string
  label: string
  status: DimensionStatus
  score: number | null
  weight: number
  issues: string[]
  evidence: Record<string, unknown>
  provider: string
}

export interface QualityFix {
  directives?: string[]
  summary?: string
}

export interface QualityReviewRow {
  id: string
  shot_id: string
  project_id: string
  stage: ReviewStage
  attempt: number
  shot_version: number
  target_path: string
  verdict: ReviewVerdict
  passed: boolean
  overall_score: number
  degraded: boolean
  dimensions: QualityDimension[]
  issues: string[]
  unsupported_dimensions: string[]
  suggestion: string
  prompt_fix: QualityFix
  gate_policy: string
  created_at: string
}

export interface QualityStageSummary {
  verdict: ReviewVerdict
  passed: boolean
  overall_score: number
  attempt: number
  degraded: boolean
  issues_count: number
  unsupported: string[]
}

export interface ShotQualitySummary {
  storyboard?: QualityStageSummary | null
  video?: QualityStageSummary | null
}

export interface ProviderCapability {
  supported: boolean
  reason?: string
  provider?: string
  vlm?: ProviderCapability
  identity_embedding?: ProviderCapability
}

export interface QualityGateSnapshot {
  threshold: number
  policy: string
  storyboard_max_retries: number
  video_max_retries: number
}

export interface QualityCapabilityResponse {
  storyboard: ProviderCapability
  video: ProviderCapability
  gate: QualityGateSnapshot
}

const VERDICT_META: Record<ReviewVerdict, { label: string; tone: string }> = {
  passed: { label: '质量通过', tone: 'passed' },
  failed: { label: '质量未通过', tone: 'failed' },
  unsupported: { label: '未检测（能力未配置）', tone: 'unsupported' },
  error: { label: '审核出错', tone: 'error' },
}

const DIMENSION_STATUS_META: Record<DimensionStatus, { label: string; tone: string }> = {
  scored: { label: '已检测', tone: 'scored' },
  unsupported: { label: '未检测', tone: 'unsupported' },
  skipped: { label: '不适用', tone: 'skipped' },
  error: { label: '检测失败', tone: 'error' },
}

export function verdictMeta(verdict: string): { label: string; tone: string } {
  return VERDICT_META[verdict as ReviewVerdict] || { label: String(verdict || '未知'), tone: 'error' }
}

export function dimensionStatusMeta(status: string): { label: string; tone: string } {
  return DIMENSION_STATUS_META[status as DimensionStatus] || { label: String(status || '未知'), tone: 'error' }
}

export function formatScore(score: number | null | undefined): string {
  if (score === null || score === undefined || Number.isNaN(Number(score))) return '—'
  return (Number(score) * 100).toFixed(0)
}

export function gateLine(gate: QualityGateSnapshot | undefined | null): string {
  if (!gate) return ''
  const policy = gate.policy === 'lenient' ? '允许降级（未检测项如实标注）' : '严格（未检测即拦截）'
  return `通过阈值 ${Math.round(gate.threshold * 100)} 分 · ${policy} · 故事板最多重试 ${gate.storyboard_max_retries} 次 / 视频 ${gate.video_max_retries} 次`
}

/** 是否存在需要向用户如实展示的未检测/降级信息。 */
export function hasUndetected(row: QualityReviewRow): boolean {
  return row.degraded || row.unsupported_dimensions.length > 0
}

export function latestByStage(rows: QualityReviewRow[]): Record<ReviewStage, QualityReviewRow | null> {
  const latest: Record<ReviewStage, QualityReviewRow | null> = { storyboard: null, video: null }
  for (const row of rows) {
    if (!latest[row.stage]) latest[row.stage] = row
  }
  return latest
}

export function stageLabel(stage: string): string {
  return stage === 'video' ? '视频' : '故事板'
}

/**
 * 缩略图角标：质量通过 / 未通过 / 未检测 / 审核出错；无审核记录返回 null。
 * 优先显示未通过与未检测（比「通过」更需要用户注意）。
 */
export function qualityBadgeFor(
  summary: ShotQualitySummary | null | undefined,
): { label: string; className: string; title: string } | null {
  const storyboard = summary?.storyboard
  const video = summary?.video
  const candidate = video || storyboard
  if (!candidate) return null
  if (candidate.unsupported?.length) {
    return {
      label: '未检测',
      className: 'unsupported',
      title: `存在未检测维度：${candidate.unsupported.join('、')}`,
    }
  }
  const meta = verdictMeta(candidate.verdict)
  return {
    label: candidate.verdict === 'passed' ? '质检通过' : meta.label,
    className: meta.tone,
    title: `${stageLabel(video && candidate === video ? 'video' : 'storyboard')}质量审核：${meta.label}（${formatScore(candidate.overall_score)} 分）`,
  }
}

/** 把审核历史整理成「每轮候选」的展示行（新在前，带阶段与轮次）。 */
export function historyEntries(rows: QualityReviewRow[]): Array<{
  id: string
  title: string
  verdict: string
  score: string
  degraded: boolean
  created_at: string
}> {
  return rows.map((row) => ({
    id: row.id,
    title: `第 ${row.attempt} 轮 · ${stageLabel(row.stage)}`,
    verdict: row.verdict,
    score: formatScore(row.overall_score),
    degraded: row.degraded,
    created_at: row.created_at,
  }))
}
