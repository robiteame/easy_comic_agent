export type ConsistencyStatus = 'pending' | 'ready' | 'failed' | 'degraded' | 'unsupported' | 'stale'

export interface ConsistencyReportItem {
  kind?: string
  asset_id?: string
  name?: string
  status?: string
  affected_shot_ids?: string[]
  affected_shot_count?: number
  shot_range?: string
  failure_reason?: string
  error_id?: string
  capability_warning?: string
}

export interface ConsistencyReport {
  status?: string
  blocking?: boolean
  degraded?: boolean
  items?: ConsistencyReportItem[]
  affected_shot_ids?: string[]
  affected_shot_count?: number
  shot_range?: string
  capability_warnings?: string[]
}

const REFERENCE_STATUS_LABELS: Record<ConsistencyStatus | 'unknown', string> = {
  ready: '参考就绪',
  failed: '参考失败',
  degraded: '已降级',
  unsupported: '能力不支持',
  stale: '参考过期',
  pending: '待生成',
  unknown: '状态未知',
}

export function normalizeReferenceStatus(value: unknown): ConsistencyStatus | 'unknown' {
  const status = String(value || '').trim().toLowerCase()
  return (Object.keys(REFERENCE_STATUS_LABELS) as Array<ConsistencyStatus | 'unknown'>).includes(status as any)
    ? (status as ConsistencyStatus | 'unknown')
    : 'unknown'
}

export function referenceStatusLabel(status?: string): string {
  return REFERENCE_STATUS_LABELS[normalizeReferenceStatus(status)]
}

export function referenceStatusClass(status?: string): string {
  return `reference-status reference-status-${normalizeReferenceStatus(status)}`
}

export function consistencyImpactText(report?: ConsistencyReport | null): string {
  const range = String(report?.shot_range || '')
  const count = Number(report?.affected_shot_count || report?.affected_shot_ids?.length || 0)
  return [range, count ? `影响 ${count} 个镜头` : ''].filter(Boolean).join(' · ')
}

export function blockingReferenceItems(items: readonly ConsistencyReportItem[] | undefined): ConsistencyReportItem[] {
  return (items || []).filter((item) => ['failed', 'unsupported', 'stale'].includes(normalizeReferenceStatus(item.status)))
}

export function consistencyReportSummary(report?: ConsistencyReport | null): string {
  const status = referenceStatusLabel(report?.status)
  const impact = consistencyImpactText(report)
  return [status, impact].filter(Boolean).join(' · ')
}
