import type { JobDebugEvent } from '../services/jobTypes'

export type DebugFilter = 'all' | 'progress' | 'api' | 'error'

export const DEBUG_FILTERS: { id: DebugFilter; label: string }[] = [
  { id: 'all', label: '全部' },
  { id: 'progress', label: '进度' },
  { id: 'api', label: 'API 请求' },
  { id: 'error', label: '异常' },
]

export const PROGRESS_STEP_LABELS: Record<string, string> = {
  generate_script: '剧本生成',
  parse_script: '剧本解析',
  generate_storyboard: '分镜生成',
  wait_asset_confirm: '等待素材确认',
  generate_storyboard_images: '故事板生成',
  wait_storyboard_approval: '等待故事板审核',
  phase2_start: '视频阶段',
  generate_voice: 'Mimo 配音',
  generate_seedance_video: 'Seedance 视频',
  compose_video: '视频合成',
  quality_check: '结构检查（仅结构，非质量认证）',
  rendering: '成片导出',
  mixing: '音频合成',
}

export function progressStepLabel(step: string): string {
  if (!step) return '待命'
  return PROGRESS_STEP_LABELS[step] || step.split('_').join(' ')
}

export function isApiDebugEvent(event: JobDebugEvent): boolean {
  return event.kind === 'api_request' || event.kind === 'api_result' || Boolean(event.api)
}

export function isErrorDebugEvent(event: JobDebugEvent): boolean {
  return event.level === 'error' || event.level === 'cancelled'
}

export function debugFilterMatches(event: JobDebugEvent, filter: DebugFilter): boolean {
  if (filter === 'all') return true
  if (filter === 'progress') return event.kind === 'progress' || event.kind === 'lifecycle'
  if (filter === 'api') return isApiDebugEvent(event)
  return isErrorDebugEvent(event)
}

export function mergeDebugEvents(...groups: (JobDebugEvent[] | undefined)[]): JobDebugEvent[] {
  const byId = new Map<string, JobDebugEvent>()
  for (const group of groups) {
    for (const event of group || []) {
      if (event?.id) byId.set(event.id, event)
    }
  }
  return Array.from(byId.values()).sort((a, b) => {
    const left = Date.parse(a.timestamp || '') || 0
    const right = Date.parse(b.timestamp || '') || 0
    return left - right || a.id.localeCompare(b.id)
  })
}

export function latestApiRequest(events: JobDebugEvent[]): JobDebugEvent | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (events[index].kind === 'api_request') return events[index]
  }
  return null
}

export function apiResultFor(events: JobDebugEvent[], request: JobDebugEvent | null): JobDebugEvent | null {
  if (!request?.request_id) return null
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]
    if (event.kind === 'api_result' && event.request_id === request.request_id) return event
  }
  return null
}

export function formatDebugTimestamp(timestamp: string): string {
  const parsed = Date.parse(timestamp || '')
  if (!Number.isFinite(parsed)) return '--:--:--'
  return new Date(parsed).toLocaleTimeString('zh-CN', { hour12: false })
}

export function formatDebugPayload(value: unknown): string {
  if (value === null || value === undefined || value === '') return '暂无数据'
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

export function promptSections(value: unknown): { label: string; content: string }[] {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const record = value as Record<string, unknown>
    return Object.entries(record).map(([key, content]) => ({
      label: key === 'system' ? '系统提示词' : key === 'user' ? '用户提示词' : key,
      content: formatDebugPayload(content),
    }))
  }
  return [{ label: '提示词', content: formatDebugPayload(value) }]
}

export function debugLevelLabel(level: string): string {
  if (level === 'request') return '请求'
  if (level === 'success') return '成功'
  if (level === 'error') return '失败'
  if (level === 'cancelled') return '取消'
  if (level === 'progress') return '进度'
  return '日志'
}
