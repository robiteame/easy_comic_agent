/**
 * 任务中心的纯逻辑：DTO 归一化、去重合并、筛选排序、统计、动作按钮状态与格式化。
 *
 * 这里不碰 DOM、不碰网络，因此可以在 node:test 下直接覆盖。组件与 store 只做
 * 「调用纯函数 + 渲染 / 发送请求」，避免把业务判断散落在 JSX 里。
 */

import type { JobDto, JobErrorDetail, JobEvent, JobQueryParams, JobStatus, JobType } from '../services/jobTypes'
import { normalizeJobCost } from '../services/costModel.ts'

export const JOB_STATUSES: JobStatus[] = [
  'queued',
  'running',
  'cancelling',
  'completed',
  'failed',
  'cancelled',
  'interrupted',
]

export const ACTIVE_STATUSES: JobStatus[] = ['queued', 'running', 'cancelling']
export const RETRYABLE_STATUSES: JobStatus[] = ['failed', 'cancelled', 'interrupted']

export const JOB_TYPES: JobType[] = [
  'script_pipeline',
  'storyboard',
  'asset_generation',
  'shot_image',
  'shot_audio',
  'shot_video',
  'render',
  'unknown',
]

export const JOB_TYPE_LABELS: Record<string, string> = {
  script_pipeline: '剧本解析',
  storyboard: '分镜与素材',
  asset_generation: '素材生成',
  shot_image: '镜头故事板',
  shot_audio: '镜头配音',
  shot_video: '镜头视频',
  render: '成片渲染',
  unknown: '其他任务',
}

export const STATUS_LABELS: Record<string, string> = {
  queued: '排队中',
  running: '进行中',
  cancelling: '取消中',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
  interrupted: '已中断',
}

// 失败类别中文标签：服务端 error_code_label 缺失（历史行/旧后端）时的前端兜底，
// 取值与后端 job_types.ERROR_CODE_LABELS 保持一致。
export const ERROR_CODE_LABELS: Record<string, string> = {
  job_failed: '任务失败',
  job_cancelled: '任务已取消',
  job_interrupted: '任务被中断',
  server_restart: '服务重启中断',
  timeout: '调用超时',
  provider_error: 'API 调用失败',
  provider_config_error: '鉴权失败或未配置',
  provider_quota_exceeded: '额度不足',
  provider_rate_limited: '触发限流',
  provider_invalid_request: 'API 参数错误',
  dependency_failed: '前置阶段失败',
  storage_error: '存储异常',
  llm_output_truncated: '模型输出超长被截断',
  budget_exceeded: '超出项目预算',
  budget_soft_exceeded: '接近预算上限',
}

// 失败原因筛选项（稳定错误码 + 标签）。
export const FAILURE_CATEGORY_OPTIONS: { value: string; label: string }[] = [
  { value: 'provider_error', label: 'API 调用失败' },
  { value: 'provider_quota_exceeded', label: '额度不足' },
  { value: 'provider_rate_limited', label: '触发限流' },
  { value: 'provider_invalid_request', label: 'API 参数错误' },
  { value: 'provider_config_error', label: '鉴权失败或未配置' },
  { value: 'timeout', label: '调用超时' },
  { value: 'dependency_failed', label: '前置阶段失败' },
  { value: 'budget_exceeded', label: '超出项目预算' },
  { value: 'storage_error', label: '存储异常' },
  { value: 'llm_output_truncated', label: '模型输出超长被截断' },
  { value: 'job_failed', label: '其他失败' },
]

/** 类别徽标的色调：账户类可自查（橙）、配置/调用类需处理（红）、环境/业务类（灰）。 */
export type ErrorCodeTone = 'warn' | 'danger' | 'muted'

/** 输出截断这类确定性失败的可执行建议：让用户知道下一步改什么，而不是重试同样的配置。 */
export const TRUNCATION_ADVICE =
  '模型输出达到上限被截断，重试同样配置无效。请在「模型与 API」中调大输出额度（max_tokens / max_output_tokens），或改用支持更大输出的端点；长剧本会自动分段解析，也可手动拆分后再试。'

export function isTruncationErrorCode(code: string): boolean {
  return code === 'llm_output_truncated'
}

export function errorCodeTone(code: string): ErrorCodeTone {
  if (code === 'provider_quota_exceeded' || code === 'provider_rate_limited') return 'warn'
  if (
    code === 'provider_error' ||
    code === 'provider_invalid_request' ||
    code === 'provider_config_error' ||
    code === 'timeout' ||
    code === 'llm_output_truncated' ||
    code === 'job_failed' ||
    code === ''
  ) {
    return 'danger'
  }
  return 'muted'
}

export type JobTone = 'active' | 'success' | 'danger' | 'muted'

export function statusTone(status: JobStatus): JobTone {
  if (ACTIVE_STATUSES.indexOf(status) >= 0) return 'active'
  if (status === 'completed') return 'success'
  if (status === 'failed') return 'danger'
  return 'muted'
}

export type JobSectionTone = 'active' | 'failed' | 'interrupted' | 'cancelled' | 'completed'

export function jobSection(status: JobStatus): JobSectionTone {
  if (ACTIVE_STATUSES.indexOf(status) >= 0) return 'active'
  if (status === 'failed') return 'failed'
  if (status === 'interrupted') return 'interrupted'
  if (status === 'cancelled') return 'cancelled'
  return 'completed'
}

export const SECTION_LABELS: Record<JobSectionTone, string> = {
  active: '活动任务',
  failed: '失败与可重试',
  interrupted: '被中断',
  cancelled: '已取消',
  completed: '最近完成',
}

export interface JobFilters {
  projectId: string
  statuses: JobStatus[]
  jobTypes: JobType[]
  /** 失败原因筛选（稳定错误码）；纯客户端过滤，不发给服务端。 */
  errorCodes: string[]
  search: string
  onlyActive: boolean
}

export const DEFAULT_FILTERS: JobFilters = {
  projectId: '',
  statuses: [],
  jobTypes: [],
  errorCodes: [],
  search: '',
  onlyActive: false,
}

export type JobSortKey = 'updated' | 'created' | 'status' | 'progress'

export const SORT_LABELS: Record<JobSortKey, string> = {
  updated: '最近更新',
  created: '最近创建',
  status: '状态优先',
  progress: '进度优先',
}

const STATUS_SORT_WEIGHT: Record<string, number> = {
  running: 0,
  cancelling: 1,
  queued: 2,
  failed: 3,
  interrupted: 4,
  cancelled: 5,
  completed: 6,
}

function asString(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number') return String(value)
  return fallback
}

/** 展示用文本：空白字符串按缺失处理，回落到稳定兜底文案。 */
function asText(value: unknown, fallback: string): string {
  const text = asString(value).trim()
  return text || fallback
}

function asBoolean(value: unknown, fallback = false): boolean {
  return typeof value === 'boolean' ? value : fallback
}

function asNumber(value: unknown, fallback = 0): number {
  const parsed = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function asIsoOrNull(value: unknown): string | null {
  return typeof value === 'string' && value ? value : null
}

/** 分析结果白名单归一化：只有带有效 summary/suggestion 才保留。 */
function normalizeErrorDetail(raw: unknown): JobErrorDetail | null {
  if (!raw || typeof raw !== 'object') return null
  const source = raw as Record<string, unknown>
  const summary = asString(source.summary)
  const suggestion = asString(source.suggestion)
  if (!summary && !suggestion) return null
  return {
    summary,
    suggestion,
    source: asString(source.source),
    model: asString(source.model),
  }
}

/**
 * 把服务端时间戳解析成绝对时刻（毫秒）。
 *
 * 服务端历史数据可能仍返回不带时区的 naive UTC 字符串；`new Date` 会把它当本地
 * 时间解释（东八区显示会偏差 8 小时）。因此这里显式补 `Z`：带时区（`Z` 或
 * `±hh:mm`）的交给标准解析，naive 的一律按 UTC 解释。无法解析返回 null。
 */
export function parseServerTime(iso: string | null | undefined): number | null {
  if (!iso) return null
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(iso)
  const time = new Date(hasZone ? iso : iso + 'Z').getTime()
  return Number.isNaN(time) ? null : time
}

/**
 * 把服务端返回的对象归一化成 DTO。
 *
 * 缺字段、类型不对或没有 id 的数据一律返回 null：宁可少显示一条任务，也不能让
 * 渲染层拿到 undefined 后崩掉，或者把 run token 之类的内部字段带进状态树。
 */
export function normalizeJob(raw: unknown): JobDto | null {
  if (!raw || typeof raw !== 'object') return null
  const source = raw as Record<string, unknown>
  const id = asString(source.id)
  if (!id) return null
  const status = asString(source.status, 'unknown') as JobStatus
  const jobType = asString(source.job_type, 'unknown') as JobType
  return {
    id,
    scope: asString(source.scope),
    project_id: asString(source.project_id),
    job_type: jobType,
    job_type_label: asText(source.job_type_label, JOB_TYPE_LABELS[jobType] || '其他任务'),
    display_name: asText(source.display_name, '未命名任务'),
    status,
    status_label: asText(source.status_label, STATUS_LABELS[status] || '未知状态'),
    progress: Math.max(0, Math.min(100, Math.round(asNumber(source.progress)))),
    current_step: asString(source.current_step),
    message: asString(source.message),
    error_code: asString(source.error_code),
    error_code_label: asString(source.error_code_label),
    error_message: asString(source.error_message),
    error_detail: normalizeErrorDetail(source.error_detail),
    report: source.report && typeof source.report === 'object' ? (source.report as Record<string, any>) : null,
    attempt: Math.max(1, Math.round(asNumber(source.attempt, 1))),
    retry_of: asIsoOrNull(source.retry_of),
    version: asNumber(source.version),
    created_at: asIsoOrNull(source.created_at),
    started_at: asIsoOrNull(source.started_at),
    updated_at: asIsoOrNull(source.updated_at),
    finished_at: asIsoOrNull(source.finished_at),
    cancel_requested_at: asIsoOrNull(source.cancel_requested_at),
    duration_seconds: Math.max(0, Math.round(asNumber(source.duration_seconds))),
    // 缺 cost 字段的老数据归一化成「暂无用量」，绝不伪造 ¥0。
    cost: normalizeJobCost(source.cost),
    eta_seconds:
      typeof source.eta_seconds === 'number' && Number.isFinite(source.eta_seconds)
        ? Math.max(0, Math.round(source.eta_seconds))
        : null,
    is_active: asBoolean(source.is_active, ACTIVE_STATUSES.indexOf(status) >= 0),
    is_terminal: asBoolean(source.is_terminal, ACTIVE_STATUSES.indexOf(status) < 0),
    has_active_successor: asBoolean(source.has_active_successor),
    can_cancel: asBoolean(source.can_cancel, ACTIVE_STATUSES.indexOf(status) >= 0),
    can_retry: asBoolean(source.can_retry, RETRYABLE_STATUSES.indexOf(status) >= 0),
    can_resume: asBoolean(source.can_resume, RETRYABLE_STATUSES.indexOf(status) >= 0),
    can_delete: asBoolean(source.can_delete, ACTIVE_STATUSES.indexOf(status) < 0),
    retry_blocked_reason: asString(source.retry_blocked_reason),
    resume_blocked_reason: asString(source.resume_blocked_reason),
    batch_id: asIsoOrNull(source.batch_id),
    queue_position: Math.max(0, Math.round(asNumber(source.queue_position))),
    priority: Math.round(asNumber(source.priority)),
    queue_order: Math.max(0, Math.round(asNumber(source.queue_order))),
    stage: asString(source.stage),
    shot_id: asIsoOrNull(source.shot_id),
    dependency_ids: Array.isArray(source.dependency_ids) ? source.dependency_ids.map(String) : [],
    blocked_reason: asString(source.blocked_reason),
    queue_concurrency: Math.max(1, Math.round(asNumber(source.queue_concurrency, 1))),
    paused: asBoolean(source.paused),
    resume_missing: asBoolean(source.resume_missing),
    reuse_audio: asBoolean(source.reuse_audio),
    force_confirmed: asBoolean(source.force_confirmed),
    requested_version: Math.max(0, Math.round(asNumber(source.requested_version))),
  }
}

export function normalizeJobList(raw: unknown): JobDto[] {
  if (!Array.isArray(raw)) return []
  const jobs: JobDto[] = []
  for (const item of raw) {
    const job = normalizeJob(item)
    if (job) jobs.push(job)
  }
  return jobs
}

/** 时间戳比较：统一解析成绝对时刻再比，兼容带时区与历史 naive 两种格式。 */
function isOlderThan(incoming: JobDto, current: JobDto): boolean {
  const a = parseServerTime(incoming.updated_at)
  const b = parseServerTime(current.updated_at)
  if (a !== null && b !== null) return a < b
  return false
}

export function upsertJob(jobs: JobDto[], incoming: JobDto): JobDto[] {
  const index = jobs.findIndex((job) => job.id === incoming.id)
  if (index < 0) return [...jobs, incoming].sort(compareByUpdatedDesc)
  const current = jobs[index]
  if (isOlderThan(incoming, current)) return jobs
  const next = jobs.slice()
  next[index] = incoming
  next.sort(compareByUpdatedDesc)
  return next
}

export function mergeJobLists(current: JobDto[], incoming: JobDto[]): JobDto[] {
  let next = current
  for (const job of incoming) {
    next = upsertJob(next, job)
  }
  return next
}

export function removeJob(jobs: JobDto[], jobId: string): JobDto[] {
  const next = jobs.filter((job) => job.id !== jobId)
  return next.length === jobs.length ? jobs : next
}

/**
 * 应用一条 WebSocket 事件；没有实际变化时返回原数组引用，避免无意义的重渲染。
 */
export function applyJobEvent(jobs: JobDto[], event: JobEvent | null | undefined): JobDto[] {
  if (!event || typeof event !== 'object') return jobs
  if (event.type === 'job_snapshot') {
    return mergeJobLists(jobs, normalizeJobList(event.jobs))
  }
  if (typeof event.type !== 'string' || event.type.indexOf('job.') !== 0) return jobs
  const job = normalizeJob(event.job)
  if (!job) return jobs
  return upsertJob(jobs, job)
}

export function isJobEvent(event: unknown): event is JobEvent {
  if (!event || typeof event !== 'object') return false
  const type = (event as JobEvent).type
  return typeof type === 'string' && (type.indexOf('job.') === 0 || type === 'job_snapshot')
}

/**
 * 项目作用域的唯一入口：任务中心的列表、徽标与项目筛选都走这里。
 *
 * 任务事件是全局的（所有项目共用一条 /ws/jobs），因此「旧项目的事件污染当前
 * 项目视图」不可能发生：事件只写进按 id 去重的全局列表，任何项目维度的展示都
 * 必须经过本函数的 projectId 过滤。
 */
export function matchesFilters(job: JobDto, filters: JobFilters): boolean {
  if (filters.projectId && job.project_id !== filters.projectId) return false
  if (filters.onlyActive && ACTIVE_STATUSES.indexOf(job.status) < 0) return false
  if (filters.statuses.length > 0 && filters.statuses.indexOf(job.status) < 0) return false
  if (filters.jobTypes.length > 0 && filters.jobTypes.indexOf(job.job_type) < 0) return false
  if (filters.errorCodes.length > 0 && filters.errorCodes.indexOf(effectiveErrorCode(job)) < 0) return false
  const term = filters.search.trim().toLowerCase()
  if (!term) return true
  const haystack = [
    job.display_name,
    job.job_type_label,
    job.current_step,
    job.message,
    job.error_message,
    job.project_id,
    job.id,
    job.scope,
  ]
    .join(' ')
    .toLowerCase()
  return haystack.indexOf(term) >= 0
}

export function filterJobs(jobs: JobDto[], filters: JobFilters): JobDto[] {
  return jobs.filter((job) => matchesFilters(job, filters))
}

function compareByUpdatedDesc(a: JobDto, b: JobDto): number {
  // 缺失/无法解析的时间按“最旧”处理，沉到列表末尾。
  const left = parseServerTime(a.updated_at || a.created_at) ?? Number.NEGATIVE_INFINITY
  const right = parseServerTime(b.updated_at || b.created_at) ?? Number.NEGATIVE_INFINITY
  if (left === right) return a.id < b.id ? -1 : a.id > b.id ? 1 : 0
  return left < right ? 1 : -1
}

export function sortJobs(jobs: JobDto[], key: JobSortKey = 'updated'): JobDto[] {
  const sorted = jobs.slice()
  if (key === 'status') {
    sorted.sort((a, b) => {
      const diff = (STATUS_SORT_WEIGHT[a.status] ?? 9) - (STATUS_SORT_WEIGHT[b.status] ?? 9)
      return diff !== 0 ? diff : compareByUpdatedDesc(a, b)
    })
    return sorted
  }
  if (key === 'progress') {
    sorted.sort((a, b) => {
      if (a.progress !== b.progress) return b.progress - a.progress
      return compareByUpdatedDesc(a, b)
    })
    return sorted
  }
  if (key === 'created') {
    sorted.sort((a, b) => {
      const left = parseServerTime(a.created_at) ?? Number.NEGATIVE_INFINITY
      const right = parseServerTime(b.created_at) ?? Number.NEGATIVE_INFINITY
      if (left === right) return compareByUpdatedDesc(a, b)
      return left < right ? 1 : -1
    })
    return sorted
  }
  sorted.sort(compareByUpdatedDesc)
  return sorted
}

export interface JobSummary {
  activeCount: number
  failedCount: number
  interruptedCount: number
  retryableCount: number
  total: number
  latest: JobDto | null
  /** 失败任务按失败类别的计数（count 降序），用于分组标题的共性原因提示。 */
  failedByCategory: { code: string; label: string; count: number }[]
}

/** 稳定错误码：历史行可能为空，失败状态按 job_failed 归类参与筛选与统计。 */
export function effectiveErrorCode(job: JobDto): string {
  return job.error_code || (job.status === 'failed' ? 'job_failed' : '')
}

export function errorCategoryLabel(job: JobDto): string {
  if (!job.error_message && !job.error_detail) return ''
  return job.error_code_label || ERROR_CODE_LABELS[effectiveErrorCode(job)] || ''
}

/** 失败原因的首行文案：优先 LLM 摘要，其次错误码标签，最后回落原始短消息。 */
export function errorHeadline(job: JobDto): string {
  return job.error_detail?.summary || job.error_message || '任务未正常完成'
}

export function summarizeJobs(jobs: JobDto[]): JobSummary {
  let activeCount = 0
  let failedCount = 0
  let interruptedCount = 0
  let retryableCount = 0
  let latest: JobDto | null = null
  const categoryCounts = new Map<string, number>()
  for (const job of jobs) {
    if (ACTIVE_STATUSES.indexOf(job.status) >= 0) activeCount += 1
    if (job.status === 'failed') {
      failedCount += 1
      const code = effectiveErrorCode(job) || 'job_failed'
      categoryCounts.set(code, (categoryCounts.get(code) || 0) + 1)
    }
    if (job.status === 'interrupted') interruptedCount += 1
    if (job.can_retry) retryableCount += 1
    if (!latest || compareByUpdatedDesc(job, latest) < 0) latest = job
  }
  const failedByCategory = Array.from(categoryCounts.entries())
    .map(([code, count]) => ({ code, label: ERROR_CODE_LABELS[code] || code, count }))
    .sort((a, b) => b.count - a.count || a.code.localeCompare(b.code))
  return {
    activeCount,
    failedCount,
    interruptedCount,
    retryableCount,
    total: jobs.length,
    latest,
    failedByCategory,
  }
}

export interface ActionButtonState {
  enabled: boolean
  reason: string
}

export interface JobActionStates {
  cancel: ActionButtonState
  retry: ActionButtonState
  resume: ActionButtonState
  remove: ActionButtonState
}

/**
 * 所有操作按钮的 disabled 状态与原因都来自服务端 DTO，前端不自行推断业务规则。
 */
export function jobActionState(job: JobDto): JobActionStates {
  return {
    cancel: {
      enabled: job.can_cancel,
      reason: job.can_cancel ? '取消该任务' : '只有进行中的任务可以取消',
    },
    retry: {
      enabled: job.can_retry,
      reason: job.can_retry ? '重新执行该任务' : job.retry_blocked_reason || '该任务当前不可重试',
    },
    resume: {
      enabled: job.can_resume,
      reason: job.can_resume ? '从已有产物继续执行' : job.resume_blocked_reason || '该任务当前不可续跑',
    },
    remove: {
      enabled: job.can_delete,
      reason: job.can_delete ? '清理该历史记录' : '任务仍在执行，请先取消',
    },
  }
}

export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds || 0))
  if (total < 60) return total + ' 秒'
  const minutes = Math.floor(total / 60)
  const rest = total % 60
  if (minutes < 60) return minutes + ' 分 ' + String(rest).padStart(2, '0') + ' 秒'
  const hours = Math.floor(minutes / 60)
  return hours + ' 小时 ' + String(minutes % 60).padStart(2, '0') + ' 分'
}

export function formatRelativeTime(iso: string | null, now: number = Date.now()): string {
  const time = parseServerTime(iso)
  if (time === null) return '—'
  const diff = Math.round((now - time) / 1000)
  if (diff < 10) return '刚刚'
  if (diff < 60) return diff + ' 秒前'
  if (diff < 3600) return Math.floor(diff / 60) + ' 分钟前'
  if (diff < 86400) return Math.floor(diff / 3600) + ' 小时前'
  if (diff < 172800) return '昨天'
  const value = new Date(time)
  return String(value.getMonth() + 1).padStart(2, '0') + '-' + String(value.getDate()).padStart(2, '0')
}

export function formatClock(iso: string | null): string {
  const time = parseServerTime(iso)
  if (time === null) return '—'
  const value = new Date(time)
  return (
    String(value.getHours()).padStart(2, '0') +
    ':' +
    String(value.getMinutes()).padStart(2, '0') +
    ':' +
    String(value.getSeconds()).padStart(2, '0')
  )
}

export function progressText(job: JobDto): string {
  if (job.status === 'completed') return '100%'
  return Math.max(0, Math.min(100, job.progress)) + '%'
}

export function stepText(job: JobDto): string {
  return job.current_step || job.message || job.status_label
}

export function errorSummary(job: JobDto): string {
  if (!job.error_message) return ''
  const label = errorCategoryLabel(job)
  return label ? label + '：' + job.error_message : job.error_message
}

export function queryFromFilters(
  filters: JobFilters,
  page = 1,
  pageSize = 50,
): JobQueryParams {
  const params: JobQueryParams = { page, page_size: pageSize }
  if (filters.projectId) params.project_id = filters.projectId
  if (filters.statuses.length > 0) params.status = filters.statuses.slice()
  if (filters.jobTypes.length > 0) params.job_type = filters.jobTypes.slice()
  if (filters.onlyActive) params.active_only = true
  const term = filters.search.trim()
  if (term) params.q = term
  return params
}

export const RECONNECT_BASE_MS = 1000
export const RECONNECT_MAX_MS = 15000

/** 指数退避（确定性，便于测试）；调用方负责在成功后归零。 */
export function nextReconnectDelay(
  attempt: number,
  baseMs: number = RECONNECT_BASE_MS,
  maxMs: number = RECONNECT_MAX_MS,
): number {
  const safeAttempt = Math.max(0, Math.floor(attempt))
  return Math.min(maxMs, baseMs * Math.pow(2, safeAttempt))
}

export function connectionLabel(state: string): string {
  if (state === 'open') return '实时连接正常'
  if (state === 'connecting') return '正在连接任务中心…'
  if (state === 'reconnecting') return '连接已断开，正在重连…'
  if (state === 'error') return '连接异常，将自动重试'
  return '未连接'
}

// --- 分组标签页（键盘可访问的 roving tabindex） --------------------------

export const SECTION_TAB_ID_PREFIX = 'task-center-tab-'
export const SECTION_PANEL_ID = 'task-center-panel-list'

export interface SectionTabAriaProps {
  id: string
  role: 'tab'
  'aria-selected': boolean
  'aria-controls': string
  tabIndex: 0 | -1
}

/** 只有选中的标签 tabIndex=0，其余为 -1；方向键导航由 resolveWorkspaceTabIndex 负责。 */
export function sectionTabAriaProps(tabId: string, activeTabId: string): SectionTabAriaProps {
  const selected = tabId === activeTabId
  return {
    id: SECTION_TAB_ID_PREFIX + tabId,
    role: 'tab',
    'aria-selected': selected,
    'aria-controls': SECTION_PANEL_ID,
    tabIndex: selected ? 0 : -1,
  }
}

/** 「跳转到对应项目」的事件载荷；任务没有项目上下文时返回 null（按钮不渲染）。 */
export function projectJumpDetail(job: JobDto): { projectId: string } | null {
  return job.project_id ? { projectId: job.project_id } : null
}

export function isActiveStatus(status: JobStatus): boolean {
  return ACTIVE_STATUSES.indexOf(status) >= 0
}

export function isRetryableStatus(status: JobStatus): boolean {
  return RETRYABLE_STATUSES.indexOf(status) >= 0
}

export const EMPTY_STATE_TEXT = '暂无任务记录'
export const EMPTY_FILTERED_TEXT = '没有符合当前筛选条件的任务'
export const LOADING_TEXT = '正在加载任务…'

export function emptyStateText(hasJobs: boolean, filtersActive: boolean): string {
  if (filtersActive) return EMPTY_FILTERED_TEXT
  return hasJobs ? EMPTY_STATE_TEXT : '还没有任何后台任务，开始生成后会显示在这里'
}
