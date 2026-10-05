/**
 * 全局任务中心状态。
 *
 * 数据流：
 * 1. REST 初始快照（``GET /api/jobs``）确定权威列表；
 * 2. 全局 WebSocket（``/ws/jobs``）推送增量事件，按 id 去重合并；
 * 3. 断线自动重连，重连成功后**重新拉一次 REST 快照**，不依赖增量事件补齐；
 * 4. 低频兜底轮询，作为 WebSocket 长时间不可用时的保底。
 *
 * 性能与安全：
 * - 进度事件先进入缓冲区，按 120ms 批量刷新，避免每条进度都触发全量渲染；
 * - 旧连接（epoch 不匹配）的迟到事件一律丢弃，避免重连竞态污染状态；
 * - 任务 DTO 本身不含 run token，store 也只保存 DTO。
 */

import { create } from 'zustand'

import { mergeDebugEvents } from '../components/progressDebugModel.ts'

import type { JobActionResult, JobDebugEvent, JobDetailDto, JobDto, JobEvent } from '../services/jobTypes.ts'
import { createJobsWebSocket, describeJobApiError, jobApi } from '../services/api.ts'
import {
  ACTIVE_STATUSES,
  DEFAULT_FILTERS,
  applyJobEvent,
  filterJobs,
  isJobEvent,
  matchesFilters,
  mergeJobLists,
  nextReconnectDelay,
  normalizeJob,
  normalizeJobList,
  queryFromFilters,
  removeJob,
  sortJobs,
  summarizeJobs,
  type JobFilters,
  type JobSortKey,
  type JobSummary,
} from '../components/taskCenterModel.ts'

export type ConnectionState = 'idle' | 'connecting' | 'open' | 'reconnecting' | 'closed' | 'error'

// 进度事件合并窗口：足够小以保证「实时」，足够大以避免每条进度都重渲染。
const PROGRESS_FLUSH_MS = 120
// WebSocket 不可用时的兜底轮询间隔。
const FALLBACK_POLL_MS = 20000
// 一次同步最多拉取的任务数量（与后端 JOB_LIST_MAX_PAGE_SIZE 对齐）。
const SYNC_PAGE_SIZE = 100

interface SocketLike {
  readyState: number
  close(): void
  send(data: string): void
  onopen: ((event: unknown) => void) | null
  onclose: ((event: unknown) => void) | null
  onerror: ((event: unknown) => void) | null
  onmessage: ((event: { data: unknown }) => void) | null
}

type SocketFactory = (
  onMessage: (data: unknown) => void,
  options: {
    onOpen?: () => void
    onClose?: () => void
    onError?: () => void
  },
) => SocketLike

const defaultSocketFactory: SocketFactory = (onMessage, options) =>
  createJobsWebSocket(onMessage, options) as unknown as SocketLike

interface DerivedState {
  visibleJobs: JobDto[]
  activeJobs: JobDto[]
  failedJobs: JobDto[]
  summary: JobSummary
}

export interface TaskStoreState extends DerivedState {
  jobs: JobDto[]
  filters: JobFilters
  sortKey: JobSortKey
  selectedJobId: string | null
  selectedJob: JobDetailDto | null
  loading: boolean
  detailLoading: boolean
  busyJobId: string | null
  error: string
  notice: string
  lastSyncedAt: number | null
  connectionState: ConnectionState
  reconnectAttempt: number
  panelOpen: boolean
  /** 单条任务的实时调试事件；REST 全量轨迹由调试面板首次加载。 */
  debugEventsByJob: Record<string, JobDebugEvent[]>

  setPanelOpen: (open: boolean) => void
  setFilters: (patch: Partial<JobFilters>) => void
  resetFilters: () => void
  setSortKey: (key: JobSortKey) => void
  sync: (options?: { silent?: boolean }) => Promise<void>
  selectJob: (jobId: string | null) => Promise<void>
  requestCancel: (jobId: string) => Promise<JobActionResult>
  requestRetry: (jobId: string) => Promise<JobActionResult>
  requestResume: (jobId: string) => Promise<JobActionResult>
  requestDelete: (jobId: string) => Promise<JobActionResult>
  cleanupHistory: () => Promise<JobActionResult>
  note: (message: string) => void
  dismissNotice: () => void
  setDebugEvents: (jobId: string, events: JobDebugEvent[]) => void
  appendDebugEvent: (jobId: string, event: JobDebugEvent) => void
  start: () => void
  stop: () => void
  dispose: () => void
}

function deriveJobs(jobs: JobDto[], filters: JobFilters, sortKey: JobSortKey): DerivedState {
  const visibleJobs = sortJobs(filterJobs(jobs, filters), sortKey)
  const scopeFilters: JobFilters = { ...filters, statuses: [], onlyActive: false }
  const scoped = filterJobs(jobs, scopeFilters)
  const activeJobs = sortJobs(
    scoped.filter((job) => ACTIVE_STATUSES.indexOf(job.status) >= 0),
    'updated',
  )
  const failedJobs = sortJobs(
    scoped.filter((job) => job.can_retry || job.status === 'failed' || job.status === 'interrupted'),
    'updated',
  )
  return { visibleJobs, activeJobs, failedJobs, summary: summarizeJobs(scoped) }
}

// --- 运行时（不进入渲染状态） ---------------------------------------------

let socket: SocketLike | null = null
let socketEpoch = 0
let reconnectTimer: ReturnType<typeof setTimeout> | null = null
let pollTimer: ReturnType<typeof setInterval> | null = null
let flushTimer: ReturnType<typeof setTimeout> | null = null
let pendingEvents: JobEvent[] = []
let syncInFlight: Promise<void> | null = null
let started = false

let socketFactory: SocketFactory = defaultSocketFactory

/** 仅测试使用：替换 socket 工厂，避免真实网络连接。 */
export function __setSocketFactoryForTests(factory: SocketFactory | null): void {
  socketFactory = factory || defaultSocketFactory
}

/** 仅测试使用：立即冲刷待处理的事件缓冲区。 */
export function __flushPendingEvents(): void {
  flushEvents()
}

function clearReconnectTimer(): void {
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer)
    reconnectTimer = null
  }
}

function clearPollTimer(): void {
  if (pollTimer !== null) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

function clearFlushTimer(): void {
  if (flushTimer !== null) {
    clearTimeout(flushTimer)
    flushTimer = null
  }
}

function applyJobs(next: JobDto[]): void {
  const state = useTaskStore.getState()
  if (next === state.jobs) return
  useTaskStore.setState({ jobs: next, ...deriveJobs(next, state.filters, state.sortKey) })
}

function flushEvents(): void {
  clearFlushTimer()
  if (pendingEvents.length === 0) return
  const events = pendingEvents
  pendingEvents = []
  const state = useTaskStore.getState()
  let next = state.jobs
  for (const event of events) {
    next = applyJobEvent(next, event)
  }
  if (next === state.jobs) return
  useTaskStore.setState({ jobs: next, ...deriveJobs(next, state.filters, state.sortKey) })
}

function scheduleFlush(): void {
  if (flushTimer !== null) return
  flushTimer = setTimeout(() => {
    flushTimer = null
    flushEvents()
  }, PROGRESS_FLUSH_MS)
  // node:test 环境没有 unref 以外的定时器语义，这里只在可用时解除引用。
  const timer = flushTimer as unknown as { unref?: () => void }
  if (typeof timer.unref === 'function') timer.unref()
}

function handleSocketMessage(epoch: number, payload: unknown): void {
  // 旧连接（已被新一轮重连取代）的迟到事件必须丢弃，否则会用过期状态覆盖新状态。
  if (epoch !== socketEpoch) return
  if (!payload || typeof payload !== 'object') return
  const event = payload as JobEvent
  if (event.type === 'pong') return
  if (!isJobEvent(event)) return
  if (event.type === 'job.debug') {
    if (event.job_id && event.event) {
      useTaskStore.getState().appendDebugEvent(event.job_id, event.event)
    }
    return
  }
  if (event.type === 'job_snapshot') {
    pendingEvents.push(event)
    flushEvents()
    return
  }
  pendingEvents.push(event)
  scheduleFlush()
}

function scheduleReconnect(): void {
  if (!started || reconnectTimer !== null) return
  const state = useTaskStore.getState()
  if (state.connectionState === 'open' || state.connectionState === 'connecting') return
  const attempt = state.reconnectAttempt
  const delay = nextReconnectDelay(attempt)
  useTaskStore.setState({ connectionState: 'reconnecting', reconnectAttempt: attempt + 1 })
  clearReconnectTimer()
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null
    openSocket()
  }, delay)
  const timer = reconnectTimer as unknown as { unref?: () => void }
  if (typeof timer.unref === 'function') timer.unref()
}

function openSocket(): void {
  if (!started) return
  socketEpoch += 1
  const epoch = socketEpoch
  useTaskStore.setState({ connectionState: 'connecting' })

  let next: SocketLike
  try {
    next = socketFactory((payload) => handleSocketMessage(epoch, payload), {
      onOpen: () => {
        if (epoch !== socketEpoch) return
        useTaskStore.setState({ connectionState: 'open', reconnectAttempt: 0, error: '' })
        // 重连（以及首次连接）后都必须重新取一次 REST 快照：只靠增量事件无法
        // 保证断线期间发生的状态变化被补齐。
        void useTaskStore.getState().sync({ silent: true })
      },
      onClose: () => {
        if (epoch !== socketEpoch) return
        useTaskStore.setState({ connectionState: 'closed' })
        scheduleReconnect()
      },
      onError: () => {
        if (epoch !== socketEpoch) return
        useTaskStore.setState({ connectionState: 'error' })
        scheduleReconnect()
      },
    })
  } catch (_error) {
    useTaskStore.setState({ connectionState: 'error' })
    scheduleReconnect()
    return
  }
  socket = next
}

function closeSocket(): void {
  socketEpoch += 1
  const current = socket
  socket = null
  if (!current) return
  current.onopen = null
  current.onclose = null
  current.onerror = null
  current.onmessage = null
  try {
    current.close()
  } catch (_error) {
    // 关闭失败不影响状态清理
  }
}

export const useTaskStore = create<TaskStoreState>((set, get) => ({
  jobs: [],
  ...deriveJobs([], DEFAULT_FILTERS, 'updated'),
  filters: { ...DEFAULT_FILTERS },
  sortKey: 'updated',
  selectedJobId: null,
  selectedJob: null,
  loading: false,
  detailLoading: false,
  busyJobId: null,
  error: '',
  notice: '',
  lastSyncedAt: null,
  connectionState: 'idle',
  reconnectAttempt: 0,
  panelOpen: false,
  debugEventsByJob: {},

  setPanelOpen: (open) => set({ panelOpen: open }),

  setFilters: (patch) => {
    const filters = { ...get().filters, ...patch }
    set({ filters, ...deriveJobs(get().jobs, filters, get().sortKey) })
  },

  resetFilters: () => {
    const filters = { ...DEFAULT_FILTERS }
    set({ filters, ...deriveJobs(get().jobs, filters, get().sortKey) })
  },

  setSortKey: (key) => {
    set({ sortKey: key, ...deriveJobs(get().jobs, get().filters, key) })
  },

  sync: async (options) => {
    const silent = options?.silent === true
    if (syncInFlight) return syncInFlight
    if (!silent) set({ loading: true, error: '' })
    syncInFlight = (async () => {
      try {
        const response = await jobApi.list({ page: 1, page_size: SYNC_PAGE_SIZE })
        const incoming = normalizeJobList(response.items)
        const state = useTaskStore.getState()
        const merged = mergeJobLists(state.jobs, incoming)
        useTaskStore.setState({
          jobs: merged,
          ...deriveJobs(merged, state.filters, state.sortKey),
          loading: false,
          error: '',
          lastSyncedAt: Date.now(),
        })
      } catch (error) {
        const described = describeJobApiError(error, '任务列表加载失败')
        useTaskStore.setState({ loading: false, error: described.message })
      } finally {
        syncInFlight = null
      }
    })()
    return syncInFlight
  },

  selectJob: async (jobId) => {
    if (!jobId) {
      set({ selectedJobId: null, selectedJob: null })
      return
    }
    set({ selectedJobId: jobId, detailLoading: true })
    try {
      const detail = (await jobApi.detail(jobId)) as JobDetailDto
      if (useTaskStore.getState().selectedJobId !== jobId) return
      const base = normalizeJob(detail)
      set({
        selectedJob: base ? { ...(detail as JobDetailDto), ...base } : null,
        detailLoading: false,
      })
    } catch (error) {
      if (useTaskStore.getState().selectedJobId !== jobId) return
      const described = describeJobApiError(error, '任务详情加载失败')
      set({ selectedJob: null, detailLoading: false, error: described.message })
    }
  },

  requestCancel: (jobId) => runJobAction(jobId, () => jobApi.cancel(jobId), '已请求取消任务'),

  requestRetry: (jobId) => runJobAction(jobId, () => jobApi.retry(jobId), '已重新派发任务'),

  requestResume: (jobId) => runJobAction(jobId, () => jobApi.resume(jobId), '已从已有产物继续执行'),

  requestDelete: (jobId) => runJobAction(jobId, () => jobApi.remove(jobId), '任务记录已清理', { dropLocalJob: true }),

  cleanupHistory: async () => {
    const filters = get().filters
    const terminalStatuses = filters.statuses.filter((item) => ACTIVE_STATUSES.indexOf(item) < 0)
    if (filters.statuses.length > 0 && terminalStatuses.length === 0) {
      const described: JobActionResult = { ok: true, status: 'noop', message: '当前筛选下没有可清理的历史记录' }
      set({ notice: described.message, error: '' })
      return described
    }
    set({ busyJobId: '__cleanup__', notice: '' })
    try {
      const params = queryFromFilters(filters, 1, 1)
      delete params.page
      delete params.page_size
      delete params.active_only
      if (terminalStatuses.length > 0) {
        // 只清理终态历史：运行中的任务永远不在这里被删除。
        params.status = terminalStatuses.slice()
      } else {
        delete params.status
      }
      const result = await jobApi.cleanup(params)
      const described: JobActionResult = {
        ok: true,
        status: 'deleted',
        message: result.deleted > 0 ? '已清理 ' + result.deleted + ' 条历史记录' : '没有可清理的历史记录',
      }
      // 服务端批量删除没有逐条回执：本地按同一组条件剪掉已删除的终态任务，
      // 再拉一次权威快照，避免历史记录在界面上「删不掉」。
      const pruneFilters = { ...filters, statuses: terminalStatuses, onlyActive: false }
      const next = get().jobs.filter((job) => job.is_active || !matchesFilters(job, pruneFilters))
      useTaskStore.setState({ jobs: next, ...deriveJobs(next, filters, get().sortKey) })
      set({ busyJobId: null, notice: described.message, error: '' })
      await get().sync({ silent: true })
      return described
    } catch (error) {
      const described = describeJobApiError(error, '清理历史失败')
      set({ busyJobId: null, error: described.message })
      return described
    }
  },

  note: (message) => set({ notice: message }),
  dismissNotice: () => set({ notice: '' }),

  setDebugEvents: (jobId, events) => {
    if (!jobId) return
    set((state) => ({
      debugEventsByJob: {
        ...state.debugEventsByJob,
        [jobId]: mergeDebugEvents(state.debugEventsByJob[jobId], events).slice(-200),
      },
    }))
  },

  appendDebugEvent: (jobId, event) => {
    if (!jobId || !event?.id) return
    set((state) => {
      const current = state.debugEventsByJob[jobId] || []
      if (current.some((item) => item.id === event.id)) return state
      return {
        debugEventsByJob: {
          ...state.debugEventsByJob,
          [jobId]: [...current, event].slice(-200),
        },
      }
    })
  },

  start: () => {
    if (started) return
    started = true
    void get().sync()
    openSocket()
    clearPollTimer()
    pollTimer = setInterval(() => {
      if (useTaskStore.getState().connectionState === 'open') return
      void useTaskStore.getState().sync({ silent: true })
    }, FALLBACK_POLL_MS)
    const timer = pollTimer as unknown as { unref?: () => void }
    if (typeof timer.unref === 'function') timer.unref()
  },

  stop: () => {
    started = false
    clearReconnectTimer()
    clearPollTimer()
    clearFlushTimer()
    pendingEvents = []
    closeSocket()
    set({ connectionState: 'idle' })
  },

  dispose: () => {
    get().stop()
    set({
      jobs: [],
      ...deriveJobs([], DEFAULT_FILTERS, 'updated'),
      filters: { ...DEFAULT_FILTERS },
      sortKey: 'updated',
      selectedJobId: null,
      selectedJob: null,
      loading: false,
      detailLoading: false,
      busyJobId: null,
      error: '',
      notice: '',
      lastSyncedAt: null,
      reconnectAttempt: 0,
      panelOpen: false,
    })
  },
}))

async function runJobAction(
  jobId: string,
  request: () => Promise<JobActionResult>,
  successMessage: string,
  options: { dropLocalJob?: boolean } = {},
): Promise<JobActionResult> {
  useTaskStore.setState({ busyJobId: jobId, notice: '' })
  try {
    const result = await request()
    if (result.ok === false) {
      useTaskStore.setState({ busyJobId: null, error: result.message || '操作失败' })
      return result
    }
    useTaskStore.setState({ busyJobId: null, error: '', notice: result.message || successMessage })
    if (options.dropLocalJob) {
      const state = useTaskStore.getState()
      const next = removeJob(state.jobs, jobId)
      useTaskStore.setState({ jobs: next, ...deriveJobs(next, state.filters, state.sortKey) })
    } else if (result.job) {
      applyJobs(applyJobEvent(useTaskStore.getState().jobs, { type: 'job.updated', job: result.job }))
    }
    // 服务端状态已落库：立即取一次权威快照，而不是停留在本地乐观状态。
    await useTaskStore.getState().sync({ silent: true })
    const selectedId = useTaskStore.getState().selectedJobId
    if (selectedId === jobId) {
      await useTaskStore.getState().selectJob(jobId)
    }
    return result
  } catch (error) {
    const described = describeJobApiError(error, '操作失败')
    useTaskStore.setState({ busyJobId: null, error: described.message })
    return described
  }
}
