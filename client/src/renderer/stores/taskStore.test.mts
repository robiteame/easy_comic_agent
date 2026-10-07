import assert from 'node:assert/strict'

import { jobApi } from '../services/api.ts'
import { __setSocketFactoryForTests, useTaskStore } from './taskStore.ts'
import { DEFAULT_FILTERS } from '../components/taskCenterModel.ts'

import { test } from 'vitest'

test('module assertions', async () => {
  // --- 测试替身 -------------------------------------------------------------

  class FakeSocket {
    readyState = 1
    onopen: ((event: unknown) => void) | null = null
    onclose: ((event: unknown) => void) | null = null
    onerror: ((event: unknown) => void) | null = null
    onmessage: ((event: { data: unknown }) => void) | null = null
    sent: string[] = []
    closed = false

    send(data: string): void {
      this.sent.push(data)
    }

    close(): void {
      this.closed = true
      this.readyState = 3
    }

    open(): void {
      this.onopen?.({})
    }

    emit(payload: unknown): void {
      this.onmessage?.({ data: JSON.stringify(payload) })
    }

    drop(): void {
      this.readyState = 3
      this.onclose?.({})
    }
  }

  const sockets: FakeSocket[] = []
  __setSocketFactoryForTests((onMessage, options) => {
    const socket = new FakeSocket()
    socket.onopen = () => options.onOpen?.()
    socket.onclose = () => options.onClose?.()
    socket.onerror = () => options.onError?.()
    // 真实实现（createJobsWebSocket）会先做 JSON 解析再交给 store，这里保持一致。
    socket.onmessage = (event) => onMessage(JSON.parse(String(event.data)))
    sockets.push(socket)
    return socket as never
  })

  let listCalls = 0
  let failList = false
  let listPayload: unknown[] = []
  const recorded: string[] = []

  jobApi.list = (async () => {
    listCalls += 1
    if (failList) throw new Error('network down')
    return {
      items: listPayload,
      total: listPayload.length,
      page: 1,
      page_size: 100,
      pages: 1,
      active_count: 1,
      status_counts: {},
      job_type_counts: {},
      generated_at: '2025-01-01T00:00:00',
    }
  }) as never

  jobApi.detail = (async (jobId: string) => ({ ...(listPayload[0] as object), id: jobId, attempts: [] })) as never
  jobApi.cancel = (async () => {
    recorded.push('cancel')
    return { ok: true, status: 'cancelling', message: '已请求取消，等待当前步骤安全退出' }
  }) as never
  jobApi.retry = (async () => {
    recorded.push('retry')
    return {
      ok: false,
      status: 'job_not_retryable',
      message: '只有失败、已取消或已中断的任务可以重试',
      error_code: 'job_not_retryable',
    }
  }) as never
  jobApi.resume = (async () => {
    recorded.push('resume')
    return { ok: true, status: 'started', message: '已从已有产物继续执行' }
  }) as never
  jobApi.remove = (async () => {
    recorded.push('remove')
    return { ok: true, status: 'deleted', message: '任务记录已清理' }
  }) as never
  jobApi.cleanup = (async () => {
    recorded.push('cleanup')
    return {
      deleted: 2,
      stats: { active_count: 0, failed_count: 0, total: 0, status_counts: {}, latest_job: null, generated_at: '' },
    }
  }) as never

  function jobPayload(overrides: Record<string, unknown> = {}) {
    return {
      id: 'job-1',
      scope: 'project:p1',
      project_id: 'p1',
      job_type: 'render',
      job_type_label: '成片渲染',
      display_name: '成片渲染 · 项目一',
      status: 'running',
      status_label: '进行中',
      progress: 10,
      current_step: 'rendering',
      message: '开始导出成片',
      error_code: '',
      error_message: '',
      attempt: 1,
      retry_of: null,
      version: 1,
      created_at: '2025-01-01T10:00:00',
      started_at: '2025-01-01T10:00:00',
      updated_at: '2025-01-01T10:00:10',
      finished_at: null,
      cancel_requested_at: null,
      duration_seconds: 10,
      eta_seconds: null,
      is_active: true,
      is_terminal: false,
      has_active_successor: false,
      can_cancel: true,
      can_retry: false,
      can_resume: false,
      can_delete: false,
      retry_blocked_reason: '',
      resume_blocked_reason: '',
      ...overrides,
    }
  }

  const tick = () => new Promise((resolve) => setTimeout(resolve, 0))
  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

  // --- REST 初始快照 --------------------------------------------------------

  useTaskStore.getState().dispose()
  assert.deepEqual(useTaskStore.getState().jobs, [], '初始状态不应有任务')

  listPayload = [jobPayload()]
  useTaskStore.getState().start()
  await tick()
  await tick()
  assert.equal(useTaskStore.getState().jobs.length, 1, '启动后应拉取一次 REST 快照')
  assert.equal(useTaskStore.getState().summary.activeCount, 1, '活动任务数量应来自快照')
  assert.equal(useTaskStore.getState().activeJobs.length, 1)
  assert.ok(useTaskStore.getState().lastSyncedAt, '同步成功后应记录 lastSyncedAt')
  assert.equal(sockets.length, 1, '启动后应建立一次任务事件连接')

  const socket = sockets[0]
  socket.open()
  await tick()
  assert.equal(useTaskStore.getState().connectionState, 'open', '连接建立后状态应为 open')
  assert.equal(listCalls, 2, '连接成功后必须再取一次 REST 快照（重连补齐语义）')

  // --- WebSocket 增量与去重 --------------------------------------------------

  socket.emit({
    type: 'job.progress',
    job: jobPayload({ progress: 55, message: '正在合成', updated_at: '2025-01-01T10:01:00' }),
  })
  assert.equal(useTaskStore.getState().jobs[0].progress, 10, '进度事件应先进入缓冲区，避免每条事件都触发全量渲染')
  await sleep(200)
  assert.equal(useTaskStore.getState().jobs[0].progress, 55, '缓冲区刷新后进度应更新')
  assert.equal(useTaskStore.getState().jobs.length, 1, '同一条任务的多次事件不得产生重复条目')

  socket.emit({
    type: 'job.progress',
    job: jobPayload({ progress: 5, message: '过期事件', updated_at: '2025-01-01T09:00:00' }),
  })
  await sleep(200)
  assert.equal(useTaskStore.getState().jobs[0].progress, 55, '更旧的事件必须被丢弃，不能回退状态')

  socket.emit({
    type: 'job_snapshot',
    jobs: [
      jobPayload({ progress: 60, updated_at: '2025-01-01T10:02:00' }),
      jobPayload({ id: 'job-2', display_name: '第二个任务', project_id: 'p2', updated_at: '2025-01-01T10:02:00' }),
    ],
  })
  await tick()
  assert.equal(useTaskStore.getState().jobs.length, 2, '快照应合并进本地列表')
  assert.equal(useTaskStore.getState().jobs.find((job) => job.id === 'job-1')?.progress, 60)

  // --- 筛选与排序 -----------------------------------------------------------

  useTaskStore.getState().setFilters({ projectId: 'p2' })
  assert.equal(useTaskStore.getState().visibleJobs.length, 1, '项目筛选应作用在可见列表上')
  assert.equal(useTaskStore.getState().activeJobs.length, 1)
  useTaskStore.getState().setFilters({ ...DEFAULT_FILTERS, search: '不存在' })
  assert.equal(useTaskStore.getState().visibleJobs.length, 0, '关键词无命中时可见列表应为空')
  useTaskStore.getState().setSortKey('status')
  assert.equal(useTaskStore.getState().sortKey, 'status')
  useTaskStore.getState().resetFilters()
  assert.equal(useTaskStore.getState().visibleJobs.length, 2, '重置筛选后应恢复完整列表')

  // --- 操作：等待服务端结果 + 刷新 -------------------------------------------

  const cancelResult = await useTaskStore.getState().requestCancel('job-1')
  assert.equal(cancelResult.ok, true)
  assert.deepEqual(recorded.slice(0, 1), ['cancel'], '取消必须真正打到服务端')
  assert.equal(useTaskStore.getState().busyJobId, null, '操作结束后应清空忙碌标记')
  assert.ok(listCalls >= 3, '操作完成后应重新拉取服务端快照，而不是停留在本地乐观状态')

  const retryResult = await useTaskStore.getState().requestRetry('job-1')
  assert.equal(retryResult.ok, false)
  assert.equal(useTaskStore.getState().error, '只有失败、已取消或已中断的任务可以重试', '服务端的拒绝原因应原样呈现')

  const resumeResult = await useTaskStore.getState().requestResume('job-1')
  assert.equal(resumeResult.ok, true)
  assert.equal(useTaskStore.getState().notice, '已从已有产物继续执行')
  useTaskStore.getState().dismissNotice()
  assert.equal(useTaskStore.getState().notice, '')

  const deleteResult = await useTaskStore.getState().requestDelete('job-2')
  assert.equal(deleteResult.ok, true)
  assert.equal(
    useTaskStore.getState().jobs.find((job) => job.id === 'job-2'),
    undefined,
    '删除成功后必须立刻从本地列表移除，不能等下一次快照（合并式同步不会删除条目）',
  )

  // 只筛活动状态时不应误删整段历史
  useTaskStore.getState().setFilters({ statuses: ['running'] })
  const noopCleanup = await useTaskStore.getState().cleanupHistory()
  assert.equal(noopCleanup.status, 'noop')
  assert.equal(recorded.indexOf('cleanup'), -1, '没有可清理的历史时不应调用服务端批量删除')
  useTaskStore.getState().resetFilters()

  listPayload = [jobPayload()]
  await useTaskStore.getState().sync()
  const cleanupResult = await useTaskStore.getState().cleanupHistory()
  assert.equal(cleanupResult.ok, true)
  assert.equal(cleanupResult.message, '已清理 2 条历史记录')

  // --- 网络失败状态 ---------------------------------------------------------

  failList = true
  await useTaskStore.getState().sync()
  assert.equal(useTaskStore.getState().loading, false)
  assert.match(useTaskStore.getState().error, /网络不可用|加载失败/, '网络失败必须给出可见错误而不是静默失败')
  failList = false
  await useTaskStore.getState().sync()
  assert.equal(useTaskStore.getState().error, '', '恢复后应清空错误')

  // --- 断线与重连 -----------------------------------------------------------

  const before = sockets.length
  useTaskStore.setState({ reconnectAttempt: 0 })
  socket.drop()
  assert.equal(useTaskStore.getState().connectionState, 'reconnecting', '断线后应进入重连状态')
  const syncsBeforeReconnect = listCalls
  await sleep(1150)
  assert.equal(sockets.length, before + 1, '应在退避后自动重连')
  const reconnected = sockets[sockets.length - 1]
  reconnected.open()
  await tick()
  await tick()
  assert.equal(useTaskStore.getState().connectionState, 'open', '重连成功后状态应恢复为 open')
  assert.ok(listCalls > syncsBeforeReconnect, '重连后必须重新获取 REST 快照，不能只依赖增量事件')

  // 旧连接的迟到事件不得污染新状态
  socket.emit({ type: 'job.progress', job: jobPayload({ progress: 99, updated_at: '2025-01-01T10:09:00' }) })
  await sleep(200)
  assert.equal(useTaskStore.getState().jobs[0].progress, 60, '旧 socket 的事件必须被丢弃')

  // --- 卸载清理 -------------------------------------------------------------

  useTaskStore.getState().dispose()
  assert.equal(useTaskStore.getState().connectionState, 'idle')
  assert.equal(useTaskStore.getState().jobs.length, 0)
  assert.equal(reconnected.closed, true, '卸载时应关闭 WebSocket')
  assert.deepEqual(useTaskStore.getState().filters, DEFAULT_FILTERS, '卸载后筛选条件应复位')

  await sleep(1200)
  assert.equal(sockets.length, before + 1, '停止后不得再自动重连')

  __setSocketFactoryForTests(null)
  console.log('taskStore.test.mts ok')
})
