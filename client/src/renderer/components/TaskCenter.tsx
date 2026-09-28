import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  BulbOutlined,
  CloseOutlined,
  DeleteOutlined,
  ExclamationCircleOutlined,
  FolderOpenOutlined,
  InfoCircleOutlined,
  LoadingOutlined,
  PlayCircleOutlined,
  RedoOutlined,
  ReloadOutlined,
  StopOutlined,
} from '@ant-design/icons'
import Button from 'antd/es/button'
import Empty from 'antd/es/empty'
import Input from 'antd/es/input'
import Progress from 'antd/es/progress'
import Select from 'antd/es/select'
import Spin from 'antd/es/spin'
import Tooltip from 'antd/es/tooltip'

import type { JobDto, JobStatus, JobType } from '../services/jobTypes'
import type { JobCostDetailDto, JobStatsCostDto } from '../services/costTypes'
import { budgetApi, jobApi, regenerationQueueApi } from '../services/api'
import {
  NO_USAGE_TEXT,
  UNKNOWN_COST_TEXT,
  describeBudgetError,
  durationSourceLabel,
  estimateCostText,
  formatCostValue,
  formatDurationText,
  formatMillisecondsText,
  jobCostComparisonText,
  jobCostText,
  jobDurationComparisonText,
  normalizeJobCostDetail,
  normalizeJobStatsCost,
  unknownCostReason,
} from '../services/costModel'
import { useTaskStore } from '../stores/taskStore'
import { useProjectStore } from '../stores/projectStore'
import {
  ERROR_CODE_LABELS,
  FAILURE_CATEGORY_OPTIONS,
  JOB_STATUSES,
  JOB_TYPES,
  JOB_TYPE_LABELS,
  SORT_LABELS,
  STATUS_LABELS,
  connectionLabel,
  emptyStateText,
  errorCategoryLabel,
  errorCodeTone,
  errorHeadline,
  effectiveErrorCode,
  formatDuration,
  formatRelativeTime,
  jobActionState,
  jobSection,
  progressText,
  projectJumpDetail,
  SECTION_PANEL_ID,
  SECTION_TAB_ID_PREFIX,
  sectionTabAriaProps,
  stepText,
  type JobSectionTone,
  type JobSortKey,
} from './taskCenterModel'
import { resolveWorkspaceTabIndex } from './workspaceTabs'

export const OPEN_TASK_CENTER_EVENT = 'workspace:open-task-center'

const SECTION_ORDER: JobSectionTone[] = ['active', 'failed', 'interrupted', 'cancelled', 'completed']
const SECTION_TITLES: Record<JobSectionTone, string> = {
  active: '活动任务',
  failed: '失败与可重试',
  interrupted: '被中断',
  cancelled: '已取消',
  completed: '最近完成',
}
type SectionFilter = JobSectionTone | 'all'

const SECTION_TABS: { id: SectionFilter; label: string }[] = [
  { id: 'all', label: '全部' },
  { id: 'active', label: '活动' },
  { id: 'failed', label: '失败' },
  { id: 'interrupted', label: '已中断' },
  { id: 'cancelled', label: '已取消' },
  { id: 'completed', label: '已完成' },
]

function statusClass(status: JobStatus): string {
  return 'task-status task-status-' + status
}

interface TaskCenterProps {
  open: boolean
  onClose: () => void
}

const TaskCenter: React.FC<TaskCenterProps> = ({ open, onClose }) => {
  const jobs = useTaskStore((state) => state.jobs)
  const visibleJobs = useTaskStore((state) => state.visibleJobs)
  const summary = useTaskStore((state) => state.summary)
  const filters = useTaskStore((state) => state.filters)
  const sortKey = useTaskStore((state) => state.sortKey)
  const loading = useTaskStore((state) => state.loading)
  const detailLoading = useTaskStore((state) => state.detailLoading)
  const error = useTaskStore((state) => state.error)
  const notice = useTaskStore((state) => state.notice)
  const connectionState = useTaskStore((state) => state.connectionState)
  const lastSyncedAt = useTaskStore((state) => state.lastSyncedAt)
  const busyJobId = useTaskStore((state) => state.busyJobId)
  const selectedJobId = useTaskStore((state) => state.selectedJobId)
  const selectedJob = useTaskStore((state) => state.selectedJob)

  const setFilters = useTaskStore((state) => state.setFilters)
  const resetFilters = useTaskStore((state) => state.resetFilters)
  const setSortKey = useTaskStore((state) => state.setSortKey)
  const sync = useTaskStore((state) => state.sync)
  const selectJob = useTaskStore((state) => state.selectJob)
  const requestCancel = useTaskStore((state) => state.requestCancel)
  const requestRetry = useTaskStore((state) => state.requestRetry)
  const requestResume = useTaskStore((state) => state.requestResume)
  const requestDelete = useTaskStore((state) => state.requestDelete)
  const cleanupHistory = useTaskStore((state) => state.cleanupHistory)
  const dismissNotice = useTaskStore((state) => state.dismissNotice)

  const currentProjectId = useProjectStore((state) => state.projectId)
  const [section, setSection] = useState<SectionFilter>('all')
  const [statsCost, setStatsCost] = useState<JobStatsCostDto | null>(null)
  const [costDetail, setCostDetail] = useState<JobCostDetailDto | null>(null)
  const [costDetailLoading, setCostDetailLoading] = useState(false)
  const [costDetailError, setCostDetailError] = useState('')
  const [busyBatchId, setBusyBatchId] = useState<string | null>(null)
  const panelRef = useRef<HTMLElement | null>(null)
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([])

  // 统计条上的项目累计成本来自 /api/jobs/stats 的 cost 字段；取不到时明确显示「—」。
  useEffect(() => {
    if (!open) return
    let cancelled = false
    void jobApi
      .stats(currentProjectId || undefined)
      .then((response) => {
        if (!cancelled) setStatsCost(normalizeJobStatsCost(response?.cost))
      })
      .catch(() => {
        if (!cancelled) setStatsCost(null)
      })
    return () => {
      cancelled = true
    }
  }, [open, currentProjectId, lastSyncedAt])

  // 任务详情里的逐能力成本明细：失败 / 取消的任务同样能查到已发生的调用成本。
  useEffect(() => {
    if (!open || !selectedJobId) {
      setCostDetail(null)
      setCostDetailError('')
      return
    }
    const jobId = selectedJobId
    let cancelled = false
    setCostDetailLoading(true)
    setCostDetailError('')
    void budgetApi
      .jobCost(jobId)
      .then((response) => {
        if (!cancelled) setCostDetail(normalizeJobCostDetail(response))
      })
      .catch((err) => {
        if (cancelled) return
        setCostDetail(null)
        setCostDetailError(describeBudgetError(err, '任务成本明细加载失败'))
      })
      .finally(() => {
        if (!cancelled) setCostDetailLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [open, selectedJobId])

  useEffect(() => {
    if (!open) return
    const node = panelRef.current
    if (node) node.focus()
  }, [open])

  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        onClose()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [open, onClose])

  const sections = useMemo(() => {
    const buckets: Record<JobSectionTone, JobDto[]> = {
      active: [],
      failed: [],
      interrupted: [],
      cancelled: [],
      completed: [],
    }
    for (const job of visibleJobs) {
      buckets[jobSection(job.status)].push(job)
    }
    return buckets
  }, [visibleJobs])

  // 失败分组标题下的共性原因提示：只统计当前筛选结果里的失败任务。
  const failedCategoryCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const job of sections.failed) {
      const code = effectiveErrorCode(job) || 'job_failed'
      counts.set(code, (counts.get(code) || 0) + 1)
    }
    return Array.from(counts.entries())
      .map(([code, count]) => ({ code, label: ERROR_CODE_LABELS[code] || code, count }))
      .sort((a, b) => b.count - a.count || a.code.localeCompare(b.code))
  }, [sections])

  const filtersActive = Boolean(
    filters.projectId ||
      filters.statuses.length ||
      filters.jobTypes.length ||
      filters.errorCodes.length ||
      filters.search.trim() ||
      filters.onlyActive,
  )

  const projectOptions = useMemo(() => {
    const ids = new Set<string>()
    for (const job of jobs) {
      if (job.project_id) ids.add(job.project_id)
    }
    if (currentProjectId) ids.add(currentProjectId)
    return [
      { value: '', label: '全部项目' },
      ...Array.from(ids)
        .sort()
        .map((id) => ({ value: id, label: id === currentProjectId ? id + '（当前项目）' : id })),
    ]
  }, [jobs, currentProjectId])

  const handleTabKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLButtonElement>, index: number) => {
      const nextIndex = resolveWorkspaceTabIndex(event.key, index, SECTION_TABS.length)
      if (nextIndex === null) return
      event.preventDefault()
      setSection(SECTION_TABS[nextIndex].id)
      const node = tabRefs.current[nextIndex]
      if (node) node.focus()
    },
    [],
  )

  const handleQueueAction = async (batchId: string, action: 'pause' | 'resume' | 'cancel' | 'retry') => {
    setBusyBatchId(batchId)
    try {
      await regenerationQueueApi[action](batchId)
      await sync({ silent: true })
    } catch (error: any) {
      const detail = error?.response?.data?.detail || error?.message || '队列操作失败'
      useTaskStore.getState().note(String(detail))
    } finally {
      setBusyBatchId(null)
    }
  }

  const statusOptions = useMemo(
    () => JOB_STATUSES.map((status) => ({ value: status, label: STATUS_LABELS[status] || status })),
    [],
  )
  const typeOptions = useMemo(
    () => JOB_TYPES.map((type: JobType) => ({ value: type, label: JOB_TYPE_LABELS[type] || type })),
    [],
  )

  if (!open) return null

  const renderJob = (job: JobDto) => {
    const actions = jobActionState(job)
    const busy = busyJobId === job.id
    const expanded = selectedJobId === job.id
    return (
      <li key={job.id} className="task-item" data-status={job.status}>
        <div className="task-item-head">
          <span className="task-item-name" title={job.display_name}>
            {job.display_name}
          </span>
          <span className={statusClass(job.status)}>{job.status_label}</span>
          {job.attempt > 1 && <span className="task-attempt">第 {job.attempt} 次尝试</span>}
        </div>

        <div className="task-item-meta">
          <span className="task-meta-chip">{job.job_type_label}</span>
          {job.stage && <span className="task-meta-chip">阶段 {job.stage}</span>}
          {job.batch_id && <span className="task-meta-chip">队列 {job.queue_position || 0}</span>}
          {typeof job.priority === 'number' && job.batch_id && <span className="task-meta-chip">优先级 {job.priority}</span>}
          {job.project_id && <span className="task-meta-chip">项目 {job.project_id}</span>}
          <span className={job.cost.cost_known ? 'task-cost' : 'task-cost task-cost-unknown'}>
            成本 {jobCostComparisonText(job.cost)}
          </span>
          <span>耗时 {jobDurationComparisonText(job.duration_seconds, job.cost.estimated_seconds)}</span>
          <span>更新 {formatRelativeTime(job.updated_at)}</span>
          {job.eta_seconds !== null && job.is_active && <span>预计剩余 {formatDuration(job.eta_seconds)}</span>}
        </div>

        <div className="task-item-progress" role="group" aria-label={job.display_name + ' 的进度'}>
          <Progress
            percent={job.progress}
            size="small"
            status={job.status === 'failed' ? 'exception' : job.status === 'completed' ? 'success' : 'active'}
            aria-label={job.display_name + ' 进度 ' + progressText(job)}
          />
          <span className="task-step">{stepText(job)}</span>
        </div>

        {job.error_message && (
          <p className="task-error" role="note">
            <ExclamationCircleOutlined aria-hidden="true" />
            {errorCategoryLabel(job) && (
              <span className={'task-error-code task-error-code-' + errorCodeTone(effectiveErrorCode(job))}>
                {errorCategoryLabel(job)}
              </span>
            )}
            <span className="task-error-text">{errorHeadline(job)}</span>
          </p>
        )}
        {!job.error_message && job.message && <p className="task-message">{job.message}</p>}
        {job.blocked_reason && <p className="task-message task-message-blocked">阻塞：{job.blocked_reason}</p>}

        <div className="task-actions">
          <Tooltip title={actions.cancel.reason}>
            <Button
              size="small"
              icon={<StopOutlined />}
              disabled={!actions.cancel.enabled || busy}
              loading={busy}
              onClick={() => void requestCancel(job.id)}
              aria-label={'取消任务 ' + job.display_name}
            >
              取消
            </Button>
          </Tooltip>
          <Tooltip title={actions.retry.reason}>
            <Button
              size="small"
              icon={<RedoOutlined />}
              disabled={!actions.retry.enabled || busy}
              onClick={() => void requestRetry(job.id)}
              aria-label={'重试任务 ' + job.display_name}
            >
              重试
            </Button>
          </Tooltip>
          <Tooltip title={actions.resume.reason}>
            <Button
              size="small"
              icon={<PlayCircleOutlined />}
              disabled={!actions.resume.enabled || busy}
              onClick={() => void requestResume(job.id)}
              aria-label={'续跑任务 ' + job.display_name}
            >
              续跑
            </Button>
          </Tooltip>
          <Tooltip title="查看任务详情与历次尝试">
            <Button
              size="small"
              icon={<InfoCircleOutlined />}
              aria-expanded={expanded}
              aria-controls="task-center-detail"
              onClick={() => void selectJob(expanded ? null : job.id)}
              aria-label={'查看任务详情 ' + job.display_name}
            >
              详情
            </Button>
          </Tooltip>
          {projectJumpDetail(job) && (
            <Tooltip title="跳转到对应项目">
              <Button
                size="small"
                icon={<FolderOpenOutlined />}
                onClick={() =>
                  window.dispatchEvent(new CustomEvent('workspace:open-project', { detail: projectJumpDetail(job) }))
                }
                aria-label={'跳转到项目 ' + job.project_id}
              >
                项目
              </Button>
            </Tooltip>
          )}
          {job.shot_id && (
            <Tooltip title="跳转到对应镜头">
              <Button
                size="small"
                icon={<PlayCircleOutlined />}
                onClick={() => {
                  window.dispatchEvent(new CustomEvent('workspace:open-project', { detail: { projectId: job.project_id, shotId: job.shot_id } }))
                  window.dispatchEvent(new CustomEvent('workspace:open-shot', { detail: { shotId: job.shot_id, projectId: job.project_id } }))
                }}
                aria-label={'跳转到镜头 ' + job.shot_id}
              >
                镜头
              </Button>
            </Tooltip>
          )}
          <Tooltip title={actions.remove.reason}>
            <Button
              size="small"
              danger
              icon={<DeleteOutlined />}
              disabled={!actions.remove.enabled || busy}
              onClick={() => void requestDelete(job.id)}
              aria-label={'清理任务记录 ' + job.display_name}
            >
              清理
            </Button>
          </Tooltip>
          {job.batch_id && (job.status === 'queued' || job.status === 'running') && (
            <Button size="small" loading={busyBatchId === job.batch_id} onClick={() => void handleQueueAction(job.batch_id as string, job.paused ? 'resume' : 'pause')}>
              {job.paused ? '继续队列' : '暂停队列'}
            </Button>
          )}
          {job.batch_id && (job.status === 'failed' || job.status === 'cancelled' || job.status === 'interrupted') && (
            <Button size="small" icon={<RedoOutlined />} loading={busyBatchId === job.batch_id} onClick={() => void handleQueueAction(job.batch_id as string, 'retry')}>
              重试队列
            </Button>
          )}
          {job.batch_id && job.status === 'running' && (
            <Button size="small" icon={<StopOutlined />} loading={busyBatchId === job.batch_id} onClick={() => void handleQueueAction(job.batch_id as string, 'cancel')}>
              取消队列
            </Button>
          )}
        </div>
      </li>
    )
  }

  const visibleSections = SECTION_ORDER.filter((key) => section === 'all' || section === key)
  const hasVisible = visibleJobs.length > 0

  return (
    <div className="task-center-backdrop" role="presentation">
      <aside
        className="task-center"
        role="dialog"
        aria-label="后台任务中心"
        aria-modal="true"
        ref={panelRef}
        tabIndex={-1}
      >
      <header className="task-center-head">
        <div>
          <span className="task-center-eyebrow">后台工作流</span>
          <h2 className="task-center-title">任务中心</h2>
          <p className="task-center-sub" aria-live="polite">
            {connectionLabel(connectionState)}
            {lastSyncedAt ? ' · 上次同步 ' + formatRelativeTime(new Date(lastSyncedAt).toISOString()) : ''}
          </p>
        </div>
        <div className="task-center-head-actions">
          <Tooltip title="刷新任务列表">
            <Button
              size="small"
              icon={loading ? <LoadingOutlined /> : <ReloadOutlined />}
              onClick={() => void sync()}
              disabled={loading}
              aria-label="刷新任务列表"
            >
              <span className="task-head-action-label">刷新</span>
            </Button>
          </Tooltip>
          <Tooltip title="清理已结束的历史任务">
            <Button
              size="small"
              icon={<DeleteOutlined />}
              onClick={() => void cleanupHistory()}
              disabled={busyJobId === '__cleanup__'}
              aria-label="清理历史任务"
            >
              <span className="task-head-action-label">清理历史</span>
            </Button>
          </Tooltip>
          <Tooltip title="关闭任务中心（任务会继续在后台运行）">
            <Button size="small" type="text" icon={<CloseOutlined />} onClick={onClose} aria-label="关闭任务中心" />
          </Tooltip>
        </div>
      </header>

      <div className="task-center-summary" role="status" aria-live="polite">
        <span className="task-summary-chip task-summary-chip-active">
          <strong>{summary.activeCount}</strong>
          <small>活动中</small>
        </span>
        <span className="task-summary-chip task-summary-chip-failed">
          <strong>{summary.failedCount}</strong>
          <small>失败</small>
        </span>
        <span className="task-summary-chip">
          <strong>{summary.retryableCount}</strong>
          <small>可重试</small>
        </span>
        <span className="task-summary-chip">
          <strong>{summary.total}</strong>
          <small>全部任务</small>
        </span>
        <span className="task-summary-chip task-summary-chip-cost">
          <strong>{statsCost ? formatCostValue(statsCost.cost_micro, statsCost.cost_known, statsCost.currency) : '—'}</strong>
          <small>项目累计成本</small>
        </span>
        {statsCost && statsCost.unknown_call_count > 0 && (
          <span className="task-summary-chip task-summary-chip-warn">
            {statsCost.unknown_call_count} 次调用{UNKNOWN_COST_TEXT}
          </span>
        )}
      </div>

      <div className="task-center-filters">
        <div className="task-filter-row task-filter-row-primary">
          <label className="task-filter-field task-filter-project">
            <span className="task-filter-label">项目范围</span>
            <Select
              size="small"
              value={filters.projectId}
              options={projectOptions}
              onChange={(value: string) => setFilters({ projectId: value || '' })}
              aria-label="按项目筛选任务"
            />
          </label>
          <label className="task-filter-field task-filter-search">
            <span className="task-filter-label">搜索任务</span>
            <Input
              size="small"
              value={filters.search}
              allowClear
              placeholder="名称、项目或错误信息"
              onChange={(event) => setFilters({ search: event.target.value })}
              aria-label="搜索任务"
            />
          </label>
        </div>
        <div className="task-filter-row task-filter-row-secondary">
          <label className="task-filter-field">
            <span className="task-filter-label">状态</span>
            <Select
              size="small"
              mode="multiple"
              allowClear
              value={filters.statuses}
              options={statusOptions}
              placeholder="全部状态"
              onChange={(value: JobStatus[]) => setFilters({ statuses: value || [] })}
              aria-label="按状态筛选任务"
            />
          </label>
          <label className="task-filter-field">
            <span className="task-filter-label">任务类型</span>
            <Select
              size="small"
              mode="multiple"
              allowClear
              value={filters.jobTypes}
              options={typeOptions}
              placeholder="全部类型"
              onChange={(value: JobType[]) => setFilters({ jobTypes: value || [] })}
              aria-label="按类型筛选任务"
            />
          </label>
          <label className="task-filter-field">
            <span className="task-filter-label">失败原因</span>
            <Select
              size="small"
              mode="multiple"
              allowClear
              value={filters.errorCodes}
              options={FAILURE_CATEGORY_OPTIONS}
              placeholder="全部原因"
              onChange={(value: string[]) => setFilters({ errorCodes: value || [] })}
              aria-label="按失败原因筛选任务"
            />
          </label>
          <label className="task-filter-field">
            <span className="task-filter-label">排序方式</span>
            <Select
              size="small"
              value={sortKey}
              options={(Object.keys(SORT_LABELS) as JobSortKey[]).map((key) => ({ value: key, label: SORT_LABELS[key] }))}
              onChange={(value: JobSortKey) => setSortKey(value)}
              aria-label="任务排序方式"
            />
          </label>
          <div className="task-filter-actions">
            <Button size="small" type={filters.onlyActive ? 'primary' : 'default'} onClick={() => setFilters({ onlyActive: !filters.onlyActive })} aria-pressed={filters.onlyActive}>
              仅看进行中
            </Button>
            <Button size="small" onClick={resetFilters} disabled={!filtersActive} aria-label="重置筛选条件">
              重置
            </Button>
          </div>
        </div>
      </div>

      <div className="task-center-tabs" role="tablist" aria-label="任务分组">
        {SECTION_TABS.map((tab, index) => (
          <button
            key={tab.id}
            type="button"
            ref={(node) => {
              tabRefs.current[index] = node
            }}
            {...sectionTabAriaProps(tab.id, section)}
            className={section === tab.id ? 'task-tab active' : 'task-tab'}
            onClick={() => setSection(tab.id)}
            onKeyDown={(event) => handleTabKeyDown(event, index)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {notice && (
        <div className="task-center-notice" role="status">
          {notice}
          <button type="button" onClick={dismissNotice} aria-label="关闭提示">
            ×
          </button>
        </div>
      )}
      {error && (
        <div className="task-center-error" role="alert">
          {error}
        </div>
      )}

      <div
        className="task-center-body"
        id={SECTION_PANEL_ID}
        role="tabpanel"
        aria-labelledby={SECTION_TAB_ID_PREFIX + section}
        tabIndex={0}
      >
        <div className="task-center-list-column">
          {loading && !jobs.length && (
            <div className="task-center-status" role="status">
              <Spin size="small" /> 正在加载任务…
            </div>
          )}

          {!loading && !hasVisible && (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={emptyStateText(jobs.length > 0, filtersActive)}
            />
          )}

          {visibleSections.map((key) => {
            const items = sections[key]
            if (items.length === 0) return null
            const categoryChips =
              key === 'failed' && failedCategoryCounts.length > 0 && failedCategoryCounts.length <= 4
                ? failedCategoryCounts
                : []
            return (
              <section key={key} className="task-section" aria-label={SECTION_TITLES[key]}>
                <h3 className="task-section-title">
                  {SECTION_TITLES[key]}
                  <span className="task-section-count">{items.length}</span>
                  {categoryChips.map((item) => (
                    <button
                      key={item.code}
                      type="button"
                      className={
                        'task-cat-chip task-cat-chip-' +
                        errorCodeTone(item.code) +
                        (filters.errorCodes.indexOf(item.code) >= 0 ? ' task-cat-chip-active' : '')
                      }
                      title={'按「' + item.label + '」筛选失败任务'}
                      onClick={() =>
                        setFilters({
                          errorCodes:
                            filters.errorCodes.indexOf(item.code) >= 0
                              ? filters.errorCodes.filter((code) => code !== item.code)
                              : [...filters.errorCodes, item.code],
                        })
                      }
                      aria-pressed={filters.errorCodes.indexOf(item.code) >= 0}
                    >
                      {item.label} ×{item.count}
                    </button>
                  ))}
                </h3>
                <ul className="task-list" role="list">
                  {items.map(renderJob)}
                </ul>
              </section>
            )
          })}
        </div>

        <div className="task-center-detail-column">
          {selectedJobId ? (
          <section className="task-detail" id="task-center-detail" aria-label="任务详情" tabIndex={0}>
            <h3 className="task-section-title">任务详情</h3>
            {detailLoading && <div className="task-center-status"><Spin size="small" /> 正在加载详情…</div>}
            {!detailLoading && selectedJob && (
              <>
                <dl className="task-detail-grid">
                  <dt>任务名称</dt>
                  <dd>{selectedJob.display_name}</dd>
                  <dt>任务类型</dt>
                  <dd>{selectedJob.job_type_label}</dd>
                  <dt>所属项目</dt>
                  <dd>{selectedJob.project_id || '—'}</dd>
                  <dt>当前状态</dt>
                  <dd>{selectedJob.status_label}</dd>
                  <dt>当前步骤</dt>
                  <dd>{stepText(selectedJob)}</dd>
                  <dt>进度</dt>
                  <dd>{progressText(selectedJob)}</dd>
                  <dt>已运行时长</dt>
                  <dd>{formatDuration(selectedJob.duration_seconds)}</dd>
                  <dt>创建时间</dt>
                  <dd>{selectedJob.created_at || '—'}</dd>
                  <dt>开始时间</dt>
                  <dd>{selectedJob.started_at || '—'}</dd>
                  <dt>更新时间</dt>
                  <dd>{selectedJob.updated_at || '—'}</dd>
                  <dt>完成时间</dt>
                  <dd>{selectedJob.finished_at || '—'}</dd>
                  <dt>尝试次数</dt>
                  <dd>
                    第 {selectedJob.attempt} 次
                    {selectedJob.retry_relationship?.retry_of_attempt
                      ? '（重试自第 ' + selectedJob.retry_relationship.retry_of_attempt + ' 次）'
                      : ''}
                  </dd>
                </dl>
                {(selectedJob.error_message || selectedJob.error_detail) && (
                  <div className="task-error-block" role="note">
                    <div className="task-error-block-head">
                      <span className={'task-error-code task-error-code-' + errorCodeTone(effectiveErrorCode(selectedJob))}>
                        {errorCategoryLabel(selectedJob) || '任务失败'}
                      </span>
                      <span className="task-error-code-id">{effectiveErrorCode(selectedJob) || 'job_failed'}</span>
                    </div>
                    <p className="task-error-text">{errorHeadline(selectedJob)}</p>
                    {selectedJob.error_detail?.suggestion && (
                      <p className="task-error-advice">
                        <BulbOutlined aria-hidden="true" /> 修复建议：{selectedJob.error_detail.suggestion}
                      </p>
                    )}
                    {selectedJob.error_detail?.summary &&
                      selectedJob.error_message &&
                      selectedJob.error_detail.summary !== selectedJob.error_message && (
                        <details className="task-error-raw">
                          <summary>原始错误信息</summary>
                          <pre>{selectedJob.error_message}</pre>
                        </details>
                      )}
                    {selectedJob.error_detail?.source && (
                      <p className="task-error-source">
                        {selectedJob.error_detail.source === 'llm'
                          ? '原因由 AI 自动识别' +
                            (selectedJob.error_detail.model ? '（' + selectedJob.error_detail.model + '）' : '')
                          : '原因由规则自动识别'}
                      </p>
                    )}
                  </div>
                )}
                <h4 className="task-detail-subtitle">历次尝试</h4>
                <ol className="task-attempt-list">
                  {(selectedJob.attempts || []).map((attempt) => (
                    <li key={attempt.id}>
                      <span>第 {attempt.attempt} 次</span>
                      <span>{STATUS_LABELS[attempt.status] || attempt.status}</span>
                      <span>{formatDuration(attempt.duration_seconds)}</span>
                      <span>{formatRelativeTime(attempt.updated_at)}</span>
                      {attempt.error_message && (
                        <em>
                          {attempt.error_code_label ? attempt.error_code_label + '：' : ''}
                          {attempt.error_message}
                        </em>
                      )}
                    </li>
                  ))}
                </ol>
                <h4 className="task-detail-subtitle">成本明细</h4>
                {costDetailLoading && (
                  <div className="task-center-status">
                    <Spin size="small" /> 正在加载成本明细…
                  </div>
                )}
                {costDetailError && (
                  <p className="task-error" role="note">
                    {costDetailError}
                  </p>
                )}
                {!costDetailLoading && !costDetailError && (
                  <>
                    <dl className="task-detail-grid">
                      <dt>实际成本</dt>
                      <dd>
                        {costDetail
                          ? formatCostValue(
                              costDetail.summary.cost_micro,
                              costDetail.summary.cost_known,
                              costDetail.summary.currency,
                            )
                          : jobCostText(selectedJob.cost)}
                        {(costDetail ? costDetail.summary.call_count : selectedJob.cost.call_count) === 0 && (
                          <em className="task-cost-hint">（{NO_USAGE_TEXT}）</em>
                        )}
                      </dd>
                      <dt>启动前估算</dt>
                      <dd>
                        {costDetail?.estimate
                          ? estimateCostText(costDetail.estimate) +
                            ' · 预计 ' +
                            formatDurationText(costDetail.estimate.estimated_seconds) +
                            '（' + durationSourceLabel(costDetail.estimate.duration_source) + '）'
                          : '没有估算记录' }
                      </dd>
                      <dt>调用次数</dt>
                      <dd>
                        {costDetail ? costDetail.summary.call_count : selectedJob.cost.call_count} 次
                        {costDetail && costDetail.summary.unknown_call_count > 0
                          ? ' · ' + costDetail.summary.unknown_call_count + ' 次成本未知'
                          : ''}
                        {costDetail && costDetail.summary.failed_call_count > 0
                          ? ' · 失败 ' + costDetail.summary.failed_call_count + ' 次'
                          : ''}
                      </dd>
                      <dt>供应商调用耗时</dt>
                      <dd>
                        {formatMillisecondsText(
                          costDetail ? costDetail.summary.duration_ms : selectedJob.cost.provider_seconds * 1000,
                        )}
                      </dd>
                    </dl>

                    <h4 className="task-detail-subtitle">逐能力成本</h4>
                    {!costDetail || costDetail.summary.by_capability.length === 0 ? (
                      <p className="task-message">这次任务没有落库的调用记录（{NO_USAGE_TEXT}）。</p>
                    ) : (
                      <table className="task-cost-table">
                        <thead>
                          <tr>
                            <th>能力</th>
                            <th>调用</th>
                            <th>数量</th>
                            <th>成本</th>
                            <th>耗时</th>
                          </tr>
                        </thead>
                        <tbody>
                          {costDetail.summary.by_capability.map((entry) => (
                            <tr key={entry.capability}>
                              <td>{entry.label}</td>
                              <td>
                                {entry.call_count} 次
                                {entry.unknown_call_count > 0 ? '（' + entry.unknown_call_count + ' 次未知）' : ''}
                              </td>
                              <td>
                                {entry.quantity} {entry.base_unit || '单位'}
                                {entry.secondary_quantity > 0 ? ' + ' + entry.secondary_quantity + ' 输出 token' : ''}
                              </td>
                              <td>{formatCostValue(entry.cost_micro, entry.cost_known, entry.currency)}</td>
                              <td>{formatDurationText(entry.seconds)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}

                    {costDetail?.estimate && costDetail.estimate.components.length > 0 && (
                      <>
                        <h4 className="task-detail-subtitle">估算拆解（启动前）</h4>
                        <table className="task-cost-table">
                          <thead>
                            <tr>
                              <th>环节</th>
                              <th>模型</th>
                              <th>数量</th>
                              <th>预计成本</th>
                              <th>预计耗时</th>
                            </tr>
                          </thead>
                          <tbody>
                            {costDetail.estimate.components.map((component, index) => (
                              <tr key={component.capability + '-' + index}>
                                <td>{component.component_label || component.label}</td>
                                <td>{[component.provider, component.model].filter(Boolean).join(' / ') || '默认'}</td>
                                <td>
                                  {component.quantity} {component.base_unit || '单位'}
                                </td>
                                <td>
                                  {component.cost_known && component.cost_micro !== null
                                    ? formatCostValue(component.cost_micro, true, costDetail.estimate?.currency)
                                    : UNKNOWN_COST_TEXT}
                                </td>
                                <td>{formatDurationText(component.estimated_seconds)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </>
                    )}

                    {costDetail?.estimate && costDetail.estimate.unknown_components.length > 0 && (
                      <p className="task-message" role="note">
                        未知成本原因：{unknownCostReason(costDetail.estimate.unknown_components)}
                      </p>
                    )}

                    {costDetail && costDetail.records.items.length > 0 && (
                      <>
                        <h4 className="task-detail-subtitle">调用记录（含失败调用）</h4>
                        <table className="task-cost-table">
                          <thead>
                            <tr>
                              <th>时间</th>
                              <th>能力</th>
                              <th>状态</th>
                              <th>数量</th>
                              <th>成本</th>
                              <th>耗时</th>
                            </tr>
                          </thead>
                          <tbody>
                            {costDetail.records.items.slice(0, 20).map((record) => (
                              <tr key={record.id}>
                                <td>{formatRelativeTime(record.created_at)}</td>
                                <td>{record.capability_label || record.capability}</td>
                                <td>
                                  {record.status === 'succeeded' ? '成功' : record.status === 'failed' ? '失败' : '已取消'}
                                </td>
                                <td>
                                  {record.quantity} {record.base_unit || '单位'}
                                </td>
                                <td>
                                  {formatCostValue(record.cost_micro, record.cost_known, record.currency)}
                                  {record.cost_source === 'local' ? '（本地）' : ''}
                                </td>
                                <td>{formatMillisecondsText(record.duration_ms)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </>
                    )}
                  </>
                )}
              </>
            )}
          </section>
          ) : (
            <div className="task-detail-empty">
              <InfoCircleOutlined aria-hidden="true" />
              <strong>选择一个任务</strong>
              <span>查看进度、尝试记录与成本明细</span>
            </div>
          )}
        </div>
      </div>
      </aside>
    </div>
  )
}

export default TaskCenter
