import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ApiOutlined,
  BugOutlined,
  CloseOutlined,
  CodeOutlined,
  LinkOutlined,
  MessageOutlined,
  ReloadOutlined,
  RightOutlined,
} from '@ant-design/icons'

import type { JobDebugEvent, JobDto } from '../services/jobTypes'
import { jobApi } from '../services/api'
import { useTaskStore } from '../stores/taskStore'
import {
  DEBUG_FILTERS,
  apiResultFor,
  debugFilterMatches,
  debugLevelLabel,
  formatDebugPayload,
  formatDebugTimestamp,
  isApiDebugEvent,
  latestApiRequest,
  mergeDebugEvents,
  progressStepLabel,
  promptSections,
  type DebugFilter,
} from './progressDebugModel'

interface ProgressDebugPanelProps {
  open: boolean
  onClose: () => void
  job: JobDto | null
  fallbackStep: string
  fallbackProgress: number
}

const ProgressDebugPanel: React.FC<ProgressDebugPanelProps> = ({
  open,
  onClose,
  job,
  fallbackStep,
  fallbackProgress,
}) => {
  const liveEvents = useTaskStore((state) => (job ? state.debugEventsByJob[job.id] : undefined))
  const setDebugEvents = useTaskStore((state) => state.setDebugEvents)
  const connectionState = useTaskStore((state) => state.connectionState)

  const [restEvents, setRestEvents] = useState<JobDebugEvent[]>([])
  const [filter, setFilter] = useState<DebugFilter>('all')
  const [selectedId, setSelectedId] = useState('')
  const [followLatest, setFollowLatest] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const listRef = useRef<HTMLDivElement | null>(null)

  const events = useMemo(() => mergeDebugEvents(restEvents, liveEvents), [restEvents, liveEvents])
  const visibleEvents = useMemo(
    () => events.filter((event) => debugFilterMatches(event, filter)),
    [events, filter],
  )
  const fallbackSelected = useMemo(() => latestApiRequest(events) || events[events.length - 1] || null, [events])
  const selected = useMemo(
    () => events.find((event) => event.id === selectedId) || fallbackSelected,
    [events, fallbackSelected, selectedId],
  )
  const result = useMemo(() => apiResultFor(events, selected?.kind === 'api_request' ? selected : null), [events, selected])

  const refresh = useCallback(async () => {
    if (!job?.id) {
      setRestEvents([])
      setError('')
      setLoading(false)
      return
    }
    setLoading(true)
    try {
      const response = await jobApi.debug(job.id)
      const next = mergeDebugEvents(response.events, useTaskStore.getState().debugEventsByJob[job.id])
      setRestEvents(next)
      setDebugEvents(job.id, next)
      setError('')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '调试日志加载失败')
    } finally {
      setLoading(false)
    }
  }, [job?.id, setDebugEvents])

  useEffect(() => {
    if (!open) return
    void refresh()
    const timer = window.setInterval(() => {
      void refresh()
    }, 1800)
    return () => window.clearInterval(timer)
  }, [open, refresh])

  useEffect(() => {
    if (!followLatest || !fallbackSelected) return
    setSelectedId(fallbackSelected.id)
  }, [fallbackSelected, followLatest])

  useEffect(() => {
    if (!followLatest || !listRef.current) return
    listRef.current.scrollTop = listRef.current.scrollHeight
  }, [followLatest, visibleEvents.length])

  const selectEvent = (event: JobDebugEvent) => {
    setSelectedId(event.id)
    setFollowLatest(false)
  }

  const currentStep = job?.current_step || fallbackStep
  const progress = job ? job.progress : fallbackProgress
  const promptPrompt = selected?.kind === 'api_request' ? promptSections(selected.prompt) : []

  return (
    <section
      id="progress-debug-panel"
      className={`progress-debug-panel${open ? ' is-open' : ''}`}
      role="dialog"
      aria-modal="false"
      aria-label="任务调试日志"
      aria-hidden={!open}
    >
      <header className="progress-debug-head">
        <div className="progress-debug-title">
          <BugOutlined aria-hidden="true" />
          <div>
            <strong>任务调试日志</strong>
            <span>进度、当前步骤、API 参数与提示词</span>
          </div>
        </div>
        <div className="progress-debug-head-actions">
          <span className={`debug-live debug-live-${connectionState}`}>
            <i aria-hidden="true" />
            {connectionState === 'open' ? '实时连接' : '自动补齐'}
          </span>
          <button type="button" onClick={() => void refresh()} aria-label="刷新调试日志">
            <ReloadOutlined spin={loading} aria-hidden="true" />
          </button>
          <button type="button" onClick={onClose} aria-label="关闭调试日志">
            <CloseOutlined aria-hidden="true" />
          </button>
        </div>
      </header>

      <div className="progress-debug-status">
        <div className="debug-status-copy">
          <span>当前步骤</span>
          <strong>{progressStepLabel(currentStep)}</strong>
          <em>{job?.message || (job ? '任务正在执行' : '当前暂无后台任务')}</em>
        </div>
        <div className="debug-progress-value">
          <strong>{Math.max(0, Math.min(100, Math.round(progress)))}%</strong>
          <div className="debug-progress-track" aria-hidden="true">
            <span style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} />
          </div>
        </div>
      </div>

      <div className="progress-debug-filters" role="tablist" aria-label="调试日志筛选">
        {DEBUG_FILTERS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={filter === item.id}
            className={filter === item.id ? 'is-active' : ''}
            onClick={() => setFilter(item.id)}
          >
            {item.label}
          </button>
        ))}
        <button
          type="button"
          className={`debug-follow${followLatest ? ' is-active' : ''}`}
          onClick={() => setFollowLatest((value) => !value)}
          aria-pressed={followLatest}
        >
          <LinkOutlined aria-hidden="true" />
          {followLatest ? '跟随最新' : '已暂停跟随'}
        </button>
      </div>

      {error && <div className="progress-debug-error" role="status">{error}</div>}

      <div className="progress-debug-body">
        <div className="progress-debug-log" ref={listRef}>
          {visibleEvents.length === 0 ? (
            <div className="progress-debug-empty">
              <CodeOutlined aria-hidden="true" />
              <strong>暂无调试记录</strong>
              <span>任务运行后，这里会实时显示步骤进度与 API 调用。</span>
            </div>
          ) : (
            visibleEvents.map((event) => (
              <button
                key={event.id}
                type="button"
                className={`debug-log-row debug-log-${event.level}${selected?.id === event.id ? ' is-selected' : ''}`}
                onClick={() => selectEvent(event)}
              >
                <span className="debug-log-time">{formatDebugTimestamp(event.timestamp)}</span>
                <span className="debug-log-icon">
                  {isApiDebugEvent(event) ? <ApiOutlined aria-hidden="true" /> : <MessageOutlined aria-hidden="true" />}
                </span>
                <span className="debug-log-copy">
                  <strong>{event.api || progressStepLabel(event.step)}</strong>
                  <span>{event.message}</span>
                </span>
                <span className="debug-log-level">{debugLevelLabel(event.level)}</span>
                <RightOutlined aria-hidden="true" />
              </button>
            ))
          )}
        </div>

        <aside className="progress-debug-detail" aria-label="调试日志详情">
          {!selected ? (
            <div className="progress-debug-empty">
              <CodeOutlined aria-hidden="true" />
              <strong>选择一条日志</strong>
              <span>查看该步骤的请求参数、提示词和返回摘要。</span>
            </div>
          ) : (
            <>
              <div className="debug-detail-head">
                <div>
                  <span>{selected.api || '任务步骤'}</span>
                  <strong>{selected.message}</strong>
                </div>
                <em className={`debug-detail-level debug-detail-${selected.level}`}>{debugLevelLabel(selected.level)}</em>
              </div>

              <dl className="debug-detail-meta">
                <div><dt>时间</dt><dd>{formatDebugTimestamp(selected.timestamp)}</dd></div>
                <div><dt>当前步骤</dt><dd>{progressStepLabel(selected.step)}</dd></div>
                {selected.provider && <div><dt>Provider</dt><dd>{selected.provider}</dd></div>}
                {selected.model && <div><dt>模型</dt><dd>{selected.model}</dd></div>}
              </dl>

              {selected.kind === 'api_request' && (
                <section className="debug-payload-section">
                  <h3><ApiOutlined aria-hidden="true" /> 请求参数</h3>
                  <pre>{formatDebugPayload(selected.params)}</pre>
                </section>
              )}

              {promptPrompt.map((section) => (
                <section className="debug-payload-section" key={section.label}>
                  <h3><MessageOutlined aria-hidden="true" /> {section.label}</h3>
                  <pre>{section.content}</pre>
                </section>
              ))}

              {result && (
                <section className="debug-payload-section">
                  <h3><CodeOutlined aria-hidden="true" /> 请求结果</h3>
                  <pre>{formatDebugPayload({ message: result.message, detail: result.detail })}</pre>
                </section>
              )}

              {!isApiDebugEvent(selected) && (
                <section className="debug-payload-section">
                  <h3><CodeOutlined aria-hidden="true" /> 日志详情</h3>
                  <pre>{formatDebugPayload(selected.detail || selected.message)}</pre>
                </section>
              )}
            </>
          )}
        </aside>
      </div>
    </section>
  )
}

export default ProgressDebugPanel
