import type React from 'react'
import { useMemo, useState } from 'react'
import { CaretUpOutlined, UnorderedListOutlined } from '@ant-design/icons'
import { useShotStore } from '../stores/shotStore'
import { useTaskStore } from '../stores/taskStore'
import { useProjectStore } from '../stores/projectStore'
import { OPEN_TASK_CENTER_EVENT } from './TaskCenter'
import ProgressDebugPanel from './ProgressDebugPanel'
import { progressStepLabel } from './progressDebugModel'

const BottomBar: React.FC = () => {
  const { isGenerating, progress, shots, currentStep } = useShotStore()
  const taskSummary = useTaskStore((state) => state.summary)
  const jobs = useTaskStore((state) => state.jobs)
  const connectionState = useTaskStore((state) => state.connectionState)
  const currentProjectId = useProjectStore((state) => state.projectId)
  const [debugOpen, setDebugOpen] = useState(false)

  const activeJob = useMemo(() => {
    const scoped = jobs.filter((job) => !currentProjectId || job.project_id === currentProjectId)
    return (
      scoped
        .filter((job) => job.is_active)
        .sort(
          (a, b) => Date.parse(b.updated_at || b.created_at || '') - Date.parse(a.updated_at || a.created_at || ''),
        )[0] ||
      scoped.sort(
        (a, b) => Date.parse(b.updated_at || b.created_at || '') - Date.parse(a.updated_at || a.created_at || ''),
      )[0] ||
      null
    )
  }, [currentProjectId, jobs])

  const openTaskCenter = () => window.dispatchEvent(new CustomEvent(OPEN_TASK_CENTER_EVENT))
  const totalDuration = shots.reduce((sum, item) => sum + item.duration, 0)
  const effectiveProgress = activeJob?.progress ?? progress
  const remaining = isGenerating ? Math.max(0, Math.round((100 - effectiveProgress) * 1.2)) : 0
  const minutes = Math.floor(remaining / 60)
  const seconds = remaining % 60
  const currentStepLabel = progressStepLabel(activeJob?.current_step || currentStep)

  return (
    <>
      <footer className="bottom-bar" aria-label="底部状态栏">
        <span>任务进度</span>

        <button
          type="button"
          className="bottom-progress-control"
          onClick={() => setDebugOpen((value) => !value)}
          aria-expanded={debugOpen}
          aria-controls="progress-debug-panel"
          aria-label={debugOpen ? '收起任务调试日志' : '打开任务调试日志'}
        >
          <span className="bottom-progress">
            <span
              className="bottom-progress-fill"
              style={{
                width: `${isGenerating || activeJob?.is_active ? effectiveProgress : shots.length > 0 ? 100 : 0}%`,
              }}
            />
          </span>
          <CaretUpOutlined className={debugOpen ? 'is-open' : ''} aria-hidden="true" />
        </button>

        <span className="bottom-step">当前步骤：{currentStepLabel}</span>

        <span>
          {isGenerating || activeJob?.is_active
            ? `预计剩余 ${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
            : shots.length > 0
              ? `总时长 ${totalDuration.toFixed(1)} 秒`
              : '等待输入剧本'}
        </span>

        <button
          type="button"
          className="bottom-task-trigger"
          onClick={openTaskCenter}
          aria-haspopup="dialog"
          aria-label="打开后台任务中心"
        >
          <UnorderedListOutlined aria-hidden="true" />
          <span>
            后台任务 {taskSummary.activeCount} 进行中
            {taskSummary.failedCount > 0 ? ' · ' + taskSummary.failedCount + ' 失败' : ''}
          </span>
          <em className={'bottom-task-link bottom-task-link-' + connectionState} aria-hidden="true">
            {connectionState === 'open' ? '实时' : '兜底轮询'}
          </em>
        </button>

        <span>系统占用 {isGenerating ? '41%' : '0%'}</span>
      </footer>

      <ProgressDebugPanel
        open={debugOpen}
        onClose={() => setDebugOpen(false)}
        job={activeJob}
        fallbackStep={currentStep}
        fallbackProgress={progress}
      />
    </>
  )
}

export default BottomBar
