import React from 'react'
import { UnorderedListOutlined } from '@ant-design/icons'
import { useShotStore } from '../stores/shotStore'
import { useTaskStore } from '../stores/taskStore'
import { OPEN_TASK_CENTER_EVENT } from './TaskCenter'

const stepLabels: Record<string, string> = {
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
}

const BottomBar: React.FC = () => {
  const { isGenerating, progress, shots, currentStep } = useShotStore()
  const taskSummary = useTaskStore((state) => state.summary)
  const connectionState = useTaskStore((state) => state.connectionState)
  const openTaskCenter = () => window.dispatchEvent(new CustomEvent(OPEN_TASK_CENTER_EVENT))

  const totalDuration = shots.reduce((sum, item) => sum + item.duration, 0)
  const remaining = isGenerating ? Math.max(0, Math.round((100 - progress) * 1.2)) : 0
  const minutes = Math.floor(remaining / 60)
  const seconds = remaining % 60
  const currentStepLabel = currentStep ? (stepLabels[currentStep] || '处理中') : '待命'

  return (
    <footer className="bottom-bar" aria-label="底部状态栏">
      <span>任务进度</span>

      <div className="bottom-progress">
        <div
          className="bottom-progress-fill"
          style={{ width: `${isGenerating ? progress : shots.length > 0 ? 100 : 0}%` }}
        />
      </div>

      <span className="bottom-step">当前步骤：{currentStepLabel}</span>

      <span>
        {isGenerating
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
  )
}

export default BottomBar
