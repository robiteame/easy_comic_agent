import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { FileSearchOutlined } from '@ant-design/icons'

import { agentGraphApi } from '../services/api'
import { useProjectStore } from '../stores/projectStore'
import { useShotStore } from '../stores/shotStore'
import AgentTracePanel from './AgentTracePanel'
import {
  AGENT_STAGE_ORDER,
  resolveStageStates,
  runStatusLabel,
  stageLabel,
  traceHasContent,
  type AgentTraceSummary,
} from './agentTraceModel'

interface FlowGraphProps {
  compact?: boolean
}

// 旧版五节点兼容步骤：没有 Agent 追踪数据（旧项目/手动模式）时的回退展示。
const STEPS = [
  { id: 'generate_script', label: '剧本' },
  { id: 'parse_script', label: '解析' },
  { id: 'generate_storyboard', label: '分镜' },
  { id: 'wait_asset_confirm', label: '资产' },
  { id: 'generate_storyboard_images', label: '定稿图' },
  { id: 'wait_storyboard_approval', label: '镜头审核' },
  { id: 'generate_voice', label: '配音' },
  { id: 'generate_seedance_video', label: '单镜头视频' },
  { id: 'compose_video', label: '合成' },
  { id: 'quality_check', label: '校验' },
]

function stepState(stepId: string, currentStep: string, isGenerating: boolean, hasVideo: boolean) {
  if (hasVideo) return 'done'
  const currentIndex = STEPS.findIndex((step) => step.id === currentStep)
  const stepIndex = STEPS.findIndex((step) => step.id === stepId)
  if (stepIndex < 0 || currentIndex < 0) return 'pending'
  if (stepIndex < currentIndex) return 'done'
  if (stepIndex === currentIndex) return isGenerating ? 'running' : 'done'
  return 'pending'
}

const FlowGraph: React.FC<FlowGraphProps> = ({ compact }) => {
  const { currentStep, isGenerating, videoPath } = useShotStore()
  const projectId = useProjectStore((state) => state.projectId)
  const [summary, setSummary] = useState<AgentTraceSummary | null>(null)
  const [showTrace, setShowTrace] = useState(false)

  const refresh = useCallback(async () => {
    if (!projectId) {
      setSummary(null)
      return
    }
    try {
      const response = await agentGraphApi.trace(projectId)
      setSummary(response?.summary || null)
    } catch {
      // 追踪拉取失败时保留上一次数据，阶段条回退到旧步骤展示。
    }
  }, [projectId])

  useEffect(() => {
    void refresh()
    if (!projectId) return
    // 阶段条只在运行中需要轮询；终态（completed/degraded/failed）停止刷新。
    const terminal = summary && ['completed', 'degraded', 'failed', 'cancelled'].includes(summary.run?.status)
    if (terminal) return
    const timer = window.setInterval(() => {
      void refresh()
    }, compact ? 5000 : 2500)
    return () => window.clearInterval(timer)
  }, [refresh, projectId, compact, summary?.run?.status])

  const stageStates = useMemo(() => resolveStageStates(summary), [summary])
  const useAgentStages = traceHasContent(summary)

  return (
    <div className="flow-steps-wrap">
      <div className="flow-steps" aria-label="Agent 执行流程">
        {useAgentStages
          ? AGENT_STAGE_ORDER.map((stage) => (
              <div
                key={stage}
                className={`flow-step flow-step-agent ${stageStates[stage] || 'pending'}`}
                title={
                  stage === summary?.run?.current_stage
                    ? `当前阶段 · 运行状态：${runStatusLabel(summary?.run?.status)}`
                    : stageLabel(stage)
                }
              >
                <span className="flow-step-dot" />
                <span>{stageLabel(stage)}</span>
                {stage === summary?.run?.current_stage && summary?.run?.status_reason ? (
                  <span className="flow-step-note">{summary.run.status_reason}</span>
                ) : null}
              </div>
            ))
          : STEPS.map((step, index) => {
              const state = stepState(step.id, currentStep, isGenerating, Boolean(videoPath))
              return (
                <React.Fragment key={step.id}>
                  <div className={`flow-step ${state}`}>
                    <span className="flow-step-dot" />
                    <span>{step.label}</span>
                  </div>
                  {index < STEPS.length - 1 && <span className={`flow-step-line ${state}`} />}
                </React.Fragment>
              )
            })}
      </div>
      {useAgentStages && (
        <button
          type="button"
          className="flow-trace-toggle"
          aria-expanded={showTrace}
          onClick={() => setShowTrace((value) => !value)}
        >
          <FileSearchOutlined aria-hidden="true" />
          {showTrace ? '收起追踪' : '展开追踪（阶段分 · 决策 · 候选 · 成本）'}
        </button>
      )}
      {useAgentStages && showTrace && <AgentTracePanel compact={compact} />}
    </div>
  )
}

export default FlowGraph
