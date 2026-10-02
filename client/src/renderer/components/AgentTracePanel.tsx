import React, { useEffect, useMemo, useState } from 'react'
import { ReloadOutlined } from '@ant-design/icons'
import Button from 'antd/es/button'
import Spin from 'antd/es/spin'

import { agentGraphApi } from '../services/api'
import { useProjectStore } from '../stores/projectStore'
import {
  candidateResultRows,
  decisionRows,
  durationComparison,
  promptChangeRows,
  referenceRows,
  runStatusLabel,
  shotStatusRows,
  stageLabel,
  stageQualityRows,
  STAGE_STATUS_LABELS,
  traceCostText,
  traceHasContent,
  type AgentTraceSummary,
} from './agentTraceModel'

interface AgentTracePanelProps {
  /** 指定项目时追踪该项目；缺省用当前项目。 */
  projectId?: string
  runId?: string
  /** 紧凑模式隐藏次要表格，只保留概览与决策。 */
  compact?: boolean
}

/**
 * Agent 可解释追踪面板：当前阶段、镜头状态、阶段质量分、Critic 问题、
 * 恢复候选与最终决策、Prompt 修改、候选结果、Provider/模型、实际发送参考图、
 * 成本、预计/实际耗时、自动降级原因、检查点与恢复次数。
 */
const AgentTracePanel: React.FC<AgentTracePanelProps> = ({ projectId, runId = 'auto', compact = false }) => {
  const currentProjectId = useProjectStore((state) => state.projectId)
  const targetProjectId = projectId || currentProjectId || ''
  const [summary, setSummary] = useState<AgentTraceSummary | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const load = React.useCallback(async () => {
    if (!targetProjectId) {
      setSummary(null)
      setError('')
      return
    }
    setLoading(true)
    try {
      const response = await agentGraphApi.trace(targetProjectId, runId)
      setSummary(response?.summary || null)
      setError('')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '追踪数据加载失败')
    } finally {
      setLoading(false)
    }
  }, [targetProjectId, runId])

  useEffect(() => {
    void load()
  }, [load])

  const shots = useMemo(() => shotStatusRows(summary), [summary])
  const quality = useMemo(() => stageQualityRows(summary), [summary])
  const decisions = useMemo(() => decisionRows(summary), [summary])
  const prompts = useMemo(() => promptChangeRows(summary), [summary])
  const candidates = useMemo(() => candidateResultRows(summary), [summary])
  const references = useMemo(() => referenceRows(summary), [summary])
  const durations = useMemo(() => durationComparison(summary), [summary])

  if (!targetProjectId) {
    return <p className="agent-trace-empty">选择项目后显示 Agent 追踪。</p>
  }

  if (loading && !summary) {
    return (
      <div className="agent-trace-empty" role="status">
        <Spin size="small" /> 正在加载追踪…
      </div>
    )
  }

  if (error) {
    return (
      <p className="agent-trace-empty agent-trace-error" role="note">
        {error}
      </p>
    )
  }

  if (!traceHasContent(summary)) {
    return <p className="agent-trace-empty">该项目还没有 Agent 运行记录。</p>
  }

  const run = summary?.run
  const counters = summary?.counters
  const totals = summary?.totals

  return (
    <div className="agent-trace" aria-label="Agent 可解释追踪">
      <div className="agent-trace-head">
        <span className={'agent-trace-status agent-trace-status-' + (run?.status || 'pending')}>
          {runStatusLabel(run?.status)}
        </span>
        <span className="agent-trace-stage">当前阶段：{stageLabel(run?.current_stage || '') || '—'}</span>
        <Button size="small" type="text" icon={<ReloadOutlined />} aria-label="刷新追踪" onClick={() => void load()} />
      </div>

      {run?.status_reason ? <p className="agent-trace-reason">{run.status_reason}</p> : null}

      <dl className="agent-trace-stats">
        <div>
          <dt>成本合计</dt>
          <dd>{traceCostText(totals?.cost_micro || null)}</dd>
        </div>
        <div>
          <dt>预计 / 实际耗时</dt>
          <dd>
            {durations.estimatedText} / {durations.actualText}
          </dd>
        </div>
        <div>
          <dt>检查点</dt>
          <dd>
            {counters?.checkpoint_records ?? 0}
            {counters?.invalidated_checkpoints ? `（失效 ${counters.invalidated_checkpoints}）` : ''}
          </dd>
        </div>
        <div>
          <dt>恢复次数</dt>
          <dd>{counters?.recoveries ?? 0}</dd>
        </div>
        <div>
          <dt>决策记录</dt>
          <dd>{counters?.decisions ?? 0}</dd>
        </div>
      </dl>

      {summary?.degradations?.length ? (
        <div className="agent-trace-degradations" role="note">
          <strong>自动降级 / 终止原因</strong>
          <ul>
            {summary.degradations.slice(0, 5).map((item, index) => (
              <li key={index}>
                {item.stage ? stageLabel(item.stage) + '：' : ''}
                {item.reason}
                {item.shot_ids?.length ? `（镜头 ${item.shot_ids.join('、')}）` : ''}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <h4 className="agent-trace-subtitle">阶段质量分与 Critic 问题</h4>
      {quality.length === 0 ? (
        <p className="agent-trace-empty">还没有阶段质量记录。</p>
      ) : (
        <ul className="agent-trace-quality">
          {quality.map((row) => (
            <li key={row.stage}>
              <span className="agent-trace-quality-stage">
                {row.label}
                <em className={'agent-trace-quality-score agent-trace-quality-' + row.status}>
                  {row.scoreText}·{STAGE_STATUS_LABELS[row.status] || row.status}
                </em>
              </span>
              {row.issues.length > 0 && (
                <ul className="agent-trace-issues">
                  {row.issues.slice(0, 3).map((issue, index) => (
                    <li key={index} className={'agent-trace-issue agent-trace-issue-' + issue.severity}>
                      {issue.shot_id ? `[${issue.shot_id}] ` : ''}
                      {issue.message}
                      {issue.recommendation ? ` → ${issue.recommendation}` : ''}
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      )}

      <h4 className="agent-trace-subtitle">最终决策（{decisions.length}）</h4>
      {decisions.length === 0 ? (
        <p className="agent-trace-empty">没有决策记录（流程尚未触发恢复决策）。</p>
      ) : (
        <ol className="agent-trace-decisions">
          {decisions.slice(-5).reverse().map((row) => (
            <li key={row.traceId}>
              <div className="agent-trace-decision-head">
                <strong>{row.stageLabel}</strong>
                <span className="agent-trace-decision-strategy">{row.selectedStrategyLabel}</span>
                {row.provider ? <span className="agent-trace-decision-provider">{row.provider}</span> : null}
                <span className="agent-trace-decision-cost">
                  预计 {row.estimatedCostText} / {row.estimatedSecondsText}
                </span>
              </div>
              <p className="agent-trace-decision-reason">{row.reason || row.failureMessage}</p>
              {row.rejected.length > 0 && (
                <p className="agent-trace-decision-rejected">
                  淘汰：{row.rejected.map((item) => `${item.label}（${item.reason}）`).join('；')}
                </p>
              )}
            </li>
          ))}
        </ol>
      )}

      {!compact && (
        <>
          <h4 className="agent-trace-subtitle">镜头状态（{shots.length}）</h4>
          {shots.length === 0 ? (
            <p className="agent-trace-empty">还没有镜头产物。</p>
          ) : (
            <table className="agent-trace-table">
              <thead>
                <tr>
                  <th>镜头</th>
                  <th>状态</th>
                  <th>Provider / 模型</th>
                  <th>参考图</th>
                  <th>成本</th>
                  <th>耗时</th>
                </tr>
              </thead>
              <tbody>
                {shots.map((row) => (
                  <tr key={row.shotId}>
                    <td title={row.failureKind || undefined}>{row.shotId}</td>
                    <td>{STAGE_STATUS_LABELS[row.status] || row.status}</td>
                    <td>{[row.provider, row.model].filter(Boolean).join(' / ') || '—'}</td>
                    <td>{row.references ? `${row.references} 张` : '—'}</td>
                    <td>{row.costText}</td>
                    <td>{row.durationText}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {candidates.length > 0 && (
            <>
              <h4 className="agent-trace-subtitle">候选结果（{candidates.length}）</h4>
              <table className="agent-trace-table">
                <thead>
                  <tr>
                    <th>镜头</th>
                    <th>候选</th>
                    <th>状态</th>
                    <th>Provider / 模型</th>
                    <th>分数</th>
                    <th>结构</th>
                    <th>耗时</th>
                    <th>选择</th>
                  </tr>
                </thead>
                <tbody>
                  {candidates.map((row) => (
                    <tr key={row.key} className={row.selected ? 'agent-trace-candidate-selected' : undefined}>
                      <td>{row.shotId}</td>
                      <td title={row.candidateId}>{row.candidateId}</td>
                      <td>{row.status}</td>
                      <td>{row.providerText}</td>
                      <td>{row.scoreText}</td>
                      <td>{row.structuralPassed}</td>
                      <td>{row.durationText}</td>
                      <td>{row.selected ? row.selectionReason || '已选中' : ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {prompts.length > 0 && (
            <>
              <h4 className="agent-trace-subtitle">Prompt 修改（{prompts.length}）</h4>
              <ul className="agent-trace-prompts">
                {prompts.slice(-6).reverse().map((row) => (
                  <li key={row.key}>
                    <span className="agent-trace-prompt-scope">
                      {row.stageLabel} · {row.shotId}
                    </span>
                    <code>
                      {row.field} {row.op} {row.valueText}
                    </code>
                    {row.reason ? <em>{row.reason}</em> : null}
                  </li>
                ))}
              </ul>
            </>
          )}

          {references.length > 0 && (
            <>
              <h4 className="agent-trace-subtitle">实际发送参考图</h4>
              <ul className="agent-trace-references">
                {references.map((row) => (
                  <li key={row.key}>
                    <span className="agent-trace-prompt-scope">
                      {row.shotId} · {row.stageLabel}
                    </span>
                    {row.count} 张：{row.names}
                  </li>
                ))}
              </ul>
            </>
          )}
        </>
      )}
    </div>
  )
}

export default AgentTracePanel
