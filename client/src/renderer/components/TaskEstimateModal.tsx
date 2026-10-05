import type React from 'react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { ExclamationCircleOutlined } from '@ant-design/icons'
import Alert from 'antd/es/alert'
import Button from 'antd/es/button'
import Modal from 'antd/es/modal'
import Spin from 'antd/es/spin'
import Tag from 'antd/es/tag'
import notification from 'antd/es/notification'

import { budgetApi } from '../services/api'
import type { EstimateRequest, TaskEstimateResponse } from '../services/costTypes'
import {
  UNKNOWN_COST_TEXT,
  budgetBadgeForState,
  budgetBlockedFromError,
  budgetWarningFromResponse,
  componentCostText,
  componentQuantityText,
  describeBudgetError,
  durationSourceLabel,
  estimateCostText,
  formatDurationText,
  formatMicroAmount,
  providerBlockedFromError,
  unknownCostReason,
  usedCostText,
} from '../services/costModel'
import { OPEN_SETTINGS_EVENT } from './TopBar'

export interface TaskEstimateRequest extends EstimateRequest {
  /** 展示用的入口名称，例如「生成故事板」。 */
  entryLabel?: string
}

interface TaskEstimateModalProps {
  open: boolean
  request: TaskEstimateRequest | null
  onCancel: () => void
  onConfirm: () => void
}

/**
 * 提交前估算确认弹窗。
 *
 * 只做展示与确认：真正的硬预算拦截在后端任务抢占时（`claim_job`）执行，因此这里
 * 的结论是提示而不是保证。blocked=true 时禁用「确认执行」并高亮原因。
 */
export const TaskEstimateModal: React.FC<TaskEstimateModalProps> = ({ open, request, onCancel, onConfirm }) => {
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState<TaskEstimateResponse | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!open || !request) {
      setResponse(null)
      setError('')
      return
    }
    let cancelled = false
    setLoading(true)
    setError('')
    setResponse(null)
    void budgetApi
      .estimate({
        job_type: request.job_type,
        project_id: request.project_id,
        shot_id: request.shot_id,
        shot_ids: request.shot_ids,
      })
      .then((result) => {
        if (cancelled) return
        setResponse(result)
      })
      .catch((err) => {
        if (cancelled) return
        setError(describeBudgetError(err, '无法获取本次任务的估算'))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [open, request])

  const estimate = response?.estimate || null
  const budget = response?.budget || null
  const badge = budgetBadgeForState(budget)
  const blocked = Boolean(response?.blocked)

  return (
    <Modal
      open={open}
      title={request?.entryLabel ? '执行前确认 · ' + request.entryLabel : '执行前确认'}
      onCancel={onCancel}
      width={680}
      maskClosable={false}
      destroyOnClose
      footer={[
        <Button key="cancel" onClick={onCancel}>
          取消
        </Button>,
        <Button key="confirm" type="primary" danger={blocked} disabled={loading || blocked} onClick={onConfirm}>
          {blocked ? '预算不足，无法执行' : '确认执行'}
        </Button>,
      ]}
    >
      {loading && (
        <div className="estimate-status" role="status">
          <Spin size="small" /> 正在估算本次任务的成本与耗时…
        </div>
      )}

      {!loading && error && (
        <>
          <Alert type="warning" showIcon message="估算失败" description={error} />
          <p className="estimate-note">
            估算接口不可用时不会阻止你继续：硬预算拦截仍会在后端启动任务时执行。你可以关闭本窗口后重试，或直接继续。
          </p>
        </>
      )}

      {!loading && !error && estimate && (
        <>
          {blocked && (
            <Alert
              type="error"
              showIcon
              icon={<ExclamationCircleOutlined />}
              className="estimate-blocked"
              message="硬预算不足，本次任务无法启动"
              description={budget?.message || '项目已用成本与本次预计成本之和超过硬预算。'}
            />
          )}
          {!blocked && response?.warning && (
            <Alert type="warning" showIcon className="estimate-warning" message={response.warning} />
          )}

          <dl className="estimate-summary">
            <dt>任务类型</dt>
            <dd>{estimate.job_type_label || estimate.job_type}</dd>
            <dt>预计成本</dt>
            <dd className={estimate.cost_known ? '' : 'estimate-unknown'}>
              {estimateCostText(estimate)}
              {!estimate.cost_known && (
                <em className="estimate-hint">
                  未配置单价的调用不会被当成 0 元，只会在成本里显示「{UNKNOWN_COST_TEXT}」。
                </em>
              )}
            </dd>
            <dt>预计耗时</dt>
            <dd>
              {estimate.estimated_seconds === null ? '暂无法估算' : formatDurationText(estimate.estimated_seconds)}
              <em className="estimate-hint">{durationSourceLabel(estimate.duration_source)}</em>
            </dd>
            <dt>预算状态</dt>
            <dd>
              <span className={'budget-badge budget-badge-' + badge.tone} style={{ color: badge.color }}>
                {badge.label}
              </span>
              {budget && <em className="estimate-hint">生效预算：{budget.budget_source_label}</em>}
            </dd>
          </dl>

          {estimate.note && <p className="estimate-note">{estimate.note}</p>}

          <h4 className="estimate-subtitle">逐项拆解</h4>
          {estimate.components.length === 0 ? (
            <p className="estimate-note">当前项目状态下没有需要执行的工作量。</p>
          ) : (
            <table className="estimate-table">
              <thead>
                <tr>
                  <th>能力 / 环节</th>
                  <th>模型</th>
                  <th>数量</th>
                  <th>预计成本</th>
                  <th>预计耗时</th>
                </tr>
              </thead>
              <tbody>
                {estimate.components.map((component, index) => (
                  <tr key={component.capability + '-' + index}>
                    <td>
                      {component.label}
                      {component.component_label ? ' · ' + component.component_label : ''}
                      {component.resolution ? '（' + component.resolution + '）' : ''}
                    </td>
                    <td>{[component.provider, component.model].filter(Boolean).join(' / ') || '默认'}</td>
                    <td>
                      {componentQuantityText(component)}
                      {component.calls > 0 ? ' · ' + component.calls + ' 次调用' : ''}
                    </td>
                    <td className={component.cost_known ? '' : 'estimate-unknown'}>{componentCostText(component)}</td>
                    <td>{formatDurationText(component.estimated_seconds)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {estimate.unknown_components.length > 0 && (
            <div className="estimate-unknown-block" role="note">
              <strong>以下环节缺少单价，成本无法计算：</strong>
              <ul>
                {estimate.unknown_components.map((item, index) => (
                  <li key={item.capability + '-' + index}>
                    {item.label}
                    {item.provider || item.model
                      ? '（' + [item.provider, item.model].filter(Boolean).join(' / ') + '）'
                      : ''}
                    ：{item.reason || unknownCostReason(estimate.unknown_components)}
                  </li>
                ))}
              </ul>
              <span>请在「系统设置 → 模型价格」补齐单价后再执行，可获得准确的成本统计。</span>
            </div>
          )}

          <h4 className="estimate-subtitle">当前预算使用情况</h4>
          {budget ? (
            <div className="estimate-budget-grid">
              <span>
                已用成本
                <strong>{usedCostText(budget.used_cost_micro, budget.cost_known, budget.currency)}</strong>
              </span>
              <span>
                已预留
                <strong>{formatMicroAmount(budget.reserved_cost_micro, budget.currency)}</strong>
              </span>
              <span>
                软预算
                <strong>
                  {budget.soft_cost_micro === null
                    ? '未设置'
                    : formatMicroAmount(budget.soft_cost_micro, budget.currency)}
                </strong>
              </span>
              <span>
                硬预算
                <strong>
                  {budget.hard_cost_micro === null
                    ? '未设置'
                    : formatMicroAmount(budget.hard_cost_micro, budget.currency)}
                </strong>
              </span>
              <span>
                本次执行后预计
                <strong>{formatMicroAmount(budget.projected_cost_micro, budget.currency)}</strong>
              </span>
              <span>
                时长（已用 / 硬上限）
                <strong>
                  {formatDurationText(budget.used_seconds)} /{' '}
                  {budget.hard_seconds === null ? '未设置' : formatDurationText(budget.hard_seconds)}
                </strong>
              </span>
            </div>
          ) : (
            <p className="estimate-note">未获取到预算状态。</p>
          )}

          {budget && budget.usage.unknown_call_count > 0 && (
            <p className="estimate-note" role="note">
              已有 <Tag color="orange">{budget.usage.unknown_call_count}</Tag> 次调用因未配置单价显示为「
              {UNKNOWN_COST_TEXT}」，这些调用没有被计入金额校验。
            </p>
          )}
        </>
      )}
    </Modal>
  )
}

/**
 * 提交前估算闸门：`await confirm({...})` 返回 true 才继续提交真正的任务。
 *
 * 用法：
 * ```tsx
 * const estimateGate = useTaskEstimateGate()
 * ...
 * if (!(await estimateGate.confirm({ job_type: 'storyboard', project_id: id }))) return
 * ...
 * {estimateGate.modal}
 * ```
 */
export function useTaskEstimateGate(): {
  confirm: (request: TaskEstimateRequest) => Promise<boolean>
  modal: React.ReactNode
} {
  const [request, setRequest] = useState<TaskEstimateRequest | null>(null)
  const resolverRef = useRef<((ok: boolean) => void) | null>(null)

  const settle = useCallback((ok: boolean) => {
    const resolver = resolverRef.current
    resolverRef.current = null
    setRequest(null)
    if (resolver) resolver(ok)
  }, [])

  const confirm = useCallback((next: TaskEstimateRequest) => {
    // 上一个还开着就先按「取消」结掉，避免 Promise 悬挂导致入口按钮永久卡住。
    if (resolverRef.current) {
      const previous = resolverRef.current
      resolverRef.current = null
      previous(false)
    }
    return new Promise<boolean>((resolve) => {
      resolverRef.current = resolve
      setRequest(next)
    })
  }, [])

  return {
    confirm,
    modal: (
      <TaskEstimateModal
        open={Boolean(request)}
        request={request}
        onCancel={() => settle(false)}
        onConfirm={() => settle(true)}
      />
    ),
  }
}

/** 提交成功后若响应带 budget_warning，用通知提示（软预算只提示不阻断）。 */
export function notifyBudgetWarning(response: unknown): void {
  const warning = budgetWarningFromResponse(response)
  if (!warning) return
  notification.warning({
    message: '已超出项目软预算',
    description: warning,
    duration: 8,
    placement: 'bottomRight',
  })
}

/**
 * 提交失败时先看是不是硬预算拦截（HTTP 409 + budget_blocked）。
 *
 * 返回被拦截的中文原因；返回空串表示这不是预算问题，调用方按原有错误提示处理。
 */
export function notifyBudgetBlocked(error: unknown): string {
  const blocked = budgetBlockedFromError(error)
  if (!blocked) return ''
  notification.error({
    message: '已超出硬预算，任务未启动',
    description: blocked.message,
    duration: 10,
    placement: 'bottomRight',
  })
  return blocked.message
}

/**
 * 提交失败时先看是不是模型端点未配置拦截（HTTP 409 + provider_not_configured）。
 *
 * 返回被拦截的中文原因；返回空串表示这不是配置问题，调用方按原有错误提示处理。
 * 通知里附「去系统设置」按钮，直接跳到模型服务配置页。
 */
export function notifyProviderBlocked(error: unknown): string {
  const blocked = providerBlockedFromError(error)
  if (!blocked) return ''
  const missingLabels = (blocked.missing || []).map((item) => item.label).filter(Boolean)
  notification.error({
    message: '模型服务未配置，任务未启动',
    description: missingLabels.length
      ? blocked.message + '（缺少：' + missingLabels.join('、') + '）'
      : blocked.message,
    btn: (
      <Button
        type="primary"
        size="small"
        onClick={() => {
          notification.destroy('provider-blocked')
          window.dispatchEvent(new CustomEvent(OPEN_SETTINGS_EVENT))
        }}
      >
        去系统设置
      </Button>
    ),
    key: 'provider-blocked',
    duration: 12,
    placement: 'bottomRight',
  })
  return blocked.message
}

export default TaskEstimateModal
