import type React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { EditOutlined, ReloadOutlined, SaveOutlined } from '@ant-design/icons'
import Button from 'antd/es/button'
import Input from 'antd/es/input'
import InputNumber from 'antd/es/input-number'
import message from 'antd/es/message'
import Spin from 'antd/es/spin'
import Switch from 'antd/es/switch'
import Tooltip from 'antd/es/tooltip'

import { budgetApi } from '../services/api'
import type { BudgetSummaryDto, EffectiveBudgetDto } from '../services/costTypes'
import {
  budgetBadgeForState,
  budgetUsedPercent,
  describeBudgetError,
  formatDurationText,
  formatMicroAmount,
  amountTextToMicro,
  microToAmountText,
  unknownCostReason,
  usedCostText,
} from '../services/costModel'

interface BudgetSummaryPanelProps {
  /** 当前项目 / 剧集 ID；为空时不渲染面板。 */
  projectId: string | null
  /** 项目 / 剧集名称，仅用于标题。 */
  title?: string
}

interface BudgetForm {
  softCostText: string
  hardCostText: string
  softSeconds: number | null
  hardSeconds: number | null
  enabled: boolean
  note: string
}

const EMPTY_FORM: BudgetForm = {
  softCostText: '',
  hardCostText: '',
  softSeconds: null,
  hardSeconds: null,
  enabled: true,
  note: '',
}

function formFromBudget(budget: EffectiveBudgetDto): BudgetForm {
  return {
    softCostText: microToAmountText(budget.soft_cost_micro),
    hardCostText: microToAmountText(budget.hard_cost_micro),
    softSeconds: budget.soft_seconds,
    hardSeconds: budget.hard_seconds,
    enabled: budget.enabled,
    note: budget.note,
  }
}

/** 金额上限展示：未设置时明确写「未设置」，不要显示 ¥0。 */
function limitCostText(micro: number | null, currency: string): string {
  if (micro === null) return '未设置'
  return formatMicroAmount(micro, currency)
}

function limitSecondsText(seconds: number | null): string {
  if (seconds === null) return '未设置'
  return formatDurationText(seconds)
}

/**
 * 项目 / 剧集页的预算与成本面板。
 *
 * 数据来自 `GET /api/budget/summary`，编辑后走 `PUT /api/budget/config`（项目作用域）
 * 并重新拉取汇总。成本未知时一律显示「成本未知」并给出原因，绝不显示 ¥0。
 */
const BudgetSummaryPanel: React.FC<BudgetSummaryPanelProps> = ({ projectId, title }) => {
  const [summary, setSummary] = useState<BudgetSummaryDto | null>(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [editing, setEditing] = useState(false)
  const [form, setForm] = useState<BudgetForm>(EMPTY_FORM)

  const load = useCallback(async () => {
    if (!projectId) {
      setSummary(null)
      return
    }
    setLoading(true)
    setError('')
    try {
      setSummary(await budgetApi.summary({ project_id: projectId }))
    } catch (err) {
      setError(describeBudgetError(err, '加载预算与成本失败'))
    } finally {
      setLoading(false)
    }
  }, [projectId])

  useEffect(() => {
    void load()
  }, [load])

  const badge = useMemo(() => budgetBadgeForState(summary?.status), [summary])
  const usedPercent = useMemo(() => budgetUsedPercent(summary?.status), [summary])

  if (!projectId) return null

  const beginEdit = () => {
    if (!summary) return
    setForm(formFromBudget(summary.budget))
    setEditing(true)
  }

  const handleSave = async () => {
    if (!summary) return
    const softCost = amountTextToMicro(form.softCostText)
    const hardCost = amountTextToMicro(form.hardCostText)
    if (form.softCostText.trim() && softCost === null) {
      message.error('软预算金额不合法（最多 6 位小数，单位为元）')
      return
    }
    if (form.hardCostText.trim() && hardCost === null) {
      message.error('硬预算金额不合法（最多 6 位小数，单位为元）')
      return
    }
    if (softCost !== null && hardCost !== null && softCost > hardCost) {
      message.error('软预算不能高于硬预算')
      return
    }
    if (form.softSeconds !== null && form.hardSeconds !== null && Number(form.softSeconds) > Number(form.hardSeconds)) {
      message.error('软时长预算不能高于硬时长预算')
      return
    }

    setSaving(true)
    try {
      await budgetApi.saveConfig({
        scope_type: 'project',
        scope_id: projectId,
        currency: summary.currency,
        soft_cost_micro: softCost,
        hard_cost_micro: hardCost,
        soft_seconds: form.softSeconds === null ? null : Math.round(Number(form.softSeconds)),
        hard_seconds: form.hardSeconds === null ? null : Math.round(Number(form.hardSeconds)),
        enabled: form.enabled,
        note: form.note,
      })
      message.success('预算已保存，硬预算不足时会阻止任务启动')
      setEditing(false)
      await load()
    } catch (err) {
      message.error(describeBudgetError(err, '保存预算失败'))
    } finally {
      setSaving(false)
    }
  }

  const usedHint = summary
    ? `${summary.used.call_count} 次调用${
        summary.used.unknown_call_count > 0 ? `，其中 ${summary.used.unknown_call_count} 次成本未知` : ''
      }${summary.used.failed_call_count > 0 ? `，失败 ${summary.used.failed_call_count} 次` : ''}`
    : ''

  const unknownReason = summary ? unknownCostReason(summary.remaining.unknown_components) : ''

  return (
    <section className="budget-panel panel-enter" aria-label="预算与成本">
      <header className="budget-panel-head">
        <div className="budget-panel-title">
          <strong>预算与成本{title ? ' · ' + title : ''}</strong>
          <span className={'budget-badge budget-badge-' + badge.tone} style={{ color: badge.color }}>
            {badge.label}
          </span>
          {summary && <span className="budget-source">生效预算：{summary.budget.source_label}</span>}
        </div>
        <div className="budget-panel-actions">
          <Tooltip title="重新读取预算与成本">
            <Button
              size="small"
              icon={<ReloadOutlined />}
              disabled={loading || saving}
              onClick={() => void load()}
              aria-label="刷新预算与成本"
            >
              刷新
            </Button>
          </Tooltip>
          {!editing && (
            <Button size="small" icon={<EditOutlined />} disabled={!summary || loading} onClick={beginEdit}>
              编辑预算
            </Button>
          )}
        </div>
      </header>

      {loading && !summary && (
        <div className="budget-status" role="status">
          <Spin size="small" /> 正在加载预算与成本…
        </div>
      )}
      {error && (
        <div className="budget-error" role="alert">
          {error}
        </div>
      )}

      {summary && (
        <>
          {summary.status.message && (
            <p className={'budget-message budget-message-' + badge.tone} role="note">
              {summary.status.message}
            </p>
          )}

          <div className="budget-grid">
            <div className="budget-metric">
              <span>已用成本</span>
              <strong>{usedCostText(summary.used.cost_micro, summary.used.cost_known, summary.currency)}</strong>
              <em>{usedHint}</em>
            </div>
            <div className="budget-metric">
              <span>已预留（进行中的任务）</span>
              <strong>{formatMicroAmount(summary.reserved.cost_micro, summary.currency)}</strong>
              <em>
                {summary.reserved.count} 个任务 · {formatDurationText(summary.reserved.seconds)}
              </em>
            </div>
            <div className="budget-metric">
              <span>预计剩余成本（剩余工作量）</span>
              <strong>
                {summary.remaining.cost_known && summary.remaining.cost_micro !== null
                  ? formatMicroAmount(summary.remaining.cost_micro, summary.currency)
                  : '成本未知（' + unknownReason + '）'}
              </strong>
              <em>
                {summary.remaining.components.length > 0
                  ? summary.remaining.components.map((item) => item.component_label || item.label).join('、')
                  : '当前没有待执行的付费工作量'}
              </em>
            </div>
            <div className="budget-metric">
              <span>预计剩余耗时</span>
              <strong>
                {summary.remaining.seconds === null ? '暂无法估算' : formatDurationText(summary.remaining.seconds)}
              </strong>
              <em>
                {summary.remaining.unknown_components.length > 0
                  ? '未知原因：' + unknownReason
                  : '按已用/在跑的调用与剩余镜头数推算'}
              </em>
            </div>
            <div className="budget-metric">
              <span>软预算（只提示）</span>
              <strong>{limitCostText(summary.budget.soft_cost_micro, summary.currency)}</strong>
              <em>时长 {limitSecondsText(summary.budget.soft_seconds)}</em>
            </div>
            <div className="budget-metric">
              <span>硬预算（超限阻止任务）</span>
              <strong>{limitCostText(summary.budget.hard_cost_micro, summary.currency)}</strong>
              <em>时长 {limitSecondsText(summary.budget.hard_seconds)}</em>
            </div>
          </div>

          {usedPercent !== null && (
            <div className="budget-progress" role="group" aria-label="已用与预留占软预算比例">
              <div className="budget-progress-track">
                <div
                  className={'budget-progress-fill budget-progress-' + badge.tone}
                  style={{ width: usedPercent + '%', background: badge.color }}
                />
              </div>
              <span>
                已用 + 预留占软预算 {usedPercent}%（
                {formatMicroAmount(summary.status.committed_cost_micro, summary.currency)} /{' '}
                {limitCostText(summary.status.soft_cost_micro ?? summary.status.hard_cost_micro, summary.currency)}）
              </span>
            </div>
          )}

          {summary.episodes.length > 0 && (
            <>
              <h4 className="budget-subtitle">各剧集成本</h4>
              <table className="budget-episode-table">
                <thead>
                  <tr>
                    <th>剧集</th>
                    <th>已用成本</th>
                    <th>预算状态</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.episodes.map((episode) => {
                    const episodeBadge = budgetBadgeForState(episode.budget_status)
                    return (
                      <tr key={episode.project_id}>
                        <td>
                          第 {episode.episode_number || 1} 集 · {episode.title}
                        </td>
                        <td>
                          {usedCostText(episode.used.cost_micro, episode.used.cost_known, summary.currency)}
                          {episode.used.unknown_call_count > 0
                            ? `（${episode.used.unknown_call_count} 次调用成本未知）`
                            : ''}
                        </td>
                        <td>
                          <span
                            className={'budget-badge budget-badge-' + episodeBadge.tone}
                            style={{ color: episodeBadge.color }}
                          >
                            {episodeBadge.label}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </>
          )}

          {!summary.used.cost_known && (
            <p className="budget-unknown-note" role="note">
              有调用没有匹配到单价，这部分成本显示为「成本未知」而不是 0；请在 「系统设置 →
              模型价格」补齐单价后重新查看。
            </p>
          )}
        </>
      )}

      {editing && summary && (
        <div className="budget-editor">
          <div className="budget-editor-grid">
            <label className="budget-field">
              <span>软预算金额（元）</span>
              <Input
                size="small"
                value={form.softCostText}
                placeholder="留空表示不限制，例如 50.00"
                onChange={(event) => setForm({ ...form, softCostText: event.target.value })}
              />
            </label>
            <label className="budget-field">
              <span>硬预算金额（元）</span>
              <Input
                size="small"
                value={form.hardCostText}
                placeholder="留空表示不限制，例如 80.00"
                onChange={(event) => setForm({ ...form, hardCostText: event.target.value })}
              />
            </label>
            <label className="budget-field">
              <span>软时长预算（秒）</span>
              <InputNumber
                size="small"
                style={{ width: '100%' }}
                min={0}
                value={form.softSeconds ?? undefined}
                onChange={(value) => setForm({ ...form, softSeconds: value === null ? null : Number(value) })}
              />
            </label>
            <label className="budget-field">
              <span>硬时长预算（秒）</span>
              <InputNumber
                size="small"
                style={{ width: '100%' }}
                min={0}
                value={form.hardSeconds ?? undefined}
                onChange={(value) => setForm({ ...form, hardSeconds: value === null ? null : Number(value) })}
              />
            </label>
            <label className="budget-field">
              <span>备注</span>
              <Input
                size="small"
                value={form.note}
                placeholder="例如「本集预算，含重跑」"
                onChange={(event) => setForm({ ...form, note: event.target.value })}
              />
            </label>
            <div className="budget-field budget-field-switch">
              <span>启用预算</span>
              <Switch
                size="small"
                checked={form.enabled}
                onChange={(checked) => setForm({ ...form, enabled: checked })}
              />
            </div>
          </div>
          <div className="budget-editor-actions">
            <em>软预算超支只提示，硬预算不足会阻止任务启动。</em>
            <Button size="small" onClick={() => setEditing(false)} disabled={saving}>
              取消
            </Button>
            <Button
              size="small"
              type="primary"
              icon={<SaveOutlined />}
              loading={saving}
              onClick={() => void handleSave()}
            >
              保存预算
            </Button>
          </div>
        </div>
      )}
    </section>
  )
}

export default BudgetSummaryPanel
