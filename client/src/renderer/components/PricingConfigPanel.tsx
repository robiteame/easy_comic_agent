import type React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { InfoCircleOutlined, ReloadOutlined, SaveOutlined } from '@ant-design/icons'
import Alert from 'antd/es/alert'
import Button from 'antd/es/button'
import Input from 'antd/es/input'
import message from 'antd/es/message'
import Spin from 'antd/es/spin'
import Switch from 'antd/es/switch'
import Tag from 'antd/es/tag'
import Tooltip from 'antd/es/tooltip'

import { budgetApi } from '../services/api'
import type { PricingCapabilityDto, PricingItemDto, PricingTableDto } from '../services/costTypes'
import {
  DEFAULT_UNKNOWN_PRICE_REASON,
  PRICING_DISCLAIMER,
  UNKNOWN_COST_TEXT,
  amountTextToMicro,
  describeBudgetError,
  formatMicroAmount,
  microToAmountText,
  microToMultiplierText,
  multipliersFromText,
  multipliersToText,
} from '../services/costModel'

/**
 * 价目表编辑草稿。
 *
 * 单价一律以「元」文本编辑，保存时才换算成整数 micro：输入框里保留用户原样的
 * 字符串，避免「输入 0.07 被浮点乘成 69999.999…」这类脏数据。
 */
interface PricingDraft {
  key: string
  capability: string
  provider: string
  model: string
  priceText: string
  secondaryText: string
  multipliersText: string
  configured: boolean
  note: string
}

function draftKey(item: { capability: string; provider: string; model: string }): string {
  return item.capability + '|' + item.provider + '|' + item.model
}

function toDraft(item: PricingItemDto): PricingDraft {
  return {
    key: draftKey(item),
    capability: item.capability,
    provider: item.provider,
    model: item.model,
    priceText: microToAmountText(item.unit_price_micro),
    secondaryText: microToAmountText(item.unit_price_secondary_micro),
    multipliersText: multipliersToText(item.resolution_multipliers),
    configured: item.configured,
    note: item.note,
  }
}

function draftMap(table: PricingTableDto): Record<string, PricingDraft> {
  const next: Record<string, PricingDraft> = {}
  for (const capability of table.capabilities) {
    for (const item of capability.items) {
      next[draftKey(item)] = toDraft(item)
    }
  }
  return next
}

const PricingConfigPanel: React.FC = () => {
  const [table, setTable] = useState<PricingTableDto | null>(null)
  const [drafts, setDrafts] = useState<Record<string, PricingDraft>>({})
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [dirty, setDirty] = useState(false)

  const applyTable = useCallback((next: PricingTableDto) => {
    setTable(next)
    setDrafts(draftMap(next))
    setDirty(false)
  }, [])

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      applyTable(await budgetApi.pricing())
    } catch (err) {
      setError(describeBudgetError(err, '加载模型价格失败'))
    } finally {
      setLoading(false)
    }
  }, [applyTable])

  useEffect(() => {
    void load()
  }, [load])

  const updateDraft = (key: string, patch: Partial<PricingDraft>) => {
    setDrafts((current) => (current[key] ? { ...current, [key]: { ...current[key], ...patch } } : current))
    setDirty(true)
  }

  const pricePreview = useMemo(() => {
    // 只做展示：把当前输入换算成整数 micro，让用户确认没有浮点尾巴。
    const previews: Record<string, string> = {}
    for (const draft of Object.values(drafts)) {
      const micro = amountTextToMicro(draft.priceText)
      previews[draft.key] = draft.priceText.trim() ? (micro === null ? '单价非法' : micro + ' micro') : '未配置'
    }
    return previews
  }, [drafts])

  const handleSave = async () => {
    if (!table) return
    const items: {
      capability: string
      provider: string
      model: string
      unit_price_micro: number | null
      unit_price_secondary_micro: number | null
      resolution_multipliers: Record<string, number>
      configured: boolean
      note: string
    }[] = []

    for (const capability of table.capabilities) {
      for (const item of capability.items) {
        const draft = drafts[draftKey(item)]
        if (!draft) continue
        const price = amountTextToMicro(draft.priceText)
        if (draft.priceText.trim() && price === null) {
          message.error(`${capability.label} / ${draft.provider || '默认'} 的单价不是合法数字（最多 6 位小数）`)
          return
        }
        const secondary = amountTextToMicro(draft.secondaryText)
        if (draft.secondaryText.trim() && secondary === null) {
          message.error(`${capability.label} / ${draft.provider || '默认'} 的次级单价不是合法数字`)
          return
        }
        const parsed = multipliersFromText(draft.multipliersText)
        if (parsed.error) {
          message.error(`${capability.label} / ${draft.provider || '默认'}：${parsed.error}`)
          return
        }
        // 后端规定「标记已配置就必须有单价」，这里按是否填了单价兜底，避免 400。
        const configured = Boolean(draft.configured && price !== null)
        if (!draft.provider.trim() && !draft.model.trim() && price === null) continue
        items.push({
          capability: capability.capability,
          provider: draft.provider.trim(),
          model: draft.model.trim(),
          unit_price_micro: price,
          unit_price_secondary_micro: capability.secondary_unit ? secondary : null,
          resolution_multipliers: parsed.multipliers,
          configured,
          note: draft.note,
        })
      }
    }

    if (items.length === 0) {
      message.warning('没有需要保存的价目行')
      return
    }

    setSaving(true)
    try {
      const saved = await budgetApi.savePricing({ currency: table.currency, items })
      applyTable(saved)
      message.success('模型价格已保存，新任务将按新单价计费')
    } catch (err) {
      message.error(describeBudgetError(err, '保存模型价格失败'))
    } finally {
      setSaving(false)
    }
  }

  const renderCapability = (capability: PricingCapabilityDto) => (
    <section className="pricing-capability" key={capability.capability} aria-label={capability.label + ' 单价配置'}>
      <header className="pricing-capability-head">
        <strong>{capability.label}</strong>
        <Tag color="blue">{capability.pricing_unit}</Tag>
        <span className="pricing-capability-note">
          基础数量单位：{capability.base_unit || '单位'}（1 个计价单位 = {capability.unit_scale}{' '}
          {capability.base_unit || '单位'}）{capability.secondary_unit ? ` · 次级：${capability.secondary_unit}` : ''}
        </span>
      </header>

      {capability.items.length === 0 ? (
        <p className="pricing-empty">该能力暂无可配置的厂商行</p>
      ) : (
        <ul className="pricing-list" role="list">
          {capability.items.map((item) => {
            const key = draftKey(item)
            const draft = drafts[key]
            if (!draft) return null
            return (
              <li className="pricing-row" key={key}>
                <label className="pricing-field">
                  <span>Provider</span>
                  <Input
                    size="small"
                    value={draft.provider}
                    placeholder="例如 ark-seedance"
                    onChange={(event) => updateDraft(key, { provider: event.target.value })}
                    aria-label={capability.label + ' provider'}
                  />
                </label>
                <label className="pricing-field">
                  <span>模型</span>
                  <Input
                    size="small"
                    value={draft.model}
                    placeholder="留空表示该厂商默认模型"
                    onChange={(event) => updateDraft(key, { model: event.target.value })}
                    aria-label={capability.label + ' 模型名'}
                  />
                </label>
                <label className="pricing-field">
                  <span>单价（元 / {capability.pricing_unit.replace(/^每/, '')}）</span>
                  <Input
                    size="small"
                    value={draft.priceText}
                    placeholder="留空 = 未配置（成本显示未知）"
                    onChange={(event) => updateDraft(key, { priceText: event.target.value })}
                    aria-label={capability.label + ' 单价（元）'}
                  />
                  <em className="pricing-hint">
                    {item.unit_price_micro === null
                      ? UNKNOWN_COST_TEXT
                      : '当前 ' + formatMicroAmount(item.unit_price_micro, item.currency)}
                    {' · '}
                    {pricePreview[key]}
                  </em>
                </label>
                {capability.secondary_unit && (
                  <label className="pricing-field">
                    <span>次级单价（元 / {capability.secondary_unit.replace(/^每/, '')}）</span>
                    <Input
                      size="small"
                      value={draft.secondaryText}
                      placeholder="仅 LLM 输出 token"
                      onChange={(event) => updateDraft(key, { secondaryText: event.target.value })}
                      aria-label={capability.label + ' 次级单价（元）'}
                    />
                  </label>
                )}
                <label className="pricing-field">
                  <span>分辨率倍率</span>
                  <Input
                    size="small"
                    value={draft.multipliersText}
                    placeholder="1080p=1，720p=0.6（可留空）"
                    onChange={(event) => updateDraft(key, { multipliersText: event.target.value })}
                    aria-label={capability.label + ' 分辨率倍率'}
                  />
                  <em className="pricing-hint">
                    {Object.keys(item.resolution_multipliers).length === 0
                      ? '当前无倍率'
                      : Object.entries(item.resolution_multipliers)
                          .map(([name, micro]) => name + '×' + microToMultiplierText(micro))
                          .join('、')}
                  </em>
                </label>
                <div className="pricing-field pricing-field-compact">
                  <span>已配置</span>
                  <Tooltip title="未配置价格的调用会显示成本未知，不会被当成 0 元">
                    <Switch
                      size="small"
                      checked={draft.configured}
                      onChange={(checked) => updateDraft(key, { configured: checked })}
                      aria-label={capability.label + ' 是否已配置单价'}
                    />
                  </Tooltip>
                </div>
                <label className="pricing-field">
                  <span>备注</span>
                  <Input
                    size="small"
                    value={draft.note}
                    placeholder="例如「官网价，含税」"
                    onChange={(event) => updateDraft(key, { note: event.target.value })}
                    aria-label={capability.label + ' 备注'}
                  />
                </label>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )

  return (
    <div className="pricing-panel">
      <div className="settings-models-head">
        <div>
          <div className="settings-section-title">模型价格</div>
          <div className="asset-board-note">
            单价按「元」填写，保存时换算成整数 micro（1 元 = 1,000,000 micro），金额不落浮点。
            {table ? ` 当前币种：${table.currency}。` : ''}
          </div>
        </div>
        <div className="pricing-toolbar">
          <Button size="small" icon={<ReloadOutlined />} onClick={() => void load()} disabled={loading || saving}>
            重新加载
          </Button>
          <Button
            size="small"
            type="primary"
            icon={<SaveOutlined />}
            loading={saving}
            disabled={loading || !table}
            onClick={() => void handleSave()}
          >
            保存模型价格
          </Button>
        </div>
      </div>

      <Alert
        type="info"
        showIcon
        icon={<InfoCircleOutlined />}
        className="pricing-disclaimer"
        message="计价单位说明"
        description={
          <ul className="pricing-unit-list">
            <li>文本模型（LLM）：每 100 万 tokens，输出 token 走次级单价</li>
            <li>图像生成：每张</li>
            <li>视频生成：每秒</li>
            <li>语音合成：每 1000 字符</li>
            <li>本地编码（FFmpeg）：每分钟编码</li>
            <li>{PRICING_DISCLAIMER}</li>
            <li>未填写的原因：{DEFAULT_UNKNOWN_PRICE_REASON}</li>
          </ul>
        }
      />

      {error && (
        <div className="pricing-error" role="alert">
          {error}
        </div>
      )}
      {loading && !table && (
        <div className="pricing-status" role="status">
          <Spin size="small" /> 正在加载模型价格…
        </div>
      )}
      {dirty && (
        <div className="pricing-dirty" role="status">
          有未保存的修改，点击「保存模型价格」后生效
        </div>
      )}

      {table?.capabilities.map(renderCapability)}
    </div>
  )
}

export default PricingConfigPanel
