/**
 * 成本、预算与用量的纯逻辑层。
 *
 * 与 `components/taskCenterModel.ts` 同样的定位：不碰 DOM、不碰网络，只做
 * 「归一化后端响应 + 格式化展示」，因此可以在 node:test 下直接覆盖。
 *
 * 三条贯穿全文件的铁律：
 *
 * 1. `cost_known === false` 或金额为 null 时，展示一律是「成本未知」/「暂无用量」，
 *    绝不出现 ¥0 或任何猜测值；
 * 2. 金额格式化只做整数 micro 的字符串拼接，不引入浮点误差；
 * 3. 缺字段、老数据、脏数据都归一化成稳定的兜底对象，绝不让 undefined 进入渲染层。
 */

import type {
  BudgetCapability,
  BudgetConfigDto,
  BudgetConfigRowDto,
  BudgetLevel,
  BudgetReason,
  BudgetSource,
  BudgetStateDto,
  CapabilityUsageDto,
  CostSource,
  DurationSource,
  EffectiveBudgetDto,
  EstimateComponentDto,
  EstimateUnknownComponentDto,
  JobCostDetailDto,
  JobCostDto,
  JobStatsCostDto,
  PricingCapabilityDto,
  PricingItemDto,
  PricingTableDto,
  TaskEstimateDto,
  UsageGroupDto,
  UsageRecordDto,
  UsageRecordPageDto,
  UsageSummaryDto,
} from './costTypes'
// node:test 用 --experimental-strip-types 直接跑 .ts，值导入必须带扩展名。
import { MICRO_PER_UNIT } from './costTypes.ts'

// --- 稳定的展示文案 --------------------------------------------------------

/** 成本未知时的统一文案：只用于「有调用但单价缺失」的情况。 */
export const UNKNOWN_COST_TEXT = '成本未知'
/** 没有任何用量记录时的文案（老数据 / 尚未执行）。 */
export const NO_USAGE_TEXT = '暂无用量'
/** 未知成本的默认原因；后端给了更具体的原因时优先用后端的。 */
export const DEFAULT_UNKNOWN_PRICE_REASON = '未配置对应模型单价'
/** 「未配置价格的调用会显示成本未知，不会按 0 计费」的固定说明。 */
export const PRICING_DISCLAIMER = '未配置价格的调用会显示「成本未知」，不会按 0 计费。'

export const HISTORY_DURATION_TEXT = '按历史耗时估算'
export const HEURISTIC_DURATION_TEXT = '按单位耗时模型估算'
export const UNKNOWN_DURATION_TEXT = '暂无法估算耗时'

// --- 基础归一化工具（与 taskCenterModel 同风格） ---------------------------

export function asString(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  return fallback
}

export function asText(value: unknown, fallback: string): string {
  const text = asString(value).trim()
  return text || fallback
}

export function asBoolean(value: unknown, fallback = false): boolean {
  return typeof value === 'boolean' ? value : fallback
}

export function asNumber(value: unknown, fallback = 0): number {
  const parsed = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

export function asCount(value: unknown): number {
  return Math.max(0, Math.round(asNumber(value)))
}

const CURRENCY_SYMBOLS: Record<string, string> = {
  CNY: '¥',
  RMB: '¥',
  USD: '$',
  EUR: '€',
  GBP: '£',
  JPY: '¥',
  HKD: 'HK$',
}

export function currencySymbol(currency: string): string {
  const code = asString(currency, 'CNY').trim().toUpperCase()
  return CURRENCY_SYMBOLS[code] || code + ' '
}

/** 整数 micro -> 人类可读金额；null / 非法 / 负数一律回落到「成本未知」。 */
export function formatMicroAmount(micro: number | null | undefined, currency = 'CNY'): string {
  if (micro === null || micro === undefined) return UNKNOWN_COST_TEXT
  if (typeof micro !== 'number' || !Number.isFinite(micro)) return UNKNOWN_COST_TEXT
  if (micro < 0) return UNKNOWN_COST_TEXT
  const value = Math.round(micro)
  const whole = Math.floor(value / MICRO_PER_UNIT)
  const fraction = value - whole * MICRO_PER_UNIT
  let fractionText = String(fraction).padStart(6, '0').replace(/0+$/, '')
  if (fractionText.length < 2) fractionText = fractionText.padEnd(2, '0')
  return currencySymbol(currency) + groupDigits(whole) + '.' + fractionText
}

function groupDigits(value: number): string {
  const text = String(Math.max(0, Math.round(value)))
  let out = ''
  for (let index = 0; index < text.length; index += 1) {
    const rest = text.length - index
    out += text[index]
    if (rest > 1 && rest % 3 === 1) out += ','
  }
  return out
}

/**
 * 金额 + 已知标记 -> 展示文案。
 *
 * `costKnown=false` 时无论 micro 是什么（包括 0）都必须显示「成本未知」。
 */
export function formatCostValue(micro: number | null | undefined, costKnown: boolean, currency = 'CNY'): string {
  if (!costKnown || micro === null || micro === undefined) return UNKNOWN_COST_TEXT
  return formatMicroAmount(micro, currency)
}

/** 耗时格式化（秒 -> 中文）；未知返回破折号，绝不显示 0 秒冒充已知。 */
export function formatDurationText(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '—'
  if (typeof seconds !== 'number' || !Number.isFinite(seconds)) return '—'
  const total = Math.max(0, Math.round(seconds))
  if (total < 60) return total + ' 秒'
  const minutes = Math.floor(total / 60)
  const rest = total % 60
  if (minutes < 60) return minutes + ' 分 ' + String(rest).padStart(2, '0') + ' 秒'
  const hours = Math.floor(minutes / 60)
  return hours + ' 小时 ' + String(minutes % 60).padStart(2, '0') + ' 分'
}

/** 供应商调用耗时（毫秒）格式化。 */
export function formatMillisecondsText(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return '—'
  return formatDurationText(Math.round(ms / 1000))
}

// --- 金额输入（元 <-> 整数 micro，全程字符串运算） -------------------------

/**
 * 「元」文本 -> 整数 micro。
 *
 * 用字符串手工换算而不是 `Number(text) * 1e6`：浮点乘法会把 0.07 变成
 * 69999.999…，落库时就成了脏数据。非法输入（空、负号、超过 6 位小数、
 * 科学计数法、超大数）一律返回 null，由调用方给出中文提示。
 */
export function amountTextToMicro(text: string): number | null {
  const raw = asString(text).trim()
  if (!raw) return null
  if (!/^[0-9]+(\.[0-9]*)?$/.test(raw)) return null
  const [wholeText, fractionText = ''] = raw.split('.')
  if (fractionText.length > 6) return null
  const whole = Number(wholeText)
  if (!Number.isSafeInteger(whole)) return null
  const fraction = Number((fractionText + '000000').slice(0, 6))
  const micro = whole * MICRO_PER_UNIT + fraction
  if (!Number.isSafeInteger(micro) || micro < 0) return null
  return micro
}

/** 整数 micro -> 「元」输入框文本；null 返回空串（表示未配置）。 */
export function microToAmountText(micro: number | null | undefined): string {
  if (micro === null || micro === undefined) return ''
  if (typeof micro !== 'number' || !Number.isFinite(micro) || micro < 0) return ''
  const value = Math.round(micro)
  const whole = Math.floor(value / MICRO_PER_UNIT)
  const fraction = value - whole * MICRO_PER_UNIT
  let fractionText = String(fraction).padStart(6, '0').replace(/0+$/, '')
  if (fractionText.length < 2) fractionText = fractionText.padEnd(2, '0')
  return whole + '.' + fractionText
}

/** 分辨率倍率：倍率文本 <-> 整数 micro，换算规则与金额一致（1.0 = 1_000_000）。 */
export const multiplierTextToMicro = amountTextToMicro
export const microToMultiplierText = microToAmountText

/** 倍率表 -> 「名称=倍率」文本；空表返回空串（表示不使用倍率）。 */
export function multipliersToText(multipliers: Record<string, number> | null | undefined): string {
  if (!multipliers) return ''
  return Object.keys(multipliers)
    .sort()
    .map((name) => name + '=' + microToMultiplierText(multipliers[name]))
    .join('，')
}

/**
 * 「名称=倍率」文本 -> 倍率表。
 *
 * 支持中英文逗号 / 分号 / 换行分隔；解析失败时返回 error，调用方据此给出中文提示，
 * 不做静默丢弃（否则用户会以为倍率保存成功了）。
 */
export function multipliersFromText(text: string): { multipliers: Record<string, number>; error: string } {
  const multipliers: Record<string, number> = {}
  const raw = asString(text).trim()
  if (!raw) return { multipliers, error: '' }
  const parts = raw.split(/[，,;；\n]/)
  for (const part of parts) {
    const entry = part.trim()
    if (!entry) continue
    const separator = entry.indexOf('=')
    if (separator <= 0) return { multipliers: {}, error: '分辨率倍率请按「名称=倍率」填写，例如 1080p=1，720p=0.6' }
    const name = entry.slice(0, separator).trim()
    const value = amountTextToMicro(entry.slice(separator + 1).trim())
    if (!name) return { multipliers: {}, error: '分辨率倍率缺少名称' }
    if (value === null) return { multipliers: {}, error: '分辨率倍率 ' + name + ' 必须是数字（最多 6 位小数）' }
    multipliers[name] = value
  }
  return { multipliers, error: '' }
}

// --- 价目表 ---------------------------------------------------------------

export function normalizePricingItem(raw: unknown): PricingItemDto | null {
  if (!raw || typeof raw !== 'object') return null
  const source = raw as Record<string, unknown>
  const capability = asString(source.capability).trim().toLowerCase()
  if (!capability) return null
  const multipliers: Record<string, number> = {}
  const rawMultipliers = source.resolution_multipliers
  if (rawMultipliers && typeof rawMultipliers === 'object') {
    for (const [key, value] of Object.entries(rawMultipliers as Record<string, unknown>)) {
      const name = String(key).trim()
      const micro = asNumber(value, -1)
      if (!name || !Number.isFinite(micro) || micro < 0) continue
      multipliers[name] = Math.round(micro)
    }
  }
  const unitPrice = source.unit_price_micro
  return {
    id: asText(source.id, capability + ':' + asString(source.provider) + ':' + asString(source.model)),
    capability,
    provider: asString(source.provider),
    model: asString(source.model),
    currency: asText(source.currency, 'CNY'),
    unit_price_micro: unitPrice === null || unitPrice === undefined ? null : Math.round(asNumber(unitPrice, 0)),
    unit_price_secondary_micro:
      source.unit_price_secondary_micro === null || source.unit_price_secondary_micro === undefined
        ? null
        : Math.round(asNumber(source.unit_price_secondary_micro, 0)),
    resolution_multipliers: multipliers,
    configured: asBoolean(source.configured, unitPrice !== null && unitPrice !== undefined),
    note: asString(source.note),
    updated_at: typeof source.updated_at === 'string' && source.updated_at ? source.updated_at : null,
  }
}

export function normalizePricingTable(raw: unknown): PricingTableDto {
  const source = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const capabilities: PricingCapabilityDto[] = []
  if (Array.isArray(source.capabilities)) {
    for (const item of source.capabilities) {
      if (!item || typeof item !== 'object') continue
      const entry = item as Record<string, unknown>
      const capability = asString(entry.capability).trim().toLowerCase()
      if (!capability) continue
      const items: PricingItemDto[] = []
      if (Array.isArray(entry.items)) {
        for (const row of entry.items) {
          // 价目行允许省略 capability（跟随所属能力分组），归一化时补齐。
          const source =
            row && typeof row === 'object' && !asString((row as Record<string, unknown>).capability)
              ? { ...(row as Record<string, unknown>), capability }
              : row
          const normalized = normalizePricingItem(source)
          if (normalized) items.push({ ...normalized, capability })
        }
      }
      capabilities.push({
        capability,
        label: asText(entry.label, capability),
        base_unit: asString(entry.base_unit),
        pricing_unit: asText(entry.pricing_unit, '每单位'),
        unit_scale: Math.max(1, Math.round(asNumber(entry.unit_scale, 1))),
        secondary_unit: typeof entry.secondary_unit === 'string' && entry.secondary_unit ? entry.secondary_unit : null,
        items,
      })
    }
  }
  return {
    currency: asText(source.currency, 'CNY'),
    micro_per_unit: Math.round(asNumber(source.micro_per_unit, MICRO_PER_UNIT)),
    rounding: asString(source.rounding),
    capabilities,
  }
}

/** 单价展示：每 100 万 tokens / 每张 / 每秒 / 每 1000 字符 / 每分钟编码。 */
export function pricingUnitLabel(capability: PricingCapabilityDto | null | undefined): string {
  if (!capability) return '每单位'
  return capability.pricing_unit || '每单位'
}

// --- 预算配置 / 状态 -------------------------------------------------------

export function normalizeBudgetConfigRow(raw: unknown): BudgetConfigRowDto | null {
  if (!raw || typeof raw !== 'object') return null
  const source = raw as Record<string, unknown>
  const microOrNull = (value: unknown): number | null =>
    value === null || value === undefined ? null : Math.round(asNumber(value, 0))
  return {
    scope_type: asString(source.scope_type) as BudgetConfigRowDto['scope_type'],
    scope_id: asString(source.scope_id),
    currency: asText(source.currency, 'CNY'),
    soft_cost_micro: microOrNull(source.soft_cost_micro),
    hard_cost_micro: microOrNull(source.hard_cost_micro),
    soft_seconds: microOrNull(source.soft_seconds),
    hard_seconds: microOrNull(source.hard_seconds),
    enabled: asBoolean(source.enabled, true),
    note: asString(source.note),
  }
}

export function normalizeEffectiveBudget(raw: unknown): EffectiveBudgetDto {
  const row = normalizeBudgetConfigRow(raw)
  const source = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const origin = asString(source.source, 'none') as BudgetSource
  const base: BudgetConfigRowDto = row || {
    scope_type: '',
    scope_id: '',
    currency: 'CNY',
    soft_cost_micro: null,
    hard_cost_micro: null,
    soft_seconds: null,
    hard_seconds: null,
    enabled: false,
    note: '',
  }
  return {
    ...base,
    source: origin === 'project' || origin === 'global' ? origin : 'none',
    source_label: asText(source.source_label, origin === 'none' ? '未设置预算' : '预算'),
  }
}

export function normalizeBudgetConfig(raw: unknown): BudgetConfigDto {
  const source = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  return {
    project_id: asString(source.project_id),
    series_id: asString(source.series_id),
    global: normalizeBudgetConfigRow(source.global),
    project: normalizeBudgetConfigRow(source.project),
    series: normalizeBudgetConfigRow(source.series),
    effective: normalizeEffectiveBudget(source.effective),
  }
}

const BUDGET_LEVELS: BudgetLevel[] = ['unlimited', 'ok', 'soft_exceeded', 'hard_exceeded']

export function normalizeBudgetState(raw: unknown): BudgetStateDto {
  const source = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const level = asString(source.level, 'ok') as BudgetLevel
  const reason = asString(source.reason) as BudgetReason
  const usage = (source.usage && typeof source.usage === 'object' ? source.usage : {}) as Record<string, unknown>
  const microOrNull = (value: unknown): number | null =>
    value === null || value === undefined ? null : Math.round(asNumber(value, 0))
  const origin = asString(source.budget_source, 'none') as BudgetSource
  return {
    level: BUDGET_LEVELS.indexOf(level) >= 0 ? level : 'ok',
    code: asString(source.code),
    message: asString(source.message),
    reason: reason === 'cost' || reason === 'seconds' ? reason : '',
    unlimited: asBoolean(source.unlimited, level === 'unlimited'),
    currency: asText(source.currency, 'CNY'),
    cost_known: asBoolean(source.cost_known, true),
    soft_cost_micro: microOrNull(source.soft_cost_micro),
    hard_cost_micro: microOrNull(source.hard_cost_micro),
    soft_seconds: microOrNull(source.soft_seconds),
    hard_seconds: microOrNull(source.hard_seconds),
    used_cost_micro: asCount(source.used_cost_micro),
    reserved_cost_micro: asCount(source.reserved_cost_micro),
    committed_cost_micro: asCount(source.committed_cost_micro),
    projected_cost_micro: asCount(source.projected_cost_micro),
    used_seconds: asCount(source.used_seconds),
    reserved_seconds: asCount(source.reserved_seconds),
    projected_seconds: asCount(source.projected_seconds),
    estimate_cost_micro: microOrNull(source.estimate_cost_micro),
    estimate_seconds: microOrNull(source.estimate_seconds),
    budget_source: origin === 'project' || origin === 'global' ? origin : 'none',
    budget_source_label: asText(source.budget_source_label, '未设置预算'),
    usage: {
      call_count: asCount(usage.call_count),
      unknown_call_count: asCount(usage.unknown_call_count),
      failed_call_count: asCount(usage.failed_call_count),
      cost_known: asBoolean(usage.cost_known, true),
    },
    reservation_count: asCount(source.reservation_count),
  }
}

export function budgetStateIsUnlimited(state: BudgetStateDto | null | undefined): boolean {
  if (!state) return true
  return state.level === 'unlimited' && state.soft_cost_micro === null && state.hard_cost_micro === null
}

// --- 估算 -----------------------------------------------------------------

export function normalizeEstimateComponent(raw: unknown): EstimateComponentDto | null {
  if (!raw || typeof raw !== 'object') return null
  const source = raw as Record<string, unknown>
  const capability = asString(source.capability).trim().toLowerCase() as BudgetCapability
  if (!capability) return null
  const micro = source.cost_micro
  return {
    capability,
    label: asText(source.label, capability),
    component_label: asString(source.component_label),
    provider: asString(source.provider),
    model: asString(source.model),
    quantity: asCount(source.quantity),
    secondary_quantity: asCount(source.secondary_quantity),
    base_unit: asString(source.base_unit),
    resolution: asString(source.resolution),
    calls: asCount(source.calls),
    cost_micro: micro === null || micro === undefined ? null : Math.round(asNumber(micro, 0)),
    cost_known: asBoolean(source.cost_known, micro !== null && micro !== undefined),
    estimated_seconds: asCount(source.estimated_seconds),
  }
}

export function normalizeUnknownComponent(raw: unknown): EstimateUnknownComponentDto | null {
  if (!raw || typeof raw !== 'object') return null
  const source = raw as Record<string, unknown>
  const capability = asString(source.capability).trim().toLowerCase() as BudgetCapability
  if (!capability) return null
  return {
    capability,
    label: asText(source.label, capability),
    provider: asString(source.provider),
    model: asString(source.model),
    quantity: asCount(source.quantity),
    reason: asText(source.reason, DEFAULT_UNKNOWN_PRICE_REASON),
  }
}

export function normalizeEstimate(raw: unknown): TaskEstimateDto | null {
  if (!raw || typeof raw !== 'object') return null
  const source = raw as Record<string, unknown>
  const components: EstimateComponentDto[] = []
  if (Array.isArray(source.components)) {
    for (const item of source.components) {
      const normalized = normalizeEstimateComponent(item)
      if (normalized) components.push(normalized)
    }
  }
  const unknownComponents: EstimateUnknownComponentDto[] = []
  if (Array.isArray(source.unknown_components)) {
    for (const item of source.unknown_components) {
      const normalized = normalizeUnknownComponent(item)
      if (normalized) unknownComponents.push(normalized)
    }
  }
  const micro = source.estimated_cost_micro
  const seconds = source.estimated_seconds
  const durationSource = asString(source.duration_source, 'unknown') as DurationSource
  const costKnown = asBoolean(source.cost_known, micro !== null && micro !== undefined)
  return {
    job_type: asString(source.job_type),
    job_type_label: asText(source.job_type_label, asString(source.job_type)),
    project_id: asString(source.project_id),
    shot_id: asString(source.shot_id),
    currency: asText(source.currency, 'CNY'),
    estimated_cost_micro: costKnown && micro !== null && micro !== undefined ? Math.round(asNumber(micro, 0)) : null,
    cost_known: costKnown && micro !== null && micro !== undefined,
    estimated_seconds: seconds === null || seconds === undefined ? null : asCount(seconds),
    duration_source:
      durationSource === 'history' || durationSource === 'heuristic' || durationSource === 'unknown'
        ? durationSource
        : 'unknown',
    components,
    unknown_components: unknownComponents,
    note: asString(source.note),
  }
}

/** 未知成本的统一原因：优先用后端给出的具体原因。 */
export function unknownCostReason(unknown: EstimateUnknownComponentDto[] | null | undefined): string {
  const reasons = (unknown || []).map((item) => asString(item?.reason).trim()).filter(Boolean)
  return reasons.length > 0 ? reasons[0] : DEFAULT_UNKNOWN_PRICE_REASON
}

/** 预计成本文案；成本未知时明确写出原因，绝不显示 ¥0。 */
export function estimateCostText(estimate: TaskEstimateDto | null | undefined): string {
  if (!estimate) return UNKNOWN_COST_TEXT
  if (!estimate.cost_known || estimate.estimated_cost_micro === null) {
    return UNKNOWN_COST_TEXT + '（' + unknownCostReason(estimate.unknown_components) + '）'
  }
  return formatMicroAmount(estimate.estimated_cost_micro, estimate.currency)
}

/** 估算来源文案：history / heuristic / unknown。 */
export function durationSourceLabel(source: DurationSource | string | null | undefined): string {
  if (source === 'history') return HISTORY_DURATION_TEXT
  if (source === 'heuristic') return HEURISTIC_DURATION_TEXT
  return UNKNOWN_DURATION_TEXT
}

/** 预计耗时文案（含来源说明）。 */
export function estimateSecondsText(estimate: TaskEstimateDto | null | undefined): string {
  if (!estimate || estimate.estimated_seconds === null) return UNKNOWN_DURATION_TEXT
  return formatDurationText(estimate.estimated_seconds) + '（' + durationSourceLabel(estimate.duration_source) + '）'
}

/** 逐项拆解文案：能力 / 数量 / 金额（未知时写原因）。 */
export function componentCostText(component: EstimateComponentDto): string {
  if (!component.cost_known || component.cost_micro === null) return UNKNOWN_COST_TEXT
  return formatMicroAmount(component.cost_micro, 'CNY')
}

export function componentQuantityText(component: EstimateComponentDto): string {
  const unit = component.base_unit || '单位'
  if (component.secondary_quantity > 0) {
    return component.quantity + ' ' + unit + ' + ' + component.secondary_quantity + ' 输出 token'
  }
  return component.quantity + ' ' + unit
}

// --- 用量汇总 -------------------------------------------------------------

export function normalizeCapabilityUsage(raw: unknown): CapabilityUsageDto {
  const source = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const capability = asString(source.capability).trim().toLowerCase() as BudgetCapability
  return {
    capability,
    label: asText(source.label, capability || '未知能力'),
    call_count: asCount(source.call_count),
    cost_micro: asCount(source.cost_micro),
    cost_known: asBoolean(source.cost_known, true),
    unknown_call_count: asCount(source.unknown_call_count),
    quantity: asCount(source.quantity),
    secondary_quantity: asCount(source.secondary_quantity),
    seconds: asCount(source.seconds),
    base_unit: asString(source.base_unit),
    currency: asText(source.currency, 'CNY'),
  }
}

export function normalizeUsageSummary(raw: unknown): UsageSummaryDto {
  const source = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const byCapability: CapabilityUsageDto[] = []
  if (Array.isArray(source.by_capability)) {
    for (const item of source.by_capability) byCapability.push(normalizeCapabilityUsage(item))
  }
  return {
    call_count: asCount(source.call_count),
    cost_micro: asCount(source.cost_micro),
    cost_known: asBoolean(source.cost_known, true),
    unknown_call_count: asCount(source.unknown_call_count),
    failed_call_count: asCount(source.failed_call_count),
    duration_ms: asCount(source.duration_ms),
    currency: asText(source.currency, 'CNY'),
    by_capability: byCapability,
  }
}

export function normalizeUsageGroup(raw: unknown): UsageGroupDto | null {
  if (!raw || typeof raw !== 'object') return null
  const source = raw as Record<string, unknown>
  return {
    key: asString(source.key),
    call_count: asCount(source.call_count),
    cost_micro: asCount(source.cost_micro),
    cost_known: asBoolean(source.cost_known, true),
    unknown_call_count: asCount(source.unknown_call_count),
    currency: asText(source.currency, 'CNY'),
  }
}

export function normalizeUsageRecord(raw: unknown): UsageRecordDto | null {
  if (!raw || typeof raw !== 'object') return null
  const source = raw as Record<string, unknown>
  const id = asString(source.id) || asString(source.usage_key)
  if (!id) return null
  const micro = source.cost_micro
  const costSource = asString(source.cost_source, 'unknown') as CostSource
  return {
    id,
    job_key: asString(source.job_key),
    job_id: asString(source.job_id),
    job_type: asString(source.job_type),
    job_status: asString(source.job_status),
    project_id: asString(source.project_id),
    series_id: asString(source.series_id),
    shot_id: asString(source.shot_id),
    capability: asString(source.capability).trim().toLowerCase(),
    capability_label: asText(source.capability_label, asString(source.capability)),
    provider: asString(source.provider),
    model: asString(source.model),
    quantity: asCount(source.quantity),
    secondary_quantity: asCount(source.secondary_quantity),
    base_unit: asString(source.base_unit),
    resolution: asString(source.resolution),
    units: (source.units && typeof source.units === 'object' ? source.units : {}) as Record<string, unknown>,
    status: asString(source.status, 'unknown'),
    error_code: asString(source.error_code),
    cost_micro: micro === null || micro === undefined ? null : Math.round(asNumber(micro, 0)),
    cost_known: asBoolean(source.cost_known, micro !== null && micro !== undefined),
    cost_source: costSource === 'pricing' || costSource === 'local' ? costSource : 'unknown',
    currency: asText(source.currency, 'CNY'),
    duration_ms: asCount(source.duration_ms),
    created_at: typeof source.created_at === 'string' && source.created_at ? source.created_at : null,
  }
}

export function normalizeUsageRecordPage(raw: unknown): UsageRecordPageDto {
  const source = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const items: UsageRecordDto[] = []
  if (Array.isArray(source.items)) {
    for (const item of source.items) {
      const normalized = normalizeUsageRecord(item)
      if (normalized) items.push(normalized)
    }
  }
  return {
    items,
    total: asCount(source.total),
    page: Math.max(1, asCount(source.page) || 1),
    page_size: Math.max(1, asCount(source.page_size) || 20),
    pages: Math.max(1, asCount(source.pages) || 1),
  }
}

export function normalizeJobCostDetail(raw: unknown): JobCostDetailDto | null {
  if (!raw || typeof raw !== 'object') return null
  const source = raw as Record<string, unknown>
  const jobId = asString(source.job_id)
  if (!jobId) return null
  return {
    job_id: jobId,
    job_key: asString(source.job_key),
    job_type: asString(source.job_type),
    status: asString(source.status),
    duration_seconds: asCount(source.duration_seconds),
    summary: normalizeUsageSummary(source.summary),
    estimate: normalizeEstimate(source.estimate),
    records: normalizeUsageRecordPage(source.records),
  }
}

// --- 任务 DTO 内嵌成本块 ---------------------------------------------------

/** 缺 `cost` 字段的老数据归一化成「暂无用量」，而不是伪造 ¥0。 */
export function normalizeJobCost(raw: unknown): JobCostDto {
  const source = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const micro = source.cost_micro
  const byCapability: JobCostDto['by_capability'] = []
  if (Array.isArray(source.by_capability)) {
    for (const item of source.by_capability) {
      if (!item || typeof item !== 'object') continue
      const entry = item as Record<string, unknown>
      byCapability.push({
        capability: asString(entry.capability).trim().toLowerCase(),
        call_count: asCount(entry.call_count),
        cost_micro: asCount(entry.cost_micro),
        cost_known: asBoolean(entry.cost_known, true),
      })
    }
  }
  const estimated = source.estimated_cost_micro
  const estimatedSeconds = source.estimated_seconds
  const durationSource = asString(source.duration_source)
  return {
    currency: asText(source.currency, 'CNY'),
    cost_micro: micro === null || micro === undefined ? null : Math.round(asNumber(micro, 0)),
    cost_known: asBoolean(source.cost_known, micro !== null && micro !== undefined),
    call_count: asCount(source.call_count),
    unknown_call_count: asCount(source.unknown_call_count),
    failed_call_count: asCount(source.failed_call_count),
    provider_seconds: asCount(source.provider_seconds),
    by_capability: byCapability,
    estimated_cost_micro: estimated === null || estimated === undefined ? null : Math.round(asNumber(estimated, 0)),
    estimated_cost_known: asBoolean(source.estimated_cost_known, estimated !== null && estimated !== undefined),
    estimated_seconds: estimatedSeconds === null || estimatedSeconds === undefined ? null : asCount(estimatedSeconds),
    duration_source:
      durationSource === 'history' || durationSource === 'heuristic' || durationSource === 'unknown'
        ? (durationSource as DurationSource)
        : '',
    has_usage: asBoolean(source.has_usage, asCount(source.call_count) > 0),
  }
}

export function normalizeJobStatsCost(raw: unknown): JobStatsCostDto {
  const source = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  return {
    currency: asText(source.currency, 'CNY'),
    cost_micro: asCount(source.cost_micro),
    cost_known: asBoolean(source.cost_known, true),
    unknown_call_count: asCount(source.unknown_call_count),
    call_count: asCount(source.call_count),
  }
}

/** 任务是否记录过「启动前估算」。 */
export function jobCostHasEstimate(cost: JobCostDto | null | undefined): boolean {
  if (!cost) return false
  return (
    cost.estimated_cost_micro !== null ||
    cost.estimated_cost_known ||
    cost.estimated_seconds !== null ||
    Boolean(cost.duration_source)
  )
}

/** 单次任务实际成本文案：无用量 -> 暂无用量；有调用但单价缺失 -> 成本未知。 */
export function jobCostText(cost: JobCostDto | null | undefined): string {
  if (!cost) return NO_USAGE_TEXT
  if (!cost.cost_known || cost.cost_micro === null) {
    if (cost.call_count === 0 && !cost.has_usage) return NO_USAGE_TEXT
    return UNKNOWN_COST_TEXT
  }
  return formatMicroAmount(cost.cost_micro, cost.currency)
}

/** 列表行文案：「预计 ¥x · 实际 ¥y」，没有估算时只显示实际值。 */
export function jobCostComparisonText(cost: JobCostDto | null | undefined): string {
  const actual = jobCostText(cost)
  if (!cost || !jobCostHasEstimate(cost)) return actual
  const expected = formatCostValue(cost.estimated_cost_micro, cost.estimated_cost_known, cost.currency)
  return '预计 ' + expected + ' · 实际 ' + actual
}

/** 耗时对比文案：「预计 2 分 00 秒 · 实际 1 分 35 秒」。 */
export function jobDurationComparisonText(
  durationSeconds: number | null | undefined,
  estimatedSeconds: number | null | undefined,
): string {
  const actual = formatDurationText(durationSeconds)
  if (estimatedSeconds === null || estimatedSeconds === undefined) return actual
  return '预计 ' + formatDurationText(estimatedSeconds) + ' · 实际 ' + actual
}
