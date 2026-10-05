import assert from 'node:assert/strict'

import {
  DEFAULT_UNKNOWN_PRICE_REASON,
  NO_USAGE_TEXT,
  UNKNOWN_COST_TEXT,
  amountTextToMicro,
  budgetBadgeForState,
  budgetBlockedFromError,
  budgetLevelBadge,
  budgetUsedPercent,
  budgetWarningFromResponse,
  componentQuantityText,
  describeBudgetError,
  durationSourceLabel,
  estimateCostText,
  estimateSecondsText,
  formatCostValue,
  formatDurationText,
  formatMicroAmount,
  formatMillisecondsText,
  jobCostComparisonText,
  jobCostHasEstimate,
  jobCostText,
  jobDurationComparisonText,
  microToAmountText,
  microToMultiplierText,
  multipliersFromText,
  multipliersToText,
  normalizeBudgetState,
  normalizeBudgetSummary,
  normalizeEstimate,
  normalizeJobCost,
  normalizeJobCostDetail,
  normalizeJobStatsCost,
  normalizePricingTable,
  providerBlockedFromError,
  usedCostText,
  unknownCostReason,
} from './costModel.ts'
import { normalizeJob } from '../components/taskCenterModel.ts'

// --- 1. micro -> 金额格式化 -------------------------------------------------

assert.equal(formatMicroAmount(null, 'CNY'), UNKNOWN_COST_TEXT, 'null 金额必须是「成本未知」')
assert.equal(formatMicroAmount(undefined, 'CNY'), UNKNOWN_COST_TEXT, 'undefined 金额必须是「成本未知」')
assert.equal(formatMicroAmount(Number.NaN, 'CNY'), UNKNOWN_COST_TEXT, 'NaN 必须兜底成「成本未知」')
assert.equal(formatMicroAmount(Number.POSITIVE_INFINITY, 'CNY'), UNKNOWN_COST_TEXT, 'Infinity 必须兜底')
assert.equal(formatMicroAmount(-1, 'CNY'), UNKNOWN_COST_TEXT, '负数金额非法，必须兜底成「成本未知」')
assert.equal(formatMicroAmount('12' as unknown as number, 'CNY'), UNKNOWN_COST_TEXT, '非数字输入必须兜底')

assert.equal(formatMicroAmount(0, 'CNY'), '¥0.00', '0 是已知的 0 元，必须如实显示')
assert.equal(formatMicroAmount(1500000, 'CNY'), '¥1.50', '1_500_000 micro = 1.50 元')
assert.equal(formatMicroAmount(1000000, 'CNY'), '¥1.00')
assert.equal(formatMicroAmount(100000, 'CNY'), '¥0.10')
assert.equal(formatMicroAmount(1, 'CNY'), '¥0.000001', '1 micro 必须精确展示，不能四舍五入成 0')
assert.equal(formatMicroAmount(1234567, 'CNY'), '¥1.234567', '整数 micro 必须分毫不差')
assert.equal(formatMicroAmount(1234567890, 'CNY'), '¥1,234.56789', '大额金额加千分位')
assert.equal(formatMicroAmount(2000000, 'USD'), '$2.00', '非人民币币种用对应符号')
assert.equal(formatMicroAmount(2000000, 'XYZ'), 'XYZ 2.00', '未知币种回落到币种代码')

assert.equal(formatCostValue(0, true, 'CNY'), '¥0.00')
assert.equal(
  formatCostValue(0, false, 'CNY'),
  UNKNOWN_COST_TEXT,
  'cost_known=false 时即使金额是 0 也必须显示「成本未知」',
)
assert.equal(formatCostValue(null, true, 'CNY'), UNKNOWN_COST_TEXT, '已知标记为真但金额缺失时同样显示「成本未知」')
assert.equal(formatCostValue(1500000, false, 'CNY'), UNKNOWN_COST_TEXT)

// --- 金额输入：字符串换算，禁止浮点落库 ------------------------------------

assert.equal(amountTextToMicro('0.07'), 70000, '0.07 元必须精确换算成 70000 micro，不能有浮点尾巴')
assert.equal(amountTextToMicro('1.5'), 1500000)
assert.equal(amountTextToMicro('1.50'), 1500000)
assert.equal(amountTextToMicro('0'), 0)
assert.equal(amountTextToMicro('1.234567'), 1234567)
assert.equal(amountTextToMicro('1.2345678'), null, '超过 6 位小数必须拒绝，而不是悄悄截断')
assert.equal(amountTextToMicro('abc'), null)
assert.equal(amountTextToMicro('-1'), null, '负数金额必须拒绝')
assert.equal(amountTextToMicro(''), null)
assert.equal(amountTextToMicro('1e3'), null, '科学计数法必须拒绝')
assert.equal(microToAmountText(70000), '0.07')
assert.equal(microToAmountText(1500000), '1.50')
assert.equal(microToAmountText(1234567), '1.234567')
assert.equal(microToAmountText(null), '', '未配置单价时输入框留空')
assert.equal(amountTextToMicro(microToAmountText(987654)), 987654, '往返换算必须无损')

// --- 耗时格式化 ------------------------------------------------------------

assert.equal(formatDurationText(0), '0 秒')
assert.equal(formatDurationText(59), '59 秒')
assert.equal(formatDurationText(95), '1 分 35 秒')
assert.equal(formatDurationText(3725), '1 小时 02 分')
assert.equal(formatDurationText(null), '—', '未知耗时不显示 0')
assert.equal(formatDurationText(Number.NaN), '—')
assert.equal(formatMillisecondsText(95000), '1 分 35 秒')
assert.equal(formatMillisecondsText(null), '—')

// --- 2. 未知成本：任何展示路径都不能出现 ¥0 -------------------------------

const unknownCostSources = [
  formatMicroAmount(null, 'CNY'),
  formatCostValue(null, false, 'CNY'),
  formatCostValue(0, false, 'CNY'),
  usedCostText(0, false, 'CNY'),
  estimateCostText(normalizeEstimate({ estimated_cost_micro: null, cost_known: false })),
  jobCostText(normalizeJobCost({ cost_micro: null, cost_known: false, call_count: 2, has_usage: true })),
]
for (const text of unknownCostSources) {
  assert.equal(text.includes('¥0'), false, '未知成本的展示路径不能出现 ¥0：' + text)
  assert.equal(text.includes('¥0.00'), false, '未知成本的展示路径不能出现 ¥0.00：' + text)
}
assert.equal(usedCostText(0, false, 'CNY'), UNKNOWN_COST_TEXT)
assert.equal(usedCostText(null, true, 'CNY'), UNKNOWN_COST_TEXT)
assert.equal(usedCostText(1500000, true, 'CNY'), '¥1.50')
assert.equal(
  usedCostText(1500000, false, 'CNY'),
  UNKNOWN_COST_TEXT + '（已知部分 ¥1.50）',
  '部分调用成本未知时必须说明已知部分，而不是当成总额',
)
assert.equal(unknownCostReason([]), DEFAULT_UNKNOWN_PRICE_REASON)
assert.equal(
  unknownCostReason([
    {
      capability: 'image',
      label: '图像生成',
      provider: 'ark-seedream',
      model: '',
      quantity: 3,
      reason: '未配置该 provider / 模型的单价',
    },
  ]),
  '未配置该 provider / 模型的单价',
  '后端给出的未知原因应优先展示',
)

// normalizeEstimate：cost_known=false 时金额必须被强制为 null
const forcedUnknown = normalizeEstimate({ cost_known: false, estimated_cost_micro: 999999, currency: 'CNY' })
assert.equal(forcedUnknown?.estimated_cost_micro, null, 'cost_known=false 时估算金额必须为 null')
assert.equal(forcedUnknown?.cost_known, false)
assert.match(estimateCostText(forcedUnknown), /成本未知/, '未知成本文案必须出现「成本未知」')

// --- 3. 估算文案：duration_source 三种来源 --------------------------------

assert.equal(durationSourceLabel('history'), '按历史耗时估算')
assert.equal(durationSourceLabel('heuristic'), '按单位耗时模型估算')
assert.equal(durationSourceLabel('unknown'), '暂无法估算耗时')
assert.equal(durationSourceLabel(''), '暂无法估算耗时', '来源缺失时按未知处理')
assert.equal(durationSourceLabel(null), '暂无法估算耗时')

const historyEstimate = normalizeEstimate({
  job_type: 'storyboard',
  job_type_label: '分镜与素材',
  currency: 'CNY',
  estimated_cost_micro: 1800000,
  cost_known: true,
  estimated_seconds: 120,
  duration_source: 'history',
  components: [
    {
      capability: 'image',
      label: '图像生成',
      component_label: '待生成故事板 3 个镜头',
      quantity: 3,
      base_unit: '张',
      calls: 3,
      cost_micro: 1800000,
      cost_known: true,
      estimated_seconds: 120,
    },
  ],
  unknown_components: [],
})
assert.equal(estimateCostText(historyEstimate), '¥1.80')
assert.equal(estimateSecondsText(historyEstimate), '2 分 00 秒（按历史耗时估算）')
assert.equal(
  estimateSecondsText(normalizeEstimate({ estimated_seconds: 65, duration_source: 'heuristic' })),
  '1 分 05 秒（按单位耗时模型估算）',
)
assert.equal(
  estimateSecondsText(normalizeEstimate({ estimated_seconds: null, duration_source: 'unknown' })),
  '暂无法估算耗时',
)
assert.equal(estimateSecondsText(null), '暂无法估算耗时')
assert.equal(historyEstimate?.components[0] ? componentQuantityText(historyEstimate.components[0]) : '', '3 张')

// --- 4. 预算状态 -> 徽标 --------------------------------------------------

assert.deepEqual(budgetLevelBadge('unlimited'), {
  level: 'unlimited',
  label: '未设置预算',
  tone: 'muted',
  color: 'var(--text-secondary)',
})
assert.deepEqual(budgetLevelBadge('ok'), { level: 'ok', label: '预算正常', tone: 'success', color: 'var(--green)' })
assert.deepEqual(budgetLevelBadge('soft_exceeded'), {
  level: 'soft_exceeded',
  label: '已超软预算',
  tone: 'warning',
  color: 'var(--amber)',
})
assert.deepEqual(budgetLevelBadge('hard_exceeded'), {
  level: 'hard_exceeded',
  label: '已超硬预算',
  tone: 'danger',
  color: 'var(--red)',
})
assert.equal(budgetLevelBadge(null).label, '预算正常', '缺状态时按正常处理，不谎报超支')
assert.equal(budgetBadgeForState(null).level, 'unlimited')

const okState = normalizeBudgetState({ level: 'ok', soft_cost_micro: 10000000, committed_cost_micro: 1000000 })
assert.equal(budgetBadgeForState(okState).level, 'ok')
const nearState = normalizeBudgetState({ level: 'ok', soft_cost_micro: 10000000, committed_cost_micro: 8500000 })
assert.equal(budgetBadgeForState(nearState).level, 'near_soft', '已用超过软预算 80% 时提示「接近软预算」')
assert.equal(budgetBadgeForState(nearState).label, '接近软预算')
assert.equal(budgetBadgeForState(normalizeBudgetState({ level: 'soft_exceeded' })).tone, 'warning')
assert.equal(budgetBadgeForState(normalizeBudgetState({ level: 'hard_exceeded' })).tone, 'danger')
assert.equal(budgetUsedPercent(normalizeBudgetState({ soft_cost_micro: 10000000, committed_cost_micro: 2500000 })), 25)
assert.equal(budgetUsedPercent(normalizeBudgetState({ soft_cost_micro: null, hard_cost_micro: null })), null)

// --- 5. 任务成本模型归一化 ------------------------------------------------

// 老数据：完全没有 cost 字段
const legacyJob = normalizeJob({ id: 'job-legacy', status: 'completed', duration_seconds: 42 })
assert.ok(legacyJob)
assert.equal(legacyJob?.cost.cost_known, false)
assert.equal(legacyJob?.cost.cost_micro, null, '缺字段时金额必须是 null，不能补 0')
assert.equal(legacyJob?.cost.call_count, 0)
assert.equal(legacyJob?.cost.has_usage, false)
assert.equal(legacyJob?.cost.estimated_cost_micro, null)
assert.equal(legacyJob?.cost.estimated_seconds, null)
assert.deepEqual(legacyJob?.cost.by_capability, [])
assert.equal(jobCostText(legacyJob?.cost), NO_USAGE_TEXT, '没有用量记录显示「暂无用量」而不是 ¥0')
assert.equal(jobCostComparisonText(legacyJob?.cost), NO_USAGE_TEXT)
assert.equal(jobCostHasEstimate(legacyJob?.cost), false)

// 有用量但单价缺失：成本未知，绝不显示 ¥0
const unknownJob = normalizeJob({
  id: 'job-unknown',
  status: 'failed',
  duration_seconds: 95,
  cost: {
    currency: 'CNY',
    cost_micro: null,
    cost_known: false,
    call_count: 4,
    unknown_call_count: 4,
    failed_call_count: 1,
    provider_seconds: 88,
    has_usage: true,
    estimated_seconds: 120,
    duration_source: 'history',
  },
})
assert.equal(jobCostText(unknownJob?.cost), UNKNOWN_COST_TEXT)
assert.equal(jobCostText(unknownJob?.cost).includes('¥'), false, '未知成本不能出现任何金额符号')
assert.equal(
  jobCostComparisonText(unknownJob?.cost),
  '预计 ' + UNKNOWN_COST_TEXT + ' · 实际 ' + UNKNOWN_COST_TEXT,
  '有估算但估算价格缺失时，两侧都必须写「成本未知」',
)
assert.equal(jobDurationComparisonText(95, 120), '预计 2 分 00 秒 · 实际 1 分 35 秒')
assert.equal(jobDurationComparisonText(95, null), '1 分 35 秒', '没有估算时只显示实际耗时')
assert.equal(jobCostHasEstimate(unknownJob?.cost), true, '有估算记录时才能显示对比')

// 正常数据：预计 / 实际对比
const knownJob = normalizeJob({
  id: 'job-known',
  status: 'completed',
  duration_seconds: 95,
  cost: {
    currency: 'CNY',
    cost_micro: 1500000,
    cost_known: true,
    call_count: 3,
    unknown_call_count: 0,
    failed_call_count: 0,
    provider_seconds: 80,
    by_capability: [{ capability: 'video', call_count: 1, cost_micro: 1500000, cost_known: true }],
    estimated_cost_micro: 1800000,
    estimated_cost_known: true,
    estimated_seconds: 120,
    duration_source: 'heuristic',
    has_usage: true,
  },
})
assert.equal(jobCostText(knownJob?.cost), '¥1.50')
assert.equal(jobCostComparisonText(knownJob?.cost), '预计 ¥1.80 · 实际 ¥1.50')
assert.equal(knownJob?.cost.by_capability[0].capability, 'video')
assert.equal(knownJob?.cost.provider_seconds, 80)
assert.equal(normalizeJobCost(null).cost_micro, null, 'raw 为 null 时归一化必须安全')
assert.equal(
  normalizeJobCost({ cost_micro: 12345, cost_known: false }).cost_micro,
  12345,
  '金额保留，但 cost_known=false 时展示层会写成「成本未知」',
)
assert.equal(
  jobCostText(normalizeJobCost({ cost_micro: 12345, cost_known: false, has_usage: true })),
  UNKNOWN_COST_TEXT,
)

// --- 统计与明细响应归一化 --------------------------------------------------

assert.equal(normalizeJobStatsCost(undefined).cost_known, true)
assert.equal(normalizeJobStatsCost(undefined).cost_micro, 0)
assert.equal(
  normalizeJobStatsCost({
    currency: 'CNY',
    cost_micro: 2500000,
    cost_known: true,
    unknown_call_count: 2,
    call_count: 5,
  }).call_count,
  5,
)

const detail = normalizeJobCostDetail({
  job_id: 'job-1',
  job_key: 'render:p1',
  job_type: 'render',
  status: 'failed',
  duration_seconds: 30,
  summary: {
    call_count: 2,
    cost_micro: 0,
    cost_known: false,
    unknown_call_count: 2,
    failed_call_count: 1,
    duration_ms: 30000,
    currency: 'CNY',
    by_capability: [],
  },
  estimate: null,
  records: {
    items: [{ id: 'u1', capability: 'ffmpeg', quantity: 60, cost_micro: null, cost_known: false, status: 'failed' }],
    total: 1,
    page: 1,
    page_size: 20,
    pages: 1,
  },
})
assert.equal(detail?.job_id, 'job-1')
assert.equal(detail?.summary.cost_known, false)
assert.equal(detail?.records.items[0].cost_known, false)
assert.equal(
  formatCostValue(detail?.records.items[0].cost_micro, false, 'CNY'),
  UNKNOWN_COST_TEXT,
  '失败任务的调用同样不能显示 ¥0',
)
assert.equal(normalizeJobCostDetail({ job_key: 'x' }), null, '缺少 job_id 的明细必须丢弃')

// --- 项目 / 剧集汇总 ------------------------------------------------------

const summary = normalizeBudgetSummary({
  project_id: 'p1',
  currency: 'CNY',
  used: {
    cost_micro: 1500000,
    cost_known: true,
    unknown_call_count: 0,
    failed_call_count: 0,
    call_count: 2,
    seconds: 40,
    by_capability: [],
  },
  reserved: { cost_micro: 500000, seconds: 10, count: 1 },
  remaining: {
    currency: 'CNY',
    cost_micro: null,
    cost_known: false,
    partial_cost_micro: 300000,
    seconds: 60,
    components: [],
    unknown_components: [
      { capability: 'video', label: '视频生成', provider: '', model: '', quantity: 3, reason: '未配置该能力的单价' },
    ],
  },
  budget: {
    scope_type: 'project',
    scope_id: 'p1',
    source: 'project',
    source_label: '项目预算',
    currency: 'CNY',
    soft_cost_micro: 10000000,
    hard_cost_micro: 20000000,
    soft_seconds: null,
    hard_seconds: null,
    enabled: true,
    note: '',
  },
  status: { level: 'ok', currency: 'CNY', soft_cost_micro: 10000000, committed_cost_micro: 2000000 },
})
assert.equal(summary.used.cost_micro, 1500000)
assert.equal(summary.budget.source_label, '项目预算')
assert.equal(summary.remaining.cost_known, false)
assert.equal(unknownCostReason(summary.remaining.unknown_components), '未配置该能力的单价')
assert.equal(budgetBadgeForState(summary.status).level, 'ok')

// --- 价目表归一化 ----------------------------------------------------------

const pricing = normalizePricingTable({
  currency: 'CNY',
  micro_per_unit: 1000000,
  rounding: '整数 micro',
  capabilities: [
    {
      capability: 'llm',
      label: '文本模型',
      base_unit: 'token',
      pricing_unit: '每 100 万 tokens',
      unit_scale: 1000000,
      secondary_unit: '每 100 万输出 tokens',
      items: [
        {
          provider: 'mimo',
          model: 'mimo-v2.5',
          currency: 'CNY',
          unit_price_micro: 2000000,
          unit_price_secondary_micro: 8000000,
          resolution_multipliers: { default: 1000000 },
          configured: true,
          note: '',
        },
      ],
    },
  ],
})
assert.equal(pricing.capabilities[0].pricing_unit, '每 100 万 tokens')
assert.equal(pricing.capabilities[0].items[0].unit_price_micro, 2000000)
assert.equal(pricing.capabilities[0].items[0].resolution_multipliers.default, 1000000)
assert.equal(microToMultiplierText(1000000), '1.00')
assert.equal(multipliersToText({ '720p': 600000, default: 1000000 }), '720p=0.60，default=1.00')
const parsedMultipliers = multipliersFromText('1080p=1，720p=0.6')
assert.deepEqual(parsedMultipliers.multipliers, { '1080p': 1000000, '720p': 600000 })
assert.equal(parsedMultipliers.error, '')
assert.notEqual(multipliersFromText('1080p').error, '', '缺少等号的倍率必须给出中文提示')
assert.notEqual(multipliersFromText('1080p=abc').error, '')
assert.deepEqual(multipliersFromText('').multipliers, {})

// --- 提交响应里的预算提示 / 硬预算拦截 -------------------------------------

assert.equal(budgetWarningFromResponse({ ok: true }), '')
assert.equal(
  budgetWarningFromResponse({ budget_level: 'soft_exceeded', budget_warning: '项目软预算已超支，任务仍会继续执行' }),
  '项目软预算已超支，任务仍会继续执行',
)
assert.match(budgetWarningFromResponse({ budget_level: 'soft_exceeded' }), /软预算/)

const blockedError = {
  response: {
    status: 409,
    data: {
      detail: {
        ok: false,
        status: 'budget_blocked',
        error_code: 'budget_exceeded',
        message: '项目硬预算 ¥20.00 已不足以启动该任务',
        budget: { level: 'hard_exceeded', currency: 'CNY', hard_cost_micro: 20000000, committed_cost_micro: 25000000 },
      },
    },
  },
}
const blocked = budgetBlockedFromError(blockedError)
assert.equal(blocked?.status, 'budget_blocked')
assert.equal(blocked?.message, '项目硬预算 ¥20.00 已不足以启动该任务')
assert.equal(blocked?.budget?.level, 'hard_exceeded')
assert.equal(describeBudgetError(blockedError, '导出成片失败'), '项目硬预算 ¥20.00 已不足以启动该任务')
assert.equal(
  budgetBlockedFromError({ response: { status: 500, data: { detail: '内部错误' } } }),
  null,
  '普通失败不能被误判成预算拦截',
)
assert.equal(
  describeBudgetError({ response: { status: 400, data: { detail: '软预算不能高于硬预算' } } }),
  '软预算不能高于硬预算',
  'PUT 价目的中文原因要原样透出',
)
assert.equal(describeBudgetError({ response: { status: 500, data: 'boom' } }, '保存失败'), '保存失败（HTTP 500）')
assert.equal(describeBudgetError(new Error('Network Error'), '保存失败'), '保存失败（网络不可用）')
assert.match(describeBudgetError(new Error('boom'), '保存失败'), /保存失败/)

// --- 模型端点未配置拦截（provider_not_configured） ---------------------------

const providerBlockedError = {
  response: {
    status: 409,
    data: {
      detail: {
        ok: false,
        status: 'provider_not_configured',
        error_code: 'provider_not_configured',
        message: '以下模型端点尚未配置 API Key：视频生成模型、配音（TTS）。',
        missing: [
          { capability: 'video', label: '视频生成模型', message: '视频生成模型未配置 API Key' },
          { capability: 'voice', label: '配音（TTS）', message: '配音（TTS）未配置 API Key' },
          { capability: '', label: '坏数据要丢弃' },
        ],
      },
    },
  },
}
const providerBlocked = providerBlockedFromError(providerBlockedError)
assert.equal(providerBlocked?.status, 'provider_not_configured')
assert.equal(providerBlocked?.message, '以下模型端点尚未配置 API Key：视频生成模型、配音（TTS）。')
assert.equal(providerBlocked?.missing?.length, 2, '缺 capability 的缺失项必须丢弃')
assert.equal(providerBlocked?.missing?.[0]?.label, '视频生成模型')
assert.equal(
  providerBlockedFromError({ response: { status: 500, data: { detail: '内部错误' } } }),
  null,
  '普通失败不能被误判成配置拦截',
)
assert.equal(providerBlockedFromError(blockedError), null, '预算拦截不能被误判成配置拦截')

console.log('costModel.test.mts ok')
