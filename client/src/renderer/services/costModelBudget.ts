/**
 * 预算徽标与项目/剧集成本汇总的纯逻辑层。
 */

import type {
  BudgetLevel,
  BudgetStateDto,
  BudgetSummaryDto,
  CapabilityUsageDto,
  EstimateComponentDto,
  EstimateUnknownComponentDto,
  EpisodeCostDto,
  RemainingWorkloadDto,
  UsageGroupDto,
} from './costTypes'
import {
  asBoolean,
  asCount,
  asString,
  asText,
  asNumber,
  formatMicroAmount,
  normalizeBudgetState,
  normalizeCapabilityUsage,
  normalizeEffectiveBudget,
  normalizeEstimateComponent,
  normalizeUnknownComponent,
  normalizeUsageGroup,
  UNKNOWN_COST_TEXT,
} from './costModelCore'

// --- 预算徽标 -------------------------------------------------------------

export type BudgetBadgeTone = 'success' | 'warning' | 'danger' | 'muted'
/** `near_soft` 不是后端状态，而是「已用 >= 80% 软预算」的前端提示。 */
export type BudgetBadgeLevel = BudgetLevel | 'near_soft'

export interface BudgetBadge {
  level: BudgetBadgeLevel
  label: string
  tone: BudgetBadgeTone
  /** 设计系统色值（与 global.css 变量一致）。 */
  color: string
}

/** 接近软预算的判定阈值（80%）。 */
export const NEAR_SOFT_RATIO = 0.8

const BUDGET_BADGES: Record<BudgetBadgeLevel, BudgetBadge> = {
  unlimited: { level: 'unlimited', label: '未设置预算', tone: 'muted', color: 'var(--text-secondary)' },
  ok: { level: 'ok', label: '预算正常', tone: 'success', color: 'var(--green)' },
  near_soft: { level: 'near_soft', label: '接近软预算', tone: 'warning', color: 'var(--amber)' },
  soft_exceeded: { level: 'soft_exceeded', label: '已超软预算', tone: 'warning', color: 'var(--amber)' },
  hard_exceeded: { level: 'hard_exceeded', label: '已超硬预算', tone: 'danger', color: 'var(--red)' },
}

export function budgetLevelBadge(level: BudgetLevel | null | undefined): BudgetBadge {
  if (level === 'unlimited') return BUDGET_BADGES.unlimited
  if (level === 'soft_exceeded') return BUDGET_BADGES.soft_exceeded
  if (level === 'hard_exceeded') return BUDGET_BADGES.hard_exceeded
  return BUDGET_BADGES.ok
}

function ratioExceedsSoft(used: number, soft: number | null): boolean {
  if (soft === null || soft <= 0) return false
  return used / soft >= NEAR_SOFT_RATIO
}

/** 预算状态 -> 徽标；`ok` 且已用/预计接近软预算时升级为「接近软预算」。 */
export function budgetBadgeForState(state: BudgetStateDto | null | undefined): BudgetBadge {
  if (!state) return BUDGET_BADGES.unlimited
  if (state.level !== 'ok') return budgetLevelBadge(state.level)
  const committedCost = Math.max(state.committed_cost_micro, state.projected_cost_micro)
  if (state.soft_cost_micro === null && state.soft_seconds === null) {
    // 没有软预算时，用硬预算的 80% 作为「接近上限」提示；两者都没有就是未设置。
    if (state.hard_cost_micro === null && state.hard_seconds === null) return BUDGET_BADGES.ok
    const nearHard =
      ratioExceedsSoft(committedCost, state.hard_cost_micro) ||
      ratioExceedsSoft(state.projected_seconds, state.hard_seconds)
    return nearHard ? BUDGET_BADGES.near_soft : BUDGET_BADGES.ok
  }
  const near =
    ratioExceedsSoft(committedCost, state.soft_cost_micro) ||
    ratioExceedsSoft(state.projected_seconds, state.soft_seconds)
  return near ? BUDGET_BADGES.near_soft : BUDGET_BADGES.ok
}

/** 已用+预留 相对生效上限的百分比（无上限时返回 null，用于进度条）。 */
export function budgetUsedPercent(state: BudgetStateDto | null | undefined): number | null {
  if (!state) return null
  const limit = state.soft_cost_micro ?? state.hard_cost_micro
  if (limit === null || limit <= 0) return null
  return Math.min(100, Math.round((state.committed_cost_micro / limit) * 100))
}

// --- 项目 / 剧集汇总 -------------------------------------------------------

function normalizeRemaining(raw: unknown): RemainingWorkloadDto {
  const source = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
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
  const micro = source.cost_micro
  const seconds = source.seconds
  return {
    currency: asText(source.currency, 'CNY'),
    cost_micro: micro === null || micro === undefined ? null : Math.round(asNumber(micro, 0)),
    cost_known: asBoolean(source.cost_known, micro !== null && micro !== undefined),
    partial_cost_micro: asCount(source.partial_cost_micro),
    seconds: seconds === null || seconds === undefined ? null : asCount(seconds),
    components,
    unknown_components: unknownComponents,
  }
}

function normalizeEpisode(raw: unknown): EpisodeCostDto | null {
  if (!raw || typeof raw !== 'object') return null
  const source = raw as Record<string, unknown>
  const projectId = asString(source.project_id)
  if (!projectId) return null
  const used = (source.used && typeof source.used === 'object' ? source.used : {}) as Record<string, unknown>
  return {
    project_id: projectId,
    title: asText(source.title, '未命名剧集'),
    episode_number: asCount(source.episode_number),
    status: asString(source.status),
    used: {
      cost_micro: asCount(used.cost_micro),
      cost_known: asBoolean(used.cost_known, true),
      unknown_call_count: asCount(used.unknown_call_count),
      seconds: asCount(used.seconds),
    },
    budget_status: normalizeBudgetState(source.budget_status),
  }
}

export function normalizeBudgetSummary(raw: unknown): BudgetSummaryDto {
  const source = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const used = (source.used && typeof source.used === 'object' ? source.used : {}) as Record<string, unknown>
  const reserved = (source.reserved && typeof source.reserved === 'object' ? source.reserved : {}) as Record<
    string,
    unknown
  >
  const byCapability: CapabilityUsageDto[] = []
  if (Array.isArray(used.by_capability)) {
    for (const item of used.by_capability) byCapability.push(normalizeCapabilityUsage(item))
  }
  const groups = (key: string): UsageGroupDto[] => {
    const list: UsageGroupDto[] = []
    if (Array.isArray(source[key])) {
      for (const item of source[key] as unknown[]) {
        const normalized = normalizeUsageGroup(item)
        if (normalized) list.push(normalized)
      }
    }
    return list
  }
  const episodes: EpisodeCostDto[] = []
  if (Array.isArray(source.episodes)) {
    for (const item of source.episodes) {
      const normalized = normalizeEpisode(item)
      if (normalized) episodes.push(normalized)
    }
  }
  return {
    project_id: asString(source.project_id),
    series_id: asString(source.series_id),
    is_episode: asBoolean(source.is_episode),
    currency: asText(source.currency, 'CNY'),
    used: {
      cost_micro: asCount(used.cost_micro),
      cost_known: asBoolean(used.cost_known, true),
      unknown_call_count: asCount(used.unknown_call_count),
      failed_call_count: asCount(used.failed_call_count),
      call_count: asCount(used.call_count),
      seconds: asCount(used.seconds),
      by_capability: byCapability,
    },
    reserved: {
      cost_micro: asCount(reserved.cost_micro),
      seconds: asCount(reserved.seconds),
      count: asCount(reserved.count),
    },
    remaining: normalizeRemaining(source.remaining),
    budget: normalizeEffectiveBudget(source.budget),
    status: normalizeBudgetState(source.status),
    by_job_type: groups('by_job_type'),
    by_shot: groups('by_shot'),
    by_capability: groups('by_capability'),
    episodes,
  }
}

/** 项目已用成本文案：全部已知才给金额，否则写「成本未知」并给出已知部分。 */
export function usedCostText(costMicro: number | null | undefined, costKnown: boolean, currency = 'CNY'): string {
  if (costKnown && costMicro !== null && costMicro !== undefined) return formatMicroAmount(costMicro, currency)
  if (costMicro === null || costMicro === undefined || costMicro === 0) return UNKNOWN_COST_TEXT
  return UNKNOWN_COST_TEXT + '（已知部分 ' + formatMicroAmount(costMicro, currency) + '）'
}
