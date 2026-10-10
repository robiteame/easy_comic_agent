/**
 * 预算/模型端点拦截错误解析的纯逻辑层。
 */

import type { BudgetBlockedDetail, JobCostDto, ProviderBlockedDetail } from './costTypes'
import { asString, asText, normalizeBudgetState, normalizeEstimate, normalizeJobCost } from './costModelCore'

// --- 提交接口的预算提示 / 拦截 ---------------------------------------------

/** 提交类接口成功响应里的软预算提示（没有则返回空串）。 */
export function budgetWarningFromResponse(response: unknown): string {
  if (!response || typeof response !== 'object') return ''
  const source = response as Record<string, unknown>
  const warning = asString(source.budget_warning).trim()
  if (warning) return warning
  return asString(source.budget_level) === 'soft_exceeded' ? '本次任务已超出项目软预算，任务仍会继续执行。' : ''
}

/**
 * 模型端点未配置拦截：HTTP 409 且 `detail` 是对象（error_code=`provider_not_configured`）。
 *
 * 返回 null 表示这只是一次普通失败，调用方按原有错误提示处理。
 */
export function providerBlockedFromError(error: unknown): ProviderBlockedDetail | null {
  const data = errorResponseData(error)
  if (!data || typeof data !== 'object') return null
  const detail = (data as { detail?: unknown }).detail
  if (!detail || typeof detail !== 'object') return null
  const source = detail as Record<string, unknown>
  const message = asString(source.message).trim()
  if (!message) return null
  if (
    asString(source.status) !== 'provider_not_configured' &&
    asString(source.error_code) !== 'provider_not_configured'
  )
    return null
  const rawMissing = Array.isArray(source.missing) ? source.missing : []
  const missing = rawMissing
    .map((item) => {
      const entry = (item || {}) as Record<string, unknown>
      const capability = asString(entry.capability).trim()
      if (!capability) return null
      return {
        capability,
        label: asString(entry.label).trim() || capability,
        message: asString(entry.message).trim(),
      }
    })
    .filter((item): item is { capability: string; label: string; message: string } => item !== null)
  return {
    ok: false,
    status: asString(source.status, 'provider_not_configured'),
    error_code: asString(source.error_code, 'provider_not_configured'),
    message,
    missing,
  }
}

function errorResponseData(error: unknown): unknown {
  return (error as { response?: { data?: unknown } } | undefined)?.response?.data
}

function errorHttpStatus(error: unknown): number | undefined {
  return (error as { response?: { status?: number } } | undefined)?.response?.status
}

/**
 * 硬预算拦截：HTTP 409 且 `detail` 是对象（status=`budget_blocked`）。
 *
 * 返回 null 表示这只是一次普通失败，调用方按原有错误提示处理。
 */
export function budgetBlockedFromError(error: unknown): BudgetBlockedDetail | null {
  const data = errorResponseData(error)
  if (!data || typeof data !== 'object') return null
  const detail = (data as { detail?: unknown }).detail
  if (!detail || typeof detail !== 'object') return null
  const source = detail as Record<string, unknown>
  const message = asString(source.message).trim()
  if (!message) return null
  if (asString(source.status) !== 'budget_blocked' && asString(source.error_code) !== 'budget_exceeded') return null
  return {
    ok: false,
    status: 'budget_blocked',
    error_code: asText(source.error_code, 'budget_exceeded'),
    message,
    budget: source.budget ? normalizeBudgetState(source.budget) : null,
    estimate: source.estimate ? normalizeEstimate(source.estimate) : null,
  }
}

/** 预算相关接口失败原因：优先用后端 detail（字符串或对象里的 message）。 */
export function describeBudgetError(error: unknown, fallback = '操作失败，请稍后重试'): string {
  const blocked = budgetBlockedFromError(error)
  if (blocked) return blocked.message
  const data = errorResponseData(error)
  if (data && typeof data === 'object') {
    // 只用结构化字段（detail / message）：整个响应体是 HTML 或纯文本时不能直接
    // 当成提示文案甩给用户。
    const detail = (data as { detail?: unknown }).detail
    if (typeof detail === 'string' && detail.trim()) return detail.trim()
    if (Array.isArray(detail)) {
      const first = detail[0] as { msg?: unknown } | undefined
      const msg = asString(first?.msg).trim()
      if (msg) return msg
    }
    if (detail && typeof detail === 'object') {
      const msg = asString((detail as { message?: unknown }).message).trim()
      if (msg) return msg
    }
    const message = asString((data as { message?: unknown }).message).trim()
    if (message) return message
  }
  if (error && typeof error === 'object') {
    const message = asString((error as { message?: unknown }).message).trim()
    const status = errorHttpStatus(error)
    if (status) return fallback + '（HTTP ' + status + '）'
    if (message && message !== 'Network Error') return fallback + '：' + message
    if (message === 'Network Error') return fallback + '（网络不可用）'
  }
  return fallback
}

export const EMPTY_JOB_COST: JobCostDto = normalizeJobCost(null)
