/**
 * 系统设置页「测试连接」的纯逻辑：请求载荷构造、响应归一化与结果展示映射。
 *
 * 这里不碰 DOM、不碰网络，可以在 node:test 下直接覆盖（参考 taskCenterModel）；
 * 组件只负责「调用纯函数 + 发请求 + 渲染」，业务判断不散落在 JSX 里。
 */

export type ConnectionTestCapability = 'llm' | 'image' | 'video' | 'tts'
export type ConnectionTestStatus = 'ok' | 'fail' | 'unsupported_check'

export interface ConnectionTestResult {
  status: ConnectionTestStatus
  provider: string
  model: string
  latency_ms: number
  message: string
}

export interface ConnectionTestPayload {
  capability: ConnectionTestCapability
  config: Record<string, any>
}

/** 表单类别 → 测试接口的能力名：script 端点对外叫 llm，voice 端点对外叫 tts。 */
const CAPABILITY_BY_CATEGORY: Record<string, ConnectionTestCapability> = {
  script: 'llm',
  image: 'image',
  video: 'video',
  voice: 'tts',
}

const CONNECTION_STATUSES: ConnectionTestStatus[] = ['ok', 'fail', 'unsupported_check']

export function connectionTestCapability(category: string): ConnectionTestCapability | null {
  return CAPABILITY_BY_CATEGORY[category] ?? null
}

/**
 * 构造「测试当前表单值」的请求载荷：测的是未保存的草稿，保存前即可测。
 * 后端回显的 capabilities 等字段不回传；掩码密钥原样交给服务端，由其判断
 * 是否复用已保存密钥（仅同一端点时复用）。
 */
export function buildConnectionTestPayload(
  category: string,
  config: Record<string, any> | null | undefined,
): ConnectionTestPayload | null {
  const capability = connectionTestCapability(category)
  if (!capability) return null
  const source = config && typeof config === 'object' ? config : {}
  const form: Record<string, any> = {}
  for (const [key, value] of Object.entries(source)) {
    if (key === 'capabilities') continue
    form[key] = value
  }
  return { capability, config: form }
}

function asString(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number') return String(value)
  return fallback
}

/** 服务端响应归一化：状态不在白名单或结构不对时返回 null，不让渲染层拿到半成品。 */
export function normalizeConnectionTestResult(raw: unknown): ConnectionTestResult | null {
  if (!raw || typeof raw !== 'object') return null
  const source = raw as Record<string, unknown>
  const status = asString(source.status) as ConnectionTestStatus
  if (CONNECTION_STATUSES.indexOf(status) < 0) return null
  const latency = typeof source.latency_ms === 'number' ? source.latency_ms : Number(source.latency_ms)
  return {
    status,
    provider: asString(source.provider),
    model: asString(source.model),
    latency_ms: Number.isFinite(latency) ? Math.max(0, Math.round(latency)) : 0,
    message: asString(source.message),
  }
}

/** 请求本身失败（网络异常 / 4xx / 5xx）时的兜底结果；文案优先取服务端 detail。 */
export function connectionTestFailureFromError(error: unknown): ConnectionTestResult {
  const source = error as { response?: { data?: { detail?: unknown } }; message?: unknown } | null
  const detail = source?.response?.data?.detail
  const message = asString(detail) || asString(source?.message) || '连接测试失败，请稍后重试'
  return { status: 'fail', provider: '', model: '', latency_ms: 0, message }
}

export type ConnectionTone = 'success' | 'error' | 'muted'

export function connectionTestTone(status: ConnectionTestStatus): ConnectionTone {
  if (status === 'ok') return 'success'
  if (status === 'fail') return 'error'
  return 'muted'
}

/** 结果展示文案：成功带延迟毫秒数；失败与「不支持检测」原样展示后端 message。 */
export function connectionTestText(result: ConnectionTestResult): string {
  if (result.status === 'ok') {
    return result.latency_ms > 0 ? `连接成功 · ${result.latency_ms}ms` : '连接成功'
  }
  if (result.status === 'fail') {
    return result.message || '连接失败，请检查配置'
  }
  return result.message || '该服务暂不支持自动检测'
}
