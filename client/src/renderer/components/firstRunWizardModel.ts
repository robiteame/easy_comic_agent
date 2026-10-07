/**
 * 首启向导纯逻辑：步骤推进、Key 表单校验、模型配置补丁构造。
 *
 * 组件层只负责渲染与调用 API；可测试的决策（能否进入下一步、把哪些
 * 类别合并进 PUT /model-configs、测试按钮何时可用）全部收敛在这里。
 */

import { STYLE_OPTIONS } from '../constants/styleTemplates'

export const WIZARD_STEP_COUNT = 3

export type WizardStepIndex = 0 | 1 | 2

export const WIZARD_STEPS: { key: WizardStepIndex; title: string; hint: string }[] = [
  { key: 0, title: '选画风', hint: '挑选一套内置画风，稍后创建的第一个项目会使用它' },
  { key: 1, title: '配置模型', hint: '只问最关键的两个 Key，暂不配置也可以先体验' },
  { key: 2, title: '开始创作', hint: '打开示例项目看看产品能做什么，或从空白开始' },
]

/** 向导内置的默认端点预设：与 server 端 .env 默认值保持一致（MiMo LLM / 火山方舟）。 */
export const WIZARD_ENDPOINT_PRESETS = {
  script: {
    protocol: 'openai-chat',
    base_url: 'https://token-plan-cn.xiaomimimo.com/v1',
    model: 'mimo-v2.5',
    auth_style: 'api-key-header',
  },
  video: {
    protocol: 'ark-seedance',
    base_url: 'https://ark.cn-beijing.volces.com/api/v3',
    model: 'doubao-seedance-2-0-260128',
  },
} as const

/** 第 3 步的一句话说明：与 README 的 Key 口径一致，不夸大也不隐瞒。 */
export const WIZARD_KEY_NOTE =
  '完整出片需要配置 MIMO_API_KEY（剧本/分镜 LLM 与配音，或改用 OPENAI_API_KEY）与 ARK_API_KEY（视频生成）；图像未配置 Key 时自动使用本地占位图。'

const MASKED_SECRET = '********'

export interface WizardKeyForm {
  mimoApiKey: string
  arkApiKey: string
}

export function wizardKeyValue(raw: string | undefined): string {
  return String(raw || '').trim()
}

/** 用户实际输入了新 Key（非空且不是回显掩码）时才需要保存。 */
export function hasFreshKey(raw: string | undefined): boolean {
  const value = wizardKeyValue(raw)
  return Boolean(value) && value !== MASKED_SECRET
}

/** 画风单选合法性：必须命中 8 套内置模板之一。 */
export function isWizardStyleValid(style: string | undefined): boolean {
  return STYLE_OPTIONS.some((option) => option.value === style)
}

export function canGoNextFromStep(step: WizardStepIndex, state: { style: string; keys: WizardKeyForm }): boolean {
  if (step === 0) return isWizardStyleValid(state.style)
  // 第 2 步允许全部留空（无 Key 也能继续，图像走本地占位图）。
  if (step === 1) return true
  return false
}

export function nextWizardStep(step: WizardStepIndex): WizardStepIndex | null {
  if (step >= WIZARD_STEP_COUNT - 1) return null
  return (step + 1) as WizardStepIndex
}

export function prevWizardStep(step: WizardStepIndex): WizardStepIndex | null {
  if (step <= 0) return null
  return (step - 1) as WizardStepIndex
}

/**
 * 构造 PUT /api/settings/model-configs 的 categories 补丁。
 *
 * 规则（诚实且不覆盖用户已有配置）：
 * - 该类 Key 没有新输入 → 不包含该类别（保持现状）；
 * - 已配置过（现配置密钥非空，无论是否掩码）且端点完整 → 只替换 api_key，
 *   其余字段原样保留；
 * - 从未配置 → 使用向导预设端点（MiMo / 方舟官方默认）+ 新 Key。
 */
export function buildWizardModelConfigPatch(
  currentCategories: Record<string, any> | undefined | null,
  keys: WizardKeyForm,
): Record<string, Record<string, any>> {
  const categories: Record<string, Record<string, any>> = {}
  const current = currentCategories || {}

  if (hasFreshKey(keys.mimoApiKey)) {
    const existing = current.script || {}
    const configured = hasExistingEndpoint(existing)
    categories.script = {
      ...(configured ? pickEndpointFields(existing) : {}),
      ...(configured ? {} : WIZARD_ENDPOINT_PRESETS.script),
      api_key: wizardKeyValue(keys.mimoApiKey),
    }
  }

  if (hasFreshKey(keys.arkApiKey)) {
    const existing = current.video || {}
    const configured = hasExistingEndpoint(existing)
    categories.video = {
      ...(configured ? pickEndpointFields(existing) : {}),
      ...(configured ? {} : WIZARD_ENDPOINT_PRESETS.video),
      api_key: wizardKeyValue(keys.arkApiKey),
    }
  }

  return categories
}

function hasExistingEndpoint(category: Record<string, any>): boolean {
  return Boolean(String(category?.base_url || '').trim() && String(category?.api_key || '').trim())
}

function pickEndpointFields(category: Record<string, any>): Record<string, any> {
  const picked: Record<string, any> = {}
  for (const field of ['protocol', 'base_url', 'model', 'auth_style']) {
    if (String(category?.[field] || '').trim()) picked[field] = category[field]
  }
  return picked
}

/** 测试按钮的可用条件：探活是真实外呼，没有新 Key 时测的是现行配置，同样有意义。 */
export function wizardTestConfig(
  capability: 'llm' | 'video',
  currentCategories: Record<string, any> | undefined | null,
  keys: WizardKeyForm,
): Record<string, any> | undefined {
  const category = capability === 'llm' ? 'script' : 'video'
  const freshKey = capability === 'llm' ? keys.mimoApiKey : keys.arkApiKey
  if (!hasFreshKey(freshKey)) {
    // 未输入新 Key：不传 config，由后端测现行生效配置。
    return undefined
  }
  const config = buildWizardModelConfigPatch(currentCategories, keys)[category]
  return config ? { ...config, api_key: wizardKeyValue(freshKey) } : undefined
}

export type ConnectionStatus = 'ok' | 'fail' | 'unsupported_check'

/** 把探活结果翻译成向导内的一句话反馈；unsupported 同样如实展示。 */
export function describeConnectionTest(result: { status: string; message?: string; latency_ms?: number }): {
  tone: 'success' | 'error' | 'warning'
  text: string
} {
  const message = String(result.message || '').trim() || '接口无返回说明'
  const latency = typeof result.latency_ms === 'number' ? `（${result.latency_ms} ms）` : ''
  if (result.status === 'ok') return { tone: 'success', text: `连通正常${latency}：${message}` }
  if (result.status === 'unsupported_check') return { tone: 'warning', text: `无法免费验证${latency}：${message}` }
  return { tone: 'error', text: `连接失败${latency}：${message}` }
}
