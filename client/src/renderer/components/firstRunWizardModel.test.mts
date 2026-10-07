import assert from 'node:assert/strict'
import { test } from 'vitest'

import {
  WIZARD_ENDPOINT_PRESETS,
  WIZARD_KEY_NOTE,
  WIZARD_STEP_COUNT,
  buildWizardModelConfigPatch,
  canGoNextFromStep,
  describeConnectionTest,
  hasFreshKey,
  isWizardStyleValid,
  nextWizardStep,
  prevWizardStep,
  wizardTestConfig,
} from './firstRunWizardModel'

// 向导纯逻辑：步骤推进、Key 校验、模型配置补丁构造与探活结果翻译。
// 这些决策必须与后端 save_model_config 的合并语义一致（空值=沿用默认）。

const EMPTY_KEYS = { mimoApiKey: '', arkApiKey: '' }

test('步骤推进：三步向导的 next/prev 边界', () => {
  assert.equal(WIZARD_STEP_COUNT, 3)
  assert.equal(nextWizardStep(0), 1)
  assert.equal(nextWizardStep(1), 2)
  assert.equal(nextWizardStep(2), null, '最后一步没有下一步')
  assert.equal(prevWizardStep(0), null, '第一步没有上一步')
  assert.equal(prevWizardStep(1), 0)
  assert.equal(prevWizardStep(2), 1)
})

test('第 1 步：画风必须是 8 套内置模板之一', () => {
  assert.ok(isWizardStyleValid('anime'))
  assert.ok(isWizardStyleValid('clay'))
  assert.ok(isWizardStyleValid(undefined) === false)
  assert.ok(isWizardStyleValid('nonexistent-style') === false)
  assert.ok(isWizardStyleValid('') === false)
})

test('第 1 步：画风非法时不能进入下一步，合法时可以', () => {
  assert.equal(canGoNextFromStep(0, { style: 'anime', keys: EMPTY_KEYS }), true)
  assert.equal(canGoNextFromStep(0, { style: 'bad-style', keys: EMPTY_KEYS }), false)
  assert.equal(canGoNextFromStep(0, { style: '', keys: EMPTY_KEYS }), false)
})

test('第 2 步：两个 Key 都留空也可以继续（图像走本地占位图）', () => {
  assert.equal(canGoNextFromStep(1, { style: 'anime', keys: EMPTY_KEYS }), true)
  assert.equal(canGoNextFromStep(1, { style: 'anime', keys: { mimoApiKey: 'sk-1', arkApiKey: '' } }), true)
})

test('Key 输入判定：空串与回显掩码都不算新 Key', () => {
  assert.equal(hasFreshKey(''), false)
  assert.equal(hasFreshKey('   '), false)
  assert.equal(hasFreshKey(undefined), false)
  assert.equal(hasFreshKey('********'), false, '掩码是回显，不是新输入')
  assert.equal(hasFreshKey(' sk-real-key '), true)
})

test('配置补丁：没有新 Key 时不包含任何类别（保持现状）', () => {
  const patch = buildWizardModelConfigPatch({ script: {}, video: {} }, EMPTY_KEYS)
  assert.deepEqual(patch, {})
})

test('配置补丁：未配置过的类别使用向导预设端点 + 新 Key', () => {
  const patch = buildWizardModelConfigPatch({}, { mimoApiKey: 'sk-mimo', arkApiKey: '' })
  assert.deepEqual(Object.keys(patch), ['script'])
  assert.equal(patch.script.api_key, 'sk-mimo')
  assert.equal(patch.script.protocol, WIZARD_ENDPOINT_PRESETS.script.protocol)
  assert.equal(patch.script.base_url, WIZARD_ENDPOINT_PRESETS.script.base_url)
  assert.equal(patch.script.auth_style, WIZARD_ENDPOINT_PRESETS.script.auth_style)
})

test('配置补丁：已配置端点只替换 Key，不覆盖用户的自定义端点', () => {
  const current = {
    script: {
      protocol: 'openai-chat',
      base_url: 'https://api.deepseek.com',
      model: 'deepseek-chat',
      auth_style: 'bearer',
      api_key: '********',
    },
    video: {
      protocol: 'ark-seedance',
      base_url: 'https://ark.example.com',
      model: 'my-model',
      api_key: 'old-key',
    },
  }
  const patch = buildWizardModelConfigPatch(current, { mimoApiKey: 'sk-new', arkApiKey: '' })
  assert.deepEqual(Object.keys(patch), ['script'])
  assert.equal(patch.script.base_url, 'https://api.deepseek.com', '已有自定义端点不得被预设覆盖')
  assert.equal(patch.script.model, 'deepseek-chat')
  assert.equal(patch.script.auth_style, 'bearer')
  assert.equal(patch.script.api_key, 'sk-new')
  // video 未输入新 Key：整个类别不进补丁，旧 Key 原样保留。
  assert.equal(patch.video, undefined)
})

test('配置补丁：端点不完整（无密钥，视为未配置）时整体回落官方预设', () => {
  const current = { script: { base_url: 'https://x.example.com' }, video: {} }
  const patch = buildWizardModelConfigPatch(current, { mimoApiKey: 'sk-m', arkApiKey: 'sk-a' })
  // 向导输入框标注的是官方 MIMO 接口：只填了地址却从未配通时，按预设
  // 端点整体写入，避免把新密钥发往语义不明的半配置地址。
  assert.equal(patch.script.base_url, WIZARD_ENDPOINT_PRESETS.script.base_url)
  assert.equal(patch.script.api_key, 'sk-m')
  assert.equal(patch.video.protocol, WIZARD_ENDPOINT_PRESETS.video.protocol)
  assert.equal(patch.video.api_key, 'sk-a')
})

test('探活配置：未输入新 Key 时不传 config（测现行生效配置）', () => {
  assert.equal(wizardTestConfig('llm', { script: { base_url: 'https://a', api_key: 'k' } }, EMPTY_KEYS), undefined)
  assert.equal(wizardTestConfig('video', {}, { mimoApiKey: '', arkApiKey: '' }), undefined)
})

test('探活配置：输入新 Key 时携带完整待测端点（保存前即可测）', () => {
  const config = wizardTestConfig('llm', {}, { mimoApiKey: 'sk-x', arkApiKey: '' })
  assert.ok(config)
  assert.equal(config!.api_key, 'sk-x')
  assert.equal(config!.base_url, WIZARD_ENDPOINT_PRESETS.script.base_url)
})

test('探活结果翻译：ok / unsupported / fail 各自的语气与文案', () => {
  assert.equal(describeConnectionTest({ status: 'ok', message: '成功', latency_ms: 120 }).tone, 'success')
  assert.ok(describeConnectionTest({ status: 'ok', message: '成功', latency_ms: 120 }).text.includes('120'))
  assert.equal(describeConnectionTest({ status: 'unsupported_check', message: '无免费探活' }).tone, 'warning')
  const fail = describeConnectionTest({ status: 'fail', message: '鉴权失败' })
  assert.equal(fail.tone, 'error')
  assert.ok(fail.text.includes('鉴权失败'))
})

test('第 3 步说明与 README 口径一致：包含 MIMO 与 ARK 两个 Key 及占位图说明', () => {
  assert.ok(WIZARD_KEY_NOTE.includes('MIMO_API_KEY'))
  assert.ok(WIZARD_KEY_NOTE.includes('ARK_API_KEY'))
  assert.ok(WIZARD_KEY_NOTE.includes('占位图'))
})
