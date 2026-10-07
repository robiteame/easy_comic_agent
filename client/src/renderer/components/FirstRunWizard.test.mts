import assert from 'node:assert/strict'
import { test } from 'vitest'

import { expectRender, visibleText } from '../test-support/ssrTestHelper.mts'
import FirstRunWizard, { OPEN_WIZARD_EVENT } from './FirstRunWizard'

// SSR 只渲染首帧（step 0），不执行 useEffect；向导的数据加载与保存逻辑
// 在 firstRunWizardModel 测试与后端 pytest 中覆盖。

test('open=false 时不渲染任何向导内容', () => {
  const html = expectRender(FirstRunWizard, { open: false, onClose: () => {} })
  assert.ok(!html.includes('首次启动向导'))
})

test('open=true 渲染三步指示器、8 套画风卡片与底部导航', () => {
  const html = visibleText(expectRender(FirstRunWizard, { open: true, onClose: () => {} }))
  assert.ok(html.includes('aria-label="首次启动向导"'), '向导对话框')
  assert.ok(html.includes('选画风'), '第 1 步指示器')
  assert.ok(html.includes('配置模型'), '第 2 步指示器')
  assert.ok(html.includes('开始创作'), '第 3 步指示器')
  assert.ok(html.includes('日系写实漫') && html.includes('定格黏土'), '8 套内置画风卡片')
  assert.ok(html.includes('细腻赛璐璐、自然肤色、柔和电影光'), '画风卡片描述')
  assert.ok(html.includes('跳过向导'), '每步可跳过')
  assert.ok(html.includes('下一步'), '可进入下一步')
  assert.ok((html.match(/role="radio"/g) || []).length === 8, '8 张卡片单选语义')
  assert.ok(html.includes('aria-checked="true"'), '默认选中一套画风')
})

test('第 2/3 步关键文案与 Key 说明在组件源内可达（口径诚实）', async () => {
  const { WIZARD_KEY_NOTE } = await import('./firstRunWizardModel')
  assert.ok(WIZARD_KEY_NOTE.includes('MIMO_API_KEY'))
  assert.ok(WIZARD_KEY_NOTE.includes('ARK_API_KEY'))
  assert.ok(WIZARD_KEY_NOTE.includes('OPENAI_API_KEY'), '与 README 口径一致：可改用 OpenAI Key')
})

test('OPEN_WIZARD_EVENT 使用 workspace 事件命名空间', () => {
  assert.equal(OPEN_WIZARD_EVENT, 'workspace:open-wizard')
})
