import assert from 'node:assert/strict'
import { test } from 'vitest'

import { expectRender, visibleText } from '../test-support/ssrTestHelper.mts'
import PricingConfigPanel from './PricingConfigPanel'

// 价目表数据在 useEffect 中拉取，SSR 渲染的是无数据的初始骨架。

test('渲染模型价格面板骨架与标题', () => {
  const html = expectRender(PricingConfigPanel)
  assert.ok(html.includes('pricing-panel'), '应渲染面板根节点')
  assert.ok(html.includes('模型价格'), '应显示面板标题')
})

test('渲染计价单位说明（LLM/图像/视频/语音/编码）', () => {
  const html = visibleText(expectRender(PricingConfigPanel))
  assert.ok(html.includes('计价单位说明'), '应显示说明标题')
  assert.ok(html.includes('文本模型（LLM）：每 100 万 tokens'), 'LLM 计价单位')
  assert.ok(html.includes('图像生成：每张'), '图像计价单位')
  assert.ok(html.includes('视频生成：每秒'), '视频计价单位')
  assert.ok(html.includes('语音合成：每 1000 字符'), '语音计价单位')
  assert.ok(html.includes('本地编码（FFmpeg）：每分钟编码'), '编码计价单位')
})

test('金额换算提示包含 micro 说明', () => {
  const html = visibleText(expectRender(PricingConfigPanel))
  assert.ok(html.includes('1 元 = 1,000,000 micro'), '应说明元到 micro 的换算')
  assert.ok(html.includes('金额不落浮点'), '应说明不落浮点')
})

test('无价目表数据时保存按钮禁用、重新加载可用', () => {
  const html = expectRender(PricingConfigPanel)
  assert.ok(html.includes('保存模型价格'), '应渲染保存按钮')
  assert.ok(html.includes('重新加载'), '应渲染重新加载按钮')
  assert.ok(html.includes('disabled'), '初始无数据时保存应禁用')
})

test('初始态不显示加载中与脏数据提示', () => {
  const html = expectRender(PricingConfigPanel)
  assert.ok(!html.includes('正在加载模型价格'), '初始 loading=false 不应显示加载中文案')
  assert.ok(!html.includes('有未保存的修改'), '初始无草稿修改不应显示脏数据提示')
})
