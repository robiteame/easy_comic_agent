import assert from 'node:assert/strict'
import { test, beforeEach } from 'vitest'

import { expectRender, visibleText } from './test-support/ssrTestHelper.mts'
import { useShotStore } from './stores/shotStore'
import { useProjectStore } from './stores/projectStore'
import App from './App'

// App 是整棵 UI 树的组装层：一次 SSR 渲染即可覆盖所有顶层子组件的
// 组合初始化（import 错误、props 传递错误、跨组件 store 初始读取都会在此暴露）。

beforeEach(() => {
  useShotStore.setState({ shots: [], selectedShotId: null, isGenerating: false })
  useProjectStore.getState().reset()
})

test('整树渲染：包含顶部、左侧、主工作区、右侧与底部区域', () => {
  const html = visibleText(expectRender(App))
  assert.ok(html.includes('aria-label="左侧导航"'), '左侧栏')
  assert.ok(html.includes('aria-label="主工作区"'), '主工作区')
  assert.ok(html.includes('aria-label="右侧运行信息"'), '右侧栏')
  assert.ok(html.includes('aria-label="底部状态栏"'), '底部状态栏')
})

test('整树渲染：底部状态栏处于等待输入剧本的空态', () => {
  const html = visibleText(expectRender(App))
  assert.ok(html.includes('等待输入剧本'), '空项目时应显示等待剧本')
})
