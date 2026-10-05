import assert from 'node:assert/strict'
import { test, beforeEach } from 'node:test'

import { expectRender } from '../test-support/ssrTestHelper.mts'
import { useProjectStore } from '../stores/projectStore'
import LeftSidebar from './LeftSidebar'

beforeEach(() => {
  useProjectStore.getState().reset()
})

test('展开态渲染侧栏骨架与未命名项目回退标题', () => {
  const html = expectRender(LeftSidebar, { collapsed: false, onToggleCollapsed: () => {} })
  assert.ok(html.includes('aria-label="左侧导航"'))
  assert.ok(!html.includes('left-sidebar collapsed'), '展开态不应带 collapsed 类')
  assert.ok(html.includes('暂无项目'), '无项目时显示空态文案')
})

test('collapsed 属性切换折叠类名', () => {
  const html = expectRender(LeftSidebar, { collapsed: true, onToggleCollapsed: () => {} })
  assert.ok(html.includes('left-sidebar collapsed'), '折叠态应带 collapsed 类')
})
