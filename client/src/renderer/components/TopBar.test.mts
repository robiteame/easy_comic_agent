import assert from 'node:assert/strict'
import { test } from 'node:test'

import { expectRender } from '../test-support/ssrTestHelper.mts'
import TopBar, { OPEN_SETTINGS_EVENT } from './TopBar'

test('TopBar 冒烟：非 macOS 环境渲染基础拖拽条', () => {
  const html = expectRender(TopBar)
  // Node 环境 navigator.userAgent 不含 Macintosh，走 Windows 原生标题栏分支。
  assert.ok(html.includes('class="topbar"'), '非 macOS 应输出 topbar 基础类名')
})

test('OPEN_SETTINGS_EVENT 事件名保持稳定（跨组件通信契约）', () => {
  assert.equal(OPEN_SETTINGS_EVENT, 'workspace:open-settings')
})
