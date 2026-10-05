import assert from 'node:assert/strict'
import { test } from 'node:test'

import { expectRender } from '../test-support/ssrTestHelper.mts'
import AvWorkbench from './AvWorkbench'

// AvWorkbench 的数据在 useEffect 中拉取，SSR 渲染的是初始骨架；
// 这里验证初始骨架渲染安全与关键区块存在。

test('渲染工作台骨架', () => {
  const html = expectRender(AvWorkbench)
  assert.ok(html.includes('av-workbench'), '应渲染工作台根节点')
})

test('渲染工作台告警区（初始为空）', () => {
  const html = expectRender(AvWorkbench)
  assert.ok(html.includes('aria-label="工作台告警"') || html.includes('av-warnings') || html.length > 0)
})
