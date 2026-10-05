import assert from 'node:assert/strict'
import { test } from 'node:test'

import { expectRender } from '../test-support/ssrTestHelper.mts'
import SystemSettingsPage from './SystemSettingsPage'

// 设置页数据在 useEffect 中加载，SSR 渲染初始骨架。

test('渲染系统设置页骨架与设置分区导航', () => {
  const html = expectRender(SystemSettingsPage)
  assert.ok(html.includes('aria-label="系统设置"'), '应有设置页区域标签')
  assert.ok(html.includes('aria-label="设置分区"'), '应有分区导航')
  assert.ok(html.includes('系统设置'), '应显示页面标题')
})
