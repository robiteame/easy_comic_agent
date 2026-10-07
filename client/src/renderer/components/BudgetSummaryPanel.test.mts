import assert from 'node:assert/strict'
import { test, beforeEach } from 'vitest'

import { expectRender } from '../test-support/ssrTestHelper.mts'
import { useProjectStore } from '../stores/projectStore'
import BudgetSummaryPanel from './BudgetSummaryPanel'

beforeEach(() => {
  useProjectStore.getState().reset()
})

test('无项目 ID 时不渲染面板', () => {
  const html = expectRender(BudgetSummaryPanel, { projectId: null })
  assert.equal(html, '')
})

test('有项目 ID 时渲染预算与成本面板标题', () => {
  const html = expectRender(BudgetSummaryPanel, { projectId: 'proj-1', title: '测试项目' })
  assert.ok(html.includes('aria-label="预算与成本"'), '应有面板区域标签')
  assert.ok(html.includes('测试项目'), '标题应包含项目名')
})
