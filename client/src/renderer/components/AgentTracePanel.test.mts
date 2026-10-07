import assert from 'node:assert/strict'
import { test, beforeEach } from 'vitest'

import { expectRender } from '../test-support/ssrTestHelper.mts'
import { useProjectStore } from '../stores/projectStore'
import AgentTracePanel from './AgentTracePanel'

beforeEach(() => {
  useProjectStore.getState().reset()
})

test('无项目时显示选择项目空态', () => {
  const html = expectRender(AgentTracePanel)
  assert.ok(html.includes('选择项目后显示 Agent 追踪'), '无项目应显示空态提示')
})

test('有项目时渲染无运行记录空态（SSR 不执行数据拉取）', () => {
  useProjectStore.setState({ projectId: 'proj-1' })
  const html = expectRender(AgentTracePanel)
  assert.ok(
    html.includes('该项目还没有 Agent 运行记录') || html.includes('aria-label="Agent 可解释追踪"'),
    '有项目时应越过选择项目空态',
  )
})

test('显式传入 projectId props 时同样越过选择项目空态', () => {
  const html = expectRender(AgentTracePanel, { projectId: 'proj-9' })
  assert.ok(!html.includes('选择项目后显示 Agent 追踪'), 'props 优先于 store')
})

test('compact 模式渲染不抛错', () => {
  expectRender(AgentTracePanel, { compact: true })
})

test('显式指定 projectId 与 runId 渲染不抛错', () => {
  expectRender(AgentTracePanel, { projectId: 'proj-1', runId: 'run-9' })
})
