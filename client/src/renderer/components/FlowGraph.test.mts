import assert from 'node:assert/strict'
import { test, beforeEach } from 'node:test'

import { expectRender } from '../test-support/ssrTestHelper.mts'
import { useShotStore } from '../stores/shotStore'
import { useProjectStore } from '../stores/projectStore'
import FlowGraph from './FlowGraph'

beforeEach(() => {
  useShotStore.setState({ shots: [], currentStep: '', isGenerating: false, videoPath: '' })
  useProjectStore.setState({ projectId: null })
})

test('无 Agent 追踪数据时回退旧版五节点流程条', () => {
  const html = expectRender(FlowGraph)
  assert.ok(html.includes('aria-label="Agent 执行流程"'), '应有流程条区域标签')
  assert.ok(html.includes('剧本'), '回退步骤条应包含「剧本」')
  assert.ok(html.includes('分镜'), '回退步骤条应包含「分镜」')
})

test('已完成视频时所有回退步骤标记为 done', () => {
  useShotStore.setState({ videoPath: '/output/final.mp4' })
  const html = expectRender(FlowGraph)
  assert.ok(html.includes('done'), '有成片时步骤应为完成态')
  assert.ok(!html.includes('running'), '不应出现运行中步骤')
})

test('生成中当前步骤标记 running，之前的步骤标记 done', () => {
  useShotStore.setState({ currentStep: 'generate_storyboard', isGenerating: true })
  const html = expectRender(FlowGraph)
  assert.ok(html.includes('running'), '当前步骤应为运行中')
  assert.ok(html.includes('done'), '前置步骤应为完成态')
})

test('compact 模式正常渲染', () => {
  const html = expectRender(FlowGraph, { compact: true })
  assert.ok(html.includes('aria-label="Agent 执行流程"'))
})
