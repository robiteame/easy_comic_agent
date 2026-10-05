import assert from 'node:assert/strict'
import { test } from 'node:test'

import { expectRender, visibleText } from '../test-support/ssrTestHelper.mts'
import { makeJob } from '../test-support/fixtures.mts'
import ProgressDebugPanel from './ProgressDebugPanel'

const baseProps = {
  onClose: () => {},
  fallbackStep: '',
  fallbackProgress: 0,
}

test('关闭状态渲染为隐藏（aria-hidden）', () => {
  const html = expectRender(ProgressDebugPanel, { ...baseProps, open: false, job: null })
  assert.ok(html.includes('aria-hidden="true"'), '关闭态面板应标记为隐藏')
})

test('打开且无任务时显示暂无后台任务', () => {
  const html = expectRender(ProgressDebugPanel, { ...baseProps, open: true, job: null })
  assert.ok(html.includes('当前暂无后台任务'), '无任务空态文案')
})

test('有任务时展示任务消息与进度百分比（截断到 0-100）', () => {
  const html = visibleText(
    expectRender(ProgressDebugPanel, {
      ...baseProps,
      open: true,
      job: makeJob({ message: '正在合成第 3 镜', progress: 47 }),
    }),
  )
  assert.ok(html.includes('正在合成第 3 镜'), '应展示任务消息')
  assert.ok(html.includes('47%'), '应展示进度百分比')
})

test('任务进度越界时被夹取到合法区间', () => {
  const html = visibleText(
    expectRender(ProgressDebugPanel, {
      ...baseProps,
      open: true,
      job: makeJob({ progress: 250 }),
    }),
  )
  assert.ok(html.includes('100%'), '超过 100 的进度应夹取为 100%')
})

test('无任务时使用 fallback 进度展示', () => {
  const html = visibleText(
    expectRender(ProgressDebugPanel, {
      ...baseProps,
      open: true,
      job: null,
      fallbackProgress: 33,
    }),
  )
  assert.ok(html.includes('33%'))
})
