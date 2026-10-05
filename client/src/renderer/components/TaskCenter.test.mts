import { test, beforeEach } from 'node:test'

import { expectRender } from '../test-support/ssrTestHelper.mts'
import { useTaskStore } from '../stores/taskStore'
import { makeJob } from '../test-support/fixtures.mts'
import TaskCenter from './TaskCenter'

function emptySummary() {
  return {
    activeCount: 0,
    failedCount: 0,
    interruptedCount: 0,
    retryableCount: 0,
    total: 0,
    latest: null,
    failedByCategory: [],
  }
}

beforeEach(() => {
  useTaskStore.setState({ jobs: [], visibleJobs: [], summary: emptySummary(), loading: false })
})

test('关闭状态渲染不抛错', () => {
  expectRender(TaskCenter, { open: false, onClose: () => {} })
})

test('打开状态渲染不抛错（弹层内容受 SSR portal 限制）', () => {
  expectRender(TaskCenter, { open: true, onClose: () => {} })
})

test('注入任务列表后渲染不抛错', () => {
  const jobs = [makeJob(), makeJob({ id: 'job-2', status: 'failed' as never, is_active: false })]
  useTaskStore.setState({ jobs })
  expectRender(TaskCenter, { open: true, onClose: () => {} })
})
