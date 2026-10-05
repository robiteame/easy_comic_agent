import { test } from 'node:test'

import { expectRender } from '../test-support/ssrTestHelper.mts'
import TaskEstimateModal from './TaskEstimateModal'

// antd Modal 在 SSR 下 portal 不输出内容，这里做渲染安全冒烟；
// 估算文案与拦截原因的纯逻辑由其他用例与后端契约保证。

const baseProps = {
  onCancel: () => {},
  onConfirm: () => {},
}

test('关闭且无请求时渲染不抛错', () => {
  expectRender(TaskEstimateModal, { ...baseProps, open: false, request: null })
})

test('打开且无请求时渲染不抛错', () => {
  expectRender(TaskEstimateModal, { ...baseProps, open: true, request: null })
})

test('打开且有估算请求时渲染不抛错', () => {
  expectRender(TaskEstimateModal, {
    ...baseProps,
    open: true,
    request: {
      job_type: 'generate_storyboard',
      project_id: 'proj-1',
      entryLabel: '生成故事板',
    },
  })
})
