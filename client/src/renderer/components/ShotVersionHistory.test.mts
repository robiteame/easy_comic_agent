import assert from 'node:assert/strict'
import { test } from 'node:test'

import { expectRender } from '../test-support/ssrTestHelper.mts'
import { makeShot } from '../test-support/fixtures.mts'
import ShotVersionHistory from './ShotVersionHistory'

// 注意：antd Modal 在 SSR（renderToString）下 portal 无法挂载到 document，
// 弹层内容不会出现在输出里，因此这里只做渲染安全冒烟；弹层内的展示逻辑
// 由 shotVersionModel.test.mts 覆盖纯逻辑部分。

const baseProps = {
  onClose: () => {},
  onRestored: () => {},
}

test('关闭状态渲染不抛错', () => {
  const html = expectRender(ShotVersionHistory, { ...baseProps, open: false, shot: null })
  assert.equal(html, '')
})

test('打开且未选中镜头渲染不抛错', () => {
  expectRender(ShotVersionHistory, { ...baseProps, open: true, shot: null })
})

test('选中镜头打开渲染不抛错', () => {
  expectRender(ShotVersionHistory, {
    ...baseProps,
    open: true,
    shot: makeShot({ id: 'shot-9', version: 5 }),
  })
})

test('beforeRestore 钩子可选，未传也能渲染', () => {
  expectRender(ShotVersionHistory, {
    ...baseProps,
    open: true,
    shot: makeShot(),
    beforeRestore: async () => true,
  })
})
