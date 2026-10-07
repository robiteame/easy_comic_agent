import assert from 'node:assert/strict'
import { test } from 'vitest'

import { expectRender } from '../test-support/ssrTestHelper.mts'
import GlobalPlayfulMotion from './GlobalPlayfulMotion'

test('渲染全局动效层不抛错（gsap 只在 effect 中执行）', () => {
  const html = expectRender(GlobalPlayfulMotion)
  assert.ok(typeof html === 'string')
})
