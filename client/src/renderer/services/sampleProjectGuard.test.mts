import assert from 'node:assert/strict'
import { test } from 'vitest'

import { mergeSampleDisabled, sampleActionDisabledReason } from './sampleProjectGuard'

// 示例项目守卫：三类依赖外部 Key 的操作（故事板/视频/配音）置灰并给出
// 原因；普通项目不受影响。

test('非示例项目：所有操作正常可用（返回 null）', () => {
  assert.equal(sampleActionDisabledReason(false, 'storyboard'), null)
  assert.equal(sampleActionDisabledReason(undefined, 'video'), null)
  assert.equal(sampleActionDisabledReason(false, 'audio'), null)
})

test('示例项目：三类操作都返回禁用原因，且原因包含操作名与解释', () => {
  const storyboard = sampleActionDisabledReason(true, 'storyboard')
  const video = sampleActionDisabledReason(true, 'video')
  const audio = sampleActionDisabledReason(true, 'audio')
  assert.ok(storyboard?.includes('故事板'))
  assert.ok(video?.includes('视频'))
  assert.ok(audio?.includes('配音'))
  for (const reason of [storyboard, video, audio]) {
    assert.ok(reason!.includes('示例项目'), '原因必须点明是示例项目的限制')
  }
})

test('mergeSampleDisabled：示例项目禁用优先；普通项目沿用原有 disabled 条件', () => {
  assert.deepEqual(mergeSampleDisabled(true, 'video', false), {
    disabled: true,
    reason: sampleActionDisabledReason(true, 'video'),
  })
  assert.deepEqual(mergeSampleDisabled(false, 'video', true), { disabled: true, reason: null })
  assert.deepEqual(mergeSampleDisabled(false, 'video', false), { disabled: false, reason: null })
})
