import assert from 'node:assert/strict'
import { test } from 'vitest'

import { getStepLabel, normalizeShot } from './mainWorkspaceModel.ts'

test('normalizeShot 对缺失字段使用稳定兜底值', () => {
  const shot = normalizeShot({ shot_id: 'shot-1', sequence: '2', duration: '4.5' })
  assert.equal(shot.id, 'shot-1')
  assert.equal(shot.sequence, 2)
  assert.equal(shot.duration, 4.5)
  assert.equal(shot.shot_type, 'medium')
  assert.deepEqual(shot.reference_weights, {})
})

test('getStepLabel 映射已知步骤并兜底处理中', () => {
  assert.equal(getStepLabel('generate_storyboard'), '分镜生成')
  assert.equal(getStepLabel('unknown-step'), '处理中')
  assert.equal(getStepLabel(undefined), '处理中')
})
