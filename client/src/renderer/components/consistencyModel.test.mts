import assert from 'node:assert/strict'
import test from 'node:test'

import {
  blockingReferenceItems,
  consistencyImpactText,
  consistencyReportSummary,
  referenceStatusLabel,
} from './consistencyModel.ts'

test('一致性状态标签不会把失败伪装成正常', () => {
  assert.equal(referenceStatusLabel('failed'), '参考失败')
  assert.equal(referenceStatusLabel('degraded'), '已降级')
  assert.equal(referenceStatusLabel('unsupported'), '能力不支持')
  assert.equal(referenceStatusLabel('stale'), '参考过期')
  assert.equal(referenceStatusLabel('ready'), '参考就绪')
})

test('UI 影响范围包含镜头区间和数量', () => {
  assert.equal(
    consistencyImpactText({
      shot_range: '镜头 2-5',
      affected_shot_ids: ['s2', 's3', 's4', 's5'],
      affected_shot_count: 4,
    }),
    '镜头 2-5 · 影响 4 个镜头',
  )
  assert.equal(
    consistencyReportSummary({
      status: 'degraded',
      shot_range: '镜头 2-5',
      affected_shot_count: 4,
    }),
    '已降级 · 镜头 2-5 · 影响 4 个镜头',
  )
})

test('只有明确确认后的 degraded 不再阻断手动流程', () => {
  const blocking = blockingReferenceItems([
    { asset_id: 'c1', status: 'failed' },
    { asset_id: 's1', status: 'degraded' },
    { asset_id: 's2', status: 'stale' },
    { asset_id: 'c2', status: 'unsupported' },
    { asset_id: 'c3', status: 'ready' },
  ])
  assert.deepEqual(blocking.map((item) => item.asset_id), ['c1', 's2', 'c2'])
})
