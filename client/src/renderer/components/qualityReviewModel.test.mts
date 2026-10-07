import assert from 'node:assert/strict'
import { test } from 'vitest'

import {
  dimensionStatusMeta,
  formatScore,
  gateLine,
  hasUndetected,
  historyEntries,
  latestByStage,
  qualityBadgeFor,
  verdictMeta,
  type QualityReviewRow,
} from './qualityReviewModel.ts'

function reviewRow(overrides: Partial<QualityReviewRow> = {}): QualityReviewRow {
  return {
    id: 'qr_1',
    shot_id: 'shot_1',
    project_id: 'proj_1',
    stage: 'storyboard',
    attempt: 1,
    shot_version: 3,
    target_path: '/tmp/shot.png',
    verdict: 'passed',
    passed: true,
    overall_score: 0.86,
    degraded: false,
    dimensions: [],
    issues: [],
    unsupported_dimensions: [],
    suggestion: '',
    prompt_fix: {},
    gate_policy: 'threshold=0.75 policy=strict',
    created_at: '2026-09-30T12:00:00',
    ...overrides,
  }
}

test('verdictMeta maps every verdict to an honest Chinese label', () => {
  assert.equal(verdictMeta('passed').label, '质量通过')
  assert.equal(verdictMeta('failed').label, '质量未通过')
  assert.equal(verdictMeta('unsupported').label, '未检测（能力未配置）')
  assert.equal(verdictMeta('error').label, '审核出错')
  // 未知值不冒充通过。
  assert.equal(verdictMeta('weird').tone, 'error')
})

test('dimensionStatusMeta marks unsupported as 未检测, never as pass', () => {
  assert.equal(dimensionStatusMeta('scored').label, '已检测')
  assert.equal(dimensionStatusMeta('unsupported').label, '未检测')
  assert.equal(dimensionStatusMeta('skipped').label, '不适用')
  assert.equal(dimensionStatusMeta('error').label, '检测失败')
})

test('formatScore renders null as dash and normalizes to percent', () => {
  assert.equal(formatScore(null), '—')
  assert.equal(formatScore(undefined), '—')
  assert.equal(formatScore(0.862), '86')
  assert.equal(formatScore(1), '100')
})

test('gateLine describes threshold, policy and retries', () => {
  const line = gateLine({ threshold: 0.75, policy: 'strict', storyboard_max_retries: 2, video_max_retries: 1 })
  assert.match(line, /通过阈值 75 分/)
  assert.match(line, /严格/)
  assert.match(line, /故事板最多重试 2 次 \/ 视频 1 次/)
})

test('hasUndetected is true for degraded rows or rows listing unsupported dimensions', () => {
  assert.equal(hasUndetected(reviewRow({ degraded: true })), true)
  assert.equal(hasUndetected(reviewRow({ unsupported_dimensions: ['角色身份一致性'] })), true)
  assert.equal(hasUndetected(reviewRow()), false)
})

test('latestByStage picks the newest row per stage', () => {
  const rows = [
    reviewRow({ id: 'a', stage: 'storyboard', attempt: 2 }),
    reviewRow({ id: 'b', stage: 'storyboard', attempt: 1 }),
    reviewRow({ id: 'c', stage: 'video', attempt: 1 }),
  ]
  const latest = latestByStage(rows)
  assert.equal(latest.storyboard?.id, 'a')
  assert.equal(latest.video?.id, 'c')
  assert.equal(latestByStage([]).storyboard, null)
})

test('qualityBadgeFor surfaces unsupported before pass and null when unreviewed', () => {
  assert.equal(qualityBadgeFor(null), null)
  assert.equal(qualityBadgeFor({}), null)

  const unsupported = qualityBadgeFor({
    storyboard: {
      verdict: 'unsupported',
      passed: false,
      overall_score: 0.8,
      attempt: 1,
      degraded: false,
      issues_count: 1,
      unsupported: ['角色身份一致性'],
    },
  })
  assert.equal(unsupported?.label, '未检测')
  assert.match(unsupported?.title || '', /角色身份一致性/)

  const failed = qualityBadgeFor({
    storyboard: {
      verdict: 'failed',
      passed: false,
      overall_score: 0.4,
      attempt: 1,
      degraded: false,
      issues_count: 2,
      unsupported: [],
    },
  })
  assert.equal(failed?.label, '质量未通过')

  const passed = qualityBadgeFor({
    storyboard: {
      verdict: 'passed',
      passed: true,
      overall_score: 0.9,
      attempt: 1,
      degraded: false,
      issues_count: 0,
      unsupported: [],
    },
  })
  assert.equal(passed?.label, '质检通过')
})

test('historyEntries lists every attempt with stage and degraded flag', () => {
  const entries = historyEntries([
    reviewRow({ id: 'b', stage: 'video', attempt: 1, degraded: true, overall_score: 0.7 }),
    reviewRow({ id: 'a', stage: 'storyboard', attempt: 2, overall_score: 0.86 }),
  ])
  assert.equal(entries.length, 2)
  assert.equal(entries[0].title, '第 1 轮 · 视频')
  assert.equal(entries[0].degraded, true)
  assert.equal(entries[1].title, '第 2 轮 · 故事板')
  assert.equal(entries[1].score, '86')
})

if (import.meta.url === `file://${process.argv[1]}`) {
  // node --test 直接执行本文件时无需额外入口。
}
