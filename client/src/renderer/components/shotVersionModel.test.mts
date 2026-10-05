import assert from 'node:assert/strict'

import {
  EMPTY_VALUE_TEXT,
  canCompare,
  changedFieldCount,
  defaultCompareSelection,
  fieldLabel,
  formatSnapshotValue,
  formatVersionTime,
  mediaPreviewOf,
  nextCompareSelection,
  normalizeVersionList,
  normalizeVersionSummary,
  restoreAvailability,
  restoreConfirmContent,
  sourceLabel,
  visibleDiffRows,
} from './shotVersionModel.ts'

function makeVersion(overrides: Record<string, unknown> = {}) {
  const normalized = normalizeVersionSummary({
    id: 'ver-1',
    shot_id: 'shot-1',
    number: 2,
    version: 5,
    source: 'regenerate',
    task_id: 'shot:shot-1:storyboard',
    parent_version_id: 'ver-0',
    content_hash: 'abc',
    created_at: '2026-09-18T08:00:00',
    has_image: true,
    has_video: false,
    ...overrides,
  })
  if (!normalized) throw new Error('normalizeVersionSummary returned null')
  return normalized
}

// --- 归一化 ---------------------------------------------------------------

assert.equal(normalizeVersionSummary(null), null, '空值不产生版本对象')
assert.equal(normalizeVersionSummary({ number: 1 }), null, '缺少 id 的记录必须丢弃')
assert.equal(makeVersion().source, 'regenerate', '合法来源原样保留')
assert.equal(makeVersion({ source: 'mystery' }).source, 'manual_edit', '未知来源回退为 manual_edit')
assert.equal(makeVersion({ has_image: 0 }).has_image, false, '媒体标记按布尔归一')

const ordered = normalizeVersionList([
  makeVersion({ id: 'ver-2', number: 1 }),
  'garbage',
  null,
  makeVersion({ id: 'ver-3', number: 3 }),
  makeVersion({ id: 'ver-1', number: 2 }),
])
assert.equal(ordered.length, 3, '列表里的脏数据应被逐个丢弃')
assert.deepEqual(
  ordered.map((item) => item.number),
  [3, 2, 1],
  '时间线按版本号从新到旧排序',
)
assert.deepEqual(normalizeVersionList('nope'), [], '非数组输入返回空列表')

// --- 文案与格式化 ---------------------------------------------------------

assert.equal(sourceLabel('restore'), '恢复')
assert.equal(sourceLabel('unknown-source'), '未知来源')
assert.equal(fieldLabel('dialogue'), '对白')
assert.equal(fieldLabel('custom_field'), 'custom_field', '未知字段回退为字段名')

assert.equal(formatSnapshotValue(''), EMPTY_VALUE_TEXT)
assert.equal(formatSnapshotValue(null), EMPTY_VALUE_TEXT)
assert.equal(formatSnapshotValue({}), EMPTY_VALUE_TEXT)
assert.equal(formatSnapshotValue([]), EMPTY_VALUE_TEXT)
assert.equal(formatSnapshotValue(true), '是')
assert.equal(formatSnapshotValue(3.5), '3.5')
assert.equal(formatSnapshotValue({ a: 1 }), '{"a":1}')
assert.equal(formatSnapshotValue(['x', 'y']), '["x","y"]')

assert.equal(formatVersionTime(null), '时间未知')
assert.equal(formatVersionTime('not-a-date'), '时间未知')
assert.ok(formatVersionTime('2026-09-18T08:00:00').includes('2026'), '合法时间应包含年份')

// --- 差异行过滤 -----------------------------------------------------------

const diff = [
  { field: 'dialogue', a: 'one', b: 'two', changed: true },
  { field: 'duration', a: 3, b: 3, changed: false },
  { field: 'visual_notes', a: 'p1', b: 'p2', changed: true },
]

assert.equal(visibleDiffRows(diff, false).length, 2, '默认只显示变化字段')
assert.equal(visibleDiffRows(diff, true).length, 3, '可切换为全量字段')
assert.deepEqual(
  visibleDiffRows(diff, false).map((row) => row.field),
  ['dialogue', 'visual_notes'],
)
assert.equal(changedFieldCount(diff), 2)
assert.deepEqual(
  visibleDiffRows(null as unknown as Parameters<typeof visibleDiffRows>[0], false),
  [],
  '空差异输入返回空列表',
)

// --- 恢复可用性 -----------------------------------------------------------

const version = makeVersion()
assert.deepEqual(
  restoreAvailability(version, { confirmed: true, currentVersionId: null }),
  { allowed: false, reason: '已审核锁定的镜头禁止修改或恢复' },
  '已审核锁定：可对比但禁止恢复',
)
assert.equal(
  restoreAvailability(version, { confirmed: false, currentVersionId: version.id }).allowed,
  false,
  '与当前状态一致的版本无需恢复',
)
assert.equal(restoreAvailability(version, { confirmed: false, currentVersionId: 'other' }).allowed, true)

const restoreText = restoreConfirmContent(makeVersion({ created_at: null }))
assert.ok(restoreText.includes('v2'), '确认文案包含版本号')
assert.ok(restoreText.includes('新的版本记录'), '确认文案强调追加新版本')
assert.ok(restoreText.includes('待审核'), '确认文案提示审核状态重置')

// --- A/B 选择 -------------------------------------------------------------

assert.deepEqual(defaultCompareSelection([]), { a: null, b: null }, '无版本时不预选')
assert.deepEqual(
  defaultCompareSelection([makeVersion({ id: 'only' })]),
  {
    a: null,
    b: 'only',
  },
  '仅一条版本时只预选 B',
)
assert.deepEqual(defaultCompareSelection(ordered), { a: ordered[1].id, b: ordered[0].id }, '默认对比最新两条')

assert.deepEqual(nextCompareSelection({ a: null, b: 'x' }, 'a', 'y'), { a: 'y', b: 'x' }, '选择空槽位直接生效')
assert.deepEqual(nextCompareSelection({ a: 'y', b: 'x' }, 'a', 'y'), { a: null, b: 'x' }, '再次点击同槽位取消选择')
assert.deepEqual(
  nextCompareSelection({ a: 'y', b: null }, 'b', 'y'),
  { a: null, b: 'y' },
  '同一版本不能同时占据 A/B 两个槽位',
)
assert.deepEqual(nextCompareSelection({ a: 'y', b: null }, 'b', 'z'), { a: 'y', b: 'z' }, '不同版本互不影响')

assert.equal(canCompare({ a: null, b: 'x' }), false)
assert.equal(canCompare({ a: 'x', b: 'x' }), false)
assert.equal(canCompare({ a: 'x', b: 'y' }), true)

// --- 媒体预览 -------------------------------------------------------------

assert.deepEqual(
  mediaPreviewOf({ storyboard_path: 'sb.png', image_path: 'img.png', video_path: 'clip.mp4' }),
  { image: 'sb.png', video: 'clip.mp4' },
  '图像预览优先定稿故事板',
)
assert.deepEqual(mediaPreviewOf({ image_path: 'img.png' }), { image: 'img.png', video: '' })
assert.deepEqual(mediaPreviewOf(null), { image: '', video: '' })

console.log('shotVersionModel tests passed')
