import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildTrackClips,
  clampMs,
  computeLocalWarnings,
  cueAtPlayhead,
  formatTimelineMs,
  groupTracksByKind,
  msToPx,
  parseMsInput,
  pxToMs,
  pushUndo,
  snapMs,
  undoOnce,
  validateCueDraft,
} from './avWorkbenchModel.ts'
import type { AudioTrackDto, AvShotInfo, SubtitleCueDto } from '../services/api.ts'

function makeTrack(overrides: Partial<AudioTrackDto> & { id: string; kind: AudioTrackDto['kind'] }): AudioTrackDto {
  return {
    project_id: 'p1',
    name: overrides.id,
    source_path: '/output/projects/p1/audio_tracks/a.mp3',
    source_url: '/output/projects/p1/audio_tracks/a.mp3',
    source_duration_ms: 30_000,
    shot_id: '',
    start_ms: 0,
    volume: 1,
    pan: 0,
    fade_in_ms: 0,
    fade_out_ms: 0,
    delay_ms: 0,
    trim_start_ms: 0,
    trim_end_ms: 0,
    loop: false,
    muted: false,
    duck_amount_db: 0,
    duck_attack_ms: 120,
    duck_release_ms: 480,
    order_index: 0,
    ...overrides,
  }
}

function makeShot(id: string, startMs: number, durationMs: number, extra: Partial<AvShotInfo> = {}): AvShotInfo {
  return {
    id,
    sequence: 1,
    dialogue: '',
    character_name: '',
    has_tts: true,
    native_audio: false,
    start_ms: startMs,
    end_ms: startMs + durationMs,
    duration_ms: durationMs,
    ...extra,
  }
}

function makeCue(start: number, end: number, text = '字幕'): SubtitleCueDto {
  return { id: `${start}`, start_ms: start, end_ms: end, text, character_name: '' }
}

test('时间线换算：ms 与像素互转且不产生负值', () => {
  assert.equal(msToPx(1500, 40), 60)
  assert.equal(pxToMs(60, 40), 1500)
  assert.equal(msToPx(-5, 40), 0)
  assert.equal(pxToMs(-3, 40), 0)
  assert.equal(pxToMs(10, 0), 0)
})

test('吸附：靠近候选值时吸附，否则保持原值', () => {
  assert.equal(snapMs(1005, [1000, 2000]), 1000)
  assert.equal(snapMs(1600, [1000, 2000], 120), 1600)
  assert.equal(clampMs(500, 1000, 2000), 1000)
  assert.equal(clampMs(2500, 1000, 2000), 2000)
})

test('时间码格式化', () => {
  assert.equal(formatTimelineMs(65_430), '01:05.4')
  assert.equal(formatTimelineMs(0), '00:00.0')
})

test('轨道片段：对白轨使用镜头区间，普通轨含延迟，循环轨铺满全片', () => {
  const clips = buildTrackClips(
    [
      makeTrack({
        id: 'd1',
        kind: 'dialogue',
        shot_id: 's2',
        shot_span: { start_ms: 2000, end_ms: 5000 },
        delay_ms: 100,
        source_duration_ms: 2500,
      }),
      makeTrack({ id: 'm1', kind: 'music', start_ms: 1000, delay_ms: 500 }),
      makeTrack({ id: 'a1', kind: 'ambient', loop: true, start_ms: 0 }),
    ],
    10_000,
  )
  const byId = new Map(clips.map((clip) => [clip.track.id, clip]))
  assert.equal(byId.get('d1')?.start_ms, 2100)
  assert.equal(byId.get('m1')?.start_ms, 1500)
  assert.equal(byId.get('m1')?.duration_ms, 30_000)
  assert.equal(byId.get('a1')?.duration_ms, 10_000)
  assert.equal(byId.get('a1')?.outOfRange, false)
})

test('越界轨道被标记 outOfRange', () => {
  const clips = buildTrackClips([makeTrack({ id: 'late', kind: 'sfx', start_ms: 12_000 })], 10_000)
  assert.equal(clips[0].outOfRange, true)
})

test('本地告警：静音、无源、越界、高增益、重叠、无伴奏', () => {
  const tracks = [
    makeTrack({ id: 'm1', kind: 'music', muted: true }),
    makeTrack({ id: 'broken', kind: 'sfx', source_path: '', source_duration_ms: 0 }),
    makeTrack({ id: 'hot', kind: 'ambient', volume: 1.8, start_ms: 12_000 }),
    makeTrack({ id: 'ov1', kind: 'music', start_ms: 0 }),
    makeTrack({ id: 'ov2', kind: 'music', start_ms: 1000 }),
  ]
  const clips = buildTrackClips(tracks, 10_000)
  const warnings = computeLocalWarnings({
    tracks,
    clips,
    shotInfos: [makeShot('s1', 0, 5000), makeShot('s2', 5000, 5000)],
    cues: [],
    totalDurationMs: 10_000,
  })
  const codes = new Set(warnings.map((item) => item.code))
  assert.ok(codes.has('muted'))
  assert.ok(codes.has('source_missing'))
  assert.ok(codes.has('hot_gain'))
  assert.ok(codes.has('out_of_range'))
  assert.ok(codes.has('overlap'))
  // m1 静音后仍有 ov1/ov2 两条 music 轨，no_bed 不应出现。
  assert.ok(!codes.has('no_bed'))
})

test('本地告警：对白轨缺配音 / 未绑定；全部静音时提示无伴奏', () => {
  const tracks = [
    makeTrack({ id: 'd1', kind: 'dialogue', shot_id: 's1', shot_span: { start_ms: 0, end_ms: 5000 } }),
    makeTrack({ id: 'd2', kind: 'dialogue', shot_id: 'missing', shot_span: undefined }),
  ]
  const warnings = computeLocalWarnings({
    tracks,
    clips: buildTrackClips(tracks, 10_000),
    shotInfos: [makeShot('s1', 0, 5000, { has_tts: false })],
    cues: [],
    totalDurationMs: 10_000,
  })
  const codes = new Set(warnings.map((item) => item.code))
  assert.ok(codes.has('dialogue_no_tts'))
  assert.ok(codes.has('dialogue_unbound'))
  assert.ok(codes.has('no_bed'))
})

test('本地告警：字幕越界与重叠', () => {
  const warnings = computeLocalWarnings({
    tracks: [makeTrack({ id: 'm1', kind: 'music' })],
    clips: [],
    shotInfos: [],
    cues: [makeCue(0, 900), makeCue(800, 11_000)],
    totalDurationMs: 10_000,
  })
  const codes = new Set(warnings.map((item) => item.code))
  assert.ok(codes.has('subtitle_beyond'))
  assert.ok(codes.has('subtitle_overlap'))
})

test('播放头命中字幕', () => {
  const cues = [makeCue(1000, 2000), makeCue(3000, 4000)]
  assert.equal(cueAtPlayhead(cues, 1500)?.text, '字幕')
  assert.equal(cueAtPlayhead(cues, 2500), null)
})

test('撤销栈：固定深度、undo 恢复上一步', () => {
  let stack = pushUndo([], 'init', 'a')
  stack = pushUndo(stack, 'edit', 'b')
  stack = pushUndo(stack, 'edit', 'c')
  assert.equal(stack.length, 3)
  let restored: string | null = null
  const step1 = undoOnce(stack, (state) => {
    restored = state
    return state
  })
  assert.equal(restored, 'b')
  assert.equal(step1.stack.length, 2)
  const empty = undoOnce([], (state) => state)
  assert.equal(empty.result, null)

  let deep: typeof stack = []
  for (let i = 0; i < 60; i += 1) deep = pushUndo(deep, `e${i}`, `v${i}`)
  assert.equal(deep.length, 40)
})

test('字幕草稿校验与服务端规则同义', () => {
  assert.equal(validateCueDraft({ text: '正常', start_ms: 0, end_ms: 1000 }), null)
  assert.match(validateCueDraft({ text: '   ', start_ms: 0, end_ms: 100 }) || '', /不能为空/)
  assert.match(validateCueDraft({ text: '带\x02控制符', start_ms: 0, end_ms: 100 }) || '', /非法/)
  assert.match(validateCueDraft({ text: '倒序', start_ms: 1000, end_ms: 900 }) || '', /结束时间/)
  assert.match(validateCueDraft({ text: '过短', start_ms: 0, end_ms: 30 }) || '', /50 毫秒/)
})

test('轨道分道：按类型分组且保持顺序', () => {
  const groups = groupTracksByKind([
    makeTrack({ id: 'm1', kind: 'music', order_index: 2 }),
    makeTrack({ id: 'd1', kind: 'dialogue', order_index: 0 }),
    makeTrack({ id: 'm2', kind: 'music', order_index: 3 }),
    makeTrack({ id: 'a1', kind: 'ambient', order_index: 1 }),
  ])
  assert.deepEqual(
    groups.map((group) => [group.kind, group.items.map((item) => item.id)]),
    [
      ['dialogue', ['d1']],
      ['music', ['m1', 'm2']],
      ['ambient', ['a1']],
    ],
  )
})

test('毫秒输入解析：非法回退默认值', () => {
  assert.equal(parseMsInput('1500', 0), 1500)
  assert.equal(parseMsInput('-3', 100), 100)
  assert.equal(parseMsInput('abc', 100), 100)
  assert.equal(parseMsInput('999999999999', 0), 86_400_000)
})
