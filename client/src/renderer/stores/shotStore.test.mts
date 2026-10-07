import assert from 'node:assert/strict'
import { test, beforeEach } from 'vitest'

import { useShotStore, type Shot } from './shotStore.ts'

function makeShot(overrides: Partial<Shot> = {}): Shot {
  return {
    id: 'shot-1',
    project_id: 'proj-1',
    sequence: 1,
    shot_type: 'normal',
    scene_description: '城市夜景',
    character_action: '主角走在街上',
    dialogue: '',
    camera_angle: 'wide',
    camera_movement: 'static',
    duration: 4.5,
    emotion: 'calm',
    transition: 'cut',
    visual_notes: '',
    image_path: '/output/images/shot-1.png',
    storyboard_path: '/output/storyboards/shot-1.png',
    video_path: '/output/videos/shot-1.mp4',
    audio_path: '/output/audio/shot-1.mp3',
    status: 'done',
    storyboard_status: 'approved',
    version: 3,
    confirmed: true,
    characters_in_scene: [],
    scene_asset_id: '',
    character_asset_ids: [],
    ...overrides,
  }
}

beforeEach(() => {
  useShotStore.setState({
    shots: [],
    selectedShotId: null,
    isGenerating: false,
    progress: 0,
    currentStep: '',
    awaitingStoryboardConfirm: false,
    videoPath: '',
    logs: [],
  })
})

test('setShots 整体替换镜头列表', () => {
  const { setShots } = useShotStore.getState()
  setShots([makeShot(), makeShot({ id: 'shot-2', sequence: 2 })])
  assert.equal(useShotStore.getState().shots.length, 2)
  assert.equal(useShotStore.getState().shots[1].id, 'shot-2')
})

test('updateShot 仅合并目标镜头字段，不影响其他镜头', () => {
  const { setShots, updateShot } = useShotStore.getState()
  setShots([makeShot(), makeShot({ id: 'shot-2', status: 'pending' })])

  updateShot('shot-2', { status: 'running', duration: 8 })

  const second = useShotStore.getState().shots.find((s) => s.id === 'shot-2')
  assert.equal(second?.status, 'running')
  assert.equal(second?.duration, 8)
  // 未指定的字段保持原值
  assert.equal(second?.version, 3)
  // 第一个镜头不受影响
  assert.equal(useShotStore.getState().shots[0].status, 'done')
})

test('updateShot 对未知 id 是无操作', () => {
  const { setShots, updateShot } = useShotStore.getState()
  setShots([makeShot()])
  updateShot('not-exist', { status: 'x' })
  assert.equal(useShotStore.getState().shots[0].status, 'done')
})

test('addShot 追加到列表末尾，removeShot 按 id 删除', () => {
  const { setShots, addShot, removeShot } = useShotStore.getState()
  setShots([makeShot()])
  addShot(makeShot({ id: 'shot-2', sequence: 2 }))

  assert.equal(useShotStore.getState().shots.length, 2)
  assert.equal(useShotStore.getState().shots[1].id, 'shot-2')

  removeShot('shot-1')
  assert.deepEqual(
    useShotStore.getState().shots.map((s) => s.id),
    ['shot-2'],
  )
})

test('applyServerShotUpdate 空媒体路径不覆盖已有有效素材', () => {
  const { setShots, applyServerShotUpdate } = useShotStore.getState()
  setShots([makeShot()])

  applyServerShotUpdate('shot-1', {
    version: 5,
    video_path: '',
    image_path: '',
    audio_path: null,
    status: 'regenerating',
  })

  const shot = useShotStore.getState().shots[0]
  assert.equal(shot.video_path, '/output/videos/shot-1.mp4')
  assert.equal(shot.image_path, '/output/images/shot-1.png')
  // 非媒体字段照常合并
  assert.equal(shot.status, 'regenerating')
})

test('applyServerShotUpdate 旧版本号更新整体丢弃', () => {
  const { setShots, applyServerShotUpdate } = useShotStore.getState()
  setShots([makeShot({ version: 7, status: 'done' })])

  applyServerShotUpdate('shot-1', { version: 3, status: 'stale-status' })

  const shot = useShotStore.getState().shots[0]
  assert.equal(shot.status, 'done', '旧任务回包必须被整体丢弃')
})

test('applyServerShotUpdate 新版本号更新合并生效', () => {
  const { setShots, applyServerShotUpdate } = useShotStore.getState()
  setShots([makeShot({ version: 3 })])

  applyServerShotUpdate('shot-1', {
    version: 4,
    video_path: '/output/videos/shot-1-v2.mp4',
  })

  assert.equal(useShotStore.getState().shots[0].video_path, '/output/videos/shot-1-v2.mp4')
})

test('applyServerShotUpdate 未知镜头 id 是无操作', () => {
  const { setShots, applyServerShotUpdate } = useShotStore.getState()
  setShots([makeShot()])
  applyServerShotUpdate('ghost', { status: 'x' })
  assert.equal(useShotStore.getState().shots[0].status, 'done')
})

test('appendLog 累积追加且上限 100 条（滚动丢弃最旧的）', () => {
  const { appendLog } = useShotStore.getState()

  for (let i = 1; i <= 105; i += 1) {
    appendLog(`line-${i}`)
  }

  const { logs } = useShotStore.getState()
  assert.equal(logs.length, 100)
  assert.equal(logs[0], 'line-6', '最旧的 5 条被滚动丢弃')
  assert.equal(logs[logs.length - 1], 'line-105')

  useShotStore.getState().clearLogs()
  assert.deepEqual(useShotStore.getState().logs, [])
})

test('reorderShots 按 id 列表重排并忽略未知 id', () => {
  const { setShots, reorderShots } = useShotStore.getState()
  setShots([makeShot(), makeShot({ id: 'shot-2', sequence: 2 }), makeShot({ id: 'shot-3', sequence: 3 })])

  reorderShots(['shot-3', 'shot-1', 'ghost-id', 'shot-2'])

  assert.deepEqual(
    useShotStore.getState().shots.map((s) => s.id),
    ['shot-3', 'shot-1', 'shot-2'],
  )
})

test('selectShot / setGenerating / setProgress / setAwaitingStoryboardConfirm / setVideoPath 状态写入', () => {
  const state = useShotStore.getState()
  state.selectShot('shot-9')
  state.setGenerating(true)
  state.setProgress(66, 'generate_storyboard')
  state.setAwaitingStoryboardConfirm(true)
  state.setVideoPath('/output/final.mp4')

  const next = useShotStore.getState()
  assert.equal(next.selectedShotId, 'shot-9')
  assert.equal(next.isGenerating, true)
  assert.equal(next.progress, 66)
  assert.equal(next.currentStep, 'generate_storyboard')
  assert.equal(next.awaitingStoryboardConfirm, true)
  assert.equal(next.videoPath, '/output/final.mp4')
})
