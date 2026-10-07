import assert from 'node:assert/strict'
import { test, beforeEach } from 'vitest'

import { useProjectStore } from './projectStore.ts'

// projectStore 是模块级单例，用 reset 恢复默认值保证用例隔离。
beforeEach(() => {
  useProjectStore.getState().reset()
})

test('初始状态为默认项目字段', () => {
  const state = useProjectStore.getState()
  assert.equal(state.projectId, null)
  assert.equal(state.projectType, 'series')
  assert.equal(state.title, '未命名项目')
  assert.equal(state.style, 'anime')
  assert.equal(state.status, 'draft')
  assert.equal(state.consistencyStatus, 'ready')
  assert.equal(state.outputFormat, '9:16')
  assert.equal(state.resolution, '1080p')
  assert.equal(state.platform, 'douyin')
  assert.equal(state.runMode, 'manual')
  assert.deepEqual(state.characters, [])
  assert.equal(state.isSample, false, '默认非示例项目')
})

test('setProject 支持部分字段合并，未给字段保持原值', () => {
  useProjectStore.getState().setProject({ title: '星际迷航', genre: '科幻' })

  const state = useProjectStore.getState()
  assert.equal(state.title, '星际迷航')
  assert.equal(state.genre, '科幻')
  // 未指定字段保持默认
  assert.equal(state.style, 'anime')
  assert.equal(state.projectId, null)
})

test('setProject 多次调用逐层累积', () => {
  const { setProject } = useProjectStore.getState()
  setProject({ projectId: 'proj-42' })
  setProject({ episodeNumber: 3, projectType: 'episode' })

  const state = useProjectStore.getState()
  assert.equal(state.projectId, 'proj-42')
  assert.equal(state.episodeNumber, 3)
  assert.equal(state.projectType, 'episode')
})

test('reset 恢复全部默认值', () => {
  const { setProject } = useProjectStore.getState()
  setProject({
    projectId: 'proj-x',
    title: '改过的标题',
    status: 'generating',
    runMode: 'auto',
    characters: [{ name: '主角' }],
    consistencyReport: { score: 0.8 },
    isSample: true,
  })

  useProjectStore.getState().reset()

  const state = useProjectStore.getState()
  assert.equal(state.projectId, null)
  assert.equal(state.title, '未命名项目')
  assert.equal(state.status, 'draft')
  assert.equal(state.runMode, 'manual')
  assert.deepEqual(state.characters, [])
  assert.deepEqual(state.consistencyReport, {})
  assert.equal(state.isSample, false, '示例项目标记随 reset 清除')
})
