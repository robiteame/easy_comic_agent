import assert from 'node:assert/strict'
import { test, beforeEach } from 'vitest'

import { expectRender } from '../test-support/ssrTestHelper.mts'
import { useShotStore } from '../stores/shotStore'
import { useProjectStore } from '../stores/projectStore'
import MainWorkspace from './MainWorkspace'

beforeEach(() => {
  useShotStore.setState({ shots: [], selectedShotId: null, isGenerating: false })
  useProjectStore.getState().reset()
})

test('渲染主工作区骨架', () => {
  const html = expectRender(MainWorkspace)
  assert.ok(html.includes('aria-label="主工作区"'), '应有主工作区区域标签')
})

test('有镜头数据时渲染不抛错', () => {
  const shot = {
    id: 'shot-1',
    project_id: 'proj-1',
    sequence: 1,
    shot_type: 'normal',
    scene_description: '',
    character_action: '',
    dialogue: '',
    camera_angle: '',
    camera_movement: '',
    duration: 3,
    emotion: '',
    transition: '',
    visual_notes: '',
    image_path: '/i.png',
    storyboard_path: '/s.png',
    video_path: '/v.mp4',
    audio_path: '/a.mp3',
    status: 'done',
    storyboard_status: 'approved',
    version: 1,
    confirmed: true,
    characters_in_scene: [],
    scene_asset_id: '',
    character_asset_ids: [],
  }
  useShotStore.setState({ shots: [shot as never], selectedShotId: 'shot-1' })
  expectRender(MainWorkspace)
})
