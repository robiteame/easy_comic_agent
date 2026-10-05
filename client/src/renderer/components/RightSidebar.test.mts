import assert from 'node:assert/strict'
import { test, beforeEach } from 'node:test'

import { expectRender } from '../test-support/ssrTestHelper.mts'
import { useShotStore } from '../stores/shotStore'
import { useProjectStore } from '../stores/projectStore'
import RightSidebar from './RightSidebar'

beforeEach(() => {
  useShotStore.setState({ shots: [], selectedShotId: null, currentStep: '', isGenerating: false })
  useProjectStore.getState().reset()
})

test('渲染右侧运行信息栏与当前步骤区块', () => {
  const html = expectRender(RightSidebar, { collapsed: false, onToggleCollapsed: () => {} })
  assert.ok(html.includes('aria-label="右侧运行信息"'))
  assert.ok(html.includes('当前步骤'))
  assert.ok(!html.includes('right-sidebar collapsed'), '展开态不应带 collapsed 类')
})

test('collapsed 属性切换折叠类名', () => {
  const html = expectRender(RightSidebar, { collapsed: true, onToggleCollapsed: () => {} })
  assert.ok(html.includes('right-sidebar collapsed'))
})

test('选中镜头后渲染镜头编辑面板', () => {
  const shot = {
    id: 'shot-1',
    project_id: 'proj-1',
    sequence: 1,
    shot_type: 'normal',
    scene_description: '城市夜景',
    character_action: '',
    dialogue: '',
    camera_angle: '',
    camera_movement: '',
    duration: 4.5,
    emotion: '',
    transition: '',
    visual_notes: '',
    image_path: '/i.png',
    storyboard_path: '/s.png',
    video_path: '/v.mp4',
    audio_path: '/a.mp3',
    status: 'done',
    storyboard_status: 'approved',
    version: 3,
    confirmed: true,
    characters_in_scene: [],
    scene_asset_id: '',
    character_asset_ids: [],
  }
  useShotStore.setState({ shots: [shot as never], selectedShotId: 'shot-1' })
  const html = expectRender(RightSidebar, { collapsed: false, onToggleCollapsed: () => {} })
  assert.ok(html.includes('shot-1') || html.includes('镜头'), '应出现选中镜头相关内容')
})
