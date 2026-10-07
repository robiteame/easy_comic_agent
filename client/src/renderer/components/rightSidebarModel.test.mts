import assert from 'node:assert/strict'
import { test } from 'vitest'

import { SHOT_AUDIO_MODE_OPTIONS, shotSaveKey, stepLabels } from './rightSidebarModel.ts'

test('RightSidebar 模型提供稳定保存键、步骤标签与音频选项', () => {
  assert.equal(shotSaveKey('project-1', 'shot-2'), 'project-1:shot-2')
  assert.equal(stepLabels.generate_storyboard, '分镜列表')
  assert.equal(SHOT_AUDIO_MODE_OPTIONS[0].value, '')
  assert.equal(SHOT_AUDIO_MODE_OPTIONS[SHOT_AUDIO_MODE_OPTIONS.length - 1].value, 'auto')
})
