import assert from 'node:assert/strict'

import { optimisticShotFieldPatch, saveFailureRollbackPatch, touchesMediaPaths } from './shotEditGuard.ts'
import { drainPendingSaves, type PendingSaveEntry } from './shotSaveQueue.ts'

function makeShot() {
  return {
    id: 'shot-1',
    visual_notes: '旧 Prompt',
    image_path: '/output/old_image.png',
    storyboard_path: '/output/old_storyboard.png',
    video_path: '/output/old_video.mp4',
    audio_path: '/output/old_audio.wav',
    last_frame_path: '/output/old_frame.png',
    confirmed: false,
    media_stale: false,
    version: 5,
  }
}

// 场景 2：编辑 + 保存失败后，前端 store 中原素材路径不丢失。
{
  const shot = makeShot()
  let backup = null as ReturnType<typeof optimisticShotFieldPatch>['backup'] | null

  // 1) 用户编辑 Prompt：乐观 patch 只含字段值与过期标记，绝不含媒体路径。
  const first = optimisticShotFieldPatch('visual_notes', '新 Prompt', shot.media_stale, backup)
  backup = first.backup
  assert.equal(touchesMediaPaths(first.patch), false, '编辑的乐观更新不得触碰媒体路径')
  assert.deepEqual(first.patch, { visual_notes: '新 Prompt', media_stale: true })
  Object.assign(shot, first.patch)

  // 2) 保存请求失败：回滚 patch 同样不触碰媒体路径，只还原过期标记。
  const rollback = saveFailureRollbackPatch(backup)
  assert.equal(touchesMediaPaths(rollback), false, '保存失败的回滚不得触碰媒体路径')
  Object.assign(shot, rollback)
  assert.equal(shot.media_stale, false, '保存失败后过期标记回到编辑前的值')

  // 3) 素材路径全程未变。
  assert.equal(shot.image_path, '/output/old_image.png')
  assert.equal(shot.storyboard_path, '/output/old_storyboard.png')
  assert.equal(shot.video_path, '/output/old_video.mp4')
  assert.equal(shot.audio_path, '/output/old_audio.wav')
  assert.equal(shot.last_frame_path, '/output/old_frame.png')
}

// 首次编辑前素材已是过期态：失败回滚保留 true（而不是清掉真实过期标记）。
{
  const { backup } = optimisticShotFieldPatch('dialogue', '新台词', true, null)
  const rollback = saveFailureRollbackPatch(backup)
  assert.equal(rollback.media_stale, true)
}

// 已有备份时后续编辑沿用最早的快照值。
{
  const first = optimisticShotFieldPatch('visual_notes', 'a', false, null)
  const second = optimisticShotFieldPatch('dialogue', 'b', true, first.backup)
  assert.equal(second.backup.media_stale, false, '回滚快照必须保留最早（编辑前）的值')
}

// 与保存队列联动：失败批次重新入队后再次失败，回滚 patch 依旧安全。
{
  const shot = makeShot()
  const { patch, backup } = optimisticShotFieldPatch('visual_notes', '新 Prompt', shot.media_stale, null)
  Object.assign(shot, patch)

  const entry: PendingSaveEntry<Record<string, unknown>> = {
    pending: { visual_notes: '新 Prompt' },
    inFlight: false,
    failed: false,
    promise: null,
    retired: false,
  }
  let attempts = 0
  const saved = drainPendingSaves(entry, async (changes) => {
    attempts += 1
    // 两次保存都失败：模拟网络错误，队列会把改动放回 pending。
    throw new Error('网络不可用 ' + Object.keys(changes).join(','))
  })
  assert.equal(await saved, false)
  assert.equal(attempts, 1)
  assert.equal(entry.failed, true)
  assert.deepEqual(entry.pending, { visual_notes: '新 Prompt' }, '失败的编辑必须保留在队列中，不能丢')

  Object.assign(shot, saveFailureRollbackPatch(backup))
  assert.equal(shot.media_stale, false)
  assert.equal(shot.image_path, '/output/old_image.png', '保存失败后素材路径仍在')
}
