import assert from 'node:assert/strict'

import { mergeShotServerUpdate, isStaleServerUpdate, SHOT_MEDIA_PATH_FIELDS } from '../services/shotUpdateGuard.ts'
import { useShotStore, type Shot } from './shotStore.ts'

import { test } from 'vitest'

test('module assertions', async () => {
  function makeShot(overrides: Partial<Shot> = {}): Shot {
    return {
      id: 'shot-1',
      project_id: 'project-1',
      sequence: 1,
      shot_type: 'medium',
      scene_description: '',
      character_action: '',
      dialogue: '',
      camera_angle: '正面',
      camera_movement: '静止',
      duration: 3,
      emotion: 'neutral',
      transition: 'cut',
      visual_notes: '',
      image_path: '/output/old_image.png',
      storyboard_path: '/output/old_storyboard.png',
      video_path: '/output/old_video.mp4',
      audio_path: '/output/old_audio.wav',
      status: 'storyboard_done',
      storyboard_status: 'done',
      version: 5,
      confirmed: false,
      media_stale: false,
      characters_in_scene: [],
      scene_asset_id: '',
      character_asset_ids: [],
      last_frame_path: '/output/old_frame.png',
      ...overrides,
    }
  }

  // 场景 6：WebSocket 返回空媒体路径时，不覆盖已有有效路径。
  {
    const current = makeShot()
    const patch = mergeShotServerUpdate(current, {
      shot_id: 'shot-1',
      version: 6,
      image_path: '',
      storyboard_path: '',
      video_path: '',
      audio_path: '',
      last_frame_path: '',
      status: 'pending',
      storyboard_status: 'queued',
      media_stale: true,
    })
    assert.ok(patch, '带新版本号的更新不应被整体丢弃')
    assert.equal(patch!.image_path, '/output/old_image.png', '空 image_path 不得覆盖有效路径')
    assert.equal(patch!.storyboard_path, '/output/old_storyboard.png', '空 storyboard_path 不得覆盖有效路径')
    assert.equal(patch!.video_path, '/output/old_video.mp4', '空 video_path 不得覆盖有效路径')
    assert.equal(patch!.audio_path, '/output/old_audio.wav', '空 audio_path 不得覆盖有效路径')
    assert.equal(patch!.last_frame_path, '/output/old_frame.png', '空 last_frame_path 不得覆盖有效路径')
    assert.equal(patch!.status, 'pending', '非媒体字段照常透传')
    assert.equal(patch!.media_stale, true, '过期标记照常透传')
  }

  // 新路径非空时正常替换。
  {
    const current = makeShot()
    const patch = mergeShotServerUpdate(current, {
      shot_id: 'shot-1',
      version: 6,
      image_path: '/output/new_image.png',
      storyboard_path: '/output/new_storyboard.png',
    })
    assert.equal(patch!.image_path, '/output/new_image.png')
    assert.equal(patch!.storyboard_path, '/output/new_storyboard.png')
  }

  // 当前也没有素材时维持空串（首次生成前的空更新）。
  {
    const current = makeShot({
      version: 1,
      image_path: '',
      storyboard_path: '',
      video_path: '',
      audio_path: '',
      last_frame_path: '',
    })
    const patch = mergeShotServerUpdate(current, { shot_id: 'shot-1', version: 2, image_path: '', storyboard_path: '' })
    assert.equal(patch!.image_path, '')
  }

  // storyboard_path 为空但 image_path 有新值：按后端口径回退补齐。
  {
    const current = makeShot({ storyboard_path: '', image_path: '' })
    const patch = mergeShotServerUpdate(current, { shot_id: 'shot-1', version: 6, image_path: '/output/fresh.png' })
    assert.equal(patch!.storyboard_path, '/output/fresh.png', 'storyboard 应回退到 image')
  }

  // 场景 7：旧任务返回的 shot_update（版本号回退）不得覆盖新版本素材。
  {
    const current = makeShot({ version: 9, storyboard_path: '/output/new_storyboard.png' })
    const staleUpdate = {
      shot_id: 'shot-1',
      version: 7, // 旧任务持有的 expected_version
      image_path: '/output/stale_image.png',
      storyboard_path: '/output/stale_storyboard.png',
      status: 'storyboard_done',
    }
    assert.equal(isStaleServerUpdate(current, staleUpdate), true, '版本回退应判定为过期更新')
    assert.equal(mergeShotServerUpdate(current, staleUpdate), null, '过期更新必须整体丢弃')

    // 相同版本号（当前状态的重复推送）不视为过期。
    assert.equal(isStaleServerUpdate(current, { version: 9 }), false)
    // 未带版本号的历史负载不视为过期，按字段级规则合并。
    const legacyPatch = mergeShotServerUpdate(current, { shot_id: 'shot-1', image_path: '/output/legacy.png' })
    assert.equal(legacyPatch!.image_path, '/output/legacy.png')
  }

  // store 集成：applyServerShotUpdate 丢弃过期更新、保留有效路径。
  {
    useShotStore.setState({ shots: [makeShot({ version: 9 })] })
    const original = useShotStore.getState().shots[0]!

    useShotStore.getState().applyServerShotUpdate('shot-1', {
      shot_id: 'shot-1',
      version: 8,
      image_path: '',
      storyboard_path: '',
      video_path: '',
    })
    let after = useShotStore.getState().shots[0]!
    assert.equal(after.version, 9, '旧任务响应不得回退版本号')
    assert.equal(after.storyboard_path, original.storyboard_path, '旧任务响应不得覆盖新版本素材')

    useShotStore.getState().applyServerShotUpdate('shot-1', {
      shot_id: 'shot-1',
      version: 10,
      storyboard_path: '/output/final_storyboard.png',
      media_stale: false,
      storyboard_status: 'done',
      estimated_speech_ms: 1250,
      consistency_status: 'degraded',
      quality_review: {
        storyboard: {
          verdict: 'passed',
          passed: true,
          overall_score: 0.91,
          attempt: 1,
          degraded: false,
          issues_count: 0,
          unsupported: [],
        },
        video: null,
      },
    })
    after = useShotStore.getState().shots[0]!
    assert.equal(after.storyboard_path, '/output/final_storyboard.png')
    assert.equal(after.version, 10)
    assert.equal(after.media_stale, false)
    assert.equal(after.estimated_speech_ms, 1250)
    assert.equal(after.consistency_status, 'degraded')
    assert.equal(after.quality_review?.storyboard?.verdict, 'passed')

    // 未知镜头：静默忽略。
    useShotStore.getState().applyServerShotUpdate('shot-missing', { shot_id: 'shot-missing', version: 1 })
  }

  // 媒体字段清单完整性：守卫覆盖全部会被清空的路径字段。
  assert.deepEqual(
    [...SHOT_MEDIA_PATH_FIELDS],
    [
      'image_path',
      'storyboard_path',
      'video_path',
      'audio_path',
      'last_frame_path',
      'continuity_reference_path',
      'pose_reference_path',
      'depth_reference_path',
    ],
  )
})
