import type { Shot } from '../stores/shotStore.ts'

/**
 * 服务端 `shot_update` 写入本地 store 前的字段级守卫。
 *
 * 两条铁律：
 * 1. 空媒体路径不允许覆盖已有的有效路径——素材只在「明确的删除 / 恢复」或
 *    「新素材生成成功」时才被替换，后端不会再下发空路径，但旧任务的迟到
 *    响应、异常分支仍可能带上空值，这里兜底。
 * 2. 带版本号且小于当前版本的更新整体丢弃（旧任务响应覆盖新状态的防护）；
 *    版本号大于等于当前、或未携带版本号（历史后端）时按字段级合并放行。
 */

export const SHOT_MEDIA_PATH_FIELDS = [
  'image_path',
  'storyboard_path',
  'video_path',
  'audio_path',
  'last_frame_path',
  'continuity_reference_path',
  'pose_reference_path',
  'depth_reference_path',
] as const

export const SHOT_META_FIELDS = [
  'status',
  'storyboard_status',
  'confirmed',
  'media_stale',
  'duration',
  'estimated_speech_ms',
  'consistency_status',
  'consistency_report',
  'storyboard_reference_manifest',
  'video_reference_manifest',
  'reference_capability_warning',
  'scene_group_id',
  'reference_weights',
  'continuity_profile',
  'quality_review',
] as const

export type ShotServerUpdate = Record<string, unknown>

export function isStaleServerUpdate(current: Shot, incoming: ShotServerUpdate): boolean {
  const incomingVersion = Number(incoming.version ?? 0)
  const currentVersion = Number(current.version || 1)
  return incomingVersion > 0 && currentVersion > 0 && incomingVersion < currentVersion
}

/**
 * 把服务端更新合并成可安全应用到当前镜头的 patch。
 * 返回 null 表示该更新已过期（旧任务响应），必须整体丢弃。
 */
export function mergeShotServerUpdate(
  current: Shot,
  incoming: ShotServerUpdate,
): Partial<Shot> | null {
  if (isStaleServerUpdate(current, incoming)) return null

  const patch: Partial<Shot> = {}

  for (const field of SHOT_MEDIA_PATH_FIELDS) {
    if (!(field in incoming)) continue
    const next = String(incoming[field] ?? '')
    const prev = String((current as unknown as Record<string, unknown>)[field] ?? '')
    // next 为空而当前已有有效路径：保留当前路径（禁止空值覆盖）。
    patch[field] = next || prev
  }

  // 后端口径：storyboard_path 优先，缺省回退 image_path。
  const storyboard = patch.storyboard_path !== undefined
    ? String(patch.storyboard_path)
    : String(current.storyboard_path || '')
  const image = patch.image_path !== undefined
    ? String(patch.image_path)
    : String(current.image_path || '')
  if (!storyboard && image) patch.storyboard_path = image

  for (const field of SHOT_META_FIELDS) {
    if (field in incoming && incoming[field] !== undefined) {
      ;(patch as Record<string, unknown>)[field] = incoming[field]
    }
  }
  if (incoming.version !== undefined) patch.version = Number(incoming.version)

  return patch
}
