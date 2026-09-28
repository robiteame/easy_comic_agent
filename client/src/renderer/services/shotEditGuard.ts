import { SHOT_MEDIA_PATH_FIELDS } from './shotUpdateGuard.ts'

/**
 * 镜头编辑写入本地 store 的守卫：纯函数，保证两条不变量可被测试锁定——
 *
 * 1. 参数编辑的乐观更新永远不包含媒体路径字段（旧素材路径不可能被编辑
 *    或保存失败提前清空）；
 * 2. 保存失败只回滚乐观写入的过期标记，同样不触碰媒体路径。
 */

export interface MediaStaleBackup {
  media_stale: boolean
}

/** 编辑参数时的乐观 store patch：字段新值 + 素材待重新生成标记。
 *
 * ``preEditMediaStale`` 为编辑前镜头的 media_stale（首次编辑时作为回滚
 * 快照），``existingBackup`` 为同一保存队列中已记录的快照（保留最早值）。
 */
export function optimisticShotFieldPatch(
  field: string,
  value: unknown,
  preEditMediaStale: boolean,
  existingBackup: MediaStaleBackup | null,
): {
  patch: Record<string, unknown>
  backup: MediaStaleBackup
} {
  return {
    patch: { [field]: value, media_stale: true },
    backup: existingBackup ?? { media_stale: Boolean(preEditMediaStale) },
  }
}

/** 保存失败时回滚乐观标记（素材路径从未被改动，无需也不得回滚）。 */
export function saveFailureRollbackPatch(backup: MediaStaleBackup | null): Record<string, unknown> {
  if (!backup) return {}
  return { media_stale: backup.media_stale }
}

/** 校验一个 patch 是否触碰媒体路径字段（测试与防御性断言用）。 */
export function touchesMediaPaths(patch: Record<string, unknown>): boolean {
  return SHOT_MEDIA_PATH_FIELDS.some((field) => field in patch)
}
