/**
 * 示例项目操作守卫：示例项目由固定数据 + 本地占位图构成，凡是要真实调用
 * 外部模型 API 的操作（重新生成故事板 / 生成视频 / 配音）一律置灰并说明
 * 原因——是禁用不是隐藏，用户能看到入口并理解为什么不可用。
 *
 * 判定依据是后端 ``is_sample`` 标记（项目创建时落库），不是标题前缀。
 */

export const SAMPLE_PROJECT_DISABLED_REASON =
  '示例项目由内置数据与本地占位图构成，不调用外部模型。请配置密钥并新建项目体验完整生成流程。'

export type SampleGuardedAction = 'storyboard' | 'video' | 'audio'

const ACTION_LABELS: Record<SampleGuardedAction, string> = {
  storyboard: '重新生成故事板',
  video: '生成视频',
  audio: '配音',
}

/**
 * 返回禁用原因；非示例项目返回 null（操作正常可用）。
 * 独立函数而非仅返回布尔，保证提示文案与置灰逻辑使用同一来源。
 */
export function sampleActionDisabledReason(isSample: boolean | undefined, action: SampleGuardedAction): string | null {
  if (!isSample) return null
  return `示例项目暂不支持「${ACTION_LABELS[action]}」。${SAMPLE_PROJECT_DISABLED_REASON}`
}

/** 合并已有 disabled 条件：示例项目的禁用优先级最高，且提示原因单独给出。 */
export function mergeSampleDisabled(
  isSample: boolean | undefined,
  action: SampleGuardedAction,
  otherwiseDisabled: boolean,
): { disabled: boolean; reason: string | null } {
  const reason = sampleActionDisabledReason(isSample, action)
  if (reason) return { disabled: true, reason }
  return { disabled: otherwiseDisabled, reason: null }
}
