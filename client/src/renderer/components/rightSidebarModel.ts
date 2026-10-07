/** RightSidebar 的选项、标签与保存队列类型。 */

import type { MediaStaleBackup } from '../services/shotEditGuard'
import type { PendingSaveEntry } from '../services/shotSaveQueue'

// 镜头级音频路径覆盖：空=继承系统设置的全局 audio_mode。
export const SHOT_AUDIO_MODE_OPTIONS = [
  { value: '', label: '继承全局设置' },
  { value: 'tts', label: 'TTS 配音合成' },
  { value: 'native', label: '原生音频（需模型支持）' },
  { value: 'auto', label: '智能 auto' },
]

export const stepLabels: Record<string, string> = {
  generate_script: '剧本生成',
  parse_script: '剧本解析',
  generate_storyboard: '分镜列表',
  wait_asset_confirm: '素材确认',
  generate_storyboard_images: '故事板生成',
  wait_storyboard_approval: '分镜审核',
  phase2_start: '视频阶段',
  generate_voice: '配音生成',
  generate_seedance_video: '单镜视频',
  compose_video: '视频合成',
  quality_check: '结构检查（仅结构，非质量认证）',
  rendering: '导出渲染',
}

export interface RightSidebarProps {
  collapsed: boolean
  onToggleCollapsed: () => void
}

export type ShotSaveEntry = PendingSaveEntry<Record<string, any>> & {
  timer: number | null
  projectId: string
  shotId: string
  // 乐观写入 store 的素材过期标记在保存失败时按此快照回滚（素材路径本身
  // 从不被乐观清空，无需回滚）。
  optimisticMediaStale: MediaStaleBackup | null
  // 保存批次中普通镜头字段是否已成功落库（资产绑定失败不影响该结论）。
  paramsPersisted: boolean
}

export const shotSaveKey = (projectId: string, shotId: string) => `${projectId}:${shotId}`
