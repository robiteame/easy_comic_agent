/**
 * 镜头版本历史的纯逻辑：DTO 归一化、来源/字段文案、A/B 选择规则、差异行
 * 过滤与取值格式化。
 *
 * 这里不碰 DOM、不碰网络，因此可以在 node:test 下直接覆盖。组件只做
 * 「调用纯函数 + 渲染 / 发送请求」，业务判断不散落在 JSX 里。
 */

import { formatDialogueForEditor } from '../services/dialogueTimeline.ts'
import type {
  ShotVersionDiffRow,
  ShotVersionSnapshot,
  ShotVersionSource,
  ShotVersionSummary,
} from '../services/shotVersionTypes'

export const VERSION_SOURCES: ShotVersionSource[] = ['manual_edit', 'regenerate', 'restore', 'import']

export const SOURCE_LABELS: Record<string, string> = {
  manual_edit: '手动编辑',
  regenerate: '重新生成',
  restore: '恢复',
  import: '剧本导入',
}

export const FIELD_LABELS: Record<string, string> = {
  visual_notes: '镜头 Prompt',
  prompt: '生成 Prompt',
  negative_prompt: '负向 Prompt',
  scene_description: '场景描述',
  character_action: '人物动作',
  dialogue: '对白',
  shot_type: '镜头类型',
  camera_angle: '机位角度',
  camera_movement: '运镜方式',
  duration: '时长（秒）',
  emotion: '情绪',
  transition: '转场方式',
  scene_asset_id: '绑定场景资产',
  character_asset_ids: '绑定角色资产',
  characters_in_scene: '场内角色',
  image_path: '图像路径',
  storyboard_path: '故事板路径',
  video_path: '视频路径',
  audio_path: '音频路径',
  last_frame_path: '最后一帧路径',
  status: '镜头状态',
  storyboard_status: '故事板状态',
  scene_group_id: '场景组',
  consistency_context: '一致性上下文',
  reference_weights: '参考权重',
  continuity_profile: '连续性配置',
  continuity_reference_path: '续帧参考',
  pose_reference_path: '骨骼参考',
  depth_reference_path: '深度参考',
  version: '镜头版本号',
}

export const EMPTY_VALUE_TEXT = '（空）'

export function sourceLabel(source: string): string {
  return SOURCE_LABELS[source] || '未知来源'
}

export function fieldLabel(field: string): string {
  return FIELD_LABELS[field] || field
}

export function normalizeVersionSummary(raw: unknown): ShotVersionSummary | null {
  if (!raw || typeof raw !== 'object') return null
  const item = raw as Record<string, unknown>
  if (typeof item.id !== 'string' || !item.id) return null
  const source = VERSION_SOURCES.indexOf(item.source as ShotVersionSource) >= 0
    ? (item.source as ShotVersionSource)
    : 'manual_edit'
  return {
    id: item.id,
    shot_id: String(item.shot_id || ''),
    number: Number(item.number || 0),
    version: Number(item.version || 1),
    source,
    task_id: String(item.task_id || ''),
    parent_version_id: String(item.parent_version_id || ''),
    content_hash: String(item.content_hash || ''),
    created_at: typeof item.created_at === 'string' ? item.created_at : null,
    has_image: Boolean(item.has_image),
    has_video: Boolean(item.has_video),
  }
}

/** 归一化并按版本序号从新到旧排序，脏数据逐条丢弃。 */
export function normalizeVersionList(raw: unknown): ShotVersionSummary[] {
  if (!Array.isArray(raw)) return []
  const items: ShotVersionSummary[] = []
  for (const entry of raw) {
    const normalized = normalizeVersionSummary(entry)
    if (normalized) items.push(normalized)
  }
  return items.sort((a, b) => b.number - a.number)
}

export function formatVersionTime(iso: string | null): string {
  if (!iso) return '时间未知'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '时间未知'
  return date.toLocaleString('zh-CN', { hour12: false })
}

/** 字段取值的展示文案：对象/数组紧凑 JSON，空值显示占位符。
 *
 * 对白字段（结构化列表）按「说话人 | 情绪 | 时间轴 | 台词」逐行展示，
 * 让版本对比能直接看出说话人差异，而不是一串难读的 JSON。
 */
export function formatSnapshotValue(value: unknown, field = ''): string {
  if (value === null || value === undefined || value === '') return EMPTY_VALUE_TEXT
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (field === 'dialogue' && Array.isArray(value)) {
    const text = formatDialogueForEditor(value)
    return text || EMPTY_VALUE_TEXT
  }
  if (typeof value === 'object') {
    const text = JSON.stringify(value)
    return text === '{}' || text === '[]' ? EMPTY_VALUE_TEXT : text
  }
  return String(value)
}

/** 差异行过滤：默认只看变化，可切换到全量字段。 */
export function visibleDiffRows(diff: ShotVersionDiffRow[], showUnchanged: boolean): ShotVersionDiffRow[] {
  if (!Array.isArray(diff)) return []
  if (showUnchanged) return diff
  return diff.filter((row) => row && row.changed)
}

export function changedFieldCount(diff: ShotVersionDiffRow[]): number {
  if (!Array.isArray(diff)) return 0
  return diff.filter((row) => row && row.changed).length
}

/**
 * 恢复按钮的可用性：已审核锁定的镜头禁止恢复；与当前状态一致的版本恢复
 * 没有意义，同样置灰并说明原因。
 */
export function restoreAvailability(
  version: ShotVersionSummary,
  options: { confirmed: boolean; currentVersionId: string | null },
): { allowed: boolean; reason: string } {
  if (options.confirmed) {
    return { allowed: false, reason: '已审核锁定的镜头禁止修改或恢复' }
  }
  if (options.currentVersionId && version.id === options.currentVersionId) {
    return { allowed: false, reason: '该版本与当前状态一致，无需恢复' }
  }
  return { allowed: true, reason: '' }
}

export interface CompareSelection {
  a: string | null
  b: string | null
}

/**
 * A/B 槽位选择：点击已选中的槽位取消选择；同一个版本不允许同时占据两个
 * 槽位（先清掉另一侧）。
 */
export function nextCompareSelection(
  current: CompareSelection,
  slot: 'a' | 'b',
  versionId: string,
): CompareSelection {
  if (slot === 'a') {
    if (current.a === versionId) return { a: null, b: current.b === versionId ? null : current.b }
    return { a: versionId, b: current.b === versionId ? null : current.b }
  }
  if (current.b === versionId) return { a: current.a === versionId ? null : current.a, b: null }
  return { a: current.a === versionId ? null : current.a, b: versionId }
}

/** 打开面板时的默认对比：最新记录为 B，上一条为 A；不足两条则留空。 */
export function defaultCompareSelection(versions: ShotVersionSummary[]): CompareSelection {
  if (!versions.length) return { a: null, b: null }
  return { a: versions[1]?.id ?? null, b: versions[0].id }
}

export function canCompare(selection: CompareSelection): boolean {
  return Boolean(selection.a && selection.b && selection.a !== selection.b)
}

/** 快照的预览媒体：优先定稿故事板，其次图像；有视频则并列展示。 */
export function mediaPreviewOf(snapshot: ShotVersionSnapshot | null | undefined): { image: string; video: string } {
  if (!snapshot || typeof snapshot !== 'object') return { image: '', video: '' }
  const image = String(snapshot.storyboard_path || snapshot.image_path || '')
  const video = String(snapshot.video_path || '')
  return { image, video }
}

/** 恢复确认弹窗的正文：强调追加新版本、历史不可变与审核状态重置。 */
export function restoreConfirmContent(version: ShotVersionSummary): string {
  const when = version.created_at ? `，${formatVersionTime(version.created_at)}` : ''
  return [
    `将把镜头恢复到 v${version.number || '?'}（${sourceLabel(version.source)}${when}）的状态。`,
    '恢复会创建一条新的版本记录，历史版本不会被修改或删除。',
    '恢复后镜头回到待审核状态，需要重新通过故事板审核。',
  ].join('\n')
}
