/** MainWorkspace 的纯状态类型、常量与归一化函数。 */

export type StyleOption = { value: string; label: string; keywords?: string; custom?: boolean }

export const STEP_LABELS: Record<string, string> = {
  start: '开始',
  parse_script: '剧本解析',
  generate_storyboard: '分镜生成',
  wait_asset_confirm: '等待素材确认',
  generate_storyboard_images: '故事板生成',
  wait_storyboard_approval: '等待故事板审核',
  phase2_start: '进入第二阶段',
  generate_voice: '配音生成',
  generate_seedance_video: 'Seedance 视频',
  compose_video: '视频合成',
  quality_check: '结构检查（仅结构，非质量认证）',
  rendering: '导出渲染',
}

export const WORKSPACE_TABS = [
  { id: 'script', label: '剧本编辑' },
  { id: 'assets', label: '角色场景资产' },
  { id: 'storyboard', label: '故事板预览' },
  { id: 'review', label: '分镜审核' },
  { id: 'av', label: '字幕与音频' },
  { id: 'video', label: '成片预览' },
] as const

export type WorkspaceTab = (typeof WORKSPACE_TABS)[number]['id']
export type PreviewMode = 'shot' | 'video'

/** 工作区标签与预览模式的唯一映射；成片页始终进入视频模式。 */
export function resolvePreviewModeForTab(tabId: WorkspaceTab): PreviewMode {
  return tabId === 'video' ? 'video' : 'shot'
}

/** 成片模式没有视频时显示空状态，绝不回退到旧故事板画面。 */
export function resolvePreviewMediaKind(
  previewMode: PreviewMode,
  hasVideo: boolean,
  hasImage: boolean,
): 'video' | 'image' | 'placeholder' {
  if (previewMode === 'video') return hasVideo ? 'video' : 'placeholder'
  return hasImage ? 'image' : 'placeholder'
}

export type ProjectOperation = {
  key: string
  token: number
  projectId: string | null
  projectEpoch: number
  navigationIntent: number
}

export function normalizeShot(shot: any) {
  return {
    id: shot.id || shot.shot_id || '',
    project_id: shot.project_id || '',
    sequence: Number(shot.sequence || 0),
    shot_type: shot.shot_type || 'medium',
    scene_description: shot.scene_description || '',
    character_action: shot.character_action || '',
    dialogue: shot.dialogue || '',
    camera_angle: shot.camera_angle || '正面',
    camera_movement: shot.camera_movement || '静止',
    duration: Number(shot.duration || 3),
    estimated_speech_ms: Number(shot.estimated_speech_ms || 0),
    emotion: shot.emotion || 'neutral',
    transition: shot.transition || 'cut',
    visual_notes: shot.visual_notes || '',
    image_path: shot.image_path || '',
    storyboard_path: shot.storyboard_path || '',
    video_path: shot.video_path || '',
    audio_path: shot.audio_path || '',
    status: shot.status || 'pending',
    storyboard_status: shot.storyboard_status || 'pending',
    version: Number(shot.version || 1),
    confirmed: Boolean(shot.confirmed),
    media_stale: Boolean(shot.media_stale),
    consistency_status: shot.consistency_status || 'pending',
    consistency_report: shot.consistency_report || {},
    storyboard_reference_manifest: Array.isArray(shot.storyboard_reference_manifest)
      ? shot.storyboard_reference_manifest
      : [],
    video_reference_manifest: Array.isArray(shot.video_reference_manifest) ? shot.video_reference_manifest : [],
    reference_capability_warning: shot.reference_capability_warning || '',
    quality_review: shot.quality_review || null,
    characters_in_scene: Array.isArray(shot.characters_in_scene) ? shot.characters_in_scene : [],
    scene_asset_id: shot.scene_asset_id || '',
    character_asset_ids: Array.isArray(shot.character_asset_ids) ? shot.character_asset_ids : [],
    scene_group_id: shot.scene_group_id || '',
    consistency_context: shot.consistency_context || '',
    reference_weights: shot.reference_weights || {},
    continuity_profile: shot.continuity_profile || {},
    continuity_reference_path: shot.continuity_reference_path || '',
    pose_reference_path: shot.pose_reference_path || '',
    depth_reference_path: shot.depth_reference_path || '',
    last_frame_path: shot.last_frame_path || '',
  }
}

export function getStepLabel(step?: string) {
  if (!step) return '处理中'
  return STEP_LABELS[step] || '处理中'
}
