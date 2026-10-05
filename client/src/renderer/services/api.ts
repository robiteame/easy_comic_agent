import axios from 'axios'

import type {
  JobActionResult,
  JobDebugLog,
  JobDetailDto,
  JobDto,
  JobListResponse,
  JobQueryParams,
  JobStatsResponse,
} from './jobTypes'
import type {
  BudgetConfigDto,
  BudgetSavePayload,
  BudgetStateDto,
  BudgetSummaryDto,
  EstimateRequest,
  JobCostDetailDto,
  PricingSavePayload,
  PricingTableDto,
  TaskEstimateResponse,
  UsageQueryParams,
  UsageResponseDto,
} from './costTypes'
import type {
  ShotVersionCompareResponse,
  ShotVersionDetail,
  ShotVersionListResponse,
  ShotVersionRestoreResponse,
} from './shotVersionTypes'
import type { QualityCapabilityResponse, QualityReviewRow, ReviewStage } from '../components/qualityReviewModel.ts'

// The packaged desktop shell spawns the backend on a per-launch random
// loopback port (main.ts reserveBackendPort) and injects the base URL here;
// development and plain-web builds keep the documented fixed endpoint.
function resolveApiBase(): string {
  if (typeof window !== 'undefined') {
    try {
      const injected = window.electronAPI?.getBackendBaseUrl?.()
      if (injected) return injected
    } catch {
      // fall through to the static default
    }
  }
  return import.meta.env?.VITE_API_BASE_URL || 'http://127.0.0.1:8011'
}

export const API_BASE = resolveApiBase()
export const WS_BASE = API_BASE.replace(/^http/, 'ws')
export const API_OUTPUT_BASE = `${API_BASE}/output/`

function getLocalAuthToken(): string {
  if (typeof window === 'undefined') return ''
  try {
    return window.electronAPI?.getLocalAuthToken?.() || ''
  } catch {
    return ''
  }
}

const LOCAL_AUTH_TOKEN = getLocalAuthToken()

function isLocalApiOrigin(value: string): boolean {
  try {
    const url = new URL(value)
    // Loopback host on any port: the local token is only ever meant for the
    // backend this shell spawned, whose port changes per launch.
    return (
      (url.protocol === 'http:' || url.protocol === 'ws:') &&
      (url.hostname === '127.0.0.1' || url.hostname === 'localhost' || url.hostname === '[::1]')
    )
  } catch {
    return false
  }
}

function withLocalAuthQuery(value: string): string {
  if (!LOCAL_AUTH_TOKEN || !isLocalApiOrigin(API_BASE)) return value
  try {
    const url = new URL(value)
    const apiOrigin = new URL(API_BASE).origin
    if (url.origin !== apiOrigin) return value
    url.searchParams.set('token', LOCAL_AUTH_TOKEN)
    return url.toString()
  } catch {
    return value
  }
}

/**
 * Convert a backend output path to a browser URL.
 *
 * The backend may return an absolute filesystem path, a path rooted at
 * `/output/`, a project-relative path, or an already resolved URL.  Do not
 * use a greedy regular expression here: project output paths themselves can
 * contain another `output/` segment (for example `projects/p1/output/...`).
 */
export function toOutputUrl(outputPath?: string | null): string | null {
  if (outputPath == null) return null
  const raw = String(outputPath).trim()
  if (!raw) return null

  if (/^https?:\/\//i.test(raw)) {
    return raw
  }

  const normalized = raw.replace(/\\/g, '/').replace(/^file:\/\//i, '')
  // Output fields are expected to contain filesystem paths or HTTP URLs. Do
  // not let an unexpected URI scheme reach an element's src attribute (for
  // example `javascript:` or `data:` from malformed/stale backend data).
  // Windows drive-letter paths are still valid local filesystem paths.
  if (/^[a-z][a-z\d+.-]*:/i.test(normalized) && !/^[a-z]:[\\/]/i.test(normalized)) {
    return null
  }
  const withoutLeadingSlash = normalized.replace(/^\/+/, '')
  let relative: string
  if (withoutLeadingSlash.startsWith('output/')) {
    relative = withoutLeadingSlash.slice('output/'.length)
  } else if (withoutLeadingSlash.startsWith('projects/')) {
    // Some API responses are already relative to the server output root.
    relative = withoutLeadingSlash
  } else {
    // For absolute filesystem paths, strip only the first output directory.
    const markerIndex = normalized.indexOf('/output/')
    relative = markerIndex >= 0 ? normalized.slice(markerIndex + '/output/'.length) : withoutLeadingSlash
  }
  relative = relative.replace(/^\/+/, '')

  if (!relative) return null
  // URL() gives callers a correctly encoded URL for spaces/non-ASCII names.
  return withLocalAuthQuery(new URL(relative, API_OUTPUT_BASE).toString())
}

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
})

api.interceptors.request.use((config) => {
  if (LOCAL_AUTH_TOKEN && isLocalApiOrigin(API_BASE)) {
    config.headers.set('X-Comic-Agent-Token', LOCAL_AUTH_TOKEN)
  }
  return config
})

export const projectApi = {
  create: (data: {
    title?: string
    first_episode_title?: string
    parent_project_id?: string
    project_type?: 'series' | 'episode'
    episode_number?: number
    style?: string
    genre?: string
    output_format?: string
    resolution?: string
    platform?: string
  }) => api.post('/api/project', data).then((r) => r.data),

  get: (id: string) => api.get(`/api/project/${id}`).then((r) => r.data),

  list: () => api.get('/api/project').then((r) => r.data),

  episodes: (id: string) => api.get(`/api/project/${id}/episodes`).then((r) => r.data),

  update: (id: string, data: Record<string, any>) => api.put(`/api/project/${id}`, data).then((r) => r.data),

  delete: (id: string) => api.delete(`/api/project/${id}`).then((r) => r.data),

  rebuildAssets: (id: string) => api.post(`/api/project/${id}/assets/rebuild`, {}).then((r) => r.data),

  importVideo: (id: string, formData: FormData) =>
    api
      .post(`/api/project/${id}/import-video`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      .then((r) => r.data),
}

export const scriptApi = {
  generate: (data: {
    project_id?: string
    prompt: string
    style?: string
    genre?: string
    target_duration?: number
    characters_hint?: string
  }) => api.post('/api/script/generate', data).then((r) => r.data),

  parse: (data: {
    project_id: string
    user_input: string
    input_type?: string
    style?: string
    output_format?: string
    resolution?: string
    platform?: string
    target_duration?: number
    mode?: 'manual' | 'auto'
  }) => api.post('/api/script/parse', data).then((r) => r.data),

  upload: (formData: FormData) =>
    api
      .post('/api/script/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      .then((r) => r.data),
}

export const shotApi = {
  list: (projectId: string) => api.get(`/api/shot/${projectId}/shots`).then((r) => r.data),

  update: (shotId: string, data: Record<string, any>) => api.put(`/api/shot/${shotId}`, data).then((r) => r.data),

  regenerate: (shotId: string, data?: Record<string, any>) =>
    api.post(`/api/shot/${shotId}/regenerate`, data || {}).then((r) => r.data),

  generationPrompt: (shotId: string) => api.get(`/api/shot/${shotId}/generation-prompt`).then((r) => r.data),

  batchRegenerate: (shotIds: string[], reason?: string) =>
    api.post('/api/shot/batch-regenerate', shotIds, { params: { reason } }).then((r) => r.data),

  generateStoryboard: (
    projectId: string,
    shotIds?: string[],
    optionsOrConfirmDegraded:
      | boolean
      | {
          confirm_degraded?: boolean
          capability_mode?: 'manual' | 'auto'
          confirm_capability_downgrade?: boolean
        } = false,
  ) => {
    const options = typeof optionsOrConfirmDegraded === 'boolean' ? {} : optionsOrConfirmDegraded || {}
    const confirmDegraded =
      typeof optionsOrConfirmDegraded === 'boolean' ? optionsOrConfirmDegraded : Boolean(options.confirm_degraded)
    return api
      .post(`/api/shot/${projectId}/generate-storyboard`, {
        shot_ids: shotIds || [],
        confirm_degraded: confirmDegraded,
        capability_mode: options.capability_mode || 'manual',
        confirm_capability_downgrade: Boolean(options.confirm_capability_downgrade),
      })
      .then((r) => r.data)
  },

  approveStoryboard: (shotId: string, approved = true) =>
    api.post(`/api/shot/${shotId}/approve-storyboard`, { approved }).then((r) => r.data),

  generateVideo: (
    shotId: string,
    force = false,
    options?: { capability_mode?: 'manual' | 'auto'; confirm_capability_downgrade?: boolean },
  ) =>
    api
      .post(`/api/shot/${shotId}/generate-video`, {
        force,
        capability_mode: options?.capability_mode || 'manual',
        confirm_capability_downgrade: Boolean(options?.confirm_capability_downgrade),
      })
      .then((r) => r.data),

  generateAudio: (shotId: string, force = false, reuseExisting = false) =>
    api.post(`/api/shot/${shotId}/generate-audio`, { force, reuse_existing: reuseExisting }).then((r) => r.data),

  confirmStoryboard: (projectId: string) => api.post(`/api/shot/${projectId}/confirm-storyboard`).then((r) => r.data),

  // --- 版本历史（只读对比 + 追加式恢复） ---

  versions: (shotId: string) => api.get(`/api/shot/${shotId}/versions`).then((r) => r.data as ShotVersionListResponse),

  versionDetail: (shotId: string, versionId: string) =>
    api.get(`/api/shot/${shotId}/versions/${encodeURIComponent(versionId)}`).then((r) => r.data as ShotVersionDetail),

  compareVersions: (shotId: string, a: string, b: string) =>
    api
      .get(`/api/shot/${shotId}/versions/compare`, { params: { a, b } })
      .then((r) => r.data as ShotVersionCompareResponse),

  restoreVersion: (shotId: string, versionId: string) =>
    api
      .post(`/api/shot/${shotId}/versions/${encodeURIComponent(versionId)}/restore`)
      .then((r) => r.data as ShotVersionRestoreResponse),
}

export type RegenerationQueueStage = 'storyboard' | 'audio' | 'video'
export interface RegenerationQueueSubmit {
  project_id: string
  shot_ids: string[]
  stages: RegenerationQueueStage[]
  priority?: number
  concurrency?: number
  order?: 'shot' | 'sequence' | 'reverse'
  reuse_audio?: boolean
  resume_missing?: boolean
  force_confirmed?: boolean
  version_map?: Record<string, number>
}

export const regenerationQueueApi = {
  submit: (data: RegenerationQueueSubmit) => api.post('/api/regeneration-queue', data).then((r) => r.data),
  detail: (batchId: string) => api.get(`/api/regeneration-queue/${encodeURIComponent(batchId)}`).then((r) => r.data),
  pause: (batchId: string) =>
    api.post(`/api/regeneration-queue/${encodeURIComponent(batchId)}/pause`).then((r) => r.data),
  resume: (batchId: string) =>
    api.post(`/api/regeneration-queue/${encodeURIComponent(batchId)}/resume`).then((r) => r.data),
  continue: (batchId: string) =>
    api.post(`/api/regeneration-queue/${encodeURIComponent(batchId)}/continue`).then((r) => r.data),
  cancel: (batchId: string) =>
    api.post(`/api/regeneration-queue/${encodeURIComponent(batchId)}/cancel`).then((r) => r.data),
  retry: (batchId: string) =>
    api.post(`/api/regeneration-queue/${encodeURIComponent(batchId)}/retry`).then((r) => r.data),
  resumeFailed: (batchId: string) =>
    api.post(`/api/regeneration-queue/${encodeURIComponent(batchId)}/resume-failed`).then((r) => r.data),
  remove: (batchId: string) => api.delete(`/api/regeneration-queue/${encodeURIComponent(batchId)}`).then((r) => r.data),
}

/** 版本接口的失败响应翻译成一句可展示的中文（后端 detail 优先）。 */
export function describeShotVersionError(error: unknown, fallback = '版本操作失败，请稍后重试'): string {
  const response = (error as { response?: { status?: number; data?: unknown } } | undefined)?.response
  const data = response?.data as { detail?: unknown } | undefined
  if (data && typeof data.detail === 'string' && data.detail) return data.detail
  const httpStatus = response?.status
  return httpStatus ? `${fallback}（HTTP ${httpStatus}）` : `${fallback}（网络不可用）`
}

export type ReferenceStatus = 'ready' | 'failed' | 'degraded' | 'unsupported' | 'stale'

export interface ReferenceActionPayload {
  project_id?: string
  action: 'retry' | 'replace_prompt' | 'regenerate' | 'skip'
  visual_prompt?: string
  reason?: string
  confirm_degraded?: boolean
}

// --- 质量审核（quality review）：评分 / 问题 / 证据 / 历史候选 ---

export const qualityReviewApi = {
  capability: () => api.get('/api/quality-review/capability').then((r) => r.data as QualityCapabilityResponse),

  projectReviews: (projectId: string) =>
    api
      .get(`/api/quality-review/project/${encodeURIComponent(projectId)}`)
      .then((r) => r.data as { project_id: string; reviews: QualityReviewRow[]; gate: Record<string, unknown> }),

  shotReviews: (shotId: string) =>
    api
      .get(`/api/quality-review/shot/${encodeURIComponent(shotId)}`)
      .then((r) => r.data as { shot_id: string; reviews: QualityReviewRow[] }),

  rerun: (shotId: string, stage: ReviewStage) =>
    api
      .post(`/api/quality-review/shot/${encodeURIComponent(shotId)}/rerun`, { stage })
      .then((r) => r.data as { shot_id: string; review: QualityReviewRow }),
}

export const assetApi = {
  board: (projectId: string) => api.get(`/api/asset/${projectId}/board`).then((r) => r.data),

  referenceAction: (kind: 'character' | 'scene', assetId: string, data: ReferenceActionPayload) =>
    api.post(`/api/asset/reference/${kind}/${encodeURIComponent(assetId)}/action`, data).then((r) => r.data),

  updateShotAssets: (
    shotId: string,
    data: { project_id?: string; scene_asset_id?: string; character_asset_ids?: string[] },
  ) => api.put(`/api/asset/shot/${shotId}`, data).then((r) => r.data),

  updateCharacter: (characterId: string, data: Record<string, any>) =>
    api.put(`/api/asset/character/${characterId}`, data).then((r) => r.data),

  updateScene: (sceneId: string, data: Record<string, any>) =>
    api.put(`/api/asset/scene/${sceneId}`, data).then((r) => r.data),
}

export const characterApi = {
  list: (projectId: string) => api.get(`/api/character/${projectId}/characters`).then((r) => r.data),

  update: (characterId: string, data: Record<string, any>) =>
    api.put(`/api/character/${characterId}`, data).then((r) => r.data),
}

export interface RenderCapabilityItem {
  value: string
  supported: boolean
  reason?: string
  fallback?: string
  prompt_strategy?: string
}

export interface RenderCapabilities {
  provider: {
    protocol: string
    model: string
    camera_movement_prompt: boolean
    timed_dialogue: boolean
    fixed_duration: number | null
    min_duration: number | null
    max_duration: number | null
    duration_step: number | null
  }
  ffmpeg: {
    available: boolean
    supported_transitions: string[]
    camera_movement: boolean
    burn_in_subtitles: boolean
    white_flash: boolean
  }
  camera_movements: RenderCapabilityItem[]
  transitions: RenderCapabilityItem[]
  fallback_policy: string
}

export const renderApi = {
  start: (data: { project_id: string; output_format?: string; resolution?: string }) =>
    api.post('/api/render', data).then((r) => r.data),

  status: (projectId: string) => api.get(`/api/render/${projectId}/status`).then((r) => r.data),

  capabilities: () => api.get('/api/render/capabilities').then((r) => r.data as RenderCapabilities),

  timeline: (projectId: string) => api.get(`/api/render/${projectId}/timeline`).then((r) => r.data),
}

// --- 字幕与音频混音工作台 ---

export interface SubtitleCueDto {
  id: string
  start_ms: number
  end_ms: number
  text: string
  character_name: string
}

export type SubtitlePosition = 'top' | 'middle' | 'bottom'

export interface SubtitleTrackDto {
  id: string
  project_id: string
  name: string
  language: string
  burn_in: boolean
  enabled: boolean
  font_family: string
  font_size: number
  primary_color: string
  outline_color: string
  outline_width: number
  bold: boolean
  position: SubtitlePosition
  safe_margin: number
  cues: SubtitleCueDto[]
}

export interface SubtitleTracksResponse {
  project_id: string
  av_config_version: number
  max_tracks: number
  tracks: SubtitleTrackDto[]
}

export type AudioTrackKind = 'dialogue' | 'music' | 'ambient' | 'sfx'

export interface AudioTrackDto {
  id: string
  project_id: string
  kind: AudioTrackKind
  name: string
  source_path: string
  source_url: string
  source_duration_ms: number
  shot_id: string
  start_ms: number
  volume: number
  pan: number
  fade_in_ms: number
  fade_out_ms: number
  delay_ms: number
  trim_start_ms: number
  trim_end_ms: number
  loop: boolean
  muted: boolean
  duck_amount_db: number
  duck_attack_ms: number
  duck_release_ms: number
  order_index: number
  shot_span?: { start_ms: number; end_ms: number }
}

export interface AvShotInfo {
  id: string
  sequence: number
  dialogue: string
  character_name: string
  has_tts: boolean
  native_audio: boolean
  start_ms: number
  end_ms: number
  duration_ms: number
}

export interface AudioTracksResponse {
  project_id: string
  av_config_version: number
  total_duration_ms: number
  shots: AvShotInfo[]
  max_tracks: number
  tracks: AudioTrackDto[]
}

export interface AvWarning {
  level: 'info' | 'warning' | 'error'
  code: string
  message: string
  track_id?: string
}

export interface AudioTrackPayload {
  kind?: AudioTrackKind
  name?: string
  source_path?: string
  shot_id?: string
  start_ms?: number
  volume?: number
  pan?: number
  fade_in_ms?: number
  fade_out_ms?: number
  delay_ms?: number
  trim_start_ms?: number
  trim_end_ms?: number
  loop?: boolean
  muted?: boolean
  duck_amount_db?: number
  duck_attack_ms?: number
  duck_release_ms?: number
}

export interface MixPreviewStatus {
  project_id: string
  status: 'idle' | 'mixing' | 'completed' | 'error' | 'cancelled'
  scope?: string
  shot_id?: string
  progress?: number
  audio_url?: string
  audio_path?: string
  message?: string
  warnings?: string[]
}

export const subtitleApi = {
  list: (projectId: string) =>
    api.get(`/api/subtitle/${projectId}/tracks`).then((r) => r.data as SubtitleTracksResponse),

  createTrack: (projectId: string, data: Record<string, unknown>) =>
    api.post(`/api/subtitle/${projectId}/tracks`, data).then((r) => r.data as SubtitleTrackDto),

  updateTrack: (trackId: string, data: Record<string, unknown>) =>
    api.put(`/api/subtitle/track/${trackId}`, data).then((r) => r.data as SubtitleTrackDto),

  deleteTrack: (trackId: string, projectId: string) =>
    api.delete(`/api/subtitle/track/${trackId}`, { params: { project_id: projectId } }).then((r) => r.data),

  replaceCues: (
    trackId: string,
    data: {
      project_id: string
      cues: Array<{ start_ms: number; end_ms: number; text: string; character_name?: string }>
    },
  ) => api.put(`/api/subtitle/track/${trackId}/cues`, data).then((r) => r.data as SubtitleTrackDto),

  importSubtitle: (trackId: string, data: { project_id: string; format: 'srt' | 'vtt'; content: string }) =>
    api.post(`/api/subtitle/track/${trackId}/import`, data).then((r) => r.data as SubtitleTrackDto),

  generateFromShots: (trackId: string, projectId: string) =>
    api
      .post(`/api/subtitle/track/${trackId}/generate`, { project_id: projectId })
      .then(
        (r) =>
          r.data as SubtitleTrackDto & { overlaps?: Array<{ index_a: number; index_b: number; overlap_ms: number }> },
      ),

  /** 导出走浏览器下载；本地鉴权 token 通过查询参数注入。 */
  exportUrl: (trackId: string, projectId: string, format: 'srt' | 'vtt') =>
    withLocalAuthQuery(
      `${API_BASE}/api/subtitle/track/${trackId}/export?project_id=${encodeURIComponent(projectId)}&format=${format}`,
    ),
}

export const audioTrackApi = {
  list: (projectId: string) =>
    api.get(`/api/audio-track/${projectId}/tracks`).then((r) => r.data as AudioTracksResponse),

  upload: (projectId: string, formData: FormData) =>
    api
      .post(`/api/audio-track/${projectId}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300000,
      })
      .then((r) => r.data as { source_path: string; source_url: string; duration_ms: number; size_bytes: number }),

  createTrack: (projectId: string, data: AudioTrackPayload) =>
    api.post(`/api/audio-track/${projectId}/tracks`, data).then((r) => r.data as AudioTrackDto),

  updateTrack: (trackId: string, data: AudioTrackPayload & { project_id: string }) =>
    api.put(`/api/audio-track/track/${trackId}`, data).then((r) => r.data as AudioTrackDto),

  deleteTrack: (trackId: string, projectId: string) =>
    api.delete(`/api/audio-track/track/${trackId}`, { params: { project_id: projectId } }).then((r) => r.data),

  analyze: (projectId: string) =>
    api
      .post(`/api/audio-track/${projectId}/analyze`)
      .then((r) => r.data as { project_id: string; total_duration_ms: number; warnings: AvWarning[] }),

  startPreview: (projectId: string, data: { scope: 'full' | 'shot'; shot_id?: string }) =>
    api.post(`/api/audio-track/${projectId}/preview`, data).then((r) => r.data),

  previewStatus: (projectId: string) =>
    api.get(`/api/audio-track/${projectId}/preview/status`).then((r) => r.data as MixPreviewStatus),
}

export const chatApi = {
  send: (data: { project_id: string; message: string; current_shots?: any[] }) =>
    api.post('/api/chat', data).then((r) => r.data),
}

export const settingsApi = {
  styleTemplates: () => api.get('/api/settings/style-templates').then((r) => r.data),

  createStyleTemplate: (data: { label: string; keywords: string; negative_prompt?: string }) =>
    api.post('/api/settings/style-templates', data).then((r) => r.data),

  skillConfigs: () => api.get('/api/settings/skill-configs').then((r) => r.data),

  saveSkillConfig: (data: Record<string, any>) => api.post('/api/settings/skill-configs', data).then((r) => r.data),

  updateSkillBindings: (data: Record<string, any>) =>
    api.put('/api/settings/skill-configs/bindings', data).then((r) => r.data),

  modelConfigs: () => api.get('/api/settings/model-configs').then((r) => r.data),

  providerCapabilities: (params?: { capability?: 'image' | 'video'; protocol?: string; model?: string }) =>
    api.get('/api/settings/provider-capabilities', { params }).then((r) => r.data),

  discoverModels: (data: {
    category: 'script' | 'image' | 'video' | 'voice'
    base_url: string
    api_key?: string
    protocol?: string
    auth_style?: string
  }) => api.post('/api/settings/model-configs/discover', data).then((r) => r.data),

  saveModelConfigs: (data: Record<string, any>) => api.put('/api/settings/model-configs', data).then((r) => r.data),
}

/**
 * 任务中心 API。
 *
 * 与项目/镜头接口分开维护：任务中心是全局视图，只读 ``/api/jobs`` 的稳定 DTO。
 * 取消 / 重试 / 续跑都必须等待服务端结果（``cancelJob`` 等返回的是服务端落库后的
 * 状态），调用方据此刷新或合并状态，而不是只做乐观更新。
 */
export const jobApi = {
  list: (params: JobQueryParams = {}) => api.get('/api/jobs', { params }).then((r) => r.data as JobListResponse),

  stats: (projectId?: string) =>
    api
      .get('/api/jobs/stats', { params: projectId ? { project_id: projectId } : undefined })
      .then((r) => r.data as JobStatsResponse),

  detail: (jobId: string) => api.get(`/api/jobs/${encodeURIComponent(jobId)}`).then((r) => r.data as JobDetailDto),
  debug: (jobId: string) => api.get(`/api/jobs/${encodeURIComponent(jobId)}/debug`).then((r) => r.data as JobDebugLog),

  cancel: (jobId: string) =>
    api.post(`/api/jobs/${encodeURIComponent(jobId)}/cancel`).then((r) => r.data as JobActionResult),

  retry: (jobId: string) =>
    api.post(`/api/jobs/${encodeURIComponent(jobId)}/retry`).then((r) => r.data as JobActionResult),

  resume: (jobId: string) =>
    api.post(`/api/jobs/${encodeURIComponent(jobId)}/resume`).then((r) => r.data as JobActionResult),

  remove: (jobId: string) =>
    api.delete(`/api/jobs/${encodeURIComponent(jobId)}`).then((r) => r.data as JobActionResult),

  cleanup: (params: JobQueryParams = {}) =>
    api.delete('/api/jobs', { params }).then((r) => r.data as { deleted: number; stats: JobStatsResponse }),
}

/**
 * 成本 / 预算 API。
 *
 * 三条与后端一致的约定，调用方必须遵守：
 * 1. 金额字段是**整数 micro**（1 元 = 1_000_000 micro）：前端展示前用
 *    `costModel.formatMicroAmount` 换算，回传前用 `costModel.amountTextToMicro`
 *    做字符串换算，绝不把浮点金额写到接口上（后端会返回 400 中文原因）；
 * 2. `cost_known=false` 时 `cost_micro` 必为 null，界面显示「成本未知」而不是 0；
 * 3. 估算（`estimate`）与实际（`usage`）是两份数据，永远分开展示。
 */
export const budgetApi = {
  /** 价目表：每类能力的主/次计价单位与已配置单价。 */
  pricing: () => api.get('/api/budget/pricing').then((r) => r.data as PricingTableDto),

  /** 保存价目表；浮点金额会被后端拒绝（400 + 中文 detail）。 */
  savePricing: (data: PricingSavePayload) =>
    api.put('/api/budget/pricing', data).then((r) => r.data as PricingTableDto),

  /** 预算配置：全局 + 项目（含父系列）的行与生效值。 */
  config: (projectId?: string) =>
    api
      .get('/api/budget/config', { params: projectId ? { project_id: projectId } : undefined })
      .then((r) => r.data as BudgetConfigDto),

  /** 保存一条预算配置（项目级 / 全局）。软预算高于硬预算会被拒绝（400）。 */
  saveConfig: (data: BudgetSavePayload) => api.put('/api/budget/config', data).then((r) => r.data as BudgetConfigDto),

  /** 项目预算状态（已用 / 已预留 / 生效上限）。 */
  status: (projectId: string) =>
    api.get('/api/budget/status', { params: { project_id: projectId } }).then((r) => r.data as BudgetStateDto),

  /** 提交前估算（只读，不占额度；硬预算不足时 blocked=true）。 */
  estimate: (data: EstimateRequest) =>
    api.post('/api/budget/estimate', data).then((r) => r.data as TaskEstimateResponse),

  /** 项目 / 剧集页汇总：已用、已预留、剩余工作量、预算与逐剧集明细。 */
  summary: (params: { project_id?: string; series_id?: string }) =>
    api.get('/api/budget/summary', { params }).then((r) => r.data as BudgetSummaryDto),

  /**
   * 用量明细与多维统计。
   *
   * `group_by` 是重复查询参数（`?group_by=job_type&group_by=shot`），axios 默认会把
   * 数组序列化成 `group_by[]=...`，后端认不出这个参数名，因此这里手工拼查询串。
   */
  usage: (params: UsageQueryParams = {}) => {
    const search = new URLSearchParams()
    const single: (keyof UsageQueryParams)[] = [
      'project_id',
      'series_id',
      'shot_id',
      'job_id',
      'job_key',
      'capability',
      'job_type',
    ]
    for (const key of single) {
      const value = params[key]
      if (typeof value === 'string' && value) search.append(key, value)
    }
    for (const dimension of params.group_by || []) {
      if (dimension) search.append('group_by', dimension)
    }
    if (params.page) search.append('page', String(params.page))
    if (params.page_size) search.append('page_size', String(params.page_size))
    const query = search.toString()
    return api.get('/api/budget/usage' + (query ? '?' + query : '')).then((r) => r.data as UsageResponseDto)
  },

  /** 单个任务的成本明细：失败 / 取消的任务同样可查已发生成本。 */
  jobCost: (jobId: string) =>
    api.get(`/api/budget/jobs/${encodeURIComponent(jobId)}`).then((r) => r.data as JobCostDetailDto),
}

/**
 * 把任务接口的失败响应翻译成统一结果。
 *
 * 后端对 404/409 也返回同样的 JSON 形状（ok/status/message/error_code），因此这里
 * 优先使用响应体；网络错误没有响应体时给一句稳定的中文提示，不做乐观假设。
 */
function statusFromHttp(httpStatus?: number): string {
  if (httpStatus === 401 || httpStatus === 403) return 'forbidden'
  if (httpStatus === 404) return 'not_found'
  if (httpStatus === 409) return 'conflict'
  return 'error'
}

export function describeJobApiError(error: unknown, fallback = '操作失败，请稍后重试'): JobActionResult {
  const response = (error as { response?: { status?: number; data?: unknown } } | undefined)?.response
  const data = response?.data as Partial<JobActionResult> | undefined
  if (data && typeof data === 'object' && typeof data.message === 'string') {
    const status = typeof data.status === 'string' && data.status ? data.status : statusFromHttp(response?.status)
    return {
      ok: false,
      status,
      message: data.message,
      error_code: data.error_code,
      job: (data.job as JobDto | null) ?? null,
    }
  }
  const httpStatus = response?.status
  return {
    ok: false,
    status: statusFromHttp(httpStatus),
    message: httpStatus ? `${fallback}（HTTP ${httpStatus}）` : `${fallback}（网络不可用）`,
  }
}

export interface WebSocketOptions {
  onOpen?: (event: Event) => void
  onClose?: (event: CloseEvent) => void
  onError?: (event: Event) => void
}

/**
 * 生成 Agent 图结构与可解释追踪。
 *
 * `/api/graph/runs/:id/trace` 的 `summary` 是展示用稳定汇总：当前阶段、镜头
 * 状态、阶段质量分、Critic 问题、恢复候选与最终决策、Prompt 修改、候选结果、
 * Provider/模型、实际发送参考图、成本、预计/实际耗时、自动降级原因、检查点
 * 与恢复次数。类型定义和展示逻辑见 `components/agentTraceModel`。
 */
export const agentGraphApi = {
  structure: () => api.get('/api/graph/structure').then((r) => r.data as unknown),

  run: (projectId: string, runId = 'auto') =>
    api
      .get(`/api/graph/runs/${encodeURIComponent(projectId)}`, { params: { run_id: runId } })
      .then((r) => r.data as unknown),

  trace: (projectId: string, runId = 'auto') =>
    api
      .get(`/api/graph/runs/${encodeURIComponent(projectId)}/trace`, { params: { run_id: runId } })
      .then((r) => r.data as import('../components/agentTraceModel').AgentTraceResponse),
}

function localTokenQuery(): string {
  return LOCAL_AUTH_TOKEN && isLocalApiOrigin(API_BASE) ? `?token=${encodeURIComponent(LOCAL_AUTH_TOKEN)}` : ''
}

function parseSocketPayload(data: unknown): any {
  try {
    return JSON.parse(String(data))
  } catch (_err) {
    // 后端可能发来非 JSON 文本帧（如纯文本心跳），忽略而非中断消息处理
    console.warn('收到无法解析的 WebSocket 消息，已忽略:', data)
    return null
  }
}

export function createWebSocket(
  projectId: string,
  onMessage: (data: any) => void,
  options: WebSocketOptions = {},
): WebSocket {
  const tokenQuery = localTokenQuery()
  const ws = new WebSocket(`${WS_BASE}/ws/${projectId}${tokenQuery}`)

  ws.onopen = (event) => {
    options.onOpen?.(event)
  }

  ws.onmessage = (event) => {
    let parsed: any
    try {
      parsed = JSON.parse(event.data)
    } catch (_err) {
      // 后端可能发来非 JSON 文本帧（如纯文本心跳），忽略而非中断消息处理
      console.warn('收到无法解析的 WebSocket 消息，已忽略:', event.data)
      return
    }
    onMessage(parsed)
  }

  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
    options.onError?.(error)
  }

  const heartbeat = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send('ping')
    }
  }, 30000)

  ws.onclose = (event) => {
    clearInterval(heartbeat)
    options.onClose?.(event)
  }

  return ws
}

/**
 * 全局任务中心 WebSocket。
 *
 * 连接建立后服务端会先推送一次当前任务快照（job_snapshot），再推送增量事件，
 * 因此前端不会因为「连接建立得比事件晚」而丢状态。断线重连由 store 负责：重连
 * 成功后必须重新拉一次 REST 快照，不能只依赖增量事件。
 */
export function createJobsWebSocket(onMessage: (data: any) => void, options: WebSocketOptions = {}): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/ws/jobs${localTokenQuery()}`)

  ws.onopen = (event) => {
    options.onOpen?.(event)
  }

  ws.onmessage = (event) => {
    const parsed = parseSocketPayload(event.data)
    if (parsed !== null) onMessage(parsed)
  }

  ws.onerror = (error) => {
    console.error('任务中心 WebSocket error:', error)
    options.onError?.(error)
  }

  const heartbeat = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send('ping')
    }
  }, 30000)

  ws.onclose = (event) => {
    clearInterval(heartbeat)
    options.onClose?.(event)
  }

  return ws
}

export default api
