import { create } from 'zustand'
import type { EditableDialogueLine } from '../services/dialogueTimeline.ts'
import { mergeShotServerUpdate, type ShotServerUpdate } from '../services/shotUpdateGuard.ts'
import type { ShotQualitySummary } from '../components/qualityReviewModel.ts'

export interface Shot {
  id: string
  project_id: string
  sequence: number
  shot_type: string
  scene_description: string
  character_action: string
  dialogue: string | EditableDialogueLine[]
  camera_angle: string
  camera_movement: string
  duration: number
  estimated_speech_ms?: number
  emotion: string
  transition: string
  visual_notes: string
  image_path: string
  storyboard_path: string
  video_path: string
  audio_path: string
  status: string
  storyboard_status: string
  version: number
  confirmed: boolean
  // 参数/配置已变更但旧素材仍保留：true 时界面在旧素材上显示「待重新生成」。
  media_stale?: boolean
  consistency_status?: 'pending' | 'ready' | 'failed' | 'degraded' | 'unsupported' | 'stale'
  consistency_report?: Record<string, any>
  storyboard_reference_manifest?: Array<Record<string, any>>
  video_reference_manifest?: Array<Record<string, any>>
  reference_capability_warning?: string
  characters_in_scene: string[]
  scene_asset_id: string
  character_asset_ids: string[]
  scene_group_id?: string
  consistency_context?: string
  reference_weights?: { environment?: number; action?: number }
  continuity_profile?: Record<string, any>
  continuity_reference_path?: string
  pose_reference_path?: string
  depth_reference_path?: string
  last_frame_path?: string
  // 质量审核摘要（最新一轮；结构检查与质量审核在界面上分开呈现）。
  quality_review?: ShotQualitySummary
}

interface ShotState {
  shots: Shot[]
  selectedShotId: string | null
  isGenerating: boolean
  progress: number
  currentStep: string
  awaitingStoryboardConfirm: boolean
  videoPath: string
  logs: string[]

  setShots: (shots: Shot[]) => void
  updateShot: (id: string, data: Partial<Shot>) => void
  applyServerShotUpdate: (id: string, payload: ShotServerUpdate) => void
  selectShot: (id: string | null) => void
  setGenerating: (v: boolean) => void
  setProgress: (progress: number, step: string) => void
  setAwaitingStoryboardConfirm: (v: boolean) => void
  setVideoPath: (path: string) => void
  appendLog: (line: string) => void
  clearLogs: () => void
  addShot: (shot: Shot) => void
  removeShot: (id: string) => void
  reorderShots: (newOrder: string[]) => void
}

export const useShotStore = create<ShotState>((set, get) => ({
  shots: [],
  selectedShotId: null,
  isGenerating: false,
  progress: 0,
  currentStep: '',
  awaitingStoryboardConfirm: false,
  videoPath: '',
  logs: [],

  setShots: (shots) => set({ shots }),

  updateShot: (id, data) =>
    set((state) => ({
      shots: state.shots.map((s) => (s.id === id ? { ...s, ...data } : s)),
    })),

  // WebSocket / 异步任务回写专用：字段级守卫（空媒体路径不覆盖有效路径、
  // 旧任务版本回退整体丢弃）后再合并。
  applyServerShotUpdate: (id, payload) =>
    set((state) => {
      const current = state.shots.find((s) => s.id === id)
      if (!current) return {}
      const patch = mergeShotServerUpdate(current, payload)
      if (!patch) return {}
      return { shots: state.shots.map((s) => (s.id === id ? { ...s, ...patch } : s)) }
    }),

  selectShot: (id) => set({ selectedShotId: id }),

  setGenerating: (v) => set({ isGenerating: v }),

  setProgress: (progress, step) => set({ progress, currentStep: step }),

  setAwaitingStoryboardConfirm: (v) => set({ awaitingStoryboardConfirm: v }),

  setVideoPath: (path) => set({ videoPath: path }),

  appendLog: (line) =>
    set((state) => ({
      logs: [...state.logs, line].slice(-100),
    })),

  clearLogs: () => set({ logs: [] }),

  addShot: (shot) => set((state) => ({ shots: [...state.shots, shot] })),

  removeShot: (id) =>
    set((state) => ({ shots: state.shots.filter((s) => s.id !== id) })),

  reorderShots: (newOrder) =>
    set((state) => {
      const shotMap = new Map(state.shots.map((s) => [s.id, s]))
      const reordered = newOrder
        .map((id) => shotMap.get(id))
        .filter(Boolean) as Shot[]
      return { shots: reordered }
    }),
}))
