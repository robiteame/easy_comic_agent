import { create } from 'zustand'

interface ProjectState {
  projectId: string | null
  parentProjectId: string
  parentProjectTitle: string
  projectType: 'series' | 'episode'
  episodeNumber: number
  title: string
  genre: string
  style: string
  status: string
  consistencyStatus: string
  consistencyReport: Record<string, any>
  outputFormat: string
  resolution: string
  platform: string
  runMode: 'manual' | 'auto'
  characters: any[]

  setProject: (data: Partial<ProjectState>) => void
  reset: () => void
}

const DEFAULT_PROJECT: Omit<ProjectState, 'setProject' | 'reset'> = {
  projectId: null,
  parentProjectId: '',
  parentProjectTitle: '',
  projectType: 'series',
  episodeNumber: 0,
  title: '未命名项目',
  genre: '',
  style: 'anime',
  status: 'draft',
  consistencyStatus: 'ready',
  consistencyReport: {},
  outputFormat: '9:16',
  resolution: '1080p',
  platform: 'douyin',
  runMode: 'manual',
  characters: [],
}

export const useProjectStore = create<ProjectState>((set) => ({
  ...DEFAULT_PROJECT,

  setProject: (data) => set((state) => ({ ...state, ...data })),
  reset: () => set(DEFAULT_PROJECT),
}))
