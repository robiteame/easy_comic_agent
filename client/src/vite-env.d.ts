/// <reference types="vite/client" />

import type { ManualCheckResult, UpdateMode, UpdaterState } from '../main/updaterModel'

declare global {
  interface Window {
    electronAPI?: {
      getLocalAuthToken?: () => string
      getBackendBaseUrl?: () => string
      retryBackend?: () => Promise<{ ok: boolean; detail?: string }>
      quitApp?: () => void
      selectFile?: () => Promise<string | null>
      selectDirectory?: () => Promise<string | null>
      // 自动更新:全部经 preload 白名单转发,渲染层不直接接触 electron-updater。
      onUpdaterState?: (listener: (state: UpdaterState) => void) => () => void
      getUpdaterInfo?: () => Promise<{ appVersion: string; mode: UpdateMode; state: UpdaterState }>
      checkForUpdates?: () => Promise<ManualCheckResult>
      startUpdateDownload?: () => void
      quitAndInstallUpdate?: () => void
      openReleasePage?: () => Promise<void>
    }
  }
}
