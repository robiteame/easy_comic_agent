import { contextBridge, ipcRenderer } from 'electron'
import type { ManualCheckResult, UpdateMode, UpdaterState } from './updaterModel'

contextBridge.exposeInMainWorld('electronAPI', {
  getLocalAuthToken: () => ipcRenderer.sendSync('get-local-auth-token') as string,
  // The packaged backend listens on a per-launch random port; the main
  // process is the only component that knows it. Empty string in dev mode
  // (the renderer then falls back to the documented 8011 endpoint).
  getBackendBaseUrl: () => ipcRenderer.sendSync('get-backend-base-url') as string,
  retryBackend: () => ipcRenderer.invoke('backend-retry') as Promise<{ ok: boolean; detail?: string }>,
  quitApp: () => ipcRenderer.send('app-quit'),
  selectFile: () => ipcRenderer.invoke('select-file'),
  selectDirectory: () => ipcRenderer.invoke('select-directory'),
  // 自动更新通道:渲染层只通过这份白名单与 electron-updater 通信,
  // 绝不直接 require('electron-updater')。openReleasePage 不接受 URL——
  // 由主进程按自身常量打开 Release 页,避免 shell.openExternal 注入面。
  onUpdaterState: (listener: (state: UpdaterState) => void) => {
    const handler = (_event: unknown, state: UpdaterState) => listener(state)
    ipcRenderer.on('updater:state', handler)
    return () => ipcRenderer.removeListener('updater:state', handler)
  },
  getUpdaterInfo: () =>
    ipcRenderer.invoke('updater:get-info') as Promise<{ appVersion: string; mode: UpdateMode; state: UpdaterState }>,
  checkForUpdates: () => ipcRenderer.invoke('updater:check') as Promise<ManualCheckResult>,
  startUpdateDownload: () => ipcRenderer.send('updater:download'),
  quitAndInstallUpdate: () => ipcRenderer.send('updater:install'),
  openReleasePage: () => ipcRenderer.invoke('updater:open-release-page') as Promise<void>,
})
