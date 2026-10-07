/**
 * 桌面端自动更新状态(store)。
 *
 * 数据流:
 * 1. initialize() 拉取一次 updater:get-info(版本号 / 模式 / 当前状态),
 *    并订阅主进程 updater:state 推送,之后状态完全由主进程状态机驱动;
 * 2. 手动「检查更新」走 invoke,结果只落在 manualResult 供设置页展示,
 *    失败也绝不弹错误框(由视图模型降级为「前往下载」);
 * 3. Web 预览(无 electronAPI)自然退化为 disabled,不发起任何请求。
 *
 * 安全:本 store 不持有任何凭证,只承载更新状态机;所有副作用都经
 * preload 白名单(window.electronAPI)转发。
 */
import { create } from 'zustand'

import type { ManualCheckResult, UpdateMode, UpdaterState } from '../../main/updaterModel'
import { bannerKey, type BannerActionKind } from '../components/updaterViewModel'

interface UpdateStore {
  appVersion: string
  mode: UpdateMode
  state: UpdaterState
  manualChecking: boolean
  manualResult: ManualCheckResult | null
  dismissedKeys: string[]
  initialize(): void
  applyUpdaterState(state: UpdaterState): void
  checkForUpdates(): Promise<void>
  runBannerAction(kind: BannerActionKind): void
}

let unsubscribe: (() => void) | null = null

// SSR / node 测试环境下 window 可能不存在,统一走安全读取。
function electronApi() {
  return (globalThis as { window?: Window }).window?.electronAPI
}

export const useUpdateStore = create<UpdateStore>((set, get) => ({
  appVersion: '',
  mode: 'disabled',
  state: { phase: 'idle' },
  manualChecking: false,
  manualResult: null,
  dismissedKeys: [],

  initialize() {
    const api = electronApi()
    if (!api?.onUpdaterState || !api.getUpdaterInfo) return
    if (!unsubscribe) {
      unsubscribe = api.onUpdaterState((state) => get().applyUpdaterState(state))
    }
    void api
      .getUpdaterInfo()
      .then((info) => set({ appVersion: info.appVersion, mode: info.mode, state: info.state }))
      .catch(() => undefined)
  },

  applyUpdaterState(state) {
    // dismissedKeys 不随新状态清理:「稍后」在本次会话内持续生效,直到
    // 状态推进到新 key(如开始下载/下载完成)横幅才重新出现。
    set({ state })
  },

  async checkForUpdates() {
    const api = electronApi()
    if (!api?.checkForUpdates) {
      set({ manualResult: { outcome: 'disabled' } })
      return
    }
    set({ manualChecking: true })
    try {
      const result = await api.checkForUpdates()
      set({ manualResult: result })
    } catch {
      set({ manualResult: { outcome: 'disabled' } })
    } finally {
      set({ manualChecking: false })
    }
  },

  runBannerAction(kind) {
    const api = electronApi()
    if (kind === 'dismiss') {
      const { state } = get()
      const version = 'version' in state ? state.version : ''
      const key = bannerKey(state.phase, version)
      set((current) => ({ dismissedKeys: [...current.dismissedKeys, key] }))
      return
    }
    if (!api) return
    if (kind === 'download') api.startUpdateDownload?.()
    else if (kind === 'install') api.quitAndInstallUpdate?.()
    else if (kind === 'open-release-page') void api.openReleasePage?.()
  },
}))
