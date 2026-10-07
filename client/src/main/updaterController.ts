// 自动更新控制器:驱动 electron-updater(或其替身)并广播状态机。
// 本文件不依赖 electron / electron-updater —— 主进程接入与单元测试都
// 通过 UpdaterControllerDeps 注入副作用,便于在 node:test 下用内存 stub
// 覆盖完整状态机。
import {
  GITHUB_OWNER,
  GITHUB_REPO,
  MAC_AUTO_UPDATE_ENABLED,
  reduceUpdaterState,
  releasePageUrl,
  resolveUpdateMode,
  type ManualCheckResult,
  type UpdateCheckSource,
  type UpdaterEvent,
  type UpdateMode,
  type UpdaterState,
} from './updaterModel'

export const INITIAL_CHECK_DELAY_MS = 10_000
export const CHECK_INTERVAL_MS = 24 * 60 * 60 * 1000

// electron-updater 的最小表面:控制器只依赖这些方法/事件,测试注入内存
// stub 即可驱动完整状态机,不真正联网、不真正重启。
export interface AutoUpdaterLike {
  on(event: string, listener: (...args: any[]) => void): unknown
  checkForUpdates(): Promise<unknown>
  downloadUpdate(): Promise<unknown>
  quitAndInstall(): void
}

export interface UpdaterControllerDeps {
  autoUpdater: AutoUpdaterLike
  platform: string
  isPackaged: boolean
  appVersion: string
  macAutoUpdate?: boolean
  initialCheckDelayMs?: number
  checkIntervalMs?: number
  broadcast: (state: UpdaterState) => void
  schedule: (fn: () => void, delayMs: number) => () => void
  repeat: (fn: () => void, intervalMs: number) => () => void
  logger?: Pick<Console, 'log' | 'error'>
}

export interface UpdaterController {
  readonly mode: UpdateMode
  start(): () => void
  check(source: UpdateCheckSource): Promise<ManualCheckResult>
  getState(): UpdaterState
  downloadUpdate(): void
  installUpdate(): void
  releaseUrlFor(state: UpdaterState): string
}

export function createUpdaterController(deps: UpdaterControllerDeps): UpdaterController {
  const log = deps.logger ?? console
  const mode = resolveUpdateMode(deps.platform, deps.isPackaged, {
    macAutoUpdate: deps.macAutoUpdate ?? MAC_AUTO_UPDATE_ENABLED,
  })
  const releaseUrl = (version?: string) => releasePageUrl(GITHUB_OWNER, GITHUB_REPO, version)

  let state: UpdaterState = { phase: 'idle' }
  let availableVersion = ''
  let lastSource: UpdateCheckSource = 'auto'
  let checkInFlight: Promise<ManualCheckResult> | null = null

  const apply = (event: UpdaterEvent) => {
    state = reduceUpdaterState(state, event)
    deps.broadcast(state)
  }

  deps.autoUpdater.on('update-available', (info: { version?: string } | undefined) => {
    availableVersion = String(info?.version ?? '')
    apply({
      type: 'update-available',
      source: lastSource,
      mode,
      version: availableVersion,
      releaseUrl: releaseUrl(availableVersion),
    })
  })
  deps.autoUpdater.on('update-not-available', () => {
    apply({ type: 'update-not-available', source: lastSource, currentVersion: deps.appVersion })
  })
  deps.autoUpdater.on('download-progress', (progress: { percent?: number } | undefined) => {
    apply({ type: 'download-progress', version: availableVersion, percent: progress?.percent ?? 0 })
  })
  deps.autoUpdater.on('update-downloaded', (info: { version?: string } | undefined) => {
    const version = String(info?.version ?? availableVersion)
    if (version) availableVersion = version
    apply({ type: 'update-downloaded', version: version || availableVersion })
  })
  deps.autoUpdater.on('error', (error: Error | undefined) => {
    apply({
      type: 'update-error',
      source: lastSource,
      message: error?.message ?? '未知更新错误',
      releaseUrl: releaseUrl(),
    })
  })

  // 手动检查的返回值由检查结束时已归约出的状态推导:electron-updater 在
  // checkForUpdates 结算前一定会派发 update-available / update-not-available
  // 或 error 事件,这里不再重复解析其结果对象。
  function deriveManualResult(): ManualCheckResult {
    if (
      state.phase === 'notified' ||
      state.phase === 'manual-check-required' ||
      state.phase === 'downloading' ||
      state.phase === 'downloaded'
    ) {
      return { outcome: 'available', state }
    }
    if (state.phase === 'up-to-date') return { outcome: 'up-to-date', currentVersion: state.currentVersion }
    return {
      outcome: 'error',
      message: state.phase === 'error' ? state.message : '更新检查结果未知',
      releaseUrl: releaseUrl(),
    }
  }

  async function runCheck(source: UpdateCheckSource): Promise<ManualCheckResult> {
    if (mode === 'disabled') {
      log.log('[updater] 更新检查已禁用(开发模式/未打包),跳过检查')
      return { outcome: 'disabled' }
    }
    lastSource = source
    apply({ type: 'check-started', source })
    try {
      await deps.autoUpdater.checkForUpdates()
    } catch (error) {
      // error 事件通常已把状态归约为 error;这里兜底(并给手动检查一个
      // 可用的失败返回值,绝不向渲染层抛出)。
      const message = error instanceof Error ? error.message : String(error)
      apply({ type: 'update-error', source, message, releaseUrl: releaseUrl() })
      return { outcome: 'error', message, releaseUrl: releaseUrl() }
    }
    return deriveManualResult()
  }

  function check(source: UpdateCheckSource): Promise<ManualCheckResult> {
    // 后台定时检查与用户手动点击可能并发,串行复用同一次请求。
    if (checkInFlight) return checkInFlight
    checkInFlight = runCheck(source).finally(() => {
      checkInFlight = null
    })
    return checkInFlight
  }

  function downloadUpdate(): void {
    if (mode !== 'auto') {
      log.log(`[updater] 当前更新模式为 ${mode},忽略下载请求`)
      return
    }
    if (state.phase !== 'notified' && state.phase !== 'error') {
      log.log(`[updater] 当前状态 ${state.phase} 不允许下载,忽略请求`)
      return
    }
    // 下载永远由用户主动触发,后续错误按 manual 归因,便于 UI 降级展示。
    lastSource = 'manual'
    apply({ type: 'download-started', version: availableVersion })
    deps.autoUpdater.downloadUpdate().catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error)
      log.error(`[updater] 下载失败: ${message}`)
      apply({ type: 'update-error', source: 'manual', message, releaseUrl: releaseUrl() })
    })
  }

  function installUpdate(): void {
    if (mode !== 'auto' || state.phase !== 'downloaded') {
      log.log(`[updater] 当前状态 ${state.phase}/${mode} 不允许安装,忽略请求`)
      return
    }
    log.log('[updater] 用户确认重启并安装新版本')
    deps.autoUpdater.quitAndInstall()
  }

  function releaseUrlFor(current: UpdaterState): string {
    if ('version' in current && current.version) return releaseUrl(current.version)
    return releaseUrl()
  }

  function start(): () => void {
    if (mode === 'disabled') {
      log.log('[updater] 未打包(开发模式),自动更新已禁用')
      return () => {}
    }
    log.log(`[updater] 更新模式: ${mode},首检延迟 ${deps.initialCheckDelayMs ?? INITIAL_CHECK_DELAY_MS}ms`)
    let cancelInterval: () => void = () => {}
    const cancelDelay = deps.schedule(() => {
      void check('auto').catch(() => undefined)
      // 首次检查启动后再进入 24h 周期,避免与首检叠加。
      cancelInterval = deps.repeat(
        () => void check('auto').catch(() => undefined),
        deps.checkIntervalMs ?? CHECK_INTERVAL_MS,
      )
    }, deps.initialCheckDelayMs ?? INITIAL_CHECK_DELAY_MS)
    return () => {
      cancelDelay()
      cancelInterval()
    }
  }

  return {
    mode,
    start,
    check,
    getState: () => state,
    downloadUpdate,
    installUpdate,
    releaseUrlFor,
  }
}
