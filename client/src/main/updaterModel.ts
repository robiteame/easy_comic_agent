// 自动更新的纯状态机:类型、平台分流、Release 链接与事件归约。
// 不依赖 electron / electron-updater,主进程、preload(仅类型)与渲染层
// 视图模型都可引用,便于在 node:test 下直接覆盖。

/** GitHub Releases 仓库,与 electron-builder.yml 的 publish 配置保持一致。 */
export const GITHUB_OWNER = 'robiteame'
export const GITHUB_REPO = 'ComicAgent'

// macOS 自动更新依赖 Squirrel.Mac,要求应用具备 Developer ID 签名;当前
// 构建未签名,自动安装必然失败,故 macOS 走「前往下载」降级路径。拿到
// 签名证书后把该开关改为 true 即可放开 mac 自动更新,其余逻辑不变。
export const MAC_AUTO_UPDATE_ENABLED = false

export type UpdateMode = 'auto' | 'manual-download' | 'disabled'
export type UpdateCheckSource = 'auto' | 'manual'

export type UpdaterState =
  | { phase: 'idle' }
  | { phase: 'checking'; source: UpdateCheckSource }
  | { phase: 'notified'; source: UpdateCheckSource; version: string; releaseUrl: string }
  | { phase: 'downloading'; version: string; percent: number }
  | { phase: 'downloaded'; version: string }
  | { phase: 'manual-check-required'; source: UpdateCheckSource; version: string; releaseUrl: string }
  | { phase: 'up-to-date'; currentVersion: string }
  | { phase: 'error'; source: UpdateCheckSource; message: string; releaseUrl: string }

export type ManualCheckResult =
  | { outcome: 'disabled' }
  | { outcome: 'up-to-date'; currentVersion: string }
  | { outcome: 'available'; state: Exclude<UpdaterState, { phase: 'idle' | 'checking' }> }
  | { outcome: 'error'; message: string; releaseUrl: string }

/** 版本号可能带 v 前缀,统一转成 release tag 用的 `v0.2.0` 形式。 */
export function releaseTag(version: string): string {
  return `v${version.replace(/^v/, '')}`
}

export function releasePageUrl(owner: string, repo: string, version?: string): string {
  const base = `https://github.com/${owner}/${repo}/releases`
  return version ? `${base}/tag/${releaseTag(version)}` : `${base}/latest`
}

/**
 * 平台分流:
 * - 未打包(electron:dev / vite dev)一律 disabled,绝不发起更新请求;
 * - Windows(NSIS 未签名可用)完整自动更新;
 * - macOS 未签名构建走 manual-download,签名后由 MAC_AUTO_UPDATE_ENABLED 放开;
 * - 其余平台保守降级为 manual-download。
 */
export function resolveUpdateMode(
  platform: string,
  isPackaged: boolean,
  options?: { macAutoUpdate?: boolean },
): UpdateMode {
  if (!isPackaged) return 'disabled'
  if (platform === 'win32') return 'auto'
  if (platform === 'darwin') return options?.macAutoUpdate ? 'auto' : 'manual-download'
  return 'manual-download'
}

/** 进度归一化:clamp 到 [0,100],非法值归 0,保留 1 位小数。 */
export function normalizePercent(percent: number): number {
  if (!Number.isFinite(percent) || percent < 0) return 0
  return Math.round(Math.min(percent, 100) * 10) / 10
}

export type UpdaterEvent =
  | { type: 'check-started'; source: UpdateCheckSource }
  | { type: 'update-available'; source: UpdateCheckSource; mode: UpdateMode; version: string; releaseUrl: string }
  | { type: 'update-not-available'; source: UpdateCheckSource; currentVersion: string }
  | { type: 'download-started'; version: string }
  | { type: 'download-progress'; version: string; percent: number }
  | { type: 'update-downloaded'; version: string }
  | { type: 'update-error'; source: UpdateCheckSource; message: string; releaseUrl: string }

/** 事件归约:决定渲染层可见的更新状态。当前每个事件无条件产出下一状态
 * (无记忆归约);保留 state 形参以维持 reducer 语义,便于将来按当前状态
 * 过滤重复事件。 */
export function reduceUpdaterState(_state: UpdaterState, event: UpdaterEvent): UpdaterState {
  switch (event.type) {
    case 'check-started':
      return { phase: 'checking', source: event.source }
    case 'update-available': {
      if (event.mode === 'manual-download' || event.mode === 'disabled') {
        return {
          phase: 'manual-check-required',
          source: event.source,
          version: event.version,
          releaseUrl: event.releaseUrl,
        }
      }
      return {
        phase: 'notified',
        source: event.source,
        version: event.version,
        releaseUrl: event.releaseUrl,
      }
    }
    case 'update-not-available':
      return { phase: 'up-to-date', currentVersion: event.currentVersion }
    case 'download-started':
      return { phase: 'downloading', version: event.version, percent: 0 }
    case 'download-progress':
      return { phase: 'downloading', version: event.version, percent: normalizePercent(event.percent) }
    case 'update-downloaded':
      return { phase: 'downloaded', version: event.version }
    case 'update-error':
      return { phase: 'error', source: event.source, message: event.message, releaseUrl: event.releaseUrl }
  }
}
