// 更新状态 → 横幅 / 设置页视图模型(纯函数,SSR 可测)。
// 文案策略:检测或下载失败一律降级为「前往下载」,绝不以错误弹窗打扰用户;
// 后台(auto 源)静默检查失败连横幅都不出。
import type { ManualCheckResult, UpdateMode, UpdaterState } from '../../main/updaterModel'

export type BannerActionKind = 'download' | 'install' | 'open-release-page' | 'dismiss'

export interface BannerAction {
  kind: BannerActionKind
  label: string
}

export interface UpdateBannerView {
  visible: boolean
  /** 供「稍后」按版本+阶段记忆,新阶段(如下载完成)重新出现。 */
  key: string
  level: 'info' | 'success'
  title: string
  description: string
  percent?: number
  actions: BannerAction[]
  /** 是否展示「查看更新日志」入口(Release 页)。 */
  showReleaseLink: boolean
  note?: string
}

export function bannerKey(phase: string, version: string): string {
  return `${phase}:${version}`
}

export function buildUpdateBannerView(
  state: UpdaterState,
  mode: UpdateMode,
  dismissedKeys: string[],
): UpdateBannerView {
  const view = buildBaseBannerView(state, mode)
  if (!view.visible) return view
  // 「稍后」只记忆可忽略的横幅(key = 阶段+版本);进入新阶段(如下载完成)
  // key 变化,横幅重新出现。
  if (view.actions.some((action) => action.kind === 'dismiss') && dismissedKeys.includes(view.key)) {
    return { ...view, visible: false }
  }
  return view
}

function buildBaseBannerView(state: UpdaterState, mode: UpdateMode): UpdateBannerView {
  const hidden: UpdateBannerView = {
    visible: false,
    key: '',
    level: 'info',
    title: '',
    description: '',
    actions: [],
    showReleaseLink: false,
  }
  if (mode === 'disabled') return hidden

  switch (state.phase) {
    case 'idle':
    case 'checking':
    case 'up-to-date':
      return hidden
    case 'notified':
      if (mode !== 'auto') return hidden
      return {
        visible: true,
        key: bannerKey(state.phase, state.version),
        level: 'info',
        title: `发现新版本 v${state.version}`,
        description: '建议尽快更新以获得最新功能与修复。',
        actions: [
          { kind: 'download', label: '立即更新' },
          { kind: 'dismiss', label: '稍后' },
        ],
        showReleaseLink: true,
      }
    case 'downloading':
      return {
        visible: true,
        key: bannerKey(state.phase, state.version),
        level: 'info',
        title: `正在下载新版本 v${state.version}`,
        description: '下载完成后会提示重启安装,期间可继续使用。',
        percent: state.percent,
        actions: [],
        showReleaseLink: false,
      }
    case 'downloaded':
      return {
        visible: true,
        key: bannerKey(state.phase, state.version),
        level: 'success',
        title: `新版本 v${state.version} 已准备就绪`,
        description: '重启应用即可完成安装。',
        actions: [
          { kind: 'install', label: '重启并安装' },
          { kind: 'dismiss', label: '稍后' },
        ],
        showReleaseLink: false,
        note: '选择「稍后」也不会丢失更新:下次退出应用时将自动安装。',
      }
    case 'manual-check-required':
      return {
        visible: true,
        key: bannerKey(state.phase, state.version),
        level: 'info',
        title: `发现新版本 v${state.version}`,
        description: '当前 macOS 构建未启用自动更新,请前往下载页获取新版。',
        actions: [
          { kind: 'open-release-page', label: '前往下载' },
          { kind: 'dismiss', label: '稍后' },
        ],
        showReleaseLink: false,
        note: '配置应用签名后将启用 macOS 自动更新。',
      }
    case 'error': {
      // 后台静默检查失败不打扰用户;用户主动操作(下载/手动检查)失败才
      // 降级展示「前往下载」。
      if (state.source !== 'manual') return hidden
      return {
        visible: true,
        key: bannerKey(state.phase, ''),
        level: 'info',
        title: '自动更新失败',
        description: '可前往下载页手动获取新版本。',
        actions: [
          { kind: 'open-release-page', label: '前往下载' },
          { kind: 'dismiss', label: '稍后' },
        ],
        showReleaseLink: false,
      }
    }
  }
}

/** 「稍后」只对可忽略的阶段生效(下载中不可关闭)。 */
export function isDismissible(view: UpdateBannerView): boolean {
  return view.actions.some((action) => action.kind === 'dismiss')
}

/** 设置页的平台通道说明文案(含 mac 未签名降级说明)。 */
export function updateModeHint(mode: UpdateMode): string {
  if (mode === 'disabled') return '开发模式(未打包)下更新检查已禁用,打包构建后自动启用。'
  if (mode === 'manual-download')
    return '当前构建未启用自动更新:检测到新版本后会引导前往下载页手动获取,配置应用签名后将启用自动更新。'
  return '支持完整自动更新:检测到新版本后可一键下载,下载完成后重启应用即可安装。'
}

export interface SettingsUpdateSummary {
  title: string
  detail: string
  showLink: boolean
  action?: BannerAction
}

export interface SettingsUpdateInputs {
  mode: UpdateMode
  appVersion: string
  checking: boolean
  manualResult: ManualCheckResult | null
}

export function describeSettingsUpdate({
  mode,
  appVersion,
  checking,
  manualResult,
}: SettingsUpdateInputs): SettingsUpdateSummary {
  const versionLine = appVersion ? `当前版本 v${appVersion}` : '当前版本未知'
  if (checking) return { title: versionLine, detail: '正在检查更新...', showLink: false }
  if (mode === 'disabled') {
    return { title: versionLine, detail: '开发模式(未打包)下更新检查已禁用。', showLink: false }
  }
  if (!manualResult) {
    return {
      title: versionLine,
      detail: '点击「检查更新」获取最新版本信息;应用也会每天在后台静默检查一次。',
      showLink: false,
    }
  }
  switch (manualResult.outcome) {
    case 'up-to-date':
      return { title: versionLine, detail: `当前已是最新版本 (v${manualResult.currentVersion})。`, showLink: false }
    case 'available': {
      const state = manualResult.state
      if (state.phase === 'manual-check-required') {
        return {
          title: versionLine,
          detail: `发现新版本 v${state.version},请前往下载页获取。`,
          showLink: true,
          action: { kind: 'open-release-page', label: '前往下载' },
        }
      }
      return {
        title: versionLine,
        detail: `发现新版本 v${'version' in state ? state.version : ''},可在上方横幅中立即更新。`,
        showLink: true,
        action: state.phase === 'notified' ? { kind: 'download', label: '立即更新' } : undefined,
      }
    }
    case 'error':
      return {
        title: versionLine,
        detail: '检查更新失败,可前往下载页手动获取新版本。',
        showLink: true,
        action: { kind: 'open-release-page', label: '前往下载' },
      }
    case 'disabled':
      return { title: versionLine, detail: '开发模式(未打包)下更新检查已禁用。', showLink: false }
  }
}
