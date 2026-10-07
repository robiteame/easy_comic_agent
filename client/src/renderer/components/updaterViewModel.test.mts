import assert from 'node:assert/strict'
import { test } from 'vitest'

import {
  bannerKey,
  buildUpdateBannerView,
  describeSettingsUpdate,
  isDismissible,
  updateModeHint,
} from './updaterViewModel.ts'
import type { UpdaterState } from '../../main/updaterModel.ts'

const URL = 'https://github.com/robiteame/ComicAgent/releases/tag/v0.3.0'

function bannerViewOf(state: UpdaterState, mode: 'auto' | 'manual-download' = 'auto', dismissed: string[] = []) {
  const view = buildUpdateBannerView(state, mode, dismissed)
  return { view, labels: view.actions.map((action) => action.label) }
}

test('横幅可见性:idle/checking/up-to-date 与开发态一律不展示', () => {
  for (const state of [
    { phase: 'idle' },
    { phase: 'checking', source: 'auto' },
    { phase: 'up-to-date', currentVersion: '0.2.0' },
  ] as UpdaterState[]) {
    assert.equal(buildUpdateBannerView(state, 'auto', []).visible, false)
  }
  assert.equal(
    buildUpdateBannerView({ phase: 'notified', source: 'auto', version: '0.3.0', releaseUrl: URL }, 'disabled', [])
      .visible,
    false,
    '开发态禁用更新时即使有新版本也不展示',
  )
})

test('Windows notified:版本号 + 立即更新/稍后 + 更新日志入口', () => {
  const { view, labels } = bannerViewOf({ phase: 'notified', source: 'auto', version: '0.3.0', releaseUrl: URL })
  assert.equal(view.visible, true)
  assert.ok(view.title.includes('v0.3.0'), '横幅应展示新版本号')
  assert.deepEqual(labels, ['立即更新', '稍后'])
  assert.equal(view.showReleaseLink, true, '应提供更新日志(Release 页)入口')
  assert.equal(view.key, bannerKey('notified', '0.3.0'))
})

test('Windows downloading:展示进度且不可关闭', () => {
  const { view, labels } = bannerViewOf({ phase: 'downloading', version: '0.3.0', percent: 42.5 })
  assert.equal(view.visible, true)
  assert.equal(view.percent, 42.5)
  assert.deepEqual(labels, [], '下载中不提供任何操作')
  assert.equal(view.showReleaseLink, false)
})

test('Windows downloaded:提示重启安装,并说明稍后也能自动生效', () => {
  const { view, labels } = bannerViewOf({ phase: 'downloaded', version: '0.3.0' })
  assert.equal(view.visible, true)
  assert.deepEqual(labels, ['重启并安装', '稍后'])
  assert.ok(view.note?.includes('自动安装'), '应说明错过后下次退出自动安装')
})

test('macOS manual-check-required:前往下载 + 签名说明', () => {
  const { view, labels } = bannerViewOf(
    { phase: 'manual-check-required', source: 'auto', version: '0.3.0', releaseUrl: URL },
    'manual-download',
  )
  assert.equal(view.visible, true)
  assert.ok(view.title.includes('v0.3.0'))
  assert.deepEqual(labels, ['前往下载', '稍后'])
  assert.ok(view.note?.includes('签名'), '文案应注明配置签名后启用自动更新')
})

test('失败降级:仅 manual 源错误展示「前往下载」,auto 源静默', () => {
  const errUrl = 'https://github.com/robiteame/ComicAgent/releases/latest'
  const manual = bannerViewOf({ phase: 'error', source: 'manual', message: 'boom', releaseUrl: errUrl })
  assert.equal(manual.view.visible, true)
  assert.deepEqual(manual.labels, ['前往下载', '稍后'])
  assert.equal(manual.view.title, '自动更新失败')

  const auto = bannerViewOf({ phase: 'error', source: 'auto', message: 'boom', releaseUrl: errUrl })
  assert.equal(auto.view.visible, false, '后台静默检查失败绝不打扰用户')
})

test('「稍后」按 key 记忆:同 key 隐藏,状态推进后重新出现', () => {
  const notified: UpdaterState = { phase: 'notified', source: 'auto', version: '0.3.0', releaseUrl: URL }
  const dismissed = [bannerKey('notified', '0.3.0')]
  assert.equal(buildUpdateBannerView(notified, 'auto', dismissed).visible, false, ' dismissed 后同版本同阶段不再显示')
  const downloading: UpdaterState = { phase: 'downloading', version: '0.3.0', percent: 1 }
  assert.equal(buildUpdateBannerView(downloading, 'auto', dismissed).visible, true, '进入下载阶段横幅重新出现')
  const nextVersion: UpdaterState = { phase: 'notified', source: 'auto', version: '0.4.0', releaseUrl: URL }
  assert.equal(buildUpdateBannerView(nextVersion, 'auto', dismissed).visible, true, '新版本号重新提醒')
})

test('isDismissible:下载中不可忽略', () => {
  assert.equal(
    isDismissible(bannerViewOf({ phase: 'notified', source: 'auto', version: '0.3.0', releaseUrl: URL }).view),
    true,
  )
  assert.equal(isDismissible(bannerViewOf({ phase: 'downloading', version: '0.3.0', percent: 1 }).view), false)
})

test('updateModeHint:平台通道说明文案', () => {
  assert.ok(updateModeHint('manual-download').includes('签名'), 'mac 文案应说明签名后启用自动更新')
  assert.ok(updateModeHint('disabled').includes('禁用'))
  assert.ok(updateModeHint('auto').includes('自动更新'))
})

test('describeSettingsUpdate:手动检查各结果的展示', () => {
  const base = { mode: 'auto' as const, appVersion: '0.2.0', checking: false, manualResult: null }
  assert.ok(describeSettingsUpdate({ ...base, checking: true }).detail.includes('正在检查'))
  assert.ok(describeSettingsUpdate(base).detail.includes('检查更新'))
  assert.ok(describeSettingsUpdate({ ...base, mode: 'disabled' }).detail.includes('禁用'))

  const upToDate = describeSettingsUpdate({ ...base, manualResult: { outcome: 'up-to-date', currentVersion: '0.2.0' } })
  assert.ok(upToDate.detail.includes('最新版本'))

  const available = describeSettingsUpdate({
    ...base,
    manualResult: {
      outcome: 'available',
      state: { phase: 'notified', source: 'manual', version: '0.3.0', releaseUrl: URL },
    },
  })
  assert.ok(available.detail.includes('v0.3.0'))
  assert.equal(available.action?.kind, 'download', 'Windows 检查到新版应提供立即更新')

  const availableMac = describeSettingsUpdate({
    ...base,
    mode: 'manual-download',
    manualResult: {
      outcome: 'available',
      state: { phase: 'manual-check-required', source: 'manual', version: '0.3.0', releaseUrl: URL },
    },
  })
  assert.ok(availableMac.detail.includes('前往下载'))
  assert.equal(availableMac.action?.kind, 'open-release-page')

  const failed = describeSettingsUpdate({
    ...base,
    manualResult: {
      outcome: 'error',
      message: 'offline',
      releaseUrl: 'https://github.com/robiteame/ComicAgent/releases/latest',
    },
  })
  assert.ok(failed.detail.includes('前往下载'), '失败降级为前往下载,不弹错误')
  assert.equal(failed.showLink, true)
})
