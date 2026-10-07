import assert from 'node:assert/strict'
import { test } from 'vitest'

import { expectRender, visibleText } from '../test-support/ssrTestHelper.mts'
import { useUpdateStore } from '../stores/updateStore.ts'
import type { UpdateMode, UpdaterState } from '../../main/updaterModel.ts'
import UpdateBanner from './UpdateBanner'

const URL = 'https://github.com/robiteame/ComicAgent/releases/tag/v0.3.0'

function setStore(mode: UpdateMode, state: UpdaterState) {
  useUpdateStore.setState({
    mode,
    state,
    dismissedKeys: [],
    manualChecking: false,
    manualResult: null,
    appVersion: '0.2.0',
  })
}

function renderBanner(): string {
  return visibleText(expectRender(UpdateBanner, {}))
}

test('默认状态(未初始化)不渲染任何横幅', () => {
  setStore('disabled', { phase: 'idle' })
  const html = expectRender(UpdateBanner, {})
  assert.equal(html, '', 'idle 时横幅完全不出现在 DOM')
})

test('Windows notified:版本号 + 立即更新/稍后 + 查看更新日志', () => {
  setStore('auto', { phase: 'notified', source: 'auto', version: '0.3.0', releaseUrl: URL })
  const text = renderBanner()
  assert.ok(text.includes('发现新版本'), '应提示发现新版本')
  assert.ok(text.includes('v0.3.0'), '应展示版本号')
  assert.ok(text.includes('立即更新'))
  assert.ok(text.includes('稍后'))
  assert.ok(text.includes('查看更新日志'), '应提供 Release 页更新日志入口')
})

test('Windows downloading:展示进度百分比', () => {
  setStore('auto', { phase: 'downloading', version: '0.3.0', percent: 42.5 })
  const text = renderBanner()
  assert.ok(text.includes('正在下载'), '应展示下载中标题')
  assert.ok(text.includes('42.5%'), '应展示进度百分比')
})

test('Windows downloaded:提示重启并安装', () => {
  setStore('auto', { phase: 'downloaded', version: '0.3.0' })
  const text = renderBanner()
  assert.ok(text.includes('已准备就绪'))
  assert.ok(text.includes('重启并安装'))
})

test('macOS:前往下载 + 配置签名说明', () => {
  setStore('manual-download', { phase: 'manual-check-required', source: 'auto', version: '0.3.0', releaseUrl: URL })
  const text = renderBanner()
  assert.ok(text.includes('发现新版本'))
  assert.ok(text.includes('前往下载'), 'mac 路径应引导前往下载页')
  assert.ok(text.includes('配置应用签名后'), '文案应注明签名后启用自动更新')
})

test('手动操作失败降级为前往下载,后台失败静默', () => {
  const errUrl = 'https://github.com/robiteame/ComicAgent/releases/latest'
  setStore('auto', { phase: 'error', source: 'manual', message: 'boom', releaseUrl: errUrl })
  const manual = renderBanner()
  assert.ok(manual.includes('自动更新失败'))
  assert.ok(manual.includes('前往下载'))

  setStore('auto', { phase: 'error', source: 'auto', message: 'boom', releaseUrl: errUrl })
  assert.equal(expectRender(UpdateBanner, {}), '', 'auto 源失败不出横幅')
})
