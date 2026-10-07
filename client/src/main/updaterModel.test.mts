import assert from 'node:assert/strict'
import { test } from 'vitest'

import {
  MAC_AUTO_UPDATE_ENABLED,
  normalizePercent,
  reduceUpdaterState,
  releasePageUrl,
  releaseTag,
  resolveUpdateMode,
  type UpdaterState,
} from './updaterModel.ts'

test('resolveUpdateMode:未打包(开发态)在所有平台禁用', () => {
  for (const platform of ['win32', 'darwin', 'linux']) {
    assert.equal(resolveUpdateMode(platform, false), 'disabled', `${platform} 未打包必须禁用`)
    assert.equal(
      resolveUpdateMode(platform, false, { macAutoUpdate: true }),
      'disabled',
      '即使放开 mac 开关,开发态也必须禁用',
    )
  }
})

test('resolveUpdateMode:打包后按平台分流', () => {
  assert.equal(resolveUpdateMode('win32', true), 'auto', 'Windows 未签名 NSIS 也可完整自动更新')
  assert.equal(resolveUpdateMode('darwin', true), 'manual-download', '当前未签名 mac 构建走手动下载')
  assert.equal(resolveUpdateMode('darwin', true, { macAutoUpdate: true }), 'auto', '配置签名后放开 mac 自动更新')
  assert.equal(MAC_AUTO_UPDATE_ENABLED, false, '签名就绪前 mac 开关保持关闭')
  assert.equal(resolveUpdateMode('linux', true), 'manual-download', '未知平台保守降级为手动下载')
})

test('releasePageUrl / releaseTag:tag 形式与仓库路径', () => {
  assert.equal(releaseTag('0.2.0'), 'v0.2.0')
  assert.equal(releaseTag('v1.0.0'), 'v1.0.0')
  assert.equal(releasePageUrl('robiteame', 'ComicAgent'), 'https://github.com/robiteame/ComicAgent/releases/latest')
  assert.equal(
    releasePageUrl('robiteame', 'ComicAgent', 'v0.3.0'),
    'https://github.com/robiteame/ComicAgent/releases/tag/v0.3.0',
  )
  assert.equal(
    releasePageUrl('robiteame', 'ComicAgent', '0.3.0'),
    'https://github.com/robiteame/ComicAgent/releases/tag/v0.3.0',
  )
})

test('normalizePercent:clamp 与非法值', () => {
  assert.equal(normalizePercent(42.456), 42.5)
  assert.equal(normalizePercent(120), 100)
  assert.equal(normalizePercent(-3), 0)
  assert.equal(normalizePercent(Number.NaN), 0)
})

test('reduceUpdaterState:Windows 完整链路 checking → notified → downloading → downloaded', () => {
  let state: UpdaterState = { phase: 'idle' }
  const url = 'https://github.com/robiteame/ComicAgent/releases/tag/v0.3.0'

  state = reduceUpdaterState(state, { type: 'check-started', source: 'auto' })
  assert.deepEqual(state, { phase: 'checking', source: 'auto' })

  state = reduceUpdaterState(state, {
    type: 'update-available',
    source: 'auto',
    mode: 'auto',
    version: '0.3.0',
    releaseUrl: url,
  })
  assert.deepEqual(state, { phase: 'notified', source: 'auto', version: '0.3.0', releaseUrl: url })

  state = reduceUpdaterState(state, { type: 'download-started', version: '0.3.0' })
  assert.deepEqual(state, { phase: 'downloading', version: '0.3.0', percent: 0 })

  state = reduceUpdaterState(state, { type: 'download-progress', version: '0.3.0', percent: 55.789 })
  assert.deepEqual(state, { phase: 'downloading', version: '0.3.0', percent: 55.8 })

  state = reduceUpdaterState(state, { type: 'update-downloaded', version: '0.3.0' })
  assert.deepEqual(state, { phase: 'downloaded', version: '0.3.0' })
})

test('reduceUpdaterState:macOS manual-download 模式下发现新版直接进入 manual-check-required', () => {
  const url = 'https://github.com/robiteame/ComicAgent/releases/tag/v0.3.0'
  const state = reduceUpdaterState(
    { phase: 'checking', source: 'auto' },
    { type: 'update-available', source: 'auto', mode: 'manual-download', version: '0.3.0', releaseUrl: url },
  )
  assert.deepEqual(state, { phase: 'manual-check-required', source: 'auto', version: '0.3.0', releaseUrl: url })
})

test('reduceUpdaterState:无新版本与错误', () => {
  assert.deepEqual(
    reduceUpdaterState({ phase: 'idle' }, { type: 'update-not-available', source: 'auto', currentVersion: '0.2.0' }),
    {
      phase: 'up-to-date',
      currentVersion: '0.2.0',
    },
  )
  const errUrl = 'https://github.com/robiteame/ComicAgent/releases/latest'
  assert.deepEqual(
    reduceUpdaterState(
      { phase: 'idle' },
      { type: 'update-error', source: 'manual', message: 'boom', releaseUrl: errUrl },
    ),
    { phase: 'error', source: 'manual', message: 'boom', releaseUrl: errUrl },
  )
})
