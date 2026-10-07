import assert from 'node:assert/strict'
import { test } from 'vitest'

import { useUpdateStore } from './updateStore.ts'
import type { ManualCheckResult, UpdateMode, UpdaterState } from '../../main/updaterModel.ts'

type AnyRecord = Record<string, any>

function resetStore(overrides: AnyRecord = {}) {
  useUpdateStore.setState({
    appVersion: '',
    mode: 'disabled' as UpdateMode,
    state: { phase: 'idle' } as UpdaterState,
    manualChecking: false,
    manualResult: null,
    dismissedKeys: [],
    ...overrides,
  })
}

function installElectronApi(overrides: AnyRecord = {}) {
  const calls: AnyRecord = { downloads: 0, installs: 0, openedRelease: 0 }
  let stateListener: ((state: UpdaterState) => void) | null = null
  const api = {
    onUpdaterState: (listener: (state: UpdaterState) => void) => {
      stateListener = listener
      return () => {
        stateListener = null
      }
    },
    getUpdaterInfo: async () => ({ appVersion: '0.2.0', mode: 'auto', state: { phase: 'idle' } }),
    checkForUpdates: async () => ({ outcome: 'up-to-date', currentVersion: '0.2.0' }) as ManualCheckResult,
    startUpdateDownload: () => {
      calls.downloads += 1
    },
    quitAndInstallUpdate: () => {
      calls.installs += 1
    },
    openReleasePage: async () => {
      calls.openedRelease += 1
    },
    ...overrides,
  }
  ;(globalThis as AnyRecord).window = { electronAPI: api }
  return { calls, emitState: (state: UpdaterState) => stateListener?.(state) }
}

test('initialize:拉取版本信息并订阅主进程状态推送', async () => {
  resetStore()
  const { emitState } = installElectronApi()
  useUpdateStore.getState().initialize()
  await Promise.resolve()
  assert.equal(useUpdateStore.getState().appVersion, '0.2.0')
  assert.equal(useUpdateStore.getState().mode, 'auto')

  emitState({ phase: 'notified', source: 'auto', version: '0.3.0', releaseUrl: 'https://example.com' })
  assert.equal(useUpdateStore.getState().state.phase, 'notified')
})

test('initialize:无 electronAPI(网页预览)时安全跳过', () => {
  resetStore()
  delete (globalThis as AnyRecord).window
  useUpdateStore.getState().initialize()
  assert.equal(useUpdateStore.getState().mode, 'disabled')
})

test('checkForUpdates:结果落 manualResult,失败不抛出', async () => {
  resetStore({ mode: 'auto' })
  installElectronApi()
  await useUpdateStore.getState().checkForUpdates()
  assert.deepEqual(useUpdateStore.getState().manualResult, { outcome: 'up-to-date', currentVersion: '0.2.0' })
  assert.equal(useUpdateStore.getState().manualChecking, false)

  installElectronApi({
    checkForUpdates: async () => {
      throw new Error('boom')
    },
  })
  await useUpdateStore.getState().checkForUpdates()
  assert.equal(useUpdateStore.getState().manualResult?.outcome, 'disabled', '异常兜底为 disabled,不向外抛错')
})

test('runBannerAction:dismiss 记忆 key;download/install/open 经 IPC 转发', () => {
  resetStore({ mode: 'auto', state: { phase: 'notified', source: 'manual', version: '0.3.0', releaseUrl: 'u' } })
  const { calls } = installElectronApi()

  useUpdateStore.getState().runBannerAction('dismiss')
  assert.deepEqual(useUpdateStore.getState().dismissedKeys, ['notified:0.3.0'])

  useUpdateStore.getState().runBannerAction('download')
  assert.equal(calls.downloads, 1)

  useUpdateStore.getState().runBannerAction('install')
  assert.equal(calls.installs, 1)

  useUpdateStore.getState().runBannerAction('open-release-page')
  assert.equal(calls.openedRelease, 1)
})
