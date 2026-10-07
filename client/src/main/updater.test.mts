import assert from 'node:assert/strict'
import { test } from 'vitest'

import { createUpdaterController, type AutoUpdaterLike, type UpdaterControllerDeps } from './updaterController.ts'
import type { UpdaterState } from './updaterModel.ts'

// electron-updater 的内存替身:事件可手动触发,副作用只计数不执行。
class StubAutoUpdater implements AutoUpdaterLike {
  checks = 0
  downloads = 0
  installs = 0
  private listeners = new Map<string, Array<(...args: any[]) => void>>()
  /** 每次 checkForUpdates 时的既定剧本(发事件 / 抛错)。 */
  checkScript: () => void = () => {}

  on(event: string, listener: (...args: any[]) => void): unknown {
    const list = this.listeners.get(event) ?? []
    list.push(listener)
    this.listeners.set(event, list)
    return this
  }

  emit(event: string, ...args: any[]): void {
    for (const listener of this.listeners.get(event) ?? []) listener(...args)
  }

  async checkForUpdates(): Promise<unknown> {
    this.checks += 1
    this.checkScript()
    return null
  }

  async downloadUpdate(): Promise<unknown> {
    this.downloads += 1
    return null
  }

  quitAndInstall(): void {
    this.installs += 1
  }
}

function fakeTimers() {
  const scheduled: { fn: () => void; delayMs: number; canceled: boolean }[] = []
  const repeated: { fn: () => void; intervalMs: number; canceled: boolean }[] = []
  return {
    scheduled,
    repeated,
    deps: {
      schedule: (fn: () => void, delayMs: number) => {
        const item = { fn, delayMs, canceled: false }
        scheduled.push(item)
        return () => {
          item.canceled = true
        }
      },
      repeat: (fn: () => void, intervalMs: number) => {
        const item = { fn, intervalMs, canceled: false }
        repeated.push(item)
        return () => {
          item.canceled = true
        }
      },
    } satisfies Pick<UpdaterControllerDeps, 'schedule' | 'repeat'>,
    runScheduled: () => {
      for (const item of scheduled) if (!item.canceled) item.fn()
    },
  }
}

function makeController(overrides?: Partial<UpdaterControllerDeps>) {
  const stub = new StubAutoUpdater()
  const timers = fakeTimers()
  const states: UpdaterState[] = []
  const controller = createUpdaterController({
    autoUpdater: stub,
    platform: 'win32',
    isPackaged: true,
    appVersion: '0.2.0',
    broadcast: (state) => states.push(state),
    schedule: timers.deps.schedule,
    repeat: timers.deps.repeat,
    logger: { log: () => {}, error: () => {} },
    ...overrides,
  })
  return { controller, stub, timers, states }
}

test('开发态(未打包)完全禁用:不排定时器、不发任何更新请求', async () => {
  const { controller, stub, timers } = makeController({ isPackaged: false })
  assert.equal(controller.mode, 'disabled')
  const cancel = controller.start()
  assert.equal(timers.scheduled.length, 0, '开发态不得安排任何延迟检查')
  assert.equal(timers.repeated.length, 0, '开发态不得安排任何周期检查')
  const result = await controller.check('manual')
  assert.deepEqual(result, { outcome: 'disabled' })
  assert.equal(stub.checks, 0, '开发态不得调用 checkForUpdates')
  controller.downloadUpdate()
  controller.installUpdate()
  assert.equal(stub.downloads, 0)
  assert.equal(stub.installs, 0)
  cancel()
})

test('Windows 启动检查:10s 延迟 + 24h 周期,发现新版进入 notified', async () => {
  const { controller, stub, timers, states } = makeController()
  assert.equal(controller.mode, 'auto')
  controller.start()
  assert.equal(timers.scheduled.length, 1)
  assert.equal(timers.scheduled[0].delayMs, 10_000)

  stub.checkScript = () => stub.emit('update-available', { version: '0.3.0' })
  timers.runScheduled()
  await Promise.resolve()
  assert.equal(stub.checks, 1)
  assert.equal(timers.repeated.length, 1, '首检后进入周期检查')
  assert.equal(timers.repeated[0].intervalMs, 24 * 60 * 60 * 1000)

  assert.deepEqual(states[states.length - 1], {
    phase: 'notified',
    source: 'auto',
    version: '0.3.0',
    releaseUrl: 'https://github.com/robiteame/ComicAgent/releases/tag/v0.3.0',
  })
})

test('手动检查复用进行中的请求,不重复调用 checkForUpdates', async () => {
  const { controller, stub } = makeController()
  stub.checkForUpdates = () =>
    new Promise((resolve) => {
      stub.checks += 1
      stub.emit('update-not-available', { version: '0.2.0' })
      setTimeout(resolve, 0)
    })
  const first = controller.check('manual')
  const second = controller.check('auto')
  assert.equal(stub.checks, 1, '并发检查应串行复用同一次请求')
  const [a, b] = await Promise.all([first, second])
  assert.equal(a.outcome, 'up-to-date')
  assert.equal(b.outcome, 'up-to-date')
})

test('Windows 完整链路:notified → 下载进度 → downloaded → quitAndInstall', async () => {
  const { controller, stub, states } = makeController()

  stub.checkScript = () => stub.emit('update-available', { version: '0.3.0' })
  const result = await controller.check('manual')
  assert.equal(result.outcome, 'available')

  // 未 downloaded 前安装请求必须被忽略。
  controller.installUpdate()
  assert.equal(stub.installs, 0)

  controller.downloadUpdate()
  assert.equal(stub.downloads, 1)
  assert.deepEqual(states[states.length - 1], { phase: 'downloading', version: '0.3.0', percent: 0 })

  stub.emit('download-progress', { percent: 42.46 })
  assert.deepEqual(states[states.length - 1], { phase: 'downloading', version: '0.3.0', percent: 42.5 })

  stub.emit('update-downloaded', { version: '0.3.0' })
  assert.deepEqual(states[states.length - 1], { phase: 'downloaded', version: '0.3.0' })

  controller.installUpdate()
  assert.equal(stub.installs, 1, 'downloaded 状态下用户确认后调用 quitAndInstall')
})

test('macOS 分流:发现新版进入 manual-check-required,不下载不安装', async () => {
  const { controller, stub, states } = makeController({ platform: 'darwin' })
  assert.equal(controller.mode, 'manual-download')

  stub.checkScript = () => stub.emit('update-available', { version: '0.3.0' })
  const result = await controller.check('manual')
  assert.equal(result.outcome, 'available')
  assert.deepEqual(states[states.length - 1], {
    phase: 'manual-check-required',
    source: 'manual',
    version: '0.3.0',
    releaseUrl: 'https://github.com/robiteame/ComicAgent/releases/tag/v0.3.0',
  })

  controller.downloadUpdate()
  controller.installUpdate()
  assert.equal(stub.downloads, 0, 'mac 未签名构建不得触发下载')
  assert.equal(stub.installs, 0)
})

test('检测失败:状态进入 error(带 releaseUrl 兜底),手动检查返回 error 而不抛出', async () => {
  const { controller, stub, states } = makeController()
  stub.checkScript = () => {
    stub.emit('error', new Error('network down'))
    throw new Error('network down')
  }
  const result = await controller.check('manual')
  assert.equal(result.outcome, 'error')
  assert.ok(result.outcome === 'error' && result.message.includes('network down'))
  assert.ok(result.outcome === 'error' && result.releaseUrl.includes('releases/latest'))
  assert.equal(states[states.length - 1]?.phase, 'error')
  const lastState = states[states.length - 1]
  if (lastState?.phase === 'error') assert.equal(lastState.source, 'manual')
})

test('后台静默检查失败同样归因为 auto 源 error,不抛出', async () => {
  const { controller, stub, states } = makeController()
  stub.checkScript = () => {
    stub.emit('error', new Error('offline'))
    throw new Error('offline')
  }
  await controller.check('auto')
  assert.equal(states[states.length - 1]?.phase, 'error')
  const lastState = states[states.length - 1]
  if (lastState?.phase === 'error') assert.equal(lastState.source, 'auto')
})

test('error 状态下允许重新发起下载(用户重试路径)', () => {
  const { controller, stub, states } = makeController()
  // 直接构造 error 状态:先触发一次失败检查。
  stub.checkScript = () => {
    stub.emit('error', new Error('boom'))
    throw new Error('boom')
  }
  void controller.check('manual').catch(() => undefined)
  controller.downloadUpdate()
  assert.equal(stub.downloads, 1, 'error 状态应允许重试下载')
  assert.equal(states[states.length - 1]?.phase, 'downloading')
})

test('start() 返回的取消函数可撤销尚未触发的检查', () => {
  const { controller, timers } = makeController()
  const cancel = controller.start()
  cancel()
  timers.runScheduled()
  assert.ok(timers.scheduled[0].canceled, '取消后延迟检查不再执行')
})

test('downloadUpdate 仅在 notified/error 状态生效', () => {
  const { controller, stub } = makeController()
  controller.downloadUpdate()
  assert.equal(stub.downloads, 0, 'idle 状态不得下载')
})
