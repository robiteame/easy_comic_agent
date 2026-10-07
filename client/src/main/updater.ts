// 主进程自动更新接线:真实 autoUpdater + IPC 通道 + 定时器。
// 状态机与平台分流逻辑在 updaterController.ts / updaterModel.ts(纯函数,
// 可单测);本文件只做副作用组装。
import { app, ipcMain, shell, type WebContents } from 'electron'
import { autoUpdater } from 'electron-updater'
import { createUpdaterController, type UpdaterController } from './updaterController'

export const UPDATER_STATE_CHANNEL = 'updater:state'

// 把控制器接入 Electron:IPC 通道 + 真实 autoUpdater + 定时器。
// getWebContents 每次取当前窗口,窗口重建后广播仍然可达;窗口缺失时静默丢弃。
export function setupUpdater(getWebContents: () => WebContents | null): UpdaterController {
  // autoDownload 必须关闭:下载由用户在横幅上点击「立即更新」触发;
  // autoInstallOnAppQuit 保持开启,用户错过重启提示时下次退出自动生效。
  autoUpdater.autoDownload = false
  autoUpdater.autoInstallOnAppQuit = true

  const controller = createUpdaterController({
    autoUpdater,
    platform: process.platform,
    isPackaged: app.isPackaged,
    appVersion: app.getVersion(),
    broadcast: (state) => {
      const webContents = getWebContents()
      if (!webContents || webContents.isDestroyed()) return
      webContents.send(UPDATER_STATE_CHANNEL, state)
    },
    schedule: (fn, delayMs) => {
      const timer = setTimeout(fn, delayMs)
      timer.unref?.()
      return () => clearTimeout(timer)
    },
    repeat: (fn, intervalMs) => {
      const timer = setInterval(fn, intervalMs)
      timer.unref?.()
      return () => clearInterval(timer)
    },
  })

  ipcMain.handle('updater:get-info', () => ({
    appVersion: app.getVersion(),
    mode: controller.mode,
    state: controller.getState(),
  }))
  ipcMain.handle('updater:check', () => controller.check('manual'))
  ipcMain.on('updater:download', () => controller.downloadUpdate())
  ipcMain.on('updater:install', () => controller.installUpdate())
  // 只打开由主进程常量推导出的 GitHub Release 页,不接受渲染层传入任意
  // URL(shell.openExternal 的注入面)。
  ipcMain.handle('updater:open-release-page', () => {
    const url = controller.releaseUrlFor(controller.getState())
    return shell.openExternal(url).catch((error: unknown) => {
      console.error('[updater] 打开 Release 页面失败:', error)
    })
  })

  return controller
}
