import { app, dialog, type BrowserWindow } from 'electron'
import { autoUpdater } from 'electron-updater'
import { acceptsPrereleaseUpdates } from './update-channel.js'

const updateIntervalMs = 6 * 60 * 60 * 1000

export class DesktopUpdater {
  private interval?: NodeJS.Timeout
  private manualCheck = false

  constructor(private readonly windowProvider: () => BrowserWindow | undefined) {
    autoUpdater.autoDownload = false
    autoUpdater.autoInstallOnAppQuit = true
    autoUpdater.allowPrerelease = acceptsPrereleaseUpdates(app.getVersion())
    autoUpdater.on('update-available', (info) => {
      const window = this.windowProvider()
      if (!window) return
      void dialog
        .showMessageBox(window, {
          type: 'info',
          title: '发现新版本',
          message: `Guitar Tab Studio ${info.version} 可用`,
          detail: '下载完成后可立即重启安装。项目数据不会被删除。',
          buttons: ['下载更新', '稍后'],
          defaultId: 0,
          cancelId: 1,
        })
        .then(({ response }) => {
          if (response === 0) void autoUpdater.downloadUpdate()
        })
    })
    autoUpdater.on('update-not-available', () => {
      if (!this.manualCheck) return
      this.manualCheck = false
      const window = this.windowProvider()
      if (window) {
        void dialog.showMessageBox(window, {
          type: 'info',
          title: '检查更新',
          message: '当前已是最新版本',
        })
      }
    })
    autoUpdater.on('update-downloaded', (info) => {
      const window = this.windowProvider()
      if (!window) return
      void dialog
        .showMessageBox(window, {
          type: 'info',
          title: '更新已就绪',
          message: `Guitar Tab Studio ${info.version} 已下载`,
          buttons: ['重启并安装', '退出时安装'],
          defaultId: 0,
          cancelId: 1,
        })
        .then(({ response }) => {
          if (response === 0) autoUpdater.quitAndInstall()
        })
    })
    autoUpdater.on('error', (error) => {
      if (!this.manualCheck) return
      this.manualCheck = false
      const window = this.windowProvider()
      if (window) {
        dialog.showErrorBox('检查更新失败', error.message)
      }
    })
  }

  start(): void {
    if (!app.isPackaged) return
    const check = () => void autoUpdater.checkForUpdates()
    setTimeout(check, 10_000)
    this.interval = setInterval(check, updateIntervalMs)
    this.interval.unref()
  }

  checkManually(): void {
    if (!app.isPackaged) {
      const window = this.windowProvider()
      if (window) {
        void dialog.showMessageBox(window, {
          type: 'info',
          title: '检查更新',
          message: '开发构建不连接更新通道',
        })
      }
      return
    }
    this.manualCheck = true
    void autoUpdater.checkForUpdates()
  }

  stop(): void {
    if (this.interval) clearInterval(this.interval)
    this.interval = undefined
  }
}
