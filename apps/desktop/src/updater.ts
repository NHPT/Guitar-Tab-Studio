import { app, dialog, type BrowserWindow } from 'electron'
import { autoUpdater } from 'electron-updater'
import type { DesktopUpdateProgress } from '@gts/core'
import { acceptsPrereleaseUpdates } from './update-channel.js'
import { normalizeUpdateProgress } from './update-progress.js'

const updateIntervalMs = 6 * 60 * 60 * 1000

export class DesktopUpdater {
  private interval?: NodeJS.Timeout
  private manualCheck = false
  private downloadInProgress = false
  private downloadVersion = ''
  private lastDownloadProgress?: DesktopUpdateProgress

  constructor(private readonly windowProvider: () => BrowserWindow | undefined) {
    autoUpdater.autoDownload = false
    autoUpdater.autoInstallOnAppQuit = true
    autoUpdater.allowPrerelease = acceptsPrereleaseUpdates(app.getVersion())
    autoUpdater.on('update-available', (info) => {
      this.manualCheck = false
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
          if (response === 0) this.startDownload(info.version)
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
    autoUpdater.on('download-progress', (metrics) => {
      if (!this.downloadInProgress) return
      const progress = normalizeUpdateProgress(this.downloadVersion, metrics)
      this.lastDownloadProgress = progress
      this.publishProgress(progress)
    })
    autoUpdater.on('update-downloaded', (info) => {
      const previous = this.lastDownloadProgress
      this.downloadInProgress = false
      this.downloadVersion = info.version
      this.lastDownloadProgress = {
        phase: 'ready',
        version: info.version,
        percent: 100,
        transferredBytes: Math.max(
          previous?.transferredBytes ?? 0,
          previous?.totalBytes ?? 0,
        ),
        totalBytes: previous?.totalBytes ?? 0,
        bytesPerSecond: 0,
      }
      this.publishProgress(this.lastDownloadProgress)
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
      const checkingManually = this.manualCheck
      const downloading = this.downloadInProgress
      this.manualCheck = false
      this.downloadInProgress = false
      if (downloading) {
        const previous = this.lastDownloadProgress
        this.lastDownloadProgress = {
          phase: 'failed',
          version: this.downloadVersion,
          percent: previous?.percent ?? 0,
          transferredBytes: previous?.transferredBytes ?? 0,
          totalBytes: previous?.totalBytes ?? 0,
          bytesPerSecond: 0,
          error: error.message,
        }
        this.publishProgress(this.lastDownloadProgress)
      }
      if (!checkingManually && !downloading) return
      const window = this.windowProvider()
      if (window) {
        dialog.showErrorBox(
          downloading ? '下载更新失败' : '检查更新失败',
          error.message,
        )
      }
    })
  }

  start(): void {
    if (!app.isPackaged) return
    const check = () => {
      if (!this.downloadInProgress) void autoUpdater.checkForUpdates()
    }
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
    if (this.downloadInProgress) {
      const window = this.windowProvider()
      if (window) {
        void dialog.showMessageBox(window, {
          type: 'info',
          title: '正在下载更新',
          message: `Guitar Tab Studio ${this.downloadVersion} 正在下载`,
          detail: `当前进度 ${Math.round(this.lastDownloadProgress?.percent ?? 0)}%`,
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
    this.windowProvider()?.setProgressBar(-1)
  }

  private startDownload(version: string): void {
    if (this.downloadInProgress) return
    this.downloadInProgress = true
    this.downloadVersion = version
    this.lastDownloadProgress = {
      phase: 'downloading',
      version,
      percent: 0,
      transferredBytes: 0,
      totalBytes: 0,
      bytesPerSecond: 0,
    }
    this.publishProgress(this.lastDownloadProgress)
    void autoUpdater.downloadUpdate().catch((error: unknown) => {
      if (!this.downloadInProgress) return
      const resolvedError =
        error instanceof Error ? error : new Error('更新下载失败')
      this.downloadInProgress = false
      this.lastDownloadProgress = {
        phase: 'failed',
        version: this.downloadVersion,
        percent: this.lastDownloadProgress?.percent ?? 0,
        transferredBytes: this.lastDownloadProgress?.transferredBytes ?? 0,
        totalBytes: this.lastDownloadProgress?.totalBytes ?? 0,
        bytesPerSecond: 0,
        error: resolvedError.message,
      }
      this.publishProgress(this.lastDownloadProgress)
      dialog.showErrorBox('下载更新失败', resolvedError.message)
    })
  }

  private publishProgress(progress: DesktopUpdateProgress): void {
    const window = this.windowProvider()
    if (!window || window.isDestroyed()) return
    window.setProgressBar(
      progress.phase === 'downloading' ? progress.percent / 100 : -1,
    )
    if (!window.webContents.isDestroyed()) {
      window.webContents.send('studio:update-progress', progress)
    }
  }
}
