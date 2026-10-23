import { resolve } from 'node:path'
import { app, dialog, session, type BrowserWindow } from 'electron'
import { autoUpdater } from 'electron-updater'
import {
  CancellationError,
  CancellationToken,
  type DownloadOptions,
} from 'builder-util-runtime'
import type { DesktopUpdateProgress } from '@gts/core'
import { ResumableUpdateDownloader } from './resumable-update-download.js'
import { acceptsPrereleaseUpdates } from './update-channel.js'
import { normalizeUpdateProgress } from './update-progress.js'

const updateIntervalMs = 6 * 60 * 60 * 1000

interface DownloadCapableExecutor {
  download(
    url: URL,
    destination: string,
    options: DownloadOptions,
  ): Promise<string>
}

interface UpdaterWithExecutor {
  httpExecutor: DownloadCapableExecutor
}

type CancellationIntent = 'pause' | 'cancel'

export class DesktopUpdater {
  private interval?: NodeJS.Timeout
  private manualCheck = false
  private downloadInProgress = false
  private downloadVersion = ''
  private lastDownloadProgress?: DesktopUpdateProgress
  private downloadCancellation?: CancellationToken
  private cancellationIntent?: CancellationIntent
  private downloadTask?: Promise<void>
  private readonly downloader: ResumableUpdateDownloader

  constructor(
    private readonly windowProvider: () => BrowserWindow | undefined,
    private readonly packagedRuntime = app.isPackaged,
  ) {
    autoUpdater.autoDownload = false
    autoUpdater.autoInstallOnAppQuit = true
    autoUpdater.allowPrerelease = acceptsPrereleaseUpdates(app.getVersion())
    autoUpdater.disableDifferentialDownload = true
    autoUpdater.disableWebInstaller = true

    const updateSession = session.fromPartition('electron-updater', {
      cache: false,
    })
    this.downloader = new ResumableUpdateDownloader(
      resolve(app.getPath('userData'), 'update-downloads'),
      (input, init) => updateSession.fetch(input.toString(), init),
    )
    const executor = (autoUpdater as unknown as UpdaterWithExecutor)
      .httpExecutor
    executor.download = (url, destination, options) =>
      this.downloader.download(url, destination, options)

    autoUpdater.on('update-available', (info) => {
      this.manualCheck = false
      if (
        this.downloadVersion === info.version &&
        this.lastDownloadProgress &&
        ['downloading', 'paused', 'ready'].includes(
          this.lastDownloadProgress.phase,
        )
      ) {
        this.publishProgress(this.lastDownloadProgress)
        return
      }
      this.downloadVersion = info.version
      this.publishProgress({
        phase: 'available',
        version: info.version,
        percent: 0,
        transferredBytes: 0,
        totalBytes: 0,
        bytesPerSecond: 0,
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
      this.publishProgress(
        normalizeUpdateProgress(this.downloadVersion, metrics),
      )
    })
    autoUpdater.on('update-downloaded', (info) => {
      const previous = this.lastDownloadProgress
      this.downloadInProgress = false
      this.downloadVersion = info.version
      this.publishProgress({
        phase: 'ready',
        version: info.version,
        percent: 100,
        transferredBytes: Math.max(
          previous?.transferredBytes ?? 0,
          previous?.totalBytes ?? 0,
        ),
        totalBytes: previous?.totalBytes ?? 0,
        bytesPerSecond: 0,
      })
    })
    autoUpdater.on('error', (error) => {
      const checkingManually = this.manualCheck
      this.manualCheck = false
      if (this.downloadInProgress) return
      if (!checkingManually) return
      const window = this.windowProvider()
      if (window) dialog.showErrorBox('检查更新失败', error.message)
    })
  }

  start(): void {
    if (!this.packagedRuntime) return
    const check = () => {
      const phase = this.lastDownloadProgress?.phase
      if (
        !this.downloadInProgress &&
        phase !== 'paused' &&
        phase !== 'ready'
      ) {
        void autoUpdater.checkForUpdates()
      }
    }
    setTimeout(check, 10_000)
    this.interval = setInterval(check, updateIntervalMs)
    this.interval.unref()
  }

  checkManually(): void {
    if (!this.packagedRuntime) {
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
    if (
      this.lastDownloadProgress &&
      ['available', 'downloading', 'paused', 'ready'].includes(
        this.lastDownloadProgress.phase,
      )
    ) {
      this.publishProgress(this.lastDownloadProgress)
      return
    }
    this.manualCheck = true
    void autoUpdater.checkForUpdates()
  }

  status(): DesktopUpdateProgress | null {
    return this.lastDownloadProgress ?? null
  }

  async download(): Promise<DesktopUpdateProgress | null> {
    if (this.lastDownloadProgress?.phase !== 'available') {
      return this.status()
    }
    void this.startDownload(this.lastDownloadProgress.version)
    return this.status()
  }

  async pause(): Promise<DesktopUpdateProgress | null> {
    if (
      this.lastDownloadProgress?.phase !== 'downloading' ||
      !this.downloadCancellation
    ) {
      return this.status()
    }
    this.cancellationIntent = 'pause'
    this.downloadCancellation.cancel()
    await this.downloadTask
    return this.status()
  }

  async resume(): Promise<DesktopUpdateProgress | null> {
    if (
      this.lastDownloadProgress?.phase !== 'paused' &&
      this.lastDownloadProgress?.phase !== 'failed'
    ) {
      return this.status()
    }
    void this.startDownload(this.lastDownloadProgress.version)
    return this.status()
  }

  async cancel(): Promise<DesktopUpdateProgress | null> {
    if (this.downloadTask && this.downloadCancellation) {
      this.cancellationIntent = 'cancel'
      this.downloadCancellation.cancel()
      await this.downloadTask
    } else {
      await this.downloader.clearPartials()
      this.publishDismissed()
    }
    return this.status()
  }

  dismiss(): DesktopUpdateProgress | null {
    this.publishDismissed()
    return this.status()
  }

  install(): void {
    if (this.lastDownloadProgress?.phase === 'ready') {
      autoUpdater.quitAndInstall()
    }
  }

  stop(): void {
    if (this.interval) clearInterval(this.interval)
    this.interval = undefined
    this.windowProvider()?.setProgressBar(-1)
  }

  private async startDownload(version: string): Promise<void> {
    if (this.downloadTask) {
      await this.downloadTask
      return
    }
    const previous =
      this.lastDownloadProgress?.version === version
        ? this.lastDownloadProgress
        : undefined
    this.downloadInProgress = true
    this.downloadVersion = version
    this.cancellationIntent = undefined
    const cancellation = new CancellationToken()
    this.downloadCancellation = cancellation
    this.publishProgress({
      phase: 'downloading',
      version,
      percent: previous?.percent ?? 0,
      transferredBytes: previous?.transferredBytes ?? 0,
      totalBytes: previous?.totalBytes ?? 0,
      bytesPerSecond: 0,
    })

    const task = autoUpdater
      .downloadUpdate(cancellation)
      .then(() => undefined)
      .catch(async (error: unknown) => {
        const intent = this.cancellationIntent
        const previousProgress = this.lastDownloadProgress
        if (
          error instanceof CancellationError ||
          cancellation.cancelled
        ) {
          if (intent === 'cancel') {
            await this.downloader.clearPartials()
            this.publishDismissed()
          } else {
            this.publishProgress({
              phase: 'paused',
              version,
              percent: previousProgress?.percent ?? 0,
              transferredBytes:
                previousProgress?.transferredBytes ?? 0,
              totalBytes: previousProgress?.totalBytes ?? 0,
              bytesPerSecond: 0,
            })
          }
          return
        }
        const resolvedError =
          error instanceof Error ? error : new Error('更新下载失败')
        this.publishProgress({
          phase: 'failed',
          version,
          percent: previousProgress?.percent ?? 0,
          transferredBytes: previousProgress?.transferredBytes ?? 0,
          totalBytes: previousProgress?.totalBytes ?? 0,
          bytesPerSecond: 0,
          error: resolvedError.message,
        })
      })
      .finally(() => {
        this.downloadInProgress = false
        this.downloadCancellation = undefined
        this.cancellationIntent = undefined
        if (this.downloadTask === task) this.downloadTask = undefined
      })
    this.downloadTask = task
    await task
  }

  private publishDismissed(): void {
    const previous = this.lastDownloadProgress
    this.publishProgress({
      phase: 'dismissed',
      version: previous?.version ?? this.downloadVersion,
      percent: previous?.percent ?? 0,
      transferredBytes: previous?.transferredBytes ?? 0,
      totalBytes: previous?.totalBytes ?? 0,
      bytesPerSecond: 0,
    })
  }

  private publishProgress(progress: DesktopUpdateProgress): void {
    this.lastDownloadProgress = progress
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
