import {
  Archive,
  Check,
  CircleAlert,
  DatabaseBackup,
  LoaderCircle,
  RefreshCcw,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import {
  createBeatBackup,
  fetchBeatBackups,
  restoreBeatBackup,
  verifyBeatBackup,
} from './api'
import type { BeatBackup } from './types'
import './SystemWorkspace.css'

function fileSize(value: number): string {
  return `${(value / 1024 / 1024).toFixed(1)} MB`
}

export default function SystemWorkspace() {
  const [backups, setBackups] = useState<BeatBackup[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const refresh = async () => setBackups(await fetchBeatBackups())

  useEffect(() => {
    let active = true
    fetchBeatBackups()
      .then((value) => {
        if (active) setBackups(value)
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : '备份列表加载失败')
        }
      })
    return () => {
      active = false
    }
  }, [])

  const run = async (
    operation: () => Promise<BeatBackup>,
    success: string,
  ) => {
    if (busy) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      await operation()
      setNotice(success)
      await refresh()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '备份操作失败')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="system-page">
      <header className="admin-page-heading">
        <div>
          <span>LOCAL OPERATIONS</span>
          <h1>系统与备份</h1>
        </div>
      </header>

      <section className="system-actions">
        <div>
          <DatabaseBackup aria-hidden="true" />
          <div>
            <strong>真实 Pilot 完整备份</strong>
            <span>包含 manifest、分析、双人草稿、金标准和评测版本</span>
          </div>
        </div>
        <button
          type="button"
          disabled={busy}
          onClick={() =>
            void run(createBeatBackup, '备份已创建并完成哈希校验')}
        >
          {busy ? <LoaderCircle className="spin" /> : <Archive />}
          创建备份
        </button>
      </section>

      {error ? (
        <p className="system-message is-error" role="alert">
          <CircleAlert />
          {error}
        </p>
      ) : null}
      {notice ? (
        <p className="system-message is-success">
          <Check />
          {notice}
        </p>
      ) : null}

      <section className="system-backups">
        <header>
          <span>备份与恢复演练</span>
          <small>{backups.length} 份</small>
        </header>
        <div>
          {backups.map((backup) => (
            <article key={backup.id}>
              <div>
                <strong>{backup.id}</strong>
                <span>
                  {new Date(backup.createdAt).toLocaleString('zh-CN', {
                    hour12: false,
                  })}
                </span>
              </div>
              <code>{fileSize(backup.archive.size)}</code>
              <code>{backup.archive.sha256.slice(0, 12)}</code>
              <i className={backup.restored ? 'is-restored' : ''}>
                {backup.restored ? '已恢复演练' : '已校验'}
              </i>
              <button
                type="button"
                disabled={busy}
                title="重新校验"
                aria-label={`校验 ${backup.id}`}
                onClick={() =>
                  void run(
                    () => verifyBeatBackup(backup.id),
                    '备份哈希校验通过',
                  )}
              >
                <Check />
              </button>
              <button
                type="button"
                disabled={busy || backup.restored}
                title="恢复演练"
                aria-label={`恢复 ${backup.id}`}
                onClick={() => {
                  if (
                    window.confirm(
                      '恢复演练会解压到隔离目录，不会覆盖当前复核数据。确认继续吗？',
                    )
                  ) {
                    void run(
                      () => restoreBeatBackup(backup.id),
                      '隔离恢复演练完成',
                    )
                  }
                }}
              >
                <RefreshCcw />
              </button>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}
