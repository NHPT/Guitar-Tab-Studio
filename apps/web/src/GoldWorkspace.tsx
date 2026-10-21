import {
  Archive,
  Check,
  CircleAlert,
  Download,
  LoaderCircle,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import {
  beatGoldReleaseUrl,
  createBeatGoldRelease,
  fetchAdminDashboard,
  fetchBeatGoldReleases,
} from './api'
import type {
  AdminDashboard,
  BeatGoldRelease,
} from './types'
import './GoldWorkspace.css'

function fileSize(value: number): string {
  if (value >= 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`
  return `${(value / 1024).toFixed(1)} KB`
}

export default function GoldWorkspace() {
  const [dashboard, setDashboard] = useState<AdminDashboard | null>(null)
  const [releases, setReleases] = useState<BeatGoldRelease[]>([])
  const [version, setVersion] = useState('beat-gold-v1')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const refresh = async () => {
    const [nextDashboard, nextReleases] = await Promise.all([
      fetchAdminDashboard(),
      fetchBeatGoldReleases(),
    ])
    setDashboard(nextDashboard)
    setReleases(nextReleases)
  }

  useEffect(() => {
    let active = true
    Promise.all([fetchAdminDashboard(), fetchBeatGoldReleases()])
      .then(([nextDashboard, nextReleases]) => {
        if (!active) return
        setDashboard(nextDashboard)
        setReleases(nextReleases)
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : '金标准版本加载失败')
        }
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  const create = async () => {
    if (saving) return
    setSaving(true)
    setError('')
    setNotice('')
    try {
      await createBeatGoldRelease(version)
      setNotice('金标准版本已冻结')
      await refresh()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '版本冻结失败')
    } finally {
      setSaving(false)
    }
  }

  if (loading || !dashboard) {
    return (
      <div className="gold-loading">
        <LoaderCircle className="spin" />
        正在加载金标准版本
      </div>
    )
  }

  const complete =
    dashboard.reviews.adjudicated === dashboard.reviews.total &&
    dashboard.reviews.total > 0

  return (
    <div className="gold-page">
      <header className="admin-page-heading">
        <div>
          <span>IMMUTABLE DATASET</span>
          <h1>金标准版本</h1>
        </div>
      </header>

      <section className="gold-status">
        <div>
          <span>裁决覆盖</span>
          <strong>
            {dashboard.reviews.adjudicated} / {dashboard.reviews.total}
          </strong>
          <small>{complete ? '允许冻结版本' : '等待全部片段裁决'}</small>
        </div>
        <dl>
          <div>
            <dt>语料版本</dt>
            <dd>{dashboard.corpus?.corpusVersion ?? '—'}</dd>
          </div>
          <div>
            <dt>语料摘要</dt>
            <dd>{dashboard.corpus?.corpusSha256.slice(0, 16) ?? '—'}</dd>
          </div>
        </dl>
      </section>

      <div className="gold-create">
        <label>
          <span>新版本标识</span>
          <input
            value={version}
            onChange={(event) => setVersion(event.target.value)}
          />
        </label>
        <button type="button" disabled={!complete || saving} onClick={() => void create()}>
          {saving ? <LoaderCircle className="spin" /> : <Archive />}
          冻结并打包
        </button>
        {error ? (
          <p className="gold-message is-error" role="alert">
            <CircleAlert />
            {error}
          </p>
        ) : null}
        {notice ? (
          <p className="gold-message is-success">
            <Check />
            {notice}
          </p>
        ) : null}
      </div>

      <section className="gold-releases">
        <header>
          <span>不可变版本</span>
          <small>{releases.length} 个</small>
        </header>
        <div>
          {releases.map((release) => (
            <article key={release.version}>
              <div>
                <strong>{release.version}</strong>
                <span>{new Date(release.createdAt).toLocaleString('zh-CN', { hour12: false })}</span>
              </div>
              <code>{release.clips} 片段</code>
              <code>{fileSize(release.archive.size)}</code>
              <code>{release.archive.sha256.slice(0, 12)}</code>
              <a href={beatGoldReleaseUrl(release.version)}>
                <Download aria-hidden="true" />
                下载
              </a>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}
