import {
  Check,
  CircleAlert,
  Database,
  FlaskConical,
  LoaderCircle,
  Play,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import {
  createBeatTrainingDataset,
  createBeatTrainingJob,
  fetchBeatTrainingStatus,
} from './api'
import type { BeatTrainingStatus } from './types'
import './ModelsWorkspace.css'

export default function ModelsWorkspace() {
  const [status, setStatus] = useState<BeatTrainingStatus | null>(null)
  const [datasetVersion, setDatasetVersion] = useState('beat-train-v1')
  const [selectedDataset, setSelectedDataset] = useState('')
  const [codeRevision, setCodeRevision] = useState('working-tree')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const refresh = async () => {
    const value = await fetchBeatTrainingStatus()
    setStatus(value)
    if (!selectedDataset && value.datasets[0]) {
      setSelectedDataset(value.datasets[0].version)
    }
  }

  useEffect(() => {
    let active = true
    fetchBeatTrainingStatus()
      .then((value) => {
        if (!active) return
        setStatus(value)
        setSelectedDataset(value.datasets[0]?.version ?? '')
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : '训练状态加载失败')
        }
      })
    return () => {
      active = false
    }
  }, [])

  const createDataset = async () => {
    if (busy) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const dataset = await createBeatTrainingDataset(datasetVersion)
      setSelectedDataset(dataset.version)
      setNotice('训练数据版本已生成')
      await refresh()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '训练数据生成失败')
    } finally {
      setBusy(false)
    }
  }

  const createJob = async () => {
    if (busy || !selectedDataset) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const job = await createBeatTrainingJob({
        datasetVersion: selectedDataset,
        codeRevision,
      })
      setNotice(
        job.status === 'queued' ? '训练任务已入队' : `任务已阻塞：${job.reason}`,
      )
      await refresh()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '训练任务创建失败')
    } finally {
      setBusy(false)
    }
  }

  if (!status) {
    return (
      <div className="models-loading">
        <LoaderCircle className="spin" />
        正在加载训练状态
      </div>
    )
  }

  const progress = Math.min(
    100,
    (status.adjudicatedTrainClips / status.minimumTrainClips) * 100,
  )

  return (
    <div className="models-page">
      <header className="admin-page-heading">
        <div>
          <span>TRAINING CONTROL</span>
          <h1>模型与实验</h1>
        </div>
      </header>

      <section className="models-readiness">
        <header>
          <div>
            <span>可训练金标准</span>
            <strong>
              {status.adjudicatedTrainClips} / {status.minimumTrainClips}
            </strong>
          </div>
          <i>{progress >= 100 ? '已就绪' : '样本不足'}</i>
        </header>
        <div>
          <span style={{ width: `${progress}%` }} />
        </div>
      </section>

      <div className="models-grid">
        <section className="models-datasets">
          <header>
            <Database aria-hidden="true" />
            <strong>训练数据版本</strong>
          </header>
          <div>
            {status.datasets.map((dataset) => (
              <article key={dataset.version}>
                <div>
                  <strong>{dataset.version}</strong>
                  <span>
                    {dataset.clips} 片段 · {dataset.recordings} 录音
                  </span>
                </div>
                <code>{dataset.manifestSha256.slice(0, 12)}</code>
                <i className={dataset.readyForTraining ? 'is-ready' : ''}>
                  {dataset.readyForTraining ? '可训练' : '仅供研究'}
                </i>
              </article>
            ))}
          </div>
          <footer>
            <input
              value={datasetVersion}
              onChange={(event) => setDatasetVersion(event.target.value)}
            />
            <button type="button" disabled={busy} onClick={() => void createDataset()}>
              <Database />
              生成 train 数据
            </button>
          </footer>
        </section>

        <aside className="models-job-form">
          <header>
            <FlaskConical aria-hidden="true" />
            <strong>创建训练任务</strong>
          </header>
          <label>
            <span>数据版本</span>
            <select
              value={selectedDataset}
              onChange={(event) => setSelectedDataset(event.target.value)}
            >
              <option value="">请选择</option>
              {status.datasets.map((dataset) => (
                <option key={dataset.version} value={dataset.version}>
                  {dataset.version}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>代码版本</span>
            <input
              value={codeRevision}
              onChange={(event) => setCodeRevision(event.target.value)}
            />
          </label>
          {error ? (
            <p className="models-message is-error" role="alert">
              <CircleAlert />
              {error}
            </p>
          ) : null}
          {notice ? (
            <p className="models-message is-success">
              <Check />
              {notice}
            </p>
          ) : null}
          <button
            type="button"
            disabled={busy || !selectedDataset}
            onClick={() => void createJob()}
          >
            {busy ? <LoaderCircle className="spin" /> : <Play />}
            创建任务
          </button>
        </aside>
      </div>

      <section className="models-jobs">
        <header>
          <span>训练任务</span>
          <small>{status.jobs.length} 个</small>
        </header>
        <div>
          {status.jobs.map((job) => (
            <article key={job.id}>
              <strong>{job.recipe}</strong>
              <span>{job.datasetVersion}</span>
              <code>{job.codeRevision}</code>
              <i className={job.status === 'queued' ? 'is-ready' : ''}>
                {job.status === 'queued' ? '已排队' : job.reason}
              </i>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}
