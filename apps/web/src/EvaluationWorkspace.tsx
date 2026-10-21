import {
  Check,
  CircleAlert,
  LoaderCircle,
  Save,
  ShieldCheck,
  X,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import {
  fetchBeatEvaluation,
  saveBeatEvaluation,
} from './api'
import type { BeatEvaluationReport } from './types'
import './EvaluationWorkspace.css'

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function milliseconds(value: number | null): string {
  return value === null ? '—' : `${value.toFixed(1)}ms`
}

export default function EvaluationWorkspace() {
  const [current, setCurrent] = useState<BeatEvaluationReport | null>(null)
  const [runs, setRuns] = useState<BeatEvaluationReport[]>([])
  const [modelVersion, setModelVersion] = useState('beat-candidate-v1')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const refresh = async () => {
    const response = await fetchBeatEvaluation()
    setCurrent(response.current)
    setRuns(response.runs)
  }

  useEffect(() => {
    let active = true
    fetchBeatEvaluation()
      .then((response) => {
        if (!active) return
        setCurrent(response.current)
        setRuns(response.runs)
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : '评测数据加载失败')
        }
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  const freeze = async () => {
    if (saving) return
    setSaving(true)
    setError('')
    setNotice('')
    try {
      await saveBeatEvaluation(modelVersion)
      setNotice('评测报告已冻结')
      await refresh()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '评测报告保存失败')
    } finally {
      setSaving(false)
    }
  }

  if (loading || !current) {
    return (
      <div className="evaluation-loading">
        <LoaderCircle className="spin" />
        正在计算节拍门禁
      </div>
    )
  }

  const checks = [
    ['coverage', '金标准覆盖完整'],
    ['beatF1', 'Beat F1'],
    ['medianError', '中位误差'],
    ['p95Error', 'P95 误差'],
    ['downbeatF1', 'Downbeat F1'],
    ['meter', '拍号准确率'],
    ['pickup', '弱起准确率'],
  ] as const

  return (
    <div className="evaluation-page">
      <header className="admin-page-heading">
        <div>
          <span>MODEL GATE</span>
          <h1>评测与门禁</h1>
        </div>
        <strong
          className={`evaluation-gate-status${
            current.gate.passed ? ' is-passed' : ''
          }`}
        >
          {current.gate.passed ? '已通过' : '未通过'}
        </strong>
      </header>

      <section className="evaluation-coverage">
        <div>
          <span>裁决覆盖</span>
          <strong>
            {current.coverage.adjudicated} / {current.coverage.total}
          </strong>
        </div>
        <dl>
          <div>
            <dt>train</dt>
            <dd>{current.coverage.splitCounts.train ?? 0}</dd>
          </div>
          <div>
            <dt>validation</dt>
            <dd>{current.coverage.splitCounts.validation ?? 0}</dd>
          </div>
          <div>
            <dt>test</dt>
            <dd>{current.coverage.splitCounts.test ?? 0}</dd>
          </div>
        </dl>
      </section>

      <section className="evaluation-metrics">
        <article>
          <span>Beat F1</span>
          <strong>{current.aggregate.beatMetrics.f1.toFixed(3)}</strong>
          <small>门槛 ≥ {current.gate.thresholds.beatF1.toFixed(2)}</small>
        </article>
        <article>
          <span>Downbeat F1</span>
          <strong>{current.aggregate.downbeatMetrics.f1.toFixed(3)}</strong>
          <small>
            门槛 ≥ {current.gate.thresholds.downbeatF1.toFixed(2)}
          </small>
        </article>
        <article>
          <span>中位误差</span>
          <strong>
            {milliseconds(current.aggregate.beatMetrics.median_error_ms)}
          </strong>
          <small>门槛 ≤ {current.gate.thresholds.medianErrorMs}ms</small>
        </article>
        <article>
          <span>P95 误差</span>
          <strong>
            {milliseconds(current.aggregate.beatMetrics.p95_error_ms)}
          </strong>
          <small>门槛 ≤ {current.gate.thresholds.p95ErrorMs}ms</small>
        </article>
      </section>

      <div className="evaluation-grid">
        <section className="evaluation-checks">
          <header>
            <ShieldCheck aria-hidden="true" />
            <strong>门禁检查</strong>
          </header>
          <div>
            {checks.map(([key, label]) => (
              <article key={key}>
                {current.gate.checks[key] ? (
                  <Check className="is-passed" aria-hidden="true" />
                ) : (
                  <X aria-hidden="true" />
                )}
                <span>{label}</span>
                <strong>
                  {current.gate.checks[key] ? '通过' : '未通过'}
                </strong>
              </article>
            ))}
          </div>
          <dl>
            <div>
              <dt>拍号准确率</dt>
              <dd>{percent(current.aggregate.meterAccuracy)}</dd>
            </div>
            <div>
              <dt>弱起准确率</dt>
              <dd>{percent(current.aggregate.pickupAccuracy)}</dd>
            </div>
          </dl>
        </section>

        <aside className="evaluation-freeze">
          <header>
            <Save aria-hidden="true" />
            <strong>冻结评测报告</strong>
          </header>
          <label>
            <span>模型版本</span>
            <input
              value={modelVersion}
              onChange={(event) => setModelVersion(event.target.value)}
            />
          </label>
          {error ? (
            <p className="evaluation-message is-error" role="alert">
              <CircleAlert />
              {error}
            </p>
          ) : null}
          {notice ? (
            <p className="evaluation-message is-success">
              <Check />
              {notice}
            </p>
          ) : null}
          <button type="button" disabled={saving} onClick={() => void freeze()}>
            {saving ? <LoaderCircle className="spin" /> : <Save />}
            冻结当前报告
          </button>
        </aside>
      </div>

      <section className="evaluation-runs">
        <header>
          <span>历史版本</span>
          <small>{runs.length} 份</small>
        </header>
        <div>
          {runs.map((run) => (
            <article key={run.reportSha256}>
              <strong>{run.modelVersion}</strong>
              <span>{run.coverage.adjudicated} 份金标准</span>
              <code>{run.aggregate.beatMetrics.f1.toFixed(3)}</code>
              <i className={run.gate.passed ? 'is-passed' : ''}>
                {run.gate.passed ? '通过' : '未通过'}
              </i>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}
