import {
  Check,
  CircleAlert,
  Database,
  LockKeyhole,
  Workflow,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import { fetchReviewLayer } from './api'
import type { ReviewLayerStatus } from './types'
import './ReviewLayerWorkspace.css'

export default function ReviewLayerWorkspace({
  layerId,
}: {
  layerId: ReviewLayerStatus['id']
}) {
  const [layer, setLayer] = useState<ReviewLayerStatus | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    fetchReviewLayer(layerId)
      .then((value) => {
        if (active) setLayer(value)
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : '审核层加载失败')
        }
      })
    return () => {
      active = false
    }
  }, [layerId])

  if (error) {
    return (
      <div className="review-layer-loading is-error">
        <CircleAlert />
        {error}
      </div>
    )
  }
  if (!layer) {
    return <div className="review-layer-loading">正在加载审核层</div>
  }

  return (
    <div className="review-layer-page">
      <header className="admin-page-heading">
        <div>
          <span>REVIEW LAYER {String(layer.order).padStart(2, '0')}</span>
          <h1>{layer.label}</h1>
        </div>
        <strong className={`review-layer-state is-${layer.status}`}>
          {layer.status === 'active'
            ? '运行中'
            : layer.status === 'scaffolded'
              ? '框架就绪'
              : '依赖锁定'}
        </strong>
      </header>

      <section className="review-layer-dependency">
        {layer.status === 'blocked' ? (
          <LockKeyhole aria-hidden="true" />
        ) : (
          <Workflow aria-hidden="true" />
        )}
        <div>
          <span>当前阻塞项</span>
          <strong>{layer.blocker ?? '无'}</strong>
        </div>
        <code>{layer.reviewKind}</code>
      </section>

      <div className="review-layer-grid">
        <section>
          <header>
            <Database aria-hidden="true" />
            <strong>Schema v{layer.schemaVersion}</strong>
          </header>
          <div className="review-layer-fields">
            {layer.fields.map((field) => (
              <article key={field.name}>
                <code>{field.name}</code>
                <span>{field.type}</span>
                <i>{field.required ? '必填' : '可选'}</i>
              </article>
            ))}
          </div>
        </section>

        <aside>
          <header>
            <Check aria-hidden="true" />
            <strong>提交检查</strong>
          </header>
          <div>
            {layer.checks.map((check) => (
              <article key={check}>
                <Check aria-hidden="true" />
                <code>{check}</code>
              </article>
            ))}
          </div>
        </aside>
      </div>

      <section className="review-layer-progress">
        <header>
          <span>已审核 / 总数</span>
          <strong>
            {layer.reviewed} / {layer.total}
          </strong>
        </header>
        <div>
          <span
            style={{
              width: `${
                layer.total > 0 ? (layer.reviewed / layer.total) * 100 : 0
              }%`,
            }}
          />
        </div>
      </section>
    </div>
  )
}
