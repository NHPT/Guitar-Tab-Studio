import {
  beatGoldStatus,
  defaultBeatGoldRoot,
  listBeatGold,
} from './beat-reviews.js'

export type ReviewLayerId =
  | 'beat'
  | 'onset'
  | 'pitch'
  | 'string-fret'
  | 'chord'
  | 'technique'

export interface ReviewLayerDefinition {
  id: ReviewLayerId
  label: string
  order: number
  schemaVersion: 1
  reviewKind: string
  dependsOn?: ReviewLayerId
  fields: Array<{
    name: string
    type: string
    required: boolean
  }>
  checks: string[]
}

export interface ReviewLayerStatus extends ReviewLayerDefinition {
  status: 'active' | 'scaffolded' | 'blocked'
  blocker?: string
  reviewed: number
  total: number
}

const DEFINITIONS: ReviewLayerDefinition[] = [
  {
    id: 'beat',
    label: '节拍',
    order: 1,
    schemaVersion: 1,
    reviewKind: 'beat-grid',
    fields: [
      { name: 'at', type: 'seconds', required: true },
      { name: 'beat', type: 'integer', required: true },
      { name: 'measure', type: 'integer', required: true },
      { name: 'downbeat', type: 'boolean', required: true },
    ],
    checks: ['beat_times', 'downbeats', 'meter', 'completeness'],
  },
  {
    id: 'onset',
    label: '起音 / 拨弦',
    order: 2,
    schemaVersion: 1,
    reviewKind: 'onset-picking',
    dependsOn: 'beat',
    fields: [
      { name: 'at', type: 'seconds', required: true },
      { name: 'attack', type: 'pick|pluck|strum|unknown', required: true },
      { name: 'confidence', type: 'number', required: false },
    ],
    checks: ['timing', 'attack_type', 'completeness'],
  },
  {
    id: 'pitch',
    label: '音高',
    order: 3,
    schemaVersion: 1,
    reviewKind: 'note-pitch',
    dependsOn: 'onset',
    fields: [
      { name: 'event_id', type: 'id', required: true },
      { name: 'midi', type: 'integer', required: true },
      { name: 'offset', type: 'seconds', required: true },
    ],
    checks: ['pitch', 'duration', 'completeness'],
  },
  {
    id: 'string-fret',
    label: '弦品',
    order: 4,
    schemaVersion: 1,
    reviewKind: 'string-fret',
    dependsOn: 'pitch',
    fields: [
      { name: 'event_id', type: 'id', required: true },
      { name: 'string', type: 'integer:1-6', required: true },
      { name: 'fret', type: 'integer:0-24', required: true },
    ],
    checks: ['playability', 'string_fret', 'completeness'],
  },
  {
    id: 'chord',
    label: '和弦',
    order: 5,
    schemaVersion: 1,
    reviewKind: 'chord-region',
    dependsOn: 'string-fret',
    fields: [
      { name: 'start', type: 'seconds', required: true },
      { name: 'end', type: 'seconds', required: true },
      { name: 'label', type: 'chord-symbol', required: true },
    ],
    checks: ['boundary', 'label', 'completeness'],
  },
  {
    id: 'technique',
    label: '技法',
    order: 6,
    schemaVersion: 1,
    reviewKind: 'guitar-technique',
    dependsOn: 'chord',
    fields: [
      { name: 'event_id', type: 'id', required: true },
      { name: 'technique', type: 'technique-label', required: true },
      { name: 'target_event_id', type: 'id', required: false },
    ],
    checks: ['technique', 'relation', 'completeness'],
  },
]

export function reviewLayerDefinitions(): ReviewLayerDefinition[] {
  return DEFINITIONS.map((definition) => ({
    ...definition,
    fields: definition.fields.map((field) => ({ ...field })),
    checks: [...definition.checks],
  }))
}

export function reviewLayerStatuses(
  root = defaultBeatGoldRoot(),
): ReviewLayerStatus[] {
  const beatStatus = beatGoldStatus(root)
  const goldCount = beatStatus.enabled ? listBeatGold(root).length : 0
  const total = beatStatus.manifest?.clips ?? 0
  const beatComplete = total > 0 && goldCount === total
  return DEFINITIONS.map((definition) => {
    if (definition.id === 'beat') {
      return {
        ...definition,
        status: beatStatus.enabled ? 'active' : 'blocked',
        ...(beatStatus.enabled ? {} : { blocker: 'Pilot v2 尚未准备' }),
        reviewed: goldCount,
        total,
      }
    }
    const dependency = DEFINITIONS.find(
      (candidate) => candidate.id === definition.dependsOn,
    )
    const dependencyReady =
      definition.dependsOn === 'beat' ? beatComplete : false
    return {
      ...definition,
      status: dependencyReady ? 'scaffolded' : 'blocked',
      blocker: dependencyReady
        ? 'schema 已就绪，候选数据尚未生成'
        : `等待${dependency?.label ?? '上游'}层完成`,
      reviewed: 0,
      total,
    }
  })
}

export function reviewLayerStatus(
  layerId: string,
  root = defaultBeatGoldRoot(),
): ReviewLayerStatus | null {
  return (
    reviewLayerStatuses(root).find((layer) => layer.id === layerId) ?? null
  )
}
