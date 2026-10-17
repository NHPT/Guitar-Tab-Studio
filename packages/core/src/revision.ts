import type { ProjectRevision, ProjectRevisionKind } from './domain.js'
import { migrateStudioProject } from './project.js'
import {
  CoreValidationError,
  enumAt,
  optionalStringAt,
  recordAt,
  stringAt,
} from './validation.js'

const revisionKinds = [
  'inference',
  'manual',
  'arrangement',
  'restore',
  'import',
] as const satisfies readonly ProjectRevisionKind[]

export function parseProjectRevision(value: unknown): ProjectRevision {
  const revision = recordAt(value, 'revision')
  if (revision.schemaVersion !== 1) {
    throw new CoreValidationError(
      'revision.schemaVersion',
      'unsupported version',
    )
  }
  const project = migrateStudioProject(revision.project)
  const projectId = stringAt(revision.projectId, 'revision.projectId')
  if (project.id !== projectId) {
    throw new CoreValidationError(
      'revision.projectId',
      'must match revision.project.id',
    )
  }
  return {
    schemaVersion: 1,
    id: stringAt(revision.id, 'revision.id'),
    projectId,
    parentRevisionId: optionalStringAt(
      revision.parentRevisionId,
      'revision.parentRevisionId',
    ),
    createdAt: stringAt(revision.createdAt, 'revision.createdAt'),
    label: stringAt(revision.label, 'revision.label'),
    kind: enumAt(revision.kind, revisionKinds, 'revision.kind'),
    project,
  }
}
