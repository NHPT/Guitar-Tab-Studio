export class CoreValidationError extends Error {
  constructor(
    readonly path: string,
    message: string,
  ) {
    super(`${path}: ${message}`)
    this.name = 'CoreValidationError'
  }
}

export function recordAt(
  value: unknown,
  path: string,
): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new CoreValidationError(path, 'expected object')
  }
  return value as Record<string, unknown>
}

export function arrayAt(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new CoreValidationError(path, 'expected array')
  }
  return value
}

export function stringAt(value: unknown, path: string): string {
  if (typeof value !== 'string') {
    throw new CoreValidationError(path, 'expected string')
  }
  return value
}

export function optionalStringAt(
  value: unknown,
  path: string,
): string | undefined {
  return value === undefined ? undefined : stringAt(value, path)
}

export function numberAt(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new CoreValidationError(path, 'expected finite number')
  }
  return value
}

export function optionalNumberAt(
  value: unknown,
  path: string,
): number | undefined {
  return value === undefined ? undefined : numberAt(value, path)
}

export function booleanAt(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') {
    throw new CoreValidationError(path, 'expected boolean')
  }
  return value
}

export function enumAt<const T extends string>(
  value: unknown,
  allowed: readonly T[],
  path: string,
): T {
  if (typeof value !== 'string' || !allowed.includes(value as T)) {
    throw new CoreValidationError(path, `expected one of ${allowed.join(', ')}`)
  }
  return value as T
}

export function stringsAt(value: unknown, path: string): string[] {
  return arrayAt(value, path).map((item, index) =>
    stringAt(item, `${path}[${index}]`),
  )
}
