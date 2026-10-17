import { createHash, randomUUID } from 'node:crypto'
import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { resolve } from 'node:path'
import {
  uploadContentValidationError,
  type MediaObject,
  type MediaRepository,
  type MediaWrite,
} from '@gts/core'

const MEDIA_ID = /^[0-9a-f-]{36}$/u
const MAX_MEDIA_BYTES = 500 * 1024 * 1024

export class DesktopMediaRepository implements MediaRepository {
  constructor(private readonly root: string) {
    mkdirSync(root, { recursive: true })
  }

  getMedia(id: string): MediaObject | undefined {
    if (!MEDIA_ID.test(id)) return undefined
    const metadataPath = this.metadataPath(id)
    const contentPath = this.contentPath(id)
    if (!existsSync(metadataPath) || !existsSync(contentPath)) return undefined
    const value = JSON.parse(readFileSync(metadataPath, 'utf8')) as MediaObject
    if (
      value.id !== id ||
      typeof value.displayName !== 'string' ||
      typeof value.mediaType !== 'string' ||
      typeof value.sizeBytes !== 'number' ||
      typeof value.sha256 !== 'string'
    ) {
      return undefined
    }
    return value
  }

  putMedia(input: MediaWrite): MediaObject {
    if (
      input.bytes.byteLength === 0 ||
      input.bytes.byteLength > MAX_MEDIA_BYTES
    ) {
      throw new Error('Desktop media size is invalid')
    }
    const validationError = uploadContentValidationError(
      input.displayName,
      input.mediaType,
      input.bytes.slice(0, 32),
    )
    if (validationError) throw new Error(validationError)

    const id = randomUUID()
    const sha256 = createHash('sha256').update(input.bytes).digest('hex')
    if (input.sha256 && input.sha256 !== sha256) {
      throw new Error('Desktop media SHA-256 mismatch')
    }
    const media: MediaObject = {
      id,
      displayName: input.displayName.slice(0, 255),
      mediaType: input.mediaType.slice(0, 128),
      sizeBytes: input.bytes.byteLength,
      sha256,
    }
    const temporaryContent = `${this.contentPath(id)}.tmp`
    const temporaryMetadata = `${this.metadataPath(id)}.tmp`
    try {
      writeFileSync(temporaryContent, input.bytes, { mode: 0o600 })
      writeFileSync(
        temporaryMetadata,
        `${JSON.stringify(media)}\n`,
        { encoding: 'utf8', mode: 0o600 },
      )
      renameSync(temporaryContent, this.contentPath(id))
      renameSync(temporaryMetadata, this.metadataPath(id))
    } catch (error) {
      rmSync(temporaryContent, { force: true })
      rmSync(temporaryMetadata, { force: true })
      throw error
    }
    return media
  }

  removeMedia(id: string): void {
    if (!MEDIA_ID.test(id)) return
    rmSync(this.contentPath(id), { force: true })
    rmSync(this.metadataPath(id), { force: true })
  }

  contentPathFor(id: string): string | undefined {
    return this.getMedia(id) ? this.contentPath(id) : undefined
  }

  private contentPath(id: string): string {
    return resolve(this.root, `${id}.media`)
  }

  private metadataPath(id: string): string {
    return resolve(this.root, `${id}.json`)
  }
}
