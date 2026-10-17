const supportedExtensions = new Set([
  '.mp3',
  '.wav',
  '.flac',
  '.ape',
  '.m4a',
  '.aac',
  '.ogg',
  '.mp4',
  '.mov',
  '.webm',
])

const encryptedExtensions = new Set([
  '.qmc0',
  '.qmc2',
  '.qmc3',
  '.qmcflac',
  '.qmcogg',
  '.tkm',
  '.tm0',
  '.tm2',
  '.tm3',
  '.tm6',
  '.bkcmp3',
  '.bkcflac',
  '.mflac',
  '.mflac0',
  '.mgg',
  '.mgg0',
  '.mgg1',
  '.ncm',
  '.kwm',
  '.kgg',
  '.kgm',
  '.kgma',
  '.vpr',
])

const encryptedFileError =
  '该文件使用专有加密容器，需要确认版权与处理声明后临时解码'

export type AuthorizedMediaProvider =
  | 'qqmusic'
  | 'netease'
  | 'moo'
  | 'kuwo'
  | 'kugou'
  | 'unknown'

const providerByExtension: Record<string, AuthorizedMediaProvider> = {
  '.qmc0': 'qqmusic',
  '.qmc2': 'qqmusic',
  '.qmc3': 'qqmusic',
  '.qmcflac': 'qqmusic',
  '.qmcogg': 'qqmusic',
  '.tkm': 'qqmusic',
  '.tm0': 'qqmusic',
  '.tm2': 'qqmusic',
  '.tm3': 'qqmusic',
  '.tm6': 'qqmusic',
  '.mflac': 'qqmusic',
  '.mflac0': 'qqmusic',
  '.mgg': 'qqmusic',
  '.mgg0': 'qqmusic',
  '.mgg1': 'qqmusic',
  '.bkcmp3': 'moo',
  '.bkcflac': 'moo',
  '.ncm': 'netease',
  '.kwm': 'kuwo',
  '.kgg': 'kugou',
  '.kgm': 'kugou',
  '.kgma': 'kugou',
  '.vpr': 'kugou',
}

const supportedScoreExtensions = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
])

function fileExtension(filename: string): string {
  const basename = filename.split(/[\\/]/u).at(-1) ?? ''
  const extensionStart = basename.lastIndexOf('.')
  return extensionStart <= 0 ? '' : basename.slice(extensionStart).toLowerCase()
}

export function requiresContentInspection(filename: string): boolean {
  return encryptedExtensions.has(fileExtension(filename))
}

export function authorizedMediaProvider(
  filename: string,
): AuthorizedMediaProvider | null {
  return providerByExtension[fileExtension(filename)] ?? null
}

export function authorizedMediaProviderForContent(
  filename: string,
  header: Uint8Array,
): AuthorizedMediaProvider | null {
  const text = (start: number, end: number) =>
    String.fromCharCode(...header.slice(start, end))
  if (
    text(0, 16) === 'yeelion-kuwo-tme' ||
    text(0, 16) === 'yeelion-kuwo\u0000\u0000\u0000\u0000'
  ) {
    return 'kuwo'
  }
  const magic = Array.from(header.slice(0, 16))
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('')
  if (
    magic === '7cd532eb86027f4ba8afa68e0fff9914' ||
    magic === '0528bc96e9e45a4391aabdd07af53631'
  ) {
    return 'kugou'
  }
  if (text(0, 8) === 'CTENFDAM') return 'netease'
  return authorizedMediaProvider(filename)
}

export function uploadPrefilterError(
  filename: string,
  mimetype: string,
): string | null {
  const extension = fileExtension(filename)

  if (encryptedExtensions.has(extension)) return null
  if (
    supportedExtensions.has(extension) ||
    mimetype.startsWith('audio/') ||
    mimetype.startsWith('video/')
  ) {
    return null
  }
  return '仅支持 MP3、WAV、FLAC、APE、M4A、AAC、OGG、MP4、MOV 和 WEBM 文件'
}

export function detectStandardAudioSignature(header: Uint8Array): boolean {
  const text = (start: number, end: number) =>
    String.fromCharCode(...header.slice(start, end))
  const mp3Frame =
    header.length >= 2 && header[0] === 0xff && (header[1] & 0xe0) === 0xe0
  return (
    text(0, 4) === 'fLaC' ||
    text(0, 4) === 'MAC ' ||
    text(0, 4) === 'OggS' ||
    text(0, 3) === 'ID3' ||
    mp3Frame ||
    (text(0, 4) === 'RIFF' && text(8, 12) === 'WAVE') ||
    text(4, 8) === 'ftyp'
  )
}

export function uploadContentValidationError(
  filename: string,
  mimetype: string,
  header: Uint8Array,
): string | null {
  if (authorizedMediaProviderForContent(filename, header)) {
    return detectStandardAudioSignature(header) ? null : encryptedFileError
  }
  return uploadPrefilterError(filename, mimetype)
}

export function requiresAuthorizedMediaDecode(
  filename: string,
  header: Uint8Array,
): boolean {
  return (
    authorizedMediaProviderForContent(filename, header) !== null &&
    !detectStandardAudioSignature(header)
  )
}

export function scoreUploadPrefilterError(
  filename: string,
  mimetype: string,
): string | null {
  const extension = fileExtension(filename)
  if (
    supportedScoreExtensions.has(extension) ||
    extension === '.pdf' ||
    mimetype.startsWith('image/') ||
    mimetype === 'application/pdf'
  ) {
    return null
  }
  return '仅支持 PNG、JPG、WEBP 和 PDF 曲谱'
}

export function scoreImageContentValidationError(
  filename: string,
  mimetype: string,
  header: Uint8Array,
): string | null {
  const signature = String.fromCharCode(...header.slice(0, 12))
  const png =
    header.length >= 8 &&
    header[0] === 0x89 &&
    signature.slice(1, 4) === 'PNG'
  const jpeg =
    header.length >= 3 &&
    header[0] === 0xff &&
    header[1] === 0xd8 &&
    header[2] === 0xff
  const webp =
    signature.slice(0, 4) === 'RIFF' &&
    signature.slice(8, 12) === 'WEBP'
  if (png || jpeg || webp) return null
  if (
    fileExtension(filename) === '.pdf' ||
    mimetype === 'application/pdf'
  ) {
    return 'PDF 需先在客户端转换为页面图片'
  }
  return '曲谱图片内容与文件格式不匹配'
}
