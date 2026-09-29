import { extname } from 'node:path'

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
  '.mflac',
  '.mgg',
  '.ncm',
  '.kgm',
  '.vpr',
])

const encryptedFileError =
  '该文件是 QQ 音乐或网易云音乐的专有加密容器，无法作为标准音频解码。会员播放权限不等于解密授权，请使用来源软件合法导出的 MP3、FLAC、WAV、APE 或 M4A'

export function requiresContentInspection(filename: string): boolean {
  return encryptedExtensions.has(extname(filename).toLowerCase())
}

export function uploadPrefilterError(filename: string, mimetype: string): string | null {
  const extension = extname(filename).toLowerCase()

  if (encryptedExtensions.has(extension)) {
    return null
  }

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
  const mp3Frame = header.length >= 2 && header[0] === 0xff && (header[1] & 0xe0) === 0xe0
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
  if (requiresContentInspection(filename)) {
    return detectStandardAudioSignature(header) ? null : encryptedFileError
  }
  return uploadPrefilterError(filename, mimetype)
}
