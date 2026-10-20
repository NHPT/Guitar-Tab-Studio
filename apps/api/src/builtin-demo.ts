import { existsSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  parseAnalysisDocument,
  type AnalysisDocument,
} from '@gts/core'

export const builtinDemoAudioFilename =
  'internationale.ogg'
export const builtinDemoStemFilenames = {
  vocals: 'internationale-vocals.m4a',
  guitar: 'internationale-guitar.m4a',
  bass: 'internationale-bass.m4a',
  drums: 'internationale-drums.m4a',
  piano: 'internationale-piano.m4a',
  other: 'internationale-other.m4a',
} as const
export const builtinDemoGuitarPartFilenames = [] as const
const ANALYSIS_FILENAME = 'internationale-analysis.json'
const moduleDirectory = dirname(fileURLToPath(import.meta.url))
const assetDirectories = [
  resolve(moduleDirectory, '../assets'),
  resolve(moduleDirectory, 'assets'),
]

export function builtinDemoAssetPath(filename: string): string {
  if (!/^[a-z0-9][a-z0-9.-]*$/u.test(filename)) {
    throw new Error('Invalid built-in demo asset name')
  }
  const path = assetDirectories
    .map((directory) => resolve(directory, filename))
    .find((candidate) => existsSync(candidate))
  if (!path) {
    throw new Error(`Built-in demo asset is unavailable: ${filename}`)
  }
  return path
}

function readAnalysis(): AnalysisDocument {
  return parseAnalysisDocument(
    JSON.parse(
      readFileSync(
        builtinDemoAssetPath(ANALYSIS_FILENAME),
        'utf8',
      ),
    ),
  )
}

export const builtinDemoAnalysis = readAnalysis()
export const builtinDemoAudioPath = builtinDemoAssetPath(
  builtinDemoAudioFilename,
)
export const builtinDemoMediaPaths = new Map(
  [
    builtinDemoAudioFilename,
    ...Object.values(builtinDemoStemFilenames),
    ...builtinDemoGuitarPartFilenames,
  ]
    .filter((filename, index, filenames) => filenames.indexOf(filename) === index)
    .map((filename) => [filename, builtinDemoAssetPath(filename)]),
)
