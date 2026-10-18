import { existsSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  parseAnalysisDocument,
  type AnalysisDocument,
} from '@gts/core'

export const builtinDemoAudioFilename =
  'take-me-home-country-roads.mp3'
export const builtinDemoStemFilenames = {
  vocals: 'take-me-home-country-roads-vocals.m4a',
  guitar: 'take-me-home-country-roads-guitar.m4a',
  bass: 'take-me-home-country-roads-bass.m4a',
  drums: 'take-me-home-country-roads-drums.m4a',
  piano: 'take-me-home-country-roads-piano.m4a',
  other: 'take-me-home-country-roads-other.m4a',
} as const
export const builtinDemoGuitarPartFilenames = [
  'take-me-home-country-roads-guitar-1.m4a',
  'take-me-home-country-roads-guitar-2.m4a',
] as const
const ANALYSIS_FILENAME = 'take-me-home-country-roads-analysis.json'
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
