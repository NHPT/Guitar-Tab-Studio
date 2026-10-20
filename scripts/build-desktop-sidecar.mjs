import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import {
  chmodSync,
  copyFileSync,
  cpSync,
  createReadStream,
  existsSync,
  lstatSync,
  mkdirSync,
  realpathSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const repositoryRoot = resolve(scriptDirectory, '..')
const workerRoot = resolve(repositoryRoot, 'services/worker')
const outputRoot = resolve(
  repositoryRoot,
  'apps/desktop/build-resources/sidecar',
)
const buildRoot = resolve(repositoryRoot, 'apps/desktop/.sidecar-build')
const localPython = resolve(
  repositoryRoot,
  process.platform === 'win32'
    ? '.venv/Scripts/python.exe'
    : '.venv/bin/python',
)
const python =
  process.env.PYTHON ??
  (existsSync(localPython)
    ? localPython
    : process.platform === 'win32'
      ? 'python.exe'
      : 'python3')
const pythonProbe = spawnSync(python, ['--version'], {
  encoding: 'utf8',
})
if (pythonProbe.status !== 0) {
  throw new Error(`Python environment is unavailable: ${python}`)
}

const result = spawnSync(
  python,
  [
    '-m',
    'PyInstaller',
    '--clean',
    '--noconfirm',
    '--distpath',
    resolve(buildRoot, 'dist'),
    '--workpath',
    resolve(buildRoot, 'work'),
    resolve(workerRoot, 'guitar-tab-worker.spec'),
  ],
  {
    cwd: repositoryRoot,
    env: {
      ...process.env,
      PYINSTALLER_CONFIG_DIR: resolve(repositoryRoot, '.cache/pyinstaller'),
    },
    encoding: 'utf8',
    stdio: 'inherit',
  },
)
if (result.status !== 0) {
  throw new Error(`PyInstaller failed with exit code ${result.status ?? 1}`)
}

mkdirSync(outputRoot, { recursive: true })
for (const name of readdirSync(outputRoot)) {
  if (name !== 'README.md') {
    rmSync(resolve(outputRoot, name), { recursive: true, force: true })
  }
}

const frozenRoot = resolve(buildRoot, 'dist/guitar-tab-worker')
for (const name of readdirSync(frozenRoot)) {
  cpSync(resolve(frozenRoot, name), resolve(outputRoot, name), {
    recursive: true,
  })
}

function materializeSymlinks(root) {
  for (const name of readdirSync(root)) {
    const path = resolve(root, name)
    const stats = lstatSync(path)
    if (stats.isDirectory()) {
      materializeSymlinks(path)
      continue
    }
    if (!stats.isSymbolicLink()) continue
    const source = realpathSync(path)
    rmSync(path)
    if (statSync(source).isDirectory()) {
      cpSync(source, path, { recursive: true })
    } else {
      copyFileSync(source, path)
    }
  }
}

materializeSymlinks(outputRoot)

const require = createRequire(import.meta.url)
const ffmpegSource = require('ffmpeg-static')
if (typeof ffmpegSource !== 'string' || !existsSync(ffmpegSource)) {
  throw new Error('ffmpeg-static did not provide a platform binary')
}
const ffmpegName = process.platform === 'win32' ? 'ffmpeg.exe' : 'ffmpeg'
copyFileSync(ffmpegSource, resolve(outputRoot, ffmpegName))
if (process.platform !== 'win32') chmodSync(resolve(outputRoot, ffmpegName), 0o755)
copyFileSync(resolve(repositoryRoot, 'LICENSE'), resolve(outputRoot, 'LICENSE.txt'))

function filesIn(root, prefix = '') {
  return readdirSync(resolve(root, prefix)).flatMap((name) => {
    const relativePath = prefix ? `${prefix}/${name}` : name
    if (relativePath === 'sidecar-manifest.json') return []
    const path = resolve(root, relativePath)
    return statSync(path).isDirectory()
      ? filesIn(root, relativePath)
      : [relativePath]
  })
}

async function sha256(path) {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path)) hash.update(chunk)
  return hash.digest('hex')
}

const files = await Promise.all(
  filesIn(outputRoot).map(async (path) => ({
    path,
    sizeBytes: statSync(resolve(outputRoot, path)).size,
    sha256: await sha256(resolve(outputRoot, path)),
    role:
      path === ffmpegName
        ? 'ffmpeg'
        : path === 'LICENSE.txt'
          ? 'license'
          : 'worker-runtime',
  })),
)
const executableName =
  process.platform === 'win32' ? 'guitar-tab-worker.exe' : 'guitar-tab-worker'
if (!files.some((file) => file.path === executableName)) {
  throw new Error(`Frozen worker executable is missing: ${executableName}`)
}

writeFileSync(
  resolve(outputRoot, 'sidecar-manifest.json'),
  `${JSON.stringify(
    {
      schemaVersion: 1,
      createdAt: new Date().toISOString(),
      platform: process.platform,
      arch: process.arch,
      analysisSchemaVersion: 10,
      builder: 'pyinstaller-onedir',
      files,
    },
    null,
    2,
  )}\n`,
  'utf8',
)

rmSync(buildRoot, { recursive: true, force: true })
process.stdout.write(
  `Built ${relative(repositoryRoot, outputRoot)} with ${files.length} audited files\n`,
)
