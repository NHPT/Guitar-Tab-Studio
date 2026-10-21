import { spawn, spawnSync } from 'node:child_process'
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs'
import { createServer } from 'node:http'
import { extname, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(fileURLToPath(new URL('..', import.meta.url)))
const cacheRoot = resolve(root, '.cache/beat-gold/review-preview')
const snapshotRoot = resolve(cacheRoot, 'snapshot')
const apiSnapshot = resolve(snapshotRoot, 'api')
const webSnapshot = resolve(snapshotRoot, 'web')
const runtimeRoot = resolve(cacheRoot, 'runtime')
const corpusRoot = resolve(root, '.cache/beat-gold/pilot-v2')
const apiPort = Number(process.env.GTS_REVIEW_API_PORT ?? 8791)
const webPort = Number(process.env.GTS_REVIEW_WEB_PORT ?? 5191)
const freezeOnly = process.argv.includes('--freeze-only')
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm'

function assertPort(value, name) {
  if (!Number.isInteger(value) || value < 1 || value > 65_535) {
    throw new Error(`${name} must be a valid TCP port`)
  }
}

function run(command, args, environment = {}) {
  const result = spawnSync(command, args, {
    cwd: root,
    env: { ...process.env, ...environment },
    stdio: 'inherit',
  })
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(' ')} failed`)
  }
}

function freezeSnapshot() {
  if (!existsSync(resolve(corpusRoot, 'pilot-manifest.json'))) {
    throw new Error('Beat Pilot v2 is not prepared')
  }
  run(npm, ['run', 'build', '-w', '@gts/core'])
  run(npm, ['run', 'build', '-w', '@gts/api'])
  run(
    npm,
    ['run', 'build', '-w', 'web'],
    { VITE_API_ROOT: `http://127.0.0.1:${apiPort}` },
  )

  rmSync(snapshotRoot, { recursive: true, force: true })
  mkdirSync(apiSnapshot, { recursive: true })
  cpSync(resolve(root, 'apps/api/dist'), resolve(apiSnapshot, 'dist'), {
    recursive: true,
  })
  cpSync(
    resolve(root, 'apps/api/package.json'),
    resolve(apiSnapshot, 'package.json'),
  )
  cpSync(resolve(root, 'apps/api/assets'), resolve(apiSnapshot, 'assets'), {
    recursive: true,
  })
  cpSync(resolve(root, 'apps/web/dist'), webSnapshot, { recursive: true })

  const manifest = JSON.parse(
    readFileSync(resolve(corpusRoot, 'pilot-manifest.json'), 'utf8'),
  )
  writeFileSync(
    resolve(snapshotRoot, 'snapshot.json'),
    `${JSON.stringify({
      createdAt: new Date().toISOString(),
      corpusVersion: manifest.corpus_version,
      corpusSha256: manifest.corpus_sha256,
      webPort,
      apiPort,
    }, null, 2)}\n`,
    { encoding: 'utf8', mode: 0o600 },
  )
}

function contentType(path) {
  return (
    {
      '.css': 'text/css; charset=utf-8',
      '.html': 'text/html; charset=utf-8',
      '.js': 'text/javascript; charset=utf-8',
      '.json': 'application/json; charset=utf-8',
      '.map': 'application/json; charset=utf-8',
      '.mjs': 'text/javascript; charset=utf-8',
      '.png': 'image/png',
      '.svg': 'image/svg+xml',
      '.woff2': 'font/woff2',
    }[extname(path)] ?? 'application/octet-stream'
  )
}

assertPort(apiPort, 'GTS_REVIEW_API_PORT')
assertPort(webPort, 'GTS_REVIEW_WEB_PORT')

if (freezeOnly) {
  freezeSnapshot()
  process.stdout.write('Frozen beat review snapshot prepared.\n')
  process.exit(0)
}

if (
  !existsSync(resolve(apiSnapshot, 'dist/index.js')) ||
  !existsSync(resolve(webSnapshot, 'index.html'))
) {
  throw new Error(
    'Frozen review snapshot is missing. Run npm run beat:review:freeze first.',
  )
}

mkdirSync(runtimeRoot, { recursive: true })
const api = spawn(
  process.execPath,
  [resolve(apiSnapshot, 'dist/index.js')],
  {
    cwd: root,
    env: {
      ...process.env,
      NODE_ENV: 'development',
      PORT: String(apiPort),
      GTS_DATA_ROOT: runtimeRoot,
      GTS_BEAT_GOLD_ROOT: corpusRoot,
    },
    stdio: 'inherit',
  },
)

const web = createServer((request, response) => {
  try {
    if (request.headers.host?.split(':')[0] === 'localhost') {
      response
        .writeHead(307, {
          Location: `http://127.0.0.1:${webPort}${request.url ?? '/'}`,
          'Cache-Control': 'no-store',
        })
        .end()
      return
    }
    const url = new URL(request.url ?? '/', `http://localhost:${webPort}`)
    const decoded = decodeURIComponent(url.pathname)
    const assetIndex = decoded.indexOf('/assets/')
    const publicPath =
      assetIndex >= 0
        ? decoded.slice(assetIndex)
        : decoded.endsWith('/favicon.svg')
          ? '/favicon.svg'
          : decoded.endsWith('/icons.svg')
            ? '/icons.svg'
            : decoded.startsWith('/admin/')
              ? decoded.slice('/admin'.length)
              : decoded
    const requested = resolve(webSnapshot, `.${publicPath}`)
    const safe =
      relative(webSnapshot, requested) !== '' &&
      !relative(webSnapshot, requested).startsWith('..')
    const file =
      safe && existsSync(requested) && statSync(requested).isFile()
        ? requested
        : resolve(webSnapshot, 'index.html')
    response
      .writeHead(200, {
        'Content-Type': contentType(file),
        'Cache-Control': file.endsWith('index.html')
          ? 'no-store'
          : 'public, max-age=31536000, immutable',
      })
      .end(readFileSync(file))
  } catch {
    response.writeHead(400).end('Bad request')
  }
})

let shuttingDown = false
function shutdown(exitCode = 0) {
  if (shuttingDown) return
  shuttingDown = true
  web.close()
  api.kill('SIGTERM')
  setTimeout(() => process.exit(exitCode), 250).unref()
}

api.once('exit', (code) => shutdown(code ?? 1))
api.once('error', () => shutdown(1))
web.once('error', () => shutdown(1))
process.once('SIGINT', () => shutdown(0))
process.once('SIGTERM', () => shutdown(0))

web.listen(webPort, '127.0.0.1', () => {
  process.stdout.write(
    `Frozen beat review preview: http://127.0.0.1:${webPort}/admin/login\n`,
  )
})
