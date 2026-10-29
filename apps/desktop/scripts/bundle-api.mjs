import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { build } from 'esbuild'

const output = resolve('build-resources/api/index.mjs')

await build({
  entryPoints: [resolve('../api/src/index.ts')],
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node22',
  minifySyntax: true,
  banner: {
    js: "import { createRequire } from 'node:module'; const require = createRequire(import.meta.url);",
  },
  outfile: output,
  define: {
    'process.env.GTS_DESKTOP_API_BUILD': JSON.stringify('1'),
  },
})

const source = readFileSync(output, 'utf8')
const forbiddenMarkers = [
  '/api/admin',
  'gts_admin_session',
  'registerAdminApi',
]
for (const marker of forbiddenMarkers) {
  if (source.includes(marker)) {
    throw new Error(`Desktop API bundle contains forbidden marker: ${marker}`)
  }
}
