import { resolve } from 'node:path'
import { build } from 'esbuild'

const commercialBuild = process.env.GTS_COMMERCIAL_BUILD === '1'

await build({
  entryPoints: [resolve('src/main.ts')],
  bundle: true,
  platform: 'node',
  format: 'cjs',
  target: 'node22',
  external: ['builder-util-runtime', 'electron', 'electron-updater'],
  outfile: resolve('dist/main.cjs'),
  define: {
    'process.env.GTS_COMMERCIAL_BUILD': JSON.stringify(
      commercialBuild ? '1' : '',
    ),
  },
})
