import assert from 'node:assert/strict'
import {
  chmodSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import test from 'node:test'
import { DesktopTranscriptionManager } from '../src/model-providers.js'
import { DesktopStateStore } from '../src/state.js'

test('manual variant overrides the automatic hardware recommendation', () => {
  const root = resolve(process.cwd(), 'data', `.test-providers-${randomUUID()}`)
  mkdirSync(root, { recursive: true })
  const state = new DesktopStateStore(resolve(root, 'desktop.sqlite'))
  try {
    const manager = new DesktopTranscriptionManager(
      resolve(root, 'providers'),
      resolve(root, 'cache'),
      state,
      () => undefined,
    )
    manager.setPreference({
      providerId: 'muscriptor',
      selectionMode: 'manual',
      variant: 'small',
      licenseAccepted: true,
    })

    const status = manager.status()
    assert.equal(status.preference.selectionMode, 'manual')
    assert.equal(status.resolvedVariant, 'small')
  } finally {
    state.close()
    rmSync(root, { recursive: true, force: true })
  }
})

test('does not run the CPU-only MR-MT3 verifier on MPS', {
  skip: process.platform === 'win32' && 'requires a POSIX worker fixture',
}, () => {
  const root = resolve(process.cwd(), 'data', `.test-providers-${randomUUID()}`)
  const worker = resolve(root, 'worker.sh')
  mkdirSync(root, { recursive: true })
  writeFileSync(
    worker,
    `#!/bin/sh
if [ "$1" = "--hardware-probe" ]; then
  printf '%s\\n' '{"accelerator":"mps","providers":{"basic-pitch":true,"muscriptor":true,"mr-mt3":true}}'
fi
`,
  )
  chmodSync(worker, 0o755)
  const state = new DesktopStateStore(resolve(root, 'desktop.sqlite'))
  try {
    const manager = new DesktopTranscriptionManager(
      resolve(root, 'providers'),
      resolve(root, 'cache'),
      state,
      () => worker,
    )
    manager.setPreference({
      providerId: 'muscriptor',
      selectionMode: 'manual',
      variant: 'medium',
      licenseAccepted: true,
    })

    assert.equal(
      manager.status().pipeline.find((item) => item.id === 'mr-mt3')
        ?.selected,
      false,
    )
    assert.equal(manager.workerConfiguration().verificationModel, undefined)
  } finally {
    state.close()
    rmSync(root, { recursive: true, force: true })
  }
})

test('imports a local MuScriptor weight into managed private storage', async () => {
  const root = resolve(process.cwd(), 'data', `.test-providers-${randomUUID()}`)
  mkdirSync(root, { recursive: true })
  const source = resolve(root, 'downloaded-medium.safetensors')
  writeFileSync(source, 'weights')
  const state = new DesktopStateStore(resolve(root, 'desktop.sqlite'))
  try {
    const manager = new DesktopTranscriptionManager(
      resolve(root, 'providers'),
      resolve(root, 'cache'),
      state,
      () => undefined,
    )
    const manifest = await manager.importWeights(source, 'medium', true)

    assert.equal(manifest.variant, 'medium')
    assert.equal(manager.status().weightSource, 'built-in')
    manager.setPreference({
      providerId: 'muscriptor',
      selectionMode: 'manual',
      variant: 'medium',
      licenseAccepted: true,
    })
    assert.equal(manager.status().weightSource, 'local-import')
    assert.deepEqual(
      JSON.parse(
        readFileSync(
          resolve(root, 'providers/muscriptor/medium/config.json'),
          'utf8',
        ),
      ),
      { dim: 1024, num_heads: 16, num_layers: 24, card: 1395 },
    )
  } finally {
    state.close()
    rmSync(root, { recursive: true, force: true })
  }
})

test('downloads and verifies weights before persisting high-accuracy mode', {
  skip: process.platform === 'win32' && 'requires a POSIX worker fixture',
}, async () => {
  const root = resolve(process.cwd(), 'data', `.test-providers-${randomUUID()}`)
  const cache = resolve(root, 'cache')
  const worker = resolve(root, 'worker.sh')
  mkdirSync(root, { recursive: true })
  const progress: Array<{ phase: string; progress: number }> = []
  writeFileSync(
    worker,
    `#!/bin/sh
if [ "$1" = "--hardware-probe" ]; then
  printf '%s\\n' '{"accelerator":"cuda","acceleratorMemoryBytes":8589934592,"providers":{"basic-pitch":true,"muscriptor":true,"mr-mt3":true}}'
  exit 0
fi
test "$HF_TOKEN" = "hf_oauth_test" || exit 9
printf '%s\\n' '{"type":"gts-model-progress","progress":42,"stageLabel":"正在下载 MuScriptor medium 权重"}'
mkdir -p "$HF_HUB_CACHE/models--MuScriptor--muscriptor-medium/snapshots/test"
printf 'weights' > "$HF_HUB_CACHE/models--MuScriptor--muscriptor-medium/snapshots/test/model.safetensors"
mkdir -p "$GTS_MODEL_PROVIDER_ROOT/noise2fret/1.0.0"
dd if=/dev/zero of="$GTS_MODEL_PROVIDER_ROOT/noise2fret/1.0.0/model.pth" bs=1 count=0 seek=60039447 2>/dev/null
printf '%s\\n' '{"schemaVersion":1,"provider":"noise2fret","version":"1.0.0","sizeBytes":60039447,"sha256":"1f454f75e9e6f44d80286e2c13188264b10d09c08fa8bdd791b1c809217bab95","license":"CC-BY-4.0"}' > "$GTS_MODEL_PROVIDER_ROOT/noise2fret/1.0.0/manifest.json"
mkdir -p "$GTS_MODEL_PROVIDER_ROOT/mr-mt3/1.0.0"
dd if=/dev/zero of="$GTS_MODEL_PROVIDER_ROOT/mr-mt3/1.0.0/model.pth" bs=1 count=0 seek=183672643 2>/dev/null
printf '%s\\n' '{"schemaVersion":1,"provider":"mr-mt3","version":"1.0.0","sizeBytes":183672643,"sha256":"b8a3807ed265059abd25ad7f68142c06c35e8f6144dcaa45bd55946a3745398f","license":"MIT"}' > "$GTS_MODEL_PROVIDER_ROOT/mr-mt3/1.0.0/manifest.json"
`,
  )
  chmodSync(worker, 0o755)
  const state = new DesktopStateStore(resolve(root, 'desktop.sqlite'))
  try {
    const manager = new DesktopTranscriptionManager(
      resolve(root, 'providers'),
      cache,
      state,
      () => worker,
      (event) => progress.push(event),
      () => undefined,
      async () => 'hf_oauth_test',
      () => 'desktop-client',
    )
    assert.equal(manager.status().weightSource, 'built-in')

    await manager.prepareAndSetPreference({
      providerId: 'muscriptor',
      selectionMode: 'manual',
      variant: 'medium',
      licenseAccepted: true,
    })

    assert.equal(state.transcriptionPreference().providerId, 'muscriptor')
    assert.equal(manager.status().weightSource, 'managed-cache')
    assert.equal(manager.status().ready, true)
    assert.ok(
      progress.some(
        (event) => event.phase === 'downloading' && event.progress === 42,
      ),
    )
    assert.equal(
      manager.status().pipeline.find((item) => item.id === 'mr-mt3')
        ?.ready,
      true,
    )
    assert.equal(
      manager.status().pipeline.find((item) => item.id === 'mr-mt3')
        ?.selected,
      true,
    )
    assert.equal(
      manager.workerConfiguration().verificationModel,
      resolve(root, 'providers', 'mr-mt3', '1.0.0', 'model.pth'),
    )
  } finally {
    state.close()
    rmSync(root, { recursive: true, force: true })
  }
})

test('accepts imported MuScriptor weights after preparing missing auxiliaries', {
  skip: process.platform === 'win32' && 'requires a POSIX worker fixture',
}, async () => {
  const root = resolve(process.cwd(), 'data', `.test-providers-${randomUUID()}`)
  const cache = resolve(root, 'cache')
  const source = resolve(root, 'imported-small.safetensors')
  const worker = resolve(root, 'worker.sh')
  mkdirSync(root, { recursive: true })
  writeFileSync(source, 'weights')
  writeFileSync(
    worker,
    `#!/bin/sh
if [ "$1" = "--hardware-probe" ]; then
  printf '%s\\n' '{"accelerator":"mps","providers":{"basic-pitch":true,"muscriptor":true,"mr-mt3":true}}'
  exit 0
fi
mkdir -p "$GTS_MODEL_PROVIDER_ROOT/noise2fret/1.0.0"
dd if=/dev/zero of="$GTS_MODEL_PROVIDER_ROOT/noise2fret/1.0.0/model.pth" bs=1 count=0 seek=60039447 2>/dev/null
printf '%s\\n' '{"schemaVersion":1,"provider":"noise2fret","version":"1.0.0","sizeBytes":60039447,"sha256":"1f454f75e9e6f44d80286e2c13188264b10d09c08fa8bdd791b1c809217bab95","license":"CC-BY-4.0"}' > "$GTS_MODEL_PROVIDER_ROOT/noise2fret/1.0.0/manifest.json"
`,
  )
  chmodSync(worker, 0o755)
  const state = new DesktopStateStore(resolve(root, 'desktop.sqlite'))
  try {
    const manager = new DesktopTranscriptionManager(
      resolve(root, 'providers'),
      cache,
      state,
      () => worker,
    )
    await manager.importWeights(source, 'small', true)

    await manager.prepareAndSetPreference({
      providerId: 'muscriptor',
      selectionMode: 'manual',
      variant: 'small',
      licenseAccepted: true,
    })

    assert.equal(state.transcriptionPreference().providerId, 'muscriptor')
    assert.equal(manager.status().weightSource, 'local-import')
    assert.equal(manager.status().ready, true)
  } finally {
    state.close()
    rmSync(root, { recursive: true, force: true })
  }
})

test('keeps Basic Pitch active when high-accuracy preparation fails', {
  skip: process.platform === 'win32' && 'requires a POSIX worker fixture',
}, async () => {
  const root = resolve(process.cwd(), 'data', `.test-providers-${randomUUID()}`)
  const worker = resolve(root, 'worker.sh')
  mkdirSync(root, { recursive: true })
  writeFileSync(
    worker,
    `#!/bin/sh
if [ "$1" = "--hardware-probe" ]; then
  printf '%s\\n' '{"accelerator":"mps","providers":{"basic-pitch":true,"muscriptor":true}}'
  exit 0
fi
printf '%s\\n' 'Hugging Face 授权未完成' >&2
exit 1
`,
  )
  chmodSync(worker, 0o755)
  const state = new DesktopStateStore(resolve(root, 'desktop.sqlite'))
  try {
    const manager = new DesktopTranscriptionManager(
      resolve(root, 'providers'),
      resolve(root, 'cache'),
      state,
      () => worker,
    )
    await assert.rejects(
      manager.prepareAndSetPreference({
        providerId: 'muscriptor',
        selectionMode: 'manual',
        variant: 'small',
        licenseAccepted: true,
      }),
      /授权未完成/,
    )
    assert.equal(
      state.transcriptionPreference().providerId,
      'basic-pitch',
    )
  } finally {
    state.close()
    rmSync(root, { recursive: true, force: true })
  }
})

test('cancels an authorization wait and allows a later retry', {
  skip: process.platform === 'win32' && 'requires a POSIX worker fixture',
}, async () => {
  const root = resolve(process.cwd(), 'data', `.test-providers-${randomUUID()}`)
  const worker = resolve(root, 'worker.sh')
  mkdirSync(root, { recursive: true })
  writeFileSync(
    worker,
    `#!/bin/sh
if [ "$1" = "--hardware-probe" ]; then
  printf '%s\\n' '{"accelerator":"mps","providers":{"basic-pitch":true,"muscriptor":true}}'
  exit 0
fi
trap 'exit 143' TERM
printf '%s\\n' '{"type":"gts-auth-required","progress":3,"stageLabel":"等待 Hugging Face 登录完成","authorizationUrl":"https://huggingface.co/device?user_code=ABCD-EFGH","userCode":"ABCD-EFGH"}'
while :; do sleep 1; done
`,
  )
  chmodSync(worker, 0o755)
  const state = new DesktopStateStore(resolve(root, 'desktop.sqlite'))
  let authorizationStarted = () => undefined
  const authorization = new Promise<void>((resolveStarted) => {
    authorizationStarted = resolveStarted
  })
  try {
    const manager = new DesktopTranscriptionManager(
      resolve(root, 'providers'),
      resolve(root, 'cache'),
      state,
      () => worker,
      (event) => {
        if (event.phase === 'authorizing') authorizationStarted()
      },
    )
    const preparation = manager.prepareAndSetPreference({
      providerId: 'muscriptor',
      selectionMode: 'manual',
      variant: 'small',
      licenseAccepted: true,
    })
    await authorization

    assert.equal(manager.cancelPreparation(), true)
    await assert.rejects(preparation, /已取消高精度模型准备/)
    assert.equal(manager.cancelPreparation(), false)
    assert.equal(state.transcriptionPreference().providerId, 'basic-pitch')
  } finally {
    state.close()
    rmSync(root, { recursive: true, force: true })
  }
})

test('activates an installed guitar-specialist research bundle', async () => {
  const root = resolve(process.cwd(), 'data', `.test-providers-${randomUUID()}`)
  const model = resolve(root, 'stringtrace-research.bundle.pt')
  mkdirSync(root, { recursive: true })
  writeFileSync(model, 'research-model')
  const state = new DesktopStateStore(resolve(root, 'desktop.sqlite'))
  try {
    const manager = new DesktopTranscriptionManager(
      resolve(root, 'providers'),
      resolve(root, 'cache'),
      state,
      () => undefined,
      () => undefined,
      () => model,
    )

    await manager.prepareAndSetPreference({
      providerId: 'guitar-specialist',
    })

    assert.equal(
      manager.status().preference.providerId,
      'guitar-specialist',
    )
    assert.deepEqual(manager.workerConfiguration(), {
      provider: 'guitar-specialist',
      tabModel: model,
      tabModelMode: 'onset-position-research',
      cacheRoot: resolve(root, 'cache'),
    })
  } finally {
    state.close()
    rmSync(root, { recursive: true, force: true })
  }
})

test('downloads and activates the multitrack research provider', {
  skip: process.platform === 'win32' && 'requires a POSIX worker fixture',
}, async () => {
  const root = resolve(process.cwd(), 'data', `.test-providers-${randomUUID()}`)
  const worker = resolve(root, 'worker.sh')
  mkdirSync(root, { recursive: true })
  writeFileSync(
    worker,
    `#!/bin/sh
if [ "$1" = "--hardware-probe" ]; then
  printf '%s\\n' '{"accelerator":"mps","providers":{"basic-pitch":true,"muscriptor":true,"mr-mt3":true,"guitar-specialist":true,"multitrack-research":true}}'
  exit 0
fi
mkdir -p "$GTS_MODEL_PROVIDER_ROOT/yourmt3/1.0.0"
dd if=/dev/zero of="$GTS_MODEL_PROVIDER_ROOT/yourmt3/1.0.0/last.ckpt" bs=1 count=0 seek=561544628 2>/dev/null
printf '%s\\n' '{"schemaVersion":1,"provider":"yourmt3","version":"1.0.0","sizeBytes":561544628,"sha256":"ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25","license":"research-only"}' > "$GTS_MODEL_PROVIDER_ROOT/yourmt3/1.0.0/manifest.json"
mkdir -p "$GTS_MODEL_PROVIDER_ROOT/noise2fret/1.0.0"
dd if=/dev/zero of="$GTS_MODEL_PROVIDER_ROOT/noise2fret/1.0.0/model.pth" bs=1 count=0 seek=60039447 2>/dev/null
printf '%s\\n' '{"schemaVersion":1,"provider":"noise2fret","version":"1.0.0","sizeBytes":60039447,"sha256":"1f454f75e9e6f44d80286e2c13188264b10d09c08fa8bdd791b1c809217bab95","license":"CC-BY-4.0"}' > "$GTS_MODEL_PROVIDER_ROOT/noise2fret/1.0.0/manifest.json"
`,
  )
  chmodSync(worker, 0o755)
  const state = new DesktopStateStore(resolve(root, 'desktop.sqlite'))
  try {
    const manager = new DesktopTranscriptionManager(
      resolve(root, 'providers'),
      resolve(root, 'cache'),
      state,
      () => worker,
    )

    await manager.prepareAndSetPreference({
      providerId: 'multitrack-research',
    })

    assert.equal(
      manager.status().preference.providerId,
      'multitrack-research',
    )
    assert.equal(manager.status().ready, true)
    assert.equal(
      manager.workerConfiguration().model,
      resolve(root, 'providers', 'yourmt3', '1.0.0', 'last.ckpt'),
    )
  } finally {
    state.close()
    rmSync(root, { recursive: true, force: true })
  }
})
