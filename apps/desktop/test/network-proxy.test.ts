import assert from 'node:assert/strict'
import test from 'node:test'
import { proxyEnvironmentFromRule } from '../src/network-proxy.js'

test('passes an Electron system proxy to model download processes', () => {
  assert.deepEqual(
    proxyEnvironmentFromRule(
      'PROXY 127.0.0.1:7890; DIRECT',
      'internal.example',
    ),
    {
      HTTP_PROXY: 'http://127.0.0.1:7890',
      HTTPS_PROXY: 'http://127.0.0.1:7890',
      NO_PROXY: 'internal.example,127.0.0.1,localhost',
      http_proxy: 'http://127.0.0.1:7890',
      https_proxy: 'http://127.0.0.1:7890',
      no_proxy: 'internal.example,127.0.0.1,localhost',
    },
  )
  assert.deepEqual(proxyEnvironmentFromRule('DIRECT'), {})
  assert.deepEqual(proxyEnvironmentFromRule('SOCKS broken proxy'), {})
})
