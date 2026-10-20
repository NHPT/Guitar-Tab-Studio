import assert from 'node:assert/strict'
import test from 'node:test'
import {
  detectDefaultMediaBrowser,
  macDefaultBrowserIdentifier,
  mediaBrowserFromIdentifier,
  windowsDefaultBrowserIdentifier,
} from '../src/default-browser.js'

test('maps supported browser identifiers without reading a browser profile', () => {
  assert.deepEqual(mediaBrowserFromIdentifier('com.google.Chrome'), {
    id: 'chrome',
    label: 'Google Chrome',
  })
  assert.deepEqual(mediaBrowserFromIdentifier('MSEdgeHTM'), {
    id: 'edge',
    label: 'Microsoft Edge',
  })
  assert.deepEqual(mediaBrowserFromIdentifier('org.mozilla.firefox'), {
    id: 'firefox',
    label: 'Firefox',
  })
  assert.equal(mediaBrowserFromIdentifier('com.example.browser'), undefined)
})

test('extracts the HTTPS handler from platform command output', () => {
  const macOutput = `(
    {
      LSHandlerRoleAll = "com.apple.Safari";
      LSHandlerURLScheme = http;
    },
    {
      LSHandlerRoleAll = "com.microsoft.edgemac";
      LSHandlerURLScheme = https;
    }
  )`
  assert.equal(
    macDefaultBrowserIdentifier(macOutput),
    'com.microsoft.edgemac',
  )
  assert.equal(
    windowsDefaultBrowserIdentifier(
      '    ProgId    REG_SZ    FirefoxURL-308046B0AF4A39CB\r\n',
    ),
    'FirefoxURL-308046B0AF4A39CB',
  )
})

test('detects a supported default browser and reports unsupported handlers', async () => {
  assert.deepEqual(
    await detectDefaultMediaBrowser(
      'linux',
      async () => 'google-chrome.desktop\n',
    ),
    {
      supported: true,
      browser: { id: 'chrome', label: 'Google Chrome' },
    },
  )
  assert.deepEqual(
    await detectDefaultMediaBrowser(
      'linux',
      async () => 'custom-browser.desktop\n',
    ),
    {
      supported: false,
      detectedLabel: 'custom-browser.desktop',
    },
  )
})

test('falls back to the macOS workspace API when LaunchServices preferences are unavailable', async () => {
  const calls: Array<{ executable: string; argumentsList: string[] }> = []
  const result = await detectDefaultMediaBrowser(
    'darwin',
    async (executable, argumentsList) => {
      calls.push({ executable, argumentsList })
      if (executable === '/usr/bin/defaults') {
        throw new Error('LaunchServices preferences unavailable')
      }
      return '/Applications/Microsoft Edge.app\n'
    },
  )

  assert.deepEqual(result, {
    supported: true,
    browser: { id: 'edge', label: 'Microsoft Edge' },
  })
  assert.deepEqual(
    calls.map((call) => call.executable),
    ['/usr/bin/defaults', '/usr/bin/osascript'],
  )
  assert.match(calls[1].argumentsList.join(' '), /NSWorkspace/u)
})
