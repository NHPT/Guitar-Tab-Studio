import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import type {
  DesktopDefaultMediaBrowser,
  MediaBrowserDescriptor,
} from '@gts/core'

const execFileAsync = promisify(execFile)

type CommandRunner = (
  executable: string,
  argumentsList: string[],
) => Promise<string>

const browserMatchers: Array<{
  id: MediaBrowserDescriptor['id']
  label: string
  pattern: RegExp
}> = [
  { id: 'edge', label: 'Microsoft Edge', pattern: /edge|msedge/iu },
  { id: 'brave', label: 'Brave', pattern: /brave/iu },
  { id: 'vivaldi', label: 'Vivaldi', pattern: /vivaldi/iu },
  { id: 'opera', label: 'Opera', pattern: /opera/iu },
  { id: 'chromium', label: 'Chromium', pattern: /chromium/iu },
  { id: 'chrome', label: 'Google Chrome', pattern: /chrome/iu },
  { id: 'firefox', label: 'Firefox', pattern: /firefox/iu },
  { id: 'safari', label: 'Safari', pattern: /safari/iu },
]
const windowsAppBoundBrowsers = new Set<MediaBrowserDescriptor['id']>([
  'brave',
  'chrome',
  'chromium',
  'edge',
  'opera',
  'vivaldi',
])

export function mediaBrowserFromIdentifier(
  identifier: string,
): MediaBrowserDescriptor | undefined {
  const normalized = identifier.trim()
  const matched = browserMatchers.find((candidate) =>
    candidate.pattern.test(normalized),
  )
  return matched ? { id: matched.id, label: matched.label } : undefined
}

export function macDefaultBrowserIdentifier(output: string): string {
  const handlers = output.match(/\{[^{}]+\}/gsu) ?? []
  const httpsHandler = handlers
    .filter((handler) => /LSHandlerURLScheme\s*=\s*"?https"?/iu.test(handler))
    .at(-1)
  return (
    httpsHandler?.match(
      /LSHandlerRole(?:All|Viewer)\s*=\s*"?([^";\n]+)"?/iu,
    )?.[1] ?? ''
  )
}

export function windowsDefaultBrowserIdentifier(output: string): string {
  return output.match(/ProgId\s+REG_\w+\s+([^\r\n]+)/iu)?.[1]?.trim() ?? ''
}

const macDefaultBrowserScript = [
  "ObjC.import('AppKit')",
  "const target = $.NSURL.URLWithString('https://example.com')",
  'const application = $.NSWorkspace.sharedWorkspace.URLForApplicationToOpenURL(target)',
  "application ? ObjC.unwrap(application.path) : ''",
].join('; ')

async function runCommand(
  executable: string,
  argumentsList: string[],
): Promise<string> {
  const result = await execFileAsync(executable, argumentsList, {
    encoding: 'utf8',
    timeout: 5_000,
    windowsHide: true,
  })
  return result.stdout
}

async function detectMacDefaultBrowserIdentifier(
  runner: CommandRunner,
): Promise<string> {
  try {
    const identifier = macDefaultBrowserIdentifier(
      await runner('/usr/bin/defaults', [
        'read',
        'com.apple.LaunchServices/com.apple.launchservices.secure',
        'LSHandlers',
      ]),
    )
    if (identifier) return identifier
  } catch {
    // Newer macOS versions may not expose LaunchServices handlers via defaults.
  }
  return (
    await runner('/usr/bin/osascript', [
      '-l',
      'JavaScript',
      '-e',
      macDefaultBrowserScript,
    ])
  ).trim()
}

export async function detectDefaultMediaBrowser(
  platform: NodeJS.Platform = process.platform,
  runner: CommandRunner = runCommand,
): Promise<DesktopDefaultMediaBrowser> {
  try {
    let identifier = ''
    if (platform === 'darwin') {
      identifier = await detectMacDefaultBrowserIdentifier(runner)
    } else if (platform === 'win32') {
      identifier = windowsDefaultBrowserIdentifier(
        await runner('reg.exe', [
          'query',
          'HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\Shell\\Associations\\UrlAssociations\\https\\UserChoice',
          '/v',
          'ProgId',
        ]),
      )
    } else {
      identifier = (
        await runner('xdg-settings', ['get', 'default-web-browser'])
      ).trim()
    }
    const browser = mediaBrowserFromIdentifier(identifier)
    return browser
      ? {
          supported: true,
          browser,
          sessionMode:
            platform === 'win32' && windowsAppBoundBrowsers.has(browser.id)
              ? 'isolated-login'
              : 'browser-cookies',
        }
      : {
          supported: false,
          ...(identifier ? { detectedLabel: identifier } : {}),
        }
  } catch {
    return { supported: false }
  }
}
