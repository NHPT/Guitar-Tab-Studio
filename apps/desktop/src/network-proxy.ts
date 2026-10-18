export function proxyEnvironmentFromRule(
  rule: string,
  existingNoProxy = '',
): NodeJS.ProcessEnv {
  const entry = rule
    .split(';')
    .map((value) => value.trim())
    .find((value) => /^(?:HTTPS|PROXY)\s+\S+$/iu.test(value))
  if (!entry) return {}

  const address = entry.replace(/^(?:HTTPS|PROXY)\s+/iu, '')
  const proxyUrl = /^[a-z][a-z0-9+.-]*:\/\//iu.test(address)
    ? address
    : `http://${address}`
  let parsed: URL
  try {
    parsed = new URL(proxyUrl)
  } catch {
    return {}
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return {}

  const noProxy = [existingNoProxy, '127.0.0.1', 'localhost']
    .filter(Boolean)
    .join(',')
  return {
    HTTP_PROXY: proxyUrl,
    HTTPS_PROXY: proxyUrl,
    NO_PROXY: noProxy,
    http_proxy: proxyUrl,
    https_proxy: proxyUrl,
    no_proxy: noProxy,
  }
}
