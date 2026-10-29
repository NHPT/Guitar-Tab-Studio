const restrictedRendererPath =
  /^\/(?:admin|community|review)(?:\/|$)/iu

function parsedUrl(value: string): URL | undefined {
  try {
    return new URL(value)
  } catch {
    return undefined
  }
}

export function isUserRendererPath(pathname: string): boolean {
  let decoded = pathname
  try {
    decoded = decodeURIComponent(pathname)
  } catch {
    return false
  }
  return !restrictedRendererPath.test(decoded)
}

export function developmentRendererUrl(
  value: string | undefined,
): URL | undefined {
  if (!value) return undefined
  const url = parsedUrl(value)
  if (
    !url ||
    url.protocol !== 'http:' ||
    !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) ||
    url.username ||
    url.password ||
    !isUserRendererPath(url.pathname)
  ) {
    throw new Error(
      'GTS_DESKTOP_RENDERER_URL must target a user-facing loopback HTTP page',
    )
  }
  return url
}

export function isDesktopNavigationAllowed(
  targetValue: string,
  entryValue: string,
): boolean {
  const target = parsedUrl(targetValue)
  const entry = parsedUrl(entryValue)
  if (
    !target ||
    !entry ||
    !isUserRendererPath(target.pathname) ||
    restrictedRendererPath.test(target.hash.slice(1))
  ) {
    return false
  }
  if (entry.protocol === 'file:') {
    return target.protocol === 'file:' && target.pathname === entry.pathname
  }
  return (
    target.protocol === entry.protocol &&
    target.origin === entry.origin &&
    target.pathname === entry.pathname
  )
}
