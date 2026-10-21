export interface SessionCookie {
  domain?: string
  expirationDate?: number
  httpOnly?: boolean
  name: string
  path?: string
  secure?: boolean
  value: string
}

export interface QQMusicWebCredential {
  musicId: string
  musicKey: string
  loginType: 1 | 2 | 3
}

function safeCookieField(value: string): string {
  return value.replaceAll(/[\u0000-\u001f\u007f]/gu, '')
}

export function netscapeCookieFile(
  cookies: SessionCookie[],
  acceptedDomains: string[],
): string {
  const domains = acceptedDomains.map((domain) => domain.toLowerCase())
  const lines = cookies
    .filter((cookie) => {
      const domain = (cookie.domain ?? '').replace(/^\./u, '').toLowerCase()
      return domains.some(
        (accepted) =>
          domain === accepted || domain.endsWith(`.${accepted}`),
      )
    })
    .map((cookie) => {
      const domain = safeCookieField(cookie.domain ?? '')
      const serializedDomain = cookie.httpOnly
        ? `#HttpOnly_${domain}`
        : domain
      return [
        serializedDomain,
        domain.startsWith('.') ? 'TRUE' : 'FALSE',
        safeCookieField(cookie.path || '/'),
        cookie.secure ? 'TRUE' : 'FALSE',
        String(Math.max(0, Math.floor(cookie.expirationDate ?? 0))),
        safeCookieField(cookie.name),
        safeCookieField(cookie.value),
      ].join('\t')
    })

  return `# Netscape HTTP Cookie File\n${lines.join('\n')}\n`
}

export function qqMusicCredentialFromCookies(
  cookies: SessionCookie[],
): QQMusicWebCredential | undefined {
  const values = new Map(
    cookies.map((cookie) => [cookie.name.toLowerCase(), cookie.value.trim()]),
  )
  const musicKey =
    values.get('qm_keyst') ??
    values.get('qqmusic_key') ??
    values.get('qqmusickey')
  let musicId =
    values.get('uin') ??
    values.get('p_uin') ??
    values.get('qqmusic_uin') ??
    values.get('wxuin')
  if (!musicKey || !musicId) return undefined
  musicId = musicId.replace(/^[oO]/u, '').replace(/^0+(?=\d{4,24}$)/u, '')
  if (!/^\d{4,24}$/u.test(musicId) || musicKey.length < 8) {
    return undefined
  }
  const rawLoginType = Number(
    values.get('tmelogintype') ?? values.get('login_type'),
  )
  const loginType =
    rawLoginType === 1 || rawLoginType === 2 || rawLoginType === 3
      ? rawLoginType
      : musicKey.startsWith('W_X')
        ? 1
        : 2
  return { musicId, musicKey, loginType }
}
