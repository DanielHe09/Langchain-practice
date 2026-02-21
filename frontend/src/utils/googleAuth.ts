/**
 * Google OAuth token storage keys (chrome.storage.local)
 */
export const GOOGLE_ACCESS_TOKEN = 'google_access_token'
export const GOOGLE_REFRESH_TOKEN = 'google_refresh_token'
export const GOOGLE_TOKEN_EXPIRES_AT = 'google_token_expires_at'

const STORAGE_KEYS = [GOOGLE_ACCESS_TOKEN, GOOGLE_REFRESH_TOKEN, GOOGLE_TOKEN_EXPIRES_AT] as const

export interface GoogleTokens {
  access_token: string | null
  refresh_token: string | null
  expires_at: number | null
}

export function getChrome(): typeof chrome | undefined {
  return typeof chrome !== 'undefined' ? chrome : (typeof (window as any).chrome !== 'undefined' ? (window as any).chrome : undefined)
}

/**
 * Get stored Google tokens from chrome.storage.local.
 * Returns nulls if not in extension context.
 */
export async function getGoogleTokens(): Promise<GoogleTokens> {
  const c = getChrome()
  if (!c?.storage?.local?.get) {
    return { access_token: null, refresh_token: null, expires_at: null }
  }
  const result = await c.storage.local.get(STORAGE_KEYS)
  return {
    access_token: result[GOOGLE_ACCESS_TOKEN] || null,
    refresh_token: result[GOOGLE_REFRESH_TOKEN] || null,
    expires_at: result[GOOGLE_TOKEN_EXPIRES_AT] ?? null,
  }
}

/**
 * Store Google tokens. expires_in is seconds from now; we store expires_at as timestamp.
 */
export async function setGoogleTokens(
  access_token: string,
  refresh_token: string | null,
  expires_in: number
): Promise<void> {
  const c = getChrome()
  if (!c?.storage?.local?.set) return
  const expires_at = Date.now() + expires_in * 1000
  await c.storage.local.set({
    [GOOGLE_ACCESS_TOKEN]: access_token,
    [GOOGLE_REFRESH_TOKEN]: refresh_token ?? undefined,
    [GOOGLE_TOKEN_EXPIRES_AT]: expires_at,
  })
}

export async function clearGoogleTokens(): Promise<void> {
  const c = getChrome()
  if (!c?.storage?.local?.remove) return
  await c.storage.local.remove(STORAGE_KEYS)
}

/** Consider token expired 60s before actual expiry */
const EXPIRY_BUFFER_MS = 60_000

/**
 * Returns a valid access token, refreshing if necessary using the backend.
 * Call from background script when sending screenshot; apiUrl is the backend base URL.
 */
export async function getValidGoogleAccessToken(apiUrl: string): Promise<string | null> {
  const tokens = await getGoogleTokens()
  const now = Date.now()
  const expired = tokens.expires_at != null && now >= tokens.expires_at - EXPIRY_BUFFER_MS
  if (tokens.access_token && !expired) return tokens.access_token
  if (!tokens.refresh_token) return null
  try {
    const r = await fetch(`${apiUrl}/api/google-auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: tokens.refresh_token }),
    })
    if (!r.ok) return null
    const body = await r.json()
    const access_token = body.access_token
    const expires_in = body.expires_in ?? 3600
    if (access_token) {
      await setGoogleTokens(access_token, tokens.refresh_token, expires_in)
      return access_token
    }
  } catch (_) {}
  return null
}
