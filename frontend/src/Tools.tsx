/**
 * Extracts the first URL from message content (plain URL or markdown link).
 * Returns null if no valid URL found.
 */
function extractUrlFromText(content: string): string | null {
  if (!content || typeof content !== 'string') return null
  // Markdown-style link: [text](url) – capture the url part
  const markdownMatch = content.match(/\]\s*\(\s*(https?:\/\/[^\s)]+)\s*\)/)
  if (markdownMatch) return markdownMatch[1].replace(/[.,;:!?)]+$/, '')
  // Plain URL – match then strip trailing punctuation
  const plainMatch = content.match(/https?:\/\/[^\s)\]]+/)
  if (plainMatch) return plainMatch[0].replace(/[.,;:!?)\]]+$/, '')
  return null
}

/**
 * Opens a new browser tab from the first URL found in the given message content.
 * No-op if not in extension context or no URL is found.
 */
export function openTabFromMessage(content: string): void {
  const DEBUG = true
  try {
    const chromeApi = typeof chrome !== 'undefined' ? chrome : (typeof (window as any).chrome !== 'undefined' ? (window as any).chrome : null)
    if (DEBUG) console.log('[openTabFromMessage] chrome.tabs available:', !!chromeApi?.tabs?.create)
    if (!chromeApi?.tabs?.create) {
      if (DEBUG) console.log('[openTabFromMessage] skipping: no chrome.tabs')
      return
    }
    const url = extractUrlFromText(content)
    if (DEBUG) console.log('[openTabFromMessage] extracted URL:', url ?? '(none)', 'from content:', content?.slice(0, 150))
    if (url) {
      chromeApi.tabs.create({ url })
      if (DEBUG) console.log('[openTabFromMessage] opened tab:', url)
    } else {
      if (DEBUG) console.log('[openTabFromMessage] no URL found in message, tab not opened')
    }
  } catch (e) {
    if (DEBUG) console.error('[openTabFromMessage] error:', e)
  }
}
