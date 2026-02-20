/**
 * Opens a new browser tab from the first URL found in the given message content.
 * No-op if not in extension context or no URL is found.
 */
export function openTabFromMessage(content: string): void {
  if (typeof chrome?.tabs?.create !== 'function') return
  const urlMatch = content.match(/https?:\/\/[^\s)]+/)
  if (urlMatch) {
    try {
      chrome.tabs.create({ url: urlMatch[0] })
    } catch (_) {}
  }
}
