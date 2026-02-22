// Background script to handle tab screenshots and Google token storage
import { getValidGoogleAccessToken, setGoogleTokens } from './utils/googleAuth'

console.log('Background script loaded!')
const API_URL = 'http://localhost:8000'

// Handle Connect Google: exchange code for tokens and store in chrome.storage (so tokens persist when popup closes)
chrome.runtime.onMessage.addListener(
  (
    msg: { type: string; code?: string; redirect_uri?: string },
    _sender: chrome.runtime.MessageSender,
    sendResponse: (response: { success: boolean; error?: string }) => void
  ) => {
    if (msg.type !== 'GOOGLE_AUTH_SAVE' || !msg.code || !msg.redirect_uri) {
      sendResponse({ success: false, error: 'Missing code or redirect_uri' })
      return true
    }
    ;(async () => {
      try {
        const res = await fetch(`${API_URL}/api/google-auth/code`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: msg.code, redirect_uri: msg.redirect_uri }),
        })
        if (!res.ok) throw new Error(await res.text())
        const data = await res.json()
        await setGoogleTokens(
          data.access_token,
          data.refresh_token ?? null,
          data.expires_in ?? 3600
        )
        sendResponse({ success: true })
      } catch (e: any) {
        sendResponse({ success: false, error: e?.message || String(e) })
      }
    })()
    return true // keep channel open for async sendResponse
  }
)

// Function to get token from chrome.storage
async function getStoredToken(): Promise<string | null> {
  try {
    const result = await chrome.storage.local.get(['supabase_token'])
    return result.supabase_token || null
  } catch (error) {
    console.error('Error getting token from storage:', error)
    return null
  }
}

// Function to send screenshot to backend
async function sendScreenshotToBackend(dataUrl: string, url: string, title?: string): Promise<void> {
  try {
    // Convert data URL to base64 (remove data:image/png;base64, prefix)
    const base64Data = dataUrl.split(',')[1]
    
    // Create ISO timestamp
    const capturedAt = new Date().toISOString()
    
    // Get JWT token from storage
    const token = await getStoredToken()
    console.log('🔑 Token retrieved for screenshot:', token ? 'Present' : 'Missing')
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    }
    
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    } else {
      console.warn('⚠️ No token found - screenshot will be sent without authentication')
    }

    const googleToken = await getValidGoogleAccessToken(API_URL)
    if (googleToken) {
      headers['X-Google-Access-Token'] = googleToken
      console.log('🔑 Google token attached for screenshot')
    } else {
      console.log('🔑 No Google token in storage – Connect Google in the popup to read Sheets/Docs')
    }

    const response = await fetch(`${API_URL}/api/embed-screenshot/`, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({
        source_url: url,
        captured_at: capturedAt,
        title: title || null,
        screenshot_data: base64Data
      })
    })
    
    if (!response.ok) {
      let errorMessage = 'Unknown error'
      try {
        const error = await response.json()
        errorMessage = (error as any).detail || (error as any).message || JSON.stringify(error)
      } catch (e) {
        errorMessage = `HTTP ${response.status}: ${response.statusText}`
      }
      console.error('Error sending screenshot to backend:', errorMessage)
      return
    }
    
    const result = await response.json()
    console.log('Screenshot sent to backend successfully:', result)
  } catch (error) {
    console.error('Error sending screenshot to backend:', error)
  }
}

// Helper function to check if URL is valid for screenshots
function isValidUrl(url?: string): boolean {
  if (!url) return false
  return !(
    url.startsWith('chrome://') ||
    url.startsWith('chrome-extension://') ||
    url.startsWith('about:') ||
    url.startsWith('devtools://') ||
    url.startsWith('edge://') ||
    url.startsWith('moz-extension://')
  )
}

// Track when a new tab becomes active
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  console.log('Tab activated:', activeInfo.tabId)
  const tabId = activeInfo.tabId
  
  try {
    const tab = await chrome.tabs.get(tabId)
    console.log('Tab info:', tab.url, 'isValid:', isValidUrl(tab.url))
    
    // Only capture if it's a valid web page
    if (isValidUrl(tab.url)) {
      console.log('✅ Scheduling screenshot for tab:', tabId)
      // Wait a bit for the page to render
      setTimeout(async () => {
        try {
          // Double-check the tab is still valid before capturing
          const currentTab = await chrome.tabs.get(tabId)
          if (!isValidUrl(currentTab.url)) {
            return // Skip if URL became invalid
          }
          
          // Capture the visible tab in the tab's window (current window when active)
          const dataUrl = await chrome.tabs.captureVisibleTab(currentTab.windowId, {
            format: 'png',
            quality: 100
          })
          
          console.log('Screenshot captured for tab:', tabId, tab.url)
          
          // Send screenshot to backend for embedding (don't store in chrome.storage to avoid quota issues)
          await sendScreenshotToBackend(dataUrl, tab.url || '', tab.title)
        } catch (error: any) {
          // Silently ignore errors for URLs we can't access
          if (error.message && error.message.includes('Cannot access contents')) {
            return
          }
          console.error('Error capturing screenshot:', error)
        }
      }, 500) // Wait 500ms for page to render
    }
    
  } catch (error: any) {
    // Silently ignore errors for tabs we can't access
    if (error.message && error.message.includes('Cannot access contents')) {
      return
    }
    console.error('Error getting tab info:', error)
  }
})

// Also capture when a tab finishes loading (for new tabs)
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  console.log('Tab updated:', tabId, 'status:', changeInfo.status, 'active:', tab.active, 'url:', tab.url)
  // Only capture when page is fully loaded and it's the active tab
  if (changeInfo.status === 'complete' && 
      tab.active &&
      isValidUrl(tab.url)) {
    console.log('✅ Conditions met for screenshot:', tabId, tab.url)
    
    try {
      // Small delay to ensure page is fully rendered
      setTimeout(async () => {
        try {
          // Double-check the tab is still valid before capturing
          const currentTab = await chrome.tabs.get(tabId)
          if (!isValidUrl(currentTab.url) || !currentTab.active) {
            return // Skip if URL became invalid or tab is no longer active
          }
          
          const dataUrl = await chrome.tabs.captureVisibleTab(currentTab.windowId, {
            format: 'png',
            quality: 100
          })
          
          console.log('Screenshot captured for loaded tab:', tabId, tab.url)
          
          // Send screenshot to backend for embedding (don't store in chrome.storage to avoid quota issues)
          await sendScreenshotToBackend(dataUrl, tab.url || '', tab.title)
        } catch (error: any) {
          // Silently ignore errors for URLs we can't access
          if (error.message && error.message.includes('Cannot access contents')) {
            return
          }
          console.error('Error capturing screenshot on update:', error)
        }
      }, 500)
    } catch (error: any) {
      // Silently ignore errors for tabs we can't access
      if (error.message && error.message.includes('Cannot access contents')) {
        return
      }
      console.error('Error in onUpdated handler:', error)
    }
  }
})
