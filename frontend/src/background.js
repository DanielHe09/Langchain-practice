// Background script to handle tab screenshots
console.log('Background script loaded!');
let previousActiveTabId = null;
const API_URL = 'http://localhost:8000';

// Function to send screenshot to backend
async function sendScreenshotToBackend(dataUrl, url, title) {
  try {
    // Convert data URL to base64 (remove data:image/png;base64, prefix)
    const base64Data = dataUrl.split(',')[1];
    
    // Create ISO timestamp
    const capturedAt = new Date().toISOString();
    
    const response = await fetch(`${API_URL}/api/embed-screenshot/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        source_url: url,
        captured_at: capturedAt,
        title: title || null,
        screenshot_data: base64Data
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      console.error('Error sending screenshot to backend:', error);
      return;
    }
    
    const result = await response.json();
    console.log('Screenshot sent to backend successfully:', result);
  } catch (error) {
    console.error('Error sending screenshot to backend:', error);
  }
}

// Helper function to check if URL is valid for screenshots
function isValidUrl(url) {
  if (!url) return false;
  return !(
    url.startsWith('chrome://') ||
    url.startsWith('chrome-extension://') ||
    url.startsWith('about:') ||
    url.startsWith('devtools://') ||
    url.startsWith('edge://') ||
    url.startsWith('moz-extension://')
  );
}

// Track when a new tab becomes active
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  console.log('Tab activated:', activeInfo.tabId);
  const tabId = activeInfo.tabId;
  
  try {
    const tab = await chrome.tabs.get(tabId);
    console.log('Tab info:', tab.url, 'isValid:', isValidUrl(tab.url));
    
    // Only capture if it's a valid web page
    if (isValidUrl(tab.url)) {
      console.log('✅ Scheduling screenshot for tab:', tabId);
      // Wait a bit for the page to render
      setTimeout(async () => {
        try {
          // Double-check the tab is still valid before capturing
          const currentTab = await chrome.tabs.get(tabId);
          if (!isValidUrl(currentTab.url)) {
            return; // Skip if URL became invalid
          }
          
          // Capture the visible tab in the current window
          const dataUrl = await chrome.tabs.captureVisibleTab(null, {
            format: 'png',
            quality: 100
          });
          
          console.log('Screenshot captured for tab:', tabId, tab.url);
          
          // Store screenshot data in chrome.storage.local
          await chrome.storage.local.set({
            [`screenshot_${tabId}`]: {
              dataUrl: dataUrl,
              url: tab.url,
              title: tab.title,
              timestamp: Date.now(),
              tabId: tabId
            }
          });
          
          console.log('Screenshot stored for tab:', tabId);
          
          // Send screenshot to backend for embedding
          await sendScreenshotToBackend(dataUrl, tab.url, tab.title);
        } catch (error) {
          // Silently ignore errors for URLs we can't access
          if (error.message && error.message.includes('Cannot access contents')) {
            return;
          }
          console.error('Error capturing screenshot:', error);
        }
      }, 500); // Wait 500ms for page to render
    }
    
    previousActiveTabId = tabId;
  } catch (error) {
    // Silently ignore errors for tabs we can't access
    if (error.message && error.message.includes('Cannot access contents')) {
      return;
    }
    console.error('Error getting tab info:', error);
  }
});

// Also capture when a tab finishes loading (for new tabs)
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  console.log('Tab updated:', tabId, 'status:', changeInfo.status, 'active:', tab.active, 'url:', tab.url);
  // Only capture when page is fully loaded and it's the active tab
  if (changeInfo.status === 'complete' && 
      tab.active &&
      isValidUrl(tab.url)) {
    console.log('✅ Conditions met for screenshot:', tabId, tab.url);
    
    try {
      // Small delay to ensure page is fully rendered
      setTimeout(async () => {
        try {
          // Double-check the tab is still valid before capturing
          const currentTab = await chrome.tabs.get(tabId);
          if (!isValidUrl(currentTab.url) || !currentTab.active) {
            return; // Skip if URL became invalid or tab is no longer active
          }
          
          const dataUrl = await chrome.tabs.captureVisibleTab(null, {
            format: 'png',
            quality: 100
          });
          
          console.log('Screenshot captured for loaded tab:', tabId, tab.url);
          
          await chrome.storage.local.set({
            [`screenshot_${tabId}`]: {
              dataUrl: dataUrl,
              url: tab.url,
              title: tab.title,
              timestamp: Date.now(),
              tabId: tabId
            }
          });
          
          console.log('Screenshot stored for loaded tab:', tabId);
          
          // Send screenshot to backend for embedding
          await sendScreenshotToBackend(dataUrl, tab.url, tab.title);
        } catch (error) {
          // Silently ignore errors for URLs we can't access
          if (error.message && error.message.includes('Cannot access contents')) {
            return;
          }
          console.error('Error capturing screenshot on update:', error);
        }
      }, 500);
    } catch (error) {
      // Silently ignore errors for tabs we can't access
      if (error.message && error.message.includes('Cannot access contents')) {
        return;
      }
      console.error('Error in onUpdated handler:', error);
    }
  }
});
