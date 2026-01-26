import { supabase } from '../lib/supabase'

/**
 * Get the current JWT access token from Supabase
 */
export async function getAccessToken() {
  const { data: { session } } = await supabase.auth.getSession()
  return session?.access_token || null
}

/**
 * Make an authenticated API call to the backend
 * Automatically includes the JWT token in the Authorization header
 */
export async function authenticatedFetch(url, options = {}) {
  const token = await getAccessToken()
  
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  }
  
  // Add Authorization header if token exists
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  
  const response = await fetch(url, {
    ...options,
    headers,
  })
  
  return response
}
