import { createContext, useContext, useEffect, useState } from 'react'
import { supabase } from '../lib/supabase'

const AuthContext = createContext({})

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [session, setSession] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(async ({ data: { session } }) => {
      setSession(session)
      setUser(session?.user ?? null)
      setLoading(false)
      
      // Store token in chrome.storage for background script access
      if (session?.access_token) {
        try {
          await chrome.storage.local.set({ 
            supabase_token: session.access_token,
            supabase_user_id: session.user.id
          })
          console.log('✅ Initial token stored in chrome.storage for user:', session.user.id)
        } catch (error) {
          // Ignore errors in non-extension contexts
          console.log('Could not store token in chrome.storage:', error)
        }
      }
    })

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (_event, session) => {
      setSession(session)
      setUser(session?.user ?? null)
      setLoading(false)
      
      // Store token in chrome.storage for background script access
      if (session?.access_token) {
        try {
          await chrome.storage.local.set({ 
            supabase_token: session.access_token,
            supabase_user_id: session.user.id
          })
          console.log('✅ Token stored in chrome.storage for user:', session.user.id)
        } catch (error) {
          // Ignore errors in non-extension contexts
          console.log('Could not store token in chrome.storage:', error)
        }
      } else {
        // Clear token on logout
        try {
          await chrome.storage.local.remove(['supabase_token', 'supabase_user_id'])
          console.log('🗑️ Token cleared from chrome.storage on logout')
        } catch (error) {
          // Ignore errors in non-extension contexts
          console.log('Could not clear token from chrome.storage:', error)
        }
      }
    })

    return () => subscription.unsubscribe()
  }, [])

  const signUp = async (email, password) => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
    })
    return { data, error }
  }

  const signIn = async (email, password) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    return { data, error }
  }

  const signOut = async () => {
    const { error } = await supabase.auth.signOut()
    return { error }
  }

  const value = {
    user,
    session,
    loading,
    signUp,
    signIn,
    signOut,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
