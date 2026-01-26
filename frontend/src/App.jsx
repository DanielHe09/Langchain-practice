import { useState, useRef, useEffect } from 'react'
import { useAuth } from './contexts/AuthContext'
import { authenticatedFetch } from './utils/api'
import Login from './components/Login'
import SignUp from './components/SignUp'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const { user, loading, signOut } = useAuth()
  const [showSignUp, setShowSignUp] = useState(false)

  // Show loading state while checking auth
  if (loading) {
    return (
      <div className="app">
        <div className="container">
          <div className="loading-container">
            <div className="spinner"></div>
            <p>Loading...</p>
          </div>
        </div>
      </div>
    )
  }

  // Show auth pages if not logged in
  if (!user) {
    return showSignUp ? (
      <SignUp onSwitchToLogin={() => setShowSignUp(false)} />
    ) : (
      <Login onSwitchToSignUp={() => setShowSignUp(true)} />
    )
  }

  // Show chatbot if logged in
  return <Chatbot signOut={signOut} />
}

function Chatbot({ signOut }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! How can I help you today?' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const sendMessage = async () => {
    const message = input.trim()
    if (!message || loading) return

    setInput('')
    setLoading(true)

    // Add user message
    const newUserMessage = { role: 'user', content: message }
    setMessages(prev => [...prev, newUserMessage])

    try {
      // Prepare conversation history (excluding the welcome message)
      const conversationHistory = messages
        .filter(msg => msg.role !== 'assistant' || msg.content !== 'Hello! How can I help you today?')
        .map(msg => ({
          role: msg.role,
          content: msg.content
        }))

      const response = await authenticatedFetch(`${API_URL}/chat`, {
        method: 'POST',
        body: JSON.stringify({
          message: message,
          conversation_history: conversationHistory
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      // Add assistant response
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }])
    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Error: ${error.message}. Make sure the backend is running on ${API_URL}` 
      }])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleSignOut = async () => {
    await signOut()
  }

  return (
    <div className="app">
      <div className="container">
        <div className="header">
          <h2>🤖 Chatbot</h2>
          <button onClick={handleSignOut} className="logout-button">
            Sign out
          </button>
        </div>
        
        <div className="messages" id="messages">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              <div className="message-content">
                {message.content}
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="message assistant">
              <div className="message-content loading">
                Thinking<span className="dots">...</span>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            disabled={loading}
            className="input-field"
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="send-button"
          >
            {loading ? '⏳' : '📤'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default App
