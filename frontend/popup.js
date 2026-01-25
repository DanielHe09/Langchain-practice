const API_URL = 'http://localhost:8000';
const messagesContainer = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');

let conversationHistory = [];

function addMessage(role, content) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${role}`;
  
  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.textContent = content;
  
  messageDiv.appendChild(contentDiv);
  messagesContainer.appendChild(messageDiv);
  
  // Scroll to bottom
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  
  return messageDiv;
}

function showLoading() {
  const loadingDiv = document.createElement('div');
  loadingDiv.className = 'message assistant';
  loadingDiv.id = 'loading-message';
  
  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content loading';
  contentDiv.textContent = 'Thinking';
  
  loadingDiv.appendChild(contentDiv);
  messagesContainer.appendChild(loadingDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeLoading() {
  const loadingMessage = document.getElementById('loading-message');
  if (loadingMessage) {
    loadingMessage.remove();
  }
}

async function sendMessage() {
  const message = messageInput.value.trim();
  if (!message) return;
  
  // Disable input
  messageInput.disabled = true;
  sendButton.disabled = true;
  
  // Add user message to UI
  addMessage('user', message);
  messageInput.value = '';
  
  // Show loading indicator
  showLoading();
  
  try {
    // Prepare conversation history for API
    const history = conversationHistory.map(msg => ({
      role: msg.role,
      content: msg.content
    }));
    
    const response = await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: message,
        conversation_history: history
      })
    });
    
    removeLoading();
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Add assistant response to UI
    addMessage('assistant', data.response);
    
    // Update conversation history
    conversationHistory.push({ role: 'user', content: message });
    conversationHistory.push({ role: 'assistant', content: data.response });
    
  } catch (error) {
    removeLoading();
    console.error('Error:', error);
    addMessage('assistant', `Error: ${error.message}. Make sure the backend is running on ${API_URL}`);
  } finally {
    // Re-enable input
    messageInput.disabled = false;
    sendButton.disabled = false;
    messageInput.focus();
  }
}

// Event listeners
sendButton.addEventListener('click', sendMessage);

messageInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    sendMessage();
  }
});

// Focus input on load
messageInput.focus();
