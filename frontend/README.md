# Chrome Extension Frontend

A simple Chrome extension chatbot that connects to your LangChain backend.

## Setup

1. **Load the extension in Chrome:**
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode" (toggle in top right)
   - Click "Load unpacked"
   - Select the `frontend` folder

2. **Make sure your backend is running:**
   ```bash
   cd backend
   python3.12 -m uvicorn main:app --reload
   ```

3. **Use the extension:**
   - Click the extension icon in Chrome toolbar
   - The chatbot popup will open
   - Start chatting!

## Files

- `manifest.json` - Chrome extension configuration
- `popup.html` - The popup UI structure
- `popup.css` - Styling for the popup
- `popup.js` - JavaScript logic for chat functionality

## API Connection

The extension connects to `http://localhost:8000/chat` by default. Make sure:
- Your backend server is running on port 8000
- CORS is enabled (already configured in your FastAPI app)
