# Chrome Extension Frontend (React)

A React-based Chrome extension chatbot that connects to your LangChain backend.

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Build the extension:**
   ```bash
   npm run build
   ```

3. **Load the extension in Chrome:**
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode" (toggle in top right)
   - Click "Load unpacked"
   - Select the `frontend/dist` folder (NOT the frontend folder)

4. **Make sure your backend is running:**
   ```bash
   cd backend
   python3.12 -m uvicorn main:app --reload
   ```

5. **Use the extension:**
   - Click the extension icon in Chrome toolbar
   - The chatbot popup will open
   - Start chatting!

## Development

For development with hot reload:
```bash
npm run dev
```

Then load the extension from the `dist` folder (Vite will rebuild on changes).

## Files

- `manifest.json` - Chrome extension configuration
- `src/App.jsx` - Main React component
- `src/popup.jsx` - React entry point
- `src/App.css` - Component styles
- `vite.config.js` - Build configuration

## API Connection

The extension connects to `http://localhost:8000/chat` by default. Make sure:
- Your backend server is running on port 8000
- CORS is enabled (already configured in your FastAPI app)
