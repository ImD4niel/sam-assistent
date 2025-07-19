# Sam Assistant Chatbot

A plug-and-play AI chatbot using FastAPI (Python) for the backend and React for the frontend. The backend uses Ollama for open-source LLMs (like Llama 2, Mistral). The frontend is a floating chat widget, ready for integration into any web app or ERPNext.

---

## Features
- Free, open-source LLM backend (Ollama)
- Modern floating chatbot widget (React)
- **Real-time streaming responses** (see your answer appear word-by-word)
- File and image upload with OCR and document extraction
- RAG (Retrieval-Augmented Generation) with ChromaDB
- **Web search integration (DuckDuckGo)** - Get real-time information from the web
- Easy REST API integration
- Ready for ERPNext or any web app

---

## 1. Backend Setup (FastAPI + Ollama)

### 1.1. Install Python dependencies
```sh
pip install -r requirements.txt
```

### 1.2. Install and Run Ollama
- Download and install from: https://ollama.com/download
- Start Ollama:
  ```sh
  ollama serve
  ```
- Pull and run a model (e.g., Mistral, Phi-3, or Llama 2):
  ```sh
  ollama pull mistral
  ollama run mistral
  ```
  Or for Phi-3:
  ```sh
  ollama pull phi3
  ollama run phi3
  ```
  Or for Llama 2 (requires more RAM):
  ```sh
  ollama pull llama2
  ollama run llama2
  ```

### 1.3. Start the FastAPI backend
```sh
python backend/main.py
```
Or (alternative):
```sh
uvicorn backend.main:app --reload
```
- The backend will run at `http://localhost:8000`
- The backend uses the model specified by the environment variable `OLLAMA_MODEL` (default: `mistral`).
- If you get a memory error (e.g., "model requires more system memory than is available"), use a smaller model like `mistral` or `phi3`.
- To change the model, update the `OLLAMA_MODEL` in your `.env` file or in `backend/main.py`.

### 1.4. **Streaming Chat Endpoint**
- **New!** Use `/chat/stream` for real-time, word-by-word streaming responses.
- Accepts both JSON and form-data (with or without file upload).
- Example (with curl):
  ```sh
  curl -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d '{"message": "Hello!"}'
  ```
- The system prompt is prepended to the user message for best LLM context.

### 1.5. **Web Search Integration**
- **DuckDuckGo Search**: The chatbot automatically uses DuckDuckGo for web searches when needed
- **No API Key Required**: DuckDuckGo is free and privacy-focused
- **Smart Detection**: Automatically triggers web search for queries containing keywords like:
  - News, latest, current events
  - Weather, stock prices, crypto
  - Sports, entertainment, technology
  - Real-time information requests
- **Dual Search**: Combines general web search and news search for comprehensive results
- **Manual Mode**: You can force web search by setting `mode: 'web'` in your request

---

## 2. Frontend Setup (React Floating Chatbot)

### 2.1. Create the React app (if not already present)
```sh
npx create-react-app frontend
```

### 2.2. Move widget files
- Place `SamAssistant.jsx` and `SamAssistant.css` into `frontend/src/`

### 2.3. Use the widget in your React app
Edit `frontend/src/App.js`:
```js
import React from 'react';
import SamAssistant from './SamAssistant';
import './SamAssistant.css';

function App() {
  return (
    <div className="App">
      {/* Your other app content */}
      <SamAssistant />
    </div>
  );
}

export default App;
```

### 2.4. Install frontend dependencies
```sh
cd frontend
npm install
```

### 2.5. Start the React app
```sh
npm start
```
- The frontend will run at `http://localhost:3000`
- You should see a floating chat widget at the bottom right
- **Now supports real-time streaming answers!**

---

## 3. GitHub Setup (Optional, for sharing your project)

### 3.1. Initialize git (if not already done)
```sh
git init
```

### 3.2. Add and commit your files
```sh
git add .
git commit -m "Initial commit"
```

### 3.3. Remove node_modules from git (if accidentally added)
```sh
git rm -r --cached frontend/node_modules
git commit -m "Remove node_modules from repository"
```

### 3.4. Add .gitignore (should already exist, but ensure it contains):
```
node_modules/
```

### 3.5. Create a new repo on GitHub and push
```sh
git remote add origin https://github.com/YOUR-USERNAME/sam-assistent.git
git branch -M main
git push -u origin main
```

---

## 4. Usage
- Open `http://localhost:3000` in your browser.
- Click the floating chat button, type a message, and interact with the AI.
- The backend must be running and Ollama must be serving a model for the chatbot to work.
- **Web Search Examples**:
  - "What's the latest news about AI?"
  - "What's the weather in New York?"
  - "Tell me about the latest iPhone release"
  - "What are the current Bitcoin prices?"
- **Enjoy real-time, streaming answers with web search capabilities!**

---

## 5. ERPNext or Web Integration
- You can embed the React widget in any web app.
- For ERPNext, use an iframe or custom app to include the widget at the bottom right.

---

## Troubleshooting
- If the chatbot says "couldn't connect to AI backend":
  - Ensure the backend is running (`python backend/main.py`)
  - Ensure Ollama is running and the model is available (`ollama run mistral` or `ollama run phi3`)
  - If you see a memory error, switch to a smaller model (see above)
  - Check browser console/network tab for errors
- If streaming does not work:
  - Make sure you are using the `/chat/stream` endpoint in both backend and frontend
  - Make sure your browser supports streaming responses (most modern browsers do)
  - Check backend logs for errors
- If web search is not working:
  - Check your internet connection
  - The `ddgs` library should be installed (included in requirements.txt)
  - Check backend logs for any DuckDuckGo API errors
- Do **not** commit `node_modules` to git

---

**Enjoy your AI chatbot with real-time streaming and web search capabilities!**
