# Sam Assistant: The Super Agent 🚀

A sophisticated **Agentic AI System** that thinks, plans, sees, and acts. It combines System 2 Reasoning, Multimodal Vision, and Autonomous Tools with a premium "ChatGPT-style" UI.

![SAm Assistant UI](./frontend/public/logo.svg)

---

## 🔥 Key Features

### 🧠 Agentic Brain (System 2)
Unlike standard chatbots, Sam doesn't just guess. It **Plans**:
1.  **Thinking Phase**: Breaks down complex queries into steps.
2.  **Tool Selection**: Decides *during runtime* which tool to use.
3.  **Self-Correction**: Reviews and polishes its own answers before showing them.

### 🛠️ Powerful Tools
-   **👁️ Vision**: Can see and analyze images (via `llava`).
-   **🌐 Browser**: Autonomous headless browser to read real websites.
-   **🐍 Code Runner**: Python REPL for complex logic and math.
-   **🔢 Calculator**: Precise mathematical operations.
-   **🔍 Deep Research**: Investigates topics by breaking them into sub-questions.

### 💾 Persistent Memory
-   **Conversation History**: Saved endlessly in **SQLite/Postgres** (`chat_history.db`).
-   **Knowledge Base**: Uploaded documents are saved in **ChromaDB** with **Strict Privacy Isolation**.
-   **Hybrid Mode**: Works 100% Offline or Online.

### 💻 Modern UI
-   **Tech**: React + TailwindCSS + Lucide Icons.
-   **Design**: Premium Dark Mode aesthetics (ChatGPT 5.2 Clone).
-   **UX**: Smooth "Thinking" animations, Markdown support, and Code highlighting.

---

## 🚀 Quick Start

### 1. Prerequisites
-   **Python 3.10+**
-   **Node.js 18+**
-   **Ollama**: [Download Here](https://ollama.com) (Required for LLM)

### 2. Backend Setup
```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Pull AI Models
ollama pull mistral  # Main Brain
ollama pull llava    # Vision

# 3. Start Server
python backend/main.py
```
> Server runs at `http://localhost:8000`

### 3. Frontend Setup
```bash
cd frontend

# 1. Install Dependencies
npm install

# 2. Start UI
npm start
```
> App runs at `http://localhost:3000`

---

## ⚙️ Configuration (`backend/.env`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OLLAMA_MODEL` | `mistral` | The LLM model to use (mistral, llama3, etc). |
| `DATABASE_URL` | `sqlite:///./chat_history.db` | Connection string (supports PostgreSQL). |
| `OFFLINE_MODE` | `false` | Set `true` to disable all internet tools. |

---

## 🛡️ Architecture
For a deep dive into how the "Thinking Loop" works, check out the `backend/ai_orchestrator.py` file.

**Flow**:
`Input` -> `Memory Check` -> `Plan` -> `Tool Execution (Vision/Web/Code)` -> `Reflection` -> `Response`
