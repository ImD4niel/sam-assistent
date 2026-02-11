import React, { useState, useEffect } from 'react';
import Sidebar from './Sidebar';
import ChatInterface from './ChatInterface';
import { Menu } from 'lucide-react';
import './index.css';

const API_Base = "http://localhost:8000";

function App() {
  const [isSidebarOpen, setSidebarOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isThinking, setThinking] = useState(false);
  const [history, setHistory] = useState([]);
  const [conversationId, setConversationId] = useState(null);

  // Load history on mount
  useEffect(() => {
    // Mock history for now, or fetch from backend if endpoint exists
    setHistory([
       { id: "1", title: "Project Brainstorming" },
       { id: "2", title: "Python Help" }
    ]);
  }, []);

  const handleSendMessage = async (text, file = null) => {
    // Add User Message
    const userMsg = { role: 'user', content: text };
    if (file) {
        // Create object URL for preview
        userMsg.image = URL.createObjectURL(file);
    }
    setMessages(prev => [...prev, userMsg]);
    setThinking(true);

    try {
        const formData = new FormData();
        formData.append("message", text);
        if (conversationId) formData.append("conversation_id", conversationId);
        if (file) formData.append("file", file);

        const response = await fetch(`${API_Base}/chat`, {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
            // Save conversation ID if new
            if (!conversationId) {
                // In a real app, backend returns the ID
            }
        } else {
            setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${data.response || "Failed to fetch"}` }]);
        }

    } catch (error) {
        setMessages(prev => [...prev, { role: 'assistant', content: "Error: Could not connect to server." }]);
    } finally {
        setThinking(false);
    }
  };

  const handleFileUpload = (e) => {
      const file = e.target.files[0];
      if (file) {
          handleSendMessage(`[Attached file: ${file.name}]`, file);
      }
  };

  return (
    <div className="flex bg-chatgpt-main h-screen text-white overflow-hidden font-sans">
      
      {/* Mobile Menu Button */}
      <button 
        className="md:hidden fixed top-4 left-4 z-50 p-2 bg-gray-800 rounded-md"
        onClick={() => setSidebarOpen(!isSidebarOpen)}
      >
        <Menu size={20} />
      </button>

      {/* Sidebar */}
      <Sidebar 
         isOpen={isSidebarOpen} 
         onNewChat={() => { setMessages([]); setConversationId(null); setSidebarOpen(false); }}
         history={history}
         loadChat={(id) => { console.log("Load chat", id); }}
      />

      {/* Main Chat */}
      <ChatInterface 
        messages={messages}
        isThinking={isThinking}
        onSendMessage={(text) => handleSendMessage(text)}
        onFileUpload={handleFileUpload}
      />
    </div>
  );
}

export default App;