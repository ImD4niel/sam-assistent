import React, { useState, useEffect, useRef } from "react";
import "./SamAssistant.css";
import { FaPaperPlane, FaPlus } from "react-icons/fa";

function SamAssistant() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [mode, setMode] = useState("auto"); // 'auto', 'web', 'docs'
  const fileInputRef = useRef();
  const chatWindowRef = useRef();

  // Clean up object URLs for file previews
  useEffect(() => {
    let url;
    if (file) {
      url = URL.createObjectURL(file);
    }
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [file]);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages, loading, uploading]);

  const renderFilePreview = (file) => {
    if (!file) return null;
    const url = URL.createObjectURL(file);
    if (file.type && file.type.startsWith("image/")) {
      return <img src={url} alt={file.name} className="bubble-file-img" />;
    }
    return (
      <a href={url} download={file.name} className="bubble-file-link">{file.name}</a>
    );
  };

  // Streaming sendMessage
  const sendMessage = async () => {
    if (!input.trim() && !file) return;
    const messageToSend = input.trim() || (file ? file.name : "");
    const userMessage = { sender: "user", text: messageToSend, fileName: file ? file.name : undefined, fileObj: file || null };
    setMessages((msgs) => [...msgs, userMessage]);
    setLoading(true);
    setUploading(false);
    setInput("");
    
    try {
      let res;
      let aiText = "";
      
      if (file) {
        setUploading(true);
        const formData = new FormData();
        formData.append("message", messageToSend);
        formData.append("file", file);
        formData.append("mode", mode);
        res = await fetch("http://localhost:8000/chat/stream", {
          method: "POST",
          body: formData,
        });
        setFile(null);
        setUploading(false);
      } else {
        res = await fetch("http://localhost:8000/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: messageToSend, mode }),
        });
      }
      
      if (!res.body || !window.ReadableStream) {
        // fallback to old method if streaming not supported
        const data = await res.json();
        setMessages((msgs) => [
          ...msgs,
          { sender: "ai", text: data.response },
        ]);
      } else {
        // Streaming response
        const reader = res.body.getReader();
        let decoder = new TextDecoder();
        let done = false;
        setMessages((msgs) => [
          ...msgs,
          { sender: "ai", text: "" },
        ]);
        let aiText = "";
        while (!done) {
          const { value, done: doneReading } = await reader.read();
          done = doneReading;
          if (value) {
            aiText += decoder.decode(value, { stream: true });
            setMessages((msgs) => {
              const updated = [...msgs];
              // Update the last AI message
              updated[updated.length - 1] = { sender: "ai", text: aiText };
              return updated;
            });
          }
        }
      }
    } catch (err) {
      setMessages((msgs) => [
        ...msgs,
        { sender: "ai", text: "Error: Could not connect to AI backend." },
      ]);
    }
    setLoading(false);
    setUploading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div className="sam-assistant-floating-container">
      {!open && (
        <button className="sam-assistant-float-btn" onClick={() => setOpen(true)}>
          💬 Chat
        </button>
      )}
      {open && (
        <div className="sam-assistant-chatbox larger">
          <div className="sam-assistant-header">
            <span>Sam Assistant</span>
            <button className="sam-assistant-close-btn" onClick={() => setOpen(false)}>
              ×
            </button>
          </div>
          
          {/* Context mode dropdown */}
          <div style={{ padding: '8px 16px', background: '#f3f4f6', borderBottom: '1px solid #e5e7eb' }}>
            <label htmlFor="context-mode-select" style={{ marginRight: 8, fontWeight: 500 }}>Context:</label>
            <select
              id="context-mode-select"
              value={mode}
              onChange={e => setMode(e.target.value)}
              style={{ fontSize: 15, padding: '4px 8px', borderRadius: 4 }}
            >
              <option value="auto">Auto (Smart)</option>
              <option value="web">Web Only</option>
              <option value="docs">Docs Only</option>
            </select>
          </div>
          
          <div className="chat-window" ref={chatWindowRef}>
            {messages.map((msg, idx) => (
              <div key={idx} className={`message-bubble ${msg.sender}`}> 
                {msg.text && <span>{msg.text}</span>}
                {msg.fileName && msg.fileObj && renderFilePreview(msg.fileObj)}
              </div>
            ))}
            {loading && <div className="message-bubble ai">Thinking...</div>}
            {uploading && file && (
              <div className="message-bubble user uploading">
                <span>Uploading {file.name}...</span>
                {renderFilePreview(file)}
              </div>
            )}
          </div>
          
          <div className="input-bar prominent">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message..."
              disabled={loading || uploading}
              style={{ fontSize: 18, padding: '12px 16px', flex: 1 }}
            />
            <button
              className="file-upload-btn"
              onClick={() => fileInputRef.current && fileInputRef.current.click()}
              disabled={loading || uploading}
              style={{ marginLeft: 8, fontSize: 22, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'none', border: 'none', cursor: 'pointer' }}
              title="Upload file"
              aria-label="Upload file"
            >
              <FaPlus color="#2563eb" />
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => setFile(e.target.files[0])}
              disabled={loading || uploading}
              style={{ display: 'none' }}
            />
            <button 
              onClick={sendMessage} 
              disabled={loading || uploading} 
              style={{ fontSize: 20, padding: '10px 16px', marginLeft: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#2563eb', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer' }}
              aria-label="Send message"
            >
              <FaPaperPlane />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default SamAssistant;
