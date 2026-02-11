import React, { useState, useEffect, useRef } from "react";
import "./SamAssistant.css";
import { FaPaperPlane, FaPlus, FaStop } from "react-icons/fa";
import { AiFillFilePdf, AiFillFileImage, AiFillFileUnknown } from "react-icons/ai";

const recognition = window.SpeechRecognition || window.webkitSpeechRecognition
  ? new (window.SpeechRecognition || window.webkitSpeechRecognition)()
  : null;

function speak(text) {
  if ('speechSynthesis' in window) {
    const utterance = new window.SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  }
}

function SamAssistant() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [files, setFiles] = useState([]); // array of files
  const [uploading, setUploading] = useState(false);
  const [mode, setMode] = useState("auto"); // 'auto', 'web', 'docs'
  const [listening, setListening] = useState(false);
  const fileInputRef = useRef();
  const chatWindowRef = useRef();
  // Add state for AbortController
  const [controller, setController] = useState(null);
  const recognitionRef = useRef(null);
  const [showPreview, setShowPreview] = useState({ open: false, file: null });
  const [dragActive, setDragActive] = useState(false);
  // Add this state to track if greeting has been shown
  const [greetingShown, setGreetingShown] = useState(false);

  // Clean up object URLs for file previews
  useEffect(() => {
    let url;
    if (showPreview.file) {
      url = URL.createObjectURL(showPreview.file);
    }
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [showPreview.file]);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages, loading, uploading]);

  // Speech recognition setup
  useEffect(() => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) return;
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US'; // Set language as needed
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognitionRef.current = recognition;

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
      setListening(false);
    };
    recognition.onerror = (event) => {
      setListening(false);
      // Optionally, show error to user
      // alert('Speech recognition error: ' + event.error);
    };
    recognition.onend = () => {
      setListening(false);
    };
    // Clean up on unmount
    return () => {
      recognition.abort();
    };
  }, []);

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

  // File type icon helper
  const getFileIcon = (file) => {
    if (!file) return <AiFillFileUnknown />;
    if (file.type.startsWith("image/")) return <AiFillFileImage color="#38bdf8" />;
    if (file.type === "application/pdf" || file.name.endsWith(".pdf")) return <AiFillFilePdf color="#f87171" />;
    return <AiFillFileUnknown />;
  };

  // File preview modal
  const renderFilePreviewModal = () => {
    if (!showPreview.open || !showPreview.file) return null;
    const file = showPreview.file;
    return (
      <div className="file-preview-modal" onClick={() => setShowPreview({ open: false, file: null })}>
        <div className="file-preview-content" onClick={e => e.stopPropagation()}>
          <button className="file-preview-close" onClick={() => setShowPreview({ open: false, file: null })}>&times;</button>
          {file.type.startsWith("image/") ? (
            <img src={URL.createObjectURL(file)} alt={file.name} style={{ maxWidth: '100%', maxHeight: 400, borderRadius: 12 }} />
          ) : file.type === "application/pdf" || file.name.endsWith(".pdf") ? (
            <iframe src={URL.createObjectURL(file)} title={file.name} width="100%" height="400px" style={{ border: 'none', borderRadius: 12 }} />
          ) : (
            <div style={{ padding: 24, textAlign: 'center' }}>No preview available</div>
          )}
        </div>
      </div>
    );
  };

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };
  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFilesUpload(e.dataTransfer.files);
    }
  };

  // Handle file input change (multiple files)
  const handleFilesUpload = (fileList) => {
    let newFiles = Array.from(fileList);
    // Prevent duplicates by name
    newFiles = newFiles.filter(f => !files.some(existing => existing.name === f.name));
    setFiles(prev => [...prev, ...newFiles]);
  };

  // Streaming sendMessage
  const sendMessage = async () => {
    if (!input.trim() && files.length === 0) return;
    const messageToSend = input.trim() || (files.length > 0 ? files.map(f => f.name).join(", ") : "");
    const userMessage = { sender: "user", text: messageToSend, files: files.length > 0 ? files : undefined };
    setMessages((msgs) => [...msgs, userMessage]);
    setLoading(true);
    setUploading(files.length > 0);
    setInput("");

    const abortController = new AbortController();
    setController(abortController);

    try {
      let res;
      let aiText = "";

      if (files.length > 0) {
        const formData = new FormData();
        formData.append("message", messageToSend);
        files.forEach(f => formData.append("file", f));
        formData.append("mode", mode);
        res = await fetch("http://localhost:8000/chat/stream", {
          method: "POST",
          body: formData,
          signal: abortController.signal,
        });
        setFiles([]);
      } else {
        res = await fetch("http://localhost:8000/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: messageToSend, mode }),
          signal: abortController.signal,
        });
      }

      if (!res.body || !window.ReadableStream) {
        // fallback to old method if streaming not supported
        const data = await res.json();
        setMessages((msgs) => [
          ...msgs,
          { sender: "ai", text: data.response },
        ]);
        setResponse(data.response || "");
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
            setResponse(aiText); // Update response for TTS
          }
        }
      }
    } catch (err) {
      let errorMsg = "Error: Could not connect to AI backend.";
      if (err.name === 'AbortError') {
        // Do not show any message if stopped by user
        setLoading(false);
        setUploading(false);
        setController(null);
        return;
      } else if (err.message) {
        errorMsg = `Error: ${err.message}`;
      }
      setMessages((msgs) => [
        ...msgs,
        { sender: "ai", text: errorMsg },
      ]);
      setResponse(errorMsg);
    }
    setLoading(false);
    setUploading(false);
    setController(null);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  function startListening() {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    if (listening) return; // Prevent double start
    setListening(true);
    try {
      recognition.abort(); // Stop any ongoing recognition
      recognition.start();
    } catch (e) {
      setListening(false);
      // Optionally, show error to user
      // alert('Could not start speech recognition: ' + e.message);
    }
  }

  // Add stopResponse function
  function stopResponse() {
    if (controller) {
      controller.abort();
      setLoading(false);
      setController(null);
    }
  }

  // Placeholder for backend file deletion
  const deleteFileFromServer = async (file) => {
    // TODO: Implement API call to delete file from server
    // Example: await fetch(`/api/files/${file.id}`, { method: 'DELETE' });
    // For now, just a placeholder
    return true;
  };

  // In the effect that handles opening the chatbox, add:
  useEffect(() => {
    if (open && !greetingShown) {
      setMessages((prev) => [
        ...prev,
        {
          id: "greeting",
          sender: "ai",
          text: (
            <span>
              <span className="waving-hand" role="img" aria-label="wave">👋</span>
              Hi there! How can I help you today?
            </span>
          ),
        },
      ]);
      setGreetingShown(true);
    }
  }, [open, greetingShown, setMessages]);

  return (
    <div className={`sam-assistant-floating-container glass-bg${dragActive ? ' drag-active' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {!open && (
        <button className="sam-assistant-float-btn glass-bg" onClick={() => setOpen(true)}>
          💬 Chat
        </button>
      )}
      {open && (
        <div className="sam-assistant-chatbox glass-bg larger">
          <div className="sam-assistant-header glass-bg">
            {/* Company logo placeholder */}
            <span
              className="company-logo"
              aria-label="Company Logo"
              title="Your Company"
              style={{ backgroundImage: `url(${process.env.PUBLIC_URL + '/logo192.png'})` }}
            />
            <span>Sam Assistant</span>
            <button className="sam-assistant-close-btn" onClick={() => setOpen(false)} aria-label="Close chat">
              ×
            </button>
          </div>
          <div className="chat-window glass-bg" ref={chatWindowRef}>
            {messages.map((msg, idx) => (
              <div key={idx} className={`message-bubble glass-bg ${msg.sender}`}> 
                {msg.text && <span>{msg.text}</span>}
                {msg.files && Array.isArray(msg.files) && msg.files.map((f, i) => renderFilePreview(f))}
                {msg.fileName && msg.fileObj && renderFilePreview(msg.fileObj)}
              </div>
            ))}
            {loading && (
              <div className="message-bubble ai glass-bg thinking-bubble">
                <span className="thinking-text">Thinking...</span>
              </div>
            )}
            {uploading && files.length > 0 && (
              <div className="message-bubble user uploading glass-bg">
                <span>Uploading {files.map(f => f.name).join(", ")}...</span>
                <div className="upload-progress-bar">
                  <div className="upload-progress-spinner" />
                </div>
              </div>
            )}
          </div>
          <div className="sam-assistant-input-bar glass-bg">
            <button
              className="icon-btn"
              onClick={() => fileInputRef.current && fileInputRef.current.click()}
              disabled={loading || uploading}
              title="Upload file"
              aria-label="Upload file"
            >
              <FaPlus />
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => handleFilesUpload(e.target.files)}
              disabled={loading || uploading}
              style={{ display: 'none' }}
            />
            {/* File chips inside input bar */}
            <div className="file-chips-row-inside" tabIndex={0} aria-label="Uploaded files">
              {files.map((file, idx) => (
                <div className="file-chip" key={file.name + idx} onClick={() => setShowPreview({ open: true, file })} title={`Preview ${file.name}`} tabIndex={0} aria-label={`Preview ${file.name}`}>
                  <span className="file-chip-icon">{getFileIcon(file)}</span>
                  <span className="file-chip-name">{file.name}</span>
                  <button
                    className="file-chip-delete"
                    onClick={async e => {
                      e.stopPropagation();
                      await deleteFileFromServer(file);
                      setFiles(files.filter((_, i) => i !== idx));
                    }}
                    title="Remove file"
                    aria-label={`Remove ${file.name}`}
                    type="button"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
            {/* Context selector bubble */}
            <div className="context-bubble">
              <select
                id="context-mode-select"
                value={mode}
                onChange={e => setMode(e.target.value)}
                className="context-select"
                aria-label="Context mode"
              >
                <option value="auto">Auto</option>
                <option value="web">Web</option>
                <option value="docs">Docs</option>
              </select>
            </div>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything..."
              disabled={loading || uploading}
              className="glass-input"
              aria-label="Type your message"
            />
            <button onClick={startListening} disabled={listening} className="icon-btn" title="Voice input" aria-label="Voice input">
              {listening ? <span className="listening-dot" /> : "🎤"}
            </button>
            {loading ? (
              <button onClick={stopResponse} className="icon-btn stop-btn" title="Stop response" aria-label="Stop response">
                <FaStop />
              </button>
            ) : (
              <button 
                onClick={sendMessage} 
                disabled={loading || uploading} 
                className="icon-btn send-btn"
                aria-label="Send message"
              >
                <FaPaperPlane />
              </button>
            )}
          </div>
          {showPreview.open && renderFilePreviewModal()}
        </div>
      )}
    </div>
  );
}

export default SamAssistant;
