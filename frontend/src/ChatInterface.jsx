import React, { useState, useEffect, useRef } from 'react';
import { Send, Paperclip } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { motion } from 'framer-motion';

const ChatInterface = ({ messages, isThinking, onSendMessage, onFileUpload }) => {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages, isThinking]);


  
  const handleKeyDown = (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          if (input.trim()) {
            onSendMessage(input);
            setInput("");
          }
      }
  };

  return (
    <div className="flex-1 flex flex-col bg-chatgpt-main relative h-screen md:pl-64">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto scroll-smooth">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-white px-4">
            <div className="bg-white/10 p-4 rounded-full mb-6">
                <img src="/logo.svg" className="w-12 h-12 invert" alt="Logo" />
            </div>
            <h2 className="text-2xl font-semibold mb-2">Where should we begin?</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-w-2xl w-full mt-8">
                {["Plan a trip", "Write python code", "Research a topic", "Analyze an image"].map((suggestion) => (
                    <button 
                        key={suggestion}
                        onClick={() => { setInput(suggestion); }}
                        className="p-3 border border-gray-600 rounded-lg text-sm text-gray-300 hover:bg-gray-700 text-left"
                    >
                        {suggestion}
                    </button>
                ))}
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-6 py-6 pb-32 max-w-3xl mx-auto px-4 w-full">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`
                    w-8 h-8 rounded-sm flex items-center justify-center shrink-0
                    ${msg.role === 'assistant' ? 'bg-green-500' : 'bg-purple-600'}
                `}>
                    {msg.role === 'assistant' ? 'AI' : 'U'}
                </div>
                <div className={`
                    prose prose-invert max-w-none rounded-lg p-0
                    ${msg.role === 'user' ? 'bg-transparent text-right' : 'w-full'}
                `}>
                    {/* Render Image if exists */}
                    {msg.image && (
                        <img src={msg.image} alt="User upload" className="max-w-xs rounded-lg mb-2 border border-gray-700"/>
                    )}
                    
                    {msg.role === 'user' ? (
                        <div className="bg-[#2F2F2F] px-4 py-2 rounded-2xl inline-block text-left">
                            {msg.content}
                        </div>
                    ) : (
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                    )}
                </div>
              </div>
            ))}
            
            {/* Thinking Indicator */}
            {isThinking && (
                 <div className="flex gap-4">
                    <div className="w-8 h-8 bg-green-500 rounded-sm flex items-center justify-center shrink-0">AI</div>
                    <div className="flex items-center gap-1">
                        <motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ duration: 1.5, repeat: Infinity }} className="w-2 h-2 bg-gray-400 rounded-full" />
                        <motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ duration: 1.5, repeat: Infinity, delay: 0.2 }} className="w-2 h-2 bg-gray-400 rounded-full" />
                        <motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ duration: 1.5, repeat: Infinity, delay: 0.4 }} className="w-2 h-2 bg-gray-400 rounded-full" />
                    </div>
                 </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="absolute bottom-0 left-0 right-0 w-full bg-gradient-to-t from-chatgpt-main via-chatgpt-main to-transparent pt-10 pb-6 md:pl-64">
        <div className="max-w-3xl mx-auto px-4">
            <div className="relative flex items-end gap-2 bg-chatgpt-input rounded-xl border border-gray-600 p-2 shadow-lg">
                {/* File Upload */}
                <button 
                    onClick={() => fileInputRef.current?.click()}
                    className="p-2 text-gray-400 hover:text-white transition-colors"
                >
                    <Paperclip size={20} />
                </button>
                <input 
                    type="file" 
                    ref={fileInputRef} 
                    className="hidden" 
                    onChange={onFileUpload}
                />

                {/* Text Input */}
                <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Message Sam..."
                    className="flex-1 bg-transparent border-0 focus:ring-0 resize-none text-white max-h-48 py-2 text-sm"
                    rows={1}
                    style={{ minHeight: '24px' }}
                />

                {/* Send Button */}
                <button 
                    onClick={() => { if(input.trim()) { onSendMessage(input); setInput(""); } }}
                    disabled={!input.trim() && !isThinking}
                    className={`p-2 rounded-lg transition-colors ${
                        input.trim() 
                        ? 'bg-white text-black hover:bg-gray-200' 
                        : 'bg-transparent text-gray-500 cursor-not-allowed'
                    }`}
                >
                    <Send size={18} />
                </button>
            </div>
            <div className="text-center text-xs text-gray-500 mt-2">
                Sam can make mistakes. Consider checking important information.
            </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
