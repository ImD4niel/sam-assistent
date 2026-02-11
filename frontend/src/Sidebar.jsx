import React from 'react';
import { Plus, MessageSquare, User } from 'lucide-react';

const Sidebar = ({ isOpen, onNewChat, history, loadChat }) => {
  return (
    <div className={`
      fixed inset-y-0 left-0 bg-chatgpt-sidebar text-gray-200 w-64 transform 
      ${isOpen ? "translate-x-0" : "-translate-x-full"} 
      md:translate-x-0 transition-transform duration-200 ease-in-out z-20 flex flex-col
    `}>
      {/* New Chat Button */}
      <div className="p-3">
        <button 
          onClick={onNewChat}
          className="w-full flex items-center gap-3 px-4 py-3 border border-gray-600 rounded-md hover:bg-gray-900 transition-colors text-sm text-white"
        >
          <Plus size={16} />
          New chat
        </button>
      </div>

      {/* History List */}
      <div className="flex-1 overflow-y-auto px-3">
        <div className="text-xs font-semibold text-gray-500 py-2">Today</div>
        
        <div className="space-y-1">
          {history.map((chat) => (
             <button 
               key={chat.id}
               onClick={() => loadChat(chat.id)}
               className="w-full text-left flex items-center gap-3 px-3 py-3 rounded-md hover:bg-[#2A2B32] transition-colors text-sm"
             >
               <MessageSquare size={16} className="text-gray-400" />
               <span className="truncate">{chat.title || "New conversation"}</span>
             </button>
          ))}
        </div>
      </div>

      {/* User Footer */}
      <div className="border-t border-gray-700 p-3">
        <button className="w-full flex items-center gap-3 px-3 py-3 rounded-md hover:bg-[#2A2B32] transition-colors text-sm">
            <User size={16} />
            <div className="flex-1 text-left font-medium">Daniel</div>
        </button>
      </div>
    </div>
  );
};

export default Sidebar;
