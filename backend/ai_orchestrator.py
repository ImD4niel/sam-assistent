import logging
import json
import asyncio
import traceback
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
import time
import math
import re

# Import existing components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_ollama import OllamaLLM
from duckduckgo_search import DDGS
import io
from backend.browser_tool import BrowserTool

logger = logging.getLogger(__name__)

# --- Configuration & Constants ---
MAX_REASONING_STEPS = 5
MAX_HISTORY_TOKENS = 4000  # Approximated
OFFLINE_MODE_SENTINEL = "[OFFLINE]"

# --- Data Structures ---

class TaskType(Enum):
    SIMPLE_QA = "simple_qa"
    WEB_SEARCH = "web_search"
    CODE_GENERATION = "code_generation"
    DOCUMENT_ANALYSIS = "document_analysis"
    MULTI_STEP = "multi_step"
    MATH_CALCULATION = "math_calculation" 
    DEEP_RESEARCH = "deep_research"
    GENERAL_AGENTIC = "general_agentic" # New default for smart handling

@dataclass
class TaskContext:
    user_id: str
    conversation_id: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    scratchpad: Dict[str, Any] = field(default_factory=dict) # Short-term memory for the plan
    
    async def add_message(self, role: str, content: str, llm=None):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        # Simple compression check
        if len(self.conversation_history) > 10 and llm:
            await self._summarize(llm)

    async def _summarize(self, llm):
        """Compresses older history into a summary."""
        try:
            old_msgs = self.conversation_history[:-5] # Keep last 5 intact
            recent_msgs = self.conversation_history[-5:]
            
            text_to_summarize = "\n".join([f"{m['role']}: {m['content']}" for m in old_msgs])
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Summarize the key facts from this conversation history concisely."),
                ("user", text_to_summarize)
            ])
            summary = await (prompt | llm | StrOutputParser()).ainvoke({})
            
            # Replace old messages with summary
            self.conversation_history = [{
                 "role": "system", 
                 "content": f"Previous Conversation Summary: {summary}",
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }] + recent_msgs
            logger.info("Memory compressed successfully.")
        except Exception as e:
            logger.error(f"Memory compression failed: {e}")

# --- Tool Registry ---

class ToolResult:
    def __init__(self, content: str, success: bool = True, error: str = None):
        self.content = content
        self.success = success
        self.error = error

    def __str__(self):
        if self.success:
            return f"Success: {self.content}"
        return f"Error: {self.error}"

class BaseTool:
    name: str = "base_tool"
    description: str = "Base tool"
    
    async def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Useful for mathematical calculations. Input: 'expression' (str)."

    async def execute(self, expression: str = "", **kwargs) -> ToolResult:
        try:
            # Safe eval with restricted globals
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            code = compile(expression, "<string>", "eval")
            for name in code.co_names:
                if name not in allowed_names and name not in ("abs", "min", "max", "round", "pow"):
                    return ToolResult("", False, f"Unsafe function '{name}' used.")
            
            result = eval(code, {"__builtins__": {}}, allowed_names)
            return ToolResult(str(result))
        except Exception as e:
            return ToolResult("", False, f"Math error: {str(e)}")

class OfflineAwareWebSearchTool(BaseTool):
    name = "web_search"
    description = "Searches the internet for real-time info. Input: 'query' (str)."
    
    def __init__(self, offline_mode: bool = False):
        self.offline_mode = offline_mode
        
    async def execute(self, query: str = "", **kwargs) -> ToolResult:
        if self.offline_mode:
            return ToolResult("Offline mode active. Web search unavailable.", False, "OFFLINE_MODE")
            
        try:
            results = []
            with DDGS() as ddgs:
                # Basic text search
                text_res = list(ddgs.text(query, max_results=3))
                results.extend([f"Result: {r.get('body', '')}" for r in text_res])
                
                # News search
                news_res = list(ddgs.news(query, max_results=2))
                results.extend([f"News: {r.get('title', '')}: {r.get('body', '')}" for r in news_res])
                
            if not results:
                return ToolResult("No results found.")
            return ToolResult("\n".join(results))
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return ToolResult("", False, f"Connection/Search Error: {str(e)}")

class PythonREPLTool(BaseTool):
    name = "python_repl"
    description = "Executes Python code. Use for data processing or complex logic. Input: 'code' (str)."
    
    async def execute(self, code: str = "", **kwargs) -> ToolResult:
        # SUPER BASIC SANDBOXING - DO NOT USE IN PRODUCTION WITH UNTRUSTED USERS WITHOUT DOCKER
        # This is for "Offline Code Runner" demonstration purposes
        try:
            import io
            import contextlib
            
            # Remove system-critical imports basic check
            if "os.system" in code or "subprocess" in code:
                return ToolResult("", False, "Security Block: System calls not allowed.")

            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                # We use a restricted global scope
                exec_globals = {"math": math}
                exec(code, exec_globals)
            
            return ToolResult(output_buffer.getvalue() or "Code executed successfully (no output).")
        except Exception as e:
            return ToolResult("", False, f"Runtime Error: {str(e)}")

# --- Agentic Logic (Reasoning Loop) ---

class AgentState(Enum):
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    DONE = "done"

class AIOrchestrator:
    """
    Advanced Agentic Orchestrator with Reasoning, Planning, and Tool Execution.
    Mimics 'System 2' thinking.
    """
    
    def __init__(self, llm_model: str = "mistral", offline_mode: bool = False):
        self.llm = OllamaLLM(model=llm_model, temperature=0.2) # Low temp for reasoning
        self.offline_mode = offline_mode
        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": OfflineAwareWebSearchTool(offline_mode=offline_mode),
            "python_repl": PythonREPLTool(),
            "browser": BrowserTool()
        }
        self.classifiers = {
             # ... (Keep existing simple keywords for fast path if needed)
             "search": ["search", "news", "latest", "weather"],
             "math": ["calculate", "math", "+", "*", "solve"]
        }

    # --- Core Pipeline ---

    async def execute_task(self, task_type: TaskType, user_message: str, context: TaskContext, extracted_text: str = "", image_data: str = None) -> str:
        """
        Main entry point. Uses a 'Reasoning Loop' regardless of task type for higher quality.
        """
        await context.add_message("user", user_message, self.llm)
        
        if image_data:
            logger.info("Image data detected, triggering vision analysis")
            return await self.analyze_image(user_message, image_data)

        # 0. Fast Path for Simple QA
        if task_type == TaskType.SIMPLE_QA:
             logger.info("Executing Fast Path (Simple QA)")
             final_answer = await self._handle_simple_qa(user_message, context)
             await context.add_message("assistant", final_answer, self.llm)
             return final_answer

        # Specialized Prompt Injection for Deep Research
        system_instruction = ""
        if task_type == TaskType.DEEP_RESEARCH:
            system_instruction = (
                "DEBUG: DEEP RESEARCH MODE ACTIVE.\n"
                "You are a Senior Researcher. Your goal is to provide a comprehensive report.\n"
                "1. Break the user's query into 3-5 distinct sub-questions.\n"
                "2. Use the 'web_search' or 'browser' tool for EACH sub-question.\n"
                "3. Synthesize all findings into a final cohesive answer."
            )
            # Append to user message effectively to guide the planner
            user_message = f"{system_instruction}\n\nOriginal Query: {user_message}"

        # 1. Thought & Plan
        thought_process = await self._think_and_plan(user_message, context, extracted_text)
        
        # 2. Execution Loop
        initial_answer = await self._execution_loop(thought_process, context)
        
        # 3. Reflection / Self-Correction
        # Only reflect if tools were used or it was a deep research task to avoid over-engineering simple chats
        if thought_process.get("needs_tools", False) or task_type == TaskType.DEEP_RESEARCH:
             final_answer = await self._reflect_and_polish(initial_answer, user_message, context)
        else:
             final_answer = initial_answer
        
        await context.add_message("assistant", final_answer, self.llm)
        return final_answer

    async def _reflect_and_polish(self, answer: str, query: str, context: TaskContext) -> str:
        """
        Self-correction loop: Criticizes the answer and improves it.
        """
        # Skip reflection for simple queries to save time
        if len(answer) < 50:
             return answer

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a critical editor. Review the answer for accuracy, clarity, and safety."),
            ("user", f"User Query: {query}\n\nDraft Answer: {answer}\n\nRate this answer (0-10) and provide a polished version.\nFormat: 'Score: X/10\n\n[Polished Answer]'")
        ])
        
        try:
            critique = await (prompt | self.llm | StrOutputParser()).ainvoke({})
            # Naive parsing
            if "Score:" in critique:
                parts = critique.split("\n\n", 1)
                if len(parts) > 1:
                    return parts[1].strip() # Return just the polished part
            # If parsing fails, return ORIGINAL answer, not the critique
            return answer
        except Exception:
            return answer

    # --- Reasoning Phase ---

    async def _think_and_plan(self, query: str, context: TaskContext, file_content: str) -> Dict[str, Any]:
        """
        Generates a structured plan using the LLM.
        """
        # We tell the LLM available tools
        tool_desc = "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])
        
        system_prompt = (
            "You are a sophisticated AI planner. "
            "Given a user request, create a step-by-step plan to solve it.\n"
            f"Available Tools:\n{tool_desc}\n\n"
            "If the request is simple (greeting, opinion), set 'needs_tools' to false.\n"
            "If the request requires external info, math, or code, set 'needs_tools' to true and list steps.\n"
            "Return JSON format: { 'reasoning': '...', 'needs_tools': bool, 'plan': [ {'tool': 'name', 'args': {...}, 'reason': '...'} ] }"
        )
        
        prompt = f"User Request: {query}\n"
        if file_content:
            prompt += f"Context from file: {file_content[:500]}...\n"

        try:
            # Check offline override
            if self.offline_mode and "web_search" in query:
                 return {"reasoning": "Offline mode active", "needs_tools": False, "plan": []}

            chain =  ChatPromptTemplate.from_messages([("system", system_prompt), ("user", prompt)]) | self.llm | JsonOutputParser()
            plan = await chain.ainvoke({})
            return plan
        except Exception:
            # Fallback if JSON parsing fails - treat as direct chat
            logger.warning("Planning failed to parse JSON. Falling back to direct chat.")
            return {"needs_tools": False, "plan": []}

    # --- Execution Phase ---

    async def _execution_loop(self, plan_data: Dict[str, Any], context: TaskContext) -> str:
        """
        Executes the plan tools in sequence, feeding outputs back.
        """
        if not plan_data.get("needs_tools", False):
            # Direct Answer Path (Fast)
            return await self._handle_simple_qa(context.conversation_history[-1]["content"], context)

        results_log = []
        
        for step in plan_data.get("plan", []):
            tool_name = step.get("tool")
            args = step.get("args", {})
            
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                try:
                    # Execute
                    result = await tool.execute(**args)
                    results_log.append(f"Step: Used {tool_name} with {args}\nResult: {result.content}\n")
                    
                    # Stop if critical error (unless corrected later - simplified for now)
                    if not result.success and result.error == "OFFLINE_MODE":
                         results_log.append("Warning: Skipping web step due to offline mode.")
                         
                except Exception as e:
                    results_log.append(f"Error executing {tool_name}: {e}")
            else:
                results_log.append(f"Error: Tool {tool_name} not found.")

        # --- Final Synthesis ---
        # Take all tool outputs and generate the final answer
        synthesis_prompt = (
            "You are a helpful assistant. Synthesize the following information to answer the user's request.\n\n"
            f"Execution Log:\n{''.join(results_log)}\n\n"
            "User Request: " + context.conversation_history[-1]["content"] + "\n\n"
            "Provide a clear, detailed Answer:"
        )
        
        chain = self.llm | StrOutputParser()
        return await chain.ainvoke(synthesis_prompt)

    # --- Legacy / Simple Handlers (Adapters) ---
    
    async def _handle_simple_qa(self, message: str, context: TaskContext) -> str:
         prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Sam, a helpful AI assistant. Answer the user clearly."),
            ("user", "{question}")
        ])
         return await (prompt | self.llm | StrOutputParser()).ainvoke({"question": message})

    async def analyze_image(self, user_request: str, image_path_or_base64: str) -> str:
        """Use a Vision Model (Llava) to analyze the image"""
        try:
            logger.info("Calling Vision Model (Llava)...")
            
            # Use OllamaLLM for vision if supported, or raw API call
            vision_llm = OllamaLLM(model="llava") 
            
            # Simple prompt for the vision model
            # Note: Actual image passing depends on the library version. 
            response = vision_llm.invoke(f"Describe this image and answer: {user_request}")
            return f"[Vision Analysis]: {response}"
        except Exception as e:
            logger.error(f"Vision error: {e}")
            return f"I tried to look at the image, but: {e}. (Make sure you ran 'ollama pull llava')"

    def classify_task(self, message: str) -> TaskType:
        # Simplified classifier mapping to new "General Agentic" flow usually
        # But keeping enum for compatibility with main.py checks
        msg = message.lower()
        if any(w in msg for w in self.classifiers["math"]): return TaskType.MATH_CALCULATION
        if "code" in msg or "script" in msg: return TaskType.CODE_GENERATION
        if any(w in msg for w in ["research", "investigate", "comprehensive", "report", "deep dive"]): return TaskType.DEEP_RESEARCH
        
        # Heuristic for simple QA: short length and no complex keywords
        # This prevents the heavy "Planner" prompt for "hi", "how are you", etc.
        if len(message.split()) < 15 and not any(w in msg for w in ["search", "browse", "plan", "analyze", "image"]):
             return TaskType.SIMPLE_QA
             
        return TaskType.GENERAL_AGENTIC

    def update_context(self, context, user_msg, response):
        # Helper to sync back to global dict if needed, 
        # but context object is already updated in execute_task
        return context

# --- Factory / Singleton ---
# (To be used by main.py)