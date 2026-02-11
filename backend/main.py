import logging
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from fastapi import UploadFile, File, Form
import tempfile
import shutil
import mimetypes
import requests
from typing import Optional
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
import asyncio
import time  # type: ignore
import json
import base64

load_dotenv()  # Load environment variables from .env

# Try to import the latest Ollama LLM
try:
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama2"))
except ImportError:  # type: ignore
    from langchain_community.llms import Ollama
    llm = Ollama(model=os.getenv("OLLAMA_MODEL", "llama2"))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS
import uuid

# Import AI Orchestration Components
from backend.ai_orchestrator import AIOrchestrator, TaskType, TaskContext
from backend.database import DatabaseManager

# Initialize Database
db_manager = DatabaseManager()

# Initialize Chroma vector store (persistent)
vectorstore = Chroma(
    collection_name="chatbot-docs",
    embedding_function=OllamaEmbeddings(model="llama2"),
    persist_directory="./chroma_db"
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def add_to_vectorstore(text, doc_id=None, conversation_id=None):
    if not text.strip():
        return
    doc_id = doc_id or str(uuid.uuid4())
    chunks = text_splitter.split_text(text)
    
    # Metadata now includes conversation_id for isolation
    metadatas = []
    for chunk in chunks:
        meta = {"source": doc_id}
        if conversation_id:
            meta["conversation_id"] = conversation_id
        metadatas.append(meta)
        
    vectorstore.add_texts(chunks, metadatas=metadatas)

def retrieve_from_vectorstore(query, k=3, conversation_id=None):
    # Add filter if conversation_id is provided
    filter_dict = None
    if conversation_id:
        filter_dict = {"conversation_id": conversation_id}
        
    results = vectorstore.similarity_search(query, k=k, filter=filter_dict)
    return "\n".join([r.page_content for r in results])

def duckduckgo_search(query, max_results=5):
    try:
        with DDGS() as ddgs:
            # Get text results
            text_results = ddgs.text(query, max_results=max_results)
            text_content = "\n".join([r["body"] for r in text_results if "body" in r])
            
            # Get news results for real-time data
            news_results = ddgs.news(query, max_results=3)
            news_content = "\n".join([f"News: {r['title']} - {r['body']}" for r in news_results if "title" in r and "body" in r])
            
            # Combine results
            combined_results = f"Text Results:\n{text_content}\n\nLatest News:\n{news_content}"
            return combined_results
    except Exception as e:
        return f"[Web search error: {e}]"

# For OCR and PDF/text extraction
try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Initialize the sophisticated AI orchestrator
is_offline = os.getenv("OFFLINE_MODE", "false").lower() == "true"
ai_provider = os.getenv("AI_PROVIDER", "ollama")
ai_model_name = os.getenv("AI_MODEL", "llama2")

orchestrator = AIOrchestrator(
    provider=ai_provider,
    model_name=ai_model_name,
    offline_mode=is_offline
)

# Store conversation contexts (in production, use a proper database)
conversation_contexts = {}

app = FastAPI(title="Sam Assistant with AI Orchestration", version="2.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = "auto"  # auto, web, docs, orchestrated
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    task_type: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

@app.get("/health")
def health():
    return {"status": "ok", "orchestrator": "active"}

@app.get("/orchestrator/capabilities")
def get_orchestrator_capabilities():
    """Get information about what the AI orchestrator can do"""
    return {
        "task_types": [task.value for task in TaskType],
        "features": [
            "Intelligent task classification",
            "Multi-step reasoning",
            "Enhanced web search",
            "Code generation with validation",
            "Document analysis",
            "Context management",
            "Fallback handling"
        ],
        "supported_modes": ["auto", "web", "docs", "orchestrated"]
    }

def get_or_create_context(user_id: str, conversation_id: str) -> TaskContext:
    """Get or create a conversation context"""
    key = f"{user_id}:{conversation_id}"
    
    # Try load from RAM first
    if key in conversation_contexts:
        return conversation_contexts[key]
        
    # Try load from DB
    data = db_manager.load_context(user_id, conversation_id)
    if data:
        conversation_contexts[key] = TaskContext(**data)
        logger.info(f"Loaded context for {key} from DB")
    else:
        conversation_contexts[key] = TaskContext(
            user_id=user_id,
            conversation_id=conversation_id,
        )
    return conversation_contexts[key]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    body: Optional[ChatRequest] = Body(None)
):
    """Enhanced chat endpoint using the sophisticated AI orchestrator"""
    start_time = time.time()
    
    try:
        # Extract user message and parameters
        user_message = message or (body.message if body else "")
        user_id = (body.user_id if body else None) or "default_user"
        conversation_id = (body.conversation_id if body else None) or str(uuid.uuid4())
        
        if not user_message:
            return JSONResponse(status_code=400, content={"response": "[Error]: No message provided."})
        
        # Get or create conversation context
        context = get_or_create_context(user_id, conversation_id)
        
        # Handle file upload (integrate with existing functionality)
        extracted_text = ""
        image_data = None
        if file:
            content_type = file.content_type or ""
            file.file.seek(0, 2)
            size = file.file.tell()
            file.file.seek(0)
            if size > 10 * 1024 * 1024:
                return JSONResponse(status_code=400, content={"response": "[Error]: File too large (max 10MB)."})
            
            allowed_types = ["image/", "application/pdf", "text/"]
            if not any(content_type.startswith(t) for t in allowed_types):
                return JSONResponse(status_code=400, content={"response": "[Error]: Unsupported file type."})
            
            suffix = mimetypes.guess_extension(content_type) or ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
            
            # Prepare image data for Vision Model
            image_data = None
            try:
                if content_type.startswith("image/"):
                    with open(tmp_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                        image_data = encoded_string
                
                if content_type.startswith("image/") and pytesseract and Image:
                    img = Image.open(tmp_path)
                    extracted_text = pytesseract.image_to_string(img)
                elif content_type == "application/pdf" and PyPDF2:
                    with open(tmp_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        extracted_text = " ".join(page.extract_text() or "" for page in reader.pages)
                elif content_type.startswith("text/"):
                    with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                        extracted_text = f.read()
                else:
                    extracted_text = "[Unsupported file type for extraction]"
            finally:
                import os
                os.remove(tmp_path)
            
            # Add extracted text to context
            if extracted_text.strip():
                context.conversation_history.append({
                    "user": f"[Uploaded file: {file.filename}]",
                    "assistant": f"[Extracted text: {extracted_text[:200]}...]",
                    "timestamp": "now"
                })
        
        # Use the sophisticated AI orchestrator
        logger.info(f"Processing message with AI orchestrator: {user_message[:100]}...")
        
        # Classify the task
        task_type = orchestrator.classify_task(user_message)
        logger.info(f"Task classified as: {task_type}")
        
        # Execute the task using the orchestrator
        response = await execute_orchestrated_task(task_type, user_message, context, extracted_text, image_data)
        
        # Update context with the interaction
        context = orchestrator.update_context(context, user_message, response)
        conversation_contexts[f"{user_id}:{conversation_id}"] = context
        
        # Save to DB
        # Convert TaskContext to dict for saving. 
        # Note: We need a better serializer if TaskContext has complex objects, but for now strict dict usage is fine.
        db_manager.save_context(user_id, conversation_id, context.__dict__)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=response,
            task_type=task_type.value,
            confidence=0.95,  # You could implement confidence scoring
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Orchestrated chat error: {str(e)}")
        return JSONResponse(status_code=500, content={"response": f"[Error]: {str(e)}"})

async def execute_orchestrated_task(task_type: TaskType, user_message: str, context: TaskContext, extracted_text: str = "", image_data: str = None) -> str:
    """Execute the task based on its type using the orchestrator"""
    try:
        # Use the orchestrator's execute_task method
        return await orchestrator.execute_task(task_type, user_message, context, extracted_text, image_data)
    except Exception as e:
        logger.error(f"Error executing task {task_type}: {str(e)}")
        return f"I encountered an error while processing your request: {str(e)}"



@app.post("/chat/stream/orchestrated")
async def chat_stream_orchestrated_endpoint(
    request: Request,
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    body: Optional[ChatRequest] = Body(None)
):
    """Streaming chat endpoint using the sophisticated AI orchestrator"""
    try:
        # Extract user message and parameters
        user_message = message or (body.message if body else "")
        user_id = (body.user_id if body else None) or "default_user"
        conversation_id = (body.conversation_id if body else None) or str(uuid.uuid4())
        
        if not user_message:
            return JSONResponse(status_code=400, content={"response": "[Error]: No message provided."})
        
        # Get or create conversation context
        context = get_or_create_context(user_id, conversation_id)
        
        # Handle file upload (same as above)
        extracted_text = ""
        if file:
            # ... (same file handling code as above)
            pass
        
        # Use the sophisticated AI orchestrator
        task_type = orchestrator.classify_task(user_message)
        logger.info(f"Streaming task classified as: {task_type}")
        
        # For streaming, we'll use a simpler approach but still leverage the orchestrator
        # In a full implementation, you'd want to stream each step of complex tasks
        
        async def gen():
            try:
                # Send task classification
                yield f"data: {json.dumps({'type': 'task_classification', 'task_type': task_type.value})}\n\n"
                
                # Execute the task (for now, non-streaming)
                response = await execute_orchestrated_task(task_type, user_message, context, extracted_text)
                
                # Update context
                context = orchestrator.update_context(context, user_message, response)
                conversation_contexts[f"{user_id}:{conversation_id}"] = context
                
                # Stream the response
                yield f"data: {json.dumps({'type': 'response', 'content': response})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        
        return StreamingResponse(gen(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Streaming orchestrated chat error: {str(e)}")
        return JSONResponse(status_code=500, content={"response": f"[Error]: {str(e)}"})

@app.post("/chat/stream")
async def chat_stream_endpoint(
    request: Request,
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    body: Optional[ChatRequest] = Body(None)
):
    # Robustly extract user_message and file from form or JSON
    user_message = message
    upload_file = file
    if user_message is None and body is not None:
        user_message = body.message
    if user_message is None:
        try:
            data = await request.json()
            user_message = data.get("message")
        except Exception:
            pass
    
    if not user_message:
        def error_no_message():
            yield "[Error]: No message provided."
        return StreamingResponse(error_no_message(), media_type="text/plain")
    
    # The rest of the logic is the same as before, but use a regular generator for StreamingResponse
    extracted_text = ""
    doc_id = None
    
    if upload_file:
        content_type = upload_file.content_type or ""
        upload_file.file.seek(0, 2)
        size = upload_file.file.tell()
        upload_file.file.seek(0)
        if size > 10 * 1024 * 1024:
            def error_file_too_large():
                yield "[Error]: File too large (max 10MB)."
            return StreamingResponse(error_file_too_large(), media_type="text/plain")
        
        allowed_types = ["image/", "application/pdf", "text/"]
        if not any(content_type.startswith(t) for t in allowed_types):
            def error_unsupported_type():
                yield "[Error]: Unsupported file type."
            return StreamingResponse(error_unsupported_type(), media_type="text/plain")
        
        suffix = mimetypes.guess_extension(content_type) or ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = tmp.name
        
        try:
            if content_type.startswith("image/") and pytesseract and Image:
                img = Image.open(tmp_path)
                extracted_text = pytesseract.image_to_string(img)
            elif content_type == "application/pdf" and PyPDF2:
                with open(tmp_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    extracted_text = " ".join(page.extract_text() or "" for page in reader.pages)
            elif content_type.startswith("text/"):
                with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                    extracted_text = f.read()
            else:
                extracted_text = "[Unsupported file type for extraction]"
        finally:
            import os
            os.remove(tmp_path)
        
        doc_id = str(uuid.uuid4())
        add_to_vectorstore(extracted_text, doc_id=doc_id, conversation_id=conversation_id)
    
    rag_context = retrieve_from_vectorstore(user_message, k=3, conversation_id=conversation_id)
    web_context = ""
    if any(word in user_message.lower() for word in ["news", "latest", "search", "find", "web", "internet"]):
        web_context = duckduckgo_search(user_message, max_results=3)
    
    # Detect if this is a web search question
    web_keywords = ["news", "latest", "search", "find", "web", "internet"]
    is_web_search = any(word in user_message.lower() for word in web_keywords)
    
    # Accept mode parameter from form, JSON, or body
    mode = None
    try:
        form = await request.form()
        if 'mode' in form:
            mode = form['mode']
    except Exception:
        form = None
    
    if mode is None:
        try:
            data = await request.json()
            if isinstance(data, dict) and 'mode' in data:
                mode = data['mode']
        except Exception:
            data = None
    
    if mode is None and body is not None and hasattr(body, 'dict'):
        body_dict = body.dict()
        if 'mode' in body_dict:
            mode = body_dict['mode']
    
    if mode is None:
        mode = 'auto'
    if not isinstance(mode, str):
        mode = 'auto'
    mode = mode.lower()
    
    # Context selection logic
    use_web = (mode == 'web') or (mode == 'auto' and is_web_search)
    use_docs = (mode == 'docs') or (mode == 'auto' and not is_web_search)
    
    prompt_parts = [user_message]
    if use_docs:
        if extracted_text.strip():
            prompt_parts.append(f"[File content]:\n{extracted_text.strip()}")
        if rag_context.strip():
            prompt_parts.append(f"[Relevant document context]:\n{rag_context.strip()}")
    if use_web and web_context.strip():
        prompt_parts.append(f"[Web search results]:\n{web_context.strip()}")
    
    user_prompt = "\n".join(prompt_parts)
    system_message = (
        "You are Sam, a helpful, reliable, and modern AI assistant. "
        "Always provide clear, relevant, and accurate answers. "
        "If you do not know the answer, say so honestly. "
        "Do not hallucinate or make up information."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{question}")
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    def gen():
        try:
            if hasattr(llm, "stream"):
                full_prompt = f"{system_message}\n\n{user_prompt}"
                for chunk in llm.stream(full_prompt):
                    yield chunk
            else:
                response = chain.invoke({"question": user_prompt})
                yield response
        except Exception as e:
            yield f"[Error]: LLM backend error: {str(e)}"
    
    return StreamingResponse(gen(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 