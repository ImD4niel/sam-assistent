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

load_dotenv()  # Load environment variables from .env

# Try to import the latest Ollama LLM
try:
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "mistral"))
except ImportError:
    from langchain_community.llms import Ollama
    llm = Ollama(model=os.getenv("OLLAMA_MODEL", "mistral"))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ddgs import DDGS
import uuid

# Initialize Chroma vector store (in-memory for demo)
vectorstore = Chroma(
    collection_name="chatbot-docs",
    embedding_function=OllamaEmbeddings(model="mistral"),
    persist_directory=None
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def add_to_vectorstore(text, doc_id=None):
    if not text.strip():
        return
    doc_id = doc_id or str(uuid.uuid4())
    chunks = text_splitter.split_text(text)
    docs = [
        {"page_content": chunk, "metadata": {"source": doc_id}} for chunk in chunks
    ]
    vectorstore.add_texts([d["page_content"] for d in docs], metadatas=[d["metadata"] for d in docs])

def retrieve_from_vectorstore(query, k=3):
    results = vectorstore.similarity_search(query, k=k)
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

app = FastAPI()

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

class ChatResponse(BaseModel):
    response: str

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    body: Optional[ChatRequest] = Body(None)
):
    try:
        # Accept both JSON and form-data
        user_message = message or (body.message if body else "")
        if not user_message:
            return JSONResponse(status_code=400, content={"response": "[Error]: No message provided."})
        
        extracted_text = ""
        doc_id = None
        
        # Handle file upload
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
            add_to_vectorstore(extracted_text, doc_id=doc_id)
        
        # RAG: Retrieve relevant context from vector store
        rag_context = retrieve_from_vectorstore(user_message, k=3)
        
        # Web search: Use DuckDuckGo if query seems web-related
        web_context = ""
        web_keywords = [
            "news", "latest", "search", "find", "web", "internet", "today", "current", 
            "recent", "update", "trending", "happening", "now", "2024", "2025",
            "weather", "stock", "price", "market", "crypto", "bitcoin", "ethereum",
            "election", "politics", "sports", "football", "basketball", "soccer",
            "movie", "film", "celebrity", "entertainment", "technology", "ai", "chatgpt"
        ]
        
        # More aggressive web search detection
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
        
        # Perform web search if needed
        if is_web_search or mode == 'web':
            web_context = duckduckgo_search(user_message, max_results=5)
        
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
        
        # Modern, clear system prompt
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
        
        logger.info(f"Prompt input: {{'question': user_prompt}}")
        
        try:
            response = chain.invoke({"question": user_prompt})
        except Exception as e:
            logger.error(f"LLM backend error: {str(e)}")
            return JSONResponse(status_code=500, content={"response": "[Error]: LLM backend error. Please try again later."})
        
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Internal error: {str(e)}")
        return JSONResponse(status_code=500, content={"response": "[Error]: Internal server error. Please try again later."})

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
        add_to_vectorstore(extracted_text, doc_id=doc_id)
    
    rag_context = retrieve_from_vectorstore(user_message, k=3)
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