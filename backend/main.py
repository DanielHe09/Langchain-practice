import os
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from mongo_client import get_embeddings_collection
from langchain_agent import retrieve_documents, llm
from datetime import datetime
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import base64
import httpx
from bs4 import BeautifulSoup
from enum import Enum
import json
import re

load_dotenv()

app = FastAPI(title="FastAPI Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to FastAPI Server", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/api/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    """Example endpoint with path and query parameters"""
    return {"item_id": item_id, "q": q}


class Message(BaseModel):
    """Message model for conversation history"""
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    conversation_history: Optional[List[Message]] = []


class ActionType(str, Enum):
    """Action types for chat responses"""
    CHAT_ONLY = "chat_only"
    OPEN_TAB = "open_tab"


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    action: ActionType
    msg: str


class ScreenshotRequest(BaseModel):
    """Request model for screenshot embedding"""
    source_url: str
    captured_at: str  # ISO string
    title: Optional[str] = None
    screenshot_data: str  # Base64 encoded image data (optional, stored but not used for text extraction)

def extract_text_from_url(url: str) -> str:
    """
    Extract text content from a webpage URL
    Falls back to basic HTML parsing if requests fail
    """
    try:
        # Try to fetch the webpage content
        response = httpx.get(url, timeout=10.0, follow_redirects=True)
        response.raise_for_status()
        
        # Parse HTML and extract text
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text if text else ""
    except Exception as e:
        print(f"Error extracting text from URL {url}: {str(e)}")
        return ""


@app.post("/api/embed-screenshot/")
async def embed_screenshot(
    request: ScreenshotRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Receive screenshot data, extract text, create embeddings, and store in MongoDB
    """
    try:
        # Extract token from Authorization header
        supabase_token = None
        if authorization and authorization.startswith("Bearer "):
            supabase_token = authorization.split(" ")[1]
        
        print(f"\n{'='*60}")
        print(f"📸 DEBUG: Screenshot received!")
        print(f"   URL: {request.source_url}")
        print(f"   Title: {request.title}")
        print(f"   Captured at: {request.captured_at}")
        print(f"   Screenshot data size: {len(request.screenshot_data) if request.screenshot_data else 0} bytes")
        print(f"   Token present: {bool(supabase_token)}")
        print(f"{'='*60}\n")
        
        # Extract text from the webpage URL (more reliable than OCR for text content)
        print(f"🔍 DEBUG: Extracting text from URL: {request.source_url}")
        text = extract_text_from_url(request.source_url)
        print(f"   Extracted text length: {len(text)} characters")
        
        if not text.strip():
            # If no text extracted, return a message (could add OCR fallback here)
            print(f"❌ DEBUG: No text extracted from URL")
            raise HTTPException(
                status_code=400, 
                detail="Could not extract text from webpage. The page might require JavaScript or be inaccessible."
            )
        
        # Initialize Google Generative AI
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        print(f"📝 DEBUG: Split text into {len(chunks)} chunks")
        
        # Create embeddings for each chunk
        embeddings_collection = get_embeddings_collection()
        document_id = str(uuid.uuid4())
        print(f"🆔 DEBUG: Generated document_id: {document_id}")
        
        documents_to_insert = []
        print(f"🔢 DEBUG: Creating embeddings for {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            # Generate embedding using Google's embedding model
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=chunk,
                task_type="retrieval_document"
            )
            embedding = result['embedding']
            
            # Prepare document for MongoDB with specified fields
            doc = {
                "supabase_token": supabase_token,
                "document_id": document_id,
                "source_type": "web_screenshot",
                "source_url": request.source_url,
                "captured_at": request.captured_at,
                "title": request.title,
                "filename": request.source_url,  # Use URL as filename for compatibility
                "chunk_index": i,
                "text": chunk,
                "embedding": embedding,
                "created_at": datetime.utcnow(),
                "metadata": {
                    "total_chunks": len(chunks),
                    "has_screenshot": True,
                    "screenshot_size": len(request.screenshot_data) if request.screenshot_data else 0
                }
            }
            documents_to_insert.append(doc)
            if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
                print(f"   ✓ Created embedding {i + 1}/{len(chunks)}")
        
        # Insert all documents into MongoDB
        if documents_to_insert:
            print(f"\n💾 DEBUG: Inserting {len(documents_to_insert)} documents into MongoDB...")
            result = embeddings_collection.insert_many(documents_to_insert)
            print(f"   ✓ Successfully inserted {len(result.inserted_ids)} documents")
            print(f"   ✓ Document IDs: {result.inserted_ids[:3]}..." if len(result.inserted_ids) > 3 else f"   ✓ Document IDs: {result.inserted_ids}")
            print(f"\n✅ DEBUG: Screenshot successfully stored in MongoDB!")
            print(f"   Document ID: {document_id}")
            print(f"   Source URL: {request.source_url}")
            print(f"   Total chunks: {len(documents_to_insert)}")
            print(f"{'='*60}\n")
            
            return {
                "status": "success",
                "message": f"Screenshot embedded and stored successfully",
                "document_id": document_id,
                "source_url": request.source_url,
                "chunks_created": len(documents_to_insert),
                "inserted_ids": len(result.inserted_ids)
            }
        else:
            print(f"❌ DEBUG: No documents to insert!")
            raise HTTPException(status_code=500, detail="Failed to create embeddings")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ DEBUG: Error processing screenshot: {str(e)}")
        print(f"{'='*60}\n")
        raise HTTPException(status_code=500, detail=f"Error processing screenshot: {str(e)}")


def retrieve_context(query: str, k: int = 4, supabase_token: str = None) -> str:
    """
    Retrieve relevant context from vector store and format it as a string
    
    Args:
        query: The search query string
        k: Number of top results to return
        supabase_token: Optional Supabase token to filter documents by user
    
    Returns:
        Formatted string with relevant context from documents
    """
    results = retrieve_documents(query, top_k=k, supabase_token=supabase_token)
    
    if not results:
        return "No relevant context found in the knowledge base."
    
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(
            f"[Document {i} from {result['filename']}]\n"
            f"{result['text']}\n"
        )
    
    return "\n".join(context_parts)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Chat endpoint that uses Claude agent with RAG (Retrieval Augmented Generation)
    
    Args:
        request: ChatRequest containing:
            - message: The user's message
            - conversation_history: Optional list of previous messages
        authorization: Authorization header containing Bearer token
    
    Returns:
        ChatResponse with the agent's response
    """
    try:
        user_message = request.message
        conversation_history = request.conversation_history or []

        # Extract token from Authorization header
        supabase_token = None
        if authorization and authorization.startswith("Bearer "):
            supabase_token = authorization.split(" ")[1]
        
        print(f"🔐 DEBUG: Chat request - Token present: {bool(supabase_token)}")

        # Step 1: Retrieve relevant context from vector store (filtered by user token)
        context = retrieve_context(user_message, k=4, supabase_token=supabase_token)
        
        # Step 2: Build the prompt with context
        # System message that instructs the agent to use the retrieved context and return structured output
        system_prompt = """You are a helpful assistant that answers questions based on the provided context from a knowledge base.

When responding, you MUST return your response in the following JSON format:
{
  "action": "chat_only" or "open_tab",
  "msg": "your response message here"
}

Action rules:
- Use "open_tab" if the user's query requires opening a new browser tab or webpage (e.g., "open YouTube", "search for Python tutorials", "go to github.com")
- Use "chat_only" for all other queries (answering questions, explanations, general conversation)

When answering:
- Use the retrieved context to provide accurate, detailed answers
- If the context contains relevant information, cite it in your response
- If the context doesn't contain enough information to answer the question, say so
- You can also use your general knowledge, but prioritize the provided context
- Be concise but thorough
- Always return valid JSON with both "action" and "msg" fields"""
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history
        for msg in conversation_history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        
        # Add current user message with context
        user_prompt = f"""Context from knowledge base:
            {context}

            User question: {user_message}

            The current time of the user request is: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

            Please answer the user's question using the context above when available."""
        
        messages.append(HumanMessage(content=user_prompt))
        
        # Step 3: Get response from LLM (the "agent")
        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Step 4: Parse structured response from LLM
        # Try to extract JSON from the response
        action = ActionType.CHAT_ONLY  # Default action
        msg = response_text  # Default message
        
        try:
            # Try to find JSON in the response (handles cases where LLM adds extra text)
            json_match = re.search(r'\{[^{}]*"action"[^{}]*"msg"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                action_str = parsed.get("action", "chat_only").lower()
                msg = parsed.get("msg", response_text)
                
                # Validate and set action
                if action_str == "open_tab":
                    action = ActionType.OPEN_TAB
                else:
                    action = ActionType.CHAT_ONLY
            else:
                # If no JSON found, try parsing the entire response as JSON
                parsed = json.loads(response_text)
                action_str = parsed.get("action", "chat_only").lower()
                msg = parsed.get("msg", response_text)
                
                if action_str == "open_tab":
                    action = ActionType.OPEN_TAB
                else:
                    action = ActionType.CHAT_ONLY
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            # If JSON parsing fails, use the raw response as message with default action
            print(f"⚠️ DEBUG: Failed to parse structured response, using default: {str(e)}")
            print(f"   Raw response: {response_text[:200]}...")
            msg = response_text
            action = ActionType.CHAT_ONLY
        
        return ChatResponse(action=action, msg=msg)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)