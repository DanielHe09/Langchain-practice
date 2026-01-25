import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from mongo_client import get_embeddings_collection
from langchain_agent import retrieve_documents, llm
from datetime import datetime
import uuid
from dotenv import load_dotenv
from io import BytesIO
from pydantic import BaseModel
from typing import Optional, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str

@app.post("/api/embed-pdf/")
async def embed_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract text, create embeddings, and store in MongoDB
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Read PDF file
        contents = await file.read()
        
        # Extract text from PDF
        pdf_reader = PdfReader(BytesIO(contents))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        # Initialize Google Generative AI
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings for each chunk
        embeddings_collection = get_embeddings_collection()
        document_id = str(uuid.uuid4())
        
        documents_to_insert = []
        for i, chunk in enumerate(chunks):
            # Generate embedding using Google's embedding model
            result = genai.embed_content(
                model="models/embedding-001",
                content=chunk,
                task_type="retrieval_document"
            )
            embedding = result['embedding']
            
            # Prepare document for MongoDB
            doc = {
                "document_id": document_id,
                "filename": file.filename,
                "chunk_index": i,
                "text": chunk,
                "embedding": embedding,
                "created_at": datetime.utcnow(),
                "metadata": {
                    "total_chunks": len(chunks),
                    "file_size": len(contents)
                }
            }
            documents_to_insert.append(doc)
        
        # Insert all documents into MongoDB
        if documents_to_insert:
            result = embeddings_collection.insert_many(documents_to_insert)
            
            return {
                "status": "success",
                "message": f"PDF embedded and stored successfully",
                "document_id": document_id,
                "filename": file.filename,
                "chunks_created": len(documents_to_insert),
                "inserted_ids": len(result.inserted_ids)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create embeddings")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


def retrieve_context(query: str, k: int = 4) -> str:
    """
    Retrieve relevant context from vector store and format it as a string
    
    Args:
        query: The search query string
        k: Number of top results to return
    
    Returns:
        Formatted string with relevant context from documents
    """
    results = retrieve_documents(query, top_k=k)
    
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
async def chat(request: ChatRequest):
    """
    Chat endpoint that uses Claude agent with RAG (Retrieval Augmented Generation)
    
    Args:
        request: ChatRequest containing:
            - message: The user's message
            - conversation_history: Optional list of previous messages
    
    Returns:
        ChatResponse with the agent's response
    """
    try:
        user_message = request.message
        conversation_history = request.conversation_history or []

        # Step 1: Retrieve relevant context from vector store
        context = retrieve_context(user_message, k=4)
        
        # Step 2: Build the prompt with context
        # System message that instructs the agent to use the retrieved context
        system_prompt = """You are a helpful assistant that answers questions based on the provided context from a knowledge base.

When answering:
- Use the retrieved context to provide accurate, detailed answers
- If the context contains relevant information, cite it in your response
- If the context doesn't contain enough information to answer the question, say so
- You can also use your general knowledge, but prioritize the provided context
- Be concise but thorough"""
        
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

            Please answer the user's question using the context above when available."""
        
        messages.append(HumanMessage(content=user_prompt))
        
        # Step 3: Get response from LLM (the "agent")
        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return ChatResponse(response=response_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))