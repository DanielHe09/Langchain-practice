import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from mongo_client import get_embeddings_collection
from langchain_agent import retrieve_documents
from datetime import datetime
import uuid
from dotenv import load_dotenv
from io import BytesIO
from pydantic import BaseModel
from typing import Optional

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

class RetrievalQuery(BaseModel):
    """Request model for retrieval endpoint"""
    query: str
    top_k: Optional[int] = 3


@app.post("/api/retrieval/")
async def retrieval(request: RetrievalQuery):
    """
    Retrieve relevant documents from MongoDB using vector similarity search
    
    Args:
        request: RetrievalQuery object containing:
            - query: The search query string
            - top_k: Number of top results to return (default: 3)
    
    Returns:
        JSON response with retrieved documents and similarity scores
    """
    try:
        # Validate top_k
        if request.top_k and (request.top_k < 1 or request.top_k > 20):
            raise HTTPException(
                status_code=400, 
                detail="top_k must be between 1 and 20"
            )
        
        # Retrieve documents
        results = retrieve_documents(request.query, top_k=request.top_k or 3)
        
        if not results:
            return {
                "status": "success",
                "query": request.query,
                "results_count": 0,
                "results": [],
                "message": "No relevant documents found."
            }
        
        # Format results for JSON response
        formatted_results = []
        for result in results:
            formatted_results.append({
                "document_id": result["document_id"],
                "filename": result["filename"],
                "chunk_index": result["chunk_index"],
                "text": result["text"],
                "similarity": round(result["similarity"], 4),
                "created_at": result.get("created_at").isoformat() if result.get("created_at") else None,
                "metadata": result.get("metadata", {})
            })
        
        return {
            "status": "success",
            "query": request.query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving documents: {str(e)}"
        )


@app.get("/api/retrieval/")
async def retrieval_get(
    query: str = Query(..., description="The search query string"),
    top_k: Optional[int] = Query(3, ge=1, le=20, description="Number of top results to return")
):
    """
    Retrieve relevant documents from MongoDB using vector similarity search (GET method)
    
    Args:
        query: The search query string (required)
        top_k: Number of top results to return (default: 3, max: 20)
    
    Returns:
        JSON response with retrieved documents and similarity scores
    """
    try:
        # Retrieve documents
        results = retrieve_documents(query, top_k=top_k)
        
        if not results:
            return {
                "status": "success",
                "query": query,
                "results_count": 0,
                "results": [],
                "message": "No relevant documents found."
            }
        
        # Format results for JSON response
        formatted_results = []
        for result in results:
            formatted_results.append({
                "document_id": result["document_id"],
                "filename": result["filename"],
                "chunk_index": result["chunk_index"],
                "text": result["text"],
                "similarity": round(result["similarity"], 4),
                "created_at": result.get("created_at").isoformat() if result.get("created_at") else None,
                "metadata": result.get("metadata", {})
            })
        
        return {
            "status": "success",
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving documents: {str(e)}"
        )


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