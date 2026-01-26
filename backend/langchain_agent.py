import os
import numpy as np
import google.generativeai as genai
from mongo_client import get_embeddings_collection
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure Google Generative AI for embeddings
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize GPT-4o-mini LLM for the chatbot
# Note: OPENAI_API_KEY should be set in your .env file or environment variable
# ChatOpenAI will automatically read from OPENAI_API_KEY environment variable if not provided
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY if OPENAI_API_KEY else None
)


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def retrieve_documents(query: str, top_k: int = 3, supabase_token: str = None) -> list:
    """
    Retrieve relevant documents from MongoDB using vector similarity search
    
    Args:
        query: The search query string
        top_k: Number of top results to return (default: 3)
        supabase_token: Optional Supabase token to filter documents by user
    
    Returns:
        List of dictionaries containing document information and similarity scores
    """
    try:
        # Generate embedding for the query
        result = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = result['embedding']
        
        # Get embeddings collection
        embeddings_collection = get_embeddings_collection()
        
        # Build query filter - only get documents for this user if token is provided
        query_filter = {}
        if supabase_token:
            query_filter['supabase_token'] = supabase_token
        
        # Retrieve documents matching the filter
        all_documents = list(embeddings_collection.find(query_filter))
        
        if not all_documents:
            print(f"No documents found for user (token: {supabase_token[:20] if supabase_token else 'None'}...)")
            return []
        
        # Calculate similarity scores
        scored_documents = []
        for doc in all_documents:
            similarity = cosine_similarity(query_embedding, doc['embedding'])
            scored_documents.append({
                'document_id': doc['document_id'],
                'filename': doc['filename'],
                'chunk_index': doc['chunk_index'],
                'text': doc['text'],
                'similarity': float(similarity),
                'created_at': doc.get('created_at'),
                'metadata': doc.get('metadata', {})
            })
        
        # Sort by similarity score (descending) and return top_k
        scored_documents.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = scored_documents[:top_k]
        
        return top_results
        
    except Exception as e:
        print(f"Error in retrieval: {str(e)}")
        return []


if __name__ == "__main__":
    # Test retrieval function
    print("=== Testing Retrieval Function ===")
    retrieval_query = "What is Daniel's contact information?"
    print(f"Query: {retrieval_query}\n")
    results = retrieve_documents(retrieval_query, top_k=3)
    print(f"Found {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Similarity: {result['similarity']:.4f}")
        print(f"   Source: {result['filename']}")
        print(f"   Chunk Index: {result['chunk_index']}")
        print(f"   Text: {result['text'][:200]}...")
        print()