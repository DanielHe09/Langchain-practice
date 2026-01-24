import os
import numpy as np
import google.generativeai as genai
from mongo_client import get_embeddings_collection
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment variable or use placeholder
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI for embeddings
genai.configure(api_key=GOOGLE_API_KEY)

# Note: Agent initialization commented out due to LangChain 1.x API changes
# If you need agent functionality, use langchain-experimental or update to new agent API
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.agents import initialize_agent, AgentType
# from langchain.tools import Tool

# def run_agent(query: str) -> str:
#     """Run the agent with a query"""
#     try:
#         response = agent.run(query)
#         return response
#     except Exception as e:
#         return f"Error: {str(e)}"


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


def retrieve_documents(query: str, top_k: int = 3) -> list:
    """
    Retrieve relevant documents from MongoDB using vector similarity search
    
    Args:
        query: The search query string
        top_k: Number of top results to return (default: 3)
    
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
        
        # Retrieve all documents (or use MongoDB vector search if index is set up)
        all_documents = list(embeddings_collection.find({}))
        
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


def retrieve_and_format(query: str, top_k: int = 3) -> str:
    """
    Retrieve documents and format them as a readable string
    
    Args:
        query: The search query string
        top_k: Number of top results to return
    
    Returns:
        Formatted string with retrieved document information
    """
    results = retrieve_documents(query, top_k)
    
    if not results:
        return "No relevant documents found."
    
    formatted_output = f"Found {len(results)} relevant document(s):\n\n"
    
    for i, result in enumerate(results, 1):
        formatted_output += f"--- Result {i} (Similarity: {result['similarity']:.4f}) ---\n"
        formatted_output += f"Source: {result['filename']}\n"
        formatted_output += f"Chunk {result['chunk_index']}\n"
        formatted_output += f"Text: {result['text'][:500]}...\n\n"
    
    return formatted_output

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