import os
import re
import numpy as np
import google.generativeai as genai
from mongo_client import get_embeddings_collection
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

# Hybrid retrieval weights: semantic (vector) vs keyword (BM25). Configurable via env.
VECTOR_WEIGHT = float(os.getenv("RETRIEVAL_VECTOR_WEIGHT", "0.7"))
TEXT_WEIGHT = float(os.getenv("RETRIEVAL_TEXT_WEIGHT", "0.3"))
MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.35"))

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
    return float(dot_product / (norm1 * norm2))


def _tokenize(text: str) -> list:
    """Simple tokenizer: lowercase, split on non-alphanumeric, drop empty. Used for BM25."""
    if not text or not isinstance(text, str):
        return []
    tokens = re.split(r"\W+", text.lower())
    return [t for t in tokens if len(t) > 0]


def retrieve_documents(
    query: str,
    top_k: int = 3,
    supabase_token: str = None,
    vector_weight: float = None,
    text_weight: float = None,
    min_score: float = None,
) -> list:
    """
    Hybrid retrieval: vector (semantic) + BM25 (keyword) with weighted scoring.
    finalScore = (vector_weight * vectorScore) + (text_weight * textScore).
    Results below min_score are filtered out. Helps both conceptual queries and
    exact matches (e.g. URLs, names, IDs).
    """
    v_w = vector_weight if vector_weight is not None else VECTOR_WEIGHT
    t_w = text_weight if text_weight is not None else TEXT_WEIGHT
    min_s = min_score if min_score is not None else MIN_SCORE

    try:
        embeddings_collection = get_embeddings_collection()
        query_filter = {}
        if supabase_token:
            query_filter["supabase_token"] = supabase_token
        all_documents = list(embeddings_collection.find(query_filter))

        if not all_documents:
            print(f"No documents found for user (token: {supabase_token[:20] if supabase_token else 'None'}...)")
            return []

        # --- Vector (semantic) scores ---
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query,
            task_type="retrieval_query",
        )
        query_embedding = result["embedding"]
        vector_scores = {}
        for doc in all_documents:
            key = (doc["document_id"], doc["chunk_index"])
            sim = cosine_similarity(query_embedding, doc["embedding"])
            vector_scores[key] = max(0.0, min(1.0, sim))  # clamp to [0,1] for weighting

        # --- BM25 (keyword) scores ---
        tokenized_corpus = [_tokenize(doc.get("text") or "") for doc in all_documents]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = _tokenize(query)
        raw_bm25 = bm25.get_scores(query_tokens)
        bm25_max = float(max(raw_bm25)) if len(raw_bm25) else 0.0
        text_scores = {}
        for i, doc in enumerate(all_documents):
            key = (doc["document_id"], doc["chunk_index"])
            s = float(raw_bm25[i]) if i < len(raw_bm25) else 0.0
            text_scores[key] = (s / bm25_max) if bm25_max > 0 else 0.0

        # --- Merge: finalScore = v_w * vector + t_w * text, filter by min_s ---
        doc_by_key = {(d["document_id"], d["chunk_index"]): d for d in all_documents}
        combined = []
        for key, doc in doc_by_key.items():
            vs = vector_scores.get(key, 0.0)
            ts = text_scores.get(key, 0.0)
            final = (v_w * vs) + (t_w * ts)
            if final < min_s:
                continue
            combined.append({
                "document_id": doc["document_id"],
                "filename": doc["filename"],
                "chunk_index": doc["chunk_index"],
                "text": doc["text"],
                "similarity": round(final, 4),
                "created_at": doc.get("created_at"),
                "metadata": doc.get("metadata", {}),
            })
        combined.sort(key=lambda x: x["similarity"], reverse=True)
        return combined[:top_k]

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