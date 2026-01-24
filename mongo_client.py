import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus

# Load environment variables from .env file
load_dotenv()

# Use standard connection string to avoid DNS resolver issues
# URL encode credentials to handle special characters
username = quote_plus(os.getenv("MONGO_USERNAME", ""))
password = quote_plus(os.getenv("MONGO_PASSWORD", ""))

# Use standard connection string (non-SRV) to avoid DNS resolver timeout issues
uri = f"mongodb://{username}:{password}@ac-qucaeld-shard-00-00.udlewkq.mongodb.net:27017,ac-qucaeld-shard-00-01.udlewkq.mongodb.net:27017,ac-qucaeld-shard-00-02.udlewkq.mongodb.net:27017/?ssl=true&authSource=admin&retryWrites=true&w=majority&appName=langchainPractice&directConnection=false"

# Lazy connection - don't connect until first use to avoid DNS timeout on import
_client = None
_db = None
_embeddings_collection = None

def get_client():
    """Get MongoDB client (lazy initialization)"""
    global _client
    if _client is None:
        _client = MongoClient(
            uri, 
            server_api=ServerApi('1'),
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=15000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000
        )
    return _client

def get_db():
    """Get database instance"""
    global _db
    if _db is None:
        _db = get_client().get_database("langchain_practice")
    return _db

def get_embeddings_collection():
    """Get embeddings collection"""
    global _embeddings_collection
    if _embeddings_collection is None:
        _embeddings_collection = get_db().embeddings
    return _embeddings_collection

# Send a ping to confirm a successful connection (only when run directly)
if __name__ == "__main__":
    try:
        client = get_client()
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
