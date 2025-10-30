import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application configuration settings."""
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_GEMINI_MODEL: str = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-pro")
    GOOGLE_GEMINI_EMBEDDING_MODEL: str = os.getenv("GOOGLE_GEMINI_EMBEDDING_MODEL", "models/embedding-001")
    
    # Directories
    PDF_FOLDER: str = "pdf_data"
    PERSIST_DIR: str = "chromadb"
    
    # Text Splitting
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    
    # Retrieval
    SIMILARITY_SEARCH_K: int = 5
    SUMMARIZE_SEARCH_K: int = 5

settings = Settings()