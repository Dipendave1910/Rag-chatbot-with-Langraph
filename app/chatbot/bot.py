import os
import shutil
import fitz
from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from app.config import settings
from app.chatbot.nodes import GraphNodes
from app.chatbot.graph import GraphBuilder

class ChatBotApp:
    """Main chatbot application handler."""
    
    def __init__(self):
        self.pdf_folder = settings.PDF_FOLDER
        self.persist_dir = settings.PERSIST_DIR
        
        # Initialize embeddings and LLM
        self.embedder = GoogleGenerativeAIEmbeddings(
            model=settings.GOOGLE_GEMINI_EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY
        )
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GOOGLE_GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Load vector database
        self.vectordb = None
        self.load_chroma()
        
        # Build graph
        self.nodes = GraphNodes(self.llm, self.vectordb)
        self.graph_builder = GraphBuilder(self.nodes)

    def load_chroma(self):
        """Load or initialize ChromaDB vector store."""
        try:
            self.vectordb = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedder,
            )
            print(f"[INFO] Vector DB loaded from {self.persist_dir}")
        except Exception as e:
            print(f"[ERROR] Could not load vector DB: {e}")
            self.vectordb = None

    def handle_query(self, user_query: str) -> str:
        """Handle a user query through the graph."""
        return self.graph_builder.invoke(user_query)

    def add_new_pdf(self, file: UploadFile) -> dict:
        """Process and add a new PDF to the vector database."""
        save_path = os.path.join(self.pdf_folder, file.filename)
        os.makedirs(self.pdf_folder, exist_ok=True)
        
        try:
            # Save uploaded file
            with open(save_path, "wb") as out_file:
                shutil.copyfileobj(file.file, out_file)
            
            # Extract text from PDF
            text = ""
            with fitz.open(save_path) as doc:
                text = "".join(page.get_text() for page in doc)
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Add to vector database
            vectordb = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedder,
            )
            vectordb.add_texts(chunks)
            # vectordb.persist()
            
            # Reload vector database
            self.load_chroma()
            
            # Update nodes with new vectordb
            self.nodes.vectordb = self.vectordb
            
            return {"status": "success", "chunks_added": len(chunks)}
        except Exception as e:
            return {"status": "error", "message": str(e)}