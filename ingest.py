import os
from pathlib import Path
import glob
from typing import List
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

class PDFHandler:
    def __init__(self, pdf_folder: str, save_dir: str = "chromadb"):
        self.pdf_folder = pdf_folder
        self.save_dir = save_dir
        try:
            self.embedder = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=os.environ["GOOGLE_API_KEY"]
            )
        except Exception as e:
            print(f"[ERROR] Could not initialize embedding: {e}")
            raise
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

    def load_pdfs(self) -> List[str]:
        files = glob.glob(f"{self.pdf_folder}/*.pdf")
        docs = []
        for file in files:
            try:
                with fitz.open(file) as doc:
                    text = "".join(page.get_text() for page in doc)
                docs.append(text)
                print(f"Loaded file: {os.path.basename(file)}")
            except Exception as e:
                print(f"[ERROR] Could not read {file}: {e}")
        if not docs:
            print("[WARN] No PDF files found or unable to load PDFs!")
        return docs

    def chunk_documents(self, documents: List[str]) -> List[str]:
        all_chunks = []
        for i, doc in enumerate(documents):
            try:
                chunks = self.text_splitter.split_text(doc)
                all_chunks.extend(chunks)
                print(f"Document {i+1}: split into {len(chunks)} chunks.")
            except Exception as e:
                print(f"[ERROR] Failed chunking document {i+1}: {e}")
        return all_chunks

    def save_to_chroma(self, chunks: List[str]):
        try:
            vectordb = Chroma.from_texts(
                texts=chunks,
                embedding=self.embedder,
                persist_directory=self.save_dir
            )
            # vectordb.persist()
            print(f"[INFO] Saved {len(chunks)} chunks to ChromaDB.")
        except Exception as e:
            print(f"[ERROR] Could not store embeddings in ChromaDB: {e}")
            raise

    def make_embeddings(self):
        print("[INFO] Loading PDFs...")
        docs = self.load_pdfs()
        if not docs:
            print("[FATAL] No documents to process. Exiting.")
            return
        print("[INFO] Splitting documents into chunks...")
        chunks = self.chunk_documents(docs)
        print(f"[INFO] {len(chunks)} chunks generated.")
        print("[INFO] Making embeddings and storing...")
        self.save_to_chroma(chunks)
        print("[SUCCESS] Embedding and storage complete!")

if __name__ == "__main__":
    handler = PDFHandler(pdf_folder="pdf_data")
    handler.make_embeddings()
