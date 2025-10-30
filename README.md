# Simple RAG Chatbot with LangGraph and LangChain

## Overview
This project is a simple Retrieval-Augmented Generation (RAG) chatbot using:
- **Google Gemini API** (LLM & Embeddings)
- **ChromaDB** (vector store)
- **LangChain** & **LangGraph** (retrieval & intent logic)
- **FastAPI** (backend)

## Project Structure
- `pdf_data/`: Place your 3 PDFs here.
- `ingest.py`: One-time script to ingest PDFs, create embeddings, and store in ChromaDB.
- `main.py`: Runs FastAPI server for chatbot.

## Setup Instructions
1. Clone this repo or unzip files.
2. Create a `.env` file (see `.env.example`) and add your Google Gemini API key.
3. Install requirements and setup env:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
4. Place your PDFs in the `pdf_data/` folder.

## Usage
**Ingest PDFs:**
```bash
python ingest.py
```

**Run the chatbot:**
```bash
uvicorn main:app --reload
```

---

## API Endpoints

### /chat
Send a user query (e.g., general, FAQ, or summarization):
```json
POST /chat
{
  "query": "What is the main topic of the PDF?"
}
```
Returns a response from the bot, choosing how to handle based on intent.

### /upload
Upload a new PDF via multipart-form POST:
```bash
curl -F "file=@yourdoc.pdf" http://127.0.0.1:8000/upload
```
Returns json with status and number of chunks added. After uploading, you may chat about the new document immediately.

---
All credentials & secrets are stored in `.env` (excluded from git).

---
**If errors occur, check:**
- Google API key is valid
- PDFs are in the expected folder (unless uploading)
- All python dependencies are installed

---
For further help or issues, contact the project maintainer.
