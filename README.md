# RAG Chatbot API

A FastAPI-based chatbot with RAG (Retrieval-Augmented Generation) capabilities using LangGraph and ChromaDB.

## Project Structure

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── config.py            # Configuration settings
│   └── chatbot/
│       ├── __init__.py
│       ├── bot.py           # Main chatbot class
│       ├── graph.py         # LangGraph workflow
│       └── nodes.py         # Graph node functions
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Setup

1. **Clone the repository**

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file**
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   GOOGLE_GEMINI_MODEL=gemini-pro
   GOOGLE_GEMINI_EMBEDDING_MODEL=models/embedding-001
   ```

5. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

   The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
- **GET** `/`
- Returns API status

### 2. Chat Endpoint
- **POST** `/chat`
- Request body:
  ```json
  {
    "query": "Your question here"
  }
  ```
- Response:
  ```json
  {
    "response": "Chatbot answer"
  }
  ```

### 3. Upload PDF
- **POST** `/upload`
- Upload a PDF file to add to the knowledge base
- Form data: `file` (PDF file)
- Response:
  ```json
  {
    "message": "Uploaded and embedded filename.pdf",
    "chunks_added": 42
  }
  ```

### 4. Health Check
- **GET** `/health`
- Returns health status

## Features

- **Intent Classification**: Automatically classifies queries as greeting, FAQ, or summarization
- **RAG Pipeline**: Uses ChromaDB for vector storage and retrieval
- **PDF Processing**: Upload PDFs to expand the knowledge base
- **LangGraph Workflow**: Structured conversation flow with conditional routing
- **Google Gemini Integration**: Uses Gemini for embeddings and chat

## Configuration

Edit `app/config.py` to modify:
- Chunk size and overlap for text splitting
- Number of documents to retrieve (k parameter)
- Model names and directories

## Development

To run in development mode with auto-reload:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Testing

Test the API using curl:

```bash
# Chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello!"}'

# Upload PDF
curl -X POST "http://localhost:8000/upload" \
  -F "file=@path/to/your/document.pdf"
```

## License

MIT