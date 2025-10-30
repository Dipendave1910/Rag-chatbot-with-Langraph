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
   py -3.12 -m venv env 
   source env/Scripts/activate  
   On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file**
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   GOOGLE_GEMINI_MODEL=gemini-2.0-flash
   GOOGLE_GEMINI_EMBEDDING_MODEL=gemini-embedding-001
   ```

5. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

   The API will be available at `http://localhost:8000`

   - Can check the api at /docs or in postment also

---

## Ingestion

If you want to ingest multiple PDFs and enable question-answering:

1. Place your PDF files in the `pdf_data` folder.  
2. Run the `ingest.py` script. All PDFs in the folder will be vectorized automatically.  
3. After ingestion, you can use the `/chat` endpoint to ask questions based on the uploaded PDFs.


## API Endpoints


### 1. Chat Endpoint
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

### 2. Upload PDF
- **POST** `/upload`
- Additional Feature - document upload andchat with it.
- Upload a PDF file to add to the knowledge base
- Form data: `file` (PDF file)
- Response:
  ```json
  {
    "message": "Uploaded and embedded filename.pdf",
    "chunks_added": 42
  }
  ```



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

---

## Notes

I have attached my `.env` file for direct testing and quick setup.

---

## Scope of Improvement

If given more time, the following areas can be improved and expanded:

- **Summarization Enhancement**  
  The summarization module can be made more detailed and context-aware to provide richer and more accurate summaries of retrieved content.

- **Intent Layer Improvement**  
  The intent classification can be enhanced by adding more nodes and intermediate layers for better accuracy and dynamic response handling.

- **Document Management APIs**  
  Additional APIs such as **delete**, **update**, and **list** can be implemented to manage the lifecycle of documents within the knowledge base more efficiently.

- **Streaming LLM Responses**  
  Implementing a streaming response mechanism will enable real-time responses from the LLM, providing a smoother and more interactive chat experience.

---

## Conclusion

This project was built in a short span of **2–3 hours**.  
During this time, I focused on building a clean and modular structure that demonstrates the RAG pipeline effectively.  

There is still room for improvement — with more time, features such as better summarization, improved intent detection, and enhanced document handling can be developed further to make the system more robust and production-ready.
