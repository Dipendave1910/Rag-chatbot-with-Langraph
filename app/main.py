from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.models import ChatRequest
from app.chatbot import ChatBotApp

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A chatbot API with PDF upload and RAG capabilities",
    version="1.0.0"
)

# Initialize chatbot
chatbot = ChatBotApp()

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RAG Chatbot API is running"}

@app.post("/chat")
async def chat_route(request: ChatRequest):
    """
    Chat endpoint to interact with the chatbot.
    
    Args:
        request: ChatRequest with user query
        
    Returns:
        JSON response with chatbot answer
    """
    try:
        answer = chatbot.handle_query(request.query)
        return {"response": answer}
    except Exception as e:
        return JSONResponse(
            content={"response": f"[ERROR] {str(e)}"},
            status_code=500
        )

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file to add to the knowledge base.
    
    Args:
        file: PDF file to upload
        
    Returns:
        JSON response with upload status
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )
    
    result = chatbot.add_new_pdf(file)
    
    if result.get("status") == "success":
        return {
            "message": f"Uploaded and embedded {file.filename}.",
            "chunks_added": result["chunks_added"]
        }
    else:
        raise HTTPException(
            status_code=500,
            detail=result.get("message", "Failed to process PDF.")
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}