import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import List, Literal, TypedDict
import shutil
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel

load_dotenv()

class ChatRequest(BaseModel):
    query: str

# --- LangGraph State (FIXED) ---
class BotState(TypedDict):
    """Holds conversation state incl. user query, intent, and response."""
    user_query: str
    response: str

# --- Main Handler Class ---
class ChatBotApp:
    def __init__(self, pdf_folder: str = "pdf_data", persist_dir: str = "chromadb"):
        print("Initializing ChatBotApp...")
        self.pdf_folder = pdf_folder
        self.persist_dir = persist_dir
        self.embedder = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        self.vectordb = None
        self.load_chroma()
        self.graph = self.build_graph()

    def load_chroma(self):
        try:
            self.vectordb = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedder,
            )
        except Exception as e:
            print(f"[ERROR] Could not load vector DB: {e}")
            self.vectordb = None

    # --- LangGraph nodes resolvers ---
    def greeting_node(self, state: BotState) -> BotState:
        print("here in the greeting node")
        print(f"[INFO] Greeting node processing query: {state}")
        user_query = state['user_query']
        prompt = (
            "You are a very friendly, helpful AI assistant. Say hello and respond gently and politely to the user's greeting or general query. "
            "Always use a warm, welcoming tone. Here is what the user said: "
            f"\nUser: {user_query}\n"
            "Respond:"
        )
        try:
            result = self.llm.invoke([{"role": "user", "content": prompt}])
            resp = result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            resp = f"[ERROR] Greeting LLM failed: {e}"
        return {"user_query": user_query, "response": resp}

    def summarize_node(self, state: BotState) -> BotState:
        query = state['user_query']
        docs = self.vectordb.similarity_search(query, k=2)
        text_to_sum = "\n".join(d.page_content for d in docs)
        prompt = f"Summarize the following text in a few lines:\n{text_to_sum}"
        try:
            result = self.llm.invoke([{"role": "user", "content": prompt}])
            resp = result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            resp = f"[ERROR] Summarization failed: {e}"
        return {"user_query": query, "response": resp}

    def faq_node(self, state: BotState) -> BotState:
        print("here in the faq node")
        print(f"[INFO] FAQ node processing query: {state}")
        query = state['user_query']
        docs = self.vectordb.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])
        print(f"[DEBUG] FAQ context retrieved: {context}")
        prompt = (
                "You are an AI assistant helping users by answering questions from a knowledge base."
                "Use the following context to answer the user's question accurately."
                "You may rephrase for clarity or flow, but do not add information not supported by the context."
                "Use bulleted lists where appropriate and give as much detail as possible."
                "If the context lacks enough information, respond politely:"
                "'Sorry, I don't have information about that in my knowledge base. Could you ask something else?'\n"
                f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )
        try:
            result = self.llm.invoke([{"role": "user", "content": prompt}])
            resp = result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            resp = f"[ERROR] FAQ LLM failed: {e}"
        return {"user_query": query, "response": resp}

    # --- LLM-based Intent router (FIXED) ---
    def classify_intent_node(self, state: BotState) -> Literal["greeting", "faq", "summarize"]:
        print("here in the intent classification node")
        print(f"[INFO] Classifying intent for query: {state}")
        query = state['user_query']
        prompt = (
            "Given the user message below, reply with ONLY one word: 'greeting', 'faq', or 'summarize'.\n"
            "Respond 'greeting' if it is a salutation or general chat. "
            "Respond 'summarize' if they want you to summarize something. "
            "Else respond 'faq'.\n\n"
            f"Message: {query}"
        )
        try:
            result = self.llm.invoke([{"role": "user", "content": prompt}])
            val = result.content.strip().lower()
            print(f"[DEBUG] Intent classification result: {val}")
        except Exception as e:
            val = "faq"
            print(f"[ERROR] Intent LLM fallback: {e}")
        if "summarize" in val:
            return "summarize"
        if "greet" in val:
            return "greeting"
        return "faq"

    # --- LangGraph Setup (FIXED) ---
    def build_graph(self):
        workflow = StateGraph(BotState)
        print("Building graph...", BotState)

        # Nodes
        workflow.add_node("greeting", self.greeting_node)
        workflow.add_node("faq", self.faq_node)
        workflow.add_node("summarize", self.summarize_node)

        # Edges - Connect START directly to conditional routing
        workflow.add_conditional_edges(
            START,
            self.classify_intent_node,
            {
                "greeting": "greeting",
                "faq": "faq",
                "summarize": "summarize",
            },
        )

        workflow.add_edge("greeting", END)
        workflow.add_edge("faq", END)
        workflow.add_edge("summarize", END)

        return workflow.compile()

    # --- API Logic ---
    def handle_query(self, user_query: str) -> str:
        print(f"[INFO] Handling query: {user_query}")
        in_state = {"user_query": user_query, "response": ""}
        print(f"[DEBUG] Initial state: {in_state}")
        result = self.graph.invoke(in_state)
        return result.get("response", "[ERROR] No response generated.")

    def add_new_pdf(self, file: UploadFile) -> dict:
        save_path = os.path.join(self.pdf_folder, file.filename)
        os.makedirs(self.pdf_folder, exist_ok=True)
        try:
            with open(save_path, "wb") as out_file:
                shutil.copyfileobj(file.file, out_file)
            text = ""
            with fitz.open(save_path) as doc:
                text = "".join(page.get_text() for page in doc)
            chunks = self.text_splitter.split_text(text)
            vectordb = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedder,
            )
            vectordb.add_texts(chunks)
            vectordb.persist()
            self.load_chroma()
            return {"status": "success", "chunks_added": len(chunks)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# --- FastAPI App ---
app = FastAPI()
chatbot = ChatBotApp()

@app.post("/chat")
async def chat_route(request: ChatRequest):
    try:
        answer = chatbot.handle_query(request.query)
        return {"response": answer}
    except Exception as e:
        return JSONResponse(content={"response": f"[ERROR] {str(e)}"}, status_code=500)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    result = chatbot.add_new_pdf(file)
    if result.get("status") == "success":
        return {"message": f"Uploaded and embedded {file.filename}.", "chunks_added": result["chunks_added"]}
    else:
        raise HTTPException(status_code=500, detail=result.get("message", "Failed to process PDF."))