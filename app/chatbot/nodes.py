from typing import Literal
from app.models import BotState
from app.config import settings

class GraphNodes:
    """Contains all LangGraph node functions."""
    
    def __init__(self, llm, vectordb):
        self.llm = llm
        self.vectordb = vectordb

    
    def greeting_node(self, state: BotState) -> BotState:
        """Handle greeting and general chat queries."""
        user_query = state['user_query']
        prompt = (
            "You are a very friendly, helpful AI assistant. Respond gently and politely to the user's greeting or general query. "
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
        """Summarize retrieved documents."""
        query = state['user_query']
        docs = self.vectordb.similarity_search(query, k=settings.SUMMARIZE_SEARCH_K)
        text_to_sum = "\n".join(d.page_content for d in docs)
        prompt = f"Summarize the following text in a few lines:\n{text_to_sum}"
        try:
            result = self.llm.invoke([{"role": "user", "content": prompt}])
            resp = result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            resp = f"[ERROR] Summarization failed: {e}"
        return {"user_query": query, "response": resp}
    
    

    def faq_node(self, state: BotState) -> BotState:
        """Answer FAQ questions using vector database."""
        query = state['user_query']
        docs = self.vectordb.similarity_search(query, k=settings.SIMILARITY_SEARCH_K)
        context = "\n".join([d.page_content for d in docs])
        
        prompt = (
            "You are an AI assistant helping users by answering questions from a knowledge base. "
            "Use the following context to answer the user's question accurately. "
            "You may rephrase for clarity or flow, but do not add information not supported by the context. "
            "Use bulleted lists where appropriate and give as much detail as possible. "
            "If the context lacks enough information, respond politely: "
            "'Sorry, I don't have information about that in my knowledge base. Could you ask something else?'\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        try:
            result = self.llm.invoke([{"role": "user", "content": prompt}])
            resp = result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            resp = f"[ERROR] FAQ LLM failed: {e}"
        return {"user_query": query, "response": resp}
    
    

    def classify_intent_node(self, state: BotState) -> Literal["greeting", "faq", "summarize"]:
        """Classify user intent using LLM."""
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
        except Exception as e:
            val = "faq"
        
        if "summarize" in val:
            return "summarize"
        if "greet" in val:
            return "greeting"
        return "faq"