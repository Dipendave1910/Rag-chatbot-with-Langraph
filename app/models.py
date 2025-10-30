from typing import TypedDict
from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str

class BotState(TypedDict):
    """Holds conversation state including user query and response."""
    user_query: str
    response: str