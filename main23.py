from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embedder = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

try:
    result = embedder.embed_query("test")
    print("SUCCESS: Embeddings work!")
    print(f"Embedding dimension: {len(result)}")
except Exception as e:
    print(f"ERROR: {e}")