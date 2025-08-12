import os
import ast
import sqlite3
import logging
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import hashlib
 
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
 
from langchain.schema.runnable import RunnableMap
import sys
sys.path.insert(0, "./langchain/libs/community")

from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path=".langchain_llm_cache.db"))

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding='utf-8'
)
 
load_dotenv()
 
logger = logging.getLogger(__name__)
 
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 

llm = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_BASE_URL"],
    model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    temperature=0.7
)
 
class MySentenceTransformerWrapper:
    def __init__(self, model):
        self.model = model
 
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
 
    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()
 
    def __call__(self, text):
        return self.embed_query(text)
 
model_path = "C:/Users/abdur.rahman/Desktop/chatbot/models/all-MiniLM-L6-v2"
model = SentenceTransformer(model_path)
embedding_model = MySentenceTransformerWrapper(model)
 
vectorstore = FAISS.load_local(
    folder_path="vectorstore",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,        # final docs returned
        "fetch_k": 20  # candidate docs considered
    }
)
 
conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS memory (
    session_id TEXT,
    role TEXT,
    message TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)''')
conn.commit()
 
def save_to_memory(session_id: str, role: str, message: str):
    cursor.execute("INSERT INTO memory (session_id, role, message) VALUES (?, ?, ?)", (session_id, role, message))
    conn.commit()
 
def get_session_history(session_id: str) -> list[dict]:
    cursor.execute("""
        SELECT role, message FROM memory
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,))
    rows = cursor.fetchall()
    last_10 = rows[-10:] if len(rows) > 10 else rows
    return [{"role": role, "content": message} for role, message in last_10]

 
def get_user_profile(email: Optional[str]) -> str:
    if not email:
        return ""
    try:
        conn2 = sqlite3.connect("jazz_telco.db")
        cursor2 = conn2.cursor()
        cursor2.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cursor2.fetchone()
        conn2.close()
        if row:
            return json.dumps(dict(zip([col[0] for col in cursor2.description], row)), indent=2)
    except Exception as e:
        logger.error(f"[user_profile] error: {e}")
    return ""
 
prompt_template = PromptTemplate.from_template(
    """    
You are Jazz Assistant — a helpful AI that answers customer queries about Jazz services like prepaid, postpaid, and internet packages.

Use the context and chat history provided. You can ask clarifying questions if needed.

If you're unsure or the info isn't in the context, just say so — do not guess.


User Profile:
{user_profile}

Relevant Info:
{context}

Conversation:
{chat_history}
User: {question}
Assistant:"""

)
 
chain = RunnableMap({
    "question": lambda x: x["message"],
    "user_profile": lambda x: get_user_profile(x.get("email")),
    "chat_history": lambda x: get_session_history(x["session_id"]),
    "context": lambda x: "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(x["message"])])
}) | prompt_template | llm


def format_history(messages):
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages[-5:]])

 
class ChatRequest(BaseModel):
    message: str
    session_id: str
    email: Optional[str] = None
 
class ChatResponse(BaseModel):
    response: str
 
from openai import BadRequestError
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    query = request.message
    session_id = request.session_id
    email = request.email
 
    save_to_memory(session_id, "human", query)
 
    try:
        result = chain.invoke({"message": query, "session_id": session_id, "email": email})
        response = result.content
    except BadRequestError as e:
        logger.error(f"BadRequestError: {e}")
        response = "Sorry, something in the input triggered a restriction. Please rephrase your question."
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        response = "Sorry, I wasn't able to process that request. Please try rephrasing your question or ask about Jazz services."
 
    save_to_memory(session_id, "ai", response)
    return ChatResponse(response=response)
 
if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run("backend_api:app", host="127.0.0.1", port=8000, reload=True)