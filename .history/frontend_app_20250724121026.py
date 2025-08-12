import chainlit as cl
import os
import uuid
import json
import datetime
import sqlite3
from dotenv import load_dotenv
from typing import Optional
import httpx
import hashlib
import logging
from sentence_transformers import SentenceTransformer
from chainlit import Text, ElementSidebar
 
os.environ["PYDANTIC_V2_FORCE_GENERATE_SCHEMAS"] = "1"
 
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding='utf-8'
)
 
load_dotenv()
 
# Embedding model
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
sentence_model = SentenceTransformer(model_path)
embedding = MySentenceTransformerWrapper(sentence_model)
 
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.load_local(
    folder_path="vectorstore",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
 
# Auth
@cl.password_auth_callback
def auth_callback(email: str, password: str) -> Optional[cl.User]:
    email = email.strip().lower()
    if email == "guest":
        return cl.User(identifier="guest", metadata={
            "user_id": "guest",
            "name": "Guest",
            "phone_number": "N/A",
            "active_plan": "N/A",
            "balance": "0",
            "cnic": "N/A",
            "region": "N/A",
            "city": "N/A",
            "signup_date": "N/A",
            "last_recharge_date": "N/A",
            "recharge_amount": "0",
            "data_plan": "N/A",
            "call_minutes_used": "N/A",
            "sms_sent": "N/A",
            "memory": "",
            "provider": "guest",
            "email": "guest"
        })
    try:
        conn = sqlite3.connect("jazz_telco.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()
 
        hashed_input = hashlib.sha256(password.strip().encode()).hexdigest()
        if row and hashed_input == row["password"].strip():
            return cl.User(identifier=email, metadata=dict(row))
    except Exception as e:
        print(f"[Auth Error] {e}")
    return None
 
@cl.on_chat_start
async def start():
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    user = cl.user_session.get("user")
    cl.user_session.set("authenticated", bool(user))
    cl.user_session.set("user_info", user.metadata if user else {})
    await cl.Message(content="👋 Welcome to Jazz Assistant!").send()
    await render_sidebar()
 
async def render_sidebar():
    user_info = cl.user_session.get("user_info", {})
    email = user_info.get("email") or user_info.get("user_id")
    sidebar_elements = []
    try:
        conn = sqlite3.connect("jazz_telco.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT memory FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()
        memory = []
        if row and row["memory"]:
            mem = json.loads(row["memory"])
            if isinstance(mem, list) and mem and isinstance(mem[0], dict) and "type" in mem[0]:
                for i in range(0, len(mem), 2):
                    if i+1 < len(mem):
                        q = mem[i].get("data", {}).get("content", "")
                        a = mem[i+1].get("data", {}).get("content", "")
                        memory.append([q, a])
            else:
                memory = mem
            for q, a in memory[-5:][::-1]:
                sidebar_elements.append(Text(content=f"**You:** {q}\n**Bot:** {a}"))
    except Exception as e:
        print(f"[Sidebar Memory Fetch Error] {e}")
    await ElementSidebar.set_title("🧠 Chat Memory")
    await ElementSidebar.set_elements(sidebar_elements)
 
@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    user_info = cl.user_session.get("user_info", {})
    email = user_info.get("email") or user_info.get("user_id")
    phone_number = user_info.get("phone_number", "unknown")
 
    user_input = message.content.strip()
    if not user_input:
        return
 
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:8000/chat",
                json={
                    "message": user_input,
                    "session_id": session_id,
                    "email": email
                },
                timeout=60
            )
            response_data = resp.json()
            response = response_data.get("response", "⚠️ No response from backend.")
 
        await cl.Message(content=response).send()
 
        try:
            conn = sqlite3.connect("jazz_telco.db")
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT memory FROM users WHERE email = ?", (email,))
            row = cursor.fetchone()
            memory = []
            if row and row["memory"]:
                mem = json.loads(row["memory"])
                if isinstance(mem, list) and mem and isinstance(mem[0], dict) and "type" in mem[0]:
                    for i in range(0, len(mem), 2):
                        if i+1 < len(mem):
                            q = mem[i].get("data", {}).get("content", "")
                            a = mem[i+1].get("data", {}).get("content", "")
                            memory.append([q, a])
                else:
                    memory = mem
            memory.append([user_input, response])
            memory = memory[-20:]
            cursor.execute("UPDATE users SET memory = ? WHERE email = ?", (json.dumps(memory), email))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Memory Update Error] {e}")
 
        await render_sidebar()
 
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
        await cl.Message(content=f"❌ Error generating response: {str(e)}").send()
 
 