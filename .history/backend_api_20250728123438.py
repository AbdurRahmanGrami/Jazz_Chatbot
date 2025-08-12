import os
import uuid
import json
import sqlite3
import datetime
from dotenv import load_dotenv
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from together import Together

load_dotenv()

# === Vectorstore Setup ===
embedding_model = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
vectorstore = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# === FastAPI Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only; tighten this for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Session DB ===
def init_db():
    conn = sqlite3.connect("chat_sessions.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT,
                    email TEXT,
                    message TEXT,
                    is_user INTEGER,
                    timestamp TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

def store_message(session_id, email, message, is_user):
    conn = sqlite3.connect("chat_sessions.db")
    c = conn.cursor()
    c.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
              (session_id, email, message, is_user, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_session_history(session_id):
    conn = sqlite3.connect("chat_sessions.db")
    c = conn.cursor()
    c.execute("SELECT message, is_user FROM sessions WHERE id = ?", (session_id,))
    rows = c.fetchall()
    conn.close()
    messages = []
    for msg, is_user in rows:
        role = "User" if is_user else "Assistant"
        messages.append(f"{role}: {msg}")
    return "\n".join(messages)

# === User DB ===
def get_user_profile(email: Optional[str]):
    if not email:
        return ""
    try:
        with open("user_db.json", "r") as f:
            users = json.load(f)
        return users.get(email, "")
    except Exception:
        return ""

# === Prompt ===
prompt_template = PromptTemplate.from_template(
    """You are Jazz Assistant, a helpful assistant for Jazz Pakistan. Use the provided context to answer the user.

User Profile:
{user_profile}

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:"""
)

# === LLM Setup ===
llm_provider = os.getenv("LLM_PROVIDER", "azure").lower()
together_client = None

if llm_provider == "azure":
    from langchain_openai import AzureChatOpenAI
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.3
    )
elif llm_provider == "together":
    together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
else:
    raise ValueError("Unsupported LLM provider in .env file")

def run_llm_chain(query: str, session_id: str, email: Optional[str] = None) -> str:
    user_profile = get_user_profile(email)
    chat_history = get_session_history(session_id)
    context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(query)])

    prompt = prompt_template.format(
        question=query,
        user_profile=user_profile,
        chat_history=chat_history,
        context=context
    )

    if llm_provider == "azure":
        return llm.invoke(prompt).content
    elif llm_provider == "together":
        response = together_client.chat.completions.create(
            model=TOGETHER_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    else:
        raise RuntimeError("No valid LLM provider configured.")

# === /chat Endpoint ===
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    query = body.get("message")
    session_id = body.get("session_id", str(uuid.uuid4()))
    email = body.get("email")

    response = run_llm_chain(query, session_id, email)

    store_message(session_id, email, query, is_user=1)
    store_message(session_id, email, response, is_user=0)

    return {"response": response, "session_id": session_id}
