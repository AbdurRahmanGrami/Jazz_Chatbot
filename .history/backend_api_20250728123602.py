import os
import uuid
import datetime
import sqlite3
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatTogether
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# === Load env ===
load_dotenv()

# === Local FAISS + SentenceTransformer ===
class MySentenceTransformerWrapper(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()

embedding_model = MySentenceTransformerWrapper(SentenceTransformer(
    r"C:\Users\abdur.rahman\Desktop\chatbot\models\all-MiniLM-L6-v2"))

vectorstore = FAISS.load_local("vectorstore", embedding=embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# === Together Chat Model ===
llm = ChatTogether(
    together_api_key=os.getenv("TOGETHER_API_KEY"),
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.3
)

# === Prompt ===
prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a Jazz support assistant. Use context to answer."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
    ("system", "Relevant context:\n{context}")
])

# === Chain ===
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

retrieval_chain = (
    RunnableLambda(lambda x: {"context": format_docs(retriever.get_relevant_documents(x["question"])), **x})
    | prompt
    | llm
)

# === Session Setup ===
db = sqlite3.connect("chat_memory.db", check_same_thread=False)
db.execute("""CREATE TABLE IF NOT EXISTS chat_memory (
    session_id TEXT,
    role TEXT,
    content TEXT,
    timestamp TEXT
)""")
db.commit()

def save_message(session_id, role, content):
    db.execute("INSERT INTO chat_memory VALUES (?, ?, ?, ?)", (session_id, role, content, str(datetime.datetime.now())))
    db.commit()

def load_chat_history(session_id):
    rows = db.execute("SELECT role, content FROM chat_memory WHERE session_id = ?", (session_id,)).fetchall()
    return [HumanMessage(content=c) if r == "user" else AIMessage(content=c) for r, c in rows]

# === FastAPI Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    session_id = data.get("session_id") or str(uuid.uuid4())
    question = data.get("message")

    history = load_chat_history(session_id)
    chain_input = {"question": question, "chat_history": history}
    response = retrieval_chain.invoke(chain_input)

    save_message(session_id, "user", question)
    save_message(session_id, "ai", response.content)

    return {"response": response.content, "session_id": session_id}
