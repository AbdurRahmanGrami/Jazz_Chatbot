# backend_api.py

import os
import ast
import sqlite3
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import messages_from_dict, messages_to_dict
from dotenv import load_dotenv

load_dotenv()
print("[DEBUG] Loaded API version:", os.getenv("AZURE_OPENAI_API_VERSION"))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure OpenAI setup
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# Embedding and Vectorstore setup
model_path = "C:/Users/abdur.rahman/Desktop/chatbot/models/all-MiniLM-L6-v2"
model = SentenceTransformer(model_path, local_files_only=True)
embedding_model = HuggingFaceEmbeddings(model_name=model_path)
vectorstore = FAISS.load_local(
    folder_path="vectorstore",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# SQLite: Chat sessions DB
chat_conn = sqlite3.connect("chat_sessions.db", check_same_thread=False)
chat_cursor = chat_conn.cursor()
chat_cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        history TEXT
    )
""")
chat_conn.commit()

# SQLite: User data DB
user_conn = sqlite3.connect("C:/Users/abdur.rahman/Pictures/chatbot/jazz_telco.db", check_same_thread=False)
user_cursor = user_conn.cursor()

# Pydantic model for input
class ChatRequest(BaseModel):
    message: str
    session_id: str
    phone_number: str

# Session memory retrieval
def get_memory(session_id: str):
    chat_cursor.execute("SELECT history FROM sessions WHERE session_id = ?", (session_id,))
    row = chat_cursor.fetchone()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    if row and row[0]:
        try:
            messages = messages_from_dict(ast.literal_eval(row[0]))
            memory.chat_memory.messages = messages
        except Exception as e:
            logger.warning(f"Could not load memory for session {session_id}: {e}")
    return memory

# Session memory saving
def save_memory(session_id: str, memory: ConversationBufferMemory):
    messages_dict = messages_to_dict(memory.chat_memory.messages)
    chat_cursor.execute(
        "REPLACE INTO sessions (session_id, history) VALUES (?, ?)",
        (session_id, str(messages_dict))
    )
    chat_conn.commit()

# Main chat endpoint
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    logger.info(f"[Chat] Incoming message: {request.message} (Session: {request.session_id})")
    memory = get_memory(request.session_id)

    # Lookup user info
    user_cursor.execute("SELECT * FROM users WHERE phone_number = ?", (request.phone_number,))
    row = user_cursor.fetchone()
    user_info = {}
    if row:
        columns = [column[0] for column in user_cursor.description]
        user_info = dict(zip(columns, row))

    # Retrieve relevant context
    docs = retriever.get_relevant_documents(request.message)
    context = "\n\n".join([doc.page_content for doc in docs])

    user_info_text = (
        f"""Name: {user_info.get('name', 'N/A')}
Phone: {user_info.get('phone_number', 'N/A')}
Plan: {user_info.get('active_plan', 'N/A')}
Balance: Rs. {user_info.get('balance', 'N/A')}
Usage: {user_info.get('usage_stats', 'N/A')}
City: {user_info.get('city', 'N/A')}
Region: {user_info.get('region', 'N/A')}
Last Recharge: {user_info.get('last_recharge_date', 'N/A')} (Rs. {user_info.get('recharge_amount', 'N/A')})
Data Plan: {user_info.get('data_plan', 'N/A')}
Call Minutes Used: {user_info.get('call_minutes_used', 'N/A')}
SMS Sent: {user_info.get('sms_sent', 'N/A')}"""
    ) if user_info else "User is anonymous or not found."

    prompt_template = PromptTemplate.from_template(
        """You are a helpful assistant for Jazz customers. Only answer queries relevant to Jazz services.
If the query is out of scope, politely decline.

User Info:
{user_info}

Relevant Info:
{context}

User: {question}
Assistant:"""
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    response_obj = chain.invoke({
        "user_info": user_info_text,
        "context": context,
        "question": request.message
    })

    # Safely extract the assistant response
    final_response = ""
    if isinstance(response_obj, dict) and "text" in response_obj:
        final_response = response_obj["text"]
    else:
        final_response = str(response_obj)

    # Ensure final_response is a clean string
    final_response = str(final_response).strip()



    memory.chat_memory.add_user_message(request.message)
    memory.chat_memory.add_ai_message(final_response)
    save_memory(request.session_id, memory)

    return {"response": final_response}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run("backend_api:app", host="127.0.0.1", port=8000, reload=True)
