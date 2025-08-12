import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.chat_models import ChatTogether
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer

# --- Load environment variables ---
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# --- Config ---
INDEX_FOLDER = "vectorstore"
MODEL_PATH = r"C:\Users\abdur.rahman\Desktop\chatbot\models\all-MiniLM-L6-v2"

# --- Wrapper for SentenceTransformer ---
class MySentenceTransformerWrapper(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()

# --- Load Embedding Model & Vectorstore ---
print("🔄 Loading local embedding model...")
model = SentenceTransformer(MODEL_PATH)
embedding_model = MySentenceTransformerWrapper(model)

print("📂 Loading FAISS vectorstore...")
vectorstore = FAISS.load_local(INDEX_FOLDER, embeddings=embedding_model, allow_dangerous_deserialization=True)

# --- Load Together Chat Model ---
llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.3
)

# --- RAG Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

# --- FastAPI App ---
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: QueryRequest):
    response = qa_chain.invoke({"query": request.query})
    return {
        "answer": response["result"],
        "sources": [doc.metadata.get("source", "") for doc in response["source_documents"]]
    }

# --- Main entry point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
