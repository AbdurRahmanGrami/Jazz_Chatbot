import os
import json
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# --- Configuration ---
faq_file = "faqs.json"
offers_file = ["prepaid_full.json", "jazz_postpaid_offers.json"]
insurance_file = "insurance_chunks.json"  # Your separated insurance chunks JSON
model_path = r"C:\Users\abdur.rahman\Desktop\chatbot\models\all-MiniLM-L6-v2"
index_folder = "vectorstore"

# --- Load model ---
print("🔄 Loading embedding model...")
model = SentenceTransformer(model_path)

# --- Helper: Load Offers with Offer-Level Chunking ---
def load_offer_documents(filepath: str) -> List[Document]:
    with open(filepath, "r", encoding="utf-8") as f:
        offers = json.load(f)

    docs = []
    for entry in offers:
        title = entry.get("Title", "").strip()
        description = entry.get("Description", "").strip()
        url = entry.get("URL", "").strip()
        full_text = entry.get("FullText", "").strip()

        metadata = {
            "type": "offer",
            "source": url,
            "title": title,
        }

        content = f"[Type: Offer]\n[Source: {url}]\n[Title: {title}]\n{description}\n{full_text}"
        docs.append(Document(page_content=content, metadata=metadata))

    return docs

# --- Helper: Load FAQs ---
def load_faq_documents(filepath: str) -> List[Document]:
    with open(filepath, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    docs = []
    for entry in faqs:
        metadata = entry.get("metadata", {})
        question = metadata.get("question", "").strip()
        answer = metadata.get("answer", "").strip()
        source = metadata.get("source", "").strip()

        content = f"[Type: FAQ]\n[Source: {source}]\nQ: {question}\nA: {answer}"
        docs.append(Document(page_content=content, metadata={
            "type": "faq",
            "source": source,
            "question": question
        }))
    return docs

# --- Helper: Load Insurance Chunks ---
def load_insurance_documents(filepath: str) -> List[Document]:
    with open(filepath, "r", encoding="utf-8") as f:
        insurance_chunks = json.load(f)

    docs = []
    for entry in insurance_chunks:
        title = entry.get("Title", "").strip()
        url = entry.get("URL", "").strip()
        full_text = entry.get("FullText", "").strip()

        metadata = {
            "type": "insurance",
            "source": url,
            "title": title,
        }

        content = f"[Type: Insurance]\n[Source: {url}]\n[Title: {title}]\n{full_text}"
        docs.append(Document(page_content=content, metadata=metadata))

    return docs

# --- LangChain-compatible wrapper ---
class MySentenceTransformerWrapper(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

# --- Main embedding pipeline ---
if __name__ == "__main__":
    print("📁 Starting embedding process...")

    documents = []

    for offer_file in offers_file:
        documents.extend(load_offer_documents(offer_file))

    documents.extend(load_faq_documents(faq_file))
    documents.extend(load_insurance_documents(insurance_file))

    print(f"🧐 Total chunks (docs): {len(documents)}")

    embedding_model = MySentenceTransformerWrapper(model)

    print("🔧 Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embedding=embedding_model)

    print(f"📂 Saving FAISS index to: {index_folder}")
    vectorstore.save_local(index_folder)

    print("✅ Done. Embeddings + index saved.")
