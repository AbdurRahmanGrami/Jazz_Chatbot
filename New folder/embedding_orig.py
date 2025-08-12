import os
import json
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# --- Configuration ---
faq_files = [
    r"C:\Users\abdur.rahman\Pictures\chatbot\chatbot\faqs.json",
    r"C:\Users\abdur.rahman\Pictures\chatbot\chatbot\jazz_prepaid_AllinOneOffers.json",
    r"C:\Users\abdur.rahman\Pictures\chatbot\chatbot\jazz_postpaid_full_data.json"
]

model_path = "C:/Users/abdur.rahman/Desktop/chatbot/models/all-MiniLM-L6-v2"
index_folder = "vectorstore"
max_tokens = 300

# --- Load model and tokenizer ---
print("Loading model and tokenizer...")

from sentence_transformers import models
word_embedding_model = models.Transformer(model_path)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# --- Token length logic ---
def is_too_long(text: str, max_tokens: int = 300) -> bool:
    return len(tokenizer.encode(text, truncation=False)) > max_tokens

def split_long_text(text: str, max_tokens: int = 300) -> List[str]:
    sentences = text.split('. ')
    chunks, current = [], ""
    for sentence in sentences:
        if is_too_long(current + sentence):
            if current:
                chunks.append(current.strip())
            current = sentence + ". "
        else:
            current += sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

# --- Metadata-aware chunk builder ---
def build_embedding_text(entry: dict) -> List[Document]:
    metadata_parts = []

    if "metadata" in entry:
        for k, v in entry["metadata"].items():
            metadata_parts.append(f"{k}: {v}")
        base_text = f"Q: {entry['metadata'].get('question', '')}\nA: {entry['metadata'].get('answer', '')}"
    else:
        for k, v in entry.items():
            if k not in ["Content", "Header Context", "URL"]:
                metadata_parts.append(f"{k}: {v}")
        audience = entry.get("Header Context", [""])[0]
        url = entry.get("URL", "")
        if audience:
            metadata_parts.append(f"Audience: {audience}")
        if url:
            metadata_parts.append(f"Source: {url}")
        base_text = entry.get("Content", "")

    # Build the full text to embed
    metadata_str = "\n".join(metadata_parts)
    full_text = f"{metadata_str}\n{base_text}".strip()

    # Chunk if needed
    chunks = split_long_text(full_text) if is_too_long(full_text) else [full_text]

    # Return each chunk as a LangChain Document with full metadata
    return [Document(page_content=chunk, metadata=entry) for chunk in chunks]

# --- Main embedding pipeline ---
if __name__ == "__main__":
    print("📁 Starting embedding generation...")

    all_documents = []

    for file in faq_files:
        print(f"📄 Processing file: {file}")
        with open(file, "r", encoding="utf-8") as f:
            entries = json.load(f)
            for entry in entries:
                docs = build_embedding_text(entry)
                all_documents.extend(docs)

    print(f"🧠 Total chunks prepared: {len(all_documents)}")

    # Define embedding wrapper for LangChain
    class MySentenceTransformerWrapper:
        def __init__(self, model):
            self.model = model

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self.model.encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text: str) -> List[float]:
            return self.model.encode(text, normalize_embeddings=True).tolist()

    embedding_model = MySentenceTransformerWrapper(model)

    print("🔧 Building FAISS index...")
    vectorstore = FAISS.from_documents(all_documents, embedding=embedding_model)

    print(f"💾 Saving index to: {index_folder}")
    vectorstore.save_local(index_folder)

    print("✅ Embedding + indexing completed successfully.")
