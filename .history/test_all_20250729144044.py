import pytest
import asyncio
from httpx import AsyncClient
from unittest.mock import patch, MagicMock
import os

# Import your backend FastAPI app
from backend_api import app

# ----------- AUTH TESTS -----------
def test_correct_sha_login():
    from backend_api import check_password, hash_password
    hashed = hash_password("password123")
    assert check_password("password123", hashed) == True

def test_wrong_sha_login():
    from backend_api import check_password, hash_password
    hashed = hash_password("password123")
    assert check_password("wrongpass", hashed) == False


# ----------- EMBEDDING PIPELINE TEST -----------
@patch("embedding.SentenceTransformer")
def test_embedding_pipeline(mock_model):
    mock_instance = MagicMock()
    mock_instance.encode.return_value = [[0.1] * 384]
    mock_model.return_value = mock_instance

    from embedding import embed_texts
    texts = ["Jazz offers unlimited internet"]
    vectors = embed_texts(texts)
    assert len(vectors) == 1
    assert len(vectors[0]) == 384


# ----------- FASTAPI /chat ENDPOINT TEST -----------
@pytest.mark.asyncio
@patch("backend_api.generate_response", return_value="Here is a mock response.")
async def test_chat_endpoint(mock_response):
    test_data = {"email": "user@example.com", "message": "What is Jazz?"}
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/chat", json=test_data)
        assert response.status_code == 200
        assert "mock response" in response.json()["response"]


# ----------- SESSION MEMORY TEST -----------
@patch("backend_api.save_to_memory")
def test_session_memory(mock_save):
    from backend_api import store_chat_memory
    store_chat_memory("session-1", "Hello", "Hi there")
    mock_save.assert_called_once()


# ----------- FAISS RETRIEVER TEST -----------
def test_faiss_retriever():
    retriever = get_faiss_retriever()
    results = retriever.get_relevant_documents("postpaid")
    assert isinstance(results, list)


# ----------- LLAMA RETRIEVER TEST -----------
def test_llama_retriever():
    retriever = get_llama_retriever()
    results = retriever.retrieve("internet packages")
    assert isinstance(results, list)


# ----------- LLM RESPONSE TEST -----------
@patch("backend_api.chat_llm")
def test_llm_response(mock_llm):
    mock_llm.predict.return_value = "This is a mock LLM response"
    response = generate_response("Tell me about postpaid", session_id="abc")
    assert "mock LLM" in response
