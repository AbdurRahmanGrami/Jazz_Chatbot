import pytest
import hashlib
from httpx import AsyncClient

# Import your FastAPI app and functions
from backend_api import app, save_to_memory, get_session_history

from fastapi.testclient import TestClient
import asyncio
import pytest
from backend_api import app

client = TestClient(app)

@pytest.mark.asyncio
async def test_chat_endpoint():
    payload = {
        "email": "test@example.com",
        "message": "What are the postpaid packages?",
        "session_id": "abc123"
    }

    response = await asyncio.to_thread(client.post, "/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)


# ------------------ SESSION MEMORY ------------------
def test_save_and_retrieve_memory():
    session_id = "test-session-id-2"
    save_to_memory(session_id, "human", "Hello?")
    save_to_memory(session_id, "ai", "Hi, how can I help you?")
    history = get_session_history(session_id)
    assert "User: Hello?" in history
    assert "Assistant: Hi, how can I help you?" in history

