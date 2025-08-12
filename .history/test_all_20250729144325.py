import pytest
from httpx import AsyncClient
from backend_api import app, save_to_memory, get_session_history

@pytest.mark.asyncio
async def test_chat_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/chat", json={
            "message": "Hi, what offers do you have?",
            "session_id": "test-session-id",
            "email": "test@example.com"
        })
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert isinstance(data["response"], str)

def test_save_and_retrieve_memory():
    session_id = "test-session-id-2"
    save_to_memory(session_id, "human", "Hello?")
    save_to_memory(session_id, "ai", "Hi, how can I help you?")
    history = get_session_history(session_id)
    assert "User: Hello?" in history
    assert "Assistant: Hi, how can I help you?" in history
