import chainlit as cl
import os
import uuid
import json
import sqlite3
import datetime
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Session logging
def log_session(session_id, session):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_info = session.get("user_info", {})
    last_query = session.get("last_query", "")
    last_response = session.get("last_response", "")
    with open("session_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Session ID: {session_id}\n")
        f.write(f"User Info: {json.dumps(user_info, ensure_ascii=False)}\n")
        f.write(f"Last Query: {last_query}\n")
        f.write(f"Last Response: {last_response}\n")
        f.write("-" * 60 + "\n")

# Global session store
sessions = {}

@cl.on_chat_start
async def start():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "user_info": {},
        "last_query": "",
        "last_response": "",
        "mode": None
    }
    cl.user_session.set("session_id", session_id)
    await cl.Message(
        content="Welcome to Jazz Assistant! Would you like to:\n1. Sign in with your phone number\n2. Continue anonymously\n\nPlease reply with `1` or `2`."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    session = sessions.get(session_id)
    user_input = message.content.strip()

    if session["mode"] is None:
        if user_input == "1":
            session["mode"] = "authenticated"
            await cl.Message(content="Please enter your phone number to sign in.").send()
        elif user_input == "2":
            session["mode"] = "anonymous"
            await cl.Message(content="You're now using the assistant anonymously. How can I help you today?").send()
        else:
            await cl.Message(content="Invalid choice. Please reply with `1` to sign in or `2` to continue anonymously.").send()
        return

    if session["mode"] == "authenticated" and not session["user_info"]:
        phone_number = user_input
        try:
            conn = sqlite3.connect("jazz_telco.db")
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE phone_number = ?", (phone_number,))
            row = cursor.fetchone()
            if row:
                columns = [column[0] for column in cursor.description]
                session["user_info"] = dict(zip(columns, row))
                await cl.Message(content="You're signed in. How can I assist you today?").send()
            else:
                await cl.Message(content="Sorry, we couldn't find your number. Please try again.").send()
            conn.close()
        except Exception as e:
            await cl.Message(content=f"Database error: {str(e)}").send()
        return

    # Send the message to FastAPI backend
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("http://127.0.0.1:8000/chat", json={
                "message": user_input,
                "session_id": session_id,
                "phone_number": session["user_info"].get("phone_number", session_id)
            })

            response.raise_for_status()
            result = response.json()
            reply = result.get("response", "Sorry, I couldn't process that.")

        session["last_query"] = user_input
        session["last_response"] = reply
        log_session(session_id, session)

        await cl.Message(content=reply).send()

    except Exception as e:
        await cl.Message(content=f"Backend error: {str(e)}").send()
