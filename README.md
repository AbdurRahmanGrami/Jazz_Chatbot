# Jazz_Chatbot

A helpful AI-powered customer support chatbot for Jazz Pakistan customers. It answers queries related to Jazz prepaid, postpaid, internet packages, and offers using a Retrieval-Augmented Generation (RAG) approach.

---

## Features

- Uses LangChain with FAISS for fast retrieval of relevant Jazz FAQs and offer documents
- Embeds text with SentenceTransformers for accurate semantic search
- Supports multi-turn conversations with memory stored in SQLite
- Handles follow-up questions by incorporating recent chat history in prompts
- Applies safety and ethical guardrails to avoid misinformation and protect user privacy
- Built with FastAPI backend and supports REST API calls for chat
- Easily switchable between different LLM providers (e.g., Azure OpenAI, Together AI, Mistral)
- Custom prompt engineering to improve answer accuracy and context understanding

---

## Tech Stack

- Python 3.9+
- FastAPI
- LangChain (for prompt chaining and retrieval)
- FAISS (vector similarity search)
- SentenceTransformers (for embeddings)
- SQLite (chat session memory and user profile storage)
- Azure OpenAI / Together AI / Other LLM providers

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/jazz-assistant.git
   cd jazz-assistant
   
2. Create and activate a virtual environment:
  python -m venv .venv
   On Linux/macOS:
  source .venv/bin/activate
   On Windows PowerShell:
  .\.venv\Scripts\activate

3. Install Dependencies
   pip install -r requirements.txt

4. Prepare environment variables in a .env file:
  OPENAI_API_KEY=your_api_key_here
  OPENAI_BASE_URL=your_api_base_url_here
  // Add other keys as needed

5. Download the SentenceTransformer model and place it in the specified folder (update the path in code if needed).

6. Load your Jazz offer documents and build the FAISS vectorstore.

7. Run the FastAPI server:
   uvicorn backend_api:app --reload

8. Access the API at http://127.0.0.1:8000.
