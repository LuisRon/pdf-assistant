# PDF Assistant

A RAG-based chatbot that answers questions about any document in PDF format.
Built with FastAPI and LangChain, powered by local LLMs via Ollama.

## Features
- Upload any PDF directly from the web interface
- Ask questions about the document content
- Conversational chat with context memory
- Auto-generated API documentation with Swagger UI
- Fast and modern async API with FastAPI

## Tech Stack
- Python + FastAPI
- Ollama (llama3.1)
- LangChain + ChromaDB
- HuggingFace Embeddings
- Pydantic for data validation

## Requirements
- Ollama installed with llama3.1 model (`ollama pull llama3.1`)
- Python 3.10+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LuisRon/pdf-assistant.git
cd pdf-assistant
```

2. Create and activate a virtual environment:
```bash
py -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
uvicorn main:app --reload
```

5. Open your browser at `http://localhost:8000`

## API Documentation
FastAPI generates interactive API documentation automatically.
Access it at `http://localhost:8000/docs`

## How it works
Upload any PDF using the interface. The app splits it into fragments, stores them
in a ChromaDB vector database, and uses llama3.1 to answer questions based on
the document content. Each new PDF upload replaces the previous database.