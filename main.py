from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import ollama
import os
import fitz
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI(title="PDF Assistant", description="RAG-based PDF chatbot", version="1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load or create vector database
print("Starting...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists("./chroma_db"):
    print("Loading existing database...")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    db = None
    print("No database found. Please upload a PDF.")

print("Ready!\n")

# Chat history with system prompt
chat_history = [{"role": "system", "content": """You are a helpful assistant that answers questions based on the provided document.
Always base your answers on the document content.
If the question is conversational, respond naturally and friendly."""}]

# Data models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

def is_conversational(question: str) -> bool:
    """Check if the message is a casual greeting or conversational message."""
    keywords = ["hello", "hi", "hey", "thanks", "goodbye", "bye", "how are you", "hola", "gracias", "adios"]
    return any(keyword in question.lower() for keyword in keywords)

# ----- ENDPOINTS ----- #

@app.get("/")
async def root():
    """Serve the main HTML page."""
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a question and return an answer based on the document."""
    global db, chat_history

    if db is None:
        return ChatResponse(answer="Please upload a PDF document first.")

    if is_conversational(request.question):
        question_with_context = f"{request.question}\n\nRespond naturally and friendly as a document assistant."
    else:
        results = db.similarity_search(request.question, k=6)
        context = "\n\n".join([r.page_content for r in results])
        question_with_context = f"""Context from the document:\n{context}\n\nQuestion: {request.question}"""

    chat_history.append({"role": "user", "content": question_with_context})

    response = ollama.chat(model="llama3.1", messages=chat_history)
    answer = response.message.content

    chat_history.append({"role": "assistant", "content": answer})

    return ChatResponse(answer=answer)

@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    """Upload a PDF and process it into the vector database."""
    global db

    if not pdf.filename.endswith(".pdf"):
        return {"error": "Only PDF files are accepted"}

    # Save the PDF
    pdf_path = f"./{pdf.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    # Process PDF
    print(f"Processing {pdf.filename}...")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    fragments = splitter.create_documents([full_text])

    # Delete existing collection and recreate
    if db is not None:
        db.delete_collection()

    db = Chroma.from_documents(fragments, embeddings, persist_directory="./chroma_db")

    print(f"{pdf.filename} processed successfully.")
    return {"message": f"{pdf.filename} processed successfully"}