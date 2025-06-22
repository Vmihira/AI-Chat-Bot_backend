from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
from datetime import datetime
import json
from io import BytesIO
import fitz  # PyMuPDF
import chromadb
from rag import generate, insert_document, query_document

client = chromadb.PersistentClient()

# main FastAPI application

app = FastAPI(title="AI Document Chat API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: str

class CreateSession(BaseModel):
    session_name: str

class SessionResponse(BaseModel):
    session_id: str
    session_name: str
    created_at: str
    message_count: int

class MessageResponse(BaseModel):
    message_id: str
    content: str
    sender: str
    timestamp: str

# In-memory storage (replace with database in production)
sessions_db: Dict[str, Dict[str, Any]] = {}
chats_db: Dict[str, List[Dict[str, Any]]] = {}

@app.get("/")
async def root():
    return {"message": "AI Document Chat API is running!"}

@app.post("/create-session")
async def create_session(session_data: CreateSession):
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    session_info = {
        "session_id": session_id,
        "session_name": session_data.session_name,
        "created_at": datetime.now().isoformat(),
        "message_count": 0,
        "documents": []
    }

    data = {}

    with open("data.json" , "r") as f:

        temp = json.load(f)
        temp[session_id] = session_info
        data = temp
    # Store session info in memory (or database)
    
    with open("data.json" , "w") as f:

        json.dump(data, f, indent=4)

    chats_data = []    

    collection = client.get_or_create_collection(name=session_id)


    with open("chats.json", "r") as f:
        temp_chats = json.load(f)
        temp_chats[session_id] = []
        chats_data = temp_chats

    with open("chats.json" , "w") as f:

        json.dump(chats_data, f, indent=4)    

    return {
        "session_id": session_id,
        "message": "Session created successfully"
    }

@app.get("/sessions")
async def get_sessions():
    """Get all chat sessions"""
    sessions_list = []

    sessions_db = {}
    with open("data.json", "r") as f:
        sessions_db = json.load(f)

    chats_db = {}
    with open("chats.json", "r") as f:
        chats_db = json.load(f)

    for session_id, session_info in sessions_db.items():
        sessions_list.append({
            "session_id": session_id,
            "session_name": session_info["session_name"],
            "created_at": session_info["created_at"],
            "message_count": len(chats_db.get(session_id, []))
        })
    
    # Sort by creation date (newest first)
    sessions_list.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {"sessions": sessions_list}

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...), session_id: str = Form(...)):
    """Upload a document to a specific session"""
    

    
    sessions_db = {}
    with open("data.json", "r") as f:
        sessions_db = json.load(f)

    pdf_bytes = await file.read()
    pdf_stream = BytesIO(pdf_bytes)

        # Open directly from memory
    doc = fitz.open(stream=pdf_stream, filetype='pdf')
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()    

    insert_document(session_id, text)
   
    document_info = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file.size if hasattr(file, 'size') else 0,
        "uploaded_at": datetime.now().isoformat()
    }
    
    sessions_db[session_id]["documents"].append(document_info)
    
    return {
        "message": f"Document {file.filename} uploaded successfully",
        "document_info": document_info
    }

@app.post("/chat")
async def chat_with_ai(chat_data: ChatMessage):
    """Send a message and get AI response"""
    
    # Store user message
    user_message = {
        "message_id": str(uuid.uuid4()),
        "content": chat_data.message,
        "sender": "user",
        "timestamp": datetime.now().isoformat()
    }

    chats_db = []

    with open("chats.json", "r") as f:
        temp_chats = json.load(f)
        if chat_data.session_id not in temp_chats:
            temp_chats[chat_data.session_id] = []
        chats_db = temp_chats


    result = query_document(chat_data.session_id, chat_data.message)

    result = chat_data.message + "\n======================================================= References ======================================================================\n" + result    


    
    chats_db[chat_data.session_id].append(user_message)
    
    # Generate AI response (replace with actual AI logic)
    # This is where you would:
    # 1. Retrieve relevant document chunks based on the query
    # 2. Create a prompt with context from documents
    # 3. Call your AI model (OpenAI, Anthropic, etc.)
    # 4. Return the generated response
    
    sample_responses = [
        "Based on the documents you've uploaded, I can help you with that question.",
        "Let me analyze the content from your documents to provide a comprehensive answer.",
        "I've found relevant information in your uploaded documents. Here's what I can tell you:",
        "According to the documents in this session, here's my analysis:",
        "I can see from the uploaded materials that this relates to several key points."
    ]
    
    ai_response = generate(result)

    ai_message = {
        "message_id": str(uuid.uuid4()),
        "content": ai_response,
        "sender": "assistant",
        "timestamp": datetime.now().isoformat()
    }
    
    chats_db[chat_data.session_id].append(ai_message)

    print("====================================================")

    with open("chats.json", "w") as f:
        json.dump(chats_db, f, indent=4)
    
    return {"response": ai_response}

@app.get("/retrieve-chats/{session_id}")
async def retrieve_chats(session_id: str):
    """Retrieve chat history for a specific session"""
    
    
    chats_db = {}
    with open("chats.json", "r") as f:
        chats_db = json.load(f)
    
    messages = chats_db.get(session_id, [])
    
    return {
        "session_id": session_id,
        "messages": messages,
        "total_messages": len(messages)
    }

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get detailed information about a specific session"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    sessions_db = {}
    with open("data.json", "r") as f:
        sessions_db = json.load(f)
    chats_db = {}
    with open("chats.json", "r") as f:
        chats_db = json.load(f)
    
    session_info = sessions_db[session_id].copy()
    session_info["message_count"] = len(chats_db.get(session_id, []))
    
    return session_info

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session and all its data"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions_db[session_id]
    if session_id in chats_db:
        del chats_db[session_id]
    
    return {"message": "Session deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
