# offline-rag-system/backend/main.py
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os
import json
import io
from cryptography.fernet import Fernet
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
import easyocr
import whisper
import logging
import uuid
from docx import Document as DocxDocument
from pypdf import PdfReader

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
SECRET_KEY = "your-secret-key"  # Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Users stored in JSON file for persistence
USERS_FILE = "users.json"
if not os.path.exists(USERS_FILE):
    # Initial users
    initial_users = {
        "admin": {"username": "admin", "hashed_password": pwd_context.hash("adminpass"), "role": "Admin"}
    }
    with open(USERS_FILE, "w") as f:
        json.dump(initial_users, f)

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# Encryption key (generate once and store securely)
ENCRYPTION_KEY = Fernet.generate_key()
fernet = Fernet(ENCRYPTION_KEY)

# Directories
os.makedirs("encrypted_files", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("user_memory", exist_ok=True)  # For persistent user memory

# Logging
logging.basicConfig(filename="logs/audit.log", level=logging.INFO)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector DB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_collection")

# Local LLM
llm = OllamaLLM(model="phi")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    username: str
    role: str

class NewUser(BaseModel):
    username: str
    password: str
    role: str  # Admin, Analyst, Guest

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username: str):
    users = load_users()
    return users.get(username)

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception
    return User(username=user["username"], role=user["role"])

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"]}, expires_delta=access_token_expires)
    logging.info(f"Login: {user['username']} at {datetime.now()}")
    return {"access_token": access_token, "token_type": "bearer"}

# User Management (Admin only)
@app.get("/users/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.get("/users")
async def list_users(current_user: User = Depends(get_current_user)):
    if current_user.role != "Admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    users = load_users()
    return [{"username": u, "role": users[u]["role"]} for u in users]

@app.post("/users")
async def create_user(new_user: NewUser, current_user: User = Depends(get_current_user)):
    if current_user.role != "Admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    users = load_users()
    if new_user.username in users:
        raise HTTPException(status_code=400, detail="User already exists")
    hashed_password = pwd_context.hash(new_user.password)
    users[new_user.username] = {"username": new_user.username, "hashed_password": hashed_password, "role": new_user.role}
    save_users(users)
    logging.info(f"User created: {new_user.username} by {current_user.username} at {datetime.now()}")
    return {"message": "User created"}

@app.delete("/users/{username}")
async def delete_user(username: str, current_user: User = Depends(get_current_user)):
    if current_user.role != "Admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    if username == "admin":
        raise HTTPException(status_code=400, detail="Cannot delete admin")
    users = load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    del users[username]
    save_users(users)
    logging.info(f"User deleted: {username} by {current_user.username} at {datetime.now()}")
    return {"message": "User deleted"}

# Ingestion
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    if current_user.role not in ["Admin", "Analyst"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    content = await file.read()
    encrypted_content = fernet.encrypt(content)
    file_id = str(uuid.uuid4())
    encrypted_path = f"encrypted_files/{file_id}"
    with open(encrypted_path, "wb") as f:
        f.write(encrypted_content)
    
    # Process based on type without temp files where possible
    text = ""
    file_stream = io.BytesIO(content)
    
    if file.filename.endswith(".pdf"):
        reader = PdfReader(file_stream)
        text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file.filename.endswith(".docx"):
        doc = DocxDocument(file_stream)
        text = " ".join(para.text for para in doc.paragraphs)
    else:
        # For image and audio, use temp file
        temp_filename = f"temp_{file_id}"
        with open(temp_filename, "wb") as temp:
            temp.write(content)
        
        if file.filename.lower().endswith((".jpg", ".png")):
            reader = easyocr.Reader(['en'])
            text = " ".join(reader.readtext(temp_filename, detail=0))
        elif file.filename.endswith(".mp3") or file.filename.endswith(".wav"):
            model = whisper.load_model("base")
            result = model.transcribe(temp_filename)
            text = result["text"]
        
        os.remove(temp_filename)
    
    # Embed and store
    if text:
        # In latest, add to collection via langchain Chroma
        vectorstore = Chroma(client=client, collection_name="rag_collection", embedding_function=embeddings)
        metadata = {"file_id": file_id, "source": file.filename, "uploader": current_user.username}
        doc = Document(page_content=text, metadata=metadata)
        vectorstore.add_documents([doc])
    
    logging.info(f"Upload: {file.filename} by {current_user.username} at {datetime.now()}")
    return {"message": "File uploaded and ingested"}

# RAG Query
class Query(BaseModel):
    question: str

@app.post("/query")
async def query_rag(query: Query, current_user: User = Depends(get_current_user)):
    if current_user.role == "Guest":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    vectorstore = Chroma(client=client, collection_name="rag_collection", embedding_function=embeddings)
    
    # Filter based on role
    search_kwargs = {"k": 5}
    if current_user.role != "Admin":
        search_kwargs["filter"] = {"uploader": current_user.username}
    
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    # Simple RAG without deprecated chain
    docs = await retriever.ainvoke(query.question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"Answer the question based only on the following context:\n{context}\n\nQuestion: {query.question}"
    answer = llm.invoke(prompt)
    citations = [{"text": f"[{i+1}] {doc.metadata['source']}", "file_id": doc.metadata['file_id']} for i, doc in enumerate(docs)]
    
    # Save to user memory (persistent history)
    timestamp = datetime.now().isoformat()
    memory_entry = {
        "timestamp": timestamp,
        "user": current_user.username,
        "query": query.question,
        "answer": answer,
        "citations": [cit["text"] for cit in citations]
    }
    memory_path = f"user_memory/{current_user.username}.json"
    history = []
    if os.path.exists(memory_path):
        with open(memory_path, "r") as f:
            history = json.load(f)
    history.append(memory_entry)
    with open(memory_path, "w") as f:
        json.dump(history, f, indent=4)
    
    logging.info(f"Query: {query.question} by {current_user.username} at {datetime.now()}")
    return {"answer": answer, "citations": citations}

# Get User Memory (History)
@app.get("/memory")
async def get_memory(current_user: User = Depends(get_current_user)):
    memory_path = f"user_memory/{current_user.username}.json"
    if not os.path.exists(memory_path):
        return {"history": []}
    with open(memory_path, "r") as f:
        history = json.load(f)
    return {"history": history}

# View source (decrypt on fly)
@app.get("/view_source/{file_id}")
async def view_source(file_id: str, current_user: User = Depends(get_current_user)):
    if current_user.role not in ["Admin", "Analyst"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Check if user can access this file
    results = collection.get(ids=[file_id], include=["metadatas"])
    if not results["metadatas"]:
        raise HTTPException(status_code=404, detail="File not found")
    metadata = results["metadatas"][0]
    if current_user.role != "Admin" and metadata.get("uploader") != current_user.username:
        raise HTTPException(status_code=403, detail="Insufficient permissions to view this file")
    
    path = f"encrypted_files/{file_id}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    
    with open(path, "rb") as f:
        encrypted_content = f.read()
    content = fernet.decrypt(encrypted_content)
    return {"content": content.decode(errors="ignore")}  # Assume text for simplicity

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)