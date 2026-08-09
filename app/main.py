import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schemas import UploadResponse, AskRequest, AskResponse
from app.ingest import build_vectorstore_from_pdf
from app.qa import answer_question

app = FastAPI(title="KT RAG Agent")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Simple in-memory store: single active vectorstore (demo scope, not multi-user)
_state = {"vectorstore": None, "filename": None}


@app.get("/")
def root():
    return {"status": "KT RAG Agent API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ui")
def rag_ui():
    return FileResponse("app/static/index.html")


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        vectorstore, chunk_count = build_vectorstore_from_pdf(save_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {e}")

    _state["vectorstore"] = vectorstore
    _state["filename"] = file.filename

    return UploadResponse(
        message=f"'{file.filename}' indexed successfully.",
        chunks_indexed=chunk_count
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    if _state["vectorstore"] is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet. Upload a PDF first via /upload.")

    try:
        result = answer_question(_state["vectorstore"], request.question)
        return AskResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {e}")