import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ingest_pdf import ingest_pdf_bytes
from qa import answer_with_retrieval
from typing import Optional

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), source: Optional[str] = None):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    content = await file.read()
    try:
        count = ingest_pdf_bytes(content, source=source or file.filename)
        return {"ingested_chunks": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(question: str):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")
    try:
        answer, sources = answer_with_retrieval(question, k=3, model_name=os.getenv("OPENAI_MODEL", "gpt-4"))
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
