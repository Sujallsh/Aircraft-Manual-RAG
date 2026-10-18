from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_system import AircraftAssistant

ROOT = Path(__file__).resolve().parent
assistant = AircraftAssistant(ROOT)
app = FastAPI(title="Aircraft Manual RAG", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    engine: str = "rag"
    top_k: int = Field(default=5, ge=1, le=10)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status() -> dict:
    return assistant.status()


@app.post("/api/ingest")
def ingest() -> dict:
    try:
        count = assistant.ingest()
        return {"message": "Manuals indexed", "chunks": count}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    try:
        return assistant.ask(request.question, request.engine, request.top_k)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

