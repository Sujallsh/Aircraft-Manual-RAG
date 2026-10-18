from __future__ import annotations

import threading
from pathlib import Path

from .generation import answer
from .ingestion import ingest_manuals, load_chunks, save_chunks
from .models import Chunk
from .retrieval import GraphRAGIndex, LightRAGIndex, TfidfIndex


class AircraftAssistant:
    def __init__(self, root: Path):
        self.root = root
        self.index_path = root / "data" / "chunks.jsonl"
        self._lock = threading.Lock()
        self.chunks: list[Chunk] = []
        self.engines: dict[str, object] = {}
        if self.index_path.exists():
            self._build(load_chunks(self.index_path))

    def _build(self, chunks: list[Chunk]) -> None:
        vector = TfidfIndex(chunks)
        self.chunks = chunks
        self.engines = {
            "rag": vector,
            "graphrag": GraphRAGIndex(chunks, vector),
            "lightrag": LightRAGIndex(chunks, vector),
        }

    def ingest(self) -> int:
        with self._lock:
            chunks = ingest_manuals(self.root)
            save_chunks(chunks, self.index_path)
            self._build(chunks)
            return len(chunks)

    def ask(self, question: str, engine: str = "rag", top_k: int = 5) -> dict:
        if engine not in self.engines:
            if not self.engines:
                raise RuntimeError("The index is empty. Download and ingest the manuals first.")
            raise ValueError(f"Unknown engine '{engine}'. Choose rag, graphrag, or lightrag.")
        results = self.engines[engine].search(question, top_k=top_k)
        return {
            "answer": answer(question, results),
            "engine": engine,
            "sources": [result.to_dict() for result in results],
        }

    def status(self) -> dict:
        return {"ready": bool(self.engines), "chunks": len(self.chunks), "engines": list(self.engines)}

