from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .models import Chunk

WORD_RE = re.compile(r"\S+")
HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 /&()\-]{5,80}$")


def _clean(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u00ad", "")
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def _section_for(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if HEADING_RE.match(line) and len(line.split()) <= 10:
            return line.title()
    return ""


def chunk_page(
    text: str,
    *,
    source: str,
    manufacturer: str,
    aircraft: str,
    page: int,
    size: int = 220,
    overlap: int = 40,
) -> list[Chunk]:
    clean = _clean(text)
    words = WORD_RE.findall(clean)
    if not words:
        return []
    chunks: list[Chunk] = []
    step = max(1, size - overlap)
    section = _section_for(clean)
    for start in range(0, len(words), step):
        part = words[start : start + size]
        if len(part) < 25 and chunks:
            break
        body = " ".join(part)
        digest = hashlib.sha1(
            f"{source}:{page}:{start}:{body[:80]}".encode("utf-8")
        ).hexdigest()[:12]
        chunks.append(
            Chunk(
                id=digest,
                text=body,
                source=source,
                manufacturer=manufacturer,
                aircraft=aircraft,
                page=page,
                section=section,
            )
        )
    return chunks


def load_pdf(path: Path, metadata: dict) -> list[Chunk]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc

    reader = PdfReader(str(path))
    chunks: list[Chunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks.extend(
            chunk_page(
                text,
                source=metadata["title"],
                manufacturer=metadata["manufacturer"],
                aircraft=metadata["aircraft"],
                page=page_number,
            )
        )
    return chunks


def ingest_manuals(root: Path) -> list[Chunk]:
    sources = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    all_chunks: list[Chunk] = []
    missing: list[str] = []
    for source in sources:
        path = root / "manuals" / source["filename"]
        if not path.exists():
            missing.append(source["filename"])
            continue
        all_chunks.extend(load_pdf(path, source))
    if missing:
        raise FileNotFoundError(
            "Missing manuals: " + ", ".join(missing) + ". Run: python download_manuals.py"
        )
    return all_chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks),
        encoding="utf-8",
    )


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]

