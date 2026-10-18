from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    source: str
    manufacturer: str
    aircraft: str
    page: int
    section: str = ""

    def citation(self) -> str:
        return f"{self.source}, p. {self.page}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class SearchResult:
    chunk: Chunk
    score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "citation": self.chunk.citation(),
            "text": self.chunk.text,
            "manufacturer": self.chunk.manufacturer,
            "aircraft": self.chunk.aircraft,
            "page": self.chunk.page,
            "reasons": self.reasons,
        }

