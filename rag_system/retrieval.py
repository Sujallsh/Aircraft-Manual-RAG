from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .models import Chunk, SearchResult

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+./-]{1,}", re.I)
STOPWORDS = {
    "about", "after", "aircraft", "also", "and", "are", "been", "being", "between",
    "can", "data", "does", "each", "for", "from", "has", "have", "into", "its", "may",
    "more", "not", "page", "shall", "should", "such", "than", "that", "the", "their",
    "then", "there", "these", "this", "through", "under", "using", "was", "were", "which",
    "will", "with", "you",
}


def tokenize(text: str) -> list[str]:
    return [token.lower().strip("./-") for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]


class TfidfIndex:
    """Small dependency-free vector index suitable for a coursework demo."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        tokenized = [tokenize(chunk.text) for chunk in chunks]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        count = max(1, len(chunks))
        self.idf = {
            term: math.log((count + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }
        self.vectors = [self._vector(tokens) for tokens in tokenized]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = max(1, sum(counts.values()))
        vector = {term: (frequency / total) * self.idf.get(term, 0) for term, frequency in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {term: value / norm for term, value in vector.items()}

    def scores(self, query: str) -> list[float]:
        query_vector = self._vector(tokenize(query))
        return [sum(weight * vector.get(term, 0) for term, weight in query_vector.items()) for vector in self.vectors]

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        ranked = sorted(enumerate(self.scores(query)), key=lambda item: item[1], reverse=True)
        return [
            SearchResult(self.chunks[index], score, ["TF-IDF similarity"])
            for index, score in ranked[:top_k]
            if score > 0
        ]


def key_terms(text: str, limit: int = 14) -> list[str]:
    counts = Counter(token for token in tokenize(text) if len(token) >= 3)
    return [term for term, _ in counts.most_common(limit)]


class GraphRAGIndex:
    """GraphRAG-style retrieval over a term co-occurrence knowledge graph.

    Terms are nodes, co-occurrence within a manual chunk forms weighted edges,
    and chunks connect evidence back to source pages.
    """

    def __init__(self, chunks: list[Chunk], vector_index: TfidfIndex):
        self.chunks = chunks
        self.vector = vector_index
        self.chunk_terms: list[set[str]] = []
        self.neighbors: dict[str, Counter[str]] = defaultdict(Counter)
        self.term_chunks: dict[str, set[int]] = defaultdict(set)
        for index, chunk in enumerate(chunks):
            terms = set(key_terms(chunk.text))
            self.chunk_terms.append(terms)
            for term in terms:
                self.term_chunks[term].add(index)
            ordered = sorted(terms)
            for left_index, left in enumerate(ordered):
                for right in ordered[left_index + 1 :]:
                    self.neighbors[left][right] += 1
                    self.neighbors[right][left] += 1

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        anchors = {term for term in query_terms if term in self.term_chunks}
        expanded: Counter[str] = Counter()
        for anchor in anchors:
            for neighbor, weight in self.neighbors[anchor].most_common(8):
                expanded[neighbor] += weight
        maximum = max(expanded.values(), default=1)
        vector_scores = self.vector.scores(query)
        ranked: list[tuple[int, float, list[str]]] = []
        for index, terms in enumerate(self.chunk_terms):
            direct = len(anchors & terms) / max(1, len(anchors))
            graph = sum(expanded[term] / maximum for term in terms if term in expanded) / max(1, len(terms))
            score = 0.50 * vector_scores[index] + 0.30 * direct + 0.20 * graph
            reasons = ["semantic similarity"]
            if direct:
                reasons.append("query entities found in chunk")
            if graph:
                reasons.append("connected graph concepts found")
            ranked.append((index, score, reasons))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [SearchResult(self.chunks[index], score, reasons) for index, score, reasons in ranked[:top_k] if score > 0]


class LightRAGIndex:
    """LightRAG-style dual-level retrieval with local and global signals."""

    def __init__(self, chunks: list[Chunk], vector_index: TfidfIndex):
        self.chunks = chunks
        self.vector = vector_index
        self.local_terms = [set(key_terms(chunk.text, 20)) for chunk in chunks]
        self.global_terms = [
            set(tokenize(f"{chunk.manufacturer} {chunk.aircraft} {chunk.section} {chunk.source}"))
            for chunk in chunks
        ]

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        vector_scores = self.vector.scores(query)
        ranked: list[tuple[int, float, list[str]]] = []
        for index, (local, global_) in enumerate(zip(self.local_terms, self.global_terms)):
            local_score = len(query_terms & local) / max(1, len(query_terms))
            global_score = len(query_terms & global_) / max(1, len(query_terms))
            score = 0.55 * vector_scores[index] + 0.30 * local_score + 0.15 * global_score
            reasons = ["vector channel"]
            if local_score:
                reasons.append("local entity channel")
            if global_score:
                reasons.append("global topic channel")
            ranked.append((index, score, reasons))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [SearchResult(self.chunks[index], score, reasons) for index, score, reasons in ranked[:top_k] if score > 0]

