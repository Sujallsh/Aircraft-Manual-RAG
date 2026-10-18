from __future__ import annotations

import os
import re

from .models import SearchResult


SYSTEM_PROMPT = """You answer questions about Airbus and Boeing aircraft using only the supplied manual excerpts.
If the excerpts do not contain the answer, say that the manuals provided do not establish it.
Keep measurements and model variants exact. Add an inline citation such as [Source, p. 12] after each factual claim.
Do not give operational safety advice beyond the source text. These airport-planning documents do not replace approved flight or maintenance manuals."""


def _context(results: list[SearchResult]) -> str:
    return "\n\n".join(f"[{result.chunk.citation()}]\n{result.chunk.text}" for result in results)


def _extractive_answer(question: str, results: list[SearchResult]) -> str:
    if not results:
        return "I could not find relevant information in the indexed manuals."
    query_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    candidates: list[tuple[int, str, SearchResult]] = []
    for result in results:
        for sentence in re.split(r"(?<=[.!?])\s+", result.chunk.text):
            overlap = len(query_terms & set(re.findall(r"[a-z0-9]+", sentence.lower())))
            if overlap and len(sentence) > 35:
                candidates.append((overlap, sentence.strip(), result))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:3] or [(1, results[0].chunk.text[:500], results[0])]
    lines = [f"{sentence} [{result.chunk.citation()}]" for _, sentence, result in selected]
    return "\n\n".join(lines) + "\n\nThis extractive answer is shown because no Gemini API key is configured."


def answer(question: str, results: list[SearchResult]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _extractive_answer(question, results)
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=f"{SYSTEM_PROMPT}\n\nQUESTION:\n{question}\n\nMANUAL EXCERPTS:\n{_context(results)}",
        )
        return response.text or _extractive_answer(question, results)
    except Exception as exc:
        fallback = _extractive_answer(question, results)
        return f"{fallback}\n\nGenerator warning: {type(exc).__name__}."

