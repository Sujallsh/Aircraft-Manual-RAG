from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_system import AircraftAssistant

ROOT = Path(__file__).resolve().parent
QUESTIONS = [
    "What information is provided about A320 ground clearances?",
    "What is the purpose of the Boeing 737 airport planning document?",
    "Compare the manuals' intended audiences.",
    "Which sections discuss terminal servicing?",
    "What engine options are mentioned for the A320?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/evaluation.json")
    args = parser.parse_args()
    assistant = AircraftAssistant(ROOT)
    if not assistant.status()["ready"]:
        raise SystemExit("Index missing. Start the app and run ingestion first.")
    rows = []
    for question in QUESTIONS:
        for engine in ("rag", "graphrag", "lightrag"):
            result = assistant.ask(question, engine, top_k=5)
            rows.append(
                {
                    "question": question,
                    "engine": engine,
                    "top_score": result["sources"][0]["score"] if result["sources"] else 0,
                    "source_diversity": len({item["manufacturer"] for item in result["sources"]}),
                    "citations": [item["citation"] for item in result["sources"]],
                }
            )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} evaluation rows to {output}")


if __name__ == "__main__":
    main()

