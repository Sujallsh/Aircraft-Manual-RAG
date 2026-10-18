# Task One: Airbus and Boeing RAG Dialogue System

This project completes Task One from Homework 2. It indexes two official, publicly available aircraft-planning manuals and provides one dialogue interface with three selectable retrieval strategies:

- **Baseline RAG:** chunk-level TF-IDF retrieval.
- **GraphRAG:** a term co-occurrence knowledge graph expands the query to connected concepts and reranks evidence.
- **LightRAG-style:** a lightweight dual-level retriever combines vector, local-entity, and global aircraft/topic signals.

The implementation is deliberately transparent and dependency-light so that every retrieval score can be explained in a report or demonstration. It is a coursework implementation inspired by GraphRAG and LightRAG, not a claim of API compatibility with the Microsoft GraphRAG or HKUDS LightRAG packages.

## Data sources

1. Airbus, *A320 Aircraft Characteristics: Airport and Maintenance Planning*, revision dated 1 July 2026. Source page: <https://www.aircraft.airbus.com/en/customer-care/fleet-wide-care/airport-operations-and-aircraft-characteristics/aircraft-characteristics>
2. Boeing, *Next-Generation 737 Airplane Characteristics for Airport Planning*, revision C dated October 2025. Source page: <https://www.boeing.com/commercial/airports/plan-manuals>

The PDFs are downloaded locally and excluded from Git. Follow each manufacturer's terms of use. These airport-planning documents are not substitutes for approved flight, maintenance, or safety manuals.

## Run the project

From the `homework_task1` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_manuals.py
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>, click **Build index**, select a retrieval engine, and ask a question.

If `GEMINI_API_KEY` exists in the environment, Gemini turns the retrieved passages into a concise cited response. Without a key, the system remains functional and returns the most relevant cited sentences directly from the manuals.

## API

- `GET /api/status` reports whether the index is ready.
- `POST /api/ingest` extracts, chunks, and indexes both manuals.
- `POST /api/chat` accepts `{"question":"...", "engine":"rag|graphrag|lightrag", "top_k":5}`.

## How it works

1. `pypdf` extracts text page by page.
2. The ingestion pipeline creates overlapping 220-word chunks and preserves manufacturer, aircraft, source, page, and section metadata.
3. The baseline index creates normalized TF-IDF vectors with no external vector database.
4. GraphRAG creates weighted edges between important terms that occur in the same chunk. Query terms seed a one-hop graph expansion before evidence reranking.
5. LightRAG-style retrieval fuses three channels: vector similarity, local terms in the passage, and global metadata such as manufacturer, aircraft, and section.
6. The response generator receives only the top evidence and must cite the original manual page. An extractive fallback prevents the demo from depending on a paid API.

## Evaluation

Run the automated tests:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

After indexing the manuals, create a comparison file:

```powershell
python evaluate.py
```

`data/evaluation.json` records the top retrieval score, source diversity, and citations for five questions under all three engines. For the written report, add a small human evaluation table rating answer correctness, citation accuracy, and completeness from 1 to 5.

## Suggested demonstration questions

- What ground-clearance information does the A320 manual provide?
- What is the purpose and intended audience of each manual?
- Compare the passenger-door information for the A320 and 737 NG.
- Which sections discuss terminal servicing and ground support?
- What do the manuals say about pavement loading?

## Limitations

- PDF extraction cannot reliably interpret every diagram or scanned table.
- The graph uses deterministic term co-occurrence rather than LLM-generated entity and relation extraction.
- TF-IDF works well for technical vocabulary but understands paraphrases less well than a neural embedding model.
- Retrieved statements must be verified against the cited page before use in real aviation work.
