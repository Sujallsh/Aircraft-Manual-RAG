from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.ingestion import chunk_page
from rag_system.retrieval import GraphRAGIndex, LightRAGIndex, TfidfIndex


def sample_chunks():
    return (
        chunk_page(
            "GENERAL AIRCRAFT DIMENSIONS The Airbus A320 wingspan and overall length are listed for airport planning. " * 5,
            source="Airbus A320 Manual", manufacturer="Airbus", aircraft="A320", page=10, size=80, overlap=10,
        )
        + chunk_page(
            "TERMINAL SERVICING The Boeing 737 passenger doors and ground service connections support airport planning. " * 5,
            source="Boeing 737 Manual", manufacturer="Boeing", aircraft="737 NG", page=20, size=80, overlap=10,
        )
    )


class RetrievalTests(unittest.TestCase):
    def test_baseline_rag_finds_airbus_dimensions(self):
        chunks = sample_chunks()
        index = TfidfIndex(chunks)
        result = index.search("A320 dimensions wingspan", top_k=1)[0]
        self.assertEqual(result.chunk.manufacturer, "Airbus")

    def test_graphrag_finds_connected_service_concepts(self):
        chunks = sample_chunks()
        vector = TfidfIndex(chunks)
        result = GraphRAGIndex(chunks, vector).search("737 terminal passenger service", top_k=1)[0]
        self.assertEqual(result.chunk.manufacturer, "Boeing")

    def test_lightrag_uses_global_aircraft_context(self):
        chunks = sample_chunks()
        vector = TfidfIndex(chunks)
        result = LightRAGIndex(chunks, vector).search("Boeing 737", top_k=1)[0]
        self.assertEqual(result.chunk.aircraft, "737 NG")


if __name__ == "__main__":
    unittest.main()
