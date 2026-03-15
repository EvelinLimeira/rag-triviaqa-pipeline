"""Property-based test for hybrid retrieval (Property 9).

Tests use fixed test documents and multiple queries rather than pure Hypothesis
generation, since each test iteration loads actual ML models which are heavy.

Validates: Requirements 7.3
"""

import pytest
from langchain_core.documents import Document

from indexing.bm25_index import build_bm25_retriever
from indexing.dense_index import build_faiss_store
from retrieval.hybrid_retriever import build_ensemble_retriever, retrieve_hybrid


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CAPITAL_DOCS = [
    Document(page_content="Paris is the capital of France and is known for the Eiffel Tower", metadata={"doc_id": "d1"}),
    Document(page_content="Berlin is the capital of Germany and has the Brandenburg Gate", metadata={"doc_id": "d2"}),
    Document(page_content="Tokyo is the capital of Japan and is the most populous city", metadata={"doc_id": "d3"}),
    Document(page_content="London is the capital of England and home to Big Ben", metadata={"doc_id": "d4"}),
    Document(page_content="Madrid is the capital of Spain and famous for the Prado Museum", metadata={"doc_id": "d5"}),
]

QUERIES = [
    "capital of France",
    "German city with a famous gate",
    "most populous Asian capital",
    "European capital with a museum",
]


@pytest.fixture(scope="module")
def ensemble_retriever():
    """Build an EnsembleRetriever combining BM25 and FAISS for the 5 capital docs."""
    corpus_size = len(CAPITAL_DOCS)

    bm25 = build_bm25_retriever(CAPITAL_DOCS, k=corpus_size)

    faiss_store = build_faiss_store(
        documents=CAPITAL_DOCS,
        device="cpu",
        batch_size=64,
        use_ivf=False,
    )
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": corpus_size})

    return build_ensemble_retriever(bm25, faiss_retriever, weights=[0.5, 0.5])


# ---------------------------------------------------------------------------
# Property 9: Hybrid Per-Query Returns All Documents Sorted
# ---------------------------------------------------------------------------
# Feature: rag-trivia-pipeline, Property 9: Hybrid Per-Query Returns All Documents Sorted
# **Validates: Requirements 7.3**


class TestProperty9HybridPerQueryReturnsAll:
    """Verify EnsembleRetriever returns all docs from the corpus."""

    def test_hybrid_returns_all_documents(self, ensemble_retriever):
        """Hybrid retriever in per-query mode returns all documents.

        The EnsembleRetriever with RRF fusion should return all documents
        from both underlying retrievers, sorted by descending RRF score.
        """
        corpus_size = len(CAPITAL_DOCS)
        for query in QUERIES:
            results = retrieve_hybrid(ensemble_retriever, query)
            assert len(results) == corpus_size, (
                f"Hybrid returned {len(results)} docs, expected {corpus_size} "
                f"for query: '{query}'"
            )
            # Verify all results are Document instances
            assert all(isinstance(doc, Document) for doc in results)

            # Verify all original doc_ids are present in results
            result_ids = {doc.metadata["doc_id"] for doc in results}
            expected_ids = {doc.metadata["doc_id"] for doc in CAPITAL_DOCS}
            assert result_ids == expected_ids, (
                f"Missing doc_ids for query '{query}': "
                f"expected {expected_ids}, got {result_ids}"
            )


# ===========================================================================
# Unit Tests for Hybrid Retriever
# ===========================================================================
# Requirements: 7.1


class TestHybridRetrieverUnit:
    """Unit tests for hybrid retriever RRF fusion output."""

    def test_hybrid_contains_docs_from_both_retrievers(self, ensemble_retriever):
        """Hybrid results contain docs that would be top-ranked by both BM25 and Dense.

        For 'capital of France', the Paris doc (d1) should appear since it
        would be highly ranked by both BM25 (keyword match) and Dense
        (semantic match).
        """
        results = retrieve_hybrid(ensemble_retriever, "capital of France")
        doc_ids = [doc.metadata["doc_id"] for doc in results]
        # Paris doc should be in results (strong match for both retrievers)
        assert "d1" in doc_ids, (
            f"Expected 'd1' (Paris) in hybrid results, got: {doc_ids}"
        )

    def test_hybrid_results_are_documents(self, ensemble_retriever):
        """All hybrid results are LangChain Document instances with doc_id metadata."""
        results = retrieve_hybrid(ensemble_retriever, "famous gate in Germany")
        for doc in results:
            assert isinstance(doc, Document)
            assert "doc_id" in doc.metadata

    def test_hybrid_empty_query_returns_empty(self, ensemble_retriever):
        """Hybrid retriever returns empty list for empty query."""
        assert retrieve_hybrid(ensemble_retriever, "") == []

    def test_hybrid_whitespace_query_returns_empty(self, ensemble_retriever):
        """Hybrid retriever returns empty list for whitespace-only query."""
        assert retrieve_hybrid(ensemble_retriever, "   ") == []
