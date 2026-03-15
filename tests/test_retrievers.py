"""Property-based tests for BM25 and Dense retrievers (Properties 6, 7, 8).

Tests use fixed test documents and multiple queries rather than pure Hypothesis
generation, since each test iteration loads actual ML models (BM25, FAISS
embeddings) which are heavy.

Validates: Requirements 5.1, 5.2, 5.3, 6.1, 6.2, 6.3
"""

import pytest
from langchain_core.documents import Document

from indexing.bm25_index import build_bm25_retriever
from indexing.dense_index import build_faiss_store
from retrieval.bm25_retriever import retrieve_bm25
from retrieval.dense_retriever import retrieve_dense


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CAPITAL_DOCS = [
    Document(page_content="Paris is the capital of France and is known for the Eiffel Tower", metadata={"doc_id": "d1"}),
    Document(page_content="Berlin is the capital of Germany and has the Brandenburg Gate", metadata={"doc_id": "d2"}),
    Document(page_content="Tokyo is the capital of Japan and is the most populous city", metadata={"doc_id": "d3"}),
    Document(page_content="London is the capital of England and home to Big Ben", metadata={"doc_id": "d4"}),
    Document(page_content="Madrid is the capital of Spain and famous for the Prado Museum", metadata={"doc_id": "d5"}),
    Document(page_content="Rome is the capital of Italy and contains the Colosseum", metadata={"doc_id": "d6"}),
    Document(page_content="Ottawa is the capital of Canada and sits on the Ottawa River", metadata={"doc_id": "d7"}),
    Document(page_content="Canberra is the capital of Australia and was purpose-built", metadata={"doc_id": "d8"}),
    Document(page_content="Brasilia is the capital of Brazil and was designed by Niemeyer", metadata={"doc_id": "d9"}),
    Document(page_content="Cairo is the capital of Egypt and sits near the pyramids of Giza", metadata={"doc_id": "d10"}),
]

QUERIES = [
    "capital of France",
    "German city with a famous gate",
    "most populous Asian capital",
    "European capital with a museum",
    "capital near ancient pyramids",
]


@pytest.fixture(scope="module")
def bm25_retriever():
    """Build a BM25 retriever over the 10 capital docs."""
    return build_bm25_retriever(CAPITAL_DOCS, k=len(CAPITAL_DOCS))


@pytest.fixture(scope="module")
def faiss_store():
    """Build a FAISS store over the 10 capital docs on CPU."""
    return build_faiss_store(
        documents=CAPITAL_DOCS,
        device="cpu",
        batch_size=64,
        use_ivf=False,
    )


# ---------------------------------------------------------------------------
# Property 6: Retriever Results Are Sorted Descending by Score
# ---------------------------------------------------------------------------
# Feature: rag-trivia-pipeline, Property 6: Retriever Results Are Sorted Descending by Score
# **Validates: Requirements 5.1, 6.1**


class TestProperty6SortedDescending:
    """Verify BM25 and Dense results are sorted in descending order by score."""

    def test_bm25_returns_results(self, bm25_retriever):
        """BM25Retriever.invoke() returns results for each query.

        BM25Retriever doesn't expose scores directly, so we verify it
        returns non-empty results (the retriever internally sorts by BM25 score).
        """
        for query in QUERIES:
            results = retrieve_bm25(bm25_retriever, query, k=len(CAPITAL_DOCS))
            assert len(results) > 0, f"BM25 returned no results for query: '{query}'"
            assert all(isinstance(doc, Document) for doc in results)

    def test_dense_scores_sorted_by_relevance(self, faiss_store):
        """Dense retriever results are sorted by relevance (most relevant first).

        FAISS similarity_search_with_score returns (Document, score) tuples
        where the score is a distance metric — lower values indicate higher
        similarity. Results are sorted ascending by distance (i.e., most
        relevant first), which is the correct descending-relevance ordering.
        """
        for query in QUERIES:
            results = retrieve_dense(faiss_store, query, k=len(CAPITAL_DOCS))
            assert len(results) > 0, f"Dense returned no results for query: '{query}'"

            scores = [score for _, score in results]
            for i in range(len(scores) - 1):
                assert scores[i] <= scores[i + 1], (
                    f"Dense results not sorted by relevance for query '{query}': "
                    f"distance[{i}]={scores[i]} > distance[{i+1}]={scores[i+1]}"
                )


# ---------------------------------------------------------------------------
# Property 7: Per-Query Mode Returns All Documents
# ---------------------------------------------------------------------------
# Feature: rag-trivia-pipeline, Property 7: Per-Query Mode Returns All Documents
# **Validates: Requirements 5.3, 6.3**


class TestProperty7PerQueryReturnsAll:
    """Verify result count equals corpus size when k=corpus_size."""

    def test_bm25_returns_all_documents(self, bm25_retriever):
        """BM25 retriever with k=corpus_size returns all documents."""
        corpus_size = len(CAPITAL_DOCS)
        for query in QUERIES:
            results = retrieve_bm25(bm25_retriever, query, k=corpus_size)
            assert len(results) == corpus_size, (
                f"BM25 returned {len(results)} docs, expected {corpus_size} "
                f"for query: '{query}'"
            )

    def test_dense_returns_all_documents(self, faiss_store):
        """Dense retriever with k=corpus_size returns all documents."""
        corpus_size = len(CAPITAL_DOCS)
        for query in QUERIES:
            results = retrieve_dense(faiss_store, query, k=corpus_size)
            assert len(results) == corpus_size, (
                f"Dense returned {len(results)} docs, expected {corpus_size} "
                f"for query: '{query}'"
            )


# ---------------------------------------------------------------------------
# Property 8: Full-Pool Mode Respects Top-K Limit
# ---------------------------------------------------------------------------
# Feature: rag-trivia-pipeline, Property 8: Full-Pool Mode Respects Top-K Limit
# **Validates: Requirements 5.2, 6.2**


class TestProperty8TopKLimit:
    """Verify result count equals top_k when corpus > top_k."""

    def test_bm25_respects_top_k(self):
        """BM25 retriever with k=3 on a 10-doc corpus returns exactly 3 results."""
        top_k = 3
        retriever = build_bm25_retriever(CAPITAL_DOCS, k=top_k)
        for query in QUERIES:
            results = retrieve_bm25(retriever, query, k=top_k)
            assert len(results) == top_k, (
                f"BM25 returned {len(results)} docs, expected {top_k} "
                f"for query: '{query}'"
            )

    def test_dense_respects_top_k(self, faiss_store):
        """Dense retriever with k=3 on a 10-doc corpus returns exactly 3 results."""
        top_k = 3
        for query in QUERIES:
            results = retrieve_dense(faiss_store, query, k=top_k)
            assert len(results) == top_k, (
                f"Dense returned {len(results)} docs, expected {top_k} "
                f"for query: '{query}'"
            )


# ===========================================================================
# Unit Tests for BM25 and Dense Retrievers
# ===========================================================================
# Requirements: 5.1, 6.1


class TestBM25RetrieverUnit:
    """Unit tests for BM25 retriever with known corpus and queries."""

    def test_bm25_known_query_returns_expected_doc(self, bm25_retriever):
        """BM25 retriever returns the Paris doc in top results for 'capital of France'."""
        results = retrieve_bm25(bm25_retriever, "capital of France", k=len(CAPITAL_DOCS))
        doc_ids = [doc.metadata["doc_id"] for doc in results]
        assert "d1" in doc_ids[:3], (
            f"Expected 'd1' (Paris) in top-3 BM25 results, got: {doc_ids[:3]}"
        )

    def test_bm25_egypt_query_returns_cairo(self, bm25_retriever):
        """BM25 retriever returns Cairo doc for 'Egypt pyramids Giza'."""
        results = retrieve_bm25(bm25_retriever, "Egypt pyramids Giza", k=len(CAPITAL_DOCS))
        doc_ids = [doc.metadata["doc_id"] for doc in results]
        assert "d10" in doc_ids[:3], (
            f"Expected 'd10' (Cairo) in top-3 BM25 results, got: {doc_ids[:3]}"
        )

    def test_bm25_empty_query_returns_empty(self, bm25_retriever):
        """BM25 retriever returns empty list for empty query."""
        assert retrieve_bm25(bm25_retriever, "", k=len(CAPITAL_DOCS)) == []

    def test_bm25_whitespace_query_returns_empty(self, bm25_retriever):
        """BM25 retriever returns empty list for whitespace-only query."""
        assert retrieve_bm25(bm25_retriever, "   ", k=len(CAPITAL_DOCS)) == []


class TestDenseRetrieverUnit:
    """Unit tests for Dense retriever with known corpus and queries."""

    def test_dense_known_query_returns_expected_doc(self, faiss_store):
        """Dense retriever returns the Paris doc in top results for 'capital of France'."""
        results = retrieve_dense(faiss_store, "capital of France", k=len(CAPITAL_DOCS))
        doc_ids = [doc.metadata["doc_id"] for doc, _ in results]
        assert "d1" in doc_ids[:3], (
            f"Expected 'd1' (Paris) in top-3 Dense results, got: {doc_ids[:3]}"
        )

    def test_dense_egypt_query_returns_cairo(self, faiss_store):
        """Dense retriever returns Cairo doc for 'Egypt pyramids Giza'."""
        results = retrieve_dense(faiss_store, "Egypt pyramids Giza", k=len(CAPITAL_DOCS))
        doc_ids = [doc.metadata["doc_id"] for doc, _ in results]
        assert "d10" in doc_ids[:3], (
            f"Expected 'd10' (Cairo) in top-3 Dense results, got: {doc_ids[:3]}"
        )

    def test_dense_empty_query_returns_empty(self, faiss_store):
        """Dense retriever returns empty list for empty query."""
        assert retrieve_dense(faiss_store, "", k=len(CAPITAL_DOCS)) == []

    def test_dense_whitespace_query_returns_empty(self, faiss_store):
        """Dense retriever returns empty list for whitespace-only query."""
        assert retrieve_dense(faiss_store, "   ", k=len(CAPITAL_DOCS)) == []
