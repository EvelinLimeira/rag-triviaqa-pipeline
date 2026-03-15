"""Property-based test for cross-encoder reranking (Property 10).

Tests use fixed test documents and multiple queries rather than pure Hypothesis
generation, since each test iteration loads the cross-encoder model which is heavy.

Validates: Requirements 8.4, 8.5
"""

import pytest
from langchain_core.documents import Document

from retrieval.reranker import Reranker


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
    "What is the capital of France?",
    "Which city has the Brandenburg Gate?",
    "What is the most populous capital in Asia?",
    "Which European capital has a famous museum?",
    "Which capital is near ancient pyramids?",
]


@pytest.fixture(scope="module")
def reranker():
    """Create a Reranker instance on CPU."""
    r = Reranker(device="cpu")
    yield r
    r.unload_model()


# ---------------------------------------------------------------------------
# Property 10: Reranker Returns Top-K Sorted Descending
# ---------------------------------------------------------------------------
# Feature: rag-trivia-pipeline, Property 10: Reranker Returns Top-K Sorted Descending
# **Validates: Requirements 8.4, 8.5**


class TestProperty10RerankerTopKSorted:
    """Verify reranker returns ≤ top_k results, sorted descending by score."""

    def test_reranker_returns_top_k_sorted_descending(self, reranker):
        """Reranker with top_k=5 on 10 candidates returns exactly 5 results
        sorted in descending order by cross-encoder score."""
        top_k = 5
        for query in QUERIES:
            results = reranker.rerank(query, CAPITAL_DOCS, top_k=top_k)

            # Verify at most top_k results
            assert len(results) <= top_k, (
                f"Reranker returned {len(results)} results, expected <= {top_k} "
                f"for query: '{query}'"
            )
            assert len(results) == top_k, (
                f"Reranker returned {len(results)} results, expected {top_k} "
                f"(corpus has {len(CAPITAL_DOCS)} docs) for query: '{query}'"
            )

            # Verify results are (Document, float) tuples
            for doc, score in results:
                assert isinstance(doc, Document)
                assert isinstance(score, float)

            # Verify scores are sorted descending
            scores = [score for _, score in results]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"Reranker results not sorted descending for query '{query}': "
                    f"score[{i}]={scores[i]} < score[{i+1}]={scores[i+1]}"
                )

    def test_reranker_fewer_than_top_k_returns_all(self, reranker):
        """When candidates < top_k, reranker returns all candidates sorted."""
        small_corpus = CAPITAL_DOCS[:3]  # Only 3 docs
        top_k = 5
        for query in QUERIES:
            results = reranker.rerank(query, small_corpus, top_k=top_k)

            # Should return all 3 candidates since 3 < 5
            assert len(results) == len(small_corpus), (
                f"Reranker returned {len(results)} results, expected {len(small_corpus)} "
                f"for query: '{query}'"
            )

            # Verify scores are sorted descending
            scores = [score for _, score in results]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"Reranker results not sorted descending for query '{query}': "
                    f"score[{i}]={scores[i]} < score[{i+1}]={scores[i+1]}"
                )

    def test_reranker_empty_candidates(self, reranker):
        """Reranker with empty candidates returns empty list."""
        results = reranker.rerank("any query", [], top_k=5)
        assert results == []
