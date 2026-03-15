"""Unit tests for indexing/bm25_index.py."""

import pytest
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from indexing.bm25_index import build_bm25_retriever


def _make_docs(n: int) -> list[Document]:
    """Create a list of simple LangChain Documents for testing."""
    return [
        Document(
            page_content=f"Document number {i} with some content",
            metadata={"doc_id": f"doc_{i}"},
        )
        for i in range(n)
    ]


class TestBuildBM25Retriever:
    """Tests for build_bm25_retriever."""

    def test_returns_bm25_retriever_instance(self) -> None:
        docs = _make_docs(5)
        retriever = build_bm25_retriever(docs, k=5)
        assert isinstance(retriever, BM25Retriever)

    def test_k_is_set_correctly(self) -> None:
        docs = _make_docs(10)
        retriever = build_bm25_retriever(docs, k=3)
        assert retriever.k == 3

    def test_k_set_to_corpus_size_for_per_query_mode(self) -> None:
        docs = _make_docs(50)
        retriever = build_bm25_retriever(docs, k=len(docs))
        assert retriever.k == 50

    def test_raises_value_error_on_empty_corpus(self) -> None:
        with pytest.raises(ValueError, match="Cannot build BM25Retriever from empty corpus"):
            build_bm25_retriever([], k=10)

    def test_custom_bm25_params(self) -> None:
        docs = _make_docs(5)
        retriever = build_bm25_retriever(docs, k=5, k1=2.0, b=0.5)
        assert isinstance(retriever, BM25Retriever)

    def test_default_bm25_params(self) -> None:
        """Default k1=1.5 and b=0.75 from config.settings."""
        docs = _make_docs(5)
        retriever = build_bm25_retriever(docs, k=5)
        # Verify it was created successfully with defaults
        assert isinstance(retriever, BM25Retriever)

    def test_retriever_can_invoke(self) -> None:
        """Verify the built retriever actually works for a query."""
        docs = [
            Document(page_content="Paris is the capital of France", metadata={"doc_id": "d1"}),
            Document(page_content="Berlin is the capital of Germany", metadata={"doc_id": "d2"}),
            Document(page_content="Tokyo is the capital of Japan", metadata={"doc_id": "d3"}),
        ]
        retriever = build_bm25_retriever(docs, k=3)
        results = retriever.invoke("capital of France")
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc in results)

    def test_single_document_corpus(self) -> None:
        docs = _make_docs(1)
        retriever = build_bm25_retriever(docs, k=1)
        assert isinstance(retriever, BM25Retriever)
        assert retriever.k == 1
