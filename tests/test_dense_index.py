"""Property-based test for FAISS save/load round-trip (Property 5).

Validates: Requirements 4.6, 4.7
"""

import pytest
from langchain_core.documents import Document

from indexing.dense_index import (
    build_faiss_store,
    create_embeddings,
    load_faiss_store,
    save_faiss_store,
    unload_embeddings,
)


# Feature: rag-trivia-pipeline, Property 5: FAISS Vector Store Save/Load Round-Trip
# **Validates: Requirements 4.6, 4.7**


class TestFAISSRoundTrip:
    """Test that saving and loading a FAISS store produces identical search results."""

    @pytest.fixture(scope="class")
    def sample_docs(self):
        return [
            Document(page_content="Paris is the capital of France", metadata={"doc_id": "d1"}),
            Document(page_content="Berlin is the capital of Germany", metadata={"doc_id": "d2"}),
            Document(page_content="Tokyo is the capital of Japan", metadata={"doc_id": "d3"}),
            Document(page_content="London is the capital of England", metadata={"doc_id": "d4"}),
            Document(page_content="Madrid is the capital of Spain", metadata={"doc_id": "d5"}),
        ]

    @pytest.fixture(scope="class")
    def queries(self):
        return [
            "capital of France",
            "German city",
            "Asian country capital",
            "European capital",
        ]

    def test_faiss_save_load_round_trip(self, sample_docs, queries, tmp_path_factory):
        """Build a FAISS store, save it, load it back, and verify
        identical search results for multiple queries.

        Property 5: For any FAISS vector store built from a corpus of LangChain
        Documents, saving with save_local() and loading with load_local() should
        produce a store that returns identical search results (same documents,
        same scores) for any query.

        # @settings(max_examples=100) — using fixed queries for practicality
        # since each iteration loads the embedding model.
        """
        tmp_dir = tmp_path_factory.mktemp("faiss_test")
        save_path = str(tmp_dir / "faiss_index")

        # Build original store
        store = build_faiss_store(
            documents=sample_docs,
            device="cpu",
            batch_size=64,
            use_ivf=False,
        )

        # Save to disk
        save_faiss_store(store, save_path)

        # Load from disk
        embeddings = create_embeddings(device="cpu")
        loaded_store = load_faiss_store(save_path, embeddings)

        # Compare search results for each query
        for query in queries:
            original_results = store.similarity_search_with_score(query, k=3)
            loaded_results = loaded_store.similarity_search_with_score(query, k=3)

            assert len(original_results) == len(loaded_results), (
                f"Different result counts for query '{query}'"
            )

            for (orig_doc, orig_score), (load_doc, load_score) in zip(
                original_results, loaded_results
            ):
                assert orig_doc.page_content == load_doc.page_content, (
                    f"Content mismatch for query '{query}': "
                    f"'{orig_doc.page_content}' != '{load_doc.page_content}'"
                )
                assert orig_doc.metadata["doc_id"] == load_doc.metadata["doc_id"], (
                    f"doc_id mismatch for query '{query}': "
                    f"'{orig_doc.metadata['doc_id']}' != '{load_doc.metadata['doc_id']}'"
                )
                assert abs(orig_score - load_score) < 1e-5, (
                    f"Score mismatch for query '{query}': "
                    f"{orig_score} != {load_score}"
                )

        # Cleanup
        unload_embeddings(embeddings)
