"""Property-based tests for indexing/chunker.py (Properties 19–20).

Tests the document chunking function using Hypothesis to verify
correctness properties across randomized inputs.
"""

from hypothesis import strategies as st, given, settings
from langchain_core.documents import Document

from indexing.chunker import chunk_documents

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

doc_ids = st.text(
    min_size=1,
    max_size=20,
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters="_-",
    ),
)

# Short documents (should pass through unchanged)
short_docs = st.builds(
    Document,
    page_content=st.text(min_size=1, max_size=512),
    metadata=st.fixed_dictionaries({"doc_id": doc_ids}),
)

# Long documents (should be chunked)
long_docs = st.builds(
    Document,
    page_content=st.text(min_size=513, max_size=2000),
    metadata=st.fixed_dictionaries({"doc_id": doc_ids}),
)


# ---------------------------------------------------------------------------
# Property 19: Chunking Respects Size Limits and Preserves Short Documents
# ---------------------------------------------------------------------------
# Feature: rag-trivia-pipeline, Property 19: Chunking Respects Size Limits and Preserves Short Documents
# Validates: Requirements 18.1, 18.2


@given(doc=short_docs)
@settings(max_examples=100)
def test_short_doc_passes_through_unchanged(doc: Document) -> None:
    """A document with page_content <= chunk_size should pass through
    as exactly one Document with the original content unchanged."""
    result = chunk_documents([doc], chunk_size=512, chunk_overlap=64)

    assert len(result) == 1, (
        f"Expected exactly 1 document for short input, got {len(result)}"
    )
    assert result[0].page_content == doc.page_content
    assert result[0].metadata["doc_id"] == doc.metadata["doc_id"]


@given(doc=long_docs)
@settings(max_examples=100)
def test_long_doc_produces_chunks_within_size_limit(doc: Document) -> None:
    """A document with page_content > chunk_size should produce multiple
    chunks, each with at most chunk_size characters."""
    chunk_size = 512
    result = chunk_documents([doc], chunk_size=chunk_size, chunk_overlap=64)

    assert len(result) > 1, (
        f"Expected multiple chunks for long input (len={len(doc.page_content)}), "
        f"got {len(result)}"
    )
    for i, chunk in enumerate(result):
        assert len(chunk.page_content) <= chunk_size, (
            f"Chunk {i} has length {len(chunk.page_content)}, "
            f"exceeds chunk_size={chunk_size}"
        )


# ---------------------------------------------------------------------------
# Property 20: Chunk IDs Are Unique and Derived From Original
# ---------------------------------------------------------------------------
# Feature: rag-trivia-pipeline, Property 20: Chunk IDs Are Unique and Derived From Original
# Validates: Requirements 18.3


@given(doc=long_docs)
@settings(max_examples=100)
def test_chunk_ids_are_unique_and_derived_from_original(doc: Document) -> None:
    """For any document chunked into multiple pieces, each chunk should
    have a unique doc_id containing the original doc_id as a substring."""
    result = chunk_documents([doc], chunk_size=512, chunk_overlap=64)

    if len(result) <= 1:
        return  # Not chunked, nothing to verify for this property

    original_id = doc.metadata["doc_id"]
    chunk_ids = [chunk.metadata["doc_id"] for chunk in result]

    # All chunk IDs must be unique
    assert len(chunk_ids) == len(set(chunk_ids)), (
        f"Chunk IDs are not unique: {chunk_ids}"
    )

    # Each chunk ID must contain the original doc_id as a substring
    for chunk_id in chunk_ids:
        assert original_id in chunk_id, (
            f"Chunk ID '{chunk_id}' does not contain original ID '{original_id}'"
        )
