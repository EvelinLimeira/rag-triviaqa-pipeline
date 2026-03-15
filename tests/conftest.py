"""Shared Hypothesis strategies and configuration for the test suite.

Defines reusable strategies for generating LangChain Document objects,
TriviaQAEntry instances, and ranked document lists. Registers a default
Hypothesis profile with ``max_examples=100`` so all property tests use
a consistent iteration count.

This file is auto-discovered by pytest, so the profile registration
happens before any tests run.
"""

from hypothesis import settings, strategies as st
from langchain_core.documents import Document

from data.loader import TriviaQAEntry

# ---------------------------------------------------------------------------
# Hypothesis profile — 100 examples per property test
# ---------------------------------------------------------------------------
settings.register_profile("default", max_examples=100)
settings.load_profile("default")

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

lc_documents: st.SearchStrategy[Document] = st.builds(
    Document,
    page_content=st.text(min_size=1, max_size=500),
    metadata=st.fixed_dictionaries({"doc_id": st.text(min_size=1, max_size=20)}),
)
"""Generate random LangChain Document objects with non-empty page_content
and a ``metadata["doc_id"]`` field."""

triviaqa_entries: st.SearchStrategy[TriviaQAEntry] = st.builds(
    TriviaQAEntry,
    question=st.text(min_size=1, max_size=200),
    answers=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=5),
    golden_docs=st.lists(lc_documents, min_size=1, max_size=5),
    noise_docs=st.lists(lc_documents, min_size=0, max_size=50),
)
"""Generate random TriviaQAEntry objects with at least one golden doc."""

ranked_doc_lists: st.SearchStrategy[list[tuple[str, float]]] = st.lists(
    st.tuples(
        st.text(min_size=1, max_size=10),
        st.floats(min_value=0, max_value=100),
    ),
    min_size=1,
    max_size=100,
    unique_by=lambda x: x[0],
)
"""Generate ranked result lists of (doc_id, score) tuples with unique doc_ids."""
