"""Dense retrieval wrapper around LangChain FAISS vector store.

Provides a thin wrapper that calls FAISS.similarity_search_with_score()
with configurable k for switching between per-query mode (all docs) and
full-pool mode (top-k filtering).
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import settings


def retrieve_dense(
    store: FAISS,
    query: str,
    k: int | None = None,
) -> list[tuple[Document, float]]:
    """Retrieve documents using dense semantic similarity search.

    Calls store.similarity_search_with_score(query, k=k) to find the
    most similar documents by inner product similarity.

    In per-query mode, pass k equal to the corpus size to return all documents.
    In full-pool mode, pass k equal to RETRIEVAL_TOP_K to limit results.

    Args:
        store: A configured LangChain FAISS vector store instance.
        query: The search query string.
        k: Number of results to return. If None, defaults to
            RETRIEVAL_TOP_K from config.settings.

    Returns:
        A list of (Document, score) tuples sorted by similarity score.
        Returns an empty list if the query is empty.
    """
    if not query or not query.strip():
        return []

    if k is None:
        k = settings.RETRIEVAL_TOP_K

    return store.similarity_search_with_score(query, k=k)
