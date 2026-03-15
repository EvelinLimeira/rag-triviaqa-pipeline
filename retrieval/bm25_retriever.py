"""BM25 retrieval wrapper around LangChain BM25Retriever.

Provides a thin wrapper that invokes BM25Retriever.invoke() with optional
k adjustment for switching between per-query mode (all docs) and full-pool
mode (top-k filtering).
"""

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def retrieve_bm25(
    retriever: BM25Retriever,
    query: str,
    k: int | None = None,
) -> list[Document]:
    """Retrieve documents using BM25 sparse search.

    Calls retriever.invoke(query), optionally adjusting the retriever's k
    parameter before invocation and restoring it afterwards.

    In per-query mode, pass k equal to the corpus size to return all documents.
    In full-pool mode, pass k equal to RETRIEVAL_TOP_K to limit results.

    Args:
        retriever: A configured LangChain BM25Retriever instance.
        query: The search query string.
        k: Optional override for the number of results to return.
            If provided, temporarily sets retriever.k before invoking.
            If None, uses the retriever's existing k value.

    Returns:
        A list of LangChain Document objects ranked by BM25 score.
        Returns an empty list if the query is empty.
    """
    if not query or not query.strip():
        return []

    if k is not None:
        original_k = retriever.k
        retriever.k = k
        try:
            results = retriever.invoke(query)
        finally:
            retriever.k = original_k
    else:
        results = retriever.invoke(query)

    return results
