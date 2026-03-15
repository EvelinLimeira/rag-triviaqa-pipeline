"""Hybrid retrieval using LangChain EnsembleRetriever with RRF fusion.

Combines BM25 sparse retrieval and FAISS dense retrieval using
LangChain's EnsembleRetriever, which applies Reciprocal Rank Fusion
(RRF) with configurable weights to produce a unified ranked list.
"""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document


def build_ensemble_retriever(
    bm25_retriever,
    faiss_retriever,
    weights: list[float] | None = None,
) -> EnsembleRetriever:
    """Create an EnsembleRetriever combining BM25 and FAISS retrievers.

    Builds a LangChain EnsembleRetriever that fuses results from both
    retrievers using Reciprocal Rank Fusion (RRF) with the given weights.

    The bm25_retriever should be a LangChain BM25Retriever instance.
    The faiss_retriever should be the result of FAISS.as_retriever()
    with appropriate search_kwargs (e.g., {"k": k}).

    Args:
        bm25_retriever: A configured LangChain BM25Retriever instance.
        faiss_retriever: A FAISS-based retriever from FAISS.as_retriever().
        weights: Weights for each retriever in the ensemble.
            Defaults to [0.5, 0.5] for equal weighting.

    Returns:
        A configured EnsembleRetriever with RRF fusion.
    """
    if weights is None:
        weights = [0.5, 0.5]

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=weights,
    )


def retrieve_hybrid(
    ensemble: EnsembleRetriever,
    query: str,
) -> list[Document]:
    """Retrieve documents using hybrid BM25 + dense search with RRF fusion.

    Calls ensemble.invoke(query) which internally queries both retrievers,
    applies Reciprocal Rank Fusion to combine their ranked results, and
    returns documents sorted by descending RRF score.

    In per-query mode, the underlying retrievers should be configured to
    return all documents so the ensemble returns the full corpus ranked
    by RRF score. In full-pool mode, each retriever returns top-500 and
    the ensemble fuses and returns the top-500 by RRF score.

    Args:
        ensemble: A configured EnsembleRetriever instance.
        query: The search query string.

    Returns:
        A list of LangChain Document objects sorted by descending RRF score.
        Returns an empty list if the query is empty.
    """
    if not query or not query.strip():
        return []

    return ensemble.invoke(query)
