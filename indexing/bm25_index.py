"""BM25 sparse index construction using LangChain BM25Retriever."""

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config.settings import BM25_B, BM25_K1


def build_bm25_retriever(
    documents: list[Document],
    k: int,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> BM25Retriever:
    """Build a BM25Retriever from a list of LangChain Documents.

    Args:
        documents: List of LangChain Document objects to index.
        k: Number of results to return per query. Set to corpus size
            in per-query mode, or RETRIEVAL_TOP_K in full-pool mode.
        k1: BM25 term-frequency saturation parameter. Defaults to 1.5.
        b: BM25 document-length normalization parameter. Defaults to 0.75.

    Returns:
        A configured BM25Retriever instance.

    Raises:
        ValueError: If the documents list is empty.
    """
    if not documents:
        raise ValueError("Cannot build BM25Retriever from empty corpus")

    retriever = BM25Retriever.from_documents(
        documents,
        bm25_params={"k1": k1, "b": b},
    )
    retriever.k = k

    return retriever
