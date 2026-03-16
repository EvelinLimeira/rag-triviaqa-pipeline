"""Hybrid index builder that orchestrates BM25 and FAISS index construction."""

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import BM25_B, BM25_K1, DEVICE, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from indexing.bm25_index import build_bm25_retriever
from indexing.dense_index import build_faiss_store


def build_hybrid_indices(
    documents: list[Document],
    k: int,
    model_name: str = EMBEDDING_MODEL,
    device: str = DEVICE,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    k1: float = BM25_K1,
    b: float = BM25_B,
    use_ivf: bool = False,
) -> tuple[BM25Retriever, FAISS]:
    """Build both BM25 and FAISS indices for a given document corpus.

    Convenience function that orchestrates construction of a sparse BM25
    retriever and a dense FAISS vector store from the same set of documents.

    Args:
        documents: List of LangChain Document objects to index.
        k: Number of results to return per query. Set to corpus size
            in per-query mode, or RETRIEVAL_TOP_K in full-pool mode.
        model_name: HuggingFace model identifier for dense embeddings.
            Defaults to bge-base-en-v1.5.
        device: Device to load the embedding model on ('cuda' or 'cpu').
        batch_size: Batch size for encoding documents. Defaults to 64.
        k1: BM25 term-frequency saturation parameter. Defaults to 1.5.
        b: BM25 document-length normalization parameter. Defaults to 0.75.
        use_ivf: If True, use IndexIVFFlat for the FAISS index. Intended
            for large corpora (>100k docs) in full-pool mode.

    Returns:
        A tuple of (BM25Retriever, FAISS vector store).

    Raises:
        ValueError: If the documents list is empty.
    """
    bm25_retriever = build_bm25_retriever(
        documents=documents,
        k=k,
        k1=k1,
        b=b,
    )

    faiss_store = build_faiss_store(
        documents=documents,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        use_ivf=use_ivf,
    )

    return bm25_retriever, faiss_store
