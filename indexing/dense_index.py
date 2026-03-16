"""Dense vector index construction using LangChain FAISS and HuggingFace embeddings."""

import logging

import faiss
import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import DEVICE, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def create_embeddings(
    model_name: str = EMBEDDING_MODEL,
    device: str = DEVICE,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> HuggingFaceEmbeddings:
    """Create a HuggingFaceEmbeddings instance with the BGE query instruction prefix.

    Args:
        model_name: HuggingFace model identifier. Defaults to bge-base-en-v1.5.
        device: Device to load the model on ('cuda' or 'cpu').
        batch_size: Batch size for encoding documents.

    Returns:
        A configured HuggingFaceEmbeddings instance.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "batch_size": batch_size,
            "normalize_embeddings": True,
        },
        query_encode_kwargs={"prompt_name": "query"},
    )


def build_faiss_store(
    documents: list[Document],
    model_name: str = EMBEDDING_MODEL,
    device: str = DEVICE,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    use_ivf: bool = False,
) -> FAISS:
    """Build a FAISS vector store from LangChain Documents.

    Creates embeddings using HuggingFaceEmbeddings and builds a FAISS index.
    Uses IndexFlatIP for per-query mode and IndexIVFFlat (nprobe=10) for
    full-pool mode when use_ivf is True.

    Args:
        documents: List of LangChain Document objects to index.
        model_name: HuggingFace model identifier. Defaults to bge-base-en-v1.5.
        device: Device to load the model on ('cuda' or 'cpu').
        batch_size: Batch size for encoding documents. Defaults to 64.
        use_ivf: If True, replace the flat index with IndexIVFFlat and set
            nprobe=10. Intended for large corpora (>100k docs).

    Returns:
        A FAISS vector store instance.

    Raises:
        ValueError: If the documents list is empty.
    """
    if not documents:
        raise ValueError("Cannot build FAISS store from empty corpus")

    embeddings = create_embeddings(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

    store = FAISS.from_documents(documents, embeddings)

    if use_ivf:
        _replace_with_ivf_index(store)

    return store


def _replace_with_ivf_index(store: FAISS) -> None:
    """Replace the flat index in a FAISS store with an IVFFlat index.

    Trains the IVF index on the existing vectors and rebuilds the index
    with nprobe=10 for approximate nearest neighbor search.

    Args:
        store: A FAISS vector store whose index will be replaced in-place.
    """
    flat_index = store.index
    d = flat_index.d
    n = flat_index.ntotal

    # Number of clusters: use sqrt(n) capped to a reasonable range
    nlist = max(1, min(int(n**0.5), 4096))

    quantizer = faiss.IndexFlatIP(d)
    ivf_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    # Reconstruct all vectors to train the IVF index
    vectors = flat_index.reconstruct_n(0, n)
    ivf_index.train(vectors)
    ivf_index.add(vectors)
    ivf_index.nprobe = 10

    store.index = ivf_index


def save_faiss_store(store: FAISS, path: str) -> None:
    """Save a FAISS vector store to disk.

    Args:
        store: The FAISS vector store to save.
        path: Directory path where the index will be saved.
    """
    store.save_local(path)


def load_faiss_store(path: str, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Load a FAISS vector store from disk.

    Args:
        path: Directory path containing the saved index.
        embeddings: The HuggingFaceEmbeddings instance to use for queries.

    Returns:
        A FAISS vector store loaded from disk.
    """
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def unload_embeddings(embeddings: HuggingFaceEmbeddings) -> None:
    """Unload the embedding model from memory and free GPU VRAM.

    Deletes the underlying model client and calls torch.cuda.empty_cache()
    to release GPU memory.

    Args:
        embeddings: The HuggingFaceEmbeddings instance to unload.
    """
    if hasattr(embeddings, "client"):
        del embeddings.client
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Embedding model unloaded and GPU cache cleared.")
