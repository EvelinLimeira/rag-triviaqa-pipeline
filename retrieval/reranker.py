"""Cross-encoder reranking using sentence-transformers.

Provides the Reranker class that rescores candidate documents using
the cross-encoder model ms-marco-MiniLM-L-6-v2. This is a manual
implementation (not LangChain-wrapped) for full control over batching
and VRAM management.
"""

import logging

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import settings
from config.settings import DEVICE

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for rescoring candidate documents.

    Loads a cross-encoder model and scores (query, document) pairs to
    produce a more precise relevance ranking than first-stage retrievers.

    Attributes:
        model: The loaded CrossEncoder model instance.
        batch_size: Number of pairs to score per batch.
    """

    def __init__(
        self,
        model_name: str = settings.RERANKER_MODEL,
        device: str = DEVICE,
        batch_size: int = settings.RERANKER_BATCH_SIZE,
    ) -> None:
        """Initialize the Reranker by loading the cross-encoder model.

        Args:
            model_name: Name or path of the cross-encoder model.
                Defaults to RERANKER_MODEL from config.settings.
            device: Device to load the model on ('cuda' or 'cpu').
            batch_size: Number of (query, document) pairs to score per batch.
                Defaults to RERANKER_BATCH_SIZE from config.settings.

        Raises:
            RuntimeError: If the cross-encoder model fails to load.
        """
        self.batch_size = batch_size
        try:
            self.model = CrossEncoder(model_name, device=device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load cross-encoder model '{model_name}': {e}"
            ) from e

    def rerank(
        self,
        query: str,
        candidates: list[Document],
        top_k: int = settings.RERANKER_TOP_K,
    ) -> list[tuple[Document, float]]:
        """Rerank candidate documents by cross-encoder relevance score.

        Creates (query, document_content) pairs for all candidates, scores
        them using the cross-encoder model in batches, sorts by descending
        score, and returns the top_k results.

        Args:
            query: The search query string.
            candidates: List of LangChain Document objects to rerank.
            top_k: Maximum number of top-scoring documents to return.
                Defaults to RERANKER_TOP_K from config.settings.

        Returns:
            A list of (Document, score) tuples sorted by descending
            cross-encoder score. Returns at most top_k results, or all
            candidates if fewer than top_k are provided. Returns an
            empty list if candidates is empty.
        """
        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        scored = list(zip(candidates, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]

    def unload_model(self) -> None:
        """Unload the cross-encoder model and free GPU memory.

        Deletes the model object and calls torch.cuda.empty_cache()
        if CUDA is available to release GPU VRAM.
        """
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Reranker model unloaded and GPU cache cleared.")
