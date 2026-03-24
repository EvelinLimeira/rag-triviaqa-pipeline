"""Document chunking for the RAG TriviaQA pipeline.

Uses LangChain's RecursiveCharacterTextSplitter to split long documents
into passage-sized chunks for indexing in full-pool mode.
"""

import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter.

    Documents with page_content length <= chunk_size are passed through
    unchanged. Longer documents are split into overlapping chunks, each
    assigned a unique ID derived from the original document ID.

    Args:
        documents: List of LangChain Documents to chunk.
        chunk_size: Maximum chunk size in characters. Defaults to CHUNK_SIZE.
        chunk_overlap: Overlap between consecutive chunks. Defaults to CHUNK_OVERLAP.

    Returns:
        List of LangChain Documents, with long documents replaced by their chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    result: list[Document] = []

    for doc in tqdm(documents, desc="Chunking documents"):
        original_doc_id = doc.metadata.get("doc_id", "")

        if len(doc.page_content) <= chunk_size:
            result.append(doc)
        else:
            chunks = splitter.split_text(doc.page_content)
            for i, chunk_text in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={**doc.metadata, "doc_id": f"{original_doc_id}_chunk_{i}"},
                )
                result.append(chunk_doc)

    return result
