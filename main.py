"""CLI entry point for the RAG TriviaQA pipeline.

Provides subcommands for each pipeline stage:
- ``download``: Download the TriviaQA dataset from HuggingFace.
- ``index``: Build and persist BM25 + FAISS indices for the full document pool.
- ``evaluate``: Run per-query evaluation on a configurable number of sample queries.
- ``evaluate-full-pool``: Run evaluation against pre-built full pool indices.
"""

import argparse
import logging
import os
import pickle
import sys

from config import settings
from data.download import download_dataset
from data.loader import load_documents_pool, load_triviaqa
from evaluation.run_eval import EvaluationOrchestrator
from indexing.bm25_index import build_bm25_retriever
from indexing.chunker import chunk_documents
from indexing.dense_index import (
    build_faiss_store,
    create_embeddings,
    load_faiss_store,
    save_faiss_store,
    unload_embeddings,
)

logger = logging.getLogger(__name__)

DATA_DIR = "data/raw"
INDEX_DIR = "data/indices"
RESULTS_DIR = "results"

TRIVIAQA_PATH = os.path.join(DATA_DIR, "final_data", "triviaqa.jsonl")
POOL_PATH = os.path.join(DATA_DIR, "final_data", "documents_pool.json")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25_retriever.pkl")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_store")


def cmd_download() -> None:
    """Execute the download subcommand."""
    logger.info("Downloading TriviaQA dataset...")
    download_dataset(data_dir=DATA_DIR)
    logger.info("Download complete.")


def cmd_index() -> None:
    """Execute the index subcommand.

    Loads the full document pool, chunks documents, builds and persists
    BM25 and FAISS indices with progress reporting.
    """
    logger.info("Loading full document pool from %s ...", POOL_PATH)
    documents = load_documents_pool(POOL_PATH)
    logger.info("Loaded %d documents.", len(documents))

    logger.info("Chunking documents (chunk_size=%d, overlap=%d) ...",
                settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    chunked_docs = chunk_documents(documents)
    logger.info("Chunking complete: %d chunks from %d documents.",
                len(chunked_docs), len(documents))

    os.makedirs(INDEX_DIR, exist_ok=True)

    # Build and save BM25 index
    logger.info("Building BM25 index over %d chunks ...", len(chunked_docs))
    bm25_retriever = build_bm25_retriever(
        chunked_docs, k=settings.RETRIEVAL_TOP_K
    )
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
    logger.info("BM25 index saved to %s", BM25_INDEX_PATH)

    # Build and save FAISS index with IVF for full pool
    logger.info("Building FAISS index (IVF) over %d chunks ...", len(chunked_docs))
    faiss_store = build_faiss_store(
        chunked_docs,
        use_ivf=True,
    )
    save_faiss_store(faiss_store, FAISS_INDEX_PATH)
    logger.info("FAISS index saved to %s", FAISS_INDEX_PATH)

    # Unload embedding model to free VRAM
    if hasattr(faiss_store, "embedding_function") and faiss_store.embedding_function is not None:
        unload_embeddings(faiss_store.embedding_function)

    logger.info("Indexing complete.")


def cmd_evaluate(sample_size: int) -> None:
    """Execute the evaluate subcommand (per-query mode).

    Args:
        sample_size: Number of queries to evaluate.
    """
    logger.info("Loading TriviaQA entries from %s ...", TRIVIAQA_PATH)
    entries = load_triviaqa(TRIVIAQA_PATH)
    logger.info("Loaded %d entries.", len(entries))

    orchestrator = EvaluationOrchestrator(sample_size=sample_size)
    logger.info("Running per-query evaluation (sample_size=%d) ...", sample_size)
    results = orchestrator.run_per_query(entries)

    orchestrator.print_comparison_table(results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    orchestrator.save_results(results, RESULTS_DIR)
    logger.info("Evaluation complete. Results saved to %s", RESULTS_DIR)


def cmd_evaluate_full_pool(sample_size: int) -> None:
    """Execute the evaluate-full-pool subcommand.

    Loads pre-built BM25 and FAISS indices and runs full-pool evaluation.

    Args:
        sample_size: Number of queries to evaluate.
    """
    logger.info("Loading TriviaQA entries from %s ...", TRIVIAQA_PATH)
    entries = load_triviaqa(TRIVIAQA_PATH)
    logger.info("Loaded %d entries.", len(entries))

    # Load pre-built BM25 index
    logger.info("Loading BM25 index from %s ...", BM25_INDEX_PATH)
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)  # noqa: S301

    # Load pre-built FAISS index
    logger.info("Loading FAISS index from %s ...", FAISS_INDEX_PATH)
    embeddings = create_embeddings()
    faiss_store = load_faiss_store(FAISS_INDEX_PATH, embeddings)

    # Build doc_map from the pool for reference
    logger.info("Loading document pool for doc_map ...")
    pool_docs = load_documents_pool(POOL_PATH)
    doc_map = {
        doc.metadata["doc_id"]: doc.page_content for doc in pool_docs
    }

    orchestrator = EvaluationOrchestrator(sample_size=sample_size)
    logger.info("Running full-pool evaluation (sample_size=%d) ...", sample_size)
    results = orchestrator.run_full_pool(entries, bm25_retriever, faiss_store, doc_map)

    orchestrator.print_comparison_table(results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    orchestrator.save_results(results, RESULTS_DIR)
    logger.info("Full-pool evaluation complete. Results saved to %s", RESULTS_DIR)


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="RAG TriviaQA Pipeline — download, index, and evaluate."
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    # download subcommand
    subparsers.add_parser("download", help="Download the TriviaQA dataset from HuggingFace.")

    # index subcommand
    subparsers.add_parser("index", help="Build and persist BM25 + FAISS indices for the full document pool.")

    # evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Run per-query evaluation.")
    eval_parser.add_argument(
        "--sample-size",
        type=int,
        default=settings.EVAL_SAMPLE_SIZE,
        help=f"Number of queries to evaluate (default: {settings.EVAL_SAMPLE_SIZE}).",
    )

    # evaluate-full-pool subcommand
    eval_fp_parser = subparsers.add_parser(
        "evaluate-full-pool",
        help="Run evaluation against pre-built full pool indices.",
    )
    eval_fp_parser.add_argument(
        "--sample-size",
        type=int,
        default=settings.EVAL_SAMPLE_SIZE,
        help=f"Number of queries to evaluate (default: {settings.EVAL_SAMPLE_SIZE}).",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_usage(sys.stderr)
        sys.exit(1)

    if args.command == "download":
        cmd_download()
    elif args.command == "index":
        cmd_index()
    elif args.command == "evaluate":
        cmd_evaluate(args.sample_size)
    elif args.command == "evaluate-full-pool":
        cmd_evaluate_full_pool(args.sample_size)


if __name__ == "__main__":
    main()
