"""Evaluation orchestrator for the RAG TriviaQA pipeline.

Runs the full evaluation pipeline across four retriever configurations
(BM25-only, Dense-only, Hybrid, Hybrid+Rerank), computing IR metrics
(Hit Rate@k, MRR) and generation metrics (EM, F1, LLM-judge correctness,
faithfulness) for each configuration.

Supports two evaluation modes:
- Per-query mode: builds fresh BM25 + FAISS indices per query (~50 docs each).
- Full-pool mode: uses pre-built global indices (~1.09M docs).

Implements sequential VRAM management to keep GPU usage under 8GB by
loading only one large model at a time.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

from config import settings
from data.loader import TriviaQAEntry, get_per_query_corpus
from evaluation.ir_metrics import hit_rate_at_k, mrr
from evaluation.llm_metrics import (
    exact_match,
    llm_judge_correctness,
    llm_judge_faithfulness,
    token_f1,
)
from generation.generator import Generator
from indexing.bm25_index import build_bm25_retriever
from indexing.dense_index import build_faiss_store, unload_embeddings
from retrieval.bm25_retriever import retrieve_bm25
from retrieval.dense_retriever import retrieve_dense
from retrieval.hybrid_retriever import build_ensemble_retriever, retrieve_hybrid
from retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Complete result for a single query across the pipeline.

    Attributes:
        question: The trivia question string.
        reference_answers: List of valid answer strings.
        golden_doc_ids: Set of ground-truth relevant document IDs.
        retrieved_ids_pre_rerank: Document IDs from the retriever before reranking.
        retrieved_ids_post_rerank: Document IDs after cross-encoder reranking.
        generated_answer: The LLM-generated answer string.
        context_docs: Top document contents sent to the LLM for generation.
        metrics: All computed metrics for this query.
    """

    question: str
    reference_answers: list[str]
    golden_doc_ids: set[str]
    retrieved_ids_pre_rerank: list[str]
    retrieved_ids_post_rerank: list[str]
    generated_answer: str
    context_docs: list[str]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResults:
    """Aggregated results for one retriever configuration.

    Attributes:
        config_name: Configuration label (e.g. "BM25", "Dense", "Hybrid", "Hybrid+Rerank").
        per_query_results: List of per-query result objects.
        aggregate_metrics: Mean metrics across all queries.
    """

    config_name: str
    per_query_results: list[QueryResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)


# Configuration labels
CONFIG_BM25 = "BM25"
CONFIG_DENSE = "Dense"
CONFIG_HYBRID = "Hybrid"
CONFIG_HYBRID_RERANK = "Hybrid+Rerank"

ALL_CONFIGS = [CONFIG_BM25, CONFIG_DENSE, CONFIG_HYBRID, CONFIG_HYBRID_RERANK]


def _extract_doc_ids(docs: list[Document]) -> list[str]:
    """Extract document IDs from a list of LangChain Documents.

    Args:
        docs: List of LangChain Document objects.

    Returns:
        List of doc_id strings from each document's metadata.
    """
    return [doc.metadata.get("doc_id", "") for doc in docs]


def _extract_doc_ids_from_scored(
    scored_docs: list[tuple[Document, float]],
) -> list[str]:
    """Extract document IDs from a list of scored (Document, float) tuples.

    Args:
        scored_docs: List of (Document, score) tuples.

    Returns:
        List of doc_id strings from each document's metadata.
    """
    return [doc.metadata.get("doc_id", "") for doc, _ in scored_docs]


def _compute_query_metrics(
    retrieved_ids: list[str],
    golden_ids: set[str],
    generated_answer: str,
    reference_answers: list[str],
    context_docs: list[str],
    question: str,
) -> dict[str, float]:
    """Compute all metrics for a single query.

    Computes Hit Rate@k for k in [1, 3, 5, 10], MRR, Exact Match,
    token-level F1, LLM-judge correctness, and faithfulness.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (post-rerank if applicable).
        golden_ids: Set of ground-truth relevant document IDs.
        generated_answer: The LLM-generated answer string.
        reference_answers: List of valid reference answer strings.
        context_docs: Context passages sent to the LLM.
        question: The original question string.

    Returns:
        Dictionary mapping metric names to their values.
    """
    metrics: dict[str, float] = {}

    # IR metrics
    for k in settings.HIT_RATE_K_VALUES:
        metrics[f"hit_rate@{k}"] = float(hit_rate_at_k(retrieved_ids, golden_ids, k))
    metrics["mrr"] = mrr(retrieved_ids, golden_ids)

    # Deterministic generation metrics
    metrics["em"] = float(exact_match(generated_answer, reference_answers))
    metrics["f1"] = token_f1(generated_answer, reference_answers)

    # LLM-judge metrics
    metrics["correctness"] = llm_judge_correctness(
        generated_answer, reference_answers, question
    )
    metrics["faithfulness"] = llm_judge_faithfulness(
        generated_answer, context_docs, question
    )

    return metrics


def _run_single_config_per_query(
    config_name: str,
    query: str,
    golden_ids: set[str],
    reference_answers: list[str],
    bm25_retriever: BM25Retriever,
    faiss_store: FAISS,
    corpus_size: int,
    reranker: Reranker | None,
    generator: Generator,
) -> QueryResult:
    """Run a single retriever configuration for one query in per-query mode.

    In per-query mode, k is set to corpus_size so all documents are retrieved.

    Args:
        config_name: Configuration label.
        query: The question string.
        golden_ids: Set of golden document IDs.
        reference_answers: Valid answer strings.
        bm25_retriever: BM25Retriever for this query's corpus.
        faiss_store: FAISS store for this query's corpus.
        corpus_size: Total number of documents in the per-query corpus.
        reranker: Reranker instance, or None if reranking is not used.
        generator: Generator instance for answer generation.

    Returns:
        A QueryResult with all metrics computed.
    """
    # Retrieve based on config
    if config_name == CONFIG_BM25:
        retrieved_docs = retrieve_bm25(bm25_retriever, query, k=corpus_size)
    elif config_name == CONFIG_DENSE:
        dense_results = retrieve_dense(faiss_store, query, k=corpus_size)
        retrieved_docs = [doc for doc, _ in dense_results]
    elif config_name in (CONFIG_HYBRID, CONFIG_HYBRID_RERANK):
        faiss_retriever = faiss_store.as_retriever(
            search_kwargs={"k": corpus_size}
        )
        ensemble = build_ensemble_retriever(bm25_retriever, faiss_retriever)
        retrieved_docs = retrieve_hybrid(ensemble, query)
    else:
        retrieved_docs = []

    retrieved_ids_pre = _extract_doc_ids(retrieved_docs)

    # Rerank if applicable
    uses_rerank = config_name in (CONFIG_BM25, CONFIG_DENSE, CONFIG_HYBRID_RERANK)
    if uses_rerank and reranker is not None:
        reranked = reranker.rerank(query, retrieved_docs, top_k=settings.RERANKER_TOP_K)
        top_docs = [doc for doc, _ in reranked]
        retrieved_ids_post = _extract_doc_ids_from_scored(reranked)
    else:
        # No reranking — take top RERANKER_TOP_K directly
        top_docs = retrieved_docs[: settings.RERANKER_TOP_K]
        retrieved_ids_post = _extract_doc_ids(top_docs)

    # Generate answer
    context = [doc.page_content for doc in top_docs]
    answer = generator.generate(query, context)

    # Compute metrics using post-rerank IDs for IR metrics
    metrics = _compute_query_metrics(
        retrieved_ids=retrieved_ids_post,
        golden_ids=golden_ids,
        generated_answer=answer,
        reference_answers=reference_answers,
        context_docs=context,
        question=query,
    )

    return QueryResult(
        question=query,
        reference_answers=reference_answers,
        golden_doc_ids=golden_ids,
        retrieved_ids_pre_rerank=retrieved_ids_pre,
        retrieved_ids_post_rerank=retrieved_ids_post,
        generated_answer=answer,
        context_docs=context,
        metrics=metrics,
    )


def _run_single_config_full_pool(
    config_name: str,
    query: str,
    golden_ids: set[str],
    reference_answers: list[str],
    bm25_retriever: BM25Retriever,
    faiss_store: FAISS,
    reranker: Reranker | None,
    generator: Generator,
) -> QueryResult:
    """Run a single retriever configuration for one query in full-pool mode.

    In full-pool mode, k is set to RETRIEVAL_TOP_K for retrieval.

    Args:
        config_name: Configuration label.
        query: The question string.
        golden_ids: Set of golden document IDs.
        reference_answers: Valid answer strings.
        bm25_retriever: Pre-built global BM25Retriever.
        faiss_store: Pre-built global FAISS store.
        reranker: Reranker instance, or None if reranking is not used.
        generator: Generator instance for answer generation.

    Returns:
        A QueryResult with all metrics computed.
    """
    top_k = settings.RETRIEVAL_TOP_K

    # Retrieve based on config
    if config_name == CONFIG_BM25:
        retrieved_docs = retrieve_bm25(bm25_retriever, query, k=top_k)
    elif config_name == CONFIG_DENSE:
        dense_results = retrieve_dense(faiss_store, query, k=top_k)
        retrieved_docs = [doc for doc, _ in dense_results]
    elif config_name in (CONFIG_HYBRID, CONFIG_HYBRID_RERANK):
        faiss_retriever = faiss_store.as_retriever(
            search_kwargs={"k": top_k}
        )
        ensemble = build_ensemble_retriever(bm25_retriever, faiss_retriever)
        retrieved_docs = retrieve_hybrid(ensemble, query)
    else:
        retrieved_docs = []

    retrieved_ids_pre = _extract_doc_ids(retrieved_docs)

    # Rerank if applicable
    uses_rerank = config_name in (CONFIG_BM25, CONFIG_DENSE, CONFIG_HYBRID_RERANK)
    if uses_rerank and reranker is not None:
        reranked = reranker.rerank(query, retrieved_docs, top_k=settings.RERANKER_TOP_K)
        top_docs = [doc for doc, _ in reranked]
        retrieved_ids_post = _extract_doc_ids_from_scored(reranked)
    else:
        top_docs = retrieved_docs[: settings.RERANKER_TOP_K]
        retrieved_ids_post = _extract_doc_ids(top_docs)

    # Generate answer
    context = [doc.page_content for doc in top_docs]
    answer = generator.generate(query, context)

    # Compute metrics
    metrics = _compute_query_metrics(
        retrieved_ids=retrieved_ids_post,
        golden_ids=golden_ids,
        generated_answer=answer,
        reference_answers=reference_answers,
        context_docs=context,
        question=query,
    )

    return QueryResult(
        question=query,
        reference_answers=reference_answers,
        golden_doc_ids=golden_ids,
        retrieved_ids_pre_rerank=retrieved_ids_pre,
        retrieved_ids_post_rerank=retrieved_ids_post,
        generated_answer=answer,
        context_docs=context,
        metrics=metrics,
    )


def _aggregate_metrics(results: list[QueryResult]) -> dict[str, float]:
    """Compute mean metrics across a list of query results.

    Args:
        results: List of QueryResult objects with populated metrics dicts.

    Returns:
        Dictionary mapping metric names to their mean values across all queries.
        NaN values are excluded from the mean computation.
    """
    if not results:
        return {}

    all_keys = results[0].metrics.keys()
    aggregated: dict[str, float] = {}

    for key in all_keys:
        values = [r.metrics[key] for r in results if not math.isnan(r.metrics.get(key, 0.0))]
        if values:
            aggregated[key] = sum(values) / len(values)
        else:
            aggregated[key] = float("nan")

    return aggregated


class EvaluationOrchestrator:
    """Orchestrates evaluation across four retriever configurations.

    Manages the full evaluation pipeline including index construction,
    retrieval, reranking, generation, and metric computation. Implements
    sequential VRAM management to keep GPU usage under 8GB.

    Args:
        sample_size: Number of queries to evaluate. Defaults to
            ``config.settings.EVAL_SAMPLE_SIZE`` (584).
    """

    def __init__(self, sample_size: int = settings.EVAL_SAMPLE_SIZE) -> None:
        self.sample_size = sample_size

    def run_per_query(
        self, entries: list[TriviaQAEntry]
    ) -> dict[str, EvaluationResults]:
        """Run per-query evaluation across all four retriever configurations.

        For each entry (up to sample_size), builds per-query BM25 + FAISS
        indices, runs all 4 retriever configs, generates answers, and
        computes all metrics.

        VRAM management: After building FAISS indices (embedding phase),
        unloads embeddings. Loads reranker, does reranking, unloads reranker.
        Then uses Generator for LLM generation.

        Args:
            entries: List of TriviaQAEntry objects from the dataset.

        Returns:
            Dictionary mapping config names to EvaluationResults.
        """
        entries = entries[: self.sample_size]
        results: dict[str, list[QueryResult]] = {cfg: [] for cfg in ALL_CONFIGS}

        logger.info(
            "Starting per-query evaluation for %d entries across %d configs.",
            len(entries),
            len(ALL_CONFIGS),
        )

        # Phase 3: Create generator (LLM stays loaded for all queries)
        generator = Generator()

        for entry in tqdm(entries, desc="Per-query evaluation"):
            corpus = get_per_query_corpus(entry)
            corpus_size = len(corpus)
            golden_ids = {
                doc.metadata["doc_id"] for doc in entry.golden_docs
            }

            # Phase 1: Build indices (embedding model loaded)
            bm25 = build_bm25_retriever(corpus, k=corpus_size)
            faiss_store = build_faiss_store(corpus, device="cuda")

            # Unload embedding model after FAISS construction
            if hasattr(faiss_store, "embedding_function") and faiss_store.embedding_function is not None:
                unload_embeddings(faiss_store.embedding_function)

            # Phase 2: Load reranker for configs that need it
            reranker = Reranker()

            # Run all 4 configs for this entry
            for config_name in ALL_CONFIGS:
                qr = _run_single_config_per_query(
                    config_name=config_name,
                    query=entry.question,
                    golden_ids=golden_ids,
                    reference_answers=entry.answers,
                    bm25_retriever=bm25,
                    faiss_store=faiss_store,
                    corpus_size=corpus_size,
                    reranker=reranker,
                    generator=generator,
                )
                results[config_name].append(qr)

            # Unload reranker after processing all configs for this entry
            reranker.unload_model()

        # Aggregate metrics per config
        eval_results: dict[str, EvaluationResults] = {}
        for config_name in ALL_CONFIGS:
            agg = _aggregate_metrics(results[config_name])
            eval_results[config_name] = EvaluationResults(
                config_name=config_name,
                per_query_results=results[config_name],
                aggregate_metrics=agg,
            )

        return eval_results


    def run_full_pool(
        self,
        entries: list[TriviaQAEntry],
        bm25_retriever: BM25Retriever,
        faiss_store: FAISS,
        doc_map: dict[str, str],
    ) -> dict[str, EvaluationResults]:
        """Run full-pool evaluation across all four retriever configurations.

        Uses pre-built global BM25 and FAISS indices. Retrieval uses
        RETRIEVAL_TOP_K for all retrievers.

        Args:
            entries: List of TriviaQAEntry objects from the dataset.
            bm25_retriever: Pre-built global BM25Retriever.
            faiss_store: Pre-built global FAISS vector store.
            doc_map: Mapping from doc_id to document content (unused but
                available for reference).

        Returns:
            Dictionary mapping config names to EvaluationResults.
        """
        entries = entries[: self.sample_size]
        results: dict[str, list[QueryResult]] = {cfg: [] for cfg in ALL_CONFIGS}

        logger.info(
            "Starting full-pool evaluation for %d entries across %d configs.",
            len(entries),
            len(ALL_CONFIGS),
        )

        # Load reranker for configs that need it
        reranker = Reranker()

        # Create generator
        generator = Generator()

        for entry in tqdm(entries, desc="Full-pool evaluation"):
            golden_ids = {
                doc.metadata["doc_id"] for doc in entry.golden_docs
            }

            for config_name in ALL_CONFIGS:
                qr = _run_single_config_full_pool(
                    config_name=config_name,
                    query=entry.question,
                    golden_ids=golden_ids,
                    reference_answers=entry.answers,
                    bm25_retriever=bm25_retriever,
                    faiss_store=faiss_store,
                    reranker=reranker,
                    generator=generator,
                )
                results[config_name].append(qr)

        # Unload reranker after all queries
        reranker.unload_model()

        # Aggregate metrics per config
        eval_results: dict[str, EvaluationResults] = {}
        for config_name in ALL_CONFIGS:
            agg = _aggregate_metrics(results[config_name])
            eval_results[config_name] = EvaluationResults(
                config_name=config_name,
                per_query_results=results[config_name],
                aggregate_metrics=agg,
            )

        return eval_results

    def run_all_configurations(
        self,
        entries: list[TriviaQAEntry],
        mode: str = "per_query",
        bm25_retriever: BM25Retriever | None = None,
        faiss_store: FAISS | None = None,
        doc_map: dict[str, str] | None = None,
    ) -> dict[str, EvaluationResults]:
        """Orchestrate evaluation across all retriever configurations.

        Dispatches to either per-query or full-pool evaluation based on
        the mode parameter.

        Args:
            entries: List of TriviaQAEntry objects from the dataset.
            mode: Evaluation mode — ``"per_query"`` or ``"full_pool"``.
            bm25_retriever: Pre-built BM25Retriever (required for full-pool mode).
            faiss_store: Pre-built FAISS store (required for full-pool mode).
            doc_map: Document ID to content mapping (required for full-pool mode).

        Returns:
            Dictionary mapping config names to EvaluationResults.

        Raises:
            ValueError: If mode is ``"full_pool"`` but required indices are not provided.
        """
        if mode == "full_pool":
            if bm25_retriever is None or faiss_store is None or doc_map is None:
                raise ValueError(
                    "full_pool mode requires bm25_retriever, faiss_store, and doc_map."
                )
            return self.run_full_pool(entries, bm25_retriever, faiss_store, doc_map)
        else:
            return self.run_per_query(entries)

    def print_comparison_table(
        self, results: dict[str, EvaluationResults]
    ) -> None:
        """Print a markdown-formatted comparison table to the console.

        Displays one row per retriever configuration with columns:
        Retriever, Hit@1, Hit@5, Hit@10, MRR, EM, F1, Correct, Faithful.
        Numeric values are formatted to 3 decimal places.

        Args:
            results: Dictionary mapping config names to EvaluationResults.
        """
        columns = [
            ("Retriever", None),
            ("Hit@1", "hit_rate@1"),
            ("Hit@5", "hit_rate@5"),
            ("Hit@10", "hit_rate@10"),
            ("MRR", "mrr"),
            ("EM", "em"),
            ("F1", "f1"),
            ("Correct", "correctness"),
            ("Faithful", "faithfulness"),
        ]

        header = "| " + " | ".join(col[0] for col in columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"

        rows: list[str] = [header, separator]

        for config_name in ALL_CONFIGS:
            if config_name not in results:
                continue
            agg = results[config_name].aggregate_metrics
            cells: list[str] = [config_name]
            for _, metric_key in columns[1:]:
                value = agg.get(metric_key, float("nan"))  # type: ignore[arg-type]
                if math.isnan(value):
                    cells.append("N/A")
                else:
                    cells.append(f"{value:.3f}")
            rows.append("| " + " | ".join(cells) + " |")

        table = "\n".join(rows)
        print(table)

    def save_results(
        self, results: dict[str, EvaluationResults], output_dir: str
    ) -> None:
        """Save per-query results as JSON and summary comparison table as markdown.

        Creates the output directory if it doesn't exist. For each configuration,
        saves a JSON file with per-query details and aggregate metrics. Also saves
        a summary markdown file containing the comparison table.

        JSON format per config::

            {
                "config": "<config_name>",
                "queries": [
                    {
                        "question": "...",
                        "generated_answer": "...",
                        "reference_answers": ["..."],
                        "retrieved_doc_ids": ["..."],
                        "metrics": {...}
                    }
                ],
                "aggregate": {...}
            }

        Args:
            results: Dictionary mapping config names to EvaluationResults.
            output_dir: Path to the directory where output files are saved.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save per-query JSON for each config
        for config_name, eval_result in results.items():
            queries_data: list[dict] = []
            for qr in eval_result.per_query_results:
                queries_data.append(
                    {
                        "question": qr.question,
                        "generated_answer": qr.generated_answer,
                        "reference_answers": qr.reference_answers,
                        "retrieved_doc_ids": qr.retrieved_ids_post_rerank,
                        "metrics": qr.metrics,
                    }
                )

            output_data = {
                "config": config_name,
                "queries": queries_data,
                "aggregate": eval_result.aggregate_metrics,
            }

            safe_name = config_name.lower().replace("+", "_plus_").replace(" ", "_")
            json_path = os.path.join(output_dir, f"results_{safe_name}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info("Saved per-query results to %s", json_path)

        # Save summary markdown table
        columns = [
            ("Retriever", None),
            ("Hit@1", "hit_rate@1"),
            ("Hit@5", "hit_rate@5"),
            ("Hit@10", "hit_rate@10"),
            ("MRR", "mrr"),
            ("EM", "em"),
            ("F1", "f1"),
            ("Correct", "correctness"),
            ("Faithful", "faithfulness"),
        ]

        header = "| " + " | ".join(col[0] for col in columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"

        rows: list[str] = ["# Evaluation Results", "", header, separator]

        for config_name in ALL_CONFIGS:
            if config_name not in results:
                continue
            agg = results[config_name].aggregate_metrics
            cells: list[str] = [config_name]
            for _, metric_key in columns[1:]:
                value = agg.get(metric_key, float("nan"))  # type: ignore[arg-type]
                if math.isnan(value):
                    cells.append("N/A")
                else:
                    cells.append(f"{value:.3f}")
            rows.append("| " + " | ".join(cells) + " |")

        md_path = os.path.join(output_dir, "summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")

        logger.info("Saved summary markdown to %s", md_path)
