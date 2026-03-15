"""Information retrieval metrics for the RAG pipeline.

Provides Hit Rate@k and Mean Reciprocal Rank (MRR) computation,
both per-query and aggregated across a set of queries.
"""

from config.settings import HIT_RATE_K_VALUES


def hit_rate_at_k(retrieved_ids: list[str], golden_ids: set[str], k: int) -> int:
    """Check whether any golden document appears in the top-k retrieved results.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first).
        golden_ids: Set of ground-truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        1 if at least one golden document is in ``retrieved_ids[:k]``, else 0.
    """
    for doc_id in retrieved_ids[:k]:
        if doc_id in golden_ids:
            return 1
    return 0


def mrr(retrieved_ids: list[str], golden_ids: set[str]) -> float:
    """Compute the reciprocal rank of the first golden document.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first).
        golden_ids: Set of ground-truth relevant document IDs.

    Returns:
        ``1 / rank`` where *rank* is the 1-based position of the first golden
        document in *retrieved_ids*, or 0.0 if no golden document appears.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in golden_ids:
            return 1.0 / rank
    return 0.0


def compute_ir_metrics(
    all_retrieved: list[list[str]],
    all_golden: list[set[str]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Aggregate IR metrics across a set of queries.

    Computes mean Hit Rate@k for each *k* in *k_values* and mean MRR.

    Args:
        all_retrieved: Per-query lists of retrieved document IDs.
        all_golden: Per-query sets of golden document IDs.
        k_values: Values of *k* for Hit Rate computation.
            Defaults to ``HIT_RATE_K_VALUES`` from config (``[1, 3, 5, 10]``).

    Returns:
        Dictionary with keys like ``"hit_rate@1"``, ``"hit_rate@3"``, …,
        ``"mrr"`` mapped to their mean values across all queries.
    """
    if k_values is None:
        k_values = HIT_RATE_K_VALUES

    n = len(all_retrieved)
    if n == 0:
        metrics: dict[str, float] = {f"hit_rate@{k}": 0.0 for k in k_values}
        metrics["mrr"] = 0.0
        return metrics

    # Accumulate per-query scores
    hit_sums: dict[int, int] = {k: 0 for k in k_values}
    mrr_sum: float = 0.0

    for retrieved_ids, golden_ids in zip(all_retrieved, all_golden):
        for k in k_values:
            hit_sums[k] += hit_rate_at_k(retrieved_ids, golden_ids, k)
        mrr_sum += mrr(retrieved_ids, golden_ids)

    metrics = {f"hit_rate@{k}": hit_sums[k] / n for k in k_values}
    metrics["mrr"] = mrr_sum / n
    return metrics
