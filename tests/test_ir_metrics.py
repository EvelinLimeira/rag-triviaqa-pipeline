"""Property-based tests for IR metrics (Hit Rate@k and MRR).

Tests Properties 15 and 16 from the design document using Hypothesis.
"""

from hypothesis import given, settings, strategies as st

from evaluation.ir_metrics import hit_rate_at_k, mrr

# --- Hypothesis strategies ---
doc_ids = st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N")))
retrieved_lists = st.lists(doc_ids, min_size=1, max_size=20, unique=True)
golden_sets = st.frozensets(doc_ids, min_size=1, max_size=5)
k_values = st.integers(min_value=1, max_value=20)


# Feature: rag-trivia-pipeline, Property 15: Hit Rate@k Correctness
# **Validates: Requirements 13.1, 13.3**
class TestHitRateAtKCorrectness:
    """Property 15: Hit Rate@k returns 1 iff intersection of top-k and golden is non-empty."""

    @given(retrieved=retrieved_lists, golden=golden_sets, k=k_values)
    @settings(max_examples=100)
    def test_hit_rate_matches_set_intersection(
        self, retrieved: list[str], golden: frozenset[str], k: int
    ) -> None:
        """hit_rate_at_k returns 1 iff set(retrieved[:k]) & golden is non-empty."""
        result = hit_rate_at_k(retrieved, set(golden), k)
        top_k_ids = set(retrieved[:k])
        has_overlap = len(top_k_ids & golden) > 0
        expected = 1 if has_overlap else 0
        assert result == expected, (
            f"hit_rate_at_k={result}, expected={expected}, "
            f"top_k_ids={top_k_ids}, golden={golden}"
        )


# Feature: rag-trivia-pipeline, Property 16: MRR Correctness
# **Validates: Requirements 13.2, 13.3**
class TestMRRCorrectness:
    """Property 16: MRR returns 1/rank of first golden doc, or 0 if none."""

    @given(retrieved=retrieved_lists, golden=golden_sets)
    @settings(max_examples=100)
    def test_mrr_matches_first_golden_rank(
        self, retrieved: list[str], golden: frozenset[str]
    ) -> None:
        """mrr returns 1/rank where rank is 1-based position of first golden doc, or 0."""
        result = mrr(retrieved, set(golden))

        # Compute expected value independently
        expected = 0.0
        for i, doc_id in enumerate(retrieved):
            if doc_id in golden:
                expected = 1.0 / (i + 1)
                break

        assert result == expected, (
            f"mrr={result}, expected={expected}, "
            f"retrieved={retrieved}, golden={golden}"
        )


# ============================================================
# Unit tests for IR metrics (Hit Rate@k and MRR)
# Validates: Requirements 13.1, 13.2
# ============================================================


class TestHitRateAtKUnit:
    """Unit tests for hit_rate_at_k with known inputs."""

    def test_golden_at_position_1(self) -> None:
        """Hit rate is 1 when golden doc is at position 1."""
        assert hit_rate_at_k(["gold", "a", "b"], {"gold"}, k=1) == 1

    def test_golden_beyond_k(self) -> None:
        """Hit rate is 0 when golden doc is beyond position k."""
        assert hit_rate_at_k(["a", "b", "gold"], {"gold"}, k=2) == 0

    def test_golden_at_exactly_k(self) -> None:
        """Hit rate is 1 when golden doc is at exactly position k."""
        assert hit_rate_at_k(["a", "b", "gold"], {"gold"}, k=3) == 1

    def test_no_golden_in_results(self) -> None:
        """Hit rate is 0 when no golden doc appears in results."""
        assert hit_rate_at_k(["a", "b", "c"], {"gold"}, k=3) == 0

    def test_multiple_golden_docs(self) -> None:
        """Hit rate is 1 when any golden doc appears in top-k."""
        assert hit_rate_at_k(["a", "gold2", "b"], {"gold1", "gold2"}, k=2) == 1


class TestMRRUnit:
    """Unit tests for mrr with known inputs."""

    def test_no_golden_in_results(self) -> None:
        """MRR is 0 when no golden doc appears in results."""
        assert mrr(["a", "b", "c"], {"gold"}) == 0.0

    def test_golden_at_position_3(self) -> None:
        """MRR is 1/3 when first golden doc is at position 3."""
        result = mrr(["a", "b", "gold"], {"gold"})
        assert abs(result - 1 / 3) < 1e-9

    def test_golden_at_position_1(self) -> None:
        """MRR is 1.0 when golden doc is at position 1."""
        assert mrr(["gold", "a", "b"], {"gold"}) == 1.0

    def test_multiple_golden_returns_first(self) -> None:
        """MRR uses the rank of the first golden doc found."""
        # gold2 at position 2, gold1 at position 4 → MRR = 1/2
        result = mrr(["a", "gold2", "b", "gold1"], {"gold1", "gold2"})
        assert abs(result - 0.5) < 1e-9

    def test_empty_retrieved_list(self) -> None:
        """MRR is 0 when retrieved list is empty."""
        assert mrr([], {"gold"}) == 0.0
