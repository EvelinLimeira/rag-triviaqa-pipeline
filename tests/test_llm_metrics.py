"""Property-based tests for EM and F1 metrics.

Tests Properties 17 and 18 from the design document using Hypothesis.
"""

from hypothesis import given, settings, strategies as st

from evaluation.llm_metrics import exact_match, token_f1

# --- Hypothesis strategies ---
# Non-empty text with at least some alphanumeric content
answer_texts = (
    st.text(
        min_size=1,
        max_size=100,
        alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
    )
    .filter(lambda s: len(s.split()) > 0)  # must have tokens after basic split
    .filter(lambda s: any(c.isalnum() for c in s))  # must have alphanumeric content
)
reference_lists = st.lists(answer_texts, min_size=1, max_size=5)


# Feature: rag-trivia-pipeline, Property 17: EM and F1 Max Across References
# **Validates: Requirements 14.1, 14.2, 14.6**
class TestEMAndF1MaxAcrossReferences:
    """Property 17: Adding more references should never decrease EM or F1 score."""

    @given(
        prediction=answer_texts,
        refs=reference_lists,
        extra_refs=reference_lists,
    )
    @settings(max_examples=100)
    def test_em_monotonic_on_adding_references(
        self,
        prediction: str,
        refs: list[str],
        extra_refs: list[str],
    ) -> None:
        """EM with a subset of references should be <= EM with the full set."""
        score_subset = exact_match(prediction, refs)
        score_superset = exact_match(prediction, refs + extra_refs)
        assert score_superset >= score_subset, (
            f"EM decreased when adding references: "
            f"subset={score_subset}, superset={score_superset}, "
            f"prediction={prediction!r}, refs={refs!r}, extra={extra_refs!r}"
        )

    @given(
        prediction=answer_texts,
        refs=reference_lists,
        extra_refs=reference_lists,
    )
    @settings(max_examples=100)
    def test_f1_monotonic_on_adding_references(
        self,
        prediction: str,
        refs: list[str],
        extra_refs: list[str],
    ) -> None:
        """F1 with a subset of references should be <= F1 with the full set."""
        score_subset = token_f1(prediction, refs)
        score_superset = token_f1(prediction, refs + extra_refs)
        assert score_superset >= score_subset, (
            f"F1 decreased when adding references: "
            f"subset={score_subset}, superset={score_superset}, "
            f"prediction={prediction!r}, refs={refs!r}, extra={extra_refs!r}"
        )


# Feature: rag-trivia-pipeline, Property 18: Token F1 Bounds and Identity
# **Validates: Requirements 14.2**
class TestTokenF1BoundsAndIdentity:
    """Property 18: F1 is in [0, 1] and f1(s, [s]) == 1.0 for non-empty strings."""

    @given(prediction=answer_texts, refs=reference_lists)
    @settings(max_examples=100)
    def test_f1_bounded_between_0_and_1(
        self,
        prediction: str,
        refs: list[str],
    ) -> None:
        """Token F1 score should be between 0.0 and 1.0 inclusive."""
        score = token_f1(prediction, refs)
        assert 0.0 <= score <= 1.0, (
            f"F1 out of bounds: {score}, "
            f"prediction={prediction!r}, refs={refs!r}"
        )

    @given(s=answer_texts)
    @settings(max_examples=100)
    def test_f1_identity(self, s: str) -> None:
        """token_f1(s, [s]) should equal 1.0 for any non-empty string."""
        score = token_f1(s, [s])
        assert score == 1.0, (
            f"F1 identity failed: token_f1({s!r}, [{s!r}]) = {score}, expected 1.0"
        )


# ============================================================
# Unit tests for EM and F1 metrics
# Validates: Requirements 14.1, 14.2
# ============================================================


class TestExactMatchUnit:
    """Unit tests for exact_match with known inputs."""

    def test_em_matching_case_insensitive(self) -> None:
        """EM returns 1 when prediction matches a reference (case-insensitive)."""
        assert exact_match("Paris", ["Paris", "paris"]) == 1

    def test_em_non_matching(self) -> None:
        """EM returns 0 when prediction does not match any reference."""
        assert exact_match("London", ["Paris"]) == 0

    def test_em_matching_with_articles(self) -> None:
        """EM returns 1 when normalization strips articles to produce a match."""
        assert exact_match("The Eiffel Tower", ["eiffel tower"]) == 1

    def test_em_empty_prediction(self) -> None:
        """EM returns 0 when prediction is empty and references are non-empty."""
        assert exact_match("", ["Paris"]) == 0

    def test_em_matching_with_punctuation(self) -> None:
        """EM returns 1 when punctuation differences are normalized away."""
        assert exact_match("Hello, World!", ["hello world"]) == 1


class TestTokenF1Unit:
    """Unit tests for token_f1 with known inputs."""

    def test_f1_high_overlap(self) -> None:
        """F1 is high when prediction and reference share most tokens."""
        score = token_f1("the capital of France", ["capital of France"])
        # pred tokens: "capital france" (2), ref tokens: "capital france" (2)
        # After normalization both become "capital france" → F1 = 1.0
        assert score == 1.0

    def test_f1_no_overlap(self) -> None:
        """F1 is 0.0 when there is no token overlap."""
        score = token_f1("apple banana", ["cherry grape"])
        assert score == 0.0

    def test_f1_partial_overlap(self) -> None:
        """F1 reflects partial token overlap correctly."""
        # pred: "big red dog" → tokens: ["big", "red", "dog"]
        # ref: "small red dog" → tokens: ["small", "red", "dog"]
        # common = 2, precision = 2/3, recall = 2/3, F1 = 2*(2/3)*(2/3)/((2/3)+(2/3)) = 2/3
        score = token_f1("big red dog", ["small red dog"])
        assert abs(score - 2 / 3) < 1e-9

    def test_f1_identical_strings(self) -> None:
        """F1 is 1.0 when prediction equals the reference."""
        assert token_f1("Paris", ["Paris"]) == 1.0

    def test_f1_empty_prediction(self) -> None:
        """F1 is 0.0 when prediction is empty."""
        assert token_f1("", ["Paris"]) == 0.0
