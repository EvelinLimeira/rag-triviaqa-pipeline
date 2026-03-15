"""Property-based tests for answer normalization.

Tests Properties 13 and 14 from the design document using Hypothesis.
"""

import re
import string

from hypothesis import given, settings
from hypothesis import strategies as st

from evaluation.llm_metrics import normalize_answer

# Strategy: arbitrary text strings up to 500 characters
texts = st.text(min_size=0, max_size=500)


class TestAnswerNormalizationOutputInvariants:
    """Property 13: Answer Normalization Output Invariants.

    For any input string, the normalized output should be entirely lowercase,
    contain no punctuation characters, contain no articles ("a", "an", "the"
    as standalone words), and contain no consecutive whitespace characters.

    **Validates: Requirements 12.1**
    """

    @given(s=texts)
    @settings(max_examples=100)
    def test_output_is_lowercase(self, s: str) -> None:
        """Normalized output should be entirely lowercase."""
        result = normalize_answer(s)
        assert result == result.lower(), (
            f"Expected lowercase output, got: {result!r}"
        )

    @given(s=texts)
    @settings(max_examples=100)
    def test_output_has_no_punctuation(self, s: str) -> None:
        """Normalized output should contain no punctuation characters."""
        result = normalize_answer(s)
        punct_found = [ch for ch in result if ch in string.punctuation]
        assert not punct_found, (
            f"Found punctuation {punct_found!r} in: {result!r}"
        )

    @given(s=texts)
    @settings(max_examples=100)
    def test_output_has_no_consecutive_whitespace(self, s: str) -> None:
        """Normalized output should contain no consecutive whitespace."""
        result = normalize_answer(s)
        assert "  " not in result, (
            f"Found consecutive whitespace in: {result!r}"
        )

    @given(s=texts)
    @settings(max_examples=100)
    def test_output_has_no_standalone_articles(self, s: str) -> None:
        """Normalized output should contain no standalone articles."""
        result = normalize_answer(s)
        # Check for standalone "a", "an", "the" using word boundaries
        for article in ("a", "an", "the"):
            pattern = rf"(?:^|(?<=\s)){re.escape(article)}(?=\s|$)"
            match = re.search(pattern, result)
            assert match is None, (
                f"Found standalone article '{article}' in: {result!r}"
            )


class TestAnswerNormalizationIdempotence:
    """Property 14: Answer Normalization Idempotence.

    For any input string s, normalize(normalize(s)) should equal normalize(s).

    **Validates: Requirements 12.2**
    """

    @given(s=texts)
    @settings(max_examples=100)
    def test_normalization_is_idempotent(self, s: str) -> None:
        """Applying normalization twice should equal applying it once."""
        once = normalize_answer(s)
        twice = normalize_answer(once)
        assert twice == once, (
            f"Not idempotent: normalize once={once!r}, twice={twice!r}"
        )


# ============================================================
# Unit tests for answer normalization with known inputs/outputs
# Validates: Requirements 12.1
# ============================================================


class TestNormalizeAnswerUnit:
    """Unit tests for normalize_answer with specific known inputs/outputs."""

    def test_removes_article_and_lowercases(self) -> None:
        """'The Quick Brown Fox' normalizes to 'quick brown fox'."""
        assert normalize_answer("The Quick Brown Fox") == "quick brown fox"

    def test_removes_punctuation(self) -> None:
        """'Hello, World!' normalizes to 'hello world'."""
        assert normalize_answer("Hello, World!") == "hello world"

    def test_collapses_whitespace(self) -> None:
        """'  multiple   spaces  ' normalizes to 'multiple spaces'."""
        assert normalize_answer("  multiple   spaces  ") == "multiple spaces"

    def test_empty_string(self) -> None:
        """Empty string normalizes to empty string."""
        assert normalize_answer("") == ""

    def test_only_articles(self) -> None:
        """String of only articles normalizes to empty string."""
        assert normalize_answer("a an the") == ""

    def test_mixed_punctuation_and_articles(self) -> None:
        """Complex input with articles and punctuation normalizes correctly."""
        assert normalize_answer("The cat's hat!") == "cats hat"
