"""Property-based tests for generation/generator.py (Property 12).

Tests the think tag stripping logic using Hypothesis to verify that
``<think>...</think>`` tags and their content are removed, and strings
without think tags are returned unchanged (after whitespace stripping).
"""

import os
import sys
from unittest.mock import MagicMock, patch

from hypothesis import strategies as st, given, settings

# Set a dummy API key so ChatOpenAI doesn't raise on construction.
# The LLM is never actually called in these tests.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")

from generation.generator import Generator

# ---------------------------------------------------------------------------
# Shared Generator instance (only strip_thinking_tags is exercised)
# ---------------------------------------------------------------------------
_generator = Generator()

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strings guaranteed to contain no think tags
no_think_tags = (
    st.text(min_size=1, max_size=200)
    .filter(lambda s: "<think>" not in s and "</think>" not in s)
)

# Components for building strings *with* think tags
think_content = st.text(min_size=0, max_size=100)
before_text = (
    st.text(min_size=0, max_size=100)
    .filter(lambda s: "<think>" not in s and "</think>" not in s)
)
after_text = (
    st.text(min_size=0, max_size=100)
    .filter(lambda s: "<think>" not in s and "</think>" not in s)
)


# ---------------------------------------------------------------------------
# Property 12: Think Tag Stripping
# ---------------------------------------------------------------------------
# Feature: rag-trivia-pipeline, Property 12: Think Tag Stripping
# **Validates: Requirements 11.4**


@given(s=no_think_tags)
@settings(max_examples=100)
def test_no_think_tags_returns_stripped_string(s: str) -> None:
    """For any string without think tags, strip_thinking_tags should return
    the string unchanged after stripping leading/trailing whitespace."""
    result = _generator.strip_thinking_tags(s)
    assert result == s.strip()


@given(before=before_text, content=think_content, after=after_text)
@settings(max_examples=100)
def test_think_tags_are_removed(before: str, content: str, after: str) -> None:
    """For any string containing ``<think>...</think>`` tags, the result
    should not contain ``<think>`` or ``</think>``, and should contain the
    text that was outside the tags (stripped)."""
    text = f"{before}<think>{content}</think>{after}"
    result = _generator.strip_thinking_tags(text)

    # Tags must be gone
    assert "<think>" not in result
    assert "</think>" not in result

    # The before/after text (stripped) should be present in the result
    expected = f"{before}{after}".strip()
    assert result == expected


# ---------------------------------------------------------------------------
# Unit tests for think tag stripping and Generator error handling
# ---------------------------------------------------------------------------
# **Validates: Requirements 10.1, 11.1, 11.3, 11.4**


class TestStripThinkingTags:
    """Unit tests for Generator.strip_thinking_tags with various patterns."""

    def test_multiple_think_tags_in_one_string(self) -> None:
        """Multiple think tags should all be removed."""
        text = "<think>first</think>Hello <think>second</think>World"
        result = _generator.strip_thinking_tags(text)
        assert result == "Hello World"

    def test_nested_looking_tags(self) -> None:
        """Nested-looking tags should be handled by the greedy-minimal regex."""
        text = "<think>outer<think>inner</think>outer</think>"
        result = _generator.strip_thinking_tags(text)
        # re.sub with .*? (non-greedy) matches first <think> to first </think>
        # so "<think>outer<think>inner</think>" is removed, leaving "outer</think>"
        # then strip removes whitespace; the remaining </think> stays as plain text
        assert "<think>" not in result

    def test_think_tags_at_start_of_string(self) -> None:
        """Think tag at the start should be removed, leaving the rest."""
        text = "<think>reasoning here</think>The answer is 42."
        result = _generator.strip_thinking_tags(text)
        assert result == "The answer is 42."

    def test_think_tags_at_end_of_string(self) -> None:
        """Think tag at the end should be removed, leaving the rest."""
        text = "The answer is 42.<think>some thought</think>"
        result = _generator.strip_thinking_tags(text)
        assert result == "The answer is 42."

    def test_empty_think_tags(self) -> None:
        """Empty think tags ``<think></think>`` should be removed."""
        text = "Before<think></think>After"
        result = _generator.strip_thinking_tags(text)
        assert result == "BeforeAfter"

    def test_no_think_tags_plain_text(self) -> None:
        """Plain text without think tags should be returned unchanged (stripped)."""
        text = "  Just a plain answer.  "
        result = _generator.strip_thinking_tags(text)
        assert result == "Just a plain answer."


class TestGeneratorConnectionError:
    """Unit tests for Generator raising ConnectionError when Ollama is down."""

    def test_raises_connection_error_on_connection_refused(self) -> None:
        """Generator.generate should raise ConnectionError when Ollama is unreachable."""
        from unittest.mock import patch

        gen = Generator()
        with patch(
            "langchain_openai.ChatOpenAI.invoke",
            side_effect=Exception("Connection refused"),
        ):
            try:
                gen.generate("What is Python?", ["Python is a language."])
                assert False, "Expected ConnectionError"
            except ConnectionError as e:
                assert "Ollama is not running" in str(e)

    def test_raises_connection_error_on_unreachable(self) -> None:
        """Generator.generate should raise ConnectionError for 'unreachable' errors."""
        from unittest.mock import patch

        gen = Generator()
        with patch(
            "langchain_openai.ChatOpenAI.invoke",
            side_effect=Exception("Host unreachable"),
        ):
            try:
                gen.generate("Test?", ["doc"])
                assert False, "Expected ConnectionError"
            except ConnectionError as e:
                assert "Ollama is not running" in str(e)

    def test_non_connection_error_is_reraised(self) -> None:
        """Non-connection exceptions should be re-raised as-is."""
        from unittest.mock import patch
        import pytest

        gen = Generator()
        with patch(
            "langchain_openai.ChatOpenAI.invoke",
            side_effect=ValueError("some other error"),
        ):
            with pytest.raises(ValueError, match="some other error"):
                gen.generate("Test?", ["doc"])
