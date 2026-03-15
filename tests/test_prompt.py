"""Property-based tests for generation/prompt_template.py (Property 11).

Tests the RAG prompt construction using Hypothesis to verify that all
required elements are present in the generated prompt messages.
"""

from hypothesis import strategies as st, given, settings

from generation.prompt_template import build_prompt

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

questions = st.text(min_size=1, max_size=200)
context_docs = st.lists(st.text(min_size=1, max_size=500), min_size=1, max_size=5)


# ---------------------------------------------------------------------------
# Property 11: Prompt Contains All Required Elements
# ---------------------------------------------------------------------------
# Feature: rag-trivia-pipeline, Property 11: Prompt Contains All Required Elements
# **Validates: Requirements 10.1, 10.2, 10.3, 10.4**


@given(question=questions, docs=context_docs)
@settings(max_examples=100)
def test_prompt_contains_all_required_elements(question: str, docs: list[str]) -> None:
    """For any question and context docs, the prompt must contain:
    (a) system instruction mentioning answering from context only,
    (b) instruction to disable chain-of-thought,
    (c) instruction to say "I don't know" when context lacks the answer,
    (d) conciseness instruction (1-2 sentences),
    (e) the context documents,
    (f) the question.
    """
    messages = build_prompt(question, docs)

    # Result is a list of exactly 2 dicts (system + user)
    assert isinstance(messages, list)
    assert len(messages) == 2

    system_msg = messages[0]
    user_msg = messages[1]

    assert system_msg["role"] == "system"
    assert user_msg["role"] == "user"

    system_content = system_msg["content"].lower()
    user_content = user_msg["content"]

    # (a) System instruction mentions answering from context only
    assert "context" in system_content

    # (b) Instruction to disable chain-of-thought
    assert "chain-of-thought" in system_content

    # (c) Instruction to say "I don't know"
    assert "i don't know" in system_content

    # (d) Conciseness instruction (1-2 sentences)
    assert "1-2 sentences" in system_content or "concise" in system_content

    # (e) User message contains each context document
    for doc in docs:
        assert doc in user_content, f"Context doc not found in user message: {doc!r}"

    # (f) User message contains the question
    assert question in user_content, f"Question not found in user message: {question!r}"


# ---------------------------------------------------------------------------
# Unit tests for prompt construction
# ---------------------------------------------------------------------------
# **Validates: Requirements 10.1**


class TestBuildPromptStructure:
    """Unit tests for build_prompt output structure and content."""

    def test_prompt_output_has_two_messages_with_correct_roles(self) -> None:
        """Prompt should return exactly 2 messages: system and user."""
        messages = build_prompt("What is Python?", ["Python is a language."])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_prompt_with_empty_context_docs_list(self) -> None:
        """Prompt should still produce valid messages when context_docs is empty."""
        messages = build_prompt("What is Python?", [])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # Question should still appear in user message
        assert "What is Python?" in messages[1]["content"]

    def test_prompt_with_single_context_doc(self) -> None:
        """Prompt should include the single context document in the user message."""
        doc = "Paris is the capital of France."
        messages = build_prompt("What is the capital of France?", [doc])
        assert len(messages) == 2
        assert doc in messages[1]["content"]
        assert "What is the capital of France?" in messages[1]["content"]
