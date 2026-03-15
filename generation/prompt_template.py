"""RAG prompt construction for the TriviaQA pipeline.

Builds OpenAI-format message lists for the LLM generator, ensuring
the model answers only from provided context, avoids chain-of-thought,
responds concisely, and falls back to "I don't know" when appropriate.
"""


def build_prompt(question: str, context_docs: list[str]) -> list[dict[str, str]]:
    """Build an OpenAI-format messages list for RAG answer generation.

    Constructs a system message with instructions for context-grounded,
    concise answering and a user message containing the retrieved context
    documents and the question.

    Args:
        question: The user's trivia question.
        context_docs: List of retrieved document content strings to use as context.

    Returns:
        A list of dicts with "role" and "content" keys, suitable for
        OpenAI-compatible chat completion APIs.
    """
    system_content = (
        "You are a helpful assistant that answers questions using ONLY the provided context. "
        "Do not use chain-of-thought. Answer directly. "
        "If the context does not contain the answer, respond with \"I don't know\". "
        "Keep your answer concise: 1-2 sentences maximum."
    )

    context_block = "\n\n".join(
        f"Document {i + 1}:\n{doc}" for i, doc in enumerate(context_docs)
    )

    user_content = f"Context:\n{context_block}\n\nQuestion: {question}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
