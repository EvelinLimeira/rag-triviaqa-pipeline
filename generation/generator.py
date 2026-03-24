"""LangChain ChatOpenAI integration for answer generation via Ollama.

Provides the Generator class that wraps a ChatOpenAI instance configured
to call a local Ollama LLM (Qwen3.5-9B) through its OpenAI-compatible API.
Handles prompt construction, message formatting, think-tag stripping, and
connection error handling.
"""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from generation.prompt_template import build_prompt

logger = logging.getLogger(__name__)


class Generator:
    """Generates answers using a local LLM via LangChain's ChatOpenAI wrapper.

    Connects to an Ollama instance through its OpenAI-compatible API endpoint,
    constructs prompts from retrieved context documents, and returns clean
    answer strings with any ``<think>`` tags stripped.

    Args:
        model: The Ollama model name. Defaults to ``config.settings.LLM_MODEL``.
        base_url: The Ollama OpenAI-compatible API base URL.
            Defaults to ``config.settings.LLM_BASE_URL``.
        temperature: Sampling temperature. Defaults to ``config.settings.LLM_TEMPERATURE``.
        max_tokens: Maximum tokens in the generated response.
            Defaults to ``config.settings.LLM_MAX_TOKENS``.
    """

    def __init__(
        self,
        model: str = settings.LLM_MODEL,
        base_url: str = settings.LLM_BASE_URL,
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = settings.LLM_MAX_TOKENS,
    ) -> None:
        self.llm = ChatOpenAI(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key="ollama",  # dummy key — Ollama ignores it
        )

    def generate(self, question: str, context_docs: list[str]) -> str:
        """Generate an answer for a question given retrieved context documents.

        Builds a prompt from the question and context, invokes the LLM via
        ChatOpenAI with SystemMessage + HumanMessage, strips any thinking
        tags from the response, and returns the answer string.

        Args:
            question: The user's trivia question.
            context_docs: List of retrieved document content strings.

        Returns:
            The generated answer text with thinking tags removed.

        Raises:
            ConnectionError: If the Ollama service is not running or unreachable.
        """
        prompt_messages = build_prompt(question, context_docs)

        messages = [
            SystemMessage(content=prompt_messages[0]["content"]),
            HumanMessage(content=prompt_messages[1]["content"]),
        ]

        try:
            response = self.llm.invoke(messages)
        except Exception as exc:
            error_msg = str(exc).lower()
            if any(
                keyword in error_msg
                for keyword in ("connection", "refused", "unreachable", "connect")
            ):
                raise ConnectionError(
                    "Ollama is not running. Start with: ollama serve"
                ) from exc
            raise

        content = response.content if response.content else ""
        if not content:
            logger.warning("LLM returned an empty response for question: %s", question)

        return self.strip_thinking_tags(content)

    def strip_thinking_tags(self, text: str) -> str:
        """Remove ``<think>...</think>`` tags and their content from text.

        Handles multiline content within think tags. If no think tags are
        present, the original text is returned unchanged (after stripping
        leading/trailing whitespace).

        Args:
            text: The raw LLM response text.

        Returns:
            The cleaned text with think tags and their content removed.
        """
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()
