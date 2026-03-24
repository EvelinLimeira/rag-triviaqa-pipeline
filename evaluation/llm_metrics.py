"""LLM-based and deterministic evaluation metrics for the RAG pipeline.

Provides answer normalization, Exact Match, token-level F1, and
LLM-judge metrics (correctness, faithfulness) via deepeval.
"""

import logging
import math
import re
import string

import warnings

try:
    from google import genai

    _USE_NEW_GENAI = True
except ImportError:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (FutureWarning, DeprecationWarning))
        import google.generativeai as genai  # type: ignore[no-redef]

    _USE_NEW_GENAI = False

from deepeval.metrics import GEval, FaithfulnessMetric, AnswerRelevancyMetric, AnswerRelevancyMetric
from deepeval.models import DeepEvalBaseLLM, GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from config.settings import (
    DEEPEVAL_MODEL,
    DEEPEVAL_BASE_URL,
    DEEPEVAL_JUDGE_PROVIDER,
    GEMINI_MODEL,
    GEMINI_API_KEY,
)

logger = logging.getLogger(__name__)


def normalize_answer(text: str) -> str:
    """Normalize an answer string for fair comparison.

    Applies the following transformations in order:
    1. Convert to lowercase
    2. Remove articles ("a", "an", "the") as standalone words
    3. Remove all punctuation characters
    4. Collapse multiple whitespace characters into a single space
    5. Strip leading/trailing whitespace

    The function is idempotent: ``normalize(normalize(s)) == normalize(s)``.

    Args:
        text: The raw answer string to normalize.

    Returns:
        The normalized answer string.
    """
    # Lowercase
    text = text.lower()
    # Remove articles as standalone words
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # Collapse whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, references: list[str]) -> int:
    """Compute Exact Match between a prediction and reference answers.

    Normalizes both the prediction and each reference, then checks if
    any normalized reference equals the normalized prediction.

    Args:
        prediction: The generated answer string.
        references: A list of valid reference answer strings.

    Returns:
        1 if any normalized reference matches the normalized prediction,
        0 otherwise.
    """
    norm_pred = normalize_answer(prediction)
    for ref in references:
        if normalize_answer(ref) == norm_pred:
            return 1
    return 0


def token_f1(prediction: str, references: list[str]) -> float:
    """Compute token-level F1 between a prediction and reference answers.

    For each reference, tokenizes both the normalized prediction and
    normalized reference by splitting on whitespace, computes precision,
    recall, and F1 based on token overlap, then returns the maximum F1
    across all references.

    Args:
        prediction: The generated answer string.
        references: A list of valid reference answer strings.

    Returns:
        The maximum token-level F1 score across all references,
        between 0.0 and 1.0 inclusive.
    """
    norm_pred = normalize_answer(prediction)
    pred_tokens = norm_pred.split()

    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for ref in references:
        norm_ref = normalize_answer(ref)
        ref_tokens = norm_ref.split()

        if not ref_tokens:
            continue

        common = sum((min(pred_tokens.count(t), ref_tokens.count(t)) for t in set(pred_tokens) & set(ref_tokens)))

        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        best_f1 = max(best_f1, f1)

    return best_f1


def create_deepeval_model() -> GPTModel:
    """Create a deepeval model configured to use Ollama via OpenAI-compatible API.

    Returns a ``GPTModel`` instance pointing at the local Ollama endpoint
    so that all deepeval metrics use the local LLM as the judge.

    Returns:
        A ``GPTModel`` configured with the Ollama base URL and model name
        from ``config.settings``.
    """
    return GPTModel(
        model=DEEPEVAL_MODEL,
        base_url=DEEPEVAL_BASE_URL,
        api_key="ollama",
    )


class GeminiModel(DeepEvalBaseLLM):
    """Wrapper to integrate Google Gemini models with DeepEval.

    Implements the ``DeepEvalBaseLLM`` interface so that any DeepEval metric
    can use a Gemini model as the LLM judge via the Google AI SDK.

    Supports both the new ``google-genai`` package and the legacy
    ``google-generativeai`` package as a fallback.

    Args:
        model_name: Name of the Gemini model (e.g. ``"gemini-2.0-flash"``).
        api_key: Google AI API key for authentication.
    """

    def __init__(self, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self._backend = self.load_model()

    def load_model(self):
        """Create and return the appropriate SDK backend.

        Returns a ``genai.Client`` when using the new SDK, or a
        ``genai.GenerativeModel`` when using the legacy SDK.
        """
        if _USE_NEW_GENAI:
            return genai.Client(api_key=self.api_key)
        else:
            genai.configure(api_key=self.api_key)
            return genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str, schema=None) -> str:
        """Generate a response synchronously.

        Args:
            prompt: The text prompt to send to the model.
            schema: Optional Pydantic model class for structured JSON output.

        Returns:
            The generated text response.
        """
        config = None
        if schema is not None:
            try:
                from google.genai import types as genai_types

                config = genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                )
            except Exception:
                # If schema config fails, append JSON instruction to prompt
                import json as _json

                schema_hint = ""
                if hasattr(schema, "model_json_schema"):
                    schema_hint = _json.dumps(schema.model_json_schema())
                prompt = (
                    f"{prompt}\n\nYou MUST respond ONLY with valid JSON "
                    f"matching this schema:\n{schema_hint}"
                )

        if _USE_NEW_GENAI:
            if config is not None:
                response = self._backend.models.generate_content(
                    model=self.model_name, contents=prompt, config=config
                )
            else:
                response = self._backend.models.generate_content(
                    model=self.model_name, contents=prompt
                )
        else:
            response = self._backend.generate_content(prompt)
        return response.text

    async def a_generate(self, prompt: str, schema=None) -> str:
        """Generate a response asynchronously.

        Args:
            prompt: The text prompt to send to the model.
            schema: Optional Pydantic model class for structured JSON output.

        Returns:
            The generated text response.
        """
        config = None
        if schema is not None:
            try:
                from google.genai import types as genai_types

                config = genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                )
            except Exception:
                import json as _json

                schema_hint = ""
                if hasattr(schema, "model_json_schema"):
                    schema_hint = _json.dumps(schema.model_json_schema())
                prompt = (
                    f"{prompt}\n\nYou MUST respond ONLY with valid JSON "
                    f"matching this schema:\n{schema_hint}"
                )

        if _USE_NEW_GENAI:
            if config is not None:
                response = await self._backend.aio.models.generate_content(
                    model=self.model_name, contents=prompt, config=config
                )
            else:
                response = await self._backend.aio.models.generate_content(
                    model=self.model_name, contents=prompt
                )
        else:
            response = await self._backend.generate_content_async(prompt)
        return response.text

    def get_model_name(self) -> str:
        """Return the model name string."""
        return self.model_name


def create_judge_model():
    """Create the appropriate judge model based on the configured provider.

    Returns a ``GPTModel`` for provider ``"local"`` (via ``create_deepeval_model()``)
    or a ``GeminiModel`` for provider ``"gemini"`` (requires ``GEMINI_API_KEY``).

    Returns:
        A model instance compatible with DeepEval metrics.

    Raises:
        ValueError: If the provider is ``"gemini"`` but ``GEMINI_API_KEY`` is not set,
            or if the provider value is not recognized.
    """
    if DEEPEVAL_JUDGE_PROVIDER == "local":
        return create_deepeval_model()
    if DEEPEVAL_JUDGE_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY é necessária quando DEEPEVAL_JUDGE_PROVIDER é 'gemini'. "
                "Defina a variável de ambiente GEMINI_API_KEY."
            )
        return GeminiModel(GEMINI_MODEL, GEMINI_API_KEY)
    raise ValueError(
        f"Provider '{DEEPEVAL_JUDGE_PROVIDER}' não reconhecido. "
        f"Valores válidos: 'local', 'gemini'."
    )


def _faithfulness_fallback(
    prediction: str, context: list[str], question: str
) -> float:
    """Fallback for faithfulness using GEval with free-text claim decomposition.

    When the structured ``FaithfulnessMetric`` fails (e.g. because the judge
    model cannot produce valid JSON), this function evaluates faithfulness
    via a plain-text prompt that asks the model to list claims, verify each
    against the retrieval context, and return a numeric score.

    Args:
        prediction: The generated answer string.
        context: A list of retrieved context passage strings.
        question: The original question.

    Returns:
        A faithfulness score between 0.0 and 1.0, or ``float('nan')``
        if the evaluation fails.
    """
    try:
        model = create_judge_model()
        metric = GEval(
            name="FaithfulnessFallback",
            criteria=(
                "Evaluate the faithfulness of the actual output to the retrieval context. "
                "List each factual claim in the actual output, check if each claim is supported "
                "by the retrieval context, and score from 0.0 (no claims supported) to 1.0 "
                "(all claims supported)."
            ),
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            model=model,
            async_mode=True,
        )
        if not prediction or not prediction.strip():
            logger.warning("Empty prediction for faithfulness fallback. Returning NaN.")
            return float("nan")
        test_case = LLMTestCase(
            input=question,
            actual_output=prediction,
            retrieval_context=context,
        )
        metric.measure(test_case)
        return metric.score
    except Exception:
        logger.warning(
            "Faithfulness fallback (claim decomposition) failed. Returning NaN.",
            exc_info=True,
        )
        return float("nan")


def llm_judge_correctness(
    prediction: str, references: list[str], question: str
) -> float:
    """Compute LLM-judge correctness using deepeval GEval.

    Uses the local Ollama model (via deepeval) to evaluate whether the
    prediction is factually correct with respect to the reference answers.

    Args:
        prediction: The generated answer string.
        references: A list of valid reference answer strings.
        question: The original question.

    Returns:
        A correctness score between 0.0 and 1.0, or ``float('nan')``
        if the LLM judge is unreachable.
    """
    try:
        model = create_judge_model()
        metric = GEval(
            name="Correctness",
            criteria=(
                "Determine whether the actual output is factually correct "
                "based on the expected output."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=model,
            async_mode=True,
        )
        if not prediction or not prediction.strip():
            logger.warning("Empty prediction for correctness. Returning NaN.")
            return float("nan")
        test_case = LLMTestCase(
            input=question,
            actual_output=prediction,
            expected_output="; ".join(references),
        )
        metric.measure(test_case)
        return metric.score
    except Exception:
        logger.warning(
            "LLM judge correctness failed (connection or model error). "
            "Returning NaN.",
            exc_info=True,
        )
        return float("nan")


def llm_judge_faithfulness(
    prediction: str, context: list[str], question: str
) -> float:
    """Compute LLM-judge faithfulness using deepeval FaithfulnessMetric.

    Uses the configured judge model to evaluate whether the prediction is
    faithful to the provided retrieval context. If the native
    ``FaithfulnessMetric`` fails (e.g. due to JSON parsing errors with
    smaller models), automatically falls back to a GEval-based claim
    decomposition approach.

    Args:
        prediction: The generated answer string.
        context: A list of retrieved context passage strings.
        question: The original question.

    Returns:
        A faithfulness score between 0.0 and 1.0, or ``float('nan')``
        if both the native metric and fallback fail.
    """
    try:
        model = create_judge_model()
        metric = FaithfulnessMetric(
            model=model,
            async_mode=True,
        )
        if not prediction or not prediction.strip():
            logger.warning("Empty prediction for faithfulness. Returning NaN.")
            return float("nan")
        test_case = LLMTestCase(
            input=question,
            actual_output=prediction,
            retrieval_context=context,
        )
        metric.measure(test_case)
        logger.info("Faithfulness computed via DeepEval native method.")
        return metric.score
    except Exception:
        logger.warning(
            "DeepEval FaithfulnessMetric failed. Trying fallback...",
            exc_info=True,
        )
        score = _faithfulness_fallback(prediction, context, question)
        if not math.isnan(score):
            logger.info("Faithfulness computed via fallback (claim decomposition).")
            return score
        logger.warning("Both faithfulness methods failed. Returning NaN.")
        return float("nan")


def llm_judge_relevancy(
    prediction: str, context: list[str], question: str
) -> float:
    """Compute LLM-judge answer relevancy using deepeval AnswerRelevancyMetric.

    Uses the configured judge model to evaluate whether the generated answer
    is relevant to the original question, considering the retrieval context.

    Args:
        prediction: The generated answer string.
        context: A list of retrieved context passage strings.
        question: The original question.

    Returns:
        A relevancy score between 0.0 and 1.0, or ``float('nan')``
        if the LLM judge is unreachable.
    """
    try:
        model = create_judge_model()
        metric = AnswerRelevancyMetric(
            model=model,
            async_mode=True,
        )
        if not prediction or not prediction.strip():
            logger.warning("Empty prediction for relevancy. Returning NaN.")
            return float("nan")
        test_case = LLMTestCase(
            input=question,
            actual_output=prediction,
            retrieval_context=context,
        )
        metric.measure(test_case)
        return metric.score
    except Exception:
        logger.warning(
            "LLM judge relevancy failed (connection or model error). "
            "Returning NaN.",
            exc_info=True,
        )
        return float("nan")
