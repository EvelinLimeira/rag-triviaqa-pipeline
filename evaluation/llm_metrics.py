"""LLM-based and deterministic evaluation metrics for the RAG pipeline.

Provides answer normalization, Exact Match, token-level F1, and
LLM-judge metrics (correctness, faithfulness) via deepeval.
"""

import logging
import re
import string

from deepeval.metrics import GEval, FaithfulnessMetric
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from config.settings import DEEPEVAL_MODEL, DEEPEVAL_BASE_URL

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
        model = create_deepeval_model()
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
            async_mode=False,
        )
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

    Uses the local Ollama model (via deepeval) to evaluate whether the
    prediction is faithful to the provided retrieval context.

    Args:
        prediction: The generated answer string.
        context: A list of retrieved context passage strings.
        question: The original question.

    Returns:
        A faithfulness score between 0.0 and 1.0, or ``float('nan')``
        if the LLM judge is unreachable.
    """
    try:
        model = create_deepeval_model()
        metric = FaithfulnessMetric(
            model=model,
            async_mode=False,
        )
        test_case = LLMTestCase(
            input=question,
            actual_output=prediction,
            retrieval_context=context,
        )
        metric.measure(test_case)
        return metric.score
    except Exception:
        logger.warning(
            "LLM judge faithfulness failed (connection or model error). "
            "Returning NaN.",
            exc_info=True,
        )
        return float("nan")
