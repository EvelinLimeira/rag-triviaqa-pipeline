"""Property-based tests for EM and F1 metrics.

Tests Properties 17 and 18 from the design document using Hypothesis.
"""

from hypothesis import given, HealthCheck, settings, strategies as st

from evaluation.llm_metrics import exact_match, normalize_answer, token_f1

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
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
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

    @given(s=answer_texts.filter(lambda s: len(normalize_answer(s).split()) > 0))
    @settings(max_examples=100)
    def test_f1_identity(self, s: str) -> None:
        """token_f1(s, [s]) should equal 1.0 for any string with tokens after normalization."""
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


# ============================================================
# Unit tests for GeminiModel and create_judge_model()
# Validates: Requirements 3.3, 3.4, 3.5, 3.6, 5.2
# ============================================================

from unittest.mock import patch, MagicMock

import pytest

from deepeval.models import DeepEvalBaseLLM, GPTModel
from evaluation.llm_metrics import GeminiModel, create_judge_model


class TestGeminiModel:
    """Unit tests for the GeminiModel wrapper class."""

    @patch("evaluation.llm_metrics.genai")
    def test_get_model_name_returns_correct_name(self, mock_genai: MagicMock) -> None:
        """get_model_name() returns the model name passed to __init__.

        Validates: Requirements 3.5
        """
        model = GeminiModel(model_name="gemini-2.0-flash", api_key="fake-key")
        assert model.get_model_name() == "gemini-2.0-flash"

    @patch("evaluation.llm_metrics._USE_NEW_GENAI", True)
    @patch("evaluation.llm_metrics.genai")
    def test_generate_calls_new_sdk_and_returns_text(self, mock_genai: MagicMock) -> None:
        """generate() delegates to the new google-genai SDK and returns response text.

        Validates: Requirements 3.5
        """
        mock_response = MagicMock()
        mock_response.text = "Generated answer"
        mock_client = mock_genai.Client.return_value
        mock_client.models.generate_content.return_value = mock_response

        model = GeminiModel(model_name="gemini-2.0-flash", api_key="fake-key")
        result = model.generate("What is 2+2?")

        assert result == "Generated answer"
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-2.0-flash", contents="What is 2+2?"
        )

    @patch("evaluation.llm_metrics._USE_NEW_GENAI", False)
    @patch("evaluation.llm_metrics.genai")
    def test_generate_calls_legacy_sdk_and_returns_text(self, mock_genai: MagicMock) -> None:
        """generate() delegates to the legacy google-generativeai SDK and returns response text.

        Validates: Requirements 3.5
        """
        mock_response = MagicMock()
        mock_response.text = "Generated answer"
        mock_genai.GenerativeModel.return_value.generate_content.return_value = (
            mock_response
        )

        model = GeminiModel(model_name="gemini-2.0-flash", api_key="fake-key")
        result = model.generate("What is 2+2?")

        assert result == "Generated answer"
        mock_genai.GenerativeModel.return_value.generate_content.assert_called_once_with(
            "What is 2+2?"
        )

    @patch("evaluation.llm_metrics.genai")
    def test_inherits_from_deepeval_base_llm(self, mock_genai: MagicMock) -> None:
        """GeminiModel is a subclass of DeepEvalBaseLLM.

        Validates: Requirements 3.4
        """
        model = GeminiModel(model_name="gemini-2.0-flash", api_key="fake-key")
        assert isinstance(model, DeepEvalBaseLLM)

    @patch("evaluation.llm_metrics._USE_NEW_GENAI", True)
    @patch("evaluation.llm_metrics.genai")
    def test_load_model_creates_client_new_sdk(self, mock_genai: MagicMock) -> None:
        """load_model() creates a genai.Client with the new SDK.

        Validates: Requirements 3.5
        """
        GeminiModel(model_name="gemini-2.0-flash", api_key="test-api-key")
        mock_genai.Client.assert_called_with(api_key="test-api-key")

    @patch("evaluation.llm_metrics._USE_NEW_GENAI", False)
    @patch("evaluation.llm_metrics.genai")
    def test_load_model_configures_legacy_sdk(self, mock_genai: MagicMock) -> None:
        """load_model() calls genai.configure with the legacy SDK.

        Validates: Requirements 3.5
        """
        GeminiModel(model_name="gemini-2.0-flash", api_key="test-api-key")
        mock_genai.configure.assert_called_with(api_key="test-api-key")
        mock_genai.GenerativeModel.assert_called_with("gemini-2.0-flash")


class TestCreateJudgeModel:
    """Unit tests for the create_judge_model() factory function."""

    @patch("evaluation.llm_metrics.DEEPEVAL_JUDGE_PROVIDER", "local")
    @patch("evaluation.llm_metrics.create_deepeval_model")
    def test_local_provider_returns_gpt_model(
        self, mock_create: MagicMock
    ) -> None:
        """Factory returns GPTModel when provider is 'local'.

        Validates: Requirements 3.3, 5.2
        """
        mock_model = MagicMock(spec=GPTModel)
        mock_create.return_value = mock_model

        result = create_judge_model()

        mock_create.assert_called_once()
        assert result is mock_model

    @patch("evaluation.llm_metrics.genai")
    @patch("evaluation.llm_metrics.GEMINI_API_KEY", "fake-key")
    @patch("evaluation.llm_metrics.GEMINI_MODEL", "gemini-2.0-flash")
    @patch("evaluation.llm_metrics.DEEPEVAL_JUDGE_PROVIDER", "gemini")
    def test_gemini_provider_returns_gemini_model(
        self, mock_genai: MagicMock
    ) -> None:
        """Factory returns GeminiModel when provider is 'gemini' with API key.

        Validates: Requirements 3.4, 5.2
        """
        result = create_judge_model()

        assert isinstance(result, GeminiModel)
        assert result.get_model_name() == "gemini-2.0-flash"

    @patch("evaluation.llm_metrics.GEMINI_API_KEY", None)
    @patch("evaluation.llm_metrics.DEEPEVAL_JUDGE_PROVIDER", "gemini")
    def test_gemini_provider_without_api_key_raises(self) -> None:
        """Factory raises ValueError when provider is 'gemini' but API key is missing.

        Validates: Requirements 3.6
        """
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            create_judge_model()

    @patch("evaluation.llm_metrics.GEMINI_API_KEY", "")
    @patch("evaluation.llm_metrics.DEEPEVAL_JUDGE_PROVIDER", "gemini")
    def test_gemini_provider_with_empty_api_key_raises(self) -> None:
        """Factory raises ValueError when provider is 'gemini' and API key is empty string.

        Validates: Requirements 3.6
        """
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            create_judge_model()

    @patch("evaluation.llm_metrics.DEEPEVAL_JUDGE_PROVIDER", "openai")
    def test_unknown_provider_raises(self) -> None:
        """Factory raises ValueError for an unrecognized provider.

        Validates: Requirements 3.3
        """
        with pytest.raises(ValueError, match="não reconhecido"):
            create_judge_model()


# ============================================================
# Property-based test for create_judge_model() factory
# Feature: llm-eval-improvements, Property 5: Factory retorna tipo correto de modelo baseado no provider
# **Validates: Requirements 3.3, 3.4, 5.2**
# ============================================================


class TestFactoryReturnsCorrectModelType:
    """Property 5: Factory retorna tipo correto de modelo baseado no provider.

    For any valid DEEPEVAL_JUDGE_PROVIDER value, create_judge_model() must
    return a GPTModel when provider is "local" and a GeminiModel (inheriting
    from DeepEvalBaseLLM) when provider is "gemini" with a valid API key.
    """

    @given(provider=st.sampled_from(["local", "gemini"]))
    @settings(max_examples=100)
    @patch("evaluation.llm_metrics.GEMINI_API_KEY", "test-api-key")
    @patch("evaluation.llm_metrics.GEMINI_MODEL", "gemini-2.0-flash")
    def test_factory_returns_correct_type_for_provider(
        self, provider: str
    ) -> None:
        """create_judge_model() returns the correct model type for each provider."""
        with patch("evaluation.llm_metrics.DEEPEVAL_JUDGE_PROVIDER", provider):
            if provider == "local":
                mock_gpt = MagicMock(spec=GPTModel)
                with patch(
                    "evaluation.llm_metrics.create_deepeval_model",
                    return_value=mock_gpt,
                ):
                    result = create_judge_model()
                    assert isinstance(result, GPTModel)
            elif provider == "gemini":
                with patch("evaluation.llm_metrics.genai"):
                    result = create_judge_model()
                    assert isinstance(result, GeminiModel)
                    assert isinstance(result, DeepEvalBaseLLM)


# ============================================================
# Property-based test for API key ausente with provider "gemini"
# Feature: llm-eval-improvements, Property 6: API key ausente com provider "gemini" lança ValueError
# **Validates: Requirements 3.6**
# ============================================================


class TestApiKeyAusenteRaisesValueError:
    """Property 6: API key ausente com provider "gemini" lança ValueError.

    For any call to create_judge_model() when DEEPEVAL_JUDGE_PROVIDER is
    "gemini" and GEMINI_API_KEY is None or empty string, the function must
    raise ValueError indicating the API key is required.
    """

    @given(api_key=st.sampled_from([None, ""]))
    @settings(max_examples=100)
    def test_missing_api_key_raises_value_error(self, api_key: str | None) -> None:
        """create_judge_model() raises ValueError when provider is 'gemini' and API key is absent."""
        with patch("evaluation.llm_metrics.DEEPEVAL_JUDGE_PROVIDER", "gemini"), \
             patch("evaluation.llm_metrics.GEMINI_API_KEY", api_key):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                create_judge_model()


# ============================================================
# Unit tests for faithfulness fallback
# Validates: Requirements 2.1, 2.2, 2.3, 2.4
# ============================================================

import math
import logging

from evaluation.llm_metrics import llm_judge_faithfulness


class TestFaithfulnessFallback:
    """Unit tests for the faithfulness fallback mechanism."""

    @patch("evaluation.llm_metrics.create_judge_model")
    @patch("evaluation.llm_metrics.FaithfulnessMetric")
    @patch("evaluation.llm_metrics._faithfulness_fallback", return_value=0.75)
    def test_primary_fails_fallback_returns_valid_score(
        self,
        mock_fallback: MagicMock,
        mock_faithfulness_cls: MagicMock,
        mock_create_model: MagicMock,
    ) -> None:
        """When FaithfulnessMetric raises, fallback returns a valid score.

        Validates: Requirements 2.1, 2.2
        """
        mock_faithfulness_cls.return_value.measure.side_effect = RuntimeError(
            "JSON parse error"
        )

        score = llm_judge_faithfulness("answer", ["context"], "question")

        assert score == 0.75
        mock_fallback.assert_called_once_with("answer", ["context"], "question")

    @patch("evaluation.llm_metrics.create_judge_model")
    @patch("evaluation.llm_metrics.FaithfulnessMetric")
    @patch(
        "evaluation.llm_metrics._faithfulness_fallback",
        return_value=float("nan"),
    )
    def test_both_fail_returns_nan(
        self,
        mock_fallback: MagicMock,
        mock_faithfulness_cls: MagicMock,
        mock_create_model: MagicMock,
    ) -> None:
        """When both FaithfulnessMetric and fallback fail, returns NaN.

        Validates: Requirements 2.3
        """
        mock_faithfulness_cls.return_value.measure.side_effect = RuntimeError(
            "JSON parse error"
        )

        score = llm_judge_faithfulness("answer", ["context"], "question")

        assert math.isnan(score)
        mock_fallback.assert_called_once()

    @patch("evaluation.llm_metrics.create_judge_model")
    @patch("evaluation.llm_metrics.FaithfulnessMetric")
    @patch("evaluation.llm_metrics._faithfulness_fallback", return_value=0.5)
    def test_logging_indicates_method_used(
        self,
        mock_fallback: MagicMock,
        mock_faithfulness_cls: MagicMock,
        mock_create_model: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Logs indicate which faithfulness method was used.

        When the primary metric fails and fallback succeeds, a warning
        should be logged for the native failure and an info message for
        the fallback success.

        Validates: Requirements 2.4
        """
        mock_faithfulness_cls.return_value.measure.side_effect = RuntimeError(
            "JSON parse error"
        )

        with caplog.at_level(logging.DEBUG, logger="evaluation.llm_metrics"):
            score = llm_judge_faithfulness("answer", ["context"], "question")

        assert score == 0.5

        # Check warning about native method failure
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("FaithfulnessMetric failed" in msg for msg in warning_msgs), (
            f"Expected warning about native failure, got: {warning_msgs}"
        )

        # Check info about fallback success
        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("fallback" in msg.lower() for msg in info_msgs), (
            f"Expected info about fallback method, got: {info_msgs}"
        )


# ============================================================
# Property-based test for fallback triggered when FaithfulnessMetric fails
# Feature: llm-eval-improvements, Property 3: Fallback é acionado quando FaithfulnessMetric falha
# **Validates: Requirements 2.1**
# ============================================================


class TestFallbackTriggeredWhenFaithfulnessFails:
    """Property 3: Fallback é acionado quando FaithfulnessMetric falha.

    For any combination of prediction, context, and question where
    FaithfulnessMetric raises an exception, the metrics module must
    execute the fallback and return the fallback's score.
    """

    @given(
        prediction=st.text(min_size=1, max_size=200),
        question=st.text(min_size=1, max_size=200),
        context=st.lists(st.text(min_size=1, max_size=200), min_size=1, max_size=3),
    )
    @settings(max_examples=100)
    def test_fallback_called_when_primary_raises(
        self,
        prediction: str,
        question: str,
        context: list[str],
    ) -> None:
        """When FaithfulnessMetric.measure raises, _faithfulness_fallback is called
        and its return value is used as the score."""
        with patch("evaluation.llm_metrics.create_judge_model") as mock_create_model, \
             patch("evaluation.llm_metrics.FaithfulnessMetric") as mock_faithfulness_cls, \
             patch("evaluation.llm_metrics._faithfulness_fallback", return_value=0.5) as mock_fallback:
            mock_faithfulness_cls.return_value.measure.side_effect = RuntimeError(
                "JSON parse error"
            )

            score = llm_judge_faithfulness(prediction, context, question)

            mock_fallback.assert_called_once_with(prediction, context, question)
            assert score == 0.5


# ============================================================
# Property-based test for faithfulness fallback score bounds
# Feature: llm-eval-improvements, Property 4: Score do fallback de faithfulness é limitado [0.0, 1.0] ou NaN
# **Validates: Requirements 2.2**
# ============================================================

from evaluation.llm_metrics import _faithfulness_fallback


class TestFaithfulnessFallbackScoreBounded:
    """Property 4: Score do fallback de faithfulness é limitado [0.0, 1.0] ou NaN.

    For any prediction, context, and question, the score returned by
    _faithfulness_fallback must be in [0.0, 1.0] or NaN (on error).
    """

    @given(
        prediction=st.text(min_size=1, max_size=200),
        question=st.text(min_size=1, max_size=200),
        context=st.lists(st.text(min_size=1, max_size=200), min_size=1, max_size=3),
        mock_score=st.one_of(
            st.floats(min_value=0.0, max_value=1.0),
            st.just(float("nan")),
        ),
    )
    @settings(max_examples=100)
    def test_fallback_score_bounded_or_nan(
        self,
        prediction: str,
        question: str,
        context: list[str],
        mock_score: float,
    ) -> None:
        """_faithfulness_fallback returns a score in [0.0, 1.0] or NaN."""
        with patch("evaluation.llm_metrics.create_judge_model") as mock_create_model, \
             patch("evaluation.llm_metrics.GEval") as mock_geval_cls:
            mock_metric = MagicMock()

            def set_score(test_case):
                mock_metric.score = mock_score

            mock_metric.measure.side_effect = set_score
            mock_geval_cls.return_value = mock_metric

            score = _faithfulness_fallback(prediction, context, question)

            assert (0.0 <= score <= 1.0) or math.isnan(score), (
                f"Score out of bounds: {score}, expected [0.0, 1.0] or NaN"
            )


# ============================================================
# Unit tests for llm_judge_relevancy()
# Validates: Requirements 1.1, 1.2
# ============================================================

from evaluation.llm_metrics import llm_judge_relevancy


class TestLlmJudgeRelevancy:
    """Unit tests for the llm_judge_relevancy function."""

    @patch("evaluation.llm_metrics.create_judge_model")
    @patch("evaluation.llm_metrics.AnswerRelevancyMetric")
    def test_relevancy_returns_known_score(
        self,
        mock_relevancy_cls: MagicMock,
        mock_create_model: MagicMock,
    ) -> None:
        """Returns the score from AnswerRelevancyMetric when successful.

        Validates: Requirements 1.1
        """
        mock_metric = MagicMock()
        mock_metric.score = 0.85
        mock_relevancy_cls.return_value = mock_metric

        score = llm_judge_relevancy("answer", ["context"], "question")

        assert score == 0.85
        mock_metric.measure.assert_called_once()

    @patch("evaluation.llm_metrics.create_judge_model")
    @patch("evaluation.llm_metrics.AnswerRelevancyMetric")
    def test_relevancy_error_returns_nan(
        self,
        mock_relevancy_cls: MagicMock,
        mock_create_model: MagicMock,
    ) -> None:
        """Returns NaN when AnswerRelevancyMetric raises an exception.

        Validates: Requirements 1.2
        """
        mock_relevancy_cls.return_value.measure.side_effect = RuntimeError(
            "Model unavailable"
        )

        score = llm_judge_relevancy("answer", ["context"], "question")

        assert math.isnan(score)


# ============================================================
# Property-based test for relevancy score bounds
# Feature: llm-eval-improvements, Property 1: Score de relevancy é limitado [0.0, 1.0] ou NaN
# **Validates: Requirements 1.1**
# ============================================================


class TestRelevancyScoreBounded:
    """Property 1: Score de relevancy é limitado [0.0, 1.0] ou NaN.

    For any prediction, context, and question, the score returned by
    llm_judge_relevancy must be in [0.0, 1.0] or NaN (on error).
    """

    @given(
        prediction=st.text(min_size=1, max_size=200),
        question=st.text(min_size=1, max_size=200),
        context=st.lists(st.text(min_size=1, max_size=200), min_size=1, max_size=3),
        mock_score=st.one_of(
            st.floats(min_value=0.0, max_value=1.0),
            st.just(float("nan")),
        ),
    )
    @settings(max_examples=100)
    def test_relevancy_score_bounded_or_nan(
        self,
        prediction: str,
        question: str,
        context: list[str],
        mock_score: float,
    ) -> None:
        """llm_judge_relevancy returns a score in [0.0, 1.0] or NaN."""
        with patch("evaluation.llm_metrics.create_judge_model"), \
             patch("evaluation.llm_metrics.AnswerRelevancyMetric") as mock_cls:
            mock_metric = MagicMock()

            def set_score(test_case):
                mock_metric.score = mock_score

            mock_metric.measure.side_effect = set_score
            mock_cls.return_value = mock_metric

            score = llm_judge_relevancy(prediction, context, question)

            assert (0.0 <= score <= 1.0) or math.isnan(score), (
                f"Score out of bounds: {score}, expected [0.0, 1.0] or NaN"
            )
