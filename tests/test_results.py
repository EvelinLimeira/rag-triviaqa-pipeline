"""Unit tests for EvaluationOrchestrator.print_comparison_table and save_results."""

import json
import math
import os

import pytest

from evaluation.run_eval import (
    ALL_CONFIGS,
    EvaluationOrchestrator,
    EvaluationResults,
    QueryResult,
)


def _make_sample_results() -> dict[str, EvaluationResults]:
    """Create sample evaluation results for testing."""
    results: dict[str, EvaluationResults] = {}
    for i, config in enumerate(ALL_CONFIGS):
        qr = QueryResult(
            question=f"What is {i}?",
            reference_answers=[str(i)],
            golden_doc_ids={f"doc_{i}"},
            retrieved_ids_pre_rerank=[f"doc_{i}", f"doc_{i+10}"],
            retrieved_ids_post_rerank=[f"doc_{i}"],
            generated_answer=str(i),
            context_docs=[f"Content about {i}"],
            metrics={
                "hit_rate@1": 1.0,
                "hit_rate@5": 1.0,
                "hit_rate@10": 1.0,
                "mrr": 1.0,
                "em": 1.0,
                "f1": 0.85 + i * 0.01,
                "correctness": 0.9,
                "faithfulness": 0.8,
                "relevancy": 0.75,
            },
        )
        results[config] = EvaluationResults(
            config_name=config,
            per_query_results=[qr],
            aggregate_metrics=qr.metrics.copy(),
        )
    return results


class TestPrintComparisonTable:
    """Tests for EvaluationOrchestrator.print_comparison_table."""

    def test_prints_all_required_columns(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify the table header contains all required columns."""
        orch = EvaluationOrchestrator(sample_size=10)
        results = _make_sample_results()
        orch.print_comparison_table(results)
        output = capsys.readouterr().out

        for col in ["Retriever", "Hit@1", "Hit@5", "Hit@10", "MRR", "EM", "F1", "Correct", "Faithful", "Relevancy"]:
            assert col in output

    def test_prints_one_row_per_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify there is one data row per configuration."""
        orch = EvaluationOrchestrator(sample_size=10)
        results = _make_sample_results()
        orch.print_comparison_table(results)
        output = capsys.readouterr().out

        for config in ALL_CONFIGS:
            assert config in output

    def test_formats_numbers_to_3_decimal_places(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify numeric values are formatted to 3 decimal places."""
        orch = EvaluationOrchestrator(sample_size=10)
        results = _make_sample_results()
        orch.print_comparison_table(results)
        output = capsys.readouterr().out

        assert "1.000" in output
        assert "0.850" in output

    def test_handles_nan_values(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify NaN metric values are displayed as N/A."""
        orch = EvaluationOrchestrator(sample_size=10)
        results = _make_sample_results()
        results["BM25"].aggregate_metrics["correctness"] = float("nan")
        orch.print_comparison_table(results)
        output = capsys.readouterr().out

        assert "N/A" in output

    def test_empty_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify table prints header even with no results."""
        orch = EvaluationOrchestrator(sample_size=10)
        orch.print_comparison_table({})
        output = capsys.readouterr().out

        assert "Retriever" in output
        assert "Hit@1" in output


class TestSaveResults:
    """Tests for EvaluationOrchestrator.save_results."""

    def test_creates_output_directory(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify output_dir is created if it doesn't exist."""
        orch = EvaluationOrchestrator(sample_size=10)
        results = _make_sample_results()
        out_dir = str(tmp_path / "new_dir")
        orch.save_results(results, out_dir)

        assert os.path.isdir(out_dir)

    def test_saves_json_per_config(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify one JSON file is saved per configuration."""
        orch = EvaluationOrchestrator(sample_size=10)
        results = _make_sample_results()
        out_dir = str(tmp_path / "output")
        orch.save_results(results, out_dir)

        json_files = [f for f in os.listdir(out_dir) if f.endswith(".json")]
        assert len(json_files) == len(ALL_CONFIGS)

    def test_json_format_matches_spec(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify JSON structure has config, queries, and aggregate keys."""
        orch = EvaluationOrchestrator(sample_size=10)
        results = _make_sample_results()
        out_dir = str(tmp_path / "output")
        orch.save_results(results, out_dir)

        json_path = os.path.join(out_dir, "results_bm25.json")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "config" in data
        assert "queries" in data
        assert "aggregate" in data
        assert data["config"] == "BM25"
        assert len(data["queries"]) == 1
        assert "question" in data["queries"][0]
        assert "generated_answer" in data["queries"][0]
        assert "reference_answers" in data["queries"][0]
        assert "retrieved_doc_ids" in data["queries"][0]
        assert "metrics" in data["queries"][0]

    def test_saves_summary_markdown(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify summary.md is created with the comparison table."""
        orch = EvaluationOrchestrator(sample_size=10)
        results = _make_sample_results()
        out_dir = str(tmp_path / "output")
        orch.save_results(results, out_dir)

        md_path = os.path.join(out_dir, "summary.md")
        assert os.path.isfile(md_path)

        with open(md_path, encoding="utf-8") as f:
            content = f.read()

        for col in ["Retriever", "Hit@1", "Hit@5", "Hit@10", "MRR", "EM", "F1", "Correct", "Faithful", "Relevancy"]:
            assert col in content

        for config in ALL_CONFIGS:
            assert config in content

    def test_hybrid_rerank_json_filename(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify Hybrid+Rerank config produces a safe filename."""
        orch = EvaluationOrchestrator(sample_size=10)
        results = _make_sample_results()
        out_dir = str(tmp_path / "output")
        orch.save_results(results, out_dir)

        expected_file = os.path.join(out_dir, "results_hybrid_plus_rerank.json")
        assert os.path.isfile(expected_file)


# ---------------------------------------------------------------------------
# Property-Based Tests
# ---------------------------------------------------------------------------

from hypothesis import HealthCheck, given, settings, strategies as st

# Feature: rag-trivia-pipeline, Property 21: Markdown Table Contains Required Columns
# **Validates: Requirements 20.1**

# Strategy: generate random metric values (floats 0–1) for each required metric key
_METRIC_KEYS = [
    "hit_rate@1",
    "hit_rate@5",
    "hit_rate@10",
    "mrr",
    "em",
    "f1",
    "correctness",
    "faithfulness",
    "relevancy",
]

_metric_dict_strategy = st.fixed_dictionaries(
    {k: st.floats(min_value=0.0, max_value=1.0) for k in _METRIC_KEYS}
)

# Pick 1–4 config names from ALL_CONFIGS (at least one so the table has data rows)
_configs_strategy = st.lists(
    st.sampled_from(ALL_CONFIGS), min_size=1, max_size=len(ALL_CONFIGS), unique=True
)


@given(
    configs=_configs_strategy,
    metrics_per_config=st.lists(
        _metric_dict_strategy, min_size=len(ALL_CONFIGS), max_size=len(ALL_CONFIGS)
    ),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property21_markdown_table_contains_required_columns(
    capsys: pytest.CaptureFixture[str],
    configs: list[str],
    metrics_per_config: list[dict[str, float]],
) -> None:
    """Property 21: The markdown comparison table always contains all required column headers.

    For any set of evaluation results built from random metric values and
    random subsets of retriever configurations, the printed table must
    include every required column header: Retriever, Hit@1, Hit@5, Hit@10,
    MRR, EM, F1, Correct, Faithful.
    """
    # Build EvaluationResults for the selected configs
    results: dict[str, EvaluationResults] = {}
    for idx, config_name in enumerate(configs):
        metrics = metrics_per_config[idx]
        qr = QueryResult(
            question="q",
            reference_answers=["a"],
            golden_doc_ids={"d"},
            retrieved_ids_pre_rerank=["d"],
            retrieved_ids_post_rerank=["d"],
            generated_answer="a",
            context_docs=["ctx"],
            metrics=metrics.copy(),
        )
        results[config_name] = EvaluationResults(
            config_name=config_name,
            per_query_results=[qr],
            aggregate_metrics=metrics.copy(),
        )

    orch = EvaluationOrchestrator(sample_size=1)
    orch.print_comparison_table(results)
    output = capsys.readouterr().out

    required_columns = [
        "Retriever",
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "MRR",
        "EM",
        "F1",
        "Correct",
        "Faithful",
        "Relevancy",
    ]
    for col in required_columns:
        assert col in output, f"Required column '{col}' missing from table output"


# ============================================================
# Property-based test for relevancy present in results
# Feature: llm-eval-improvements, Property 2: Métrica relevancy presente nos resultados
# **Validates: Requirements 1.3**
# ============================================================

from unittest.mock import patch, MagicMock

from evaluation.run_eval import _compute_query_metrics


class TestRelevancyPresentInResults:
    """Property 2: Métrica relevancy presente nos resultados.

    For any query result computed by the orchestrator, the metrics
    dictionary must contain the key "relevancy" with a float value.
    """

    @given(
        prediction=st.text(min_size=1, max_size=100),
        question=st.text(min_size=1, max_size=100),
        context=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=3),
        references=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=3),
    )
    @settings(max_examples=100)
    def test_relevancy_key_present_in_metrics(
        self,
        prediction: str,
        question: str,
        context: list[str],
        references: list[str],
    ) -> None:
        """_compute_query_metrics always includes 'relevancy' in the result dict."""
        with patch("evaluation.run_eval.llm_judge_correctness", return_value=0.5), \
             patch("evaluation.run_eval.llm_judge_faithfulness", return_value=0.5), \
             patch("evaluation.run_eval.llm_judge_relevancy", return_value=0.5):
            metrics = _compute_query_metrics(
                retrieved_ids=["doc1"],
                golden_ids={"doc1"},
                generated_answer=prediction,
                reference_answers=references,
                context_docs=context,
                question=question,
            )

        assert "relevancy" in metrics, (
            f"'relevancy' key missing from metrics: {list(metrics.keys())}"
        )
        assert isinstance(metrics["relevancy"], float)


# ============================================================
# Unit tests for orchestrator integration
# Validates: Requirements 1.3, 1.4, 4.1, 4.3
# ============================================================


class TestOrchestratorRelevancyIntegration:
    """Unit tests for relevancy integration in the orchestrator."""

    def test_compute_query_metrics_includes_relevancy(self) -> None:
        """_compute_query_metrics includes 'relevancy' in the metrics dict.

        Validates: Requirements 1.3, 4.1
        """
        with patch("evaluation.run_eval.llm_judge_correctness", return_value=0.9), \
             patch("evaluation.run_eval.llm_judge_faithfulness", return_value=0.8), \
             patch("evaluation.run_eval.llm_judge_relevancy", return_value=0.75):
            metrics = _compute_query_metrics(
                retrieved_ids=["doc1"],
                golden_ids={"doc1"},
                generated_answer="Paris",
                reference_answers=["Paris"],
                context_docs=["Paris is the capital of France"],
                question="What is the capital of France?",
            )

        assert "relevancy" in metrics
        assert metrics["relevancy"] == 0.75

    def test_comparison_table_contains_relevancy_column(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """print_comparison_table includes a 'Relevancy' column.

        Validates: Requirements 1.4
        """
        orch = EvaluationOrchestrator(sample_size=10)
        results = _make_sample_results()
        orch.print_comparison_table(results)
        output = capsys.readouterr().out

        assert "Relevancy" in output

    def test_save_results_markdown_contains_relevancy(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """save_results markdown includes 'Relevancy' column.

        Validates: Requirements 1.4, 4.3
        """
        orch = EvaluationOrchestrator(sample_size=10)
        results = _make_sample_results()
        out_dir = str(tmp_path / "output")
        orch.save_results(results, out_dir)

        md_path = os.path.join(out_dir, "summary.md")
        with open(md_path, encoding="utf-8") as f:
            content = f.read()

        assert "Relevancy" in content
