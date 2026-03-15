"""Property-based tests for data/loader.py (Properties 1–4).

Tests the TriviaQA data loading and parsing functions using Hypothesis
to verify correctness properties across randomized inputs.
"""

import json
import tempfile
import os

from hypothesis import strategies as st, given, settings
from langchain_core.documents import Document

from data.loader import TriviaQAEntry, load_triviaqa, load_documents_pool, get_per_query_corpus

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

lc_documents = st.builds(
    Document,
    page_content=st.text(min_size=1, max_size=500),
    metadata=st.fixed_dictionaries(
        {
            "doc_id": st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(
                    whitelist_categories=("L", "N"),
                    whitelist_characters="_-",
                ),
            )
        }
    ),
)

triviaqa_entries = st.builds(
    TriviaQAEntry,
    question=st.text(min_size=1, max_size=200),
    answers=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=5),
    golden_docs=st.lists(lc_documents, min_size=1, max_size=5),
    noise_docs=st.lists(lc_documents, min_size=0, max_size=10),
)


# ---------------------------------------------------------------------------
# Property 1: TriviaQA Entry Round-Trip
# ---------------------------------------------------------------------------
# Feature: rag-trivia-pipeline, Property 1: TriviaQA Entry Round-Trip
# Validates: Requirements 2.1, 2.2, 2.6


@given(entry=triviaqa_entries)
@settings(max_examples=100)
def test_triviaqa_entry_round_trip(entry: TriviaQAEntry) -> None:
    """Serialize a TriviaQAEntry to dict, write as JSONL, parse back,
    and verify equivalence of question, answers, and all Document
    page_content / metadata['doc_id']."""

    # Serialize entry to the JSONL dict format expected by load_triviaqa
    serialized = {
        "question": entry.question,
        "answer": entry.answers,
        "golden_docs": [
            {"doc_id": d.metadata["doc_id"], "content": d.page_content}
            for d in entry.golden_docs
        ],
        "noise_docs": [
            {"doc_id": d.metadata["doc_id"], "content": d.page_content}
            for d in entry.noise_docs
        ],
    }

    # Write to a temp JSONL file and parse back
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps(serialized) + "\n")
        tmp_path = f.name

    try:
        parsed = load_triviaqa(tmp_path)
        assert len(parsed) == 1, f"Expected 1 entry, got {len(parsed)}"
        result = parsed[0]

        # Verify question
        assert result.question == entry.question

        # Verify answers
        assert result.answers == entry.answers

        # Verify golden_docs
        assert len(result.golden_docs) == len(entry.golden_docs)
        for orig, loaded in zip(entry.golden_docs, result.golden_docs):
            assert loaded.page_content == orig.page_content
            assert loaded.metadata["doc_id"] == orig.metadata["doc_id"]

        # Verify noise_docs
        assert len(result.noise_docs) == len(entry.noise_docs)
        for orig, loaded in zip(entry.noise_docs, result.noise_docs):
            assert loaded.page_content == orig.page_content
            assert loaded.metadata["doc_id"] == orig.metadata["doc_id"]
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Property 2: Per-Query Corpus is Union of Golden and Noise Docs
# ---------------------------------------------------------------------------
# Feature: rag-trivia-pipeline, Property 2: Per-Query Corpus is Union of Golden and Noise Docs
# Validates: Requirements 2.3


@given(entry=triviaqa_entries)
@settings(max_examples=100)
def test_per_query_corpus_is_union(entry: TriviaQAEntry) -> None:
    """Verify get_per_query_corpus returns exactly golden_docs ∪ noise_docs
    with no duplicates by doc_id."""

    corpus = get_per_query_corpus(entry)

    # Collect expected unique doc_ids from golden + noise (golden first)
    expected_ids: set[str] = set()
    for doc in entry.golden_docs + entry.noise_docs:
        expected_ids.add(doc.metadata["doc_id"])

    # Corpus doc_ids should match the expected unique set
    corpus_ids = [d.metadata["doc_id"] for d in corpus]
    assert set(corpus_ids) == expected_ids

    # No duplicate doc_ids in corpus
    assert len(corpus_ids) == len(set(corpus_ids))

    # Every corpus doc should come from either golden_docs or noise_docs
    all_source_ids = {d.metadata["doc_id"] for d in entry.golden_docs + entry.noise_docs}
    for doc in corpus:
        assert doc.metadata["doc_id"] in all_source_ids


# ---------------------------------------------------------------------------
# Property 3: Invalid JSONL Entries Are Skipped
# ---------------------------------------------------------------------------
# Feature: rag-trivia-pipeline, Property 3: Invalid JSONL Entries Are Skipped
# Validates: Requirements 2.5

# Strategy: generate a subset of required fields to omit (at least one missing)
_required_fields = ["question", "answer", "golden_docs", "noise_docs"]

invalid_entries_strategy = st.fixed_dictionaries(
    {},
    optional={
        "question": st.text(min_size=1, max_size=100),
        "answer": st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=3),
        "golden_docs": st.just([{"doc_id": "g1", "content": "golden text"}]),
        "noise_docs": st.just([{"doc_id": "n1", "content": "noise text"}]),
    },
).filter(
    lambda d: not {"question", "answer", "golden_docs", "noise_docs"}.issubset(d.keys())
)


@given(invalid_entry=invalid_entries_strategy)
@settings(max_examples=100)
def test_invalid_jsonl_entries_are_skipped(invalid_entry: dict) -> None:
    """Generate JSONL entries missing one or more required fields and verify
    they are excluded from parsed output."""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps(invalid_entry) + "\n")
        tmp_path = f.name

    try:
        parsed = load_triviaqa(tmp_path)
        assert len(parsed) == 0, (
            f"Expected 0 entries for invalid input with keys {list(invalid_entry.keys())}, "
            f"got {len(parsed)}"
        )
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Property 4: Full-Pool Documents Are Valid LangChain Documents
# ---------------------------------------------------------------------------
# Feature: rag-trivia-pipeline, Property 4: Full-Pool Documents Are Valid LangChain Documents
# Validates: Requirements 2.4

pool_doc_dicts = st.lists(
    st.fixed_dictionaries(
        {
            "doc_id": st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(
                    whitelist_categories=("L", "N"),
                    whitelist_characters="_-",
                ),
            ),
            "content": st.text(min_size=1, max_size=500),
        }
    ),
    min_size=1,
    max_size=20,
)


@given(raw_docs=pool_doc_dicts)
@settings(max_examples=100)
def test_full_pool_documents_are_valid(raw_docs: list[dict]) -> None:
    """Verify loaded pool docs have correct page_content and
    metadata['doc_id'] matching the source data."""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(raw_docs, f)
        tmp_path = f.name

    try:
        documents = load_documents_pool(tmp_path)

        assert len(documents) == len(raw_docs)

        for raw, doc in zip(raw_docs, documents):
            assert isinstance(doc, Document)
            assert doc.page_content == raw["content"]
            assert doc.metadata["doc_id"] == raw["doc_id"]
    finally:
        os.unlink(tmp_path)


# ===========================================================================
# Unit tests for data/loader.py
# ===========================================================================
# Requirements: 2.1, 2.2, 2.3, 2.4, 2.5


class TestLoadTriviaqa:
    """Unit tests for load_triviaqa — parsing known TriviaQA entries."""

    def test_parse_single_valid_entry(self, tmp_path):
        """Parse a single well-formed JSONL line and verify all fields."""
        entry = {
            "question": "What is the capital of France?",
            "answer": ["Paris", "paris"],
            "golden_docs": [
                {"doc_id": "wiki_123", "content": "Paris is the capital of France."}
            ],
            "noise_docs": [
                {"doc_id": "wiki_456", "content": "Lyon is a city in France."}
            ],
        }
        filepath = tmp_path / "triviaqa.jsonl"
        filepath.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        result = load_triviaqa(str(filepath))

        assert len(result) == 1
        e = result[0]
        assert e.question == "What is the capital of France?"
        assert e.answers == ["Paris", "paris"]
        assert len(e.golden_docs) == 1
        assert e.golden_docs[0].page_content == "Paris is the capital of France."
        assert e.golden_docs[0].metadata["doc_id"] == "wiki_123"
        assert len(e.noise_docs) == 1
        assert e.noise_docs[0].metadata["doc_id"] == "wiki_456"

    def test_parse_multiple_entries(self, tmp_path):
        """Parse multiple valid JSONL lines."""
        entries = [
            {
                "question": "Q1",
                "answer": ["A1"],
                "golden_docs": [{"doc_id": "g1", "content": "gold1"}],
                "noise_docs": [],
            },
            {
                "question": "Q2",
                "answer": ["A2", "A2b"],
                "golden_docs": [{"doc_id": "g2", "content": "gold2"}],
                "noise_docs": [{"doc_id": "n2", "content": "noise2"}],
            },
        ]
        filepath = tmp_path / "triviaqa.jsonl"
        filepath.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )

        result = load_triviaqa(str(filepath))
        assert len(result) == 2
        assert result[0].question == "Q1"
        assert result[1].answers == ["A2", "A2b"]

    def test_multiple_golden_and_noise_docs(self, tmp_path):
        """Entry with several golden and noise docs parses all of them."""
        entry = {
            "question": "Q",
            "answer": ["A"],
            "golden_docs": [
                {"doc_id": "g1", "content": "c1"},
                {"doc_id": "g2", "content": "c2"},
            ],
            "noise_docs": [
                {"doc_id": "n1", "content": "nc1"},
                {"doc_id": "n2", "content": "nc2"},
                {"doc_id": "n3", "content": "nc3"},
            ],
        }
        filepath = tmp_path / "triviaqa.jsonl"
        filepath.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        result = load_triviaqa(str(filepath))
        assert len(result[0].golden_docs) == 2
        assert len(result[0].noise_docs) == 3


class TestSkipMalformedLines:
    """Unit tests for skip behavior on malformed JSONL lines."""

    def test_skip_invalid_json(self, tmp_path):
        """Lines with invalid JSON are skipped."""
        valid = {
            "question": "Q",
            "answer": ["A"],
            "golden_docs": [{"doc_id": "g1", "content": "c"}],
            "noise_docs": [],
        }
        filepath = tmp_path / "triviaqa.jsonl"
        filepath.write_text(
            "NOT VALID JSON\n" + json.dumps(valid) + "\n", encoding="utf-8"
        )

        result = load_triviaqa(str(filepath))
        assert len(result) == 1
        assert result[0].question == "Q"

    def test_skip_missing_question(self, tmp_path):
        """Entry missing 'question' field is skipped."""
        entry = {
            "answer": ["A"],
            "golden_docs": [{"doc_id": "g1", "content": "c"}],
            "noise_docs": [],
        }
        filepath = tmp_path / "triviaqa.jsonl"
        filepath.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        result = load_triviaqa(str(filepath))
        assert len(result) == 0

    def test_skip_missing_answer(self, tmp_path):
        """Entry missing 'answer' field is skipped."""
        entry = {
            "question": "Q",
            "golden_docs": [{"doc_id": "g1", "content": "c"}],
            "noise_docs": [],
        }
        filepath = tmp_path / "triviaqa.jsonl"
        filepath.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        result = load_triviaqa(str(filepath))
        assert len(result) == 0

    def test_skip_missing_golden_docs(self, tmp_path):
        """Entry missing 'golden_docs' field is skipped."""
        entry = {"question": "Q", "answer": ["A"], "noise_docs": []}
        filepath = tmp_path / "triviaqa.jsonl"
        filepath.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        result = load_triviaqa(str(filepath))
        assert len(result) == 0

    def test_skip_missing_noise_docs(self, tmp_path):
        """Entry missing 'noise_docs' field is skipped."""
        entry = {
            "question": "Q",
            "answer": ["A"],
            "golden_docs": [{"doc_id": "g1", "content": "c"}],
        }
        filepath = tmp_path / "triviaqa.jsonl"
        filepath.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        result = load_triviaqa(str(filepath))
        assert len(result) == 0

    def test_valid_entries_survive_among_invalid(self, tmp_path):
        """Valid entries are kept even when surrounded by invalid ones."""
        valid = {
            "question": "Q",
            "answer": ["A"],
            "golden_docs": [{"doc_id": "g1", "content": "c"}],
            "noise_docs": [],
        }
        invalid_json = "{broken"
        missing_field = json.dumps({"question": "Q"})

        filepath = tmp_path / "triviaqa.jsonl"
        filepath.write_text(
            "\n".join([invalid_json, json.dumps(valid), missing_field]) + "\n",
            encoding="utf-8",
        )

        result = load_triviaqa(str(filepath))
        assert len(result) == 1

    def test_empty_lines_are_skipped(self, tmp_path):
        """Blank lines in the JSONL file are silently skipped."""
        valid = {
            "question": "Q",
            "answer": ["A"],
            "golden_docs": [{"doc_id": "g1", "content": "c"}],
            "noise_docs": [],
        }
        filepath = tmp_path / "triviaqa.jsonl"
        filepath.write_text(
            "\n\n" + json.dumps(valid) + "\n\n", encoding="utf-8"
        )

        result = load_triviaqa(str(filepath))
        assert len(result) == 1


class TestLoadDocumentsPool:
    """Unit tests for load_documents_pool."""

    def test_load_sample_pool(self, tmp_path):
        """Load a small JSON array and verify Documents."""
        pool = [
            {"doc_id": "wiki_001", "content": "First document content."},
            {"doc_id": "wiki_002", "content": "Second document content."},
        ]
        filepath = tmp_path / "pool.json"
        filepath.write_text(json.dumps(pool), encoding="utf-8")

        docs = load_documents_pool(str(filepath))

        assert len(docs) == 2
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "First document content."
        assert docs[0].metadata["doc_id"] == "wiki_001"
        assert docs[1].page_content == "Second document content."
        assert docs[1].metadata["doc_id"] == "wiki_002"

    def test_load_empty_pool(self, tmp_path):
        """An empty JSON array produces an empty list."""
        filepath = tmp_path / "pool.json"
        filepath.write_text("[]", encoding="utf-8")

        docs = load_documents_pool(str(filepath))
        assert docs == []

    def test_load_single_document_pool(self, tmp_path):
        """A pool with a single document works correctly."""
        pool = [{"doc_id": "only_one", "content": "Solo doc."}]
        filepath = tmp_path / "pool.json"
        filepath.write_text(json.dumps(pool), encoding="utf-8")

        docs = load_documents_pool(str(filepath))
        assert len(docs) == 1
        assert docs[0].metadata["doc_id"] == "only_one"


class TestGetPerQueryCorpus:
    """Unit tests for get_per_query_corpus edge cases."""

    def test_empty_noise_docs(self):
        """Corpus with no noise docs returns only golden docs."""
        entry = TriviaQAEntry(
            question="Q",
            answers=["A"],
            golden_docs=[
                Document(page_content="gold1", metadata={"doc_id": "g1"}),
                Document(page_content="gold2", metadata={"doc_id": "g2"}),
            ],
            noise_docs=[],
        )

        corpus = get_per_query_corpus(entry)
        assert len(corpus) == 2
        ids = {d.metadata["doc_id"] for d in corpus}
        assert ids == {"g1", "g2"}

    def test_deduplication_golden_takes_precedence(self):
        """When golden and noise share a doc_id, golden version is kept."""
        entry = TriviaQAEntry(
            question="Q",
            answers=["A"],
            golden_docs=[
                Document(page_content="golden content", metadata={"doc_id": "dup"})
            ],
            noise_docs=[
                Document(page_content="noise content", metadata={"doc_id": "dup"}),
                Document(page_content="unique noise", metadata={"doc_id": "n1"}),
            ],
        )

        corpus = get_per_query_corpus(entry)
        assert len(corpus) == 2
        dup_doc = next(d for d in corpus if d.metadata["doc_id"] == "dup")
        assert dup_doc.page_content == "golden content"

    def test_all_unique_docs(self):
        """All docs are unique — corpus size equals golden + noise."""
        entry = TriviaQAEntry(
            question="Q",
            answers=["A"],
            golden_docs=[
                Document(page_content="g1", metadata={"doc_id": "g1"}),
            ],
            noise_docs=[
                Document(page_content="n1", metadata={"doc_id": "n1"}),
                Document(page_content="n2", metadata={"doc_id": "n2"}),
            ],
        )

        corpus = get_per_query_corpus(entry)
        assert len(corpus) == 3

    def test_single_golden_no_noise(self):
        """Minimal case: one golden doc, no noise."""
        entry = TriviaQAEntry(
            question="Q",
            answers=["A"],
            golden_docs=[
                Document(page_content="only", metadata={"doc_id": "g1"})
            ],
            noise_docs=[],
        )

        corpus = get_per_query_corpus(entry)
        assert len(corpus) == 1
        assert corpus[0].page_content == "only"
