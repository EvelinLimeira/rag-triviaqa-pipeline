"""Parse TriviaQA dataset files into LangChain Document objects.

Provides functions to load the TriviaQA JSONL dataset and the full
document pool JSON file, converting raw entries into structured
:class:`TriviaQAEntry` objects and LangChain ``Document`` instances.
"""

import json
import logging
from dataclasses import dataclass

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"question", "answer", "golden_docs", "noise_docs"}


@dataclass
class TriviaQAEntry:
    """A parsed TriviaQA dataset entry.

    Attributes:
        question: The trivia question string.
        answers: List of valid answer strings.
        golden_docs: LangChain Documents that contain the answer,
            each with ``metadata["doc_id"]``.
        noise_docs: LangChain distractor Documents,
            each with ``metadata["doc_id"]``.
    """

    question: str
    answers: list[str]
    golden_docs: list[Document]
    noise_docs: list[Document]


def _parse_docs(raw_docs: list[dict]) -> list[Document]:
    """Convert a list of raw doc dicts to LangChain Documents.

    Args:
        raw_docs: List of dicts each containing ``"doc_id"`` and
            ``"content"`` keys.

    Returns:
        List of :class:`Document` objects with ``page_content`` set to
        the content field and ``metadata["doc_id"]`` set to the doc id.
    """
    return [
        Document(page_content=d["content"], metadata={"doc_id": d["doc_id"]})
        for d in raw_docs
    ]


def load_triviaqa(filepath: str) -> list[TriviaQAEntry]:
    """Parse a TriviaQA JSONL file into a list of entries.

    Each line of the file is expected to be a JSON object with fields:
    ``question``, ``answer``, ``golden_docs``, and ``noise_docs``.
    Lines with invalid JSON or missing required fields are skipped with
    a logged warning that includes the 1-based line number.

    Args:
        filepath: Path to the ``triviaqa.jsonl`` file.

    Returns:
        List of :class:`TriviaQAEntry` objects parsed from valid lines.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
    """
    entries: list[TriviaQAEntry] = []

    with open(filepath, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            # Parse JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping line %d: invalid JSON", line_num
                )
                continue

            # Validate required fields
            missing = REQUIRED_FIELDS - data.keys()
            if missing:
                logger.warning(
                    "Skipping line %d: missing required fields %s",
                    line_num,
                    sorted(missing),
                )
                continue

            entries.append(
                TriviaQAEntry(
                    question=data["question"],
                    answers=data["answer"],
                    golden_docs=_parse_docs(data["golden_docs"]),
                    noise_docs=_parse_docs(data["noise_docs"]),
                )
            )

    return entries


def load_documents_pool(filepath: str) -> list[Document]:
    """Load the full document pool from a JSON array file.

    The file is expected to contain a JSON array of objects, each with
    ``"doc_id"`` and ``"content"`` fields.

    Args:
        filepath: Path to the ``documents_pool.json`` file.

    Returns:
        List of LangChain :class:`Document` objects with
        ``page_content`` and ``metadata["doc_id"]`` populated.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        raw_pool = json.load(fh)

    return _parse_docs(raw_pool)


def get_per_query_corpus(entry: TriviaQAEntry) -> list[Document]:
    """Return the union of golden and noise documents for a query.

    Combines ``entry.golden_docs`` and ``entry.noise_docs`` into a
    single list, removing duplicates by ``metadata["doc_id"]``.
    Documents from ``golden_docs`` take precedence when duplicates
    exist.

    Args:
        entry: A :class:`TriviaQAEntry` whose documents to merge.

    Returns:
        Deduplicated list of LangChain :class:`Document` objects.
    """
    seen: set[str] = set()
    corpus: list[Document] = []

    for doc in entry.golden_docs + entry.noise_docs:
        doc_id = doc.metadata["doc_id"]
        if doc_id not in seen:
            seen.add(doc_id)
            corpus.append(doc)

    return corpus
