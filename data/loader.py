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

REQUIRED_FIELDS = {"query", "golden_doc", "reference", "ground_truth"}


@dataclass
class TriviaQAEntry:
    """A parsed TriviaQA dataset entry.

    Attributes:
        question: The trivia question string.
        answers: List of valid answer strings.
        golden_doc_ids: List of document IDs that contain the answer.
        noise_doc_ids: List of distractor document IDs.
        golden_docs: LangChain Documents that contain the answer
            (populated after resolving against the document pool).
        noise_docs: LangChain distractor Documents
            (populated after resolving against the document pool).
    """

    question: str
    answers: list[str]
    golden_doc_ids: list[str]
    noise_doc_ids: list[str]
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


def load_triviaqa(
    filepath: str,
    doc_pool: dict[str, str] | None = None,
) -> list[TriviaQAEntry]:
    """Parse a TriviaQA JSONL file into a list of entries.

    Each line is a JSON object with fields: ``query``, ``golden_doc``
    (list of doc IDs), ``reference`` (list of noise doc IDs), and
    ``ground_truth`` (list of answer strings).

    If *doc_pool* is provided (mapping doc_id → content), the entry's
    ``golden_docs`` and ``noise_docs`` are populated with LangChain
    Documents. Otherwise they remain empty lists.

    Args:
        filepath: Path to the ``triviaqa.jsonl`` file.
        doc_pool: Optional mapping from document ID to content string.

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

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping line %d: invalid JSON", line_num)
                continue

            missing = REQUIRED_FIELDS - data.keys()
            if missing:
                logger.warning(
                    "Skipping line %d: missing required fields %s",
                    line_num,
                    sorted(missing),
                )
                continue

            golden_doc_ids = data["golden_doc"]
            noise_doc_ids = data["reference"]

            golden_docs: list[Document] = []
            noise_docs: list[Document] = []

            if doc_pool is not None:
                for doc_id in golden_doc_ids:
                    if doc_id in doc_pool:
                        golden_docs.append(
                            Document(
                                page_content=doc_pool[doc_id],
                                metadata={"doc_id": doc_id},
                            )
                        )
                for doc_id in noise_doc_ids:
                    if doc_id in doc_pool:
                        noise_docs.append(
                            Document(
                                page_content=doc_pool[doc_id],
                                metadata={"doc_id": doc_id},
                            )
                        )

            entries.append(
                TriviaQAEntry(
                    question=data["query"],
                    answers=data["ground_truth"],
                    golden_doc_ids=golden_doc_ids,
                    noise_doc_ids=noise_doc_ids,
                    golden_docs=golden_docs,
                    noise_docs=noise_docs,
                )
            )

    return entries


def load_documents_pool(filepath: str) -> list[Document]:
    """Load the full document pool from a JSON file.

    The file is expected to contain a JSON object mapping document IDs
    to their content strings, e.g. ``{"Document_1": "content...", ...}``.

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

    if isinstance(raw_pool, dict):
        return [
            Document(page_content=content, metadata={"doc_id": doc_id})
            for doc_id, content in raw_pool.items()
        ]

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
