"""Download TriviaQA dataset files from HuggingFace."""

import logging
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

logger = logging.getLogger(__name__)

REPO_ID = "AQ-MedAI/RAG-QA-Leaderboard"
REPO_TYPE = "dataset"
FILES = [
    "final_data/triviaqa.jsonl",
    "final_data/documents_pool.json",
]


def download_dataset(data_dir: str = "data/raw") -> None:
    """Download TriviaQA dataset files from HuggingFace Hub.

    Downloads ``triviaqa.jsonl`` and ``documents_pool.json`` from the
    ``AQ-MedAI/RAG-QA-Leaderboard`` dataset repository into *data_dir*.
    Files that already exist on disk are skipped.

    Args:
        data_dir: Local directory where the downloaded files are stored.
            Defaults to ``"data/raw"``.

    Raises:
        SystemExit: If a network error prevents downloading.
    """
    os.makedirs(data_dir, exist_ok=True)

    for filename in FILES:
        # hf_hub_download with local_dir preserves the repo path structure
        local_path = os.path.join(data_dir, filename)

        if os.path.exists(local_path):
            logger.info("File already exists, skipping download: %s", local_path)
            continue

        try:
            logger.info("Downloading %s from %s ...", filename, REPO_ID)
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type=REPO_TYPE,
                local_dir=data_dir,
                token=os.environ.get("HF_TOKEN"),
            )
            logger.info("Downloaded %s to %s", filename, data_dir)
        except Exception as exc:
            logger.error(
                "Failed to download %s from %s: %s", filename, REPO_ID, exc
            )
            sys.exit(1)
