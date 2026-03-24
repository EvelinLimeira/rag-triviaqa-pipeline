"""Centralized configuration for the RAG TriviaQA pipeline."""

import logging
import os

from dotenv import load_dotenv
import torch

load_dotenv()

logger = logging.getLogger(__name__)

# Device auto-detection
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    _gpu_name = torch.cuda.get_device_name(0)
    _gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info("Using GPU: %s (%.1f GB VRAM)", _gpu_name, _gpu_mem)
else:
    logger.warning(
        "CUDA not available — running on CPU. "
        "Performance will be significantly slower. "
        "Connect your NVIDIA GPU for faster execution."
    )

# Models
LLM_MODEL: str = "qwen3.5:9b"
LLM_BASE_URL: str = "http://localhost:11434/v1"
EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval
RETRIEVAL_TOP_K: int = 500
RERANKER_TOP_K: int = 5
RRF_K: int = 60
BM25_K1: float = 1.5
BM25_B: float = 0.75

# Encoding
EMBEDDING_BATCH_SIZE: int = 256
RERANKER_BATCH_SIZE: int = 128

# Chunking (full-pool only)
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 64

# LLM
LLM_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 256
LLM_NUM_CTX: int = 8192

# Evaluation
EVAL_SAMPLE_SIZE: int = 584
HIT_RATE_K_VALUES: list[int] = [1, 3, 5, 10]

# deepeval
DEEPEVAL_MODEL: str = "qwen3.5:9b"
DEEPEVAL_BASE_URL: str = "http://localhost:11434/v1"

# Judge model (external API support)
DEEPEVAL_JUDGE_PROVIDER: str = os.getenv("DEEPEVAL_JUDGE_PROVIDER", "local")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY", None)
