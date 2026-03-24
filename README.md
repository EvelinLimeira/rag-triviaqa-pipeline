# RAG TriviaQA Pipeline

RAG pipeline for TriviaQA with BM25, Dense, and Hybrid retrieval, cross-encoder reranking, and evaluation via deepeval with configurable LLM-judge (Gemini or local model).

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│  Data Load  │──> │   Indexing   │───>│  Retrieval  │───>│ Generation │
│  (HF Hub)   │    │ BM25 + FAISS │    │ + Reranking │    │  (Ollama)  │
└─────────────┘    └──────────────┘    └─────────────┘    └─────┬──────┘
                                                                │
                                                          ┌─────▼──────┐
                                                          │ Evaluation │
                                                          │ IR + LLM   │
                                                          │ (Gemini /  │
                                                          │  Local)    │
                                                          └────────────┘
```

## Features

- **Data**: TriviaQA dataset from [AQ-MedAI/RAG-QA-Leaderboard](https://huggingface.co/datasets/AQ-MedAI/RAG-QA-Leaderboard)
- **Indexing**: BM25 (sparse) + FAISS (dense) with document chunking
- **Retrieval**: 4 configurations — BM25, Dense, Hybrid (RRF fusion), Hybrid+Rerank
- **Reranking**: Cross-encoder (ms-marco-MiniLM-L-6-v2)
- **Generation**: Local LLM via Ollama (Qwen3.5-9B) through LangChain
- **Evaluation**:
  - LLM-judge metrics: Correctness (GEval), Faithfulness, and Answer Relevancy via **deepeval**
  - Configurable judge model: Google Gemini (API) or local Ollama model
  - Automatic fallback: if FaithfulnessMetric fails, falls back to GEval-based claim decomposition
  - IR metrics: Hit Rate@k (1, 3, 5, 10) and MRR
  - Deterministic: Exact Match and token-level F1

## Tech Stack

| Component | Tool |
|---|---|
| Framework | LangChain |
| Sparse Index | BM25Retriever |
| Dense Index | FAISS + HuggingFace Embeddings (bge-base-en-v1.5) |
| Hybrid Fusion | EnsembleRetriever (RRF) |
| Reranker | sentence-transformers CrossEncoder |
| LLM (generation) | Ollama (Qwen3.5-9B) |
| LLM-judge | Google Gemini 2.5 Flash (default) or local Ollama model |
| LLM Evaluation | deepeval (GEval, FaithfulnessMetric, AnswerRelevancyMetric) |
| Testing | pytest + Hypothesis (property-based) |

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- GPU recommended for faster execution (runs on CPU too)
- Google Gemini API key (optional, for external LLM-judge)

### Installation

```bash
pip install -r requirements.txt
ollama pull qwen3.5:9b
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Required for dataset download
HF_TOKEN=your_huggingface_token

# LLM-judge configuration (optional — defaults to local Ollama model)
DEEPEVAL_JUDGE_PROVIDER=gemini          # "gemini" or "local"
GEMINI_API_KEY=your_gemini_api_key      # Required when provider is "gemini"
GEMINI_MODEL=gemini-2.5-flash           # Gemini model to use as judge

# Timeout for DeepEval metric calls (seconds)
DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=300
```

When `DEEPEVAL_JUDGE_PROVIDER` is set to `"local"` (or omitted), the pipeline uses the local Ollama model for evaluation. Set it to `"gemini"` to use Google Gemini as the LLM-judge, which is faster and more reliable for structured evaluation.

## Usage

### 1. Download dataset

```bash
python main.py download
```

### 2. Build indices (full-pool mode)

```bash
python main.py index
```

### 3. Run evaluation

Per-query mode (builds indices per question, ~50 docs each):
```bash
python main.py evaluate --sample-size 30
```

Full-pool mode (uses pre-built indices, ~1M+ docs):
```bash
python main.py evaluate-full-pool --sample-size 30
```

## Evaluation Metrics

| Category | Metric | Description |
|---|---|---|
| IR | Hit Rate@k | Whether a golden doc appears in top-k results (k=1,3,5,10) |
| IR | MRR | Mean Reciprocal Rank of the first golden doc |
| Deterministic | Exact Match (EM) | Normalized exact string match |
| Deterministic | F1 | Token-level F1 between prediction and reference |
| LLM-judge | Correctness | GEval-based factual correctness vs. reference answers |
| LLM-judge | Faithfulness | Whether the answer is grounded in the retrieval context |
| LLM-judge | Relevancy | Whether the answer is relevant to the original question |

## Output

Results are saved to `results/`:
- `summary.md` — Comparison table across all retriever configurations
- `results_<config>.json` — Per-query detailed results with all metrics

Example output:

| Retriever | Hit@1 | Hit@5 | Hit@10 | MRR | EM | F1 | Correct | Faithful | Relevancy |
|---|---|---|---|---|---|---|---|---|---|
| BM25 | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Dense | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Hybrid | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Hybrid+Rerank | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Project Structure

```
├── config/             # Centralized settings (model, retrieval, eval params)
├── data/               # Dataset download, loading, and indices
├── indexing/            # BM25 and FAISS index construction + chunking
├── retrieval/           # Retriever wrappers (BM25, Dense, Hybrid) + reranker
├── generation/          # LLM prompt template and generator
├── evaluation/          # IR metrics, LLM-judge metrics (deepeval), orchestrator
├── tests/               # 151 tests (unit + property-based with Hypothesis)
├── results/             # Evaluation output (summary + per-config JSONs)
├── main.py              # CLI entry point
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables (API keys, judge config)
```

## Tests

```bash
pytest tests/ -v
```

The test suite includes:
- Unit tests for all modules (loader, chunker, retrievers, generator, metrics, orchestrator)
- Property-based tests with Hypothesis (round-trip parsing, score bounds, fallback behavior)
- All LLM-judge tests use mocks (no real API calls required)

## License

MIT
