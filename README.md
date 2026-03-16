# RAG TriviaQA Pipeline

RAG pipeline for TriviaQA with BM25, Dense, and Hybrid retrieval, cross-encoder reranking, and evaluation via deepeval.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│  Data Load   │───▶│   Indexing    │───▶│  Retrieval   │───▶│ Generation │
│  (HF Hub)    │    │ BM25 + FAISS │    │ + Reranking  │    │  (Ollama)  │
└─────────────┘    └──────────────┘    └─────────────┘    └─────┬──────┘
                                                                │
                                                          ┌─────▼──────┐
                                                          │ Evaluation  │
                                                          │ IR + LLM   │
                                                          └────────────┘
```

## Features

- **Data**: TriviaQA dataset from [AQ-MedAI/RAG-QA-Leaderboard](https://huggingface.co/datasets/AQ-MedAI/RAG-QA-Leaderboard)
- **Indexing**: BM25 (sparse) + FAISS (dense) with document chunking
- **Retrieval**: 4 configurations — BM25, Dense, Hybrid (RRF fusion), Hybrid+Rerank
- **Reranking**: Cross-encoder (ms-marco-MiniLM-L-6-v2)
- **Generation**: Local LLM via Ollama (Qwen3.5-9B) through LangChain
- **Evaluation**:
  - LLM metrics: Correctness (GEval) and Faithfulness via **deepeval**
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
| LLM | Ollama (Qwen3.5-9B) |
| LLM Evaluation | deepeval (GEval, FaithfulnessMetric) |
| Testing | pytest + hypothesis |

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- GPU with ~8GB VRAM (recommended)

### Installation

```bash
pip install -r requirements.txt
ollama pull qwen3.5:9b
```

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
python main.py evaluate --sample-size 100
```

Full-pool mode (uses pre-built indices, ~1M+ docs):
```bash
python main.py evaluate-full-pool --sample-size 100
```

## Output

Results are saved to `results/`:
- `summary.md` — Comparison table across all retriever configurations
- `results_<config>.json` — Per-query detailed results

Example output:

| Retriever | Hit@1 | Hit@5 | Hit@10 | MRR | EM | F1 | Correct | Faithful |
|---|---|---|---|---|---|---|---|---|
| BM25 | ... | ... | ... | ... | ... | ... | ... | ... |
| Dense | ... | ... | ... | ... | ... | ... | ... | ... |
| Hybrid | ... | ... | ... | ... | ... | ... | ... | ... |
| Hybrid+Rerank | ... | ... | ... | ... | ... | ... | ... | ... |

## Project Structure

```
├── config/             # Centralized settings
├── data/               # Dataset download and loading
├── indexing/            # BM25 and FAISS index construction
├── retrieval/          # Retriever wrappers + reranker
├── generation/         # LLM prompt template and generator
├── evaluation/         # IR metrics + LLM metrics (deepeval)
├── tests/              # Test suite
├── main.py             # CLI entry point
└── requirements.txt
```

## Tests

```bash
pytest tests/ -v
```

## License

MIT
