# code-rag

Evaluation framework for measuring code search performance using keyword, semantic, and hybrid retrieval strategies against the [CodeSearchNet](https://github.com/github/CodeSearchNet) benchmark.

## Overview

`code-rag` implements and evaluates three retrieval algorithms:

- **BM25** — PostgreSQL full-text search (`tsvector` + GIN index)
- **Embedding** — HNSW vector search via pgvector (BGE-Code-v1 + BGE-Reranker-v2-m3)
- **Hybrid** — Reciprocal Rank Fusion (RRF) combining BM25 and embedding rankings

Evaluation uses two CodeSearchNet query sets:

| Suite | Type | Queries | Relevance |
|-------|------|---------|-----------|
| `csn_expert` | Human-annotated | 99 | Graded 0–3 |
| `csn_docstring` | Docstring-derived | 1,000 | Binary |

## Architecture

```
src/
  coderag/
    interfaces.py          # SearchResult + SearchAlgorithm protocol
    algo/
      bm25.py              # PostgreSQL FTS (tsvector/ts_rank_cd)
      embedding.py         # pgvector HNSW + cross-encoder reranking
      hybrid.py            # RRF fusion of BM25 + embedding
  eval/
    suite.py               # EvalSuite (queries + qrels)
    runner.py              # run_eval() → ranx metrics
    cli.py                 # CLI entry point (coderag eval ...)
  csn/
    download.py            # Extract CSN corpus, download annotations
    expert.py              # Build expert evaluation suite
    docstring.py           # Build docstring evaluation suite
tests/                     # pytest test suite (86 tests)
```

## Prerequisites

- Python 3.12+
- PostgreSQL 16+ with the [pgvector](https://github.com/pgvector/pgvector) extension
- ~4 GB RAM for embedding model inference (no GPU required)

### PostgreSQL setup

The quickest way is Docker:

```bash
docker run -d --name coderag-postgres \
  -e POSTGRES_USER=coderag \
  -e POSTGRES_PASSWORD=coderag \
  -e POSTGRES_DB=coderag \
  -p 5432:5432 \
  pgvector/pgvector:pg17
```

## Installation

```bash
# Clone
git clone https://github.com/igorlima/code-rag.git
cd code-rag

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

Key variables:

| Variable | Description |
|----------|-------------|
| `CODERAG_PG_URL` | PostgreSQL connection URL (e.g. `postgresql://user:pass@localhost:5432/coderag`) |
| `CODERAG_DATASETS_DIR` | Directory containing raw CSN zip files |
| `CODERAG_DATA_DIR` | Output directory for eval suites and run results |

## Data setup

### 1. Download the CodeSearchNet corpus

Download `python.zip` from [CodeSearchNet releases](https://github.com/github/CodeSearchNet#data-download) and place it at:

```
$CODERAG_DATASETS_DIR/codesearchnet/python.zip
```

### 2. Extract corpus and download annotations

```bash
uv run coderag eval csn init
```

This extracts the corpus to `$CODERAG_DATASETS_DIR/codesearchnet/python.jsonl` and downloads the expert annotations.

### 3. Build evaluation suites

```bash
uv run coderag eval csn generate --docstring-sample-count 1000
```

Produces `$CODERAG_DATA_DIR/eval_suites/csn_expert` and `$CODERAG_DATA_DIR/eval_suites/csn_docstring`.

## Building indexes

### BM25

```bash
uv run coderag eval index \
  --algorithm bm25 \
  --corpus $CODERAG_DATASETS_DIR/codesearchnet/python.jsonl
```

### Embedding (HNSW)

```bash
uv run coderag eval index \
  --algorithm embedding \
  --corpus $CODERAG_DATASETS_DIR/codesearchnet/python.jsonl \
  --model-name BAAI/bge-code-v1 \
  --batch-size 32 \
  --max-length 4096
```

Both indexes are stored in PostgreSQL. BM25 uses `bm25_docs` (GIN index on `tsvector`); embedding uses `embedding_docs` (HNSW index via pgvector).

## Running evaluations

```bash
# BM25
uv run coderag eval run bm25 \
  --suite csn_expert --top-k 10 \
  --metrics ndcg@10,mrr@10 --workers 4

# Embedding
uv run coderag eval run embedding \
  --suite csn_expert --top-k 10 \
  --metrics ndcg@10,mrr@10 --workers 1 \
  --model-name BAAI/bge-code-v1 \
  --reranker-name BAAI/bge-reranker-v2-m3 \
  --rerank-candidates 50 --gap-threshold 0.1 \
  --reranker-max-length 512 --ef-search 200

# Hybrid (RRF)
uv run coderag eval run hybrid \
  --suite csn_expert --top-k 10 \
  --metrics ndcg@10,mrr@10 --workers 1 \
  --model-name BAAI/bge-code-v1 \
  --reranker-name BAAI/bge-reranker-v2-m3 \
  --rerank-candidates 50 --gap-threshold 0.1 \
  --reranker-max-length 512 --ef-search 200 \
  --rrf-k 60 --bm25-depth 100 --embedding-depth 50
```

### Compare runs

```bash
uv run coderag eval compare
```

## Benchmark results

Results on CodeSearchNet Python (457K functions), BAAI/bge-code-v1 embeddings, BAAI/bge-reranker-v2-m3 reranker, ef_search=200.

| Suite | Algorithm | NDCG@10 | MRR@10 |
|-------|-----------|---------|--------|
| csn_expert | bm25 | 0.1020 | 0.2318 |
| csn_expert | embedding | 0.0344 | 0.1035 |
| csn_expert | hybrid (rrf k=60) | 0.0919 | 0.2105 |
| csn_docstring | bm25 | 0.9256 | 0.9091 |
| csn_docstring | embedding | 0.5100 | 0.5036 |
| csn_docstring | hybrid (rrf k=60) | **0.9315** | **0.9150** |

## Running tests

```bash
uv run pytest tests/ -v
```

86 tests across unit, integration, and end-to-end levels. All tests mock PostgreSQL and model inference — no running database or GPU required.

## License

Apache 2.0 — see [LICENSE](LICENSE).
