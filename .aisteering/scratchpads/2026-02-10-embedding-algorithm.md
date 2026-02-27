# 2026-02-10 — Embedding Search Algorithm

## What changed

Added two-stage embedding search algorithm: BGE-Code-v1 (dense retrieval via sqlite-vec) + BGE-Reranker-v2-m3 (cross-encoder reranking) with score-gap gating.

### Files created
- `src/coderag/algo/embedding.py` — `init()` + `EmbeddingAlgorithm` class
- `tests/test_embedding.py` — 12 tests (4 init + 8 search)
- `docs/plans/2026-02-10-embedding-algorithm-design.md` — design doc
- `docs/plans/2026-02-10-embedding-algorithm-plan.md` — implementation plan

### Files modified
- `src/coderag/algo/bm25.py` — renamed `build_index` → `init` (standardized algo API)
- `src/eval/cli.py` — added embedding branches to index/run/resolve, new CLI flags
- `tests/test_bm25.py` — updated imports for renamed `init`
- `tests/test_end_to_end.py` — updated imports for renamed `init`
- `tests/test_cli.py` — 3 new embedding CLI tests
- `pyproject.toml` — added transformers, torch, sqlite-vec deps

### Key decisions
- sqlite-vec for vector storage (consistent with BM25's SQLite pattern)
- Two-table schema: `embedding_docs` (rowid → func_id) + `embedding_vec` (vec0 virtual)
- Document text for reranking read from BM25 FTS5 index (not duplicated)
- Query instruction: "Given a textual explanation of code functionality, retrieve the corresponding code implementation."
- Score-gap gating: if cosine similarity gap between #1 and #2 >= threshold, pin #1
- Embedding index opened read-only, BM25 db loaded into memory
- `_resolve_algorithm` now takes full `args` namespace (embedding needs extra params)

### Architecture
```
Query → <instruct>...\n<query>... → BGE-Code-v1 embed
  → sqlite-vec KNN (rerank_candidates neighbors)
  → score-gap check (pin if gap >= threshold)
  → BGE-Reranker-v2-m3 (cross-encoder rescore)
  → gating (pin original #1 if flagged)
  → top_k results
```

## How to validate
```bash
uv run pytest tests/ -v                     # 75 pass, 2 pre-existing fail
uv run coderag eval index --help            # shows embedding flags
uv run coderag eval run --help              # shows reranker flags

# Build embedding index (requires real model + GPU):
uv run coderag eval index --algorithm embedding \
    --corpus $CODERAG_DATASETS_DIR/codesearchnet/python.jsonl \
    --model-name BAAI/bge-code-v1 --batch-size 32

# Run evaluation:
uv run coderag eval run embedding --suite csn_expert --top-k 10 \
    --metrics "ndcg@10,mrr@10" --workers 1 \
    --model-name BAAI/bge-code-v1 \
    --reranker-name BAAI/bge-reranker-v2-m3 \
    --rerank-candidates 50 --gap-threshold 0.1 \
    --bm25-db-path $CODERAG_DATA_DIR/algorithms/bm25/index.db
```

## Notes/risks
- Embedding index at 500K docs × 1536-dim × 4 bytes ≈ 3GB
- Reranker limited to 512 tokens — long functions will be truncated
- BM25 index must be built before embedding search (text lookup dependency)
- 2 pre-existing test failures in test_cli.py (env var leakage from .env, not related)
- Real model testing not done yet — manual validation needed against CSN data

## Next steps
- Run against real CSN corpus to get embedding baseline metrics
- Compare embedding vs BM25 via `coderag eval compare`
- Tune gap_threshold based on real results
- Consider hybrid BM25 + embedding algorithm
