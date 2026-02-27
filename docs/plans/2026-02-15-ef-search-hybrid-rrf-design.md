# 2026-02-15 — ef_search tuning + Hybrid RRF retrieval

## Context

Embedding algorithm (BGE-Code-v1 + pgvector HNSW) underperforms BM25 by 5.5x on csn_expert and 2x on csn_docstring. Root cause analysis identified KNN recall failure as the bottleneck — the reranker works well (92.2% rank-1 accuracy when the right doc is in the candidate pool), but pgvector's default `ef_search=40` limits HNSW graph exploration.

Additionally, BM25 and embedding retrieve from almost disjoint document sets (3.1% overlap on expert, 11.9% on docstring), making them strong candidates for fusion via Reciprocal Rank Fusion (RRF).

## Part 1: `ef_search` parameter

### Changes

- **`EmbeddingAlgorithm.__init__`**: new required param `ef_search: int`. Set once on the pgvector connection via `SET hnsw.ef_search = {value}` (session-level setting, no per-query overhead).
- **CLI**: new `--ef-search` arg on the run subparser, required for `embedding` and `hybrid` algorithms.
- **Tests**: update all `EmbeddingAlgorithm` constructor calls to include `ef_search`.

### Rationale

With 457K vectors, `ef_search=40` (pgvector default) explores too few HNSW graph nodes, severely limiting recall. Values of 100-400 are recommended for corpora of this size. Making it a required CLI param (no default) aligns with project policy.

## Part 2: Hybrid RRF retrieval

### Architecture

Two components:

1. **`rrf_combine()`** — reusable generic function
   - Signature: `rrf_combine(runs: list[list[SearchResult]], k: int, top_n: int) -> list[SearchResult]`
   - RRF score formula: `score(d) = sum(1/(k + rank_i(d)))` across all runs
   - Works with any number of ranked lists
   - Returns top_n results sorted by RRF score descending

2. **`HybridAlgorithm`** — composes BM25 + Embedding
   - Implements `SearchAlgorithm` protocol
   - Constructor takes both sub-algorithms + RRF params
   - `.search()` calls each sub-algorithm with its own depth, merges via `rrf_combine`

### Constructor

```python
HybridAlgorithm(
    bm25: Bm25Algorithm,
    embedding: EmbeddingAlgorithm,
    rrf_k: int,           # RRF constant (typically 60)
    bm25_depth: int,      # results to fetch from BM25
    embedding_depth: int,  # results to fetch from embedding
)
```

### CLI integration

New args on the run subparser (all required for hybrid):
- `--rrf-k` (int) — RRF constant
- `--bm25-depth` (int) — BM25 retrieval depth
- `--embedding-depth` (int) — embedding retrieval depth

The hybrid algorithm also requires all embedding flags (`--model-name`, `--reranker-name`, etc.) plus `--ef-search`.

`_resolve_algorithm("hybrid", ...)` constructs BM25Algorithm, EmbeddingAlgorithm, then HybridAlgorithm composing both.

### Files

| File | Change |
|------|--------|
| `src/coderag/algo/embedding.py` | Add `ef_search` param to `__init__`, set on pg connection |
| `src/coderag/algo/hybrid.py` | **New** — `rrf_combine()` + `HybridAlgorithm` |
| `src/eval/cli.py` | Add `--ef-search`, `--rrf-k`, `--bm25-depth`, `--embedding-depth`; hybrid resolution branch |
| `tests/test_embedding.py` | Add `ef_search` to all constructor calls |
| `tests/test_cli.py` | Update embedding tests with `ef_search`, add hybrid CLI tests |
| `tests/test_hybrid.py` | **New** — `rrf_combine` unit tests + `HybridAlgorithm` integration tests |

### Validation

- `uv run pytest tests/ -v` — all tests pass
- Run embedding eval with `--ef-search 200` and compare against baseline
- Run hybrid eval and compare against both individual algorithms
