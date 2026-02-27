# 2026-02-12 — Migrate embedding KNN from sqlite-vec to pgvector

## What changed
- **`pyproject.toml`**: Removed `sqlite-vec>=0.1.6` dependency (psycopg[binary] and pgvector already present)
- **`src/coderag/algo/embedding.py`**: Full rewrite of storage backend
  - Replaced sqlite-vec with pgvector (psycopg + pgvector.psycopg)
  - New `_get_pg_conn(pg_url)` helper for connection + vector type registration
  - `init()`: second arg changed from `db_path: Path` to `pg_url: str`; uses DROP/CREATE TABLE + executemany + HNSW index creation
  - `EmbeddingAlgorithm.__init__()`: first arg changed from `db_path: Path` to `pg_url: str`; added `reranker_max_length: int` param; validates table via information_schema
  - `_knn_search()`: pgvector `<=>` cosine distance operator instead of sqlite-vec MATCH
  - `_rerank()`: uses `self._reranker_max_length` instead of hardcoded 4096
- **`src/eval/cli.py`**:
  - `_eval_index` embedding branch: uses `CODERAG_PG_URL` env var
  - `_resolve_algorithm` embedding branch: uses `CODERAG_PG_URL`, passes `reranker_max_length`
  - Added `--reranker-max-length` to run subparser
  - Added `reranker_max_length` to required flags validation
- **`tests/test_embedding.py`**: Replaced sqlite-vec with psycopg mocks
  - New `_mock_pg_conn()` helper that simulates KNN via numpy cosine similarity
  - `_build_test_dbs()` returns `(mock_pg_conn, bm25_db_path)` instead of file paths
  - All `EmbeddingAlgorithm()` calls use `pg_url="postgresql://test"` + `reranker_max_length=512`
  - `test_init_missing_db` → `test_init_missing_table` (checks info_schema response)
- **`tests/test_cli.py`**:
  - `test_cli_eval_index_embedding`: mocks `_get_pg_conn`, sets `CODERAG_PG_URL`
  - `test_cli_eval_run_embedding_no_index` → `test_cli_eval_run_embedding_missing_pg_url`

## Why
- sqlite-vec brute-force KNN over 457K × 1536-dim vectors takes ~7 min/query
- pgvector HNSW indexing reduces to milliseconds
- Reranker `max_length=4096` caused huge padded tensors; now configurable

## How to validate
- `uv run pytest tests/ -v` — 75 pass, 2 pre-existing env leakage failures
- Full re-index: `coderag eval index --algorithm embedding --corpus <path> --model-name BAAI/bge-code-v1 --batch-size 32 --max-length 4096`
- Requires `CODERAG_PG_URL` env var (Docker: `coderag-postgres`, pgvector/pgvector:pg17)

## Bug fix (2026-02-15)
- `_knn_search()` passed `query_vec.tolist()` (Python list) to pgvector `<=>` operator
- pgvector adapter registered by `register_vector()` only handles numpy arrays, not lists
- Error: `operator does not exist: vector <=> double precision[]`
- Fix: pass `query_vec` (numpy array) directly instead of `.tolist()`

## Eval results (2026-02-15)

| Suite | Algorithm | ndcg@10 | mrr@10 |
|-------|-----------|---------|--------|
| csn_expert | bm25 | 0.1711 | 0.4073 |
| csn_expert | embedding | 0.0309 | 0.0892 |
| csn_docstring | bm25 | 0.9056 | 0.8854 |
| csn_docstring | embedding | 0.4481 | 0.4428 |

- Embedding underperforms BM25 on both suites — needs investigation
- pgvector KNN queries now take <1s (was ~7 min with sqlite-vec brute-force)
- 1000-query csn_docstring eval completed in minutes (previously infeasible)

## Notes/risks
- pgvector extension must be installed in target PostgreSQL (`CREATE EXTENSION vector`)
- HNSW index build may take a few minutes on 457K rows (3.5 GB index at 457K rows)
- No migration path from old sqlite-vec indexes — must re-index

## ef_search + Hybrid RRF (2026-02-15)

Root cause analysis identified KNN recall failure as bottleneck — pgvector default `ef_search=40` too low for 457K vectors.

### Changes
- **`EmbeddingAlgorithm`**: new required `ef_search: int` param, sets `SET hnsw.ef_search` on pgvector connection
- **`src/coderag/algo/hybrid.py`** (new): `rrf_combine()` generic RRF function + `HybridAlgorithm` composing BM25 + embedding
- **CLI**: `--ef-search`, `--rrf-k`, `--bm25-depth`, `--embedding-depth` flags; hybrid algorithm resolution
- **Tests**: 10 new hybrid tests, updated all embedding constructor calls with `ef_search`

### How to validate
- `uv run pytest tests/ -v` — 85 pass, 3 pre-existing env leakage failures
- Embedding eval: `coderag eval run embedding --suite csn_expert --top-k 10 --metrics ndcg@10,mrr@10 --workers 1 --model-name BAAI/bge-code-v1 --reranker-name BAAI/bge-reranker-v2-m3 --rerank-candidates 50 --gap-threshold 0.1 --reranker-max-length 512 --ef-search 200`
- Hybrid eval: `coderag eval run hybrid --suite csn_expert --top-k 10 --metrics ndcg@10,mrr@10 --workers 1 --model-name BAAI/bge-code-v1 --reranker-name BAAI/bge-reranker-v2-m3 --rerank-candidates 50 --gap-threshold 0.1 --reranker-max-length 512 --ef-search 200 --rrf-k 60 --bm25-depth 100 --embedding-depth 50`

## Eval results (2026-02-16) — ef_search=200 + hybrid RRF

### Full comparison

| Suite | Algorithm | ndcg@10 | mrr@10 | Notes |
|-------|-----------|---------|--------|-------|
| csn_expert | bm25 | 0.1711 | 0.4073 | baseline |
| csn_expert | embedding (ef=40) | 0.0309 | 0.0892 | 2026-02-15 baseline |
| csn_expert | embedding (ef=200) | 0.0344 | 0.1035 | +11% ndcg, +16% mrr vs ef=40 |
| csn_expert | hybrid (rrf k=60) | 0.1241 | 0.2828 | bm25_depth=100, emb_depth=50 |
| csn_docstring | bm25 | 0.9056 | 0.8854 | baseline |
| csn_docstring | embedding (ef=40) | 0.4481 | 0.4428 | 2026-02-15 baseline |
| csn_docstring | embedding (ef=200) | 0.5100 | 0.5036 | +14% ndcg, +14% mrr vs ef=40 |
| csn_docstring | hybrid (rrf k=60) | 0.7781 | 0.7225 | bm25_depth=100, emb_depth=50 |

### Analysis

**ef_search=200 vs ef_search=40 (embedding only)**:
- Consistent ~11-16% improvement on both suites
- Still far below BM25 on both suites — embedding retrieval quality is the bottleneck, not HNSW recall

**Hybrid RRF**:
- csn_expert: hybrid (0.1241) < bm25 (0.1711) — embedding drags down fusion
- csn_docstring: hybrid (0.7781) < bm25 (0.9056) — same pattern, embedding hurts more than it helps
- On both suites, hybrid sits between embedding-only and BM25-only
- The weak embedding signal adds noise to the strong BM25 signal via RRF

**Key insight**: The embedding model (BAAI/bge-code-v1) performs poorly on code search tasks relative to BM25. This isn't a recall or infrastructure problem (ef_search=200 helps modestly) — the embedding representations themselves don't capture code search relevance well. The reranker (bge-reranker-v2-m3) + gap-based filtering may also be discarding good candidates.

### Possible next directions
- Try different embedding models (e.g., CodeSage, StarEncoder, or code-specific models)
- Ablation: disable reranker to isolate KNN vs reranker contribution
- Ablation: increase rerank-candidates (currently 50) to give reranker more to work with
- Weighted RRF: give BM25 higher weight in fusion (currently equal)
- Tune gap-threshold (currently 0.1) — may be too aggressive

## Next steps
- Investigate why embedding retrieval quality is poor (model choice? encoding strategy?)
- Consider ablation experiments (no reranker, higher rerank-candidates)
- Evaluate alternative embedding models for code search

---

## BM25 → PostgreSQL migration (2026-02-27)

Migrated BM25 from SQLite FTS5 to PostgreSQL full-text search. BM25 and embedding now share the same postgres instance.

### What changed
- **`src/coderag/algo/bm25.py`**: Full rewrite — SQLite FTS5 (`bm25()`) → PostgreSQL `tsvector`/`ts_rank_cd`/GIN index
  - New `bm25_docs` table: `func_id, func_name, code, docstring, tsv (generated tsvector)`
  - Search uses `plainto_tsquery` + `ts_rank_cd` (handles special chars safely, no FTS5 syntax errors)
  - `search_batch()` uses `ThreadPoolExecutor` with per-worker pg connections
- **`src/coderag/algo/embedding.py`**: Removed SQLite dependency entirely
  - `_fetch_doc_texts()` now queries `bm25_docs` via the existing pg connection (no separate sqlite3 conn)
- **`src/eval/cli.py`**: Removed `--bm25-db-path` flag; BM25 uses `CODERAG_PG_URL`
- **Tests**: `test_bm25.py` fully rewritten for pg mocks; `_FakePgBm25` functional mock added to `test_end_to_end.py`

### Why
- Eliminates the `data/` filesystem dependency (NAS symlink gone → SQLite index inaccessible)
- Single infrastructure concern: all persistent state in PostgreSQL

### How to validate
```bash
uv run pytest tests/ -v   # 86 tests, all passing
uv run coderag eval index --algorithm bm25 --corpus <python.jsonl>
uv run coderag eval run bm25 --suite csn_expert --top-k 10 --metrics ndcg@10,mrr@10 --workers 4
```

### Eval results (2026-02-27) — postgres FTS vs previous SQLite FTS5

| Suite | Algorithm | ndcg@10 | mrr@10 | vs SQLite FTS5 |
|-------|-----------|---------|--------|----------------|
| csn_expert | bm25 | 0.1020 | 0.2318 | ↓ (ts_rank_cd ≠ bm25()) |
| csn_expert | embedding (ef=200) | 0.0344 | 0.1035 | = |
| csn_expert | hybrid (rrf k=60) | 0.0919 | 0.2105 | ↓ (weaker BM25 component) |
| csn_docstring | bm25 | 0.9256 | 0.9091 | ↑ |
| csn_docstring | embedding (ef=200) | 0.5100 | 0.5036 | = |
| csn_docstring | hybrid (rrf k=60) | 0.9315 | 0.9150 | ↑↑ (+20% ndcg) |

**Key insight**: `ts_rank_cd` performs differently from FTS5 `bm25()` — weaker on expert (graded queries) but stronger on docstring (keyword queries). Hybrid gains significantly on docstring because both BM25 and embedding are strong there.

### Notes/risks
- `ts_rank_cd` is NOT the same as BM25 — it's a normalized TF-IDF variant. The "bm25" algorithm name now refers to the general concept rather than the specific scoring function.
- csn_expert BM25 regression (0.1711 → 0.1020) may be recoverable by switching to `ts_rank` (unnormalized) or tuning the text combination strategy
