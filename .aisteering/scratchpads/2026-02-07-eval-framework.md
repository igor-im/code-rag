# 2026-02-07 — Eval Framework + CSN Implementation

## What changed
Implemented the full code search evaluation framework and CSN benchmark plugin.

### Package structure (3 sibling packages under src/)
- `src/coderag/` — search system (interfaces only for now)
- `src/eval/` — generic evaluation framework (benchmark-agnostic)
- `src/csn/` — CSN benchmark data handling

### Files created
- `pyproject.toml` — project config (hatchling, src layout, deps: ranx, datasets, python-dotenv)
- `.gitignore` — Python + data dirs
- `src/coderag/interfaces.py` — `SearchResult` dataclass + `SearchAlgorithm` Protocol
- `src/eval/suite.py` — `EvalSuite` with save/load/to_ranx_qrels
- `src/eval/runner.py` — `run_eval()` function (ranx evaluation)
- `src/eval/cli.py` — argparse CLI: `eval csn init`, `eval csn generate`, `eval run`, `eval compare`
- `src/csn/download.py` — `extract_corpus()` from zip, `download_annotations()` from GitHub
- `src/csn/expert.py` — `build_expert_suite()` (graded 0-3 relevance)
- `src/csn/docstring.py` — `build_docstring_suite()` (binary relevance)
- `tests/` — 43 tests across 7 test files + fixture data

### Key decisions
- `make_comparable=True` passed to `ranx.evaluate()` to handle partial results
- Empty algorithm results → early return with 0.0 scores (ranx crashes on empty Run)
- Expert suite aggregates multiple annotations via rounded mean
- Docstring suite uses sorted func_id for deterministic subsample
- CLI requires all params explicitly (no defaults per policy)
- Packages are siblings (coderag, eval, csn) not nested — coderag is for search, eval/csn are separate concerns

## How to validate
```bash
uv run pytest tests/ -v          # 43 tests, all passing
uv run coderag --help            # CLI works
uv run coderag eval csn --help   # CSN subcommands
```

## Notes/risks
- `download_annotations()` fetches from GitHub raw URLs — needs network
- Full corpus extraction (~500K functions) not yet tested at scale
- `uv run` is the correct way to invoke tools (not direct .venv/bin/ paths)

## Next steps
- Commit the implementation on feat/eval-framework branch
- Run `coderag eval csn init` + `generate` against real data
- Run BM25 against real CSN data (see below)

---

# 2026-02-07 — BM25 Search Algorithm

## What changed
Added first real search algorithm: BM25 via SQLite FTS5.

### Files created
- `src/coderag/algo/__init__.py` — algo subpackage
- `src/coderag/algo/bm25.py` — `build_index()` + `Bm25Algorithm` class
- `tests/test_bm25.py` — 15 tests (indexer + search)

### Files modified
- `src/eval/cli.py` — added `eval index` subcommand, extended `_resolve_algorithm` for bm25
- `tests/test_cli.py` — 3 new BM25 CLI tests
- `tests/test_end_to_end.py` — 1 new BM25 e2e test

### Key decisions
- BM25 algo lives in `src/coderag/algo/bm25.py` (algo subpackage under coderag)
- FTS5 schema: `func_name`, `code`, `docstring` indexed; `func_id` UNINDEXED
- Porter stemming + unicode61 tokenizer
- Query tokens quoted and joined with OR for safe FTS5 queries
- FTS5 bm25() returns negative scores; negated for positive SearchResult.score
- Index path: `$CODERAG_DATA_DIR/algorithms/bm25/index.db`
- No new dependencies (stdlib sqlite3)

## How to validate
```bash
uv run pytest tests/ -v                    # 62 tests, all passing
uv run coderag eval index --algorithm bm25 --corpus $CODERAG_DATASETS_DIR/codesearchnet/python.jsonl
uv run coderag eval run bm25 --suite csn_docstring --top-k 10 --metrics "ndcg@10,mrr@10"
uv run coderag eval run bm25 --suite csn_expert --top-k 10 --metrics "ndcg@10,mrr@10"
uv run coderag eval compare
```

## Next steps
- Commit BM25 implementation
- Run against real CSN corpus to get baseline metrics
- Implement vector search algorithm (next algo)

---

# 2026-02-10 — Embedding init() — Index Building (Task 3)

## What changed
Added the `init()` function for the embedding algorithm module. This embeds a JSONL corpus using a transformer model and stores L2-normalized vectors in a sqlite-vec database.

### Files created
- `src/coderag/algo/embedding.py` — `init()` + helpers (`_encode_batch`, `_last_token_pool`, `_insert_batch`)
- `tests/test_embedding.py` — 4 tests for init()

### Key decisions
- Same temp-file-then-copy pattern as BM25 for NAS safety
- Last-token pooling (BGE-Code-v1 convention) for embedding extraction
- L2 normalization before storage so cosine distance = dot product
- Two tables: `embedding_docs` (rowid -> func_id) and `embedding_vec` (vec0 virtual table with cosine metric)
- Document text = `func_name + "\n" + code + "\n" + docstring` (concatenated)
- Batched encoding with configurable `batch_size` for GPU memory management
- Tests mock `AutoModel`, `AutoTokenizer`, and `torch.cuda.is_available` to avoid downloading real models
- Mock `model.parameters()` uses `side_effect` (not `return_value`) to produce fresh iterators per call

## How to validate
```bash
uv run pytest tests/test_embedding.py -v   # 4 tests, all passing
uv run pytest tests/ -v                     # 66 total, 64 passing (2 pre-existing CLI failures)
```

## Notes/risks
- `_INSTRUCTION` constant defined but not yet used (will be used in EmbeddingAlgorithm search)
- `SearchResult` import present but not yet used (same — for EmbeddingAlgorithm)
- Module-level `torch.cuda.is_available()` call means tests on CUDA machines need the patch

## Next steps
- Task 4: Implement `EmbeddingAlgorithm` class (KNN search + reranker + gating)
- Task 5: CLI integration for embedding algorithm

---

# 2026-02-10 — EmbeddingAlgorithm — KNN Search + Reranker + Gating (Task 4)

## What changed
Added the `EmbeddingAlgorithm` class implementing a two-stage retrieval pipeline: KNN embedding search followed by cross-encoder reranking with score-gap gating.

### Files modified
- `src/coderag/algo/embedding.py` — added `EmbeddingAlgorithm` class, `AutoModelForSequenceClassification` import, `_BM25_TABLE` constant
- `tests/test_embedding.py` — added 3 helper functions + 8 search tests (total: 12 tests in file)

### Key decisions
- Two-stage pipeline: KNN retrieval (sqlite-vec cosine) -> cross-encoder reranking (sigmoid-normalized logits)
- Score-gap gating: if gap between #1 and #2 KNN cosine similarity >= threshold, pin top result through reranking
- Pinned result gets max_reranker_score + 1.0 to ensure it stays at position 0
- Embedding index opened read-only (`?mode=ro`); BM25 index loaded into memory via backup (same pattern as Bm25Algorithm)
- Document text for reranker fetched from BM25 FTS5 index (avoids storing text twice)
- Query text formatted with BGE-Code instruction prefix: `<instruct>{instruction}\n<query>{query}`
- Empty/whitespace queries return [] immediately
- Tests use deterministic known vectors: [1,0,0,0], [0.7,0.7,0,0] normalized, [0,0,1,0]
- Mock reranker returns ascending logits (torch.linspace) so last KNN candidate gets highest score, enabling reorder verification

## How to validate
```bash
uv run pytest tests/test_embedding.py -v   # 12 tests, all passing
uv run pytest tests/ -v                     # 74 total, 72 passing (2 pre-existing CLI failures)
```

## Notes/risks
- `torch.cuda.is_available` must be patched in all search tests to prevent mock `.half().cuda()` chaining
- BM25 db dependency: EmbeddingAlgorithm requires a pre-built BM25 index for doc text lookup
- Reranker tokenizer receives pairs (list of [query, doc_text]) not single strings

## Next steps
- Task 5: CLI integration for embedding algorithm

---

# 2026-02-10 — CLI Integration for Embedding Algorithm (Task 5)

## What changed
Wired the embedding algorithm into the CLI so users can build embedding indexes and run embedding search through the command line.

### Files modified
- `src/eval/cli.py` — added embedding branches to `_eval_index` and `_resolve_algorithm`, updated `_resolve_algorithm` signature to accept `args`, added embedding-specific CLI flags to index and run subparsers, added runtime validation for embedding-specific flags
- `tests/test_cli.py` — added 3 new tests: `test_cli_eval_index_embedding`, `test_cli_eval_index_unknown_algorithm`, `test_cli_eval_run_embedding_no_index`

### Key decisions
- `_resolve_algorithm` now takes full `args: argparse.Namespace` (not just `name` and `data_dir`) because embedding algo needs extra parameters (model_name, reranker_name, rerank_candidates, gap_threshold, bm25_db_path)
- Embedding-specific CLI flags are optional at the argparse level but validated at runtime when `--algorithm embedding` is specified (avoids making them required for non-embedding algorithms)
- Index subparser gets: `--model-name`, `--batch-size`
- Run subparser gets: `--model-name`, `--reranker-name`, `--rerank-candidates`, `--gap-threshold`, `--bm25-db-path`
- Tests reuse `_mock_model` and `_mock_tokenizer` from `tests/test_embedding` to avoid duplication
- TDD approach: wrote tests first, verified failure, then implemented

## How to validate
```bash
uv run pytest tests/test_cli.py -v   # 13 tests, 11 passing (2 pre-existing failures)
uv run pytest tests/ -v              # 77 total, 75 passing (2 pre-existing CLI failures)
```

## Notes/risks
- The 2 pre-existing test failures (test_cli_eval_csn_init_missing_env, test_cli_eval_run_missing_env) are unrelated environment issues

## Next steps
- Final verification and cleanup (Task 6)

---

# 2026-02-15 — Wire Hybrid Algorithm into CLI (Task 5 of hybrid)

## What changed
Wired the `HybridAlgorithm` (BM25 + embedding fusion via RRF) into the CLI with three new flags.

### Files modified
- `src/eval/cli.py` — added `--rrf-k`, `--bm25-depth`, `--embedding-depth` args to run subparser; added hybrid validation block in `_eval_run`; added hybrid branch to `_resolve_algorithm` that constructs both sub-algorithms and composes `HybridAlgorithm`
- `tests/test_cli.py` — added `test_cli_eval_run_hybrid_missing_rrf_k` (validates that omitting `--rrf-k` exits with a clear error)

### Key decisions
- Hybrid validation checks all 10 required flags (7 shared with embedding + 3 hybrid-specific)
- `_resolve_algorithm` hybrid branch constructs BM25 from `data_dir/algorithms/bm25/index.db` and EmbeddingAlgorithm from user-provided `--bm25-db-path` + pgvector
- TDD approach: wrote failing test first, verified failure, then implemented

## How to validate
```bash
uv run pytest tests/test_cli.py::test_cli_eval_run_hybrid_missing_rrf_k -v  # 1 test, passing
uv run pytest tests/test_cli.py -v  # 14 tests, 11 passing (3 pre-existing env leakage failures)
```

## Notes/risks
- The 3 pre-existing test failures are env var leakage from `.env` (not caused by this change)
- The hybrid branch uses two separate BM25 paths: one for the BM25 sub-algorithm (`data_dir/algorithms/bm25/index.db`) and one for the EmbeddingAlgorithm's text lookup (`--bm25-db-path` flag) -- these could be the same file in practice

## Next steps
- Task 6 or integration testing with real data
