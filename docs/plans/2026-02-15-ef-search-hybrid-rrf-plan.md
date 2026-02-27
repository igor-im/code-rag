# ef_search Tuning + Hybrid RRF Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve embedding recall via configurable `ef_search` and add hybrid BM25+embedding retrieval using Reciprocal Rank Fusion.

**Architecture:** Two changes: (1) thread a new `ef_search` parameter from CLI through `EmbeddingAlgorithm` to pgvector, (2) create a reusable `rrf_combine()` function and a `HybridAlgorithm` that composes BM25 + embedding sub-algorithms via RRF.

**Tech Stack:** pgvector (HNSW ef_search), Python (SearchAlgorithm protocol), argparse CLI

**Design doc:** `docs/plans/2026-02-15-ef-search-hybrid-rrf-design.md`

---

## Task 1: Add `ef_search` to `EmbeddingAlgorithm`

**Files:**
- Modify: `src/coderag/algo/embedding.py:207-275` (EmbeddingAlgorithm.__init__)
- Test: `tests/test_embedding.py`

**Step 1: Update all existing test constructor calls to include `ef_search=40`**

In `tests/test_embedding.py`, every `EmbeddingAlgorithm(...)` constructor call needs `ef_search=40` added. There are 8 calls at approximately lines 349, 393, 437, 481, 524, 566, 606, 622. Add `ef_search=40,` after `reranker_max_length=512,` in each.

Also update the `_mock_pg_conn` execute side effect (line 112) to handle `SET hnsw.ef_search` SQL — it should be a no-op (just don't crash on it). Add this before the KNN check:

```python
        if sql_str.startswith("SET"):
            return  # session-level settings are no-ops in mock
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding.py -v`
Expected: FAIL — `EmbeddingAlgorithm.__init__()` got unexpected keyword argument `ef_search`

**Step 3: Add `ef_search` parameter to `EmbeddingAlgorithm.__init__`**

In `src/coderag/algo/embedding.py`, modify the `__init__` method:

1. Add `ef_search: int,` parameter after `reranker_max_length: int,` (line 216)
2. After the pgvector connection is established and table validated (after line 275), add:

```python
        # Set HNSW search breadth for the session
        cur.execute(f"SET hnsw.ef_search = {ef_search}")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding.py -v`
Expected: All 12 tests PASS

**Step 5: Commit**

```bash
git add src/coderag/algo/embedding.py tests/test_embedding.py
git commit -m "feat: add ef_search parameter to EmbeddingAlgorithm"
```

---

## Task 2: Add `--ef-search` to CLI

**Files:**
- Modify: `src/eval/cli.py:140-144,186-201,280-285`
- Test: `tests/test_cli.py`

**Step 1: Update existing CLI tests for ef_search**

In `tests/test_cli.py`:

1. In `test_cli_eval_run_embedding_missing_pg_url` (line 225), add `"--ef-search", "40",` to the CLI args list.
2. In `test_cli_eval_index_embedding` — no change needed (ef_search is a run-time param, not index-time).

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_cli_eval_run_embedding_missing_pg_url -v`
Expected: FAIL — `error: unrecognized arguments: --ef-search`

**Step 3: Add `--ef-search` CLI arg and wire it through**

In `src/eval/cli.py`:

1. Add to run subparser args (after line 285):
```python
    run_parser.add_argument("--ef-search", type=int, help="HNSW ef_search parameter (embedding algo)")
```

2. Add `"ef_search"` to the required flags list for embedding in `_eval_run` (line 141):
```python
        for flag in ("model_name", "reranker_name", "rerank_candidates", "gap_threshold", "bm25_db_path", "reranker_max_length", "ef_search"):
```

3. Pass `ef_search` in `_resolve_algorithm` embedding branch (line 193-201):
```python
        return EmbeddingAlgorithm(
            pg_url,
            model_name=args.model_name,
            reranker_name=args.reranker_name,
            rerank_candidates=args.rerank_candidates,
            gap_threshold=args.gap_threshold,
            bm25_db_path=bm25_db_path,
            reranker_max_length=args.reranker_max_length,
            ef_search=args.ef_search,
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All CLI tests PASS

**Step 5: Commit**

```bash
git add src/eval/cli.py tests/test_cli.py
git commit -m "feat: add --ef-search CLI flag for embedding algorithm"
```

---

## Task 3: Implement `rrf_combine()`

**Files:**
- Create: `src/coderag/algo/hybrid.py`
- Create: `tests/test_hybrid.py`

**Step 1: Write failing tests for `rrf_combine`**

Create `tests/test_hybrid.py`:

```python
"""Tests for coderag.algo.hybrid — RRF combiner and hybrid search."""

from __future__ import annotations

from coderag.algo.hybrid import rrf_combine
from coderag.interfaces import SearchResult


def test_rrf_combine_single_run() -> None:
    """Single run: RRF scores = 1/(k + rank)."""
    run = [
        SearchResult(doc_id="a", score=10.0),
        SearchResult(doc_id="b", score=5.0),
    ]
    results = rrf_combine([run], k=60, top_n=2)
    assert len(results) == 2
    assert results[0].doc_id == "a"
    assert results[1].doc_id == "b"
    # score("a") = 1/(60+1) ≈ 0.01639
    assert abs(results[0].score - 1 / 61) < 1e-9
    assert abs(results[1].score - 1 / 62) < 1e-9


def test_rrf_combine_two_runs_overlap() -> None:
    """Two runs with overlapping docs: scores are summed."""
    run1 = [SearchResult(doc_id="a", score=1.0), SearchResult(doc_id="b", score=0.5)]
    run2 = [SearchResult(doc_id="b", score=1.0), SearchResult(doc_id="c", score=0.5)]
    results = rrf_combine([run1, run2], k=60, top_n=3)
    # "b" appears in both runs: rank 2 in run1 + rank 1 in run2
    # score("b") = 1/(60+2) + 1/(60+1) = 1/62 + 1/61
    # "a": rank 1 in run1 only = 1/61
    # "c": rank 2 in run2 only = 1/62
    assert results[0].doc_id == "b"  # highest combined score
    assert results[1].doc_id == "a"
    assert results[2].doc_id == "c"


def test_rrf_combine_top_n_truncates() -> None:
    """top_n limits the output length."""
    run = [
        SearchResult(doc_id="a", score=3.0),
        SearchResult(doc_id="b", score=2.0),
        SearchResult(doc_id="c", score=1.0),
    ]
    results = rrf_combine([run], k=60, top_n=1)
    assert len(results) == 1
    assert results[0].doc_id == "a"


def test_rrf_combine_empty_runs() -> None:
    """Empty runs produce empty results."""
    results = rrf_combine([], k=60, top_n=10)
    assert results == []
    results = rrf_combine([[]], k=60, top_n=10)
    assert results == []


def test_rrf_combine_three_runs() -> None:
    """Works with 3+ runs (reusability)."""
    run1 = [SearchResult(doc_id="a", score=1.0)]
    run2 = [SearchResult(doc_id="a", score=1.0)]
    run3 = [SearchResult(doc_id="a", score=1.0)]
    results = rrf_combine([run1, run2, run3], k=60, top_n=1)
    assert len(results) == 1
    assert abs(results[0].score - 3 / 61) < 1e-9
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hybrid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'coderag.algo.hybrid'`

**Step 3: Implement `rrf_combine`**

Create `src/coderag/algo/hybrid.py`:

```python
"""Hybrid search via Reciprocal Rank Fusion (RRF).

Provides:
- ``rrf_combine``: generic RRF over any number of ranked lists
- ``HybridAlgorithm``: BM25 + embedding fusion, implementing ``SearchAlgorithm``
"""

from __future__ import annotations

from coderag.interfaces import SearchResult


def rrf_combine(
    runs: list[list[SearchResult]],
    k: int,
    top_n: int,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion over multiple ranked result lists.

    For each document, the RRF score is ``sum(1 / (k + rank_i))``
    across all runs where it appears (1-indexed ranks).

    Returns the top *top_n* results sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    for run in runs:
        for rank, result in enumerate(run, start=1):
            scores[result.doc_id] = scores.get(result.doc_id, 0.0) + 1.0 / (k + rank)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        SearchResult(doc_id=doc_id, score=score)
        for doc_id, score in sorted_docs[:top_n]
    ]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hybrid.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/coderag/algo/hybrid.py tests/test_hybrid.py
git commit -m "feat: add rrf_combine() for reciprocal rank fusion"
```

---

## Task 4: Implement `HybridAlgorithm`

**Files:**
- Modify: `src/coderag/algo/hybrid.py`
- Modify: `tests/test_hybrid.py`

**Step 1: Write failing tests for `HybridAlgorithm`**

Append to `tests/test_hybrid.py`:

```python
from unittest.mock import MagicMock

from coderag.algo.hybrid import HybridAlgorithm


def _mock_bm25(results: list[SearchResult]) -> MagicMock:
    """Create a mock BM25 algorithm returning fixed results."""
    algo = MagicMock()
    algo.name = "bm25"
    algo.search.return_value = results
    return algo


def _mock_embedding(results: list[SearchResult]) -> MagicMock:
    """Create a mock embedding algorithm returning fixed results."""
    algo = MagicMock()
    algo.name = "embedding"
    algo.search.return_value = results
    return algo


def test_hybrid_name() -> None:
    """algo.name == 'hybrid'."""
    bm25 = _mock_bm25([])
    emb = _mock_embedding([])
    algo = HybridAlgorithm(
        bm25=bm25, embedding=emb,
        rrf_k=60, bm25_depth=10, embedding_depth=10,
    )
    assert algo.name == "hybrid"


def test_hybrid_search_merges_results() -> None:
    """Hybrid merges BM25 + embedding via RRF."""
    bm25_results = [
        SearchResult(doc_id="a", score=5.0),
        SearchResult(doc_id="b", score=3.0),
    ]
    emb_results = [
        SearchResult(doc_id="b", score=0.9),
        SearchResult(doc_id="c", score=0.5),
    ]
    bm25 = _mock_bm25(bm25_results)
    emb = _mock_embedding(emb_results)

    algo = HybridAlgorithm(
        bm25=bm25, embedding=emb,
        rrf_k=60, bm25_depth=10, embedding_depth=10,
    )
    results = algo.search("test query", top_k=3)

    # "b" is in both runs, so it should have the highest RRF score
    assert results[0].doc_id == "b"
    assert len(results) == 3

    # Verify sub-algorithms were called with their respective depths
    bm25.search.assert_called_once_with("test query", 10)
    emb.search.assert_called_once_with("test query", 10)


def test_hybrid_search_respects_top_k() -> None:
    """top_k limits output from hybrid."""
    bm25_results = [SearchResult(doc_id="a", score=5.0)]
    emb_results = [SearchResult(doc_id="b", score=0.9)]
    bm25 = _mock_bm25(bm25_results)
    emb = _mock_embedding(emb_results)

    algo = HybridAlgorithm(
        bm25=bm25, embedding=emb,
        rrf_k=60, bm25_depth=10, embedding_depth=10,
    )
    results = algo.search("test", top_k=1)
    assert len(results) == 1


def test_hybrid_search_separate_depths() -> None:
    """BM25 and embedding are called with their own depths."""
    bm25 = _mock_bm25([])
    emb = _mock_embedding([])

    algo = HybridAlgorithm(
        bm25=bm25, embedding=emb,
        rrf_k=60, bm25_depth=200, embedding_depth=50,
    )
    algo.search("query", top_k=10)
    bm25.search.assert_called_once_with("query", 200)
    emb.search.assert_called_once_with("query", 50)


def test_hybrid_search_both_empty() -> None:
    """Both empty sub-results produce empty hybrid results."""
    bm25 = _mock_bm25([])
    emb = _mock_embedding([])

    algo = HybridAlgorithm(
        bm25=bm25, embedding=emb,
        rrf_k=60, bm25_depth=10, embedding_depth=10,
    )
    results = algo.search("query", top_k=10)
    assert results == []
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hybrid.py::test_hybrid_name -v`
Expected: FAIL — `ImportError: cannot import name 'HybridAlgorithm'`

**Step 3: Implement `HybridAlgorithm`**

Append to `src/coderag/algo/hybrid.py`:

```python
class HybridAlgorithm:
    """BM25 + embedding fusion via Reciprocal Rank Fusion.

    Implements the :class:`~coderag.interfaces.SearchAlgorithm` protocol.

    Calls each sub-algorithm with its own retrieval depth, then
    merges the results using :func:`rrf_combine`.
    """

    def __init__(
        self,
        bm25: object,
        embedding: object,
        rrf_k: int,
        bm25_depth: int,
        embedding_depth: int,
    ) -> None:
        self._bm25 = bm25
        self._embedding = embedding
        self._rrf_k = rrf_k
        self._bm25_depth = bm25_depth
        self._embedding_depth = embedding_depth

    @property
    def name(self) -> str:
        return "hybrid"

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Run both sub-algorithms and merge via RRF."""
        bm25_results = self._bm25.search(query, self._bm25_depth)
        emb_results = self._embedding.search(query, self._embedding_depth)
        return rrf_combine(
            [bm25_results, emb_results], self._rrf_k, top_k
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hybrid.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add src/coderag/algo/hybrid.py tests/test_hybrid.py
git commit -m "feat: add HybridAlgorithm composing BM25 + embedding via RRF"
```

---

## Task 5: Wire hybrid algorithm into CLI

**Files:**
- Modify: `src/eval/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Write failing CLI test for hybrid**

Append to `tests/test_cli.py`:

```python
def test_cli_eval_run_hybrid_missing_rrf_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid algo requires --rrf-k."""
    data_dir = tmp_path / "data"
    suites_dir = data_dir / "eval_suites" / "test_suite"
    suites_dir.mkdir(parents=True)
    (suites_dir / "queries.json").write_text(
        json.dumps({"name": "test_suite", "queries": {"q1": "hello"}})
    )
    (suites_dir / "qrels.json").write_text(
        json.dumps({"q1": {"d1": 1}})
    )
    monkeypatch.setenv("CODERAG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CODERAG_PG_URL", "postgresql://test")
    with pytest.raises(SystemExit) as exc:
        main([
            "eval", "run", "hybrid",
            "--suite", "test_suite",
            "--top-k", "10",
            "--metrics", "ndcg@10",
            "--workers", "1",
            "--model-name", "test-model",
            "--reranker-name", "test-reranker",
            "--rerank-candidates", "10",
            "--gap-threshold", "1.0",
            "--bm25-db-path", "/tmp/nonexistent/bm25.db",
            "--reranker-max-length", "512",
            "--ef-search", "200",
            "--bm25-depth", "100",
            "--embedding-depth", "50",
            # --rrf-k intentionally missing
        ])
    assert "rrf-k" in str(exc.value).lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_cli_eval_run_hybrid_missing_rrf_k -v`
Expected: FAIL — `error: unrecognized arguments: --bm25-depth` (or the hybrid branch doesn't exist yet)

**Step 3: Add hybrid CLI args and resolution**

In `src/eval/cli.py`:

1. Add new args to run subparser (after the `--ef-search` line):
```python
    run_parser.add_argument("--rrf-k", type=int, help="RRF constant (hybrid algo)")
    run_parser.add_argument("--bm25-depth", type=int, help="BM25 retrieval depth (hybrid algo)")
    run_parser.add_argument("--embedding-depth", type=int, help="Embedding retrieval depth (hybrid algo)")
```

2. Add required flags validation for hybrid in `_eval_run` (after the embedding validation block, line 144):
```python
    if args.algorithm == "hybrid":
        for flag in ("model_name", "reranker_name", "rerank_candidates", "gap_threshold", "bm25_db_path", "reranker_max_length", "ef_search", "rrf_k", "bm25_depth", "embedding_depth"):
            if getattr(args, flag, None) is None:
                pretty = flag.replace("_", "-")
                sys.exit(f"Error: --{pretty} is required for hybrid algorithm")
```

3. Add hybrid branch to `_resolve_algorithm` (before the final `sys.exit`):
```python
    if name == "hybrid":
        from coderag.algo.bm25 import Bm25Algorithm
        from coderag.algo.embedding import EmbeddingAlgorithm
        from coderag.algo.hybrid import HybridAlgorithm

        bm25_db_path = data_dir / "algorithms" / "bm25" / "index.db"
        if not bm25_db_path.exists():
            sys.exit(
                f"Error: BM25 index not found at {bm25_db_path}\n"
                "Build it first with: coderag eval index --algorithm bm25 --corpus <path>"
            )
        bm25 = Bm25Algorithm(bm25_db_path)

        pg_url = _require_env("CODERAG_PG_URL")
        emb_bm25_path = Path(args.bm25_db_path)
        if not emb_bm25_path.exists():
            sys.exit(f"Error: BM25 index not found at {emb_bm25_path}")
        embedding = EmbeddingAlgorithm(
            pg_url,
            model_name=args.model_name,
            reranker_name=args.reranker_name,
            rerank_candidates=args.rerank_candidates,
            gap_threshold=args.gap_threshold,
            bm25_db_path=emb_bm25_path,
            reranker_max_length=args.reranker_max_length,
            ef_search=args.ef_search,
        )

        return HybridAlgorithm(
            bm25=bm25,
            embedding=embedding,
            rrf_k=args.rrf_k,
            bm25_depth=args.bm25_depth,
            embedding_depth=args.embedding_depth,
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All CLI tests PASS

**Step 5: Commit**

```bash
git add src/eval/cli.py tests/test_cli.py
git commit -m "feat: wire hybrid algorithm into CLI with --rrf-k, --bm25-depth, --embedding-depth"
```

---

## Task 6: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: 85+ tests pass (75 existing + 10 new hybrid + updates), 2 pre-existing env failures

**Step 2: Update scratchpad**

Update `.aisteering/scratchpads/2026-02-12-pgvector-migration.md` next steps section to reflect completed work and add note about running eval with new parameters.

**Step 3: Commit scratchpad**

```bash
git add .aisteering/scratchpads/2026-02-12-pgvector-migration.md
git commit -m "docs: update scratchpad with ef_search + hybrid RRF implementation"
```
