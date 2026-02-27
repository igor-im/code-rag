"""Tests for coderag.algo.hybrid — RRF combiner and hybrid search."""

from __future__ import annotations

from unittest.mock import MagicMock

from coderag.algo.hybrid import HybridAlgorithm, rrf_combine
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
    assert abs(results[0].score - 1 / 61) < 1e-9
    assert abs(results[1].score - 1 / 62) < 1e-9


def test_rrf_combine_two_runs_overlap() -> None:
    """Two runs with overlapping docs: scores are summed."""
    run1 = [SearchResult(doc_id="a", score=1.0), SearchResult(doc_id="b", score=0.5)]
    run2 = [SearchResult(doc_id="b", score=1.0), SearchResult(doc_id="c", score=0.5)]
    results = rrf_combine([run1, run2], k=60, top_n=3)
    assert results[0].doc_id == "b"
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

    assert results[0].doc_id == "b"
    assert len(results) == 3

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
