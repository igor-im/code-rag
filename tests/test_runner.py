from __future__ import annotations

from pathlib import Path

from eval.runner import run_eval
from eval.suite import EvalSuite
from coderag.interfaces import SearchResult


class PerfectAlgorithm:
    """Returns the exact docs in the qrels, scored by relevance."""

    def __init__(self, qrels: dict[str, dict[str, int]]) -> None:
        self._qrels = qrels

    @property
    def name(self) -> str:
        return "perfect"

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        # We don't know qid from query text alone, so return all known docs
        # sorted by relevance. The runner matches by doc_id.
        all_docs: dict[str, int] = {}
        for qid_docs in self._qrels.values():
            for doc_id, rel in qid_docs.items():
                all_docs[doc_id] = max(all_docs.get(doc_id, 0), rel)
        ranked = sorted(all_docs.items(), key=lambda x: -x[1])[:top_k]
        return [SearchResult(doc_id=d, score=float(s)) for d, s in ranked]


class EmptyAlgorithm:
    """Returns no results."""

    @property
    def name(self) -> str:
        return "empty"

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return []


def _suite() -> EvalSuite:
    return EvalSuite(
        name="test",
        queries={"q1": "sort a list", "q2": "read a file"},
        qrels={
            "q1": {"d1": 3, "d2": 1, "d3": 0},
            "q2": {"d4": 2, "d5": 0},
        },
    )


def test_run_eval_returns_metrics() -> None:
    suite = _suite()
    algo = PerfectAlgorithm(suite.qrels)
    scores = run_eval(algo, suite, top_k=10, metrics=["ndcg@10", "mrr@10"])
    assert "ndcg@10" in scores
    assert "mrr@10" in scores
    assert all(isinstance(v, float) for v in scores.values())


def test_run_eval_perfect_algo_high_scores() -> None:
    suite = _suite()
    algo = PerfectAlgorithm(suite.qrels)
    scores = run_eval(algo, suite, top_k=10, metrics=["ndcg@10", "mrr@10"])
    # Perfect algo should score reasonably well (not necessarily 1.0
    # since it returns all docs merged, not per-query perfect)
    assert scores["mrr@10"] > 0.0


def test_run_eval_empty_algo_zero_scores() -> None:
    suite = _suite()
    algo = EmptyAlgorithm()
    scores = run_eval(algo, suite, top_k=10, metrics=["ndcg@10", "mrr@10"])
    assert scores["ndcg@10"] == 0.0
    assert scores["mrr@10"] == 0.0


def test_run_eval_saves_to_disk(tmp_path: Path) -> None:
    suite = _suite()
    algo = PerfectAlgorithm(suite.qrels)
    scores = run_eval(
        algo, suite, top_k=10, metrics=["ndcg@10", "mrr@10"], runs_dir=tmp_path
    )
    # Should have created a directory with run.json and metrics.json
    dirs = list(tmp_path.iterdir())
    assert len(dirs) == 1
    run_dir = dirs[0]
    assert run_dir.name.startswith("perfect_test_")
    assert (run_dir / "run.json").exists()
    assert (run_dir / "metrics.json").exists()


def test_run_eval_single_metric() -> None:
    """ranx returns scalar for single metric — runner should handle it."""
    suite = _suite()
    algo = PerfectAlgorithm(suite.qrels)
    scores = run_eval(algo, suite, top_k=10, metrics=["ndcg@10"])
    assert "ndcg@10" in scores
    assert isinstance(scores["ndcg@10"], float)
