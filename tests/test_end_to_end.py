"""End-to-end test: build suite from fixture data, run mock algo, check metrics."""

from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from coderag.algo.bm25 import Bm25Algorithm, init as bm25_init
from coderag.interfaces import SearchResult
from csn.docstring import build_docstring_suite
from csn.download import extract_corpus, func_id
from csn.expert import build_expert_suite
from eval.runner import run_eval
from eval.suite import EvalSuite

FIXTURES = Path(__file__).parent / "fixtures"


class _FakePgBm25:
    """Functional in-memory postgres mock that stores docs and does simple keyword search."""

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, str]] = {}
        self._table_exists = False

    def make_conn(self) -> MagicMock:
        store = self
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        def execute(sql, params=None):
            s = str(sql)
            if "DROP TABLE" in s:
                store._docs.clear()
                store._table_exists = False
            elif "CREATE TABLE" in s:
                store._table_exists = True
            elif "CREATE INDEX" in s:
                pass
            elif "information_schema" in s:
                cursor.fetchone.return_value = (store._table_exists,)
            elif "ts_rank_cd" in s and params:
                query_tokens = set(params[1].lower().split())
                results = []
                for fid, doc in store._docs.items():
                    text = (
                        doc.get("func_name", "") + " " +
                        doc.get("code", "") + " " +
                        doc.get("docstring", "")
                    ).lower()
                    hits = sum(1 for t in query_tokens if t in text)
                    if hits:
                        results.append((fid, hits / len(query_tokens)))
                results.sort(key=lambda x: x[1], reverse=True)
                cursor.fetchall.return_value = results[: params[4]]

        def executemany(sql, rows):
            for row in rows:
                func_id_, func_name, code, docstring = row
                store._docs[func_id_] = {
                    "func_name": func_name,
                    "code": code,
                    "docstring": docstring,
                }

        cursor.execute.side_effect = execute
        cursor.executemany.side_effect = executemany
        return conn


class FixtureLookupAlgorithm:
    """Algorithm that returns results based on a pre-built lookup table."""

    def __init__(self, lookup: dict[str, list[SearchResult]]) -> None:
        self._lookup = lookup

    @property
    def name(self) -> str:
        return "fixture_lookup"

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return self._lookup.get(query, [])[:top_k]


def _make_zip(tmp_path: Path) -> Path:
    """Build a CSN-format zip from fixture JSONL."""
    records = FIXTURES.joinpath("sample_csn_records.jsonl").read_text().strip().split("\n")
    gz_path = tmp_path / "test.jsonl.gz"
    with gzip.open(gz_path, "wt") as gz:
        for rec in records:
            gz.write(rec + "\n")
    zip_path = tmp_path / "python.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(gz_path, "python/final/jsonl/test/python_test_0.jsonl.gz")
    return zip_path


def test_end_to_end_expert(tmp_path: Path) -> None:
    """Full pipeline: extract → build expert suite → run → metrics."""
    # 1. Extract corpus
    zip_path = _make_zip(tmp_path)
    corpus_dir = tmp_path / "datasets" / "codesearchnet"
    extract_corpus(zip_path, corpus_dir)

    # 2. Build expert suite
    suite = build_expert_suite(
        FIXTURES / "sample_annotations.csv",
        FIXTURES / "sample_queries.csv",
    )
    suite_dir = tmp_path / "suites" / "csn_expert"
    suite.save(suite_dir)

    # 3. Reload
    loaded = EvalSuite.load(suite_dir)
    assert loaded.queries == suite.queries

    # 4. Run with a lookup algo that returns relevant results
    url1 = "https://github.com/owner/repo1/blob/abc123/src/module.py#L1-L2"
    url2 = "https://github.com/owner/repo1/blob/abc123/src/module.py#L4-L5"
    lookup = {
        "sort a list": [
            SearchResult(doc_id=func_id(url1), score=2.0),
            SearchResult(doc_id=func_id(url2), score=1.0),
        ],
    }
    algo = FixtureLookupAlgorithm(lookup)

    scores = run_eval(
        algo, loaded, top_k=10,
        metrics=["ndcg@10", "mrr@10"],
        runs_dir=tmp_path / "runs",
    )

    assert "ndcg@10" in scores
    assert "mrr@10" in scores
    # We return relevant docs for "sort a list" but nothing for "read a file"
    # so scores should be > 0 but < 1
    assert 0.0 < scores["ndcg@10"] < 1.0
    assert 0.0 < scores["mrr@10"] <= 1.0

    # 5. Verify run was saved
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "metrics.json").exists()


def test_end_to_end_docstring(tmp_path: Path) -> None:
    """Full pipeline: extract → build docstring suite → run → metrics."""
    # 1. Extract corpus
    zip_path = _make_zip(tmp_path)
    corpus_dir = tmp_path / "datasets" / "codesearchnet"
    corpus_path = extract_corpus(zip_path, corpus_dir)

    # 2. Build docstring suite
    suite = build_docstring_suite(corpus_path, sample_count=3)
    assert len(suite.queries) == 3

    # 3. Run with a perfect lookup
    lookup: dict[str, list[SearchResult]] = {}
    for qid, query_text in suite.queries.items():
        lookup[query_text] = [SearchResult(doc_id=qid, score=1.0)]

    algo = FixtureLookupAlgorithm(lookup)
    scores = run_eval(
        algo, suite, top_k=10,
        metrics=["ndcg@10", "mrr@10"],
    )

    # Perfect results should yield perfect scores
    assert scores["ndcg@10"] == 1.0
    assert scores["mrr@10"] == 1.0


def test_end_to_end_bm25(tmp_path: Path) -> None:
    """Full pipeline: extract → build index → build suite → BM25 search → metrics."""
    fake_pg = _FakePgBm25()

    with patch("coderag.algo.bm25._get_pg_conn", side_effect=lambda _url: fake_pg.make_conn()):
        # 1. Extract corpus
        zip_path = _make_zip(tmp_path)
        corpus_dir = tmp_path / "datasets" / "codesearchnet"
        corpus_path = extract_corpus(zip_path, corpus_dir)

        # 2. Build BM25 index
        count = bm25_init(corpus_path, "postgresql://test")
        assert count == 5

        # 3. Build docstring suite (only functions with docstrings ≥3 words)
        suite = build_docstring_suite(corpus_path, sample_count=5)
        assert len(suite.queries) > 0

        # 4. Run BM25 against the suite
        algo = Bm25Algorithm("postgresql://test")
        scores = run_eval(
            algo, suite, top_k=10,
            metrics=["ndcg@10", "mrr@10"],
            runs_dir=tmp_path / "runs",
        )

    # BM25 should find at least some relevant results
    assert "ndcg@10" in scores
    assert "mrr@10" in scores
    assert scores["ndcg@10"] > 0.0
    assert scores["mrr@10"] > 0.0

    # 5. Verify run was saved
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "metrics.json").exists()
    assert runs[0].name.startswith("bm25_")
