"""Tests for coderag.algo.bm25 — PostgreSQL FTS indexer and search algorithm."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from coderag.algo.bm25 import Bm25Algorithm, init


def _write_corpus(path: Path, records: list[dict[str, str]]) -> Path:
    """Write JSONL corpus file and return its path."""
    corpus = path / "corpus.jsonl"
    with open(corpus, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return corpus


def _sample_records() -> list[dict[str, str]]:
    return [
        {
            "func_id": "aaa111",
            "func_name": "foo",
            "code": "def foo():\n    pass",
            "docstring": "Do foo things.",
        },
        {
            "func_id": "bbb222",
            "func_name": "bar",
            "code": "def bar(x):\n    return x",
            "docstring": "Return x unchanged.",
        },
        {
            "func_id": "ccc333",
            "func_name": "hello",
            "code": "def hello(name):\n    print(name)",
            "docstring": "Print a greeting.",
        },
        {
            "func_id": "ddd444",
            "func_name": "qux",
            "code": "def qux(a, b):\n    return a + b",
            "docstring": "Add two numbers together.",
        },
        {
            "func_id": "eee555",
            "func_name": "sort_list",
            "code": "def sort_list(items):\n    return sorted(items)",
            "docstring": "Sort a list of items in ascending order.",
        },
    ]


def _mock_pg_conn(table_exists: bool = True) -> MagicMock:
    """Create a mock psycopg connection for BM25 tests."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.fetchone.return_value = (table_exists,)
    cursor.fetchall.return_value = []
    return conn


# ── Indexer tests ──────────────────────────────────────────────────────


@patch("coderag.algo.bm25._get_pg_conn")
def test_build_index_returns_count(mock_get_conn, tmp_path: Path) -> None:
    conn = _mock_pg_conn()
    mock_get_conn.return_value = conn
    corpus = _write_corpus(tmp_path, _sample_records())

    count = init(corpus, "postgresql://test")

    assert count == 5


@patch("coderag.algo.bm25._get_pg_conn")
def test_build_index_issues_correct_sql(mock_get_conn, tmp_path: Path) -> None:
    conn = _mock_pg_conn()
    mock_get_conn.return_value = conn
    corpus = _write_corpus(tmp_path, _sample_records())

    init(corpus, "postgresql://test")

    cur = conn.cursor()
    sql_text = " ".join(str(c) for c in cur.execute.call_args_list)
    assert "DROP TABLE" in sql_text
    assert "CREATE TABLE" in sql_text
    assert "gin" in sql_text.lower()
    assert "CREATE INDEX" in sql_text


@patch("coderag.algo.bm25._get_pg_conn")
def test_build_index_inserts_all_rows(mock_get_conn, tmp_path: Path) -> None:
    conn = _mock_pg_conn()
    mock_get_conn.return_value = conn
    corpus = _write_corpus(tmp_path, _sample_records())

    init(corpus, "postgresql://test")

    cur = conn.cursor()
    total_rows = sum(len(c.args[1]) for c in cur.executemany.call_args_list)
    assert total_rows == 5


def test_build_index_corpus_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Corpus not found"):
        init(tmp_path / "nope.jsonl", "postgresql://test")


@patch("coderag.algo.bm25._get_pg_conn")
def test_build_index_replaces_existing(mock_get_conn, tmp_path: Path) -> None:
    conn = _mock_pg_conn()
    mock_get_conn.return_value = conn
    corpus = _write_corpus(tmp_path, _sample_records())

    first = init(corpus, "postgresql://test")

    conn2 = _mock_pg_conn()
    mock_get_conn.return_value = conn2
    second = init(corpus, "postgresql://test")

    assert first == second == 5
    for c in [conn, conn2]:
        sql_text = " ".join(str(ca) for ca in c.cursor().execute.call_args_list)
        assert "DROP TABLE" in sql_text


@patch("coderag.algo.bm25._get_pg_conn")
def test_build_index_rollback_on_error(mock_get_conn, tmp_path: Path) -> None:
    conn = _mock_pg_conn()
    conn.cursor().executemany.side_effect = RuntimeError("DB error")
    mock_get_conn.return_value = conn

    corpus = _write_corpus(tmp_path, _sample_records())

    with pytest.raises(RuntimeError, match="DB error"):
        init(corpus, "postgresql://test")

    conn.rollback.assert_called()


# ── Algorithm tests ────────────────────────────────────────────────────


@patch("coderag.algo.bm25._get_pg_conn")
def test_init_missing_table(mock_get_conn) -> None:
    conn = _mock_pg_conn(table_exists=False)
    mock_get_conn.return_value = conn

    with pytest.raises(FileNotFoundError, match="bm25_docs"):
        Bm25Algorithm("postgresql://test")


@patch("coderag.algo.bm25._get_pg_conn")
def test_name_property(mock_get_conn) -> None:
    conn = _mock_pg_conn()
    mock_get_conn.return_value = conn

    algo = Bm25Algorithm("postgresql://test")
    assert algo.name == "bm25"


@patch("coderag.algo.bm25._get_pg_conn")
def test_search_returns_results(mock_get_conn) -> None:
    conn = _mock_pg_conn()
    conn.cursor().fetchall.return_value = [("eee555", 0.5), ("ddd444", 0.3)]
    mock_get_conn.return_value = conn

    algo = Bm25Algorithm("postgresql://test")
    results = algo.search("sort list", top_k=10)

    assert len(results) == 2
    assert results[0].doc_id == "eee555"
    assert results[0].score == pytest.approx(0.5)


@patch("coderag.algo.bm25._get_pg_conn")
def test_search_respects_top_k(mock_get_conn) -> None:
    conn = _mock_pg_conn()
    conn.cursor().fetchall.return_value = [("aaa111", 0.4), ("bbb222", 0.3)]
    mock_get_conn.return_value = conn

    algo = Bm25Algorithm("postgresql://test")
    # top_k is passed to LIMIT in the query; mock returns 2
    results = algo.search("def", top_k=2)
    assert len(results) <= 2


@patch("coderag.algo.bm25._get_pg_conn")
def test_search_scores_positive(mock_get_conn) -> None:
    conn = _mock_pg_conn()
    conn.cursor().fetchall.return_value = [("bbb222", 0.2)]
    mock_get_conn.return_value = conn

    algo = Bm25Algorithm("postgresql://test")
    results = algo.search("return", top_k=10)
    assert all(r.score > 0 for r in results)


@patch("coderag.algo.bm25._get_pg_conn")
def test_search_empty_query(mock_get_conn) -> None:
    conn = _mock_pg_conn()
    mock_get_conn.return_value = conn

    algo = Bm25Algorithm("postgresql://test")
    assert algo.search("", top_k=10) == []
    assert algo.search("   ", top_k=10) == []


@patch("coderag.algo.bm25._get_pg_conn")
def test_search_no_match(mock_get_conn) -> None:
    conn = _mock_pg_conn()
    conn.cursor().fetchall.return_value = []
    mock_get_conn.return_value = conn

    algo = Bm25Algorithm("postgresql://test")
    results = algo.search("xyznonexistent", top_k=10)
    assert results == []


@patch("coderag.algo.bm25._get_pg_conn")
def test_search_special_characters(mock_get_conn) -> None:
    conn = _mock_pg_conn()
    conn.cursor().fetchall.return_value = []
    mock_get_conn.return_value = conn

    algo = Bm25Algorithm("postgresql://test")
    # plainto_tsquery handles these safely (no FTS syntax errors)
    algo.search("foo AND bar", top_k=5)
    algo.search("foo OR bar", top_k=5)
    algo.search('foo "bar"', top_k=5)
    algo.search("foo(bar)", top_k=5)
    algo.search("a + b", top_k=5)
