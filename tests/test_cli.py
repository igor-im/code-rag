from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from eval.cli import main


def test_cli_no_args(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 1


def test_cli_eval_csn_init_missing_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODERAG_DATASETS_DIR", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["eval", "csn", "init"])
    assert "CODERAG_DATASETS_DIR" in str(exc.value)


def test_cli_eval_run_missing_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODERAG_DATA_DIR", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["eval", "run", "dummy", "--suite", "x", "--top-k", "10", "--metrics", "ndcg@10", "--workers", "1"])
    assert "CODERAG_DATA_DIR" in str(exc.value)


def test_cli_eval_run_unknown_algorithm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    suites_dir = data_dir / "eval_suites" / "test_suite"
    suites_dir.mkdir(parents=True)
    # Create a minimal suite
    (suites_dir / "queries.json").write_text(
        json.dumps({"name": "test_suite", "queries": {"q1": "hello"}})
    )
    (suites_dir / "qrels.json").write_text(
        json.dumps({"q1": {"d1": 1}})
    )
    monkeypatch.setenv("CODERAG_DATA_DIR", str(data_dir))
    with pytest.raises(SystemExit) as exc:
        main(["eval", "run", "nonexistent", "--suite", "test_suite", "--top-k", "10", "--metrics", "ndcg@10", "--workers", "1"])
    assert "unknown algorithm" in str(exc.value)


def test_cli_eval_run_dummy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
    main(["eval", "run", "dummy", "--suite", "test_suite", "--top-k", "10", "--metrics", "ndcg@10,mrr@10", "--workers", "1"])
    captured = capsys.readouterr()
    assert "ndcg@10" in captured.out
    assert "mrr@10" in captured.out


def _write_corpus(tmp_path: Path) -> Path:
    """Write a small extracted corpus JSONL and return its path."""
    corpus = tmp_path / "corpus.jsonl"
    records = [
        {"func_id": "aaa", "func_name": "foo", "code": "def foo(): pass", "docstring": "Do foo."},
        {"func_id": "bbb", "func_name": "bar", "code": "def bar(x): return x", "docstring": "Return x."},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return corpus


@patch("coderag.algo.bm25._get_pg_conn")
def test_cli_eval_index_bm25(
    mock_pg_conn,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from unittest.mock import MagicMock
    conn = MagicMock()
    mock_pg_conn.return_value = conn

    data_dir = tmp_path / "data"
    corpus = _write_corpus(tmp_path)
    monkeypatch.setenv("CODERAG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CODERAG_PG_URL", "postgresql://test")
    main(["eval", "index", "--algorithm", "bm25", "--corpus", str(corpus)])
    captured = capsys.readouterr()
    assert "2" in captured.out  # 2 documents indexed


def test_cli_eval_index_missing_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODERAG_DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        main(["eval", "index", "--algorithm", "bm25", "--corpus", "/no/such/file.jsonl"])
    assert "corpus not found" in str(exc.value).lower()


@patch("coderag.algo.bm25._get_pg_conn")
def test_cli_eval_run_bm25_missing_table(
    mock_pg_conn,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock
    conn = MagicMock()
    conn.cursor().fetchone.return_value = (False,)  # table_exists=False
    mock_pg_conn.return_value = conn

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
        main(["eval", "run", "bm25", "--suite", "test_suite", "--top-k", "10", "--metrics", "ndcg@10", "--workers", "1"])
    assert "bm25_docs" in str(exc.value).lower() or "BM25" in str(exc.value)


def test_cli_eval_compare_no_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODERAG_DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        main(["eval", "compare"])
    assert "no runs directory" in str(exc.value).lower()


def test_cli_eval_compare_with_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    runs_dir = data_dir / "runs"
    run1 = runs_dir / "algo1_suite1_20250101T000000Z"
    run1.mkdir(parents=True)
    (run1 / "metrics.json").write_text(json.dumps({"ndcg@10": 0.5, "mrr@10": 0.4}))

    monkeypatch.setenv("CODERAG_DATA_DIR", str(data_dir))
    main(["eval", "compare"])
    captured = capsys.readouterr()
    assert "algo1_suite1" in captured.out
    assert "0.5000" in captured.out


@patch("torch.cuda.is_available", return_value=False)
@patch("coderag.algo.embedding._get_pg_conn")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_cli_eval_index_embedding(
    mock_tok_cls,
    mock_model_cls,
    mock_pg_conn,
    _mock_cuda,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.test_embedding import _mock_model, _mock_tokenizer, _mock_pg_conn

    mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
    mock_model_cls.from_pretrained.return_value = _mock_model()
    mock_pg_conn.return_value = _mock_pg_conn()

    data_dir = tmp_path / "data"
    corpus = _write_corpus(tmp_path)
    monkeypatch.setenv("CODERAG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CODERAG_PG_URL", "postgresql://test")
    main([
        "eval", "index",
        "--algorithm", "embedding",
        "--corpus", str(corpus),
        "--model-name", "test-model",
        "--batch-size", "10",
        "--max-length", "512",
    ])
    captured = capsys.readouterr()
    assert "2" in captured.out  # 2 documents indexed


def test_cli_eval_index_unknown_algorithm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _write_corpus(tmp_path)
    monkeypatch.setenv("CODERAG_DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        main([
            "eval", "index",
            "--algorithm", "nonexistent",
            "--corpus", str(corpus),
        ])
    assert "unknown algorithm" in str(exc.value).lower()


def test_cli_eval_run_embedding_missing_pg_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.delenv("CODERAG_PG_URL", raising=False)
    with pytest.raises(SystemExit) as exc:
        main([
            "eval", "run", "embedding",
            "--suite", "test_suite",
            "--top-k", "10",
            "--metrics", "ndcg@10",
            "--workers", "1",
            "--model-name", "test-model",
            "--reranker-name", "test-reranker",
            "--rerank-candidates", "10",
            "--gap-threshold", "1.0",
            "--reranker-max-length", "512",
            "--ef-search", "40",
        ])
    assert "CODERAG_PG_URL" in str(exc.value)


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
            "--reranker-max-length", "512",
            "--ef-search", "200",
            "--bm25-depth", "100",
            "--embedding-depth", "50",
            # --rrf-k intentionally missing
        ])
    assert "rrf-k" in str(exc.value).lower()
