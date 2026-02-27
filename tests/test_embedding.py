"""Tests for coderag.algo.embedding — embedding index and search."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBED_DIM = 4  # Small dimension for tests


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
    ]


def _mock_model(embed_dim: int = EMBED_DIM) -> MagicMock:
    """Create a mock transformer model that returns deterministic embeddings."""
    model = MagicMock()
    model.config.hidden_size = embed_dim
    # Simulate CPU device — return a fresh iterator each call
    model.parameters.side_effect = lambda: iter([torch.tensor([1.0])])
    model.eval.return_value = model
    model.to.return_value = model

    call_count = 0

    def forward(**kwargs):
        nonlocal call_count
        input_ids = kwargs["input_ids"]
        batch_size, seq_len = input_ids.shape
        # Deterministic embeddings: each doc gets a unique direction
        gen = torch.Generator().manual_seed(42 + call_count)
        call_count += 1
        hidden = torch.randn(batch_size, seq_len, embed_dim, generator=gen)
        result = MagicMock()
        result.last_hidden_state = hidden
        return result

    model.side_effect = forward
    return model


def _mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer that returns fake input tensors."""
    tokenizer = MagicMock()

    def tokenize(texts, **kwargs):
        batch_size = len(texts) if isinstance(texts, list) else 1
        seq_len = 10
        return {
            "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }

    tokenizer.side_effect = tokenize
    return tokenizer


_SAMPLE_DOC_TEXTS = {
    "aaa111": ("foo", "def foo():\n    pass", "Do foo things."),
    "bbb222": ("bar", "def bar(x):\n    return x", "Return x unchanged."),
    "ccc333": ("hello", "def hello(name):\n    print(name)", "Print a greeting."),
}


def _mock_pg_conn(known_vectors: dict[str, np.ndarray] | None = None, table_exists: bool = True) -> MagicMock:
    """Create a mock psycopg connection that simulates pgvector KNN and bm25_docs text lookups.

    If *known_vectors* is provided, KNN queries (detected by ``<=>`` in SQL)
    compute actual cosine similarity against the known vectors.
    """
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor

    if known_vectors is None:
        known_vectors = {}

    def execute_side_effect(sql, params=None):
        sql_str = sql if isinstance(sql, str) else str(sql)
        if sql_str.startswith("SET"):
            return  # session-level settings are no-ops in mock
        # Handle KNN queries
        if "<=>" in sql_str and params:
            query_vec = np.array(params[0], dtype=np.float32)
            k = params[2]
            results = []
            for fid, vec in known_vectors.items():
                vec = vec.astype(np.float32)
                dot = np.dot(query_vec, vec)
                norm_q = np.linalg.norm(query_vec)
                norm_v = np.linalg.norm(vec)
                sim = float(dot / (norm_q * norm_v)) if norm_q > 0 and norm_v > 0 else 0.0
                results.append((fid, sim))
            results.sort(key=lambda x: x[1], reverse=True)
            cursor.fetchall.return_value = results[:k]
        # Handle bm25_docs text lookups
        elif "bm25_docs" in sql_str and params:
            rows = [
                (fid, *_SAMPLE_DOC_TEXTS[fid])
                for fid in params
                if fid in _SAMPLE_DOC_TEXTS
            ]
            cursor.fetchall.return_value = rows
        # Handle table existence check
        elif "information_schema" in sql_str:
            cursor.fetchone.return_value = (table_exists,)

    cursor.execute.side_effect = execute_side_effect
    return conn


# ---------------------------------------------------------------------------
# init() tests
# ---------------------------------------------------------------------------


@patch("torch.cuda.is_available", return_value=False)
@patch("coderag.algo.embedding._get_pg_conn")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_init_creates_table(mock_tok_cls, mock_model_cls, mock_pg_conn, _mock_cuda, tmp_path: Path) -> None:
    mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
    mock_model_cls.from_pretrained.return_value = _mock_model()

    conn = _mock_pg_conn()
    mock_pg_conn.return_value = conn

    from coderag.algo.embedding import init

    corpus = _write_corpus(tmp_path, _sample_records())
    count = init(corpus, "postgresql://test", model_name="test-model", batch_size=2, max_length=512)

    assert count == 3

    # Verify SQL calls: DROP, CREATE TABLE, executemany (inserts), CREATE INDEX
    cur = conn.cursor()
    sql_calls = [str(c) for c in cur.execute.call_args_list]
    sql_text = " ".join(sql_calls)
    assert "DROP TABLE" in sql_text
    assert "CREATE TABLE" in sql_text
    assert "hnsw" in sql_text.lower()


@patch("torch.cuda.is_available", return_value=False)
@patch("coderag.algo.embedding._get_pg_conn")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_init_inserts_embeddings(mock_tok_cls, mock_model_cls, mock_pg_conn, _mock_cuda, tmp_path: Path) -> None:
    mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
    mock_model_cls.from_pretrained.return_value = _mock_model()

    conn = _mock_pg_conn()
    mock_pg_conn.return_value = conn

    from coderag.algo.embedding import init

    corpus = _write_corpus(tmp_path, _sample_records())
    init(corpus, "postgresql://test", model_name="test-model", batch_size=10, max_length=512)

    # Verify executemany was called with 3 rows
    cur = conn.cursor()
    executemany_calls = cur.executemany.call_args_list
    assert len(executemany_calls) >= 1
    # All rows across all executemany calls
    total_rows = sum(len(c.args[1]) for c in executemany_calls)
    assert total_rows == 3


@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_init_corpus_not_found(mock_tok_cls, mock_model_cls, tmp_path: Path) -> None:
    mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
    mock_model_cls.from_pretrained.return_value = _mock_model()

    from coderag.algo.embedding import init

    with pytest.raises(FileNotFoundError, match="Corpus not found"):
        init(tmp_path / "nope.jsonl", "postgresql://test",
             model_name="test-model", batch_size=10, max_length=512)


@patch("torch.cuda.is_available", return_value=False)
@patch("coderag.algo.embedding._get_pg_conn")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_init_replaces_existing(mock_tok_cls, mock_model_cls, mock_pg_conn, _mock_cuda, tmp_path: Path) -> None:
    mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
    mock_model_cls.from_pretrained.return_value = _mock_model()

    conn = _mock_pg_conn()
    mock_pg_conn.return_value = conn

    from coderag.algo.embedding import init

    corpus = _write_corpus(tmp_path, _sample_records())
    first = init(corpus, "postgresql://test", model_name="test-model", batch_size=10, max_length=512)

    # Reset mock model for second run
    mock_model_cls.from_pretrained.return_value = _mock_model()
    conn2 = _mock_pg_conn()
    mock_pg_conn.return_value = conn2
    second = init(corpus, "postgresql://test", model_name="test-model", batch_size=10, max_length=512)

    assert first == second == 3
    # Verify DROP TABLE was called on both runs
    for c in [conn, conn2]:
        cur = c.cursor()
        sql_calls = " ".join(str(ca) for ca in cur.execute.call_args_list)
        assert "DROP TABLE" in sql_calls


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


def _build_test_mock() -> MagicMock:
    """Build a mock pgvector connection with known vectors.

    Vectors (dimension=4):
    - aaa111: [1, 0, 0, 0]  -> cosine sim 1.0 from query [1,0,0,0]
    - bbb222: [0.7, 0.7, 0, 0] normalized -> cosine sim ~0.707
    - ccc333: [0, 0, 1, 0]  -> cosine sim 0.0 (orthogonal)
    """
    vecs = {
        "aaa111": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "bbb222": np.array([0.7, 0.7, 0.0, 0.0], dtype=np.float32),
        "ccc333": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    bbb = vecs["bbb222"]
    vecs["bbb222"] = bbb / np.linalg.norm(bbb)
    return _mock_pg_conn(known_vectors=vecs, table_exists=True)


def _mock_reranker() -> MagicMock:
    """Mock reranker that returns ascending logits.

    The LAST KNN candidate gets the highest score, so the reranker
    will reverse the KNN ordering.
    """
    reranker = MagicMock()
    reranker.eval.return_value = reranker
    reranker.parameters.side_effect = lambda: iter([torch.tensor([1.0])])

    def forward(**kwargs):
        input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        result = MagicMock()
        result.logits = torch.linspace(0.0, 2.0, batch_size).unsqueeze(-1)
        return result

    reranker.side_effect = forward
    return reranker


def _mock_reranker_tokenizer() -> MagicMock:
    """Mock tokenizer for reranker pairs."""
    tokenizer = MagicMock()

    def tokenize(pairs, **kwargs):
        batch_size = len(pairs)
        seq_len = 10
        return {
            "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }

    tokenizer.side_effect = tokenize
    return tokenizer


# ---------------------------------------------------------------------------
# EmbeddingAlgorithm search tests
# ---------------------------------------------------------------------------


@patch("torch.cuda.is_available", return_value=False)
@patch("coderag.algo.embedding._get_pg_conn")
@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_search_returns_results(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, mock_pg_conn, _mock_cuda, tmp_path: Path
) -> None:
    """Basic: search returns SearchResult objects."""
    mock_conn = _build_test_mock()
    mock_pg_conn.return_value = mock_conn

    embedder = _mock_model()

    def embed_query(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        result = MagicMock()
        hidden = torch.zeros(batch_size, 10, EMBED_DIM)
        hidden[:, -1, 0] = 1.0
        result.last_hidden_state = hidden
        return result

    embedder.side_effect = embed_query
    mock_model_cls.from_pretrained.return_value = embedder

    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()
    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]

    from coderag.algo.embedding import EmbeddingAlgorithm
    from coderag.interfaces import SearchResult

    algo = EmbeddingAlgorithm(
        "postgresql://test",
        model_name="test-model",
        reranker_name="test-reranker",
        rerank_candidates=10,
        gap_threshold=10.0,
        reranker_max_length=512,
        ef_search=40,
    )
    results = algo.search("test query", top_k=3)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


@patch("torch.cuda.is_available", return_value=False)
@patch("coderag.algo.embedding._get_pg_conn")
@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_search_reranker_reorders(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, mock_pg_conn, _mock_cuda, tmp_path: Path
) -> None:
    """With no gating (gap_threshold=10.0), reranker reverses order so ccc333 is #1."""
    mock_conn = _build_test_mock()
    mock_pg_conn.return_value = mock_conn

    embedder = _mock_model()

    def embed_query(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        result = MagicMock()
        hidden = torch.zeros(batch_size, 10, EMBED_DIM)
        hidden[:, -1, 0] = 1.0
        result.last_hidden_state = hidden
        return result

    embedder.side_effect = embed_query
    mock_model_cls.from_pretrained.return_value = embedder

    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()
    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]

    from coderag.algo.embedding import EmbeddingAlgorithm

    algo = EmbeddingAlgorithm(
        "postgresql://test",
        model_name="test-model",
        reranker_name="test-reranker",
        rerank_candidates=10,
        gap_threshold=10.0,
        reranker_max_length=512,
        ef_search=40,
    )
    results = algo.search("test query", top_k=3)
    # Reranker gives ascending logits, so the last KNN candidate (ccc333) gets highest
    assert results[0].doc_id == "ccc333"


@patch("torch.cuda.is_available", return_value=False)
@patch("coderag.algo.embedding._get_pg_conn")
@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_search_gating_pins_top(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, mock_pg_conn, _mock_cuda, tmp_path: Path
) -> None:
    """With gating (gap_threshold=0.0), aaa111 stays #1 despite reranker reversal."""
    mock_conn = _build_test_mock()
    mock_pg_conn.return_value = mock_conn

    embedder = _mock_model()

    def embed_query(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        result = MagicMock()
        hidden = torch.zeros(batch_size, 10, EMBED_DIM)
        hidden[:, -1, 0] = 1.0
        result.last_hidden_state = hidden
        return result

    embedder.side_effect = embed_query
    mock_model_cls.from_pretrained.return_value = embedder

    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()
    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]

    from coderag.algo.embedding import EmbeddingAlgorithm

    algo = EmbeddingAlgorithm(
        "postgresql://test",
        model_name="test-model",
        reranker_name="test-reranker",
        rerank_candidates=10,
        gap_threshold=0.0,
        reranker_max_length=512,
        ef_search=40,
    )
    results = algo.search("test query", top_k=3)
    # Gating threshold 0.0: gap between aaa111 (sim=1.0) and bbb222 (sim~0.707) > 0.0 => pin aaa111
    assert results[0].doc_id == "aaa111"


@patch("torch.cuda.is_available", return_value=False)
@patch("coderag.algo.embedding._get_pg_conn")
@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_search_empty_query(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, mock_pg_conn, _mock_cuda, tmp_path: Path
) -> None:
    """Empty string and whitespace queries return empty list."""
    mock_conn = _build_test_mock()
    mock_pg_conn.return_value = mock_conn

    embedder = _mock_model()

    def embed_query(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        result = MagicMock()
        hidden = torch.zeros(batch_size, 10, EMBED_DIM)
        hidden[:, -1, 0] = 1.0
        result.last_hidden_state = hidden
        return result

    embedder.side_effect = embed_query
    mock_model_cls.from_pretrained.return_value = embedder

    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()
    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]

    from coderag.algo.embedding import EmbeddingAlgorithm

    algo = EmbeddingAlgorithm(
        "postgresql://test",
        model_name="test-model",
        reranker_name="test-reranker",
        rerank_candidates=10,
        gap_threshold=10.0,
        reranker_max_length=512,
        ef_search=40,
    )
    assert algo.search("", top_k=3) == []
    assert algo.search("   ", top_k=3) == []


@patch("torch.cuda.is_available", return_value=False)
@patch("coderag.algo.embedding._get_pg_conn")
@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_name_property(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, mock_pg_conn, _mock_cuda, tmp_path: Path
) -> None:
    """algo.name == 'embedding'."""
    mock_conn = _build_test_mock()
    mock_pg_conn.return_value = mock_conn

    embedder = _mock_model()

    def embed_query(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        result = MagicMock()
        hidden = torch.zeros(batch_size, 10, EMBED_DIM)
        hidden[:, -1, 0] = 1.0
        result.last_hidden_state = hidden
        return result

    embedder.side_effect = embed_query
    mock_model_cls.from_pretrained.return_value = embedder

    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()
    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]

    from coderag.algo.embedding import EmbeddingAlgorithm

    algo = EmbeddingAlgorithm(
        "postgresql://test",
        model_name="test-model",
        reranker_name="test-reranker",
        rerank_candidates=10,
        gap_threshold=10.0,
        reranker_max_length=512,
        ef_search=40,
    )
    assert algo.name == "embedding"

    # Verify SET hnsw.ef_search was issued with the provided value
    set_calls = [
        c.args[0] for c in mock_conn.cursor().execute.call_args_list
        if isinstance(c.args[0], str) and c.args[0].startswith("SET")
    ]
    assert len(set_calls) == 1
    assert "40" in set_calls[0]


@patch("torch.cuda.is_available", return_value=False)
@patch("coderag.algo.embedding._get_pg_conn")
@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_search_respects_top_k(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, mock_pg_conn, _mock_cuda, tmp_path: Path
) -> None:
    """top_k=1 returns exactly 1 result."""
    mock_conn = _build_test_mock()
    mock_pg_conn.return_value = mock_conn

    embedder = _mock_model()

    def embed_query(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        result = MagicMock()
        hidden = torch.zeros(batch_size, 10, EMBED_DIM)
        hidden[:, -1, 0] = 1.0
        result.last_hidden_state = hidden
        return result

    embedder.side_effect = embed_query
    mock_model_cls.from_pretrained.return_value = embedder

    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()
    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]

    from coderag.algo.embedding import EmbeddingAlgorithm

    algo = EmbeddingAlgorithm(
        "postgresql://test",
        model_name="test-model",
        reranker_name="test-reranker",
        rerank_candidates=10,
        gap_threshold=10.0,
        reranker_max_length=512,
        ef_search=40,
    )
    results = algo.search("test query", top_k=1)
    assert len(results) == 1


@patch("torch.cuda.is_available", return_value=False)
@patch("coderag.algo.embedding._get_pg_conn")
@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_init_missing_table(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, mock_pg_conn, _mock_cuda, tmp_path: Path
) -> None:
    """FileNotFoundError when embedding_docs table doesn't exist in pg."""
    mock_conn = _mock_pg_conn(table_exists=False)
    mock_pg_conn.return_value = mock_conn

    embedder = _mock_model()
    mock_model_cls.from_pretrained.return_value = embedder
    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()
    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]

    from coderag.algo.embedding import EmbeddingAlgorithm

    with pytest.raises(FileNotFoundError, match="embedding_docs"):
        EmbeddingAlgorithm(
            "postgresql://test",
            model_name="test-model",
            reranker_name="test-reranker",
            rerank_candidates=10,
            gap_threshold=1.0,
            reranker_max_length=512,
            ef_search=40,
        )
