# Embedding Algorithm Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a two-stage embedding search algorithm (BGE-Code-v1 + BGE-Reranker-v2-m3 with score-gap gating) that implements the SearchAlgorithm protocol.

**Architecture:** Dense retrieval via sqlite-vec KNN, then reranking with cross-encoder. Score-gap gating protects high-confidence top results from being displaced by the reranker. Document text for reranking is read from the existing BM25 index.

**Tech Stack:** transformers, torch, sqlite-vec, sqlite3

**Design doc:** `docs/plans/2026-02-10-embedding-algorithm-design.md`

---

### Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml:10-14`

**Step 1: Add transformers, torch, sqlite-vec to dependencies**

```toml
dependencies = [
    "ranx>=0.3",
    "datasets>=3.0",
    "python-dotenv>=1.0",
    "transformers>=4.40",
    "torch>=2.0",
    "sqlite-vec>=0.1.6",
]
```

**Step 2: Install**

Run: `uv sync`
Expected: resolves and installs all deps

**Step 3: Verify import**

Run: `uv run python -c "import sqlite_vec; import transformers; import torch; print('ok')"`
Expected: `ok`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add transformers, torch, sqlite-vec dependencies"
```

---

### Task 2: Rename BM25 build_index → init

**Files:**
- Modify: `src/coderag/algo/bm25.py:23` (rename function)
- Modify: `src/eval/cli.py:94` (update import + call)
- Modify: `tests/test_bm25.py:11,66-67,75,89,91` (update imports + calls)
- Modify: `tests/test_end_to_end.py:10,133` (update import + call)
- Modify: `tests/test_cli.py:94,96` (update reference in assertion context)

**Step 1: Rename function in bm25.py**

In `src/coderag/algo/bm25.py`, rename `build_index` to `init`:
- Line 23: `def build_index(` → `def init(`
- Line 6: update docstring: `- ``build_index``:` → `- ``init``:`

**Step 2: Update test_bm25.py imports and calls**

In `tests/test_bm25.py`:
- Line 11: `from coderag.algo.bm25 import Bm25Algorithm, build_index` → `from coderag.algo.bm25 import Bm25Algorithm, init`
- Line 66: `build_index(corpus_path, p)` → `init(corpus_path, p)`
- Line 75: `count = build_index(corpus_path, db)` → `count = init(corpus_path, db)`
- Line 82: `with pytest.raises(FileNotFoundError, match="Corpus not found"):` (no change)
- Line 83: `build_index(tmp_path / "nope.jsonl", tmp_path / "index.db")` → `init(tmp_path / "nope.jsonl", tmp_path / "index.db")`
- Line 89: `first = build_index(corpus_path, db)` → `first = init(corpus_path, db)`
- Line 90: `second = build_index(corpus_path, db)` → `second = init(corpus_path, db)`
- Line 99: `build_index(bad_corpus, db)` → `init(bad_corpus, db)`

**Step 3: Update test_end_to_end.py**

In `tests/test_end_to_end.py`:
- Line 10: `from coderag.algo.bm25 import Bm25Algorithm, build_index` → `from coderag.algo.bm25 import Bm25Algorithm, init as bm25_init`
- Line 133: `count = build_index(corpus_path, db_path)` → `count = bm25_init(corpus_path, db_path)`

**Step 4: Update cli.py**

In `src/eval/cli.py`:
- Line 94: `from coderag.algo.bm25 import build_index` → `from coderag.algo.bm25 import init as bm25_init`
- Line 98: `count = build_index(corpus_path, db_path)` → `count = bm25_init(corpus_path, db_path)`

**Step 5: Run tests to verify nothing broke**

Run: `uv run pytest tests/ -v`
Expected: 62 tests pass

**Step 6: Commit**

```bash
git add src/coderag/algo/bm25.py src/eval/cli.py tests/test_bm25.py tests/test_end_to_end.py
git commit -m "refactor: rename BM25 build_index to init for standardized algo API"
```

---

### Task 3: Embedding init() — index building

**Files:**
- Create: `src/coderag/algo/embedding.py`
- Create: `tests/test_embedding.py`

**Step 1: Write the failing tests for init()**

Create `tests/test_embedding.py`:

```python
"""Tests for coderag.algo.embedding — embedding index and search."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sqlite_vec
import torch

from coderag.algo.bm25 import init as bm25_init


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
    # Simulate CPU device
    model.parameters.return_value = iter([torch.tensor([1.0])])
    model.eval.return_value = model

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


# ---------------------------------------------------------------------------
# init() tests
# ---------------------------------------------------------------------------


@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_init_creates_db(mock_tok_cls, mock_model_cls, tmp_path: Path) -> None:
    mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
    mock_model_cls.from_pretrained.return_value = _mock_model()

    from coderag.algo.embedding import init

    corpus = _write_corpus(tmp_path, _sample_records())
    db = tmp_path / "algorithms" / "embedding" / "index.db"
    count = init(corpus, db, model_name="test-model", batch_size=2)

    assert db.exists()
    assert count == 3


@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_init_stores_embeddings(mock_tok_cls, mock_model_cls, tmp_path: Path) -> None:
    mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
    mock_model_cls.from_pretrained.return_value = _mock_model()

    from coderag.algo.embedding import init

    corpus = _write_corpus(tmp_path, _sample_records())
    db = tmp_path / "index.db"
    init(corpus, db, model_name="test-model", batch_size=10)

    # Verify we can read back the func_ids
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    rows = conn.execute("SELECT func_id FROM embedding_docs ORDER BY rowid").fetchall()
    conn.close()
    assert [r[0] for r in rows] == ["aaa111", "bbb222", "ccc333"]


@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_init_corpus_not_found(mock_tok_cls, mock_model_cls, tmp_path: Path) -> None:
    mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
    mock_model_cls.from_pretrained.return_value = _mock_model()

    from coderag.algo.embedding import init

    with pytest.raises(FileNotFoundError, match="Corpus not found"):
        init(tmp_path / "nope.jsonl", tmp_path / "index.db",
             model_name="test-model", batch_size=10)


@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_init_replaces_existing(mock_tok_cls, mock_model_cls, tmp_path: Path) -> None:
    mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
    mock_model_cls.from_pretrained.return_value = _mock_model()

    from coderag.algo.embedding import init

    corpus = _write_corpus(tmp_path, _sample_records())
    db = tmp_path / "index.db"
    first = init(corpus, db, model_name="test-model", batch_size=10)
    # Reset mock call counts
    mock_model_cls.from_pretrained.return_value = _mock_model()
    second = init(corpus, db, model_name="test-model", batch_size=10)
    assert first == second == 3
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'coderag.algo.embedding'`

**Step 3: Write minimal init() implementation**

Create `src/coderag/algo/embedding.py`:

```python
"""Embedding search using BGE-Code-v1 + BGE-Reranker-v2-m3.

Provides two operations:
- ``init``: embed a JSONL corpus and store vectors in sqlite-vec
- ``EmbeddingAlgorithm``: two-stage search (KNN + rerank),
  implementing ``SearchAlgorithm``
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import sqlite_vec
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from coderag.interfaces import SearchResult

_INSTRUCTION = (
    "Given a textual explanation of code functionality, "
    "retrieve the corresponding code implementation."
)


def _last_token_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool using the last non-padding token (BGE-Code-v1 convention)."""
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device),
        sequence_lengths,
    ]


def _encode_batch(
    model: object,
    tokenizer: object,
    texts: list[str],
    max_length: int = 4096,
) -> np.ndarray:
    """Encode texts and return L2-normalized embeddings as numpy array.

    Shape: ``(len(texts), hidden_size)``.
    """
    inputs = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = _last_token_pool(
        outputs.last_hidden_state, inputs["attention_mask"]
    )
    normed = F.normalize(embeddings, p=2, dim=1)
    return normed.cpu().float().numpy()


def init(
    corpus_path: Path,
    db_path: Path,
    *,
    model_name: str,
    batch_size: int,
) -> int:
    """Embed all documents in a JSONL corpus and store in sqlite-vec.

    Each line in *corpus_path* must be a JSON object with keys
    ``func_id``, ``func_name``, ``code``, ``docstring``.

    The database contains two tables:
    - ``embedding_docs``: maps rowid → func_id (regular table)
    - ``embedding_vec``: stores vectors for KNN (vec0 virtual table)

    Returns the number of documents indexed.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    if torch.cuda.is_available():
        model = model.half().cuda()
    model.eval()

    embed_dim = model.config.hidden_size
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db = Path(tmpdir) / "index.db"
        conn = sqlite3.connect(tmp_db)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        conn.execute(
            "CREATE TABLE embedding_docs ("
            "  rowid INTEGER PRIMARY KEY,"
            "  func_id TEXT NOT NULL UNIQUE"
            ")"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE embedding_vec USING vec0("
            f"  embedding FLOAT[{embed_dim}] distance_metric=cosine"
            f")"
        )

        count = 0
        batch_texts: list[str] = []
        batch_ids: list[str] = []

        with open(corpus_path) as f:
            for line in f:
                record = json.loads(line)
                text = (
                    record["func_name"]
                    + "\n"
                    + record["code"]
                    + "\n"
                    + record["docstring"]
                )
                batch_texts.append(text)
                batch_ids.append(record["func_id"])

                if len(batch_texts) == batch_size:
                    _insert_batch(conn, model, tokenizer, batch_ids, batch_texts, count)
                    count += len(batch_texts)
                    batch_texts.clear()
                    batch_ids.clear()

        if batch_texts:
            _insert_batch(conn, model, tokenizer, batch_ids, batch_texts, count)
            count += len(batch_texts)

        conn.commit()
        conn.close()

        if db_path.exists():
            db_path.unlink()
        shutil.copy2(tmp_db, db_path)

    return count


def _insert_batch(
    conn: sqlite3.Connection,
    model: object,
    tokenizer: object,
    func_ids: list[str],
    texts: list[str],
    offset: int,
) -> None:
    """Encode and insert a batch of documents."""
    embeddings = _encode_batch(model, tokenizer, texts)
    for i, (func_id, emb) in enumerate(zip(func_ids, embeddings)):
        rowid = offset + i + 1  # 1-based
        conn.execute(
            "INSERT INTO embedding_docs (rowid, func_id) VALUES (?, ?)",
            (rowid, func_id),
        )
        conn.execute(
            "INSERT INTO embedding_vec (rowid, embedding) VALUES (?, ?)",
            (rowid, emb.astype(np.float32).tobytes()),
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding.py -v`
Expected: 4 tests pass

**Step 5: Run full suite**

Run: `uv run pytest tests/ -v`
Expected: 66 tests pass (62 existing + 4 new)

**Step 6: Commit**

```bash
git add src/coderag/algo/embedding.py tests/test_embedding.py
git commit -m "feat: add embedding init() for sqlite-vec index building"
```

---

### Task 4: EmbeddingAlgorithm — KNN search + reranker + score-gap gating

**Files:**
- Modify: `src/coderag/algo/embedding.py` (add EmbeddingAlgorithm class)
- Modify: `tests/test_embedding.py` (add search tests)

**Step 1: Write the failing tests**

Append to `tests/test_embedding.py`:

```python
# ---------------------------------------------------------------------------
# Helpers for search tests
# ---------------------------------------------------------------------------

def _build_test_dbs(tmp_path: Path) -> tuple[Path, Path]:
    """Build a BM25 index + embedding index with known data for search tests.

    Returns (embedding_db_path, bm25_db_path).
    """
    records = _sample_records()
    corpus = _write_corpus(tmp_path, records)

    # Build BM25 index (source of doc text for reranker)
    bm25_db = tmp_path / "bm25" / "index.db"
    bm25_init(corpus, bm25_db)

    # Build embedding index with known vectors
    embed_db = tmp_path / "embedding" / "index.db"
    conn = sqlite3.connect(embed_db.parent.mkdir(parents=True, exist_ok=True) or str(embed_db))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute(
        "CREATE TABLE embedding_docs ("
        "  rowid INTEGER PRIMARY KEY,"
        "  func_id TEXT NOT NULL UNIQUE"
        ")"
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE embedding_vec USING vec0("
        f"  embedding FLOAT[{EMBED_DIM}] distance_metric=cosine"
        f")"
    )

    # Insert vectors with known cosine similarities to a query vector [1,0,0,0]:
    # aaa111: [1, 0, 0, 0]  → cosine distance 0.0 (most similar)
    # bbb222: [0.7, 0.7, 0, 0] → cosine distance ~0.293
    # ccc333: [0, 0, 1, 0]  → cosine distance 1.0 (orthogonal)
    vectors = {
        "aaa111": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "bbb222": np.array([0.7, 0.7, 0.0, 0.0], dtype=np.float32),
        "ccc333": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    # Normalize
    for k in vectors:
        vectors[k] = vectors[k] / np.linalg.norm(vectors[k])

    for i, (func_id, vec) in enumerate(vectors.items(), start=1):
        conn.execute(
            "INSERT INTO embedding_docs (rowid, func_id) VALUES (?, ?)",
            (i, func_id),
        )
        conn.execute(
            "INSERT INTO embedding_vec (rowid, embedding) VALUES (?, ?)",
            (i, vec.tobytes()),
        )

    conn.commit()
    conn.close()

    return embed_db, bm25_db


def _mock_reranker() -> MagicMock:
    """Create a mock reranker model.

    Returns logits that REVERSE the KNN ordering:
    - For 3 candidates: logits [0.0, 2.0, 1.0] → after sigmoid: [0.5, 0.88, 0.73]
    - For 2 candidates: logits [0.0, 2.0] → after sigmoid: [0.5, 0.88]
    So the reranker prefers the second candidate over the first.
    """
    model = MagicMock()
    model.eval.return_value = model
    model.parameters.return_value = iter([torch.tensor([1.0])])  # CPU

    def forward(**kwargs):
        input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        # Assign ascending logits: last candidate gets highest score
        logits = torch.linspace(0.0, 2.0, batch_size).unsqueeze(1)
        result = MagicMock()
        result.logits = logits
        return result

    model.side_effect = forward
    return model


def _mock_reranker_tokenizer() -> MagicMock:
    """Mock tokenizer for the reranker (takes pairs)."""
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
# EmbeddingAlgorithm tests
# ---------------------------------------------------------------------------


@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_search_returns_results(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, tmp_path: Path
) -> None:
    embed_db, bm25_db = _build_test_dbs(tmp_path)

    # Embedder mock: returns query vector [1, 0, 0, 0]
    embedder = _mock_model()

    def embed_query(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        result = MagicMock()
        hidden = torch.zeros(batch_size, 10, EMBED_DIM)
        # Set last token to [1, 0, 0, 0]
        hidden[:, -1, 0] = 1.0
        result.last_hidden_state = hidden
        return result

    embedder.side_effect = embed_query

    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]
    mock_model_cls.from_pretrained.return_value = embedder
    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()

    from coderag.algo.embedding import EmbeddingAlgorithm

    algo = EmbeddingAlgorithm(
        db_path=embed_db,
        model_name="test-model",
        reranker_name="test-reranker",
        rerank_candidates=10,
        gap_threshold=10.0,  # Very high → no gating
        bm25_db_path=bm25_db,
    )

    results = algo.search("test query", top_k=3)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_search_reranker_reorders(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, tmp_path: Path
) -> None:
    """With no gating, the reranker should freely reorder results."""
    embed_db, bm25_db = _build_test_dbs(tmp_path)

    # Query vector [1,0,0,0] → KNN order: aaa111, bbb222, ccc333
    embedder = _mock_model()

    def embed_query(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        result = MagicMock()
        hidden = torch.zeros(batch_size, 10, EMBED_DIM)
        hidden[:, -1, 0] = 1.0
        result.last_hidden_state = hidden
        return result

    embedder.side_effect = embed_query

    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]
    mock_model_cls.from_pretrained.return_value = embedder
    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()

    from coderag.algo.embedding import EmbeddingAlgorithm

    algo = EmbeddingAlgorithm(
        db_path=embed_db,
        model_name="test-model",
        reranker_name="test-reranker",
        rerank_candidates=10,
        gap_threshold=10.0,  # Very high → no gating
        bm25_db_path=bm25_db,
    )

    results = algo.search("test query", top_k=3)
    # Reranker assigns ascending scores → last KNN candidate gets highest
    # So reranker reverses the KNN order: ccc333, bbb222, aaa111
    assert results[0].doc_id == "ccc333"


@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_search_gating_pins_top(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, tmp_path: Path
) -> None:
    """When score gap exceeds threshold, embedding #1 stays on top."""
    embed_db, bm25_db = _build_test_dbs(tmp_path)

    embedder = _mock_model()

    def embed_query(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        result = MagicMock()
        hidden = torch.zeros(batch_size, 10, EMBED_DIM)
        hidden[:, -1, 0] = 1.0
        result.last_hidden_state = hidden
        return result

    embedder.side_effect = embed_query

    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]
    mock_model_cls.from_pretrained.return_value = embedder
    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()

    from coderag.algo.embedding import EmbeddingAlgorithm

    algo = EmbeddingAlgorithm(
        db_path=embed_db,
        model_name="test-model",
        reranker_name="test-reranker",
        rerank_candidates=10,
        gap_threshold=0.0,  # Very low → always gate
        bm25_db_path=bm25_db,
    )

    results = algo.search("test query", top_k=3)
    # Despite reranker wanting to reorder, gating keeps aaa111 on top
    assert results[0].doc_id == "aaa111"


@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_search_empty_query(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, tmp_path: Path
) -> None:
    embed_db, bm25_db = _build_test_dbs(tmp_path)

    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]
    mock_model_cls.from_pretrained.return_value = _mock_model()
    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()

    from coderag.algo.embedding import EmbeddingAlgorithm

    algo = EmbeddingAlgorithm(
        db_path=embed_db,
        model_name="test-model",
        reranker_name="test-reranker",
        rerank_candidates=10,
        gap_threshold=0.1,
        bm25_db_path=bm25_db,
    )

    assert algo.search("", top_k=10) == []
    assert algo.search("   ", top_k=10) == []


@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_name_property(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, tmp_path: Path
) -> None:
    embed_db, bm25_db = _build_test_dbs(tmp_path)

    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]
    mock_model_cls.from_pretrained.return_value = _mock_model()
    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()

    from coderag.algo.embedding import EmbeddingAlgorithm

    algo = EmbeddingAlgorithm(
        db_path=embed_db,
        model_name="test-model",
        reranker_name="test-reranker",
        rerank_candidates=10,
        gap_threshold=0.1,
        bm25_db_path=bm25_db,
    )
    assert algo.name == "embedding"


@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_search_respects_top_k(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, tmp_path: Path
) -> None:
    embed_db, bm25_db = _build_test_dbs(tmp_path)

    embedder = _mock_model()

    def embed_query(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        result = MagicMock()
        hidden = torch.zeros(batch_size, 10, EMBED_DIM)
        hidden[:, -1, 0] = 1.0
        result.last_hidden_state = hidden
        return result

    embedder.side_effect = embed_query

    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]
    mock_model_cls.from_pretrained.return_value = embedder
    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()

    from coderag.algo.embedding import EmbeddingAlgorithm

    algo = EmbeddingAlgorithm(
        db_path=embed_db,
        model_name="test-model",
        reranker_name="test-reranker",
        rerank_candidates=10,
        gap_threshold=10.0,
        bm25_db_path=bm25_db,
    )

    results = algo.search("test query", top_k=1)
    assert len(results) == 1


@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_init_missing_db(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, tmp_path: Path
) -> None:
    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]
    mock_model_cls.from_pretrained.return_value = _mock_model()
    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()

    from coderag.algo.embedding import EmbeddingAlgorithm

    with pytest.raises(FileNotFoundError, match="Embedding index not found"):
        EmbeddingAlgorithm(
            db_path=tmp_path / "nope.db",
            model_name="test-model",
            reranker_name="test-reranker",
            rerank_candidates=10,
            gap_threshold=0.1,
            bm25_db_path=tmp_path / "also-nope.db",
        )


@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_init_missing_bm25_db(
    mock_tok_cls, mock_model_cls, mock_reranker_cls, tmp_path: Path
) -> None:
    embed_db, _ = _build_test_dbs(tmp_path)

    mock_tok_cls.from_pretrained.side_effect = [_mock_tokenizer(), _mock_reranker_tokenizer()]
    mock_model_cls.from_pretrained.return_value = _mock_model()
    mock_reranker_cls.from_pretrained.return_value = _mock_reranker()

    from coderag.algo.embedding import EmbeddingAlgorithm

    with pytest.raises(FileNotFoundError, match="BM25 index not found"):
        EmbeddingAlgorithm(
            db_path=embed_db,
            model_name="test-model",
            reranker_name="test-reranker",
            rerank_candidates=10,
            gap_threshold=0.1,
            bm25_db_path=tmp_path / "nope.db",
        )
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedding.py::test_search_returns_results -v`
Expected: FAIL — `cannot import name 'EmbeddingAlgorithm'`

**Step 3: Implement EmbeddingAlgorithm**

Add to `src/coderag/algo/embedding.py` (after the `init` function, add the import and class):

Add to imports at the top:

```python
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
```

Add the class:

```python
_BM25_TABLE = "code_search"


class EmbeddingAlgorithm:
    """Two-stage search: embedding KNN + cross-encoder reranker.

    Implements the :class:`~coderag.interfaces.SearchAlgorithm` protocol.

    Requires a pre-built embedding index (from :func:`init`) and a
    pre-built BM25 index (from :func:`coderag.algo.bm25.init`) for
    document text lookup during reranking.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        model_name: str,
        reranker_name: str,
        rerank_candidates: int,
        gap_threshold: float,
        bm25_db_path: Path,
    ) -> None:
        if not db_path.exists():
            raise FileNotFoundError(
                f"Embedding index not found at {db_path}"
            )
        if not bm25_db_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {bm25_db_path}"
            )

        # Load embedding index (read-only, extension needed for vec0)
        self._conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True
        )
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # Load BM25 db for text lookups
        bm25_source = sqlite3.connect(
            f"file:{bm25_db_path}?mode=ro", uri=True
        )
        self._bm25_conn = sqlite3.connect(":memory:")
        bm25_source.backup(self._bm25_conn)
        bm25_source.close()

        # Load embedder
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        )
        if torch.cuda.is_available():
            self._model = self._model.half().cuda()
        self._model.eval()

        # Load reranker
        self._reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
        self._reranker = AutoModelForSequenceClassification.from_pretrained(
            reranker_name
        )
        if torch.cuda.is_available():
            self._reranker = self._reranker.half().cuda()
        self._reranker.eval()

        self._rerank_candidates = rerank_candidates
        self._gap_threshold = gap_threshold

    @property
    def name(self) -> str:
        return "embedding"

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return up to *top_k* results via KNN retrieval + reranking."""
        if not query.strip():
            return []

        # 1. Embed query (with instruction prefix)
        query_text = f"<instruct>{_INSTRUCTION}\n<query>{query}"
        query_vec = _encode_batch(
            self._model, self._tokenizer, [query_text]
        )[0]

        # 2. KNN retrieval
        candidates = self._knn_search(query_vec, self._rerank_candidates)
        if not candidates:
            return []

        # 3. Score-gap check
        pin_top = False
        if len(candidates) >= 2:
            gap = candidates[0][1] - candidates[1][1]
            pin_top = gap >= self._gap_threshold

        pinned_id = candidates[0][0] if pin_top else None

        # 4. Rerank
        func_ids = [c[0] for c in candidates]
        doc_texts = self._fetch_doc_texts(func_ids)
        reranked = self._rerank(query, candidates, doc_texts)

        # 5. Apply gating — pin original #1 to top position
        if pinned_id is not None:
            reranked = [
                (fid, s) for fid, s in reranked if fid != pinned_id
            ]
            pinned_score = (reranked[0][1] + 1.0) if reranked else 1.0
            reranked.insert(0, (pinned_id, pinned_score))

        # 6. Return top_k
        return [
            SearchResult(doc_id=fid, score=score)
            for fid, score in reranked[:top_k]
        ]

    def _knn_search(
        self, query_vec: np.ndarray, k: int
    ) -> list[tuple[str, float]]:
        """Return up to *k* nearest neighbors as (func_id, cosine_similarity)."""
        cur = self._conn.execute(
            "WITH knn AS ("
            "  SELECT rowid, distance"
            "  FROM embedding_vec"
            "  WHERE embedding MATCH ?"
            "  AND k = ?"
            ")"
            "SELECT d.func_id, (1.0 - knn.distance) AS similarity "
            "FROM knn "
            "JOIN embedding_docs d ON d.rowid = knn.rowid "
            "ORDER BY knn.distance",
            (query_vec.astype(np.float32).tobytes(), k),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]

    def _fetch_doc_texts(self, func_ids: list[str]) -> dict[str, str]:
        """Fetch concatenated document text from the BM25 index."""
        placeholders = ",".join("?" * len(func_ids))
        cur = self._bm25_conn.execute(
            f"SELECT func_id, func_name, code, docstring "
            f"FROM {_BM25_TABLE} "
            f"WHERE func_id IN ({placeholders})",
            func_ids,
        )
        return {
            row[0]: f"{row[1]}\n{row[2]}\n{row[3]}"
            for row in cur.fetchall()
        }

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        doc_texts: dict[str, str],
    ) -> list[tuple[str, float]]:
        """Rerank candidates using the cross-encoder reranker.

        Returns ``(func_id, sigmoid_score)`` sorted descending.
        """
        pairs = []
        valid_ids = []
        for func_id, _ in candidates:
            if func_id in doc_texts:
                pairs.append([query, doc_texts[func_id]])
                valid_ids.append(func_id)

        if not pairs:
            return [(fid, score) for fid, score in candidates]

        inputs = self._reranker_tokenizer(
            pairs,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        device = next(self._reranker.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._reranker(**inputs).logits.view(-1).float()

        scores = torch.sigmoid(logits).cpu().tolist()

        scored = list(zip(valid_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedding.py -v`
Expected: All 12 tests pass

**Step 5: Run full suite**

Run: `uv run pytest tests/ -v`
Expected: 74 tests pass (62 existing + 12 new)

**Step 6: Commit**

```bash
git add src/coderag/algo/embedding.py tests/test_embedding.py
git commit -m "feat: add EmbeddingAlgorithm with KNN search, reranker, and score-gap gating"
```

---

### Task 5: CLI integration

**Files:**
- Modify: `src/eval/cli.py:85-103,133-159,238-244`
- Modify: `tests/test_cli.py` (add embedding CLI tests)

**Step 1: Write failing CLI tests**

Append to `tests/test_cli.py`:

```python
@patch("coderag.algo.embedding.AutoModelForSequenceClassification")
@patch("coderag.algo.embedding.AutoModel")
@patch("coderag.algo.embedding.AutoTokenizer")
def test_cli_eval_index_embedding(
    mock_tok_cls,
    mock_model_cls,
    mock_reranker_cls,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    corpus = _write_corpus(tmp_path)
    monkeypatch.setenv("CODERAG_DATA_DIR", str(data_dir))

    # Mock the transformer models
    from tests.test_embedding import _mock_model, _mock_tokenizer
    mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
    mock_model_cls.from_pretrained.return_value = _mock_model()

    main([
        "eval", "index",
        "--algorithm", "embedding",
        "--corpus", str(corpus),
        "--model-name", "test-model",
        "--batch-size", "10",
    ])
    captured = capsys.readouterr()
    assert "2" in captured.out
    assert (data_dir / "algorithms" / "embedding" / "index.db").exists()


def test_cli_eval_index_unknown_algorithm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _write_corpus(tmp_path)
    monkeypatch.setenv("CODERAG_DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        main(["eval", "index", "--algorithm", "nonexistent", "--corpus", str(corpus)])
    assert "unknown algorithm" in str(exc.value).lower()


def test_cli_eval_run_embedding_no_index(
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
    with pytest.raises(SystemExit) as exc:
        main([
            "eval", "run", "embedding",
            "--suite", "test_suite",
            "--top-k", "10",
            "--metrics", "ndcg@10",
            "--workers", "1",
            "--reranker-name", "test-reranker",
            "--rerank-candidates", "50",
            "--gap-threshold", "0.1",
            "--bm25-db-path", str(data_dir / "algorithms" / "bm25" / "index.db"),
        ])
    assert "Embedding index not found" in str(exc.value)
```

Also add this import at the top of `tests/test_cli.py`:

```python
from unittest.mock import patch
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py::test_cli_eval_index_embedding -v`
Expected: FAIL — argparse error (flags not defined yet)

**Step 3: Implement CLI changes**

Modify `src/eval/cli.py`:

In `_eval_index` (line 85), add the embedding branch after the bm25 branch:

```python
    elif args.algorithm == "embedding":
        from coderag.algo.embedding import init as embedding_init

        db_path = data_dir / "algorithms" / "embedding" / "index.db"
        print(f"Building embedding index from {corpus_path} …")
        count = embedding_init(
            corpus_path, db_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
        )
        print(f"  → {db_path}")
        print(f"  → {count:,} documents indexed")
        print("Done.")
```

In `_resolve_algorithm` (line 133), add the embedding branch:

```python
    if name == "embedding":
        from coderag.algo.embedding import EmbeddingAlgorithm

        db_path = data_dir / "algorithms" / "embedding" / "index.db"
        if not db_path.exists():
            sys.exit(
                f"Error: Embedding index not found at {db_path}\n"
                "Build it first with: coderag eval index --algorithm embedding --corpus <path>"
                " --model-name <model> --batch-size <n>"
            )
        # These args are only present on the run subparser
        bm25_db_path = Path(args.bm25_db_path)
        if not bm25_db_path.exists():
            sys.exit(f"Error: BM25 index not found at {bm25_db_path}")
        return EmbeddingAlgorithm(
            db_path,
            model_name=args.model_name,
            reranker_name=args.reranker_name,
            rerank_candidates=args.rerank_candidates,
            gap_threshold=args.gap_threshold,
            bm25_db_path=bm25_db_path,
        )
```

Note: `_resolve_algorithm` signature needs to accept `args` instead of just `name` and `data_dir`. Update the signature:

```python
def _resolve_algorithm(name: str, data_dir: Path, args: argparse.Namespace) -> object:
```

And update the call in `_eval_run` (line 119):

```python
    algo = _resolve_algorithm(args.algorithm, data_dir, args)
```

In `_build_parser`, update the `index` subparser to add embedding-specific flags:

```python
    # eval index --algorithm <name> --corpus <path>
    index_parser = eval_sub.add_parser("index", help="Build search index from corpus")
    index_parser.add_argument("--algorithm", required=True, help="Algorithm name (e.g. bm25, embedding)")
    index_parser.add_argument("--corpus", required=True, help="Path to JSONL corpus file")
    index_parser.add_argument("--model-name", help="Model name for embedding algorithm")
    index_parser.add_argument("--batch-size", type=int, help="Batch size for embedding algorithm")
```

Update the `run` subparser to add embedding-specific flags:

```python
    run_parser.add_argument("--model-name", help="Embedder model name (embedding algo)")
    run_parser.add_argument("--reranker-name", help="Reranker model name (embedding algo)")
    run_parser.add_argument("--rerank-candidates", type=int, help="Number of KNN candidates to rerank (embedding algo)")
    run_parser.add_argument("--gap-threshold", type=float, help="Score gap threshold for gating (embedding algo)")
    run_parser.add_argument("--bm25-db-path", help="Path to BM25 index for text lookups (embedding algo)")
```

Add validation in `_eval_index` for required embedding args:

```python
    if args.algorithm == "embedding":
        if not args.model_name:
            sys.exit("Error: --model-name is required for embedding algorithm")
        if not args.batch_size:
            sys.exit("Error: --batch-size is required for embedding algorithm")
```

Add validation in `_eval_run` for required embedding args (before `_resolve_algorithm`):

```python
    if args.algorithm == "embedding":
        for flag in ("model_name", "reranker_name", "rerank_candidates", "gap_threshold", "bm25_db_path"):
            if getattr(args, flag, None) is None:
                pretty = flag.replace("_", "-")
                sys.exit(f"Error: --{pretty} is required for embedding algorithm")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All CLI tests pass

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/eval/cli.py tests/test_cli.py
git commit -m "feat: add embedding algorithm CLI support (index + run)"
```

---

### Task 6: Final verification and cleanup

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests pass, no warnings

**Step 2: Check imports are clean**

Run: `uv run python -c "from coderag.algo.embedding import init, EmbeddingAlgorithm; print('imports ok')"`
Expected: `imports ok`

**Step 3: Verify CLI help**

Run: `uv run coderag eval index --help`
Expected: shows --algorithm, --corpus, --model-name, --batch-size flags

Run: `uv run coderag eval run --help`
Expected: shows --reranker-name, --rerank-candidates, --gap-threshold, --bm25-db-path flags

**Step 4: Update scratchpad**

Write session summary to `.aisteering/scratchpads/2026-02-10-embedding-algorithm.md`.

**Step 5: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: embedding algorithm cleanup and scratchpad update"
```
