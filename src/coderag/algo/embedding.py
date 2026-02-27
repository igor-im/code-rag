"""Embedding search using BGE-Code-v1 + BGE-Reranker-v2-m3.

Provides two operations:
- ``init``: embed a JSONL corpus and store vectors in pgvector
- ``EmbeddingAlgorithm``: two-stage search (KNN + rerank),
  implementing ``SearchAlgorithm``
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import psycopg
import torch
import torch.nn.functional as F
from pgvector.psycopg import register_vector
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

from coderag.interfaces import SearchResult

_BM25_TABLE = "bm25_docs"

_INSTRUCTION = (
    "Given a textual explanation of code functionality, "
    "retrieve the corresponding code implementation."
)


def _get_pg_conn(pg_url: str) -> psycopg.Connection:
    """Open a psycopg connection with pgvector type registration."""
    conn = psycopg.connect(pg_url, autocommit=False)
    register_vector(conn)
    return conn


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
    with torch.inference_mode():
        outputs = model(**inputs)
        embeddings = _last_token_pool(
            outputs.last_hidden_state, inputs["attention_mask"]
        )
        normed = F.normalize(embeddings, p=2, dim=1)
    return normed.cpu().float().numpy()


def init(
    corpus_path: Path,
    pg_url: str,
    *,
    model_name: str,
    batch_size: int,
    max_length: int,
) -> int:
    """Embed all documents in a JSONL corpus and store in pgvector.

    Each line in *corpus_path* must be a JSON object with keys
    ``func_id``, ``func_name``, ``code``, ``docstring``.

    The table ``embedding_docs`` stores (func_id, embedding vector).
    An HNSW index is created for fast cosine-similarity KNN.

    Returns the number of documents indexed.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=False, attn_implementation="sdpa",
    )
    embed_dim = model.config.hidden_size
    if torch.cuda.is_available():
        model = model.half().cuda()
    else:
        model = model.to(dtype=torch.bfloat16)
    model.eval()

    # Read all records upfront so we can sort by length to minimise padding.
    logger.info("Reading corpus …")
    records: list[tuple[str, str]] = []  # (func_id, text)
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
            records.append((record["func_id"], text))
    logger.info("Loaded %d records, sorting by length …", len(records))
    records.sort(key=lambda r: len(r[1]))

    total = len(records)
    num_batches = (total + batch_size - 1) // batch_size

    conn = _get_pg_conn(pg_url)
    cur = conn.cursor()

    # Recreate table
    cur.execute("DROP TABLE IF EXISTS embedding_docs")
    cur.execute(
        f"CREATE TABLE embedding_docs ("
        f"  func_id TEXT PRIMARY KEY,"
        f"  embedding vector({embed_dim})"
        f")"
    )
    conn.commit()

    count = 0
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_ids = [records[i][0] for i in range(start, end)]
        batch_texts = [records[i][1] for i in range(start, end)]

        embeddings = _encode_batch(model, tokenizer, batch_texts, max_length=max_length)
        rows = [
            (fid, emb.tolist())
            for fid, emb in zip(batch_ids, embeddings)
        ]
        cur.executemany(
            "INSERT INTO embedding_docs (func_id, embedding) VALUES (%s, %s)",
            rows,
        )
        conn.commit()
        count += len(batch_texts)

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            pct = count * 100 // total
            print(
                f"\r  [{count:>7d}/{total}] {pct:3d}%  "
                f"(batch {batch_idx + 1}/{num_batches})",
                end="",
                flush=True,
                file=sys.stderr,
            )

    print(file=sys.stderr)  # newline after progress

    # Build HNSW index for fast KNN
    cur.execute(
        "CREATE INDEX embedding_docs_hnsw_idx ON embedding_docs "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 200)"
    )
    conn.commit()
    conn.close()

    return count


class EmbeddingAlgorithm:
    """Two-stage search: KNN embedding retrieval + cross-encoder reranking.

    Implements the :class:`~coderag.interfaces.SearchAlgorithm` protocol.

    Pipeline:
    1. Embed query with BGE-Code instruction prefix
    2. KNN retrieval for ``rerank_candidates`` nearest neighbours
    3. Score-gap gating: if gap between #1 and #2 cosine similarity
       exceeds ``gap_threshold``, pin the top result
    4. Rerank all candidates with cross-encoder
    5. If pinned: force original #1 back to position 0
    6. Return top_k results
    """

    def __init__(
        self,
        pg_url: str,
        *,
        model_name: str,
        reranker_name: str,
        rerank_candidates: int,
        gap_threshold: float,
        reranker_max_length: int,
        ef_search: int,
    ) -> None:
        self._rerank_candidates = rerank_candidates
        self._gap_threshold = gap_threshold
        self._reranker_max_length = reranker_max_length

        # --- Embedding model ---
        print("[init] loading embedding tokenizer…", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=False
        )
        print("[init] loading embedding model…", flush=True)
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=False,
            attn_implementation="sdpa",
        )
        if torch.cuda.is_available():
            print("[init] moving embedding model to GPU…", flush=True)
            self._model = self._model.half().cuda()
        self._model.eval()
        print("[init] embedding model ready", flush=True)

        # --- Reranker model ---
        print("[init] loading reranker tokenizer…", flush=True)
        self._reranker_tokenizer = AutoTokenizer.from_pretrained(
            reranker_name, trust_remote_code=False
        )
        print("[init] loading reranker model…", flush=True)
        self._reranker = AutoModelForSequenceClassification.from_pretrained(
            reranker_name, trust_remote_code=False,
            attn_implementation="sdpa",
        )
        if torch.cuda.is_available():
            print("[init] moving reranker to GPU…", flush=True)
            self._reranker = self._reranker.half().cuda()
        self._reranker.eval()
        print("[init] reranker ready", flush=True)

        # --- pgvector connection ---
        print("[init] connecting to pgvector…", flush=True)
        self._pg_conn = _get_pg_conn(pg_url)
        # Validate table exists
        cur = self._pg_conn.cursor()
        cur.execute(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables "
            "  WHERE table_name = 'embedding_docs'"
            ")"
        )
        exists = cur.fetchone()[0]
        if not exists:
            raise FileNotFoundError(
                "Embedding table 'embedding_docs' not found in database"
            )
        # Set HNSW search breadth for the session
        cur.execute(f"SET hnsw.ef_search = {ef_search}")
        print("[init] pgvector index loaded", flush=True)

    @property
    def name(self) -> str:
        return "embedding"

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return up to *top_k* results using KNN + rerank pipeline."""
        if not query.strip():
            return []

        # 1. Embed query with instruction prefix
        query_text = f"<instruct>{_INSTRUCTION}\n<query>{query}"
        query_vec = _encode_batch(
            self._model, self._tokenizer, [query_text]
        )[0]

        # 2. KNN retrieval
        candidates = self._knn_search(query_vec, self._rerank_candidates)
        if not candidates:
            return []

        # 3. Score-gap check
        pinned_id: str | None = None
        if len(candidates) >= 2:
            top_sim = candidates[0][1]
            second_sim = candidates[1][1]
            gap = top_sim - second_sim
            if gap >= self._gap_threshold:
                pinned_id = candidates[0][0]

        # 4. Fetch doc texts and rerank
        func_ids = [fid for fid, _ in candidates]
        doc_texts = self._fetch_doc_texts(func_ids)
        reranked = self._rerank(query, candidates, doc_texts)

        # 5. If pinned: force original #1 back to position 0
        if pinned_id is not None:
            max_score = reranked[0][1] if reranked else 0.0
            reranked = [
                (fid, score) for fid, score in reranked if fid != pinned_id
            ]
            reranked.insert(0, (pinned_id, max_score + 1.0))

        # 6. Return top_k
        return [
            SearchResult(doc_id=fid, score=score)
            for fid, score in reranked[:top_k]
        ]

    def _knn_search(
        self, query_vec: np.ndarray, k: int
    ) -> list[tuple[str, float]]:
        """pgvector KNN returning (func_id, cosine_similarity)."""
        cur = self._pg_conn.cursor()
        cur.execute(
            "SELECT func_id, 1 - (embedding <=> %s) AS similarity "
            "FROM embedding_docs ORDER BY embedding <=> %s LIMIT %s",
            (query_vec, query_vec, k),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]

    def _fetch_doc_texts(self, func_ids: list[str]) -> dict[str, str]:
        """Read document text from the bm25_docs table."""
        placeholders = ", ".join("%s" for _ in func_ids)
        cur = self._pg_conn.cursor()
        cur.execute(
            f"SELECT func_id, func_name, code, docstring "
            f"FROM {_BM25_TABLE} "
            f"WHERE func_id IN ({placeholders})",
            func_ids,
        )
        result: dict[str, str] = {}
        for row in cur.fetchall():
            func_id, func_name, code, docstring = row
            result[func_id] = f"{func_name}\n{code}\n{docstring}"
        return result

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        doc_texts: dict[str, str],
    ) -> list[tuple[str, float]]:
        """Cross-encoder reranking with sigmoid normalization."""
        pairs = [
            [query, doc_texts.get(fid, "")]
            for fid, _ in candidates
        ]
        func_ids = [fid for fid, _ in candidates]

        inputs = self._reranker_tokenizer(
            pairs,
            max_length=self._reranker_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        device = next(self._reranker.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._reranker(**inputs)

        logits = outputs.logits.squeeze(-1)
        scores = torch.sigmoid(logits).cpu().float().tolist()

        scored = list(zip(func_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
