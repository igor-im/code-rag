"""BM25 search using PostgreSQL full-text search (tsvector/tsquery).

Provides two operations:
- ``init``: build a full-text index in PostgreSQL from a JSONL corpus
- ``Bm25Algorithm``: search the index, implementing ``SearchAlgorithm``
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg

from coderag.interfaces import SearchResult

_TABLE = "bm25_docs"
_FTS_CONFIG = "english"


def _get_pg_conn(pg_url: str) -> psycopg.Connection:
    """Open a psycopg connection."""
    return psycopg.connect(pg_url, autocommit=False)


def init(corpus_path: Path, pg_url: str) -> int:
    """Build a full-text search index in PostgreSQL from a JSONL corpus.

    Each line in *corpus_path* must be a JSON object with keys
    ``func_id``, ``func_name``, ``code``, ``docstring`` (as produced
    by :func:`csn.download.extract_corpus`).

    Creates the ``bm25_docs`` table with a GIN tsvector index.
    Drops and recreates the table if it already exists.

    Returns the number of documents indexed.

    Raises:
        FileNotFoundError: If *corpus_path* does not exist.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    rows: list[tuple[str, str, str, str]] = []
    with open(corpus_path) as f:
        for line in f:
            record = json.loads(line)
            rows.append((
                record["func_id"],
                record["func_name"],
                record["code"],
                record.get("docstring", ""),
            ))

    conn = _get_pg_conn(pg_url)
    try:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {_TABLE}")
        cur.execute(f"""
            CREATE TABLE {_TABLE} (
                func_id   TEXT PRIMARY KEY,
                func_name TEXT,
                code      TEXT,
                docstring TEXT,
                tsv       tsvector GENERATED ALWAYS AS (
                    to_tsvector('{_FTS_CONFIG}',
                        coalesce(func_name, '') || ' ' ||
                        coalesce(code, '')      || ' ' ||
                        coalesce(docstring, ''))
                ) STORED
            )
        """)
        cur.executemany(
            f"INSERT INTO {_TABLE} (func_id, func_name, code, docstring) "
            "VALUES (%s, %s, %s, %s)",
            rows,
        )
        conn.commit()
        cur.execute(
            f"CREATE INDEX bm25_docs_tsv_idx ON {_TABLE} USING gin(tsv)"
        )
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise

    conn.close()
    return len(rows)


class Bm25Algorithm:
    """BM25-style full-text search via PostgreSQL tsvector/tsquery.

    Implements the :class:`~coderag.interfaces.SearchAlgorithm` protocol.
    """

    def __init__(self, pg_url: str) -> None:
        self._pg_url = pg_url
        self._conn = _get_pg_conn(pg_url)
        cur = self._conn.cursor()
        cur.execute(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables "
            "  WHERE table_name = %s"
            ")",
            (_TABLE,),
        )
        if not cur.fetchone()[0]:
            self._conn.close()
            raise FileNotFoundError(
                f"BM25 index (table '{_TABLE}') not found in database"
            )

    @property
    def name(self) -> str:
        return "bm25"

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return up to *top_k* results ranked by ts_rank_cd."""
        if not query.strip():
            return []

        cur = self._conn.cursor()
        cur.execute(
            f"SELECT func_id, ts_rank_cd(tsv, plainto_tsquery(%s, %s)) AS score "
            f"FROM {_TABLE} "
            f"WHERE tsv @@ plainto_tsquery(%s, %s) "
            f"ORDER BY score DESC LIMIT %s",
            (_FTS_CONFIG, query, _FTS_CONFIG, query, top_k),
        )
        return [
            SearchResult(doc_id=row[0], score=float(row[1]))
            for row in cur.fetchall()
        ]

    def search_batch(
        self,
        queries: list[tuple[str, str]],
        top_k: int,
        n_workers: int,
    ) -> dict[str, dict[str, float]]:
        """Run *queries* in parallel using a thread pool.

        Each worker opens its own PostgreSQL connection.

        Returns:
            ``{qid: {doc_id: score}}`` for queries that had results.
        """
        pg_url = self._pg_url

        def _search_one(item: tuple[str, str]) -> tuple[str, dict[str, float]]:
            qid, text = item
            if not text.strip():
                return qid, {}
            conn = _get_pg_conn(pg_url)
            try:
                cur = conn.cursor()
                cur.execute(
                    f"SELECT func_id, ts_rank_cd(tsv, plainto_tsquery(%s, %s)) AS score "
                    f"FROM {_TABLE} "
                    f"WHERE tsv @@ plainto_tsquery(%s, %s) "
                    f"ORDER BY score DESC LIMIT %s",
                    (_FTS_CONFIG, text, _FTS_CONFIG, text, top_k),
                )
                rows = cur.fetchall()
            finally:
                conn.close()
            return qid, {row[0]: float(row[1]) for row in rows}

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_search_one, queries))

        return {qid: hits for qid, hits in results if hits}
