"""Hybrid search via Reciprocal Rank Fusion (RRF).

Provides:
- ``rrf_combine``: generic RRF over any number of ranked lists
- ``HybridAlgorithm``: BM25 + embedding fusion, implementing ``SearchAlgorithm``
"""

from __future__ import annotations

from coderag.interfaces import SearchAlgorithm, SearchResult


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


class HybridAlgorithm:
    """BM25 + embedding fusion via Reciprocal Rank Fusion.

    Implements the :class:`~coderag.interfaces.SearchAlgorithm` protocol.

    Calls each sub-algorithm with its own retrieval depth, then
    merges the results using :func:`rrf_combine`.
    """

    def __init__(
        self,
        bm25: SearchAlgorithm,
        embedding: SearchAlgorithm,
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
