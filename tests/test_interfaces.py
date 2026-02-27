from __future__ import annotations

from coderag.interfaces import SearchAlgorithm, SearchResult


class FakeAlgorithm:
    """Minimal implementation of the SearchAlgorithm protocol."""

    @property
    def name(self) -> str:
        return "fake"

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return [SearchResult(doc_id="d1", score=1.0)]


def test_search_result_fields() -> None:
    r = SearchResult(doc_id="abc", score=0.5)
    assert r.doc_id == "abc"
    assert r.score == 0.5


def test_search_result_is_frozen() -> None:
    r = SearchResult(doc_id="abc", score=0.5)
    try:
        r.doc_id = "xyz"  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except AttributeError:
        pass


def test_search_algorithm_protocol_conformance() -> None:
    algo: SearchAlgorithm = FakeAlgorithm()
    assert algo.name == "fake"
    results = algo.search("test query", top_k=5)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)


def test_search_result_equality() -> None:
    a = SearchResult(doc_id="x", score=1.0)
    b = SearchResult(doc_id="x", score=1.0)
    assert a == b


def test_search_result_inequality() -> None:
    a = SearchResult(doc_id="x", score=1.0)
    b = SearchResult(doc_id="y", score=1.0)
    assert a != b
