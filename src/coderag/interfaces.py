from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result returned by an algorithm."""

    doc_id: str
    score: float


class SearchAlgorithm(Protocol):
    """Interface every search algorithm must implement."""

    @property
    def name(self) -> str:
        """Algorithm identifier (used in run output filenames)."""
        ...

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return top_k results ranked by relevance, highest score first."""
        ...
