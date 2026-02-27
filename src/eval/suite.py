from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ranx


@dataclass
class EvalSuite:
    """Evaluation suite: queries paired with golden relevance judgments."""

    name: str
    queries: dict[str, str]  # {qid: query_text}
    qrels: dict[str, dict[str, int]]  # {qid: {doc_id: relevance}}

    def save(self, path: Path) -> None:
        """Serialize suite to queries.json + qrels.json in *path*."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "queries.json", "w") as f:
            json.dump(
                {"name": self.name, "queries": self.queries},
                f,
                indent=2,
                sort_keys=True,
            )
        with open(path / "qrels.json", "w") as f:
            json.dump(self.qrels, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> EvalSuite:
        """Deserialize suite from queries.json + qrels.json in *path*."""
        with open(path / "queries.json") as f:
            q_data = json.load(f)
        with open(path / "qrels.json") as f:
            qrels = json.load(f)
        return cls(
            name=q_data["name"],
            queries=q_data["queries"],
            qrels=qrels,
        )

    def to_ranx_qrels(self) -> ranx.Qrels:
        """Convert to ranx Qrels for evaluation."""
        return ranx.Qrels(self.qrels)
