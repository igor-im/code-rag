"""Build a docstring-based evaluation suite from the CSN corpus.

Each function with a non-empty docstring becomes a query (first paragraph)
paired with binary relevance (1) for that function.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.suite import EvalSuite


def _first_paragraph(docstring: str) -> str:
    """Extract the first paragraph from a docstring.

    Paragraphs are separated by blank lines.  Returns the stripped
    first paragraph, or the whole stripped docstring if there are
    no blank-line separators.
    """
    lines: list[str] = []
    for line in docstring.strip().splitlines():
        if not line.strip():
            break
        lines.append(line.strip())
    return " ".join(lines)


def build_docstring_suite(
    corpus_path: Path,
    sample_count: int,
) -> EvalSuite:
    """Build binary-relevance suite from corpus docstrings.

    *corpus_path* must be a JSONL file produced by ``extract_corpus``.
    *sample_count* controls how many functions to include (deterministic
    subsample via sorted func_id).
    """
    candidates: list[tuple[str, str]] = []  # (func_id, first_paragraph)

    with open(corpus_path) as f:
        for line in f:
            record = json.loads(line)
            doc = record.get("docstring", "").strip()
            if not doc:
                continue
            para = _first_paragraph(doc)
            if len(para.split()) < 3:
                continue
            candidates.append((record["func_id"], para))

    # Deterministic subsample: sort by func_id (hex hash), take first N
    candidates.sort(key=lambda x: x[0])
    selected = candidates[:sample_count]

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    for fid, para in selected:
        qid = fid  # use func_id as query ID (unique per function)
        queries[qid] = para
        qrels[qid] = {fid: 1}

    return EvalSuite(name="csn_docstring", queries=queries, qrels=qrels)
