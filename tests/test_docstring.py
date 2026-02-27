from __future__ import annotations

import json
from pathlib import Path

from csn.docstring import _first_paragraph, build_docstring_suite
from csn.download import func_id

FIXTURES = Path(__file__).parent / "fixtures"


def _make_corpus(tmp_path: Path) -> Path:
    """Create a small corpus JSONL from the fixture records."""
    records = []
    with open(FIXTURES / "sample_csn_records.jsonl") as f:
        for line in f:
            raw = json.loads(line)
            records.append(
                {
                    "func_id": func_id(raw["url"]),
                    "url": raw["url"],
                    "repo": raw["repo"],
                    "path": raw["path"],
                    "func_name": raw["func_name"],
                    "code": raw["code"],
                    "docstring": raw.get("docstring", ""),
                }
            )
    corpus_path = tmp_path / "python.jsonl"
    with open(corpus_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return corpus_path


def test_first_paragraph_single_line() -> None:
    assert _first_paragraph("Do foo things.") == "Do foo things."


def test_first_paragraph_multi_paragraph() -> None:
    doc = "Return x unchanged.\n\nThis is the second paragraph."
    assert _first_paragraph(doc) == "Return x unchanged."


def test_first_paragraph_with_args_block() -> None:
    doc = "Print a greeting.\n\nArgs:\n    name: the name to greet"
    assert _first_paragraph(doc) == "Print a greeting."


def test_first_paragraph_multiline_first() -> None:
    doc = "Line one\nline two.\n\nSecond paragraph."
    assert _first_paragraph(doc) == "Line one line two."


def test_build_docstring_suite(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    suite = build_docstring_suite(corpus, sample_count=100)
    assert suite.name == "csn_docstring"
    # From 5 fixture records: 4 have docstrings, 1 is empty
    # "Do foo things." has only 3 tokens — just at the threshold
    # "Add two numbers together." — 4 tokens
    # "Return x unchanged." — 3 tokens
    # "Print a greeting." — 3 tokens
    assert len(suite.queries) == 4


def test_docstring_suite_binary_relevance(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    suite = build_docstring_suite(corpus, sample_count=100)
    for qid, docs in suite.qrels.items():
        for fid, rel in docs.items():
            assert rel == 1, "Docstring suite must use binary relevance"
            assert fid == qid, "Each query's doc should be its own func_id"


def test_docstring_suite_subsample(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    suite = build_docstring_suite(corpus, sample_count=2)
    assert len(suite.queries) == 2


def test_docstring_suite_deterministic(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    a = build_docstring_suite(corpus, sample_count=3)
    b = build_docstring_suite(corpus, sample_count=3)
    assert a.queries == b.queries
    assert a.qrels == b.qrels


def test_docstring_suite_skips_empty(tmp_path: Path) -> None:
    """Functions with empty docstrings should be excluded."""
    corpus = _make_corpus(tmp_path)
    suite = build_docstring_suite(corpus, sample_count=100)
    # "baz" has empty docstring
    baz_url = "https://github.com/owner/repo2/blob/def456/lib/utils.py#L1-L2"
    baz_fid = func_id(baz_url)
    assert baz_fid not in suite.queries
