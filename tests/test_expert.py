from __future__ import annotations

from pathlib import Path

from csn.download import func_id
from csn.expert import build_expert_suite

FIXTURES = Path(__file__).parent / "fixtures"


def test_build_expert_suite_shape() -> None:
    suite = build_expert_suite(
        FIXTURES / "sample_annotations.csv",
        FIXTURES / "sample_queries.csv",
    )
    assert suite.name == "csn_expert"
    # Only Python rows, and only queries present in queries.csv
    # "sort a list" has 3 Python annotations, "read a file" has 1
    assert "sort a list" in suite.queries
    assert "read a file" in suite.queries
    # "convert int to string" is in queries.csv but has no annotations
    assert "convert int to string" not in suite.queries


def test_expert_suite_relevance_range() -> None:
    suite = build_expert_suite(
        FIXTURES / "sample_annotations.csv",
        FIXTURES / "sample_queries.csv",
    )
    for qid, docs in suite.qrels.items():
        for doc_id, rel in docs.items():
            assert 0 <= rel <= 3, f"Relevance {rel} out of range for {qid}/{doc_id}"


def test_expert_suite_filters_non_python() -> None:
    suite = build_expert_suite(
        FIXTURES / "sample_annotations.csv",
        FIXTURES / "sample_queries.csv",
    )
    # The Go annotation for "sort a list" should be excluded
    go_url = "https://github.com/other/go-repo/blob/aaa/sort.go#L1-L10"
    go_fid = func_id(go_url)
    for docs in suite.qrels.values():
        assert go_fid not in docs


def test_expert_suite_func_ids_are_sha256() -> None:
    suite = build_expert_suite(
        FIXTURES / "sample_annotations.csv",
        FIXTURES / "sample_queries.csv",
    )
    for docs in suite.qrels.values():
        for fid in docs:
            assert len(fid) == 64


def test_expert_suite_doc_count() -> None:
    suite = build_expert_suite(
        FIXTURES / "sample_annotations.csv",
        FIXTURES / "sample_queries.csv",
    )
    # "sort a list" has 3 Python annotations (3 distinct URLs)
    assert len(suite.qrels["sort a list"]) == 3
    # "read a file" has 1
    assert len(suite.qrels["read a file"]) == 1
