from __future__ import annotations

from pathlib import Path

import ranx

from eval.suite import EvalSuite


def _sample_suite() -> EvalSuite:
    return EvalSuite(
        name="test_suite",
        queries={"q1": "sort a list", "q2": "read a file"},
        qrels={
            "q1": {"d1": 3, "d2": 1},
            "q2": {"d3": 2},
        },
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    suite = _sample_suite()
    suite.save(tmp_path / "out")
    loaded = EvalSuite.load(tmp_path / "out")
    assert loaded.name == suite.name
    assert loaded.queries == suite.queries
    assert loaded.qrels == suite.qrels


def test_save_creates_directory(tmp_path: Path) -> None:
    suite = _sample_suite()
    dest = tmp_path / "nested" / "dir"
    suite.save(dest)
    assert (dest / "queries.json").exists()
    assert (dest / "qrels.json").exists()


def test_to_ranx_qrels() -> None:
    suite = _sample_suite()
    qrels = suite.to_ranx_qrels()
    assert isinstance(qrels, ranx.Qrels)


def test_to_ranx_qrels_content() -> None:
    suite = EvalSuite(
        name="tiny",
        queries={"q1": "hello"},
        qrels={"q1": {"d1": 1, "d2": 0}},
    )
    qrels = suite.to_ranx_qrels()
    # ranx Qrels should have the same structure
    assert "q1" in qrels.to_dict()
    assert qrels.to_dict()["q1"]["d1"] == 1


def test_save_json_is_sorted(tmp_path: Path) -> None:
    """Verify JSON keys are sorted for stable diffs."""
    import json

    suite = EvalSuite(
        name="s",
        queries={"z": "zzz", "a": "aaa"},
        qrels={"z": {"d2": 1, "d1": 0}, "a": {"d3": 2}},
    )
    suite.save(tmp_path / "out")
    with open(tmp_path / "out" / "queries.json") as f:
        raw = f.read()
    data = json.loads(raw)
    assert list(data["queries"].keys()) == ["a", "z"]
