from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

from csn.download import extract_corpus, func_id

FIXTURES = Path(__file__).parent / "fixtures"


def test_func_id_deterministic() -> None:
    url = "https://github.com/owner/repo/blob/abc/file.py#L1-L5"
    assert func_id(url) == func_id(url)


def test_func_id_differs_for_different_urls() -> None:
    a = func_id("https://github.com/a/b/blob/c/d.py#L1")
    b = func_id("https://github.com/x/y/blob/z/w.py#L1")
    assert a != b


def test_func_id_is_hex_sha256() -> None:
    h = func_id("some-url")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def _make_test_zip(tmp_path: Path) -> Path:
    """Create a minimal CSN-like zip for testing."""
    records = []
    with open(FIXTURES / "sample_csn_records.jsonl") as f:
        for line in f:
            records.append(line.strip())

    gz_path = tmp_path / "test.jsonl.gz"
    with gzip.open(gz_path, "wt") as gz:
        for rec in records:
            gz.write(rec + "\n")

    zip_path = tmp_path / "python.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(gz_path, "python/final/jsonl/test/python_test_0.jsonl.gz")

    return zip_path


def test_extract_corpus(tmp_path: Path) -> None:
    zip_path = _make_test_zip(tmp_path)
    dest = tmp_path / "output"
    out = extract_corpus(zip_path, dest)

    assert out.exists()
    assert out.name == "python.jsonl"

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 5

    first = json.loads(lines[0])
    assert "func_id" in first
    assert "url" in first
    assert "code" in first
    assert "docstring" in first
    assert "repo" in first
    assert "path" in first
    assert "func_name" in first
    # func_id should match SHA256 of the url
    assert first["func_id"] == func_id(first["url"])


def test_extract_corpus_creates_dest_dir(tmp_path: Path) -> None:
    zip_path = _make_test_zip(tmp_path)
    dest = tmp_path / "nested" / "dir"
    out = extract_corpus(zip_path, dest)
    assert out.exists()
