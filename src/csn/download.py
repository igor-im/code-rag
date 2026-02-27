"""CSN data acquisition: extract corpus from zip and fetch annotations."""

from __future__ import annotations

import gzip
import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def func_id(url: str) -> str:
    """Stable function ID: SHA256 hex digest of the GitHub URL."""
    return hashlib.sha256(url.encode()).hexdigest()


def extract_corpus(zip_path: Path, dest: Path) -> Path:
    """Extract python JSONL records from *zip_path* into *dest*/python.jsonl.

    Each output line is a JSON object with keys:
    func_id, url, repo, path, func_name, code, docstring.
    """
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "python.jsonl"

    with zipfile.ZipFile(zip_path) as zf:
        jsonl_names = sorted(
            n for n in zf.namelist() if n.endswith(".jsonl.gz")
        )
        with open(out_path, "w") as out:
            for name in jsonl_names:
                with zf.open(name) as raw, gzip.open(raw, "rt") as gz:
                    for line in gz:
                        record = json.loads(line)
                        out.write(
                            json.dumps(
                                {
                                    "func_id": func_id(record["url"]),
                                    "url": record["url"],
                                    "repo": record["repo"],
                                    "path": record["path"],
                                    "func_name": record["func_name"],
                                    "code": record["code"],
                                    "docstring": record.get("docstring", ""),
                                }
                            )
                            + "\n"
                        )
    return out_path


def download_annotations(dest: Path, annotations_url: str, queries_url: str) -> tuple[Path, Path]:
    """Fetch annotationStore.csv and queries.csv into *dest*/annotations/.

    Returns (annotations_path, queries_path).
    """
    ann_dir = dest / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    ann_path = ann_dir / "annotationStore.csv"
    queries_path = ann_dir / "queries.csv"

    urlretrieve(annotations_url, ann_path)
    urlretrieve(queries_url, queries_path)

    return ann_path, queries_path
