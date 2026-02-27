"""Build the CSN expert-annotated evaluation suite (graded relevance 0-3)."""

from __future__ import annotations

import csv
from pathlib import Path

from csn.download import func_id
from eval.suite import EvalSuite


def build_expert_suite(
    annotations_path: Path,
    queries_path: Path,
) -> EvalSuite:
    """Parse CSN annotations and build a graded-relevance EvalSuite.

    Only Python annotations are included.  When multiple annotators
    scored the same (query, url) pair, the *rounded mean* is used.
    """
    # Load query texts
    query_texts: dict[str, str] = {}
    with open(queries_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["query"].strip()
            # Use the query text itself as qid (stable, human-readable)
            query_texts[q] = q

    # Collect annotations: (query, url) -> list of relevance scores
    raw: dict[tuple[str, str], list[int]] = {}
    with open(annotations_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Language"] != "Python":
                continue
            q = row["Query"].strip()
            url = row["GitHubUrl"].strip()
            rel = int(row["Relevance"])
            raw.setdefault((q, url), []).append(rel)

    # Build qrels: aggregate multiple annotations per pair via rounded mean
    qrels: dict[str, dict[str, int]] = {}
    queries: dict[str, str] = {}
    for (q, url), scores in raw.items():
        if q not in query_texts:
            continue
        fid = func_id(url)
        mean_rel = round(sum(scores) / len(scores))
        qrels.setdefault(q, {})[fid] = mean_rel
        queries[q] = q

    return EvalSuite(name="csn_expert", queries=queries, qrels=qrels)
