from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import ranx

from eval.suite import EvalSuite
from coderag.interfaces import SearchAlgorithm


def run_eval(
    algorithm: SearchAlgorithm,
    suite: EvalSuite,
    top_k: int,
    metrics: list[str],
    runs_dir: Path | None = None,
    n_workers: int = 1,
) -> dict[str, float]:
    """Run *algorithm* against every query in *suite*, score with ranx.

    If *runs_dir* is provided, saves the ranx Run and metrics JSON
    under ``runs_dir/<algo>_<suite>_<timestamp>/``.

    When *n_workers* > 1 and the algorithm supports ``search_batch``,
    queries are distributed across multiple processes.
    """
    if n_workers > 1 and hasattr(algorithm, "search_batch"):
        query_list = list(suite.queries.items())
        run_dict = algorithm.search_batch(query_list, top_k, n_workers)
    else:
        run_dict: dict[str, dict[str, float]] = {}
        for qid, query_text in suite.queries.items():
            results = algorithm.search(query_text, top_k)
            if results:
                run_dict[qid] = {r.doc_id: r.score for r in results}

    if not run_dict:
        return {m: 0.0 for m in metrics}

    qrels = suite.to_ranx_qrels()
    run = ranx.Run(run_dict)

    scores = ranx.evaluate(qrels, run, metrics, make_comparable=True)
    if isinstance(scores, float):
        # ranx returns a scalar when given a single metric
        scores = {metrics[0]: scores}

    if runs_dir is not None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = runs_dir / f"{algorithm.name}_{suite.name}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        run.save(str(out_dir / "run.json"))
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(scores, f, indent=2)

    return scores
