"""CLI entry point for the coderag evaluation framework."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _require_env(name: str) -> str:
    """Return an env var or exit with a clear error."""
    val = os.environ.get(name)
    if not val:
        sys.exit(f"Error: required environment variable {name} is not set")
    return val


# ---------------------------------------------------------------------------
# CSN subcommands
# ---------------------------------------------------------------------------

def _csn_init(args: argparse.Namespace) -> None:
    """Download / extract CSN data."""
    from csn.download import download_annotations, extract_corpus

    datasets_dir = Path(_require_env("CODERAG_DATASETS_DIR"))
    zip_path = datasets_dir / "codesearchnet" / "python.zip"
    if not zip_path.exists():
        sys.exit(f"Error: {zip_path} not found — download the CSN zip first")

    dest = datasets_dir / "codesearchnet"
    print(f"Extracting corpus from {zip_path} …")
    out = extract_corpus(zip_path, dest)
    print(f"  → {out}  ({out.stat().st_size:,} bytes)")

    annotations_url = _require_env("CODERAG_CSN_ANNOTATIONS_URL")
    queries_url = _require_env("CODERAG_CSN_QUERIES_URL")
    print("Downloading annotations …")
    ann, queries = download_annotations(dest, annotations_url, queries_url)
    print(f"  → {ann}")
    print(f"  → {queries}")
    print("Done.")


def _csn_generate(args: argparse.Namespace) -> None:
    """Build evaluation suites from CSN data."""
    from csn.docstring import build_docstring_suite
    from csn.expert import build_expert_suite

    datasets_dir = Path(_require_env("CODERAG_DATASETS_DIR"))
    data_dir = Path(_require_env("CODERAG_DATA_DIR"))
    csn_dir = datasets_dir / "codesearchnet"

    suites_dir = data_dir / "eval_suites"

    # Expert suite
    ann_path = csn_dir / "annotations" / "annotationStore.csv"
    queries_path = csn_dir / "annotations" / "queries.csv"
    if ann_path.exists() and queries_path.exists():
        print("Building expert suite …")
        expert = build_expert_suite(ann_path, queries_path)
        expert.save(suites_dir / "csn_expert")
        print(f"  → {suites_dir / 'csn_expert'}  "
              f"({len(expert.queries)} queries)")
    else:
        print(f"Skipping expert suite — annotations not found at {ann_path}")

    # Docstring suite
    corpus_path = csn_dir / "python.jsonl"
    if corpus_path.exists():
        sample_count = args.docstring_sample_count
        print(f"Building docstring suite (sample={sample_count}) …")
        docstring = build_docstring_suite(corpus_path, sample_count)
        docstring.save(suites_dir / "csn_docstring")
        print(f"  → {suites_dir / 'csn_docstring'}  "
              f"({len(docstring.queries)} queries)")
    else:
        sys.exit(f"Error: corpus not found at {corpus_path} — run 'eval csn init' first")

    print("Done.")


def _eval_index(args: argparse.Namespace) -> None:
    """Build a search index from a corpus."""
    data_dir = Path(_require_env("CODERAG_DATA_DIR"))
    corpus_path = Path(args.corpus)

    if not corpus_path.exists():
        sys.exit(f"Error: corpus not found at {corpus_path}")

    if args.algorithm == "embedding":
        if not args.model_name:
            sys.exit("Error: --model-name is required for embedding algorithm")
        if not args.batch_size:
            sys.exit("Error: --batch-size is required for embedding algorithm")
        if not args.max_length:
            sys.exit("Error: --max-length is required for embedding algorithm")

    if args.algorithm == "bm25":
        from coderag.algo.bm25 import init as bm25_init

        pg_url = _require_env("CODERAG_PG_URL")
        print(f"Building BM25 index from {corpus_path} …")
        count = bm25_init(corpus_path, pg_url)
        print(f"  → {count:,} documents indexed (postgres FTS)")
        print("Done.")
    elif args.algorithm == "embedding":
        from coderag.algo.embedding import init as embedding_init

        pg_url = _require_env("CODERAG_PG_URL")
        print(f"Building embedding index from {corpus_path} …")
        count = embedding_init(
            corpus_path, pg_url,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        print(f"  → {count:,} documents indexed (pgvector)")
        print("Done.")
    else:
        sys.exit(f"Error: unknown algorithm '{args.algorithm}'")


def _eval_run(args: argparse.Namespace) -> None:
    """Run an algorithm against an eval suite."""
    from eval.runner import run_eval
    from eval.suite import EvalSuite

    data_dir = Path(_require_env("CODERAG_DATA_DIR"))

    suite_path = data_dir / "eval_suites" / args.suite
    if not suite_path.exists():
        sys.exit(f"Error: suite not found at {suite_path}")

    suite = EvalSuite.load(suite_path)

    if args.algorithm == "embedding":
        for flag in ("model_name", "reranker_name", "rerank_candidates", "gap_threshold", "reranker_max_length", "ef_search"):
            if getattr(args, flag, None) is None:
                pretty = flag.replace("_", "-")
                sys.exit(f"Error: --{pretty} is required for embedding algorithm")

    if args.algorithm == "hybrid":
        for flag in ("model_name", "reranker_name", "rerank_candidates", "gap_threshold", "reranker_max_length", "ef_search", "rrf_k", "bm25_depth", "embedding_depth"):
            if getattr(args, flag, None) is None:
                pretty = flag.replace("_", "-")
                sys.exit(f"Error: --{pretty} is required for hybrid algorithm")

    algo = _resolve_algorithm(args.algorithm, data_dir, args)

    metrics = [m.strip() for m in args.metrics.split(",")]
    runs_dir = data_dir / "runs"

    print(f"Running {algo.name} against {suite.name} "
          f"(top_k={args.top_k}, metrics={metrics}, workers={args.workers}) …")
    scores = run_eval(algo, suite, args.top_k, metrics, runs_dir, args.workers)

    print("\nResults:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")


def _resolve_algorithm(name: str, data_dir: Path, args: argparse.Namespace) -> object:
    """Look up a SearchAlgorithm by name."""
    if name == "dummy":
        from coderag.interfaces import SearchResult

        class DummyAlgorithm:
            @property
            def name(self) -> str:
                return "dummy"

            def search(self, query: str, top_k: int) -> list[SearchResult]:
                return []

        return DummyAlgorithm()

    if name == "bm25":
        from coderag.algo.bm25 import Bm25Algorithm

        pg_url = _require_env("CODERAG_PG_URL")
        try:
            return Bm25Algorithm(pg_url)
        except FileNotFoundError as e:
            sys.exit(f"Error: {e}\nBuild it first with: coderag eval index --algorithm bm25 --corpus <path>")

    if name == "embedding":
        from coderag.algo.embedding import EmbeddingAlgorithm

        pg_url = _require_env("CODERAG_PG_URL")
        return EmbeddingAlgorithm(
            pg_url,
            model_name=args.model_name,
            reranker_name=args.reranker_name,
            rerank_candidates=args.rerank_candidates,
            gap_threshold=args.gap_threshold,
            reranker_max_length=args.reranker_max_length,
            ef_search=args.ef_search,
        )

    if name == "hybrid":
        from coderag.algo.bm25 import Bm25Algorithm
        from coderag.algo.embedding import EmbeddingAlgorithm
        from coderag.algo.hybrid import HybridAlgorithm

        pg_url = _require_env("CODERAG_PG_URL")
        try:
            bm25 = Bm25Algorithm(pg_url)
        except FileNotFoundError as e:
            sys.exit(f"Error: {e}\nBuild it first with: coderag eval index --algorithm bm25 --corpus <path>")
        embedding = EmbeddingAlgorithm(
            pg_url,
            model_name=args.model_name,
            reranker_name=args.reranker_name,
            rerank_candidates=args.rerank_candidates,
            gap_threshold=args.gap_threshold,
            reranker_max_length=args.reranker_max_length,
            ef_search=args.ef_search,
        )

        return HybridAlgorithm(
            bm25=bm25,
            embedding=embedding,
            rrf_k=args.rrf_k,
            bm25_depth=args.bm25_depth,
            embedding_depth=args.embedding_depth,
        )

    sys.exit(f"Error: unknown algorithm '{name}'")


def _eval_compare(args: argparse.Namespace) -> None:
    """Compare metrics across runs."""
    data_dir = Path(_require_env("CODERAG_DATA_DIR"))
    runs_dir = data_dir / "runs"

    if not runs_dir.exists():
        sys.exit(f"Error: no runs directory at {runs_dir}")

    rows: list[tuple[str, dict[str, float]]] = []
    for run_path in sorted(runs_dir.iterdir()):
        metrics_file = run_path / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            rows.append((run_path.name, metrics))

    if not rows:
        print("No runs found.")
        return

    # Collect all metric names
    all_metrics = sorted({m for _, ms in rows for m in ms})
    header = f"{'Run':<50} " + " ".join(f"{m:>12}" for m in all_metrics)
    print(header)
    print("-" * len(header))
    for name, metrics in rows:
        vals = " ".join(f"{metrics.get(m, 0.0):>12.4f}" for m in all_metrics)
        print(f"{name:<50} {vals}")


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coderag", description="Code search evaluation framework")
    sub = parser.add_subparsers(dest="command")

    # eval subcommand
    eval_parser = sub.add_parser("eval", help="Evaluation commands")
    eval_sub = eval_parser.add_subparsers(dest="eval_command")

    # eval csn
    csn_parser = eval_sub.add_parser("csn", help="CodeSearchNet benchmark")
    csn_sub = csn_parser.add_subparsers(dest="csn_command")

    # eval csn init
    csn_sub.add_parser("init", help="Download and extract CSN data")

    # eval csn generate
    gen_parser = csn_sub.add_parser("generate", help="Build evaluation suites")
    gen_parser.add_argument(
        "--docstring-sample-count",
        type=int,
        required=True,
        help="Number of docstring queries to sample",
    )

    # eval run <algorithm> --suite <name>
    run_parser = eval_sub.add_parser("run", help="Run algorithm against a suite")
    run_parser.add_argument("algorithm", help="Algorithm name (e.g. dummy)")
    run_parser.add_argument("--suite", required=True, help="Suite directory name")
    run_parser.add_argument("--top-k", type=int, required=True, help="Number of results per query")
    run_parser.add_argument(
        "--metrics",
        required=True,
        help="Comma-separated metrics (e.g. ndcg@10,mrr@10)",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        required=True,
        help="Number of parallel worker processes (1 = sequential)",
    )
    run_parser.add_argument("--model-name", help="Embedder model name (embedding algo)")
    run_parser.add_argument("--reranker-name", help="Reranker model name (embedding algo)")
    run_parser.add_argument("--rerank-candidates", type=int, help="Number of KNN candidates to rerank (embedding algo)")
    run_parser.add_argument("--gap-threshold", type=float, help="Score gap threshold for gating (embedding algo)")
    run_parser.add_argument("--reranker-max-length", type=int, help="Max token length for reranker (embedding algo)")
    run_parser.add_argument("--ef-search", type=int, help="HNSW ef_search parameter (embedding algo)")
    run_parser.add_argument("--rrf-k", type=int, help="RRF constant (hybrid algo)")
    run_parser.add_argument("--bm25-depth", type=int, help="BM25 retrieval depth (hybrid algo)")
    run_parser.add_argument("--embedding-depth", type=int, help="Embedding retrieval depth (hybrid algo)")

    # eval index --algorithm <name> --corpus <path>
    index_parser = eval_sub.add_parser("index", help="Build search index from corpus")
    index_parser.add_argument("--algorithm", required=True, help="Algorithm name (e.g. bm25)")
    index_parser.add_argument("--corpus", required=True, help="Path to JSONL corpus file")
    index_parser.add_argument("--model-name", help="Model name for embedding algorithm")
    index_parser.add_argument("--batch-size", type=int, help="Batch size for embedding algorithm")
    index_parser.add_argument("--max-length", type=int, help="Max token sequence length for embedding algorithm")

    # eval compare
    eval_sub.add_parser("compare", help="Compare runs across benchmarks")

    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "eval":
        if args.eval_command is None:
            parser.parse_args(["eval", "--help"])
        elif args.eval_command == "csn":
            if args.csn_command is None:
                parser.parse_args(["eval", "csn", "--help"])
            elif args.csn_command == "init":
                _csn_init(args)
            elif args.csn_command == "generate":
                _csn_generate(args)
        elif args.eval_command == "index":
            _eval_index(args)
        elif args.eval_command == "run":
            _eval_run(args)
        elif args.eval_command == "compare":
            _eval_compare(args)


if __name__ == "__main__":
    main()
