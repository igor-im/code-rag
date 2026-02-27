Plan: Code Search Evaluation Framework                                        

 Context

 We need an algorithm-agnostic evaluation harness for code search. CSN (CodeSearchNet) is the first benchmark — others (manual suites, CoSQA,
 etc.) will follow. The generic eval framework lives in eval/, each benchmark gets its own subpackage (e.g. csn/).

 Flow:
 EvalSuite (queries + golden refs)
     → SearchAlgorithm.search(query) → ranked results
     → ranx.evaluate(qrels, run) → NDCG, MRR, MAP

 Architecture

 Two layers

 1. Generic eval framework (eval/) — suite format, runner, metrics. Benchmark-agnostic.
 2. Benchmarks (e.g. csn/) — download data, build EvalSuites. Each benchmark is a plugin that produces EvalSuites.

 The Interface Contract (centerpiece)

 There are two sides of the interface. Every algorithm implements one side, every benchmark produces the other. The runner connects them.

 ┌─────────────────┐                         ┌─────────────────┐
 │   Benchmark      │    produces             │   Algorithm      │
 │   (CSN, manual)  │───────────► EvalSuite   │   (FTS, vector)  │
 └─────────────────┘                         └────────┬────────┘
                                                      │ implements
                          ┌───────────┐               │
                          │  Runner   │◄──────────────┘
                          │           │  SearchAlgorithm.search()
                          │  ranx     │
                          │  evaluate │
                          └───────────┘

 Algorithm side: SearchAlgorithm protocol

 Every search algorithm must implement this. It's the only thing required to plug into the eval.

 # interfaces.py

 @dataclass
 class SearchResult:
     doc_id: str    # unique document ID (matches IDs in EvalSuite qrels)
     score: float   # algorithm's relevance/ranking score (higher = more relevant)

 class SearchAlgorithm(Protocol):
     """Interface every search algorithm must implement."""

     @property
     def name(self) -> str:
         """Algorithm identifier (used in run output filenames)."""
         ...

     def search(self, query: str, top_k: int) -> list[SearchResult]:
         """Return top_k results ranked by relevance, highest score first."""
         ...

 The algorithm owns indexing, corpus access, and ranking. The eval framework is blind to how search works — it only sees the returned
 list[SearchResult].

 Eval side: EvalSuite

 Every benchmark produces this. It's the standardized format for golden references.

 # eval/suite.py

 @dataclass
 class EvalSuite:
     name: str                                # e.g. "csn_expert", "csn_docstring", "manual_v1"
     queries: dict[str, str]                  # {qid: query_text}
     qrels: dict[str, dict[str, int]]         # {qid: {doc_id: relevance_score}}

     def save(self, path: Path) -> None       # serialize to queries.json + qrels.json
     @classmethod
     def load(cls, path: Path) -> EvalSuite   # deserialize
     def to_ranx_qrels(self) -> ranx.Qrels    # convert for ranx evaluation

 - doc_id in qrels must match doc_id returned by algorithms — this is the join key.
 - Relevance scores: 0-3 for graded (NDCG), binary 0/1 for simpler suites (MRR/MAP).

 Runner: connects both sides

 # eval/runner.py

 def run_eval(
     algorithm: SearchAlgorithm,
     suite: EvalSuite,
     top_k: int,
     metrics: list[str],       # e.g. ["ndcg@10", "mrr@10", "map@10"]
 ) -> dict[str, float]:
     """Run algorithm against every query in suite, score with ranx."""

 Document IDs

 Each benchmark defines its own ID scheme, but IDs must be stable and collision-free:
 - CSN: SHA256(github_url) — the function's GitHub URL is the canonical identifier
 - Manual (future): SHA256(file_path + "\0" + qualname)

 Files to Create

 1. Project scaffolding

 pyproject.toml — Python project config (src layout)
 - Dependencies: ranx, datasets (HuggingFace), python-dotenv
 - Dev deps: pytest, pytest-cov

 .gitignore — standard Python + data dirs

 2. src/coderag/__init__.py — empty

 3. src/coderag/interfaces.py — Core protocols

 SearchResult — dataclass(function_id: str, score: float)
 SearchAlgorithm — Protocol with search(query, top_k) -> list[SearchResult]

 4. src/coderag/eval/__init__.py — empty

 5. src/coderag/eval/suite.py — EvalSuite

 @dataclass EvalSuite:
     name: str
     queries: dict[str, str]           # qid -> query text
     qrels: dict[str, dict[str, int]]  # qid -> {func_id: relevance}

     def save(path: Path) -> None     # write queries.json + qrels.json
     @classmethod load(path: Path)    # read them back
     def to_ranx_qrels() -> ranx.Qrels

 Two JSON files per suite (queries.json, qrels.json) — human-readable, diffable.

 6. src/coderag/eval/runner.py — EvalRunner

 def run_eval(algorithm: SearchAlgorithm, suite: EvalSuite,
              top_k: int, metrics: list[str]) -> dict[str, float]:
     # For each query in suite:
     #   results = algorithm.search(query, top_k)
     #   Collect into ranx Run
     # return ranx.evaluate(qrels, run, metrics)

 Also saves the Run to disk (JSON) for later comparison.

 7. src/coderag/csn/__init__.py — empty

 8. src/coderag/csn/download.py — Data acquisition

 - download_corpus(dest: Path) — HuggingFace datasets.load_dataset("code_search_net", "python"), save as JSONL
 - download_annotations(dest: Path) — fetch annotationStore.csv + queries.csv from CSN GitHub repo raw URLs
 - Each function stores: func_id (SHA256 of url), url, func_name, code, docstring, path, repo

 9. src/coderag/csn/expert.py — Expert suite builder

 - Parse annotationStore.csv (filter Language=Python)
 - Parse queries.csv for the 99 query texts
 - Map GitHub URLs → func_ids (SHA256)
 - Build EvalSuite(name="csn_expert", queries={...}, qrels={...})
 - Relevance scores: 0-3 from annotations (graded, for NDCG)

 10. src/coderag/csn/docstring.py — Docstring suite builder

 - Read corpus JSONL
 - For each function with a docstring: query=first_paragraph(docstring), func_id=SHA256(url)
 - Binary relevance: qrels[qid][func_id] = 1
 - Subsample (configurable count, deterministic via sorted hash)
 - Build EvalSuite(name="csn_docstring", queries={...}, qrels={...})

 11. src/coderag/cli.py — CLI entry point

 Benchmarks are namespaced under eval <benchmark>:

 coderag eval csn init          # download CSN data to data_sets/codesearchnet/
 coderag eval csn generate      # build eval suites → data/eval_suites/csn_*/
 coderag eval csn run <algo>    # run algo against CSN suites, output metrics
 coderag eval compare           # compare runs across any benchmark

 Future benchmarks plug in the same way (eval manual ..., eval cosqa ..., etc.).

 Uses argparse with subparsers. Config (data paths) from env vars.

 Data Layout

 data_sets/codesearchnet/
     python.jsonl              # corpus (~500K functions)
     annotations/
         annotationStore.csv   # expert annotations
         queries.csv           # 99 queries

 data/eval_suites/
     csn_expert/
         queries.json
         qrels.json
     csn_docstring/
         queries.json
         qrels.json

 data/runs/
     <algo_name>_<suite>_<timestamp>/
         run.json              # ranx Run (serialized)
         metrics.json          # evaluation results

 Testing Strategy

 Unit tests

 - test_suite.py — EvalSuite save/load roundtrip, to_ranx_qrels conversion
 - test_runner.py — EvalRunner with a mock SearchAlgorithm (returns known results), verify metric computation
 - test_expert.py — Parse annotations CSV, verify suite shape (query count, relevance range 0-3)
 - test_docstring.py — Build suite from sample corpus, verify IDs and binary relevance
 - test_interfaces.py — Verify protocol conformance

 Integration test

 - test_end_to_end.py — Download a tiny slice of CSN, build suite, run mock algo, check metrics are sane

 Test data

 - Small fixture CSN samples (5-10 functions) in tests/fixtures/
 - Mock annotation CSV with 2-3 queries

 Verification

 # After implementation:
 pytest tests/ -v                                    # all tests pass
 coderag eval csn init                               # downloads CSN data
 coderag eval csn generate                           # builds both suites
 coderag eval csn run dummy --suite csn_expert       # runs dummy algo, prints NDCG/MRR

 Implementation Order

 1. Project scaffolding (pyproject.toml, dirs, .gitignore)
 2. interfaces.py — SearchResult, SearchAlgorithm protocol
 3. eval/suite.py — EvalSuite with save/load/to_ranx
 4. eval/runner.py — run_eval function
 5. csn/download.py — data acquisition
 6. csn/expert.py — expert suite builder
 7. csn/docstring.py — docstring suite builder
 8. cli.py — CLI wiring
 9. Tests throughout (written alongside each module)
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌