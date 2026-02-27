Plan: Embedding Search Algorithm (BGE-Code-v1 + Reranker)

Context

Add a two-stage embedding-based search algorithm to compare against BM25.
Stage 1: dense retrieval via BGE-Code-v1 embeddings stored in sqlite-vec.
Stage 2: reranking via BGE-Reranker-v2-m3, with score-gap gating to protect
high-confidence top results from being displaced.

Architecture

 Query
   |
   v
 BGE-Code-v1 (embed query with instruction prefix)
   |
   v
 sqlite-vec KNN (top `rerank_candidates` nearest neighbors)
   |
   v
 BGE-Reranker-v2-m3 (rescore all candidates)
   |
   v
 Score-gap gating (pin #1 if embedding gap >= threshold)
   |
   v
 Return top_k results

Models

- Embedder: BAAI/bge-code-v1 (2B params, 1536-dim, 4096 max tokens)
- Reranker: BAAI/bge-reranker-v2-m3 (0.6B params, 512 max tokens)
- Both loaded via `transformers` (AutoModel / AutoModelForSequenceClassification)
- FP16 on GPU, FP32 on CPU

Module Structure

File: src/coderag/algo/embedding.py

Two public entry points (standardized pattern, same as BM25):

    def init(corpus_path, db_path, *, model_name, batch_size) -> int
        Embed all documents, store in sqlite-vec. Returns doc count.

    class EmbeddingAlgorithm:
        Implements SearchAlgorithm protocol.

        __init__(db_path, *, model_name, reranker_name,
                 rerank_candidates, gap_threshold, bm25_db_path)
        name -> "embedding"
        search(query, top_k) -> list[SearchResult]

BM25 standardization: rename build_index -> init in bm25.py.

Index Building (init)

1. Load BGE-Code-v1 via AutoModel + AutoTokenizer from transformers.
2. Create sqlite-vec database:
   - vec0 virtual table: func_id + embedding FLOAT[1536]
   - Uses temp-file-then-copy pattern (NAS safety, same as BM25).
3. Stream corpus JSONL in batches of `batch_size`:
   - Concatenate: func_name + "\n" + code + "\n" + docstring
   - Documents get NO instruction prefix (only queries do).
   - Tokenize, encode, extract last-token embedding, L2-normalize.
   - Insert (func_id, embedding) into sqlite-vec.
4. Return document count.

The embedding index stores vectors only. Document text (for reranking)
is read from the BM25 index at search time.

Search Flow (search)

1. Embed query:
   - Prefix with: <instruct>{instruction}\n<query>{query_text}
   - Instruction: "Given a textual explanation of code functionality,
     retrieve the corresponding code implementation."
   - Encode with BGE-Code-v1, L2-normalize.

2. KNN retrieval:
   - sqlite-vec query for top `rerank_candidates` nearest neighbors.
   - Returns [(func_id, cosine_score), ...].

3. Score-gap check:
   - gap = score[0] - score[1]
   - pin_top = (gap >= gap_threshold)

4. Rerank:
   - Look up doc texts from BM25 index (func_name + code + docstring).
   - Build pairs: [(query, doc_text), ...] for all candidates.
   - Score with BGE-Reranker-v2-m3 (AutoModelForSequenceClassification).
   - Sigmoid-normalize scores to [0, 1].
   - Sort by reranker score descending.

5. Apply gating:
   - If pin_top: force original embedding #1 into position 0,
     shift others down.

6. Return top_k as list[SearchResult].

Prerequisites:
- BM25 index must be built first (provides text for reranker lookups).

Dependencies

New in pyproject.toml:
- transformers
- torch
- sqlite-vec

CLI Changes

src/eval/cli.py modifications:

- _eval_index: add "embedding" branch, calls embedding.init()
  with --model-name and --batch-size flags (both required).
- _resolve_algorithm: add "embedding" branch, instantiates
  EmbeddingAlgorithm with --reranker-name, --rerank-candidates,
  --gap-threshold, --bm25-db-path flags (all required).
- BM25 references updated for build_index -> init rename.

Testing Strategy

test_embedding.py (mock models, no real GPU/downloads):

- init: mock transformer, verify sqlite-vec table created,
  doc count returned, embeddings stored correctly.
- search without gating: verify KNN -> rerank -> correct ordering
  when score gap is below threshold.
- search with gating: verify pinned #1 stays on top when
  gap exceeds threshold.
- BM25 text lookup: verify reranker fetches doc text from BM25 db.
- Edge cases: empty query, single candidate, all same score.

BM25 rename: update existing test imports (build_index -> init).

No integration tests requiring real model downloads.
Manual validation against real CSN data.

Validation

    uv run pytest tests/ -v
    uv run coderag eval index --algorithm bm25 --corpus <path>
    uv run coderag eval index --algorithm embedding --corpus <path> \
        --model-name BAAI/bge-code-v1 --batch-size 32
    uv run coderag eval run embedding --suite csn_expert --top-k 10 \
        --metrics "ndcg@10,mrr@10" --workers 1 \
        --reranker-name BAAI/bge-reranker-v2-m3 \
        --rerank-candidates 50 --gap-threshold 0.1
    uv run coderag eval compare
