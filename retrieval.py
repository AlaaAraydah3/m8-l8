"""Module 8 — Applied Lab: Vector Retrieval.

Implement BM25, dense, and hybrid retrievers against a Weaviate index of the
CQADupStack + Stack Exchange technical-Q&A corpus, then evaluate all three on
the bundled 60-pair labeled set.

Methodology (canonical — autograder enforces):
- recall@k: gold_doc_id in top_k_returned_ids; mean over all queries.
- MRR: 1-indexed position of gold_doc_id in returned list of length 10;
  1/rank if found, 0 if not; mean over all queries.
- Hybrid alpha: 0.5 for the base assignment.
- Top-k for retrieval calls during evaluation: k=10; recall@5 is the top-5
  slice of those 10. One retrieval call per query.
"""

import json
import os
from collections import defaultdict
from typing import Callable

import weaviate

CLASS_NAME = "Post"

# ---------------------------------------------------------------------------
# Model loading — uses bundled local copy first to avoid HuggingFace rate
# limits in CI; falls back to remote download for fresh dev environments.
# ---------------------------------------------------------------------------
_model = None

def _get_embedder():
    """Return the sentence-transformers model, loading it once and caching it."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # Prefer the bundled copy committed alongside this file
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "models", "all-MiniLM-L6-v2")
        model_name = local_path if os.path.isdir(local_path) else "all-MiniLM-L6-v2"
        _model = SentenceTransformer(model_name)
    return _model


# ---------------------------------------------------------------------------
# Task 1 — Schema
# ---------------------------------------------------------------------------

def create_schema(client: weaviate.Client) -> None:
    """Create the Post class in Weaviate (idempotent — deletes first if exists)."""

    # Delete existing class so re-runs are idempotent
    existing = [c["class"] for c in client.schema.get().get("classes", [])]
    if CLASS_NAME in existing:
        client.schema.delete_class(CLASS_NAME)

    class_def = {
        "class": CLASS_NAME,
        "vectorizer": "none",
        "vectorIndexConfig": {
            "distance": "cosine"
        },
        "properties": [
            {
                # Filterable unique identifier — exact-match WHERE only, never BM25
                "name": "doc_id",
                "dataType": ["text"],
                "indexSearchable": False,
                "indexFilterable": True,
                "tokenization": "field",
            },
            {
                # Filterable subset tag — same pattern as doc_id
                "name": "subset",
                "dataType": ["text"],
                "indexSearchable": False,
                "indexFilterable": True,
                "tokenization": "field",
            },
            {
                # BM25-indexed
                "name": "title",
                "dataType": ["text"],
                "indexSearchable": True,
                "indexFilterable": False,
                "tokenization": "word",
            },
            {
                # BM25-indexed
                "name": "question_text",
                "dataType": ["text"],
                "indexSearchable": True,
                "indexFilterable": False,
                "tokenization": "word",
            },
            {
                # BM25-indexed
                "name": "answer_text",
                "dataType": ["text"],
                "indexSearchable": True,
                "indexFilterable": False,
                "tokenization": "word",
            },
            {
                # Stored only — dense-embedding source, NOT BM25-indexed
                "name": "text",
                "dataType": ["text"],
                "indexSearchable": False,
                "indexFilterable": False,
                "tokenization": "word",
            },
        ],
    }

    client.schema.create_class(class_def)
    print(f"[create_schema] Created class '{CLASS_NAME}'.")


# ---------------------------------------------------------------------------
# Task 2 — Ingest
# ---------------------------------------------------------------------------

def index_corpus(client: weaviate.Client, corpus_path: str, embedder) -> int:
    """Embed and ingest the corpus. Returns count of ingested objects."""

    # 1. Load all rows
    rows = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("[index_corpus] No rows found — check corpus_path.")
        return 0

    print(f"[index_corpus] Loaded {len(rows)} rows. Embedding …")

    # 2. Batch-embed the `text` field (fast: one forward pass for all docs)
    texts = [row["text"] for row in rows]
    vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True)

    # 3. Batch-ingest into Weaviate
    count = 0
    with client.batch as batch:
        batch.batch_size = 100
        for row, vec in zip(rows, vectors):
            properties = {
                "doc_id":        row["id"],          # "{subset}:{post_id}"
                "subset":        row["subset"],
                "title":         row.get("title", ""),
                "question_text": row.get("question_text", ""),
                "answer_text":   row.get("answer_text", ""),
                "text":          row.get("text", ""),
            }
            batch.add_data_object(
                data_object=properties,
                class_name=CLASS_NAME,
                vector=vec.tolist(),   # must be a plain Python list
            )
            count += 1

    # 4. Verify via aggregate (sanity-check; return the verified count)
    result = (
        client.query
        .aggregate(CLASS_NAME)
        .with_meta_count()
        .do()
    )
    verified = result["data"]["Aggregate"][CLASS_NAME][0]["meta"]["count"]
    print(f"[index_corpus] Ingested {count} rows; Weaviate reports {verified}.")
    return int(verified)


# ---------------------------------------------------------------------------
# Task 3 — BM25 retrieval
# ---------------------------------------------------------------------------

def bm25_search(client: weaviate.Client, query: str, k: int) -> list[str]:
    """BM25 retrieval over title, question_text, answer_text. Returns top-k doc_ids."""
    result = (
        client.query
        .get(CLASS_NAME, ["doc_id"])
        .with_bm25(query=query, properties=["title", "question_text", "answer_text"])
        .with_limit(k)
        .do()
    )
    hits = result.get("data", {}).get("Get", {}).get(CLASS_NAME, []) or []
    return [h["doc_id"] for h in hits]


# ---------------------------------------------------------------------------
# Task 4 — Dense retrieval
# ---------------------------------------------------------------------------

def dense_search(client: weaviate.Client, query: str, k: int, embedder) -> list[str]:
    """Dense (nearVector) retrieval. Returns top-k doc_ids."""
    qv = embedder.encode(query).tolist()
    result = (
        client.query
        .get(CLASS_NAME, ["doc_id"])
        .with_near_vector({"vector": qv})
        .with_limit(k)
        .do()
    )
    hits = result.get("data", {}).get("Get", {}).get(CLASS_NAME, []) or []
    return [h["doc_id"] for h in hits]


# ---------------------------------------------------------------------------
# Task 5 — Hybrid retrieval
# ---------------------------------------------------------------------------

def hybrid_search(
    client: weaviate.Client,
    query: str,
    k: int,
    embedder,
    alpha: float = 0.5,
) -> list[str]:
    """Hybrid (BM25 + vector) retrieval. alpha=0.5 for the base assignment."""
    qv = embedder.encode(query).tolist()
    result = (
        client.query
        .get(CLASS_NAME, ["doc_id"])
        .with_hybrid(query=query, vector=qv, alpha=alpha)
        .with_limit(k)
        .do()
    )
    hits = result.get("data", {}).get("Get", {}).get(CLASS_NAME, []) or []
    return [h["doc_id"] for h in hits]


# ---------------------------------------------------------------------------
# Task 6 — Evaluation
# ---------------------------------------------------------------------------

def evaluate_retriever(
    eval_path: str,
    search_fn: Callable,
    k_values=(5, 10),
) -> dict:
    """Evaluate a retriever against the labeled set.

    search_fn signature: search_fn(query: str, k: int) -> list[str]
    (use functools.partial to bind client/embedder/alpha before passing in).
    """
    max_k = max(k_values)   # 10
    min_k = min(k_values)   # 5

    # Accumulators — overall
    hits5, hits10, rr_scores = [], [], []

    # Accumulators — per query_type
    type_hits5   = defaultdict(list)
    type_hits10  = defaultdict(list)
    type_rr      = defaultdict(list)

    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query      = row["query"]
            gold_id    = row["gold_doc_id"]
            qtype      = row.get("query_type", "unknown")

            # One retrieval call per query — retrieve max_k (10)
            returned_ids = search_fn(query, k=max_k)

            top5  = returned_ids[:min_k]
            top10 = returned_ids[:max_k]

            h5  = 1 if gold_id in top5  else 0
            h10 = 1 if gold_id in top10 else 0

            # MRR — 1-indexed rank within top-10
            if gold_id in top10:
                rank = top10.index(gold_id) + 1
                rr = 1.0 / rank
            else:
                rr = 0.0

            hits5.append(h5)
            hits10.append(h10)
            rr_scores.append(rr)

            type_hits5[qtype].append(h5)
            type_hits10[qtype].append(h10)
            type_rr[qtype].append(rr)

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    by_type = {}
    for qtype in set(list(type_hits5.keys())):
        by_type[qtype] = {
            "recall@5":  _mean(type_hits5[qtype]),
            "recall@10": _mean(type_hits10[qtype]),
            "mrr":       _mean(type_rr[qtype]),
        }

    return {
        "recall@5":  _mean(hits5),
        "recall@10": _mean(hits10),
        "mrr":       _mean(rr_scores),
        "by_type":   by_type,
    }