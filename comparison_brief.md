# Comparison Brief — Module 8 Lab: Vector Retrieval

## Metrics Table

| Retriever | recall@5 | recall@10 | MRR    | factoid recall@5 | paraphrastic recall@5 |
|-----------|----------|-----------|--------|------------------|-----------------------|
| BM25      | 0.5667   | 0.6333    | 0.5474 | 1.0000           | 0.1333                |
| Dense     | 0.9000   | 0.9333    | 0.6703 | 0.8333           | 0.9667                |
| Hybrid    | 0.8500   | 0.9833    | 0.6863 | 1.0000           | 0.7000                |

---

## Where BM25 Wins

**Query type: factoid — BM25 recall@5 = 1.00 vs Dense recall@5 = 0.8333**

BM25 achieves a perfect 1.0 recall@5 on factoid queries while dense drops to 0.8333. Factoid queries contain exact identifiers, API names, error codes, or rare technical tokens that appear verbatim in the indexed title and answer fields. For example, a query like *"android:layout_weight attribute LinearLayout"* hits immediately because the XML attribute name is a distinctive low-frequency token — BM25's IDF weighting assigns it very high importance, and the token appears literally in the gold document's title. Dense embeddings average over many semantic dimensions and dilute this sharp lexical signal.

**Query type: factoid — BM25 MRR = 0.9833**

On factoid queries BM25 not only finds the gold document but ranks it first or second almost every time (MRR 0.9833). When a query is essentially a lookup by exact term — a specific function name, a numbered error code, a versioned library — BM25 places the matching document at rank 1 because no other document contains that exact token cluster. Dense retrieval finds the right document but sometimes ranks it third or fourth (MRR 0.7659), likely because semantically similar but non-gold documents score nearly as high.

---

## Where Dense Wins

**Query type: paraphrastic — Dense recall@5 = 0.9667 vs BM25 recall@5 = 0.1333**

Dense retrieval dominates paraphrastic queries by an enormous margin (0.9667 vs 0.1333). A paraphrastic query rephrases the document's content without sharing its vocabulary — for example, *"How do I make my site load faster for mobile users?"* maps to a gold document discussing HTTP caching, asset minification, and CDN configuration. None of those terms appear in the query. BM25 finds no token overlap and returns irrelevant results; the dense embedding captures the shared intent across the vocabulary gap.

**Query type: paraphrastic — Dense recall@10 = 1.00**

Dense retrieval achieves perfect recall@10 on paraphrastic queries (1.0000 vs BM25's 0.2667). Every single gold document for a paraphrastic query appears somewhere in the dense top-10, confirming that `all-MiniLM-L6-v2` encodes semantic meaning reliably enough to bridge natural-language paraphrases to their technical answers. BM25 misses 73% of these documents entirely even in the top 10.

---

## Alpha Recommendation

**Recommended alpha: 0.65**

The results show a strongly asymmetric corpus: BM25 is near-perfect on factoid queries but nearly useless on paraphrastic ones; dense is excellent across both but slightly weaker on factoid (0.8333 vs BM25's 1.0). Hybrid at alpha=0.5 already recovers BM25's factoid perfection (1.0) while keeping paraphrastic recall@5 at 0.70 — a large gain over pure BM25. However, paraphrastic recall@5 drops from dense's 0.9667 to hybrid's 0.7000, meaning the BM25 component is pulling some paraphrastic results down. Pushing alpha to 0.65 (slightly more weight on the dense/vector side) should recover more of that paraphrastic recall while keeping factoid recall near 1.0, since even a small BM25 contribution is enough to surface exact-token matches. Values above 0.75 risk losing the factoid precision advantage that makes hybrid worth using over dense alone.

---

## Schema Choice — Cosine vs. Dot Product

The schema uses `"vectorIndexConfig": {"distance": "cosine"}`.

`all-MiniLM-L6-v2` outputs L2-normalized vectors, which means cosine similarity and dot product produce **identical rankings** on this corpus (the dot product of two unit vectors equals their cosine similarity by definition). However, cosine is the safer production default: if the embedding model is ever swapped for one that does not normalize outputs, or if vectors are quantized or re-scaled during indexing, dot product rankings will silently degrade while cosine similarity remains invariant to magnitude. Specifying cosine documents the design intent — "rank by angle, not magnitude" — and prevents silent ranking regressions if the pipeline changes.