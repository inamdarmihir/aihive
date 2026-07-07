---
title: "Duplicate Issue Detection Without the Triage Tax: A Qdrant Hybrid Search Sidecar for GitHub"
date: 2026-06-25
description: "How to build a GitHub webhook sidecar that uses Qdrant BM42 sparse-dense hybrid search and async scalar quantization to automatically surface duplicate issues before they ever reach the triage queue — with concrete collection design, scoring thresholds, and a FastAPI webhook handler."
tags: ["qdrant", "vector-search", "bm42", "hybrid-search", "github", "deduplication", "system-design", "fastapi"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Manual duplicate triage is an untracked engineering cost: on repositories with more than a few thousand open issues, GitHub's built-in title-matching catches copy-paste duplicates and misses every case where two reporters describe the same root cause in different words. In this post, I walk through building a Qdrant sidecar that runs a BM42 sparse-dense hybrid query on every `issues.opened` webhook event, surfacing likely duplicates before any human reads the new report. The focus is collection design, the BM42 mechanism, the hybrid query with Reciprocal Rank Fusion, scalar quantization tradeoffs, threshold calibration, and async archive backfill. This post does not cover issue classification, priority routing, or label inference. Familiarity with vector databases and GitHub webhook integration is assumed.

---

## Table of Contents

1. [Why Exact-String Matching Fails](#why-exact-string-fails)
2. [Architecture: The Sidecar Pattern](#architecture)
3. [How BM42 Works: Attention Weights as Sparse Signals](#bm42-mechanism)
   - [BM25 Baseline](#bm25-baseline)
   - [What BM42 Changes](#what-bm42-changes)
   - [Sparse Vector Structure](#sparse-vector-structure)
4. [The Qdrant Collection Design](#collection-design)
   - [Named Vectors: Dense and Sparse Together](#named-vectors)
   - [Scalar Quantization: INT8 vs FLOAT32](#quantization)
   - [Payload Schema](#payload-schema)
5. [Backfilling the Archive with Async Quantization](#async-quantization)
6. [The Webhook Handler](#webhook-handler)
7. [The Hybrid Search Query](#hybrid-search)
   - [Why BM42 + Dense, Not Just Dense](#why-bm42)
   - [RRF vs. Linear Combination](#fusion-strategies)
8. [Scoring and Thresholds](#scoring)
9. [The GitHub Bot Response](#github-bot)
10. [Challenges and Open Problems](#challenges)

---

## Why Exact-String Matching Fails {#why-exact-string-fails}

GitHub already does substring and edit-distance matching on titles when you open a new issue. It works for copy-paste duplicates. The failures that cost the most time are not copy-paste duplicates; they are semantically equivalent reports written by different people who describe the same thing differently.

Consider a real failure mode: a library has a bug where calling `.close()` on an already-closed connection throws an unhandled exception. In a month, five people report it:

```
"UnhandledRejection when calling close() on disconnected socket"
"app crashes after network timeout, close() throws"
"TypeError: Cannot read property 'destroy' of null"
"Fatal error on socket close after idle timeout"
"Crash in connection pool teardown, reproducible"
```

These share no useful n-grams. The error messages differ because the exact stack trace varies by Node version. The symptom descriptions differ because users observe the crash at different call sites. GitHub's similarity suggestions, which operate on the title text, surface none of them to each other.

A dense embedding of each title places all five within cosine distance $\approx 0.12$ of each other and $\approx 0.45$ away from unrelated issues. This is exactly what embedding-based retrieval is for.

But dense alone has a different failure mode: it loses *specific* tokens. If you search for `TypeError: Cannot read property 'destroy' of null`, a dense model will score that against "socket teardown failure" as high similarity, because both are about connection cleanup. That is the right behavior for semantic dedup. It is the *wrong* behavior when a developer wants to find the exact error message to verify whether a fix applies to their specific stack, or to cross-reference a known CVE.

**BM42** ([Qdrant's sparse embedding model](https://qdrant.tech/articles/bm42/)) is built for exactly this split. It produces sparse vectors using transformer-derived attention weights rather than raw term frequency, preserving the specificity of rare tokens like `destroy`, `UnhandledRejection`, or `ERR_SOCKET_CLOSED` while still providing term-level recall that dense models dilute. Running BM42 and dense in the same Qdrant hybrid query with Reciprocal Rank Fusion gives you both signals over a single round-trip.

---

## Architecture: The Sidecar Pattern {#architecture}

Qdrant sits entirely outside the issue tracker's critical path. The GitHub issues database is unchanged. The sidecar adds exactly one external touch: a bot comment on new issues that match an existing one above the confidence threshold.

```
  GitHub Issues
  ┌───────────────────────────────────────────────┐
  │  issues.opened webhook ──────────────────────┼──► FastAPI
  │                                               │    /webhook
  │  Issues API (read)  ◄─────────────────────────┼──   │
  │  Issues API (write) ◄─────────────────────────┼──   │ upsert new issue
  │    (bot comment)                              │     │ hybrid search
  └───────────────────────────────────────────────┘     │
                                                         ▼
                                               ┌──────────────────┐
                                               │  Qdrant           │
                                               │  issues_index     │
                                               │                   │
                                               │  dense:  1536-d   │
                                               │  sparse: BM42     │
                                               │  quant:  SQ8      │
                                               └──────────────────┘
```

The webhook handler does four things in order on every `issues.opened` event:

1. **Upsert** the new issue into Qdrant (so future issues can match against it)
2. **Hybrid search** for the most similar existing issues
3. **Score and filter** candidates against the confidence threshold
4. **Post a bot comment** if any candidate exceeds the threshold

Nothing is gated on Qdrant availability. If the vector store is unreachable, the handler returns 200 and the issue flows through normally. The sidecar is advisory, not blocking.

---

## How BM42 Works: Attention Weights as Sparse Signals {#bm42-mechanism}

Understanding what BM42 computes, and how it differs from BM25, is necessary for making good decisions about when to use it and how to weight it against dense retrieval.

### BM25 Baseline {#bm25-baseline}

BM25 is a bag-of-words retrieval model. For a query $q$ and document $d$, it scores each query term $t$ against the document using:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

where $f(t, d)$ is the raw count of term $t$ in document $d$, $k_1 \in [1.2, 2.0]$ controls TF saturation (diminishing returns from repeated occurrences), $b = 0.75$ normalizes for document length relative to the corpus average $\text{avgdl}$, and $\text{IDF}(t)$ is:

$$\text{IDF}(t) = \log\frac{N - n(t) + 0.5}{n(t) + 0.5}$$

with $N$ the total number of documents and $n(t)$ the number containing term $t$.

The term frequency component is BM25's main weakness for GitHub issues. A title like "crash on close" and a body that mentions "close" fourteen times will rank identically in many TF-weighted systems, because TF saturates but still dominates rare documents. More critically, BM25 has no way to know that `ERR_SOCKET_CLOSED` is a more *specific* token than `error`, beyond what IDF captures from corpus statistics. IDF works at the population level; it knows `error` is common and `ERR_SOCKET_CLOSED` is rare, but it cannot assess the semantic *role* that each token plays in the specific document being indexed.

### What BM42 Changes {#what-bm42-changes}

BM42 makes one targeted replacement: it drops term frequency entirely and substitutes per-token attention weights from a transformer. The model used by the `fastembed` reference implementation is **all-MiniLM-L6-v2**, a 22M-parameter sentence transformer. The process is:

1. Tokenize the input text with the model's WordPiece tokenizer.
2. Run a forward pass through the 6-layer transformer.
3. Extract the attention weights from the final layer's [CLS] token row. Each attention head produces a distribution over all tokens; the values are averaged across heads to produce a single scalar weight per token.
4. Multiply each per-token attention weight by the token's IDF score (computed from a background corpus shipped with the model, not from the local collection).
5. Emit a sparse vector with one non-zero entry per unique token in the input, weighted by attention × IDF.

The resulting scoring formula is approximately:

$$s_{\text{BM42}}(t, d) = \text{IDF}(t) \cdot \alpha_t(d)$$

where $\alpha_t(d)$ is the mean attention weight that the [CLS] token assigns to token $t$ when encoding document $d$.

What this achieves: the transformer's attention mechanism learns, during pretraining, which tokens are semantically central to a piece of text. For "UnhandledRejection when calling close() on disconnected socket", the model's attention concentrates on `UnhandledRejection` and `disconnected` because they carry the distinctive semantic load; stopwords and common verbs receive near-zero attention. Multiplied by IDF, which is also high for `UnhandledRejection` (a rare term in any corpus), this token receives a disproportionately high sparse weight.

BM25 would assign `UnhandledRejection` a high IDF score too, but its TF contribution would be 1 (it appears once), indistinguishable from `close` or `calling`. BM42 lets the model's learned attention push `UnhandledRejection` significantly higher.

### Sparse Vector Structure {#sparse-vector-structure}

The output of BM42 is a standard sparse vector: a parallel array of token indices and floating-point weights. For a typical GitHub issue title (8 to 20 tokens after WordPiece tokenization), the sparse vector has 8 to 20 non-zero entries. For a title plus 512-character body excerpt, it might have 60 to 150 non-zero entries.

Qdrant stores sparse vectors in a compressed index that supports dot-product scoring. At query time, the query sparse vector is intersected with stored sparse vectors; only tokens present in both the query and the stored document contribute to the score. This makes sparse retrieval fast even at scale: for a 100k-issue collection, a BM42 query touches only the posting lists for the specific tokens in the query, analogous to a traditional inverted index lookup.

The contrast with dense vectors is instructive. A 1536-dimensional dense vector requires computing the dot product across all 1536 dimensions for every candidate in the HNSW search. Sparse vectors only compute over the non-zero intersection. For short queries with few high-signal tokens, sparse retrieval is faster and more precise; for long natural-language queries, dense retrieval generalizes better.

---

## The Qdrant Collection Design {#collection-design}

### Named Vectors: Dense and Sparse Together {#named-vectors}

Qdrant's named vector API lets you store multiple vector representations per point and query against any combination of them. The collection needs exactly two:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    ScalarQuantizationConfig,
    ScalarType,
    QuantizationConfig,
)

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="issues_index",
    vectors_config={
        "dense": VectorParams(
            size=1536,
            distance=Distance.COSINE,
        ),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(),
    },
    quantization_config=QuantizationConfig(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        )
    ),
    on_disk_payload=True,
)
```

The `"dense"` vector name and `"sparse"` vector name are referenced by exact string in every subsequent upsert and query call. Name mismatches fail silently in some client versions, returning empty results; using constants rather than string literals throughout the codebase is worth the discipline.

### Scalar Quantization: INT8 vs FLOAT32 {#quantization}

Scalar quantization compresses each component of a dense FLOAT32 vector to an INT8 value. The memory savings are proportional to the precision reduction: FLOAT32 uses 4 bytes per dimension, INT8 uses 1 byte per dimension, giving a 4× compression ratio.

For a 1536-dimensional dense vector (the output size of `text-embedding-3-small`):

| Format | Bytes per vector | 100k issue collection |
|---|---|---|
| FLOAT32 | 1536 × 4 = 6,144 B (≈ 6 KB) | ≈ 600 MB |
| INT8 | 1536 × 1 = 1,536 B (≈ 1.5 KB) | ≈ 150 MB |

The `quantile=0.99` parameter controls how the FLOAT32 range maps to the INT8 range. Qdrant computes the distribution of all component values across the collection, takes the 0.5th percentile as the minimum and the 99.5th percentile as the maximum, and maps that range linearly to the signed INT8 range $[-127, 127]$. Values outside the percentile window are clamped to the extremes.

Setting `quantile=0.99` rather than `1.0` prevents outlier components from dilating the mapping range. If a small fraction of vector components have extreme values (which happens with distributions that have heavy tails), a `quantile=1.0` mapping would compress the range for the 99% of values that are well-behaved. At `quantile=0.99`, the 1% of outlier values are clamped, and the remaining 99% use the full INT8 precision. The tradeoff is a slight distortion for outlier components, which is almost always preferable.

The recall impact is modest. On typical text embedding distributions, INT8 scalar quantization with `quantile=0.99` achieves recall@100 (the fraction of true top-100 nearest neighbors recovered) of approximately 0.97 versus full FLOAT32, measured over random query samples. HNSW traversal is already approximate, with recall typically 0.95 to 0.99 depending on `ef` configuration. Adding INT8 quantization drops total recall by another 1 to 3 percentage points at comparable HNSW settings, which is acceptable for deduplication: missed duplicates (false negatives) are inconvenient; wrongly surfaced non-duplicates (false positives) can be filtered by the downstream threshold check.

**Quantization rescoring** is the mechanism that keeps recall high despite compressed scoring during graph traversal. The `always_ram=True` flag keeps the INT8 quantized vectors pinned in memory. During HNSW search, Qdrant uses INT8 dot products for fast candidate scoring and graph navigation. Once the HNSW traversal completes, Qdrant loads the full FLOAT32 originals for only the final top-k candidates and rescores them at full precision. This two-phase approach means the ANN graph traversal (the expensive part, touching many vectors) runs on compact INT8 data, while the final ranking uses exact FLOAT32 values. The FLOAT32 originals can be stored on disk without impacting search latency because they are only read for the small final candidate set.

```python
# always_ram=True: INT8 quantized vectors stay in RAM
# full FLOAT32 originals can page to disk, read only for rescoring
ScalarQuantizationConfig(
    type=ScalarType.INT8,
    quantile=0.99,
    always_ram=True,
)
```

For the issues sidecar, the practical consequence is that a 100k-issue collection fits comfortably in 150 MB of vector RAM plus payload on disk, versus the 600 MB that would be required without quantization. This keeps the sidecar hostable on a single small instance.

The `on_disk_payload=True` flag at the collection level moves the text payload (titles, bodies, URLs) to disk. Payloads are only needed on the final candidates returned after scoring; they are never accessed during the HNSW traversal itself. This is a free memory saving for large archives and does not affect search latency under normal query loads.

### Payload Schema {#payload-schema}

Each point in `issues_index` maps to one GitHub issue:

```python
from qdrant_client.models import PointStruct

point = PointStruct(
    id=issue_number,           # GitHub issue number as the Qdrant point ID
    vector={
        "dense": dense_embedding,   # float32[1536] from text-embedding-3-small
    },
    sparse_vector={
        "sparse": sparse_embedding, # SparseVector from BM42
    },
    payload={
        "repo":        "owner/repo",
        "number":      1234,
        "title":       "UnhandledRejection when calling close() on disconnected socket",
        "body_excerpt": "Steps to reproduce: ...",   # first 512 chars
        "state":       "open",       # "open" | "closed"
        "labels":      ["bug", "networking"],
        "created_at":  1716000000,
        "html_url":    "https://github.com/owner/repo/issues/1234",
    }
)
```

Using the GitHub issue number directly as the Qdrant point ID removes any ID translation layer. `client.upsert()` idempotently writes or overwrites: if an issue gets edited after opening, the next webhook event re-embeds and overwrites the stale vector cleanly.

The `repo` field in the payload is the key for per-repository filtering. Because a Qdrant collection can store issues from multiple repositories, the `repo_filter` in every query restricts results to the same repository as the new issue. This filter is applied as a *pre-filter* inside the HNSW traversal, not as a post-filter on results, so the graph navigation itself only considers points matching the filter.

---

## Backfilling the Archive with Async Quantization {#async-quantization}

A new deployment starts with an empty `issues_index`. Before the sidecar is useful, every existing issue needs to be backfilled. A repo with 20k issues at roughly 1 second per embed (batched via the OpenAI embedding API) takes about three to four hours to fully index. During that window, you still want the collection queryable for the issues that are already in.

This is exactly the problem **async quantization** solves. By default, Qdrant applies quantization synchronously during indexing: each batch of upserts blocks until the INT8 representations are computed. With the `optimizer_config` set to run quantization asynchronously, the quantized vectors are built in the background while the collection stays fully queryable:

```python
from qdrant_client.models import OptimizersConfigDiff

client.update_collection(
    collection_name="issues_index",
    optimizer_config=OptimizersConfigDiff(
        indexing_threshold=20_000,   # build HNSW after 20k points
    ),
)
```

During the backfill, Qdrant serves queries against unquantized vectors for the in-progress segment and quantized vectors for completed segments. Recall is slightly lower on the unquantized segments (because the HNSW graph is still being built), but the collection never goes offline and never serves degraded results past the first few minutes. Once the backfill completes and the optimizer catches up, all segments are quantized and the query path is consistent.

The `indexing_threshold=20_000` setting tells Qdrant not to build the HNSW index for a segment until it contains 20,000 points. Below that threshold, Qdrant uses a flat brute-force search on the segment, which is fast enough for small collections and avoids repeatedly rebuilding the graph during active ingestion. Tune this value based on expected collection size: for a 100k-issue repo, a threshold of 10,000 to 20,000 is reasonable.

The backfill script itself is straightforward:

```python
import asyncio
import httpx
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.models import PointStruct, SparseVector

dense_model  = TextEmbedding("BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

async def backfill_repo(owner: str, repo: str, github_token: str):
    headers = {"Authorization": f"Bearer {github_token}"}
    page = 1

    async with httpx.AsyncClient() as http:
        while True:
            resp = await http.get(
                f"https://api.github.com/repos/{owner}/{repo}/issues",
                params={"state": "all", "per_page": 100, "page": page},
                headers=headers,
            )
            issues = resp.json()
            if not issues:
                break

            texts = [
                f"{i['title']} {(i['body'] or '')[:512]}"
                for i in issues
            ]

            dense_vecs  = list(dense_model.embed(texts))
            sparse_vecs = list(sparse_model.embed(texts))

            points = [
                PointStruct(
                    id=issue["number"],
                    vector={"dense": dense_vecs[idx].tolist()},
                    sparse_vector={
                        "sparse": SparseVector(
                            indices=sparse_vecs[idx].indices.tolist(),
                            values=sparse_vecs[idx].values.tolist(),
                        )
                    },
                    payload={
                        "repo":         f"{owner}/{repo}",
                        "number":       issue["number"],
                        "title":        issue["title"],
                        "body_excerpt": (issue["body"] or "")[:512],
                        "state":        issue["state"],
                        "labels":       [l["name"] for l in issue["labels"]],
                        "created_at":   issue["created_at"],
                        "html_url":     issue["html_url"],
                    },
                )
                for idx, issue in enumerate(issues)
            ]

            client.upsert(collection_name="issues_index", points=points)
            page += 1
            await asyncio.sleep(0.1)   # stay within GitHub rate limits
```

---

## The Webhook Handler {#webhook-handler}

The FastAPI handler verifies the GitHub webhook signature, extracts the issue, embeds it, and triggers the search pipeline:

```python
import hashlib
import hmac
import os

from fastapi import FastAPI, Header, HTTPException, Request
from fastembed import SparseTextEmbedding, TextEmbedding

app = FastAPI()

WEBHOOK_SECRET  = os.environ["GITHUB_WEBHOOK_SECRET"].encode()
dense_model     = TextEmbedding("BAAI/bge-small-en-v1.5")
sparse_model    = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

def _verify_signature(body: bytes, sig_header: str) -> None:
    expected = "sha256=" + hmac.new(WEBHOOK_SECRET, body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig_header):
        raise HTTPException(status_code=401, detail="Invalid signature")

@app.post("/webhook")
async def handle_webhook(
    request: Request,
    x_hub_signature_256: str = Header(...),
    x_github_event: str = Header(...),
):
    body = await request.body()
    _verify_signature(body, x_hub_signature_256)

    if x_github_event != "issues":
        return {"ok": True}

    payload = await request.json()
    if payload.get("action") != "opened":
        return {"ok": True}

    issue = payload["issue"]
    repo  = payload["repository"]["full_name"]

    await process_new_issue(issue, repo)
    return {"ok": True}
```

The HMAC verification is non-negotiable. Without it, any caller can POST arbitrary payloads to the webhook endpoint and trigger bot comments on arbitrary issues. `hmac.compare_digest` is used instead of `==` to prevent timing-based secret oracle attacks.

---

## The Hybrid Search Query {#hybrid-search}

### Why BM42 + Dense, Not Just Dense {#why-bm42}

The text being searched is GitHub issue titles and body excerpts. This surface has two distinct retrieval needs that do not overlap cleanly:

**Lexical precision:** Error codes, exception class names, function names, and exact library versions are high-signal tokens. `ERR_SOCKET_CLOSED`, `UnhandledRejection`, and `v18.12.0` are almost never synonymous with anything else. A dense model encodes these into a high-dimensional manifold where similar-sounding things cluster, which is exactly wrong here. `ERR_SOCKET_CLOSED` and `ERR_CONNECTION_RESET` are *not* the same bug.

**Semantic recall:** Symptom descriptions use natural language that varies freely. "crashes after network timeout," "fatal error on connection teardown," and "app hangs on close" can all describe the same underlying issue. Dense retrieval handles this well; BM42 alone would miss it because the token overlap between these phrases is minimal.

The hybrid query fires both vectors in a single Qdrant request and fuses the results with Reciprocal Rank Fusion (RRF).

### RRF vs. Linear Combination {#fusion-strategies}

Two fusion strategies are standard for combining ranked lists from different retrievers.

**Reciprocal Rank Fusion (RRF)** ([Cormack et al. 2009](https://dl.acm.org/doi/10.1145/1571941.1572114)) combines ranked lists from multiple retrievers by position rather than score. For a candidate $d$ appearing at rank $r_i$ in list $i$:

$$\text{RRF}(d) = \sum_{i} \frac{1}{k + r_i}$$

where $k = 60$ is the standard constant that dampens the influence of very top ranks. A rank-1 result contributes $\frac{1}{61} \approx 0.016$ to the fused score; a rank-20 result contributes $\frac{1}{80} \approx 0.012$. The key property of RRF is score-agnosticism: it does not matter that sparse BM42 dot products and dense cosine similarities are on completely different scales. Candidates that rank consistently near the top in both lists receive the highest fused scores.

**Linear combination** is the alternative: $\text{score}(d) = \alpha \cdot s_{\text{dense}}(d) + (1-\alpha) \cdot s_{\text{sparse}}(d)$. This requires that both component scores be on a comparable scale, which they are not by default. Cosine similarity is bounded to $[-1, 1]$ for L2-normalized vectors; BM42 dot products are unbounded and depend on document length and vocabulary. A linear combination without normalization would let the BM42 score dominate on longer documents and be negligible on short titles.

The practical difference between the two:

| Property | RRF | Linear combination |
|---|---|---|
| Score calibration required | No | Yes (normalize both scores) |
| Threshold on fused score | Not meaningful | Meaningful if scores are calibrated |
| Sensitivity to outlier scores | Low | High |
| Tuning required | Just $k$ (rarely changed) | $\alpha$ per use case |

For this system, RRF is the right choice. The fused score is used only for ranking; the actual duplicate confidence check runs a separate cosine similarity computation on the top-k candidates. If you wanted to use the fused score directly as a threshold, linear combination would be more appropriate, at the cost of per-repo score normalization.

To use linear combination in Qdrant, you would run two separate `Prefetch` queries and apply a custom rescoring formula in application code, or use Qdrant's `Score` query type with explicit per-prefetch weights. RRF is available natively via `FusionQuery(fusion=Fusion.RRF)`.

```python
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
)

async def search_duplicates(
    title: str,
    body: str,
    repo: str,
    new_issue_number: int,
    top_k: int = 5,
) -> list[dict]:
    text = f"{title} {body[:512]}"

    dense_vec  = next(iter(dense_model.embed([text]))).tolist()
    sparse_vec = next(iter(sparse_model.embed([text])))

    repo_filter = Filter(
        must=[FieldCondition(key="repo", match=MatchValue(value=repo))]
    )

    results = client.query_points(
        collection_name="issues_index",
        prefetch=[
            Prefetch(
                query=dense_vec,
                using="dense",
                filter=repo_filter,
                limit=20,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                filter=repo_filter,
                limit=20,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    return [
        r for r in results.points
        if r.payload["number"] != new_issue_number
    ]
```

The `repo_filter` is applied as a pre-filter inside both `Prefetch` blocks rather than as a post-filter on the final fused result. Qdrant evaluates filters during HNSW traversal, so restricting to the current repo before scoring avoids comparing issues across unrelated repositories. Two completely different projects can have superficially similar bugs that are not actionable duplicates; cross-repo results at this stage are noise, not signal.

`limit=20` per sub-query and `top_k=5` on the fused result is a deliberate oversampling pattern. RRF needs candidates from both lists to fuse meaningfully; fetching 20 from each side before fusing to 5 gives the algorithm room to promote candidates that rank consistently in both lists over candidates that rank very high in only one. A candidate in rank 3 on both dense and sparse lists gets a fused score of $\frac{1}{63} + \frac{1}{63} \approx 0.032$, beating a candidate at rank 1 on dense only ($\frac{1}{61} \approx 0.016$).

---

## Scoring and Thresholds {#scoring}

RRF returns a ranked list, not calibrated scores. A rank-1 result from RRF is the candidate that appeared most consistently near the top in both the dense and sparse sub-queries, but whether it is a genuine duplicate depends on the actual similarity, not just the rank.

To gate the bot comment, a secondary dense similarity check runs against the top-1 RRF result:

```python
DUPLICATE_THRESHOLD = 0.88
POSSIBLE_THRESHOLD  = 0.78

async def evaluate_top_match(
    new_text: str,
    candidate: dict,
) -> str | None:
    """Returns 'duplicate' | 'possible' | None."""
    candidate_text = f"{candidate['title']} {candidate['body_excerpt']}"

    new_vec       = next(iter(dense_model.embed([new_text]))).tolist()
    candidate_vec = next(iter(dense_model.embed([candidate_text]))).tolist()

    dot = sum(a * b for a, b in zip(new_vec, candidate_vec))
    # both vectors are L2-normalized by the model; dot product == cosine similarity

    if dot >= DUPLICATE_THRESHOLD:
        return "duplicate"
    if dot >= POSSIBLE_THRESHOLD:
        return "possible"
    return None
```

The two-tier threshold is intentional:

| Tier | Threshold | Bot behavior |
|---|---|---|
| `duplicate` | >= 0.88 | "This looks like a duplicate of #N. Closing." |
| `possible` | >= 0.78 | "This may be related to #N, linking for visibility." |
| None | < 0.78 | No comment posted |

The `possible` tier exists to surface likely-but-not-certain duplicates for human review without auto-closing. Auto-close only triggers above 0.88, where the false-positive rate on real issue data is low enough to be acceptable. The right thresholds for your specific repository require calibration against a labeled sample of known duplicates; these defaults are a starting point, not a production guarantee.

Why re-embed at threshold time rather than using the dense score from the original `Prefetch`? The `Prefetch` dense score is affected by HNSW approximation and possibly INT8 quantization during graph traversal. The rescored value at retrieval time is more reliable for the final top-k, but for a single binary threshold decision, a fresh exact dot product is the most defensible choice. The additional embedding call costs roughly 10 ms for a short text, which is acceptable here.

---

## The GitHub Bot Response {#github-bot}

The bot comment is posted via the GitHub REST API. The handler distinguishes the `duplicate` and `possible` tiers and formats the comment accordingly:

```python
import httpx

GITHUB_TOKEN = os.environ["GITHUB_BOT_TOKEN"]

async def post_duplicate_comment(
    repo: str,
    issue_number: int,
    match: dict,
    tier: str,
) -> None:
    match_url    = match["html_url"]
    match_title  = match["title"]
    match_number = match["number"]

    if tier == "duplicate":
        body = (
            f"**Possible duplicate detected**: this issue looks very similar to "
            f"[#{match_number}: {match_title}]({match_url}).\n\n"
            f"If it describes the same problem, please close this one and add any "
            f"new reproduction details to the original. If it is distinct, feel free to ignore this comment."
        )
    else:
        body = (
            f"**Related issue**: this may be connected to "
            f"[#{match_number}: {match_title}]({match_url}).\n\n"
            f"Linking for triage visibility."
        )

    async with httpx.AsyncClient() as http:
        await http.post(
            f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments",
            json={"body": body},
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
            },
        )
```

The comment language is deliberately hedged. The bot never asserts "this is a duplicate" with certainty; it says "looks very similar to." Developers who file a new issue are usually confident it is distinct. An aggressive bot that auto-closes on weak matches destroys trust in the system within a week. The goal is to surface the information, not to make the decision.

---

## Challenges and Open Problems {#challenges}

**Embedding drift.** The index is built with one embedding model. If you upgrade the model, existing vectors and new vectors are no longer in the same space: cosine similarity becomes meaningless across the boundary. The fix is a full reindex on model change, which the async quantization pattern handles cleanly (backfill into a new collection, swap aliases when done), but it is not free. A 20k-issue repo takes three to four hours to reindex. Tracking the model version in a separate Qdrant collection metadata field and rejecting mismatched queries at the handler level prevents silent accuracy degradation during the transition.

**Multi-repo search.** The current design is scoped to a single repository via the `repo_filter`. For a monorepo organization where the same bug might be reported across several related repos, removing the repo filter and switching to a `must_not` on a per-repo blocklist is straightforward, but the threshold calibration needs to account for the noisier cross-repo signal. Issues in adjacent repos often use the same terminology without describing the same root cause, which pushes the optimal `DUPLICATE_THRESHOLD` upward.

**Closed issue handling.** The index includes both open and closed issues. Matching against a closed issue is often the right outcome ("this was fixed in v3.2, close and link to the fix PR"), but matching against a closed issue that was closed as "won't fix" or "not reproducible" creates a different user experience. Filtering by `state: "open"` at search time (via payload filter inside the `Prefetch`) reduces noise at the cost of missing known-fixed duplicates. The right behavior depends on the project's issue hygiene conventions.

**Body quality variance.** Some reporters write detailed reproduction steps; others write three words. The body excerpt embedding is noisy for short-body issues. A fallback to title-only search when `body_excerpt` is under 50 characters is a practical mitigation. Alternatively, the sparse BM42 vector can be given higher weight relative to dense for short-body issues, because BM42's attention mechanism degrades more gracefully on sparse text than a dense model trying to embed three words into 1536 dimensions.

**Comment spam on noisy repos.** At the `possible` threshold of 0.78, a very active repo with many superficially similar bug reports will generate a comment on nearly every new issue. The threshold should be tuned per-repo, not globally, and the `possible` tier should be disabled for repos where the false-positive rate exceeds the signal value. A per-repo configuration map keyed by `"owner/repo"` is the simplest structure; persistent storage in the payload of a dedicated Qdrant point works if you want to avoid a separate configuration store.

**Quantization recall at scale.** INT8 scalar quantization with `quantile=0.99` achieves recall@100 of approximately 0.97 on well-distributed embedding spaces. At 1 million issues (for an organization-wide deployment), HNSW graph quality becomes the binding constraint: recall depends heavily on the `ef_construction` parameter used during indexing and the `ef` parameter at query time. Higher `ef` values improve recall at the cost of query latency. The current design does not expose `ef` as a tunable parameter per query, which should be remedied for scale.

The most open problem across all of the above is threshold calibration: the right values for `DUPLICATE_THRESHOLD` and `POSSIBLE_THRESHOLD` depend on the repo's issue vocabulary, reporter population, and tolerance for false positives, none of which are stable over time. A per-repo calibration loop (periodically sampling recently-closed issues labeled "duplicate" and adjusting thresholds to minimize recall loss) is a natural next step that the current design does not include.

---

## Citation

```
@misc{bm42,
  title   = {BM42: New Baseline for Hybrid Search},
  author  = {Qdrant Team},
  year    = {2024},
  url     = {https://qdrant.tech/articles/bm42/}
}

@misc{rrf,
  title   = {Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods},
  author  = {Cormack, Gordon V. and Clarke, Charles L.A. and Buettcher, Stefan},
  year    = {2009},
  url     = {https://dl.acm.org/doi/10.1145/1571941.1572114}
}
```
