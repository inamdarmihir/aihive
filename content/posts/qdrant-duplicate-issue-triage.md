---
title: "Duplicate Issue Detection Without the Triage Tax: A Qdrant Hybrid Search Sidecar for GitHub"
date: 2026-06-27
description: "How to build a GitHub webhook sidecar that uses Qdrant BM42 sparse-dense hybrid search and async scalar quantization to automatically surface duplicate issues before they ever reach the triage queue — with concrete collection design, scoring thresholds, and a FastAPI webhook handler."
tags: ["qdrant", "vector-search", "bm42", "hybrid-search", "github", "deduplication", "system-design", "fastapi"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Manual duplicate triage is an untracked engineering cost: on repositories with more than a few thousand open issues, GitHub's built-in title-matching catches copy-paste duplicates and misses every case where two reporters describe the same root cause in different words. In this post, I walk through building a Qdrant sidecar that runs a BM42 sparse-dense hybrid query on every `issues.opened` webhook event, surfacing likely duplicates before any human reads the new report — I will focus on collection design, the hybrid query, threshold calibration, and async archive backfill, and will not cover issue classification, priority routing, or label inference. Familiarity with vector databases and GitHub webhook integration is assumed.

---

## Table of Contents

1. [Why Exact-String Matching Fails](#why-exact-string-fails)
2. [Architecture: The Sidecar Pattern](#architecture)
3. [The Qdrant Collection Design](#collection-design)
   - [Named Vectors: Dense and Sparse Together](#named-vectors)
   - [Payload Schema](#payload-schema)
4. [Backfilling the Archive with Async Quantization](#async-quantization)
5. [The Webhook Handler](#webhook-handler)
6. [The Hybrid Search Query](#hybrid-search)
   - [Why BM42 + Dense, Not Just Dense](#why-bm42)
   - [Reciprocal Rank Fusion](#rrf)
7. [Scoring and Thresholds](#scoring)
8. [The GitHub Bot Response](#github-bot)
9. [Challenges and Open Problems](#challenges)

---

## Why Exact-String Matching Fails {#why-exact-string-fails}

GitHub already does substring and edit-distance matching on titles when you open a new issue. It works for copy-paste duplicates. The failures that cost the most time are not copy-paste duplicates — they are semantically equivalent reports written by different people who describe the same thing differently.

Consider a real failure mode: a library has a bug where calling `.close()` on an already-closed connection throws an unhandled exception. In a month, five people report it:

```
"UnhandledRejection when calling close() on disconnected socket"
"app crashes after network timeout — close() throws"
"TypeError: Cannot read property 'destroy' of null"
"Fatal error on socket close after idle timeout"
"Crash in connection pool teardown — reproducible"
```

These share no useful n-grams. The error messages differ because the exact stack trace varies by Node version. The symptom descriptions differ because users observe the crash at different call sites. GitHub's similarity suggestions, which operate on the title text, surface none of them to each other.

A dense embedding of each title places all five within cosine distance $\approx 0.12$ of each other and $\approx 0.45$ away from unrelated issues. This is exactly what embedding-based retrieval is for.

But dense alone has a different failure mode: it loses *specific* tokens. If you search for `TypeError: Cannot read property 'destroy' of null`, a dense model will score that against "socket teardown failure" as high similarity — because both are about connection cleanup. That is the right behavior for semantic dedup. It is the *wrong* behavior when a developer specifically wants to find the exact error message — to verify the fix applies to their specific stack, or to cross-reference a known CVE.

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

`always_ram=True` on the scalar quantization config means the compressed INT8 vectors stay in memory for fast ANN scoring, while the full float32 originals (used for rescoring) can page to disk. For a collection of 100k issues, the INT8 layer runs at roughly 150 MB RAM versus 600 MB for full precision — a 4× reduction that keeps the sidecar hostable on a single small instance.

The `on_disk_payload=True` flag moves the text payload (titles, bodies, URLs) to disk. Payloads are only needed on the final candidates returned after scoring; they are never accessed during the ANN traversal itself. This is a free memory saving for large archives.

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

Using the GitHub issue number directly as the Qdrant point ID removes any ID translation layer. `client.upsert()` idempotently writes or overwrites — if an issue gets edited after opening, the next webhook event re-embeds and overwrites the stale vector cleanly.

---

## Backfilling the Archive with Async Quantization {#async-quantization}

A new deployment starts with an empty `issues_index`. Before the sidecar is useful, every existing issue needs to be backfilled. A repo with 20k issues at ~1 second per embed (batched via the OpenAI embedding API) takes about three to four hours to fully index. During that window, you still want the collection queryable for the issues that are already in.

This is exactly the problem **async quantization** solves. By default, Qdrant applies quantization synchronously during indexing — each batch of upserts blocks until the INT8 representations are computed. With the `optimizer_config` set to run quantization asynchronously, the quantized vectors are built in the background while the collection stays fully queryable:

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

The text being searched is GitHub issue titles and body excerpts. This surface has two distinct retrieval needs that don't overlap cleanly:

**Lexical precision:** Error codes, exception class names, function names, and exact library versions are high-signal tokens. `ERR_SOCKET_CLOSED`, `UnhandledRejection`, and `v18.12.0` are almost never synonymous with anything else. A dense model encodes these into a high-dimensional manifold where similar-sounding things cluster — which is exactly wrong here. `ERR_SOCKET_CLOSED` and `ERR_CONNECTION_RESET` are *not* the same bug.

**Semantic recall:** Symptom descriptions use natural language that varies freely. "crashes after network timeout," "fatal error on connection teardown," and "app hangs on close" can all describe the same underlying issue. Dense retrieval handles this well; BM42 alone would miss it.

The hybrid query fires both vectors in a single Qdrant request and fuses the results with Reciprocal Rank Fusion (RRF).

### Reciprocal Rank Fusion {#rrf}

RRF combines ranked lists from multiple retrievers by position rather than score. For a candidate $d$ appearing at rank $r_i$ in list $i$:

$$\text{RRF}(d) = \sum_{i} \frac{1}{k + r_i}$$

where $k = 60$ is the standard constant that dampens the influence of very top ranks. The key property: RRF is score-agnostic. It doesn't matter that sparse and dense scores are not on the same scale — they never need to be compared directly.

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

The `repo_filter` is applied as a pre-filter inside both `Prefetch` blocks rather than as a post-filter on the final fused result. Qdrant evaluates filters during HNSW traversal, so restricting to the current repo before scoring avoids comparing issues across unrelated repositories — two completely different projects can have superficially similar bugs that are not actionable duplicates.

`limit=20` per sub-query and `top_k=5` on the fused result is a deliberate oversampling pattern. RRF needs candidates from both lists to fuse meaningfully; fetching 20 from each side before fusing to 5 gives the algorithm room to promote candidates that rank consistently in both lists over candidates that rank very high in only one.

---

## Scoring and Thresholds {#scoring}

RRF returns a ranked list, not calibrated scores. A rank-1 result from RRF is the candidate that appeared most consistently near the top in both the dense and sparse sub-queries — but whether it is a genuine duplicate depends on the actual similarity, not just the rank.

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
| `duplicate` | ≥ 0.88 | "This looks like a duplicate of #N. Closing." |
| `possible` | ≥ 0.78 | "This may be related to #N — linking for visibility." |
| None | < 0.78 | No comment posted |

The `possible` tier exists to surface likely-but-not-certain duplicates for human review without auto-closing. Auto-close only triggers above 0.88, where the false-positive rate on real issue data is low enough to be acceptable. The right thresholds for your specific repository require calibration against a labeled sample of known duplicates — these defaults are a starting point, not a production guarantee.

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
            f"**Possible duplicate detected** — this issue looks very similar to "
            f"[#{match_number}: {match_title}]({match_url}).\n\n"
            f"If it describes the same problem, please close this one and add any "
            f"new reproduction details to the original. If it is distinct, feel free to ignore this comment."
        )
    else:
        body = (
            f"**Related issue** — this may be connected to "
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

The comment language is deliberately hedged. The bot never says "this is a duplicate" with certainty — it says "looks very similar to." Developers who file a new issue are usually confident it is distinct; an aggressive bot that auto-closes on weak matches destroys trust in the system within a week. The goal is to surface the information, not to make the decision.

---

## Challenges and Open Problems {#challenges}

**Embedding drift.** The index is built with one embedding model. If you upgrade the model, existing vectors and new vectors are no longer in the same space — cosine similarity becomes meaningless across the boundary. The fix is a full reindex on model change, which the async quantization pattern handles cleanly (backfill into a new collection, swap aliases when done), but it is not free.

**Multi-repo search.** The current design is scoped to a single repository via the `repo_filter`. For a monorepo organization where the same bug might be reported across several related repos, removing the repo filter and switching to a `must_not` on a per-repo blocklist is straightforward — but the threshold calibration needs to account for the noisier cross-repo signal.

**Closed issue handling.** The index includes both open and closed issues. Matching against a closed issue is often the right outcome — "this was fixed in v3.2, close and link to the fix PR." But matching against a closed issue that was closed as "won't fix" or "not reproducible" creates a different user experience. Filtering by `state: "open"` at search time reduces noise at the cost of missing known-fixed duplicates.

**Body quality variance.** Some reporters write detailed reproduction steps; others write three words. The body excerpt embedding is noise for short-body issues. A fallback to title-only search when `body_excerpt` is under 50 characters is a practical mitigation.

**Comment spam on noisy repos.** At the `possible` threshold of 0.78, a very active repo with many superficially similar bug reports will generate a comment on nearly every new issue. The threshold should be tuned per-repo, not globally, and the `possible` tier should be disabled for repos where the false-positive rate exceeds the signal value.

The most open problem across all of the above is threshold calibration: the right values for `DUPLICATE_THRESHOLD` and `POSSIBLE_THRESHOLD` depend on the repo's issue vocabulary, reporter population, and tolerance for false positives, none of which are stable over time. A per-repo calibration loop — periodically sampling recently-closed issues labeled "duplicate" and adjusting thresholds to minimize recall loss — is a natural next step that the current design does not include.

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
