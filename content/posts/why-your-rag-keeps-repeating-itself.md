---
title: "Why your RAG keeps repeating itself"
date: 2026-08-11
description: "Near-duplicate chunks crowd the top of a similarity search ranking because nothing in plain top-k retrieval penalizes redundancy. Maximal Marginal Relevance fixes it, but proving it helped needs a metric built for how an LLM actually consumes retrieved passages, not nDCG."
tags: ["qdrant", "rag", "retrieval", "evaluation"]
author: "Mihir Inamdar"
showToc: false
---

Ask a RAG system a question and you'll sometimes get back five chunks that
all say roughly the same thing. Nothing in a standard top-k retrieval
pipeline penalizes that. Similarity search finds what's closest to the
query vector, and near-duplicate chunks (the same paragraph indexed twice,
three sections of a document making the same point) sit close to each
other and close to the query at the same time. The top of the ranking
fills up with redundancy instead of coverage.

Generation doesn't catch it either. Hand an LLM five chunks that repeat one
fact and it writes a fluent, confident answer built on that one fact. The
answer looks grounded because, technically, it is, it's just grounded in a
fifth of the information the retrieval budget could have carried.

Maximal Marginal Relevance is the standard fix here, and it isn't new. It
reranks a candidate set by relevance to the query minus similarity to what
you've already selected, so the second result has to earn its spot by
adding something the first one didn't cover. Applied to a Qdrant result
set, that means over-fetching a wider candidate pool than your final k,
then selecting from it in a way that penalizes near-duplicates before the
context window ever sees them.

I went looking for public implementations of plain MMR and found almost
nothing with real usage behind it, a handful of GitHub repos in the low
double digits of stars, mostly teaching examples. MMR itself isn't hard.
Proving it actually helped is the part most of those repos skip, and it's
the part that matters more than the reranking code.

What actually took the time here wasn't the reranking function, it was
getting the vectors back out of Qdrant in the first place. `query_points`
doesn't return vectors by default, they cost bandwidth you usually don't
want on a normal search request, so the first version of this I wrote
silently got `None` back for every point's `.vector` and threw a
confusing error three lines later. `with_vectors=True` on the query is the
fix, and it's now the first thing `mmr_rerank` checks for and complains
about loudly if it's missing.

Here's a working version anyway, wired to real Qdrant `ScoredPoint` objects
instead of a toy list of vectors:

```python
import numpy as np
from qdrant_client import models


def mmr_rerank(
    query_vector: list[float],
    candidates: list[models.ScoredPoint],
    k: int,
    lambda_mult: float = 0.5,
) -> list[models.ScoredPoint]:
    """Rerank a Qdrant candidate set with Maximal Marginal Relevance.

    candidates must have been fetched with with_vectors=True, otherwise
    there's nothing here to compute similarity against. lambda_mult trades
    off relevance to the query (1.0) against diversity from what's already
    selected (0.0). 0.5 is a reasonable starting point, not a proven default.
    """
    if not candidates:
        return []

    query_vec = np.asarray(query_vector, dtype=float)
    vectors = {}
    for point in candidates:
        if point.vector is None:
            raise ValueError(
                f"Point {point.id} has no vector attached. "
                "Query with with_vectors=True."
            )
        if isinstance(point.vector, dict):
            raise ValueError(
                f"Point {point.id} carries named vectors. Pick one name's "
                "vector before calling mmr_rerank; this implementation "
                "assumes a single unnamed vector per point."
            )
        vectors[point.id] = np.asarray(point.vector, dtype=float)

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    remaining = list(candidates)
    selected: list[models.ScoredPoint] = []

    while remaining and len(selected) < k:
        best_point, best_score = None, -float("inf")
        for point in remaining:
            relevance = cosine(vectors[point.id], query_vec)
            redundancy = max(
                (cosine(vectors[point.id], vectors[s.id]) for s in selected),
                default=0.0,
            )
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            if mmr_score > best_score:
                best_point, best_score = point, mmr_score
        selected.append(best_point)
        remaining.remove(best_point)

    return selected
```

Used against a live collection, it looks like this:

```python
result = client.query_points(
    collection_name="docs",
    query=query_vector,
    limit=40,
    with_vectors=True,
)
diversified = mmr_rerank(query_vector, result.points, k=8, lambda_mult=0.5)
```

Over-fetch before reranking. `limit=40` for a final `k=8` gives MMR a wide
enough pool to actually find non-redundant candidates in; ask it to
diversify a pool of 10 for a final 8 and there's barely anything to trade
off. `lambda_mult` is the knob that decides how aggressively it diversifies:
push it toward 1.0 and you're basically back to plain similarity ranking,
push it toward 0.0 and it'll happily surface a weaker match just because
it's different from what's already picked. 0.5 is a coin flip between the
two, and it's a knob to sweep against your own data, not a constant to
trust from a blog post, this one included.

Proving any of this worked is harder than it sounds, because the obvious
metric is the wrong one. Standard IR metrics like nDCG, MAP, and MRR were
built around how a human scans a ranked list top to bottom. An LLM doesn't
scan, it reads everything in the context window at once, so a metric built
for human browsing behavior doesn't capture what actually matters for
generation quality: how much genuinely new information is in the set, not
how well-ordered it is. A [recent paper proposing UDCG](https://arxiv.org/abs/2510.21440)
makes exactly this case, arguing for a metric closer to how RAG pipelines
actually consume retrieved passages. An independent eval tool built around
Qdrant, `Evret`, has already been running experiments in this direction.

There's a related finding worth being precise about. Retrieval failure and
generation failure get conflated in most end-to-end RAG evaluation, because
faithfulness and answer-relevance scores can look fine even when context
recall has quietly dropped, since the model still sounds grounded on
whatever incomplete context it got. [A 2025 paper on this exact gap](https://arxiv.org/pdf/2507.06554)
argues retrieval quality needs its own measurement, separate from whatever
the generator does with it afterward.

Which means "just add MMR" isn't quite the takeaway. Measure redundancy in
your retrieved set directly, with a metric built for how the generator
consumes it, before and after adding diversification. Most teams skip
straight to assuming reranking helped because the answers subjectively
read better, which is the same trap as judging a compressed embedding
model by whether it loads fast instead of whether its rankings are still
correct. The failure mode that doesn't throw an error is the one that
ships unnoticed.

If you're deciding whether this is worth adding at all: it's cheap
insurance for any corpus where the same fact tends to appear in more than
one place, changelogs, FAQ-style docs, anything with revision history
sitting in the same collection. It's much less useful over a corpus that's
already deduplicated upstream. Check which one you have before reaching
for `lambda_mult` as the fix for an answer quality problem that might
actually be a data hygiene problem.
