---
title: "Qdrant's hybrid search docs stop at \"sweep it yourself\""
date: 2026-08-18
description: "Qdrant's own tuning guide hands you the RRF-vs-DBSF sweep and admits there's no script to run it for you. A second set of failure modes, a missing IDF modifier, a wrong avg_len, fusion placed wrong relative to sharding, doesn't throw an error either. A preflight check for the checkable subset of it."
tags: ["qdrant", "hybrid-search", "rrf", "vector-search"]
author: "Mihir Inamdar"
showToc: false
---

I went looking for a script to sweep RRF against DBSF in Qdrant and found
something more useful than a script: Qdrant's own article on the topic,
[How to Tune Hybrid Search in Qdrant](https://qdrant.tech/articles/how-to-tune-hybrid-search/),
just telling you straight up that no such script exists. It walks through
how to sweep RRF against DBSF, how to sweep the `k` constant, and how to
sweep fusion weights once `k` is settled, Python snippets included for each
step. What it doesn't give you is something that runs the sweep for you.
That's the vendor's own documentation saying so, not me guessing.

The findings underneath are worth knowing regardless. Across five public
benchmark datasets, hybrid retrieval beat both pure dense and pure sparse
search on four of them, by 2 to 4 percent on nDCG@10. DBSF beat default RRF
on three of the five. And `k` behaves differently depending on how many
relevant documents a typical query actually has: around one relevant
document per query favored `k` at 2 or 5, dozens or hundreds favored `k` at
20 or 61. None of that is intuitive, and none of it transfers to a
different dataset without checking your own.

There's a second Qdrant article, from the same team, about a shorter and
meaner list: settings that fail silently. A sparse vector missing its IDF
modifier treats a rare, specific term the same as a common one, and nothing
errors. BM25's `avg_len` defaults to 256 in Qdrant; across the same five
datasets, the actual correct value ranged from 35.3 to 151.4, meaning the
default overestimated typical document length by 15 to 43 percent, every
single time. That doesn't throw an error either. It just quietly changes
every score you get back.

Going through both articles side by side took longer than I expected,
mostly because the tuning guide and the silent-failures piece are solving
adjacent but different problems, one is "which fusion setting wins," the
other is "which settings quietly corrupt the comparison you're making
between fusion settings." You need both pieces read together before a
sweep result means anything, and I don't think that's obvious from either
article on its own.

The sharding case is the one I'd most easily have missed. Fusion placed at
the query root runs once, across the full result set, after all shards
return candidates, which is what most people assume happens. Fusion nested
inside a prefetch instead runs once per shard, combining only that shard's
local candidates. Which ranking you get depends on shard count, and nothing
in the response tells you which one you got.

None of this throws an error. It shows up later as an A/B test result that
looks real and isn't. With 25 labeled queries, the 95 percent confidence
interval on a fusion-tuning gain was wider than most of the gains being
measured, meaning a lot of "hybrid search improved things by 2 percent"
claims out there are statistically unresolved, not confirmed.

Qdrant published the method and the caveats clearly. What's missing is
something that checks a live collection against this exact list before
anyone trusts what a fusion sweep hands back. So here's a start, and I want
to be upfront about which of the four it can actually see and which it
can't:

```python
from qdrant_client import QdrantClient, models


def preflight_hybrid_search(
    client: QdrantClient,
    collection_name: str,
    sparse_vector_name: str,
    query_request: models.QueryRequest | None = None,
    bm25_avg_len: float | None = None,
    measured_avg_len: float | None = None,
    labeled_query_count: int | None = None,
) -> list[str]:
    """Flag the checkable subset of the four silent hybrid-search failures.

    Only the IDF modifier is something a live collection's config actually
    exposes. The other three are not collection settings Qdrant's API can
    hand back, so this function is honest about needing extra input for
    them instead of pretending to detect them from the collection alone.
    """
    warnings: list[str] = []

    # 1. IDF modifier: genuinely inspectable, this is real collection config.
    info = client.get_collection(collection_name)
    sparse_config = info.config.params.sparse_vectors or {}
    params = sparse_config.get(sparse_vector_name)
    if params is None:
        warnings.append(
            f"No sparse vector named '{sparse_vector_name}' on '{collection_name}'."
        )
    elif params.modifier != models.Modifier.IDF:
        warnings.append(
            f"Sparse vector '{sparse_vector_name}' has modifier={params.modifier!r}, "
            "not Modifier.IDF. Rare terms are scored the same as common ones."
        )

    # 2. BM25 avg_len: NOT stored on the collection. It's a parameter to
    # whatever encoded the sparse vectors client-side (fastembed's Bm25
    # class, default 256), and Qdrant's API has no endpoint that returns
    # it after the fact. This can only be checked if the caller supplies
    # both the value that was actually used and a real measured average.
    if bm25_avg_len is None:
        warnings.append(
            "avg_len can't be read back from a live collection at all. "
            "Pass bm25_avg_len= with the value your encoder used if you "
            "want this checked."
        )
    elif measured_avg_len:
        drift = abs(bm25_avg_len - measured_avg_len) / measured_avg_len
        if drift > 0.15:
            warnings.append(
                f"bm25_avg_len={bm25_avg_len} is {drift:.0%} off the measured "
                f"corpus average of {measured_avg_len}."
            )

    # 3. Fusion placement: not a collection setting either, it's a property
    # of the query object itself. Checkable only by inspecting the actual
    # request being built, not by asking the collection anything.
    if query_request is not None and query_request.prefetch:
        prefetches = (
            query_request.prefetch
            if isinstance(query_request.prefetch, list)
            else [query_request.prefetch]
        )
        for p in prefetches:
            if getattr(p, "prefetch", None):
                warnings.append(
                    "A prefetch entry contains its own nested prefetch list. "
                    "If a FusionQuery lives at that inner level, fusion is "
                    "running once per shard, not once across the full result set."
                )

    # 4. Labeled query set size: not a Qdrant setting at all, this is an
    # evaluation methodology question. Flagged as a heuristic, not detected.
    if labeled_query_count is not None and labeled_query_count < 50:
        warnings.append(
            f"Only {labeled_query_count} labeled queries. Qdrant's own tuning "
            "benchmark saw confidence intervals wider than the measured gain "
            "at 25. Below roughly 50, treat any fusion-tuning result as unresolved."
        )

    return warnings
```

Two of the four get a real programmatic answer: the IDF modifier, because
it's config Qdrant actually stores and `get_collection` returns; the
fusion nesting, because it's structure in a Python object you already
built and can walk. The other two, `avg_len` and labeled-set size, were
never going to be things the Qdrant API exposes, they live outside the
database entirely, in the encoder config and your eval harness. A
preflight check that pretended otherwise would just be one more silent
failure in a post about silent failures.

In practice I'd run `preflight_hybrid_search` as a pre-merge check any
time someone touches fusion config, not just once before the initial
sweep. Config drifts, someone changes the sparse encoder's defaults months
later, and the IDF modifier check especially is cheap enough to run on
every deploy. The other three warnings need a human to supply numbers,
which means they're a checklist item for whoever's running the sweep, not
something CI can catch on its own.

Sources: [Qdrant, How to Tune Hybrid Search](https://qdrant.tech/articles/how-to-tune-hybrid-search/); Qdrant team posts on silent-failure settings in hybrid search configuration.
