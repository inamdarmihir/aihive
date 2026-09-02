---
title: "Server-side score boosting: skip the re-ranking microservice"
date: 2026-08-21
description: "A lot of production search stacks run a separate re-ranking service for business-rule boosting Qdrant already supports natively as a formula query. Recency decay, geo boost, and inventory boost, expressed server-side, without the network hop."
tags: ["qdrant", "reranking", "search-relevance", "vector-search"]
author: "Mihir Inamdar"
showToc: false
---

A lot of production search stacks run a second service whose entire job is
taking the results the vector database already ranked and re-ranking them
again based on business logic: recency, distance from the user, whether the
item is in stock. That's a network hop, a deploy, and an on-call rotation
for logic that could just run inside the query that already touched the
data once.

Qdrant supports formula queries, score boosting expressed as an arithmetic
combination of the vector similarity score and payload fields, evaluated
server-side as part of the search itself. Recency decay, geo boost,
inventory boost, and arbitrary combinations of business rules are all
expressible this way, no shipping results out to a separate process to get
re-scored.

Here's recency decay specifically, boosting the vector score with an
exponential falloff over a `published_at` payload field, using the actual
`FormulaQuery` syntax from [Qdrant's hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/) and [search relevance](https://qdrant.tech/documentation/search/search-relevance/) docs:

```python
from datetime import datetime, timezone
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

results = client.query_points(
    collection_name="articles",
    prefetch=models.Prefetch(
        query=query_vector,
        limit=50,
    ),
    query=models.FormulaQuery(
        formula=models.SumExpression(
            sum=[
                "$score",
                models.MultExpression(
                    mult=[
                        0.3,
                        models.ExpDecayExpression(
                            exp_decay=models.DecayParamsExpression(
                                x=models.DatetimeKeyExpression(datetime_key="published_at"),
                                target=models.DatetimeExpression(datetime=now),
                                scale=60 * 60 * 24 * 30,  # 30 days, in seconds
                                midpoint=0.5,
                            )
                        ),
                    ]
                ),
            ]
        )
    ),
    limit=10,
)
```

`"$score"` is the literal string Qdrant's formula language uses to
reference the prefetch's own similarity score, not a placeholder to swap
out. `scale` and `midpoint` are what actually shape the decay curve: at
`|x - target| == scale`, the decay term equals `midpoint`, here 0.5 at 30
days out, so a 30-day-old article's recency contribution is worth half of a
brand-new one's. `published_at` has to actually be stored as an RFC 3339
datetime string in the payload for `DatetimeKeyExpression` to read it. Swap
`ExpDecayExpression` for `GaussDecayExpression` or `LinDecayExpression` for
a softer or harder falloff shape, same `DecayParamsExpression` underneath.

Wiring this into an existing service is mostly a matter of moving logic
that already exists somewhere in your reranking service into the query
itself. The `prefetch` limit matters here as much as the formula does:
`limit=50` inside prefetch versus `limit=10` on the outer query means the
formula gets 50 real candidates to re-score before trimming down to the
10 you actually return, so the boost has room to actually change the
ranking instead of just re-sorting a set that's already been cut too
short.

This isn't a niche feature, and it's underused for what it does: almost no
community content exists on it despite it being one of the more
commercially useful things Qdrant ships. That tracks with what shows up
when you actually go looking for formula query examples in the wild, a lot
of general reranking tutorials, very little on Qdrant's server-side formula
syntax specifically.

I don't think the gap is because formula queries are hard to use. I think
it's that "vector search, then a separate reranking pass" is the default
mental model most people carry over from other systems, where the vector
database genuinely can't score beyond similarity. Qdrant can, and that
changes the question from "how do we build a fast reranking service" to
"do we need a reranking service at all, for the cases formula queries
already cover."

Not every case, to be clear. A learned reranker, a cross-encoder scoring
query-document pairs directly, is still doing something formula queries
can't: judging semantic relevance beyond vector similarity. The comparison
that actually matters is narrower than "formula queries versus reranking"
in general. It's formula queries versus the specific, common case of
business-rule boosting: recency, geography, inventory, and similar
deterministic adjustments that don't need a model call, just a payload
field and an arithmetic expression.

For that narrower case, the tradeoff is concrete: one query instead of a
query plus a round trip to a separate service, one system to operate
instead of two, boosting logic that lives next to the data it depends on
instead of application code that has to fetch that data separately just to
apply the same rule. What I don't have yet is a real latency comparison,
formula query versus external reranker, run on the same dataset. Qdrant's
docs cover the syntax well. Nobody seems to have published the number that
tells you whether skipping the microservice is worth it at your actual
query volume, and that's the number I'd want a real follow-up to produce.

The case where I'd reach for this first is a content feed with a recency
requirement, articles, listings, anything where "newer, all else equal"
is a real ranking rule. Instead of a reranking pass that pulls
`published_at` back out of a database after the vector search already
ran, the formula reads the payload field that's already sitting on the
point. One fewer round trip, one fewer place for the recency logic and the
similarity logic to silently drift out of sync with each other.
