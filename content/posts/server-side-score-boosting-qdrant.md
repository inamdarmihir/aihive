---
title: "Server-side score boosting: skip the re-ranking microservice"
date: 2026-08-21
description: "A lot of production search stacks run a separate re-ranking service for business-rule boosting Qdrant already supports natively as a formula query. Recency decay, geo boost, and inventory boost, expressed server-side, without the network hop."
tags: ["qdrant", "reranking", "search-relevance", "vector-search"]
author: "Mihir Inamdar"
showToc: false
---

A lot of production search stacks have a second service whose entire job is: take the results the vector database already ranked, and re-rank them again based on business logic. Recency, distance from the user, whether the item is in stock. That's a network hop, a deploy, and an on-call rotation for logic that could run inside the query that already touched the data once.

Qdrant supports formula queries: score boosting expressed as an arithmetic combination of the vector similarity score and payload fields, evaluated server-side as part of the search itself. Recency decay, geo boost, inventory boost, and arbitrary combinations of business rules are all expressible this way, without shipping search results out to a separate process to get re-scored.

Here's recency decay specifically, boosting the vector score with an exponential falloff over a `published_at` payload field, using the actual `FormulaQuery` syntax from [Qdrant's hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/) and [search relevance](https://qdrant.tech/documentation/search/search-relevance/) docs:

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

`"$score"` is the literal string Qdrant's formula language uses to reference the prefetch's own similarity score, it's not a placeholder to swap out. `scale` and `midpoint` set the decay curve: at `|x - target| == scale`, the decay term equals `midpoint`, here 0.5 at 30 days out. `published_at` has to actually be stored as an RFC 3339 datetime string in the payload for `DatetimeKeyExpression` to read it. Swap `ExpDecayExpression` for `GaussDecayExpression` or `LinDecayExpression` for a softer or harder falloff shape, same `DecayParamsExpression` underneath.

This isn't a niche feature. Qdrant's own internal content brief flags it directly: "almost no community content exists yet despite being one of the most commercially useful features." That's a specific, checkable claim, not a vague one, and it lines up with what shows up searching for real-world formula query examples: a lot of general reranking tutorials, very little on Qdrant's server-side formula syntax specifically.

The reason this gap probably exists isn't that formula queries are hard to use. It's that the pattern of "vector search, then a separate reranking pass" is the default mental model most people bring in from other systems, where the vector database genuinely doesn't support scoring beyond similarity. Qdrant does, and that changes the architecture question from "how do we build a fast reranking service" to "do we need a reranking service at all, for the cases formula queries already cover."

Not every case. A learned reranker, a cross-encoder scoring query-document pairs directly, is still doing something formula queries can't: judging semantic relevance beyond vector similarity. The comparison that actually matters is narrower than "formula queries versus reranking" in general. It's formula queries versus the specific, common case of business-rule boosting: recency, geography, inventory, and similar deterministic adjustments that don't require a model call to compute, just a payload field and an arithmetic expression.

For that narrower case, the tradeoff is concrete: one query instead of a query plus a network round trip to a separate service, one system to operate instead of two, and boosting logic that lives next to the data it depends on instead of in application code that has to fetch that data separately to apply the same rule. What this piece doesn't have yet is a real latency comparison, formula query versus external reranker, run on the same dataset. Qdrant's own docs cover the syntax. Nobody seems to have published the number that tells you whether "skip the microservice" is worth doing for your actual query volume, and that's the number a genuinely useful follow-up would need to produce, not assume.
