---
title: "Multi-tenant Qdrant: the query that leaks"
date: 2026-08-06
description: "Payload partitioning, collection-per-tenant, and 1.19's tiered multitenancy all trade off differently, but the isolation boundary itself is a filter nobody enforces at the code level. A query built without it doesn't error, it just searches every tenant sharing the collection."
tags: ["qdrant", "multi-tenancy", "vector-search", "reliability"]
author: "Mihir Inamdar"
showToc: false
---

A shared vector index doesn't fail loudly when tenant isolation breaks. It just returns another customer's data, ranked and scored like it belongs there.

Qdrant gives you three ways to isolate tenants in one collection: payload partitioning with `is_tenant=true` on the field that marks ownership, a separate collection per tenant, or, as of 1.19, tiered multitenancy for accounts that don't fit either extreme cleanly. Qdrant's own internal brief calls this "one of the most common questions in enterprise sales calls," which tells you two things: teams hit this constantly, and there isn't a settled default answer yet.

The three options trade off differently, and the tradeoff isn't cosmetic.

Payload partitioning keeps every tenant's vectors in one HNSW graph. That's cheap to run and easy to add a tenant to, but it means every query traverses a graph built partly out of vectors it will never be allowed to return. `is_tenant=true` tells Qdrant to build filter-aware edges so that traversal stays mostly inside the matching tenant's data, which helps, but the isolation is still enforced at query time by a filter, not by the data being physically separate.

Collection-per-tenant is the opposite bet. Real isolation, because there's no other tenant's data in the collection to accidentally return. The cost shows up at the fleet level: thousands of small tenants means thousands of collections, each with its own HNSW overhead, and a lot of operational surface area for something as simple as "list every tenant's collection and check it's healthy."

Tiered multitenancy is Qdrant's answer to the actual shape of most real customer bases: a handful of huge tenants next to a long tail of tiny ones. Big tenants get their own collection. Small ones share one, partitioned. It's the option most teams should probably be using and the one with the least public documentation.

Here's the part that isn't a performance question. `is_tenant=true` and payload partitioning are a filter, and filters are something application code has to remember to apply. A query built without the tenant filter, whether from a bug, a copy-pasted debug script, or a background job that skips the usual query path, doesn't error. It searches across every tenant sharing that collection and returns whatever scores best. Nothing about the response shape tells you it happened.

That's the actual gap. Qdrant gives you the primitives for isolation. It doesn't give you a way to guarantee, at the code level, that every query path used them. A tenant boundary that depends on every call site remembering a filter is a tenant boundary that will eventually get skipped once, by someone, under deadline pressure.

What a real fix looks like: a thin wrapper around the Qdrant client that refuses to run a query without an explicit tenant filter, so the leak becomes an exception at call-time instead of a silent wrong answer at read-time. Here's a minimal version, built against the current qdrant-client API (`query_points`, `Filter`, `FieldCondition`, `MatchValue`, verified against Qdrant's own client docs):

```python
from __future__ import annotations

from qdrant_client import QdrantClient, models


class MissingTenantFilter(Exception):
    """Raised when a query has no explicit tenant scope."""


def tenant_safe_query(
    client: QdrantClient,
    collection_name: str,
    query: list[float],
    tenant_id: str,
    tenant_field: str = "tenant_id",
    limit: int = 10,
    additional_conditions: list[models.FieldCondition] | None = None,
):
    """Run a Qdrant vector search that cannot execute without a tenant filter.

    Raises MissingTenantFilter instead of silently searching every tenant
    sharing this collection.
    """
    if not tenant_id:
        raise MissingTenantFilter(
            "tenant_safe_query() requires a non-empty tenant_id. "
            "Refusing to run an unscoped search."
        )

    conditions = [
        models.FieldCondition(
            key=tenant_field,
            match=models.MatchValue(value=tenant_id),
        )
    ]
    if additional_conditions:
        conditions.extend(additional_conditions)

    return client.query_points(
        collection_name=collection_name,
        query=query,
        query_filter=models.Filter(must=conditions),
        limit=limit,
        with_payload=True,
    )
```

Called without a tenant_id, it raises before the request ever reaches Qdrant. Called with one, it behaves like a normal `query_points()` call, scoped:

```python
client = QdrantClient(url="http://localhost:6333")

# Raises MissingTenantFilter. Never reaches Qdrant.
tenant_safe_query(client, "documents", query_vector, tenant_id="")

# Runs, scoped to one tenant.
results = tenant_safe_query(client, "documents", query_vector, tenant_id="acme-corp")
```

It's deliberately narrow. It doesn't pick a partitioning strategy, it just makes the one filter that enforces isolation impossible to skip. Whether payload partitioning, collection-per-tenant, or the 1.19 tiered approach is the right base layer depends on tenant count and size distribution, and that's a benchmark question this piece hasn't run yet. The isolation-boundary problem is separate from that choice, and it's the one worth fixing first regardless of which partitioning strategy ends up underneath it.

Source: Qdrant's internal Stars content brief, and [Multi-Tenant Vector Search in Practice](https://kulekci.medium.com/multi-tenant-vector-search-in-practice-building-a-shared-knowledge-base-with-qdrant-7b7928ba00fe) on the shared-HNSW-graph tradeoff.
