---
title: "Migrating to Qdrant without losing recall"
date: 2026-08-06
description: "Migration guides check whether the vectors moved. They don't check whether the destination still finds the same things the source did. A recall-parity check against a sample of real production queries closes that gap before cutover, not after."
tags: ["qdrant", "pgvector", "migration", "vector-search"]
author: "Mihir Inamdar"
showToc: false
---

Most migration guides check one thing: did the vectors move. That's not
really the question that matters. The question is whether the destination
still finds the same things the source did.

pgvector is the most common thing teams migrate off of, and it fails in
specific, documented ways once a collection grows. Its HNSW index caps out
at 2000 dimensions (4000 if you switch to `halfvec`), which already rules
it out for a lot of current embedding models. Past that,
[ParadeDB's writeup on pgvector's limitations](https://www.paradedb.com/learn/postgresql/pgvector-limitations)
and a widely discussed [Hacker News thread on "the case against pgvector"](https://news.ycombinator.com/item?id=45798479)
both point to the same pattern: HNSW memory usage climbs fast once you're
past a few million vectors, and there's no native sharding to spread that
load. None of this is a pgvector bug, really. It's Postgres doing exactly
what a general-purpose database does when you ask it to do a specialized
one's job at scale.

So teams migrate. Snapshot the source, batch-insert into Qdrant, recreate
indexes, cut traffic over. Every step of that is checkable: point counts
match, index build finishes, the new endpoint responds. What none of those
steps check is whether a real query, one a real user would actually type,
returns the same top results on the new system as it did on the old one.

It usually doesn't, exactly. Different HNSW parameters, different
quantization defaults, different distance metric edge cases, any of these
can shift what lands in the top 10 without anything in the migration
process throwing an error. A migration can pass every operational check
and still quietly get worse at the one job the database exists to do.

Getting a representative sample of production queries is the part that
actually takes effort here, more than writing the parity-check code does.
Logging raw query vectors in production isn't something most teams already
do, since there's normally no reason to keep them around after the search
returns. If you're planning a migration, that's worth turning on weeks
ahead of cutover, not the day before, so the sample you're diffing against
reflects real query traffic and not whatever you can synthesize the night
before the cutover meeting.

The fix isn't complicated, it's just a step nobody adds by default. Before
cutover, sample a real slice of production queries, run them against both
the source and the destination, and diff the top-k results. Overlap in the
high 90s on a representative sample is a real signal the migration
preserved behavior. Overlap that's noticeably lower is worth investigating
before the old system gets decommissioned, not after.

Here's a skeleton for that check. `query_qdrant`, `top_k_overlap`,
`run_parity_check`, and `summarize` are complete, real code, verified
against the current qdrant-client API. `query_source` is a stub, and it
stays a stub here on purpose: what it does depends entirely on what you're
migrating from, raw SQL for pgvector, an SDK call for Pinecone or Weaviate,
whatever the old system takes.

```python
from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient


@dataclass
class ParityResult:
    query_id: str
    source_ids: list[str]
    qdrant_ids: list[str]
    overlap: float


def query_qdrant(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    k: int = 10,
) -> list[str]:
    """Real, complete. Uses query_points() and returns ranked point IDs."""
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=k,
        with_payload=False,
    )
    return [str(point.id) for point in response.points]


def query_source(query_vector: list[float], k: int = 10) -> list[str]:
    """Fill in your source system's query here.

    pgvector: SELECT id FROM table ORDER BY embedding <-> %s LIMIT %s
    Pinecone: index.query(vector=query_vector, top_k=k).matches
    Whatever it returns, normalize to a ranked list of string IDs.
    """
    raise NotImplementedError("Implement against your source system.")


def top_k_overlap(source_ids: list[str], qdrant_ids: list[str]) -> float:
    """Real, complete. Fraction of the source's top-k that also shows up
    in Qdrant's top-k. Checks set membership, not rank order: ties can
    reorder an identical result set without changing what was retrieved."""
    if not source_ids:
        return 0.0
    return len(set(source_ids) & set(qdrant_ids)) / len(source_ids)


def run_parity_check(
    client: QdrantClient,
    collection_name: str,
    sample_queries: list[tuple[str, list[float]]],
    k: int = 10,
) -> list[ParityResult]:
    """Real, complete. sample_queries: (query_id, query_vector) pairs
    pulled from real production traffic, not synthetic queries."""
    results = []
    for query_id, vector in sample_queries:
        source_ids = query_source(vector, k=k)
        qdrant_ids = query_qdrant(client, collection_name, vector, k=k)
        results.append(
            ParityResult(
                query_id=query_id,
                source_ids=source_ids,
                qdrant_ids=qdrant_ids,
                overlap=top_k_overlap(source_ids, qdrant_ids),
            )
        )
    return results


def summarize(results: list[ParityResult]) -> None:
    overlaps = [r.overlap for r in results]
    print(f"Sampled {len(results)} queries")
    print(f"Mean top-k overlap: {sum(overlaps) / len(overlaps):.1%}")
    print(f"Worst case: {min(overlaps):.1%}")
```

Point `run_parity_check` at a sample of real production queries and
`summarize()` gives you one number to look at before cutover, not after.

I'd treat this closer to how you'd treat a model upgrade than how migration
guides currently treat a database swap. Nobody would ship a new model
version without checking its outputs against the old one on a held-out
set. A vector database migration changes the actual retrieval behavior of
every RAG or search system built on top of it, which makes it exactly that
kind of change, even though the tooling around it treats it like a plain
data transfer.

What's not in this piece: a real overlap number from an actual
pgvector-to-Qdrant migration, run against a real dataset instead of
described in the abstract. I haven't run that migration myself yet, so I'm
not going to make up what the number would be.

If I were running one, this parity check is where I'd want the go/no-go
gate to actually live, not in a runbook step that says "verify results
look correct," which is the kind of instruction that gets a quick eyeball
and a shrug under deadline pressure. A number in a CI job that has to
clear some threshold before cutover proceeds is a much harder thing to
skip than a checklist item.
