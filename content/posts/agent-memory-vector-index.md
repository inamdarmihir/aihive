---
title: "What agent memory actually does to a vector index"
date: 2026-08-27
description: "AI agents write to memory constantly and read from it rarely, the opposite of a batch-loaded RAG corpus. We measured what that write pattern does to a real Qdrant collection's recall, latency, and footprint against LoCoMo's real long-term dialogue data. The honest answer, at this scale: nothing."
tags: ["qdrant", "agent-memory", "vector-search", "benchmarking"]
author: "Mihir Inamdar"
showToc: false
---

AI agents write to their memory constantly and read from it rarely,
compared to the batch-loaded corpora every RAG benchmark measures. Tools
like Mem0, Letta, and Qdrant's own local-first `engram` all generate the
same pattern: small, continuous writes, interleaved with reads, plus
periodic re-writes as an agent updates or confirms something it already
stored. No public benchmark had measured what that write pattern actually
does to a vector index's recall, latency, or footprint, as distinct from
whatever a specific memory framework's design contributes. We measured it
directly at the Qdrant layer, and the honest answer, at least at the scale
we tested, is: nothing.

## The setup

Real data: [LoCoMo](https://arxiv.org/abs/2402.17753), ten long-term
dialogues with 5,882 real conversational turns and a frozen, held-out
sample of 300 real grounded questions. Same embedder, same 5,882 vectors,
two Qdrant collections, only the write pattern differs:

- **Batch**: one `upsert()` call, all 5,882 points, the way a RAG corpus
  normally loads.
- **Streamed**: the same points, same order, upserted in batches of 1-5
  (roughly agent-memory write granularity), with a real query run and a
  footprint snapshot every 500 records, plus a real re-upsert on 5% of
  writes, simulating an agent re-confirming a memory it already stored.

## The result

| | Batch | Streamed |
|---|---:|---:|
| recall@1 | 0.2433 | 0.2433 |
| recall@10 | 0.5667 | 0.5667 |
| query p50 | 3.575 ms | 3.680 ms |
| disk | 136,519,258 B | 136,493,733 B |
| memory resident | 185,532,416 B | 192,544,768 B |
| write time | 0.52s (1 call) | 18.83s (1,700+ calls) |

Recall and disk are effectively identical. The latency and memory deltas
are inside normal run-to-run noise, not a pattern. Full results, every
checkpoint, and reproduction steps:
[qdrant-agent-memory-ingest-bench](https://github.com/inamdarmihir/qdrant-agent-memory-ingest-bench).

The more interesting number is the trajectory during the streamed write,
not just the endpoint. Recall climbs steadily from 0.0233 at 502 records
written to 0.2433 at the full 5,882, which is expected and correct: a
question's grounding evidence can't be retrieved before it's been written.
What matters is query latency across that same climb: 2.951 ms at 502
records, 3.759 ms at 5,501, 3.680 ms at the end. Flat, not degrading, from
the first checkpoint to the last.

## Why this is a real finding, not a non-finding

The honest hypothesis going in was that streamed, agent-style writes would
measurably cost something past some point, the way index fragmentation or
segment sprawl costs something in other systems. That didn't hold here.
Qdrant's segment architecture appears to absorb continuous small-batch
writes, interleaved reads, and periodic re-upserts without the recall,
latency, or footprint penalty that pattern might reasonably be expected to
carry. That's worth publishing as-is: a benchmark that assumed a cost and
found none is still a real answer to the question every agent-memory tool
builder is implicitly asking.

It's not a blanket "write pattern never matters" result, though. Segment
count was pinned to 1 on both collections deliberately, to isolate the
write-pattern variable from a segment-count variable. Whether the same
parity holds under Qdrant's default multi-segment optimizer, or at 100k+
records instead of 5,882, is untested here and is the natural next
question. Checkpoint reads also ran between write batches rather than on a
genuinely concurrent thread, so this measures the state after streamed
writes settle, not what a reader sees mid-write. Both are real scope
boundaries on this result, not asterisks to gloss over.

## What this means for building agent memory on Qdrant

If your agent's memory system writes the way these two collections modeled
(small batches, occasional re-writes, real read traffic in between), the
write pattern itself isn't the thing to worry about, at least not at
single-digit-thousands of records. Whatever costs an agent-memory system
does incur are more likely coming from the memory framework's own design
choices, embedding quality, or retrieval logic, not from how Qdrant
handles the write traffic underneath it.

Sources: [LoCoMo](https://arxiv.org/abs/2402.17753) · [qdrant-agent-memory-ingest-bench](https://github.com/inamdarmihir/qdrant-agent-memory-ingest-bench), full method, checkpoint data, and reproduction steps.
