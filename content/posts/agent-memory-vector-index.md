---
title: "What agent memory actually does to a vector index"
date: 2026-08-31
description: "AI agents write to memory constantly and read from it rarely, the opposite of a batch-loaded RAG corpus. I measured what that write pattern does to a real Qdrant collection's recall, latency, and footprint against LoCoMo's real long-term dialogue data. At this scale, it does basically nothing."
tags: ["qdrant", "agent-memory", "vector-search", "benchmarking"]
author: "Mihir Inamdar"
showToc: false
---

I kept noticing the same shape across Mem0, Letta, and Qdrant's own local-first
`engram`: an agent writes to memory constantly, in small batches, and reads
from it comparatively rarely. That's basically the inverse of how every RAG
benchmark builds its corpus, load everything once, query forever after. Nobody
seemed to have actually measured what that write pattern does to a vector
index's recall, latency, or footprint, separate from whatever a given memory
framework's design contributes on top. So I measured it directly at the
Qdrant layer. At the scale I tested, it does basically nothing.

## The setup

Real data: [LoCoMo](https://arxiv.org/abs/2402.17753), ten long-term
dialogues with 5,882 real conversational turns and a frozen, held-out sample
of 300 real grounded questions. Same embedder, same 5,882 vectors, two Qdrant
collections, and the only thing that differs between them is how the writes
land:

- **Batch**: one `upsert()` call, all 5,882 points, the way a RAG corpus
  normally loads.
- **Streamed**: the same points, same order, upserted in batches of 1-5
  (roughly agent-memory write granularity), with a real query run and a
  footprint snapshot every 500 records, plus a real re-upsert on 5% of
  writes, simulating an agent re-confirming a memory it already stored.

Getting the two collections genuinely comparable took more care than the
setup description makes it sound. Same distance metric, same HNSW config,
same embedder called on the same text in the same order, checkpointed at
the same 500-record marks. The only degree of freedom I wanted open was
how the writes arrived. If I'd let anything else drift, a difference in
the result table wouldn't tell me anything about write pattern specifically.

## The result

| | Batch | Streamed |
|---|---:|---:|
| recall@1 | 0.2433 | 0.2433 |
| recall@10 | 0.5667 | 0.5667 |
| query p50 | 3.575 ms | 3.680 ms |
| disk | 136,519,258 B | 136,493,733 B |
| memory resident | 185,532,416 B | 192,544,768 B |
| write time | 0.52s (1 call) | 18.83s (1,700+ calls) |

Recall and disk are basically identical. The latency and memory gaps sit
inside normal run-to-run noise, not a pattern. Full results, every
checkpoint, and reproduction steps live here:
[qdrant-agent-memory-ingest-bench](https://github.com/inamdarmihir/qdrant-agent-memory-ingest-bench).

The number I actually cared about is the trajectory during the streamed
write, not just where it lands. Recall climbs steadily from 0.0233 at 502
records written to 0.2433 at the full 5,882, which is expected: you can't
retrieve a question's grounding evidence before it's been written. What I
was watching for was query latency across that same climb, and it stayed
flat: 2.951 ms at 502 records, 3.759 ms at 5,501, 3.680 ms at the end. No
degradation from the first checkpoint to the last.

## Worth publishing even though nothing broke

I'll admit I expected to find something here. My working hypothesis going
in was that streamed, agent-style writes would cost something eventually,
the way index fragmentation or segment sprawl costs something in other
systems. It didn't hold up. Qdrant's segment
architecture seems to absorb continuous small-batch writes, interleaved
reads, and periodic re-upserts without the recall, latency, or footprint
penalty that pattern would reasonably be expected to carry. A benchmark that
assumes a cost and finds none is still a real answer, and it happens to be
the question every agent-memory tool builder is implicitly asking whether
they've said so or not.

It's not a "write pattern never matters" result, though. I pinned segment
count to 1 on both collections on purpose, to keep the write-pattern
variable isolated from a segment-count variable. Whether the same parity
holds under Qdrant's default multi-segment optimizer, or at 100k+ records
instead of 5,882, I don't know, I haven't tested it. My checkpoint reads
also ran between write batches rather than on a genuinely concurrent
thread, so this measures the state after streamed writes settle, not what a
reader sees mid-write. Both are real boundaries on what this result covers.

## What this means for building agent memory on Qdrant

If your agent's memory system writes the way these two collections modeled
(small batches, occasional re-writes, real read traffic in between), the
write pattern itself isn't the thing to worry about, at least not at
single-digit-thousands of records. Whatever costs an agent-memory system
does incur are more likely coming from the memory framework's own design
choices, embedding quality, or retrieval logic, not from how Qdrant handles
the write traffic underneath it.

Practically, that changes what I'd spend engineering time on if I were
building this today. I wouldn't add batching or write-coalescing logic in
front of Qdrant to protect it from an agent's chattiness, there's nothing
in this data suggesting Qdrant needs protecting from that, at this scale.
I'd spend that time instead on the memory framework's own decisions: what
gets written, what gets deduplicated before it ever reaches the vector
store, and how stale memories get pruned. Those are the levers that
actually move recall, not the write cadence underneath them.

Sources: [LoCoMo](https://arxiv.org/abs/2402.17753) · [qdrant-agent-memory-ingest-bench](https://github.com/inamdarmihir/qdrant-agent-memory-ingest-bench), full method, checkpoint data, and reproduction steps.
