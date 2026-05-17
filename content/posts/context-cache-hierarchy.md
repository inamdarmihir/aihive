-----

## title: “Context Windows Are a Cache Hierarchy”
date: 2026-05-17
description: “A first-principles redesign of agent memory management, treating the context window as L1 cache and Qdrant as L2, with a concrete ContextAllocator implementation in LangGraph.”
tags: [“agents”, “langgraph”, “qdrant”, “memory”, “context-management”]
author: “Mihir Inamdar”
showToc: true
math: true

Every agent system has a context budget, and every production agent runs out of it at the wrong time. The standard response is to retrieve less, summarize more, or raise the token limit. None of these are principled solutions because they treat the symptom rather than the structure.

This post reframes the problem: a context window is an L1 cache. Qdrant is L2. The problem is not retrieval; it is cache coherence. When do you evict from L1? What is your write-back policy when an agent mutates retrieved content? How do you prefetch across subgraph boundaries before a node needs data it has not asked for yet?

I will focus on the LangGraph case, building a concrete `ContextAllocator` node and `SemanticCache` class that implement the full hierarchy. The post assumes familiarity with LangGraph’s graph and state primitives and basic vector store concepts. It does not assume prior knowledge of cache design.

## The Problem with Stateless Retrieval

Current practice in LangGraph agents is a single retrieval step at the start of a run: embed the query, fetch the top-K chunks from a vector store, prepend them to the context, and proceed. Three problems follow from this.

**No eviction.** Chunks loaded at step 1 stay in context through step 20, consuming tokens regardless of whether any downstream node uses them. There is no mechanism to remove a chunk when it becomes irrelevant.

**No write-back.** When an agent reasons over a retrieved document and produces a refined answer, the new understanding exists only in the current context. It is never written back to the vector store. The next run starts from the same original chunks.

**No cross-node coordination.** In a multi-node LangGraph graph, each node that needs context retrieves independently. Two nodes fetching overlapping content waste tokens. A node that could benefit from content another node already loaded gets nothing.

**MemGPT** ([Packer et al., 2023](https://arxiv.org/abs/2310.08560)) introduced the framing of the LLM as an operating system managing a memory hierarchy, but it operates at the abstraction level of main memory and external storage, not at the level of cache coherence protocols. The work here is different: applying cache design primitives to the specific structure of a LangGraph execution graph.

## Component One: The Cache Hierarchy Model

### L1: The Context Window

The context window is fast (zero-latency access), small (bounded by model token limit), and expensive (every token costs compute). It maps directly to an L1 cache. Content in L1 is directly readable by the model without a tool call.

For a 200k-token model running a complex multi-step task, a working budget breakdown might look like:

|Segment                                  |Typical allocation|
|-----------------------------------------|------------------|
|System prompt + tool definitions         |8,000 tokens      |
|Task input                               |2,000 tokens      |
|Active working memory (retrieved context)|40,000 tokens     |
|Reasoning buffer (prior turns)           |60,000 tokens     |
|Reserved for output                      |90,000 tokens     |

The 40,000-token working memory slot is L1. It is managed by the `ContextAllocator`. Content not in this slot is either in L2 (Qdrant, retrievable in ~1 LLM turn) or L3 (cold storage, retrievable in multiple turns or via tool calls).

### L2: Qdrant as Semantic Cache

Qdrant serves as L2: larger than L1, semantically addressable, and with lookup latency of one embedding call plus one vector search. Content lives in Qdrant when it is likely to be needed but is not currently active.

The key property that makes Qdrant a cache rather than a retrieval store is payload-driven metadata. Each Qdrant point stores not just the chunk embedding and text, but cache metadata: when it was last accessed, how many times it has been accessed, which graph node loaded it, and whether it is dirty (modified by the agent since it was fetched from L3).

### L3: External Storage and Cold Retrieval

L3 is whatever the original data source is: a document store, a database, a file system. Content in L3 has no embedding index; fetching it requires a structured tool call. In the cache framing, L3 is the backing store. When L2 evicts a dirty entry, it is written back to L3. When L2 misses on a lookup, the miss handler fetches from L3 and populates L2.

## Component Two: Eviction Policy

### What Makes a Context Segment Evictable

A context segment (a retrieved chunk in L1) is a candidate for eviction when:

1. No node downstream in the current graph has declared a read dependency on it.
1. It has not been accessed in the last N node executions.
1. The context budget is above a high-watermark threshold.

Condition 1 requires that nodes declare their read dependencies at graph construction time, which I cover in Component Three. Conditions 2 and 3 are runtime signals tracked by the `ContextAllocator`.

A segment is **pinned** and not evictable when:

- It was loaded by the currently executing node.
- It is referenced in the current node’s output.
- It is marked `pin=True` by the allocator based on access frequency above a threshold.

### Priority-Weighted LRU

Standard LRU evicts the least recently used entry. For a context window, recency is not the only signal. A segment that has been accessed by three different nodes is more valuable than one accessed once, even if the single-access was more recent.

The eviction score is:

$$\text{evict_score}(s) = \frac{1}{\text{access_count}(s) \cdot w_a + \text{recency}(s) \cdot w_r}$$

Where $\text{recency}(s)$ is the inverse of turns since last access (higher is more recent), $w_a$ and $w_r$ are tunable weights (defaults: 0.6, 0.4), and higher `evict_score` means evict first.

The segment with the highest `evict_score` is evicted when L1 is above the high-watermark. Eviction is not instantaneous: the allocator batches evictions at node boundaries, not mid-execution. Evicting mid-node would require the model to regenerate its reasoning state.

### Write-Back vs. Write-Through

In CPU cache design, **write-through** immediately propagates every write to the backing store. **Write-back** defers writes until the cache line is evicted, reducing write traffic at the cost of potential data loss on failure.

For agent context, write-through is too expensive: every intermediate reasoning step would trigger an embedding call and Qdrant upsert. Write-back is the right default. A segment is marked dirty when:

- The agent generates output that explicitly references or modifies the segment’s content.
- A tool call returns data that supersedes the segment.

Dirty segments are written back to Qdrant at eviction time. The allocator detects dirtiness via a simple signal: if the segment ID appears in the assistant’s output (by reference tag, see Component Four), the segment is marked dirty and the updated version is extracted and written back on eviction.

## Component Three: Prefetching Across Subgraph Boundaries

### The Boundary Problem

In a LangGraph graph with subgraphs, each subgraph is a separate execution scope. By default, a subgraph that needs context must retrieve it from scratch. If the parent graph already loaded relevant content, the subgraph has no access to it unless it is explicitly passed through state.

This is the prefetch problem: can the allocator load content into L1 *before* the subgraph that needs it starts executing?

### Lookahead Prefetch

The allocator performs a lookahead pass at graph construction time. Each node and subgraph in the graph is annotated with a `context_hint`: a short description of what it is likely to need.

```python
@node(context_hint="merchant chargeback history, category-specific thresholds")
def risk_assessment_node(state: AgentState) -> AgentState:
    ...
```

Before a node executes, the allocator checks whether the next N nodes (lookahead depth N=2 by default) have context hints that are not already satisfied by current L1 contents. If a hint is not satisfied, the allocator issues a background retrieval from Qdrant and stages the result in a prefetch buffer. When the target node starts, the prefetch buffer flushes into L1.

The staging step matters: the content is not placed in L1 until the node that needs it begins executing. Premature loading would consume budget for a node that might not run (e.g., if a conditional edge routes elsewhere).

### Invalidation on State Mutation

A prefetched segment can become stale before the target node runs. If an intermediate node updates a state field that the prefetched content depends on, the cached segment is invalid.

The allocator tracks a dependency map at write time: when a segment is fetched in response to a query over a state field (e.g., `merchant_id`), the segment is tagged with that field name. When any node writes to that field, all segments tagged with it are invalidated: removed from the prefetch buffer and from L1 if present.

```python
class ContextAllocator:
    def invalidate_by_field(self, field_name: str) -> None:
        stale_ids = [
            seg_id for seg_id, seg in self.l1.items()
            if field_name in seg.dependency_fields
        ]
        for seg_id in stale_ids:
            self._evict(seg_id, write_back=seg.dirty)
        self.prefetch_buffer = {
            k: v for k, v in self.prefetch_buffer.items()
            if field_name not in v.dependency_fields
        }
```

## Component Four: The ContextAllocator Node

### State Schema Design

The allocator needs to track L1 contents, budget usage, access history, and dirty bits. This state lives in the LangGraph state schema, not in a separate store, so it participates in checkpointing and can be resumed after interruption.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class ContextSegment:
    id: str
    text: str
    token_count: int
    source_query: str
    access_count: int = 0
    last_accessed_turn: int = 0
    dirty: bool = False
    dependency_fields: List[str] = field(default_factory=list)
    pinned: bool = False

@dataclass
class AllocatorState:
    l1: Dict[str, ContextSegment] = field(default_factory=dict)
    prefetch_buffer: Dict[str, ContextSegment] = field(default_factory=dict)
    token_budget: int = 40000
    tokens_used: int = 0
    current_turn: int = 0
    high_watermark: float = 0.85
    low_watermark: float = 0.65
```

The full agent state includes `AllocatorState` as a nested field alongside the task-specific fields.

### The Allocator Node

The `ContextAllocator` runs as a node at two points in the graph: before the first task node (initial load) and at each node boundary (eviction + prefetch). It does not run inside subgraphs; the parent graph passes a resolved L1 snapshot to subgraph inputs.

```python
def context_allocator_node(state: AgentState) -> AgentState:
    alloc = state.allocator

    # Flush prefetch buffer for current node
    alloc = flush_prefetch(alloc, state.current_node)

    # Run eviction if above high watermark
    if alloc.tokens_used / alloc.token_budget > alloc.high_watermark:
        alloc = evict_to_watermark(alloc, target=alloc.low_watermark)

    # Issue prefetch for next N nodes
    next_nodes = get_next_nodes(state.graph, state.current_node, lookahead=2)
    alloc = schedule_prefetch(alloc, next_nodes, state)

    # Increment turn counter
    alloc = alloc.replace(current_turn=alloc.current_turn + 1)

    return state.replace(allocator=alloc)
```

### Budget Accounting

The allocator tracks token counts at segment granularity, not character count. Segment token counts are computed at load time using `tiktoken` or the model’s native tokenizer. This is slightly expensive but prevents budget overruns from rough estimates.

When the allocator needs to evict, it computes `evict_score` for all non-pinned segments and evicts in order until `tokens_used` drops below `low_watermark * token_budget`. The low watermark is set below the high watermark to create hysteresis, avoiding repeated single-segment evict-and-refetch cycles.

## Component Five: SemanticCache in Qdrant

### Collection Design

Each Qdrant collection corresponds to one task domain. A segment stored in Qdrant has the following payload structure:

```json
{
  "text": "...",
  "source_url": "...",
  "source_type": "document | tool_output | agent_derived",
  "token_count": 412,
  "access_count": 7,
  "last_accessed": "2026-05-14T10:22:00Z",
  "dirty": false,
  "dependency_fields": ["merchant_id", "assessment_date"],
  "agent_version": "merchant-risk-v7"
}
```

The `source_type` field distinguishes original source content from content derived by the agent during a prior run. Agent-derived content is the write-back product: summaries, extracted entities, corrected analyses. It is stored alongside source content but retrievable separately via a payload filter.

### Eviction as a Qdrant Payload Filter

When the allocator evicts a segment from L1 and writes it back to Qdrant, it upserts the point with updated metadata: incremented `access_count`, updated `last_accessed`, and `dirty=false` (since it has just been written back).

When the L2 cache itself needs to evict (Qdrant has a configured max collection size), it runs a scheduled background job that deletes points with the lowest access counts and oldest `last_accessed` timestamps. This is implemented as a Qdrant scroll + delete with a payload filter:

```python
def evict_l2(client: QdrantClient, collection: str, target_size: int) -> None:
    current_count = client.count(collection).count
    if current_count <= target_size:
        return

    evict_count = current_count - target_size
    results, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="dirty", match=MatchValue(value=False))]
        ),
        with_payload=True,
        limit=evict_count * 2,
        order_by=OrderBy(key="access_count", direction="asc")
    )

    ids_to_delete = [r.id for r in results[:evict_count]]
    client.delete(collection_name=collection, points_selector=PointIdsList(points=ids_to_delete))
```

Dirty segments are excluded from L2 eviction by the payload filter. They must be written back to L3 before they can be evicted from Qdrant.

### Write-Back on Dirty Segments

A dirty segment in Qdrant represents agent-derived knowledge that has not yet been persisted to L3. The write-back handler runs asynchronously, triggered when a dirty segment is evicted from Qdrant:

```python
async def writeback_handler(segment: QdrantPoint, l3_client: L3Client) -> None:
    if not segment.payload.get("dirty"):
        return

    await l3_client.upsert(
        source_url=segment.payload["source_url"],
        content=segment.payload["text"],
        derived=segment.payload["source_type"] == "agent_derived",
        metadata=segment.payload
    )

    client.set_payload(
        collection_name=collection,
        payload={"dirty": False},
        points=[segment.id]
    )
```

The async pattern matters here: write-back should not block the agent’s next retrieval. If the write-back fails (L3 unavailable), the segment remains dirty in Qdrant and the handler retries on the next eviction cycle.

## End-to-End Data Flow

A complete request cycle through the cache hierarchy:

1. **Task arrives.** The `ContextAllocator` node runs first. L1 is empty; tokens used = 0.
1. **Initial retrieval.** The allocator queries Qdrant for segments relevant to the task. Five segments returned, totaling 6,200 tokens. All loaded into L1.
1. **Prefetch issued.** Lookahead detects the next node has `context_hint="category-specific thresholds"`. A background Qdrant query runs; three segments staged in prefetch buffer.
1. **Node 1 executes.** Accesses two L1 segments. Allocator updates `access_count` and `last_accessed_turn` for both.
1. **Node boundary.** Allocator flushes prefetch buffer: three segments added to L1. Total L1 now 9,800 tokens. Below high watermark.
1. **Node 2 executes.** Produces a corrected threshold table that supersedes one of the L1 segments. Allocator marks that segment dirty.
1. **Node 3 executes.** L1 reaches 38,400 tokens (96% of 40,000 budget). Allocator triggers eviction: three lowest-scoring segments removed. One is dirty; it is written back to Qdrant before removal. Tokens used drops to 27,100.
1. **Task completes.** Remaining dirty segments in L1 are written back to Qdrant. Qdrant metadata updated.
1. **Next task.** The corrected threshold table is now in Qdrant as `source_type=agent_derived`. The next run retrieves it at stage 2 and gets the agent’s improved version, not the original.

Step 9 is the key payoff: the agent gets better at the task across runs without a fine-tune, because write-back means its best work persists into the semantic cache.

## Challenges and Open Problems

**Embedding staleness.** When a segment is written back to Qdrant as `agent_derived`, the embedding reflects the original chunk embedding, not the derived content. The write-back handler should re-embed the updated text before upserting. This doubles the write-back cost. An open question is whether a lightweight re-embedding model (e.g., a smaller encoder) provides sufficient recall to justify the asymmetry with the read-time encoder.

**Budget estimation under branching.** The prefetch lookahead issues retrieval for the next N nodes in the default execution path. In a graph with conditional edges, the actual path may diverge. Prefetching for a branch that is never taken wastes Qdrant bandwidth and slightly biases the allocator’s budget estimates. One approach is to prefetch at reduced priority for non-default branches (weighted by the probability the edge is taken, if that can be estimated from prior runs).

**Cache invalidation in parallel subgraphs.** LangGraph supports parallel node execution via the `Send` API. Two nodes running in parallel can both update a state field, triggering mutual invalidation of each other’s prefetch buffers in a way that interleaves unpredictably. The allocator needs a lock-free invalidation protocol for parallel execution, which is not addressed in this post.

**Dirty bit accuracy.** The current dirty detection relies on the agent referencing a segment ID in its output. This is a coarse signal. A more precise approach would diff the segment text against the agent’s output to detect semantic modification, but this requires an additional LLM call per segment per turn. The right threshold for when to pay that cost is unclear.

**Cold start with no L2 state.** The first run in a new domain has an empty Qdrant collection. The allocator falls back to direct L3 retrieval, which is slower and may not be semantically indexed. A useful initialization step is a background indexing job that populates Qdrant with the most commonly accessed L3 documents before the agent runs its first task, estimated from prior query logs if available.

## References

- Packer et al. (2023). *MemGPT: Towards LLMs as Operating Systems*. [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- Shinn et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Qdrant Engineering (2024). *Payload Filtering and Scroll API*. [qdrant.tech/documentation](https://qdrant.tech/documentation/concepts/filtering/)
- LangGraph Documentation (2025). *Subgraphs and State Passing*. [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/how-tos/subgraph/)
- LangGraph Documentation (2025). *Send API and Parallel Execution*. [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)
- Shazeer et al. (2017). *Attention Is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Wang et al. (2023). *Voyager: An Open-Ended Embodied Agent with Large Language Models*. [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)
- Park et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. UIST 2023. [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)