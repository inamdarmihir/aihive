---
title: "Curing Context Thrashing: Pointer-Based Handoffs in Agent Swarms"
date: 2026-06-30
description: "How to prevent multi-agent frameworks from choking on their own state by replacing pass-by-value context windows with vector-backed pointers and BM42 hybrid retrieval."
tags: ["agents", "langgraph", "qdrant", "architecture", "state-management"]
author: "Mihir Inamdar"
showToc: true
math: true
---

*In this post, I examine a structural failure mode in multi-agent systems — what I call **context thrashing** — and a concrete architectural fix: replacing pass-by-value state graphs with a pointer-based handoff scheme backed by a vector store. I'll use Qdrant and its BM42 hybrid retrieval as the implementation vehicle, but the pattern applies to any retrieval backend that supports hybrid search. The post assumes familiarity with LangGraph-style state graphs and with the dense vs. sparse retrieval distinction. I won't rederive BM25 or embedding models from scratch.*

---

## Table of Contents

- [The Problem: Monolithic State Graphs](#the-problem-monolithic-state-graphs)
- [Why This Scales Badly](#why-this-scales-badly)
- [Architecture: Pointer-Based Handoffs](#architecture-pointer-based-handoffs)
- [The Retrieval Layer: Why Hybrid Search Is Mandatory](#the-retrieval-layer-why-hybrid-search-is-mandatory)
  - [Dense-Only Failure](#dense-only-failure)
  - [BM42 and Reciprocal Rank Fusion](#bm42-and-reciprocal-rank-fusion)
- [Tradeoffs](#tradeoffs)
  - [Input Token Cost](#input-token-cost)
  - [Latency](#latency)
  - [Thick vs. Thin State](#thick-vs-thin-state)
- [Open Problems](#open-problems)
- [Further Reading](#further-reading)

---

## The Problem: Monolithic State Graphs

<#the-problem-monolithic-state-graphs>

Multi-agent frameworks model shared state as a single growing object — a `TypedDict` wrapping a message list, a plain string buffer, or some application-specific dict. Every node reads the full object and appends its output to it. If $\text{State}_n$ is the state after the $n$-th node executes and $\text{Output}_n$ is that node's output:

$$
\text{State}_n = \text{State}_{n-1} \cup \text{Output}_n
$$

State grows monotonically. There is no eviction policy, no scoping, and no mechanism to signal that most of what node $k$ produced is irrelevant to node $k+5$. This design works fine in demos, where pipelines are short and each node produces a few hundred tokens. In production, it breaks.

A concrete example: a `ResearchAgent` reads three API references and a dozen forum threads. It produces 60k tokens of raw notes, dead ends, and half-finished reasoning. It appends all of this to the shared state and hands it to the `CoderAgent`. The `CoderAgent`'s context window is now dominated by research exhaust. **Lost in the Middle** (**Liu et al. 2023**, [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)) documents what happens next: LLMs systematically underweight information buried in the middle of long contexts, regardless of how relevant it is. The agent loses track of the one snippet it needed, hallucinates variable names, and burns $0.50 in input tokens before writing a single line of code.

I call this failure mode **context thrashing**, by analogy to OS thrashing: the swarm spends more compute re-reading the history of its own execution than advancing the task, just as an OS page-faults itself into uselessness when physical memory runs out.

---

## Why This Scales Badly

<#why-this-scales-badly>

The damage compounds as the pipeline grows. A `DeploymentAgent` four hops downstream inherits the full output of every upstream node, even though it needs one Dockerfile from the pile. The LLM is being used as a linear scanner: it reads 80k tokens of accumulated history to locate roughly 500 tokens of actionable instruction.

More precisely, if each node adds $c$ tokens of output on average, the context window seen by node $n$ grows as $O(nc)$. The total input tokens consumed across an $N$-node pipeline is:

$$
\sum_{n=1}^{N} n \cdot c = \frac{N(N+1)}{2} \cdot c \;\in\; O(N^2)
$$

Token cost scales quadratically in pipeline depth, while actual useful information delivered to each node is roughly constant. This is not a prompting problem — no instruction set can fix it. It is a state-management problem: the graph conflates *the history of what happened* with *what a given node actually needs right now*, and offloads the separation to the LLM at inference time on every call.

---

## Architecture: Pointer-Based Handoffs

<#architecture-pointer-based-handoffs>

The fix is to stop passing the payload between nodes and instead pass a reference to where the payload lives. This is pass-by-reference rather than pass-by-value, with the vector store serving as the shared memory heap for the entire swarm.

Concretely, the orchestrator writes each node's full output to Qdrant as it is produced and threads only a lightweight **pointer state** through the graph:

```python
class PointerState(TypedDict):
    session_id: str
    current_objective: str
    active_pointers: List[str]  # UUIDs mapping to Qdrant payloads
```

Compare the two data flows. Under pass-by-value, every arrow in the graph carries the full accumulated payload:

```
[Researcher] ──(80k tokens)──▶ [Coder] ──(85k tokens)──▶ [Reviewer]
```

Under pass-by-reference, arrows carry only pointers and a stated objective. Heavy payloads move sideways into Qdrant:

```
[Researcher] ──(writes 80k tokens)──▶ QDRANT
       │
       (passes session_id + objective ≈ 150 tokens)
       ▼
    [Coder] ──(hybrid search for specific functions)──▶ QDRANT
       │      ◀──(returns ~2k targeted tokens)──────────
       │
       (passes session_id + commit_hash ≈ 150 tokens)
       ▼
  [Reviewer]
```

*Pass-by-value forces every node to inherit the full history; pass-by-reference lets each node pull only the slice its current objective requires.*

When the `CoderAgent` receives its `PointerState`, it never sees the raw research dump. It uses `current_objective` to issue a targeted query against Qdrant and retrieves the specific slice of context relevant to the code it is about to write. The context window changes its role: instead of being a passive accumulator, it becomes an active working set — loaded with what matters now, evicted when the node finishes.

---

## The Retrieval Layer: Why Hybrid Search Is Mandatory

<#the-retrieval-layer-why-hybrid-search-is-mandatory>

Pointer-based handoffs only deliver on their promise if the retrieval step reliably surfaces the right slice of the heap. This is where a subtle but consequential failure mode appears.

### Dense-Only Failure

<#dense-only-failure>

If the shared heap is indexed with dense embeddings alone — say, `text-embedding-3-small` — the architecture quietly fails on a common class of query. Suppose the `CoderAgent` asks: *"What were the exact environment variables required for the Redis connection?"*

A pure dense search retrieves passages semantically close to "Redis configuration" in the abstract: conceptual paragraphs about caching strategies, connection pooling, and database setup. The actual line containing the literal variable name `REDIS_PASSWORD_PROD` may rank lower, because dense vectors encode topical similarity rather than exact lexical identity. The agent gets plausible-looking context that does not contain the specific fact it needs, and either hallucinates the variable name or fails the task.

This is the same gap that motivated hybrid retrieval in classic RAG: **dense embeddings** capture semantic intent, **sparse representations** (à la BM25) capture exact token overlap, and robust retrieval systems need both to avoid systematic blind spots.

### BM42 and Reciprocal Rank Fusion

<#bm42-and-reciprocal-rank-fusion>

Qdrant's **BM42** applies this combination natively inside the vector store, computing both a dense vector (semantic intent) and a sparse vector (exact keyword match) for every payload written to the heap. At query time, the two ranked lists are fused using **Reciprocal Rank Fusion (RRF)**.

For a document $d$ with dense-search rank $r_\text{dense}(d)$ and sparse-search rank $r_\text{sparse}(d)$, the fused score is:

$$
\text{RRF}(d) = \sum_{r \,\in\, \{r_\text{dense}(d),\; r_\text{sparse}(d)\}} \frac{1}{k + r}
$$

where $k$ is a smoothing constant (typically 60) that dampens the influence of any single ranking signal and prevents low-rank documents from dominating the sum.

In plain terms: a document that ranks highly in *either* channel contributes a large term to its score. Results that are strong on lexical match, strong on semantic match, or both, all surface near the top. The $k$ constant ensures that a document ranked 1st in dense and 50th in sparse still wins over a document ranked 5th in both — the top of one channel is not entirely thrown away.

With this fusion running against the shared heap, the Redis query above retrieves the exact block containing `REDIS_PASSWORD_PROD` via the sparse channel, and the surrounding paragraph explaining *why* the connection is configured that way via the dense channel — in the same call.

---

## Tradeoffs

<#tradeoffs>

No architecture ships without costs. These are the real ones.

### Input Token Cost

<#input-token-cost>

| Metric | Pass-by-Value | Pointer-Based (Qdrant) |
|---|---|---|
| Input token cost | Grows near-quadratically across the pipeline | Flat — each node retrieves a bounded, targeted set |
| Agent focus (Lost in Middle risk) | High: irrelevant history dilutes the objective | Low: retrieved context is filtered by construction |
| Dominant latency source | LLM time-to-first-token on a huge prompt | Vector DB read/write |
| Architectural complexity | Low — state lives in memory | Higher — requires an external, consistent state store |

### Latency

<#latency>

Passing state in memory is instantaneous; writing to and querying Qdrant adds real network time, typically 50–200ms depending on deployment topology. This is not free. But feeding 80k tokens to an LLM incurs its own latency tax in time-to-first-token, often 5–10 seconds for a prompt that size at current API throughputs. The database round-trip is dwarfed by the inference time it eliminates, so the net effect on wall-clock latency is generally a win. Collocating the vector store with the agent workers (same availability zone or same machine) keeps the DB overhead below 20ms in most configurations, which is inconsequential.

### Thick vs. Thin State

<#thick-vs-thin-state>

Not everything belongs in the heap. If two nodes are passing a boolean back and forth — "did the test pass?" — writing that to Qdrant is pure overhead with no retrieval benefit. Pointer-based handoffs pay off for **thick data**: research notes, full codebases, logs, structured reports — anything with enough volume that a downstream node would otherwise have to scan past it. **Thin data** — flags, enums, short status fields, iteration counters — should stay in the `TypedDict` where it costs nothing to pass directly.

The heuristic I use: if the data would not meaningfully benefit from semantic search, it does not belong in the heap. The pointer state is not a replacement for the `TypedDict`; it is an escape valve for payloads that would otherwise bloat it beyond usefulness.

---

## Open Problems

<#open-problems>

The pointer-based pattern shifts which things can go wrong, and the new failure modes are worth naming.

**Consistency and staleness.** This post assumes a single writer per session — one node writes, then the next reads. In truly parallel swarms where multiple nodes write concurrently to overlapping regions of the heap, you need an explicit invalidation or versioning scheme. Without it, a downstream node can retrieve a pointer to data that has been superseded by a concurrent write. Multi-writer session management for vector stores is not a solved problem in most production frameworks.

**Retrieval recall as a new failure mode.** Pass-by-value never drops information — everything is always in context. Pass-by-reference can drop information if the query formulated by `current_objective` fails to surface the right pointer. Debugging a wrong answer now requires inspecting retrieval quality in addition to the prompt. The failure is less visible: instead of a model hallucinating despite having the correct fact in context, the model hallucinated because the correct fact was never retrieved. Distinguishing these two cases requires logging retrieval results, which most agent observability tooling does not do today.

**Where to draw the thick/thin line.** The heuristic above is qualitative and works for the clear cases. As swarms grow more complex — agents with dozens of inter-dependent subtasks, hierarchical memory systems, shared world models — deciding what belongs in the heap versus the `TypedDict` becomes its own non-trivial design decision. There is no principled threshold today, and the choice can meaningfully affect both latency and retrieval recall.

**Query formulation quality.** The retrieval step is only as good as the query the agent issues. `current_objective` is a concise string, but a complex retrieval need may require a more structured or decomposed query. An agent that issues a vague objective to `current_objective` will get vague retrieval results, with no obvious signal that the query was the problem. Investing in query formulation — perhaps a dedicated reformulation step before the retrieval call — is worth considering for high-stakes pipelines.

---

The core move here mirrors a shift the rest of software engineering already made: monolithic applications gave way to services communicating through message buses and shared databases rather than passing entire application state in memory. Agent swarms built on pass-by-value state graphs are re-running the monolith pattern, with tokens instead of RAM. Treating the context window as something closer to an L1 cache than a hard drive — a node loads only what its current objective requires, executes, and hands off a pointer rather than a payload — is the same discipline, applied one layer up the stack.

---

## Further Reading

<#further-reading>

- **Lost in the Middle: How Language Models Use Long Contexts** — Liu et al. 2023 ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172))
- [Qdrant: BM42 and Hybrid Search](https://qdrant.tech/)
- [LangGraph: Managing Complex State](https://www.langchain.com/langgraph)

---

```bibtex
@article{liu2023lost,
  title   = {Lost in the Middle: How Language Models Use Long Contexts},
  author  = {Liu, Nelson F. and Lin, Kevin and Hewitt, John and Paranjape, Ashwin
             and Bevilacqua, Michele and Petroni, Fabio and Liang, Percy},
  journal = {arXiv preprint arXiv:2307.03172},
  year    = {2023}
}
```
