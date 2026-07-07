---
title: "Building a Stateful AI Co-worker: Lessons from Specter"
date: 2026-06-21
description: "How Specter wires together LangGraph orchestration, Qdrant vector memory, SSE streaming, and headless browser access to build an AI agent that actually persists context across sessions."
tags: ["agents", "langgraph", "qdrant", "memory", "streaming", "browser-automation"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Most "AI assistant" demos share a silent assumption: the conversation is ephemeral. Ask a question, get an answer, close the tab, everything resets. This is convenient for demos and catastrophic for anything resembling real work, where context accumulates, tasks span multiple sessions, and the assistant is supposed to actually *know* you by now.

Specter is a small but architecturally interesting personal agent that takes the opposite stance. Built by Mihir Inamdar, it wires together **LangGraph** for agent orchestration, **Qdrant** for semantic long-term memory, **FastAPI** with SSE streaming for the backend, and a **Next.js 14** chat UI, plus an optional headless browser for web interaction. This post covers how those pieces fit together, where the interesting design decisions live, and what this stack implies for building agents that feel less like toys.

Pre-reads: familiarity with LLM tool-calling is assumed. Some knowledge of graph-based agent frameworks (LangGraph, or conceptually similar) will help.

## Table of Contents

- [The Core Problem: Statefulness in Agents](#the-core-problem-statefulness-in-agents)
- [Architecture Overview](#architecture-overview)
- [LangGraph as the Orchestration Backbone](#langgraph-as-the-orchestration-backbone)
- [Memory via Qdrant: The Persistence Layer](#memory-via-qdrant-the-persistence-layer)
  - [Collection Design: Named Vectors for Dual Memory Tiers](#collection-design-named-vectors-for-dual-memory-tiers)
  - [Payload Schema](#payload-schema)
  - [Importance-Score Filtering](#importance-score-filtering)
  - [MMR Retrieval: Avoiding Redundant Context Injection](#mmr-retrieval-avoiding-redundant-context-injection)
  - [A Memory Write and Retrieval Cycle](#a-memory-write-and-retrieval-cycle)
  - [The Retrieval Boundary Problem](#the-retrieval-boundary-problem)
- [Server-Sent Events and Streaming](#server-sent-events-and-streaming)
- [Browser Automation as a Tool](#browser-automation-as-a-tool)
- [Self-Evaluation: Agents Grading Their Own Outputs](#self-evaluation-agents-grading-their-own-outputs)
- [Frontend Design: The Memory Side Panel](#frontend-design-the-memory-side-panel)
- [Challenges and Open Problems](#challenges-and-open-problems)

## The Core Problem: Statefulness in Agents

The dominant pattern for LLM applications is stateless: a user sends a message, the model produces a response, done. What makes it feel stateful is usually just stuffing the conversation history into the context window. This works until it doesn't. Context windows have limits, full conversation replay gets expensive fast, and there's no structural distinction between what the model *knows* and what it merely saw in the last few turns.

Real work sessions look different. You mention your name and your job once, and you expect the assistant to still know both of those things six sessions later. You ask it to browse a site for you, read the output, and remember the relevant parts. You want it to evaluate whether its own response was actually good.

These are distinct problems with distinct solutions:

- **Cross-session memory**: vector store with semantic retrieval
- **Live web access**: headless browser automation
- **Output quality**: structured self-evaluation

Specter assembles these into a single agent without making any of them the entire point. That's what makes it worth studying.

## Architecture Overview

```
┌──────────────────────────────────────────────┐
│  Next.js 14 (port 3000)                      │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ ChatPanel│  │MemoryPanel│  │SuccessCrit │ │
│  └────┬─────┘  └──────────┘  └────────────┘ │
│       │ fetch (SSE)                          │
│  ┌────▼───────────────────────────────────┐  │
│  │  /api/chat   /api/memory  (Next routes)│  │
└──┴────┬────────────────────────────────────┴──┘
        │ HTTP proxy
┌───────▼──────────────────────────────────────┐
│  FastAPI (port 8000)                          │
│                                               │
│  POST /api/chat  ──►  SpectorAgent            │
│                        │                      │
│                   LangGraph StateGraph         │
│                   ┌────┴──────┐               │
│                   │   agent   │◄── GPT-4o-mini │
│                   └────┬──────┘               │
│                   ┌────▼──────┐               │
│                   │   tools   │               │
│                   │ ┌────────┐│               │
│                   │ │memory  ││ remember/recall│
│                   │ ├────────┤│               │
│                   │ │browser ││ open/read/     │
│                   │ │        ││ snapshot/click │
│                   │ └────────┘│               │
│                   └────┬──────┘               │
│                   ┌────▼──────┐               │
│                   │ evaluate  │ structured-out │
│                   └────┬──────┘               │
│                        │                      │
│  GET /api/memory ──►  QdrantStore             │
│                   AsyncQdrantClient           │
│                   (:memory: or remote)        │
│                                               │
│  browser tools ──►  agent-browser daemon      │
│                      Chrome (headless)        │
└──────────────────────────────────────────────┘
```

The data flow is straightforward. The Next.js frontend maintains a SSE connection to its own API routes, which proxy to the FastAPI backend. The backend runs a LangGraph `StateGraph` that can call memory tools, browser tools, or both, then pipes the token stream back up. The Qdrant store runs in-memory by default (`:memory:`), or points at a remote server for persistence.

This separation is clean. The frontend knows nothing about LangGraph. The backend knows nothing about the React component tree. They communicate through SSE events with three message types: `token`, `done`, and `error`.

## LangGraph as the Orchestration Backbone

**LangGraph** ([LangChain AI, 2023](https://github.com/langchain-ai/langgraph)) is a graph-based agent runtime built on top of LangChain. The key idea is representing agent execution as a directed graph of nodes and edges, where nodes are computation steps and edges control flow, including cycles. This matters because real tool-calling agents loop: the model calls a tool, sees the result, decides whether to call another tool or stop, and so on.

In Specter's implementation, the graph has three nodes:

1. **`agent`**: calls GPT-4o-mini with the current message and tool schemas, decides whether to use tools or generate a final response
2. **`tools`**: executes whichever tool the agent selected (memory operations or browser actions), feeds the result back
3. **`evaluate`**: after a final response is ready, scores it against any success criteria the user provided

The execution cycle for a simple turn looks like:

```
agent → tools → agent → tools → ... → agent → evaluate → END
```

Each loop iteration adds new messages to the graph's shared state. This is LangGraph's `MessagesState`: a list of `BaseMessage` objects that accumulates tool calls and tool results alongside user and assistant turns.

What's useful about this approach is that the graph handles conditional routing automatically. The `agent` node returns control to `tools` if there's a pending tool call, or routes to `evaluate` (and then `END`) when the model produces a final answer. You define the routing logic once as a conditional edge function, and LangGraph calls it after every node execution.

One capability Specter doesn't currently use: LangGraph's **checkpointing** mechanism, which lets you persist graph state across execution runs to a durable store. This would allow the agent to resume interrupted tasks. Specter instead offloads persistence to Qdrant, which is appropriate for memory but wouldn't help you resume a half-finished web browsing task.

## Memory via Qdrant: The Persistence Layer

The memory system is where Specter separates itself from a basic chat wrapper. It exposes two tools to the agent: `remember` and `recall`. The agent decides autonomously when to call these. If you say "my name is Mihir," the agent calls `remember`. If you later ask "what do you know about me?", it calls `recall`.

This section covers the full design: how the Qdrant collection is structured, how payloads encode metadata, how importance-score filtering prevents stale facts from dominating context, and how Maximal Marginal Relevance retrieval avoids injecting redundant memories.

### Collection Design: Named Vectors for Dual Memory Tiers

Qdrant supports *named vectors*, meaning each point in a collection can carry multiple independent vector embeddings under distinct names. Specter's memory benefits from a two-tier design that maps directly onto this feature:

| Vector name | Dimension | Purpose |
|---|---|---|
| `"episodic"` | 1536 | Short-term, session-scoped memories: recent user inputs, per-session context, transient observations |
| `"semantic"` | 1536 | Long-term facts: user profile, preferences, stable knowledge extracted across sessions |

Both use `text-embedding-3-small` (1536-dim, configurable down to 512 with Matryoshka distillation). The distance metric is cosine similarity for both, matching the normalization properties of OpenAI embeddings.

Creating the collection:

```python
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance

client = AsyncQdrantClient(":memory:")

await client.create_collection(
    collection_name="specter_memory",
    vectors_config={
        "episodic": VectorParams(size=1536, distance=Distance.COSINE),
        "semantic": VectorParams(size=1536, distance=Distance.COSINE),
    },
)
```

Storing a memory under a specific vector name:

```python
from qdrant_client.models import PointStruct
from uuid import uuid4

await client.upsert(
    collection_name="specter_memory",
    points=[
        PointStruct(
            id=str(uuid4()),
            vector={"semantic": embedding},   # or {"episodic": embedding}
            payload={...},
        )
    ],
)
```

The named-vector design lets the agent query against either tier independently. A recall for "what are this user's long-term preferences?" searches the `"semantic"` vector, ignoring noisy episodic observations from the current session. A recall for "what did we discuss earlier in this session?" searches `"episodic"`, ignoring stable semantic facts from months ago. Qdrant handles both under one collection with no join overhead.

For brevity, the base Specter implementation uses a single unnamed vector. The named-vector schema above is the natural extension for deployments where session isolation and memory tier separation matter.

Switching to a remote Qdrant instance for true persistence across process restarts is a one-line config change:

```python
# backend/.env
QDRANT_URL=http://localhost:6333
```

The embedding dimension depends on the model. `text-embedding-3-small` produces 1536-dimensional vectors by default. Qdrant's HNSW index handles approximate nearest-neighbor search at this dimensionality comfortably, with queries completing well under 10 ms at typical personal-assistant scales.

### Payload Schema

Each memory point carries a structured JSON payload alongside its vector. The full schema:

```json
{
    "content":          "User works in AI at a startup in Bangalore",
    "user_id":          "mihir",
    "session_id":       "sess_20260625_001",
    "timestamp":        1750838400,
    "memory_type":      "semantic",
    "importance_score": 0.87,
    "source":           "user_statement"
}
```

Field semantics:

- **`content`**: the raw text that was embedded. Always stored verbatim so retrieval can surface it directly.
- **`user_id`**: scopes retrieval to a single user. In multi-user deployments, this is the primary isolation boundary.
- **`session_id`**: identifies the session in which the memory was recorded. Used for episodic lookups and for debugging ("why did the agent say that?").
- **`timestamp`**: Unix epoch at storage time. Enables time-decay ranking, recency filtering, and explicit forgetting of stale memories.
- **`memory_type`**: `"episodic"` or `"semantic"`. Mirrors the vector name, stored here as a payload field for filter-only queries that don't involve ANN search.
- **`importance_score`**: a float in $[0, 1]$ representing estimated importance. This is the primary lever for preventing stale or low-signal memories from polluting context; higher scores survive filtering and lower scores are pruned before semantic search runs.
- **`source`**: provenance tag (`"user_statement"`, `"tool_result"`, `"agent_inference"`). Allows selective recall, e.g., excluding inferred memories in high-stakes contexts.

Memory is scoped by `user_id` and `session_id` on write, but recall filters only by `user_id`. This means facts from previous sessions are retrievable. The session boundary organizes when things were stored; it does not reset what the agent knows.

### Importance-Score Filtering

The central problem with naive memory retrieval: over time, the vector store accumulates noise. Greetings, filler statements, partial observations, and factual corrections all get stored alongside genuinely useful facts. A cosine-similarity search over this unfiltered collection will surface whichever memory happens to be most geometrically close to the query, regardless of whether it's still accurate or ever mattered.

Importance-score filtering addresses this at query time by restricting the ANN search to points whose `importance_score` exceeds a threshold. Qdrant's payload filter API makes this a low-overhead operation: the HNSW graph traversal is gated by the filter condition, so low-importance points are never scored, not just post-filtered.

```python
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

results = await client.search(
    collection_name="specter_memory",
    query_vector=("semantic", query_embedding),
    query_filter=Filter(
        must=[
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id),
            ),
            FieldCondition(
                key="importance_score",
                range=Range(gte=0.5),
            ),
        ]
    ),
    limit=20,
    with_vectors=True,
)
```

The `limit=20` here is intentionally larger than the final context injection count. A wider candidate pool is required for the MMR pass described below.

Assigning `importance_score` at storage time is itself a design choice. Options include:

1. **Heuristic rules**: named entities get 0.8+, statements about user preferences get 0.7+, conversational filler gets 0.2.
2. **LLM scoring**: a secondary prompt rates each memory's importance before storage. Slower but more accurate.
3. **Recency-weighted decay**: score starts at 1.0 and decays as $e^{-\lambda \Delta t}$, so older memories need to be reinforced by repeated mentions to remain above threshold.

A combined approach (heuristic initial score, time decay applied at query time via payload arithmetic) is feasible. Qdrant's payload filtering can compare `timestamp` against a computed recency boundary, and the importance score encodes the non-temporal signal. Both operate at query time without modifying stored points.

### MMR Retrieval: Avoiding Redundant Context Injection

Even after filtering by importance, the top-$k$ cosine-similarity results may be highly redundant. If the user mentioned their job role in five separate turns, those five nearly-identical memories will all score high against a "what does the user do?" query. Injecting all five into context wastes tokens and adds noise without adding information.

**Maximal Marginal Relevance** (MMR) ([Carbonell & Goldstein, 1998](https://dl.acm.org/doi/10.1145/290941.291025)) addresses this by iteratively selecting documents that are simultaneously relevant to the query and dissimilar to already-selected documents. The scoring criterion at each selection step is:

$$\text{MMR}_i = \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \text{sim}(d_i, q) - (1-\lambda) \cdot \max_{d_j \in S} \text{sim}(d_i, d_j) \right]$$

where $R$ is the candidate pool, $S$ is the set of already-selected memories, $q$ is the query vector, $\text{sim}$ is cosine similarity, and $\lambda \in [0,1]$ controls the relevance/diversity tradeoff. At $\lambda = 1$, MMR degenerates to standard similarity ranking; at $\lambda = 0$, it maximizes diversity with no relevance weighting. Values around $\lambda = 0.7$ work well in practice for memory retrieval.

Qdrant does not implement MMR natively; it returns ANN results ranked by similarity. MMR runs as a post-processing step in Python over the candidate pool:

```python
import numpy as np

def mmr_select(
    query_vec: np.ndarray,
    candidate_vecs: list[np.ndarray],
    candidates: list[dict],
    k: int = 5,
    lambda_: float = 0.7,
) -> list[dict]:
    """Iterative MMR selection over a pre-fetched candidate pool."""
    selected: list[dict] = []
    selected_vecs: list[np.ndarray] = []
    remaining = list(zip(candidate_vecs, candidates))

    while len(selected) < k and remaining:
        scores = []
        for vec, doc in remaining:
            relevance = float(np.dot(query_vec, vec))  # vectors are L2-normalized
            if selected_vecs:
                redundancy = max(float(np.dot(vec, sv)) for sv in selected_vecs)
            else:
                redundancy = 0.0
            scores.append(lambda_ * relevance - (1.0 - lambda_) * redundancy)

        best_idx = int(np.argmax(scores))
        best_vec, best_doc = remaining.pop(best_idx)
        selected.append(best_doc)
        selected_vecs.append(best_vec)

    return selected
```

The tradeoff: MMR's greedy loop is $O(|R| \cdot k)$ in cosine operations, which is negligible when $|R|$ is a few dozen memory candidates. It becomes a concern only if the candidate pool grows into the thousands, at which point approximate MMR via clustering is more appropriate.

### A Memory Write and Retrieval Cycle

To make the above concrete, here is a full end-to-end example tracing a single fact from storage to context injection.

**Scenario**: the user says "My name is Mihir and I work in AI at a startup in Bangalore." The agent calls `remember`.

**Write path:**

```python
import time
from uuid import uuid4
from qdrant_client.models import PointStruct

async def remember(
    content: str,
    user_id: str,
    session_id: str,
    memory_type: str = "semantic",
    importance_score: float = 0.85,
) -> str:
    embedding = await embed(content)   # text-embedding-3-small → list[float] len 1536

    await client.upsert(
        collection_name="specter_memory",
        points=[
            PointStruct(
                id=str(uuid4()),
                vector={memory_type: embedding},
                payload={
                    "content": content,
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": int(time.time()),
                    "memory_type": memory_type,
                    "importance_score": importance_score,
                    "source": "user_statement",
                },
            )
        ],
    )
    return f"Remembered: {content}"
```

The point lands in Qdrant's HNSW graph under the `"semantic"` named vector. Qdrant's HNSW construction is incremental; the new point is wired into the existing graph in $O(\log n)$ expected edge updates.

**Three sessions later**: the user asks "What do you know about me?" The agent calls `recall`.

**Retrieval path:**

```python
async def recall(query: str, user_id: str, k: int = 5) -> list[str]:
    query_vec = np.array(await embed(query))

    # Step 1: ANN search with importance filter, fetch 4x the final k
    candidates = await client.search(
        collection_name="specter_memory",
        query_vector=("semantic", query_vec.tolist()),
        query_filter=Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="importance_score", range=Range(gte=0.5)),
            ]
        ),
        limit=k * 4,
        with_vectors=True,
    )

    if not candidates:
        return []

    # Step 2: MMR over the candidate pool
    cand_vecs = [np.array(r.vector["semantic"]) for r in candidates]
    cand_docs = [r.payload for r in candidates]
    selected = mmr_select(query_vec, cand_vecs, cand_docs, k=k, lambda_=0.7)

    return [doc["content"] for doc in selected]
```

**What the agent receives** (after three sessions of varied inputs):

```
recalled memories:
1. "My name is Mihir and I work in AI at a startup in Bangalore"  [importance 0.87]
2. "Prefers concise responses, no bullet points for casual questions"  [importance 0.72]
3. "Working on a LangGraph-based agent project called Specter"  [importance 0.91]
4. "Interested in retrieval-augmented generation and vector databases"  [importance 0.78]
5. "Located in India, works in IST timezone"  [importance 0.65]
```

The MMR pass ensures that even if the user's name was mentioned five times across sessions (resulting in five nearly-identical high-similarity points), only one variant is selected. The remaining four slots go to semantically distinct facts. Context injection is both relevant and non-repetitive.

This also works across sessions: as long as the agent is hitting the same non-ephemeral Qdrant instance, facts from prior sessions are retrievable without any special session-handoff logic.

### The Retrieval Boundary Problem

One issue the current Specter implementation leaves open: how the agent decides *what* is worth remembering. The agent calls `remember` based on prompt instructions and its own judgment. This is flexible but unpredictable. More structured approaches define explicit categories (what to remember: named entities, user preferences, constraints) and store them separately from conversational filler.

**MemGPT** ([Packer et al. 2023](https://arxiv.org/abs/2310.08560)) explored a related design where memory was divided into core memory (always in context) and archival memory (retrieved on demand). Specter's approach is simpler and probably sufficient for a personal co-worker at this scale, but would need tightening for multi-user or production deployments.

A natural extension: classify each incoming statement before storage using a lightweight classifier or LLM prompt, assign it to `"episodic"` or `"semantic"` accordingly, and set the `importance_score` heuristically from the category. The named-vector schema is already in place to support this routing.

## Server-Sent Events and Streaming

The streaming model is worth examining, because it's where most AI chat implementations make a choice they later regret.

**Polling** is simple but creates visible lag: you wait for the whole response, then it appears. **WebSockets** are real-time but require stateful connections that complicate load balancing and reconnection. **Server-Sent Events (SSE)** sit in between: a long-lived HTTP response where the server pushes newline-delimited data chunks as they become available. SSE is unidirectional (server to client only), which is appropriate here because the user doesn't need to send data while the stream is in progress.

Specter's backend uses `sse-starlette` to yield events from FastAPI. Each event carries a `type` field and a `content` field:

```json
{"type": "token",  "content": "<text chunk>"}
{"type": "done",   "content": ""}
{"type": "error",  "content": "<message>"}
```

The frontend's `ChatPanel` component reads this stream using the browser's built-in `EventSource` API (or a fetch-based polyfill for more control). Tokens accumulate in React state and render progressively: users see words arriving as the model generates them, rather than waiting for the full response.

The LangGraph streaming integration uses `astream_events`, which emits events at each node transition. Specter filters for `on_chat_model_stream` events to extract individual tokens from the underlying LLM stream. Other event types (tool calls, tool results) could be surfaced to the UI for a more transparent "thinking" display, though Specter keeps this simple.

One practical issue with SSE: reconnection. If the connection drops mid-stream, `EventSource` reconnects automatically, but the stream itself is gone. The backend has no mechanism to resume a stream mid-generation. For a personal tool running on localhost this rarely matters; in production, you'd want a session store that can replay missed events or a client-side buffer.

## Browser Automation as a Tool

The browser integration is handled by **agent-browser** (Vercel Labs), a CLI tool that wraps headless Chrome and exposes it over a local API. Specter wraps each operation as an async Python tool using `asyncio.to_thread`, since the CLI calls are blocking.

The agent gets access to 10 tools:

| Tool | What it does |
|---|---|
| `browser_open` | Navigate to a URL |
| `browser_read` | Fetch readable text from a URL (no browser launch) |
| `browser_snapshot` | Get accessibility tree with element refs (`@e1`, `@e2`, …) |
| `browser_click` | Click an element by ref or CSS selector |
| `browser_fill` | Clear and fill an input field |
| `browser_get_text` | Get visible text from an element |
| `browser_screenshot` | Take a screenshot, returns file path |
| `browser_scroll` | Scroll in any direction |
| `browser_wait` | Wait N milliseconds |
| `browser_close` | Close the browser session |

The agent-browser daemon keeps Chrome alive between tool calls, so startup overhead (~5 s) only hits on the first `browser_open`. Subsequent operations complete in 1–2 s.

The accessibility tree output from `browser_snapshot` is particularly useful. Rather than dumping raw HTML (verbose and LLM-unfriendly), it produces a structured representation of the page's interactive elements with stable refs like `@e1`, `@e2`. The agent can then click `@e1` rather than generating a fragile CSS selector.

The README shows a representative interaction:

```
You: Open https://example.com and summarize what you find on the page.

Specter: The page is titled "Example Domain" and states that this domain
         is intended for use in documentation examples...

Specter autonomously called browser_open → navigated → read content → responded.
Total: ~9 s.
```

That's real-world latency: one `browser_open` call plus an LLM inference pass. For a personal agent where you're thinking in seconds anyway, this is acceptable.

### What the browser can't do reliably

The hard cases for browser automation haven't changed much: CAPTCHA challenges, login flows with MFA, sites that fingerprint headless browsers, and highly dynamic single-page apps where the accessibility tree is empty until JavaScript hydrates. `browser_read` sidesteps some of this by using a plain HTTP fetch to extract readable text without launching Chrome at all, which is faster and more reliable for read-only tasks on standard pages.

## Self-Evaluation: Agents Grading Their Own Outputs

Every response optionally passes through a structured self-evaluation step. The user can supply a `success_criteria` string with their message, and after the agent generates a response, Specter submits both the response and the criteria to GPT-4o-mini as a second call with structured output enabled.

The output schema:

```python
class EvaluationResult(BaseModel):
    success: bool
    score: float   # 0.0 to 1.0
    reasoning: str
```

The score and success flag appear alongside the response in the UI. The README example shows a weekly planning response scoring `0.95` against the supplied criteria.

This is a lightweight version of what the research community calls *LLM-as-judge*: using a language model to evaluate another language model's output. **G-Eval** ([Liu et al. 2023](https://arxiv.org/abs/2303.16634)) formalized this approach with chain-of-thought prompting for more reliable scoring. **MT-Bench** ([Zheng et al. 2023](https://arxiv.org/abs/2306.05685)) showed that strong models can score outputs at near-human agreement on many dimensions, though they inherit known biases: preferring longer responses, favoring their own outputs, and struggling with factual accuracy verification.

Specter's self-evaluation is simpler than any of these: a single score from the same model that generated the response. The model cannot verify its own factual claims, and there's an inherent conflict of interest. But for the personal co-worker use case, it remains useful. A planner that scores its own output at 0.3 is telling you something.

A more considered approach would separate the generator and evaluator: use a smaller model for generation and a stronger model for evaluation, or use a different prompt with explicit chain-of-thought scoring criteria. Neither is difficult to add here.

## Frontend Design: The Memory Side Panel

The frontend is Next.js 14 with React 18, TypeScript, and Tailwind CSS. The component structure:

```
components/
├── ChatPanel.tsx         # Main chat orchestrator + SSE reader
├── MessageList.tsx
├── MessageBubble.tsx
├── InputBar.tsx
├── MemoryPanel.tsx       # Live view of recalled memories
└── SuccessCriteria.tsx   # Optional input for self-eval criteria
```

`ChatPanel` owns the SSE connection and the message state. It reads the token stream, accumulates text, and updates the message list in real time. The SSE connection is managed with a `useEffect` cleanup that closes the connection when the component unmounts, preventing stale streams on navigation.

The **MemoryPanel** is the most distinctive UI element. It hits `GET /api/memory/search?q=&user_id=&limit=5` to show what facts the agent has retrieved (or recently stored). This makes memory visible and auditable, which matters: if the agent has an incorrect fact about you, you want to see it before it affects ten more turns.

The `SuccessCriteria` component is a simple textarea that, when filled in, gets included in the `/api/chat` request body. It's optional and unintrusive: if you don't fill it in, evaluation is skipped.

## Challenges and Open Problems

**Memory staleness.** Qdrant stores facts but has no built-in expiration or conflict resolution. If you tell the agent you work in AI on Monday and change careers on Friday, both facts live in the vector store with equal weight. Cosine-similarity search will surface whichever is more semantically close to the query, which may not be the newer one. The `importance_score` and `timestamp` payload fields provide the data needed to implement decay and conflict detection, but Specter currently has no `forget` or `update` primitive. Systems like **A-MEM** ([Xu et al. 2024](https://arxiv.org/abs/2404.00573)) address this with explicit memory update operations.

**MMR parameter sensitivity.** The $\lambda$ parameter in MMR is set globally. A single value cannot simultaneously optimize for a query about "what are my recent session topics" (where diversity is highly desirable) and "summarize what I've said about my job" (where relevance should dominate). Adaptive $\lambda$ selection conditioned on query type is an open design question.

**Importance score assignment.** In the current design, the agent assigns `importance_score` heuristically or via a secondary LLM call. Both have failure modes: heuristics miss context-dependent importance, and LLM scoring adds latency and cost to every `remember` call. A learned importance model trained on which memories were actually used in downstream recalls would be more accurate but requires instrumentation not present in this implementation.

**No checkpointing.** LangGraph supports graph-level checkpointing, which would allow Specter to recover an interrupted multi-step task. As it stands, if a browser task takes 30 seconds and the connection drops at second 25, there's no resume path. The user has to re-issue the request.

**Single-agent architecture.** The LangGraph graph has one agent node. For complex multi-step tasks ("research this topic, write a summary, save it to a doc, email it to me"), a single agent calling all tools sequentially works but lacks parallelism and specialization. A multi-agent architecture with separate researcher, writer, and executor subgraphs would be better suited to this workflow, though it adds significant coordination complexity.

**Self-evaluation reliability.** The current evaluator uses the same model that generated the response. It cannot catch its own factual errors and may score outputs higher than warranted. Even a simple cross-check against retrieved memory ("does the response contradict known facts?") would improve reliability.

**Browser tool surface area.** Ten browser tools cover most common interactions, but there are gaps: file downloads, JavaScript execution, cookie management, and anything requiring authenticated sessions. The Vercel agent-browser daemon abstracts Chrome well but exposes only a subset of the full DevTools Protocol.

Despite these limitations, Specter is a well-constructed example of what a personal agent with memory, browsing, and self-evaluation actually looks like when you wire it together: not a prototype that glosses over the plumbing, but something that runs, has a real API, and shows the seams. That's most of what matters at this stage of the field.

## Further Reading

- [Specter on GitHub](https://github.com/inamdarmihir/specter)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- **Towards LLMs as Operating Systems** ([Packer et al. 2023](https://arxiv.org/abs/2310.08560))
- **G-Eval: NLG Evaluation using GPT-4** ([Liu et al. 2023](https://arxiv.org/abs/2303.16634))
- **MT-Bench** ([Zheng et al. 2023](https://arxiv.org/abs/2306.05685))
- **A-MEM: Agentic Memory for LLM Agents** ([Xu et al. 2024](https://arxiv.org/abs/2404.00573))
- **The Use of MMR, Diversity-Based Reranking for Reordering Documents** ([Carbonell & Goldstein, 1998](https://dl.acm.org/doi/10.1145/290941.291025))
- [Server-Sent Events (MDN Web Docs)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
