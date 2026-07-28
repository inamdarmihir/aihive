---
title: "Content-Addressed Subagent Result Caching in Deep Agents"
date: 2026-07-28
description: "A content-addressed result cache for Deep Agents subagent dispatch, implemented as wrap_tool_call middleware on Qdrant — covering cache-key design, purity classification, invalidation, false-hit risk, and break-even cost analysis."
tags: ["agents", "deepagents", "caching", "qdrant", "middleware", "cost-optimization"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Multi-agent frameworks solve context rot by isolating work into subagents — each with its own context window, its own tool access, its own billing. The isolation is the point. But it comes with a structural cost that nobody has directly addressed: every subagent re-derives results from scratch, even when the underlying inputs haven't changed. This post proposes a content-addressed result cache for Deep Agents dispatch, implemented as a `wrap_tool_call` middleware backed by Qdrant, and analyzes the conditions under which it is safe, unsafe, and economically significant.

I will focus specifically on the **static-dispatch** path (the `task` tool in `SubAgentMiddleware`) and the **dynamic-dispatch** path (programmatic fan-out via `CodeInterpreterMiddleware`). I won't cover caching for the model itself — provider-side prompt caching already handles prefix deduplication at the API layer. What's missing is result-level memoization above the API.

---

## Table of Contents

1. [The Structural 4× Cost Multiplier](#1-the-structural-4-cost-multiplier)
2. [Why Provider-Side Prompt Caching Doesn't Close the Gap](#2-why-provider-side-prompt-caching-doesnt-close-the-gap)
3. [Content-Addressed Caching: The Core Idea](#3-content-addressed-caching-the-core-idea)
4. [Cache Key Design](#4-cache-key-design)
   - [4.1 Components of a Canonical Key](#41-components-of-a-canonical-key)
   - [4.2 Exact vs. Semantic Matching](#42-exact-vs-semantic-matching)
   - [4.3 Purity Classification](#43-purity-classification)
5. [Why Qdrant for the Cache Store](#5-why-qdrant-for-the-cache-store)
   - [5.1 Filterable HNSW](#51-filterable-hnsw)
   - [5.2 Payload-Indexed Namespacing](#52-payload-indexed-namespacing)
   - [5.3 Named Vectors for Multi-Signal Lookup](#53-named-vectors-for-multi-signal-lookup)
6. [Implementation: A Deep Agents Middleware](#6-implementation-a-deep-agents-middleware)
   - [6.1 Middleware Hook Points](#61-middleware-hook-points)
   - [6.2 Cache Lookup and Insert](#62-cache-lookup-and-insert)
   - [6.3 Dynamic-Dispatch Fan-out (REPL Path)](#63-dynamic-dispatch-fan-out-repl-path)
7. [Invalidation Strategy](#7-invalidation-strategy)
   - [7.1 Provenance-Based vs. TTL-Based](#71-provenance-based-vs-ttl-based)
   - [7.2 Staleness Propagation](#72-staleness-propagation)
8. [False Hit Risk: Where Semantic Similarity Breaks](#8-false-hit-risk-where-semantic-similarity-breaks)
9. [Cost Model and Break-Even Analysis](#9-cost-model-and-break-even-analysis)
10. [Open Problems](#10-open-problems)
11. [Citation](#11-citation)

---

## 1. The Structural 4× Cost Multiplier

Deep Agents' primary primitive for managing long-running work is the **subagent**: a stateless, isolated agent invocation that receives a task description, a system prompt, its own tools, and its own context window, then returns a single result to the orchestrator. Each subagent bills its own API calls independently. A session that spawns three subagents in parallel for a task the main agent could handle sequentially therefore incurs roughly four API call budgets: one orchestrator plus three workers.

This isn't a bug or a configuration problem — it is the mechanism. The isolation that prevents context rot is structurally identical to the isolation that multiplies cost. LangChain's own benchmark for RLM-enabled agents at 128k tokens shows the tradeoff plainly: the RLM-enabled agent scores 0.79 vs. 0.44 for the plain agent, but it is also definitively slower and, despite using fewer *total* tokens, costs more due to output token pricing. The cheap path got better; the per-call economics got worse.

What compounds the problem at fleet scale is **redundancy**. Across a developer's daily sessions, subagents repeatedly re-derive the same results:

- "Summarize the changes in `src/api/pagination.ts` since last commit" — run every time the orchestrator branches into the file-review subagent, even if the file hasn't changed.
- "Classify this GitHub issue into one of four triage buckets" — dispatched N times for N issues, many of which share near-identical descriptions.
- "Lint this module for security anti-patterns" — invoked per-file in a map-style workflow, with identical results for files that didn't touch the relevant patterns.

The current answer is model routing: use cheaper models for workers (Haiku is 5× cheaper than Opus). This is the right lever, but it requires humans to configure a static policy at definition time, and one environment variable can silently undo it. More fundamentally, it reduces per-call cost; it doesn't eliminate redundant calls.

---

## 2. Why Provider-Side Prompt Caching Doesn't Close the Gap

LangChain's Deep Agents prompt caching blog post describes prefix-level caching via the Anthropic API's `cache_control` blocks. The mechanism discounts input tokens when the same prefix appears again in a subsequent request. This is meaningful for long-form system prompts that are re-sent with every subagent invocation.

But the cost structure of a typical subagent call looks like this:

$$\text{Cost} = \underbrace{C_{\text{in}} \cdot T_{\text{sys}}}_{\text{cached prefix}} + \underbrace{C_{\text{in}} \cdot T_{\text{task}}}_{\text{task description (uncached)}} + \underbrace{C_{\text{out}} \cdot T_{\text{result}}}_{\text{output tokens}}$$

where $C_{\text{in}}$ and $C_{\text{out}}$ are the input and output token prices, and $T_{\text{sys}}$, $T_{\text{task}}$, $T_{\text{result}}$ are the token counts for the system prompt, task payload, and generated result respectively.

Provider-side caching reduces $C_{\text{in}} \cdot T_{\text{sys}}$. It does nothing for $C_{\text{out}} \cdot T_{\text{result}}$, and $C_{\text{out}}$ is typically 3–5× higher than $C_{\text{in}}$ on current frontier models. The expensive part of running a subagent is the reasoning and generation it produces, not the preamble it receives.

Furthermore, provider-side caching is *stateless across sessions*. Each new session rebuilds the KV cache from scratch. An invocation on Monday and the same invocation on Thursday both pay full output token cost.

---

## 3. Content-Addressed Caching: The Core Idea

A **content-addressed cache** stores a function's output keyed by a deterministic hash of its inputs. If the inputs hash to the same key, the stored output is returned without re-executing. This is the same principle behind build systems like Bazel and Nix, applied here to subagent invocations.

Formally, let a subagent invocation be a function:

$$f: (\theta, \sigma, x) \rightarrow r$$

where $\theta$ is the subagent type (name, system prompt, model, tool schema version), $\sigma$ is any shared state the invocation reads, $x$ is the task-specific input payload, and $r$ is the result. If $f$ is *pure* — deterministic and free of side effects — then for identical $(\theta, \sigma, x)$ tuples, $r$ is always the same, and we can cache.

The challenge is that most useful subagents are not obviously pure. They may read files, search the web, or call external APIs. The cache design must either restrict to pure subagents or carry provenance about what was read so that the result can be invalidated when those inputs change.

---

## 4. Cache Key Design

### 4.1 Components of a Canonical Key

The cache key $k$ should capture everything that could cause two invocations to produce different outputs:

$$k = \text{Hash}(\theta_{\text{type}}, \theta_{\text{prompt}}, \theta_{\text{model}}, \theta_{\text{tools\_version}}, x_{\text{canonical}})$$

- `θ_type` — subagent name or profile identifier
- `θ_prompt` — the full system prompt, since a prompt change invalidates all prior results for that subagent type
- `θ_model` — model string including version pin (e.g., `anthropic:claude-sonnet-4-6`), because the same inputs to a different model are not guaranteed to produce equivalent results
- `θ_tools_version` — a version hash of the tool schemas exposed to the subagent, since a tool change can alter reachable behaviors
- `x_canonical` — a canonicalized serialization of the task input, with deterministic key ordering and normalized whitespace

SHA-256 of the concatenated canonical form is sufficient. For performance, `θ_type || θ_prompt || θ_model || θ_tools_version` can be computed once per subagent type at middleware initialization and cached in memory, reducing per-dispatch work to hashing `x_canonical`.

### 4.2 Exact vs. Semantic Matching

Exact hash matching handles the case where inputs are literally identical across invocations — the same file path, the same content, the same task description. This covers the map-reduce pattern well: `pages.map(page => task(...))` generates structurally identical invocations for each page element, and across runs pages already reviewed are exact matches.

*Semantic* matching is a different and riskier proposition. Two task descriptions may be linguistically similar but semantically distinct: "summarize security issues in `auth.ts`" vs. "summarize security issues in `auth.ts` focusing on session tokens" are close in embedding space but should not share a cached result. This risk is not hypothetical — a [recent paper on semantic caching for enterprise BI systems](https://arxiv.org/pdf/2606.28370) demonstrates that queries exceeding 0.95 cosine similarity under `text-embedding-3-large` can produce silent factual errors when cached hits are returned across them.

My recommendation is **exact-first, semantic-never by default**. The cache should check exact hash first; on a miss, the invocation runs. Semantic lookup is an opt-in, per-subagent-type flag, restricted to subagents where the operator can reason that near-duplicate inputs reliably produce equivalent outputs (e.g., a classification subagent with a closed label set where input variations don't change the label).

### 4.3 Purity Classification

Not every subagent is cache-eligible. A sensible default classification:

| Subagent behavior | Cache-eligible? | Reason |
|---|---|---|
| Pure summarization (no tool calls) | ✅ Yes | Output is a deterministic function of the input text |
| Classification (closed label set) | ✅ Yes | Stable for same input, no external reads |
| File review (reads local VFS) | ✅ with provenance | Safe if input hash includes file content hash, not just path |
| Web search subagent | ❌ No | External state changes unpredictably |
| Code execution subagent | ❌ No | Has side effects by construction |
| Multi-step research subagent | ❌ No | Accumulates external reads; result depends on retrieval order |

The middleware can expose a `pure: bool` flag per subagent definition. Subagents not flagged `pure` are bypassed transparently — the middleware adds no overhead to their dispatch path.

---

## 5. Why Qdrant for the Cache Store

### 5.1 Filterable HNSW

Qdrant's ANN index is HNSW with an important extension: it augments the graph with additional edges derived from indexed payload fields, guaranteeing that the graph remains connected and traversable under filtered queries. This means a payload filter like `subagent_type = "summarizer" AND model = "anthropic:claude-sonnet-4-6"` is applied *during* graph traversal rather than as a post-filter over the full result set — subgraph connectivity per payload value is pre-built at index time.

For the cache use case, this matters because most production deployments run multiple subagent types. Without payload-aware filtering, a vector lookup for a `"security-reviewer"` result could return a high-similarity result from a `"summarizer"` subagent. Qdrant's filterable HNSW eliminates this at the retrieval layer rather than requiring application-level rejection.

### 5.2 Payload-Indexed Namespacing

Each cached entry stores the following payload alongside its vector:

```json
{
  "subagent_type":   "file-summarizer",
  "model":           "anthropic:claude-sonnet-4-6",
  "tools_version":   "a3f9c2d1",
  "exact_hash":      "sha256:e3b0c44298fc...",
  "source_refs":     ["vfs://src/api/pagination.ts@commit:abc123"],
  "created_at":      "2026-07-17T10:00:00Z",
  "result":          "..."
}
```

Payload indexes on `subagent_type`, `model`, `tools_version`, and `exact_hash` enable:

1. **Exact lookup first**: filter on `exact_hash` directly, bypassing ANN entirely. An exact match returns in O(1) index lookup time.
2. **Namespace isolation**: cache entries for different models or tool versions never pollute each other.
3. **Bulk invalidation**: when a subagent's system prompt is updated (changing `tools_version`), a single payload-filtered delete removes all stale entries for that profile.

### 5.3 Named Vectors for Multi-Signal Lookup

When semantic matching is opted in for a specific subagent type, Qdrant's **named vectors** allow storing both a task-description embedding and a result-summary embedding in the same point:

```python
client.upsert(
    collection_name="subagent_cache",
    points=[
        PointStruct(
            id=entry_id,
            vector={
                "task_embed":   task_embedding,    # embed(x_canonical)
                "result_embed": result_embedding,  # embed(r[:512])
            },
            payload=payload,
        )
    ],
)
```

At lookup time, the query can be issued against `task_embed` only, with the result vector available for optional post-hoc similarity verification: if `cosine(query_result_embed, cached_result_embed) < threshold`, reject the hit even if the task similarity passes. This double-gate is the approach recommended by the enterprise BI caching paper cited above, and it materially reduces false hit rate for borderline cases.

---

## 6. Implementation: A Deep Agents Middleware

### 6.1 Middleware Hook Points

The `AgentMiddleware` protocol in Deep Agents exposes:

- `wrap_tool_call(request, handler)` — intercepts any tool execution before it reaches the underlying implementation
- `before_agent(state, runtime)` — runs once at session start, suitable for initializing the Qdrant client connection

The `task` tool, surfaced by `SubAgentMiddleware`, is just a tool call from the orchestrator's perspective. This means `wrap_tool_call` provides a clean interception point for both the static dispatch path (turn-by-turn `task` calls) and the dynamic dispatch path (REPL-generated `task` calls inside the QuickJS interpreter).

```python
from deepagents.middleware import AgentMiddleware
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

class SubagentResultCacheMiddleware(AgentMiddleware):
    def __init__(
        self,
        qdrant_url: str,
        collection: str = "subagent_cache",
        exact_only: bool = True,
        similarity_threshold: float = 0.97,
    ) -> None:
        self._url = qdrant_url
        self._collection = collection
        self._exact_only = exact_only
        self._threshold = similarity_threshold
        self._client: AsyncQdrantClient | None = None

    async def before_agent(self, state, runtime) -> None:
        self._client = AsyncQdrantClient(url=self._url)
        await self._ensure_collection()

    async def awrap_tool_call(self, request, handler):
        if request.tool_name != "task":
            return await handler(request)

        args = request.tool_input
        subagent_type = args.get("subagent_type", "default")

        if not self._is_pure(subagent_type):
            return await handler(request)

        key = self._compute_key(args)
        cached = await self._lookup_exact(key, subagent_type)

        if cached is not None:
            return cached

        result = await handler(request)
        await self._store(key, args, subagent_type, result)
        return result
```

### 6.2 Cache Lookup and Insert

The **exact lookup** path queries Qdrant as a key-value store using payload index, not ANN:

```python
async def _lookup_exact(
    self,
    exact_hash: str,
    subagent_type: str,
) -> str | None:
    hits, _ = await self._client.scroll(
        collection_name=self._collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="exact_hash",    match=MatchValue(value=exact_hash)),
                FieldCondition(key="subagent_type", match=MatchValue(value=subagent_type)),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if hits:
        return hits[0].payload["result"]
    return None
```

Note that `with_vectors=False` skips vector deserialization entirely — the payload index lookup is a conventional database point query. When `exact_only=False` and no exact hit is found, the middleware falls through to an ANN search:

```python
async def _lookup_semantic(
    self,
    task_text: str,
    subagent_type: str,
) -> str | None:
    query_vec = await self._embed(task_text)
    hits = await self._client.search(
        collection_name=self._collection,
        query_vector=("task_embed", query_vec),
        query_filter=Filter(
            must=[FieldCondition(key="subagent_type", match=MatchValue(value=subagent_type))]
        ),
        limit=1,
        score_threshold=self._threshold,
        with_payload=True,
    )
    if hits:
        return hits[0].payload["result"]
    return None
```

The **insert** path upserts a new point with both the embedding and the full payload:

```python
async def _store(
    self,
    exact_hash: str,
    args: dict,
    subagent_type: str,
    result: str,
) -> None:
    task_text = args.get("description", "")
    task_vec  = await self._embed(task_text)

    await self._client.upsert(
        collection_name=self._collection,
        points=[
            PointStruct(
                id=uuid4().hex,
                vector={"task_embed": task_vec},
                payload={
                    "exact_hash":    exact_hash,
                    "subagent_type": subagent_type,
                    "model":         args.get("model", "default"),
                    "tools_version": self._tools_version,
                    "source_refs":   self._extract_source_refs(args),
                    "created_at":    datetime.utcnow().isoformat(),
                    "result":        result,
                },
            )
        ],
    )
```

### 6.3 Dynamic-Dispatch Fan-out (REPL Path)

Dynamic subagents via `CodeInterpreterMiddleware` dispatch `task` calls from inside the QuickJS interpreter's event loop. From the middleware's perspective, these arrive through the same `wrap_tool_call` hook — the REPL bridges to native Python `task` handler functions registered with the interpreter. No additional integration is needed; the cache intercepts at the Python boundary before the call crosses into the subagent harness.

This is the most economically significant path: a programmatic fan-out over N items generates N structurally identical tool call invocations, and the cache hit rate across a `pages.map(...)` with unchanged pages approaches 100% on repeated runs.

---

## 7. Invalidation Strategy

### 7.1 Provenance-Based vs. TTL-Based

TTL-based invalidation (expire entries after $\Delta t$) is the simplest implementation but is wrong for this use case. A summarization of an unchanged file is just as valid at $t + 30\text{d}$ as at $t$; a summarization of a file that changed five minutes ago is stale immediately regardless of TTL. Expiring aggressively wastes cache value; expiring loosely causes stale results.

The correct invalidation model is **provenance-based**: an entry is valid as long as all of its `source_refs` point to content that hasn't changed. For the virtual filesystem backend Deep Agents provides, this translates to:

$$\text{valid}(e) \iff \forall r \in e.\texttt{source\_refs}: \text{hash}(\text{read}(r)) = e.\texttt{content\_hash}[r]$$

At dispatch time, the middleware resolves any file paths in the task arguments, computes their content hashes, and folds them into the cache key. This makes staleness detectable by hash mismatch rather than elapsed time.

For subagents that read from the VFS through tool calls (rather than receiving content in the task description), provenance is harder to track exactly — the middleware would need to intercept the subagent's own tool calls during a cache-priming run. This is a second-order concern; the simplest safe default is to mark VFS-reading subagents as `pure=False` unless the operator explicitly takes on provenance responsibility.

### 7.2 Staleness Propagation

When a subagent type's system prompt changes, all prior results are semantically stale. The middleware can invalidate in bulk using Qdrant's `delete_by_payload_filter`:

```python
await client.delete(
    collection_name="subagent_cache",
    points_selector=Filter(
        must=[
            FieldCondition(key="subagent_type",  match=MatchValue(value="file-summarizer")),
            FieldCondition(key="tools_version",   match=MatchValue(value=old_version_hash)),
        ]
    ),
)
```

This is an O(matching entries) operation that Qdrant executes as a payload index scan, not an ANN search — it doesn't degrade the vector index quality.

---

## 8. False Hit Risk: Where Semantic Similarity Breaks

I want to be direct about the principal failure mode: **a false cache hit is strictly worse than a cache miss**. A miss wastes cost (the subagent runs redundantly). A false hit injects a confidently-wrong result into the orchestrator's context, which the orchestrator will act on without visibility into the substitution.

The risk is highest under two conditions:

1. **Exact-hash collision** — SHA-256 collisions are theoretically possible but computationally negligible in practice. Not a real concern at any fleet scale below cryptographic adversarial pressure.

2. **Semantic match with non-equivalent semantics** (opt-in mode only) — this is the real risk. "Summarize the auth module for security issues" and "Summarize the auth module for performance issues" may embed close to each other. The mitigation is a conservative similarity threshold ($\geq 0.97$, not $0.85$), per-subagent-type opt-in rather than global, and an optional result-embedding double-gate as described in §5.3.

A recommended production posture: **run `exact_only=True` for 30 days, instrument miss rate, then evaluate whether semantic matching is worth enabling for specific subagent types based on empirical hit-rate data**. The 30-day exact run also gives you the baseline cost data to calculate whether the cache is paying for itself.

---

## 9. Cost Model and Break-Even Analysis

Let:
- $N$ = number of subagent dispatches per day across the fleet
- $\rho$ = cache hit rate (fraction of dispatches that find a valid cache entry)
- $C_{\text{agent}}$ = average cost per subagent invocation (input + output tokens at provider rates)
- $C_{\text{embed}}$ = cost to embed the task description for each dispatch
- $C_{\text{qdrant}}$ = amortized Qdrant storage + query cost per dispatch

The daily net saving is:

$$\Delta = N \cdot \left[ \rho \cdot (C_{\text{agent}} - C_{\text{embed}} - C_{\text{qdrant}}) - (1 - \rho) \cdot (C_{\text{embed}} + C_{\text{qdrant}}) \right]$$

Simplifying:

$$\Delta = N \cdot \left[ \rho \cdot C_{\text{agent}} - (C_{\text{embed}} + C_{\text{qdrant}}) \right]$$

The cache breaks even when $\rho > \frac{C_{\text{embed}} + C_{\text{qdrant}}}{C_{\text{agent}}}$.

For current representative numbers: $C_{\text{agent}} \approx \$0.01$–$\$0.05$ per subagent invocation (depending on model and task length), $C_{\text{embed}} \approx \$0.00002$ (FastEmbed on-prem or small embedding model), $C_{\text{qdrant}} \approx \$0.000005$ (self-hosted at scale). The break-even hit rate is approximately **0.2%–0.5%** — that is, the cache pays for itself if even 1 in 200 dispatches is a hit.

Map-reduce workflows over unchanged inputs should approach hit rates in the 50–95% range on repeated runs. The cache is almost certainly economically justified for any team running daily scheduled agent workflows or CI-integrated code review agents.

The point of the analysis isn't to claim specific numbers — fleet data will vary — but to note that **the marginal cost of embedding is so low relative to subagent invocation cost that the break-even bar is trivially low**. This is unlike semantic caching at the end-user query layer, where the embedding cost is non-trivial relative to simple queries.

---

## 10. Open Problems

Several design questions remain genuinely open, and I think they are more interesting than the implementation described above.

**Tool-call provenance inside a running subagent.** The cache key is computed from the *input* to a subagent, not from what it reads during execution. For subagents that call VFS tools autonomously, the actual result depends on file contents at read time. Tracking this requires either (a) intercepting the subagent's own tool calls during a designated "priming run" to extract the read set, or (b) restricting caching to subagents that receive all required content in their task description. Neither is fully satisfying. A principled solution would look like lineage tracking in data pipeline frameworks (dbt, Airflow) applied to agent tool access patterns.

**Non-deterministic model outputs.** The same inputs to the same model at `temperature > 0` produce different outputs across runs. The cache returns one specific realization. For summarization and classification at moderate temperatures, this is unlikely to matter operationally — the outputs are semantically equivalent. But for generative tasks (drafting code, writing prose), returning a cached result can suppress a better answer the model would have produced today with a different random seed. The cache trades output variability for cost savings; this is often the right trade, but the operator should make it consciously.

**Cross-model cache equivalence.** A result produced by `claude-sonnet-4-6` is cached separately from the same task on `gpt-5.5`. They could share a cache if you believe the outputs are equivalent for that subagent type — which is sometimes true for pure classification but rarely true for nuanced generation. Practical cross-model caching would require human-validated equivalence for specific (subagent type, model pair) combinations, which is a significant operational investment.

**Adaptive purity classification.** Currently, `pure: bool` is a static flag set at subagent definition time. A more sophisticated system would learn purity probabilistically from trace data — classifying a subagent as "effectively pure" if its outputs show low variance across identical inputs in practice, even if it formally reads external state. This reduces reliance on operator judgment and could recover cache value for impure subagents on quiescent inputs.

**Multi-tenant cache isolation.** A fleet where multiple users share subagent types raises the question of whether cached results should be shared across users. For code review (VFS-backed, user-specific files), sharing is unsafe. For classification against a shared label taxonomy, sharing across the org is probably fine and dramatically improves fleet-wide hit rates. The OKF model (per-scope, per-resource metadata) suggests a framing: cache entries with `scope: personal` vs. `scope: org` propagate differently.

---

## 11. Citation

```
@misc{subagent-result-cache-2026,
  title   = {Content-Addressed Subagent Result Caching in Deep Agents},
  author  = {Inamdar, Mihir},
  year    = {2026},
  url     = {https://inamdarmihir.github.io/aihive/posts/subagent-result-cache/},
  note    = {Blog post, July 2026}
}
```

**Referenced work:**

- LangChain Team (2026). *Introducing Dynamic Subagents in Deep Agents*. [langchain.com/blog](https://www.langchain.com/blog/introducing-dynamic-subagents-in-deep-agents)
- LangChain Team (2026). *How to Use RLMs in Deep Agents*. [langchain.com/blog](https://www.langchain.com/blog/how-to-use-rlms-in-deep-agents)
- LangChain Team (2026). *Running Untrusted Agent Code Without a Sandbox*. [langchain.com/blog](https://www.langchain.com/blog/running-untrusted-agent-code-without-a-sandbox)
- LangChain Team (2026). *Prompt Caching with Deep Agents*. [langchain.com/blog](https://www.langchain.com/blog/deep-agents-prompt-caching)
- Zhang, A. et al. (2025). *Recursive Language Models*. [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
- Anonymous (2026). *Conversational Query Engine for Mixed-Modality Heterogeneous Enterprise Data Sources*. [arXiv:2606.28370](https://arxiv.org/pdf/2606.28370) — §6 on verified semantic cache false-hit risk.
- Qdrant Team (2025). *Combining Vector Search and Filtering*. [qdrant.tech/documentation](https://qdrant.tech/course/essentials/day-2/filterable-hnsw/)
- Qdrant Team (2025). *Collections: Named Vectors and Quantization*. [qdrant.tech/documentation](https://qdrant.tech/documentation/manage-data/collections/)
