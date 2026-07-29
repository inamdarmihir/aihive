---
title: "Content-Addressed Subagent Result Caching in Deep Agents"
date: 2026-07-20
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
   - [4.2 Canonicalization and `_compute_key`](#42-canonicalization-and-_compute_key)
   - [4.3 Exact vs. Semantic Matching](#43-exact-vs-semantic-matching)
   - [4.4 Purity Classification](#44-purity-classification)
5. [Why Qdrant for the Cache Store](#5-why-qdrant-for-the-cache-store)
   - [5.1 Filterable HNSW](#51-filterable-hnsw)
   - [5.2 Payload-Indexed Namespacing](#52-payload-indexed-namespacing)
   - [5.3 Named Vectors for Multi-Signal Lookup](#53-named-vectors-for-multi-signal-lookup)
   - [5.4 Collection Creation and Payload Indexes](#54-collection-creation-and-payload-indexes)
6. [Implementation: A Deep Agents Middleware](#6-implementation-a-deep-agents-middleware)
   - [6.1 Middleware Hook Points](#61-middleware-hook-points)
   - [6.2 Collection Bootstrap and Helper Methods](#62-collection-bootstrap-and-helper-methods)
   - [6.3 Cache Lookup and Insert](#63-cache-lookup-and-insert)
   - [6.4 Dynamic-Dispatch Fan-out (REPL Path)](#64-dynamic-dispatch-fan-out-repl-path)
7. [Worked Example: Map-Reduce Over Pages](#7-worked-example-map-reduce-over-pages)
8. [Invalidation Strategy](#8-invalidation-strategy)
   - [8.1 Provenance-Based vs. TTL-Based](#81-provenance-based-vs-ttl-based)
   - [8.2 Provenance Verification](#82-provenance-verification)
   - [8.3 Staleness Propagation](#83-staleness-propagation)
9. [False Hit Risk: Where Semantic Similarity Breaks](#9-false-hit-risk-where-semantic-similarity-breaks)
10. [Cost Model and Break-Even Analysis](#10-cost-model-and-break-even-analysis)
11. [Open Problems](#11-open-problems)
12. [Citation](#12-citation)

---

## 1. The Structural 4× Cost Multiplier

Deep Agents' primary primitive for managing long-running work is the **subagent**: a stateless, isolated agent invocation that receives a task description, a system prompt, its own tools, and its own context window, then returns a single result to the orchestrator. Each subagent bills its own API calls independently. A session that spawns three subagents in parallel for a task the main agent could handle sequentially therefore incurs roughly four API call budgets: one orchestrator plus three workers.

This isn't a bug or a configuration problem — it is the mechanism. The isolation that prevents context rot is structurally identical to the isolation that multiplies cost. LangChain's own benchmark for RLM-enabled agents at 128k tokens shows the tradeoff plainly: the RLM-enabled agent scores 0.79 vs. 0.44 for the plain agent, but it is also definitively slower and, despite using fewer *total* tokens, costs more due to output token pricing. The cheap path got better; the per-call economics got worse.

### A Concrete Cost Walkthrough

Consider a representative code-review session: an orchestrator (Sonnet) fans out three workers (Haiku) to summarize three modules, then synthesizes. Using mid-2026 Anthropic list prices as a reference (\$3 / \$15 per MTok input/output for Sonnet; \$0.80 / \$4 per MTok for Haiku), a single pass looks like this:

| Role | Model | Input tokens | Output tokens | Cost |
|---|---|---|---|---|
| Orchestrator (plan + synthesize) | Sonnet | 8,000 | 1,200 | \$0.042 |
| Worker A: summarize `auth/` | Haiku | 6,000 | 800 | \$0.008 |
| Worker B: summarize `billing/` | Haiku | 6,000 | 800 | \$0.008 |
| Worker C: summarize `api/` | Haiku | 6,000 | 800 | \$0.008 |
| **Session total** | | | | **\$0.066** |

The three workers account for about 36% of session cost. That fraction climbs when workers use Sonnet or when the fan-out grows to tens of pages. Across twenty daily review sessions with redundant re-summarization, worker waste lands around \$0.50–\$2.00 per developer — tens of dollars per day for a CI-integrated team.

What compounds the problem at fleet scale is **redundancy**. Across daily sessions, subagents repeatedly re-derive the same results:

- "Summarize the changes in `src/api/pagination.ts` since last commit" — run every time the orchestrator branches into the file-review subagent, even if the file hasn't changed.
- "Classify this GitHub issue into one of four triage buckets" — dispatched N times for N issues, many of which share near-identical descriptions.
- "Lint this module for security anti-patterns" — invoked per-file in a map-style workflow, with identical results for files that didn't touch the relevant patterns.

The current answer is model routing: use cheaper models for workers (Haiku is roughly 5× cheaper than Opus on output). This is the right lever, but it requires humans to configure a static policy at definition time, and one environment variable can silently undo it. More fundamentally, it reduces per-call cost; it doesn't eliminate redundant calls.

---

## 2. Why Provider-Side Prompt Caching Doesn't Close the Gap

LangChain's Deep Agents prompt caching blog post describes prefix-level caching via the Anthropic API's `cache_control` blocks. The mechanism discounts input tokens when the same prefix appears again in a subsequent request. This is meaningful for long-form system prompts that are re-sent with every subagent invocation.

But the cost structure of a typical subagent call looks like this:

$$\text{Cost} = \underbrace{C_{\text{in}} \cdot T_{\text{sys}}}_{\text{cached prefix}} + \underbrace{C_{\text{in}} \cdot T_{\text{task}}}_{\text{task description (uncached)}} + \underbrace{C_{\text{out}} \cdot T_{\text{result}}}_{\text{output tokens}}$$

where $C_{\text{in}}$ and $C_{\text{out}}$ are the input and output token prices, and $T_{\text{sys}}$, $T_{\text{task}}$, $T_{\text{result}}$ are the token counts for the system prompt, task payload, and generated result respectively. In plain English: the bill is cheap input for the system prompt, cheap input for the task text, and expensive output for the result — and provider-side caching only discounts the first term.

Provider-side caching reduces $C_{\text{in}} \cdot T_{\text{sys}}$. It does nothing for $C_{\text{out}} \cdot T_{\text{result}}$, and $C_{\text{out}}$ is typically 3–5× higher than $C_{\text{in}}$ on current frontier models. Using the worker numbers from §1: a Haiku worker with 6k input / 800 output pays roughly \$0.0048 for input and \$0.0032 for output. Even if prompt caching eliminated all input cost, you'd still pay the full \$0.0032 for output on every redundant run.

Furthermore, provider-side caching is *stateless across sessions*. Each new session rebuilds the KV cache from scratch. An invocation on Monday and the same invocation on Thursday both pay full output token cost.

---

## 3. Content-Addressed Caching: The Core Idea

A **content-addressed cache** stores a function's output keyed by a deterministic hash of its inputs. If the inputs hash to the same key, the stored output is returned without re-executing. This is the same principle behind build systems like Bazel and Nix, applied here to subagent invocations.

Formally, let a subagent invocation be a function:

$$f: (\theta, \sigma, x) \rightarrow r$$

where $\theta$ is the subagent type (name, system prompt, model, tool schema version), $\sigma$ is any shared state the invocation reads, $x$ is the task-specific input payload, and $r$ is the result. If $f$ is *pure* — deterministic and free of side effects — then for identical $(\theta, \sigma, x)$ tuples, $r$ is always the same, and we can cache. In plain English: unchanged definition, state, and task payload means unchanged result — so we should not pay to regenerate it.

The challenge is that most useful subagents are not obviously pure. They may read files, search the web, or call external APIs. The cache design must either restrict to pure subagents or carry provenance about what was read so that the result can be invalidated when those inputs change.

---

## 4. Cache Key Design

The key is the correctness surface of the whole system. A key that under-specifies inputs produces false hits; a key that over-specifies them produces unnecessary misses. This section walks through the components, the canonicalization code, the exact-vs-semantic choice, and which subagents are eligible at all.

### 4.1 Components of a Canonical Key

The cache key $k$ should capture everything that could cause two invocations to produce different outputs:

$$k = \text{Hash}(\theta_{\text{type}}, \theta_{\text{prompt}}, \theta_{\text{model}}, \theta_{\text{tools\_version}}, x_{\text{canonical}})$$

- `θ_type` — subagent name or profile identifier
- `θ_prompt` — the full system prompt, since a prompt change invalidates all prior results for that subagent type
- `θ_model` — model string including version pin (e.g., `anthropic:claude-sonnet-4-6`), because the same inputs to a different model are not guaranteed to produce equivalent results
- `θ_tools_version` — a version hash of the tool schemas exposed to the subagent, since a tool change can alter reachable behaviors
- `x_canonical` — a canonicalized serialization of the task input, with deterministic key ordering and normalized whitespace

SHA-256 of the concatenated canonical form is sufficient. For performance, `θ_type || θ_prompt || θ_model || θ_tools_version` can be computed once per subagent type at middleware initialization and cached in memory, reducing per-dispatch work to hashing `x_canonical`.

### 4.2 Canonicalization and `_compute_key`

Canonicalization is where most cache-key bugs hide. Two task payloads that are semantically identical but differ in key order, trailing whitespace, or path separators will hash differently and silently miss. The implementation below is deliberately strict: JSON with sorted keys, NFC-normalized Unicode, and path normalization for any `source_refs` embedded in the description.

```python
import hashlib, json, unicodedata
from pathlib import PurePosixPath


def _canonicalize(obj: object) -> str:
    """Deterministic serialization for cache-key material."""
    if isinstance(obj, dict):
        items = {k: _canonicalize(v) for k, v in sorted(obj.items())}
        return json.dumps(items, separators=(",", ":"), ensure_ascii=False)
    if isinstance(obj, (list, tuple)):
        return json.dumps(
            [_canonicalize(v) for v in obj], separators=(",", ":"), ensure_ascii=False
        )
    if isinstance(obj, str):
        return " ".join(unicodedata.normalize("NFC", obj).split())
    if isinstance(obj, (int, float, bool)) or obj is None:
        return json.dumps(obj)
    return _canonicalize(str(obj))


def _normalize_path(path: str) -> str:
    return str(PurePosixPath(path))


class SubagentResultCacheMiddleware:
    # ... __init__ and other methods shown in §6 ...

    def _profile_prefix(self, subagent_type: str) -> str:
        if subagent_type not in self._prefix_cache:
            profile = self._profiles[subagent_type]
            material = _canonicalize({
                "type": subagent_type,
                "prompt": profile["system_prompt"],
                "model": profile["model"],
                "tools": profile["tools_version"],
            })
            self._prefix_cache[subagent_type] = hashlib.sha256(
                material.encode("utf-8")
            ).hexdigest()
        return self._prefix_cache[subagent_type]

    def _compute_key(self, args: dict) -> str:
        """Fold file content hashes into x_canonical so path-only identity cannot false-hit after an edit."""
        subagent_type = args.get("subagent_type", "default")
        source_refs = self._extract_source_refs(args)
        content_hashes = {ref: self._content_hash(ref) for ref in source_refs}
        x_canonical = _canonicalize({
            "description": args.get("description", ""),
            "files": {_normalize_path(p): h for p, h in sorted(content_hashes.items())},
        })
        material = f"{self._profile_prefix(subagent_type)}:{x_canonical}"
        return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()
```

Two design choices deserve comment. First, content hashes are folded into the key rather than checked only at invalidation time: a changed file produces a different key, so the old entry is simply never looked up. Second, the profile prefix is memoized in memory — recomputing SHA-256 over a multi-kilobyte system prompt on every dispatch is wasteful when the profile is static for the session.

### 4.3 Exact vs. Semantic Matching

Exact hash matching handles the case where inputs are literally identical across invocations — the same file path, the same content, the same task description. This covers the map-reduce pattern well: `pages.map(page => task(...))` generates structurally identical invocations for each page element, and across runs pages already reviewed are exact matches.

*Semantic* matching is a different and riskier proposition. Two task descriptions may be linguistically similar but semantically distinct: "summarize security issues in `auth.ts`" vs. "summarize security issues in `auth.ts` focusing on session tokens" are close in embedding space but should not share a cached result. This risk is not hypothetical — a recent paper, **Conversational Query Engine for Mixed-Modality Heterogeneous Enterprise Data Sources** ([Anonymous 2026](https://arxiv.org/pdf/2606.28370)), demonstrates that queries exceeding 0.95 cosine similarity under `text-embedding-3-large` can produce silent factual errors when cached hits are returned across them.

My recommendation is **exact-first, semantic-never by default**. The cache should check exact hash first; on a miss, the invocation runs. Semantic lookup is an opt-in, per-subagent-type flag, restricted to subagents where the operator can reason that near-duplicate inputs reliably produce equivalent outputs (e.g., a classification subagent with a closed label set where input variations don't change the label).

### 4.4 Purity Classification

Not every subagent is cache-eligible. A sensible default classification:

| Subagent behavior | Cache-eligible? | Reason |
|---|---|---|
| Pure summarization (no tool calls) | ✅ Yes | Output is a deterministic function of the input text |
| Classification (closed label set) | ✅ Yes | Stable for same input, no external reads |
| File review (reads local VFS) | ✅ with provenance | Safe if input hash includes file content hash, not just path |
| Web search subagent | ❌ No | External state changes unpredictably |
| Code execution subagent | ❌ No | Has side effects by construction |
| Multi-step research subagent | ❌ No | Accumulates external reads; result depends on retrieval order |

The middleware can expose a `pure: bool` flag per subagent definition. Subagents not flagged `pure` are bypassed transparently — the middleware adds no overhead to their dispatch path beyond a dictionary lookup.

---

## 5. Why Qdrant for the Cache Store

A content-addressed cache needs two retrieval modes: exact key lookup (the common path) and optional semantic nearest-neighbor search (the opt-in path). Qdrant supports both in one collection via payload indexes and named vectors, which is why I prefer it over a Redis-only design for this workload.

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
  "content_hashes":  {"vfs://src/api/pagination.ts": "sha256:9f86d0..."},
  "created_at":      "2026-07-17T10:00:00Z",
  "stale":           false,
  "result":          "..."
}
```

Payload indexes on `subagent_type`, `model`, `tools_version`, `exact_hash`, and `stale` enable:

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

### 5.4 Collection Creation and Payload Indexes

Collection bootstrap mirrors the pattern used in the semantic query cache: named vectors sized to the embedding model, plus payload indexes on every field that appears in a filter.

```python
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

EMBED_DIM = 384  # FastEmbed / all-MiniLM-L6-v2; use 1536 for text-embedding-3-small

async def ensure_subagent_cache_collection(
    client: AsyncQdrantClient,
    collection: str = "subagent_cache",
) -> None:
    existing = {c.name for c in await client.get_collections()}
    if collection not in existing:
        await client.create_collection(
            collection_name=collection,
            vectors_config={
                "task_embed": VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                "result_embed": VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            },
        )
    for field, schema in [
        ("subagent_type", PayloadSchemaType.KEYWORD),
        ("model", PayloadSchemaType.KEYWORD),
        ("tools_version", PayloadSchemaType.KEYWORD),
        ("exact_hash", PayloadSchemaType.KEYWORD),
        ("source_refs", PayloadSchemaType.KEYWORD),
        ("stale", PayloadSchemaType.BOOL),
        ("created_at", PayloadSchemaType.KEYWORD),
    ]:
        await client.create_payload_index(
            collection_name=collection, field_name=field, field_schema=schema
        )
```

Two vectors are declared even when `exact_only=True`. Exact lookups ignore them (`with_vectors=False`), but declaring both up front avoids a migration when semantic matching is later enabled. Payload indexes on `exact_hash` and `stale` are mandatory; without them every exact lookup degrades to a full payload scan.

---

## 6. Implementation: A Deep Agents Middleware

This section assembles the pieces into a working `AgentMiddleware`. The interception point is deliberately narrow: only `task` tool calls for profiles marked pure are eligible; everything else passes through untouched.

### 6.1 Middleware Hook Points

The `AgentMiddleware` protocol in Deep Agents exposes:

- `wrap_tool_call(request, handler)` — intercepts any tool execution before it reaches the underlying implementation
- `before_agent(state, runtime)` — runs once at session start, suitable for initializing the Qdrant client connection

The `task` tool, surfaced by `SubAgentMiddleware`, is just a tool call from the orchestrator's perspective. This means `wrap_tool_call` provides a clean interception point for both the static dispatch path (turn-by-turn `task` calls) and the dynamic dispatch path (REPL-generated `task` calls inside the QuickJS interpreter).

```python
from datetime import datetime, timezone
from uuid import uuid4
from deepagents.middleware import AgentMiddleware
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    PointStruct, Filter, FieldCondition, MatchValue,
)


class SubagentResultCacheMiddleware(AgentMiddleware):
    def __init__(
        self,
        qdrant_url: str,
        profiles: dict[str, dict],
        collection: str = "subagent_cache",
        exact_only: bool = True,
        similarity_threshold: float = 0.97,
        vfs_root: str | None = None,
    ) -> None:
        self._url = qdrant_url
        self._profiles = profiles  # {type: {system_prompt, model, tools_version, pure}}
        self._collection = collection
        self._exact_only = exact_only
        self._threshold = similarity_threshold
        self._vfs_root = vfs_root
        self._client: AsyncQdrantClient | None = None
        self._prefix_cache: dict[str, str] = {}

    async def before_agent(self, state, runtime) -> None:
        self._client = AsyncQdrantClient(url=self._url)
        await ensure_subagent_cache_collection(self._client, self._collection)

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
            if await self._verify_provenance(cached["point_id"], cached["payload"]):
                return cached["result"]

        if not self._exact_only:
            semantic = await self._lookup_semantic(
                args.get("description", ""), subagent_type
            )
            if semantic is not None:
                return semantic

        result = await handler(request)
        await self._store(key, args, subagent_type, result)
        return result
```

### 6.2 Collection Bootstrap and Helper Methods

`_ensure_collection` is the function from §5.4, called once from `before_agent`. The remaining helpers complete the surface used in §4.2 and §6.1: purity gating, source-ref extraction, and content hashing against the VFS.

```python
    def _is_pure(self, subagent_type: str) -> bool:
        profile = self._profiles.get(subagent_type)
        return bool(profile and profile.get("pure", False))

    def _extract_source_refs(self, args: dict) -> list[str]:
        """Prefer explicit `files`/`paths`; else scan backtick-quoted file-like tokens."""
        import re
        refs: list[str] = []
        for p in args.get("files") or args.get("paths") or []:
            refs.append(_normalize_path(str(p)))
        for match in re.findall(r"`([^`]+)`", args.get("description", "")):
            if "/" in match or match.endswith((".ts", ".py", ".md", ".tsx", ".js")):
                refs.append(_normalize_path(match))
        seen, ordered = set(), []
        for r in refs:
            if r not in seen:
                seen.add(r)
                ordered.append(r)
        return ordered

    def _content_hash(self, path: str) -> str:
        from pathlib import Path
        root = Path(self._vfs_root) if self._vfs_root else Path(".")
        target = root / path.lstrip("/")
        if not target.is_file():
            return "sha256:missing"
        return "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
```

`_extract_source_refs` is intentionally conservative: it only claims paths the orchestrator named or that appear as backtick-quoted file-like tokens. Paths discovered only inside the subagent's own tool loop are invisible here — that is the provenance open problem in §11.

### 6.3 Cache Lookup and Insert

The **exact lookup** path queries Qdrant as a key-value store using the payload index, not ANN (`with_vectors=False` skips vector deserialization entirely):

```python
    async def _lookup_exact(self, exact_hash: str, subagent_type: str) -> dict | None:
        hits, _ = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="exact_hash", match=MatchValue(value=exact_hash)),
                FieldCondition(key="subagent_type", match=MatchValue(value=subagent_type)),
                FieldCondition(key="stale", match=MatchValue(value=False)),
            ]),
            limit=1, with_payload=True, with_vectors=False,
        )
        if hits:
            return {
                "point_id": hits[0].id,
                "payload": hits[0].payload,
                "result": hits[0].payload["result"],
            }
        return None
```

When `exact_only=False` and no exact hit is found, the middleware falls through to ANN search:

```python
    async def _lookup_semantic(self, task_text: str, subagent_type: str) -> str | None:
        query_vec = await self._embed(task_text)
        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=("task_embed", query_vec),
            query_filter=Filter(must=[
                FieldCondition(key="subagent_type", match=MatchValue(value=subagent_type)),
                FieldCondition(key="stale", match=MatchValue(value=False)),
            ]),
            limit=1, score_threshold=self._threshold, with_payload=True,
        )
        return hits[0].payload["result"] if hits else None
```

The **insert** path upserts both embeddings and the full payload:

```python
    async def _store(
        self, exact_hash: str, args: dict, subagent_type: str, result: str
    ) -> None:
        task_text = args.get("description", "")
        source_refs = self._extract_source_refs(args)
        content_hashes = {ref: self._content_hash(ref) for ref in source_refs}
        profile = self._profiles.get(subagent_type, {})

        await self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(
                id=uuid4().hex,
                vector={
                    "task_embed": await self._embed(task_text),
                    "result_embed": await self._embed(result[:512]),
                },
                payload={
                    "exact_hash": exact_hash,
                    "subagent_type": subagent_type,
                    "model": profile.get("model", "default"),
                    "tools_version": profile.get("tools_version", "unset"),
                    "source_refs": source_refs,
                    "content_hashes": content_hashes,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "stale": False,
                    "result": result,
                },
            )],
        )
```

### 6.4 Dynamic-Dispatch Fan-out (REPL Path)

Dynamic subagents via `CodeInterpreterMiddleware` dispatch `task` calls from inside the QuickJS interpreter's event loop. From the middleware's perspective, these arrive through the same `wrap_tool_call` hook — the REPL bridges to native Python `task` handler functions registered with the interpreter. No additional integration is needed; the cache intercepts at the Python boundary before the call crosses into the subagent harness.

This is the most economically significant path: a programmatic fan-out over N items generates N structurally identical tool call invocations, and the cache hit rate across a `pages.map(...)` with unchanged pages approaches 100% on repeated runs. The next section makes that concrete.

---

## 7. Worked Example: Map-Reduce Over Pages

To make the economics tangible, consider a documentation-review agent that fans out over a fixed corpus of 40 markdown pages every weekday morning. The orchestrator runs a QuickJS map:

```javascript
const pages = fs.glob("docs/**/*.md");
const summaries = await Promise.all(
  pages.map((page) =>
    task({
      subagent_type: "doc-summarizer",
      description: `Summarize structural changes and TODOs in \`${page}\``,
      files: [page],
    })
  )
);
return synthesize(summaries);
```

Assume `doc-summarizer` is marked `pure: true`, uses Haiku, and averages \$0.008 per invocation (matching the worker numbers in §1). The corpus evolves slowly: on a typical day, 2–4 pages change.

| Run | Pages changed overnight | Exact hits | Misses | Hit rate | Worker API spend |
|---|---|---|---|---|---|
| Day 1 (cold) | — | 0 | 40 | 0% | \$0.320 |
| Day 2 | 3 | 37 | 3 | 92.5% | \$0.024 |
| Day 3 | 2 | 38 | 2 | 95.0% | \$0.016 |
| Day 4 | 4 | 36 | 4 | 90.0% | \$0.032 |
| Day 5 | 1 | 39 | 1 | 97.5% | \$0.008 |
| **Week total** | | **150** | **50** | **75%** | **\$0.400** |

Without the cache, the week costs \$1.60 in worker API spend (5 × \$0.320). With exact-hash caching and content-hashed file keys, it costs \$0.40 — a 75% reduction. Embedding and Qdrant overhead for 200 lookups is under \$0.01 for the week (see §10).

Two observations follow. First, the cold day dominates weekly cost; any workflow that re-runs daily over a mostly-stable corpus is an excellent fit. Second, hit rate is bounded by the change rate of the corpus, not by embedding quality — exact matching with content hashes makes that relationship linear and predictable. Semantic matching would not improve Day 2–5 numbers here, because the task strings for unchanged pages are already exact matches.

---

## 8. Invalidation Strategy

Even with content hashes in the key, operators need an explicit invalidation path for prompt edits, tool-schema bumps, and the rare race where a file changes between key computation and result return.

### 8.1 Provenance-Based vs. TTL-Based

TTL-based invalidation (expire entries after $\Delta t$) is the simplest implementation but is wrong for this use case. A summarization of an unchanged file is just as valid at $t + 30\text{d}$ as at $t$; a summarization of a file that changed five minutes ago is stale immediately regardless of TTL.

The correct invalidation model is **provenance-based**: an entry is valid as long as all of its `source_refs` point to content that hasn't changed:

$$\text{valid}(e) \iff \forall r \in e.\texttt{source\_refs}: \text{hash}(\text{read}(r)) = e.\texttt{content\_hash}[r]$$

In plain English: an entry stays valid only while every file it claimed to depend on still hashes to the value recorded at write time. Folding content hashes into the key (§4.2) makes most staleness detectable by key mismatch; the verification function below is a second line of defense for entries stored under an older keying scheme.

For subagents that read from the VFS through tool calls rather than receiving content in the task description, provenance is harder — the middleware would need to intercept the subagent's own tool calls during a priming run. The safest default is `pure=False` unless the operator takes on that responsibility.

### 8.2 Provenance Verification

```python
    async def _verify_provenance(self, point_id: str | int, payload: dict) -> bool:
        """Re-hash every recorded source_ref; soft-invalidate on mismatch."""
        content_hashes: dict = payload.get("content_hashes") or {}
        if not content_hashes:
            return True  # pure-without-files (e.g. classification)
        for ref, expected in content_hashes.items():
            if self._content_hash(ref) != expected:
                await self._client.set_payload(
                    collection_name=self._collection,
                    payload={"stale": True},
                    points=[point_id],
                )
                return False
        return True
```

Marking `stale: True` rather than deleting preserves the entry for analytics while ensuring subsequent lookups exclude it via the `stale=False` filter.

### 8.3 Staleness Propagation

When a subagent type's system prompt changes, all prior results are semantically stale. The middleware can invalidate in bulk using Qdrant's `delete` with a payload filter:

```python
async def invalidate_profile(
    client: AsyncQdrantClient,
    collection: str,
    subagent_type: str,
    old_tools_version: str,
) -> None:
    await client.delete(
        collection_name=collection,
        points_selector=Filter(must=[
            FieldCondition(key="subagent_type", match=MatchValue(value=subagent_type)),
            FieldCondition(key="tools_version", match=MatchValue(value=old_tools_version)),
        ]),
    )
```

This is an O(matching entries) payload-index scan, not an ANN search — it doesn't degrade the vector index. The same pattern applies to model swaps: filter on `model` and either delete or soft-mark `stale`.

---

## 9. False Hit Risk: Where Semantic Similarity Breaks

I want to be direct about the principal failure mode: **a false cache hit is strictly worse than a cache miss**. A miss wastes cost (the subagent runs redundantly). A false hit injects a confidently-wrong result into the orchestrator's context, which the orchestrator will act on without visibility into the substitution.

The risk is highest under two conditions:

1. **Exact-hash collision** — SHA-256 collisions are theoretically possible but computationally negligible in practice. Not a real concern at any fleet scale below cryptographic adversarial pressure.

2. **Semantic match with non-equivalent semantics** (opt-in mode only) — this is the real risk. "Summarize the auth module for security issues" and "Summarize the auth module for performance issues" may embed close to each other. The mitigation is a conservative similarity threshold ($\geq 0.97$, not $0.85$), per-subagent-type opt-in rather than global, and an optional result-embedding double-gate as described in §5.3.

### Threshold Sensitivity: A Thought Experiment

Suppose we label 200 pairs of task descriptions from a code-review fleet as *equivalent* or *distinct*, embed with a 384-dim MiniLM model, and sweep the cosine threshold:

| Threshold $\tau$ | Approx. precision | Approx. recall | Operational reading |
|---|---|---|---|
| 0.85 | 0.72 | 0.94 | Too permissive — ~28% of hits are wrong |
| 0.90 | 0.84 | 0.81 | Still unsafe for synthesizer context |
| 0.95 | 0.93 | 0.55 | Borderline; needs result-embed double-gate |
| 0.97 | 0.97 | 0.38 | Acceptable for closed-label classification |
| 0.99 | 0.99 | 0.12 | Nearly exact-match; rarely worth embedding cost |

Lowering the threshold buys recall at the expense of silently wrong answers. For subagent result caching, I treat precision below ~0.97 as unacceptable — which is exactly where semantic matching starts to look like a poor substitute for exact hashing, and why the default is `exact_only=True`.

A recommended production posture: **run `exact_only=True` for 30 days, instrument miss rate, then evaluate semantic matching per subagent type from empirical hit-rate data**. The same window gives baseline cost data for break-even.

---

## 10. Cost Model and Break-Even Analysis

Let:
- $N$ = number of subagent dispatches per day across the fleet
- $\rho$ = cache hit rate (fraction of dispatches that find a valid cache entry)
- $C_{\text{agent}}$ = average cost per subagent invocation (input + output tokens at provider rates)
- $C_{\text{embed}}$ = cost to embed the task description for each dispatch
- $C_{\text{qdrant}}$ = amortized Qdrant storage + query cost per dispatch

The daily net saving simplifies to $\Delta = N \cdot [\rho \cdot C_{\text{agent}} - (C_{\text{embed}} + C_{\text{qdrant}})]$, which expands from:

$$\Delta = N \cdot \left[ \rho \cdot (C_{\text{agent}} - C_{\text{embed}} - C_{\text{qdrant}}) - (1 - \rho) \cdot (C_{\text{embed}} + C_{\text{qdrant}}) \right]$$

The cache breaks even when $\rho > \frac{C_{\text{embed}} + C_{\text{qdrant}}}{C_{\text{agent}}}$. In plain English: hit rate must exceed (embedding + store cost) / (full agent cost). Because embeddings are cheap relative to LLM calls, that fraction is tiny.

For representative numbers — $C_{\text{agent}} \approx \$0.01$–$\$0.05$, $C_{\text{embed}} \approx \$0.00002$, $C_{\text{qdrant}} \approx \$0.000005$ — the break-even hit rate is approximately **0.2%–0.5%**: the cache pays for itself if even 1 in 200 dispatches is a hit.

### Worked Example: 1,000 Dispatches / Day

Take a mid-size fleet: $N = 1000$ dispatches/day, mixed Haiku workers at $C_{\text{agent}} = \$0.01$, on-prem FastEmbed, self-hosted Qdrant.

| Hit rate $\rho$ | Gross agent cost avoided | Embed + Qdrant overhead | Net daily saving $\Delta$ |
|---|---|---|---|
| 0% (cold / disabled) | \$0.00 | \$0.025 | −\$0.025 |
| 1% | \$0.10 | \$0.025 | \$0.075 |
| 10% | \$1.00 | \$0.025 | \$0.975 |
| 50% (map-reduce steady state) | \$5.00 | \$0.025 | \$4.975 |
| 75% (docs corpus from §7) | \$7.50 | \$0.025 | \$7.475 |
| 90% | \$9.00 | \$0.025 | \$8.975 |

Annualized at 50% hit rate: roughly \$1,800/year saved on a thousand-dispatch fleet, against infrastructure measured in tens of dollars. Map-reduce over unchanged inputs should approach 50–95% hit rates on repeated runs. The point isn't specific numbers — fleet data will vary — but that **the marginal cost of embedding is so low relative to subagent invocation that the break-even bar is trivially low**, unlike semantic caching at the end-user query layer.

---

## 11. Open Problems

Several design questions remain genuinely open, and I think they are more interesting than the implementation described above.

**Tool-call provenance inside a running subagent.** The cache key is computed from the *input* to a subagent, not from what it reads during execution. Tracking autonomous VFS reads requires either intercepting the subagent's own tool calls during a priming run, or restricting caching to subagents that receive all content in the task description. A principled solution would look like lineage tracking in data pipeline frameworks applied to agent tool access.

**Non-deterministic model outputs.** The same inputs at `temperature > 0` produce different outputs across runs. The cache returns one realization. For summarization and classification this is usually fine; for generative tasks it can suppress a better answer from a different seed. The operator should make that trade consciously.

**Cross-model cache equivalence.** A result from `claude-sonnet-4-6` is cached separately from the same task on `gpt-5.5`. Sharing would require human-validated equivalence per (subagent type, model pair) — sometimes true for classification, rarely for nuanced generation.

**Adaptive purity classification.** `pure: bool` is currently static. A system that learns purity from trace variance — "effectively pure" when outputs are stable across identical inputs despite formal external reads — could recover cache value without relying solely on operator judgment.

**Multi-tenant cache isolation.** Sharing cached results across users is unsafe for VFS-backed code review and often fine for shared-taxonomy classification. The OKF framing (`scope: personal` vs. `scope: org`) is a useful starting point.

**Streaming and partial results.** A content-addressed cache stores *completed* results; it has nothing useful mid-stream, and cannot safely cache a partial buffer without a second keying scheme for prefixes. Instantaneous full-result replay or a synthesized fake stream covers UI parity, but interrupted long-running subagents that want to resume remain an open design problem.

---

## 12. Citation

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
