---
title: "Recall: A Working Content-Addressed Cache for Deep Agents Subagent Dispatch"
date: 2026-07-04
description: "The implementation spec for Recall, a content-addressed result cache for Deep Agents subagent dispatch: canonical cache keys, a Qdrant-backed store with named vectors, a wrap_tool_call middleware, provenance-based invalidation, and a break-even cost model."
tags: ["agents", "deepagents", "caching", "qdrant", "middleware", "cost-optimization"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Multi-agent frameworks solve context rot by isolating work into subagents — each with its own context window, its own tool access, its own billing. The isolation is the point. But it comes with a structural cost that nobody addresses by default: every subagent re-derives results from scratch, even when the underlying inputs haven't changed. This post is the implementation spec for **Recall**, a content-addressed result cache for Deep Agents dispatch, packaged as an `AgentMiddleware` backed by Qdrant. I cover the module layout, cache-key design, the Qdrant collection schema, the middleware's hook points, and the conditions under which the cache is safe, unsafe, and economically significant — precise enough to build, not just discuss.

I focus specifically on the **static-dispatch** path (the `task` tool in `SubAgentMiddleware`) and the **dynamic-dispatch** path (programmatic fan-out via `CodeInterpreterMiddleware`). I won't cover caching for the model itself — provider-side prompt caching already handles prefix deduplication at the API layer. What's missing is result-level memoization above the API, which is what this library provides.

## Table of Contents

1. [The Structural 4× Cost Multiplier](#1-the-structural-4-cost-multiplier)
2. [Why Provider-Side Prompt Caching Doesn't Close the Gap](#2-why-provider-side-prompt-caching-doesnt-close-the-gap)
3. [Content-Addressed Caching: The Core Idea](#3-content-addressed-caching-the-core-idea)
4. [Package Layout and Design Overview](#4-package-layout-and-design-overview)
5. [Installing and Using Recall](#5-installing-and-using-recall)
6. [keys.py: Cache Key Design](#6-keyspy-cache-key-design)
7. [store.py: Qdrant Collection Design](#7-storepy-qdrant-collection-design)
8. [middleware.py: The wrap_tool_call Middleware](#8-middlewarepy-the-wrap_tool_call-middleware)
9. [Worked Example: Map-Reduce Over Pages](#9-worked-example-map-reduce-over-pages)
10. [invalidation.py: Provenance-Based Invalidation](#10-invalidationpy-provenance-based-invalidation)
11. [False Hit Risk: Where Semantic Similarity Breaks](#11-false-hit-risk-where-semantic-similarity-breaks)
12. [Cost Model and Break-Even Analysis](#12-cost-model-and-break-even-analysis)
13. [Challenges and Open Problems](#13-challenges-and-open-problems)
14. [References](#14-references)

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

The current answer is model routing: use cheaper models for workers (Haiku is roughly 5× cheaper than Opus on output). This is the right lever, but it requires humans to configure a static policy at definition time, and one environment variable can silently undo it. More fundamentally, it reduces per-call cost; it doesn't eliminate redundant calls. That's the gap `recall` closes.

## 2. Why Provider-Side Prompt Caching Doesn't Close the Gap

LangChain's Deep Agents prompt caching blog post describes prefix-level caching via the Anthropic API's `cache_control` blocks. The mechanism discounts input tokens when the same prefix appears again in a subsequent request. This is meaningful for long-form system prompts that are re-sent with every subagent invocation.

But the cost structure of a typical subagent call looks like this:

$$\text{Cost} = \underbrace{C_{\text{in}} \cdot T_{\text{sys}}}_{\text{cached prefix}} + \underbrace{C_{\text{in}} \cdot T_{\text{task}}}_{\text{task description (uncached)}} + \underbrace{C_{\text{out}} \cdot T_{\text{result}}}_{\text{output tokens}}$$

where $C_{\text{in}}$ and $C_{\text{out}}$ are the input and output token prices, and $T_{\text{sys}}$, $T_{\text{task}}$, $T_{\text{result}}$ are the token counts for the system prompt, task payload, and generated result respectively. In plain English: the bill is cheap input for the system prompt, cheap input for the task text, and expensive output for the result — and provider-side caching only discounts the first term.

Provider-side caching reduces $C_{\text{in}} \cdot T_{\text{sys}}$. It does nothing for $C_{\text{out}} \cdot T_{\text{result}}$, and $C_{\text{out}}$ is typically 3–5× higher than $C_{\text{in}}$ on current frontier models. Using the worker numbers from §1: a Haiku worker with 6k input / 800 output pays roughly \$0.0048 for input and \$0.0032 for output. Even if prompt caching eliminated all input cost, you'd still pay the full \$0.0032 for output on every redundant run.

Furthermore, provider-side caching is *stateless across sessions*. Each new session rebuilds the KV cache from scratch. An invocation on Monday and the same invocation on Thursday both pay full output token cost. `recall` targets exactly this gap: a persistent, cross-session, result-level cache.

## 3. Content-Addressed Caching: The Core Idea

A **content-addressed cache** stores a function's output keyed by a deterministic hash of its inputs. If the inputs hash to the same key, the stored output is returned without re-executing. This is the same principle behind build systems like Bazel and Nix, applied here to subagent invocations.

Formally, let a subagent invocation be a function:

$$f: (\theta, \sigma, x) \rightarrow r$$

where $\theta$ is the subagent type (name, system prompt, model, tool schema version), $\sigma$ is any shared state the invocation reads, $x$ is the task-specific input payload, and $r$ is the result. If $f$ is *pure* — deterministic and free of side effects — then for identical $(\theta, \sigma, x)$ tuples, $r$ is always the same, and we can cache. In plain English: unchanged definition, state, and task payload means unchanged result — so we should not pay to regenerate it.

The challenge is that most useful subagents are not obviously pure. They may read files, search the web, or call external APIs. `recall` restricts to subagents an operator flags `pure`, and carries provenance about what those subagents read so results can be invalidated when inputs change — that's `invalidation.py`.

## 4. Package Layout and Design Overview

`recall` is four modules: pure key computation, the Qdrant-backed store, the middleware that wires into Deep Agents, and invalidation logic kept deliberately separate from the hot path.

```
recall/
  __init__.py          # public re-exports
  keys.py              # canonicalize(), compute_key(), purity classification
  store.py             # Qdrant collection setup, lookup_exact/lookup_semantic, insert
  middleware.py        # SubagentResultCacheMiddleware (wrap_tool_call hook)
  invalidation.py       # provenance verification, staleness propagation
```

```
┌────────────────────────────────────────────────────────────────┐
│  Orchestrator                                                  │
│    │ task(subagent_type, description, files=[...])             │
│    ▼                                                            │
│  SubagentResultCacheMiddleware.awrap_tool_call  (middleware.py) │
│    1. is_pure(subagent_type)?           ── no ──▶ handler(req)  │
│    2. compute_key(args)                          (keys.py)      │
│    3. lookup_exact(key)                          (store.py)     │
│    4. verify_provenance(hit)                     (invalidation) │
│    5. hit + valid  ──▶ return cached result                     │
│       miss/invalid ──▶ handler(req) ──▶ store.insert(...)       │
└────────────────────────────────────────────────────────────────┘
                    │                              ▲
                    ▼                              │
             recall collection (Qdrant, named vectors)
```

## 5. Installing and Using Recall

The package targets Python 3.11+, depends on `qdrant-client>=1.18` for the current `query_points` API, and the `deepagents` middleware protocol. Embeddings are pluggable and only required when semantic matching is opted in.

```bash
pip install recall-agents
# optional: bundled local embeddings for semantic-match mode
pip install "recall-agents[fastembed]"
```

Attaching it to a Deep Agents pipeline is a single middleware registration:

```python
from deepagents import create_deep_agent
from recall import SubagentResultCacheMiddleware

cache_middleware = SubagentResultCacheMiddleware(
    qdrant_url="http://localhost:6333",
    profiles={
        "doc-summarizer": {
            "system_prompt": DOC_SUMMARIZER_PROMPT,
            "model": "anthropic:claude-haiku-4-5",
            "tools_version": "a3f9c2d1",
            "pure": True,
        },
        "web-researcher": {
            "system_prompt": WEB_RESEARCHER_PROMPT,
            "model": "anthropic:claude-sonnet-4-6",
            "tools_version": "a3f9c2d1",
            "pure": False,  # reads live web state; never cached
        },
    },
    exact_only=True,   # start conservative; see §11 before enabling semantic mode
    vfs_root="./workspace",
)

agent = create_deep_agent(
    tools=[...],
    subagents=[...],
    middleware=[cache_middleware],
)
```

`before_agent` (called once at session start) opens the Qdrant client and ensures the collection exists; every subsequent `task` tool call for a profile marked `pure` is transparently intercepted. Non-pure profiles and non-`task` tool calls pass straight through with a single dictionary lookup of overhead. The rest of this post is the implementation behind these four modules.

## 6. keys.py: Cache Key Design

The key is the correctness surface of the whole system. A key that under-specifies inputs produces false hits; a key that over-specifies them produces unnecessary misses.

### 6.1 Components of a Canonical Key

The cache key $k$ should capture everything that could cause two invocations to produce different outputs:

$$k = \text{Hash}(\theta_{\text{type}}, \theta_{\text{prompt}}, \theta_{\text{model}}, \theta_{\text{tools\_version}}, x_{\text{canonical}})$$

- `θ_type` — subagent name or profile identifier
- `θ_prompt` — the full system prompt, since a prompt change invalidates all prior results for that subagent type
- `θ_model` — model string including version pin (e.g., `anthropic:claude-sonnet-4-6`), because the same inputs to a different model are not guaranteed to produce equivalent results
- `θ_tools_version` — a version hash of the tool schemas exposed to the subagent, since a tool change can alter reachable behaviors
- `x_canonical` — a canonicalized serialization of the task input, with deterministic key ordering and normalized whitespace

SHA-256 of the concatenated canonical form is sufficient. `θ_type || θ_prompt || θ_model || θ_tools_version` is computed once per subagent type at middleware initialization and memoized, reducing per-dispatch work to hashing `x_canonical`.

### 6.2 canonicalize() and compute_key()

Canonicalization is where most cache-key bugs hide. Two task payloads that are semantically identical but differ in key order, trailing whitespace, or path separators will hash differently and silently miss. `canonicalize` is deliberately strict: JSON with sorted keys, NFC-normalized Unicode, and path normalization for any file references embedded in the description.

```python
from __future__ import annotations
import hashlib, json, unicodedata
from pathlib import PurePosixPath
from typing import Any


def canonicalize(obj: object) -> str:
    """Deterministic serialization for cache-key material."""
    if isinstance(obj, dict):
        items = {k: canonicalize(v) for k, v in sorted(obj.items())}
        return json.dumps(items, separators=(",", ":"), ensure_ascii=False)
    if isinstance(obj, (list, tuple)):
        return json.dumps([canonicalize(v) for v in obj], separators=(",", ":"), ensure_ascii=False)
    if isinstance(obj, str):
        return " ".join(unicodedata.normalize("NFC", obj).split())
    if isinstance(obj, (int, float, bool)) or obj is None:
        return json.dumps(obj)
    return canonicalize(str(obj))


def normalize_path(path: str) -> str:
    return str(PurePosixPath(path))


def profile_prefix(subagent_type: str, profile: dict[str, Any]) -> str:
    """SHA-256 over the static (θ_type, θ_prompt, θ_model, θ_tools_version) tuple."""
    material = canonicalize({
        "type": subagent_type,
        "prompt": profile["system_prompt"],
        "model": profile["model"],
        "tools": profile["tools_version"],
    })
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def compute_key(
    subagent_type: str,
    profile: dict[str, Any],
    args: dict[str, Any],
    content_hash_fn,
) -> str:
    """Fold file content hashes into x_canonical so path-only identity cannot false-hit after an edit."""
    source_refs = extract_source_refs(args)
    content_hashes = {ref: content_hash_fn(ref) for ref in source_refs}
    x_canonical = canonicalize({
        "description": args.get("description", ""),
        "files": {normalize_path(p): h for p, h in sorted(content_hashes.items())},
    })
    material = f"{profile_prefix(subagent_type, profile)}:{x_canonical}"
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def extract_source_refs(args: dict[str, Any]) -> list[str]:
    """Prefer explicit `files`/`paths`; else scan backtick-quoted file-like tokens."""
    import re
    refs: list[str] = []
    for p in args.get("files") or args.get("paths") or []:
        refs.append(normalize_path(str(p)))
    for match in re.findall(r"`([^`]+)`", args.get("description", "")):
        if "/" in match or match.endswith((".ts", ".py", ".md", ".tsx", ".js")):
            refs.append(normalize_path(match))
    seen, ordered = set(), []
    for r in refs:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return ordered
```

Two design choices deserve comment. First, content hashes are folded into the key rather than checked only at invalidation time: a changed file produces a different key, so the old entry is simply never looked up. Second, the profile prefix is memoized in `middleware.py` — recomputing SHA-256 over a multi-kilobyte system prompt on every dispatch is wasteful when the profile is static for the session. `extract_source_refs` is intentionally conservative: it only claims paths the orchestrator named or that appear as backtick-quoted file-like tokens. Paths discovered only inside the subagent's own tool loop are invisible here — that is the provenance open problem in §13.

### 6.3 Exact vs. Semantic Matching

Exact hash matching handles the case where inputs are literally identical across invocations — the same file path, the same content, the same task description. This covers the map-reduce pattern well: `pages.map(page => task(...))` generates structurally identical invocations for each page element, and across runs pages already reviewed are exact matches.

*Semantic* matching is a different and riskier proposition. Two task descriptions may be linguistically similar but semantically distinct: "summarize security issues in `auth.ts`" vs. "summarize security issues in `auth.ts` focusing on session tokens" are close in embedding space but should not share a cached result. This risk is not hypothetical — a recent paper, **Conversational Query Engine for Mixed-Modality Heterogeneous Enterprise Data Sources** ([Anonymous 2026](https://arxiv.org/pdf/2606.28370)), demonstrates that queries exceeding 0.95 cosine similarity under `text-embedding-3-large` can produce silent factual errors when cached hits are returned across them.

My recommendation is **exact-first, semantic-never by default** — `exact_only=True` on `SubagentResultCacheMiddleware`. The cache checks exact hash first; on a miss, the invocation runs. Semantic lookup is an opt-in, per-subagent-type flag, restricted to subagents where the operator can reason that near-duplicate inputs reliably produce equivalent outputs.

### 6.4 Purity Classification

Not every subagent is cache-eligible. A sensible default classification:

| Subagent behavior | Cache-eligible? | Reason |
|---|---|---|
| Pure summarization (no tool calls) | Yes | Output is a deterministic function of the input text |
| Classification (closed label set) | Yes | Stable for same input, no external reads |
| File review (reads local VFS) | Yes, with provenance | Safe if input hash includes file content hash, not just path |
| Web search subagent | No | External state changes unpredictably |
| Code execution subagent | No | Has side effects by construction |
| Multi-step research subagent | No | Accumulates external reads; result depends on retrieval order |

`profiles[subagent_type]["pure"]` is the switch. Subagents not flagged `pure` are bypassed transparently — `middleware.py` adds no overhead to their dispatch path beyond a dictionary lookup.

## 7. store.py: Qdrant Collection Design

A content-addressed cache needs two retrieval modes: exact key lookup (the common path) and optional semantic nearest-neighbor search (the opt-in path). Qdrant supports both in one collection via payload indexes and named vectors, which is why I prefer it over a Redis-only design for this workload.

### 7.1 Filterable HNSW and Payload-Indexed Namespacing

Qdrant's ANN index is HNSW with an important extension: it augments the graph with additional edges derived from indexed payload fields, guaranteeing that the graph remains connected and traversable under filtered queries. This means a payload filter like `subagent_type = "summarizer" AND model = "anthropic:claude-sonnet-4-6"` is applied *during* graph traversal rather than as a post-filter over the full result set. Without payload-aware filtering, a vector lookup for a `"security-reviewer"` result could return a high-similarity result from a `"summarizer"` subagent.

Each cached entry stores this payload alongside its vectors:

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

Payload indexes on `subagent_type`, `model`, `tools_version`, `exact_hash`, and `stale` enable exact lookup by index scan rather than ANN, namespace isolation across models and tool versions, and bulk invalidation via a single payload-filtered delete.

### 7.2 Named Vectors for Multi-Signal Lookup

When semantic matching is opted in, Qdrant's named vectors store both a task-description embedding and a result-summary embedding on the same point. At lookup time the query runs against `task_embed`; the `result_embed` vector supports an optional post-hoc double-gate: if `cosine(query_result_embed, cached_result_embed) < threshold`, reject the hit even if the task similarity passed. This is the approach recommended by the enterprise BI caching paper cited in §11, and it materially reduces false-hit rate for borderline cases.

### 7.3 Collection Bootstrap

```python
from __future__ import annotations
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PayloadSchemaType, PointStruct,
    Filter, FieldCondition, MatchValue,
)

EMBED_DIM = 384  # FastEmbed / all-MiniLM-L6-v2; use 1536 for text-embedding-3-small


async def ensure_collection(client: AsyncQdrantClient, collection: str = "recall") -> None:
    existing = {c.name for c in (await client.get_collections()).collections}
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
            collection_name=collection, field_name=field, field_schema=schema,
        )
```

Two vectors are declared even when `exact_only=True`. Exact lookups ignore them (`with_vectors=False`), but declaring both up front avoids a migration when semantic matching is later enabled. Payload indexes on `exact_hash` and `stale` are mandatory; without them every exact lookup degrades to a full payload scan.

### 7.4 Lookup and Insert — query_points, Not search

The **exact lookup** path queries Qdrant as a key-value store via `scroll` on the payload index, skipping ANN entirely:

```python
class SubagentCacheStore:
    def __init__(self, client: AsyncQdrantClient, collection: str = "recall"):
        self.client, self.collection = client, collection

    async def lookup_exact(self, exact_hash: str, subagent_type: str) -> dict | None:
        hits, _ = await self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="exact_hash", match=MatchValue(value=exact_hash)),
                FieldCondition(key="subagent_type", match=MatchValue(value=subagent_type)),
                FieldCondition(key="stale", match=MatchValue(value=False)),
            ]),
            limit=1, with_payload=True, with_vectors=False,
        )
        if hits:
            return {"point_id": hits[0].id, "payload": hits[0].payload, "result": hits[0].payload["result"]}
        return None

    async def lookup_semantic(
        self, task_embed: list[float], subagent_type: str, *, threshold: float,
    ) -> str | None:
        """Opt-in ANN path — uses query_points against the named `task_embed` vector."""
        result = await self.client.query_points(
            collection_name=self.collection,
            query=task_embed,
            using="task_embed",
            query_filter=Filter(must=[
                FieldCondition(key="subagent_type", match=MatchValue(value=subagent_type)),
                FieldCondition(key="stale", match=MatchValue(value=False)),
            ]),
            limit=1, score_threshold=threshold, with_payload=True,
        )
        return result.points[0].payload["result"] if result.points else None

    async def insert(
        self, *, point_id: str, exact_hash: str, subagent_type: str, model: str,
        tools_version: str, source_refs: list[str], content_hashes: dict[str, str],
        task_embed: list[float], result_embed: list[float], result: str, created_at: str,
    ) -> None:
        await self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=point_id,
                vector={"task_embed": task_embed, "result_embed": result_embed},
                payload={
                    "exact_hash": exact_hash, "subagent_type": subagent_type, "model": model,
                    "tools_version": tools_version, "source_refs": source_refs,
                    "content_hashes": content_hashes, "created_at": created_at,
                    "stale": False, "result": result,
                },
            )],
        )
```

`lookup_semantic` is the one place the original design used the deprecated `qdrant_client.search()` method with a `("task_embed", vector)` tuple argument. The current API — `query_points(collection_name=..., query=[...], using="task_embed", query_filter=Filter(...), limit=...)`, reading `.points` off the return value — replaces it everywhere in `store.py`. `qdrant-client>=1.18` is required for this signature.

## 8. middleware.py: The wrap_tool_call Middleware

This section assembles `keys.py` and `store.py` into a working `AgentMiddleware`. The interception point is deliberately narrow: only `task` tool calls for profiles marked `pure` are eligible; everything else passes through untouched.

### 8.1 Hook Points

The `AgentMiddleware` protocol in Deep Agents exposes `wrap_tool_call(request, handler)` — intercepts any tool execution before it reaches the underlying implementation — and `before_agent(state, runtime)` — runs once at session start, suitable for opening the Qdrant client connection.

The `task` tool, surfaced by `SubAgentMiddleware`, is just a tool call from the orchestrator's perspective. This means `wrap_tool_call` provides a clean interception point for both the static dispatch path (turn-by-turn `task` calls) and the dynamic dispatch path (REPL-generated `task` calls inside the QuickJS interpreter).

```python
from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4
from deepagents.middleware import AgentMiddleware
from qdrant_client import AsyncQdrantClient
from recall.keys import compute_key, extract_source_refs, canonicalize
from recall.store import SubagentCacheStore, ensure_collection
from recall.invalidation import verify_provenance


class SubagentResultCacheMiddleware(AgentMiddleware):
    def __init__(
        self,
        qdrant_url: str,
        profiles: dict[str, dict],
        collection: str = "recall",
        exact_only: bool = True,
        similarity_threshold: float = 0.97,
        vfs_root: str | None = None,
    ) -> None:
        self._url = qdrant_url
        self._profiles = profiles  # {type: {system_prompt, model, tools_version, pure}}
        self._collection_name = collection
        self._exact_only = exact_only
        self._threshold = similarity_threshold
        self._vfs_root = vfs_root
        self._store: SubagentCacheStore | None = None

    async def before_agent(self, state, runtime) -> None:
        client = AsyncQdrantClient(url=self._url)
        await ensure_collection(client, self._collection_name)
        self._store = SubagentCacheStore(client, self._collection_name)

    def _is_pure(self, subagent_type: str) -> bool:
        profile = self._profiles.get(subagent_type)
        return bool(profile and profile.get("pure", False))

    def _content_hash(self, path: str) -> str:
        from pathlib import Path
        import hashlib
        root = Path(self._vfs_root) if self._vfs_root else Path(".")
        target = root / path.lstrip("/")
        if not target.is_file():
            return "sha256:missing"
        return "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()

    async def awrap_tool_call(self, request, handler):
        if request.tool_name != "task":
            return await handler(request)
        args = request.tool_input
        subagent_type = args.get("subagent_type", "default")
        if not self._is_pure(subagent_type):
            return await handler(request)

        profile = self._profiles[subagent_type]
        key = compute_key(subagent_type, profile, args, self._content_hash)
        cached = await self._store.lookup_exact(key, subagent_type)
        if cached is not None and await verify_provenance(self._store, self._content_hash, cached):
            return cached["result"]

        if not self._exact_only:
            embed_vec = await self._embed(args.get("description", ""))
            semantic = await self._store.lookup_semantic(embed_vec, subagent_type, threshold=self._threshold)
            if semantic is not None:
                return semantic

        result = await handler(request)
        source_refs = extract_source_refs(args)
        await self._store.insert(
            point_id=uuid4().hex, exact_hash=key, subagent_type=subagent_type,
            model=profile.get("model", "default"), tools_version=profile.get("tools_version", "unset"),
            source_refs=source_refs, content_hashes={r: self._content_hash(r) for r in source_refs},
            task_embed=await self._embed(args.get("description", "")),
            result_embed=await self._embed(result[:512]),
            result=result, created_at=datetime.now(timezone.utc).isoformat(),
        )
        return result

    async def _embed(self, text: str) -> list[float]:
        raise NotImplementedError("wire an embedder, e.g. fastembed.TextEmbedding, in a subclass")
```

`_embed` is left abstract in the reference implementation on purpose — bind it to whatever embedding backend the deployment already uses, matching `EMBED_DIM` in `store.py`.

### 8.2 Dynamic-Dispatch Fan-out (REPL Path)

Dynamic subagents via `CodeInterpreterMiddleware` dispatch `task` calls from inside the QuickJS interpreter's event loop. From the middleware's perspective, these arrive through the same `wrap_tool_call` hook — the REPL bridges to native Python `task` handler functions registered with the interpreter. No additional integration is needed; the cache intercepts at the Python boundary before the call crosses into the subagent harness.

This is the most economically significant path: a programmatic fan-out over N items generates N structurally identical tool call invocations, and the cache hit rate across a `pages.map(...)` with unchanged pages approaches 100% on repeated runs.

## 9. Worked Example: Map-Reduce Over Pages

To make the economics tangible, consider a documentation-review agent that fans out over a fixed corpus of 40 markdown pages every weekday morning:

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

Without the cache, the week costs \$1.60 in worker API spend (5 × \$0.320). With `compute_key`'s content-hashed file keys, it costs \$0.40 — a 75% reduction. Embedding and Qdrant overhead for 200 lookups is under \$0.01 for the week (see §12).

Two observations follow. First, the cold day dominates weekly cost; any workflow that re-runs daily over a mostly-stable corpus is an excellent fit. Second, hit rate is bounded by the change rate of the corpus, not by embedding quality — exact matching with content hashes makes that relationship linear and predictable. Semantic matching would not improve Day 2–5 numbers here, because the task strings for unchanged pages are already exact matches.

## 10. invalidation.py: Provenance-Based Invalidation

Even with content hashes in the key, operators need an explicit invalidation path for prompt edits, tool-schema bumps, and the rare race where a file changes between key computation and result return.

### 10.1 Provenance-Based vs. TTL-Based

TTL-based invalidation (expire entries after $\Delta t$) is the simplest implementation but is wrong for this use case. A summarization of an unchanged file is just as valid at $t + 30\text{d}$ as at $t$; a summarization of a file that changed five minutes ago is stale immediately regardless of TTL.

The correct invalidation model is **provenance-based**: an entry is valid as long as all of its `source_refs` point to content that hasn't changed:

$$\text{valid}(e) \iff \forall r \in e.\texttt{source\_refs}: \text{hash}(\text{read}(r)) = e.\texttt{content\_hash}[r]$$

In plain English: an entry stays valid only while every file it claimed to depend on still hashes to the value recorded at write time. Folding content hashes into the key (§6.2) makes most staleness detectable by key mismatch; `verify_provenance` below is a second line of defense for entries stored under an older keying scheme.

For subagents that read from the VFS through tool calls rather than receiving content in the task description, provenance is harder — the middleware would need to intercept the subagent's own tool calls during a priming run. The safest default is `pure=False` unless the operator takes on that responsibility.

### 10.2 Verification and Staleness Propagation

```python
from __future__ import annotations
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


async def verify_provenance(store, content_hash_fn, cached: dict) -> bool:
    """Re-hash every recorded source_ref; soft-invalidate on mismatch."""
    content_hashes: dict = cached["payload"].get("content_hashes") or {}
    if not content_hashes:
        return True  # pure-without-files (e.g. classification)
    for ref, expected in content_hashes.items():
        if content_hash_fn(ref) != expected:
            await store.client.set_payload(
                collection_name=store.collection,
                payload={"stale": True},
                points=[cached["point_id"]],
            )
            return False
    return True


async def invalidate_profile(
    client: AsyncQdrantClient, collection: str, subagent_type: str, old_tools_version: str,
) -> None:
    """Bulk-invalidate on a prompt or tool-schema change: O(matching entries), not an ANN search."""
    await client.delete(
        collection_name=collection,
        points_selector=Filter(must=[
            FieldCondition(key="subagent_type", match=MatchValue(value=subagent_type)),
            FieldCondition(key="tools_version", match=MatchValue(value=old_tools_version)),
        ]),
    )
```

Marking `stale: True` rather than deleting preserves the entry for analytics while ensuring subsequent lookups exclude it via the `stale=False` filter. `invalidate_profile` is a payload-index scan that doesn't touch the vector index — the same pattern applies to model swaps: filter on `model` and either delete or soft-mark `stale`.

## 11. False Hit Risk: Where Semantic Similarity Breaks

I want to be direct about the principal failure mode: **a false cache hit is strictly worse than a cache miss**. A miss wastes cost (the subagent runs redundantly). A false hit injects a confidently-wrong result into the orchestrator's context, which the orchestrator will act on without visibility into the substitution.

The risk is highest under two conditions. **Exact-hash collision** — SHA-256 collisions are theoretically possible but computationally negligible in practice; not a real concern at any fleet scale below cryptographic adversarial pressure. **Semantic match with non-equivalent semantics** (opt-in mode only) — this is the real risk. "Summarize the auth module for security issues" and "Summarize the auth module for performance issues" may embed close to each other. The mitigation is a conservative similarity threshold ($\geq 0.97$, not $0.85$), per-subagent-type opt-in rather than global, and the result-embedding double-gate from §7.2.

### Threshold Sensitivity: A Thought Experiment

Suppose we label 200 pairs of task descriptions from a code-review fleet as *equivalent* or *distinct*, embed with a 384-dim MiniLM model, and sweep the cosine threshold used by `lookup_semantic`:

| Threshold $\tau$ | Approx. precision | Approx. recall | Operational reading |
|---|---|---|---|
| 0.85 | 0.72 | 0.94 | Too permissive — ~28% of hits are wrong |
| 0.90 | 0.84 | 0.81 | Still unsafe for synthesizer context |
| 0.95 | 0.93 | 0.55 | Borderline; needs result-embed double-gate |
| 0.97 | 0.97 | 0.38 | Acceptable for closed-label classification |
| 0.99 | 0.99 | 0.12 | Nearly exact-match; rarely worth embedding cost |

Lowering the threshold buys recall at the expense of silently wrong answers. For subagent result caching, I treat precision below ~0.97 as unacceptable — which is exactly where semantic matching starts to look like a poor substitute for exact hashing, and why `SubagentResultCacheMiddleware` defaults to `exact_only=True`.

A recommended production posture: run `exact_only=True` for 30 days, instrument miss rate, then evaluate semantic matching per subagent type from empirical hit-rate data. The same window gives baseline cost data for break-even.

## 12. Cost Model and Break-Even Analysis

Let $N$ be the number of subagent dispatches per day across the fleet, $\rho$ the cache hit rate, $C_{\text{agent}}$ the average cost per subagent invocation, $C_{\text{embed}}$ the cost to embed the task description per dispatch, and $C_{\text{qdrant}}$ the amortized Qdrant storage and query cost per dispatch.

The daily net saving is $\Delta = N \cdot [\rho \cdot C_{\text{agent}} - (C_{\text{embed}} + C_{\text{qdrant}})]$, which expands from:

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
| 75% (docs corpus from §9) | \$7.50 | \$0.025 | \$7.475 |
| 90% | \$9.00 | \$0.025 | \$8.975 |

Annualized at 50% hit rate: roughly \$1,800/year saved on a thousand-dispatch fleet, against infrastructure measured in tens of dollars. Map-reduce over unchanged inputs should approach 50–95% hit rates on repeated runs. The point isn't specific numbers — fleet data will vary — but that the marginal cost of embedding is so low relative to subagent invocation that the break-even bar is trivially low, unlike semantic caching at the end-user query layer.

## 13. Challenges and Open Problems

**Tool-call provenance inside a running subagent.** `compute_key` derives from the *input* to a subagent, not from what it reads during execution. Tracking autonomous VFS reads requires either intercepting the subagent's own tool calls during a priming run, or restricting caching to subagents that receive all content in the task description. A principled solution would look like lineage tracking in data pipeline frameworks applied to agent tool access.

**Non-deterministic model outputs.** The same inputs at `temperature > 0` produce different outputs across runs. The cache returns one realization. For summarization and classification this is usually fine; for generative tasks it can suppress a better answer from a different seed. The operator should make that trade consciously via the `pure` flag.

**Cross-model cache equivalence.** A result from `claude-sonnet-4-6` is cached separately from the same task on `gpt-5.5`, because `θ_model` is part of the key. Sharing would require human-validated equivalence per (subagent type, model pair) — sometimes true for classification, rarely for nuanced generation.

**Adaptive purity classification.** `pure: bool` is currently static per profile. A system that learns purity from trace variance — "effectively pure" when outputs are stable across identical inputs despite formal external reads — could recover cache value without relying solely on operator judgment.

**Multi-tenant cache isolation.** Sharing cached results across users is unsafe for VFS-backed code review and often fine for shared-taxonomy classification. A `scope: personal` vs. `scope: org` payload field, filtered the same way `subagent_type` is filtered today, is a useful starting point not yet in the reference `store.py`.

**Streaming and partial results.** A content-addressed cache stores *completed* results; it has nothing useful mid-stream, and cannot safely cache a partial buffer without a second keying scheme for prefixes. Instantaneous full-result replay or a synthesized fake stream covers UI parity, but interrupted long-running subagents that want to resume remain an open design problem outside this library's scope.

## 14. References

```bibtex
@misc{recall-2026,
  title   = {recall: A Working Content-Addressed Cache for Deep Agents Subagent Dispatch},
  author  = {Inamdar, Mihir},
  year    = {2026},
  note    = {Blog post, July 2026}
}
```

- LangChain Team (2026). *Introducing Dynamic Subagents in Deep Agents*. [langchain.com/blog](https://www.langchain.com/blog/introducing-dynamic-subagents-in-deep-agents)
- LangChain Team (2026). *How to Use RLMs in Deep Agents*. [langchain.com/blog](https://www.langchain.com/blog/how-to-use-rlms-in-deep-agents)
- LangChain Team (2026). *Running Untrusted Agent Code Without a Sandbox*. [langchain.com/blog](https://www.langchain.com/blog/running-untrusted-agent-code-without-a-sandbox)
- LangChain Team (2026). *Prompt Caching with Deep Agents*. [langchain.com/blog](https://www.langchain.com/blog/deep-agents-prompt-caching)
- Zhang, A. et al. (2025). *Recursive Language Models*. [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
- Anonymous (2026). *Conversational Query Engine for Mixed-Modality Heterogeneous Enterprise Data Sources*. [arXiv:2606.28370](https://arxiv.org/pdf/2606.28370) — §6 on verified semantic cache false-hit risk.
- Qdrant Team (2025). *Combining Vector Search and Filtering*. [qdrant.tech/documentation](https://qdrant.tech/course/essentials/day-2/filterable-hnsw/)
- Qdrant Team (2025). *Collections: Named Vectors and Quantization*. [qdrant.tech/documentation](https://qdrant.tech/documentation/manage-data/collections/)
- Qdrant Team (2025). *Query API — `query_points`*. [qdrant.tech/documentation](https://qdrant.tech/documentation/concepts/search/)
