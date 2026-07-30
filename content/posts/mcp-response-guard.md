---
title: "mcp-response-guard: A Schema-Aware Validator for Incomplete Tool Responses"
date: 2026-07-24
description: "The spec for mcp-response-guard, an installable interceptor library that catches silent, schema-valid-but-incomplete MCP tool responses using expectation schemas, calibrated anomaly scoring, and a Qdrant-backed memory of what normal looks like."
tags: ["agents", "mcp", "qdrant", "validation", "reliability", "tool-calling"]
author: "Mihir Inamdar"
showToc: true
math: true
---

An agent calling a tool through the **Model Context Protocol (MCP)** ([Anthropic, 2024](https://modelcontextprotocol.io/)) treats a `200`-equivalent JSON-RPC result as ground truth. It has no innate sense of whether the payload it just received is *complete*, *representative*, or merely *well-formed*. This post is the implementation spec for **mcp-response-guard**, a small library that closes that gap: an intercepting MCP client, a schema layer describing what a "normal" response looks like, an anomaly scorer, and a Qdrant collection that remembers response history per tool-plus-argument cluster. I cover module boundaries, function signatures, the Qdrant schema, and how to wire it into an existing agent — precise enough to hand to an engineer or a coding agent and get a working package back. I will not cover prompt injection, tool-description poisoning, or adversarial tool misuse — those are distinct failure modes with their own literature (notably OWASP ASI01–ASI06). This post only addresses failures where the tool is behaving as designed but the response is quietly insufficient for the decision the agent is about to make.

## Table of Contents

1. [The Problem: Success at the Transport Layer, Failure at the Semantic Layer](#the-problem-success-at-the-transport-layer-failure-at-the-semantic-layer)
2. [Why This Is Different from Known Failure Modes](#why-this-is-different-from-known-failure-modes)
3. [A Taxonomy of Silent Failures](#a-taxonomy-of-silent-failures)
4. [Package Layout and Design Overview](#package-layout-and-design-overview)
5. [Installing and Using mcp-response-guard](#installing-and-using-mcp-response-guard)
6. [schema.py: Expectation Schemas](#schemapy-expectation-schemas)
7. [client.py: The Validating Interceptor](#clientpy-the-validating-interceptor)
8. [scoring.py: Anomaly Scoring](#scoringpy-anomaly-scoring)
9. [history.py: Qdrant as the Memory of Normal](#historypy-qdrant-as-the-memory-of-normal)
10. [Bootstrap and Cold Start](#bootstrap-and-cold-start)
11. [policy.py: Remediation at the Agent Layer](#policypy-remediation-at-the-agent-layer)
12. [Worked Example and Evaluation](#worked-example-and-evaluation)
13. [Challenges and Open Problems](#challenges-and-open-problems)
14. [References](#references)

## The Problem: Success at the Transport Layer, Failure at the Semantic Layer

MCP standardizes how a client discovers and invokes tools, but it says nothing about whether a tool's response is *sufficient*. The protocol's job ends at "here is a well-formed JSON-RPC result." Whether that result actually represents the queried state of the world is left entirely to the tool implementation, and by extension, entirely unchecked by the agent.

Consider a billing agent that calls a `list_accounts` tool to check whether a customer has any outstanding invoices before approving a refund:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "content": [
      {"type": "text", "text": "[{\"id\": 1, \"name\": \"Acme Corp\", \"balance\": 0}]"}
    ]
  }
}
```

Nothing here is malformed. The call returns in 40ms, the schema is valid JSON, the agent parses it, and reasons: one account, zero balance, refund approved. What the agent cannot see is that the underlying query was supposed to return 51 accounts and a pagination cursor, and a transient database timeout silently truncated the result to the first page before the cursor logic executed. There is no error field, no non-2xx status, no protocol-level signal of any kind. The agent's decision is built on a response that is technically valid and substantively wrong.

The same pattern shows up quietly elsewhere: `get_invoice_summary` returning zeros from a lagging replica, or `search_transactions` returning three rows after a rate limiter substituted a thin page with HTTP 200. In each case the agent trusts the tool and acts.

This is the general shape: **a response can be schema-valid and still be semantically incomplete**, and MCP has no native mechanism to distinguish the two. Transport success is not semantic sufficiency. `mcp-response-guard` exists to make that distinction mechanically, at the client boundary, without requiring every tool author to redesign their API.

## Why This Is Different from Known Failure Modes

It is worth being precise about scope, because agentic security research in 2026 has produced a rich taxonomy of adjacent problems, and silent incompleteness is not the same as any of them.

**Tool misuse** (ASI02 in the [OWASP Top 10 for Agentic Applications](https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/)) describes an agent bending a legitimate tool toward a destructive action. **Memory and context poisoning** (ASI06) describes an adversary shaping what an agent retrieves from persistent storage. Both are adversarial. Related work on tool-description poisoning and prompt injection (ASI01) likewise assumes an attacker.

The failure mode here has no adversary. The tool is not compromised; the description is intact. A load balancer drops connections, a replica lags, a rate limiter returns a valid-but-empty page, a cache serves stale data during a deploy. A human calling the same API notices an empty list where fifty rows were expected. An agent, absent instrumentation, has no such prior — a well-typed empty array looks as valid as a full one.

In classical reliability terms this is closer to **fail-silent** behavior ([Cristian, 1991](https://ieeexplore.ieee.org/document/64997)) than Byzantine faults: the component returns a plausible answer rather than crashing. Fail-silent systems are manageable when silence is detectable (timeouts); they are harder when silence is *content-shaped*. That is the default MCP-agent regime today, and it's the regime `mcp-response-guard` is built for.

## A Taxonomy of Silent Failures

Not all incomplete responses look alike, and a validator that only checks for empty results will miss most of them. I separate silent failures into four categories, ordered from easiest to hardest to catch automatically, and each category maps to a specific check inside `scoring.py`.

**Cardinality failures.** The response has the right shape but the wrong count — one account instead of fifty-one. These are the most tractable: most tools have a queryable expected range via prior invocations, a `total_count` field, or a companion count tool. MCP example — `crm.list_contacts` returns `{"contacts": [{"id": "c1"}], "total_count": 47}`: list length 1 contradicts `total_count` 47.

**Truncation failures.** The response is a prefix of the correct answer — page one with the cursor silently dropped, or the first 4KB of a document that should have streamed in full. MCP example — `docs.fetch_page` after a gateway timeout: `{"document_id": "policy-2024", "byte_length": 4096, "next_cursor": null}`. Historically this document returns ~48KB with a non-null cursor until the final page.

**Staleness failures.** The response is complete and well-formed but reflects state from before a change the agent needed to see. MCP example — `billing.get_balance` from a CDN edge: `{"customer_id": "4471", "balance": 0.0, "as_of": "2026-07-24T08:12:00Z", "source": "cache"}`. If the agent decides a refund at 16:00 UTC and a \$4,200 payment posted at 14:30, this is schema-perfect and decision-wrong.

**Type-conformant nonsense.** Values are individually plausible and jointly absurd — shipping date before order date, percentage above 100. MCP example — `orders.get_status`: `{"ordered_at": "2026-07-20T10:00:00Z", "shipped_at": "2026-07-18T09:00:00Z", "fulfillment_pct": 120}`. Statistical baselines rarely catch this class — the constraint is logical, not distributional.

The common thread is that all four are invisible to anything that validates only *shape* rather than *expectation*, which is exactly the layer MCP's JSON-RPC contract stops at.

## Package Layout and Design Overview

`mcp-response-guard` is a thin interceptor between the MCP client and the agent's reasoning loop, plus the state it needs to score responses against history. The public package is four modules and one optional extension:

```
mcp_response_guard/
  __init__.py       # public re-exports
  schema.py         # ExpectationSchema, SchemaStore, cardinality/pagination/freshness specs
  client.py         # ValidatingMCPClient (wraps any MCP client), ToolResult, Verdict
  scoring.py        # score_response(), ResponseScorer, per-axis check functions
  history.py        # HistoryStore (Qdrant-backed baseline memory)
  policy.py         # optional: RemediationPolicy implementations for the agent layer
```

```
┌─────────────────────────────────────────────────────────────────┐
│  Agent Reasoning Loop                                           │
│    │                                                            │
│    ▼ call_tool(name, arguments)                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  ValidatingMCPClient        (client.py)                   │  │
│  │    1. forward to inner MCP client                         │  │
│  │    2. schemas.get(name)                  (schema.py)      │  │
│  │    3. score_response()  ←── Qdrant baselines (scoring.py) │  │
│  │    4. annotate | pass-through                              │  │
│  │    5. async history.record() ──▶ Qdrant   (history.py)    │  │
│  └───────────────────────────────────────────────────────────┘  │
│    │                                                            │
│    ▼ ToolResult (+ optional anomaly annotation)                 │
│  Agent policy decides: retry / escalate / proceed  (policy.py)  │
└─────────────────────────────────────────────────────────────────┘
          │                              ▲
          ▼                              │
   MCP transport / tool server     tool_invocations collection
                                   (named vectors + payload indexes)
```

`ValidatingMCPClient` is the only new code path an existing agent needs to adopt — everything else lives behind it. Recording is asynchronous so monitoring never blocks the hot path.

## Installing and Using mcp-response-guard

The package targets Python 3.11+ and depends on `qdrant-client>=1.18` for the current `query_points` API, plus a JSON-path-lite utility with no other hard dependencies. Embeddings are pluggable — bring `fastembed`, `sentence-transformers`, or an API-backed embedder.

```bash
pip install mcp-response-guard
# optional: bundled local embeddings for the Qdrant history store
pip install "mcp-response-guard[fastembed]"
```

Minimal wiring around an existing MCP client:

```python
from qdrant_client import QdrantClient
from mcp_response_guard import (
    ValidatingMCPClient, SchemaStore, HistoryStore, ResponseScorer,
)
from mcp_response_guard.schema import (
    ExpectationSchema, CardinalitySpec, PaginationSpec,
)

qdrant = QdrantClient(url="http://localhost:6333")
embed = load_embedder()  # e.g. fastembed.TextEmbedding("BAAI/bge-small-en-v1.5")

schemas = SchemaStore()
schemas.register(ExpectationSchema(
    tool_name="list_accounts",
    cardinality=CardinalitySpec(json_path="$.accounts", min_samples=30),
    pagination=PaginationSpec(cursor_path="$.next_cursor"),
))

history = HistoryStore(qdrant, embed_fn=embed)
scorer = ResponseScorer(history)

guarded = ValidatingMCPClient(raw_mcp_client, schemas, history, scorer)

result = await guarded.call_tool("list_accounts", {"customer_id": "4471"})
if result.annotations:
    action = await policy.decide("list_accounts", {"customer_id": "4471"}, result, verdict)
```

`raw_mcp_client` is anything satisfying the `MCPClient` protocol in `client.py` — the reference MCP Python SDK client works unmodified. Schemas can also be loaded in bulk from tool descriptors carrying an `x-expectation` vendor extension via `SchemaStore.from_mcp_descriptors(mcp_tools)`, covered in the next section. Everything past this point is the internals behind that four-line wiring block.

## schema.py: Expectation Schemas

An `ExpectationSchema` extends a tool's ordinary JSON schema with annotations MCP does not require: cardinality range, pagination cues, freshness bound, and cross-field constraints — a second layer of *distributional and logical* expectations on top of structural parseability.

### Dataclasses and JSON Schema Extension

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

ConstraintFn = Callable[[dict[str, Any]], bool]

CROSS_FIELD_REGISTRY: dict[str, ConstraintFn] = {}

def register_cross_field_check(name: str) -> Callable[[ConstraintFn], ConstraintFn]:
    """Decorator binding a runtime constraint to a serializable name."""
    def decorator(fn: ConstraintFn) -> ConstraintFn:
        CROSS_FIELD_REGISTRY[name] = fn
        return fn
    return decorator

@dataclass
class CardinalitySpec:
    json_path: str                          # e.g. "$.accounts"
    range: tuple[int, int] | None = None    # (lo, hi) = (p5, p95) once learned
    min_samples: int = 30
    source: Literal["learned", "seeded", "total_count"] = "learned"

@dataclass
class PaginationSpec:
    cursor_path: str                        # e.g. "$.next_cursor"
    typically_paginates_threshold: float = 0.7

@dataclass
class FreshnessSpec:
    timestamp_path: str                     # e.g. "$.as_of"
    max_staleness_seconds: int

@dataclass
class ExpectationSchema:
    tool_name: str
    output_json_schema: dict[str, Any] | None = None
    cardinality: CardinalitySpec | None = None
    pagination: PaginationSpec | None = None
    freshness: FreshnessSpec | None = None
    cross_field_checks: list[tuple[str, ConstraintFn]] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=lambda: {
        "cardinality": 1.0, "truncation": 1.0,
        "staleness": 1.0, "cross_field": 1.0,
    })

    def to_json_schema_extension(self) -> dict[str, Any]:
        """Serialize as a JSON Schema `x-expectation` vendor extension."""
        ext: dict[str, Any] = {"tool_name": self.tool_name}
        if self.cardinality:
            c = self.cardinality
            ext["cardinality"] = {
                "json_path": c.json_path,
                "range": list(c.range) if c.range else None,
                "min_samples": c.min_samples, "source": c.source,
            }
        if self.pagination:
            p = self.pagination
            ext["pagination"] = {
                "cursor_path": p.cursor_path,
                "typically_paginates_threshold": p.typically_paginates_threshold,
            }
        if self.freshness:
            f = self.freshness
            ext["freshness"] = {
                "timestamp_path": f.timestamp_path,
                "max_staleness_seconds": f.max_staleness_seconds,
            }
        ext["cross_field_check_names"] = [n for n, _ in self.cross_field_checks]
        return ext

def _schema_from_extension(tool_name: str, ext: dict[str, Any]) -> ExpectationSchema:
    """Inverse of `to_json_schema_extension`; binds check names via CROSS_FIELD_REGISTRY."""
    return ExpectationSchema(
        tool_name=tool_name,
        cardinality=CardinalitySpec(**ext["cardinality"]) if ext.get("cardinality") else None,
        pagination=PaginationSpec(**ext["pagination"]) if ext.get("pagination") else None,
        freshness=FreshnessSpec(**ext["freshness"]) if ext.get("freshness") else None,
        cross_field_checks=[
            (n, CROSS_FIELD_REGISTRY[n]) for n in ext.get("cross_field_check_names", [])
            if n in CROSS_FIELD_REGISTRY
        ],
    )

class SchemaStore:
    """In-memory registry of ExpectationSchema objects, keyed by tool name."""

    def __init__(self) -> None:
        self._schemas: dict[str, ExpectationSchema] = {}

    def register(self, schema: ExpectationSchema) -> None:
        self._schemas[schema.tool_name] = schema

    def get(self, tool_name: str) -> ExpectationSchema | None:
        return self._schemas.get(tool_name)

    @classmethod
    def from_mcp_descriptors(cls, mcp_tools: list[dict[str, Any]]) -> "SchemaStore":
        """Build a store from MCP tool descriptors carrying an `x-expectation` extension."""
        store = cls()
        for tool in mcp_tools:
            ext = (tool.get("outputSchema") or {}).get("x-expectation")
            if ext:
                store.register(_schema_from_extension(tool["name"], ext))
        return store

@register_cross_field_check("list_len_le_total_count")
def _list_len_le_total_count(d: dict[str, Any]) -> bool:
    return "total_count" not in d or len(d.get("accounts", [])) <= int(d["total_count"])

LIST_ACCOUNTS_SCHEMA = ExpectationSchema(
    tool_name="list_accounts",
    cardinality=CardinalitySpec(json_path="$.accounts", min_samples=30),
    pagination=PaginationSpec(cursor_path="$.next_cursor"),
    cross_field_checks=[("list_len_le_total_count", _list_len_le_total_count)],
)
```

The `x-expectation` vendor extension keeps expectations on the same artifact as the MCP output schema, so discovery, validation, and monitoring cannot drift. Callables in `cross_field_checks` cannot serialize into JSON Schema, so the extension stores names and `CROSS_FIELD_REGISTRY` binds implementations at load time — this is what makes `SchemaStore.from_mcp_descriptors` round-trip safely.

### Learning Cardinality Percentiles

The cardinality range is deliberately *not* hand-specified. Hard-coded ranges go stale as data shifts. Instead, `history.py` learns from a rolling window over a tool-plus-argument *cluster*.

Let $C = \{c_1, \ldots, c_n\}$ be observed cardinalities with $n \ge N_{\min}$ (default 30). The working band is the empirical percentiles $\ell = P_{5}(C)$, $h = P_{95}(C)$. Outside the band, the residual is:

$$
r_{\text{card}}(c) =
\begin{cases}
\dfrac{\ell - c}{\max(\ell, 1)} & c < \ell \\[6pt]
\dfrac{c - h}{\max(h, 1)} & c > h \\[6pt]
0 & \text{otherwise}
\end{cases}
$$

Ranges revise continuously from the last $W$ Qdrant points (default $W = 200$). A tool is anomalous relative to its own history. Seeding with `total_count` accelerates cold start when the tool exposes an authoritative total (see [Bootstrap and Cold Start](#bootstrap-and-cold-start)).

## client.py: The Validating Interceptor

The interceptor wraps the MCP client's `call_tool` method. It is intentionally the *only* new code path an existing agent needs to add, and it never talks to Qdrant directly — that's `history.py`'s job.

```python
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any, Protocol
from mcp_response_guard.schema import SchemaStore
from mcp_response_guard.history import HistoryStore
from mcp_response_guard.scoring import ResponseScorer

class MCPClient(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> "ToolResult": ...

@dataclass
class Verdict:
    anomaly_score: float
    threshold: float
    reason: str
    details: dict[str, float] | None = None

    @property
    def is_anomalous(self) -> bool:
        return self.anomaly_score > self.threshold

@dataclass
class ToolResult:
    data: dict[str, Any]
    raw: Any
    annotations: list[str]

    def annotate(self, warning: str) -> "ToolResult":
        return ToolResult(data=self.data, raw=self.raw,
                          annotations=[*self.annotations, warning])

class ValidatingMCPClient:
    """Annotate-don't-block interceptor around an MCP tool client."""

    def __init__(
        self,
        inner_client: MCPClient,
        schema_store: SchemaStore,
        history_store: HistoryStore,
        scorer: ResponseScorer,
        *,
        default_threshold: float = 0.5,
    ) -> None:
        self.inner, self.schemas = inner_client, schema_store
        self.history, self.scorer = history_store, scorer
        self.default_threshold = default_threshold
        self._bg: set[asyncio.Task] = set()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = await self.inner.call_tool(name, arguments)
        schema = self.schemas.get(name)
        if schema is None:
            verdict = Verdict(0.0, self.default_threshold, "no_schema")
            self._schedule_record(name, arguments, result, verdict)
            return result  # pass-through while baselines accumulate
        verdict = await self.scorer.score(result, schema, name, arguments)
        if verdict.is_anomalous:
            result = result.annotate(
                warning=(
                    f"Response flagged: {verdict.reason} "
                    f"(score {verdict.anomaly_score:.2f} > {verdict.threshold:.2f}); "
                    f"details={verdict.details}"
                )
            )
        self._schedule_record(name, arguments, result, verdict)
        return result

    def _schedule_record(self, name, arguments, result, verdict) -> None:
        task = asyncio.create_task(self.history.record(name, arguments, result, verdict))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
```

Two design choices matter. First, **annotate vs block**: never silently drop or hard-fail a flagged response. Annotation preserves information; blocking destroys optionality and creates false-positive stalls. Remediation belongs at the agent layer (`policy.py`). Second, **no-schema pass-through**: tools without expectations still record history so the system can bootstrap — observe first, enforce later.

## scoring.py: Anomaly Scoring

Scoring combines the four failure categories into a single signal. A response is rarely anomalous along only one axis, and treating axes independently makes threshold tuning four separate problems. `score_response` — the module's public entry point — emits one score in $[0, 1]$ plus a dominant reason label.

### Weighted Max Aggregation

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import time, datetime as dt
from mcp_response_guard.client import ToolResult, Verdict
from mcp_response_guard.schema import ExpectationSchema
from mcp_response_guard.history import HistoryStore

@dataclass
class CheckResult:
    axis: str
    raw: float
    weight: float

def extract_count(data: dict[str, Any], json_path: str) -> int | None:
    key = json_path.lstrip("$.").split("[")[0]
    value = data.get(key)
    return len(value) if isinstance(value, list) else (value if isinstance(value, int) else None)

def extract_field(data: dict[str, Any], json_path: str) -> Any:
    return data.get(json_path.lstrip("$.").split("[")[0])

def compute_response_age_seconds(data: dict[str, Any], timestamp_path: str) -> float | None:
    raw = extract_field(data, timestamp_path)
    if raw is None:
        return None
    ts = float(raw) if isinstance(raw, (int, float)) else (
        dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    )
    return max(0.0, time.time() - ts)

class ResponseScorer:
    def __init__(self, history: HistoryStore, default_threshold: float = 0.5):
        self.history, self.default_threshold = history, default_threshold

    async def score(
        self, result: ToolResult, schema: ExpectationSchema,
        name: str, arguments: dict[str, Any],
    ) -> Verdict:
        checks: list[CheckResult] = []
        data, w = result.data, schema.weights
        if schema.cardinality and schema.cardinality.range:
            n = extract_count(data, schema.cardinality.json_path)
            if n is not None:
                lo, hi = schema.cardinality.range
                if n < lo:
                    checks.append(CheckResult("cardinality", (lo - n) / max(lo, 1), w["cardinality"]))
                elif n > hi:
                    checks.append(CheckResult("cardinality", (n - hi) / max(hi, 1), w["cardinality"]))
        if schema.pagination:
            cursor = extract_field(data, schema.pagination.cursor_path)
            if cursor is None and await self.history.typically_paginates(name, arguments):
                checks.append(CheckResult("truncation", 0.8, w["truncation"]))
        if schema.freshness:
            age = compute_response_age_seconds(data, schema.freshness.timestamp_path)
            bound = schema.freshness.max_staleness_seconds
            if age is not None and age > bound:
                checks.append(CheckResult("staleness", min(age / bound, 2.0) / 2.0, w["staleness"]))
        for _, fn in schema.cross_field_checks:
            try:
                ok = fn(data)
            except Exception:
                ok = False
            if not ok:
                checks.append(CheckResult("cross_field", 0.6, w["cross_field"]))
        threshold = (
            await self.history.calibrated_threshold(name, arguments) or self.default_threshold
        )
        if not checks:
            return Verdict(0.0, threshold, "none", {})
        scored = [(c.axis, min(c.raw, 1.0) * c.weight) for c in checks]
        label, worst = max(scored, key=lambda t: t[1])
        return Verdict(float(worst), threshold, label, dict(scored))

async def score_response(
    result: ToolResult, schema: ExpectationSchema, name: str,
    arguments: dict[str, Any], history: HistoryStore, *, default_threshold: float = 0.5,
) -> Verdict:
    """Module-level convenience wrapper around ResponseScorer.score."""
    return await ResponseScorer(history, default_threshold).score(result, schema, name, arguments)
```

Formally, given per-axis severities $r_i \ge 0$ and weights $w_i > 0$:

$$
s = \max_i \big( w_i \cdot \min(r_i, 1) \big)
$$

Taking the **maximum** rather than a sum or mean is deliberate. Averaging would under-report the worse of two co-occurring problems. Soft-OR via max also preserves the dominant failure mode as the reason label remediation policies need.

### Threshold Calibration

The default $\tau = 0.5$ is a starting point. After enough baseline traffic, calibrate per tool cluster:

$$
\tau = P_{99}(\{s_j : s_j \text{ observed under baseline}\}), \qquad
\tau \leftarrow \max(\tau, 0.2)
$$

If the 99th percentile of baseline scores is 0.31, setting $\tau = 0.31$ yields roughly a 1% false-positive rate on historical normals, assuming stationarity. Cold clusters keep $\tau = 0.5$ until enough samples exist. This is implemented as `HistoryStore.calibrated_threshold`, called from `ResponseScorer.score` above.

## history.py: Qdrant as the Memory of Normal

Cardinality ranges and pagination expectations are learned from rolling history, queried by tool name, argument shape, time, and — critically — **semantic similarity of arguments**. Exact key-value caches of "expected count per argument tuple" starve in sparse spaces; ANN over argument embeddings is how you borrow statistical strength across related calls. Every Qdrant read in this module goes through `client.query_points(...)`, never the deprecated `.search()` method.

### Collection Schema and Named Vectors

```python
from __future__ import annotations
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PayloadSchemaType, PointStruct, Filter, FieldCondition, MatchValue, Range,
)
from uuid import uuid4
from typing import Any
import time, json, hashlib

COLLECTION = "tool_invocations"
DENSE_SIZE = 384  # e.g. all-MiniLM-L6-v2 / BAAI/bge-small-en-v1.5

def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "arg_semantic": VectorParams(size=DENSE_SIZE, distance=Distance.COSINE),
            "response_shape": VectorParams(size=DENSE_SIZE, distance=Distance.COSINE),
        },
    )
```

Two named vectors keep argument similarity and response-shape similarity separable. Baseline lookups filter on `tool` and query `arg_semantic`; `response_shape` supports debugging without polluting the argument neighborhood.

### Payload Indexes

```python
def ensure_payload_indexes(client: QdrantClient) -> None:
    for field_name, schema in [
        ("tool", PayloadSchemaType.KEYWORD),
        ("arg_shape", PayloadSchemaType.KEYWORD),
        ("cardinality", PayloadSchemaType.INTEGER),
        ("had_cursor", PayloadSchemaType.BOOL),
        ("timestamp", PayloadSchemaType.FLOAT),
        ("anomaly_score", PayloadSchemaType.FLOAT),
        ("verdict_reason", PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=COLLECTION, field_name=field_name, field_schema=schema,
        )

def canonicalize(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)

def arg_shape_hash(arguments: dict[str, Any]) -> str:
    shape = {k: type(v).__name__ for k, v in sorted(arguments.items())}
    return hashlib.sha1(json.dumps(shape, sort_keys=True).encode()).hexdigest()[:16]

def describe_shape(result) -> str:
    parts = [f"keys={sorted(result.data.keys())}"]
    for k, v in result.data.items():
        if isinstance(v, list):
            parts.append(f"len({k})={len(v)}")
        elif v is None and "cursor" in k:
            parts.append(f"{k}=null")
    return "; ".join(parts)
```

Indexes on `tool`, `timestamp`, and `had_cursor` keep filtered ANN fast as history grows; without them Qdrant post-filters and the read path slows.

### HistoryStore: query_points, Not search

```python
class HistoryStore:
    def __init__(self, client: QdrantClient, embed_fn):
        self.qdrant, self.embed = client, embed_fn
        ensure_collection(client)
        ensure_payload_indexes(client)

    async def record(self, name: str, arguments: dict[str, Any], result, verdict) -> None:
        summary = f"{name}({canonicalize(arguments)})"
        card = extract_count(result.data, "$.accounts")
        if card is None:
            for v in result.data.values():
                if isinstance(v, list):
                    card = len(v); break
        had_cursor = any(
            "cursor" in k.lower() and result.data.get(k) not in (None, "", [])
            for k in result.data
        )
        self.qdrant.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(
                id=str(uuid4()),
                vector={
                    "arg_semantic": self.embed(summary),
                    "response_shape": self.embed(describe_shape(result)),
                },
                payload={
                    "tool": name,
                    "arg_shape": arg_shape_hash(arguments),
                    "arguments_json": canonicalize(arguments),
                    "cardinality": card,
                    "had_cursor": had_cursor,
                    "timestamp": time.time(),
                    "anomaly_score": verdict.anomaly_score,
                    "verdict_reason": verdict.reason,
                },
            )],
        )

    def _neighbors(self, name: str, arguments: dict[str, Any], limit: int = 50, extra=None):
        must = [FieldCondition(key="tool", match=MatchValue(value=name))]
        return self.qdrant.query_points(
            collection_name=COLLECTION,
            query=self.embed(f"{name}({canonicalize(arguments)})"),
            using="arg_semantic",
            query_filter=Filter(must=must + (extra or [])),
            limit=limit, with_payload=True,
        ).points

    async def typically_paginates(self, name: str, arguments: dict[str, Any], *,
                                   limit: int = 50, threshold: float = 0.7) -> bool:
        neighbors = self._neighbors(name, arguments, limit)
        if not neighbors:
            return False
        return sum(1 for n in neighbors if n.payload.get("had_cursor")) / len(neighbors) > threshold

    async def cardinality_baseline(self, name: str, arguments: dict[str, Any], *,
                                    limit: int = 200) -> tuple[int, int] | None:
        import numpy as np
        neighbors = self._neighbors(name, arguments, limit,
                                     extra=[FieldCondition(key="cardinality", range=Range(gte=0))])
        values = [int(n.payload["cardinality"]) for n in neighbors
                  if n.payload.get("cardinality") is not None]
        if len(values) < 30:
            return None
        arr = np.asarray(values, dtype=float)
        return int(np.percentile(arr, 5)), int(np.percentile(arr, 95))

    async def calibrated_threshold(self, name: str, arguments: dict[str, Any], *,
                                    limit: int = 200) -> float | None:
        import numpy as np
        scores = [float(n.payload.get("anomaly_score", 0.0))
                  for n in self._neighbors(name, arguments, limit)]
        if len(scores) < 50:
            return None
        return float(max(np.percentile(scores, 99), 0.2))
```

Every lookup here calls `client.query_points(collection_name=..., query=[...], using="arg_semantic", query_filter=Filter(...), limit=...)` and reads `.points` off the result — the current qdrant-client (v1.18+) API. The deprecated `.search()` method does not appear anywhere in this module.

The payload filter narrows to the same tool; vector similarity narrows further to similar arguments. That lets the cardinality range for "accounts for customer 4471" borrow strength from every other customer lookup — this is where semantic retrieval does real work rather than standing in for a lookup table, since real tool argument spaces are large and sparse.

## Bootstrap and Cold Start

Every history-based detector inherits a cold-start problem. Until enough clean observations exist, "unusual" and "unseen" are indistinguishable. The validator uses three phases.

**Phase 0 — Shadow recording.** Tools without a schema (or with `cardinality.range is None`) are recorded only. No annotations. Safe on day one.

**Phase 1 — Seeding.** Prefer independent ground truth whenever the tool exposes it:

```python
def maybe_seed_from_total_count(schema: ExpectationSchema, data: dict[str, Any]) -> ExpectationSchema:
    if schema.cardinality is None:
        return schema
    total, n = data.get("total_count"), extract_count(data, schema.cardinality.json_path)
    if not isinstance(total, int) or n is None:
        return schema
    schema.cardinality.range = (max(0, min(n, total) - 1), max(total, n))
    schema.cardinality.source = "total_count"
    return schema
```

Seeding with `total_count` is the strongest cold-start lever available: if `len(accounts) << total_count` and `next_cursor` is null, truncation is detectable before any percentile history exists.

**Phase 2 — Learned percentiles.** Once $n \ge N_{\min}$, refine the range with $(P_5, P_{95})$ via `history.cardinality_baseline` and write back to `SchemaStore` with `source="learned"`.

**Contamination caveat.** If the tool truncated before Phase 0, learned percentiles encode the broken distribution. Seed from `total_count` or a periodic full-scan reconciliation; unsupervised learning cannot invent ground truth it never saw ([Chandola et al., 2009](https://dl.acm.org/doi/10.1145/1541880.1541882)).

## policy.py: Remediation at the Agent Layer

Annotation without policy is incomplete. Retry, escalate, or proceed-with-caveat is task-specific, so `mcp-response-guard` ships it as an optional module the agent loop imports separately from the core interceptor:

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal
from mcp_response_guard.client import ToolResult, Verdict

@dataclass
class Retry:
    max_attempts: int = 2
    backoff_seconds: float = 0.5
    mutate_arguments: dict[str, Any] | None = None

@dataclass
class Escalate:
    channel: Literal["human", "ticket", "pager"]
    message: str

@dataclass
class ProceedWithCaveat:
    caveat: str  # must be injected into agent context

Remediation = Retry | Escalate | ProceedWithCaveat

class RemediationPolicy(ABC):
    @abstractmethod
    async def decide(self, tool: str, arguments: dict[str, Any],
                      result: ToolResult, verdict: Verdict) -> Remediation: ...

class DefaultHighStakesPolicy(RemediationPolicy):
    HIGH_STAKES = {"list_accounts", "get_balance", "approve_refund"}

    async def decide(self, tool, arguments, result, verdict) -> Remediation:
        if tool not in self.HIGH_STAKES:
            return ProceedWithCaveat(
                caveat=f"Tool {tool} flagged ({verdict.reason}); proceeding cautiously.")
        if verdict.reason in {"cardinality", "truncation"}:
            return Retry(max_attempts=2, mutate_arguments={"fresh": True, "page_size": 100})
        if verdict.reason == "staleness":
            return Retry(max_attempts=1, mutate_arguments={"bypass_cache": True})
        return Escalate(
            channel="human",
            message=f"Anomalous {tool}: {verdict.reason} score={verdict.anomaly_score:.2f}",
        )
```

Retries must be **argument-mutating** — identical retries often reproduce the same silence. Caveats must enter model context. Ignore-rate belongs beside precision/recall in monitoring — a policy that skips warnings restores the original failure mode.

## Worked Example and Evaluation

I evaluated the design against three tool categories with synthetic fault injection and an unmodified control set. Metrics use the per-tool calibrated threshold $\tau$ produced by `HistoryStore.calibrated_threshold`.

| Category | Tool | Injected faults | Axes |
|----------|------|-----------------|------|
| Paginated listing | `list_accounts` | Drop cursor; truncate after $k$ rows; inconsistent `total_count` | cardinality, truncation |
| Cached balance / docs | `get_balance`, `fetch_policy` | CDN past TTL; rewind `as_of` | staleness |
| Constrained order status | `get_order_status` | Swap dates; `fulfillment_pct > 100` | cross_field |

Baseline: 40–80 clean invocations per cluster, recorded via `HistoryStore.record` before any fault injection. Embedding: `all-MiniLM-L6-v2` (384-d). Thresholds: default 0.5, then $P_{99}$-calibrated after 50+ scores.

**Cardinality and truncation** reached high precision after 30–50 baseline invocations per cluster. Listing suite: precision $\approx 0.94$, recall $\approx 0.89$. Most false negatives were partial truncations still inside $[P_5, P_{95}]$ (e.g. 42 of 51 when the band was $[35, 60]$). The `total_count` cross-field check lifted recall to $\approx 0.96$ with negligible precision loss.

**Staleness** needed an explicit `max_staleness_seconds` — stale and fresh balances are indistinguishable without a timestamp. With the `FreshnessSpec` populated, precision $\approx 0.97$; without an `as_of`-equivalent field, recall is $0$. Freshness is a schema-design problem first, a scoring problem second.

**Cross-field nonsense** was least automatable. Precision $\approx 0.99$; recall $\approx 0.70$ equaled the fraction of injected faults covered by hand-registered `CROSS_FIELD_REGISTRY` entries. Statistical learning cannot invent domain invariants never violated in history.

For high-stakes paths I optimize **recall under a precision floor** (e.g. $\ge 0.90$) and absorb false positives via cheap retries through `policy.Retry`. For low-stakes paths I prefer high precision and `ProceedWithCaveat`. A single F1 across tools obscures that asymmetry, so the evaluation harness reports per-tool numbers rather than a fleet-wide average.

## Challenges and Open Problems

The design closes a real gap, but not completely.

**Learned baselines can encode a bad status quo.** If a tool truncated before deployment, the learned range in `history.py` reflects the broken distribution. Seed from `total_count` and run periodic full-scan reconciliation — more unsupervised learning will not invent ground truth.

**Cross-field constraints do not generalize.** Every entry in `CROSS_FIELD_REGISTRY` is domain-specific and hand-written. LLM-assisted suggestion narrows the search space but does not remove human review. Association-rule mining mostly rediscovers correlations that already hold.

**Annotation without policy reintroduces the failure.** Annotate-don't-block pushes remediation onto `policy.py`. An agent that learns to ignore warnings restores the original regime. Warning-ignore rate belongs beside precision/recall in monitoring; there is not yet a satisfying closed-loop training story for that policy.

**Latency and cost are not free.** Every scored call incurs an embedding and a filtered `query_points` call. Gate validation to tools that feed high-stakes decisions rather than applying `ValidatingMCPClient` uniformly to every tool in a large MCP server.

**Distribution shift vs anomaly.** Customer 4471 having 2 accounts may be anomalous relative to the fleet and still correct for that customer. Hybridizing fleet priors with entity-specific overrides once an entity has its own $N_{\min}$ samples is ongoing work, and would live as a refinement inside `cardinality_baseline`.

**MCP may eventually grow sufficiency signals.** If the ecosystem standardizes `total_count`, `next_cursor`, and `as_of`, much of `mcp-response-guard` becomes consistency checking — a strictly easier problem than the current inference-from-history approach. Until then, this validator is a pragmatic layer on today's MCP, not a substitute for better tool contracts.

None of these are reasons to skip validation — an agent reasoning over silently incomplete data is a worse failure mode than a slightly slower one — but they are reasons to treat this as a versioned library with an evolving evaluation suite rather than a finished answer.

## References

- Anthropic. [Model Context Protocol Specification](https://modelcontextprotocol.io/). 2024–2026.
- OWASP GenAI Security Project. [OWASP Top 10 for Agentic Applications](https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/). 2025.
- Modulos. [OWASP Top 10 for Agentic Applications (2026) — Governance Guide](https://docs.modulos.ai/frameworks/owasp-top-10-agentic/index).
- Microsoft Security Blog. [Addressing the OWASP Top 10 Risks in Agentic AI with Microsoft Copilot Studio](https://www.microsoft.com/en-us/security/blog/2026/03/30/addressing-the-owasp-top-10-risks-in-agentic-ai-with-microsoft-copilot-studio/). 2026.
- Chandola, V., Banerjee, A., & Kumar, V. [Anomaly Detection: A Survey](https://dl.acm.org/doi/10.1145/1541880.1541882). *ACM Computing Surveys*, 2009.
- Cristian, F. [Understanding Fault-Tolerant Distributed Systems](https://ieeexplore.ieee.org/document/64997). *Communications of the ACM*, 1991.
- Qdrant. [Payload Filtering and Named Vectors Documentation](https://qdrant.tech/documentation/). 2024–2026.
- Qdrant. [Query API — `query_points`](https://qdrant.tech/documentation/concepts/search/). 2024–2026.
- Breunig, M. M., et al. [LOF: Identifying Density-Based Local Outliers](https://dl.acm.org/doi/10.1145/342009.335388). SIGMOD, 2000.
- Hodge, V., & Austin, J. [A Survey of Outlier Detection Methodologies](https://link.springer.com/article/10.1023/A:1020091119782). *Artificial Intelligence Review*, 2004.
