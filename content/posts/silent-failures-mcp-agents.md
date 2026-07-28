---
title: "Silent Failures in MCP Agent Workflows: Detecting Incomplete Tool Responses"
date: 2026-07-24
description: "Why a successful MCP tool call can still be a silent failure — incomplete, truncated, or unrepresentative responses — and how a schema-aware response validator backed by Qdrant detects them before the agent acts."
tags: ["agents", "mcp", "qdrant", "validation", "reliability", "tool-calling"]
author: "Mihir Inamdar"
showToc: true
math: true
---

An agent calling a tool through the Model Context Protocol treats a `200`-equivalent response as ground truth. It has no innate sense of whether the payload it just received is *complete*, *representative*, or merely *well-formed*. This post is about the gap between "the call succeeded" and "the call returned what the agent needed," why that gap is invisible by default, and how to close it with a schema-aware response validator backed by Qdrant. I will not cover prompt injection or tool-description poisoning here — that is a distinct failure mode with its own literature — this post only focuses on failures where the tool itself is behaving as designed, but the response is quietly insufficient for the decision the agent is about to make.

## Table of Contents

- [The Problem: Success at the Transport Layer, Failure at the Semantic Layer](#the-problem-success-at-the-transport-layer-failure-at-the-semantic-layer)
- [Why This Is Different from Known Failure Modes](#why-this-is-different-from-known-failure-modes)
- [A Taxonomy of Silent Failures](#a-taxonomy-of-silent-failures)
- [Design: A Schema-Aware Response Validator](#design-a-schema-aware-response-validator)
- [Component One: Expectation Schemas](#component-one-expectation-schemas)
- [Component Two: The Interceptor](#component-two-the-interceptor)
- [Component Three: Anomaly Scoring](#component-three-anomaly-scoring)
- [Component Four: Qdrant as the Memory of "What Normal Looks Like"](#component-four-qdrant-as-the-memory-of-what-normal-looks-like)
- [Putting It Together](#putting-it-together)
- [Evaluation](#evaluation)
- [Challenges and Open Problems](#challenges-and-open-problems)

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

This is the general shape of the problem: **a response can be schema-valid and still be semantically incomplete**, and MCP — like most RPC protocols before it — has no native mechanism to distinguish the two.

## Why This Is Different from Known Failure Modes

It is worth being precise about scope, because agentic security research in 2026 has produced a rich taxonomy of adjacent problems, and silent incompleteness is not the same as any of them.

**Tool misuse** (ASI02 in the [OWASP Top 10 for Agentic Applications](https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/)) describes an agent bending a legitimate tool toward a destructive or unintended action — the tool does something it shouldn't. **Memory and context poisoning** (ASI06) describes an adversary shaping what an agent retrieves from persistent storage so that future reasoning is misled — the *store* has been corrupted. Both are adversarial: something or someone is actively trying to manipulate the agent.

The failure mode in this post has no adversary. The tool is not compromised, the description has not been tampered with, no one is injecting instructions. A load balancer drops connections under pressure, a database index is temporarily unavailable, a third-party API rate-limits and returns a valid-but-empty page, a cache serves stale data during a deploy. These are the ordinary failure modes of distributed systems, and they have always required careful handling — the difference is that a human calling an API directly will *notice* an empty list where fifty rows were expected, because a human has priors about what "makes sense." An agent, absent explicit instrumentation, has no such prior. It reads a well-typed JSON array, and a well-typed empty array looks exactly as valid as a well-typed full one.

## A Taxonomy of Silent Failures

Not all incomplete responses look alike, and a validator that only checks for empty results will miss most of them. I find it useful to separate silent failures into four categories, ordered from easiest to hardest to catch automatically.

**Cardinality failures.** The response has the right shape but the wrong count — one account instead of fifty-one, three search results instead of an expected hundred. These are the most tractable, because most tool calls have a queryable expected range (via prior invocations, via a `total_count` field the tool exposes but the agent doesn't check, or via a companion tool that reports counts independently).

**Truncation failures.** The response is a prefix of the correct answer — page one of a paginated result with the cursor silently dropped, the first 4KB of a document that was supposed to stream in full. These look structurally identical to a legitimately short answer, and are usually only detectable by cross-referencing a pagination or length field the caller failed to inspect.

**Staleness failures.** The response is complete and well-formed, but reflects state from before a change the agent needed to see — a cached account balance from six hours ago, a document version before the edit currently under discussion. These require a notion of recency that the schema alone does not encode.

**Type-conformant nonsense.** The response passes every structural check but contains values that are individually plausible and jointly absurd — a shipping date before the order date, a percentage above 100, a user ID that does not match any format ever issued. This category is the hardest, because it requires cross-field or domain-level constraints rather than per-field type checks.

The common thread is that all four are invisible to anything that validates only *shape* (does this parse as the declared JSON schema?) rather than *expectation* (does this match what a response to this query, at this point, should look like?).

## Design: A Schema-Aware Response Validator

The system I am describing sits as a thin interceptor between the MCP client and the agent's reasoning loop. It does four things: it defines what a "normal" response looks like for each tool call, it intercepts the actual response before the agent sees it, it scores the response against the expectation, and it remembers enough history to make that scoring better over time. I will walk through each in turn.

### Component One: Expectation Schemas

An expectation schema extends a tool's ordinary JSON schema with three additional annotations that MCP itself does not require: a cardinality range, a set of cross-field constraints, and a freshness bound.

```python
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class ExpectationSchema:
    tool_name: str
    cardinality_field: str | None = None       # e.g. "$.accounts" (JSONPath)
    cardinality_range: tuple[int, int] | None = None  # observed 5th–95th percentile
    max_staleness_seconds: int | None = None
    cross_field_checks: list[Callable[[dict], bool]] = field(default_factory=list)
    pagination_field: str | None = None        # e.g. "$.next_cursor"
```

The cardinality range is deliberately *not* hand-specified up front. Engineers are bad at guessing how many rows a query "usually" returns, and a hard-coded range goes stale the moment the underlying data distribution shifts. Instead, the range is learned — the validator observes the first N invocations of a given tool-plus-parameter shape, computes the 5th and 95th percentile of the observed cardinality, and uses that as the working expectation, revising it continuously as new calls arrive. A tool is anomalous relative to its own history, not relative to a number a developer typed into a config file eight months ago and forgot about.

### Component Two: The Interceptor

The interceptor wraps the MCP client's `call_tool` method. It is intentionally the *only* new code path an existing agent needs to add — everything else in this design lives behind it.

```python
class ValidatingMCPClient:
    def __init__(self, inner_client, schema_store, history_store):
        self.inner = inner_client
        self.schemas = schema_store       # ExpectationSchema per tool
        self.history = history_store      # Qdrant-backed, see Component Four

    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        result = await self.inner.call_tool(name, arguments)
        schema = self.schemas.get(name)
        if schema is None:
            return result  # no expectation yet — record and pass through

        verdict = score_response(result, schema, self.history, name, arguments)
        self.history.record(name, arguments, result, verdict)

        if verdict.anomaly_score > verdict.threshold:
            result = result.annotate(
                warning=f"Response flagged: {verdict.reason} "
                        f"(score {verdict.anomaly_score:.2f})"
            )
        return result
```

Two design choices are worth calling out. First, the interceptor never silently drops or blocks a flagged response — it annotates it and lets the agent's own reasoning decide what to do, because in practice the right response to "this looks unusually short" is often task-specific (retry with different parameters, ask the user, or proceed with an explicit caveat), and hard-coding that decision inside the transport layer removes flexibility the agent needs. Second, a tool with no schema yet degrades to a pure pass-through that still records history — the system bootstraps its own expectations from traffic rather than requiring every tool to be manually annotated before it can be monitored at all.

### Component Three: Anomaly Scoring

Scoring combines the four failure categories from the taxonomy above into a single weighted signal, because in practice a response is rarely anomalous along only one axis, and treating them independently makes the threshold-tuning problem four times harder than it needs to be.

```python
def score_response(result, schema, history, name, arguments) -> Verdict:
    checks = []

    if schema.cardinality_field and schema.cardinality_range:
        n = extract_count(result, schema.cardinality_field)
        lo, hi = schema.cardinality_range
        if n < lo:
            checks.append(("cardinality", (lo - n) / max(lo, 1)))

    if schema.pagination_field:
        cursor = extract_field(result, schema.pagination_field)
        if cursor is None and history.typically_paginates(name, arguments):
            checks.append(("truncation", 0.8))

    if schema.max_staleness_seconds:
        age = compute_response_age(result)
        if age is not None and age > schema.max_staleness_seconds:
            checks.append(("staleness", min(age / schema.max_staleness_seconds, 2.0) / 2))

    for check in schema.cross_field_checks:
        if not check(result.data):
            checks.append(("cross_field", 0.6))

    if not checks:
        return Verdict(anomaly_score=0.0, threshold=0.5, reason="none")

    label, worst = max(checks, key=lambda c: c[1])
    return Verdict(anomaly_score=worst, threshold=0.5, reason=label)
```

Taking the maximum rather than summing across checks is a deliberate choice: a response that is merely a little stale is very different from a response that is both stale and truncated, and averaging the two would under-report the worse of the two problems. The threshold of 0.5 is a starting point, not a constant — Component Four is what lets it move.

### Component Four: Qdrant as the Memory of "What Normal Looks Like"

The cardinality ranges, staleness bounds, and pagination expectations in Component One are not static configuration; they are learned from a rolling history of prior invocations, and that history needs to be queried along several axes at once: by tool name, by argument shape, by time window, and — critically — by semantic similarity of the arguments, since "get accounts for customer 4471" and "get accounts for customer 4472" should share statistical history even though their argument strings are not identical.

This is where a vector store earns its place in the design rather than being bolted on for its own sake. Each recorded invocation is embedded — tool name, argument shape, and a short summary of the response — and stored with the raw cardinality, timestamp, and verdict as payload:

```python
def record(self, name: str, arguments: dict, result: ToolResult, verdict: Verdict):
    summary = f"{name}({arguments}) -> {describe_shape(result)}"
    vector = embed(summary)
    self.qdrant.upsert(
        collection_name="tool_invocations",
        points=[{
            "id": uuid4(),
            "vector": vector,
            "payload": {
                "tool": name,
                "arg_shape": canonicalize(arguments),
                "cardinality": extract_any_count(result),
                "had_cursor": extract_any_cursor(result) is not None,
                "timestamp": time.time(),
                "anomaly_score": verdict.anomaly_score,
            },
        }],
    )

def typically_paginates(self, name: str, arguments: dict) -> bool:
    neighbors = self.qdrant.search(
        collection_name="tool_invocations",
        query_vector=embed(f"{name}({arguments})"),
        query_filter={"must": [{"key": "tool", "match": {"value": name}}]},
        limit=50,
    )
    return sum(1 for n in neighbors if n.payload.get("had_cursor")) / max(len(neighbors), 1) > 0.7
```

The payload filter narrows the search to the same tool; the vector similarity narrows it further to invocations with *similar arguments*, which is what lets the cardinality range for "accounts for customer 4471" borrow statistical strength from every other customer lookup rather than requiring its own independent history before it can be judged anomalous at all. This is the one place in the system where semantic retrieval is doing real work rather than standing in for a lookup table — the argument space for real tools is large and sparse, and a plain key-value cache of "expected count per exact argument tuple" would need to see each argument combination dozens of times before it had any statistical basis to flag deviations.

## Putting It Together

The full request path looks like this:

```
Agent
  │
  ▼
ValidatingMCPClient.call_tool()
  │
  ├──▶ inner MCP client ──▶ actual tool ──▶ response
  │
  ├──▶ score_response() ──▶ verdict
  │        │
  │        ├──▶ query Qdrant for historical baseline
  │        └──▶ compare against ExpectationSchema
  │
  ├──▶ history.record() ──▶ Qdrant (async, non-blocking)
  │
  └──▶ annotated response ──▶ Agent
```

*The interceptor sits between the agent's tool-call and the underlying MCP transport; scoring and recording both consult the same Qdrant collection, but recording is fire-and-forget so it never adds latency to the agent's reasoning loop.*

The recording step is deliberately asynchronous and best-effort — a lost history write degrades the quality of future baselines slightly, but a blocked or slow history write degrades the latency of every single tool call, which is a much worse trade for a monitoring system to make.

## Evaluation

I tested the validator against three tool categories with different failure characteristics: a paginated account-listing endpoint (cardinality and truncation failures), a document-fetch endpoint behind a CDN cache (staleness failures), and a cross-field-constrained order-status endpoint (type-conformant nonsense). Failures were injected synthetically — truncating pagination at the database layer, serving stale cache entries past their TTL, and swapping order/shipment date fields — while a control set of calls ran unmodified.

The cardinality and truncation checks reached high precision quickly, typically after 30–50 baseline invocations per tool-argument cluster, because pagination behavior is a strong, low-variance signal once a tool has been observed a handful of times. Staleness detection required an explicit `max_staleness_seconds` annotation per tool rather than being learnable from traffic alone — there is no way to infer "how fresh should this be" purely from response shape, since a stale-but-plausible balance and a fresh one are byte-for-byte indistinguishable without an external timestamp. Cross-field checks were the least automatable of the four: every one of them had to be hand-written for the domain (shipment date cannot precede order date), which is the expected cost of the "type-conformant nonsense" category described earlier — no amount of statistical learning over past responses substitutes for a domain constraint that was never violated in the training history because it never needed to be.

## Challenges and Open Problems

The design above closes a real gap, but it does not close it completely, and it is worth being explicit about where it falls short.

**Learned baselines can encode a bad status quo.** If a tool has been silently truncating responses since before the validator was deployed, the learned cardinality range will reflect the truncated distribution, not the correct one, and the validator will happily declare the broken behavior "normal." This is the same cold-start problem every anomaly-detection-from-history system has, and the only real mitigation is seeding schemas with an independent ground truth wherever one is obtainable — a `total_count` field the tool exposes but the agent previously ignored, or a periodic full-scan reconciliation job.

**Cross-field constraints do not generalize across tools.** Every one of them is domain-specific, hand-written, and requires someone who understands the business logic to enumerate. This does not scale linearly with the number of tools an organization runs, and I do not have a good answer for automating it beyond narrowing the search space with an LLM-assisted constraint suggester that a human still has to review.

**The system has no opinion on what the agent should do with a flagged response.** Annotating rather than blocking was a deliberate choice in Component Two, but it pushes the actual remediation decision — retry, escalate, proceed with a caveat — back onto agent-level policy that this post does not attempt to specify, and getting that policy wrong (for instance, an agent that learns to simply ignore warnings because retries are expensive) would quietly reintroduce the exact failure mode this system is meant to catch.

**Latency and cost are not free.** Every tool call now incurs an embedding computation and a vector search before the agent can proceed, and while the write path is asynchronous, the read path (scoring) is not. For latency-sensitive agents, this is a real tax, and I would not recommend applying it uniformly to every tool without first identifying which tool calls actually feed high-stakes decisions.

None of these are reasons to skip validation — an agent reasoning over silently incomplete data is a worse failure mode than a slightly slower one — but they are reasons to treat this as a starting framework rather than a finished answer.

---

**References**

- OWASP GenAI Security Project. [OWASP Top 10 for Agentic Applications](https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/). 2025.
- Modulos. [OWASP Top 10 for Agentic Applications (2026) — Governance Guide](https://docs.modulos.ai/frameworks/owasp-top-10-agentic/index).
- Microsoft Security Blog. [Addressing the OWASP Top 10 Risks in Agentic AI with Microsoft Copilot Studio](https://www.microsoft.com/en-us/security/blog/2026/03/30/addressing-the-owasp-top-10-risks-in-agentic-ai-with-microsoft-copilot-studio/). 2026.
