---
title: "Ledger: A Consumer-Side Contract Guard for MCP Tool-Schema Drift"
date: 2026-07-10
description: "The MCP specification's 2026-07-28 revision lets clients cache tools/list results — a real efficiency win that also widens the window during which your agent can silently reason against a stale schema for a remote server you don't control. Ledger is a pip-installable, consumer-side contract guard: daily schema snapshots, a deterministic structural diff, and a Qdrant-backed semantic-drift check that catches reworded tool descriptions a JSON-schema differ cannot."
tags: ["agents", "mcp", "reliability", "observability", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

The **Model Context Protocol**'s [2026-07-28 specification revision](https://modelcontextprotocol.io/specification/2026-07-28/changelog) added caching hints — `ttlMs` and `cacheScope` — to `tools/list` results, letting a client hold onto a server's declared tool set instead of re-fetching it every session. For a server with a large, stable tool surface, that's a genuine efficiency win: fewer round trips, better prompt-cache hit rates, less redundant traffic on a stateless transport. It is also, if you consume MCP servers you don't control, a change that quietly widens an existing blind spot.

This post is about that blind spot specifically: schema drift on a remote MCP server maintained by someone else — a third-party connector, another team's internal server, anything not built and versioned by you — and **Ledger**, a small library that monitors for it from the consuming side by keeping an append-only ledger of every tool's schema history. It does not cover how to design your *own* tool schemas well (that's a naming and scoping discipline, covered elsewhere); it does not cover prompt injection or adversarial tool misuse; and it is not a tutorial on building an MCP server or client. Familiarity with MCP's basic `tools/list` / `tools/call` shape and with Qdrant's filtering and search API is assumed.

## Table of Contents

1. [A Different Failure Mode: Monitoring, Not Design](#a-different-failure-mode-monitoring-not-design)
2. [What "Breaking" Means for a Tool Schema](#what-breaking-means-for-a-tool-schema)
3. [Design: Ledger's Contract Guard](#design-ledgers-contract-guard)
4. [Component One: Daily Snapshots as Ground Truth](#component-one-daily-snapshots-as-ground-truth)
5. [Component Two: The Structural Diff](#component-two-the-structural-diff)
6. [Component Three: Qdrant as the Semantic-Drift Side-Piece](#component-three-qdrant-as-the-semantic-drift-side-piece)
7. [Wiring It Together](#wiring-it-together)
8. [Installing and Using Ledger](#installing-and-using-ledger)
9. [Worked Example: search_customers Quietly Changes Its Manners](#worked-example-search_customers-quietly-changes-its-manners)
10. [Why Cacheable tools/list Makes This More Necessary, Not Less](#why-cacheable-toolslist-makes-this-more-necessary-not-less)
11. [Challenges and Open Problems](#challenges-and-open-problems)
12. [References](#references)

## A Different Failure Mode: Monitoring, Not Design

There's a large body of advice, including **Anthropic**'s own engineering guidance, about writing tool schemas your own agents won't misuse — clear names, tight enums, bounded queries. That's a design-time problem you solve once, for tools you own, before shipping them. It has nothing to do with what this post covers.

The problem here starts after you've stopped designing anything. You're a *consumer* of an MCP server someone else maintains, and that server's tool schemas can change at any point, for reasons entirely outside your visibility, with no obligation on the maintainer's part to tell you. This is the same posture any engineering team already takes toward a third-party REST API it depends on — you watch for breaking changes because the vendor isn't going to page you when they ship one — applied specifically to MCP tool schemas, which are less mature as a monitored surface than REST APIs are, precisely because the ecosystem is younger.

MCP standardizes *discovery* (`tools/list`) and *invocation* (`tools/call`). It does not standardize a change-notification contract or a semantic-versioning scheme between an independent server maintainer and every client that depends on that server. A server operator is free to rename a parameter, tighten a constraint, or reword a description in a point release, and the only signal a consumer gets is whatever the next `tools/list` response happens to contain.

The July 2026 caching change makes this sharper rather than causing it. Before caching hints existed, a client that re-fetched `tools/list` on every session would at least see a changed schema promptly, even without being told anything changed. Once a client is entitled to treat a cached response as fresh for up to `ttlMs` milliseconds — and `cacheScope: "public"` responses are explicitly meant to be reusable across callers, which invites longer effective cache lifetimes in gateways and shared clients — there's a real window during which your agent keeps reasoning against a tool definition the server has already moved past. Nothing in the spec is wrong here; `ttlMs` is a freshness *hint*, and a well-behaved server will still send `notifications/tools/list_changed` when it can. But a hint a client is permitted to honor for an extended period, plus a notification mechanism that depends on the server bothering to fire it and your client bothering to act on it immediately, is not the same guarantee as "you will always be looking at the current schema." The gap this post addresses — nobody watching for drift on a server you don't own — existed before the caching change. The caching change just makes staleness an intended, spec-sanctioned behavior instead of an implementation quirk, which is exactly why the monitoring discipline below needs to live outside whatever your MCP client library does for live traffic.

## What "Breaking" Means for a Tool Schema

Not every schema change is equally dangerous, and treating them all the same either drowns you in noise or lets the important ones through. It helps to have a working taxonomy before building any tooling around it.

**Renaming a required parameter is a hard break.** If `search_customers` renames `query` to `search_term` and `query` was required, every in-flight or newly-planned call built against the old name fails outright — either at JSON-Schema validation on the server side, or downstream if the server accepts the unrecognized field and silently ignores it (arguably worse, since that fails without even an error). There's no ambiguity in classifying this: the field the model was told to fill in no longer exists.

**Narrowing an accepted type or enum is a soft break.** If a `status` parameter's enum shrinks from `["open", "pending", "closed", "archived"]` to `["open", "closed"]`, a call the model would have correctly made yesterday — passing `"archived"` — is rejected today. The model didn't do anything wrong; the space of valid inputs moved out from under a previously valid choice. Type narrowing (a parameter that accepted `string | number` now only accepts `string`) behaves the same way: some previously-valid calls now fail.

**Widening or adding an optional parameter is typically safe.** Adding an optional `region` filter, or widening `limit`'s maximum from 50 to 200, doesn't invalidate any call that was working before. This is the additive case REST API versioning discipline treats as non-breaking, and the same logic transfers directly.

**Rewording a description without touching the schema shape is the case a structural differ cannot see at all, and it can still matter.** Suppose `search_customers`'s JSON Schema is byte-for-byte identical before and after, but the description changes from "Search the customer database" to "Search the customer database. Use sparingly — this endpoint is now rate-limited and expensive to call at high volume." Nothing broke. No call that worked yesterday fails today. But the model's *behavior* can shift meaningfully: an agent that previously called this tool inside a loop to check several customers in sequence might now avoid it, batch differently, or ask a clarifying question it wouldn't have asked before — because the description is a prompt the model reads before every decision to invoke the tool, and prompts change inference even when types don't change at all. A JSON-Schema diff reports zero changes here. This is precisely why a monitoring system built only on structural diffing has a blind spot, and why the design below adds a second, semantic layer specifically to cover it.

## Design: Ledger's Contract Guard

The shape of **Ledger**'s fix follows directly from the taxonomy above: one deterministic layer for the changes a JSON-Schema diff can classify with certainty, and one similarity-based layer for the changes that require comparing meaning, not shape. Both layers read from the same ground truth — a snapshot history that is captured independently of whatever your live MCP client caches for ordinary traffic.

```
                    ┌─────────────────────────────┐
                    │   Daily snapshot job          │
                    │   (forces a live tools/list,  │
                    │    bypasses client-side cache)│
                    └───────────────┬─────────────┘
                                    │
                     writes one ToolSnapshot per tool
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                             ▼
    ┌───────────────────┐                       ┌───────────────────────┐
    │  Structural diff    │                       │  Embed raw_definition   │
    │  (yesterday vs today,│                      │  → Qdrant per-tool       │
    │   pure JSON compare) │                      │   embedding history      │
    └─────────┬──────────┘                       └───────────┬───────────┘
              │                                              │
      hard/soft break flags                        semantic-distance flag
              │                                              │
              └───────────────────┬──────────────────────────┘
                                   ▼
                          human review queue
```

The snapshot job is the only piece that talks to the MCP server. The structural diff and the semantic-drift check are both pure functions over stored data — neither needs live network access, which matters because it means the monitoring cadence is entirely decoupled from the server's actual availability or your own client's cache TTL.

## Component One: Daily Snapshots as Ground Truth

Every tool on every MCP server your agents depend on gets one recorded snapshot per day: its name, its description, and its full JSON Schema for arguments, flattened into a single definition string for later embedding. This is deliberately not fancy — it's an append-only structured log, not a database with update semantics, because the entire point is to keep an immutable history to diff against, not a single current-state row that would overwrite yesterday's version.

```python
from dataclasses import dataclass
from typing import Any
import json


@dataclass
class ToolSnapshot:
    server_name: str
    tool_name: str
    snapshot_date: str        # ISO date, one snapshot per tool per day
    description: str
    input_schema: dict[str, Any]
    raw_definition: str        # name + description + schema, flattened for embedding


def build_snapshot(server_name: str, tool_name: str, description: str,
                    input_schema: dict[str, Any], snapshot_date: str) -> ToolSnapshot:
    raw_definition = (
        f"{tool_name}\n\n{description}\n\n"
        f"{json.dumps(input_schema, sort_keys=True)}"
    )
    return ToolSnapshot(
        server_name=server_name, tool_name=tool_name, snapshot_date=snapshot_date,
        description=description, input_schema=input_schema, raw_definition=raw_definition,
    )


class SnapshotStore:
    """Append-only JSONL log. This is the raw ground truth the rest of the
    pipeline reasons over. It is just stored data — nothing here is Qdrant."""

    def __init__(self, path: str):
        self.path = path

    def append(self, snapshot: ToolSnapshot) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(snapshot.__dict__) + "\n")

    def load_history(self, server_name: str, tool_name: str) -> list[ToolSnapshot]:
        history = []
        with open(self.path) as f:
            for line in f:
                row = json.loads(line)
                if row["server_name"] == server_name and row["tool_name"] == tool_name:
                    history.append(ToolSnapshot(**row))
        return sorted(history, key=lambda s: s.snapshot_date)

    def latest_pair(self, server_name: str, tool_name: str) -> tuple[ToolSnapshot | None, ToolSnapshot | None]:
        """Returns (today, yesterday) for the structural diff. Either may be
        None on the first day a tool is observed."""
        history = self.load_history(server_name, tool_name)
        if not history:
            return None, None
        if len(history) == 1:
            return history[-1], None
        return history[-1], history[-2]
```

The snapshot job itself runs on a fixed daily cadence, forcing a live `tools/list` call against every server regardless of what a caching-aware client would otherwise reuse from its own cache:

```python
def snapshot_all_servers(clients: dict[str, "MCPClientLike"],
                          store: SnapshotStore, snapshot_date: str) -> list[ToolSnapshot]:
    """`clients` maps server_name -> an MCP client. Each client's `list_tools`
    is called with force_refresh=True: the monitoring cadence is independent
    of the client's own tools/list cache TTL, by design — a caching-aware
    client used for live traffic is exactly the thing we don't want to trust
    here, since honoring its cache is the failure mode this job exists to
    catch."""
    snapshots = []
    for server_name, client in clients.items():
        result = client.list_tools(force_refresh=True)
        for tool in result.tools:
            snap = build_snapshot(
                server_name=server_name,
                tool_name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
                snapshot_date=snapshot_date,
            )
            store.append(snap)
            snapshots.append(snap)
    return snapshots
```

Daily is a reasonable default, not a law — a server you invoke a thousand times an hour deserves a tighter cadence than one you call twice a week. The point of decoupling this job from live-traffic caching is that the cadence is a decision you make, not a side effect of a TTL some server operator picked for a different reason.

## Component Two: The Structural Diff

Given two snapshots of the same tool, most of the interesting changes can be classified deterministically by walking the JSON Schema — no embedding needed, no ambiguity, no threshold to tune.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ChangeSeverity(str, Enum):
    HARD_BREAK = "hard_break"    # previously-valid calls will now fail outright
    SOFT_BREAK = "soft_break"    # some previously-valid calls may now fail or behave differently
    SAFE = "safe"                # additive, does not invalidate prior calls
    NONE = "none"


@dataclass
class StructuralChange:
    field_path: str
    change_type: str
    severity: ChangeSeverity
    before: Any
    after: Any


def diff_schemas(before: dict[str, Any], after: dict[str, Any]) -> list[StructuralChange]:
    changes: list[StructuralChange] = []
    before_props = before.get("properties", {})
    after_props = after.get("properties", {})
    before_required = set(before.get("required", []))
    after_required = set(after.get("required", []))

    for name in before_required - after_required:
        if name in after_props:
            changes.append(StructuralChange(
                name, "required_to_optional", ChangeSeverity.SOFT_BREAK, True, False))
        else:
            changes.append(StructuralChange(
                name, "required_field_removed", ChangeSeverity.HARD_BREAK, name, None))

    for name in after_required - before_required:
        changes.append(StructuralChange(
            name, "optional_to_required", ChangeSeverity.HARD_BREAK, False, True))

    for name in before_props.keys() - after_props.keys() - before_required:
        changes.append(StructuralChange(
            name, "optional_field_removed", ChangeSeverity.SOFT_BREAK, name, None))

    for name in after_props.keys() - before_props.keys():
        severity = ChangeSeverity.HARD_BREAK if name in after_required else ChangeSeverity.SAFE
        changes.append(StructuralChange(name, "field_added", severity, None, name))

    for name in before_props.keys() & after_props.keys():
        b, a = before_props[name], after_props[name]

        if b.get("type") != a.get("type"):
            changes.append(StructuralChange(
                f"{name}.type", "type_changed", ChangeSeverity.HARD_BREAK,
                b.get("type"), a.get("type")))

        b_enum, a_enum = set(b.get("enum") or []), set(a.get("enum") or [])
        if b_enum and a_enum and a_enum < b_enum:
            changes.append(StructuralChange(
                f"{name}.enum", "enum_narrowed", ChangeSeverity.SOFT_BREAK,
                sorted(b_enum), sorted(a_enum)))
        elif b_enum and a_enum and a_enum > b_enum:
            changes.append(StructuralChange(
                f"{name}.enum", "enum_widened", ChangeSeverity.SAFE,
                sorted(b_enum), sorted(a_enum)))

        if b.get("default") != a.get("default"):
            changes.append(StructuralChange(
                f"{name}.default", "default_changed", ChangeSeverity.SOFT_BREAK,
                b.get("default"), a.get("default")))

        for bound, direction in [("maximum", "lower"), ("minimum", "higher")]:
            if bound in b and bound in a and a[bound] != b[bound]:
                tightened = a[bound] < b[bound] if direction == "lower" else a[bound] > b[bound]
                if tightened:
                    changes.append(StructuralChange(
                        f"{name}.{bound}", f"{bound}_tightened", ChangeSeverity.SOFT_BREAK,
                        b[bound], a[bound]))

    return changes


def worst_severity(changes: list[StructuralChange]) -> ChangeSeverity:
    if any(c.severity == ChangeSeverity.HARD_BREAK for c in changes):
        return ChangeSeverity.HARD_BREAK
    if any(c.severity == ChangeSeverity.SOFT_BREAK for c in changes):
        return ChangeSeverity.SOFT_BREAK
    if changes:
        return ChangeSeverity.SAFE
    return ChangeSeverity.NONE
```

A default-value change is classified as a soft break rather than something more severe, deliberately: it doesn't reject any call that specifies the parameter explicitly, but it silently changes the outcome of every call that *omits* it, which is exactly the kind of change a model relying on its prior understanding of "what happens if I don't pass `limit`" would walk straight into. This is the category the worked example below lands in.

The version above deliberately handles flat, top-level parameters — the common case for MCP tools, which tend to favor a small number of scalar or enum arguments over deeply nested structures. Nested `object` or `array` parameters need the same walk applied recursively (diff `properties.<name>.properties` the same way once you've confirmed both sides declare `type: "object"`), and `oneOf` / `anyOf` unions need their own branch-matching logic rather than a direct type comparison, since a schema that replaces one arm of a union with a differently-shaped one isn't a simple type change. Neither extension changes the underlying classification scheme — hard break, soft break, safe, none — only the traversal needed to reach every field worth checking.

## Component Three: Qdrant as the Semantic-Drift Side-Piece

The structural diff has nothing to say about a description rewrite that leaves the schema shape untouched. That's the gap **Qdrant** fills: each tool's full definition text — name, description, and parameter documentation, the same `raw_definition` string built during snapshotting — gets embedded and stored, and every new snapshot for that tool is compared against its *own* prior-snapshot embedding history, filtered to that exact `server_name` + `tool_name` pair. A large semantic distance from a tool's own established trajectory, even with an identical JSON Schema, is the signal that a rewording may be shifting what the model infers about correct usage.

The payload carries `server_name`, `tool_name`, `snapshot_date`, and `raw_definition` — enough to filter tightly and to show a human reviewer exactly what changed in context.

```python
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, PayloadSchemaType,
)


class ToolDefinitionHistory:
    """Qdrant holds embeddings of each tool's definition text over time —
    not the raw snapshots themselves, which live in SnapshotStore. This
    collection exists to answer one question: does today's definition read
    like a continuation of this tool's own history, or a departure from it?"""

    def __init__(self, client: QdrantClient, embed_fn, collection: str = "tool_definition_history"):
        self.client, self.embed_fn, self.collection = client, embed_fn, collection
        existing = {c.name for c in client.get_collections().collections}
        if collection not in existing:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            for field_name, schema in [
                ("server_name", PayloadSchemaType.KEYWORD),
                ("tool_name", PayloadSchemaType.KEYWORD),
                ("snapshot_date", PayloadSchemaType.KEYWORD),
            ]:
                client.create_payload_index(
                    collection_name=collection, field_name=field_name, field_schema=schema,
                )

    def record(self, snapshot: ToolSnapshot) -> None:
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=self.embed_fn(snapshot.raw_definition),
                payload={
                    "server_name": snapshot.server_name,
                    "tool_name": snapshot.tool_name,
                    "snapshot_date": snapshot.snapshot_date,
                    "raw_definition": snapshot.raw_definition,
                },
            )],
        )

    def check_drift(self, snapshot: ToolSnapshot, top_k: int = 14,
                     similarity_floor: float = 0.90) -> dict:
        """Embed today's definition and compare it against this exact tool's
        own historical embeddings — nothing else in the collection is
        eligible, because a tool drifting relative to a *different* tool's
        history would be meaningless."""
        hits = self.client.query_points(
            collection_name=self.collection,
            query=self.embed_fn(snapshot.raw_definition),
            query_filter=Filter(must=[
                FieldCondition(key="server_name", match=MatchValue(value=snapshot.server_name)),
                FieldCondition(key="tool_name", match=MatchValue(value=snapshot.tool_name)),
            ]),
            limit=top_k,
            with_payload=True,
        ).points

        if not hits:
            return {"drifted": False, "reason": "no_history_yet", "mean_similarity": None}

        mean_similarity = sum(h.score for h in hits) / len(hits)
        return {
            "drifted": mean_similarity < similarity_floor,
            "reason": "semantic_drift" if mean_similarity < similarity_floor else "stable",
            "mean_similarity": mean_similarity,
            "most_similar_prior_date": hits[0].payload["snapshot_date"],
            "most_similar_prior_text": hits[0].payload["raw_definition"],
        }
```

`query_points` here is doing something slightly different from the deduplication use case this pattern usually shows up in on this blog: instead of asking "does anything *else* in the registry look like this," it's asking "does this tool's *own past self* still look like this," which is why the filter pins both `server_name` and `tool_name` rather than searching the whole collection.

## Wiring It Together

The daily job runs both layers over every snapshot and produces one flagged item per tool that needs a human's attention — never an automated block on anything, since neither layer has enough context to safely act unilaterally.

```python
@dataclass
class DriftReport:
    server_name: str
    tool_name: str
    snapshot_date: str
    structural_changes: list[StructuralChange]
    structural_severity: ChangeSeverity
    semantic_drift: dict


def run_contract_guard(store: SnapshotStore, history: ToolDefinitionHistory,
                        snapshots: list[ToolSnapshot]) -> list[DriftReport]:
    reports = []
    for snap in snapshots:
        today, yesterday = store.latest_pair(snap.server_name, snap.tool_name)
        structural_changes = (
            diff_schemas(yesterday.input_schema, today.input_schema) if yesterday else []
        )
        semantic = history.check_drift(snap)
        history.record(snap)  # record after checking, so today isn't compared to itself

        severity = worst_severity(structural_changes)
        if severity != ChangeSeverity.NONE or semantic["drifted"]:
            reports.append(DriftReport(
                server_name=snap.server_name, tool_name=snap.tool_name,
                snapshot_date=snap.snapshot_date, structural_changes=structural_changes,
                structural_severity=severity, semantic_drift=semantic,
            ))
    return reports
```

Both flags can fire independently, on the same tool, for entirely different reasons — which is exactly what the worked example below does.

### Turning Reports Into a Review Queue, Not an Alert Flood

A `DriftReport` per tool per day is still too granular to hand a human directly once you're watching more than a handful of servers — most days, most tools produce nothing to report, and the ones that do shouldn't all land with equal urgency. Routing hard breaks to an immediate, high-visibility channel and batching soft breaks and semantic flags into a daily digest keeps the signal-to-noise ratio workable:

```python
def route_report(report: DriftReport) -> str:
    if report.structural_severity == ChangeSeverity.HARD_BREAK:
        return "page_immediately"
    if report.structural_severity == ChangeSeverity.SOFT_BREAK or report.semantic_drift["drifted"]:
        return "daily_digest"
    return "log_only"


def format_for_digest(report: DriftReport) -> str:
    lines = [f"### {report.server_name} / {report.tool_name} ({report.snapshot_date})"]
    for change in report.structural_changes:
        lines.append(f"- structural: {change.change_type} on `{change.field_path}` "
                      f"({change.before!r} -> {change.after!r}), severity={change.severity.value}")
    if report.semantic_drift["drifted"]:
        sim = report.semantic_drift["mean_similarity"]
        lines.append(f"- semantic: mean similarity to own history dropped to {sim:.2f}")
    return "\n".join(lines)
```

A hard break — a required parameter renamed or removed — genuinely warrants interrupting someone, because any agent still holding the old schema in a cached `tools/list` response is about to start failing calls it thinks are valid. A soft break or a semantic flag warrants a look before the next planning cycle, not a pager alert; both are "something changed and a human should form an opinion about it," not "something is actively broken right now."

## Installing and Using Ledger

Ledger ships as a single pip-installable package around the components built up in this post:

```bash
pip install ledger-mcp
```

```
ledger/
├── snapshot.py     # ToolSnapshot, SnapshotStore, snapshot_all_servers()
├── diff.py           # diff_schemas(), StructuralChange, worst_severity()
├── semantic.py          # ToolDefinitionHistory (Qdrant-backed drift check)
├── report.py              # DriftReport, run_contract_guard()
└── routing.py                # route_report(), format_for_digest()
```

A minimal daily job wired against your existing MCP clients:

```python
from ledger.snapshot import SnapshotStore, snapshot_all_servers
from ledger.semantic import ToolDefinitionHistory
from ledger.report import run_contract_guard
from ledger.routing import route_report, format_for_digest

store = SnapshotStore(path="tool_snapshots.jsonl")
history = ToolDefinitionHistory(qdrant_client, embed_fn)

snapshots = snapshot_all_servers(your_mcp_clients, store, snapshot_date=today)
reports = run_contract_guard(store, history, snapshots)

for report in reports:
    route = route_report(report)
    if route == "page_immediately":
        page_oncall(format_for_digest(report))
    elif route == "daily_digest":
        append_to_digest(format_for_digest(report))
```

`your_mcp_clients` is whatever dictionary of MCP client instances your agent framework already maintains — Ledger doesn't implement its own MCP client, it wraps around one you already have with `force_refresh=True` calls on its own independent schedule.

## Worked Example: search_customers Quietly Changes Its Manners

A different team runs a shared internal MCP server exposing `search_customers(query: str, limit: int = 20)`. Neither `query` nor `limit` is renamed, and `limit`'s type stays `integer` throughout. Two things change quietly in the same release: the default for `limit` drops from `20` to `5`, and the description is reworded from "Search customer records by name or account ID." to "Search customer records by name or account ID. Use sparingly — this endpoint is now rate-limited and expensive to call at high volume."

Yesterday's schema:

```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string"},
    "limit": {"type": "integer", "default": 20, "maximum": 100}
  },
  "required": ["query"]
}
```

Today's schema:

```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string"},
    "limit": {"type": "integer", "default": 5, "maximum": 100}
  },
  "required": ["query"]
}
```

`diff_schemas` runs first and returns exactly one change:

```python
[StructuralChange(
    field_path="limit.default",
    change_type="default_changed",
    severity=ChangeSeverity.SOFT_BREAK,
    before=20,
    after=5,
)]
```

Nothing about `limit`'s name or type moved, so a naive "did the shape change" check would report nothing at all. But any caller that has been omitting `limit` and relying on getting a reasonably sized page — a common, previously reasonable pattern — now silently gets a quarter of the rows it used to, with no error and no obvious symptom beyond "the agent seems to be missing customers it used to find." That's precisely the shape of bug a `default_changed` flag exists to catch before someone spends an afternoon debugging a downstream agent instead of a five-minute schema diff.

`check_drift` runs independently and, because the description changed even though `query` and `limit`'s names and types didn't, returns something like:

```python
{
    "drifted": True,
    "reason": "semantic_drift",
    "mean_similarity": 0.83,
    "most_similar_prior_date": "2026-07-15",
    "most_similar_prior_text": "search_customers\n\nSearch customer records by name or account ID.\n\n{...}",
}
```

The two signals are catching two different things. The structural diff caught a concrete, checkable fact: the default value changed. The semantic-drift check caught something a structural differ has no vocabulary for at all: the tool is now telling the model, in plain language, to change how it uses it — call it less, batch differently, treat it as an expensive resource rather than a cheap lookup. A model that hasn't re-read this description recently (because its client cached the old `tools/list` response) is reasoning against neither of these facts. A model that has read the new description but whose calling agent still assumes the old default is caught by neither signal alone, which is exactly why both layers run on every snapshot rather than either one being treated as sufficient on its own.

## Why Cacheable tools/list Makes This More Necessary, Not Less

It would be easy to read the July 2026 caching change as making a monitoring system like this less relevant — if the client is caching more aggressively, doesn't that mean it's re-fetching less often anyway, so why build separate infrastructure to check for changes?

The reasoning runs the other way. Caching is optimized for the overwhelmingly common case where nothing changed: a stable tool surface, re-fetched needlessly every session, now reused for up to `ttlMs` milliseconds at effectively no cost. That's a real, defensible optimization, and adopting it is the right call for most servers with large, slow-moving tool sets. But optimizing for the common case necessarily means the uncommon case — the schema *did* change — now has a longer runway before anyone notices, because the very mechanism that makes the common case cheap is the one that makes the client's live view of the world stale for longer. A `cacheScope: "public"` response is explicitly designed to be reusable across callers and intermediaries, which is good for reducing load and bad for how quickly a change propagates to every consumer that touches it.

An independent, consumer-side snapshot cadence sidesteps this entirely by never relying on the live client's cache state in the first place. It doesn't care whether your MCP client is honoring a five-minute TTL or a five-day one, because it forces its own fresh `tools/list` call on its own schedule. The caching change is a reason to want this discipline in place, not a reason it's now less needed — the size of the blind spot it closes gets larger, not smaller, as more servers adopt aggressive caching hints.

## Prioritizing Which Servers to Snapshot

Not every MCP server you touch deserves the same monitoring investment, and being explicit about that upfront keeps the system from either missing what matters or drowning in low-value snapshots of tools nobody calls.

A reasonable prioritization axis combines two things you already know without any new instrumentation: how often your agents actually invoke a given server's tools, and how consequential a silent break there would be. A rarely-used internal reporting tool failing loudly the next time someone calls it is an annoyance. A `search_customers`-style tool sitting in the hot path of a customer-facing agent failing silently, or degrading to a fifth of its previous result size, is a production incident wearing a shrug.

```python
@dataclass
class ServerPriority:
    server_name: str
    call_volume_last_30d: int
    in_customer_facing_path: bool
    snapshot_cadence_hours: int


def recommend_cadence(volume: int, customer_facing: bool) -> int:
    if customer_facing and volume > 1000:
        return 6       # four snapshots a day for high-traffic, high-stakes servers
    if customer_facing or volume > 1000:
        return 24      # daily for anything either high-traffic or customer-facing
    return 72          # every three days for low-traffic internal servers
```

This is a starting heuristic, not a formula to defend rigorously — the point is that snapshot cadence should be a deliberate allocation of a scarce resource (how often you're willing to hit someone else's server just to check whether anything changed) rather than a single global constant applied uniformly regardless of how much any given server actually matters.

## Challenges and Open Problems

**Thresholding the semantic-drift signal needs care, and some false positives should be expected rather than eliminated.** A tool whose description gets rewritten for a typo fix or a formatting cleanup — no change in meaning at all — will still register some nonzero embedding distance from its own history, because the text changed even if nothing about correct usage did. Chasing a threshold that produces zero false positives risks pushing it high enough to also miss real drift; a `similarity_floor` around 0.90 (illustrative, not universal) is a starting point to calibrate against your own tools' description-editing habits, not a constant to import unmodified.

**Snapshot cadence bounds what this system can catch, and that bound is real.** A server behind aggressive rate limits, or one your agents invoke rarely enough that a daily snapshot job is itself a meaningful fraction of its total traffic, may not get snapshotted often enough to catch a fast-moving change before it's already caused a problem. Widening the cadence for high-traffic or high-risk servers and accepting a longer detection window for low-traffic ones is a reasonable tradeoff, not a flaw to engineer away — snapshotting every server every minute defeats the purpose of decoupling from live traffic in the first place.

**None of this creates an actual obligation on the server maintainer.** A contract guard built this way is a monitoring discipline running entirely on the consumer's side, compensating for a real gap at the protocol level: MCP has no semantic-versioning or change-notification contract binding an independent server maintainer to every consumer of that server. Detecting drift quickly is valuable and worth building, but it is fundamentally a defensive measure, not a fix — the underlying gap between "the server changed" and "every consumer was told" remains open regardless of how good any single consumer's monitoring gets.

## References

- Model Context Protocol. *Specification.* [modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification)
- Model Context Protocol Blog. *The 2026-07-28 Specification.* [blog.modelcontextprotocol.io/posts/2026-07-28](https://blog.modelcontextprotocol.io/posts/2026-07-28/)
- Model Context Protocol. *Caching (2026-07-28 specification).* [modelcontextprotocol.io/specification/2026-07-28/server/utilities/caching](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/caching)
- Model Context Protocol. *2026-07-28 Changelog.* [modelcontextprotocol.io/specification/2026-07-28/changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- Qdrant. *Qdrant Vector Database Documentation.* [qdrant.tech/documentation](https://qdrant.tech/documentation)
