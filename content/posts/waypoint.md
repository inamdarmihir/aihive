---
title: "Waypoint, For Real: A Working Checkpoint-and-Resume Library for Agent Pipelines"
date: 2026-07-02
description: "The implementation spec for Waypoint, tightened from architecture essay into precise module boundaries: a Workflow engine, a Qdrant-backed CheckpointStore with deterministic point IDs, a HashingEmbedder, and resume/diff/pause semantics an engineer could build from directly."
tags: ["agents", "qdrant", "checkpointing", "durable-execution", "debugging", "workflows"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Agentic systems fail differently than the software we're used to building. A REST endpoint either returns 200 or it doesn't, and when it doesn't, a stack trace tells you exactly where. An agent pipeline that chains an LLM call, a tool call, a retrieval step, and another LLM call can fail at step 5 of 7 after ninety seconds of real work, and the only artifact left behind is usually a log line and a shrug. I described **Waypoint**, a small library for making agent and LLM pipelines durable and debuggable, in an earlier post on this idea; this post tightens that design into something with real, precise module boundaries an engineer — or a coding agent — could actually build. The core insight survives unchanged: checkpoint and embed in the same write, so a durable-execution log doubles as a semantic index over failures. What changes here is precision — real signatures, a real package layout, and the current Qdrant query API throughout. I'll assume basic familiarity with vector databases and LLM agent frameworks (LangChain / LangGraph-style step orchestration), but no prior exposure to durable execution systems is needed.

This post only covers the checkpoint-and-resume problem and the semantic-search-over-failures problem. It does not cover distributed multi-worker execution, automatic retry/backoff policies, or cost-aware model routing — those are related but separate problems I consider out of scope here.

## Table of Contents

1. [The Problem: Non-Determinism Without a Stack Trace](#the-problem-non-determinism-without-a-stack-trace)
2. [Why Checkpoint and Embed in the Same Write](#why-checkpoint-and-embed-in-the-same-write)
3. [Package Layout and Design Overview](#package-layout-and-design-overview)
4. [Installing and Using Waypoint](#installing-and-using-waypoint)
5. [workflow.py: The Workflow Engine](#workflowpy-the-workflow-engine)
6. [checkpoint.py: The Checkpoint Store](#checkpointpy-the-checkpoint-store)
7. [embedder.py: The Embedder Protocol](#embedderpy-the-embedder-protocol)
8. [Resume Semantics: Retry vs Advance](#resume-semantics-retry-vs-advance)
9. [Semantic Search Over Failures](#semantic-search-over-failures)
10. [diffing.py: Time-Travel Debugging via Run Diffs](#diffingpy-time-travel-debugging-via-run-diffs)
11. [Human-in-the-Loop as a First-Class State](#human-in-the-loop-as-a-first-class-state)
12. [Integrating with LangGraph](#integrating-with-langgraph)
13. [Worked Example: Invoice Pipeline Walkthrough](#worked-example-invoice-pipeline-walkthrough)
14. [Related Work](#related-work)
15. [Challenges and Open Problems](#challenges-and-open-problems)
16. [References](#references)

## The Problem: Non-Determinism Without a Stack Trace

Enterprises adopting agentic AI in 2026 report a consistent set of blockers, and they aren't the ones most demos optimize for. In Confluent's 2026 Data Streaming Report, surveying over 4,600 IT leaders, concerns about LLM reliability and non-determinism sit at 68%, and among organizations that have gotten as far as running agents in production, 77% report stalled projects tied to exactly these reliability problems ([Help Net Security, 2026](https://www.helpnetsecurity.com/2026/06/18/report-agentic-ai-in-production/)). A recurring framing in policy discussions of agentic AI reliability captures the core engineering complaint directly: when an AI system goes wrong, there's no stack trace you can examine ([The Hill, 2026](https://thehill.com/opinion/technology/5950143-ai-reliability-national-security/)).

The reason is structural, not incidental. Agentic behavior is non-deterministic by construction — the same input can produce a different execution path each time, so you can't snapshot a failure and replay it the way you'd replay a deterministic function call ([Machine Learning Mastery, 2026](https://machinelearningmastery.com/5-production-scaling-challenges-for-agentic-ai-in-2026/)). Durable-execution vendors like Temporal describe this as agentic AI entering a "rebuild era," where teams that shipped a working demo discover that production requires durable execution, state management, workflow visibility, and recovery mechanisms that their original prototype never needed ([VentureBeat, 2026](https://venturebeat.com/orchestration/ai-agents-are-entering-their-rebuild-era-as-enterprises-confront-the-reliability-problem)).

Three consequences follow, and Waypoint addresses the first two while leaving the third to surrounding infrastructure:

1. **Restart cost.** Without checkpointing, a failure at step 5 means re-running steps 1–4 — and re-paying for any LLM or API work that already succeeded. At a few thousand runs per day, mid-pipeline crashes become real money.
2. **No failure memory.** Without a searchable failure history, every incident starts from scratch, even if a structurally identical failure occurred last week under a different run ID. Agent errors are often LLM-generated prose, so exact `grep` misses near-duplicates.
3. **No visibility into divergence.** When two runs of the "same" pipeline disagree, teams rarely have intermediate state in a form that supports step-by-step comparison.

The industry response so far has been either (a) adopt a full durable-execution platform (Temporal, Restate, Inngest) and accept the operational surface, or (b) bolt a key-value checkpointer onto LangGraph and accept that failed runs leave no searchable trail. Waypoint occupies the gap: a single-process checkpoint log that is also a semantic failure index, with no external service beyond Qdrant itself.

## Why Checkpoint and Embed in the Same Write

The obvious fix for restart cost is a checkpoint log — write state after every step, key it by run ID and step index, resume from the last successful key. This is a solved problem; task queues and workflow engines have done it for a decade.

The less obvious opportunity is that a checkpoint log and a failure index have almost identical write patterns. Every checkpoint already has a natural-language description of what happened: step, status, and — on failure — error. Embedding that description in the same write yields semantic search over failures without a second pipeline or store.

```
save_checkpoint(run_id, step_index, step_name, status, state, error)
        │
        ├──► payload  → ordered, filterable checkpoint log   (resume from here)
        └──► vector   → semantic index over failures         (search from here)
```

Qdrant fits because one store does both jobs: `payload` for exact-match, filterable, sortable checkpoint access; `vector` for ANN failure search; one client, no separate indexing pipeline. Local mode runs embedded in-process, clearing the "try this in one line" bar. A single `upsert` also keeps checkpoint persistence and failure indexing atomic at the point level — not distributed consensus, but enough to keep debugging coherent with resume.

Formally, each checkpoint $c$ is a pair $(p, v)$ where $p$ is the payload and $v \in \mathbb{R}^{d}$ embeds summary $s(c)$:

$$
s(c) = \text{step\_name} \;\|\; \text{status} \;\|\; \text{error\_message}, \qquad
v = \mathrm{embed}\big(s(c)\big)
$$

Successful steps are still embedded; search defaults to `status="failed"`. Keeping every status indexed means you can later search pauses without a schema migration.

## Package Layout and Design Overview

Waypoint's original architecture — a `Workflow` engine, a `CheckpointStore` over Qdrant, and an `Embedder` protocol — is correct; what it lacked was a settled module boundary and a diffing utility split out from the engine. The concrete package:

```
waypoint/
  __init__.py         # public re-exports: Workflow, PauseWorkflow, RunHandle
  workflow.py         # Workflow, @wf.step decorator, StepStatus, resume()
  checkpoint.py       # CheckpointStore (Qdrant-backed, deterministic uuid5 IDs)
  embedder.py         # Embedder protocol, HashingEmbedder, FastEmbedEmbedder
  diffing.py          # diff_runs(), localize_state_diff()
```

Control flow for one run:

```
┌──────────────────┐     for each step i      ┌──────────────────────────┐
│  Workflow.run    │ ───────────────────────► │ step_fn(state) → state   │
│  ordered steps   │      (workflow.py)       │ raises FAILED / PAUSED   │
└──────────────────┘                          └────────────┬─────────────┘
                                                           │
                                              ┌────────────▼─────────────┐
                                              │ CheckpointStore.save_…   │
                                              │ uuid5(run_id,i)+embed()  │
                                              │      (checkpoint.py)     │
                                              └────────────┬─────────────┘
                                            ┌──────────────┴──────────────┐
                                       Qdrant payload              Qdrant vector
                                       (resume / diffing.py)     (failure search)
```

A `RunHandle` is returned to the caller and can also be reconstructed via `Workflow.get_run(run_id)` with no re-execution.

## Installing and Using Waypoint

The package targets Python 3.11+ and depends on `qdrant-client>=1.18` for the current `query_points` API. `Embedder` is a two-method protocol, so `HashingEmbedder` ships with zero external dependencies and real embedding models are opt-in extras.

```bash
pip install waypoint-agents
# optional: FastEmbed-backed embedder for better failure clustering
pip install "waypoint-agents[fastembed]"
```

A complete pipeline in a few lines:

```python
from waypoint import Workflow, PauseWorkflow
from waypoint.checkpoint import CheckpointStore
from waypoint.embedder import HashingEmbedder

store = CheckpointStore(
    collection_name="waypoint_invoice_processor",
    embedder=HashingEmbedder(dim=384),
    path="./waypoint_invoice_data",  # or client=QdrantClient(url="http://localhost:6333")
)
wf = Workflow("invoice_processor", store=store)

@wf.step("fetch_invoice")
def fetch_invoice(state: dict) -> dict:
    state["invoice"] = load_invoice(state["invoice_id"])
    return state

@wf.step("validate_total")
def validate_total(state: dict) -> dict:
    if state["invoice"].get("total") is None:
        raise ValueError("missing total field on invoice")
    return state

handle = wf.run({"invoice_id": "INV-2"})
if handle.failed:
    for match in wf.search_similar_failures(handle.error):
        print(match["run_id"], match["score"], match["error"])
    wf.resume(handle.run_id)  # retries validate_total once the input is fixed
```

Everything past this point is the internals behind that block: how `Workflow`, `CheckpointStore`, and `Embedder` are actually implemented, module by module.

## workflow.py: The Workflow Engine

A `Workflow` is an ordered list of plain functions, each taking and returning a state dict — the same shape as a LangGraph node or Temporal activity, so it slots into existing pipelines rather than replacing them. State is a mutable `dict[str, Any]`; steps may mutate in place and must return the dict. Keys are opaque to Waypoint. Steps should be idempotent enough to retry safely, or gate on keys already present (e.g. skip an LLM call if `state["summary"]` exists) — same caller contract as Temporal activities.

```python
from __future__ import annotations
import traceback, uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
from waypoint.checkpoint import CheckpointStore
from waypoint.embedder import Embedder, HashingEmbedder

class StepStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PAUSED = "paused"

class PauseWorkflow(Exception):
    def __init__(self, state: dict, reason: str = ""):
        self.state, self.reason = state, reason
        super().__init__(reason or "workflow paused")

@dataclass
class RunHandle:
    run_id: str
    workflow_name: str
    status: StepStatus
    step_index: int
    step_name: str
    state: dict
    error: Optional[str] = None
    failed: bool = False
    paused: bool = False

StepFn = Callable[[dict], dict]

class Workflow:
    def __init__(self, name: str, store: Optional[CheckpointStore] = None,
                 embedder: Optional[Embedder] = None):
        self.name = name
        self.steps: list[tuple[str, StepFn]] = []
        self.embedder = embedder or HashingEmbedder(dim=384)
        self.store = store or CheckpointStore(
            collection_name=f"waypoint_{name}", embedder=self.embedder)

    def step(self, name: str):
        def decorator(fn: StepFn) -> StepFn:
            self.steps.append((name, fn))
            return fn
        return decorator

    def _ckpt(self, run_id, i, step_name, status, state, error=None, tb=None):
        self.store.save_checkpoint(
            run_id=run_id, workflow_name=self.name, step_index=i,
            step_name=step_name, status=status, state=state,
            error=error, traceback_str=tb)

    def run(self, state: dict, *, run_id: Optional[str] = None,
            start_index: int = 0, raise_on_error: bool = False) -> RunHandle:
        run_id = run_id or str(uuid.uuid4())
        current = dict(state)
        for i in range(start_index, len(self.steps)):
            step_name, step_fn = self.steps[i]
            try:
                current = step_fn(current)
                self._ckpt(run_id, i, step_name, StepStatus.SUCCESS, current)
            except PauseWorkflow as e:
                current = e.state
                self._ckpt(run_id, i, step_name, StepStatus.PAUSED, current, e.reason)
                return RunHandle(run_id, self.name, StepStatus.PAUSED, i, step_name,
                                 current, error=e.reason, paused=True)
            except Exception as e:
                self._ckpt(run_id, i, step_name, StepStatus.FAILED, current,
                           str(e), traceback.format_exc())
                handle = RunHandle(run_id, self.name, StepStatus.FAILED, i, step_name,
                                   current, error=str(e), failed=True)
                if raise_on_error:
                    raise
                return handle
        last = self.steps[-1][0] if self.steps else ""
        return RunHandle(run_id, self.name, StepStatus.SUCCESS,
                         len(self.steps) - 1, last, current)
```

`run` checkpoints after each step. On raise, state *as of that step* is written as `failed` and a `RunHandle` returns — no exception unless `raise_on_error=True`. Failed steps are expected first-class outcomes in agent pipelines; forcing every caller into `try/except` treats the common case as rare. The engine never talks to Qdrant directly — `checkpoint.py` owns that boundary entirely.

## checkpoint.py: The Checkpoint Store

`CheckpointStore` wraps `qdrant_client.QdrantClient`. The two behaviors that matter most are deterministic point IDs and a filterable payload schema.

Each checkpoint ID is derived from `(run_id, step_index)` via `uuid5`, not a fresh `uuid4`:

```python
import uuid

def point_id(run_id: str, step_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"waypoint:{run_id}:{step_index}"))
```

Re-executing step 3 overwrites the same point — idempotent resume without duplicate history. Waypoint keeps the *latest* checkpoint per `(run_id, step_index)`. For forensic attempt history, fold an attempt counter into the uuid5 key; the default optimizes for resume correctness.

Full payload schema and store implementation, including collection creation and payload indexes (every filter/sort field must be indexed or scrolls and filtered ANN degrade to full scans):

```python
from __future__ import annotations
from typing import Any, Optional
import time
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PayloadSchemaType, PointStruct, Filter, FieldCondition, MatchValue,
)
from waypoint.embedder import Embedder
from waypoint.workflow import StepStatus

# Payload fields per point:
#   run_id, workflow_name, step_index, step_name, status,
#   state (JSON-serializable dict), error, traceback, created_at, summary

class CheckpointStore:
    def __init__(self, collection_name: str, embedder: Embedder,
                 client: Optional[QdrantClient] = None, path: str = ":memory:"):
        self.collection_name, self.embedder = collection_name, embedder
        self.client = client or QdrantClient(path=path)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in names:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )
        for f, s in [("run_id", PayloadSchemaType.KEYWORD),
                     ("workflow_name", PayloadSchemaType.KEYWORD),
                     ("step_index", PayloadSchemaType.INTEGER),
                     ("step_name", PayloadSchemaType.KEYWORD),
                     ("status", PayloadSchemaType.KEYWORD),
                     ("created_at", PayloadSchemaType.INTEGER)]:
            self.client.create_payload_index(
                collection_name=self.collection_name, field_name=f, field_schema=s)

    def save_checkpoint(self, *, run_id: str, workflow_name: str, step_index: int,
                        step_name: str, status: StepStatus, state: dict,
                        error: Optional[str] = None,
                        traceback_str: Optional[str] = None) -> str:
        summary = f"{step_name} | {status.value} | {error or 'ok'}"
        pid = point_id(run_id, step_index)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=pid,
                vector=self.embedder.embed(summary),
                payload={
                    "run_id": run_id, "workflow_name": workflow_name,
                    "step_index": step_index, "step_name": step_name,
                    "status": status.value, "state": state,
                    "error": error, "traceback": traceback_str,
                    "created_at": int(time.time()), "summary": summary,
                },
            )],
        )
        return pid

    def get_checkpoints(self, run_id: str) -> list[dict]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=[
                FieldCondition(key="run_id", match=MatchValue(value=run_id))]),
            with_payload=True, with_vectors=False, limit=10_000,
        )
        return sorted([p.payload for p in points], key=lambda r: r["step_index"])

    def search_failures(self, query_vector: list[float], *,
                        workflow_name: Optional[str] = None,
                        limit: int = 10, score_threshold: float = 0.55) -> list[dict]:
        must = [FieldCondition(key="status", match=MatchValue(value="failed"))]
        if workflow_name:
            must.append(FieldCondition(
                key="workflow_name", match=MatchValue(value=workflow_name)))
        hits = self.client.query_points(
            collection_name=self.collection_name, query=query_vector,
            query_filter=Filter(must=must), limit=limit,
            score_threshold=score_threshold, with_payload=True,
        ).points
        return [{
            "run_id": h.payload["run_id"], "step_name": h.payload["step_name"],
            "step_index": h.payload["step_index"], "error": h.payload.get("error"),
            "score": h.score, "summary": h.payload.get("summary"),
        } for h in hits]
```

`search_failures` is the one place the earlier design used the deprecated `.search(query_vector=...)` method. It's replaced here with `client.query_points(collection_name=..., query=query_vector, query_filter=Filter(...), limit=...)`, reading `.points` off the response — the collection here uses a single unnamed vector, so `query_points` takes `query` directly without a `using=` argument. `qdrant-client>=1.18` is required for this signature.

Store large blobs elsewhere and keep pointers in `state`. Persist `summary` so you can debug search quality without re-embedding. Write `traceback` only on `FAILED`. `get_checkpoints` then reconstructs a `RunHandle` with no re-execution.

## embedder.py: The Embedder Protocol

The `Embedder` is a two-method protocol — `embed(text) -> list[float]` and `dim` — so the semantic layer stays swappable.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    def embed(self, text: str) -> list[float]: ...
```

### HashingEmbedder Mathematics

The default `HashingEmbedder` is a deterministic offline vectorizer: tokenize into words and character trigrams, hash each token into one of `dim` buckets with SHA-256, accumulate a signed count per bucket, L2-normalize.

Let $T(x)$ be tokens from text $x$ (lowercased words plus overlapping character trigrams). For each $t \in T(x)$, let $h(t)$ be the SHA-256 digest:

$$
i(t) = \big(\mathrm{int}_{64}(h(t)[0:8])\big) \bmod d, \qquad
\mathrm{sign}(t) =
\begin{cases}
+1 & \text{if } h(t)[8] \bmod 2 = 0 \\
-1 & \text{otherwise}
\end{cases}
$$

$$
\tilde{v}_{i} = \sum_{t \in T(x)\,:\, i(t)=i} \mathrm{sign}(t), \qquad
\mathbf{v} = \frac{\tilde{\mathbf{v}}}{\lVert \tilde{\mathbf{v}} \rVert_{2} + \varepsilon}
$$

Equivalently, $v_i = \frac{1}{\lVert \mathbf{v} \rVert_2} \sum_{t \,:\, h(t) \bmod d = i} \text{sign}(t)$ — the classical hashing trick / signed feature map (Weinberger et al., 2009). No model download, network, or GPU. Cosine similarity is high when texts share vocabulary (error type, step name): the right signal for "have I seen this failure before?" It will not cluster paraphrases that share meaning but not tokens.

```python
import hashlib, math, re

_WORD = re.compile(r"[a-z0-9_]+", re.I)

class HashingEmbedder:
    def __init__(self, dim: int = 384, use_trigrams: bool = True):
        self._dim, self.use_trigrams = dim, use_trigrams

    @property
    def dim(self) -> int:
        return self._dim

    def _tokens(self, text: str) -> list[str]:
        words = _WORD.findall(text.lower())
        tokens = list(words)
        if self.use_trigrams:
            c = "".join(words)
            tokens.extend(c[i:i+3] for i in range(max(0, len(c) - 2)))
        return tokens

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in self._tokens(text):
            digest = hashlib.sha256(tok.encode()).digest()
            bucket = int.from_bytes(digest[:8], "big") % self._dim
            vec[bucket] += 1.0 if digest[8] % 2 == 0 else -1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1e-12
        return [x / norm for x in vec]
```

### FastEmbed Swap

```python
from fastembed import TextEmbedding

class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model = TextEmbedding(model_name=model_name)
        self._dim = 384  # keep collection dim in sync

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        return list(self._model.embed([text]))[0].tolist()

wf = Workflow("invoice_processor", embedder=FastEmbedEmbedder())
```

Vector size is fixed at creation — a 768-dim model needs a new collection or named-vector migration (see Challenges on model drift).

## Resume Semantics: Retry vs Advance

The subtlety is what "resume" means when the latest checkpoint is not a success:

| Latest status | Resume starts at | Rationale |
|---|---|---|
| `SUCCESS` | `step_index + 1` | Completed; advance. |
| `FAILED` | `step_index` | Did not complete; retry. |
| `PAUSED` | `step_index` | Gate unsatisfied; retry gate. |

An earlier version treated `PAUSED` like `SUCCESS` and advanced past the gate — wrong. A pause means the step could not proceed yet; resume must re-run that gate once the external condition changes. The fix is one line (`retry_statuses = (FAILED, PAUSED)`), invisible until a test exercises pause → mutate state → resume → assert the gate re-ran.

```python
RETRY_STATUSES = {StepStatus.FAILED.value, StepStatus.PAUSED.value}

class Workflow:
    def resume(self, run_id: str, *, state_updates: Optional[dict] = None,
               raise_on_error: bool = False) -> RunHandle:
        checkpoints = self.store.get_checkpoints(run_id)
        if not checkpoints:
            raise KeyError(f"no checkpoints for run_id={run_id}")
        latest = checkpoints[-1]
        state = dict(latest["state"])
        if state_updates:
            state.update(state_updates)
        start = (latest["step_index"] if latest["status"] in RETRY_STATUSES
                 else latest["step_index"] + 1)
        if start >= len(self.steps):
            return RunHandle(run_id, self.name, StepStatus.SUCCESS,
                             latest["step_index"], latest["step_name"], state)
        return self.run(state, run_id=run_id, start_index=start,
                        raise_on_error=raise_on_error)

    def get_run(self, run_id: str) -> RunHandle:
        latest = self.store.get_checkpoints(run_id)[-1]
        status = StepStatus(latest["status"])
        return RunHandle(
            run_id, self.name, status, latest["step_index"], latest["step_name"],
            latest["state"], error=latest.get("error"),
            failed=status == StepStatus.FAILED, paused=status == StepStatus.PAUSED)
```

Edge cases to test: resume after full success is a no-op; `state_updates` merge *before* the retried step (HITL injection); retry upserts the same uuid5 point.

```python
def test_failed_resume_retries_same_step():
    wf = Workflow("t", store=CheckpointStore("t", HashingEmbedder(), path=":memory:"))
    calls = {"n": 0}
    @wf.step("flaky")
    def flaky(state):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        state["ok"] = True
        return state
    h1 = wf.run({})
    assert h1.failed and h1.step_index == 0
    assert not wf.resume(h1.run_id).failed and calls["n"] == 2

def test_paused_resume_does_not_advance():
    wf = Workflow("t2", store=CheckpointStore("t2", HashingEmbedder(), path=":memory:"))
    seen = []
    @wf.step("gate")
    def gate(state):
        seen.append("gate")
        if not state.get("approved"):
            raise PauseWorkflow(state, reason="need approval")
        return state
    @wf.step("after")
    def after(state):
        seen.append("after"); return state
    h1 = wf.run({})
    assert h1.paused
    wf.resume(h1.run_id, state_updates={"approved": True})
    assert seen == ["gate", "gate", "after"]
```

## Semantic Search Over Failures

`search_similar_failures` (on `Workflow`, in `workflow.py`) embeds the query and delegates to `CheckpointStore.search_failures`, defaulting to `status="failed"`:

```python
class Workflow:
    def search_similar_failures(self, text: str, *, limit: int = 10,
                                score_threshold: float = 0.55) -> list[dict]:
        return self.store.search_failures(
            self.embedder.embed(text), workflow_name=self.name,
            limit=limit, score_threshold=score_threshold)

run = wf.run({"invoice_id": "INV-2"})
for match in wf.search_similar_failures(run.error):
    print(match["run_id"], match["score"], match["error"])
```

HNSW retrieves by cosine similarity of the summary embedding via `query_points`; the payload filter keeps failures (and optionally `workflow_name`). With `HashingEmbedder`, thresholds around $0.55$–$0.65$ work for shared vocabulary; with FastEmbed / BGE prefer $0.75$–$0.85$.

This turns incidents from "grep exact strings" into "ask whether this failure shape has a history." Value scales with team size — one developer debugging their own pipeline gets less than a team running dozens of workflows where today's on-call is not last week's. It helps most when errors are natural-language prose and step names overlap; least when every message is a unique UUID-laden string.

## diffing.py: Time-Travel Debugging via Run Diffs

`diff_runs` compares two runs' checkpoints step-by-step and reports whether state and status matched at each index — a storage-only replay debugger without re-execution. It lives in its own module because, unlike everything else in Waypoint, it never touches Qdrant directly; it operates purely on `CheckpointStore.get_checkpoints` output.

```python
from __future__ import annotations
from typing import Any, Iterator
from waypoint.checkpoint import CheckpointStore

def diff_runs(store: CheckpointStore, run_id_a: str, run_id_b: str) -> list[dict]:
    a = {c["step_index"]: c for c in store.get_checkpoints(run_id_a)}
    b = {c["step_index"]: c for c in store.get_checkpoints(run_id_b)}
    diffs = []
    for i in sorted(set(a) | set(b)):
        ca, cb = a.get(i), b.get(i)
        if ca is None or cb is None:
            diffs.append({"step_index": i, "equal": False,
                          "reason": "missing_checkpoint", "a": ca, "b": cb})
            continue
        equal = ca["state"] == cb["state"] and ca["status"] == cb["status"]
        diffs.append({
            "step_index": i,
            "step_name": ca.get("step_name") or cb.get("step_name"),
            "equal": equal, "status_a": ca["status"], "status_b": cb["status"],
            "state_a": ca["state"], "state_b": cb["state"],
        })
    return diffs
```

`Workflow.diff_runs(run_id_a, run_id_b)` is a thin convenience wrapper that passes `self.store`. Shallow equality answers "where did these runs start disagreeing?" For nested state, localize *which key* with a recursive path walk:

```python
def iter_paths(obj: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])):
            yield from iter_paths(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_paths(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj

def localize_state_diff(a: dict, b: dict, *, max_paths: int = 20) -> list[dict]:
    ma, mb = dict(iter_paths(a)), dict(iter_paths(b))
    out = []
    for path in sorted(set(ma) | set(mb)):
        if path not in ma:
            out.append({"path": path, "kind": "missing_in_a", "b": mb[path]})
        elif path not in mb:
            out.append({"path": path, "kind": "missing_in_b", "a": ma[path]})
        elif ma[path] != mb[path]:
            out.append({"path": path, "kind": "changed", "a": ma[path], "b": mb[path]})
        if len(out) >= max_paths:
            break
    return out

def diff_runs_localized(store: CheckpointStore, run_id_a: str, run_id_b: str) -> list[dict]:
    diffs = diff_runs(store, run_id_a, run_id_b)
    for d in diffs:
        if not d["equal"] and d.get("state_a") is not None and d.get("state_b") is not None:
            d["changed_paths"] = localize_state_diff(d["state_a"], d["state_b"])
    return diffs
```

Treat localization as an extension: path walks get noisy with full chat transcripts. Shallow-diff first, then localize only the first divergent step.

## Human-in-the-Loop as a First-Class State

Approval gates belong in agent pipelines. Modeling them as `PauseWorkflow` rather than a boolean threaded through every downstream step keeps the common path clean:

```python
@wf.step("needs_approval")
def needs_approval(state):
    if not state.get("approved"):
        raise PauseWorkflow(state, reason="waiting for manager approval")
    return state
```

Caught on the same path as errors, checkpointed as `paused`, resumed with `wf.resume(run_id)` — status label differs, and both `FAILED` and `PAUSED` retry rather than advance.

Three interrupt patterns recur: **approval gates** (`approved=True` via `state_updates`); **missing input** (form/Slack fills fields); **policy holds**:

```python
@wf.step("policy_check")
def policy_check(state):
    if state.get("risk_score", 0) >= 0.8 and not state.get("policy_override"):
        raise PauseWorkflow(state, reason="high risk — compliance review required")
    return state
```

LangGraph's `interrupt()` has the same shape: refuse to proceed, persist, resume with a value. Mapping: `interrupt({"reason": ...})` ↔ `raise PauseWorkflow(...)`; `Command(resume=...)` ↔ `wf.resume(run_id, state_updates=...)`. Storage differs — LangGraph checkpointers are typically key-value with no semantic index. Waypoint pauses are also payload-filterable (`status="paused"`) for "invoices waiting on compliance."

## Integrating with LangGraph

Both systems share the state-dict contract, so a LangGraph node can wrap a Waypoint workflow:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class LGState(TypedDict, total=False):
    invoice_id: str
    invoice: dict
    approved: bool
    waypoint_run_id: str
    error: str

def make_waypoint_node(wf: Workflow, *, run_id_key: str = "waypoint_run_id"):
    def node(state: LGState) -> LGState:
        run_id = state.get(run_id_key)
        handle = (wf.resume(run_id, state_updates=dict(state))
                  if run_id else wf.run(dict(state)))
        update: LGState = dict(handle.state)  # type: ignore
        update[run_id_key] = handle.run_id
        if handle.failed or handle.paused:
            update["error"] = handle.error or handle.status.value
        return update
    return node

graph = StateGraph(LGState)
graph.add_node("invoice_pipeline", make_waypoint_node(build_invoice_workflow()))
graph.set_entry_point("invoice_pipeline")
graph.add_edge("invoice_pipeline", END)
app = graph.compile()
```

Whole-workflow wrapping keeps Waypoint resume authoritative. Per-step wrapping fits when LangGraph owns branching and you only want the Qdrant failure index as a side effect.

## Worked Example: Invoice Pipeline Walkthrough

A complete invoice pipeline — fetch, validate, optional approval, extract line items, post payment — focusing on checkpoint/resume under failure and pause:

```python
store = CheckpointStore(
    collection_name="waypoint_invoice_processor",
    embedder=HashingEmbedder(dim=384), path="./waypoint_invoice_data")
wf = Workflow("invoice_processor", store=store)

def load_invoice(invoice_id: str) -> dict:
    return {
        "INV-1": {"id": "INV-1", "total": 1200.0, "vendor": "acme"},
        "INV-2": {"id": "INV-2", "total": None, "vendor": "acme"},
        "INV-3": {"id": "INV-3", "total": 50_000.0, "vendor": "globex"},
    }[invoice_id]

@wf.step("fetch_invoice")
def fetch_invoice(state):
    state["invoice"] = load_invoice(state["invoice_id"])
    return state

@wf.step("validate_total")
def validate_total(state):
    if state["invoice"].get("total") is None:
        raise ValueError("missing total field on invoice")
    return state

@wf.step("approval_gate")
def approval_gate(state):
    if state["invoice"]["total"] >= 10_000 and not state.get("approved"):
        raise PauseWorkflow(state, reason="waiting for manager approval")
    return state

@wf.step("extract_line_items")
def extract_line_items(state):
    state["line_items"] = [{"sku": "WIDGET", "amount": state["invoice"]["total"]}]
    return state

@wf.step("post_payment")
def post_payment(state):
    state["payment_id"] = f"PAY-{state['invoice']['id']}"
    return state
```

**Happy path.** `wf.run({"invoice_id": "INV-1"})` → five success checkpoints → `payment_id="PAY-INV-1"`.

**Validation failure.** `INV-2` fails at validate. `search_similar_failures("missing total field")` retrieves it. After ERP repair, `resume` retries `validate_total` (not fetch) because latest status is `FAILED`.

**Human approval.** `INV-3` pauses at `approval_gate`. After review: `wf.resume(run_id, state_updates={"approved": True, "approver": "alice"})` → `PAY-INV-3`.

**Time-travel.** `diff_runs(store, good_id, bad_id)` shows fetch payloads differ and validate status diverges; `diff_runs_localized` narrows to the changed key if nested fields matter. That is the loop: run → fail/pause → search → diff → patch → resume.

## Related Work

Waypoint sits at the intersection of two established ideas rather than inventing either from scratch.

**Durable execution engines.** **Temporal** is the most prominent in the agentic-AI context: distributed checkpoint-and-resume with strong exactly-once activity guarantees ([VentureBeat, 2026](https://venturebeat.com/orchestration/ai-agents-are-entering-their-rebuild-era-as-enterprises-confront-the-reliability-problem)). Temporal records event histories and replays workflows deterministically. Waypoint is narrower: single-process, no worker fleets, no service beyond Qdrant — trading Temporal's generality for near-zero setup and a semantic failure index Temporal does not provide out of the box. Multi-worker failover → Temporal / Restate / Inngest. Checkpoint agent steps and find similar past failures in one store → Waypoint's niche.

**Agent orchestration checkpointers.** **LangGraph** checkpointers (`MemorySaver`, `SqliteSaver`, `PostgresSaver`) persist graph state and support `interrupt()` for HITL. Those backends are key-value: excellent for resume, silent on semantic search. Waypoint's failure-search layer is additive — dual-writing is redundant for resume but useful when you want LangGraph graph semantics plus Qdrant's filtered ANN. Logging platforms (LangSmith, Helicone, OpenTelemetry) answer "what happened in run X?" but do not give resume-from-step-3 or deterministic overwrite of `(run_id, step_index)`. Waypoint is persistence for control flow; it composes with tracing rather than replacing it.

## Challenges and Open Problems

**Embedding quality.** `HashingEmbedder` clusters by shared vocabulary, not shared cause. Same root cause with different wording may not score similarly. Deliberate zero-dependency default; production should supply a real model via `embedder.Embedder`.

**Local mode is single-writer.** `:memory:` and on-disk `path=` use an embedded instance with a file lock. Fine for one app process plus a CLI reader; concurrent multi-worker writers need a real Qdrant server via `client=QdrantClient(url=...)`.

**No automatic retry policy.** Resume is always explicit — intentional scope-narrowing. Compose with an external scheduler that calls `resume()` on a cadence.

**Diffing depth vs. noise.** Shallow diffs find *that* a step diverged; `localize_state_diff` helps but drowns you when state carries full chat transcripts. Ignoring high-churn keys is application-specific and not yet configurable in `diffing.py`.

**State serialization and schema evolution.** Payloads must be JSON-serializable. Renaming a state key breaks clean diffs across runs that straddle the rename; there is no migration helper.

**Embedding model drift.** Changing embedders invalidates cosine comparability. Options: re-embed-all, versioned collections, dual-read during migration — none automated today.

**Exactly-once side effects and multi-tenancy.** Retrying a step that already charged a payment API is unsafe unless the step is idempotent — Waypoint records state but cannot make external systems idempotent. Shared collections need `tenant_id` in every filter and careful point-ID namespacing, or failure summaries can leak across tenants.

## References

```bibtex
@article{inamdar2026waypoint,
  title   = {Waypoint, For Real: A Working Checkpoint-and-Resume Library for Agent Pipelines},
  author  = {Inamdar, Mihir},
  year    = {2026},
  note    = {Personal blog},
}
```

[1] Help Net Security. ["Most agentic AI projects in production have stalled over data problems."](https://www.helpnetsecurity.com/2026/06/18/report-agentic-ai-in-production/) 2026.
[2] The Hill. ["To unlock agentic AI's promise for government, America must build reliability."](https://thehill.com/opinion/technology/5950143-ai-reliability-national-security/) 2026.
[3] Machine Learning Mastery. ["5 Production Scaling Challenges for Agentic AI in 2026."](https://machinelearningmastery.com/5-production-scaling-challenges-for-agentic-ai-in-2026/) 2026.
[4] VentureBeat. ["AI agents are entering their rebuild era as enterprises confront the reliability problem."](https://venturebeat.com/orchestration/ai-agents-are-entering-their-rebuild-era-as-enterprises-confront-the-reliability-problem) 2026.
[5] Qdrant Documentation. *Payload Filtering*. [qdrant.tech](https://qdrant.tech/documentation/concepts/filtering/)
[6] Qdrant Documentation. *Payload Indexes*. [qdrant.tech](https://qdrant.tech/documentation/concepts/indexing/#payload-index)
[7] Qdrant Documentation. *Query API — `query_points`*. [qdrant.tech](https://qdrant.tech/documentation/concepts/search/)
[8] LangGraph Documentation. *Persistence / Checkpointers*. [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/concepts/persistence/)
[9] LangGraph Documentation. *Interrupts (Human-in-the-Loop)*. [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
[10] Temporal Documentation. *What is Temporal?* [docs.temporal.io](https://docs.temporal.io/temporal)
[11] Weinberger, Kilian et al. (2009). *Feature Hashing for Large Scale Multitask Learning*. ICML 2009 / PMLR.
