---
title: "Durable Agent Workflows: Checkpointing and Semantic Failure Search with Qdrant"
date: 2026-07-27
description: "How Waypoint uses Qdrant as both a checkpoint log and a semantic failure index to make agent and LLM pipelines durable and debuggable — covering resume semantics, time-travel diffs, and human-in-the-loop pauses."
tags: ["agents", "qdrant", "checkpointing", "durable-execution", "debugging", "workflows"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Agentic systems fail differently than the software we're used to building. A REST endpoint either returns 200 or it doesn't, and when it doesn't, a stack trace tells you exactly where. An agent pipeline that chains an LLM call, a tool call, a retrieval step, and another LLM call can fail at step 5 of 7 after ninety seconds of real work, and the only artifact left behind is usually a log line and a shrug. This post walks through **Waypoint**, a small library I built to make agent and LLM pipelines durable and debuggable, and the design decisions behind using Qdrant as both the checkpoint log and the failure index. I'll assume basic familiarity with vector databases and LLM agent frameworks (LangChain / LangGraph-style step orchestration), but no prior exposure to durable execution systems is needed.

This post only covers the checkpoint-and-resume problem and the semantic-search-over-failures problem. It does not cover distributed multi-worker execution, automatic retry/backoff policies, or cost-aware model routing — those are related but separate problems I consider out of scope here.

## Table of Contents

- [The Problem: Non-Determinism Without a Stack Trace](#the-problem-non-determinism-without-a-stack-trace)
- [Why Checkpoint *and* Embed in the Same Write](#why-checkpoint-and-embed-in-the-same-write)
- [System Design](#system-design)
  - [Component One: The Workflow Engine](#component-one-the-workflow-engine)
  - [Component Two: The Checkpoint Store](#component-two-the-checkpoint-store)
  - [Component Three: The Embedder](#component-three-the-embedder)
- [Resume Semantics: Retry vs. Advance](#resume-semantics-retry-vs-advance)
- [Semantic Search Over Failures](#semantic-search-over-failures)
- [Time-Travel Debugging via Run Diffs](#time-travel-debugging-via-run-diffs)
- [Human-in-the-Loop as a First-Class State](#human-in-the-loop-as-a-first-class-state)
- [Related Work](#related-work)
- [Challenges and Open Problems](#challenges-and-open-problems)
- [Citation](#citation)

## The Problem: Non-Determinism Without a Stack Trace

Enterprises adopting agentic AI in 2026 report a consistent set of blockers, and they aren't the ones most demos optimize for. In Confluent's 2026 Data Streaming Report, surveying over 4,600 IT leaders, concerns about LLM reliability and non-determinism sit at 68%, and among organizations that have gotten as far as running agents in production, 77% report stalled projects tied to exactly these reliability problems ([Help Net Security, 2026](https://www.helpnetsecurity.com/2026/06/18/report-agentic-ai-in-production/)). A recurring framing in policy discussions of agentic AI reliability captures the core engineering complaint directly: when an AI system goes wrong, there's no stack trace you can examine ([The Hill, 2026](https://thehill.com/opinion/technology/5950143-ai-reliability-national-security/)).

The reason is structural, not incidental. Agentic behavior is non-deterministic by construction — the same input can produce a different execution path each time, so you can't snapshot a failure and replay it the way you'd replay a deterministic function call ([Machine Learning Mastery, 2026](https://machinelearningmastery.com/5-production-scaling-challenges-for-agentic-ai-in-2026/)). Durable-execution vendors like Temporal describe this as agentic AI entering a "rebuild era," where teams that shipped a working demo discover that production requires durable execution, state management, workflow visibility, and recovery mechanisms that their original prototype never needed ([VentureBeat, 2026](https://venturebeat.com/orchestration/ai-agents-are-entering-their-rebuild-era-as-enterprises-confront-the-reliability-problem)).

Two consequences follow directly from this, and Waypoint is built to address both:

1. **Restart cost.** Without checkpointing, a failure at step 5 means re-running steps 1 through 4 — which, if any of those steps call an LLM or an external API, means re-paying for work that already succeeded.
2. **No failure memory.** Without a searchable failure history, every incident is investigated from scratch, even if a structurally identical failure occurred last week under a different run ID.

## Why Checkpoint *and* Embed in the Same Write

The obvious fix for restart cost is a checkpoint log — write the state after every step, key it by run ID and step index, and resume from the last successful key. This is a solved problem; it's exactly what task queues and workflow engines have done for a decade.

The less obvious opportunity is that a checkpoint log and a failure index have almost identical write patterns. Every checkpoint write already has, as a side effect, a natural language description of what happened: which step, what status, and — on failure — what error. If that same write also produces an embedding of that description, you get semantic search over failures for free, without running a second pipeline or standing up a second store.

Qdrant is a natural fit here because it's one store that does both jobs well: `payload` fields give you exact-match, filterable, sortable access for the checkpoint log, and the `vector` field gives you approximate nearest-neighbor search for the failure index, and both are available from a single client with no separate indexing pipeline. In local mode it also runs embedded in-process with no server to stand up, which matters for the "can I try this in one line" bar a library like this needs to clear.

```python
save_checkpoint(run_id, step_index, step_name, status, state, error)
        │
        ├──► payload  → ordered, filterable checkpoint log   (resume from here)
        └──► vector   → semantic index over failures         (search from here)
```

## System Design

Waypoint has three components: a `Workflow` engine that orchestrates step execution, a `CheckpointStore` that wraps the Qdrant client, and an `Embedder` protocol that turns a checkpoint summary into a vector.

### Component One: The Workflow Engine

A `Workflow` is an ordered list of plain functions, each taking and returning a state dictionary — deliberately the same shape as a LangGraph node or a Temporal activity, so it slots into an existing pipeline rather than replacing it.

```python
from waypoint import Workflow

wf = Workflow("invoice_processor")

@wf.step("fetch_invoice")
def fetch_invoice(state):
    state["invoice"] = load_invoice(state["invoice_id"])
    return state

@wf.step("validate_total")
def validate_total(state):
    if state["invoice"].get("total") is None:
        raise ValueError("missing total field on invoice")
    return state
```

`Workflow.run(state)` executes steps in order, calling `CheckpointStore.save_checkpoint` after each one. If a step raises, the exception is caught, the state *as of that step* is checkpointed with `status="failed"`, and a `RunHandle` is returned describing what happened — no exception propagates unless the caller opts in via `raise_on_error=True`. This is a deliberate API choice: a failed step in an agent pipeline is an expected, first-class outcome, not an exceptional one, and forcing every caller into a `try/except` treats the common case as the rare one.

### Component Two: The Checkpoint Store

`CheckpointStore` is a thin wrapper over `qdrant_client.QdrantClient`. Its two most important behaviors are point-ID determinism and payload structure.

Each checkpoint's point ID is derived deterministically from `(run_id, step_index)` via `uuid5`, not a fresh `uuid4` per write:

```python
def _point_id(run_id: str, step_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"waypoint:{run_id}:{step_index}"))
```

This makes checkpointing idempotent: re-executing step 3 of a run — whether because of a retry or a resume — overwrites the same point rather than accumulating duplicate history. The payload carries the full state dict, the step name, status, error message and traceback (on failure), and a timestamp, giving `get_checkpoints(run_id)` everything needed to reconstruct a `RunHandle` purely from storage, with no re-execution (`Workflow.get_run`).

### Component Three: The Embedder

The `Embedder` is a two-method protocol — `embed(text) -> list[float]` and a `dim` property — which keeps the semantic-search layer swappable. The default, `HashingEmbedder`, is a deterministic offline vectorizer: it tokenizes text into words and word-trigrams, hashes each token into one of `dim` buckets with SHA-256, and accumulates a signed count per bucket before L2-normalizing:

$$
v_i = \frac{1}{\lVert \mathbf{v} \rVert_2} \sum_{t \,:\, h(t) \bmod d = i} \text{sign}(t)
$$

where $h$ is SHA-256, $d$ is the embedding dimension, and $\text{sign}(t) \in \{-1, +1\}$ is derived from a separate byte of the same digest to reduce collision bias. This is a bag-of-hashed-features representation — closer to the hashing trick from classical NLP than to a learned embedding model — but it requires no model download, no network call, and no GPU, which means the library installs and runs anywhere `qdrant-client` does. Cosine similarity between two such vectors is high when both texts share vocabulary (error type, step name), which is precisely the signal needed to answer "have I seen a failure like this before?" The `Embedder` protocol makes it a one-line swap to a real model (FastEmbed, a hosted embeddings API) when production search quality matters more than zero-dependency installation.

## Resume Semantics: Retry vs. Advance

The one subtlety worth dwelling on is what "resume" means for a checkpoint that isn't a success. Waypoint distinguishes three checkpoint statuses — `SUCCESS`, `FAILED`, `PAUSED` — and resume behavior depends on which one is latest:

- **`SUCCESS`**: the step completed; resume continues at `step_index + 1`.
- **`FAILED`**: the step did not complete; resume *retries the same step*, at `step_index`, not the next one.
- **`PAUSED`** (see [Human-in-the-Loop](#human-in-the-loop-as-a-first-class-state) below): also did not complete; also retries at `step_index`.

An earlier version of this library treated `PAUSED` like `SUCCESS` and advanced past the gating step on resume — which is wrong. A pause means the step itself decided it couldn't proceed yet; the correct resume behavior is to re-run that same gating logic once the external condition changes, not to skip it. Getting this distinction right is a one-line change (`retry_statuses = (FAILED, PAUSED)`), but it's the kind of one-line change that's invisible until a test exercises the exact sequence: pause, mutate state externally, resume, and assert the gate step actually re-ran.

## Semantic Search Over Failures

`Workflow.search_similar_failures(text)` embeds the query text and issues a filtered vector search against the same collection used for checkpointing, restricting results to `status="failed"` by default:

```python
run = wf.run({"invoice_id": "INV-2"})
# run.failed is True, run.error == "missing total field on invoice"

for match in wf.search_similar_failures(run.error):
    print(match["run_id"], match["score"], match["error"])
```

This turns an incident from "grep the logs for an exact string" into "ask the system whether this shape of failure has a history." The practical value scales with team size and pipeline count — a single developer debugging their own pipeline gets less out of this than a team running dozens of agent workflows where the person hitting a failure today isn't the person who last debugged something similar.

## Time-Travel Debugging via Run Diffs

`diff_runs(run_id_a, run_id_b)` compares two runs' checkpoint histories step-by-step and reports, for each step index, whether the state at that point was equal between the two runs. Pointed at a known-good run and a known-bad run, it identifies the first step where their state diverged — a minimal, storage-only version of the "replay debugger" idea, without needing to actually replay any execution:

```python
diffs = wf.diff_runs(good_run_id, bad_run_id)
first_divergence = next(d for d in diffs if not d["equal"])
```

This is deliberately simple — a dict equality check per step, not a structural diff — but it answers the single most useful debugging question ("where did these two runs start disagreeing?") without any additional instrumentation beyond the checkpoints Waypoint already writes.

## Human-in-the-Loop as a First-Class State

Agent pipelines increasingly need approval gates — a step that shouldn't proceed without a human decision. Modeling this as an exception (`PauseWorkflow`) rather than a boolean flag threaded through every downstream step keeps the common path clean:

```python
from waypoint import PauseWorkflow

@wf.step("needs_approval")
def needs_approval(state):
    if not state.get("approved"):
        raise PauseWorkflow(state, reason="waiting for manager approval")
    return state
```

`PauseWorkflow` is caught by the same exception-handling path as a real error, checkpointed with `status="paused"` instead of `"failed"`, and resumed with the same `wf.resume(run_id)` call — the only difference from a genuine failure is the status label and, as covered above, that both cases retry rather than advance.

## Related Work

Waypoint sits at the intersection of two established ideas rather than inventing either from scratch. **Durable execution engines** — Temporal being the most prominent in the agentic-AI context — solve the checkpoint-and-resume problem generally, for arbitrary distributed workflows, with strong guarantees around exactly-once execution and cross-service state recovery ([VentureBeat, 2026](https://venturebeat.com/orchestration/ai-agents-are-entering-their-rebuild-era-as-enterprises-confront-the-reliability-problem)). Waypoint is deliberately narrower and lighter: single-process, no distributed coordination, and no external service beyond Qdrant itself, trading generality for a near-zero setup cost. **Agent orchestration frameworks** like LangGraph provide their own checkpointing primitives (`checkpointer` backends), but those are typically key-value stores with no semantic layer — the failure-search capability here is additive to, not a replacement for, that layer, and a LangGraph node can wrap a Waypoint-checkpointed step directly since both share the "function of state dict in, state dict out" shape.

## Challenges and Open Problems

**Embedding quality is the clearest limitation.** The default `HashingEmbedder` is a bag-of-hashed-features signal, not a learned semantic representation — it clusters failures by shared vocabulary, not by shared underlying cause. Two failures with the same root cause but very different error message wording will not necessarily score as similar. This is a deliberate default (zero network dependency, deterministic, testable offline) rather than an oversight, but production deployments that care about search recall should supply a real embedding model through the `Embedder` protocol.

**Local mode is single-writer.** Both `:memory:` and on-disk `path=` modes use an embedded Qdrant instance with a file lock; only one process can hold a given local-mode collection open at a time. This is fine for a single application process plus an out-of-process CLI reading after the fact, but it rules out concurrent multi-worker writers to the same collection without switching to a real Qdrant server.

**No automatic retry policy.** Resume is always explicit — Waypoint does not retry a failed step on a schedule or with backoff. This is intentional scope-narrowing (see the opening of this post) rather than a gap to be filled later within this library; it composes naturally with an external scheduler that calls `resume()` on a cadence, but doesn't attempt to be one.

**Diffing is shallow.** `diff_runs` does a dict equality check per step, not a structural or semantic diff of state contents. For state dicts containing large nested structures, this correctly identifies *that* a step diverged but doesn't localize *which key* first differed — a natural extension, not yet implemented.

## Citation

Cited as:

> Inamdar, Mihir. (2026). "Durable Agent Workflows: Checkpointing and Semantic Failure Search with Qdrant." Personal blog.

```bibtex
@article{inamdar2026waypoint,
  title   = {Durable Agent Workflows: Checkpointing and Semantic Failure Search with Qdrant},
  author  = {Inamdar, Mihir},
  year    = {2026},
  note    = {Personal blog},
}
```

### References

[1] Help Net Security. ["Most agentic AI projects in production have stalled over data problems."](https://www.helpnetsecurity.com/2026/06/18/report-agentic-ai-in-production/) 2026.

[2] The Hill. ["To unlock agentic AI's promise for government, America must build reliability."](https://thehill.com/opinion/technology/5950143-ai-reliability-national-security/) 2026.

[3] Machine Learning Mastery. ["5 Production Scaling Challenges for Agentic AI in 2026."](https://machinelearningmastery.com/5-production-scaling-challenges-for-agentic-ai-in-2026/) 2026.

[4] VentureBeat. ["AI agents are entering their rebuild era as enterprises confront the reliability problem."](https://venturebeat.com/orchestration/ai-agents-are-entering-their-rebuild-era-as-enterprises-confront-the-reliability-problem) 2026.
