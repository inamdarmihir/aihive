---
title: "Verifiability-Gated Loops: An Escalation Contract for Agentic Software Factories"
date: 2026-07-28
description: "Most agentic loops only check what they were told to check. This post proposes a verifiability-gated architecture — risk classification, bounded execution, escalation, checkpoint commits, and calibration — so loops fail loudly when stop conditions cannot measure what matters."
tags: ["agents", "verifiability", "human-in-the-loop", "software-engineering", "agent-loops", "reliability"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Coding agents can now run for hours without a human reading a line of what they produce. The natural next question is how far that can go before something breaks that nobody notices until much later. This post is not about prompting technique. It is about a structural gap in how agentic loops are built: most loops only know how to check the thing they were told to check, and nothing tells them when that check is not measuring what actually matters.

I want to narrow the scope considerably. This post does not cover model training, RLHF, or benchmark design in depth — those appear only insofar as they explain *why* the gap exists. The focus is architectural and implementational: a five-component LangGraph supervisor with risk scoring, Harbor-style verifier contracts, escalation artifacts, checkpoint commits as blast-radius boundaries, and a calibration loop that mines labeled outcomes back into the classifier.
## Table of Contents

1. [Background: the loop engineering moment](#background-the-loop-engineering-moment)
2. [The verifiability gap](#the-verifiability-gap)
3. [Why more review agents don't close the gap](#why-more-review-agents-dont-close-the-gap)
4. [Framework overview: five-component supervisor graph](#framework-overview-five-component-supervisor-graph)
5. [Component One: The Risk Classifier](#component-one-the-risk-classifier)
   - [Checkable signals](#checkable-signals)
   - [Static fan-out analysis](#static-fan-out-analysis)
   - [Scoring and routing](#scoring-and-routing)
6. [Component Two: Bounded Execution](#component-two-bounded-execution)
   - [Harbor-style verifier contracts](#harbor-style-verifier-contracts)
   - [Iteration caps and reclassification](#iteration-caps-and-reclassification)
7. [Component Three: The Escalation Subgraph](#component-three-the-escalation-subgraph)
   - [Minimal review artifacts](#minimal-review-artifacts)
   - [LangGraph interrupt and HITL pause](#langgraph-interrupt-and-hitl-pause)
8. [Component Four: Checkpoint Commits as Blast-Radius Boundaries](#component-four-checkpoint-commits-as-blast-radius-boundaries)
9. [Component Five: The Calibration Loop](#component-five-the-calibration-loop)
10. [Worked Example: Architectural Decision vs Mechanical Rename](#worked-example-architectural-decision-vs-mechanical-rename)
11. [Related Directions](#related-directions)
12. [Challenges and Open Problems](#challenges-and-open-problems)
13. [References](#references)
## Background: the loop engineering moment

In mid-2026 a phrase spread quickly through the agent-engineering community: stop prompting your coding agent, start designing the loop that prompts it for you. The idea itself predates the phrase — agents inside feedback loops with tools, retries, and stop conditions is not new — but the framing crystallized something practitioners had already converged on. A loop, in this sense, is a small system: a trigger, a verification step, some memory, and a stop condition, wrapped around a model.

Loops work extremely well on bounded, mechanically checkable work. Triage a failing test, migrate a deprecated API call, fix a lint violation — in each case a program can tell you, in seconds, whether the loop succeeded. That is also the shape of task reinforcement learning for coding agents optimizes against: a base commit, an issue description, and a test suite that returns a scalar. **SWE-bench** (Jimenez et al., 2023) operationalized that contract; **Harbor**-style environments later packaged it as an explicit bundle of environment, instruction, and scoring function.

The trouble starts once you point the same loop at something that does not reduce to pass/fail. Several practitioner reports through 2026 — most notably a widely discussed essay from the **HumanLayer** team on lights-off agent coding, alongside data from **Faros AI**'s code-review research — describe teams that went "lights-off" (no human reading agent-generated code before merge) and later found review quality, incident rates, and bugs-per-developer trending the wrong way. None of this is because the agents were failing their tests. It is because the tests were never checking the thing that eventually cost them time: whether the codebase stayed easy to change.

I would consider that moment less a failure of agent capability and more a failure of loop engineering. The loops were honest about what they measured. They were silent about what they could not measure.
## The verifiability gap

Call this the **verifiability gap**: the difference between what a loop's stop condition actually measures and what "success" means for the task. I find it useful to write that difference explicitly:

$$
G(t) = S(t) - M(\sigma_t)
$$

where $t$ is a step in an agentic plan, $S(t) \in [0,1]$ is the *success meaning* of the step (the latent property we actually care about — correctness under future change, interface stability, operational safety), $M(\sigma_t) \in [0,1]$ is what the stop condition $\sigma_t$ can measure (tests, type checks, lint, schema validation), and $G(t)$ is the residual gap. For a narrow bug fix with `FAIL_TO_PASS` / `PASS_TO_PASS` oracles, $G(t) \approx 0$. For an architectural decision — introducing a new service boundary, choosing a data model, deciding where logic should live — $G(t)$ is large, because there is no fast oracle for "will this be easy to extend in three months."

The cost of a bad architectural decision surfaces weeks or months later, when a one-line change requires touching eleven files. RL cannot optimize against a signal that arrives that late, and neither can a loop's retry logic. The asymmetry is invisible from inside the loop: an agent iterating against a test suite has no internal signal that says "you are currently making a decision this suite cannot evaluate." It converges on a passing, poorly designed solution with the same confidence it shows for a well-designed one.

A useful operational corollary: **a loop should refuse to terminate successfully when $G(t)$ exceeds a threshold**, even if $M(\sigma_t) = 1$. That refusal is escalation. The rest of this post makes that refusal structural.

## Why more review agents don't close the gap

The natural response — more review agents, more linters, an "adversarial review" pass — raises the floor. It catches obviously bad code. It does not raise the ceiling: adding a second pass/fail check on top of an already-blind stop condition does not create the missing signal. In practice this means (1) a second LLM critiquing the first (bounded by shared priors), (2) a static-analysis pass that was already cheap, or (3) an adversarial agent that produces prose rather than a scalar oracle. The failure mode I care about is not "the review agent missed a bug." It is "the review agent approved a design the stop condition was never capable of evaluating." The architecture needs a routing decision that admits when measurement is insufficient.

## Framework overview: five-component supervisor graph

If the gap cannot be closed by training a better verifier for maintainability, the loop needs to know when it has hit the gap and hand the decision to something that can evaluate it. That reframes the engineering problem: instead of one loop with one stop condition, build a small graph with a routing decision at its center.

I implement that as a LangGraph `StateGraph` with five components: risk classification, bounded execution, escalation, checkpoint commit, and calibration. Task decomposition feeds the classifier; the classifier routes to bounded execution or escalation; both merge into a checkpoint; calibration feeds historical incident rates back into the classifier.

```
                    START → decompose → classify
                                        │
                    risk < τ ───────────┼─────────── risk ≥ τ
                         ▼              │                ▼
                  bounded_exec          │            escalate
                  (Harbor contract      │         ArtifactBuilder
                   + iter cap)          │          + interrupt()
                         │              │                │
                   exhausted? ──────────┘───────────────┘
                         │
                         ▼
                    checkpoint  →  calibrate  →  next step / END
                    (git SHA)      (Qdrant / τ)
```

State and graph wiring:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt


class Route(str, Enum):
    BOUNDED = "bounded"
    ESCALATE = "escalate"


class EscalationOutcome(str, Enum):
    APPROVED = "approved"
    REDIRECTED = "redirected"
    REJECTED = "rejected"


@dataclass
class Step:
    step_id: str
    description: str
    planned_files: list[str]
    proposed_diff: str | None = None
    task_class: str = "general"  # rename | migration | boundary | bugfix | …


@dataclass
class VerifierResult:
    score: float
    passed: bool
    details: str


@dataclass
class RiskAssessment:
    score: float
    verifier_exists: bool
    fan_out: int
    historical_incident_rate: float
    route: Route
    reasons: list[str] = field(default_factory=list)


@dataclass
class SupervisorState:
    repo_path: str
    steps: list[Step]
    step_index: int = 0
    risk: RiskAssessment | None = None
    iteration: int = 0
    max_iterations: int = 3
    last_verifier: VerifierResult | None = None
    review_artifact: dict[str, Any] | None = None
    escalation_outcome: EscalationOutcome | None = None
    human_notes: str | None = None
    checkpoint_sha: str | None = None
    checkpoint_stack: list[str] = field(default_factory=list)
    labeled_outcomes: list[dict[str, Any]] = field(default_factory=list)


def route_after_bounded(state: SupervisorState) -> str:
    if state.last_verifier and state.last_verifier.passed:
        return "checkpoint"
    if state.iteration >= state.max_iterations:
        return "escalate"  # reclassify-by-exhaustion
    return "bounded_exec"


def build_supervisor_graph(classifier, executor, escalator, checkpointer, calibrator):
    g = StateGraph(SupervisorState)
    for name, node in [
        ("decompose", decompose_node),
        ("classify", classifier.as_node),
        ("bounded_exec", executor.as_node),
        ("escalate", escalator.as_node),
        ("checkpoint", checkpointer.as_node),
        ("calibrate", calibrator.as_node),
    ]:
        g.add_node(name, node)
    g.set_entry_point("decompose")
    g.add_edge("decompose", "classify")
    g.add_conditional_edges(
        "classify", lambda s: s.risk.route.value,
        {"bounded": "bounded_exec", "escalate": "escalate"},
    )
    g.add_conditional_edges(
        "bounded_exec", route_after_bounded,
        {"checkpoint": "checkpoint", "escalate": "escalate", "bounded_exec": "bounded_exec"},
    )
    g.add_edge("escalate", "checkpoint")
    g.add_edge("checkpoint", "calibrate")
    g.add_conditional_edges(
        "calibrate",
        lambda s: "classify" if s.step_index < len(s.steps) else END,
        {"classify": "classify", END: END},
    )
    return g.compile(checkpointer=MemorySaver())
```

Each node is a pure function from `SupervisorState` to a partial update. The interesting control flow lives in the conditional edges after `classify` and after `bounded_exec`.

## Component One: The Risk Classifier

Every step gets scored *before* it runs, not after. The score should be built from checkable signals rather than an LLM's self-assessment, because self-assessment is exactly the thing that fails silently. I treat the classifier as a small deterministic program with optional learned inputs (historical rates), not as another agent.

### Checkable signals

Three signals form the core:

- **Verifier existence** — can a deterministic check (test, schema validation, type check) even be written for this step? If no, that fact alone is stronger than any confidence estimate the model could offer about its own plan.
- **Fan-out** — how many files or call sites does the change touch? A static-analysis proxy for the "shotgun surgery" smell: work that ripples across many unrelated places is disproportionately likely to be an architectural decision wearing a bug-fix costume.
- **Historical incident rate** — pulled from mined traces of similar past steps. This is the one signal that improves over time, which makes the classifier part of a learning system rather than a static rule engine.

Formally:

$$
R(t) = w_v \cdot (1 - \mathbb{1}_{\text{verifier}}(t)) + w_f \cdot \tilde{f}(t) + w_h \cdot h(t)
$$

where $\mathbb{1}_{\text{verifier}}(t) \in \{0,1\}$ indicates whether a verifier contract exists, $\tilde{f}(t) = \min(1, f(t)/F_{\max})$ is normalized fan-out, $h(t) \in [0,1]$ is the historical incident rate for similar steps, and $w_v + w_f + w_h = 1$. Route to escalation when $R(t) \ge \tau$. Defaults I would start with: $w_v = 0.45$, $w_f = 0.30$, $w_h = 0.25$, $\tau = 0.55$, $F_{\max} = 8$. These are not sacred; calibration exists because $\tau$ should move.

### Static fan-out analysis

Fan-out should not be "number of files the agent *says* it will touch." Agents understate scope. Prefer static analysis: parse proposed edit sites, resolve symbols, count call sites / import dependents.

```python
import ast
from pathlib import Path


def extract_changed_symbols(diff_text: str) -> list[str]:
    symbols: list[str] = []
    for line in diff_text.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        s = line[1:].lstrip()
        if s.startswith("def "):
            symbols.append(s[4:].split("(")[0].strip())
        elif s.startswith("class "):
            symbols.append(s[6:].split("(")[0].split(":")[0].strip())
    return [x for x in symbols if x.isidentifier()]


def python_fan_out(repo_path: str, symbols: list[str]) -> int:
    """Count files referencing any of `symbols` via AST name/attr matches.

    Production should prefer tree-sitter + a real name resolver; AST shows the contract.
    """
    root, wanted, hits = Path(repo_path), set(symbols), set()
    skip = {".", "venv", ".venv", "node_modules", "__pycache__"}
    for path in root.rglob("*.py"):
        if any(p in skip or p.startswith(".") for p in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        if wanted & (names | attrs):
            hits.add(str(path.relative_to(root)))
    return len(hits)
```

Verifier existence is similarly mechanical: given a step, do we have a `VerifierContract` registered for its `task_class`? Absence is not a soft preference; it is hard evidence that $M(\sigma_t)$ is undefined.

### Scoring and routing

```python
@dataclass
class ClassifierConfig:
    w_v: float = 0.45
    w_f: float = 0.30
    w_h: float = 0.25
    tau: float = 0.55
    f_max: float = 8.0


class RiskClassifier:
    def __init__(self, config: ClassifierConfig, contract_registry: dict, incident_store):
        self.config, self.contract_registry, self.incident_store = (
            config, contract_registry, incident_store
        )

    def assess(self, state: SupervisorState, step: Step) -> RiskAssessment:
        verifier_exists = step.task_class in self.contract_registry
        symbols = extract_changed_symbols(step.proposed_diff or "")
        fan_out = (
            python_fan_out(state.repo_path, symbols)
            if symbols else max(1, len(step.planned_files))
        )
        h = self.incident_store.historical_rate(
            task_class=step.task_class, description=step.description, fan_out=fan_out
        )
        cfg = self.config
        f_tilde = min(1.0, fan_out / cfg.f_max)
        score = (
            cfg.w_v * (0.0 if verifier_exists else 1.0)
            + cfg.w_f * f_tilde
            + cfg.w_h * h
        )
        reasons = []
        if not verifier_exists:
            reasons.append(f"no verifier contract for task_class={step.task_class}")
        if f_tilde >= 0.5:
            reasons.append(f"fan_out={fan_out} (normalized={f_tilde:.2f})")
        if h >= 0.3:
            reasons.append(f"historical_incident_rate={h:.2f}")
        route = Route.ESCALATE if score >= cfg.tau else Route.BOUNDED
        return RiskAssessment(score, verifier_exists, fan_out, h, route, reasons)

    def as_node(self, state: SupervisorState) -> dict:
        return {"risk": self.assess(state, state.steps[state.step_index]), "iteration": 0}
```

The classifier is deliberately boring. Boring is the point: a silent failure in the router reproduces the original problem.

## Component Two: Bounded Execution

Steps that clear the classifier run inside a capped loop with a verifier contract attached — not a general instruction to "write good code," but a Harbor-style bundle of an environment, an instruction, and a scoring function specific to that step.

### Harbor-style verifier contracts

I borrow the three-part shape popularized by **Harbor**-style agent environments: *(environment, instruction, scoring function)*. The environment is the sandbox. The instruction is the step description plus constraints. The scoring function returns a scalar in $[0,1]$. The contract is attached *before* the agent iterates, so the stop condition is not improvised mid-loop.

```python
@dataclass
class VerifierContract:
    name: str
    task_class: str
    environment: dict[str, Any]
    instruction_template: str
    score_fn: Callable[[SupervisorState, Step], VerifierResult]
    max_iterations: int = 3


def pytest_contract(test_path: str) -> VerifierContract:
    def score_fn(state: SupervisorState, step: Step) -> VerifierResult:
        from subprocess import run
        proc = run(
            ["pytest", test_path, "-q"],
            cwd=state.repo_path, capture_output=True, text=True, timeout=120,
        )
        passed = proc.returncode == 0
        return VerifierResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            details=(proc.stdout[-4000:] + proc.stderr[-2000:]),
        )

    return VerifierContract(
        name=f"pytest:{test_path}",
        task_class="bugfix",
        environment={"cwd": ".", "network": False, "timeout_s": 120},
        instruction_template=(
            "Apply a minimal patch for: {description}. "
            "Do not change public APIs. Stop when the attached verifier passes."
        ),
        score_fn=score_fn,
        max_iterations=3,
    )
```

The important property is not that every contract is a unit test. It is that **every autonomous step names its oracle up front**. If you cannot name one, the classifier should already have routed you to escalation.

### Iteration caps and reclassification

Capping iteration count matters as much as the verifier. An agent that cannot pass its own contract in a small, fixed number of tries should not keep grinding. It should be reclassified and routed to escalation — the safety net for a classifier that misjudged risk before work started.

$$
\text{exhaust}(t) = \mathbb{1}\!\left[i_t \ge I_{\max} \land M(\sigma_t) < 1\right]
$$

When $\text{exhaust}(t)=1$, the step enters escalation with reason `verifier_exhausted`. Calibration later treats exhaustion as a positive label for "should have been higher risk."

```python
class BoundedExecutor:
    def __init__(self, contract_registry: dict[str, VerifierContract], agent_fn):
        self.contract_registry, self.agent_fn = contract_registry, agent_fn

    def as_node(self, state: SupervisorState) -> dict:
        step = state.steps[state.step_index]
        contract = self.contract_registry[step.task_class]
        instruction = contract.instruction_template.format(description=step.description)
        self.agent_fn(state, step, instruction)  # one attempt; mutates working tree
        result = contract.score_fn(state, step)
        return {
            "iteration": state.iteration + 1,
            "max_iterations": contract.max_iterations,
            "last_verifier": result,
        }
```

Bounded execution is where most volume should live in a healthy factory: mechanical work with real oracles. The graph's job is to keep architectural work *out* of this path, and to eject misrouted work quickly when the oracle stops cooperating.

## Component Three: The Escalation Subgraph

Steps that do not clear the classifier — or that exhaust their retries — get routed here instead of continuing to iterate blind. The subgraph produces the smallest artifact that lets a human decide quickly, then pauses.

### Minimal review artifacts

I deliberately avoid full design documents. Prefer three compact views: **call-stack diff** (control-flow edges gained/lost), **file-tree diff** (added/removed/moved paths), and **interface signatures** (before/after for functions at stake).

```python
@dataclass
class ReviewArtifact:
    step_id: str
    risk: RiskAssessment
    summary: str
    file_tree_diff: list[str]
    interface_signatures: list[dict[str, str]]
    call_stack_diff: list[str]
    proposed_diff_excerpt: str
    questions_for_reviewer: list[str]


class ArtifactBuilder:
    def build(self, state: SupervisorState, step: Step) -> ReviewArtifact:
        risk = state.risk
        assert risk is not None
        symbols = extract_changed_symbols(step.proposed_diff or "")
        qs = [
            "Is this change mechanical, or does it set a new boundary/invariant?",
            "What would a correct verifier for this step look like if we had one?",
        ]
        if not risk.verifier_exists:
            qs.append("Approve proceeding without a stop-condition oracle?")
        if risk.fan_out >= 5:
            qs.append("Is the fan-out expected (rename) or a smell (shotgun surgery)?")
        return ReviewArtifact(
            step_id=step.step_id,
            risk=risk,
            summary=step.description,
            file_tree_diff=[f"+/- {f}" for f in step.planned_files],
            interface_signatures=[
                {"symbol": s, "after": f"{s}(...)", "before": "unknown"} for s in symbols
            ],
            call_stack_diff=[f"approx call-site fan-out for {symbols}: {risk.fan_out} files"],
            proposed_diff_excerpt=(step.proposed_diff or "")[:6000],
            questions_for_reviewer=qs,
        )
```

None of this replaces judgment. It compresses the input to judgment.

### LangGraph interrupt and HITL pause

The escalation node builds the artifact, then calls LangGraph's `interrupt()` so the graph pauses until a human resumes with an outcome. The contribution is *routing* to HITL systematically rather than leaving the decision to whether a developer happens to remember to look.

```python
class EscalationSubgraph:
    def __init__(self, artifacts: ArtifactBuilder):
        self.artifacts = artifacts

    def as_node(self, state: SupervisorState) -> dict:
        step = state.steps[state.step_index]
        artifact = self.artifacts.build(state, step)
        decision = interrupt({
            "type": "verifiability_escalation",
            "artifact": artifact.__dict__,
            "risk_score": state.risk.score if state.risk else None,
            "reasons": state.risk.reasons if state.risk else [],
        })
        # resume payload: {"outcome": "approved"|"redirected"|"rejected", "notes"?, "diff"?}
        outcome = EscalationOutcome(decision["outcome"])
        updates: dict[str, Any] = {
            "review_artifact": artifact.__dict__,
            "escalation_outcome": outcome,
            "human_notes": decision.get("notes"),
        }
        if outcome is EscalationOutcome.REDIRECTED and "diff" in decision:
            step.proposed_diff = decision["diff"]
            updates["steps"] = state.steps
        if outcome is EscalationOutcome.REJECTED:
            updates["last_verifier"] = VerifierResult(
                0.0, False, "rejected at escalation"
            )
        return updates
```

Escalation rate over a window of $N$ steps:

$$
E = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\!\left[\text{route}_i = \text{escalate} \lor \text{exhaust}_i\right]
$$

Calibration's job is to keep $E$ informative: high enough to catch real gaps, low enough that humans remain the scarce resource rather than the default path.

## Component Four: Checkpoint Commits as Blast-Radius Boundaries

Every merge point between the two branches is also a checkpoint: a known-good state the system can roll back to if a downstream verifier later fails. In most agent frameworks, checkpoints mean *resumability*. In a verifiability-gated factory, they also mean *containment*: bound how much bad work can accumulate before anyone notices.

Without this, a misclassified step becomes the foundation later steps quietly build on. Blast radius after step $k$:

$$
B(k) = \left|\left\{ j > k : \text{depends}(j, k) \right\}\right|
$$

Checkpointing after every gated step keeps $B(k)$ small by construction.

```python
import subprocess


class CheckpointCommitter:
    def __init__(self, git: bool = True):
        self.git = git

    def as_node(self, state: SupervisorState) -> dict:
        step = state.steps[state.step_index]
        if state.escalation_outcome is EscalationOutcome.REJECTED:
            return {
                "checkpoint_sha": self._rollback(state),
                "step_index": state.step_index + 1,
                "escalation_outcome": None,
                "review_artifact": None,
            }
        sha = self._commit(state, step)
        return {
            "checkpoint_sha": sha,
            "checkpoint_stack": state.checkpoint_stack + [sha],
            "step_index": state.step_index + 1,
            "escalation_outcome": None,
            "review_artifact": None,
            "last_verifier": None,
        }

    def _commit(self, state: SupervisorState, step: Step) -> str:
        if not self.git:
            return f"snap-{state.step_index}-{step.step_id}"
        subprocess.run(["git", "add", "-A"], cwd=state.repo_path, check=True)
        msg = f"agent-checkpoint: {step.step_id} — {step.description[:72]}"
        subprocess.run(
            ["git", "commit", "-m", msg, "--allow-empty"], cwd=state.repo_path, check=True
        )
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=state.repo_path, text=True
        ).strip()

    def _rollback(self, state: SupervisorState) -> str:
        prev = state.checkpoint_stack[-1] if state.checkpoint_stack else (state.checkpoint_sha or "ROOT")
        if self.git and state.checkpoint_stack:
            subprocess.run(["git", "reset", "--hard", prev], cwd=state.repo_path, check=True)
        return prev
```

Rollback should be boring and total: hard reset to the previous checkpoint SHA, discard uncommitted agent edits, preserve the escalation label for calibration. LangGraph's checkpointer persists *graph state* for resume; git persists *repo state* for blast-radius control. Keep both.

## Component Five: The Calibration Loop

Every escalation outcome — approved, redirected, or rejected — and every autonomous failure caught by the retry cap becomes a labeled example. Mining these back into the risk classifier keeps the boundary between "loop it" and "escalate it" from being a fixed guess. A team that consistently sees database-migration steps rejected at escalation should see the classifier tighten for that class automatically.

| Event | Label $\ell$ | Effect |
|---|---|---|
| Escalation rejected / verifier exhausted | $1$ | raise $h(t)$; consider lowering $\tau$ |
| Escalation redirected | $0.5$ | mild increase in $h(t)$ |
| Escalation approved | $0$ | slight decrease (possible over-escalation) |
| Bounded pass, no later revert | $0$ | confirm low-risk path |

Online update for class-conditional incident rate: $h_{c} \leftarrow (1-\alpha)\, h_{c} + \alpha\, \ell$. If rolling false-negative rate among formerly bounded steps exceeds budget $\epsilon$, decrease $\tau$ by $\Delta$; if escalation rate $E$ exceeds a cost budget, increase $\tau$ cautiously — never so far that verifier-absent steps route to bounded execution.

I store step embeddings plus incident labels in Qdrant so `historical_rate` can query *similar* past steps, not only exact `task_class` matches:

```python
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue,
    PointStruct, VectorParams, PayloadSchemaType,
)


class IncidentStore:
    def __init__(self, client: QdrantClient, embed_fn, collection: str = "step_incidents"):
        self.client, self.embed_fn, self.collection = client, embed_fn, collection
        names = {c.name for c in client.get_collections().collections}
        if collection not in names:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            for field_name, schema in [
                ("task_class", PayloadSchemaType.KEYWORD),
                ("incident", PayloadSchemaType.FLOAT),
                ("fan_out", PayloadSchemaType.INTEGER),
                ("repo", PayloadSchemaType.KEYWORD),
            ]:
                client.create_payload_index(
                    collection_name=collection, field_name=field_name, field_schema=schema
                )

    def record(self, *, repo: str, step: Step, risk: RiskAssessment, label: float, notes: str | None):
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=self.embed_fn(f"{step.task_class}: {step.description}"),
                payload={
                    "task_class": step.task_class, "description": step.description,
                    "incident": label, "fan_out": risk.fan_out, "risk_score": risk.score,
                    "repo": repo, "notes": notes or "",
                },
            )],
        )

    def historical_rate(self, *, task_class: str, description: str, fan_out: int, top_k: int = 20) -> float:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=self.embed_fn(f"{task_class}: {description}"),
            query_filter=Filter(must=[
                FieldCondition(key="task_class", match=MatchValue(value=task_class))
            ]),
            limit=top_k, with_payload=True,
        )
        if not hits:
            return 0.2  # cold-start prior
        num = sum(h.score * float(h.payload["incident"]) for h in hits)
        den = sum(h.score for h in hits) or 1.0
        return max(0.0, min(1.0, num / den))


class Calibrator:
    def __init__(self, store: IncidentStore, classifier: RiskClassifier):
        self.store, self.classifier = store, classifier

    def as_node(self, state: SupervisorState) -> dict:
        prev = state.step_index - 1
        if prev < 0 or state.risk is None:
            return {}
        step, label = state.steps[prev], self._label(state)
        self.store.record(
            repo=state.repo_path, step=step, risk=state.risk,
            label=label, notes=state.human_notes,
        )
        if label >= 1.0 and state.iteration >= state.max_iterations:
            self.classifier.config.tau = max(0.35, self.classifier.config.tau - 0.02)
        row = {"step_id": step.step_id, "label": label, "tau": self.classifier.config.tau}
        return {"labeled_outcomes": state.labeled_outcomes + [row], "risk": None}

    def _label(self, state: SupervisorState) -> float:
        if state.escalation_outcome is EscalationOutcome.REJECTED:
            return 1.0
        if state.escalation_outcome is EscalationOutcome.REDIRECTED:
            return 0.5
        if state.escalation_outcome is EscalationOutcome.APPROVED:
            return 0.0
        if state.last_verifier and not state.last_verifier.passed:
            return 1.0
        return 0.0
```

Calibration does not invent a maintainability oracle. It reallocates human attention toward regions where past silence was expensive.

## Worked Example: Architectural Decision vs Mechanical Rename

Consider two steps an unconstrained coding agent might treat as "just another patch."

**Step A — mechanical rename.** Rename `CustomerDTO` → `CustomerRecord`. Fan-out is 14 files, but every change is an identifier rewrite. A verifier exists (unit tests + grep gate for leftover `CustomerDTO`). Historical rate for `task_class="rename"` is low ($h \approx 0.05$):

$$
R_A = 0.45\cdot 0 + 0.30\cdot\min(1, 14/8) + 0.25\cdot 0.05 = 0.3125
$$

With $\tau = 0.55$, Step A routes to **bounded execution**. High fan-out alone does not escalate when a verifier exists and history is clean.

**Step B — architectural boundary.** "Extract billing into a separate service and leave a façade in the monolith." Planned files look small. No honest verifier exists for "façade will age well." Historical rate for `task_class="boundary"` is high ($h \approx 0.62$):

$$
R_B = 0.45\cdot 1 + 0.30\cdot\min(1, 3/8) + 0.25\cdot 0.62 = 0.7175
$$

Step B routes to **escalation**. A human redirects: keep billing in-process but isolate a module boundary with explicit interfaces. Checkpoint records the redirected state; calibration stores $\ell=0.5$.

The contrast is the thesis: **fan-out without verifiers is not the same object as fan-out with verifiers.** Lights-off factories that optimize only for "tests green" systematically promote Step-B work into Step-A paths. The supervisor exists to make that promotion expensive and loud.

## Related Directions

A few efforts point at the same problem from adjacent axes. **Judge models over diffs** raise the floor for obvious defects but remain bounded by the same priors that produced the design, so they do not close $G(t)$ for latent maintainability. **Mutation testing** improves $M(\sigma_t)$ when the missing signal is test weakness — helpful for mechanical work, not an oracle for architectural properties absent from near-term tests. **Permission boundaries** in remediation pipelines keep write credentials and deploy triggers in a deterministic layer the agent never holds; that is an authority problem rather than a verifiability problem, but it rhymes with escalation. **LangGraph HITL and durable checkpointers** supply the pause/resume mechanism; this post is about *when* to pause for verifiability reasons. I would not treat any of these as substitutes for verifiability gating.

## Challenges and Open Problems

**False negatives in the classifier.** Fan-out and verifier-existence are proxies — a one-file change can still be a bad architectural decision, and a ten-file change can be entirely mechanical. Getting the false-negative rate down is an open calibration problem, and it is the one place where a bad call has the same silent-failure characteristic that motivated this framework. Exhaustion-based reclassification mitigates some misses; it does not help when a weak verifier passes.

**Escalation cost.** A system that escalates too aggressively reproduces the original bottleneck — a human reviewing everything, with extra steps in between. Whether calibration converges to a stable low $E$ for a given codebase, or whether high architectural churn simply requires a permanently higher rate, is an empirical question I do not yet have multi-team data to answer.

**Cold start and checkpoint throughput.** Qdrant-backed incident memory is empty on day one; priors (returning $0.2$ for unknown classes) are placeholders. Checkpoint-per-step maximizes containment and can thrash git history on high-volume mechanical factories; batching across proven-mechanical sequences re-opens blast radius inside the batch. Adaptive batching conditioned on $R(t)$ is plausible future work — and a new source of silent accumulation if the conditioner is wrong.

**The training gap remains.** None of this addresses the underlying training gap — it routes around it, at the loop level, for a single team's codebase. If a future model generation acquires a robust sense of maintainability, most of the escalation subgraph becomes redundant. Until a benchmark result makes that credible rather than aspirational, treating verifiability as an explicit, checkable property of each step seems like the more defensible place to put the engineering effort.

The framework is a contract about silence: when the stop condition cannot speak to what matters, the loop should stop speaking for itself.

## References

- Jimenez, Carlos E. et al. (2023). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* arXiv:2310.06770
- Yang, John et al. (2024). *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*. NeurIPS 2024.
- Shinn, Noah et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023.
- LangGraph Documentation. *Interrupts (Human-in-the-Loop)*. [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- LangGraph Documentation. *Persistence / Checkpointers*. [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- Qdrant. *Qdrant Vector Database Documentation*. [qdrant.tech/documentation](https://qdrant.tech/documentation)
- Fowler, Martin. *Code Smells: Shotgun Surgery*. [martinfowler.com](https://martinfowler.com/bliki/ShotgunSurgery.html)
- Cristian, Flaviu (1991). Fail-silent / fault-tolerant distributed systems survey literature.
- HumanLayer. Practitioner writing on lights-off / human-in-the-loop agent coding (2025–2026).
- Faros AI. Code review and engineering metrics research on AI-assisted development (2025–2026).

```bibtex
@article{inamdar2026verifiability,
  title={Verifiability-Gated Loops: An Escalation Contract for Agentic Software Factories},
  author={Inamdar, Mihir}, year={2026}, note={Personal blog}}
```
