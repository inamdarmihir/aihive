---
title: "risk-gate: A Working Risk Classifier for Agentic Software Factories"
date: 2026-07-08
description: "This post is the design spec for risk-gate, a small library for agentic coding loops that scores every proposed step for verifiability before it runs, routes unverifiable or high-fan-out work to human escalation instead of grinding blind against a stop condition that can't see what matters, and calibrates its own thresholds against a Qdrant-backed history of past outcomes."
tags: ["agents", "verifiability", "human-in-the-loop", "software-engineering", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Coding agents can now run for hours without a human reading a line of what they produce. This post is the design spec for **risk-gate**, a small library I'm building that gives an agentic coding loop a structural way to recognize when its own stop condition cannot measure what actually matters for a step — and to route that step to a human instead of grinding forward on a passing-but-blind signal.

I want to narrow the scope precisely, because the temptation with a post like this is to write an architecture essay with illustrative snippets. That is not the goal here. The goal is a library specification precise enough that the code shown could become a real repository: module boundaries, function and class signatures, a stated dependency footprint, an installation story. This post does not cover model training, RLHF, or benchmark design in depth — those appear only insofar as they explain *why* the verifiability gap exists. It covers the five-component design of `risk-gate` itself: a risk classifier, bounded execution under a verifier contract, an escalation subgraph, checkpoint commits as blast-radius boundaries, and a Qdrant-backed calibration loop.

## Table of Contents

1. [Background: the loop engineering moment](#background-the-loop-engineering-moment)
2. [The verifiability gap](#the-verifiability-gap)
3. [Why more review agents don't close the gap](#why-more-review-agents-dont-close-the-gap)
4. [risk-gate: Library Overview and Module Layout](#risk-gate-library-overview-and-module-layout)
5. [Installing and Using risk-gate](#installing-and-using-risk-gate)
6. [Component One: The Risk Classifier](#component-one-the-risk-classifier)
   - [Checkable signals](#checkable-signals)
   - [Static fan-out analysis](#static-fan-out-analysis)
   - [Scoring and routing](#scoring-and-routing)
7. [Component Two: Bounded Execution](#component-two-bounded-execution)
   - [Verifier contracts](#verifier-contracts)
   - [Iteration caps and reclassification](#iteration-caps-and-reclassification)
8. [Component Three: The Escalation Subgraph](#component-three-the-escalation-subgraph)
   - [Minimal review packages](#minimal-review-packages)
9. [Component Four: Checkpoint Commits as Blast-Radius Boundaries](#component-four-checkpoint-commits-as-blast-radius-boundaries)
10. [Component Five: The Calibration Loop and Qdrant-Backed Incident Memory](#component-five-the-calibration-loop-and-qdrant-backed-incident-memory)
11. [Worked Example: Architectural Decision vs Mechanical Rename](#worked-example-architectural-decision-vs-mechanical-rename)
12. [Challenges and Open Problems](#challenges-and-open-problems)
13. [References](#references)

## Background: the loop engineering moment

In mid-2026 a phrase spread quickly through the agent-engineering community: stop prompting your coding agent, start designing the loop that prompts it for you. The idea itself predates the phrase — agents inside feedback loops with tools, retries, and stop conditions is not new — but the framing crystallized something practitioners had already converged on. A loop, in this sense, is a small system: a trigger, a verification step, some memory, and a stop condition, wrapped around a model.

Loops work extremely well on bounded, mechanically checkable work. Triage a failing test, migrate a deprecated API call, fix a lint violation — in each case a program can tell you, in seconds, whether the loop succeeded. That is also the shape of task reinforcement learning for coding agents optimizes against: a base commit, an issue description, and a test suite that returns a scalar. **SWE-bench** (Jimenez et al. 2023) operationalized that contract, and **SWE-agent** (Yang et al. 2024) built the agent-computer interface that made iterating against it practical at scale.

The trouble starts once you point the same loop at something that does not reduce to pass/fail. Self-reported accounts from teams running lights-off agentic pipelines through 2025–2026 — no human reading agent-generated code before merge — describe review quality, incident rates, and bugs-per-developer trending the wrong way after the switch, even while test suites stayed green. None of this is because the agents were failing their tests. It is because the tests were never checking the thing that eventually cost them time: whether the codebase stayed easy to change.

I would consider that moment less a failure of agent capability and more a failure of loop engineering. The loops were honest about what they measured. They were silent about what they could not measure. `risk-gate` is my attempt to make that silence structural instead of accidental — a library, not a one-off supervisor script, precisely because the failure mode recurs across every codebase that adopts agentic loops, not just the one I happen to be working in.

## The verifiability gap

Call this the **verifiability gap**: the difference between what a loop's stop condition actually measures and what "success" means for the task. I find it useful to write that difference explicitly:

$$
G(t) = S(t) - M(\sigma_t)
$$

where $t$ is a step in an agentic plan, $S(t) \in [0,1]$ is the *success meaning* of the step (the latent property we actually care about — correctness under future change, interface stability, operational safety), $M(\sigma_t) \in [0,1]$ is what the stop condition $\sigma_t$ can measure (tests, type checks, lint, schema validation), and $G(t)$ is the residual gap. For a narrow bug fix with `FAIL_TO_PASS` / `PASS_TO_PASS` oracles, $G(t) \approx 0$. For an architectural decision — introducing a new service boundary, choosing a data model, deciding where logic should live — $G(t)$ is large, because there is no fast oracle for "will this be easy to extend in three months."

The cost of a bad architectural decision surfaces weeks or months later, when a one-line change requires touching eleven files. RL cannot optimize against a signal that arrives that late, and neither can a loop's retry logic. The asymmetry is invisible from inside the loop: an agent iterating against a test suite has no internal signal that says "you are currently making a decision this suite cannot evaluate." It converges on a passing, poorly designed solution with the same confidence it shows for a well-designed one.

A useful operational corollary, and the thesis `risk-gate` exists to enforce: **a loop should refuse to terminate successfully when $G(t)$ exceeds a threshold**, even if $M(\sigma_t) = 1$. That refusal is escalation.

## Why more review agents don't close the gap

The natural response — more review agents, more linters, an adversarial-review pass — raises the floor. It catches obviously bad code. It does not raise the ceiling: adding a second pass/fail check on top of an already-blind stop condition does not create the missing signal. In practice this means a second LLM critiquing the first (bounded by shared priors), a static-analysis pass that was already cheap, or an adversarial agent producing prose rather than a scalar oracle. The failure mode I care about is not "the review agent missed a bug." It is "the review agent approved a design the stop condition was never capable of evaluating." The architecture needs a routing decision that admits when measurement is insufficient — which is what a risk classifier is for, and why it's the first module in the library rather than an afterthought bolted onto an existing loop.

## risk-gate: Library Overview and Module Layout

`risk-gate` is deliberately small: five modules, each owning exactly one of the five components, with a dependency footprint limited to `qdrant-client` for the calibration store and whatever the host loop framework already provides (I build against **LangGraph**, but nothing in the classifier, executor, or checkpoint modules requires it). The intended package layout:

```
risk_gate/
  __init__.py
  classifier.py      # RiskClassifier: scores a proposed step
  contracts.py         # VerifierContract: environment + instruction + scoring fn
  executor.py           # BoundedExecutor: runs a step under contract + iteration cap
  escalation.py          # EscalationArtifact: builds minimal review packages
  checkpoint.py           # CheckpointCommit: blast-radius boundary via git
  calibration.py           # CalibrationStore: Qdrant-backed historical incident rate
```

Each module has exactly one public class and a small, stable set of dataclasses it owns. `classifier.py` owns `ProposedStep` and `RiskScore`; `contracts.py` owns `VerifierContract`; `executor.py` owns `ExecutionResult`; `escalation.py` owns `ReviewPackage`. No module reaches into another's internals — the classifier depends on `contracts.VerifierContract` only through a registry it's handed at construction time, and on `calibration.CalibrationStore` only through its `historical_rate` method, which keeps the calibration store swappable (a team without Qdrant available yet can hand the classifier a stub that always returns a fixed prior).

```
                    ┌─────────────┐
   ProposedStep ──▶ │  classifier  │──▶ RiskScore
                    └──────┬───────┘
                    risk.route.value
              ┌────────────┴────────────┐
              ▼                          ▼
       ┌────────────┐            ┌─────────────┐
       │  executor   │            │ escalation   │
       │ (bounded,   │            │ (human-in-   │
       │  contract-  │            │  the-loop)   │
       │  scored)    │            └──────┬──────┘
       └──────┬─────┘                    │
              └─────────────┬────────────┘
                             ▼
                       ┌────────────┐
                       │ checkpoint  │  (git SHA, blast-radius boundary)
                       └─────┬──────┘
                             ▼
                       ┌────────────┐
                       │ calibration │  (Qdrant: label the step, adjust τ)
                       └────────────┘
```

## Installing and Using risk-gate

The library ships as a normal Python package with a minimal dependency set — `qdrant-client` and standard library only, with the host framework (LangGraph, or any other graph/agent orchestration layer) provided by the caller rather than pulled in as a hard dependency:

```bash
pip install risk-gate qdrant-client
```

Wiring it into an existing LangGraph supervisor is meant to be a few lines at the node level, not a rewrite of the graph:

```python
from risk_gate.classifier import RiskClassifier, ClassifierConfig
from risk_gate.calibration import CalibrationStore
from qdrant_client import QdrantClient

calibration_store = CalibrationStore(QdrantClient(url="localhost:6333"), embed_fn=my_embed_fn)
classifier = RiskClassifier(ClassifierConfig(), contract_registry, calibration_store)

def classify_node(state: SupervisorState) -> dict:
    step = state.steps[state.step_index]
    return {"risk": classifier.score(step)}

graph.add_node("classify", classify_node)
graph.add_conditional_edges(
    "classify",
    lambda s: s.risk.route.value,
    {"bounded": "bounded_exec", "escalate": "escalate"},
)
```

`classifier.score` is a pure function of a `ProposedStep` plus the classifier's own state (its config and its handle on the calibration store); it never touches the graph's `SupervisorState` directly, which is what makes it usable outside LangGraph too — a CI hook or a plain Python script can call `RiskClassifier.score` on a proposed diff without a graph runtime attached at all.

## Component One: The Risk Classifier

Every step gets scored *before* it runs, not after. The score should be built from checkable signals rather than an LLM's self-assessment, because self-assessment is exactly the thing that fails silently. `RiskClassifier` is a small deterministic program with optional learned inputs (historical rates), not another agent.

### Checkable signals

Three signals form the core:

- **Verifier existence** — can a deterministic check (test, schema validation, type check) even be written for this step? If no, that fact alone is stronger than any confidence estimate the model could offer about its own plan.
- **Fan-out** — how many files or call sites does the change touch? A static-analysis proxy for the "shotgun surgery" smell: work that ripples across many unrelated places is disproportionately likely to be an architectural decision wearing a bug-fix costume.
- **Historical incident rate** — pulled from mined traces of similar past steps, via `CalibrationStore`. This is the one signal that improves over time, which makes the classifier part of a learning system rather than a static rule engine.

Formally:

$$
R(t) = w_v \cdot (1 - \mathbb{1}_{\text{verifier}}(t)) + w_f \cdot \tilde{f}(t) + w_h \cdot h(t)
$$

where $\mathbb{1}_{\text{verifier}}(t) \in \{0,1\}$ indicates whether a verifier contract exists, $\tilde{f}(t) = \min(1, f(t)/F_{\max})$ is normalized fan-out, $h(t) \in [0,1]$ is the historical incident rate for similar steps, and $w_v + w_f + w_h = 1$. Route to escalation when $R(t) \ge \tau$. Defaults I would start with: $w_v = 0.45$, $w_f = 0.30$, $w_h = 0.25$, $\tau = 0.55$, $F_{\max} = 8$. These are not sacred; the calibration module exists because $\tau$ should move.

### Static fan-out analysis

Fan-out should not be "number of files the agent *says* it will touch." Agents understate scope. Prefer static analysis: parse proposed edit sites, resolve symbols, count call sites and import dependents.

```python
# risk_gate/classifier.py
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

    A real release should prefer tree-sitter plus a proper name resolver;
    the stdlib AST version here shows the contract this function fulfills.
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

Verifier existence is similarly mechanical: given a step, do we have a `VerifierContract` registered for its `task_class`? Absence is not a soft preference; it is hard evidence that $M(\sigma_t)$ is undefined for this step.

### Scoring and routing

The public surface of `classifier.py` is small on purpose: two dataclasses, a config, and one class with one entry point.

```python
# risk_gate/classifier.py (continued)
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Route(str, Enum):
    BOUNDED = "bounded"
    ESCALATE = "escalate"


@dataclass
class ProposedStep:
    step_id: str
    description: str
    planned_files: list[str]
    proposed_diff: str | None = None
    task_class: str = "general"  # rename | migration | boundary | bugfix | …


@dataclass
class RiskScore:
    value: float
    verifier_exists: bool
    fan_out: int
    historical_incident_rate: float
    route: Route
    reasons: list[str] = field(default_factory=list)


@dataclass
class ClassifierConfig:
    w_v: float = 0.45
    w_f: float = 0.30
    w_h: float = 0.25
    tau: float = 0.55
    f_max: float = 8.0


class RiskClassifier:
    def __init__(self, config: ClassifierConfig, contract_registry: dict[str, Any],
                 calibration_store: "CalibrationStore"):
        self.config = config
        self.contract_registry = contract_registry
        self.calibration_store = calibration_store

    def score(self, step: ProposedStep) -> RiskScore:
        verifier_exists = step.task_class in self.contract_registry
        symbols = extract_changed_symbols(step.proposed_diff or "")
        fan_out = (
            python_fan_out(step_repo_path(step), symbols)
            if symbols else max(1, len(step.planned_files))
        )
        h = self.calibration_store.historical_rate(
            task_class=step.task_class, description=step.description, fan_out=fan_out,
        )
        cfg = self.config
        f_tilde = min(1.0, fan_out / cfg.f_max)
        value = (
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
        route = Route.ESCALATE if value >= cfg.tau else Route.BOUNDED
        return RiskScore(value, verifier_exists, fan_out, h, route, reasons)
```

`step_repo_path` is a one-line accessor left out above for brevity (`ProposedStep` in a real repo would carry a `repo_path` field or the classifier would take one at construction time). The classifier is deliberately boring. Boring is the point: a silent failure in the router reproduces the exact problem the library exists to prevent.

## Component Two: Bounded Execution

Steps that clear the classifier run inside `BoundedExecutor`, a capped loop with a `VerifierContract` attached — not a general instruction to "write good code," but a bundle of an environment, an instruction, and a scoring function specific to that step.

### Verifier contracts

I borrow the three-part shape popularized by **Harbor**-style agent environments: *(environment, instruction, scoring function)*. The environment is the sandbox. The instruction is the step description plus constraints. The scoring function returns a scalar in $[0,1]$. The contract is attached *before* the agent iterates, so the stop condition is not improvised mid-loop.

```python
# risk_gate/contracts.py
from dataclasses import dataclass
from typing import Any, Callable
from risk_gate.classifier import ProposedStep


@dataclass
class ExecutionResult:
    passed: bool
    score: float
    details: str
    iterations_used: int = 0


@dataclass
class VerifierContract:
    name: str
    task_class: str
    environment: dict[str, Any]
    instruction_template: str
    score_fn: Callable[[ProposedStep], ExecutionResult]
    max_iterations: int = 2


def pytest_contract(test_path: str, repo_path: str) -> VerifierContract:
    def score_fn(step: ProposedStep) -> ExecutionResult:
        from subprocess import run
        proc = run(
            ["pytest", test_path, "-q"],
            cwd=repo_path, capture_output=True, text=True, timeout=120,
        )
        passed = proc.returncode == 0
        return ExecutionResult(
            passed=passed,
            score=1.0 if passed else 0.0,
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
        max_iterations=2,
    )
```

The important property is not that every contract is a unit test. It is that **every autonomous step names its oracle up front**. If you cannot name one, `RiskClassifier.score` should already have routed you to escalation.

### Iteration caps and reclassification

Capping iteration count matters as much as the verifier. An agent that cannot pass its own contract in a small, fixed number of tries should not keep grinding — it should be reclassified and routed to escalation, the safety net for a classifier that misjudged risk before work started.

$$
\text{exhaust}(t) = \mathbb{1}\!\left[i_t \ge I_{\max} \land M(\sigma_t) < 1\right]
$$

When $\text{exhaust}(t)=1$, the step enters escalation with reason `verifier_exhausted`. Calibration later treats exhaustion as a positive label for "should have been higher risk."

```python
# risk_gate/executor.py
from typing import Callable
from risk_gate.classifier import ProposedStep
from risk_gate.contracts import VerifierContract, ExecutionResult


class BoundedExecutor:
    def __init__(self, agent_fn: Callable[[ProposedStep, str], None]):
        """`agent_fn` mutates the working tree in place, given a step and a
        rendered instruction — the executor doesn't care how it does it."""
        self.agent_fn = agent_fn

    def run(self, step: ProposedStep, contract: VerifierContract,
            max_iters: int = 2) -> ExecutionResult:
        instruction = contract.instruction_template.format(description=step.description)
        result = ExecutionResult(passed=False, score=0.0, details="", iterations_used=0)
        for i in range(1, max_iters + 1):
            self.agent_fn(step, instruction)
            result = contract.score_fn(step)
            result.iterations_used = i
            if result.passed:
                return result
        return result
```

`BoundedExecutor.run` returns an `ExecutionResult` whether or not the step passed — exhaustion is a normal, expected return value, not an exception, since the escalation module needs the full attempt history rather than a stack trace to build a useful review package.

## Component Three: The Escalation Subgraph

Steps that do not clear the classifier — or that exhaust their retries in `BoundedExecutor` — get routed to `escalation.py` instead of continuing to iterate blind. The module produces the smallest artifact that lets a human decide quickly, then hands control back to whatever pause/resume mechanism the host loop uses.

### Minimal review packages

I deliberately avoid full design documents. Prefer three compact views: **call-stack diff** (control-flow edges gained/lost), **file-tree diff** (added/removed/moved paths), and **interface signatures** (before/after for functions at stake).

```python
# risk_gate/escalation.py
from dataclasses import dataclass
from risk_gate.classifier import ProposedStep, RiskScore, extract_changed_symbols
from risk_gate.contracts import ExecutionResult


@dataclass
class ReviewPackage:
    step_id: str
    risk: RiskScore
    summary: str
    file_tree_diff: list[str]
    interface_signatures: list[dict[str, str]]
    call_stack_diff: list[str]
    proposed_diff_excerpt: str
    questions_for_reviewer: list[str]


class EscalationArtifact:
    @staticmethod
    def build(step: ProposedStep, execution_history: list[ExecutionResult],
              risk: RiskScore) -> ReviewPackage:
        symbols = extract_changed_symbols(step.proposed_diff or "")
        questions = [
            "Is this change mechanical, or does it set a new boundary/invariant?",
            "What would a correct verifier for this step look like if we had one?",
        ]
        if not risk.verifier_exists:
            questions.append("Approve proceeding without a stop-condition oracle?")
        if risk.fan_out >= 5:
            questions.append("Is the fan-out expected (rename) or a smell (shotgun surgery)?")
        if execution_history and not execution_history[-1].passed:
            questions.append(
                f"Verifier exhausted after {execution_history[-1].iterations_used} attempts "
                "— redirect, or accept without a passing oracle?"
            )
        return ReviewPackage(
            step_id=step.step_id,
            risk=risk,
            summary=step.description,
            file_tree_diff=[f"+/- {f}" for f in step.planned_files],
            interface_signatures=[
                {"symbol": s, "after": f"{s}(...)", "before": "unknown"} for s in symbols
            ],
            call_stack_diff=[f"approx call-site fan-out for {symbols}: {risk.fan_out} files"],
            proposed_diff_excerpt=(step.proposed_diff or "")[:6000],
            questions_for_reviewer=questions,
        )
```

`EscalationArtifact.build` is a static method rather than something requiring an instance, because it has no state of its own to hold — it's a pure transform from a step, its execution history, and its risk score into a `ReviewPackage`. None of this replaces judgment. It compresses the input to judgment.

Wired into LangGraph, the pause itself is a single `interrupt()` call around the built package:

```python
from langgraph.types import interrupt
from risk_gate.escalation import EscalationArtifact

def escalate_node(state) -> dict:
    step = state.steps[state.step_index]
    package = EscalationArtifact.build(step, state.execution_history, state.risk)
    decision = interrupt({"type": "risk_gate_escalation", "package": package.__dict__})
    return {"escalation_outcome": decision["outcome"], "human_notes": decision.get("notes")}
```

## Component Four: Checkpoint Commits as Blast-Radius Boundaries

Every merge point between the bounded and escalation paths is also a checkpoint: a known-good state the system can roll back to if a downstream verifier later fails. In most agent frameworks, checkpoints mean *resumability*. In `risk-gate`, they also mean *containment*: bound how much bad work can accumulate before anyone notices.

Without this, a misclassified step becomes the foundation later steps quietly build on. Blast radius after step $k$:

$$
B(k) = \left|\left\{ j > k : \text{depends}(j, k) \right\}\right|
$$

Checkpointing after every gated step keeps $B(k)$ small by construction.

```python
# risk_gate/checkpoint.py
import subprocess


class CheckpointCommit:
    def __init__(self, repo_path: str, use_git: bool = True):
        self.repo_path = repo_path
        self.use_git = use_git
        self._stack: list[str] = []

    def commit(self, step_id: str, description: str) -> str:
        if not self.use_git:
            sha = f"snap-{len(self._stack)}-{step_id}"
        else:
            subprocess.run(["git", "add", "-A"], cwd=self.repo_path, check=True)
            msg = f"risk-gate-checkpoint: {step_id} — {description[:72]}"
            subprocess.run(
                ["git", "commit", "-m", msg, "--allow-empty"], cwd=self.repo_path, check=True,
            )
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.repo_path, text=True,
            ).strip()
        self._stack.append(sha)
        return sha

    def rollback(self) -> str:
        prev = self._stack[-1] if self._stack else "ROOT"
        if self.use_git and self._stack:
            subprocess.run(["git", "reset", "--hard", prev], cwd=self.repo_path, check=True)
        return prev
```

Rollback should be boring and total: hard reset to the previous checkpoint SHA, discard uncommitted agent edits, preserve the escalation label for calibration. If the host loop's own framework has a checkpointer (LangGraph's `MemorySaver` and friends persist *graph state* for resume), keep both — `CheckpointCommit` persists *repo state* for blast-radius control, which is a different axis entirely.

## Component Five: The Calibration Loop and Qdrant-Backed Incident Memory

Every escalation outcome — approved, redirected, or rejected — and every autonomous failure caught by `BoundedExecutor`'s retry cap becomes a labeled example. Mining these back into the risk classifier keeps the boundary between "loop it" and "escalate it" from being a fixed guess. A team that consistently sees database-migration steps rejected at escalation should see the classifier tighten for that class automatically.

| Event | Label $\ell$ | Effect |
|---|---|---|
| Escalation rejected / verifier exhausted | $1$ | raise $h(t)$; consider lowering $\tau$ |
| Escalation redirected | $0.5$ | mild increase in $h(t)$ |
| Escalation approved | $0$ | slight decrease (possible over-escalation) |
| Bounded pass, no later revert | $0$ | confirm low-risk path |

Online update for class-conditional incident rate: $h_{c} \leftarrow (1-\alpha)\, h_{c} + \alpha\, \ell$. If rolling false-negative rate among formerly bounded steps exceeds budget $\epsilon$, decrease $\tau$ by $\Delta$; if escalation rate $E$ exceeds a cost budget, increase $\tau$ cautiously — never so far that verifier-absent steps route to bounded execution.

`CalibrationStore` stores step embeddings plus incident labels in Qdrant so `historical_rate` can query *similar* past steps, not only exact `task_class` matches:

```python
# risk_gate/calibration.py
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue,
    PointStruct, VectorParams, PayloadSchemaType,
)
from risk_gate.classifier import ProposedStep, RiskScore, RiskClassifier


class CalibrationStore:
    def __init__(self, client: QdrantClient, embed_fn, collection: str = "risk_gate_incidents"):
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
                    collection_name=collection, field_name=field_name, field_schema=schema,
                )

    def record(self, *, repo: str, step: ProposedStep, risk: RiskScore,
               label: float, notes: str | None = None) -> None:
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=self.embed_fn(f"{step.task_class}: {step.description}"),
                payload={
                    "task_class": step.task_class, "description": step.description,
                    "incident": label, "fan_out": risk.fan_out, "risk_score": risk.value,
                    "repo": repo, "notes": notes or "",
                },
            )],
        )

    def historical_rate(self, *, task_class: str, description: str,
                         fan_out: int, top_k: int = 20) -> float:
        hits = self.client.query_points(
            collection_name=self.collection,
            query=self.embed_fn(f"{task_class}: {description}"),
            query_filter=Filter(must=[
                FieldCondition(key="task_class", match=MatchValue(value=task_class))
            ]),
            limit=top_k,
            with_payload=True,
        ).points
        if not hits:
            return 0.2  # cold-start prior
        num = sum(h.score * float(h.payload["incident"]) for h in hits)
        den = sum(h.score for h in hits) or 1.0
        return max(0.0, min(1.0, num / den))

    def recalibrate(self, classifier: RiskClassifier, label: float,
                     iteration: int, max_iterations: int) -> None:
        """Called after a step resolves. Nudges tau, never the weights, since
        the weights encode a modeling choice about which signals matter and
        tau encodes a cost tradeoff the team is entitled to move."""
        if label >= 1.0 and iteration >= max_iterations:
            classifier.config.tau = max(0.35, classifier.config.tau - 0.02)
```

Calibration does not invent a maintainability oracle. It reallocates human attention toward regions where past silence was expensive. Note that `recalibrate` only ever adjusts $\tau$, never $w_v$, $w_f$, or $w_h$ — the weights are a modeling assumption about which signals matter for risk, and I'd rather have a human revisit that assumption deliberately than have it drift silently under the same online-update mechanism that's appropriate for a cost threshold.

## Worked Example: Architectural Decision vs Mechanical Rename

Consider two steps an unconstrained coding agent might treat as "just another patch," run through `RiskClassifier.score`.

**Step A — mechanical rename.** Rename `CustomerDTO` to `CustomerRecord`. Fan-out is 14 files, but every change is an identifier rewrite. A verifier exists (unit tests plus a grep gate for leftover `CustomerDTO`). Historical rate for `task_class="rename"` is low ($h \approx 0.05$):

$$
R_A = 0.45\cdot 0 + 0.30\cdot\min(1, 14/8) + 0.25\cdot 0.05 = 0.3125
$$

With $\tau = 0.55$, Step A routes to **bounded execution** via `BoundedExecutor.run`. High fan-out alone does not escalate when a verifier exists and history is clean.

**Step B — architectural boundary.** "Extract billing into a separate service and leave a façade in the monolith." Planned files look small. No honest verifier exists for "façade will age well." Historical rate for `task_class="boundary"` is high ($h \approx 0.62$):

$$
R_B = 0.45\cdot 1 + 0.30\cdot\min(1, 3/8) + 0.25\cdot 0.62 = 0.7175
$$

Step B routes to `EscalationArtifact.build`, producing a `ReviewPackage` with three questions for the reviewer, including the mandatory one for any step lacking a verifier. A human redirects: keep billing in-process but isolate a module boundary with explicit interfaces. `CheckpointCommit.commit` records the redirected state; `CalibrationStore.record` stores $\ell=0.5$ for this step, which nudges $h(t)$ for `task_class="boundary"` slightly upward the next time a similar step is scored.

The contrast is the thesis `risk-gate` is built to enforce mechanically: **fan-out without verifiers is not the same object as fan-out with verifiers.** Lights-off factories that optimize only for "tests green" systematically promote Step-B work into Step-A paths. The library exists to make that promotion expensive and loud, at the level of a reusable classifier rather than a one-off check somebody has to remember to write into each new project.

## Challenges and Open Problems

**False negatives in the classifier.** Fan-out and verifier-existence are proxies — a one-file change can still be a bad architectural decision, and a ten-file change can be entirely mechanical. Getting the false-negative rate down is an open calibration problem, and it is the one place where a bad call has the same silent-failure characteristic that motivated the whole library. Exhaustion-based reclassification in `BoundedExecutor` mitigates some misses; it does not help when a weak verifier passes cleanly on work it shouldn't have approved.

**Escalation cost.** A deployment that escalates too aggressively reproduces the original bottleneck — a human reviewing everything, with extra library plumbing in between. Whether `CalibrationStore.recalibrate` converges $\tau$ to a stable low escalation rate $E$ for a given codebase, or whether high architectural churn simply requires a permanently higher rate, is an empirical question I don't yet have multi-repo data to answer, since `risk-gate` hasn't run in enough independent codebases yet to say.

**Cold start and checkpoint throughput.** `CalibrationStore` is empty on day one; the $0.2$ prior for unknown classes is a placeholder, not a calibrated number. `CheckpointCommit.commit`-per-step maximizes containment and can thrash git history on high-volume mechanical factories; batching across proven-mechanical sequences re-opens blast radius inside the batch. Adaptive batching conditioned on $R(t)$ is plausible future work for the `checkpoint` module — and a new source of silent accumulation if the conditioner is wrong.

**The training gap remains.** None of this addresses the underlying training gap — `risk-gate` routes around it, at the loop level, for a single team's codebase. If a future model generation acquires a robust sense of maintainability, most of the escalation and calibration modules become redundant. Until a benchmark result makes that credible rather than aspirational, treating verifiability as an explicit, checkable property of each step — enforced by a small library rather than reinvented per project — seems like the more defensible place to put the engineering effort.

## References

- Jimenez, Carlos E. et al. (2023). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* arXiv:2310.06770
- Yang, John et al. (2024). *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*. NeurIPS 2024.
- Shinn, Noah et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023.
- LangGraph Documentation. *Interrupts (Human-in-the-Loop)*. [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- LangGraph Documentation. *Persistence / Checkpointers*. [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- Qdrant. *Qdrant Vector Database Documentation*. [qdrant.tech/documentation](https://qdrant.tech/documentation)
- Fowler, Martin. *Code Smells: Shotgun Surgery*. [martinfowler.com](https://martinfowler.com/bliki/ShotgunSurgery.html)
