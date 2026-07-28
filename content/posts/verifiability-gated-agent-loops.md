---
title: "Verifiability-Gated Loops: An Escalation Contract for Agentic Software Factories"
date: 2026-07-28
description: "Most agentic loops only check what they were told to check. This post proposes a verifiability-gated architecture — risk classification, bounded execution, escalation, checkpoint commits, and calibration — so loops fail loudly when stop conditions cannot measure what matters."
tags: ["agents", "verifiability", "human-in-the-loop", "software-engineering", "agent-loops", "reliability"]
author: "Mihir Inamdar"
showToc: true
---

Coding agents can now run for hours without a human reading a line of what they produce. The natural next question — and the one most teams are asking in mid-2026 — is how far that can go before something breaks that nobody notices until much later. This post is not about prompting technique. It is about a structural gap in how agentic loops are built: most loops only know how to check the thing they were told to check, and nothing tells them when that check isn't measuring what actually matters.

I want to narrow the scope considerably. This post does not cover model training, RLHF, or benchmark design in depth — those are treated only as far as they explain *why* the gap exists. The focus is architectural: given that gap is real, what does a loop need to look like so it fails loudly instead of quietly?

## Table of Contents

- [Background: the loop engineering moment](#background-the-loop-engineering-moment)
- [The verifiability gap](#the-verifiability-gap)
- [A framework for verifiability-gated loops](#a-framework-for-verifiability-gated-loops)
  - [Component one: the risk classifier](#component-one-the-risk-classifier)
  - [Component two: bounded execution](#component-two-bounded-execution)
  - [Component three: the escalation subgraph](#component-three-the-escalation-subgraph)
  - [Component four: checkpoint commits as blast-radius boundaries](#component-four-checkpoint-commits-as-blast-radius-boundaries)
  - [Component five: the calibration loop](#component-five-the-calibration-loop)
- [Related directions](#related-directions)
- [Open problems](#open-problems)

## Background: the loop engineering moment

In mid-2026 a phrase spread quickly through the agent-engineering community: stop prompting your coding agent, start designing the loop that prompts it for you. The idea itself predates the phrase — agents acting inside feedback loops with tools, retries, and stop conditions is not new — but the framing crystallized something practitioners had already converged on independently. A loop, in this sense, is a small system: a trigger, a verification step, some memory, and a stop condition, wrapped around a model.

Loops work extremely well on a specific class of problem: bounded, mechanically checkable work. Triage a failing test, migrate a deprecated API call, fix a lint violation — in each case, there is a program that can tell you, in seconds, whether the loop succeeded. This is also exactly the shape of task that reinforcement learning for coding agents optimizes against: a base commit, an issue description, and a test suite that returns a scalar. Run enough of these episodes and the model gets very good at driving pass/fail signals toward 1.

The trouble starts once you point the same loop at something that doesn't reduce to pass/fail. Several practitioner reports through 2026 — most notably a widely discussed essay from the HumanLayer team, alongside data from Faros AI's code-review research — describe teams that went "lights-off" (no human reading agent-generated code before merge) and later found review quality, incident rates, and bugs-per-developer all trending in the wrong direction. None of this is because the agents were failing their tests. It is because the tests were never checking the thing that eventually cost them time: whether the codebase stayed easy to change.

## The verifiability gap

Call this the **verifiability gap**: the difference between what a loop's stop condition actually measures and what "success" means for the task. For a narrow bug fix, the gap is close to zero — `FAIL_TO_PASS` and `PASS_TO_PASS` genuinely capture whether the fix worked. For an architectural decision — introducing a new service boundary, choosing a data model, deciding where a piece of logic should live — the gap is large, because there is no fast oracle for "will this be easy to extend in three months." The cost of a bad architectural decision surfaces weeks or months later, when someone tries to make a one-line change and discovers it requires touching eleven files. RL cannot optimize against a signal that arrives that late, and neither can a loop's retry logic.

This asymmetry matters because it is invisible from inside the loop. An agent iterating against a test suite has no internal signal that says "you are currently making a decision this test suite cannot evaluate." It will happily converge on a passing, poorly designed solution with the same confidence it shows for a well-designed one. The loop cannot tell the difference, because nothing in its stop condition is *built* to tell the difference.

The natural response — more review agents, more linters, an "adversarial review" pass — raises the floor. It catches obviously bad code. It does not raise the ceiling, because the ceiling is set by what the loop was ever able to check in the first place, and adding a second pass/fail check on top of an already-blind stop condition does not create the missing signal.

## A framework for verifiability-gated loops

If the gap can't be closed by training a better verifier for maintainability (nobody has one yet — more on this below), the loop needs to know when it has hit the gap, and hand the decision to something that can actually evaluate it. That reframes the engineering problem: instead of one loop with one stop condition, build a small graph with a routing decision at its center.

![A five-node graph: task decomposition feeds a risk classifier, which routes to either a bounded execution subgraph or an escalation subgraph, both merging into a checkpoint commit, with a dashed feedback edge back to the classifier.](loop-supervisor-diagram)
*The five components described below, wired as a supervisor graph around a normal agent execution loop.*

### Component one: the risk classifier

Every step gets scored before it runs, not after. The score should be built from checkable signals rather than an LLM's self-assessment, because self-assessment is exactly the thing that fails silently:

- **Verifier existence** — can a deterministic check (test, schema validation, type check) even be written for this step? If the answer is no, that fact alone is the strongest available signal, stronger than any confidence estimate the model could offer about its own plan.
- **Fan-out** — how many files or call sites does the change touch? A static-analysis proxy for the "shotgun surgery" smell: work that ripples across many unrelated places is disproportionately likely to be an architectural decision wearing a bug-fix costume.
- **Historical incident rate** — pulled from mined traces of similar past steps. This is the one signal that improves over time rather than staying fixed, which is what makes the classifier a component of a learning system rather than a static rule engine.

### Component two: bounded execution

Steps that clear the classifier run inside a capped loop with a verifier contract attached — not a general instruction to "write good code," but a Harbor-style bundle of an environment, an instruction, and a scoring function specific to that step. Capping the iteration count matters as much as the verifier itself: an agent that cannot pass its own contract in a small, fixed number of tries should not be allowed to keep grinding on the assumption that one more attempt will fix it. It should be reclassified and routed to escalation. This is the safety net for a classifier that misjudged a step's risk in the first place — verifiability gaps aren't always visible before the work starts.

### Component three: the escalation subgraph

Steps that don't clear the classifier — or that exhaust their retries — get routed here instead of continuing to iterate blind. The subgraph's job is to produce the smallest artifact that lets a human make the decision quickly: a call-stack diff for control-flow changes, a file-tree diff for layout changes, interface signatures for the functions actually at stake. None of this is a full design document; it is the minimum context a reviewer needs to approve, redirect, or reject in a few minutes rather than a few hours. The subgraph then pauses on an interrupt and waits. This is a direct application of human-in-the-loop primitives already available in modern agent frameworks — the contribution here is *routing* to them systematically rather than leaving the decision to whether a developer happens to remember to look.

### Component four: checkpoint commits as blast-radius boundaries

Every merge point between the two branches is also a checkpoint: a known-good state the system can roll back to if a downstream verifier later fails. This turns checkpointing from a resumability feature (survive a crash, resume a session) into a containment feature (bound how much bad work can accumulate before anyone notices). Without this, a single misclassified step doesn't just fail once — it becomes the foundation the next several steps quietly build on top of, and the eventual discovery cost compounds with every step layered on.

### Component five: the calibration loop

Every escalation outcome — approved as-is, redirected, or rejected — and every autonomous failure that had to be caught by the retry cap becomes a labeled example. Mining these examples back into the risk classifier is what keeps the boundary between "loop it" and "escalate it" from being a fixed guess: a team that consistently sees a certain class of database-migration step get rejected at escalation should see the classifier's fan-out threshold for that class tighten automatically, not manually. This closes the system into something closer to a genuine feedback loop rather than a one-time router.

## Related directions

A few efforts point at the same underlying problem from the training side rather than the architecture side. Benchmarks built around long-horizon, multi-PR tasks — scoring code quality with judge models over diffs, or penalizing tests that don't fail against the pre-patch code (a form of mutation testing) — are early attempts to give RL a maintainability signal it currently lacks. None of them close the gap yet: a judge model evaluating design quality is itself bounded by whatever the underlying model already knows about good design, and if it reliably recognized bad design, it would plausibly have avoided producing it in the first place. This is the same asymmetry the loop-level framework above is built to route around rather than solve — it does not try to manufacture a fast oracle for maintainability where none exists; it tries to make sure the absence of one gets escalated instead of ignored.

Separately, teams building automated remediation pipelines against production systems have converged on a related but distinct pattern: keeping dangerous authority (write credentials, deployment triggers) in a deterministic layer the agent never directly holds, with audit trails that don't depend on the agent's own account of what it did. That is a permission-boundary problem rather than a verifiability problem, but it rhymes with the escalation subgraph above — both are ways of admitting that some decisions cannot be delegated to the loop, and building the graph so that admission is structural rather than a matter of the developer remembering to be careful.

## Open problems

The classifier is the weakest link in this design, and it's worth being direct about that. Fan-out and verifier-existence are reasonable proxies, but they are proxies — a change that touches one file can still be a bad architectural decision, and a change that touches ten files can be entirely mechanical (a rename, a dependency bump). Getting the false-negative rate down — steps that should have escalated but didn't — is an open calibration problem, and it is the one place where a bad call has the same silent-failure characteristic that motivated this whole framework in the first place.

There is also a cost question this post hasn't addressed: escalation is not free. A system that escalates too aggressively reproduces the original bottleneck it was trying to route around — a human reviewing everything, just with extra steps in between. The interesting long-run question is whether the calibration loop in component five converges toward a stable, low escalation rate for a given codebase, or whether codebases with high genuine architectural churn simply need a permanently higher rate, and no amount of trace mining changes that. I don't have data on this yet; it would need runs across teams with materially different rates of architectural change to separate the two possibilities.

Finally, none of this addresses the underlying training gap — it routes around it, at the loop level, for a single team's codebase. If a future model generation genuinely acquires a robust sense of maintainability, most of the escalation subgraph becomes redundant. Until there is a benchmark result that makes that credible rather than aspirational, treating verifiability as an explicit, checkable property of each step — rather than an assumption baked into the loop — seems like the more defensible place to put the engineering effort.
