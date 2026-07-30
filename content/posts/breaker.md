---
title: "Breaker: A Session-Local Loop Guard for Coding Agents"
date: 2026-07-12
description: "A coding agent chasing a failing test rarely repeats the exact same tool call twice when it's stuck — it paraphrases its own unsuccessful attempt, which defeats exact-match and simple-diff loop detectors while still being the same unproductive action repeated. Breaker is a pip-installable, session-local, embedding-based loop guard you wrap around an existing tool-calling loop; it checks the meaning of consecutive actions and outcomes, backed by an ephemeral, session-scoped Qdrant collection rather than a persistent corpus."
tags: ["agents", "coding-agents", "reliability", "cost-optimization", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

An autonomous coding agent chasing a failing test, a build error, or an elusive bug can get stuck in a loop: repeatedly attempting near-identical fixes, re-reading the same file without changing approach, or oscillating between two edits that each break what the other one fixed. This burns tokens and wall-clock time for many turns before a human notices, or before a blunt fixed-iteration cap eventually kicks in — sometimes too late, after real spend; sometimes too early, cutting off a legitimately converging fix that needed two more turns. This is a widely-observed operational reality of running coding agents unattended on real repositories, not something tied to a single dated incident, so I'm not going to anchor this post on one. This post is the spec for **Breaker**, a session-local loop guard that wraps around an existing agent's tool-calling loop and evaluates incrementally after every tool call — as opposed to cross-session learning from past failures (a different, complementary problem) or hard iteration ceilings (a blunt instrument this post argues Breaker complements, not replaces). Required background: basic familiarity with how agent frameworks structure a tool-calling loop and with the general concept of a recursion or iteration limit as a safety net.

## Table of Contents

1. [The Loop Problem in Production Coding Agents](#the-loop-problem-in-production-coding-agents)
2. [Why Exact-Match and Textual-Similarity Checks Both Fail](#why-exact-match-and-textual-similarity-checks-both-fail)
3. [Designing Breaker: A Session-Local Loop Guard](#designing-breaker-a-session-local-loop-guard)
4. [Qdrant as the Side-Piece: Ephemeral and Session-Scoped, Not a Persistent Corpus](#qdrant-as-the-side-piece-ephemeral-and-session-scoped-not-a-persistent-corpus)
5. [Component One: The Action-Outcome Signature](#component-one-the-action-outcome-signature)
6. [Component Two: The Ephemeral Per-Session Collection](#component-two-the-ephemeral-per-session-collection)
7. [Component Three: The Rolling-Window Similarity Check](#component-three-the-rolling-window-similarity-check)
8. [Defining the Trigger Condition Precisely](#defining-the-trigger-condition-precisely)
9. [What Happens When Breaker Fires: an Escalation Ladder](#what-happens-when-breaker-fires-an-escalation-ladder)
10. [Component Four: Escalation Logic](#component-four-escalation-logic)
11. [Breaker Complements a Fixed Cap, It Doesn't Replace One](#breaker-complements-a-fixed-cap-it-doesnt-replace-one)
12. [Installing and Wiring Breaker Into the Tool-Calling Loop](#installing-and-wiring-breaker-into-the-tool-calling-loop)
13. [A Quick Cost Model for When Breaker Pays for Itself](#a-quick-cost-model-for-when-breaker-pays-for-itself)
14. [Calibrating k, m, and the Thresholds](#calibrating-k-m-and-the-thresholds)
15. [Worked Example](#worked-example)
16. [Producing a Structured Stall Summary](#producing-a-structured-stall-summary)
17. [Session Scope With Parallel Tool Calls and Sub-Agents](#session-scope-with-parallel-tool-calls-and-sub-agents)
18. [Challenges and Open Problems](#challenges-and-open-problems)
19. [References](#references)

## The Loop Problem in Production Coding Agents

A coding agent working through a failing test tries something, observes the result, and tries again. That loop is the entire mechanism by which these agents get anything done — the failure mode isn't looping itself, it's looping *unproductively*: the same underlying attempt, dressed differently each time, against a failure that isn't actually moving. Anyone who has run a coding agent unattended against a real repository for more than a few dozen turns has seen a version of this: five attempts at "fix the import," each syntactically distinct, all failing with the identical `ImportError`; a function edited back and forth between two states because each edit "fixes" a symptom the other edit reintroduces; a file re-read four times in a row with no new action taken in between, as if re-reading might reveal something a fresh read of unchanged content cannot.

A few recognizable shapes cover most of what shows up in practice: **paraphrased retries**, where each attempt is textually distinct but semantically identical to the last (the `fix_import` example above); **thrashing**, where the agent oscillates between two states, each edit "fixing" a symptom the previous edit reintroduced, so the net diff across several turns is close to zero even though each individual turn looks like forward motion; **unproductive re-reading**, where the agent re-reads the same file or re-runs the same diagnostic command repeatedly with no new action taken in between, as though a second look at unchanged content might reveal something the first look didn't; and **strategy tunnel vision**, where the agent keeps refining variations of one approach well past the point where the approach itself, not its execution, is the problem. All four defeat exact-match detection for the same underlying reason — the literal tool call or diff differs each time — while all four are, in substance, the same unproductive pattern repeated.

This isn't a hypothetical concern the framework maintainers are unaware of. A real, dated GitHub issue against LangGraph's agent runtime (`langchain-ai/langgraph`, issue #6731, opened early 2026) reports exactly this pattern in production: an agent generating SQL against Databricks kept "generating slight variations of the same broken SQL" in response to a `REQUIRES_SINGLE_PART_NAMESPACE` error it couldn't reason its way past, looping until the graph's `recursion_limit` finally killed it. The maintainer's diagnosis is worth quoting directly, because it states the current state of the art plainly: an earlier version of LangChain's `create_agent` had a heuristic for "N identical tool calls → terminate," and the newer `create_react_agent` rewrite dropped it, "leaving only `recursion_limit` as the last line of defense." The suggested fix at the time was `ToolCallLimitMiddleware`, capping `max_consecutive_tool_calls` and `max_total_tool_calls` — a blunt count-based ceiling, not a check on whether those calls were actually converging toward anything.

That's the state most production coding-agent deployments are in today: either no loop-specific check at all, relying purely on a fixed iteration or recursion cap as the backstop, or — where a smarter check exists — one built on exact repetition, which the next section argues is close to useless against how a stuck agent actually behaves.

## Why Exact-Match and Textual-Similarity Checks Both Fail

The naive loop detector — stop after $N$ identical tool calls, same tool name and same arguments byte-for-byte — sounds reasonable and almost never fires on a genuinely stuck agent, for a simple reason: a stuck agent rarely repeats itself exactly. It paraphrases its own unsuccessful attempt. Trying `fix_import("from foo import bar")`, failing, then trying `fix_import(module="foo", name="bar")` is, in every way that matters for productivity, the identical action — the agent is doing the same thing it just tried, expressed with different argument shapes — but it will never trigger an exact-match check, because the call is textually and structurally different from the one before it. The same applies to a code edit: rewriting the same function with a renamed local variable, reordered statements that have no semantic effect, or a cosmetically different but functionally equivalent diff each time defeats exact matching completely while representing zero actual progress.

A step up — comparing the raw text similarity of consecutive tool-call arguments, via edit distance or token overlap — does better against pure copy-paste repetition and still fails against the common case, because trivial variable renaming, whitespace differences, or reordering keyword arguments changes the surface text substantially while leaving the underlying action unchanged. Two calls that are semantically identical can have a large edit distance; two calls that are semantically unrelated (different files, different fixes, coincidentally similar phrasing) can have a small one. Textual similarity is a proxy for semantic similarity that breaks down exactly in the direction that matters here: an agent avoiding detection by accident, simply because paraphrasing is what generating slightly-different-sounding attempts naturally produces, needs no adversarial intent to defeat a text-similarity check.

What's actually needed is a check on the *meaning* of consecutive actions, not their literal text — embedding-based semantic similarity, which places `fix_import("from foo import bar")` and `fix_import(module="foo", name="bar")` close together in vector space despite near-zero token overlap, because they express the same underlying intent. And it needs to be evaluated incrementally, after every single tool call, scoped to the one session currently running — not as a periodic batch check every $N$ turns (which can let a loop run for up to $N-1$ turns past the point it became detectable) and not as a global check across every session an agent framework has ever run (which conflates "is this session stuck" with an entirely different question about historical failure patterns across unrelated tasks).

Laying the three approaches side by side against the same paraphrased-retry example makes the gap concrete:

| Approach | Catches exact repeats | Catches `fix_import("from foo import bar")` → `fix_import(module="foo", name="bar")` | Catches cosmetic diff variation (renamed vars, reordered statements) | Main failure mode |
|---|---|---|---|---|
| Exact match (tool + args) | Yes | No | No | Misses nearly every real stuck-agent pattern, which rarely repeats verbatim |
| Raw text similarity (edit distance / token overlap) | Yes | Partially, and unreliably | No | Fooled by trivial renaming; also flags coincidentally similar but unrelated calls |
| Semantic embedding similarity | Yes | Yes | Yes | Requires a reasonable outcome-summarization step to avoid over-firing on normal retries (addressed below) |

The last column matters as much as the first two — a check that catches everything a loop guard should catch but also fires on every single retry, including the one legitimate retry that follows a normal, expected first failure, is not an improvement on a blunt iteration cap, just a differently-blunt one. The sustained-similarity condition developed later in this post exists specifically to keep the semantic check from over-firing on that normal case.

## Designing Breaker: A Session-Local Loop Guard

The guard I want sits directly in the tool-call loop: after every tool call completes, construct a compact signature of what the agent just attempted and what happened as a result, compare that signature against the last handful of signatures from the *same session*, and raise a signal when several consecutive actions are both similar to each other and failing to produce a meaningfully different outcome.

Two things distinguish this from a single retry, which is completely normal agent behavior and should never be flagged on its own: the check only fires on a **sustained run** of several consecutive high-similarity turns, not any single repeated attempt in isolation, and it requires high similarity on **both** the attempted action and the resulting outcome — an agent whose fixes are superficially similar-looking (same file, same function) but whose outcomes are genuinely changing turn over turn (a different assertion failing, a different line number, eventual success) is converging, not looping, and should not trigger the guard even though its edits might look repetitive to a naive text diff.

## Qdrant as the Side-Piece: Ephemeral and Session-Scoped, Not a Persistent Corpus

Every other Qdrant-backed design on this blog uses a persistent, growing collection: a corpus of known injection patterns, a cache of past query results, an incident registry that gets more useful the longer it accumulates history. This one is different by design, and it's worth being explicit about why, because the instinct to make everything a persistent, cross-session store is exactly wrong here.

The loop guard's entire job is to answer one question — "is *this* session, right now, going in circles" — using only the last handful of actions from *that* session. It has no need for, and actively should not have, visibility into what happened in other sessions on other tasks: mixing in historical data from unrelated runs doesn't sharpen the detection of an in-progress loop, it dilutes the rolling window's signal with irrelevant neighbors and reintroduces exactly the kind of cross-session conflation the previous section argued against. Building a persistent library of past failures for future retrieval — so that a *new* session can recognize "this looks like a fix pattern that failed on a similar bug three weeks ago" — is a genuinely different and complementary problem, closer in spirit to the calibration loop described in [Verdict](/posts/verdict/) on this blog, which mines labeled outcomes across many steps and many runs into a classifier that improves over time. That's a *cross-session learning* problem. This is an *in-session, right-now* problem, and conflating the two designs would make both worse: the loop guard doesn't need to remember anything past this session's end, and a cross-session incident store shouldn't be repurposed to also gate a live tool-calling loop turn-by-turn.

Practically, this means the collection backing the guard is created fresh at session start and discarded at session end — either a genuinely ephemeral named collection dropped on session close, or Qdrant's in-memory mode (`QdrantClient(location=":memory:")`) when the agent framework runs one session per process, which avoids any persistence step at all. Either way, the collection never needs to hold more than a handful of points at once, since only the last $k$ actions matter.

| | Persistent corpus (e.g. the incident memory in Verifiability-Gated Loops) | Session-local loop guard (this post) |
|---|---|---|
| Lifetime | Grows indefinitely across many runs | Created at session start, discarded at session end |
| Size | Unbounded, accumulates over time | Bounded at $k$ points, never grows |
| Query scope | All historical steps, filtered by task class or similarity | Only the current session's last $k$ actions |
| Purpose | Learn from past outcomes to improve future risk classification | Detect an in-progress stall before it wastes further budget |
| Failure if scoped wrong | N/A — cross-session accumulation is the point | Diluted signal, false negatives on real loops, false positives from unrelated sessions |

Getting the row order backwards — treating the loop guard's collection as something worth keeping around, or treating the incident-memory corpus as something to reset per session — breaks each design for the problem it's actually solving, which is the strongest argument for keeping the two intentionally separate rather than merging them into one general-purpose "agent memory" collection that tries to serve both purposes at once. A team building both should feel free to run them against the same underlying Qdrant deployment — there's no operational reason they need separate infrastructure — as long as the collections themselves stay logically distinct: one keyed by session and bounded in size, the other keyed by task class or embedding similarity and intentionally unbounded.

## Component One: The Action-Outcome Signature

Each turn produces two things worth embedding separately: what the agent attempted, and what happened as a result. Keeping these as two independent vectors — rather than one embedding of a concatenated blob — is what lets the trigger condition later check action-similarity and outcome-similarity independently, which turns out to matter a great deal for avoiding false positives on legitimately converging attempts.

```python
import json
import re


def normalize_args(tool_name: str, args: dict) -> str:
    """Canonicalize call arguments so cosmetic variation — key order,
    quoting, incidental whitespace, keyword vs. positional shape — doesn't
    make two functionally identical calls look novel to an embedding model
    any more than it would to a human skimming both calls side by side."""
    normalized = {}
    for key in sorted(args):
        value = args[key]
        if isinstance(value, str):
            value = re.sub(r"\s+", " ", value.strip())
        normalized[key] = value
    return f"{tool_name}({json.dumps(normalized, sort_keys=True)})"


def summarize_outcome(result: dict) -> str:
    """Compress a tool result to the signal that matters for loop detection:
    whether the failure mode changed, not the full stdout/stderr dump. A poor
    summary here — one that can't distinguish 'still failing the same way'
    from 'failing differently now' — undermines the whole mechanism, since
    the outcome half of the similarity check is only as good as this text."""
    if result.get("success"):
        return "success"
    error_type = result.get("error_type", "unknown_error")
    error_detail = (result.get("error_message") or "")[:200]
    return f"failed:{error_type}:{error_detail}"


def action_signature(tool_name: str, args: dict, result: dict) -> tuple[str, str]:
    return normalize_args(tool_name, args), summarize_outcome(result)
```

`summarize_outcome` deliberately keeps only the error type and a short slice of the message, discarding line numbers, timestamps, and other high-entropy detail that would make two occurrences of the *same* underlying failure look different for reasons that have nothing to do with whether the agent is actually making progress. This is a lossy compression on purpose — the goal is "did the failure mode change," not "reproduce the exact stack trace."

## Component Two: The Ephemeral Per-Session Collection

The collection uses two named vectors, `action` and `outcome`, so both halves of a turn's signature can be queried independently against the rolling window without maintaining two separate collections.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class SessionLoopGuard:
    def __init__(self, session_id: str, embed_fn, window_size: int = 6,
                 sustain_turns: int = 3, action_threshold: float = 0.85,
                 outcome_threshold: float = 0.85, in_memory: bool = True):
        self.client = (
            QdrantClient(location=":memory:") if in_memory
            else QdrantClient(url="http://localhost:6333")
        )
        self.collection = f"loop_guard_{session_id}"
        self.embed_fn = embed_fn
        self.window_size = window_size
        self.sustain_turns = sustain_turns
        self.action_threshold = action_threshold
        self.outcome_threshold = outcome_threshold
        self.turn = 0
        self.point_ids: list[int] = []
        self.high_similarity_streak = 0

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "action": VectorParams(size=768, distance=Distance.COSINE),
                "outcome": VectorParams(size=768, distance=Distance.COSINE),
            },
        )

    def close(self):
        """Called at session end — or skipped entirely in in-memory mode,
        where the whole client (and its state) is discarded with the process."""
        self.client.delete_collection(self.collection)
```

`window_size` bounds the collection at a handful of points — never more than $k$ — which is the practical consequence of the design argument in the previous section: this is a small, rolling window, not a corpus that's meant to grow.

## Component Three: The Rolling-Window Similarity Check

After every tool call, the guard embeds the new turn's signature, queries the existing window for similarity on both vectors *before* inserting the new point (so the new turn is compared only against prior turns, not against itself), then evicts the oldest point once the window exceeds its bound.

```python
from qdrant_client.models import PointStruct


# Continuing the SessionLoopGuard class defined above.
class SessionLoopGuard:
    def record_and_check(self, tool_name: str, args: dict, result: dict) -> dict:
        action_text, outcome_text = action_signature(tool_name, args, result)
        action_vec = self.embed_fn(action_text)
        outcome_vec = self.embed_fn(outcome_text)
        self.turn += 1

        sim_action, sim_outcome = 0.0, 0.0
        if self.point_ids:
            action_hits = self.client.query_points(
                collection_name=self.collection, query=action_vec,
                using="action", limit=len(self.point_ids),
            ).points
            outcome_hits = self.client.query_points(
                collection_name=self.collection, query=outcome_vec,
                using="outcome", limit=len(self.point_ids),
            ).points
            sim_action = sum(p.score for p in action_hits) / len(action_hits)
            sim_outcome = sum(p.score for p in outcome_hits) / len(outcome_hits)

        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=self.turn,
                vector={"action": action_vec, "outcome": outcome_vec},
                payload={
                    "turn": self.turn,
                    "action_text": action_text,
                    "outcome_text": outcome_text,
                },
            )],
        )
        self.point_ids.append(self.turn)
        if len(self.point_ids) > self.window_size:
            stale_id = self.point_ids.pop(0)
            self.client.delete(collection_name=self.collection, points_selector=[stale_id])

        is_loopy_turn = (
            sim_action >= self.action_threshold and sim_outcome >= self.outcome_threshold
        )
        self.high_similarity_streak = (
            self.high_similarity_streak + 1 if is_loopy_turn else 0
        )

        return {
            "turn": self.turn,
            "sim_action": round(sim_action, 3),
            "sim_outcome": round(sim_outcome, 3),
            "is_loopy_turn": is_loopy_turn,
            "streak": self.high_similarity_streak,
            "loop_detected": self.high_similarity_streak >= self.sustain_turns,
        }
```

`sim_action` and `sim_outcome` here are each the mean similarity of the current turn against every point still in the window, not just the single most recent turn — averaging over the window smooths out a one-off coincidental match and better reflects "does this turn resemble the recent pattern of attempts," which is closer to what the loop signal is actually about than a pairwise comparison to the immediately preceding turn alone.

## Defining the Trigger Condition Precisely

Writing the trigger condition out explicitly makes clear what has to be jointly true before the guard fires. Let $a_t$ and $o_t$ be the action and outcome embeddings at turn $t$, and let $W_t = \{t-k, \dots, t-1\}$ be the current rolling window of the last $k$ turns. Define:

$$
\bar{s}_a(t) = \frac{1}{|W_t|}\sum_{j \in W_t} \cos(a_t, a_j), \qquad
\bar{s}_o(t) = \frac{1}{|W_t|}\sum_{j \in W_t} \cos(o_t, o_j)
$$

A turn is flagged as loopy when both similarities clear their thresholds:

$$
L(t) = \mathbb{1}\!\left[\bar{s}_a(t) \ge \tau_a \; \land \; \bar{s}_o(t) \ge \tau_o\right]
$$

and the guard actually fires only once loopy turns have run consecutively for $m$ turns (`sustain_turns` in the code above):

$$
\text{fire}(t) = \mathbb{1}\!\left[\forall\, i \in \{0, \dots, m-1\}: L(t-i) = 1\right]
$$

Requiring the outcome condition, not just the action condition, is what keeps a legitimately converging agent from tripping the guard: an agent making five different edits to the same function, each targeting a different sub-issue, can easily have a high $\bar{s}_a(t)$ (same file, same function, structurally similar diffs) while $\bar{s}_o(t)$ stays low, because the assertion failing, the error type, or the test outcome genuinely changes turn over turn. $L(t)$ only fires when *both* conditions hold, and $\text{fire}(t)$ only escalates that into an actual guard trigger once it's been true for $m$ consecutive turns rather than a single coincidental spike — a single retry after one failure, which is normal and expected, produces at most one loopy turn and never reaches $\text{fire}(t)=1$ unless the pattern actually persists.

## What Happens When Breaker Fires: an Escalation Ladder

A firing guard should not default to a hard stop. Aborting a task at the first sign of repetition wastes legitimate near-convergent work as readily as it stops a genuine loop, and the whole point of scoping this to a sustained-similarity condition rather than a single retry was to avoid exactly that overcorrection. A graduated response, in increasing order of severity and cost:

1. **Force articulation.** Ask the agent to state, in one or two sentences, specifically what will differ about its next attempt and why that difference should change the outcome — before it's allowed to take the next action. This is a cheap circuit breaker: an agent that genuinely has a new idea can usually articulate it concretely, and an agent that's actually stuck usually cannot produce anything more specific than a restatement of what it already tried, which is itself useful evidence that the loop is real.
2. **Escalate to a different strategy or tool.** If articulation fails, or the guard fires again shortly after a forced articulation that didn't actually change anything, switch approach explicitly — a different tool, a broader read of surrounding context the agent hasn't consulted, or an instruction to step back and reconsider the diagnosis rather than the fix.
3. **Pause for human confirmation.** If the strategy switch doesn't break the pattern within a further handful of turns, stop autonomous execution and surface the situation to a human, with the recent action-outcome history attached so the human isn't starting from zero.
4. **Abort and summarize.** As a last resort, end the task and produce a structured summary of what was attempted and why each attempt failed — so whoever picks up the task next, human or agent, doesn't have to rediscover the same dead ends from scratch.

## Component Four: Escalation Logic

Tracking how many times the guard has already fired within a session — not just whether it's currently firing — is what lets the response climb the ladder rather than repeating the same first-level intervention indefinitely.

```python
from enum import Enum


class EscalationLevel(str, Enum):
    NONE = "none"
    ARTICULATE_DIFFERENCE = "articulate_difference"
    SWITCH_STRATEGY = "switch_strategy"
    HUMAN_CONFIRM = "human_confirm"
    ABORT = "abort"


LADDER = [
    EscalationLevel.ARTICULATE_DIFFERENCE,
    EscalationLevel.SWITCH_STRATEGY,
    EscalationLevel.HUMAN_CONFIRM,
    EscalationLevel.ABORT,
]


def next_escalation(loop_state: dict, prior_escalations_this_session: int) -> EscalationLevel:
    if not loop_state["loop_detected"]:
        return EscalationLevel.NONE
    rung = min(prior_escalations_this_session, len(LADDER) - 1)
    return LADDER[rung]


ARTICULATION_PROMPT = (
    "Your last several attempts have been highly similar to each other and "
    "none has resolved the failure. Before trying again: state in one "
    "sentence what specifically will be different about your next attempt, "
    "and why you expect that difference to change the outcome. If you "
    "cannot state a concrete difference, say so explicitly rather than "
    "guessing at one."
)
```

A session that has already been through `ARTICULATE_DIFFERENCE` once and fires again shortly after climbs straight to `SWITCH_STRATEGY` rather than asking the agent to articulate a difference for a second time — an agent that failed to produce a real answer the first time is unlikely to produce one on request a second time, and repeating the same escalation step is itself a small loop worth avoiding.

## Breaker Complements a Fixed Cap, It Doesn't Replace One

Everything above argues that a fixed iteration or recursion cap is a poor *primary* loop detector — it's blunt, and it fires only after the fact, at a count chosen without any reference to whether the session is actually stuck. None of that is an argument for removing the cap. A fixed ceiling is what catches the guard's own failure modes: an embedding service outage that silently degrades `embed_fn` to returning constant or near-random vectors, a misconfigured threshold that never fires, a session whose behavior doesn't resemble any of the patterns this design anticipates. The LangGraph issue cited earlier makes the layering explicit without necessarily framing it this way: `recursion_limit` was already "the last line of defense" once the old identical-tool-call heuristic was removed, and `ToolCallLimitMiddleware`'s count-based ceiling is a reasonable *second* line, not a replacement for a semantic check — it's simply the layer that's cheap to implement and independent of any embedding infrastructure, which is exactly the property you want in a backstop.

The practical arrangement, then, is two layers: the semantic guard in this post as the primary, fast-acting detector that catches most loops well before they'd exhaust a generously-sized cap, and a fixed recursion or iteration limit set generously enough to rarely bind in practice, existing purely to bound worst-case cost when the primary layer fails silently for any reason. Neither layer alone is sufficient — the fixed cap alone reproduces the too-late/too-early problem this post opened with, and the semantic guard alone has no protection against its own infrastructure failing quietly.

## Installing and Wiring Breaker Into the Tool-Calling Loop

Breaker ships as a single pip-installable package built around `SessionLoopGuard` and the escalation ladder:

```bash
pip install breaker-agents
```

```
breaker/
├── signature.py     # action-outcome signature construction
├── store.py           # ephemeral, session-scoped Qdrant collection
├── guard.py             # SessionLoopGuard, record_and_check()
└── escalation.py           # EscalationLevel, next_escalation(), ARTICULATION_PROMPT
```

Breaker is only useful if something actually calls it after every tool invocation and acts on the result — a design that lives in a document but not in the agent's actual control flow prevents nothing. A minimal integration around a generic tool-calling loop:

```python
from breaker.guard import SessionLoopGuard
from breaker.escalation import EscalationLevel, next_escalation, ARTICULATION_PROMPT

def run_agent_session(agent, task, tools, embed_fn, session_id: str):
    guard = SessionLoopGuard(session_id=session_id, embed_fn=embed_fn)
    escalations_this_session = 0
    trace: list[dict] = []

    try:
        while not agent.is_done(task):
            tool_name, args = agent.propose_next_action(task, trace)
            result = tools[tool_name](**args)
            trace.append({"tool": tool_name, "args": args, "result": result})

            loop_state = guard.record_and_check(tool_name, args, result)
            level = next_escalation(loop_state, escalations_this_session)

            if level is EscalationLevel.NONE:
                continue

            escalations_this_session += 1
            if level is EscalationLevel.ARTICULATE_DIFFERENCE:
                answer = agent.ask(ARTICULATION_PROMPT, context=trace[-guard.window_size:])
                trace.append({"escalation": level.value, "response": answer})
            elif level is EscalationLevel.SWITCH_STRATEGY:
                agent.force_strategy_change(reason="sustained low-progress loop detected")
            elif level is EscalationLevel.HUMAN_CONFIRM:
                if not agent.request_human_confirmation(trace[-guard.window_size:]):
                    break
            elif level is EscalationLevel.ABORT:
                summary = build_stall_summary(trace, guard)
                agent.abort_with_summary(summary)
                break
    finally:
        guard.close()

    return trace
```

The escalation counter resets naturally at the top of the ladder only when the guard stops firing — a session that breaks out of its loop after a forced articulation simply never triggers `next_escalation` again above `NONE`, and `escalations_this_session` never needs to be decremented, since the point of the ladder is to make each subsequent intervention within one stuck episode progressively stronger, not to forgive past escalations mid-session.

## A Quick Cost Model for When Breaker Pays for Itself

The guard itself isn't free — an embedding call and a couple of small Qdrant queries on every tool call adds latency and a marginal compute cost to every single turn, whether or not a loop is ever detected. Whether that overhead is worth paying comes down to a simple comparison: the guard's per-turn cost against the expected savings from catching loops earlier than a fixed iteration cap would.

For a session with per-turn cost $c$ (tokens plus tool execution), a fixed iteration cap of $N$, and a loop that — left unchecked — would otherwise run to the cap before being killed, the wasted spend under a pure fixed-cap policy is roughly the turns spent between when the loop actually became detectable (turn $t^*$) and when the cap finally fires:

$$
\text{waste}_{\text{cap-only}} = (N - t^*) \cdot c
$$

Under the guard, assuming it fires at $t^* + m$ (after $m$ sustained loopy turns) and the escalation ladder resolves the situation within a small constant number of additional turns $\delta$ rather than running to $N$:

$$
\text{waste}_{\text{guarded}} = m \cdot c_{\text{guard-overhead}} + \delta \cdot c
$$

where $c_{\text{guard-overhead}}$ is the small marginal cost of the embedding calls and Qdrant queries added to every turn (guard overhead applies whether or not a loop occurs, so in practice it should be amortized over the full session length, not just the loop's duration — the formula above isolates just the loop episode for clarity). The guard is worth deploying whenever $\text{waste}_{\text{guarded}} \ll \text{waste}_{\text{cap-only}}$, which holds comfortably whenever $t^*$ sits well below $N$ — exactly the common case a fixed cap handles badly, since a cap tuned generously enough to avoid cutting off legitimate multi-step convergence (a large $N$) is, by the same generosity, slow to catch a loop that stalls out early in a long session.

Putting illustrative numbers to it: a session with per-turn cost $c \approx \$0.08$ (a moderate tool-call-plus-reasoning turn on a mid-sized coding task), a generously-set recursion cap of $N=60$, a loop that becomes detectable at $t^*=12$, and a guard with $m=3$, $\delta=4$ (three sustained loopy turns to fire, four more turns for the escalation ladder to resolve it):

$$
\text{waste}_{\text{cap-only}} = (60 - 12) \cdot 0.08 = \$3.84
$$

$$
\text{waste}_{\text{guarded}} \approx 3 \cdot c_{\text{guard-overhead}} + 4 \cdot 0.08 \approx \$0.02 + \$0.32 = \$0.34
$$

(taking $c_{\text{guard-overhead}}$ — one embedding call plus two small Qdrant queries per turn — as roughly a cent, an order of magnitude below the cost of a full reasoning-plus-tool-execution turn). An order-of-magnitude reduction in wasted spend per stalled session, for a per-turn overhead measured in fractions of a cent applied to every turn whether or not a loop ever occurs — the arithmetic favors the guard heavily whenever loops are a real, recurring feature of a deployment's workload, and is closer to a wash on a workload where loops are already rare, since the overhead is paid on every turn but the savings only materialize on the (hopefully small) fraction of sessions that actually stall.

## Calibrating k, m, and the Thresholds

None of $k$ (`window_size`), $m$ (`sustain_turns`), $\tau_a$, or $\tau_o$ have a defensible universal default — they depend on how repetitive legitimate work looks in a given codebase and how noisy the embedding model's similarity scores are on short, code-heavy text. A practical calibration pass, run against a sample of past sessions before turning the guard on in blocking mode:

1. **Replay past sessions offline** with the guard in observe-only mode (compute `sim_action`, `sim_outcome`, and whether `loop_detected` would have fired, without actually intervening), against a mix of sessions already known to have looped and sessions known to have converged normally.
2. **Check the false-positive rate on known-good sessions first**, the same discipline any threshold-based screener needs: a guard that fires on legitimately converging multi-step fixes is worse than no guard at all, since it burns the same articulation/escalation overhead on productive work and trains whoever operates the agent to ignore its warnings.
3. **Check detection latency on known-loop sessions** — how many turns into the loop does `fire(t)` actually trigger, relative to how many turns a fixed iteration cap would have let the loop run. This is the number that justifies the guard's overhead: if it fires meaningfully earlier than the cap would have, on average, across the sample, it's paying for itself per the cost model above.
4. **Adjust $\tau_a$ and $\tau_o$ independently** rather than moving them together — a guard that's too trigger-happy on converging work usually needs $\tau_o$ raised (demanding a tighter outcome match before considering a turn loopy) more than it needs $\tau_a$ raised, since action similarity is expected to be high in a narrow codebase even for legitimate work.

Illustrative figures from a calibration pass structured this way, representative of the shape reported for similarity-threshold tuning in adjacent screening problems rather than a specific measured deployment:

| Threshold setting | False-positive rate on converging sessions | Mean detection latency vs. fixed cap |
|---|---|---|
| $\tau_a=0.75$, $\tau_o=0.75$ (loose) | ~18% — flags several legitimately converging multi-step fixes | Fires very early, often within 2–3 turns of a real stall |
| $\tau_a=0.85$, $\tau_o=0.85$ (this post's defaults) | ~4% | Fires within 3–5 turns of a real stall |
| $\tau_a=0.92$, $\tau_o=0.92$ (tight) | <1% | Fires late, sometimes not meaningfully earlier than a generous fixed cap |

The middle row is a reasonable starting point precisely because the tails are each bad in a different way: too loose wastes the escalation ladder's credibility on false alarms, too tight reproduces the fixed-cap problem this guard exists to improve on. Neither tail is a correctness bug — it's a tuning problem specific to whatever codebase and task distribution the guard is actually running against, which is why step 1 above insists on calibrating against real past sessions rather than shipping the defaults from this post unexamined.

## Worked Example

Consider an agent fixing a failing test with assertion `assert calc_total(items) == 5`, currently returning `3`. Five consecutive attempts, all touching the same function, with `window_size=4`, `sustain_turns=3`, `action_threshold=0.85`, `outcome_threshold=0.85`:

| Attempt | Action | Outcome | $\bar{s}_a$ | $\bar{s}_o$ | Loopy? | Streak |
|---|---|---|---|---|---|---|
| 1 | Fix off-by-one: change loop bound from `range(n)` to `range(n+1)` | `AssertionError: expected 5, got 3` | — | — | — | 0 |
| 2 | Adjust the same loop bound to `range(0, n+1)` | `AssertionError: expected 5, got 3` | 0.74 | 0.98 | No (below $\tau_a$) | 0 |
| 3 | Rewrite the loop as `for i in range(len(items)+1)` | `AssertionError: expected 5, got 3` | 0.86 | 0.98 | Yes | 1 |
| 4 | Rename loop variable, restructure as a `while` loop with equivalent bound | `AssertionError: expected 5, got 3` | 0.89 | 0.99 | Yes | 2 |
| 5 | Revert to attempt 3's structure with a cosmetic comment added | `AssertionError: expected 5, got 3` | 0.91 | 0.99 | Yes | 3 → **fires** |

The guard doesn't fire on attempt 3's first loopy turn — it takes three consecutive loopy turns to reach `sustain_turns=3`, which is exactly the point: a single high-similarity retry after a normal first attempt is unremarkable, and it's the *run* of three in a row, each still failing with the identical `AssertionError`, that constitutes real evidence of a stall. Every attempt is superficially different code, and a naive text-diff check would see five distinct patches; the semantic signature sees the same off-by-one confusion, argued five different ways, against an outcome that never once changed.

Contrast that with a genuinely converging agent working the same bug, attempts that also touch `calc_total` repeatedly but move the failure each time:

| Attempt | Action | Outcome | $\bar{s}_a$ | $\bar{s}_o$ | Loopy? | Streak |
|---|---|---|---|---|---|---|
| 1 | Fix off-by-one in the loop bound | `AssertionError: expected 5, got 3` | — | — | — | 0 |
| 2 | Fix a truncating integer division used inside the loop | `AssertionError: expected 5, got 4` | 0.79 | 0.41 | No | 0 |
| 3 | Fix rounding applied before summation | `AssertionError: expected 5.0, got 4.99` | 0.81 | 0.33 | No | 0 |
| 4 | Add explicit float-to-int conversion at the return statement | Test passes for the base case; new failure on empty input: `AttributeError: 'NoneType' has no len` | 0.77 | 0.22 | No | 0 |
| 5 | Add an empty-list guard at the top of `calc_total` | `success` | 0.68 | 0.15 | No | 0 |

Here $\bar{s}_a$ stays in a similar range to the stuck example — all five attempts touch the same function, which any text-similarity check would also flag as repetitive — but $\bar{s}_o$ never clears the threshold, because the outcome is genuinely different every single turn: a different error type, a different assertion value, eventually success. `is_loopy_turn` is `False` throughout, the streak never advances past zero, and the guard correctly stays silent through what looks, on action text alone, like exactly the kind of repetitive pattern the guard exists to catch. This is the entire argument for checking outcome similarity independently rather than relying on action similarity alone: the two examples above have comparable $\bar{s}_a$ values and completely different $\bar{s}_o$ values, and only the second number tells you which one is actually a loop.

## Producing a Structured Stall Summary

The last rung of the escalation ladder, abort, is only useful to whoever picks up the task next if it hands over more than "the agent gave up." The whole point of catching a loop is that the agent has already, involuntarily, run an experiment on several variations of one approach — throwing that information away at exactly the moment it becomes most valuable defeats a large part of the reason to detect the loop in the first place.

```python
from collections import Counter


def build_stall_summary(trace: list[dict], guard: SessionLoopGuard) -> dict:
    recent = trace[-guard.window_size:]
    outcomes = [summarize_outcome(t["result"]) for t in recent if "result" in t]
    files_touched = sorted({
        t["args"].get("path") or t["args"].get("file")
        for t in recent if "args" in t and (t["args"].get("path") or t["args"].get("file"))
    })
    dominant_failure, occurrences = (
        Counter(outcomes).most_common(1)[0] if outcomes else ("unknown", 0)
    )
    return {
        "attempts_in_stalled_window": len(recent),
        "distinct_actions_tried": [
            t["args"] for t in recent if "args" in t
        ],
        "dominant_failure_signature": dominant_failure,
        "dominant_failure_occurrences": occurrences,
        "files_touched": files_touched,
        "recommendation": (
            "All recent attempts converged on the same failure signature "
            "above — the next attempt should target a different hypothesis "
            "about the root cause, not another variation of the same fix."
            if occurrences >= guard.sustain_turns
            else "Attempts varied in outcome; review individually rather than "
                 "assuming a single dead end."
        ),
    }
```

Handing a human (or a fresh agent session starting from this summary rather than from scratch) the dominant failure signature and the list of already-tried variations is what turns an aborted, stuck session into a useful artifact instead of a wasted one — the next attempt, whoever or whatever makes it, starts already knowing which approach has been exhausted, which is exactly the information a stuck agent accumulates for free over the course of failing repeatedly and that a bare "task aborted" message throws away.

## Session Scope With Parallel Tool Calls and Sub-Agents

"Session" needs a precise definition once an agent framework allows more than one thread of tool calls to be in flight at once, and getting this wrong quietly reintroduces the exact cross-session dilution this design argues against. A single agent issuing several tool calls in parallel within one turn (reading three files at once, say) is still one session for loop-guard purposes — those calls share one rolling window, since they're all part of the same reasoning thread's progress on one task. A supervising agent that fans a task out into several independent sub-agents, each working a different piece of the problem, is a different situation entirely: each sub-agent should get its own `SessionLoopGuard` instance with its own `session_id` and its own ephemeral collection, because a sub-agent stuck on its sub-task and a sibling sub-agent making fine progress on a different sub-task have no meaningful shared "recent action" history — mixing their windows would let one sub-agent's stall dilute another's genuinely-converging signal, or vice versa, the same category of mistake as sharing a rolling window across unrelated sessions in the first place.

This has a direct practical consequence for how the guard should be instantiated in a fan-out architecture: keyed per sub-agent, created when that sub-agent's sub-task begins and closed when it ends, exactly mirroring the sub-task's own lifecycle rather than the parent orchestrator's. A supervisor that wants visibility into "is any of my sub-agents currently stuck" aggregates `loop_detected` flags across its children's individual guards after the fact — it does not, and should not, feed all children's actions into one shared collection to answer that question, for the same reason a shared corpus would defeat the point of session-scoping in the first place.

## Challenges and Open Problems

None of the mechanisms above eliminate the need for a human to occasionally look at what an agent has been doing — they change how much wasted work happens before that look occurs, and how informative the resulting review is once it does. That's a meaningfully smaller problem than the one this post opened with, and worth stating plainly as the actual scope of what a loop guard buys, rather than implying it as a solved problem.

Tuning $k$ (`window_size`) and the two thresholds $\tau_a$, $\tau_o$ is workload-specific in a way that resists a single global default. A codebase with a narrow, repetitive fix vocabulary — a large generated-code module where every legitimate fix looks structurally similar to every other fix in that module — has a naturally higher baseline similarity between genuinely different, legitimate attempts than a codebase with broad architectural variety, which means a threshold tuned on one repository can be too trigger-happy on the first and too permissive on the second. This argues for per-repository or per-task-class calibration rather than a single shipped default, and for treating the thresholds as something to monitor and adjust over time rather than fix once at launch.

The outcome-summarization step in `summarize_outcome` carries as much weight as the similarity math built on top of it, and a poor summarizer undermines the entire mechanism regardless of how well-tuned the thresholds are. If the summary can't distinguish "still failing with the identical assertion" from "failing differently now" — because it's too coarse (collapsing all `AssertionError`s to one bucket regardless of what's actually being asserted) or too noisy (including line numbers or object identities that vary incidentally between otherwise-identical failures) — the outcome-similarity signal becomes either uselessly flat (everything looks the same) or uselessly volatile (nothing looks the same), and either failure mode defeats the joint action-and-outcome condition this whole design depends on.

A determined-but-genuinely-stuck agent that varies its phrasing specifically to defeat a loop guard it's aware of — or that has, through training, implicitly learned patterns that route around this kind of check without any explicit awareness of doing so — is a real, if currently mostly theoretical, adversarial-adjacent failure mode worth naming even without a documented case of it happening in the wild. Everything in this post assumes the paraphrasing behavior that defeats naive checks is an unintentional byproduct of how a stuck model generates varied-sounding attempts, not a deliberate evasion strategy; if that assumption stops holding as agents get better at reasoning about their own tool-calling patterns, an embedding-based check faces the same arms-race dynamic any pattern-based detector eventually faces; the honest conclusion is that this guard raises the bar past naive repetition rather than making evasion impossible.

The guard is also a heuristic about *productivity*, not a proof of it, and it's worth being precise about that distinction rather than overselling what a firing (or non-firing) guard actually establishes. A `loop_detected=True` result is evidence that recent actions and outcomes both look stuck by the similarity measures this post defines — it is not a formal guarantee that no progress is being made, the same way a `loop_detected=False` result is not a guarantee that progress *is* being made, only that the specific stuck-looking pattern this guard checks for hasn't shown up yet. An agent could, in principle, be making slow, genuine progress that this guard's outcome-similarity check is too coarse to distinguish from stagnation, particularly if `summarize_outcome` is compressing away exactly the detail that would reveal the difference — which loops back to why the outcome-summarization step deserves as much design attention as the similarity math, not less.

Finally, the added latency from an embedding call plus two Qdrant queries on every single tool call is a real cost for workloads with very fast, very frequent tool calls — a coding agent making dozens of sub-second file reads per second would see the guard's overhead become a non-trivial fraction of total turn time, even if the dollar cost stays small. Batching several turns' worth of signatures into a single embedding call, or falling back to a cheaper local similarity model for the action/outcome signatures specifically (rather than the same embedding model used elsewhere in a pipeline), are reasonable mitigations worth considering for latency-sensitive deployments, at the cost of some additional implementation complexity this post doesn't attempt to specify in detail.

## References

- LangGraph GitHub Issue #6731. *Agent infinite looping until recursion limit error is hit.* [github.com/langchain-ai/langgraph/issues/6731](https://github.com/langchain-ai/langgraph/issues/6731)
- LangGraph Documentation. *Recursion limit and `GraphRecursionError`.* [docs.langchain.com/oss/python/langgraph/graph-api](https://docs.langchain.com/oss/python/langgraph/graph-api)
- Anthropic (2024). *Building Effective Agents.* [anthropic.com/engineering/building-effective-agents](https://www.anthropic.com/engineering/building-effective-agents)
- This blog's companion post on session-level architecture for agentic loops: [Verdict: A Working Risk Classifier for Agentic Software Factories](/posts/verdict/)
- Qdrant. *Qdrant Vector Database Documentation.* [qdrant.tech/documentation](https://qdrant.tech/documentation)
