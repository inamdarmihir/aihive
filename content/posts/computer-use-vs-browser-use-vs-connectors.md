---
title: "Computer Use vs. Browser Use vs. API Connectors: Picking the Right Integration Layer"
date: 2026-07-30
description: "Five browser-and-desktop agent stacks dominate 2026, and the temptation is to pick the most general one for every task. This post lays out the three integration layers an agent task actually chooses between, the real benchmark and cost gap separating them, and a Qdrant-backed router that picks the cheapest layer likely to succeed based on similarity to past tasks."
tags: ["agents", "browser-automation", "computer-use", "mcp", "system-design", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Given a task like "check whether this vendor's invoice matches our purchase order," there are at least three structurally different ways to build the agent that does it: call the vendor's API through an MCP connector if one exists, drive a browser against their web portal if it doesn't, or hand the whole desktop to a vision-based computer-use agent if the portal is a legacy Java applet with no accessible DOM at all. These are not three flavors of the same solution — they differ by an order of magnitude in cost, latency, and reliability, and the decision of which to reach for is usually made by habit (whichever the team built last) rather than by the task's actual shape. This post lays out the decision explicitly, with the benchmark numbers that justify it, and a router that can make the call automatically for a large, heterogeneous task set.

## Table of Contents

1. [Three Layers, Not One Spectrum](#three-layers-not-one-spectrum)
2. [The Benchmark Reality Check](#the-benchmark-reality-check)
3. [A Decision Framework](#a-decision-framework)
4. [The Cost Model](#the-cost-model)
5. [A Concrete Latency Budget, Layer by Layer](#a-concrete-latency-budget-layer-by-layer)
6. [Hybrid Architectures: Fall Through the Layers](#hybrid-architectures-fall-through-the-layers)
7. [When a Vendor Changes the Rules Underneath You](#when-a-vendor-changes-the-rules-underneath-you)
8. [A Qdrant-Backed Layer Router](#a-qdrant-backed-layer-router)
9. [A Worked Example](#a-worked-example)
10. [Challenges and Open Problems](#challenges-and-open-problems)
11. [References](#references)

## Three Layers, Not One Spectrum

**API / MCP connectors** — the agent calls a structured tool (covered in depth in the [MCP server](/posts/building-mcp-server-claude-connector/) and [ChatGPT connector](/posts/building-chatgpt-connector-apps-sdk/) posts) that talks to a well-defined backend. Deterministic, fast, cheap, and only available when the target system exposes one. This is the layer to default to whenever it exists — everything below is a fallback for when it doesn't.

**Browser-use / DOM-grounded agents** — `browser-use`, Stagehand, or a raw Playwright-plus-LLM loop reads the page's DOM or accessibility tree and decides actions without hardcoded selectors. Handles any web UI without requiring the target to expose an API, at meaningfully higher cost and lower reliability than a direct connector call, and with the full failure taxonomy covered in the [companion post](/posts/browser-agent-failure-taxonomy/) on browser-agent failure modes.

**Computer Use / vision-based desktop control** — Anthropic's Computer Use or OpenAI's Computer-Using-Agent (Operator) reasons over rendered screenshots and issues mouse/keyboard actions, with no reliance on DOM access at all. The most general layer — it works against native desktop applications, canvas-rendered UIs, and anything else with no accessible structure to read — and the slowest and most expensive, because every step is a full vision-model call over a screenshot rather than a structured read.

Five stacks dominate the current landscape across these layers: Playwright + Claude (deterministic scripting with agentic fallback), Stagehand (Browserbase's SDK bridging Playwright precision and AI-driven control), Browserbase itself (managed cloud browser runtime), Anthropic Computer Use (vision-driven, screen-general), and OpenAI's Computer-Using-Agent (cloud-only, web-focused). None of these is strictly dominant — each sits at a different point on the structure-vs-generality tradeoff this section describes.

## The Benchmark Reality Check

Published comparisons put concrete numbers on the generality-vs-reliability tradeoff. On OSWorld (general computer tasks across arbitrary native applications), Operator scores 38.1% against Computer Use's 22.0%. On WebVoyager (browser-specific tasks), Operator reaches 87% against Computer Use's 56%. The gap runs the other direction for coding and software-development tasks, where Computer Use's tighter integration with a coding-focused model shows an advantage Operator's web-centric design doesn't match. Neither number should be read as "Operator is better" or "Computer Use is better" in the abstract — they're evidence that vision-based computer-use agents, across both vendors, solve well under half of general computer tasks even in a benchmark setting with no adversarial content and no production time pressure. That's the generality tax: the layer that can act on literally anything on a screen pays for that generality in a success rate meaningfully below what a scoped browser-DOM agent achieves on browser-specific tasks, which is itself below what a direct API call achieves on the (necessarily narrower) tasks an API actually exposes.

## A Decision Framework

The practical routing logic, in order of preference:

1. **Does a structured API or MCP connector exist for this target system?** Use it. This is not close — a direct API call is deterministic, doesn't degrade with UI redesigns, and costs a fraction of a vision-model inference loop.
2. **If not, is the target a standard web UI with an accessible DOM?** Use a browser-use / Stagehand-style agent. Accept the failure taxonomy and determinism-gap costs covered elsewhere on this blog as the price of not having a connector, and consider building one if this task recurs often enough to amortize the engineering cost.
3. **If the target has no accessible DOM at all** — a canvas-rendered app, a native desktop application, a legacy system with no web presence — **only then** reach for Computer Use or Operator-style vision control, understanding that OSWorld's sub-40% pass rate for even the stronger of the two vendors is the honest expectation for general tasks in this regime, not an edge case to plan around.

The mistake this framework is meant to prevent is reaching for the most general (and most expensive, least reliable) layer by default because it's the most exciting one to build with, when a structured connector for the same target was available the whole time.

## The Cost Model

Approximate per-step cost and latency scale roughly an order of magnitude apart across the three layers: an MCP tool call is a single structured request/response, typically tens to low hundreds of milliseconds and negligible token cost beyond the tool's own arguments and result. A browser-use step involves reading a DOM snapshot (often several KB to tens of KB of trimmed HTML/accessibility-tree text) through the model's context, typically hundreds of milliseconds to low seconds per step once network and page-settle time are included. A computer-use step involves a full screenshot (tens to hundreds of KB of image data), a vision-model inference call, and often multiple seconds of latency per action given the added rendering and vision-processing overhead. For an $n$-step task, total cost scales as:

$$
C_{\text{total}} = n \cdot c_{\text{layer}}
$$

where $c_{\text{layer}}$ differs by roughly one to two orders of magnitude between the cheapest (API) and most expensive (computer use) layer. Combined with the compounding-reliability mathematics from the [determinism gap](/posts/browser-agent-determinism-gap/) post — end-to-end success probability falling as $\prod p_i$ across $n$ steps — the expected *cost per successful completion* (not just cost per attempt) widens the gap between layers further still, because the least reliable layer also needs the most retries to reach the same expected number of successful completions.

## A Concrete Latency Budget, Layer by Layer

Abstract order-of-magnitude claims are easy to nod along with and easy to under-estimate in an actual latency budget. Walking through one step of a hypothetical "check order status" task at each layer, with rough, representative figures rather than a specific benchmark's numbers:

**API/MCP connector.** One HTTP round trip to a structured endpoint: DNS/connection reuse from a warm pool (~0ms amortized), request serialization and network round-trip (~50–150ms depending on region), response parsing (~negligible). Total: **well under 200ms**, and this figure barely changes whether the task has one step or ten, since each step is an independent, cheap call.

**Browser-use / DOM agent.** Page navigation and render settle (~500ms–2s depending on the site's own performance), DOM/accessibility-tree extraction and trimming (~50–100ms), a model inference call over that extracted context to decide the next action (~500ms–2s depending on model and context size), then the actual action execution and a wait for the resulting state change (~200ms–1s). Total per step: **roughly 1.5–5 seconds**, and — critically — this doesn't shrink much with practice, because each step's dominant costs (page render time, model inference) are close to fixed regardless of how many times you've automated this exact flow before.

**Computer Use / vision agent.** Screenshot capture (~100–300ms), image encoding and transfer to a vision-model endpoint (~200–500ms depending on resolution), vision-model inference over the full image — meaningfully slower than a text-only inference call given the larger effective context (~1–4 seconds), then action execution and a fresh screenshot to confirm the result before proceeding (another full round of the above). Total per step: **roughly 3–10 seconds**, before accounting for the lower per-step success rate discussed above, which means a meaningful fraction of steps also pay a retry's worth of this cost on top.

For a 10-step task, these per-step figures compound to roughly 1–2 seconds (API), 15–50 seconds (browser-use), and 30–100+ seconds (computer use) — a real, user-perceptible difference in how long a task takes to complete even before factoring in the differing failure and retry rates from the cost model above. This is the number to put in front of a stakeholder asking "why can't we just use computer use for everything" — it isn't only less reliable, it's an order of magnitude slower per step even when it works on the first try.

Combining this latency budget with the benchmark success rates from earlier — and the all-$k$ reliability mathematics from the [determinism gap](/posts/browser-agent-determinism-gap/) post — gives an expected *time-to-successful-completion* per layer, which is the number that actually determines user-facing experience, not raw per-attempt latency:

| Layer | Per-attempt latency (10-step task) | Illustrative per-attempt success rate | Expected attempts to succeed ($1/p$) | Expected time to success |
|---|---|---|---|---|
| API/MCP connector | ~1.5s | ~0.99 (structured, deterministic) | ~1.01 | ~1.5s |
| Browser-use / DOM agent | ~30s | ~0.85 (WebVoyager-consistent range for a scoped task) | ~1.18 | ~35s |
| Computer Use / vision agent | ~65s | ~0.40 (OSWorld-consistent range for a general task) | ~2.5 | ~163s |

The gap between browser-use and computer-use widens further than the raw per-attempt latency ratio suggests once expected retries are folded in — computer-use isn't just roughly twice as slow per attempt, it needs roughly two and a half attempts on average to reach the same outcome a browser-use agent reaches in just over one, compounding a 2x per-step latency disadvantage into something closer to 4-5x in expected wall-clock time to a successful result. This is the number that should end the "just use the most general layer" argument decisively for any task where a narrower layer is available at all.

## Hybrid Architectures: Fall Through the Layers

Production systems handling a heterogeneous task set rarely commit to one layer exclusively — they fall through the hierarchy per task: attempt the API connector, and only invoke a browser-use fallback for the subset of accounts or vendors that don't have one onboarded yet; within browser-use, escalate to computer-use only for the specific pages that turn out to be canvas-rendered or otherwise DOM-inaccessible, rather than running every task through the most general (and most expensive) layer uniformly. This fallthrough pattern is exactly the kind of decision a router benefits from automating once the task set is large and heterogeneous enough that a human curating "which layer for which vendor" doesn't scale.

## When a Vendor Changes the Rules Underneath You

The layer hierarchy isn't static per vendor over the lifetime of an integration, and treating a routing decision as permanent once made is a quiet source of technical debt. Three directions of change are common enough to design for rather than treat as edge cases:

**An API gets deprecated or rate-limited more aggressively**, pushing a previously connector-routed task down to the browser-use layer as a stopgap — this is a downgrade in reliability and cost that should trigger an alert, not a silent fallback that nobody notices until someone asks why the finance-ops pipeline got slower and more expensive without an obvious code change.

**A vendor ships an API where none existed**, which should trigger the opposite: a previously browser-use-routed task becoming eligible for promotion to the connector layer. This is easy to miss precisely because nothing broke — the browser-use path kept working, just at a worse cost and reliability profile than necessary, and there's no natural trigger to revisit the routing decision unless something is actively watching for "does this vendor have an API now" on some cadence.

**A site redesign breaks the DOM structure a browser-use agent depended on**, which the router above will eventually detect through a dip in recorded success rate for that layer on that task — but only after enough failed attempts have accumulated to move the average, which for a low-volume task can take longer than a human would notice by simply trying the task once and seeing it fail.

The practical implication: the router's `recommend_layer` output should be periodically re-validated against reality, not trusted indefinitely once a routing decision has stabilized — a lightweight monthly job that re-checks whether a connector now exists for browser-use-routed vendors, and whether previously-reliable connectors are still returning success, catches both directions of drift before they've silently cost significant reliability or budget.

## A Qdrant-Backed Layer Router

Embedding a task description alongside which layer succeeded, at what cost, and how reliably, turns "which layer should handle this new task" into a similarity lookup against historical outcomes rather than a fixed, hand-maintained rule table that has to be updated every time a new target system is onboarded:

```python
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class LayerRouter:
    def __init__(self, client: QdrantClient, embed_fn, collection: str = "layer_outcomes"):
        self.client, self.embed_fn, self.collection = client, embed_fn, collection
        names = {c.name for c in client.get_collections().collections}
        if collection not in names:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def record_outcome(self, task_description: str, layer: str, succeeded: bool,
                        cost_usd: float, latency_ms: int):
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=str(uuid.uuid4()), vector=self.embed_fn(task_description),
                payload={"layer": layer, "succeeded": succeeded,
                         "cost_usd": cost_usd, "latency_ms": latency_ms},
            )],
        )

    def recommend_layer(self, task_description: str, top_k: int = 30) -> dict:
        hits = self.client.search(
            collection_name=self.collection, query_vector=self.embed_fn(task_description), limit=top_k,
        )
        if not hits:
            return {"layer": "api_connector", "reason": "no history — try the cheapest layer first"}
        by_layer: dict[str, list] = {}
        for h in hits:
            by_layer.setdefault(h.payload["layer"], []).append(h.payload)
        scored = {
            layer: {
                "success_rate": sum(r["succeeded"] for r in rows) / len(rows),
                "avg_cost": sum(r["cost_usd"] for r in rows) / len(rows),
                "n": len(rows),
            }
            for layer, rows in by_layer.items()
        }
        # Prefer the cheapest layer whose historical success rate clears a bar,
        # not just the highest-success-rate layer regardless of cost.
        acceptable = {l: s for l, s in scored.items() if s["success_rate"] >= 0.85}
        if acceptable:
            best = min(acceptable, key=lambda l: acceptable[l]["avg_cost"])
        else:
            best = max(scored, key=lambda l: scored[l]["success_rate"])
        return {"layer": best, "stats": scored}
```

The `acceptable`-then-cheapest selection is the important design decision here, not the similarity search itself — a naive router that always picks the historically highest-success-rate layer will happily route every task to computer-use once it has enough data showing computer-use eventually succeeds, ignoring that an API connector might clear an 85% bar at a fraction of the cost and latency. The router's objective should be "cheapest layer that's reliable enough," not "most reliable layer regardless of cost," matching the actual decision framework above rather than optimizing a metric that silently biases toward the most expensive option.

## A Worked Example

A finance-ops team automates invoice reconciliation across 40 vendors. Six have modern APIs (routed to MCP connectors, near-100% success, sub-second latency). Twenty-two have standard web portals with no API (routed to browser-use, ~90% success per attempt, several seconds per task). Twelve are legacy vendor portals built on old Java applets or Flash-derived rendering with no accessible DOM (routed to computer-use as the only option that works at all, ~60% success per attempt given the OSWorld-consistent ceiling on general vision-driven tasks, tens of seconds per task). A router seeded with even a few weeks of outcome data across these 40 vendors correctly keeps the six API-backed vendors off the expensive layers entirely, and only escalates the twelve DOM-inaccessible ones to computer-use — versus a naive "use the most capable agent for everything" policy that would run all 40 vendors through computer-use, at roughly the OSWorld-implied cost and failure rate, for the 28 vendors that never needed it.

## Challenges and Open Problems

**Published benchmark numbers are a ceiling for a general task distribution, not a prediction for your specific vendor portal.** OSWorld and WebVoyager scores describe performance across a broad task sample; your twelve legacy vendor portals might individually perform better or worse than the aggregate number, and the router above only starts reflecting that once it has accumulated outcome history specific to *your* tasks — early routing decisions are necessarily made on thinner evidence than the benchmark literature.

**Cold start biases toward under-escalation.** A router with no history defaults to the cheapest layer, which is correct in expectation but means the first attempt at any genuinely computer-use-only task will predictably fail before the router has evidence to route around the cheaper layers — worth pairing with a fast-failure detection (rather than a long timeout) so the cold-start cost is one quick failed attempt, not a multi-minute one.

**The three layers aren't as cleanly separable in practice as the framework implies.** A single task can legitimately need an API call for part of the workflow and a browser step for another part (log in via the portal, then hit an authenticated internal API the portal itself calls) — the router above scores whole tasks, not sub-steps, and a more granular per-step router is a real extension this post doesn't build out.

## References

- WorkOS. *Anthropic's Computer Use versus OpenAI's Computer Using Agent (CUA).* [workos.com/blog/anthropics-computer-use-versus-openais-computer-using-agent-cua](https://workos.com/blog/anthropics-computer-use-versus-openais-computer-using-agent-cua)
- Helicone. *The Best Web Agents: Computer Use vs Operator vs Browser Use.* [helicone.ai/blog/browser-use-vs-computer-use-vs-operator](https://www.helicone.ai/blog/browser-use-vs-computer-use-vs-operator)
- notte.cc. *The browser automation stack in 2026: which tool for which job.* [notte.cc/blog/browser-agent-stack-2026](https://www.notte.cc/blog/browser-agent-stack-2026)
- digitalapplied.com. *Browser Automation AI Agents: Playwright vs Stagehand.* [digitalapplied.com/blog/browser-automation-ai-agents-playwright-stagehand-2026](https://www.digitalapplied.com/blog/browser-automation-ai-agents-playwright-stagehand-2026)
- Claude Platform Docs. *Computer use tool.* [platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)
- Qdrant. *Qdrant Vector Database Documentation.* [qdrant.tech/documentation](https://qdrant.tech/documentation)
