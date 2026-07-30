---
title: "The Determinism Gap: Why the Same Browser-Agent Task Succeeds Once and Fails the Next"
date: 2026-07-30
description: "A browser agent that passes a task nine times and fails the tenth isn't flaky in some vague sense — it's exhibiting one of two structurally different non-determinisms, each measurable and each requiring a different fix. This post separates them, argues pass@k is the wrong metric for unattended production use, and covers a Qdrant-backed trajectory clustering approach for catching a divergent run early rather than after it fails."
tags: ["agents", "browser-automation", "reliability", "evaluation", "benchmarks", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

"It worked when I tested it" is the most dangerous sentence in browser-agent engineering, because a run that succeeds is not evidence the task is solved — it's one sample from a distribution you haven't measured. This post is about that distribution: why the same task, same agent, same target site can succeed on one run and fail on the next with no code change in between, why the standard `pass@k` metric flatters exactly the systems you shouldn't trust for unattended production use, and a Qdrant-backed approach to catching a run that's drifting toward failure early — mid-trajectory — rather than discovering it only after the task completes badly.

This is a companion piece to the [failure taxonomy](/posts/browser-agent-failure-taxonomy/) post, not a restatement of it — that post catalogs *what* fails; this one is about *measuring and detecting* the fact that the same task doesn't fail the same way twice, which is a distinct engineering problem from knowing the failure categories exist.

## Table of Contents

1. [The Uncomfortable Statistic](#the-uncomfortable-statistic)
2. [Two Structurally Different Non-Determinisms](#two-structurally-different-non-determinisms)
3. [Why pass@k Is the Wrong Metric for Production](#why-passk-is-the-wrong-metric-for-production)
4. [Compounding Reliability Across a Multi-Step Task](#compounding-reliability-across-a-multi-step-task)
5. [Temperature and Sampling as a Lever, and Its Limits](#temperature-and-sampling-as-a-lever-and-its-limits)
6. [Separating Environment Drift from Model Non-Determinism](#separating-environment-drift-from-model-non-determinism)
7. [How Many Runs Before You Trust a Success-Rate Estimate](#how-many-runs-before-you-trust-a-success-rate-estimate)
8. [A Qdrant-Backed Trajectory Clustering Approach](#a-qdrant-backed-trajectory-clustering-approach)
9. [A Worked Example](#a-worked-example)
10. [Challenges and Open Problems](#challenges-and-open-problems)
11. [References](#references)

## The Uncomfortable Statistic

BrowserArena's finding that GPT-4o, used as an automated judge of whether an agent trace succeeded, agrees with human judgment only 68% of the time is worth sitting with before anything else in this post. It means that even in an offline evaluation setting — unhurried, full trace available, purpose-built judge prompt — the field's own tooling for measuring "did this work" is wrong roughly one time in three. WebArena's maintainers independently found brittle evaluation mechanics producing noisy, inconsistent metrics from underspecified success criteria. Real-website benchmarks compound this with **content drift**: reports show approximately 12% of Mind2Web tasks expired within a single year as target sites changed underneath the benchmark. The consequence for anyone building a production browser agent: you cannot borrow academic benchmark methodology wholesale and expect it to tell you, reliably, whether your agent is getting better or worse over time — the measurement instrument has its own error bars, and they're not small.

## Two Structurally Different Non-Determinisms

Conflating these two is the single most common mistake in reasoning about browser-agent flakiness, because they look identical from the outside (same task, different outcome) and require opposite fixes.

**Environment non-determinism**: the world the agent is acting in changed between runs, independent of anything the agent itself does differently. A/B test cohort assignment changed which layout variant rendered; a background job updated inventory count mid-task; network latency shifted timing enough that a race condition (see failure class 2 in the companion taxonomy post) resolved differently. The agent's decision-making was identical; the environment handed it a different situation. The fix here is environmental: better wait conditions, idempotent retries, detecting and adapting to layout variants rather than assuming a single fixed DOM shape.

**Model non-determinism**: the environment was identical (replayable, deterministic test fixture) but the agent's own reasoning path diverged — a different token sampled at a decision point, a slightly different interpretation of an ambiguous instruction, a different element chosen among several plausible matches. The fix here is *not* environmental at all; it's about tightening the decision surface itself — sampling temperature, more specific task instructions, disambiguation logic — because no amount of "wait for network idle" helps if the agent's own judgment is the thing that varied.

Distinguishing these two requires running the identical task against a *replayed, fixed* environment (a recorded HAR file or a snapshot test fixture, not the live site) and checking whether the same divergence still occurs. If it does, you have model non-determinism. If a fixed environment eliminates the divergence entirely, it was environmental all along, and no amount of prompt tuning would have fixed it.

## Why pass@k Is the Wrong Metric for Production

`pass@k` — the task counts as solved if *at least one* of $k$ independent attempts succeeds — is a reasonable metric for a human-supervised or human-selectable workflow (show the user the successful attempt, discard the rest) and actively misleading for anything meant to run unattended. An agent with a true per-attempt success rate of $p = 0.6$ has:

$$
\text{pass@}k = 1 - (1-p)^k
$$

At $k=5$, $\text{pass@}5 = 1 - 0.4^5 \approx 0.99$ — a system that fails four times out of ten on any single attempt reports a 99% "pass rate" under this metric, which is true in the narrow sense that *someone watching all five attempts and picking the best one* would almost always find a success, and completely irrelevant to a production deployment that runs the task once, unattended, and needs *that* attempt to work. The metric that actually matters for unattended use is closer to **all-$k$ reliability** — the probability that $k$ independent runs *all* succeed, which for the same $p=0.6$ at $k=5$ is $0.6^5 \approx 0.078$. The gap between 99% and 7.8% for the identical underlying agent is the gap between "looks production-ready in a benchmark leaderboard" and "fails the overwhelming majority of unattended runs."

## Compounding Reliability Across a Multi-Step Task

A multi-step browser task (navigate → search → select → fill form → submit → confirm) is not one Bernoulli trial, it's a chain of them, and even a per-step success probability that looks comfortable compounds badly:

$$
P(\text{task success}) = \prod_{i=1}^{n} p_i
$$

At a uniform $p_i = 0.95$ per step — which sounds like a strong number for any individual step — a six-step task succeeds end to end at $0.95^6 \approx 0.735$. A twelve-step task at the same per-step reliability drops to $0.95^{12} \approx 0.54$, worse than a coin flip, purely from chaining steps that each individually look fine. This is the same compounding-fan-out mathematics that shows up in agentic loop reliability generally; browser agents are simply where it's most visible, because each click, wait, and read is a distinct step with its own independent failure surface (the six categories in the companion taxonomy post), and long real-world workflows routinely run to a dozen-plus steps.

## Temperature and Sampling as a Lever, and Its Limits

If a diagnostic run against a replayed, fixed environment still shows divergence, sampling temperature is the first and cheapest lever to pull, precisely because it's the most direct source of model non-determinism as defined above: at temperature 0 (or as close to greedy decoding as the provider exposes), the model's token-selection at each step is deterministic given identical input, which collapses one entire axis of variance — a fixed environment plus greedy decoding should, in principle, reproduce the identical action sequence run after run. In practice this reduces but does not eliminate divergence, for two reasons worth naming rather than glossing over: first, provider-side infrastructure (batching, hardware nondeterminism in floating-point accumulation across different batch compositions) can introduce tiny numerical differences even at temperature 0 that occasionally flip a close call between two similarly-scored next tokens; second, and more consequentially for browser agents specifically, the *input* to the model at each step is a fresh DOM or screenshot read, and if that input has any variance at all (a millisecond-different animation frame captured in a screenshot, a DOM attribute ordering that isn't guaranteed stable), greedy decoding over a slightly different input is not the same as greedy decoding over an identical one, so environment-level determinism is a precondition for temperature-based determinism to actually deliver on its promise, not a separate, independent fix.

The practical upshot: lowering temperature is worth doing for any task where consistency matters more than exploring alternative approaches, but it should be validated against actual replayed-environment runs before being trusted as *the* fix — a team that lowers temperature, sees the failure rate drop from 40% to 15%, and declares victory has improved something real, but 15% is still far from deterministic, and the remaining gap is very likely the environment-non-determinism component this lever was never going to touch.

## Separating Environment Drift from Model Non-Determinism

In practice, distinguishing the two categories at scale — not just for one debugging session but as an ongoing reliability signal — means comparing runs structurally, not just by final outcome:

- **DOM/structure hashing per step**: fingerprint the page structure (not content — content legitimately changes) at each decision point across repeated runs of the same task. Structural divergence between runs against the live site, absent in replayed-fixture runs, is evidence of environment drift.
- **Action-sequence comparison**: two runs that hit structurally identical pages at every step but choose different actions at some point is evidence of model non-determinism, not environment drift — the world was the same; the decision wasn't.

## How Many Runs Before You Trust a Success-Rate Estimate

A single successful demo run tells you almost nothing about a task's true success probability $p$, and "run it a few more times and see" is only useful if you know roughly how many "a few" needs to be. Treating each run as an independent Bernoulli trial, the standard error of an observed success rate $\hat{p}$ from $n$ runs is:

$$
SE(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

For $\hat{p} = 0.8$ (a task that looks quite reliable from small-sample testing), $n=5$ gives $SE \approx 0.18$ — a 95% confidence interval of roughly $[0.44, 1.0]$, which is not a meaningful claim about reliability at all; the true rate could plausibly be anywhere from "worse than a coin flip" to "perfect." Reaching a genuinely tight interval — say, $\pm 0.05$ at 95% confidence — for $\hat{p}=0.8$ requires solving $1.96 \cdot SE(\hat p) \le 0.05$, which needs roughly $n \approx 246$ independent runs. This is the uncomfortable arithmetic behind "we tested it ten times and it worked nine of those": ten runs is nowhere near enough to distinguish a genuinely 90%-reliable task from one that's 60% reliable and got lucky, and any reliability claim made from single-digit run counts should be treated as a rough directional signal, not a number to build a production SLA around.

The required sample size to reach a fixed confidence-interval width grows with $\hat p (1-\hat p)$, so the runs-needed table looks different depending on where the true rate sits — a genuinely near-deterministic task needs far fewer confirming runs than a middling one, which is itself a useful diagnostic:

| Observed $\hat{p}$ | Runs for ±0.10 interval (95% CI) | Runs for ±0.05 interval (95% CI) |
|---|---|---|
| 0.95 | ~18 | ~73 |
| 0.90 | ~35 | ~139 |
| 0.80 | ~61 | ~246 |
| 0.60 | ~92 | ~369 |
| 0.50 | ~96 | ~384 |

The practical reading: a task that looks 95%-reliable from casual testing can be confirmed to a tight interval relatively cheaply (order of tens of runs), while a task sitting closer to the 50–60% range — exactly the range where it matters most to know whether you're above or below a "good enough to automate" bar — needs several hundred runs to pin down with the same confidence. This is backwards from how most teams actually allocate testing effort: the temptation is to run a handful of confirming tests on a task that already looks reliable and move on, when the task that most needs the larger sample is the mediocre one nobody wants to spend the compute confirming.

## A Qdrant-Backed Trajectory Clustering Approach

The failure-memory pattern in the companion taxonomy post matches a single failed *action* against known failure signatures after the fact. A complementary, earlier-warning approach: embed the full trajectory (the sequence of abstracted actions and page-state signatures) of *successful* runs for a given task and site, and check a new run's partial trajectory against that cluster as it progresses — flagging divergence while the task is still executing, which is cheap to abort or escalate, rather than only after it completes (successfully-looking but wrong, or visibly failed):

```python
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams,
)


class TrajectoryClusters:
    def __init__(self, client: QdrantClient, embed_fn, collection: str = "task_trajectories"):
        self.client, self.embed_fn, self.collection = client, embed_fn, collection
        names = {c.name for c in client.get_collections().collections}
        if collection not in names:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    def _trace_text(self, steps: list[str]) -> str:
        return " -> ".join(steps)

    def record_successful_run(self, task_id: str, steps: list[str]):
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=str(uuid.uuid4()), vector=self.embed_fn(self._trace_text(steps)),
                payload={"task_id": task_id, "outcome": "success", "n_steps": len(steps)},
            )],
        )

    def divergence_from_success_cluster(self, task_id: str, partial_steps: list[str],
                                         top_k: int = 10) -> float:
        """Called mid-run, after each step. Returns 0 (matches known-good
        pattern closely) to 1 (no resemblance to any recorded successful run
        for this task) — a rising trend across consecutive calls is the
        signal, not any single value in isolation."""
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=self.embed_fn(self._trace_text(partial_steps)),
            query_filter=Filter(must=[
                FieldCondition(key="task_id", match=MatchValue(value=task_id)),
                FieldCondition(key="outcome", match=MatchValue(value="success")),
            ]),
            limit=top_k,
        )
        if not hits:
            return 1.0
        return max(0.0, 1.0 - (sum(h.score for h in hits) / len(hits)))
```

A rising divergence score across consecutive steps of a single in-progress run — not a one-time low score, since a legitimately novel-but-valid path exists too — is the trigger for early intervention: pause for human confirmation, or abort and retry from a checkpoint, before the agent sinks another five steps of latency and cost into a trajectory that's already left the region where past runs succeeded.

## A Worked Example

A form-filling task historically completes in 8 steps with a tight trajectory cluster (the site's flow is stable). A new run, at step 4, shows a divergence score climbing from 0.1 (steps 1–3, matching the cluster closely) to 0.6 (step 4 diverges — the agent clicked into an options panel no successful run has ever entered). Aborting at step 4 and retrying from a checkpoint costs four wasted steps; letting the run continue to its natural conclusion at step 8 or 9, only to discover it produced an unexpected result, costs the full task plus whatever cleanup the wrong path required. The trajectory signal makes that a step-4 decision instead of a step-9 discovery.

## Challenges and Open Problems

**The measurement instrument is part of the problem.** With GPT-4o-as-judge agreeing with humans only 68% of the time on trace success, any automated pipeline that both runs a browser agent *and* automatically labels its own success (to feed the trajectory clusters above, for instance) is training on noisy labels — a divergence-detection system built on unreliable ground truth inherits that unreliability, and there's no clean way around this short of human-labeled outcomes for at least a calibration subset.

**Trajectory clusters decay at the same rate the underlying benchmarks do.** If 12% of task flows on real sites change meaningfully within a year, a "successful trajectory" cluster recorded six months ago may itself be the divergent pattern now, and the detector above has no built-in way to distinguish "this run diverges because it's failing" from "this run diverges because the site changed and every run now looks different from the stale cluster" — the same staleness problem that shows up in [incremental embedding sync](/posts/driftsync/) generally, applied here to behavioral rather than content embeddings.

**Distinguishing the two non-determinisms cleanly requires infrastructure most teams don't have.** Replaying a task against a frozen environment snapshot is straightforward for a simple static page and genuinely hard for a modern SPA with live backend state, third-party scripts, and session-dependent rendering — in practice, most teams will only ever get an approximate signal, not a clean binary answer, about which category a given flake belongs to.

## References

- *WebArena Verified: Reliable Evaluation for Web Agents.* OpenReview. [openreview.net/pdf?id=94tlGxmqkN](https://openreview.net/pdf?id=94tlGxmqkN)
- *WAREX: Web Agent Reliability Evaluation on Existing Benchmarks.* arXiv:2510.03285
- *StressWeb: A Diagnostic Benchmark for Web Agent Robustness under Realistic Interaction Variability.* arXiv:2604.16385
- Emergent Mind. *Online-Mind2Web Benchmark.* [emergentmind.com/topics/online-mind2web-benchmark](https://www.emergentmind.com/topics/online-mind2web-benchmark)
- *WebChoreArena: Evaluating Web Browsing Agents on Realistic Tedious Web Tasks.* arXiv:2506.01952
- Qdrant. *Qdrant Vector Database Documentation.* [qdrant.tech/documentation](https://qdrant.tech/documentation)
