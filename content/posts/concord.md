---
title: "Concord: A Consolidation Gateway for Multi-Bot PR Review Fatigue"
date: 2026-07-24
description: "Running four AI code review bots on the same pull request means the same real bug gets flagged three different ways in three different comments, while each bot's own distinct false positives pile on top. Concord is a review-consolidation gateway — a GitHub App you install alongside your existing bots (CodeRabbit, Cursor Bugbot, Copilot, Devin Reviewer) — that buffers incoming bot comments, matches them by a combination of code-location proximity and semantic similarity, and collapses true duplicates deterministically, using Qdrant as the per-PR similarity index rather than another LLM call."
tags: ["agents", "code-review", "github", "developer-productivity", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Running one AI code review bot on a pull request was, for a while, a clear productivity win — a tireless first pass that caught the obvious stuff before a human spent any attention on it. Running two to four of them at once, which is now the normal configuration on teams using **GitHub Copilot**'s PR review, **CodeRabbit**, **Cursor Bugbot**, **Devin Reviewer**, and **Graphite**'s reviewer in various combinations, produces a different and worse outcome than "four times the coverage." Each bot reads the same diff independently and posts its own comments, and the empirical pattern documented across multiple teams running this setup is consistent: false positives rarely overlap between bots, but true positives mostly do. A genuinely correct finding gets posted three or four times in slightly different wording, while each bot also contributes its own distinct noise on top. A developer opening the PR sees ten to fifteen comments, has to manually work out which are duplicates, which are real, and which are noise — which is close to the exact triage burden AI review was supposed to remove.

This post designs and specs **Concord**, a consolidation layer that sits between the bots and the developer: a GitHub App that buffers incoming review comments for a short window, matches genuine duplicates using a combination of code-location proximity and semantic text similarity, and collapses them before anything reaches a human — installed alongside whatever review bots your team already runs, with no changes required to those bots themselves. It covers the matching design, the Qdrant-backed filtered-then-ranked query at the center of it, and the threshold tradeoffs involved in getting the aggressiveness of the merge right. It does not cover the review bots' own detection logic, prompt engineering for better first-pass findings, or CI/static-analysis integration — those are separate, already well-covered concerns. Familiarity with GitHub's Apps/webhooks model (`pull_request_review_comment` events specifically) is assumed. The matching problem here is a close cousin of the one covered in the [companion post on duplicate GitHub issue detection](/posts/qdrant-duplicate-issue-triage/) — both use a hybrid of exact-field filtering and semantic similarity rather than either alone — but the constraint that makes this problem distinct is the hard requirement that two comments in different code locations must never merge regardless of textual similarity, which the issue-dedup design doesn't need to enforce in the same way.

## Table of Contents

1. [The Shape of the Problem: Redundant Signal, Independent Noise](#the-shape-of-the-problem-redundant-signal-independent-noise)
2. [Why Naive Deduplication Fails](#why-naive-deduplication-fails)
3. [Prior Art: Distill and Quorum](#prior-art-distill-and-quorum)
4. [Design: Concord's Review-Consolidation Gateway](#design-concords-review-consolidation-gateway)
   - [Buffering Incoming Comments](#buffering-incoming-comments)
   - [The Qdrant Schema](#the-qdrant-schema)
   - [The Filtered-Then-Ranked Query](#the-filtered-then-ranked-query)
   - [Consolidation Logic](#consolidation-logic)
5. [Threshold Tuning: The Conservative Bias](#threshold-tuning-the-conservative-bias)
6. [Installing Concord](#installing-concord)
7. [Worked Example: Four Bots, One Line](#worked-example-four-bots-one-line)
8. [Deterministic Consensus, Not Another LLM Call](#deterministic-consensus-not-another-llm-call)
9. [Challenges and Open Problems](#challenges-and-open-problems)
10. [References](#references)

## The Shape of the Problem: Redundant Signal, Independent Noise

The core dynamic is worth stating precisely, because it's counterintuitive if you assume more reviewers should mean strictly better coverage with proportional noise. In practice, teams report the opposite correlation structure: correctness correlates across bots, because a genuinely broken null check or an off-by-one is objectively there in the diff for any reasonably capable model to find, so multiple bots independently converge on it. Noise doesn't correlate, because each bot's specific hallucinations, stylistic nitpicks, and misreadings of context are artifacts of that bot's own training and prompting, essentially uncorrelated with any other bot's mistakes. The net effect on a PR with four active bots: one real issue becomes three or four near-identical comments, and each bot's independent noise stacks additively rather than canceling out.

The cost of this is best framed through **Return on Attention (ROA)** — a framing from a widely-circulated [dev.to post](https://dev.to/cseeman/return-on-attention-why-ai-code-reviews-are-wearing-us-out-2hh0) on exactly this fatigue pattern, worth quoting directly because it names the actual scarce resource precisely: "every word you ask someone else to read has to be worth what it costs them to read it." Human reviewer attention doesn't scale the way bot output does. Four bots can each generate a comment in seconds at effectively zero marginal cost; a human still has to read, evaluate, and decide on each one at the same fixed cost per comment regardless of how many bots produced it. Multiplying the comment count by four without multiplying the *distinct information* by four is a straightforward tax on the one resource that didn't get cheaper.

The behavioral consequence compounds the problem rather than self-correcting. Teams have reported having to explicitly "engineer nitpicking out of" their own review bots because noisy comments were burying the ones that mattered — a tuning response to the *symptom*, not the duplication that caused it. And once a majority of a given bot's comments get dismissed without action, developers stop reading that bot's comments carefully at all, including the valid ones — a trust-erosion dynamic where the bot's genuinely useful findings start underperforming purely because of the noise-to-signal ratio the developer has learned to expect from it. A consolidation layer's entire value proposition rests on interrupting this loop before it sets in, which means it has to be conservative enough to never be the thing that erodes trust itself — a constraint that shapes the threshold-tuning discussion later in this post more than any other single factor.

## Why Naive Deduplication Fails

The instinctive first approach — dedupe by exact string match, or by a lexical similarity measure like Jaccard overlap or edit distance on the comment text — fails for a structural reason, not a tuning reason: different bots phrase the *same* underlying finding in ways that share almost no surface vocabulary. Three real-shaped comments on the identical line:

- "This line could cause a null pointer exception if `x` is undefined."
- "Potential NPE: `x` is not null-checked before use."
- "Consider guarding against undefined `x` here."

These describe one issue at one code location. Their token overlap is thin — "null," "x," maybe "undefined" — nowhere near what a Jaccard threshold tuned to avoid false merges would accept. Lexical similarity is measuring the wrong thing: it's sensitive to *phrasing*, and phrasing is exactly the dimension along which different bots vary the most while agreeing on substance. This needs semantic matching — comparing what the comments *mean*, via embeddings, not what words they share.

But semantic similarity on the text alone introduces a different failure mode, and this is the design point that matters most in this post: two comments about superficially similar *kinds* of issues, on completely unrelated parts of the codebase, can score highly similar in embedding space purely because they're both, say, "missing error handling around an async call." A null-check comment on `auth.ts:42` and a conceptually similar null-check comment on `billing.ts:210` are not the same finding and must never be merged, no matter how close their embeddings sit — collapsing them would silently make one real bug disappear from view. Location — file path, and line number or the specific code span a comment anchors to — has to be at least as strong a matching signal as text similarity, arguably stronger, because a location mismatch is a hard, unambiguous signal that two comments are not the same finding, while text similarity is inherently fuzzy. The matching key needs to combine both: text similarity to establish that two comments plausibly describe the same *kind* of issue, and location proximity to confirm they're describing it at the *same place*. Neither signal alone is safe to merge on.

Stated as a matching function rather than prose, for two comments $c_1$ and $c_2$ with embeddings $\vec{v}_1, \vec{v}_2$, file paths $f_1, f_2$, anchor lines $\ell_1, \ell_2$, and a line-tolerance $\tau$:

$$\text{match}(c_1, c_2) = \mathbb{1}[f_1 = f_2] \cdot \mathbb{1}[\,|\ell_1 - \ell_2| \le \tau\,] \cdot \cos(\vec{v}_1, \vec{v}_2)$$

The two indicator terms are hard gates, not soft weights — either of them evaluating to zero collapses the whole expression to zero regardless of how high the cosine similarity is, which is exactly the property that makes the design safe against the two-different-bugs-phrased-similarly failure mode. This is deliberately *not* a weighted sum like $\alpha \cdot \text{location\_score} + (1-\alpha) \cdot \text{text\_score}$, because a weighted sum lets a very high text similarity partially compensate for a middling location mismatch, which is precisely the tradeoff that must never be available here — no amount of textual similarity should be able to buy its way past a location mismatch. Only when both gates pass does cosine similarity get to make the final call, compared against $\text{DUPLICATE\_THRESHOLD}$.

## Prior Art: Distill and Quorum

This is a live, current problem with tooling already emerging around it, and it's worth grounding the design against two real examples rather than presenting the approach below as if it were the first attempt.

**Distill**, marketed as "One review. Every bot. Every commit.," is a commercial GitHub App built specifically to consolidate this noise. Its own positioning states plainly that "modern dev teams use 2-4 AI code review bots simultaneously, and the noise drowns the signal," and describes its consolidation mechanism as using "AI to understand semantic similarity" so that, in its own example, "Add null check" and "Handle undefined" get merged into a single comment rather than shown as two — the exact phrasing-variance problem described above, from a product built to solve it in production.

**Quorum**, an open-source project on GitHub by developer Wenjix, frames the same problem from a slightly different angle: "Modern teams often run several AI reviewers on the same pull request: Cursor Bugbot, GitHub Copilot, Devin Reviewer, CodeRabbit... Quorum adds a consensus layer. It groups duplicate findings, highlights where multiple reviewers agree." The detail worth carrying directly into this post's design is an explicit statement in Quorum's own documentation: **"Quorum is computed in code from distinct reviewers, not by the model."** The consensus/dedup computation itself is deterministic — clustering and counting run as plain code against structured inputs, not as a prompt to an LLM asking "are these the same finding?" That's a design principle worth treating as load-bearing, not incidental, and it's the subject of its own section below.

Both tools converge on the same two ideas independently: semantic matching is necessary because lexical matching misses true duplicates, and the actual grouping decision should not itself be a source of new inconsistency. The design in this post takes those as given and focuses on the mechanism — specifically, on making location proximity a first-class, hard constraint in the matching key rather than an implicit assumption.

The design in this post and the two prior-art tools land on the same core mechanism from different starting points, which is worth laying out explicitly rather than leaving as an implicit claim:

| Property | Distill | Quorum | This design |
|---|---|---|---|
| Matching signal | Semantic similarity ("AI to understand semantic similarity") | Deterministic clustering of line-anchored comments | Location filter (hard gate) + semantic similarity (ranking), combined |
| Consensus computation | Not publicly specified in detail | Explicitly "computed in code from distinct reviewers, not by the model" | Deterministic threshold comparison over Qdrant-returned scores |
| Scope | Cross-PR, ongoing product | Single-PR synthesis skill, plus optional deeper cloud-agent exploration | Single-PR, real-time buffering gateway |
| Output shape | One consolidated summary comment | One idempotent synthesis comment with reviewer quorum counts | Representative comment per cluster, footnoted with other contributing bots |

The overlap on "consensus should be computed deterministically, not delegated to another model call" across two independently-built tools is a reasonable signal that it's the right default rather than a coincidence, and it's the one design principle this post treats as closer to a hard constraint than a preference.

It's also worth being clear that a consolidation layer is a complement to a handful of already-documented mitigation habits, not a replacement for them. Teams dealing with this problem also scope individual bots down — narrower path filters so a bot only comments on the parts of the codebase it's actually reliable on, and higher severity thresholds so it stays quiet on the borderline cases most likely to be dismissed. Smaller, more atomic pull requests help independently, since large monolithic diffs are exactly the condition under which review bots (like human reviewers) tend to produce more generic, less anchored feedback — a bot skimming a 2,000-line diff hallucinates and over-generalizes more than the same bot given a focused 80-line change. And the most durable habit is cultural rather than technical: treating every bot comment, consolidated or not, as a hypothesis a human still has to verify, not a fact to act on unquestioned — a consolidation layer that successfully merges four instances of the same wrong claim into one comment has not made the claim any more true, only quieter.

## Design: Concord's Review-Consolidation Gateway

**Concord** is a GitHub App installed alongside the review bots, subscribed to `pull_request_review_comment` and review-submission webhook events from each one. Rather than letting each bot's comments post directly and visibly the moment they're created, Concord intercepts them, holds them in a short buffer, and only then decides what actually gets shown.

### Buffering Incoming Comments

Bots don't all finish reviewing a PR at the same instant — one might return in 15 seconds, another in over a minute, depending on diff size and the bot's own queueing. A consolidation layer that tries to compare a new comment only against what's already arrived will miss duplicates from bots that haven't finished yet. The gateway holds each incoming comment in a buffer keyed by PR, waiting 30-90 seconds (tunable per repo, longer for larger diffs where slower bots take longer) before finalizing which comments actually get posted:

```python
import asyncio
import time
from dataclasses import dataclass, field

BUFFER_WINDOW_SECONDS = 60

@dataclass
class BufferedComment:
    pr_id: str
    bot_name: str
    file_path: str
    line_start: int
    line_end: int
    raw_text: str
    has_suggestion: bool
    received_at: float = field(default_factory=time.time)

pr_buffers: dict[str, list[BufferedComment]] = {}
pr_flush_tasks: dict[str, asyncio.Task] = {}

async def handle_incoming_comment(comment: BufferedComment):
    pr_buffers.setdefault(comment.pr_id, []).append(comment)

    if comment.pr_id not in pr_flush_tasks or pr_flush_tasks[comment.pr_id].done():
        pr_flush_tasks[comment.pr_id] = asyncio.create_task(
            flush_after_window(comment.pr_id)
        )

async def flush_after_window(pr_id: str):
    await asyncio.sleep(BUFFER_WINDOW_SECONDS)
    buffered = pr_buffers.pop(pr_id, [])
    await consolidate_and_post(pr_id, buffered)
```

Restarting the flush timer isn't done here on every new arrival — a fixed window from the *first* comment on a PR is deliberate, since an unbounded reset (waiting another 60 seconds every time any bot posts anything) risks the buffer never draining on a PR that attracts a steady trickle of comments from slow or retrying bots. A late arrival after the window closes is handled separately, as a follow-up comparison against what already posted, discussed under commit-awareness in the challenges section.

### The Qdrant Schema

Each buffered comment gets embedded and stored in a collection scoped by PR, with the payload carrying exactly the fields the matching logic needs to filter and rank on:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="https://your-cluster.qdrant.io", api_key="...")

client.create_collection(
    collection_name="pr_review_comments",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
client.create_payload_index(collection_name="pr_review_comments", field_name="pr_id", field_schema="keyword")
client.create_payload_index(collection_name="pr_review_comments", field_name="file_path", field_schema="keyword")
client.create_payload_index(collection_name="pr_review_comments", field_name="line_start", field_schema="integer")
```

A single shared collection, filtered by `pr_id` on every query, is the right shape here rather than one collection per PR — PRs are short-lived, and creating and tearing down a Qdrant collection per PR adds operational overhead a payload filter avoids entirely, while the `pr_id` index keeps the filtered lookup cheap regardless of how many other PRs' comments sit in the same collection.

The `size=384` in the collection config reflects a deliberate model choice, not an arbitrary default: a small sentence-transformer (something in the `all-MiniLM-L6-v2` class, the same family used for BM42 sparse embeddings elsewhere on this blog) is a better fit here than a larger 1536-dimensional embedding model, precisely because of the latency constraint from the buffering design — this embedding call sits on the critical path for every single incoming comment, and review comments are short (a sentence or two, rarely more), which is exactly the regime where a smaller model's reduced representational capacity costs little accuracy while its lower inference latency and smaller vector size compound favorably with the per-comment cost accounting later in this section.

```python
import uuid

def upsert_comment(client: QdrantClient, embed_fn, comment: BufferedComment):
    signature = f"{comment.file_path}:{comment.line_start}-{comment.line_end} :: {comment.raw_text}"
    client.upsert(
        collection_name="pr_review_comments",
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=embed_fn(signature),
            payload={
                "pr_id": comment.pr_id,
                "file_path": comment.file_path,
                "line_start": comment.line_start,
                "line_end": comment.line_end,
                "bot_name": comment.bot_name,
                "raw_text": comment.raw_text,
                "has_suggestion": comment.has_suggestion,
            },
        )],
    )
```

Embedding the composite `{file_path}:{line_range} :: {comment_text}` signature rather than the comment text alone gives the vector itself a weak location signal too, but this is not relied on as the primary location match — the payload filter in the query below is the actual, hard constraint, because a signature-level embedding can still drift two nearby-but-distinct locations close together in vector space, which a strict filter cannot.

### The Filtered-Then-Ranked Query

This is the core operation: for each newly buffered comment, search the same PR's already-buffered comments, restricted by a hard filter on file path and a tolerance window on line number, and only *then* rank the filtered candidates by text-embedding similarity.

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

LINE_TOLERANCE = 3          # lines of anchor drift tolerated as "same region"
DUPLICATE_THRESHOLD = 0.87  # conservative — see threshold-tuning discussion below

def find_duplicate_candidates(
    client: QdrantClient, embed_fn, pr_id: str, comment: BufferedComment,
) -> list[dict]:
    signature = f"{comment.file_path}:{comment.line_start}-{comment.line_end} :: {comment.raw_text}"

    location_filter = Filter(must=[
        FieldCondition(key="pr_id", match=MatchValue(value=pr_id)),
        FieldCondition(key="file_path", match=MatchValue(value=comment.file_path)),
        FieldCondition(
            key="line_start",
            range=Range(
                gte=comment.line_start - LINE_TOLERANCE,
                lte=comment.line_start + LINE_TOLERANCE,
            ),
        ),
    ])

    results = client.query_points(
        collection_name="pr_review_comments",
        query=embed_fn(signature),
        query_filter=location_filter,
        limit=10,
        with_payload=True,
    )

    return [
        {"payload": p.payload, "similarity": p.score}
        for p in results.points
        if p.score >= DUPLICATE_THRESHOLD and p.payload["bot_name"] != comment.bot_name
    ]
```

The `file_path` exact match plus a small `line_start` tolerance window together form the hard gate — this is a pre-filter Qdrant applies during the vector search itself, not a post-hoc filter on results, so two comments on different files never even get compared on text similarity in the first place, regardless of how similar their phrasing might coincidentally be. The `bot_name != comment.bot_name` check at the end excludes a bot's own prior comment on the same spot (relevant when a bot re-posts or amends), since the goal here is cross-bot duplication, not a bot repeating itself.

### Consolidation Logic

Once duplicate candidates are found for a newly buffered comment, the gateway picks one representative to actually post and folds the rest into a footnote rather than dropping them silently — preserving the information that multiple bots agreed, which is itself a useful confidence signal for the developer, without showing the same finding four times:

```python
def select_representative(comment: BufferedComment, duplicates: list[dict]) -> dict:
    """Among a duplicate cluster, keep the most useful single comment."""
    all_candidates = [
        {"bot_name": comment.bot_name, "text": comment.raw_text, "has_suggestion": comment.has_suggestion},
        *[{"bot_name": d["payload"]["bot_name"], "text": d["payload"]["raw_text"],
           "has_suggestion": d["payload"]["has_suggestion"]} for d in duplicates],
    ]
    # Prefer a comment with an inline code suggestion over prose-only feedback.
    with_suggestions = [c for c in all_candidates if c["has_suggestion"]]
    representative = with_suggestions[0] if with_suggestions else all_candidates[0]

    others = [c["bot_name"] for c in all_candidates if c is not representative]
    return {
        "post_text": representative["text"],
        "footnote": f"_Also flagged by: {', '.join(others)}_" if others else None,
    }
```

The tie-break rule above — prefer whichever comment carries an inline code suggestion — is a reasonable default because a suggestion is directly actionable in a way prose alone isn't. A natural refinement, worth flagging as an extension rather than baking in from day one, is picking the representative based on which bot has historically had the better resolved-vs-dismissed ratio for that specific finding category on the given repo — a bot that's been right about null-check findings 90% of the time on this codebase is a better default representative for a null-check cluster than one whose null-check comments get dismissed more often, but that requires tracking outcome history per bot per category, which is a meaningfully bigger system than the gateway described here.

**Making the merge actually invisible to the developer** deserves a specific technical note, because the buffering description above slightly oversimplifies what "intercepting before it's shown" means in practice. Each bot is typically its own GitHub App, posting comments directly via the REST API the moment its review finishes — the gateway doesn't sit in front of that API call in the general case, since it doesn't control the bots' own posting logic. What it *can* do, having subscribed to the same `pull_request_review_comment` webhook event every bot's post triggers, is react within the buffer window: let the comment land, then immediately minimize it via GitHub's `minimizeComment` GraphQL mutation with a `DUPLICATE` classifier if the consolidation logic determines it's a duplicate, collapsing it to a single greyed-out "marked as duplicate" line in the GitHub UI, and post the gateway's own consolidated comment (with the footnote crediting all contributing bots) as the one that stays visible and expanded:

```python
MINIMIZE_COMMENT_MUTATION = """
mutation($id: ID!, $classifier: ReportedContentClassifiers!) {
  minimizeComment(input: {subjectId: $id, classifier: $classifier}) {
    minimizedComment { isMinimized }
  }
}
"""

async def minimize_duplicate(github_client, comment_node_id: str):
    await github_client.graphql(
        MINIMIZE_COMMENT_MUTATION,
        variables={"id": comment_node_id, "classifier": "DUPLICATE"},
    )
```

This means the true end-to-end sequence is: bot posts → gateway observes via webhook and buffers → window closes → gateway computes clusters → gateway posts one consolidated comment per cluster → gateway minimizes every other comment in that cluster. The duplicate comments technically exist for the buffer window's duration, but they're minimized before the window closes and before most developers are actively looking at the PR, which is a close practical approximation of "never shown" even though it isn't a literal pre-publish intercept.

**Handling review-level, non-line-anchored comments** is a related wrinkle the design above glosses over by focusing on inline `pull_request_review_comment` events. Some bots also post a PR-level summary comment — an overall review body not anchored to any specific line, often a paragraph or two of high-level observations. These have no `file_path` or `line_start` to filter on, so the location-based matching key doesn't apply to them at all. The practical answer is to treat review-level summaries as a separate matching pool from inline comments: embed the summary text alone (no location component in the signature), filter only by `pr_id`, and use a similarity threshold within that pool if multiple bots' summaries substantially restate each other — but hold this pool to an even more conservative threshold than the inline case, precisely because there's no location signal available to backstop a borderline text match.

**Latency budget.** The gateway sits between the bots posting and the developer seeing anything, so its own added delay is a real cost against the buffer window it already introduces. A rough per-comment accounting for the consolidation step itself, separate from the deliberate 30-90 second buffering wait:

| Step | Typical latency | Notes |
|---|---|---|
| Embedding the composite signature | 10–25ms | One embedding call per incoming comment |
| Filtered `query_points` against the PR's buffered comments | 5–15ms | Small candidate set — a single PR rarely has more than a few dozen buffered comments at once |
| Representative selection + footnote assembly | <5ms | Pure in-memory logic, no external calls |
| **Total added latency per comment** | **~20–45ms** | Negligible relative to the buffer window itself |

The buffer window, not the consolidation math, is where essentially all of the end-to-end delay lives — which is the right place for it, since the window's entire purpose is waiting for slower bots to finish, not computing anything. A repo with unusually slow bots (or a very large diff that pushes every bot's own review time out) is better served by lengthening `BUFFER_WINDOW_SECONDS` for that repo than by optimizing the consolidation query itself, which is already far from the bottleneck.

### The Webhook Handler: Tying It Together

The pieces above — buffering, embedding, the filtered-then-ranked query, minimizing duplicates — come together in the actual webhook endpoint the gateway exposes, which is what GitHub calls on every `pull_request_review_comment` event from every installed bot:

```python
import hashlib
import hmac
import os

from fastapi import FastAPI, Header, HTTPException, Request

app = FastAPI()
WEBHOOK_SECRET = os.environ["GITHUB_WEBHOOK_SECRET"].encode()

KNOWN_BOTS = {
    "coderabbitai[bot]", "cursor[bot]",
    "devin-ai-integration[bot]", "copilot-pull-request-reviewer[bot]",
}

def _verify_signature(body: bytes, sig_header: str) -> None:
    expected = "sha256=" + hmac.new(WEBHOOK_SECRET, body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig_header):
        raise HTTPException(status_code=401, detail="Invalid signature")

@app.post("/webhook")
async def handle_webhook(
    request: Request,
    x_hub_signature_256: str = Header(...),
    x_github_event: str = Header(...),
):
    body = await request.body()
    _verify_signature(body, x_hub_signature_256)

    if x_github_event != "pull_request_review_comment":
        return {"ok": True}

    payload = await request.json()
    if payload.get("action") != "created":
        return {"ok": True}

    author = payload["comment"]["user"]["login"]
    if author not in KNOWN_BOTS:
        return {"ok": True}  # never touch a human reviewer's comment thread

    comment = BufferedComment(
        pr_id=str(payload["pull_request"]["id"]),
        bot_name=author,
        file_path=payload["comment"]["path"],
        line_start=payload["comment"].get("start_line") or payload["comment"]["line"],
        line_end=payload["comment"]["line"],
        raw_text=payload["comment"]["body"],
        has_suggestion="```suggestion" in payload["comment"]["body"],
    )
    await handle_incoming_comment(comment)
    return {"ok": True}
```

Two details here matter beyond the now-familiar HMAC signature check. First, `author not in KNOWN_BOTS` is a hard gate evaluated before anything touches the buffer, and it's arguably the single most important line in the whole system: this gateway exists to consolidate machine-generated noise, and it must never minimize or otherwise touch a human reviewer's comment, even one that happens to resemble a bot's phrasing. A false minimize on a real person's feedback would be a far more damaging trust failure than any amount of duplicate bot noise, and no similarity threshold, however conservative, is a substitute for this categorical author check running first. Second, `has_suggestion` is computed once at ingestion by checking for GitHub's `` ```suggestion `` code-fence marker directly in the raw comment body, so the representative-selection logic later doesn't need to re-parse comment text — a small detail, but it keeps the actual consolidation step (the part with real decision logic worth reviewing carefully) free of incidental string-parsing.

### Extension: Outcome-Weighted Representative Selection

The representative-selection rule in the previous section defaults to preferring whichever comment carries an inline suggestion, which is a reasonable static heuristic but ignores a signal the gateway is well-positioned to accumulate over time: which bot's comments in a given finding category actually get acted on versus dismissed on this specific repository. Tracking that requires listening for one more signal — whether a posted comment's underlying suggestion was ultimately applied (via a follow-up commit touching the same lines) or the PR merged without any change at that location — and storing an outcome per `(bot_name, category)` pair:

```python
def record_outcome(client: QdrantClient, bot_name: str, category: str, resolved: bool):
    client.upsert(
        collection_name="bot_outcome_history",
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.0],  # outcome history doesn't need similarity search, only aggregation
            payload={"bot_name": bot_name, "category": category, "resolved": resolved},
        )],
    )

def resolved_ratio(client: QdrantClient, bot_name: str, category: str) -> float:
    records, _ = client.scroll(
        collection_name="bot_outcome_history",
        scroll_filter=Filter(must=[
            FieldCondition(key="bot_name", match=MatchValue(value=bot_name)),
            FieldCondition(key="category", match=MatchValue(value=category)),
        ]),
        limit=1_000,
    )
    if not records:
        return 0.5  # no history yet — neutral prior
    return sum(1 for r in records if r.payload["resolved"]) / len(records)
```

With this in place, `select_representative` can break ties by `resolved_ratio` instead of (or in addition to) suggestion presence: among a duplicate cluster, prefer the bot whose comments in that finding category have historically led to an actual code change on this repo. This is presented here as a real, buildable extension rather than something the core gateway needs from day one — it requires a category-classification step (grouping "null check," "missing test," "SQL injection risk," and so on into consistent buckets across differently-worded bot output) that's a meaningfully separate piece of engineering, and a cold-start period during which the neutral 0.5 prior governs every decision until enough outcome history accumulates to be trustworthy.

## Threshold Tuning: The Conservative Bias

`DUPLICATE_THRESHOLD` is the one number in this design with a real, asymmetric cost on both sides, and it's worth being explicit about which direction of error is worse before picking a default.

Set the threshold too low, and comments that are phrased similarly but describe genuinely different underlying issues get merged into one. This is the dangerous direction, not the annoying one: if a bot flags a real null-check bug and another bot, coincidentally using similar language, flags a *different* real bug at a nearby line that happens to fall inside the location-tolerance window, collapsing them means one of the two bugs silently disappears from the developer's view entirely. A false merge doesn't just create minor redundancy — it actively hides a finding that would otherwise have surfaced.

Set the threshold too high, and true duplicates stop getting merged, which means the consolidation layer simply fails to do its job and the original noise problem persists unaddressed — a failure mode that's real but bounded: it degrades back to the status quo rather than actively hiding information.

Given that asymmetry — a false merge hides a real bug, a false non-merge just wastes some attention — the right default leans conservative, and the location filter is what makes a conservative text threshold viable rather than merely safe-but-useless. Because two comments can only ever be compared for merging if they already passed the hard file-path-and-line-tolerance gate, a genuinely high `DUPLICATE_THRESHOLD` (0.85-0.90 range) on the text side doesn't need to do all the discriminating work alone — location has already ruled out the large majority of false-merge risk before text similarity is even computed. This is the practical version of the design principle from earlier: **a location mismatch cannot be salvaged by text similarity alone, but a location match plus a high text threshold is a defensible combined bar.** Pushing the text threshold down to catch more phrasing variation, once location is already doing real work as a filter, buys little additional recall at real risk to precision — the two knobs aren't substitutes for each other, and tightening the wrong one to compensate for a design choice made elsewhere is a common miscalibration to watch for.

| `DUPLICATE_THRESHOLD` | Behavior | Risk |
|---|---|---|
| 0.75 | Merges most same-topic comments within the location window | High collapse-risk — two distinct bugs described in similar general terms at nearby lines merge routinely |
| 0.87 (default above) | Merges clear phrasing variants of one underlying finding (the null-check example throughout this post) | A true duplicate phrased unusually differently from the others in a cluster can occasionally slip through unmerged |
| 0.95+ | Merges only near-identical text | Misses most real duplicates, since independently-prompted bots rarely produce more than roughly 90% textual similarity even when describing the same issue |

`LINE_TOLERANCE` deserves the same scrutiny as a secondary knob, though it moves the failure modes around less dramatically than the text threshold does. A tolerance of 0 (exact line match only) is defensible when bots reliably anchor comments to the precise line of the underlying issue, but real anchoring varies — one bot might anchor a multi-line conditional's null-check comment to the opening `if`, another to the specific line where the unguarded access happens two lines later. A tolerance of 2-5 lines accommodates that variance without meaningfully increasing collapse-risk, since it's still a small enough window that two genuinely unrelated bugs landing inside it purely by coincidence is rare on any but the most densely-packed diffs.

Neither number should be picked from a table alone and shipped. The more reliable path is running the matching logic in shadow mode — logging what clusters it *would* form without actually minimizing anything or posting a consolidated comment — against a sample of already-closed PRs that had active multi-bot review, then having a human spot-check whether those clusters look right. A few dozen historical PRs are usually enough to reveal whether the 0.87 default is producing too many false merges or too many missed duplicates for a given repo's specific bot mix and codebase vocabulary, before the gateway is trusted to act on its own judgment against live PRs.

## Installing Concord

Concord installs as a standard GitHub App, and its scoring internals ship as a separate pip-installable package for teams that want to run the matching logic themselves (self-hosted webhook receiver, custom posting logic) rather than use the hosted App directly:

```bash
pip install concord-review
```

```python
from concord.buffer import handle_incoming_comment, BufferedComment
from concord.matching import find_duplicates
from concord.store import CommentIndex

index = CommentIndex(url="https://your-cluster.qdrant.io", api_key="...")

await handle_incoming_comment(BufferedComment(
    pr_id="org/repo#4821",
    bot_name="cursor-bugbot",
    file_path="auth/session.ts",
    line_start=87, line_end=89,
    raw_text="Potential NPE: `user` is not null-checked before use here.",
    has_suggestion=False,
))
```

No changes are required to CodeRabbit, Cursor Bugbot, Copilot, or Devin Reviewer's own configuration — Concord subscribes to the same `pull_request_review_comment` webhook events GitHub already delivers for each of them and only changes what a developer sees after the buffer window closes.

## Worked Example: Four Bots, One Line

A concrete pass through the pipeline, with four bots returning findings on the same PR within the buffer window.

**CodeRabbit**, on `auth/session.ts:88-88`: *"This line could cause a null pointer exception if `user` is undefined."* Embedded and upserted first; no prior comments exist for this PR yet, so it posts with no duplicates found.

**Cursor Bugbot**, on `auth/session.ts:87-89`: *"Potential NPE: `user` is not null-checked before use here."* The location filter checks `file_path == "auth/session.ts"` (match) and `line_start` within `±3` of 87 (88 is within range — match). Text similarity against CodeRabbit's comment on the composite signature comes back around 0.91, comfortably above the 0.87 threshold. Verdict: duplicate. This comment gets folded into a footnote on CodeRabbit's original rather than posted separately.

**GitHub Copilot**, on `auth/session.ts:88-88`: *"Consider guarding against undefined `user` here before accessing its properties."* Same location match, and — despite sharing almost no tokens with either prior comment — a text similarity around 0.89 against the representative, since all three describe the identical semantic issue (an unguarded null dereference on the same variable at the same line). Also folded in.

**Devin Reviewer**, on `db/queries.ts:88-88`: *"This query could cause a null pointer exception if the result is undefined."* Same *phrasing pattern* as CodeRabbit's original comment, and the raw text embedding might score reasonably high in isolation — but the file path is `db/queries.ts`, not `auth/session.ts`. The location filter excludes it from the candidate set entirely, before text similarity is ever computed. This posts as a fully separate, top-level comment, correctly, because it's a different bug in a different file that happens to share a phrasing template.

The result posted to the developer: one comment from CodeRabbit with an inline note reading `Also flagged by: Cursor Bugbot, GitHub Copilot`, plus Devin Reviewer's distinct finding on the other file — two things to read and act on, down from four, with the cross-bot agreement on the first one preserved as a genuine confidence signal rather than discarded.

## Quantifying the Effect: What Consolidation Actually Removes

The worked example above is one instance; it's worth a short, deliberately illustrative model of why consolidation's leverage concentrates specifically on the correlated true-positive share of comments and does comparatively little to the uncorrelated false-positive share — which also explains why consolidation is a complement to, not a substitute for, the scoping and severity-tuning habits mentioned earlier.

Take a PR with $k$ active bots, $m$ genuine issues actually present in the diff, a per-bot, per-issue detection rate $p$ (the probability any given bot independently catches a given real issue), and an average false-positive rate $\mu$ per bot per PR, roughly independent across bots since each bot's specific hallucinations stem from its own idiosyncratic misreading of the diff rather than a shared cause. Without any consolidation, the expected total comment count is:

$$E[\text{comments}]_{\text{no dedup}} = k \cdot p \cdot m + k \cdot \mu$$

With consolidation collapsing every true duplicate perfectly (an optimistic assumption the real threshold-based system only approximates), each real issue produces exactly one surfaced comment as long as at least one bot caught it, while false positives — being idiosyncratic and rarely landing on the same location with high text similarity — largely fail to cluster and so pass through the gateway roughly unaffected:

$$E[\text{comments}]_{\text{dedup}} \approx m \cdot \left(1 - (1-p)^k\right) + k \cdot \mu$$

Plugging in illustrative values in the range these tools describe — $k=4$ bots, $p=0.75$ (a real, obvious issue is individually pretty likely to be caught by a capable reviewer bot), $m=2$ genuine issues in a typical mid-sized PR, and $\mu=1.5$ average false positives per bot per PR:

$$E[\text{comments}]_{\text{no dedup}} = 4 \cdot 0.75 \cdot 2 + 4 \cdot 1.5 = 6 + 6 = 12$$

$$E[\text{comments}]_{\text{dedup}} \approx 2 \cdot \left(1 - 0.25^4\right) + 6 = 2 \cdot 0.996 + 6 \approx 7.99$$

The true-positive contribution collapses from 6 expected comments down to essentially 2 (one per real issue, since at $p=0.75$ across 4 bots almost every real issue gets caught by at least one), while the false-positive contribution stays at 6, entirely untouched by deduplication. This is not a rigorous model of real bot behavior — real detection rates vary hugely by issue category and real false positives aren't perfectly independent — but the qualitative point holds regardless of the exact numbers: **consolidation's entire leverage is on the $k \cdot p \cdot m$ term, and it has essentially no effect on the $k \cdot \mu$ term.** A team relying on consolidation alone, without also working on the false-positive rate through path scoping and severity thresholds, will find the noise floor stays stubbornly high even after deduplication does everything it can — which is exactly why the mitigation habits mentioned earlier (narrower bot scope, smaller PRs that give bots less surface area to misread) are addressing a different term in this equation than the gateway is, and both matter.

## Deterministic Consensus, Not Another LLM Call

Every step in Concord's pipeline above — the location filter, the similarity threshold comparison, the representative-selection tie-break — is plain code operating on numbers Qdrant returned, not a prompt asking a model "are comments A and B the same finding?" This mirrors Quorum's explicit design choice, stated directly in its own documentation, that "quorum is computed in code from distinct reviewers, not by the model," and it's worth treating as a hard constraint on this design rather than an implementation detail.

The reasoning is the same one that shows up whenever a system is built specifically to reduce noise or hallucination: the consolidation layer itself must not become a new source of exactly the problem it exists to solve. An LLM asked to judge semantic equivalence between two review comments will sometimes get it wrong in ways that are neither predictable nor reproducible — the same pair of comments might get merged on one call and kept separate on a re-run with a different sampling seed, which makes the system's behavior impossible to reason about or debug when a developer asks "why did these two get merged?" A cosine similarity score against a location-filtered candidate set is deterministic: the same two comments, the same embeddings, the same threshold, always produce the same verdict, and the verdict is directly inspectable — the similarity number and the location distance are both concrete values a developer or an on-call engineer can look at and understand, not a model's unexplained judgment call. The embedding step itself does involve a model, but its output feeds a deterministic downstream comparison rather than being asked to render the merge decision directly, which keeps the one genuinely model-dependent part of the pipeline contained to a step whose failure mode (a slightly worse embedding) degrades gracefully into a slightly worse similarity score, rather than an outright wrong yes/no verdict.

## Challenges and Open Problems

**Near-duplicate comments about genuinely different bugs, phrased similarly, are a real and not-fully-solved failure mode.** The location filter handles the common case — different bugs are usually at different locations — but it doesn't fully eliminate the risk within a single tight code region. Two distinct issues on the same few lines (a null-check problem and a separate type-coercion problem, say, both plausibly describable in overlapping vocabulary) can still collide inside the location tolerance window and score above the text threshold. This is the collapse-risk discussed under threshold tuning, and no threshold setting eliminates it entirely — it only trades off how often it happens against how much redundancy survives.

**Bots that update or re-post comments on new commits need dedup state that's commit-aware, not just PR-aware.** The design above treats a PR as one buffering scope, but real PRs accumulate multiple commits, and a bot re-reviewing after a push may re-post a comment on a line that shifted, or restate a finding that a previous commit already triggered a (now possibly stale) consolidated comment for. Tracking dedup state keyed by `(pr_id, commit_sha)` rather than `pr_id` alone, and reconciling "is this a genuinely new finding on the new commit, or the same finding re-surfacing because the diff shifted," is real additional complexity this post's design doesn't fully address — it's scoped to the single-buffer-window case within one review round.

**A consolidation layer that's wrong even occasionally erodes exactly the trust it exists to protect.** This is worth stating as a standalone principle rather than folding into the threshold discussion, because it argues for a specific operational posture, not just a specific number: when the gateway is uncertain — a borderline similarity score, an edge-of-tolerance location match — the safer default is to show both comments rather than merge, even at the cost of occasional redundancy. A developer who discovers that the consolidation layer silently hid a real, distinct bug because two comments looked similar will stop trusting the layer entirely, which is a worse outcome than the noise problem it was built to fix — noisy-but-complete review output is recoverable by a developer choosing to skim faster; a review layer that occasionally deletes real findings without any indication it did so is not something a developer can compensate for by being more careful, since there's no visible signal that anything was hidden.

**The minimize-then-post mechanism has a brief visibility gap that a fast-moving developer can still hit.** Because true interception before publish isn't generally available across independently-operated bot GitHub Apps, there's an unavoidable window — the buffer duration — during which duplicate comments are technically live and un-minimized. A developer who opens the PR and starts reading during that window, rather than after it closes, sees the pre-consolidation state. This is a real gap, not just a theoretical one, on PRs where a human happens to be actively watching the review bots run in real time rather than checking back after they've finished; the mitigation is keeping the buffer window as short as the slowest common bot reasonably allows, not eliminating the gap, since eliminating it entirely would require every bot's own posting pipeline to route through the gateway first, which is not something a consolidation layer built independently of the bots can enforce.

## References

- dev.to (cseeman). "Return on Attention: Why AI Code Reviews Are Wearing Us Out." [dev.to/cseeman/return-on-attention-why-ai-code-reviews-are-wearing-us-out-2hh0](https://dev.to/cseeman/return-on-attention-why-ai-code-reviews-are-wearing-us-out-2hh0)
- Distill. "One review. Every bot. Every commit." [distillbot.com](https://distillbot.com/)
- Wenjix. Quorum — Consensus layer for AI code reviews. [github.com/Wenjix/quorum](https://github.com/Wenjix/quorum)
- Qdrant. Qdrant Vector Database Documentation. [qdrant.tech/documentation](https://qdrant.tech/documentation)
- GitHub Docs. Webhook events and payloads — `pull_request_review_comment`. [docs.github.com/en/webhooks/webhook-events-and-payloads](https://docs.github.com/en/webhooks/webhook-events-and-payloads)
