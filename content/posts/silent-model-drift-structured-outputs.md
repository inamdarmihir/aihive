---
title: "Silent Model-Version Drift: Catching Structured-Output Regressions After Provider Upgrades"
date: 2026-07-18
description: "A provider swaps the checkpoint behind a model alias, your schema validation still passes, and your parser starts breaking anyway. This post covers why structural validation can't catch conventions-level drift, and a Qdrant-backed canary harness that treats a model's output habits as an empirical baseline cluster rather than a single golden answer."
tags: ["llm", "reliability", "structured-outputs", "observability", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Every major provider — **OpenAI**, **Anthropic**, **Google** — ships model updates behind aliases that are not hard version pins: a `-latest` suffix, a default model name, or simply "the model behind this API key" can resolve to a materially different checkpoint on a schedule the provider controls, not you. None of this is secret; it's documented lifecycle policy. What's less discussed is the specific failure mode it creates for teams built on structured output, JSON mode, or function calling: a routine upgrade can change a model's output *conventions* — whether it wraps JSON in a markdown fence, how it represents an absent value, how verbose its string fields are, which casing an enum-like field uses — while the output remains perfectly schema-valid. Nothing throws. Nothing alerts. The pipeline just starts producing subtly different data than it did yesterday.

This post covers why schema validation alone cannot catch this class of regression, and how to build a canary evaluation harness that can — using a small, fixed, production-realistic prompt suite run on a schedule, and a Qdrant-backed behavioral baseline that treats "what does this model normally do" as an empirical cluster rather than a single fixed golden output. I'm scoping this deliberately: this is not a post about prompt injection, not about hallucination detection, and not a takedown of any specific provider or a specific dated incident — it's a systemic risk that shows up repeatedly across every provider's model lifecycle, and the mitigation is provider-agnostic. Familiarity with structured output / function calling and with basic vector search concepts is assumed.

## Table of Contents

1. [The Failure Mode: Schema-Valid, Convention-Broken](#the-failure-mode-schema-valid-convention-broken)
2. [Why Schema Validation Can't See This](#why-schema-validation-cant-see-this)
3. [A Taxonomy of Convention Drift](#a-taxonomy-of-convention-drift)
4. [Design: A Canary Eval Harness](#design-a-canary-eval-harness)
5. [Component One: The Canary Prompt Suite](#component-one-the-canary-prompt-suite)
6. [Component Two: Qdrant as the Behavioral Baseline](#component-two-qdrant-as-the-behavioral-baseline)
7. [Component Three: Baseline Cluster Construction](#component-three-baseline-cluster-construction)
8. [Component Four: The Drift-Scoring Query](#component-four-the-drift-scoring-query)
9. [Putting It Together: The Canary Runner](#putting-it-together-the-canary-runner)
10. [Worked Example: The Order-Status Prompt](#worked-example-the-order-status-prompt)
11. [Where This Fits Relative to Standard Practices](#where-this-fits-relative-to-standard-practices)
12. [Challenges and Open Problems](#challenges-and-open-problems)
13. [References](#references)

## The Failure Mode: Schema-Valid, Convention-Broken

Consider a support-ticket triage system that calls a model with a JSON-mode prompt asking for `{"category": string, "priority": "low"|"medium"|"high", "summary": string}`. The system has run against a pinned model version for months. A parser downstream does `json.loads(response.strip())`, reads `priority` as a value in a fixed enum set, and routes the ticket accordingly.

The team eventually migrates to a newer pinned snapshot — because the old one hit its retirement date, because a teammate bumped a `-latest` alias without realizing it wasn't actually pinned, or because a provider's default routing changed underneath an unpinned integration. The new checkpoint is, by every metric the team already tracks, an improvement: better instruction-following, fewer refusals, higher benchmark scores. It also, incidentally, now:

- Prefixes its JSON with a one-line acknowledgment ("Here's the categorized ticket:") before the payload, where the old version emitted bare JSON.
- Represents an unset `assignee` field as `"assignee": null` where the old version simply omitted the key.
- Writes `summary` fields that are on average 3x longer, because the new version's instruction-following is more literal about "provide a summary" without inferring the implicit brevity constraint the old version had learned to apply.
- Emits `"In Progress"` instead of `"in_progress"` for a status-like field the schema declared as a free-form string rather than a strict enum, because nothing in the schema actually pinned the casing convention — it was always an implicit contract, never an explicit one.

None of these breaks the JSON Schema. All four are individually plausible model behaviors that a capability benchmark would never flag as a regression — some of them look like improvements in isolation. Every one of them can break a downstream consumer that was written against the *previous* model's specific habits, silently: a parser that doesn't strip wrapper text before calling `json.loads` throws on the fenced or prefixed response, or — worse — a lenient parser that does regex-extract the JSON succeeds, and the pipeline continues on data whose enum values or null-handling convention no longer match what every downstream `if` statement was written to expect.

This is the shape of the risk this post addresses: **the model's contract with your code was never fully specified by the schema you validate against, and a provider-side model swap can silently violate the unspecified part.**

## Why Schema Validation Can't See This

It's worth being precise about the boundary, because "we validate against a JSON Schema" is the standard answer teams give when asked how they'd catch this, and it's an answer that addresses a different, narrower problem.

JSON Schema validation checks **structure**: are the required keys present, are the types correct, does an enum value (if declared as a strict `enum` in the schema) fall within the allowed set. It says nothing about:

- **Framing text.** A schema describes the shape of a JSON value; it has no vocabulary for "and there should be zero characters before the opening brace." A markdown-fenced or narrated response is, from the schema's perspective, not even a candidate for validation until *something* has already extracted the JSON substring — and whether that extraction step exists, and how lenient it is, is an implementation detail of your parser that most schema-validation setups treat as solved and never revisit.
- **Null vs. omission.** Unless a schema explicitly sets `"required": [...]` and separately forbids `null` as a type for optional fields (most don't bother, because both are "valid" under a loose schema), a model that switches from omitting an unset field to explicitly nulling it — or vice versa — passes validation either way. Code that does `"assignee" in response` behaves differently than code that does `response.get("assignee") is not None`, and only one of those survives the switch.
- **Verbosity within a field.** A `summary: string` field accepts a one-sentence string and a four-paragraph string identically. If a downstream UI truncates at 200 characters, or a database column has a length constraint, or a second LLM call embeds this field expecting it to be short, verbosity drift is invisible to the schema and fully visible to the downstream system.
- **Value-string conventions.** Unless a field is declared as a strict `enum`, nothing constrains `"in_progress"` vs `"In Progress"` vs `"IN_PROGRESS"` — they're all valid strings. Many real schemas under-specify enum-like fields precisely because the field's actual value space was inferred from observed model behavior rather than deliberately designed, which means the "contract" downstream code relies on was never written down anywhere a validator could check it.

The generalization: **schema validation enforces the parts of the contract someone thought to write down. Every convention a team's code implicitly depends on but never encoded as a schema constraint is, by definition, unprotected by that validation.** And because most of these conventions were never *chosen* — they were *observed*, from whatever the original model happened to do — nobody thought to write them down until they changed.

## A Taxonomy of Convention Drift

Four categories cover the vast majority of what shows up after a model-version bump, roughly ordered by how likely a naive pipeline is to break outright versus silently degrade:

**Wrapper and narration text.** The model surrounds otherwise-valid JSON with prose — a markdown code fence, an introductory sentence, a closing caveat. Depending on how strict the downstream parser is, this either raises immediately (strict `json.loads`) or gets silently stripped by a lenient extraction step, in which case the *break* is invisible but the *behavioral change* is real and worth knowing about even when nothing crashes.

**Null vs. omission conventions.** A field that used to be omitted when empty starts being emitted as `null`, or vice versa. This is invisible to type-only schema checks and often invisible in casual manual testing too, because a human skimming a sample response rarely notices whether a key is present-with-null versus absent.

**Verbosity shifts in string fields.** Free-text fields (summaries, descriptions, rationale fields) get longer or shorter as instruction-following characteristics shift between checkpoints. This rarely breaks a parser outright but can break UI truncation logic, downstream embedding-based retrieval (a summary field embedded for search behaves differently at 3x length), or per-token cost assumptions baked into a pricing model.

**Enum-value and casing convention drift.** Free-string fields that function as de facto enums (`status`, `category`, `priority`) shift casing, tokenization (`snake_case` vs `Title Case` vs space-separated), or the actual vocabulary used for a given concept (`"urgent"` vs `"high"` for the same underlying severity). This is the category most likely to cause silent misrouting — a `case status == "in_progress":` branch that never matches `"In Progress"` fails closed, without an exception anywhere in the stack.

The common thread across all four: they are differences in *habit*, not in *validity*, and a validator built to check validity has no mechanism to notice a change in habit.

## Cheap Heuristic Pre-Checks as a Complement

Before reaching for embeddings at all, a handful of hand-written heuristics catch a meaningful slice of drift cases cheaply, and are worth running unconditionally on every structured-output call in production, not just on the nightly canary suite — they cost microseconds and need no vector search.

```python
import re


def has_wrapper_text(raw_output: str) -> bool:
    """True if there is any non-whitespace content before the first '{'
    or after the last '}'."""
    stripped = raw_output.strip()
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace == -1 or last_brace == -1:
        return True  # no JSON object found at all is its own signal
    return bool(stripped[:first_brace].strip()) or bool(stripped[last_brace + 1:].strip())


MARKDOWN_FENCE = "`" * 3


def has_markdown_fence(raw_output: str) -> bool:
    return MARKDOWN_FENCE in raw_output


def string_field_lengths(parsed: dict, keys: list[str]) -> dict[str, int]:
    return {k: len(str(parsed[k])) for k in keys if k in parsed and isinstance(parsed[k], str)}


def enum_like_value_shape(value: str) -> str:
    """Classify a free-string enum-like value's casing/tokenization
    convention, so a shift between conventions is detectable without
    needing to know the specific vocabulary in advance."""
    if re.fullmatch(r"[a-z]+(_[a-z]+)*", value):
        return "snake_case"
    if re.fullmatch(r"[A-Z][a-z]*( [A-Z][a-z]*)*", value):
        return "Title Case"
    if re.fullmatch(r"[A-Z]+(_[A-Z]+)*", value):
        return "UPPER_SNAKE"
    return "other"
```

These heuristics are deliberately narrow and interpretable — `has_wrapper_text` alone would have caught the order-status regression in the worked example below before any embedding comparison ran. Run them inline in production, not just in the nightly canary, and log a lightweight counter (fraction of calls with wrapper text this hour, distribution of `enum_like_value_shape` outputs for a given field) that can itself be watched for a step change. They're cheap enough to run on 100% of production traffic where the embedding-based drift detector, scoped to a fixed canary suite, only samples a handful of prompts nightly.

What heuristics can't do is generalize past the specific patterns someone thought to encode — a verbosity shift, a subtly different word choice for the same underlying value, or a wrapper phrase that doesn't match any regex written in advance all slip past a fixed heuristic set. That's the gap the embedding-based canary detector below is built to close: it doesn't need to know in advance what kind of drift to look for, only that the new output is textually distant from what this exact prompt has produced before.

## Design: A Canary Eval Harness

The mitigation this post develops is a **canary eval harness**: a small, fixed suite of production-realistic structured-output prompts, run repeatedly against the exact model configuration used in production, on a schedule — nightly, and additionally whenever a provider announces or is suspected of rotating a model version behind an alias you use. This is emphatically not one-off ad hoc testing done once after a migration and never revisited; the value is in the repetition, because a canary suite run once tells you whether *this* output looks reasonable, while a canary suite run nightly against a stable baseline tells you the moment behavior starts moving.

The harness has four jobs:

1. **Run the same prompts, the same way, every time** — same temperature, same system prompt, same schema, same model configuration string used in production, so that any observed difference is attributable to the model rather than to harness drift.
2. **Check hard schema validity** — this is standard practice already and stays in place unchanged; it catches actual structural breaks.
3. **Check behavioral similarity to the trusted baseline** — this is the new layer, and it's where Qdrant comes in: embed the raw output (or a normalized representation of it) and compare it against a rolling cluster of past accepted outputs for that exact prompt, from the currently-trusted model version.
4. **Flag, don't auto-block** — a canary that falls outside its historical behavioral cluster is a signal for human review, not an automatic deployment gate, because a genuine model improvement can look identical to a regression from a pure similarity-distance perspective. More on this in the challenges section.

```
┌─────────────────────────────────────────────────────────────┐
│  Nightly / on-alias-rotation trigger                        │
│    │                                                         │
│    ▼                                                         │
│  Canary Runner                                               │
│    for each CanaryPrompt in suite:                           │
│      1. call model (prod config: model_version, temp, ...)   │
│      2. schema_valid = validate(output, prompt.schema)       │
│      3. drift = DriftDetector.score(output, prompt.id)  ──┐  │
│      4. if drift.is_anomalous or not schema_valid:         │  │
│             flag for human review                          │  │
│         else:                                               │  │
│             DriftDetector.accept(output, prompt.id)  ───┐  │  │
└────────────────────────────────────────────────────────┼──┼──┘
                                                           ▼  ▼
                                            Qdrant: canary_baselines
                                            (rolling window per
                                             prompt_id + model_version)
```

## Component One: The Canary Prompt Suite

The suite should be small enough to run nightly without meaningful cost, and representative enough that the prompts actually exercise the structured-output patterns production traffic depends on — not a generic "write me some JSON" smoke test.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class CanaryPrompt:
    prompt_id: str            # stable identifier, e.g. "order_status_v1"
    system_prompt: str
    user_prompt: str
    json_schema: dict[str, Any]
    model_config: dict[str, Any]   # {"model": "...", "temperature": 0.2, ...}
    tags: list[str]


ORDER_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "status": {"type": "string"},
        "eta_days": {"type": "integer"},
    },
    "required": ["order_id", "status"],
}

CANARY_SUITE: list[CanaryPrompt] = [
    CanaryPrompt(
        prompt_id="order_status_v1",
        system_prompt=(
            "You are an order-status API. Respond with a single JSON object "
            "matching the given schema. No other text."
        ),
        user_prompt='Order ORD-48213 shipped yesterday, arriving in 2 days.',
        json_schema=ORDER_STATUS_SCHEMA,
        model_config={"model": "gpt-x-2026-05-01", "temperature": 0.2},
        tags=["json_mode", "order_status"],
    ),
    CanaryPrompt(
        prompt_id="ticket_triage_v1",
        system_prompt=(
            "Classify the support ticket. Respond with a single JSON object "
            "matching the given schema. No other text."
        ),
        user_prompt=(
            "Customer reports checkout fails with a 500 error on mobile "
            "Safari only, started this morning, blocking all purchases."
        ),
        json_schema={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "priority": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["category", "priority", "summary"],
        },
        model_config={"model": "gpt-x-2026-05-01", "temperature": 0.2},
        tags=["json_mode", "ticket_triage"],
    ),
    # Realistic suites hold 20-60 of these, spanning every distinct
    # structured-output call site the production system actually makes —
    # not synthetic variety for its own sake.
]
```

The prompts are pinned verbatim, including the exact `model_config` used in production. Running the canary against anything other than the production configuration string defeats the point: you want to know what the production traffic is actually about to receive, not what some other, cleaner configuration would produce.

### Scheduling: Nightly and Event-Triggered

Two triggers matter, and they catch different kinds of drift.

**Nightly, on a fixed schedule**, catches an unannounced or unnoticed alias rotation — the case where nobody on the team did anything, but the provider's routing changed underneath a `-latest` alias, or a nominally-pinned snapshot's serving stack shifted in some subtler way. This is the trigger that matters most for the systemic risk this post is about, precisely because it requires no human action to fire.

**Event-triggered, on a provider's deprecation or version-rotation announcement**, catches the case where a migration is imminent and known in advance — a retirement date on the calendar, a new snapshot the team is about to adopt. Running the canary suite against the *new* candidate version, ahead of the actual cutover, turns "did anything change" from a question answered after the fact into one answered before the migration goes live. This is the same harness, pointed at a different `model_config.model` string, with results compared against the *current* production baseline rather than accumulated into it — a dry run, not a baseline update.

A simple cron-triggered job covers the nightly case; the event-triggered case is best wired to whatever the team already uses to track upstream deprecation notices (a scraped changelog, an internal ticket queue fed by the provider's deprecation page), so a canary run against the replacement model is scheduled automatically, or at minimum reminded, the moment a retirement date is known — rather than depending on someone remembering to do it manually in the scramble before a hard cutoff.

## Component Two: Qdrant as the Behavioral Baseline

Instead of a single fixed "golden" output per canary prompt, the baseline is a small rolling window of past *accepted* outputs — accepted meaning schema-valid and not itself flagged as drift at the time it was produced. This matters because even a stable model has legitimate variance at nonzero temperature: two runs of the same prompt against the same checkpoint will not be byte-identical, and a single golden-output comparison would either be too strict (flagging normal variance constantly) or too loose (a fixed string-similarity threshold wide enough to tolerate normal variance also tolerates a lot of real drift).

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

COLLECTION = "canary_baselines"
EMBED_DIM = 384  # e.g. all-MiniLM-L6-v2 or an equivalent small embedding model


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    for field_name, schema in [
        ("prompt_id", PayloadSchemaType.KEYWORD),
        ("model_version", PayloadSchemaType.KEYWORD),
        ("captured_at", PayloadSchemaType.FLOAT),
    ]:
        client.create_payload_index(
            collection_name=COLLECTION, field_name=field_name, field_schema=schema,
        )
```

Each point's payload holds exactly what's needed to scope a query to "the trusted baseline for this exact prompt, from this exact model version," plus the raw text for human review when something gets flagged:

```python
def normalize(raw_output: str) -> str:
    """Collapse whitespace but preserve wrapper text, casing, and punctuation —
    all of it is signal for convention drift, not noise to strip away."""
    return " ".join(raw_output.split())


def make_point(prompt_id: str, model_version: str, raw_output: str,
               captured_at: float, embed_fn) -> PointStruct:
    import uuid
    normalized = normalize(raw_output)
    return PointStruct(
        id=str(uuid.uuid4()),
        vector=embed_fn(normalized),
        payload={
            "prompt_id": prompt_id,
            "model_version": model_version,
            "raw_output": raw_output,
            "captured_at": captured_at,
        },
    )
```

`normalize` deliberately does *not* strip a markdown fence or narration prefix before embedding — the whole point is that the wrapper text is itself the signal a schema-only validator would miss. If a baseline is entirely bare JSON and a new output prefixes it with "Here's the order status:", embedding the raw (whitespace-normalized) text is exactly what makes that difference visible to a similarity search; embedding only the *extracted* JSON payload would throw the signal away before the comparison ever runs.

## Component Three: Baseline Cluster Construction

The baseline for a given `(prompt_id, model_version)` pair is the last $N$ accepted outputs, evicting the oldest once the window is full. $N = 50$ is a reasonable default — large enough to characterize normal temperature-induced variance, small enough that a genuine, intentional model-version transition can be fully re-seeded within a day or two of canary runs at typical nightly cadence.

```python
import time


class BaselineStore:
    def __init__(self, client: QdrantClient, embed_fn, window_size: int = 50):
        self.client, self.embed_fn, self.window_size = client, embed_fn, window_size
        ensure_collection(client)

    def _scope_filter(self, prompt_id: str, model_version: str) -> Filter:
        return Filter(must=[
            FieldCondition(key="prompt_id", match=MatchValue(value=prompt_id)),
            FieldCondition(key="model_version", match=MatchValue(value=model_version)),
        ])

    def accept(self, prompt_id: str, model_version: str, raw_output: str) -> None:
        """Add an output to the trusted baseline, evicting the oldest member
        of the window if it's already full."""
        point = make_point(prompt_id, model_version, raw_output, time.time(), self.embed_fn)
        self.client.upsert(collection_name=COLLECTION, points=[point])
        self._evict_overflow(prompt_id, model_version)

    def _evict_overflow(self, prompt_id: str, model_version: str) -> None:
        members, _ = self.client.scroll(
            collection_name=COLLECTION,
            scroll_filter=self._scope_filter(prompt_id, model_version),
            limit=self.window_size + 25,
            with_payload=True,
        )
        if len(members) <= self.window_size:
            return
        oldest_first = sorted(members, key=lambda p: p.payload["captured_at"])
        overflow_ids = [p.id for p in oldest_first[: len(members) - self.window_size]]
        self.client.delete(collection_name=COLLECTION, points_selector=overflow_ids)

    def reset(self, prompt_id: str, model_version: str) -> None:
        """Explicitly wipe a baseline — used when intentionally adopting a
        new model version, so the old cluster is never silently carried
        forward as if it still described current behavior."""
        self.client.delete(
            collection_name=COLLECTION,
            points_selector=self._scope_filter(prompt_id, model_version),
        )
```

`reset` exists for a reason discussed at length in the challenges section: a baseline built entirely from a model version you've already decided to retire needs to be explicitly torn down, not left to gradually age out, because until it's reset every output from the *new*, now-current version looks like drift relative to a baseline you've already chosen to move past.

## Component Four: The Drift-Scoring Query

Scoring a new output means asking "how similar is this to the cluster of outputs we already trust for this exact prompt and model version" and calibrating that similarity against how much the cluster naturally varies against itself.

```python
from dataclasses import dataclass
import numpy as np


@dataclass
class DriftVerdict:
    mean_similarity: float
    percentile: float | None
    is_anomalous: bool
    reason: str


class DriftDetector:
    def __init__(self, store: BaselineStore, embed_fn, k: int = 10):
        self.store, self.embed_fn, self.k = store, embed_fn, k

    def _mean_similarity(self, raw_output: str, prompt_id: str, model_version: str,
                          exclude_id: str | None = None) -> float | None:
        query_vec = self.embed_fn(normalize(raw_output))
        hits = self.store.client.query_points(
            collection_name=COLLECTION,
            query=query_vec,
            query_filter=self.store._scope_filter(prompt_id, model_version),
            limit=self.k + (1 if exclude_id else 0),
        ).points
        hits = [h for h in hits if h.id != exclude_id][: self.k]
        if not hits:
            return None
        return float(np.mean([h.score for h in hits]))

    def _historical_self_similarity(self, prompt_id: str, model_version: str) -> list[float]:
        """Leave-one-out mean similarity of every baseline member to its own
        k nearest neighbors within the cluster — this is the empirical
        distribution of 'how similar does a normal output look to the rest
        of the cluster,' used to calibrate what counts as anomalous."""
        members, _ = self.store.client.scroll(
            collection_name=COLLECTION,
            scroll_filter=self.store._scope_filter(prompt_id, model_version),
            limit=self.store.window_size,
            with_payload=True,
        )
        sims = []
        for m in members:
            sim = self._mean_similarity(m.payload["raw_output"], prompt_id, model_version,
                                         exclude_id=m.id)
            if sim is not None:
                sims.append(sim)
        return sims

    def score(self, raw_output: str, prompt_id: str, model_version: str) -> "DriftVerdict":
        new_sim = self._mean_similarity(raw_output, prompt_id, model_version)
        history = self._historical_self_similarity(prompt_id, model_version)
        if new_sim is None or len(history) < 10:
            # Cold start: not enough baseline to calibrate against yet.
            return DriftVerdict(mean_similarity=new_sim or 0.0, percentile=None,
                                 is_anomalous=False, reason="insufficient_baseline")
        percentile = float(np.mean([1 for s in history if s < new_sim]) / len(history))
        is_anomalous = percentile <= 0.05  # new output sits at/below the 5th percentile
        return DriftVerdict(mean_similarity=new_sim, percentile=percentile,
                             is_anomalous=is_anomalous,
                             reason="behavioral_drift" if is_anomalous else "within_baseline")
```

Formally, let $S = \{s_1, \ldots, s_n\}$ be the leave-one-out mean similarity of each baseline member to its $k$ nearest neighbors within its own cluster, and let $s_{\text{new}}$ be the mean similarity of the new output to its $k$ nearest neighbors in that same cluster. The drift signal is the percentile rank of $s_{\text{new}}$ within $S$:

$$
p = \frac{|\{s \in S : s < s_{\text{new}}\}|}{|S|}
$$

Flag as anomalous when $p \le 0.05$ — the new output is less self-similar to the trusted cluster than 95% of the cluster's own members are to each other. This calibrates the threshold to *this specific prompt's* natural variance rather than a single fixed cosine-similarity cutoff applied uniformly: a canary prompt that legitimately produces more varied phrasing (a summary field, say) will have a wider historical self-similarity spread than a canary prompt whose schema is tightly numeric, and the percentile approach adapts to that automatically rather than requiring a hand-tuned threshold per prompt.

## Putting It Together: The Canary Runner

```python
async def run_canary_suite(
    suite: list[CanaryPrompt],
    model_client,          # thin wrapper around the provider SDK in use
    store: BaselineStore,
    detector: DriftDetector,
    validate_schema,       # e.g. jsonschema.validate, wrapped to return bool
) -> list[dict]:
    results = []
    for prompt in suite:
        raw_output = await model_client.complete(
            system=prompt.system_prompt, user=prompt.user_prompt,
            **prompt.model_config,
        )
        model_version = prompt.model_config["model"]

        schema_ok = validate_schema(raw_output, prompt.json_schema)
        drift = detector.score(raw_output, prompt.prompt_id, model_version)

        flagged = (not schema_ok) or drift.is_anomalous
        if not flagged:
            store.accept(prompt.prompt_id, model_version, raw_output)

        results.append({
            "prompt_id": prompt.prompt_id,
            "model_version": model_version,
            "schema_valid": schema_ok,
            "drift_percentile": drift.percentile,
            "drift_flagged": drift.is_anomalous,
            "needs_review": flagged,
            "raw_output": raw_output,
        })
    return results
```

The runner never blocks a deployment by itself — it produces a report. A flagged row routes to whatever review surface the team already uses (a Slack alert, a dashboard, a ticket), with the raw output attached so a human can see exactly what changed. Only outputs that pass *both* checks get added to the baseline, which keeps the cluster from slowly absorbing drifted outputs as if they were normal — an important property, discussed further below.

## Worked Example: The Order-Status Prompt

Take the `order_status_v1` canary from Component One, run nightly for three weeks against the trusted model version. The baseline cluster accumulates 50 outputs, all bare JSON of the shape:

```json
{"order_id": "ORD-48213", "status": "shipped", "eta_days": 2}
```

Minor variance exists — `eta_days` differs across nights depending on the (deterministic, hardcoded) prompt input's implied dates, and occasionally `status` reads `"in_transit"` instead of `"shipped"` at temperature 0.2 — but every member is bare JSON, no wrapper text, snake_case status values. The historical self-similarity distribution $S$ for this cluster sits tightly around a mean of roughly 0.93, with a 5th percentile around 0.86.

The provider rotates the alias behind `gpt-x-2026-05-01` to a newer checkpoint (this can happen even with what looks like a pinned dated snapshot, if the pin is to a family alias rather than a true immutable snapshot ID — see the references for how this distinction is documented per provider). The next nightly canary run produces:

```
Here's the order status:

{"order_id": "ORD-48213", "status": "Shipped", "eta_days": 2}
```

Two things changed at once: a narration prefix appeared, and `status` shifted from `"shipped"` to `"Shipped"`. Walking through both checks:

**Schema validation.** If the parser does naive `json.loads(response)`, this raises immediately — a hard, loud break, caught the moment anyone looks at the canary run's error log. If the parser instead does a regex-based JSON extraction before validating (a common leniency added specifically to tolerate minor formatting noise), the extracted `{"order_id": "ORD-48213", "status": "Shipped", "eta_days": 2}` is schema-valid under a schema that types `status` as a plain string. The pipeline reports success. This is the dangerous branch: nothing in the schema check indicates anything happened.

**Drift scoring.** Independent of which parsing branch the team happens to be running, the drift detector embeds the raw, un-extracted text and compares it against the baseline cluster. The new output's mean similarity to its 10 nearest baseline neighbors comes out around 0.58 — well below the cluster's own 5th percentile of 0.86 — because the wrapper sentence and the casing change both show up as substantial textual distance from a cluster that has never contained either. The output is flagged, `percentile ≈ 0.0`, `is_anomalous = True`, regardless of which parser branch would have technically "succeeded."

A representative drift-score table across that night's canary run, alongside two other unaffected canaries in the suite:

| Canary prompt | Baseline mean self-similarity | New output similarity | Percentile vs. baseline | Schema valid? | Flagged |
|---|---|---|---|---|---|
| `order_status_v1` | 0.93 | 0.58 | ~0.0 | Yes (lenient parser) / No (strict) | **Yes** |
| `ticket_triage_v1` | 0.89 | 0.91 | 0.62 | Yes | No |
| `refund_eligibility_v1` | 0.91 | 0.87 | 0.34 | Yes | No |

The two unaffected canaries show the detector isn't hair-triggered on ordinary temperature-induced variance — their new outputs land comfortably within the historical spread. Only the prompt whose output actually changed shape gets flagged, and it gets flagged *even in the lenient-parser scenario where the schema check alone would have reported a clean pass.* That's the specific gap this system closes: a human sees the flag, opens the raw output, and immediately understands what changed and why it matters, well before it shows up as a customer-facing ticket-routing bug three weeks later when someone finally notices `case "in_progress":` branches silently stopped matching.

## Human Review Workflow

A flagged canary is only useful if it reaches a human who can act on it quickly, with enough context to decide in seconds rather than minutes whether it's a real regression, a benign improvement, or noise. The report from `run_canary_suite` carries everything needed for that decision — the prompt, the raw output, the drift percentile, and (for comparison) a sample of recent baseline members — so the review surface doesn't need a second round-trip to Qdrant to render a useful diff.

```python
FENCE = "`" * 3


def render_review_card(result: dict, baseline_sample: list[str]) -> str:
    lines = [
        f"**Canary flagged: `{result['prompt_id']}`** (model: `{result['model_version']}`)",
        f"Schema valid: {result['schema_valid']} | Drift percentile: {result['drift_percentile']}",
        "",
        "**New output:**",
        FENCE,
        result["raw_output"],
        FENCE,
        "",
        "**Recent trusted baseline (sample):**",
    ]
    for sample in baseline_sample[:3]:
        lines += [FENCE, sample, FENCE]
    lines.append(
        "\nReply `accept` to adopt this behavior into the baseline going forward, "
        "or `reject` to leave the baseline unchanged and escalate to engineering."
    )
    return "\n".join(lines)
```

The two possible human responses map directly onto the two operations already defined on `BaselineStore`: `accept` calls `store.accept(...)` for the flagged output (folding this specific instance into the trusted cluster without necessarily resetting the whole window — useful for a one-off legitimate variation), while a broader "we're adopting this model version going forward" decision calls `store.reset(...)` followed by a run of fresh canary calls to rebuild the baseline cleanly under the new version. Keeping these as two distinct actions, rather than collapsing "accept this one output" and "adopt this new baseline wholesale" into a single button, avoids quietly widening the baseline one flagged sample at a time until it no longer represents anything coherent.

## Where This Fits Relative to Standard Practices

This system is additive, not a replacement for anything a mature LLM-integrated team should already be doing.

**Pinning explicit model version strings is necessary but not sufficient.** Pinning to a dated snapshot ID rather than a rolling alias removes the most common source of *surprise* drift — you stop finding out about a swap because your parser broke in production. It does not remove the risk entirely: a provider's serving stack can shift subtly even for a nominally fixed snapshot (sampling implementation details, infrastructure-level changes to how a checkpoint is served), and every pinned version, however stable while active, is eventually retired on the provider's own schedule — Anthropic and OpenAI both publish deprecation calendars with advance-notice windows, but "advance notice" still means an eventual, sometimes tight-deadline migration to a new version whose behavior you must validate, not an option to stay on the old one forever. Pinning buys you control over *when* you migrate; it doesn't buy you certainty about *what changes* when you do.

**A canary-eval-on-every-deploy discipline is standard practice this system extends, not competes with.** Most teams that take structured output seriously already run some form of canary or smoke test before promoting a new model version to production — checking schema validity, checking a handful of expected-value assertions, sometimes checking latency and cost. The Qdrant-backed behavioral layer described here is an addition to that discipline: it catches the category of regression that passes every check a standard canary suite already runs, because the standard suite is validating structure and, at best, a few hand-picked value assertions — not the full space of conventions a downstream system has come to depend on.

**This is a human-review trigger layered on top of automated gates, not a replacement for either.** The schema check remains the hard gate that should block a deploy outright on structural failure. The drift check is a softer, complementary signal that something worth a human's attention has changed, even when nothing structurally broke.

## Challenges and Open Problems

**The canary suite is only as representative as the prompts chosen.** A drift that manifests specifically on production traffic patterns the canary set doesn't cover — an unusual input shape, a rare branch of the schema, a locale-specific phrasing — will not be caught by this system no matter how well-calibrated the drift scoring is. The canary suite needs periodic review against actual production traffic to stay representative, and that review is a manual, ongoing cost this system doesn't eliminate.

**A genuine model improvement triggers the same flag as a regression.** A newer checkpoint with better instruction-following might produce a response that's meaningfully *better* — more accurate, more appropriately concise — while still sitting outside the old baseline's cluster, because "better" and "different" are not distinguishable from a pure embedding-similarity signal. This is precisely why the system is designed as a human-review trigger and not an automatic deployment block: the detector's job is to make sure a human looks at every case where behavior changed, not to adjudicate whether the change is good or bad. Treating a drift flag as an automatic rollback trigger would block legitimate improvements as readily as it blocks regressions.

**Baseline clusters need explicit staleness handling.** Once a team reviews a flagged drift and decides the new behavior is acceptable — intentionally adopting the new model version — the old baseline, built entirely from the retired version's outputs, must be explicitly reset (via the `reset` method in Component Three) rather than silently left in place. If it's left in place, every subsequent output from the newly-adopted version will continue to score as drift relative to a baseline the team has already decided to move past, producing a permanent stream of false-positive flags that will, in practice, train the team to ignore the alert entirely — which defeats the entire system the next time a *real* regression occurs. The reset step is a deliberate, logged action tied to a human decision ("we are adopting version X as of today"), not something that should ever happen automatically based on volume of flagged outputs alone, since automatic reset-on-drift would just as readily paper over a real regression as it would clear a stale baseline after an intentional upgrade.

## References

- OpenAI. [API Deprecations](https://developers.openai.com/api/docs/deprecations). Documents provider-side model retirement schedules and the distinction between rolling aliases and dated snapshot IDs.
- Anthropic. [Model IDs and Versions](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions). Documents the pinned-snapshot vs. convenience-alias distinction and the guarantee (and its limits) that a given model ID's weights don't change underneath you.
- Anthropic. [Model Deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations). Current retirement schedule and notice-period policy for Claude models.
- Qdrant. [Vector Database Documentation](https://qdrant.tech/documentation/). Reference for `query_points`, filtering, and collection configuration used throughout this post.
