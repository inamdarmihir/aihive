---
title: "Fixing Progressive Latency Drift in Long Realtime Voice Agent Sessions"
date: 2026-07-30
description: "Voice agents built on OpenAI's Realtime API get slower the longer a session runs, and the standard fix — pruning or summarizing conversation history — doesn't reliably reset it. This post proposes a structurally different mitigation: a live, retrieval-seeded session profile backed by Qdrant, plus a per-turn latency probe that rotates sessions before drift becomes audible."
tags: ["agents", "voice", "realtime-api", "latency", "reliability", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

A voice agent that feels conversational in the first two minutes of a call and noticeably laggy by minute eight is not a rare failure mode — it is, as of mid-2026, a reported and unresolved characteristic of long-running sessions on OpenAI's Realtime API. In this post I look at what's actually been documented about this drift, why the mitigation most teams reach for first (pruning or summarizing conversation history) does not reliably fix it, and propose a structurally different approach: replace the growing, compacted transcript with a small, continuously retrieved slice of structured session state. This post does not cover speech-to-text accuracy, voice cloning, or turn-taking/VAD tuning in general — those are separate problems with their own literature. I also want to be upfront about scope on the harder question: OpenAI has not published a root-cause explanation for why latency climbs even after context is pruned, so what follows is a plausible, evidence-informed mitigation for one hypothesis, not a confirmed fix. Familiarity with the Realtime API's event model (`conversation.item.create`, `conversation.item.delete`, `response.create`) is assumed.

## Table of Contents

1. [The 800ms Budget and Why It's Already Tight](#the-800ms-budget-and-why-its-already-tight)
2. [What Progressive Latency Drift Actually Looks Like](#what-progressive-latency-drift-actually-looks-like)
3. [The Standard Mitigation, and Its Documented Limit](#the-standard-mitigation-and-its-documented-limit)
4. [Transport and Tool Calls: The Other Latency Sources](#transport-and-tool-calls-the-other-latency-sources)
5. [Reframing the Problem: Symptom vs. Structural Fix](#reframing-the-problem-symptom-vs-structural-fix)
   - [Naming the Competing Hypotheses Explicitly](#naming-the-competing-hypotheses-explicitly)
6. [Component One: The Session Fact Store](#component-one-the-session-fact-store)
   - [Deduplicating Facts Before They Accumulate](#deduplicating-facts-before-they-accumulate)
   - [Facts That Persist Across Calls, Not Just Within One](#facts-that-persist-across-calls-not-just-within-one)
7. [Component Two: Incremental Fact Extraction](#component-two-incremental-fact-extraction)
8. [Component Three: Retrieval-Based Session Reseeding](#component-three-retrieval-based-session-reseeding)
   - [Why Facts Alone Aren't Quite Enough: A Hybrid Reseed](#why-facts-alone-arent-quite-enough-a-hybrid-reseed)
9. [Component Four: A Per-Turn Latency Drift Monitor](#component-four-a-per-turn-latency-drift-monitor)
   - [The Window-Size Tradeoff](#the-window-size-tradeoff)
   - [Measuring Turn Latency Precisely](#measuring-turn-latency-precisely)
10. [Putting It Together: The Rotation Flow](#putting-it-together-the-rotation-flow)
    - [Where This Logic Has to Live: Server-Mediated Sessions](#where-this-logic-has-to-live-server-mediated-sessions)
11. [Validating the Rotation Trigger Offline](#validating-the-rotation-trigger-offline)
12. [Worked Example](#worked-example)
13. [Does This Generalize Beyond OpenAI's Realtime API?](#does-this-generalize-beyond-openais-realtime-api)
14. [Challenges and Open Problems](#challenges-and-open-problems)
15. [References](#references)

## The 800ms Budget and Why It's Already Tight

Before discussing what goes wrong over a long session, it's worth establishing the budget a voice agent is working with even in the best case, because that budget is the reason drift is operationally serious rather than cosmetic. Human conversational turn-taking latency is roughly 200ms — the gap most people leave between the end of someone else's sentence and the start of their own. A voice agent responding in around 800ms feels "digital but acceptable"; above roughly 1.2 seconds, users reliably start talking over the agent, because the pause reads as "finished," not "still thinking." An 800ms voice-to-voice budget is achievable but not free. Time-to-first-byte from the Realtime API is roughly 500ms in US regions on its own, which leaves only about 300ms for everything else: microphone capture and encoding (20-40ms), network transit from client to agent and from agent to OpenAI's endpoint (30-140ms combined, depending on regional co-location), voice-activity-detection and turn-detection silence thresholds (200-400ms, tunable via `silence_duration_ms`), and audio streaming back to the client for rendering (50-100ms) ([Forasoft, *OpenAI Realtime API: Production Voice Agents*, 2026](https://www.forasoft.com/blog/article/openai-realtime-api-voice-agent-production-guide-2026)).

The practical upshot: there is very little slack in this budget to begin with. A 500ms model-side TTFB consumes over 60% of the target before a single millisecond of client-side work happens. This is why the phenomenon in the next section — turn latency climbing well past this budget as a session runs longer — is not a minor degradation; it's the difference between an agent that stays inside the "acceptable" zone for the entire call and one that crosses into "users start talking over it" territory somewhere in the middle.

## What Progressive Latency Drift Actually Looks Like

The clearest documentation of this problem is not a paper — it's a developer's own words in [OpenAI's Developer Community forum](https://community.openai.com/t/progressive-latency-increase-in-long-gpt-realtime-1-5-gpt-realtime-voice-sessions-including-non-tool-turns-not-fully-reset-by-conversation-item-deletion/1376004), reporting behavior across both **gpt-realtime** and **gpt-realtime-1.5**:

> "We are seeing a consistent progressive latency increase in long-running voice sessions with gpt-realtime-1.5. We use the model in a real-time phone / voice interview workflow. Early in the session, the model is fast and responsive. As the session gets longer, the latency grows significantly. By the later parts of the call, the end-to-end delay from user answer to next assistant question can be roughly 3x or more compared to the beginning."

Concretely: a session that opens with turn latency around the 800ms budget outlined above can, by turn 20 or later, be running at 2 seconds or more per turn — squarely past the point where users start interrupting because the pause reads as a dropped call rather than a thinking agent. This isn't limited to tool-call turns, where you might expect extra round-trip cost; the report is explicit that plain conversational turns with no function calling exhibit the same climb. And it isn't an artifact of the reporting team's own infrastructure — the same thread notes: "Receiving and processing ws events are decoupled on our end, so we can ensure the delay is happening on OpenAI's end because the delay happens on the receiving side (not processing)," and separately, "our own server-side processing time increases only moderately over the same period, so the main slowdown appears to be on the model side, not in our orchestration layer." A related, though mechanically distinct, class of reports on the same forum describes intermittent multi-second delays in the `output_audio_buffer.stopped` event specifically — for instance a [thread reporting 6-10 second delays](https://community.openai.com/t/big-delays-receiving-output-audio-buffer-stopped-event-since-apr-18th/1379823) between a response finishing and the client being told it finished, which breaks turn-taking logic that gates re-enabling input detection on that event. I treat these as adjacent but separate symptoms in this post — the fact-store approach below targets the progressive, session-length-correlated drift specifically, not buffer-event delivery bugs, which are a transport/infrastructure issue outside the scope of anything a client-side architecture change can fix.

## The Standard Mitigation, and Its Documented Limit

The mitigation most teams reach for first is exactly what you'd expect from the "it's the growing transcript" hypothesis: periodically delete old conversation items via `conversation.item.delete`, or replace a block of old turns with a short compacted summary injected into instructions, keeping the live item count bounded. A second variant, common enough to be closer to a house style than a workaround, is to rotate to an entirely fresh Realtime session every 8-12 turns, reseeding the new session's system instructions from a summary of the old one rather than pruning in place.

Both variants share the same underlying assumption: that latency is a function of *how much raw or summarized text the model has to carry forward*, and that shrinking that text should shrink the latency. The forum report above tested this directly and found it doesn't hold cleanly:

> "What makes this especially confusing is that we already clear old conversation items during the session, and we also compact old context into a short summary. So we would expect latency to improve after a memory cleanup, but in practice the model often remains slow even after cleanup... So the issue does not appear to be explained simply by 'too many raw conversation items still present.'"

This is the load-bearing fact for everything that follows. If pruning-in-place reliably reset latency, the engineering problem would already be solved and this post wouldn't need to exist. It doesn't, at least not reliably, which means whatever's driving the slowdown on the model side is either not fully captured by "conversation item count" as OpenAI's own API models it, or the model-side session state that accumulates cost isn't cleanly reset by client-issued deletion events even when the visible, enumerable items are gone. Neither the forum thread's authors nor OpenAI (as of this writing, in the same thread) have published a confirmed explanation. That's an important admission to make explicitly before proposing anything: I am not claiming to know the root cause. I'm proposing an approach that removes one plausible contributing factor — the requirement that the model re-derive meaning from an accumulated or compacted block of prior-turn text at the exact moment a fresh session starts — on the theory that even a *partial* mitigation of a poorly understood problem is worth having, provided its cost is bounded and its failure mode is graceful.

## Transport and Tool Calls: The Other Latency Sources

It's worth being precise about which parts of a voice agent's latency budget are fixed, bounded, per-turn costs, and which are unbounded, growing-with-session-length costs — because the correct amount of engineering effort to spend on each is different, and conflating them leads to over-investing in the wrong place.

**Transport (WebRTC vs. WebSocket)** is a fixed-per-turn architectural choice, not something that degrades over a session's lifetime. For browser and mobile clients, native WebRTC is the right default: it skips a buffering layer that WebSocket introduces client-side, giving more consistent frame-by-frame audio delivery and lower jitter. But WebSocket isn't strictly worse — it's necessary whenever a server needs to sit in the connection path to mediate compliance requirements, audit logging, or custom business logic before audio or tool calls reach the model, and the extra 80-150ms hop that server-mediation costs is a reasonable trade for teams that need it. This is a one-time architectural decision made per deployment, not a per-turn variable, and it doesn't get worse the longer a call runs.

**Tool-call round-trips** add a further 400-800ms to any turn that invokes a function, when the tool itself responds in under 200ms — the model has to pause audio generation, wait on the tool result, and then resume speaking coherently, and users feel that pause distinctly even when the total time is well within the tool's own SLA. This cost is also roughly fixed per tool-call turn: it doesn't compound as the session gets longer, it's just a tax on whichever specific turns happen to invoke a function.

Contrast both of these with **accumulated session context**, which is the one latency source in this list that is *not* bounded per-turn — it's a function of how many turns have already happened, and per the forum reports, it appears to keep growing even under the standard prune/summarize mitigation. That asymmetry is the argument for where to spend engineering effort here: transport choice and tool-call latency are worth optimizing once and then treating as fixed costs, while the growing-with-session-length cost is the one that, left unaddressed, eventually dominates every other line item in the budget, on a long enough call. That's the piece this post is about attacking structurally, not the others.

## Reframing the Problem: Symptom vs. Structural Fix

Both variants of the standard mitigation — pruning items in place, or rotating to a new session reseeded from a summary — attack the same symptom: a long raw transcript. What neither variant changes is the *shape* of the fix: at the moment a session needs fresh state, the model is handed a compacted, lossy text representation of "everything that happened before" and has to re-derive meaning from it before it can respond coherently. A summary is, definitionally, an imperfect compression of the original conversation — details get dropped, structure gets flattened into prose, and the exact facts that matter for the current turn are mixed in with facts that don't matter at all right now. Asking the model to reconstruct working context from that compressed blob, at precisely the moment you want a fast, fresh session, is asking it to do meaningful inference work disguised as a formality.

The alternative I want to develop here is structurally different, not just a smaller version of the same thing: instead of accumulating a transcript and periodically compacting it, maintain a **live, continuously-updated session profile** per caller — the concrete facts, stated preferences, and current task state, extracted incrementally as the conversation happens — stored as structured records plus embeddings, never as a growing block of prose. When a session rotates, the new session isn't seeded with a summary of everything that happened; it's seeded with a small, semantically retrieved slice of facts relevant to *what's actually being discussed right now*, queried against the fact store using the current turn's content. This is a bet on a specific hypothesis about the drift: that some meaningful share of the "growing state" that isn't reset by pruning is a *side effect of the summarization/reconstruction workload itself*, not purely a property of raw item count — and that retrieval-based seeding sidesteps that workload entirely, because the new session is never asked to re-derive anything from an accumulated text block. It hands the model a short list of already-distilled facts instead.

I want to flag directly that this is a hypothesis test, not a guaranteed fix, and come back to that honestly in the closing section. What it does unambiguously do, independent of whether it fully resolves the model-side drift, is bound the size and freshness of what gets reseeded on every rotation — which is a property worth having on its own even if the deeper cause turns out to be something entirely outside client control.

### Naming the Competing Hypotheses Explicitly

It helps to write out the two hypotheses this design is choosing between, since the forum evidence rules out one but doesn't confirm the other. Let $C(n)$ denote the model-side latency cost at turn $n$ of a session.

**H1 — item-count hypothesis:** $C(n) \approx f(\text{items}(n))$, where $\text{items}(n)$ is the number of live conversation items at turn $n$. Under H1, pruning items back down (via `conversation.item.delete` or summarization) should push $C(n)$ back down proportionally, because cost is modeled as a function of what's currently enumerable in the conversation. The forum's own cleanup test — delete items, replace with a summary, observe latency stay elevated — is direct evidence against a *pure* version of H1: if item count were the whole story, cleanup would have worked.

**H2 — accumulated-session-state hypothesis:** $C(n) \approx g(n)$ directly, as a function of turns elapsed or wall-clock session duration, largely independent of how many items happen to be enumerable at any instant. Under H2, the only way to reset $C(n)$ is to end the session and start a structurally new one — pruning within a still-open session doesn't touch whatever state $g(n)$ depends on.

The evidence favors rejecting a pure H1 over confirming H2 specifically — ruling out one explanation is not the same as proving another. It's entirely possible the true function has a form closer to $C(n) \approx g(n) + \epsilon(\text{provider infrastructure effects uncorrelated with anything client-visible})$, in which case full session rotation helps because it resets $g(n)$, but the retrieval-vs-summary distinction proposed in this post turns out not to matter much, since the summarization approach *also* opens a fresh session on rotation. I think the retrieval-based reseed is still worth preferring over summary-based reseed even under this more pessimistic reading, because it's strictly cheaper to compute (no summarization pass needed) and bounds context size more tightly regardless of which hypothesis turns out to be true — but I want to be honest that the strongest form of the argument in this post (that retrieval-seeding specifically, as opposed to any full session rotation, is what closes the gap) rests on H2 in a fairly specific form that the public evidence doesn't yet nail down.

## Component One: The Session Fact Store

The fact store is a Qdrant collection holding one point per extracted fact, scoped by caller and session, queryable by semantic similarity to whatever the current turn is about. The schema is deliberately narrow — five payload fields, one vector per point:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PayloadSchemaType,
)

client = QdrantClient(url="http://localhost:6333")

FACTS_COLLECTION = "session_facts"

client.create_collection(
    collection_name=FACTS_COLLECTION,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

for field, schema in [
    ("caller_id", PayloadSchemaType.KEYWORD),
    ("session_id", PayloadSchemaType.KEYWORD),
    ("fact_type", PayloadSchemaType.KEYWORD),
]:
    client.create_payload_index(
        collection_name=FACTS_COLLECTION,
        field_name=field,
        field_schema=schema,
    )
```

Each point's payload holds `caller_id` (the durable identity across sessions — an account ID or phone number, not the ephemeral Realtime session ID), `session_id` (the specific call this fact was extracted during, useful for debugging and for scoping facts to "this call" versus "this caller's history across calls"), `turn_index` (which turn in the session produced this fact, used to recency-weight retrieval later), `fact_text` (a short natural-language statement — "caller's account tier is Pro," not a raw transcript excerpt), and `fact_type` (a coarse category: `preference`, `identity`, `task_state`, `constraint`). The vector embeds `fact_text`. Indexing `caller_id`, `session_id`, and `fact_type` as payload indexes means retrieval queries can pre-filter on any of them without a full collection scan, which matters once a deployment accumulates facts across thousands of callers.

The point of keeping this schema narrow is that it never grows in the way a transcript does. A caller who's had six calls with the support line over a year might accumulate thirty or forty facts total, most of them stable (account tier, preferred contact method) and a handful session-specific (current open ticket, last stated intent) — a fixed, bounded footprint per caller, not a linearly growing one per turn.

### Deduplicating Facts Before They Accumulate

A naive extraction step run on every turn will happily re-extract "caller wants to cancel their subscription" five separate times across five turns where the caller keeps circling back to the same point — each extraction is individually correct, but storing all five bloats the fact store with redundant points that all retrieve for the same query, crowding out genuinely distinct facts from the top-`k` results at reseed time. The fix is to check a newly extracted fact against the caller's existing facts before upserting, and skip (or update in place) anything that's a near-duplicate of something already stored:

```python
DEDUP_THRESHOLD = 0.90


class SessionFactStore:
    # ... __init__ as before ...

    def _is_near_duplicate(self, caller_id: str, fact_text: str, vector: list[float]) -> bool:
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=Filter(
                must=[FieldCondition(key="caller_id", match=MatchValue(value=caller_id))]
            ),
            limit=1,
            with_payload=False,
        ).points
        return bool(hits) and hits[0].score >= DEDUP_THRESHOLD

    def extract_and_upsert(self, caller_id: str, session_id: str,
                            turn_index: int, turn_transcript: str) -> list[dict]:
        facts = self.extractor_fn(FACT_EXTRACTION_PROMPT.format(turn_transcript=turn_transcript))
        if not facts:
            return []

        points, kept = [], []
        for fact in facts:
            vector = self.embed_fn(fact["text"])
            if self._is_near_duplicate(caller_id, fact["text"], vector):
                continue  # already captured; don't inflate the store with a restatement
            points.append(PointStruct(
                id=str(uuid.uuid4()), vector=vector,
                payload={
                    "caller_id": caller_id, "session_id": session_id,
                    "turn_index": turn_index, "fact_text": fact["text"],
                    "fact_type": fact["type"], "recorded_at": time.time(),
                },
            ))
            kept.append(fact)

        if points:
            self.client.upsert(collection_name=self.collection, points=points)
        return kept
```

This adds one extra query per extracted fact, which is a small, fixed cost (single-digit milliseconds against a per-caller-filtered collection) relative to the extraction call itself, and it's what keeps the "thirty or forty facts per caller per year" estimate above realistic rather than optimistic — without dedup, a caller who restates their intent three times in one call would otherwise triple the count.

### Facts That Persist Across Calls, Not Just Within One

Because facts are keyed by `caller_id` rather than `session_id`, a retrieval query at the start of a brand-new call — not just a rotation mid-call — can pull in facts from the caller's prior calls, which is a genuine capability upgrade over transcript-based approaches (a fresh Realtime session has no transcript from a call that ended yesterday, but it does have the fact store). This needs a decay term, though, or a fact from eight months ago ("caller mentioned they're traveling next week") competes on equal footing with something said thirty seconds ago. A simple recency-weighted rescoring of the retrieved candidates, applied after the vector search rather than baked into the vector itself, keeps this tunable independently of the embedding model:

$$
\text{score}_{\text{final}} = \text{score}_{\text{cosine}} \cdot e^{-\lambda \cdot \Delta t}
$$

where $\Delta t$ is the age of the fact in days and $\lambda$ controls how fast old facts fade — a $\lambda$ around 0.01 halves a fact's effective weight roughly every 70 days, appropriate for `preference`/`identity` facts that should stay relevant for months, while `task_state` facts (which usually stop being relevant the moment a call ends) are better handled by scoping the retrieval filter to `session_id` rather than relying on decay to suppress them. In practice this means `fact_type` isn't just a label — it determines which retrieval strategy applies: `task_state` facts filtered to the current session only, `preference`/`identity` facts retrieved caller-wide with recency decay applied.

## Component Two: Incremental Fact Extraction

Facts are extracted turn-by-turn during the live session, not as a batch operation at the end of a call or at rotation time — extraction has to be cheap and asynchronous enough that it doesn't itself become a latency source on the critical path of a turn.

```python
import time
import uuid

from qdrant_client.models import PointStruct

FACT_EXTRACTION_PROMPT = """\
Given this single conversation turn, extract 0-3 short factual statements
worth remembering for the rest of this call or future calls with this caller.
Only extract facts that are stable or task-relevant — not filler, not
acknowledgments, not anything already obvious from context.

Return a JSON list of objects: {"text": str, "type": "identity"|"preference"|"task_state"|"constraint"}.
If nothing is worth extracting, return [].

Turn transcript:
{turn_transcript}
"""


class SessionFactStore:
    def __init__(self, client: QdrantClient, embed_fn, extractor_fn,
                 collection: str = FACTS_COLLECTION):
        self.client = client
        self.embed_fn = embed_fn
        self.extractor_fn = extractor_fn  # cheap/small model call, e.g. gpt-4o-mini
        self.collection = collection

    def extract_and_upsert(self, caller_id: str, session_id: str,
                            turn_index: int, turn_transcript: str) -> list[dict]:
        facts = self.extractor_fn(FACT_EXTRACTION_PROMPT.format(turn_transcript=turn_transcript))
        if not facts:
            return []

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=self.embed_fn(fact["text"]),
                payload={
                    "caller_id": caller_id,
                    "session_id": session_id,
                    "turn_index": turn_index,
                    "fact_text": fact["text"],
                    "fact_type": fact["type"],
                    "recorded_at": time.time(),
                },
            )
            for fact in facts
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return facts
```

Extraction runs off the critical audio path — fired asynchronously after each turn's transcript is available, using a small, fast model, since the extraction result only needs to be committed before the *next* rotation, not before the current turn's audio response goes out. This decoupling matters: if extraction were on the hot path of every turn, it would add exactly the kind of per-turn latency tax this whole design is trying to avoid introducing elsewhere.

## Component Three: Retrieval-Based Session Reseeding

At rotation time, instead of assembling a summary of the full prior transcript, the new session's context is seeded by querying the fact store for whatever is semantically closest to the content of the turn that triggered rotation:

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue


class SessionReseeder:
    def __init__(self, client: QdrantClient, embed_fn, collection: str = FACTS_COLLECTION):
        self.client = client
        self.embed_fn = embed_fn
        self.collection = collection

    def relevant_facts(self, caller_id: str, current_turn_text: str,
                        limit: int = 6) -> list[str]:
        hits = self.client.query_points(
            collection_name=self.collection,
            query=self.embed_fn(current_turn_text),
            query_filter=Filter(
                must=[FieldCondition(key="caller_id", match=MatchValue(value=caller_id))]
            ),
            limit=limit,
            with_payload=True,
        ).points
        # Most-relevant-first is already the query order; break ties toward
        # more recent facts within similar relevance bands.
        hits.sort(key=lambda h: (-h.score, -h.payload["turn_index"]))
        return [h.payload["fact_text"] for h in hits]

    def seed_instructions(self, base_instructions: str, caller_id: str,
                           current_turn_text: str) -> str:
        facts = self.relevant_facts(caller_id, current_turn_text)
        if not facts:
            return base_instructions
        fact_block = "\n".join(f"- {f}" for f in facts)
        return (
            f"{base_instructions}\n\n"
            f"Relevant context for this caller (retrieved, not a full transcript):\n{fact_block}"
        )
```

`limit=6` is a deliberate ceiling, not a tuned constant — the point of retrieval-based seeding is that the injected context stays small and bounded regardless of how long the prior session ran or how many facts have accumulated for this caller over their history. A caller with forty accumulated facts across a year of calls still only injects the six most relevant to *this* turn, not a growing subset of all of them. This is the structural difference from summarization: a summary's size tends to creep upward as more gets folded in over a session's lifetime (or requires progressively lossier compression to stay bounded); a top-`k` retrieval query's output size is fixed by construction, independent of how much has accumulated in the store.

### Why Facts Alone Aren't Quite Enough: A Hybrid Reseed

Pure fact-only reseeding has a real gap worth naming rather than glossing over: a list of six retrieved facts captures *what* is true about the caller and the task, but not the immediate conversational texture of the exchange that just happened — if the caller's last utterance was a half-finished sentence interrupted by a clarifying question, a fact list has nothing to say about that, and the new session may respond in a way that feels like it missed the last few seconds of the call, because in a meaningful sense it did. The practical fix is a hybrid seed rather than a purely fact-based one: keep the last one or two raw turns verbatim (as actual `conversation.item.create` events on the new session) for immediate continuity, and use retrieved facts only for everything older than that:

```python
async def seed_new_session(new_session, reseeder: SessionReseeder,
                            caller_id: str, base_instructions: str,
                            current_turn_text: str, raw_tail: list[dict]) -> None:
    """raw_tail: the last 1-2 turns as literal transcript entries, kept verbatim
    for conversational continuity; everything older is represented only via
    retrieved facts, never as raw or summarized text."""
    seeded_instructions = reseeder.seed_instructions(base_instructions, caller_id, current_turn_text)
    await new_session.update_instructions(seeded_instructions)
    for turn in raw_tail:
        await new_session.send_conversation_item(role=turn["role"], content=turn["content"])
```

This keeps the core structural claim intact — the *bulk* of prior session history is never carried forward as accumulating raw or compacted text, only the most recent one or two turns are, and that tail is fixed in size regardless of how long the session has run so far. It's a deliberate concession that "facts only, zero raw text" trades a small amount of conversational continuity for a larger reduction in what has to be reconstructed at rotation time, and the size of `raw_tail` (I use 1-2 turns) is the knob that trades one against the other.

## Component Four: A Per-Turn Latency Drift Monitor

Rotating on a fixed cadence (every 8-12 turns, say) rotates whether or not drift has actually started — which means short calls that never drift pay the rotation cost for nothing, while calls that drift faster than the fixed cadence assumes still get caught late. A better trigger compares each session's own recent latency against its own early-session baseline, since acceptable absolute latency varies by network conditions and region and a fixed millisecond threshold would misfire across deployments.

Define the trailing-window mean latency at turn $n$ over a window of size $w$:

$$
\bar{L}_{n}^{(w)} = \frac{1}{w}\sum_{i=n-w+1}^{n} L_i
$$

and the drift ratio relative to the session's own first-$b$-turn baseline:

$$
\rho_n = \frac{\bar{L}_{n}^{(w)}}{\bar{L}_{b}^{(b)}}
$$

Rotation triggers when $\rho_n$ crosses a threshold (I use 1.5, i.e. trailing latency running 50% above the session's own opening baseline) for a required number of consecutive windows, to avoid triggering on a single noisy turn:

```python
from collections import deque


class LatencyDriftMonitor:
    def __init__(self, baseline_window: int = 5, trailing_window: int = 5,
                 drift_ratio: float = 1.5, confirm_windows: int = 2):
        self.baseline: list[float] = []
        self.trailing: deque[float] = deque(maxlen=trailing_window)
        self.baseline_window = baseline_window
        self.drift_ratio = drift_ratio
        self.confirm_windows = confirm_windows
        self._consecutive_breaches = 0

    def record_turn(self, latency_ms: float) -> None:
        if len(self.baseline) < self.baseline_window:
            self.baseline.append(latency_ms)
        self.trailing.append(latency_ms)

    def should_rotate(self) -> bool:
        if (len(self.baseline) < self.baseline_window
                or len(self.trailing) < self.trailing.maxlen):
            return False  # not enough data yet to trust a ratio

        baseline_mean = sum(self.baseline) / len(self.baseline)
        trailing_mean = sum(self.trailing) / len(self.trailing)
        ratio = trailing_mean / baseline_mean

        if ratio >= self.drift_ratio:
            self._consecutive_breaches += 1
        else:
            self._consecutive_breaches = 0
        return self._consecutive_breaches >= self.confirm_windows

    def reset_baseline(self) -> None:
        # Called immediately after a rotation completes, so the new
        # session establishes its own fresh baseline rather than
        # inheriting the old session's (now-irrelevant) numbers.
        self.baseline = []
        self.trailing.clear()
        self._consecutive_breaches = 0
```

Requiring `confirm_windows >= 2` consecutive breaches before triggering rotation is the guard against a single slow turn (a transient network hiccup, a longer-than-usual tool call) causing an unnecessary rotation — the monitor only acts once the drift shows up consistently across two overlapping trailing windows, which is a much stronger signal than one bad data point.

### The Window-Size Tradeoff

`trailing_window` and `baseline_window` trade detection speed against noise sensitivity in the usual way rolling statistics do. A short trailing window (three turns) reacts to genuine drift faster — fewer turns spent in a degraded state before rotation kicks in — but its mean is more sensitive to any single outlier turn, which is exactly what `confirm_windows` is compensating for. A longer trailing window (eight to ten turns) produces a smoother, more trustworthy signal but necessarily lags real drift by however many turns it takes to fill the window with post-drift data; if turn latency genuinely triples starting at turn 12, a ten-turn trailing window won't fully reflect that until turn 21 or so, by which point the caller has already sat through nine degraded turns the monitor hasn't yet flagged. For voice specifically — where every additional degraded turn is directly felt by a human on the call in a way that, say, a degraded background batch job is not — I lean toward the shorter end (four to six turns) paired with `confirm_windows=2`, accepting slightly more sensitivity to noise in exchange for catching real drift within a handful of turns rather than a dozen.

The standard error of a window mean scales as $\sigma / \sqrt{w}$ for window size $w$, which is the formal version of the same tradeoff: halving the window doubles the estimate's variance-driven noise (√2× larger standard error), and that's the price paid for roughly halving detection lag.

### Measuring Turn Latency Precisely

`turn_latency_ms` needs a precise, consistent definition, or the drift ratio computed from it is measuring noise as much as signal. The cleanest anchor points are both already present in the Realtime API's own event stream: the timestamp of the `input_audio_buffer.speech_stopped` event (the moment the API's own VAD decided the caller finished speaking) and the timestamp of the first `response.audio.delta` event for the resulting response (the first byte of audio the model produced in reply). The difference between those two is the model's actual "thinking" latency, cleanly separated from client-side capture and render time on either end, which is the number that should feed the drift monitor — mixing in client-side network jitter would make the ratio noisier without making it more informative about model-side drift specifically.

```python
class TurnLatencyTracker:
    def __init__(self):
        self._speech_stopped_at: float | None = None

    def on_event(self, event: dict) -> float | None:
        """Feed every Realtime server event through this; returns a turn
        latency in ms exactly once per turn, when the first audio delta
        for the matching response arrives."""
        if event["type"] == "input_audio_buffer.speech_stopped":
            self._speech_stopped_at = event["_received_at"]  # client-stamped receive time
            return None
        if event["type"] == "response.audio.delta" and self._speech_stopped_at is not None:
            latency_ms = (event["_received_at"] - self._speech_stopped_at) * 1000
            self._speech_stopped_at = None  # consumed; next turn re-arms on the next speech_stopped
            return latency_ms
        return None
```

This tracker is what feeds `monitor.record_turn(latency_ms)` in the orchestration loop below — keeping the measurement isolated in its own small class rather than inlined into the rotation logic makes it straightforward to unit test against a fixed sequence of recorded events, independent of anything related to fact extraction or rotation.

## Putting It Together: The Rotation Flow

The three components compose into a single control loop that runs alongside the live session: latency is recorded every turn, drift is checked every turn, and rotation — when triggered — reseeds from retrieval rather than from a transcript summary.

```python
async def maybe_rotate_session(session, monitor: LatencyDriftMonitor,
                                fact_store: SessionFactStore,
                                reseeder: SessionReseeder,
                                caller_id: str, session_id: str,
                                turn_index: int, turn_latency_ms: float,
                                current_turn_text: str, base_instructions: str):
    monitor.record_turn(turn_latency_ms)
    fact_store.extract_and_upsert(caller_id, session_id, turn_index, current_turn_text)

    if not monitor.should_rotate():
        return session

    new_session_id = str(uuid.uuid4())
    seeded_instructions = reseeder.seed_instructions(
        base_instructions, caller_id, current_turn_text
    )
    new_session = await open_realtime_session(instructions=seeded_instructions)
    await session.close()
    monitor.reset_baseline()

    return new_session
```

Two design choices here are worth calling out explicitly. First, rotation happens *between* turns, at a natural conversational boundary, not mid-response — the caller experiences it, at worst, as a very brief pause rather than an audible interruption of the agent's own speech. Second, the fact extraction call for the *current* turn happens before the rotation check, so that if this turn's content is itself relevant to what comes next, it's already in the store and eligible for retrieval by the new session immediately, rather than lagging one turn behind.

A subtlety worth handling explicitly rather than discovering in production: rotation should never be triggered while a response is still streaming audio to the caller, even if the drift monitor's `should_rotate()` happened to flip to `True` mid-response due to timing. The orchestration wrapper checks the session's response state and, if a response is in flight, defers the rotation check to the next turn boundary rather than issuing `response.cancel` to force an early cutover — cutting off the agent mid-sentence to save on drift is a strictly worse experience than tolerating one more turn at degraded latency.

### Where This Logic Has to Live: Server-Mediated Sessions

This orchestration only works if something sits in the connection path with the authority to intercept turn boundaries, call out to a fact-extraction model, query Qdrant, and swap the underlying Realtime connection out from under the caller without the caller noticing. That's a direct implication for the transport choice discussed earlier: a browser client connected via native WebRTC straight to OpenAI's endpoint has no natural place to run this logic — the whole point of that transport choice is to minimize what sits between the client and the model. Retrieval-seeded rotation, by contrast, requires exactly the kind of server-mediated architecture that the transport section flagged as the case for choosing WebSocket over WebRTC: a server-side process holding the live session, watching turn latencies, and able to open a replacement session and redirect audio to it transparently. For a phone-based deployment (Realtime API over SIP or bridged through Twilio, as in the forum reports cited above) this is already the natural architecture, since a telephony bridge is inherently server-mediated. For a browser-first product built around direct WebRTC for its latency advantage, adopting this design means accepting the server-mediation overhead specifically to gain the rotation capability — a real trade against the leaner transport, and one only worth making for deployments where call length routinely pushes into the range where drift becomes noticeable.

## Validating the Rotation Trigger Offline

Because there's no production baseline yet for how well retrieval-seeded rotation actually reduces model-side latency (the honest caveat repeated throughout this post), the one piece that *can* be validated before deployment is the drift monitor's triggering behavior itself — whether it fires at reasonable points, with reasonable frequency, against real turn-latency traces, independent of whether rotation subsequently helps. A simple offline replay harness against logged latency data from existing (non-rotating) sessions answers "how often would this fire, and how early relative to when a human would notice" without requiring any change to production traffic:

```python
def replay_drift_trigger(latency_trace: list[float], **monitor_kwargs) -> list[int]:
    """Returns the turn indices at which should_rotate() would have fired,
    given a recorded latency trace from a real (non-rotating) session."""
    monitor = LatencyDriftMonitor(**monitor_kwargs)
    trigger_turns = []
    for turn_index, latency_ms in enumerate(latency_trace):
        monitor.record_turn(latency_ms)
        if monitor.should_rotate():
            trigger_turns.append(turn_index)
            monitor.reset_baseline()  # simulate what a real rotation would do
    return trigger_turns
```

Running this against a corpus of logged sessions before shipping the live rotation logic answers several calibration questions cheaply: does `drift_ratio=1.5` fire too eagerly on sessions that never actually degrade badly (a false-positive-rate check), does it fire late enough on sessions the forum reports describe (a sanity check against the "3x by the later part of the call" pattern), and how does `confirm_windows` trade off against detection lag on real, noisy latency traces rather than the idealized ones used to reason about the math above. This doesn't validate the *fact-store reseeding* half of the design — that genuinely can't be validated without live sessions to rotate — but it de-risks the *triggering* half considerably before any of it touches a real call.

## Worked Example

Consider a customer-support voice agent handling a 25-turn call: a caller wants to downgrade a subscription, has questions about a refund, and needs to confirm a mailing address change. I'm presenting the numbers below as illustrative, not measured — this is a proposed mitigation for a problem OpenAI itself hasn't publicly root-caused, so there is no production dataset yet that isolates the effect of retrieval-seeded rotation specifically. The *shape* of the naive-approach numbers is drawn directly from the forum report's own description (roughly 3x latency by the later part of a long call); the proposed-approach numbers are a hypothesis about what bounding the reseed payload should achieve, not a claim about what it definitely does achieve.

| Turn | Naive (prune-only, no rotation) | Proposed (retrieval-seeded rotation) | Notes |
|---|---|---|---|
| 1-5 | ~800ms | ~800ms | Both approaches start identical — no drift yet, no rotation has happened |
| 6-10 | ~950-1100ms | ~800-850ms | Drift monitor's baseline ratio crosses 1.5x around turn 9 in the naive trace; proposed approach triggers its first rotation here |
| 11-15 | ~1300-1500ms | ~800-900ms | Naive approach keeps accumulating transcript/summary state; proposed approach has rotated once, reset to a fresh baseline |
| 16-20 | ~1700-1900ms | ~850-950ms | Second rotation triggers in the proposed approach around turn 18 as drift begins again post-first-rotation |
| 21-25 | ~2100-2400ms (matches the "3x or more" forum report) | ~800-900ms | Naive approach is now well past the 1.2s talk-over threshold for most of the back half of the call |

The mechanism behind the proposed column staying flat isn't that rotation makes each individual session faster in some absolute sense — it's that rotation, combined with retrieval-based reseeding instead of summary reseeding, keeps resetting the session back toward its own early-session baseline before drift compounds, and does so without asking the new session to process a growing block of prior-turn text on every reset. Two rotations across 25 turns is also a modest overhead to budget for: each rotation costs a fact-retrieval query (single-digit milliseconds against a modestly sized per-caller collection) plus the cost of establishing a new Realtime connection, both of which are small and fixed compared to the multi-second drift they're avoiding.

It's worth putting an explicit cost accounting next to the latency numbers, since the mitigation isn't free even where it works — again, illustrative, not measured:

| Cost item | Naive prune-only | Proposed retrieval-seeded rotation |
|---|---|---|
| Extraction calls (small model, per turn) | 0 (no extraction step) | 25 calls × ~\$0.0001 ≈ \$0.0025/call |
| Embedding calls (per extracted/queried fact) | 0 | ~40-50 calls × ~\$0.00002 ≈ \$0.001/call |
| Extra Realtime session establishments | 0 (single session) | 2 (one per rotation) × negligible connection overhead |
| Degraded-call abandonment risk (illustrative, tied to talk-over threshold) | Elevated in the back half of the call, per the forum-reported 3x drift | Low throughout, per the hypothesis this design tests |

The direct infrastructure cost of the proposed approach (roughly half a cent per call in this illustration) is trivial next to the cost of a caller abandoning a support interaction because the agent felt broken by minute six — the same asymmetry that shows up in most latency-engineering cost arguments: the mitigation's direct cost is easy to price, the thing it's protecting against is not, and the two aren't remotely comparable in magnitude when the protection actually works.

## Does This Generalize Beyond OpenAI's Realtime API?

The specific forum evidence cited throughout this post is about OpenAI's Realtime API and the `gpt-realtime`/`gpt-realtime-1.5` models specifically, and I want to be careful not to overclaim generalization. That said, the *structural* pattern this post is responding to — a stateful, long-running conversational session where cost appears to correlate with session duration or turn count in a way that isn't fully explained by the size of enumerable context — is not obviously specific to one vendor's implementation. Any realtime voice API built on a similarly stateful session model (as opposed to a fully stateless request-response pattern re-sent in full on every turn) has at least the structural precondition for the same class of drift to appear, whether or not it's been publicly reported yet for that provider. The fact-store-and-retrieval architecture proposed here doesn't depend on anything OpenAI-specific beyond the event names used to open a session and inject seeded instructions — the same pattern (bounded, retrieval-seeded context instead of an accumulating or periodically-summarized transcript) applies unchanged to any comparable stateful voice session API, provided it exposes some mechanism for rotating to a fresh session and injecting initial context into it.

## Challenges and Open Problems

**The root cause is not publicly confirmed, and this approach targets a hypothesis, not a diagnosis.** The forum thread's own authors asked OpenAI directly whether progressive latency increase is expected behavior and whether `conversation.item.delete` fully reduces the model's effective context cost — as of this writing, neither question has a public, confirmed answer. This design removes one plausible contributing factor (the requirement to re-derive meaning from an accumulated or compacted transcript at rotation time) from the picture. If the actual cause is something else entirely — provider-side infrastructure behavior, caching effects tied to session duration rather than content volume, or something in how the model's own internal state representation scales with wall-clock session length rather than token count — this approach may reduce drift without eliminating it, or in the worst case may not meaningfully help at all. I'd treat it as worth deploying because its cost is bounded and its downside is limited, not because it's guaranteed to solve the underlying problem.

**Fact-extraction quality is a new failure surface.** An extraction step that misses an important fact, or mis-labels a `constraint` as a `preference`, means the reseeded session is missing context the caller reasonably assumes persists — "I already told you I want a refund, not a replacement" is a much worse experience than a few hundred milliseconds of extra latency. This trades one failure mode (drift) for a different one (incomplete memory), and the second is arguably more visible to the caller when it goes wrong, since it shows up as the agent asking a question it should already know the answer to.

**Session rotation has its own small latency cost and a small risk of an audible seam.** Establishing a new Realtime connection isn't free, and even with rotation timed to fall between turns rather than mid-response, a caller paying close attention may notice a very brief pause or a subtle shift in the agent's phrasing style as it picks up with retrieved facts instead of full prior context. For most calls this is a good trade against multi-second drift, but it is a real, if small, cost — not a free lunch.

**This adds infrastructure to a problem that shorter calls might not have.** A Qdrant-backed fact store, an extraction step running on every turn, and a drift monitor are all additional moving parts. For a voice agent whose calls reliably wrap up in under ten turns — well within the range where the forum reports suggest drift hasn't yet become severe — the simplest available options (do nothing, or prune conversation items on a fixed schedule) may be entirely sufficient, and this architecture is solving a problem those deployments don't actually have. The complexity here is justified specifically by session length and by the documented fact that pruning alone doesn't reliably bound the drift once sessions run long — it is not a default worth reaching for on every voice agent regardless of typical call length.

**Cross-call fact persistence raises data-retention questions this post doesn't resolve.** Storing caller facts beyond the lifetime of a single call — precisely the capability that makes cross-session reseeding useful — means the fact store becomes a durable record of what callers have said, subject to whatever data-retention, deletion-request, and PII-handling obligations already apply to the rest of a support system's call records. A `fact_type` of `identity` in particular deserves the same handling as any other stored PII: encryption at rest, a defined retention window, and a deletion path that actually removes the caller's points from Qdrant (not just marks them inactive) when a deletion request comes in. This is solvable with standard data-governance practice, but it is a real obligation the naive prune-only approach doesn't create, since that approach never persists anything past the life of the conversation items it eventually deletes.

**There's a cold-start gap for first-time callers.** The entire retrieval-seeding mechanism has nothing to retrieve for a caller with no prior facts, which is precisely the case where session rotation matters least (a first-time caller's session hasn't had time to drift much either, if the call is short) but is worth naming as a boundary condition: the hybrid reseed's raw-tail component still works for a first-time caller (there's always a recent turn or two to carry forward verbatim), but the fact-retrieval component contributes nothing on a caller's very first rotation within their very first call, since there's no history yet to have extracted facts from beyond what's already happened in that same session.

## References

- OpenAI Developer Community. *Progressive latency increase in long gpt-realtime-1.5 (+gpt-realtime) voice sessions, including non-tool turns, not fully reset by conversation item deletion.* [community.openai.com/t/progressive-latency-increase-in-long-gpt-realtime-1-5-gpt-realtime-voice-sessions-including-non-tool-turns-not-fully-reset-by-conversation-item-deletion/1376004](https://community.openai.com/t/progressive-latency-increase-in-long-gpt-realtime-1-5-gpt-realtime-voice-sessions-including-non-tool-turns-not-fully-reset-by-conversation-item-deletion/1376004)
- OpenAI Developer Community. *Big delays receiving output_audio_buffer.stopped event since Apr 18th.* [community.openai.com/t/big-delays-receiving-output-audio-buffer-stopped-event-since-apr-18th/1379823](https://community.openai.com/t/big-delays-receiving-output-audio-buffer-stopped-event-since-apr-18th/1379823)
- Forasoft. *OpenAI Realtime API: Production Voice Agents (2026).* [forasoft.com/blog/article/openai-realtime-api-voice-agent-production-guide-2026](https://www.forasoft.com/blog/article/openai-realtime-api-voice-agent-production-guide-2026)
- OpenAI Developer Docs. *Realtime and audio.* [developers.openai.com/api/docs/guides/realtime](https://developers.openai.com/api/docs/guides/realtime)
- OpenAI Developer Docs. *Realtime API with WebRTC.* [developers.openai.com/api/docs/guides/realtime-webrtc](https://developers.openai.com/api/docs/guides/realtime-webrtc)
- OpenAI Agents SDK Docs. *Realtime Transport Layer.* [openai.github.io/openai-agents-js/guides/voice-agents/transport](https://openai.github.io/openai-agents-js/guides/voice-agents/transport/)
- Qdrant. *Qdrant Vector Database Documentation.* [qdrant.tech/documentation](https://qdrant.tech/documentation)
