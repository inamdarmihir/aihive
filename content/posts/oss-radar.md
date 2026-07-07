---
title: "OSS Radar: A Multi-Agent System for Continuous Open-Source Intelligence"
date: 2026-06-29
description: "A deep dive into OSS Radar — a production multi-agent system that runs nightly GitHub scans, uses Qdrant vector search for semantic relevance gating and deduplication, and surfaces findings through a Next.js digest dashboard. Covers the orchestration architecture, the two-collection knowledge layer, subagent design, the human-in-the-loop security gate, and the schedule-to-dashboard notification chain."
tags: ["agents", "qdrant", "vector-search", "multi-agent", "eve", "mcp", "github", "semantic-search", "human-in-the-loop"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Keeping track of a fast-moving OSS ecosystem is expensive. If you care about AI agent frameworks, TypeScript runtimes, and vector databases, you are potentially watching hundreds of repositories. Doing this manually means opening GitHub trending every morning, reading changelogs when you remember, and finding out about critical CVEs days after they were published. The signal is real but scattered; aggregating it by hand does not scale.

The alternative is an autonomous system that does the watching for you: discovers new repositories matching your interests, synthesizes emerging issue patterns, catches new releases, and surfaces security advisories, all filtered against a set of topics you define, deduplicated across runs, and delivered as a daily digest. That is what **OSS Radar** is.

This post covers the design in full: the two-collection Qdrant knowledge layer that handles semantic relevance and deduplication, the multi-agent orchestration pattern using the eve framework, how each of the four specialist subagents works, the human-in-the-loop gate for security findings, and the schedule-to-dashboard notification chain. I will focus on the design decisions and their tradeoffs. Familiarity with vector databases, Qdrant, and basic agent frameworks is assumed.

---

## Table of Contents

1. [The Problem: OSS Intelligence at Scale](#the-problem)
2. [System Overview](#system-overview)
3. [Why Qdrant: Hybrid Search and Payload Filtering in One Engine](#why-qdrant)
4. [The Two-Collection Qdrant Knowledge Layer](#qdrant-knowledge-layer)
   - [Collection Schema and Initialization](#collection-schema)
   - [ID Keying Strategy](#id-keying)
   - [watchlist: Semantic Topic Registry](#watchlist-collection)
   - [findings: The Deduplication Target](#findings-collection)
   - [Embedding and Sparse Tokenization Strategy](#embedding-strategy)
5. [Relevance Gating with qdrant_watchlist_match](#relevance-gating)
   - [BM42 Sparse Vectors: Why Hybrid Search](#bm42-hybrid)
   - [The Hybrid Query and RRF Fusion](#hybrid-query)
   - [Threshold Design](#threshold-design)
6. [Deduplication with qdrant_dedup_check](#deduplication)
7. [The Orchestrator and Dispatch Loop](#orchestrator)
8. [The Four Specialist Subagents](#subagents)
   - [trending_scout: Repository Discovery](#trending-scout)
   - [issue_analyst: Issue Pattern Synthesis](#issue-analyst)
   - [release_watcher: Release Monitoring](#release-watcher)
   - [security_watcher: Advisory Scanning](#security-watcher)
9. [Human-in-the-Loop for Security Findings](#hitl)
10. [GitHub via MCP: Two Scoped Connections](#github-mcp)
11. [The eve Framework: Schedules, Hooks, and the Build Contract](#eve-framework)
12. [Dashboard and ISR Revalidation](#dashboard)
13. [Challenges and Open Problems](#challenges)

---

## The Problem: OSS Intelligence at Scale {#the-problem}

The information asymmetry in OSS tracking is real. A new agent framework can go from 0 to 10k stars in a week; a critical CVE in a widely-used package might be published on a Tuesday afternoon; a library you depend on might drop a major version with breaking changes the same day you have three other things to review. Individual tracking tools exist for pieces of this: GitHub notifications for repos you already follow, dependabot for CVEs in your own dependencies. None of them surface *new* things you have not yet decided to follow, or synthesize patterns across a portfolio of tracked projects.

OSS Radar is designed for the specific problem of *topic-scoped, continuous intelligence*. You define topics ("AI agent frameworks," "TypeScript runtimes and bundlers," "vector databases") and the system finds what is worth knowing every day, across four dimensions: new repos, issue patterns, releases, and security advisories. Findings are deduplicated across runs so you do not see the same repo three days in a row, and relevance is gated semantically so keyword overlap alone is not sufficient to generate a finding.

I want to be precise about what this post does *not* cover: it does not cover the frontend UX in depth, it does not discuss the GitHub MCP server internals, and it does not benchmark the relevance scoring against alternative approaches. Those are real topics but they would double the length without proportionate value here.

---

## System Overview {#system-overview}

```
                         ┌──────────────────────────────────────────┐
                         │            OSS Radar (eve app)           │
                         │                                          │
   daily cron            │  ┌──────────────────────────────────┐   │
  (13:00 UTC)  ──────────┼─►│        Orchestrator Agent         │   │
                         │  │    reads watchlist → dispatches   │   │
                         │  └──┬─────────┬──────────┬───────┬──┘   │
                         │     │         │          │       │       │
                         │     ▼         ▼          ▼       ▼       │
                         │  trending  issue_    release  security   │
                         │  _scout    analyst   _watcher  _watcher  │
                         │  (daily)   (daily)   (daily)  (Sundays)  │
                         │     │         │          │       │       │
                         │     └────┬────┘          └───┬───┘       │
                         │          ▼                   ▼           │
                         │   qdrant_dedup_check   qdrant_write_     │
                         │   qdrant_watchlist_    advisory (gated)  │
                         │   match                                  │
                         │   qdrant_write_finding                   │
                         │                                          │
                         └──────────────┬───────────────────────────┘
                                        │ POST /api/revalidate
                                        ▼
                         ┌──────────────────────────────────────────┐
                         │         Qdrant (two collections)         │
                         │   watchlist            findings          │
                         └──────────────────────────────────────────┘
                                        │
                                        │ scroll (status: "new")
                                        ▼
                         ┌──────────────────────────────────────────┐
                         │      Next.js Dashboard (ISR 24h)         │
                         │   /api/digest → grouped by topic         │
                         └──────────────────────────────────────────┘
```

The orchestrator reads the `watchlist` Qdrant collection for active topics, dispatches one to four subagents per topic depending on the day of the week, and writes findings back to the `findings` collection through typed tools that enforce semantic relevance and deduplication before every write. After all subagents complete, a session hook POSTs to the dashboard's revalidate endpoint to bust its 24-hour ISR cache.

---

## Why Qdrant: Hybrid Search and Payload Filtering in One Engine {#why-qdrant}

The persistence layer is entirely Qdrant. This deserves an explanation, because the workload here has specific characteristics that make the choice non-obvious.

OSS Radar needs to answer two classes of queries on every agent tick:

1. **Relevance check**: "Does this candidate repository conceptually match any topic I am tracking?" This requires semantic similarity (a repository named `mini-agent-framework` should match the topic "AI agent frameworks" even without the exact phrase).

2. **Deduplication check**: "Have I already reported on this repository recently?" This requires exact match on `repo_full_name` combined with a time-range filter on `first_seen_at`, scoped to a semantic similarity threshold on the finding text.

Both queries need *filtered* similarity search: not just "find the nearest vector" but "find the nearest vector among points that satisfy a boolean predicate." Qdrant evaluates filter conditions during HNSW graph traversal rather than as a post-filter. When you pass a `filter` to a `search` or `query` call, Qdrant's HNSW implementation prunes ineligible nodes before scoring them. The consequence is that a filter on `active: true` (which passes roughly half of watchlist points on average) does not double the number of distance computations; it halves them. The same applies to the dedup filter on `repo_full_name`, which prunes to a handful of points before any cosine scoring happens.

The second reason is **hybrid search**: the system needs both semantic similarity (dense vectors) and exact-term matching (sparse vectors) in a single query. GitHub repository names, language names, and framework identifiers are exact terms that dense embeddings handle inconsistently. "langchain", "llamaindex", "qdrant": embeddings may place these correctly, but they may also conflate adjacent terms at moderate similarity scores. A sparse BM42 vector over the candidate text catches exact token matches with high precision; the dense vector catches conceptual adjacency. Together, via Reciprocal Rank Fusion, they produce a better relevance signal than either alone.

Alternatives I considered:

- **Pinecone**: native dense search with metadata filters. Does not support sparse vectors as first-class citizens in the same collection (Pinecone Sparse-Dense is a separate architecture). Managed service only.
- **pgvector**: dense vectors in PostgreSQL. No native sparse vector support; payload filtering is SQL `WHERE`, which post-filters unless you add a GIN index. A good fit for systems already on Postgres, but not the right choice when hybrid search is a core requirement.
- **Weaviate**: supports sparse and dense hybrid search with BM25. Schema-heavy configuration for what is a simple two-collection setup; more operational overhead.
- **ChromaDB**: no production-grade payload index semantics; designed for experimentation, not for a system running daily with sub-second filter queries.

Qdrant's combination of filter-during-traversal, named vectors (multiple vector representations per point), sparse vectors with IDF weighting via BM42, and the Query API's `prefetch` plus `fusion: "rrf"` makes it the right choice for this workload.

---

## The Two-Collection Qdrant Knowledge Layer {#qdrant-knowledge-layer}

The entire persistence layer is two Qdrant collections. One more and you need joins; one fewer and you conflate intent (what to watch) with output (what was found).

### Collection Schema and Initialization {#collection-schema}

Both collections are initialized in `scripts/setup_collections.ts`, run once on first deploy. Each collection is created with two named vector configurations: `dense` for 1536-dimensional cosine similarity, and `bm42` for sparse IDF-weighted token vectors. Payload indexes are created immediately after collection creation; they are required for filter-during-traversal to work correctly.

```typescript
// watchlist collection: small, stable, keep in memory
await qdrant.createCollection("watchlist", {
  vectors: {
    dense: {
      size: 1536,       // text-embedding-3-small output dimensionality
      distance: "Cosine",
      on_disk: false,   // small collection; RAM is fine
    },
  },
  sparse_vectors: {
    bm42: {
      modifier: "idf",  // Qdrant BM42: BERT attention weights + IDF scaling
    },
  },
});

// Payload indexes for filter-during-HNSW-traversal
await qdrant.createPayloadIndex("watchlist", {
  field_name: "active",
  field_schema: "bool",
});
await qdrant.createPayloadIndex("watchlist", {
  field_name: "topic_name",
  field_schema: "keyword",
});
```

```typescript
// findings collection: unbounded growth, offload dense index to disk
await qdrant.createCollection("findings", {
  vectors: {
    dense: {
      size: 1536,
      distance: "Cosine",
      on_disk: true,    // findings accumulate daily; avoid RAM pressure
    },
  },
  sparse_vectors: {
    bm42: {
      modifier: "idf",
    },
  },
  optimizers_config: {
    indexing_threshold: 20_000,  // build HNSW after 20k vectors
    memmap_threshold: 50_000,    // mmap above 50k for reduced footprint
  },
});

await qdrant.createPayloadIndex("findings", {
  field_name: "repo_full_name",
  field_schema: "keyword",   // used in every dedup filter
});
await qdrant.createPayloadIndex("findings", {
  field_name: "first_seen_at",
  field_schema: "integer",   // range filter for lookback window
});
await qdrant.createPayloadIndex("findings", {
  field_name: "status",
  field_schema: "keyword",   // dashboard scroll: status = "new"
});
await qdrant.createPayloadIndex("findings", {
  field_name: "entity_type",
  field_schema: "keyword",   // orchestrator queries by entity_type
});
await qdrant.createPayloadIndex("findings", {
  field_name: "matched_topic",
  field_schema: "keyword",   // grouping key for digest display
});
```

Two decisions are worth calling out. First, `on_disk: true` for the `dense` vector in `findings`. The watchlist is small and stable (tens of topics at most); keeping it in memory is fine. The findings collection grows with every daily scan and is unbounded; using on-disk storage for the dense index trades some query latency for much lower memory pressure in steady-state operation. For a batch workload running at 13:00 UTC (not a latency-sensitive real-time API), this is the right tradeoff.

Second, **payload indexes are mandatory**, not optional, for filter-during-traversal to engage. Without them, Qdrant falls back to scoring all HNSW candidates and then discarding those that fail the predicate. Creating an index on `repo_full_name`, `first_seen_at`, `status`, `entity_type`, and `matched_topic` ensures that every relevance and deduplication query prunes the candidate set before vector scoring begins.

### ID Keying Strategy {#id-keying}

Qdrant point IDs must be either unsigned 64-bit integers or UUID strings. The choice of ID scheme has consequences for idempotency and update semantics.

For `watchlist`: points use sequential integer IDs assigned at seed time. The watchlist is small and static; seed order is stable; integer IDs are simple.

For `findings`: points use **deterministic UUID v5** derived from the finding's natural key. The natural key for a repo finding is `{repo_full_name}:{entity_type}`. For issue and release findings, a short qualifier is appended: `{repo_full_name}:{entity_type}:{content_hash}`, where `content_hash` is a CRC32 of the synthesized headline. This ensures that two different issue patterns for the same repo produce distinct point IDs while the same pattern encountered on a subsequent scan resolves to the same ID.

```typescript
import { v5 as uuidv5 } from 'uuid';

// Stable namespace UUID for OSS Radar
const RADAR_NAMESPACE = '9b2a4c5e-1f8a-4c3d-8e7b-a12f34567890';

export function findingId(
  repoFullName: string,
  entityType: string,
  qualifier?: string
): string {
  const key = qualifier
    ? `${repoFullName}:${entityType}:${qualifier}`
    : `${repoFullName}:${entityType}`;
  return uuidv5(key, RADAR_NAMESPACE);
}
```

Deterministic IDs enable idempotent upserts. When `qdrant_write_finding` is called for a repo that already has a finding, the derived UUID resolves to the same point. The tool calls `qdrant.setPayload()` on the existing point, updating only `last_seen_at`, `trend_signal`, and `provenance`. No separate lookup is needed to discover whether the point exists; the ID encodes the existence check implicitly.

The dedup check still runs before the write, because its role is different: it determines whether the finding is *new enough* to surface in the digest (status `"new"`) or already tracked (status `"tracked"`). A deterministic ID alone cannot answer that question.

### watchlist: Semantic Topic Registry {#watchlist-collection}

The `watchlist` collection stores the topics the system monitors. Each point represents one topic:

| Payload field | Type | Purpose |
|---|---|---|
| `topic_name` | string | Human-readable label |
| `description` | string | Full natural-language description of what the topic covers |
| `active` | boolean | If false, this topic is skipped entirely |
| `created_at` | number | Unix ms timestamp |
| `last_scanned_at` | number or null | Updated after every successful scan; used as the cutoff for the next scan |

Each point carries two named vector representations: `dense` (the embedding of `description`) and `bm42` (the sparse tokenization of the same text). At query time, both are combined via the hybrid Query API. The choice to embed and tokenize `description` rather than `topic_name` is the key design decision: the description is what gets semantically matched against candidate findings, so it must be in the same representational space as the candidates.

Topics are seeded manually via `scripts/seed_watchlist.ts` and do not change at runtime. The starter set covers three domains: AI agent frameworks, TypeScript runtimes and bundlers, and vector databases, each defined with enough breadth to cover adjacent terminology. The AI agent frameworks topic explicitly lists "LLM orchestration," "multi-agent coordination," and "tool-calling" in its description, so a repository named `mini-react-agent` scores well on both the dense semantic dimension and the BM42 sparse dimension even without the exact phrase "framework."

### findings: The Deduplication Target {#findings-collection}

The `findings` collection stores everything the subagents surface. Every point is one finding:

| Payload field | Type | Purpose |
|---|---|---|
| `entity_type` | enum | `repo`, `issue`, `release`, `advisory` |
| `repo_full_name` | string | GitHub `owner/repo` |
| `title` | string | Finding headline |
| `summary` | string | 1-3 sentence description |
| `url` | string | Link to the source |
| `source_subagent` | enum | Which subagent produced this |
| `matched_topic` | string | Which watchlist topic this finding matched |
| `first_seen_at` | number | Unix ms of first insertion |
| `last_seen_at` | number | Updated on duplicate detection |
| `status` | enum | `new`, `tracked`, `superseded` |
| `trend_signal` | object | `stars`, `star_delta`, `comment_count` (all nullable) |
| `provenance` | array | Append-only log of `{subagent, note, at}` entries |
| `severity` | enum or null | Advisory-only: `critical`, `high`, `medium`, `low`, `unknown` |
| `cve_ids` | string[] or null | Advisory-only: CVE identifiers |

Each point carries named vectors: `dense` (the embedding of `title + summary`) and `bm42` (the sparse tokenization of the same text). The dual-use is intentional: the same vector pair that enables semantic deduplication also enables hybrid search over the findings corpus. The dashboard does not exploit this yet, but the capability is present.

Provenance deserves specific attention. It is an append-only array, never overwritten. When an existing finding is updated (because a subagent encountered the same repo again with a significantly higher star count), the new subagent call, timestamp, and note are *appended* to `provenance`. This creates a lightweight audit trail without any separate history collection.

### Embedding and Sparse Tokenization Strategy {#embedding-strategy}

Both collections use two vector representations per point. The dense embedding uses OpenAI's `text-embedding-3-small` (1536 dimensions) via the AI SDK's `embed()` call. The sparse BM42 vector is computed via Qdrant's FastEmbed inference using the `Qdrant/bm42-all-minilm-l6-v2-attentions` model.

**BM42** is Qdrant's sparse embedding model. Unlike BM25, which counts raw token frequencies, BM42 uses BERT-style attention weights to assign token importance, combined with IDF weighting across the corpus. The result is a sparse vector where high-weight dimensions correspond to tokens that are both attention-attended (contextually relevant) and rare in the corpus (informationally distinctive). For a text like "multi-agent LLM orchestration with tool-calling," BM42 assigns high weight to "orchestration" and "tool-calling" and lower weight to "multi" and "with." This makes BM42 more discriminative than BM25 for short technical descriptions while remaining a sparse representation that Qdrant indexes efficiently.

The shared helpers in `lib/qdrant-client.ts`:

```typescript
// Dense embedding via OpenAI text-embedding-3-small
export async function embed(text: string): Promise<number[]> {
  const result = await aiEmbed({
    model: openai.embedding(embeddingModel),
    value: text,
  });
  return result.embedding;
}

// BM42 sparse tokenization via Qdrant FastEmbed inference
export async function sparseEmbed(
  text: string,
  inputType: "document" | "query" = "document"
): Promise<{ indices: number[]; values: number[] }> {
  const result = await qdrant.inference({
    input: [text],
    model: "Qdrant/bm42-all-minilm-l6-v2-attentions",
    input_type: inputType,
  });
  return result.data[0].embedding as { indices: number[]; values: number[] };
}
```

The `inputType` parameter matters. At index time (when inserting a finding), use `"document"` to apply full IDF scaling. At query time, use `"query"` to apply the lighter query-side weighting. This asymmetry is a standard sparse retrieval convention: queries are typically shorter than documents and should not receive full IDF penalization. All four tools call both helpers. Every insert and every search uses the same two models with the same parameters, so vectors across both collections are in compatible embedding spaces, a prerequisite for cross-collection similarity to be meaningful.

---

## Relevance Gating with qdrant_watchlist_match {#relevance-gating}

**`qdrant_watchlist_match`** is the first checkpoint every candidate finding must pass. It computes both a dense embedding and a BM42 sparse tokenization of the candidate's text, then runs a hybrid query against the `watchlist` collection, returning the best-matching active topic and its fused relevance score.

### BM42 Sparse Vectors: Why Hybrid Search {#bm42-hybrid}

The relevance gating problem has two failure modes with a dense-only approach:

**False positives from semantic proximity.** A repository about "generalist robotics agents" scores near the "AI agent frameworks" topic because "agent" is a shared high-frequency token that dense embeddings tend to anchor on. The semantic embedding places them close in vector space even though the robotics context is outside scope.

**False negatives from exact-term mismatch.** A repository named `qwik-city-mcp-plugin` might not score highly against the "TypeScript runtimes and bundlers" topic even though `qwik-city` is a known framework in that ecosystem, because the embedding model has learned to represent "Qwik" as a generic web framework rather than specifically adjacent to "bundler."

BM42 addresses both. For the false positive case, a sparse vector for "generalist robotics agents" will assign high weight to tokens like "robotics" and "generalist" that do not appear in the "AI agent frameworks" topic description, depressing the sparse score even when the dense score is high. For the false negative case, exact token overlap on "TypeScript" between the candidate and the topic description boosts the sparse score even when the semantic embedding falls short.

Reciprocal Rank Fusion (RRF) combines the two rankings:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r_r(d)}$$

where $r_r(d)$ is the rank of document $d$ in ranking $r$ and $k = 60$ is a smoothing constant. A candidate that ranks highly in both dense and sparse search scores well under RRF; a candidate that ranks highly in only one is penalized. This is the right behavior for relevance gating: documents that are both semantically related (dense) and lexically specific (sparse) should win.

### The Hybrid Query and RRF Fusion {#hybrid-query}

```typescript
const denseVector = await embed(candidateText);
const sparseVector = await sparseEmbed(candidateText, "query");

const hits = await qdrant.query("watchlist", {
  prefetch: [
    {
      query: {
        sparse: {
          indices: sparseVector.indices,
          values: sparseVector.values,
        },
      },
      using: "bm42",
      filter: {
        must: [{ key: "active", match: { value: true } }],
      },
      limit: 10,
    },
    {
      query: denseVector,
      using: "dense",
      filter: {
        must: [{ key: "active", match: { value: true } }],
      },
      limit: 10,
    },
  ],
  query: { fusion: "rrf" },
  limit: 1,
  with_payload: true,
});

if (hits.length === 0 || hits[0].score < MATCH_THRESHOLD) {
  return { matched: false, score: hits[0]?.score ?? 0, threshold: MATCH_THRESHOLD };
}
```

The `prefetch` structure is Qdrant's Query API mechanism for multi-stage retrieval. Each prefetch block runs its vector search independently: sparse BM42 and dense cosine each restricted to `active: true` points, each returning 10 candidates. Qdrant then applies RRF fusion across the two candidate sets and returns the top-1 result by fused score.

The `active: true` filter is evaluated during HNSW traversal in both prefetch blocks. Qdrant prunes deactivated topic points from the candidate set before scoring them, so the cost of filtering is proportional to the number of *active* points, not the total collection size. At small watchlist sizes this is immaterial; at hundreds of topics with many deactivated entries, it matters.

### Threshold Design {#threshold-design}

The match threshold is $0.75$ on the fused RRF score. Qdrant's Query API normalizes the fused score to $[0, 1]$ in the response, so this threshold is directly comparable across queries. The $0.75$ value is calibrated empirically on the starter watchlist.

The conservative setting is intentional. The tool's instructions say: "start conservative; tune down once you have real false-negative data." A missed finding surfaces on the next daily scan; a false positive erodes trust in the digest and trains the user to ignore it. The asymmetry favors precision over recall during early operation.

The returned `topic_name` becomes the `matched_topic` field written to every finding, creating a stable link between findings and the watchlist entry that spawned them. The dashboard uses this field for grouping.

---

## Deduplication with qdrant_dedup_check {#deduplication}

**`qdrant_dedup_check`** runs immediately after a candidate passes the relevance gate. It embeds the candidate's `title + summary`, applies a compound payload filter on the `findings` collection, and looks for any existing point with cosine similarity above $0.9$.

```typescript
const cutoff = Date.now() - input.lookback_days * 86_400_000;
const vector = await embed(`${input.title} ${input.summary}`);

const hits = await qdrant.search("findings", {
  vector: { name: "dense", vector },
  filter: {
    must: [
      { key: "repo_full_name", match: { value: input.repo_full_name } },
      { key: "first_seen_at", range: { gte: cutoff } },
    ],
  },
  limit: 1,
  score_threshold: 0.9,
  with_payload: false,
});
```

The filter expression is a compound `must` clause with two conditions:

- `repo_full_name` is a `keyword` field with a payload index. Qdrant uses the index to retrieve only the points for the specified repository before touching the HNSW graph. For a findings collection with 10,000 points spread across 100 repositories, this reduces the candidate set from 10,000 to approximately 100 before any vector computation.
- `first_seen_at` is an `integer` field with a payload index. Combined with the `repo_full_name` filter, the range clause further restricts to findings for this repo created within the lookback window. For a 7-day lookback, this is typically fewer than 10 points.

Deduplication uses dense-only search, not hybrid. The rationale: deduplication is an identity check, not a relevance ranking. Two findings about the same repo with the same synthesized headline will score above $0.9$ on their dense embeddings. Adding sparse search here would introduce noise from minor wording differences in the headline without improving dedup accuracy.

Two thresholds govern the overall system:

- **Relevance gate**: $0.75$ (hybrid RRF score): "is this candidate related to any active topic?"
- **Dedup gate**: $0.9$ (dense cosine similarity): "is this essentially the same finding we have already written?"

The higher dedup threshold is deliberate. $0.75$ is the right bar for "conceptually related," but deduplication needs "essentially identical." You want to deduplicate the same repo appearing on trending twice in a week, but *not* two different repos in the same ecosystem. $0.9$ is tight enough to avoid false deduplication while still catching genuine repeats.

When a duplicate is found, `qdrant_dedup_check` returns `{ duplicate: true, existing_id }`. Callers pass `existing_id` to `qdrant_write_finding`, which enters update mode: it calls `qdrant.setPayload(existing_id, { last_seen_at, trend_signal })` and appends to `provenance`, but does not overwrite `first_seen_at` or `status`.

One case where the dedup logic intentionally allows re-surfacing: **trending_scout** is instructed to skip duplicates *unless* the repo's star count has grown by more than 20% since last seen. Significant momentum is a signal worth re-surfacing even if the repo was seen recently. This override logic lives in the subagent's instructions rather than in the tool; the tool is policy-neutral and the subagent decides when to pass an `existing_id`.

---

## The Orchestrator and Dispatch Loop {#orchestrator}

The orchestrator runs as the root agent in the eve app. Its role is coordination, not execution: it reads the watchlist, dispatches subagents, and updates `last_scanned_at` after each topic completes. It does not directly call the GitHub MCP tools; that is delegated entirely to the subagents.

The dispatch procedure per active topic:

1. **trending_scout**: always dispatched; receives `topic_name`, `topic_description`, and a cutoff timestamp derived from `last_scanned_at` (or 7 days ago if null)
2. **issue_analyst**: dispatched with the topic and the current list of `repo_full_name` values in `findings` with `entity_type: "repo"` and `matched_topic` equal to this topic
3. **release_watcher**: same repo list as issue_analyst
4. **security_watcher**: dispatched only on Sundays (UTC day-of-week check), with the same repo list

The cutoff timestamp propagated to subagents prevents re-processing already-seen content. The `last_scanned_at` field is updated *after all subagents for a topic complete*, not per-subagent. If the scan is interrupted mid-topic, the next run will re-scan from the same cutoff rather than skipping content that was never actually processed. This is the safe choice, at the cost of some duplicate API calls on recovery.

Topics are intended to be processed in parallel when possible. For a watchlist with three topics, the ideal is three concurrent orchestration branches running their four subagent dispatches in parallel. The constraint is the GitHub MCP connection's rate limits, which parallel requests exhaust faster than sequential ones. Sequential topics minimize rate limit pressure at the cost of scan duration.

The orchestrator reads the watchlist using a scroll query filtered to `active: true`:

```typescript
const { points } = await qdrant.scroll("watchlist", {
  filter: {
    must: [{ key: "active", match: { value: true } }],
  },
  with_payload: true,
  with_vector: false,   // orchestrator only needs payload fields
  limit: 100,
});
```

Not loading vectors (`with_vector: false`) is a correct optimization: the orchestrator only needs `topic_name`, `description`, and `last_scanned_at` to build dispatch instructions. Skipping vector retrieval eliminates deserialization cost for 1536-float arrays that would be immediately discarded.

---

## The Four Specialist Subagents {#subagents}

Each subagent is a `defineAgent` declaration with its own `instructions.md` and a description that constrains how the orchestrator dispatches it. All four use `anthropic/claude-sonnet-4.6`. None of the subagents define their own tools; they use the shared tools from `agent/tools/` and the GitHub MCP connections, which are available across the whole app.

### trending_scout: Repository Discovery {#trending-scout}

**trending_scout** discovers repositories not yet tracked. It searches GitHub using the `pushed:>CUTOFF_DATE sort:stars-desc` and `created:>CUTOFF_DATE sort:stars-desc` query patterns, scanning the top 20-30 results per search.

For each candidate, the scout runs the full two-checkpoint pipeline: `qdrant_watchlist_match` first (reject below the hybrid RRF threshold), then `qdrant_dedup_check` (pass `existing_id` if duplicate, but re-surface if star growth exceeds 20%). Only candidates that clear both checks become findings written with `entity_type: "repo"`.

The discipline instruction in trending_scout's system prompt is worth quoting directly: "A missed repo is recoverable on the next daily scan; a false positive erodes trust in the digest." The quality invariant is precision, not recall. The daily cadence means recall is eventually achieved; trust, once damaged by noisy findings, is harder to restore.

### issue_analyst: Issue Pattern Synthesis {#issue-analyst}

**issue_analyst** operates only on repos already in the `findings` collection (passed explicitly by the orchestrator). It fetches 15-30 recent open issues per repo (sorted by `updated`) and synthesizes *patterns*, not individual issues.

The constraint "do not report individual issues" is explicit in the instructions. A digest of 40 issue links is noise; a digest of 4 synthesized patterns like "5 issues this week about OOM crashes under concurrent load in Qdrant" is useful. A pattern requires at least 2-3 distinct issues pointing at the same root problem to qualify.

The `entity_type: "issue"` finding that results is a synthesized headline, not a link to one issue. The `trend_signal.comment_count` is the sum across all issues in the pattern, providing a rough proxy for severity. The `provenance_note` records the synthesis count: "Synthesized from N issues filed in the last X days."

Pattern recognition over issue threads is inherently approximate. Claude Sonnet 4.6 is good at it with 15-30 issues in context, but it will miss subtle patterns requiring domain knowledge the model lacks, and it will occasionally hallucinate a pattern that does not exist. The dedup check provides a floor against re-surfacing the same non-pattern, but it cannot filter hallucinated patterns. This is the weakest link in the current design; more on it in [Challenges](#challenges).

### release_watcher: Release Monitoring {#release-watcher}

**release_watcher** is the most constrained subagent. It operates only on repos it is explicitly given, never runs discovery searches, and skips patch releases without notable changes (security fixes or breaking changes trigger a report regardless of version semantics).

Priority signals in the instructions are explicit:

- **Always report**: major version bumps (`v2.0.0`, `v3.0.0`), releases mentioning breaking changes or security fixes, first stable release after a long beta
- **De-prioritize**: pure bug-fix patch releases, pre-releases unless they have been in RC for more than 30 days

The `title` format `v2.1.0 (vercel/next.js)` is specified in the instructions rather than left to the model, for consistency in dashboard display and dedup matching. The title is part of the text embedded for deduplication; a consistent format ensures that two reports of the same release produce near-identical embeddings and score above the $0.9$ dedup threshold.

### security_watcher: Advisory Scanning {#security-watcher}

**security_watcher** runs once a week (Sundays) rather than daily. Security advisory publication is not correlated with daily cadence, and the GitHub code security API has more limited tooling than the repo/issue search API. Running weekly is sufficient for practical security awareness.

The subagent uses a separate, narrowly scoped MCP connection (`github-security`) rather than the main GitHub connection. This is a least-privilege design: security_watcher gets only the `code_security` toolset, not the broader repo/issue search scope. If the security subagent produces anomalous behavior, the blast radius is limited to the security data scope.

All security_watcher findings use `qdrant_write_advisory` rather than `qdrant_write_finding`. The dedup lookback is extended to 30 days (versus 7 for other entity types) because security advisories can be re-published or updated without new CVE IDs, and a 7-day window would allow the same advisory to re-surface frequently.

The priority filter is hardcoded in the instructions:

| Severity | Has public exploit | Action |
|---|---|---|
| Critical or High | any | Always report |
| Medium | yes | Report |
| Medium | no | Skip unless widely-used package |
| Low | any | Skip |

Missing a critical finding because it was filtered is worse than missing a low-severity one. The bar is set accordingly.

---

## Human-in-the-Loop for Security Findings {#hitl}

**`qdrant_write_advisory`** is the only tool in the system with an approval gate. It uses the `once()` helper from eve's approval system:

```typescript
export default defineTool({
  description: "Write a security advisory finding to Qdrant. REQUIRES HUMAN APPROVAL before executing (once per session).",
  inputSchema: ...,
  approval: once(),
  async execute(input) { ... },
});
```

`once()` means the gate fires exactly one time per session. The first `qdrant_write_advisory` call in a security scan session pauses and waits for a human to approve or deny before executing. All subsequent calls in the same session proceed automatically. This is the right granularity: you want a human to confirm "yes, run the security advisory scan and write what you find," not to approve every individual advisory one by one, which would be unusable at scale.

The approval gate has a consequence that is easy to miss: when a session is dispatched by the daily schedule (an `"app"` authenticator principal), the session is paused waiting for a human response. Since the schedule-triggered session uses the app principal rather than a user principal, nobody is watching to approve the first call. This means security_watcher findings will not be written without explicit human engagement, which is the intended behavior. The system emits an approval request to whatever channel is configured (the eve TUI during development, or a configured Slack/Teams/Discord channel in production) and waits.

This is an explicit design choice: advisory findings require a human to be in the loop before they reach the digest. Auto-approving advisory writes when triggered by the scheduler would mean the system publishes security information without human validation. For a system used to inform remediation decisions, that is a risk not worth taking. If you want fully automated advisory surfacing, the tool provides a comment showing the conditional pattern from the eve docs:

```typescript
approval: ({ session }) => {
  const auth = session.auth.current;
  return auth?.authenticator === "app" &&
    auth.principalId === "eve:app" &&
    auth.principalType === "runtime"
    ? "not-applicable"  // auto-approve on schedule
    : "user-approval";  // require approval for human-initiated runs
},
```

I left this as `once()` rather than the conditional pattern because the conservative default is the right one for security data.

---

## GitHub via MCP: Two Scoped Connections {#github-mcp}

The system uses two GitHub MCP connections rather than one, differentiated by scope:

**`github`** (main connection):
```typescript
defineMcpClientConnection({
  url: "https://api.githubcopilot.com/mcp/x/repos,issues,pull_requests/readonly",
  description: "GitHub repository, issue, and pull request search. Read-only.",
  auth: { getToken: async () => ({ token: process.env.GITHUB_MCP_TOKEN! }) },
})
```

**`github-security`** (security-only connection):
```typescript
defineMcpClientConnection({
  url: "https://api.githubcopilot.com/mcp/x/code_security/readonly",
  description: "GitHub code security advisories. Read-only, scoped to security data only.",
  auth: { getToken: async () => ({ token: process.env.GITHUB_SECURITY_MCP_TOKEN! }) },
})
```

The path-based scope syntax (`/x/repos,issues,pull_requests`) on GitHub's remote MCP server controls which toolsets are exposed. The main connection exposes repository search, issue search, and pull request access. The security connection exposes only the code security advisory API.

Using separate tokens for the two connections makes it possible (though not currently implemented) to issue a narrower token to security_watcher, one with only security advisory read access, while the main connection token retains broader read access. Whether this matters depends on the threat model of the deployment.

One operational note: the connection file must be named `github-security.ts`, not `github_security.ts`. Eve requires connection filenames to use lowercase letters, digits, and dashes only; an underscore is rejected at build time with a descriptive error.

---

## The eve Framework: Schedules, Hooks, and the Build Contract {#eve-framework}

OSS Radar runs on the **eve framework**, which handles the agent runtime, MCP connection management, session lifecycle, and the build pipeline that compiles the TypeScript agent definitions into a deployable Nitro server.

The three primitives that matter most for this system:

**`defineSchedule`** declares the daily cron. The `cron` field is a standard 5-field cron expression:

```typescript
export default defineSchedule({
  cron: "0 13 * * *",  // 13:00 UTC daily
  markdown: `Run the daily OSS Radar scan. ...`,
});
```

The `markdown` field is the system prompt injected into the orchestrator's context when the schedule fires. This is the full scan procedure, written in prose: query the watchlist, dispatch subagents in the right order, check if it is Sunday for security, update `last_scanned_at`. The orchestrator receives this as instructions and executes accordingly.

**`defineHook`** attaches lifecycle callbacks to session events. The `on_schedule_complete` hook fires on `session.completed` for schedule-triggered sessions and POSTs to the dashboard's revalidate endpoint:

```typescript
export default defineHook({
  events: {
    "session.completed": async (_event, ctx) => {
      if (ctx.channel.kind !== "schedule") return;
      // POST to ${DASHBOARD_URL}/api/revalidate with shared secret
    },
  },
});
```

The `ctx.channel.kind !== "schedule"` guard is important: the hook fires for every session completion, not just scheduled ones. Without the guard, it would also fire on interactive chat sessions, sending spurious revalidation requests.

**`eve build`** compiles the agent graph into a Nitro server bundle. The build fetches the AI Gateway model catalog from `https://ai-gateway.vercel.sh/v1/models/catalog` to verify that the model ID specified in each `defineAgent` call has known context window metadata, which it needs to configure session compaction thresholds. Model IDs must match the catalog's slug format exactly: `anthropic/claude-sonnet-4.6` (dot-separated version), not `anthropic/claude-sonnet-4-6` (hyphen). A mismatch produces a build-time error, not a runtime one, which is the right failure mode.

---

## Dashboard and ISR Revalidation {#dashboard}

The **Next.js 16 dashboard** is a separate app in `dashboard/` that reads from Qdrant directly (bypassing the agent runtime entirely) and renders findings grouped by topic.

The data flow:

```
Qdrant findings collection
        │
        │  scroll(filter: {status: "new"}, limit: 200)
        ▼
fetchNewFindings()  →  sorted by first_seen_at desc
        │
groupByTopic()      →  Record<topic_name, Finding[]>
        │
GET /api/digest     →  returned as JSON
        │
HomePage (Server Component, revalidate: 86400)
```

The digest page is a Next.js Server Component with `revalidate: 86400`. It is rendered once and served as a static page, updated at most every 24 hours by ISR. The 24-hour default is intentional: scans run once daily, so fetching from Qdrant on every request would be wasteful. The catch is that with a 24-hour ISR window, the dashboard would show yesterday's findings until the next natural ISR revalidation even though a fresh scan just completed.

The `on_schedule_complete` hook resolves this by calling `POST /api/revalidate` immediately after the scan finishes:

```typescript
// dashboard/src/app/api/revalidate/route.ts
export async function POST(req: Request) {
  const secret = req.headers.get("x-radar-secret");
  if (!secret || secret !== process.env.RADAR_REVALIDATE_SECRET) {
    return new Response("Unauthorized", { status: 401 });
  }
  revalidatePath("/");
  return new Response("ok", { status: 200 });
}
```

The shared secret (`RADAR_REVALIDATE_SECRET`) prevents arbitrary cache busting. `revalidatePath("/")` triggers Next.js's on-demand ISR: the next request to `/` re-renders the page server-side, fetching fresh findings from Qdrant, and caches the result for another 24 hours.

The result is a dashboard that is static (fast, cacheable, no per-request Qdrant queries) but updates within seconds of each scan completing.

The `fetchNewFindings()` function reads only findings with `status: "new"` and a hard limit of 200 points. The status filter keeps the digest focused on freshly discovered content rather than showing the full historical archive. The 200-point limit is a practical guard against a scenario where the findings collection grows large enough that a scroll query would time out or return more items than a useful digest can display. In steady state, 200 findings per digest is already far more than any reader will consume; the limit is a ceiling, not a target.

---

## Challenges and Open Problems {#challenges}

**Issue pattern synthesis is model-dependent and hard to validate.** The issue_analyst subagent's core value (synthesizing recurring themes from raw GitHub issue text) relies on the model's judgment about what constitutes a pattern. There is no ground truth to validate against, and a model that synthesizes confident-sounding but incorrect patterns is worse than no synthesis at all. One partial mitigation would be to require the subagent to cite specific issue numbers in the `provenance_note` for every pattern claim, making patterns auditable. This is not currently enforced.

**The dedup threshold is not adaptive.** A fixed $0.9$ cosine similarity threshold works well for exact repeats but may allow near-duplicates through in domains with specialized vocabulary. Two CVE advisories for the same vulnerability reported by different sources may have sufficiently different summaries to score below $0.9$. An adaptive threshold that tightens for `advisory` entity types and loosens for `repo` findings would be more appropriate than a single global number.

**Hybrid search threshold calibration requires labeled data.** The relevance gate uses $0.75$ on the RRF fused score, calibrated manually on the starter watchlist. As the watchlist grows beyond the initial three topics, the RRF score distribution may shift because IDF weights in the BM42 sparse index depend on corpus statistics: adding topics changes IDF values for existing tokens, which changes sparse scores for existing queries. A mechanism to re-evaluate the threshold when the watchlist changes significantly would prevent precision drift.

**Rate limits on the GitHub MCP connection are unaddressed.** The orchestrator is designed to run topics in parallel, but GitHub's API has rate limits that parallel subagent dispatches will hit sooner than sequential ones. There is no retry logic, no rate-limit-aware pacing, and no fallback when a subagent fails due to rate limiting versus failing due to a genuine absence of findings. Structured error reporting from failed subagent runs would let the orchestrator distinguish "nothing found" from "rate limited."

**`last_scanned_at` update timing creates a re-scan risk.** The orchestrator updates `last_scanned_at` after all subagents for a topic complete. If the scan is interrupted between subagent dispatch and completion (a process crash, a timeout, or a transient error), `last_scanned_at` is not updated and the next scan will re-scan from the same cutoff. This produces duplicate candidates, which the dedup check handles, but it also means unnecessary API calls and embedding costs. A per-subagent `scanned_at` marker in the watchlist payload would make recovery more precise.

**No mechanism for aging findings out.** The `status` field has a `"tracked"` state defined in the schema, but nothing currently transitions findings from `"new"` to `"tracked"` over time. A finding from six months ago with no updates should not appear in the same digest as a finding from yesterday. A time-based status transition (move to `"tracked"` after 30 days without a `last_seen_at` update) would keep the digest focused on genuinely current content.

**Security scan on Sundays only, with no compensating mechanism for zero-day events.** The weekly security scan is a practical choice, but critical vulnerabilities can be published on any day of the week. A manual trigger for the security subagent (allowing a user to run `security_watcher` on demand for specific repositories) would close the gap without requiring daily execution of the full security scan for every tracked repo.

**The watchlist is static.** Topics are seeded manually and not updated by the agent. A trending_scout that consistently surfaces repos in a domain not yet in the watchlist has no mechanism to suggest adding a new topic. Some form of topic suggestion (even just a channel message when the scout finds multiple highly-scored repos that match no existing topic above $0.75$) would help the watchlist evolve as the ecosystem does.

Despite these, the core design is sound: hybrid semantic relevance gating and Qdrant payload-filtered deduplication keep the findings collection clean; subagent specialization keeps each agent's scope narrow and its instructions clear; and the human approval gate for security findings reflects appropriate caution for content that influences remediation decisions. The system is in production, running daily, and generating genuine signal across three topic areas.

---

## References

- **Qdrant vector database**: [qdrant.tech](https://qdrant.tech), the vector store powering both collections, with named vectors, sparse BM42, and payload-filtered HNSW traversal
- **BM42** (Qdrant, 2024): [qdrant.tech/articles/bm42](https://qdrant.tech/articles/bm42/), Qdrant's sparse embedding model combining BERT attention weights with IDF weighting
- **Reciprocal Rank Fusion** (Cormack, Clarke & Buettcher, 2009): [doi.org/10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114), the fusion algorithm used to combine BM42 sparse and dense rankings in the hybrid query
- **eve agent framework** (v0.17.1): [eve.dev](https://eve.dev), schedules, hooks, connections, and the MCP client runtime
- **AI SDK** (Vercel, 2024): [sdk.vercel.ai](https://sdk.vercel.ai), the `embed()` call and AI Gateway model routing
- **GitHub Remote MCP Server**: [github.com/modelcontextprotocol](https://github.com/modelcontextprotocol), the remote MCP endpoint for repository, issue, and security data
- **Model Context Protocol** (Anthropic, 2024): [modelcontextprotocol.io](https://modelcontextprotocol.io), the protocol underlying the GitHub and eve tool connections
- **Next.js Incremental Static Regeneration**: [nextjs.org/docs/app/building-your-application/caching#on-demand-revalidation](https://nextjs.org/docs/app/building-your-application/caching#on-demand-revalidation), the dashboard's cache invalidation mechanism
- **text-embedding-3-small** (OpenAI): [platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings), the default dense embedding model for both Qdrant collections
