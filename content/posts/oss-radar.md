---
title: "OSS Radar: A Multi-Agent System for Continuous Open-Source Intelligence"
date: 2026-06-30
description: "A deep dive into OSS Radar — a production multi-agent system that runs nightly GitHub scans, uses Qdrant vector search for semantic relevance gating and deduplication, and surfaces findings through a Next.js digest dashboard. Covers the orchestration architecture, the two-collection knowledge layer, subagent design, the human-in-the-loop security gate, and the schedule-to-dashboard notification chain."
tags: ["agents", "qdrant", "vector-search", "multi-agent", "eve", "mcp", "github", "semantic-search", "human-in-the-loop"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Keeping track of a fast-moving OSS ecosystem is expensive. If you care about, say, AI agent frameworks, TypeScript runtimes, and vector databases, you are potentially watching hundreds of repositories. Doing this manually means opening GitHub trending every morning, reading changelogs when you remember, and finding out about critical CVEs days after they were published. The signal is real but scattered; aggregating it by hand doesn't scale.

The alternative is an autonomous system that does the watching for you: discovers new repositories matching your interests, synthesizes emerging issue patterns, catches new releases, and surfaces security advisories — all filtered against a set of topics you define, deduplicated across runs, and delivered as a daily digest. That is what **OSS Radar** is.

This post covers the design in full: the two-collection Qdrant knowledge layer that handles semantic relevance and deduplication, the multi-agent orchestration pattern using the eve framework, how each of the four specialist subagents works, the human-in-the-loop gate for security findings, and the schedule-to-dashboard notification chain. I will focus on the design decisions and their tradeoffs. Familiarity with vector databases, Qdrant, and basic agent frameworks is assumed.

---

## Table of Contents

1. [The Problem: OSS Intelligence at Scale](#the-problem)
2. [System Overview](#system-overview)
3. [The Two-Collection Qdrant Knowledge Layer](#qdrant-knowledge-layer)
   - [watchlist: Semantic Topic Registry](#watchlist-collection)
   - [findings: The Deduplication Target](#findings-collection)
   - [Embedding Strategy](#embedding-strategy)
4. [Relevance Gating with qdrant_watchlist_match](#relevance-gating)
5. [Deduplication with qdrant_dedup_check](#deduplication)
6. [The Orchestrator and Dispatch Loop](#orchestrator)
7. [The Four Specialist Subagents](#subagents)
   - [trending_scout: Repository Discovery](#trending-scout)
   - [issue_analyst: Issue Pattern Synthesis](#issue-analyst)
   - [release_watcher: Release Monitoring](#release-watcher)
   - [security_watcher: Advisory Scanning](#security-watcher)
8. [Human-in-the-Loop for Security Findings](#hitl)
9. [GitHub via MCP: Two Scoped Connections](#github-mcp)
10. [The eve Framework: Schedules, Hooks, and the Build Contract](#eve-framework)
11. [Dashboard and ISR Revalidation](#dashboard)
12. [Challenges and Open Problems](#challenges)

---

## The Problem: OSS Intelligence at Scale {#the-problem}

The information asymmetry in OSS tracking is real. A new agent framework can go from 0 to 10k stars in a week; a critical CVE in a widely-used package might be published on a Tuesday afternoon; a library you depend on might drop a major version with breaking changes the same day you have three other things to review. Individual tracking tools exist for pieces of this — GitHub notifications for repos you follow, dependabot for CVEs in your own dependencies — but none of them surface *new* things you haven't yet decided to follow, or synthesize patterns across a portfolio of tracked projects.

OSS Radar is designed for the specific problem of *topic-scoped, continuous intelligence*. You define topics — "AI agent frameworks," "TypeScript runtimes and bundlers," "vector databases" — and the system finds what's worth knowing, every day, across all four dimensions: new repos, issue patterns, releases, and security advisories. Findings are deduplicated across runs so you don't see the same repo three days in a row, and relevance is gated semantically so keyword overlap alone is not sufficient to generate a finding.

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

## The Two-Collection Qdrant Knowledge Layer {#qdrant-knowledge-layer}

The entire persistence layer is two Qdrant collections. This is the right number: one more and you need joins; one fewer and you conflate intent (what to watch) with output (what was found).

### watchlist: Semantic Topic Registry {#watchlist-collection}

The `watchlist` collection stores the topics the system monitors. Each point represents one topic:

| Payload field | Type | Purpose |
|---|---|---|
| `topic_name` | string | Human-readable label |
| `description` | string | Full natural-language description of what the topic covers |
| `active` | boolean | If false, this topic is skipped entirely |
| `created_at` | number | Unix ms timestamp |
| `last_scanned_at` | number \| null | Updated after every successful scan; used as the cutoff for the next scan |

The vector stored per point is the embedding of the `description` field. This is the key design decision: the description is what gets semantically matched against candidate findings, so it is also what gets embedded. When the semantic relevance check runs, it embeds the candidate's text and finds its nearest neighbor in this collection — that neighbor is the matched topic.

Topics are seeded manually (via `scripts/seed_watchlist.ts`) and do not change at runtime. The starter set covers three domains: AI agent frameworks, TypeScript runtimes and bundlers, and vector databases — all defined with enough breadth in their description to cover adjacent terminology (e.g., the AI agent frameworks topic explicitly lists "LLM orchestration" and "multi-agent coordination" in addition to the obvious names).

### findings: The Deduplication Target {#findings-collection}

The `findings` collection stores everything the subagents surface. Every point is one finding:

| Payload field | Type | Purpose |
|---|---|---|
| `entity_type` | enum | `repo`, `issue`, `release`, `advisory` |
| `repo_full_name` | string | GitHub `owner/repo` |
| `title` | string | Finding headline |
| `summary` | string | 1–3 sentence description |
| `url` | string | Link to the source |
| `source_subagent` | enum | Which subagent produced this |
| `matched_topic` | string | Which watchlist topic this finding matched |
| `first_seen_at` | number | Unix ms of first insertion |
| `last_seen_at` | number | Updated on duplicate detection |
| `status` | enum | `new`, `tracked`, `superseded` |
| `trend_signal` | object | `stars`, `star_delta`, `comment_count` (all nullable) |
| `provenance` | array | Append-only log of `{subagent, note, at}` entries |
| `severity` | enum \| null | Advisory-only: `critical`, `high`, `medium`, `low`, `unknown` |
| `cve_ids` | string[] \| null | Advisory-only: CVE identifiers |

The vector stored per point is the embedding of `title + summary`. This dual-use is intentional: the same vector that enables semantic deduplication (cosine similarity against existing findings) also enables semantic search over the findings corpus. The dashboard doesn't exploit this yet, but the capability is there.

Provenance deserves specific attention. It is an append-only array, never overwritten. When an existing finding is updated — because a subagent encountered the same repo again with a significantly higher star count — the new subagent call, timestamp, and note are *appended* to `provenance`. This creates a lightweight audit trail without any separate history collection.

### Embedding Strategy {#embedding-strategy}

Both collections use OpenAI's `text-embedding-3-small` via the AI SDK's `embed()` call. The embedding model is configurable via the `EMBEDDING_MODEL` environment variable, but the default is appropriate: `text-embedding-3-small` produces 1536-dimensional vectors, runs cheaply at scale, and has strong performance on the kind of short, technical descriptions that topic and finding text consists of.

The shared `embed()` helper in `lib/qdrant-client.ts` wraps the AI SDK call:

```typescript
export async function embed(text: string): Promise<number[]> {
  const result = await aiEmbed({
    model: openai.embedding(embeddingModel),
    value: text,
  });
  return result.embedding;
}
```

All four tools call this function directly. Every embed call goes to the same model with the same parameters, so vectors in `watchlist` and `findings` are in the same embedding space — a prerequisite for cross-collection similarity search to be meaningful.

---

## Relevance Gating with qdrant_watchlist_match {#relevance-gating}

**`qdrant_watchlist_match`** is the first checkpoint every candidate finding must pass. It embeds the candidate's text and performs a filtered nearest-neighbor search against the `watchlist` collection, returning the best-matching active topic and its similarity score. If the score is below the threshold, the finding is rejected.

```typescript
const hits = await qdrant.search("watchlist", {
  vector,
  filter: {
    must: [{ key: "active", match: { value: true } }],
  },
  limit: 1,
  with_payload: true,
});

if (hits.length === 0 || hits[0].score < MATCH_THRESHOLD) {
  return { matched: false, score: hits[0]?.score ?? 0, threshold: MATCH_THRESHOLD };
}
```

The threshold is $0.75$ cosine similarity. This is explicitly conservative — the tool's description says "start conservative; tune down once you have real false-negative data." The rationale: a missed finding surfaces on the next daily scan; a false positive erodes trust in the digest and trains the user to ignore it. The asymmetry favors precision over recall during early operation.

The filter on `active: true` is important for efficiency. Without it, the nearest-neighbor search could match a deactivated topic. More subtly, because Qdrant evaluates filter conditions during HNSW graph traversal rather than as a post-filter over results, restricting to active points is effectively free: it narrows the candidate set before scoring rather than scoring all points and then discarding inactive ones. At small watchlist sizes this doesn't matter; at hundreds of topics it does.

The returned `topic_name` becomes the `matched_topic` field written to every finding. This creates a stable link between findings and the watchlist entry that spawned them, which the dashboard uses for grouping.

---

## Deduplication with qdrant_dedup_check {#deduplication}

**`qdrant_dedup_check`** runs immediately after a candidate passes the relevance gate. It embeds the candidate's `title + summary`, filters the `findings` collection by `repo_full_name` and a `first_seen_at` range (the lookback window), and looks for any existing point with cosine similarity above $0.9$.

```typescript
const cutoff = Date.now() - input.lookback_days * 86_400_000;
const vector = await embed(`${input.title} ${input.summary}`);

const hits = await qdrant.search("findings", {
  vector,
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

Two thresholds govern the system's behavior:

- **Relevance gate**: $0.75$ — "is this candidate related to any active topic?"
- **Dedup gate**: $0.9$ — "is this essentially the same finding we've already written?"

The higher dedup threshold is deliberate. $0.75$ is the right bar for "conceptually related," but deduplication needs "essentially identical" — you want to deduplicate the same repo appearing on trending twice in a week, but *not* two different repos in the same ecosystem. $0.9$ is tight enough to avoid false deduplication while still catching genuine repeats.

The `repo_full_name` filter pre-screens by repository before cosine scoring. This is both a precision improvement (two findings about different repos can be semantically similar without being duplicates) and a performance improvement (the scoring set is bounded by the number of prior findings for that repo, not the whole collection).

When a duplicate is found, `qdrant_dedup_check` returns `{ duplicate: true, existing_id }`. Callers then pass `existing_id` to `qdrant_write_finding`, which enters update mode: it refreshes `last_seen_at`, updates `trend_signal`, and appends to `provenance` — but does not overwrite the original `first_seen_at` or `status`.

One case where the dedup logic intentionally allows re-surfacing: **trending_scout** is instructed to skip duplicates *unless* the repo's star count has grown by more than 20% since last seen. Significant momentum is a signal worth re-surfacing even if the repo was seen recently. This override logic lives in the subagent's instructions rather than in the tool — the tool is policy-neutral; the subagent decides when to pass an `existing_id`.

---

## The Orchestrator and Dispatch Loop {#orchestrator}

The orchestrator runs as the root agent in the eve app. Its role is coordination, not execution: it reads the watchlist, dispatches subagents, and updates `last_scanned_at` after each topic completes. It does not directly call the GitHub MCP tools — that is delegated to the subagents.

The dispatch procedure per active topic:

1. **trending_scout**: always dispatched; receives `topic_name`, `topic_description`, and a cutoff timestamp derived from `last_scanned_at` (or 7 days ago if null)
2. **issue_analyst**: dispatched with the topic and the current list of `repo_full_name` values in `findings` with `entity_type: "repo"` and `matched_topic` equal to this topic
3. **release_watcher**: same repo list as issue_analyst
4. **security_watcher**: dispatched only on Sundays (UTC day-of-week check), with the same repo list

The cutoff timestamp propagated to subagents is the mechanism that prevents re-processing already-seen content. The `last_scanned_at` field is updated *after all subagents for a topic complete*, not per-subagent. This is the safe choice: if the scan is interrupted mid-topic, the next run will re-scan from the same cutoff rather than skipping content that was never actually processed.

Topics are intended to be processed in parallel when possible. For a watchlist with three topics, the ideal is three concurrent orchestration branches running their four subagent dispatches in parallel. The constraint is the GitHub MCP connection, which has rate limits that parallel requests will hit faster. The tradeoff is worth noting: sequential topics minimize rate limit pressure at the cost of scan duration.

---

## The Four Specialist Subagents {#subagents}

Each subagent is a `defineAgent` declaration with its own `instructions.md` and a description that constrains how the orchestrator dispatches it. All four use `anthropic/claude-sonnet-4.6`. None of the subagents define their own tools — they use the shared tools from `agent/tools/` and the GitHub MCP connections, which are available across the whole app.

### trending_scout: Repository Discovery {#trending-scout}

**trending_scout** discovers repositories not yet tracked. It searches GitHub using the `pushed:>CUTOFF_DATE sort:stars-desc` and `created:>CUTOFF_DATE sort:stars-desc` query patterns, scanning the top 20–30 results per search.

For each candidate, the scout runs the full two-checkpoint pipeline: `qdrant_watchlist_match` first (reject below 0.75), then `qdrant_dedup_check` (pass `existing_id` if duplicate, but re-surface if star growth > 20%). Only candidates that clear both checks become findings written with `entity_type: "repo"`.

The discipline instruction in trending_scout's system prompt is worth quoting directly: "A missed repo is recoverable on the next daily scan; a false positive erodes trust in the digest." This framing is conscious. The quality invariant is precision, not recall. The daily cadence means recall is eventually achieved; trust, once damaged by noisy findings, is harder to restore.

### issue_analyst: Issue Pattern Synthesis {#issue-analyst}

**issue_analyst** operates only on repos already in the `findings` collection (passed explicitly by the orchestrator). It fetches 15–30 recent open issues per repo (sorted by `updated`) and synthesizes *patterns* — not individual issues.

This is the most distinctively designed subagent. The constraint "do not report individual issues" is explicit in the instructions. A digest of 40 issue links is noise; a digest of 4 synthesized patterns like "5 issues this week about OOM crashes under concurrent load in Qdrant" is useful. A pattern requires at least 2–3 distinct issues pointing at the same root problem to qualify.

The `entity_type: "issue"` finding that results is a synthesized headline, not a link to one issue. The `trend_signal.comment_count` is the sum across all issues in the pattern, providing a rough proxy for severity. The `provenance_note` records the synthesis count: "Synthesized from N issues filed in the last X days."

Pattern recognition over issue threads is inherently approximate. Claude Sonnet 4.6 is good at it with 15–30 issues in context, but it will miss subtle patterns requiring domain knowledge the model lacks, and it will occasionally hallucinate a pattern that doesn't exist. The dedup check provides a floor against re-surfacing the same non-pattern, but it can't filter hallucinated patterns. This is the weakest link in the current design — more on it in [Challenges](#challenges).

### release_watcher: Release Monitoring {#release-watcher}

**release_watcher** is the most constrained subagent. It operates only on repos it is explicitly given, never runs discovery searches, and skips patch releases without notable changes (security fixes or breaking changes trigger a report regardless of version semantics).

Priority signals in the instructions are explicit:

- **Always report**: major version bumps (`v2.0.0`, `v3.0.0`), releases mentioning breaking changes or security fixes, first stable release after a long beta
- **De-prioritize**: pure bug-fix patch releases, pre-releases unless they've been in RC for > 30 days

The `title` format `v2.1.0 — vercel/next.js` is specified in the instructions rather than left to the model, for consistency in the dashboard display and dedup matching (the title is part of the embedding used for deduplication).

### security_watcher: Advisory Scanning {#security-watcher}

**security_watcher** runs once a week (Sundays) rather than daily. This is a deliberate choice: security advisory publication is not correlated with daily cadence, and the GitHub code security API has more limited tooling than the repo/issue search API. Running weekly is sufficient for practical security awareness.

The subagent uses a separate, narrowly scoped MCP connection (`github-security`) rather than the main GitHub connection. This is a least-privilege design: security_watcher gets only the `code_security` toolset, not the broader repo/issue search scope. If the security subagent were compromised or produced anomalous behavior, the blast radius is limited to the security data scope.

All security_watcher findings use `qdrant_write_advisory` rather than `qdrant_write_finding`. The dedup lookback is extended to 30 days (versus 7 for other entity types) because security advisories can be re-published or updated without new CVE IDs, and a 7-day window would allow the same advisory to re-surface frequently.

The priority filter is hardcoded in the instructions:

| Severity | Has public exploit | Action |
|---|---|---|
| Critical or High | any | Always report |
| Medium | yes | Report |
| Medium | no | Skip unless widely-used package |
| Low | any | Skip |

This is a deliberate trade: missing a critical finding because it was filtered is worse than missing a low-severity one. The bar is set accordingly.

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

`once()` means the gate fires exactly one time per session. The first `qdrant_write_advisory` call in a security scan session pauses and waits for a human to approve or deny before executing. All subsequent calls in the same session proceed automatically. This is the right granularity for the use case: you want a human to confirm "yes, run the security advisory scan and write what you find," not to approve every individual advisory one by one — that would be unusable at scale.

The approval gate has a consequence that's easy to miss: when a session is dispatched by the daily schedule (an `"app"` authenticator principal), the session is paused waiting for a human response. Since the schedule-triggered session uses the app principal rather than a user principal, nobody is watching to approve the first call. This means **security_watcher findings will not be written without explicit human engagement** — which is the intended behavior. The system will emit an approval request to whatever channel is configured (the eve TUI during development, or a configured Slack/Teams/Discord channel in production) and wait.

This is an explicit design choice: advisory findings require a human to be in the loop before they reach the digest. The alternative — auto-approving advisory writes when triggered by the scheduler — would mean the system publishes security information without human validation. For a system used to inform remediation decisions, that's a risk not worth taking. If you want fully automated advisory surfacing, the tool provides a comment showing the conditional pattern from the eve docs:

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

I left this as `once()` rather than the conditional pattern because I think the conservative default is the right one for security data.

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

The path-based scope syntax (`/x/repos,issues,pull_requests`) on GitHub's remote MCP server controls which toolsets are exposed. The main connection exposes repository search, issue search, and pull request access — the tools trending_scout, issue_analyst, and release_watcher need. The security connection exposes only the code security advisory API.

Using separate tokens for the two connections means it is possible (though not currently implemented) to issue a narrower token to security_watcher — one with only security advisory read access — while the main connection token retains broader read access. Whether this matters depends on the threat model of the deployment.

One operational note: the connection file must be named `github-security.ts`, not `github_security.ts`. Eve requires connection filenames to use lowercase letters, digits, and dashes only — an underscore is rejected at build time with a descriptive error. This is the kind of constraint that is easy to get wrong the first time.

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

The `markdown` field is the system prompt injected into the orchestrator's context when the schedule fires. This is the full scan procedure, written in prose: query the watchlist, dispatch subagents in the right order, check if it's Sunday for security, update `last_scanned_at`. The orchestrator receives this as instructions and executes accordingly.

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

**`eve build`** compiles the agent graph into a Nitro server bundle. The build fetches the AI Gateway model catalog from `https://ai-gateway.vercel.sh/v1/models/catalog` to verify that the model ID specified in each `defineAgent` call has known context window metadata — metadata it needs to configure the session compaction thresholds. Model IDs must match the catalog's slug format exactly: `anthropic/claude-sonnet-4.6` (dot-separated version), not `anthropic/claude-sonnet-4-6` (hyphen). A mismatch produces a build-time error, not a runtime one, which is the right failure mode.

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

The digest page is a Next.js Server Component with `revalidate: 86400` — it is rendered once and served as a static page, updated at most every 24 hours by ISR. The 24-hour default is intentional: scans run once daily, so fetching from Qdrant on every request would be wasteful. The catch is that with a 24-hour ISR window, the dashboard would show yesterday's findings until the next natural ISR revalidation even though a fresh scan just completed.

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

The shared secret (`RADAR_REVALIDATE_SECRET`) prevents arbitrary cache busting. `revalidatePath("/")` triggers Next.js's on-demand ISR: the next request to `/` will re-render the page server-side, fetching fresh findings from Qdrant, and cache the result for another 24 hours.

The result is a dashboard that is static (fast, cacheable, no per-request Qdrant queries) but updates within seconds of each scan completing — the best of both.

The `fetchNewFindings()` function reads only findings with `status: "new"` and a hard limit of 200 points. The status filter keeps the digest focused on freshly discovered content rather than showing the full historical archive. The 200-point limit is a practical guard against a scenario where the findings collection grows large enough that a scroll query would timeout or return too many items for a useful digest display. In steady state, 200 findings per digest is already far more than any reader will consume; the limit is a ceiling, not a target.

---

## Challenges and Open Problems {#challenges}

**Issue pattern synthesis is model-dependent and hard to validate.** The issue_analyst subagent's core value — synthesizing recurring themes from raw GitHub issue text — relies on the model's judgment about what constitutes a pattern. There is no ground truth to validate against, and a model that synthesizes confident-sounding but incorrect patterns is worse than no synthesis at all. One partial mitigation would be to require the subagent to cite specific issue numbers in the `provenance_note` for every pattern claim, making patterns auditable, but this is not currently enforced.

**The dedup threshold is not adaptive.** A fixed $0.9$ cosine similarity threshold works well for exact repeats but may allow near-duplicates through in domains with specialized vocabulary (two CVE advisories for the same vulnerability reported by different sources may have different enough summaries to score below $0.9$). An adaptive threshold that tightens for advisory entity types and loosens for repo findings would be more appropriate than a single global number.

**Rate limits on the GitHub MCP connection are unaddressed.** The orchestrator is designed to run topics in parallel, but GitHub's API has rate limits that parallel subagent dispatches will hit sooner than sequential ones. There is no retry logic, no rate-limit-aware pacing, and no fallback when a subagent fails due to rate limiting versus failing due to a genuine absence of findings. Structured error reporting from failed subagent runs would let the orchestrator distinguish "nothing found" from "rate limited."

**`last_scanned_at` update timing creates a re-scan risk.** The orchestrator updates `last_scanned_at` after all subagents for a topic complete. If the scan is interrupted between subagent dispatch and completion — a process crash, a timeout, a transient error — `last_scanned_at` is not updated and the next scan will re-scan from the same cutoff. This produces duplicate candidates, which the dedup check handles, but it also means unnecessary API calls and embedding costs. A per-subagent `scanned_at` marker in the watchlist payload would make recovery more precise.

**No mechanism for aging findings out.** The `status` field has a `"tracked"` state defined in the schema, but nothing currently transitions findings from `"new"` to `"tracked"` over time. A finding surfaces as new until something explicitly changes its status. A finding from six months ago with no updates should not appear in the same digest as a finding from yesterday. A time-based status transition (e.g., move to `"tracked"` after 30 days without a `last_seen_at` update) would keep the digest focused on genuinely current content.

**Security scan on Sundays only, with no compensating mechanism for zero-day events.** The weekly security scan is a practical choice, but critical vulnerabilities can be published on any day of the week. A manual trigger for the security subagent — allowing a user to run `security_watcher` on demand for specific repositories — would close the gap without requiring daily execution of the full security scan for every tracked repo.

**The watchlist is static.** Topics are seeded manually and not updated by the agent. A trending_scout that consistently surfaces repos in a domain not yet in the watchlist has no mechanism to suggest adding a new topic. Some form of topic suggestion — even just a channel message when the scout finds multiple highly-scored repos that match no existing topic above 0.75 — would help the watchlist evolve as the ecosystem does.

Despite these, the core design is sound: semantic relevance gating and deduplication ensure the findings collection stays clean; the subagent specialization keeps each agent's scope narrow and its instructions clear; and the human approval gate for security findings reflects appropriate caution for content that influences remediation decisions. The system is in production, running daily, and generating genuine signal across three topic areas — which is the right test.

---

## References

- **Qdrant vector database**: [qdrant.tech](https://qdrant.tech) — the vector store powering both collections
- **eve agent framework** (v0.17.1): [eve.dev](https://eve.dev) — schedules, hooks, connections, and the MCP client runtime
- **AI SDK** (Vercel, 2024): [sdk.vercel.ai](https://sdk.vercel.ai) — the `embed()` call and AI Gateway model routing
- **GitHub Remote MCP Server**: [github.com/modelcontextprotocol](https://github.com/modelcontextprotocol) — the remote MCP endpoint for repository, issue, and security data
- **Model Context Protocol** (Anthropic, 2024): [modelcontextprotocol.io](https://modelcontextprotocol.io) — the protocol underlying the GitHub and eve tool connections
- **Next.js Incremental Static Regeneration**: [nextjs.org/docs/app/building-your-application/caching#on-demand-revalidation](https://nextjs.org/docs/app/building-your-application/caching#on-demand-revalidation) — the dashboard's cache invalidation mechanism
- **BM25** (Robertson & Zaragoza, 2009): [doi.org/10.1561/1500000019](https://doi.org/10.1561/1500000019) — the term-weighting function referenced in the DocuLayer post and mentioned here for completeness; OSS Radar uses dense vector search rather than BM25
- **text-embedding-3-small** (OpenAI): [platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings) — the default embedding model for both Qdrant collections
