---
title: "Grounding Agents in Live Documentation: The DocuLayer Approach"
date: 2026-06-22
description: "How DocuLayer eliminates AI agent hallucination of stale API signatures by fetching live documentation on demand and running BM25 search locally -- no embeddings, no vector database, no generated text."
tags: ["agents", "mcp", "documentation", "bm25", "retrieval", "hallucination"]
author: "Mihir Inamdar"
showToc: true
math: true
---

One of the more reliable failure modes of AI coding agents is confidently wrong API usage. Not garbled syntax, not logical errors -- just a function signature from six months ago, a parameter that was renamed, a method that didn't exist at training time. The model doesn't know it's wrong. Its training data said this is how `httpx.AsyncClient.get()` works, and nothing in the conversation contradicts that.

The standard mitigation is retrieval-augmented generation: build a vector index of documentation, embed queries, retrieve the nearest chunks, paste them into context. This works. It also requires a vector database, an embedding model, an ingestion pipeline, and some scheme for keeping the index fresh. It's a lot of infrastructure to solve what is fundamentally a staleness problem.

DocuLayer, by Mihir Inamdar, takes a simpler route. It sits between the agent and live documentation, fetches content on demand, runs BM25 search locally, and returns verbatim text -- no embeddings, no disk storage, no generated text. This post covers how it works, where the design choices are interesting, and what it trades away.

Pre-reads: familiarity with MCP (Model Context Protocol) and basic IR concepts (TF-IDF, BM25) will help.

## The Problem: Documentation Drift

Language models are trained on static snapshots of the web. The snapshot for a given model might be six months old, a year old, older. Documentation for actively maintained libraries changes faster than that -- sometimes dramatically. FastAPI added a dependency injection redesign. Pydantic v2 broke the v1 API in numerous places. React's hooks documentation has evolved continuously. `httpx` renamed parameters. OpenAI has deprecated and replaced entire API surfaces.

When an agent is asked about these libraries, it pulls from training data. If training data has the old version, it answers with confidence about behavior that no longer exists. The generated code compiles. It might even run. It just doesn't do what the documentation says it should.

The naive fix is to put the current docs in the system prompt. This works for small libraries or targeted tasks. For anything at scale, you're immediately fighting the context window. A full Next.js documentation site is tens of thousands of words. You can't paste it in wholesale.

RAG solves the scale problem: embed the docs, retrieve the relevant chunks, insert only what's needed. But this adds the freshness problem back in -- embeddings are of a snapshot, not the live docs. You need scheduled re-indexing, staleness detection, delta pipelines.

DocuLayer's approach: don't index offline at all. Fetch on demand, cache in memory for a configurable TTL, and search the freshly fetched content locally.

## Architecture: No Database, No Embeddings

```
Your agent  (Claude, Cursor, Codex, any MCP client…)
    │   "what parameters does httpx.AsyncClient.get() take?"
    ▼
┌──────────────────────────────────────────────────────────────┐
│  DocuLayer                                                   │
│  ──────────────────────────────────────────────────────────  │
│  resolve_identifier  →  shortcut table / PyPI / npm         │
│                                                              │
│  discover_llms_txt   →  targeted page index (when present)  │
│       keyword score entries  →  fetch only relevant pages   │
│                                                              │
│  DocParser  (HTML → Markdown, heading split)                │
│  DocSearcher  (BM25 -- no ML, no embeddings, no network)     │
│                                                              │
│  TTLCache  (in-memory only -- zero disk writes)              │
└──────────────────────────────────────────────────────────────┘
    │   verbatim section + source URL + "fetched 3s ago"
    ▼
LLM  --  reads real docs, answers correctly
```

The pipeline on every query is:

1. **Resolve the identifier** -- `"httpx"` goes through a shortcut table first; on a miss, falls back to PyPI JSON API or npm registry to find the real docs URL
2. **Discover llms.txt** -- probe the root of the docs site for `/llms.txt`; if present, parse the indexed entries
3. **Select candidate pages** -- keyword-score the llms.txt entries against the query, pick the top 1–3
4. **Fetch and parse** -- download those pages in parallel, convert HTML to Markdown, split by heading into `DocSection` objects
5. **BM25 search** -- run Okapi BM25 over all sections from the fetched pages, return the top-$k$ ranked by score
6. **Return verbatim** -- no rewriting, no summarizing; the raw section text goes back to the caller with a source attribution header

The TTLCache holds `FetchResult` objects keyed by URL. Default TTL is one hour. On a cache hit, steps 4–5 run over cached content; steps 1–3 are still needed to determine which URLs to look up. On a miss, it fetches live.

Nothing writes to disk. The cache is process-local. When the process restarts, it's empty.

## llms.txt as a Navigation Layer

**The llms.txt standard** (Howard, 2024, [llmstxt.org](https://llmstxt.org)) proposes that documentation sites publish a Markdown file at `/llms.txt` containing structured links to machine-readable versions of their pages. The format is intentionally simple: an H1 name, a blockquote summary, and lists of Markdown links, optionally organized under heading sections. The goal is to give LLMs a navigable index without having to parse the site's full HTML structure.

A growing number of libraries now publish llms.txt indexes. DocuLayer maintains a shortlist of packages with confirmed llms.txt files: `anthropic`, `astro`, `fastapi`, `httpx`, `langchain`, `nextjs`, `openai`, `pydantic`, `react`, `shadcn`, `supabase`, `svelte`, `tailwindcss`, `vite`, `vue`.

For a query like `"dependency injection"` against `fastapi`, the flow is:

1. Fetch `https://fastapi.tiangolo.com/llms.txt` -- 88 indexed entries
2. Keyword-score each entry title and description against `"dependency injection"`
3. Take top 3 candidate URLs -- probably `tutorial/dependencies/`, `advanced/dependencies/`, and something adjacent
4. Fetch those three pages, parse into sections
5. BM25 rank sections across all three fetched pages, return top 5

This is why DocuLayer can be significantly cheaper than naively fetching the whole docs site. A large documentation site might have 200 pages and several megabytes of HTML. A targeted three-page fetch for a specific query is maybe 50–100 KB. The llms.txt index acts as a pre-filter before any HTTP is made.

For packages without llms.txt, the fallback is to parse the root HTML page of the docs site, extract links, and try to identify doc pages by URL pattern. This is more brittle but covers the common case where the root page has a top-level nav.

## BM25 Over Live Fetched Content

*BM25* (Best Match 25), specifically **BM25Okapi** ([Robertson & Zaragoza, 2009](https://doi.org/10.1561/1500000019)), is a probabilistic term-weighting retrieval function. Given a query $q$ and a corpus of documents $D$, BM25 scores each document $d$ as:

$$\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

where $f(t, d)$ is term frequency in document $d$, $|d|$ is document length, $\text{avgdl}$ is the average document length in the corpus, and $k_1$, $b$ are free parameters controlling term frequency saturation and length normalization (typical values: $k_1 = 1.5$, $b = 0.75$).

The IDF term weights against terms that appear in many documents:

$$\text{IDF}(t) = \log \frac{N - n(t) + 0.5}{n(t) + 0.5}$$

where $N$ is the total number of documents and $n(t)$ is the number containing term $t$.

DocuLayer's corpus at query time is not the entire index -- it's just the sections parsed from the 1–3 freshly fetched pages. This is a small corpus, typically 20–80 sections. BM25 runs over that and finishes in under a millisecond. There's no embedding inference, no approximate nearest-neighbor index, no GPU.

The tradeoff versus dense retrieval is well-known: BM25 is exact term matching. It scores `AsyncClient` highly for a query containing `AsyncClient`, but it doesn't know that `AsyncHttpClient` from a different library is conceptually related. Semantic similarity -- the core strength of embedding-based retrieval -- is absent. For the documentation use case this is usually fine: if you're asking about `AsyncClient`, you know to say `AsyncClient`. The vocabulary alignment between query and documentation is tighter than, say, a conversational search over a general knowledge corpus.

There's also no cross-document IDF here: the corpus is rebuilt per query from freshly fetched pages, so IDF is calculated over sections from those pages only. This means a term that appears in every section of the three fetched pages (say, `httpx`) gets downweighted, which is actually correct behavior -- it's noise in this local corpus.

## MCP as the Agent Interface

**The Model Context Protocol** (Anthropic, 2024, [modelcontextprotocol.io](https://modelcontextprotocol.io)) is an open protocol for connecting LLMs to external tools and data sources. Clients (Claude Code, Cursor, Windsurf, VS Code with MCP support, Zed) speak a common stdio or HTTP+SSE protocol to MCP servers. The server exposes a list of tools with JSON Schema descriptions; the LLM client decides when to call them.

DocuLayer runs as an MCP server via `doculayer mcp`. After `doculayer setup` writes the IDE-specific JSON config and the IDE restarts, the four DocuLayer tools appear in the agent's tool palette. No API keys, no cloud service, no remote dependency -- it's a local subprocess communicating over stdio.

The setup command auto-detects the IDE:

```bash
doculayer setup           # auto-detect
doculayer setup --ide cursor
doculayer setup --ide vscode
doculayer setup --all-ides
```

It writes to the appropriate config file -- `~/.cursor/mcp.json` for Cursor, `~/.codeium/windsurf/mcp_config.json` for Windsurf, `~/.config/zed/settings.json` for Zed. The generated config is the same in all cases:

```json
{
  "mcpServers": {
    "doculayer": {
      "command": "doculayer",
      "args": ["mcp"]
    }
  }
}
```

The simplicity here is deliberate. There's no authentication, no server to configure, no environment variables required (all optional). From `pip install doculayer` to functional MCP tools is two commands and a restart.

## The Four Tools

| Tool | Signature | What it does |
|---|---|---|
| `doculayer_search` | `(query, source, max_results=5)` | BM25 search across live fetched sections |
| `doculayer_fetch` | `(source, section=None)` | Fetch a whole page or a named heading |
| `doculayer_symbol` | `(symbol, source=None)` | Look up a function, class, or method |
| `doculayer_sources` | `()` | List known sources, identifier formats, cache stats |

Every response from all four tools includes an attribution block:

```
> **Source**: https://docs.pydantic.dev/latest/concepts/validators/
> **Fetched**: 4s ago
```

The `source` argument accepts multiple identifier formats:

| Format | Example | Resolves via |
|---|---|---|
| bare name | `fastapi` | shortcut table → PyPI → npm |
| `pypi:` prefix | `pypi:httpx` | PyPI JSON API |
| `npm:` prefix | `npm:react` | npm registry |
| `gh:` prefix | `gh:owner/repo` | GitHub URL construction |
| direct URL | `https://docs.example.com` | passthrough |

`doculayer_symbol` is worth calling out specifically. Given `symbol="AsyncClient"` and `source="httpx"`, it constructs a targeted BM25 query, fetches the likely page, and returns the relevant section. Source can also be inferred from dotted notation: `symbol="httpx.AsyncClient"` doesn't need `source=` specified.

`doculayer_sources()` returns live cache statistics -- which URLs are cached, when they expire, how many sections are indexed. This is useful for debugging and for understanding what the agent is drawing from.

## DocuLayer vs. RAG: What You Actually Trade

The README includes a comparison table. It's accurate but worth expanding on what the tradeoffs actually feel like in practice.

|  | RAG | DocuLayer |
|---|---|---|
| Storage | Vector DB required | None -- in-memory TTL only |
| Freshness | Depends on indexing schedule | Always live (TTL-bounded) |
| Accuracy | Semantic similarity | Verbatim text from source |
| Setup | Embedding model + DB + ingestion pipeline | `pip install doculayer` |
| Hallucination risk | Embedding drift, chunking artifacts | Zero -- no generated text |

**On freshness**: DocuLayer wins cleanly. A one-hour TTL means the agent reads docs that are at most an hour old. RAG freshness depends on how often you re-index -- daily is common, weekly is common, and "we haven't touched the index in three months" is also common.

**On semantic coverage**: RAG wins. If you ask about "rate limiting" and the documentation says "request throttling," BM25 misses. Embedding-based retrieval handles this. For programming documentation specifically, this gap is smaller than in general search -- the vocabulary in docs tends to match the vocabulary in developer queries closely -- but it's not zero.

**On accuracy**: DocuLayer returns verbatim text, so there's no paraphrasing drift or chunk boundary artifacts. RAG chunks have a well-known problem: a paragraph that reads cleanly in context loses meaning when extracted as a 512-token chunk that starts mid-thought. Heading-based splitting (which DocuLayer uses) tends to produce more semantically coherent sections.

**On infrastructure**: DocuLayer wins for personal and small-team use. Running a vector database for documentation retrieval is real overhead that most developers don't want. The tradeoff shows at scale: for a large team sharing a doc index across hundreds of agents, a persistent shared vector store starts making sense.

**On latency**: DocuLayer adds HTTP fetch time on a cache miss. First fetch for a given URL is typically 1–3 seconds. Subsequent queries hit the cache and are essentially free at query time. RAG retrieval from a local vector store is under 100ms at most scales, with no per-query network dependency. For interactive agent use (where you're waiting a few seconds for a response anyway), DocuLayer's latency profile is fine. For high-throughput batch processing, the difference matters.

**On zero hallucination**: This is the most interesting design property. DocuLayer returns no generated text whatsoever. Every byte in a response came from the fetched URL. The agent can still hallucinate when it *reads* that content and answers a question -- that's a different problem -- but DocuLayer can't inject incorrect information by paraphrasing. The content is either right or the fetch got the wrong page.

## Challenges and Open Problems

**Cache cold starts.** On first query after process restart, every URL needs a live fetch. For a multi-step agentic task that touches five different libraries, that's five sequential or parallel HTTP requests before any answering happens. For a long-running service this amortizes quickly; for a tool launched fresh with each conversation, the cold start adds meaningful latency.

**Packages without llms.txt.** The fallback to parsing root HTML and inferring doc structure by URL pattern works poorly for documentation sites with unusual layouts, JavaScript-heavy navigation, or nested subdomains (e.g., `api.docs.example.com` vs. `docs.example.com`). Coverage here is best-effort. The llms.txt ecosystem is growing -- there were around 5,000 sites publishing it as of late 2025 -- but most packages still don't have it.

**BM25 vocabulary mismatch.** As noted, exact term matching misses synonyms. A developer asking about "connection pooling" might not know that httpx calls it "connection limits." DocuLayer returns nothing useful for that query even though the documentation exists. A lightweight hybrid approach -- BM25 plus a few rounds of embedding re-ranking over the top candidates -- would close most of this gap without requiring a full embedding pipeline on every query.

**Section granularity.** Heading-based splits produce variable-length sections. Some headings in technical documentation introduce a single sentence and then delegate to subsections. Others introduce 2,000-word API references. BM25 length normalization helps, but a section that's 50 words versus one that's 2,000 words is a structural mismatch. Adaptive chunking -- splitting long sections while preserving heading context -- would improve recall on dense API documentation pages.

**No versioned docs.** `doculayer_search("validators", "pydantic")` fetches from wherever the pydantic shortcut resolves -- likely the current stable docs. If the project is pinned to pydantic 1.x, the returned documentation is wrong in ways the agent can't detect. Identifier formats like `pypi:pydantic==1.10` that resolve to version-specific docs would be a straightforward extension.

**Read-only.** DocuLayer is purely retrieval. It has no mechanism for agents to contribute back -- to flag a doc section as stale, to request that a source be indexed, or to annotate retrieved content. For individual developer use this is fine. For a shared team deployment, the ability to curate the source list and annotate known issues would add real value.

Despite these limitations, the core design philosophy is sound: for the "agent confidently uses stale API" failure mode, the direct fix is to give the agent live documentation access with zero hallucination risk. DocuLayer does that, in under 500 lines of Python, with a two-command install. That's a useful thing to have.

## Further reading

- [DocuLayer on GitHub](https://github.com/inamdarmihir/doculayer)
- [The /llms.txt file standard (Howard, 2024)](https://llmstxt.org)
- [The Probabilistic Relevance Framework: BM25 and Beyond (Robertson & Zaragoza, 2009)](https://doi.org/10.1561/1500000019)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [Okapi BM25 -- Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
