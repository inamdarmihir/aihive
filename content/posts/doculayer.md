---
title: "Grounding Agents in Live Documentation: The DocuLayer Approach"
date: 2026-06-19
description: "How DocuLayer eliminates AI agent hallucination of stale API signatures by fetching live documentation on demand and running BM25 search locally -- no embeddings, no vector database, no generated text."
tags: ["agents", "mcp", "documentation", "bm25", "retrieval", "hallucination", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

One of the more reliable failure modes of AI coding agents is confidently wrong API usage. Not garbled syntax, not logical errors: just a function signature from six months ago, a parameter that was renamed, a method that didn't exist at training time. The model doesn't know it's wrong. Its training data said this is how `httpx.AsyncClient.get()` works, and nothing in the conversation contradicts that.

The standard mitigation is retrieval-augmented generation: build a vector index of documentation, embed queries, retrieve the nearest chunks, paste them into context. This works. It also requires a vector database, an embedding model, an ingestion pipeline, and some scheme for keeping the index fresh. That is a lot of infrastructure to solve what is fundamentally a staleness problem.

DocuLayer takes a more direct route. It sits between the agent and live documentation, fetches content on demand, runs BM25 search locally, and returns verbatim text with no embeddings, no disk storage, and no generated text. For teams running multi-agent systems at scale, there is a second tier available: Qdrant as a persistent documentation cache, so that fetched sections survive process restarts and are searchable via hybrid BM42 plus dense vectors. This post covers the full design, from the single-process live-fetch case to the distributed warm-cache architecture.

Pre-reads: familiarity with MCP (Model Context Protocol) and basic IR concepts (TF-IDF, BM25) will help.

## Table of Contents

1. [The Problem: Documentation Drift](#the-problem-documentation-drift)
2. [Architecture: No Database, No Embeddings](#architecture-no-database-no-embeddings)
3. [llms.txt as a Navigation Layer](#llmstxt-as-a-navigation-layer)
4. [BM25 Over Live Fetched Content](#bm25-over-live-fetched-content)
5. [MCP as the Agent Interface](#mcp-as-the-agent-interface)
6. [The Four Tools](#the-four-tools)
7. [DocuLayer vs. RAG: What You Actually Trade](#doculayer-vs-rag-what-you-actually-trade)
8. [Scaling with Qdrant: A Persistent Documentation Cache](#scaling-with-qdrant-a-persistent-documentation-cache)
9. [Challenges and Open Problems](#challenges-and-open-problems)
10. [References](#references)

## The Problem: Documentation Drift

Language models are trained on static snapshots of the web. The snapshot for a given model might be six months old, a year old, or older. Documentation for actively maintained libraries changes faster than that, sometimes dramatically. FastAPI added a dependency injection redesign. Pydantic v2 broke the v1 API in numerous places. React's hooks documentation has evolved continuously. `httpx` renamed parameters. OpenAI has deprecated and replaced entire API surfaces.

When an agent is asked about these libraries, it pulls from training data. If training data has the old version, it answers with confidence about behavior that no longer exists. The generated code compiles. It might even run. It just doesn't do what the documentation says it should.

The naive fix is to put the current docs in the system prompt. This works for small libraries or targeted tasks. For anything at scale, you are immediately fighting the context window. A full Next.js documentation site is tens of thousands of words; you cannot paste it in wholesale.

RAG solves the scale problem: embed the docs, retrieve the relevant chunks, insert only what is needed. But this reintroduces the freshness problem. Embeddings are of a snapshot, not the live docs. You need scheduled re-indexing, staleness detection, and delta pipelines.

DocuLayer's approach: don't index offline at all. Fetch on demand, cache in memory for a configurable TTL, and search the freshly fetched content locally.

## Architecture: No Database, No Embeddings

```
Your agent  (Claude, Cursor, Codex, any MCP client)
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
│  DocSearcher  (BM25, no ML, no embeddings, no network)      │
│                                                              │
│  TTLCache  (in-memory only, zero disk writes)               │
└──────────────────────────────────────────────────────────────┘
    │   verbatim section + source URL + "fetched 3s ago"
    ▼
LLM  reads real docs, answers correctly
```

The pipeline on every query:

1. **Resolve the identifier**: `"httpx"` goes through a shortcut table first; on a miss, falls back to PyPI JSON API or npm registry to find the real docs URL.
2. **Discover llms.txt**: probe the root of the docs site for `/llms.txt`; if present, parse the indexed entries.
3. **Select candidate pages**: keyword-score the llms.txt entries against the query, pick the top 1–3.
4. **Fetch and parse**: download those pages in parallel, convert HTML to Markdown, split by heading into `DocSection` objects.
5. **BM25 search**: run Okapi BM25 over all sections from the fetched pages, return the top-$k$ ranked by score.
6. **Return verbatim**: no rewriting, no summarizing; the raw section text goes back to the caller with a source attribution header.

The TTLCache holds `FetchResult` objects keyed by URL. Default TTL is one hour. On a cache hit, steps 4–5 run over cached content; steps 1–3 are still needed to determine which URLs to look up. On a miss, it fetches live. Nothing writes to disk. The cache is process-local, and when the process restarts, it is empty.

## llms.txt as a Navigation Layer

**The llms.txt standard** ([Howard, 2024](https://llmstxt.org)) proposes that documentation sites publish a Markdown file at `/llms.txt` containing structured links to machine-readable versions of their pages. The format is intentionally simple: an H1 name, a blockquote summary, and lists of Markdown links, optionally organized under heading sections. The goal is to give LLMs a navigable index without having to parse the site's full HTML structure.

A growing number of libraries now publish llms.txt indexes. DocuLayer maintains a shortlist of packages with confirmed llms.txt files: `anthropic`, `astro`, `fastapi`, `httpx`, `langchain`, `nextjs`, `openai`, `pydantic`, `react`, `shadcn`, `supabase`, `svelte`, `tailwindcss`, `vite`, `vue`.

For a query like `"dependency injection"` against `fastapi`, the flow is:

1. Fetch `https://fastapi.tiangolo.com/llms.txt` (88 indexed entries).
2. Keyword-score each entry title and description against `"dependency injection"`.
3. Take the top 3 candidate URLs, likely `tutorial/dependencies/`, `advanced/dependencies/`, and something adjacent.
4. Fetch those three pages, parse into sections.
5. BM25-rank sections across all three fetched pages, return top 5.

This is why DocuLayer can be significantly cheaper than naively fetching the whole docs site. A large documentation site might have 200 pages and several megabytes of HTML. A targeted three-page fetch for a specific query is maybe 50–100 KB. The llms.txt index acts as a pre-filter before any HTTP request is made.

For packages without llms.txt, the fallback is to parse the root HTML page of the docs site, extract links, and try to identify doc pages by URL pattern. This is more brittle but covers the common case where the root page has a top-level nav.

## BM25 Over Live Fetched Content

*BM25* (Best Match 25), specifically **BM25Okapi** ([Robertson & Zaragoza, 2009](https://doi.org/10.1561/1500000019)), is a probabilistic term-weighting retrieval function. Given a query $q$ and a corpus of documents $D$, BM25 scores each document $d$ as:

$$\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

where $f(t, d)$ is term frequency in document $d$, $|d|$ is document length, $\text{avgdl}$ is the average document length in the corpus, and $k_1$, $b$ are free parameters controlling term frequency saturation and length normalization (typical values: $k_1 = 1.5$, $b = 0.75$).

The IDF term weights against terms that appear in many documents:

$$\text{IDF}(t) = \log \frac{N - n(t) + 0.5}{n(t) + 0.5}$$

where $N$ is the total number of documents and $n(t)$ is the number containing term $t$.

DocuLayer's corpus at query time is not the entire index: it is just the sections parsed from the 1–3 freshly fetched pages. This is a small corpus, typically 20–80 sections. BM25 runs over that and finishes in under a millisecond. There is no embedding inference, no approximate nearest-neighbor index, no GPU.

The tradeoff versus dense retrieval is well-known. BM25 is exact term matching. It scores `AsyncClient` highly for a query containing `AsyncClient`, but it does not know that `AsyncHttpClient` from a different library is conceptually related. Semantic similarity, the core strength of embedding-based retrieval, is absent. For the documentation use case this is usually fine: if you are asking about `AsyncClient`, you know to say `AsyncClient`. The vocabulary alignment between query and documentation is tighter than in a conversational search over a general knowledge corpus.

There is also no cross-document IDF here. The corpus is rebuilt per query from freshly fetched pages, so IDF is calculated over sections from those pages only. A term that appears in every section of the three fetched pages (say, `httpx`) gets downweighted, which is correct behavior: it is noise in this local corpus.

## MCP as the Agent Interface

**The Model Context Protocol** (Anthropic, 2024, [modelcontextprotocol.io](https://modelcontextprotocol.io)) is an open protocol for connecting LLMs to external tools and data sources. Clients (Claude Code, Cursor, Windsurf, VS Code with MCP support, Zed) speak a common stdio or HTTP+SSE protocol to MCP servers. The server exposes a list of tools with JSON Schema descriptions; the LLM client decides when to call them.

DocuLayer runs as an MCP server via `doculayer mcp`. After `doculayer setup` writes the IDE-specific JSON config and the IDE restarts, the four DocuLayer tools appear in the agent's tool palette. No API keys, no cloud service, no remote dependency: it is a local subprocess communicating over stdio.

The setup command auto-detects the IDE:

```bash
doculayer setup           # auto-detect
doculayer setup --ide cursor
doculayer setup --ide vscode
doculayer setup --all-ides
```

It writes to the appropriate config file: `~/.cursor/mcp.json` for Cursor, `~/.codeium/windsurf/mcp_config.json` for Windsurf, `~/.config/zed/settings.json` for Zed. The generated config is the same in all cases:

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

The simplicity here is deliberate. There is no authentication, no server to configure, no environment variables required (all optional). From `pip install doculayer` to functional MCP tools is two commands and a restart.

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

`doculayer_symbol` is worth calling out specifically. Given `symbol="AsyncClient"` and `source="httpx"`, it constructs a targeted BM25 query, fetches the likely page, and returns the relevant section. Source can also be inferred from dotted notation: `symbol="httpx.AsyncClient"` does not need `source=` specified.

`doculayer_sources()` returns live cache statistics: which URLs are cached, when they expire, how many sections are indexed. This is useful for debugging and for understanding what the agent is drawing from.

## DocuLayer vs. RAG: What You Actually Trade

The README includes a comparison table. The tradeoffs are worth expanding on.

|  | RAG | DocuLayer (in-memory) | DocuLayer + Qdrant |
|---|---|---|---|
| Storage | Vector DB required | None, in-memory TTL only | Qdrant collection (persistent) |
| Freshness | Depends on indexing schedule | Always live (TTL-bounded) | Live fetch + TTL in payload |
| Accuracy | Semantic similarity | Verbatim text from source | Verbatim text, hybrid retrieval |
| Setup | Embedding model + DB + ingestion pipeline | `pip install doculayer` | DocuLayer + Qdrant instance |
| Hallucination risk | Embedding drift, chunking artifacts | Zero, no generated text | Zero, no generated text |
| Cold start | None (index preloaded) | Slow (live fetch per URL) | Warm cache hits, fast |
| Multi-agent sharing | Yes (shared DB) | No (process-local) | Yes (shared Qdrant) |

**On freshness**: DocuLayer wins cleanly over traditional RAG. A one-hour TTL means the agent reads docs that are at most an hour old. RAG freshness depends on re-indexing cadence; daily is common, weekly is common, and three months without a touch is also common.

**On semantic coverage**: RAG wins over pure BM25. If you ask about "rate limiting" and the documentation says "request throttling," BM25 misses. Embedding-based retrieval handles synonyms and paraphrase. For programming documentation the gap is smaller than in general search: vocabulary in docs tends to match developer queries closely. But the gap is not zero, and the Qdrant-backed tier (described below) closes most of it by adding dense vectors alongside the BM42 sparse index.

**On accuracy**: DocuLayer returns verbatim text, so there is no paraphrasing drift or chunk boundary artifacts. RAG chunks have a well-known problem: a paragraph that reads cleanly in context loses meaning when extracted as a 512-token chunk that starts mid-thought. Heading-based splitting, which DocuLayer uses, tends to produce more semantically coherent sections.

**On infrastructure**: DocuLayer wins for personal and small-team use. Running a vector database for documentation retrieval is real overhead. The tradeoff shows at scale: for a large team sharing a doc index across many agents, a persistent shared vector store starts making sense. That is exactly what the Qdrant tier provides.

**On latency**: DocuLayer adds HTTP fetch time on a cache miss. First fetch for a given URL is typically 1–3 seconds. Subsequent queries hit the cache and are essentially free at query time. RAG retrieval from a local vector store is under 100ms at most scales, with no per-query network dependency. For interactive agent use (where you are waiting a few seconds for a response anyway), DocuLayer's latency profile is fine. For high-throughput batch processing, the difference matters, and the Qdrant warm cache addresses it directly.

**On zero hallucination**: DocuLayer returns no generated text. Every byte in a response came from the fetched URL. The agent can still hallucinate when it reads that content and answers a question, but DocuLayer cannot inject incorrect information by paraphrasing. The content is either right or the fetch got the wrong page.

## Scaling with Qdrant: A Persistent Documentation Cache

The in-memory TTL cache in DocuLayer's base design is process-local and ephemeral. This is fine for a single developer running one IDE process. It becomes a bottleneck in two scenarios:

1. **Multi-agent deployments**: ten agent workers sharing no state each cold-start their own cache and make redundant HTTP fetches for the same documentation pages.
2. **High-throughput pipelines**: batch jobs processing thousands of queries need sub-second latency on doc lookups; live HTTP fetches on cache miss are unacceptable at that rate.

The solution is to add Qdrant as a *warm cache* between the agent and the live-fetch path. Fetched and parsed `DocSection` objects are indexed into a Qdrant collection. Subsequent queries check Qdrant first; a hit with a valid TTL avoids the live fetch entirely. A miss (or an expired hit) falls through to the existing DocuLayer live-fetch pipeline, and the fresh result is indexed back into Qdrant.

```
Agent query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  DocuLayer (query router)                                   │
│                                                             │
│  1. Hybrid search in Qdrant doc_cache                       │
│     (BM42 sparse + dense, filter: fetched_at > now - TTL)  │
│                                                             │
│  2a. HIT  → return verbatim section from payload           │
│  2b. MISS → live fetch via DocuLayer pipeline               │
│             → index result into doc_cache                   │
│             → return verbatim section                       │
└─────────────────────────────────────────────────────────────┘
```

Qdrant is the warm path (fast, persistent, shared across agents). DocuLayer live fetch is the cold path (always current, higher latency). The separation is clean: freshness is guaranteed by the TTL filter on `fetched_at`; accuracy is guaranteed by the verbatim-text payload.

### Collection Design

The Qdrant collection `doc_cache` stores one point per `DocSection`. The payload carries all metadata needed for TTL filtering and attribution. The vectors capture both keyword and semantic similarity for retrieval.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
)

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="doc_cache",
    vectors_config={
        "text-dense": VectorParams(
            size=384,           # e.g., all-MiniLM-L6-v2
            distance=Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "text-sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        )
    },
)
```

Each point has the following payload schema:

| Field | Type | Purpose |
|---|---|---|
| `url` | string | Canonical source URL of the documentation page |
| `section` | string | Heading text (e.g., `"AsyncClient.get"`) |
| `text` | string | Full verbatim section body |
| `source_id` | string | DocuLayer source identifier (e.g., `"httpx"`) |
| `version` | string | Package version if known, else `"latest"` |
| `fetched_at` | int | Unix timestamp of the live fetch |

The `fetched_at` field drives TTL invalidation (described in the next section). The `text` field stores the full section body verbatim, so a cache hit returns exactly the same content the live-fetch path would return.

The point ID is a stable hash of the URL concatenated with the section heading. This ensures that re-indexing a section (after a live re-fetch) is an upsert, not a duplicate insert.

### Hybrid Search with BM42

Qdrant supports named sparse vectors using the **BM42** weighting scheme ([Qdrant Labs, 2024](https://qdrant.tech/articles/bm42/)). BM42 is a reformulation of BM25 for token-level embeddings rather than raw term frequencies, producing sparse vectors that preserve keyword recall properties while fitting naturally into Qdrant's unified scoring framework. This is the right choice here because documentation retrieval is vocabulary-sensitive: `AsyncClient.get`, `dependency_overrides`, `model_validator` are exact identifiers that must match, not paraphrases of ideas.

The indexing step, called after a live fetch, encodes each `DocSection` into both a sparse BM42 vector and a dense semantic vector:

```python
from qdrant_client.models import PointStruct, SparseVector
import time

def index_section(
    client: QdrantClient,
    section: DocSection,
    dense_encoder,    # e.g. SentenceTransformer("all-MiniLM-L6-v2")
    sparse_encoder,   # e.g. FastEmbed BM42 model
    source_id: str,
    version: str = "latest",
) -> None:
    dense_vec = dense_encoder.encode(section.text).tolist()
    sparse_result = sparse_encoder.embed(section.text)
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    client.upsert(
        collection_name="doc_cache",
        points=[
            PointStruct(
                id=section.id,       # stable hash of url + section heading
                payload={
                    "url": section.url,
                    "section": section.heading,
                    "text": section.text,
                    "source_id": source_id,
                    "version": version,
                    "fetched_at": int(time.time()),
                },
                vector={
                    "text-dense": dense_vec,
                    "text-sparse": sparse_vec,
                },
            )
        ],
    )
```

Retrieval uses Qdrant's hybrid search with prefetch and RRF (Reciprocal Rank Fusion) to merge sparse and dense result lists:

```python
from qdrant_client.models import (
    Prefetch,
    FusionQuery,
    Filter,
    FieldCondition,
    Range,
    SparseVector,
)

def search_doc_cache(
    client: QdrantClient,
    query: str,
    dense_encoder,
    sparse_encoder,
    ttl_seconds: int = 3600,
    top_k: int = 5,
) -> list[dict]:
    freshness_threshold = int(time.time()) - ttl_seconds
    ttl_filter = Filter(
        must=[
            FieldCondition(
                key="fetched_at",
                range=Range(gte=freshness_threshold),
            )
        ]
    )

    dense_vec = dense_encoder.encode(query).tolist()
    sparse_result = sparse_encoder.embed(query)
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    results = client.query_points(
        collection_name="doc_cache",
        prefetch=[
            Prefetch(
                query=sparse_vec,
                using="text-sparse",
                filter=ttl_filter,
                limit=20,
            ),
            Prefetch(
                query=dense_vec,
                using="text-dense",
                filter=ttl_filter,
                limit=20,
            ),
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k,
        with_payload=True,
    )
    return [hit.payload for hit in results.points]
```

The `ttl_filter` is a payload range filter on `fetched_at`. Points older than `ttl_seconds` are excluded from both the sparse and dense prefetch legs before RRF fusion runs. This is not a hard delete: the point remains in the collection and is reachable without the filter. The benefit is that you can tune the TTL per query (tighter for volatile docs, looser for stable APIs) without modifying the index.

### TTL Invalidation via Payload Filtering

Rather than scheduling periodic deletes (which add operational complexity), the design uses payload filtering to simulate TTL at query time. When `search_doc_cache` returns zero results for a given query, the caller knows the warm cache either has no matching section or has only stale matches. Either way, it falls through to the live-fetch cold path, and the fresh result is upserted back into `doc_cache` with a new `fetched_at` timestamp.

This upsert-on-miss pattern means the collection grows over time. A background cleanup job can prune genuinely old points periodically without affecting serving latency. Qdrant's `delete` API accepts a filter directly:

```python
from qdrant_client.models import FilterSelector

stale_threshold = int(time.time()) - 7 * 24 * 3600  # older than 7 days

client.delete(
    collection_name="doc_cache",
    points_selector=FilterSelector(
        filter=Filter(
            must=[
                FieldCondition(
                    key="fetched_at",
                    range=Range(lt=stale_threshold),
                )
            ]
        )
    ),
)
```

Running this daily keeps the collection size bounded. The payload filter at query time already handles freshness independently; the cleanup job is purely a storage hygiene operation.

### Why Qdrant for This Workload

Several properties make Qdrant the right choice for the doc cache role specifically.

**Named vectors** allow the same point to carry both sparse BM42 and dense representations without duplicating payload. The section text, URL, heading, and timestamp are stored once; the two vectors attach to the same point ID. This matters for the upsert-on-miss pattern: a single `upsert` call writes or overwrites both vector representations atomically.

**Payload filtering inside HNSW** is a first-class primitive in Qdrant's indexing, not a post-hoc filter applied over ANN results. The `fetched_at` range filter runs inside the graph traversal. For the documentation cache workload, where a large fraction of points in the collection may be stale (TTL expired), this matters significantly: without filter-aware indexing, ANN search would return mostly stale candidates that are then discarded, wasting traversal budget. Qdrant avoids this by maintaining a filterable HNSW index.

**BM42 sparse vectors** (via Qdrant's FastEmbed integration) are the correct sparse representation for identifier-heavy technical documentation. Classic BM25 as computed by a library like `rank-bm25` requires building the IDF table over a fixed corpus; adding a new document requires recomputing IDF or approximating. BM42 sparse vectors are computed per-document and stored independently in the inverted index Qdrant maintains internally, so new sections are indexed incrementally without rebuilding any global table.

**On-disk indexing** for the sparse vector index (configured via `on_disk=True` in `SparseIndexParams`) lets the collection scale beyond available RAM without degrading recall, using Qdrant's memory-mapped file access for the sparse index structure. For a small team's deployment that fits in RAM, `on_disk=False` gives faster search; for a large shared deployment covering many libraries and versions, on-disk keeps the memory footprint bounded.

### The Full Warm/Cold Routing Flow

The complete routing logic for a production DocuLayer deployment with Qdrant:

```python
def doc_query(
    query: str,
    source: str,
    qdrant_client: QdrantClient,
    doculayer_client,           # existing DocuLayer live-fetch client
    dense_encoder,
    sparse_encoder,
    ttl_seconds: int = 3600,
    top_k: int = 5,
) -> list[DocSection]:
    # 1. Warm path: check Qdrant
    cached = search_doc_cache(
        qdrant_client, query, dense_encoder, sparse_encoder,
        ttl_seconds=ttl_seconds, top_k=top_k,
    )
    if cached:
        return [DocSection.from_payload(p) for p in cached]

    # 2. Cold path: live fetch via DocuLayer
    fresh_sections = doculayer_client.search(query=query, source=source)

    # 3. Index fresh results into Qdrant for future warm hits
    for section in fresh_sections:
        index_section(
            qdrant_client, section, dense_encoder, sparse_encoder,
            source_id=source,
        )

    return fresh_sections
```

The agent's tool call hits the warm path on virtually every query after the first fetch for a given library. Qdrant's hybrid search (BM42 sparse + dense, RRF fusion) handles both exact-identifier queries (`AsyncClient.get`) and concept queries ("how do I configure connection timeouts") better than BM25 alone. The live-fetch cold path remains the source of truth for freshness, and the `fetched_at` payload field ensures the warm cache is bounded by the configured TTL.

## Challenges and Open Problems

**Cache cold starts.** On first query after process restart, every URL needs a live fetch. For a multi-step agentic task that touches five different libraries, that is five sequential or parallel HTTP requests before any answering happens. For a long-running service this amortizes quickly; for a tool launched fresh with each conversation, the cold start adds meaningful latency. The Qdrant tier solves this for shared deployments: the warm cache persists across process restarts, and the first agent to fetch a page populates the cache for all subsequent agents.

**Packages without llms.txt.** The fallback to parsing root HTML and inferring doc structure by URL pattern works poorly for documentation sites with unusual layouts, JavaScript-heavy navigation, or nested subdomains (for example, `api.docs.example.com` vs. `docs.example.com`). Coverage here is best-effort. The llms.txt ecosystem is growing (around 5,000 sites publishing it as of late 2025), but most packages still don't have it.

**BM25 vocabulary mismatch.** Exact term matching misses synonyms. A developer asking about "connection pooling" might not know that httpx calls it "connection limits." DocuLayer returns nothing useful for that query even though the documentation exists. The Qdrant hybrid tier partially addresses this: the dense vector leg handles paraphrase and synonym matching that BM25 cannot. For the single-process in-memory case, a lightweight re-ranking step over BM25 candidates using a small embedding model would close most of the gap without requiring a full embedding pipeline on every query.

**Section granularity.** Heading-based splits produce variable-length sections. Some headings in technical documentation introduce a single sentence and then delegate to subsections. Others introduce 2,000-word API references. BM25 length normalization helps, but a 50-word section versus a 2,000-word section is a structural mismatch. Adaptive chunking, splitting long sections while preserving heading context, would improve recall on dense API documentation pages. This applies equally to the Qdrant-indexed sections, since the `text` payload field stores whatever the heading-based splitter produced.

**No versioned docs.** `doculayer_search("validators", "pydantic")` fetches from wherever the pydantic shortcut resolves: likely the current stable docs. If the project is pinned to pydantic 1.x, the returned documentation is wrong in ways the agent cannot detect. Identifier formats like `pypi:pydantic==1.10` that resolve to version-specific docs would be a straightforward extension. In the Qdrant tier, the `version` payload field is already in the schema; routing by version just requires propagating the pinned version from the query context through the `source_id` filter.

**Read-only interface.** DocuLayer is purely retrieval. It has no mechanism for agents to contribute back: to flag a doc section as stale, to request that a source be indexed, or to annotate retrieved content. For individual developer use this is fine. For a shared team deployment with the Qdrant backend, the ability to curate the source list and annotate known issues would add real value. A write path through the MCP interface, writing annotations as Qdrant payload updates against an existing point, is a natural extension of the current design.

**Qdrant index memory footprint.** For large documentation corpora covering many libraries and versions, the `doc_cache` collection grows significantly. Sparse BM42 vectors for technical documentation can be high-dimensional, and storing them for thousands of sections requires careful configuration of Qdrant's sparse index. On-disk indexing mitigates RAM pressure at the cost of I/O overhead on search. The right operating point depends on deployment scale: a small team's Qdrant instance can likely keep the full sparse index in memory; a large shared deployment may need on-disk sparse vectors with an explicit hot-set policy for the most frequently accessed libraries.

The core design philosophy is sound: for the "agent confidently uses stale API" failure mode, the direct fix is to give the agent live documentation access with zero hallucination risk. DocuLayer does that, in under 500 lines of Python, with a two-command install. The Qdrant layer extends this to multi-agent deployments without sacrificing the freshness guarantee or the verbatim-text accuracy that makes DocuLayer's output trustworthy.

## References

- [DocuLayer on GitHub](https://github.com/inamdarmihir/doculayer)
- Howard, J. (2024). [The /llms.txt file standard](https://llmstxt.org). llmstxt.org.
- Robertson, S. & Zaragoza, H. (2009). [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019). *Foundations and Trends in Information Retrieval*, 3(4), 333–389.
- Anthropic. (2024). [Model Context Protocol](https://modelcontextprotocol.io). modelcontextprotocol.io.
- Qdrant. (2024). [BM42: New Baseline for Hybrid Search](https://qdrant.tech/articles/bm42/). qdrant.tech.
- Qdrant. [Collections API](https://qdrant.tech/documentation/concepts/collections/). qdrant.tech.
- Wikipedia contributors. [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25). Wikipedia.
