---
title: "Building a Custom ChatGPT Connector: From Actions to the Apps SDK"
date: 2026-07-30
description: "OpenAI's connector story has gone through three names in two years — Plugins, Actions, Connectors, now Apps — and converged, as of the Apps SDK, on the same Model Context Protocol that Claude uses. This post covers the current build-and-publish path, what actually differs from building for Claude, and a Qdrant-backed telemetry layer for spotting when the same backend gets used differently across platforms."
tags: ["agents", "chatgpt", "openai", "connectors", "mcp", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

If you built a ChatGPT integration in 2023 and haven't touched it since, almost nothing about the current path matches what you remember. OpenAI's connector story renamed itself three times — Plugins (2023), Actions for GPTs (2024), Connectors (2025) — and as of December 17, 2025, the term of art is **Apps**, built against the **Apps SDK**, which is the recommended way to package and publish app experiences including apps that use MCP-backed tools. The app directory itself migrated again on July 9, 2026, into what's now called the Plugin directory, with plugins as the primary discovery surface across both ChatGPT and Codex while "apps" refers specifically to the integrations that connect ChatGPT to external data and actions. This is not academic trivia — if you're searching for documentation and landing on an "Actions" or "Plugins" tutorial from 2024, you are reading about a deprecated surface.

The genuinely useful news buried in that naming churn: OpenAI converged on MCP as the underlying protocol for connector-backed apps, the same protocol Claude uses (covered in the [companion post on building an MCP server](/posts/building-mcp-server-claude-connector/)). That means the *server* you build is largely reusable across both ecosystems — what differs is packaging, discovery, auth specifics, and UI. This post covers that delta directly, using the same Qdrant-backed search connector as the running example, and does not re-derive MCP protocol mechanics already covered in the Claude-focused post.

## Table of Contents

1. [A Brief and Necessary History](#a-brief-and-necessary-history)
2. [The Current Model: Apps SDK and MCP](#the-current-model-apps-sdk-and-mcp)
3. [Developer Mode and the Publishing Path](#developer-mode-and-the-publishing-path)
4. [Reusing the Qdrant MCP Server for ChatGPT](#reusing-the-qdrant-mcp-server-for-chatgpt)
5. [Implementing a Widget: Rendering Results Inline](#implementing-a-widget-rendering-results-inline)
6. [What Actually Differs from Claude's Connector Model](#what-actually-differs-from-claudes-connector-model)
7. [Migrating a Legacy GPT Action to the Apps SDK](#migrating-a-legacy-gpt-action-to-the-apps-sdk)
8. [A Qdrant-Backed Cross-Platform Usage Telemetry Layer](#a-qdrant-backed-cross-platform-usage-telemetry-layer)
9. [Challenges and Open Problems](#challenges-and-open-problems)
10. [References](#references)

## A Brief and Necessary History

**Plugins (2023)** were the first attempt — a manifest-driven model where ChatGPT called your API directly based on an OpenAPI spec, discovered via a `.well-known/ai-plugin.json` file. **Actions for GPTs (2024)** folded the same underlying mechanism into the custom-GPT builder, letting a GPT author attach an OpenAPI-defined action without the separate plugin store. **Connectors (2025)** introduced OpenAI-maintained MCP wrappers for common services (Google Workspace, Dropbox) as first-party integrations distinct from developer-built ones. **Apps**, formalized December 17, 2025, is the current umbrella term for developer-built integrations, built on the **Apps SDK**, which explicitly supports MCP-backed tools as its primary building block rather than raw OpenAPI specs. If your mental model of "how do I integrate with ChatGPT" still starts with an OpenAPI manifest, that's the 2023–2024 model; the current path starts with an MCP server.

## The Current Model: Apps SDK and MCP

The practical shape: you build an MCP server — the same primitives covered in the Claude post (tools, resources, prompts, JSON-RPC over Streamable HTTP for anything remote) — and the Apps SDK is the layer that packages that server as a ChatGPT-installable app, adds OpenAI-specific manifest metadata, and (where relevant) lets you attach interactive UI components ("widgets" in Apps SDK terminology) that render inline in a ChatGPT conversation rather than returning plain text or JSON. Custom MCP connectors reachable through **Developer Mode** are currently in beta across Plus, Pro, Business, Enterprise, and Edu tiers on the web — meaning the audience for a connector you publish today is gated by which of those tiers your target users are on, not universal across all ChatGPT users yet.

## Developer Mode and the Publishing Path

Two distinct paths exist depending on your audience:

**Private, org-internal deployment.** Admins and authorized developers (Enterprise/Edu only) can upload and test MCP apps privately in Developer Mode without any public listing — the right path for an internal tool (say, a company-specific Qdrant knowledge base) that should never appear in a public directory at all.

**Public distribution via the Plugin directory.** As of the July 9, 2026 migration, public discovery happens through the Plugin directory rather than a separate "app store" — submission there is how a connector becomes discoverable to users who haven't been given a direct install link. Because this migration is recent as of this writing, expect the submission requirements and review turnaround to still be stabilizing; treat any specific procedural detail here as more likely to have shifted than the protocol-level facts above.

## Reusing the Qdrant MCP Server for ChatGPT

The tool-registration code from the Claude post — `search_documents(collection, query, limit)` backed by a Qdrant client — needs no logical changes to work against ChatGPT's Apps SDK, because both platforms are MCP clients talking to the same server shape. What changes is the wrapper metadata:

```python
# apps_sdk_manifest.py — OpenAI-specific packaging on top of the same MCP server
APP_MANIFEST = {
    "name": "qdrant-search-connector",
    "display_name": "Document Search",
    "description": "Semantic search over your organization's Qdrant-backed document collections.",
    "mcp_server_url": "https://connectors.yourcompany.com/mcp",
    "auth": {
        "type": "oauth2",
        "authorization_url": "https://auth.yourcompany.com/oauth/authorize",
        "token_url": "https://auth.yourcompany.com/oauth/token",
        "scopes": ["qdrant:search"],
    },
    # Widget registration is additive — omit entirely and the tool still
    # works, returning plain structured results rendered as text.
    "widgets": [
        {"tool": "search_documents", "component": "SearchResultsList"}
    ],
}
```

The underlying `search_documents` tool implementation is untouched. This is the actual payoff of both ecosystems standardizing on MCP: the platform-specific work is a metadata and packaging layer on top of a server you'd have built anyway, not a second parallel implementation of your search logic.

## Implementing a Widget: Rendering Results Inline

The manifest entry in the previous section declares that `search_documents` results should render through a `SearchResultsList` component rather than as plain text — the widget itself is a small, sandboxed frontend component the Apps SDK loads and feeds the tool's structured output:

```jsx
// SearchResultsList.jsx — registered against search_documents in the manifest
export default function SearchResultsList({ toolResult }) {
  const results = toolResult?.results ?? [];
  if (!results.length) {
    return <EmptyState message="No matching documents found." />;
  }
  return (
    <div className="search-results">
      {results.map((r) => (
        <ResultCard
          key={r.id}
          title={r.text.slice(0, 80)}
          score={r.score}
          onOpen={() => window.openAppsSdkLink(r.source_url)}
        />
      ))}
    </div>
  );
}
```

Two constraints shape what a widget can reasonably do, both worth designing around from the start rather than discovering mid-build: the component runs in a sandboxed context with no arbitrary network access of its own — any additional data it needs beyond what the tool call already returned has to come from the tool result itself, not a follow-up fetch the widget issues independently — and the component should degrade gracefully to the tool's plain structured output if a client doesn't support widget rendering at all (an older ChatGPT client, or a non-widget-aware surface). Treat the widget purely as a presentation layer over `search_documents`'s existing return value, not as a place to put logic the tool itself should own; a widget that silently reimplements query logic the tool already does creates two places that can disagree about what the "same" search returned.

## What Actually Differs from Claude's Connector Model

| Aspect | Claude | ChatGPT |
|---|---|---|
| Underlying protocol | MCP | MCP |
| Local/private packaging | Desktop Extension (`.mcpb`) | Developer Mode private upload (Enterprise/Edu) |
| Public discovery | Connector directory (MCP directory submission form) | Plugin directory (post July 9, 2026 migration) |
| Auth model | OAuth 2.1 resource server, RFC 8707 + 8693 | OAuth2, app-manifest-declared scopes |
| Rich UI in-conversation | MCP Apps extension | Apps SDK widgets |
| Terminology stability | Stable since MCP's introduction | Renamed three times since 2023 |

The terminology-stability row is not a throwaway line — it's the single biggest practical risk of building for the ChatGPT side specifically: documentation, SDK method names, and even the term for "the thing you're building" have changed on roughly an annual cadence. Isolate OpenAI-specific integration code behind a clean interface in your own codebase, the same discipline recommended for the still-draft OAuth extension in the [OAuth for agents](/posts/oauth-for-agents/) post, so a future rename costs you an adapter update rather than a rewrite.

## Migrating a Legacy GPT Action to the Apps SDK

A team that built a custom GPT Action against the 2024-era OpenAPI-manifest model faces a genuine migration, not a config tweak, because the two models express the integration differently at the root: an Action describes endpoints via an OpenAPI schema that ChatGPT calls more or less directly, while an Apps SDK connector describes tools via MCP's own primitive (name, description, JSON Schema for arguments) that an MCP server exposes. The practical migration path:

1. **Extract the OpenAPI operation's parameters into an MCP tool signature**, applying the naming and scoping discipline from the [tool schema design](/posts/tool-schema-design-mcp/) post along the way — an OpenAPI spec written for a REST client doesn't automatically produce a well-named, unambiguous tool for a model to call; this is the moment to fix parameter names like `id` or `type` that a REST-focused API design tolerated but a tool-calling model shouldn't have to guess about.
2. **Stand up the MCP server** (the same server built for Claude in the companion post is directly reusable here — that's the entire point of both platforms converging on MCP) rather than keeping the OpenAPI-direct-call path running in parallel indefinitely.
3. **Re-declare auth** in the Apps SDK manifest format rather than the OpenAPI security scheme the Action used — the OAuth flow itself may be unchanged on the authorization-server side, but the manifest that tells ChatGPT how to obtain and attach a token is a different artifact.
4. **Run both integration paths in parallel for a transition window**, if the legacy Action already has real users, rather than cutting over atomically — the same additive-then-deprecate discipline recommended for MCP tool versioning generally applies here, just at the platform-migration scale rather than the single-tool scale.

Teams that skip step 1 and mechanically wrap the existing OpenAPI operations as MCP tools without revisiting parameter naming tend to import every ambiguity the REST API had into the new integration, which defeats a real part of the reason to migrate in the first place — the point of the migration is not just "same functionality, new transport," it's an opportunity to fix tool-calling-specific design debt that an OpenAPI-first Action never had to address.

A rough effort budget for a mid-sized Action (6–10 operations, one auth flow) migrating through the four steps above, useful for setting expectations before committing a sprint to this:

| Step | Typical effort | Where teams underestimate |
|---|---|---|
| 1. Parameter extraction and renaming | 0.5–1 day per operation with real review | Skipping the naming pass to hit a deadline — this is the step most likely to get cut, and cutting it is exactly what reintroduces the ambiguity the migration was supposed to fix |
| 2. MCP server stand-up | 2–4 days if reusing an existing Claude-side server; 1–2 weeks from scratch | Assuming the Claude-side server needs zero changes — platform-specific rate limits and payload size limits sometimes force small adjustments |
| 3. Auth re-declaration | 1–3 days | Discovering the OAuth scopes granted to the old Action don't map cleanly onto the new manifest's scope model, requiring a re-consent flow for existing users |
| 4. Parallel-run transition window | 2–6 weeks calendar time (not engineering effort) | Setting the transition window based on engineering readiness rather than actual usage telemetry showing the old Action's call volume has genuinely dropped to near zero |

The pattern across all four rows: the effort that's easy to estimate (writing code) is rarely where migrations actually slip; the risk concentrates in the steps that involve waiting on evidence (usage telemetry) or coordination (re-consent) rather than pure implementation work.

## A Qdrant-Backed Cross-Platform Usage Telemetry Layer

Once the same MCP server serves both Claude and ChatGPT users, a genuinely useful side-piece is understanding whether the two populations use it the same way — different platforms attract different query patterns, different session lengths, different tolerance for ambiguous results. Embedding each query alongside its originating platform and outcome, stored in the same collection family used elsewhere on this blog for [semantic query caching](/posts/semantic-query-cache/), turns that into an answerable question rather than a guess:

```python
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams,
)


class CrossPlatformQueryLog:
    def __init__(self, client: QdrantClient, embed_fn, collection: str = "connector_queries"):
        self.client, self.embed_fn, self.collection = client, embed_fn, collection
        names = {c.name for c in client.get_collections().collections}
        if collection not in names:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def record(self, query: str, platform: str, result_count: int, took_ms: int):
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=str(uuid.uuid4()), vector=self.embed_fn(query),
                payload={"platform": platform, "result_count": result_count, "took_ms": took_ms},
            )],
        )

    def platform_divergence(self, sample_query: str, top_k: int = 50) -> dict:
        """For queries semantically similar to `sample_query`, compare result
        counts and latency across platforms — a cheap way to spot whether one
        platform is systematically getting worse results for the same intent."""
        hits = self.client.search(
            collection_name=self.collection, query_vector=self.embed_fn(sample_query), limit=top_k,
        )
        by_platform: dict[str, list] = {}
        for h in hits:
            by_platform.setdefault(h.payload["platform"], []).append(h.payload)
        return {
            platform: {
                "n": len(rows),
                "avg_results": sum(r["result_count"] for r in rows) / len(rows),
                "avg_ms": sum(r["took_ms"] for r in rows) / len(rows),
            }
            for platform, rows in by_platform.items()
        }
```

A divergence here is a real signal worth investigating — for instance, if ChatGPT-originated queries for a semantically similar intent consistently return fewer results than Claude-originated ones, that's evidence the query phrasing patterns differ enough between the two model families that your `search_documents` tool's query-normalization step (or the embedding call itself) needs platform-aware tuning, not evidence that one platform is inherently worse at using the tool.

## Challenges and Open Problems

**Naming and API churn is not over.** Three renames in roughly two years is a pattern, not a one-time transition — anything written about "the current OpenAI connector model," including this post, should be read with an expiry date in mind.

**Maintaining two platform integrations is real ongoing cost, even when the core server is shared.** Manifest formats, widget/UI component models, and auth particulars diverge and will keep diverging independently, since the two companies have no obligation to coordinate their packaging layers even where they agree on MCP as the wire protocol underneath.

**The Plugin directory migration is recent enough that process details are still settling.** Submission review criteria, turnaround time, and even long-term stability of the directory-vs-app-store distinction are the kind of thing likely to have shifted by the time you read this — verify against current OpenAI documentation rather than trusting the procedural specifics here to still be accurate.

**Feature parity between platforms is not guaranteed and probably shouldn't be assumed.** MCP Apps (Claude) and Apps SDK widgets (ChatGPT) both let a connector render rich UI, but the component models are not identical, and a widget built for one is not portable to the other without platform-specific adaptation — budget for that as separate frontend work, not a shared codebase.

## References

- OpenAI Help Center. *Developer mode and MCP apps in ChatGPT.* [help.openai.com/en/articles/12584461](https://help.openai.com/en/articles/12584461-developer-mode-apps-and-full-mcp-connectors-in-chatgpt-beta)
- OpenAI Help Center. *Apps in ChatGPT.* [help.openai.com/en/articles/11487775](https://help.openai.com/en/articles/11487775-connectors-in-chatgpt)
- OpenAI Developer Docs. *MCP and Connectors.* [developers.openai.com/api/docs/guides/tools-connectors-mcp](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)
- Eigent.ai. *OpenAI Workspace Agents in ChatGPT: Full Guide (2026).* [eigent.ai/blog/openai-workspace-agents-chatgpt](https://www.eigent.ai/blog/openai-workspace-agents-chatgpt)
- usecarly.com. *ChatGPT Connectors: The Complete List of App Integrations (2026).* [usecarly.com/blog/chatgpt-connectors](https://www.usecarly.com/blog/chatgpt-connectors/)
- Qdrant. *Qdrant Vector Database Documentation.* [qdrant.tech/documentation](https://qdrant.tech/documentation)
