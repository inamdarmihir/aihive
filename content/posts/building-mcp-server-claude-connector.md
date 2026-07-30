---
title: "Building an MCP Server From Scratch: A Working Connector for Claude"
date: 2026-07-30
description: "The mechanics of building a Model Context Protocol server end to end — transport choice, tool registration, the 2026-07-28 spec's stateless core and cacheable list results, packaging as a Desktop Extension vs a remote connector, and OAuth for multi-user access — using a Qdrant-backed search tool as the working example throughout."
tags: ["agents", "mcp", "claude", "connectors", "qdrant", "protocol"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Most MCP tutorials show you `@mcp.tool()` and a docstring and call it done. That gets a local stdio server running against Claude Desktop in ten minutes, and it teaches you almost nothing about the decisions that determine whether the same server survives contact with a second user, a remote deployment, or the protocol's own evolution — MCP shipped a new specification revision on 2026-07-28 that changes the performance and statefulness assumptions a server built six months ago was implicitly relying on. This post builds a real connector end to end: transport selection, tool registration, packaging for distribution through Claude's connector directory or as a Desktop Extension, and the OAuth 2.1 resource-server pattern a remote server needs to serve more than one user safely.

I use a Qdrant-backed document search tool as the running example throughout, because it's a realistic connector shape (a stateful backend, non-trivial auth scoping, structured query parameters) without being a toy `add(a, b)` demo. This post is about protocol mechanics, packaging, and auth — it is not about designing the search/retrieval logic itself, which the companion posts on [interpreting natural-language queries against Qdrant](/posts/qdrant-interpreter-agentic-rag/) and [validating tool responses](/posts/silent-failures-mcp-agents/) cover in depth.

## Table of Contents

1. [What MCP Actually Standardizes](#what-mcp-actually-standardizes)
2. [Choosing a Transport](#choosing-a-transport)
3. [A Minimal Server: Qdrant Search as an MCP Tool](#a-minimal-server-qdrant-search-as-an-mcp-tool)
4. [What Changed in the 2026-07-28 Specification](#what-changed-in-the-2026-07-28-specification)
5. [Packaging for Claude: Desktop Extension vs Remote Connector](#packaging-for-claude-desktop-extension-vs-remote-connector)
6. [Auth for a Multi-User Remote Server](#auth-for-a-multi-user-remote-server)
7. [Token Validation Middleware, Written Out](#token-validation-middleware-written-out)
8. [Testing with the MCP Inspector](#testing-with-the-mcp-inspector)
9. [Versioning a Tool Without Breaking Deployed Agents](#versioning-a-tool-without-breaking-deployed-agents)
10. [Challenges and Open Problems](#challenges-and-open-problems)
11. [References](#references)

## What MCP Actually Standardizes

The Model Context Protocol is JSON-RPC 2.0 over a transport, plus three primitives that give the JSON-RPC calls consistent meaning across every client and server that implements them: **tools** (model-invocable functions with a JSON Schema for their arguments), **resources** (addressable content a client can read, like files or URLs, without necessarily invoking a tool), and **prompts** (reusable prompt templates a server exposes for the client to surface). A connector, in the sense this blog and Anthropic's own documentation use the word, is almost always a **tools** server: you're giving Claude the ability to *do something* — search a collection, create a record, call an internal API — not just serve it static content.

What the protocol standardizes is the negotiation layer around those primitives: capability discovery (`initialize`, `tools/list`), invocation (`tools/call`), and the transport-level framing so that a Python server and a TypeScript client interoperate without either side knowing the other's implementation. What it does *not* standardize is your tool's internal logic, your auth backend, or your data model — all of that is yours to design, and most of the decisions in this post live in that space.

## Choosing a Transport

MCP servers run over one of two transports, and the choice determines almost everything downstream about packaging and auth:

**stdio** — the server is a local subprocess the client (Claude Desktop) spawns and talks to over stdin/stdout. No network exposure, no auth needed beyond OS-level process permissions, but the server only runs on the machine where Claude Desktop is installed and only for that one user. This is the shape Desktop Extensions package.

**Streamable HTTP** — the server is a long-running network service the client connects to over HTTP, with the ability to stream multiple messages per request. This is what a remote connector needs: reachable from claude.ai in a browser, usable by more than one person, deployable independently of any individual's machine. It's also the shape that requires OAuth, because an HTTP endpoint reachable from claude.ai is reachable from anywhere, and the server needs to know who's calling before it does anything backed by real data.

The Server-Sent Events (SSE) transport that earlier MCP tooling used for remote servers has been superseded by Streamable HTTP in current SDKs — if you're starting a new remote server today, build on Streamable HTTP directly rather than SSE, which is now the legacy path.

## A Minimal Server: Qdrant Search as an MCP Tool

Using the Python SDK, a tool is a decorated function; the SDK introspects the signature and docstring to build the JSON Schema the client sees during `tools/list`:

```python
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

mcp = FastMCP("qdrant-search-connector")
qdrant = QdrantClient(url="https://your-cluster.qdrant.io", api_key="...")


@mcp.tool()
def search_documents(collection: str, query: str, limit: int = 5) -> list[dict]:
    """Search a Qdrant collection by semantic similarity to the query text.

    Args:
        collection: name of the Qdrant collection to search (must be one the
            caller has access to — see auth section for scoping).
        query: natural-language search text.
        limit: maximum number of results to return (default 5, max 20).
    """
    limit = min(limit, 20)
    vector = embed(query)  # your embedding function
    hits = qdrant.search(collection_name=collection, query_vector=vector, limit=limit)
    return [{"score": h.score, "text": h.payload.get("text", ""), "id": h.id} for h in hits]


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Two details in this docstring matter more than the code: the parameter descriptions and the explicit `limit`/`max 20` cap. Anthropic's own guidance on writing tools for agents is blunt about this — tool descriptions are prompts, and small refinements to them have produced measurable jumps in benchmark performance (their write-up cites a SWE-bench Verified improvement from tool-description refinement alone). An unambiguously named parameter (`collection`, not `db` or `source`) and an explicit bound on `limit` do more for reliability than any amount of server-side validation logic, because they shape what the model decides to pass in the first place, before your code ever runs.

## What Changed in the 2026-07-28 Specification

If you're building today rather than porting an existing server, four changes in the current spec are worth designing around from the start rather than retrofitting later:

- **Stateless protocol core.** Earlier server implementations often kept per-session state implicitly (a connection object holding context between calls). The current spec pushes toward a stateless core, which means your tool handlers should not assume any server-side memory persists between one `tools/call` and the next unless you've explicitly built and wired up that persistence (a database, a cache, or — as in the failure-memory and calibration patterns elsewhere on this blog — a Qdrant collection keyed by session or user ID).
- **Cacheable list results.** `tools/list` responses can now be cached by the client rather than re-fetched every session, which matters for a server with a large or slowly-changing tool set: design your capability metadata to change only when the underlying tool set actually changes, not on every request, or clients may serve stale tool lists past your cache-invalidation window.
- **Authorization hardening.** Authorization servers are now expected to return the `iss` parameter per RFC 9207, and clients must validate it before redeeming a code — closing an authorization-server mix-up vulnerability class. If you're standing up your own auth server rather than delegating to an existing IdP, this is not optional.
- **Client ID Metadata Documents replacing Dynamic Client Registration.** The older DCR flow is deprecated in favor of Client ID Metadata Documents, and `application_type` is now set during registration so authorization servers stop incorrectly rejecting `localhost` redirects from desktop and CLI clients — a real, previously-annoying friction point for anyone testing a remote server against a local client.

The cacheable-list-results change is worth quantifying rather than taking on faith, because its payoff scales directly with how large and stable a server's tool set is. For the Qdrant search connector with a modest 8-tool surface, a rough per-session accounting:

| Scenario | `tools/list` calls per 20-turn session | Bytes transferred (≈600 bytes/tool schema) | Added round-trip latency |
|---|---|---|---|
| Pre-2026-07-28 (re-fetched every session) | 1 (at session start, typical client behavior) | ~4.8 KB | ~50–150ms once |
| Cacheable, cold client cache | 1 (first session only) | ~4.8 KB | ~50–150ms, one time |
| Cacheable, warm client cache | 0 | 0 | 0ms |

For an 8-tool server this is a rounding error either way — the real payoff shows up for a connector exposing dozens of tools across several domains (a shared internal platform team's aggregate MCP surface, say 60+ tools), where the same accounting turns a repeated ~35–40 KB, 100ms+ fetch into a one-time cost amortized across every subsequent session from a client that respects the cache. The design implication from earlier in this section — keep tool-set metadata stable across requests — is what makes that amortization actually materialize instead of silently reverting to pre-2026-07-28 behavior because every request looks like a fresh tool set to the client's cache logic.

## Packaging for Claude: Desktop Extension vs Remote Connector

Claude supports connectors as three distinct artifact types, and the packaging path differs meaningfully:

**Desktop Extensions (`.mcpb` bundles)** package a local, stdio-transport MCP server for one-click installation in Claude Desktop — no configuration file editing, no separate hosting. This is the right choice for a connector that only ever needs to run on the user's own machine with their own local credentials (a local file-search tool, a personal Qdrant instance running on localhost). Submission goes through the desktop extension submission form, distinct from the remote-server path.

**Remote MCP servers** are internet-hosted and reachable from Claude Desktop, Claude Code, and claude.ai in the browser — the right choice for a connector serving more than one user or backed by infrastructure you host (a shared Qdrant cluster, an internal API). These go through the MCP directory submission form and, if approved, appear in Claude's connector directory alongside official ("Made by Anthropic") and third-party connectors.

**MCP Apps** are a further extension for MCP servers that render interactive UI elements directly inside a Claude conversation, not just structured JSON, submitted via the same directory form with the MCP Apps extension declared.

For the Qdrant search example above, a remote server is almost always the right shape once more than one teammate needs access to the same collections — which immediately raises the auth question below.

## Auth for a Multi-User Remote Server

A remote MCP server is, per the current spec, an OAuth 2.1 **resource server**: it accepts and validates access tokens, and it delegates the actual authentication (who is this user, did they consent) to a separate authorization server rather than implementing login itself. The MCP authorization spec — developed jointly by Anthropic, Arcade.dev, Microsoft, and Okta/Auth0, among others — layers RFC 8707 (Resource Indicators, so a token is scoped to *this* server specifically, not usable against any resource server that trusts the same authorization server) and RFC 8693 (Token Exchange, for delegation semantics when an agent needs to act on a user's behalf without holding the user's raw credentials).

For the Qdrant connector, this resolves concretely into per-user collection scoping: the access token's claims determine which collections `search_documents` is allowed to query, checked inside the tool handler before the Qdrant call is issued — not left to Qdrant's own access control alone, since a shared API key gives every caller the same backend privileges regardless of what the token said:

```python
@mcp.tool()
def search_documents(collection: str, query: str, limit: int = 5, *, ctx) -> list[dict]:
    """..."""
    allowed = ctx.auth_claims.get("qdrant_collections", [])
    if collection not in allowed:
        raise PermissionError(f"token is not scoped for collection '{collection}'")
    ...
```

This is a thin sketch, not a complete auth implementation — a full remote server needs the authorization-server redirect flow, token validation middleware, and the `iss`-parameter check from the 2026-07-28 hardening above, all of which is genuinely substantial enough to be its own post (see the companion piece on [OAuth for agents](/posts/oauth-for-agents/) for the delegation and multi-user token-scoping problem in depth).

## Token Validation Middleware, Written Out

The permission-check sketch in the previous section assumes `ctx.auth_claims` already exists by the time a tool handler runs — that's the job of middleware sitting in front of every tool call, and it's worth writing out fully rather than waving at it, because most of the actual security-relevant work in a remote MCP server lives here, not in individual tool bodies:

```python
import time
import httpx
from jose import jwt, JWTError

JWKS_URL = "https://auth.yourcompany.com/.well-known/jwks.json"
EXPECTED_ISSUER = "https://auth.yourcompany.com"
EXPECTED_AUDIENCE = "mcp://qdrant-search-connector"  # RFC 8707 resource indicator

_jwks_cache = {"keys": None, "fetched_at": 0}


def _get_jwks():
    if _jwks_cache["keys"] is None or time.time() - _jwks_cache["fetched_at"] > 3600:
        resp = httpx.get(JWKS_URL, timeout=5)
        resp.raise_for_status()
        _jwks_cache["keys"] = resp.json()
        _jwks_cache["fetched_at"] = time.time()
    return _jwks_cache["keys"]


def validate_token(bearer_token: str) -> dict:
    try:
        claims = jwt.decode(
            bearer_token,
            _get_jwks(),
            audience=EXPECTED_AUDIENCE,
            issuer=EXPECTED_ISSUER,
            options={"verify_exp": True},
        )
    except JWTError as e:
        raise PermissionError(f"token validation failed: {e}")

    # 2026-07-28 spec hardening: confirm the authorization server that issued
    # this token is the one this client actually redirected to, per RFC 9207 —
    # without this check, a mix-up attack can present a validly-signed token
    # from a *different* authorization server the client also trusts.
    if claims.get("iss") != EXPECTED_ISSUER:
        raise PermissionError("issuer mismatch — possible authorization-server mix-up")

    return claims


async def auth_middleware(request, call_next):
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return unauthorized_response("missing bearer token")
    try:
        claims = validate_token(auth_header.removeprefix("Bearer "))
    except PermissionError as e:
        return unauthorized_response(str(e))
    request.state.auth_claims = claims
    return await call_next(request)
```

Two details here matter more than the JWT-decoding boilerplate. First, `audience=EXPECTED_AUDIENCE` is the RFC 8707 resource-indicator check in practice — it's what stops a token minted for some *other* MCP server, but signed by an authorization server this server also trusts, from being replayed here. Second, the explicit `iss` comparison after decode is the specific 2026-07-28 hardening measure: the JWT library's own audience/expiry checks don't by themselves protect against an authorization-server mix-up, so this needs to be an explicit line of code, not an assumption that the library's defaults cover it.

## Testing with the MCP Inspector

Before submitting anywhere, the MCP Inspector (the reference client bundled with the SDK tooling) connects to a running server over either transport, lists its declared tools and their schemas exactly as a real client would see them, and lets you invoke a tool manually with hand-constructed arguments — the fastest way to catch a malformed JSON Schema or a docstring that doesn't actually describe what the parameter needs, before Claude itself is the one discovering the mismatch mid-conversation.

## Versioning a Tool Without Breaking Deployed Agents

A tool's schema is a contract that, unlike a typical internal API, you cannot version by simply telling every caller to upgrade — a deployed Claude session or a cached `tools/list` result (a real thing, given the 2026-07-28 spec's cacheable-list-results change discussed above) may keep referencing the old schema for some window after you change it server-side. Renaming `search_documents`'s `limit` parameter to `max_results`, for instance, silently breaks any in-flight session that was relying on the previous name, because the model formed its plan to call `limit=5` against a schema that no longer accepts that argument.

The pattern that avoids this: additive changes only within a tool version, and a new tool name entirely for breaking changes, rather than mutating an existing tool's contract in place.

```python
@mcp.tool()
def search_documents(collection: str, query: str, limit: int = 5, max_results: int | None = None) -> list[dict]:
    """Search a Qdrant collection by semantic similarity.
    ... (unchanged existing docstring) ...

    Note: `max_results` is the preferred name for the result-count parameter
    and takes precedence if both are supplied; `limit` remains supported for
    backward compatibility with existing callers.
    """
    effective_limit = max_results if max_results is not None else limit
    effective_limit = min(effective_limit, 20)
    ...


@mcp.tool()
def search_documents_v2(collection: str, query: str, filters: dict | None = None, max_results: int = 5) -> list[dict]:
    """Search a Qdrant collection with structured filter support.

    This is the successor to search_documents — prefer this tool for new
    integrations. search_documents remains available unchanged for existing
    callers and is not deprecated on a fixed timeline.
    """
    ...
```

Deprecating the old tool outright (removing it from `tools/list`) should happen on a schedule communicated well in advance and validated against real usage telemetry showing the old tool has actually stopped being called — not on a fixed migration deadline picked without evidence that callers have moved off it, since a cached `tools/list` result on some client can mean "stopped being called" is harder to verify than it sounds.

## Challenges and Open Problems

**Statelessness pushes memory into your own infrastructure, and that infrastructure needs its own reliability story.** The 2026-07-28 spec's stateless core is a protocol-level guarantee, not a guarantee that your tool logic is stateless — if your connector genuinely needs memory across calls (a multi-turn search refinement, a running incident log), that memory now lives explicitly in a database or vector store you operate, with its own availability and consistency concerns that the protocol itself has no opinion about.

**Cacheable list results assume your tool set is genuinely stable.** If your server's available tools depend on the calling user's permissions (a natural fit for the auth-scoped Qdrant collections above — a user might only ever see the tools relevant to collections they can access), naive client-side caching of `tools/list` can serve one user a tool list computed for a different one, unless your server design accounts for per-identity cache keys rather than a single global list.

**The directory review process is a real gate, not a formality.** Submission to Claude's connector directory involves annotation requirements and a review pass distinct from "does it technically work" — budget for that as a separate step from development, not an afterthought after the code is done.

## References

- Model Context Protocol. *Specification.* [modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification/draft/basic/authorization)
- Model Context Protocol Blog. *The 2026-07-28 Specification.* [blog.modelcontextprotocol.io/posts/2026-07-28](https://blog.modelcontextprotocol.io/posts/2026-07-28/)
- Claude Help Center. *Build custom connectors via remote MCP servers.* [support.claude.com/en/articles/11503834](https://support.claude.com/en/articles/11503834-build-custom-connectors-via-remote-mcp-servers)
- Claude Help Center. *When to use desktop and web connectors.* [support.claude.com/en/articles/11725091](https://support.claude.com/en/articles/11725091-when-to-use-desktop-and-web-connectors)
- Claude.ai Documentation. *MCP, plugins, skills, and hooks.* [claude.com/docs/third-party/claude-desktop/extensions](https://claude.com/docs/third-party/claude-desktop/extensions)
- Anthropic. *Writing effective tools for AI agents — using AI agents.* [anthropic.com/engineering/writing-tools-for-agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Stack Overflow Blog. *Is that allowed? Authentication and authorization in Model Context Protocol.* [stackoverflow.blog/2026/01/21](https://stackoverflow.blog/2026/01/21/is-that-allowed-authentication-and-authorization-in-model-context-protocol/)
- Qdrant. *Qdrant Vector Database Documentation.* [qdrant.tech/documentation](https://qdrant.tech/documentation)
