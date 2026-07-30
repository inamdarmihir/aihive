---
title: "OAuth for Agents: The Authentication Problem Nobody Solved Cleanly"
date: 2026-07-30
description: "Standard OAuth models a two-party relationship: a client acting for itself. An agent acting on a user's behalf, calling a chain of connectors that each call further tools, is a three-plus-party delegation problem OAuth 2.1 alone doesn't express. This post covers RFC 8693 token exchange, the draft On-Behalf-Of extension for AI agents, how MCP adopted both, and a Qdrant-backed anomaly detector for delegation chains that drift from a user's historical scope."
tags: ["agents", "oauth", "authentication", "mcp", "security", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Every developer who has connected an agent to a real user's real accounts has hit the same wall: OAuth as commonly implemented assumes the entity requesting access *is* the entity that will use it. A user authorizes an app, the app gets a token, the app calls an API with that token. An agent breaks this cleanly at the second step — the user authorizes an *agent*, but the agent then delegates to a chain of tools, sub-agents, and connectors, each of which may need to act with some subset of the user's authority, and none of which is the original authorized party. Standard OAuth 2.1 has no native vocabulary for "this token was issued to a user but is being presented by an agent acting on their behalf, which is itself calling a downstream connector that needs to know that." That gap is what this post is about.

I focus on the delegation and scoping problem specifically — not general OAuth setup, not session-cookie auth for a single-user local tool, both of which are well-covered ground. The running example is a connector-calling agent (the shape from the [MCP server](/posts/building-mcp-server-claude-connector/) post) that needs to act across multiple downstream services on a user's behalf without ever holding that user's raw password or a single all-powerful token.

## Table of Contents

1. [Why Standard OAuth Doesn't Fit](#why-standard-oauth-doesnt-fit)
2. [RFC 8693: Token Exchange and the `act` Claim](#rfc-8693-token-exchange-and-the-act-claim)
3. [The Draft Extension: On-Behalf-Of Authorization for AI Agents](#the-draft-extension-on-behalf-of-authorization-for-ai-agents)
4. [How MCP Adopted This](#how-mcp-adopted-this)
5. [A Practical Scoping Pattern](#a-practical-scoping-pattern)
6. [Token Storage and Rotation for Long-Running Agents](#token-storage-and-rotation-for-long-running-agents)
7. [A Qdrant-Backed Delegation Anomaly Detector](#a-qdrant-backed-delegation-anomaly-detector)
8. [A Worked Example: Three-Hop Delegation](#a-worked-example-three-hop-delegation)
9. [Challenges and Open Problems](#challenges-and-open-problems)
10. [References](#references)

## Why Standard OAuth Doesn't Fit

Classic OAuth 2.1 has two parties beyond the authorization server: the resource owner (the user) and the client (the app the user authorized). A token issued in this model answers one question: *is this client allowed to act as this user, within this scope?* An agent architecture adds a party the spec never modeled — the agent itself is not the resource owner, and it is often not a single client either, because a single user request can fan out into an agent calling three separate MCP connectors, each of which is a distinct OAuth client with its own authorization server relationship. A token that says "this client can act as this user" doesn't say which *agent*, acting through which client, is actually presenting it right now, nor does it express that the agent's authority should be narrower than the user's own (an agent reading a calendar should not silently inherit the user's ability to delete calendar entries just because the user's own token could).

## RFC 8693: Token Exchange and the `act` Claim

RFC 8693 defines an HTTP- and JSON-based Security Token Service protocol: a client presents a `subject_token` (representing the party the action is *about*) and, for delegation, an `actor_token` (representing the party *doing* the acting), and receives back a new token that encodes both via an `act` claim. Critically, `act` claims nest — a token can express "actor A, acting on behalf of actor B, acting on behalf of the original resource owner," which is exactly the shape of an agent that itself invokes a sub-agent or a chain of connectors, each hop adding one more link to a verifiable delegation chain rather than silently collapsing into "some request came in with a valid-looking token."

This matters operationally: a downstream resource server can inspect the full `act` chain and make an authorization decision based on *who is actually in the loop*, not just whether the presented token validates. A connector that would grant access to a token acting directly for the user might reasonably deny the same scope if the chain shows three agent hops with no human confirmation step in between.

## The Draft Extension: On-Behalf-Of Authorization for AI Agents

RFC 8693 alone is "primarily designed for server-side communication or impersonation scenarios" and doesn't fully address the *initial* authorization step specific to AI agents — how a user grants an agent delegated authority in the first place, distinct from an app requesting its own scope. An active IETF draft, *OAuth 2.0 Extension: On-Behalf-Of User Authorization for AI Agents*, extends the authorization framework specifically for this: it introduces a `requested_actor` parameter in the authorization request (identifying which specific agent is asking for delegated authority) and an `actor_token` parameter in the token request (authenticating that agent during the exchange of an authorization code for an access token). The practical effect is that a user's consent screen can say "Agent X wants to act on your behalf for scope Y" rather than the generic "App X wants access to Y" — and the resulting token carries the agent's identity as a first-class, verifiable claim rather than something inferred from which API key happened to make the call.

This is a **draft**, not a ratified standard — worth stating plainly, because building against a moving IETF draft means budgeting for the field names or flow details to shift before it stabilizes.

## How MCP Adopted This

MCP's own authorization spec, built with input from Anthropic, Arcade.dev, Microsoft, and Okta/Auth0, composes the pieces above rather than inventing new ones: a protected MCP server is an OAuth 2.1 **resource server**, audience-bound via **RFC 8707 Resource Indicators** (so a token issued for one MCP server can't be replayed against a different one that happens to trust the same authorization server), with delegation expressed via **RFC 8693 Token Exchange**. The 2026-07-28 spec revision hardens the authorization-server relationship further: authorization servers must return the `iss` parameter per **RFC 9207**, and clients must validate it before redeeming a code — closing a mix-up attack where a malicious authorization server could otherwise trick a client into redeeming a code meant for a different, legitimate authorization server. None of this is MCP-specific cryptography; it's MCP adopting the existing delegation and audience-binding machinery the wider identity ecosystem already built, rather than reinventing agent auth from scratch.

## A Practical Scoping Pattern

The scoping decision that actually matters in production is narrower than "does this token work" — it's "what is the *minimum* the agent needs for *this specific action*, right now." A pattern that holds up:

- **Per-connector, per-session tokens**, not one long-lived token reused across every tool call in a conversation. A token minted for a single MCP session, scoped to the collections/resources that session's task actually touches, bounds the blast radius of a leaked or misused token to that session's declared scope.
- **Short-lived access tokens with refresh, not long-lived tokens.** An agent that runs for hours (a long research task, a multi-step workflow) should be refreshing frequently against a refresh token the user can revoke, not holding a single access token that remains valid for the task's full duration regardless of what happens mid-task.
- **Revocation that actually propagates.** Revoking a user's consent should invalidate not just the top-level agent token but the full `act` chain beneath it — a delegation model that lets a revoked user's agent keep operating through already-issued downstream tokens defeats the point of having a chain at all.

The blast-radius argument for short-lived, narrowly-scoped tokens is easier to internalize with numbers attached. Consider a research agent compromised mid-session — via a prompt injection, a dependency vulnerability, or a leaked log line — and compare exposure under three token policies:

| Token policy | Token lifetime | Scope | Exposure window if compromised | What the attacker gets |
|---|---|---|---|---|
| Single standing token (anti-pattern) | Indefinite, until manually rotated | Full user account | Until someone notices and manually revokes — often days | Everything the user could do, indefinitely |
| Session-scoped, long-lived | Duration of one agent run (hours) | Task-relevant resources only | Remainder of the session, up to hours | Bounded to the session's declared scope, but for a long window |
| Session-scoped, short-lived + refresh | Minutes, refreshed via broker | Task-relevant resources only | Until the current access token expires — minutes | Bounded to scope *and* to a short window; broker-side revocation closes it immediately |

The middle row is what most teams building their first agent-to-connector integration actually ship — scoped correctly, but with a token lifetime chosen for engineering convenience (fewer refresh calls to write) rather than security. The gap between "hours" and "minutes" of exposure window is the entire value of the token-broker pattern in the previous section, and it's a gap that doesn't show up in a functional test — a compromised-token scenario only gets exercised in an incident, which is exactly when the difference between minutes and hours of standing access matters most.

## Token Storage and Rotation for Long-Running Agents

A research or workflow agent that runs for hours needs its refresh token available at the moment it needs to refresh, without holding that token in the same process memory space as the model's own context — a compromised prompt (see the [prompt injection](/posts/prompt-injection-browser-agents/) post's lethal-trifecta framing) should not be able to read out a refresh token just because it shares an address space with the agent loop that uses it.

The pattern that separates these concerns: a dedicated token-broker service holds refresh tokens in encrypted storage, and the agent process only ever holds a short-lived access token it exchanges for a fresh one on expiry, via a call to the broker rather than direct access to the refresh token itself.

```python
import time
from dataclasses import dataclass


@dataclass
class ScopedSession:
    access_token: str
    expires_at: float
    session_id: str


class TokenBroker:
    """Runs as a separate service/process from the agent loop. Holds refresh
    tokens; never exposes them directly to a calling agent process."""

    def __init__(self, encrypted_store, oauth_client):
        self.store, self.oauth_client = encrypted_store, oauth_client

    def mint_session(self, user_id: str, requested_scope: str, session_id: str) -> ScopedSession:
        refresh_token = self.store.get_refresh_token(user_id)
        token_response = self.oauth_client.refresh(
            refresh_token=refresh_token, scope=requested_scope,
        )
        return ScopedSession(
            access_token=token_response["access_token"],
            expires_at=time.time() + token_response["expires_in"],
            session_id=session_id,
        )

    def refresh_if_needed(self, session: ScopedSession, user_id: str, requested_scope: str) -> ScopedSession:
        if time.time() < session.expires_at - 30:  # 30s safety margin before actual expiry
            return session
        return self.mint_session(user_id, requested_scope, session.session_id)
```

The agent loop calls `refresh_if_needed` before each round of tool calls and only ever holds the resulting short-lived `access_token` — the refresh token itself never enters the agent's process, which bounds what a successful prompt injection or a compromised agent process can actually exfiltrate to "one access token, valid for minutes, scoped to one session" rather than "a long-lived refresh token that can mint new access tokens indefinitely." Revoking a compromised session, correspondingly, is a broker-side operation (invalidate that `session_id`'s refresh-token association) rather than something that requires locating and killing whatever process happened to be holding the token when the compromise was detected.

## A Qdrant-Backed Delegation Anomaly Detector

Individual token validation (signature, expiry, audience) is necessary and already handled by standard OAuth libraries; it says nothing about whether a *particular* delegation chain, even a cryptographically valid one, looks like this user's normal usage. A lightweight side-layer: embed a compact representation of each request's delegation shape — the actor chain, the requested scope, the resource being accessed — and flag chains that sit far from a user's historical pattern, the same anomaly-detection shape used elsewhere on this blog for [tool-response anomalies](/posts/mcp-response-guard/):

```python
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams,
)


class DelegationAnomalyDetector:
    def __init__(self, client: QdrantClient, embed_fn, collection: str = "delegation_history"):
        self.client, self.embed_fn, self.collection = client, embed_fn, collection
        names = {c.name for c in client.get_collections().collections}
        if collection not in names:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    def _signature(self, user_id: str, act_chain: list[str], scope: str, resource: str) -> str:
        return f"user:{user_id} chain:{' -> '.join(act_chain)} scope:{scope} resource:{resource}"

    def record(self, user_id: str, act_chain: list[str], scope: str, resource: str):
        text = self._signature(user_id, act_chain, scope, resource)
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=str(uuid.uuid4()), vector=self.embed_fn(text),
                payload={"user_id": user_id, "scope": scope, "resource": resource, "raw": text},
            )],
        )

    def anomaly_score(self, user_id: str, act_chain: list[str], scope: str, resource: str,
                       top_k: int = 20) -> float:
        text = self._signature(user_id, act_chain, scope, resource)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=self.embed_fn(text),
            query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
            limit=top_k,
        )
        if not hits:
            return 1.0  # no history for this user — treat first delegation as maximally novel
        # Mean similarity to the k most similar past requests; low mean → anomalous shape.
        mean_sim = sum(h.score for h in hits) / len(hits)
        return max(0.0, 1.0 - mean_sim)
```

A high anomaly score doesn't block the request by itself — it's a signal to route into the human-confirmation layer (the same pattern used for state-changing browser-agent actions elsewhere on this blog) rather than an automatic denial, because a genuinely new but legitimate delegation chain (a user connecting a new tool for the first time) should look anomalous on this metric without being wrong.

## A Worked Example: Three-Hop Delegation

A user authorizes a research agent to act on their behalf (`requested_actor=research-agent`, per the draft extension). The research agent calls an MCP-based Qdrant connector to search internal documents (hop one), which in turn needs to call a separate summarization connector to condense a long result (hop two). The token presented to the summarization connector carries an `act` chain: `summarizer ← qdrant-connector ← research-agent ← user`. The summarization connector's resource server can now decide, from the chain alone, whether to honor the request at the scope requested — for instance, denying write access even if the top-level user token nominally had it, because nothing in the chain indicates the user explicitly authorized a *write* action three hops removed from their own session. Without RFC 8693's `act` claim, the summarization connector would see only "some agent, somewhere, presented a token that validates" and have no structural basis for that distinction.

## Challenges and Open Problems

**The AI-agent OAuth extension is still a draft.** Building against `draft-oauth-ai-agents-on-behalf-of-user-02` today means the parameter names and flow details described here may change before ratification — worth isolating behind an abstraction in your own codebase rather than hardcoding the current draft's field names throughout.

**Consent fatigue is a real UX cost of doing this correctly.** Per-connector, per-session, minimally-scoped tokens are the technically correct answer and also mean a user may face more distinct consent prompts than the "authorize once, trust forever" pattern most non-agentic OAuth apps train users to expect. Whether product design can make fine-grained consent feel lightweight rather than like nagging is an open UX problem, not just a protocol one.

**Revocation propagation across a multi-hop chain is harder in practice than the RFC 8693 model implies on paper.** If hop two's downstream token was already issued and cached before the user revoked consent at the top level, whether that token actually stops working depends on whether every resource server in the chain checks against a live revocation list rather than trusting a token until its (possibly long) expiry — a detail easy to get wrong under the pressure of "just make the demo work."

**Anomaly detection on delegation shape is a heuristic, not a security boundary.** The Qdrant-backed detector above is a triage signal for routing to human review, not a replacement for correct scoping and the RFC 8693/8707 machinery — treating it as a substitute for narrow, principled scopes would be the same mistake as treating a prompt-injection screener as a substitute for privilege minimization.

## References

- IETF Datatracker. *OAuth 2.0 Extension: On-Behalf-Of User Authorization for AI Agents.* [datatracker.ietf.org/doc/html/draft-oauth-ai-agents-on-behalf-of-user-02](https://datatracker.ietf.org/doc/html/draft-oauth-ai-agents-on-behalf-of-user-02)
- RFC Editor. *RFC 8693: OAuth 2.0 Token Exchange.* [rfc-editor.org/info/rfc8693](https://www.rfc-editor.org/info/rfc8693/)
- ZITADEL Docs. *OAuth 2.0 Token Exchange (RFC 8693): Impersonation & Delegation.* [zitadel.com/docs/guides/integrate/token-exchange](https://zitadel.com/docs/guides/integrate/token-exchange)
- Model Context Protocol. *Authorization Specification.* [modelcontextprotocol.io/specification/draft/basic/authorization](https://modelcontextprotocol.io/specification/draft/basic/authorization)
- Model Context Protocol Blog. *The 2026-07-28 Specification.* [blog.modelcontextprotocol.io/posts/2026-07-28](https://blog.modelcontextprotocol.io/posts/2026-07-28/)
- Descope. *Diving Into the MCP Authorization Specification.* [descope.com/blog/post/mcp-auth-spec](https://www.descope.com/blog/post/mcp-auth-spec)
- DEV Community / Arcade. *How to manage multi-user AI agent authentication and authorization in 2026.* [dev.to/arcade](https://dev.to/arcade/how-to-manage-multi-user-ai-agent-authentication-and-authorization-in-2026-oauth-21-oidc-and-2943)
- Qdrant. *Qdrant Vector Database Documentation.* [qdrant.tech/documentation](https://qdrant.tech/documentation)
