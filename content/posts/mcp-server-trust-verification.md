---
title: "Before You Connect: Verifying MCP Server Identity Against Lookalikes"
date: 2026-07-16
description: "MCP server discovery has moved from a colleague handing you a URL to searchable public directories that agents browse on their own. A server calling itself 'gdrive-search' and describing itself almost identically to the trusted 'gdrive-connector' should not get the same casual click-to-connect treatment. This post covers a Qdrant-backed trust-verification layer that runs before any OAuth flow starts."
tags: ["agents", "mcp", "security", "supply-chain", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Discovering an MCP server used to mean a colleague pasted you a URL, or you read the README of a repo you already trusted for other reasons. In 2026 it increasingly means searching a directory — **Claude's** connector directory, and, since a July 9, 2026 migration, **OpenAI's** Plugin directory serving both ChatGPT and Codex — and clicking the top result, or letting an agent do the searching and connecting on your behalf without a human ever looking at the URL at all. That shift changes the trust model completely. A hand-off URL from a colleague carries an implicit vouch. A search result carries none: the only thing distinguishing a legitimate server from a lookalike is a self-declared name, a self-declared description, and a self-declared tool list, none of which the protocol certifies as accurate.

This post is about that specific, narrow decision point — whether to trust and connect to a server you've just discovered — and a Qdrant-backed verification layer that sits client-side, before any authorization flow begins. It is explicitly not about OAuth scoping (which governs what an already-trusted server's token can do) and not about tool-schema design (which governs how a tool you built describes itself to a model). Those are covered well elsewhere; this post covers the moment before either of them applies, which is also the moment where the least tooling currently exists to help. Familiarity with MCP's basic client/server/tool model and with vector similarity search is assumed.

## Table of Contents

1. [The Decision Point: Trusting a Server You've Just Discovered](#the-decision-point-trusting-a-server-youve-just-discovered)
2. [Why This Is a Different Problem](#why-this-is-a-different-problem)
3. [The Ecosystem Reality: A Young, Unvetted Long Tail](#the-ecosystem-reality-a-young-unvetted-long-tail)
4. [Design: An MCP Server Trust-Verification Layer](#design-an-mcp-server-trust-verification-layer)
5. [Component One: The Trusted Corpus](#component-one-the-trusted-corpus)
6. [Component Two: The Lookalike-Identity Check](#component-two-the-lookalike-identity-check)
7. [Component Three: The Tool-Set Overlap Check](#component-three-the-tool-set-overlap-check)
8. [Putting It Together: The Pre-Connection Gate](#putting-it-together-the-pre-connection-gate)
9. [Worked Example: gdrive-search-mcp vs. gdrive-searchmcp](#worked-example-gdrive-search-mcp-vs-gdrive-searchmcp)
10. [Composing With OAuth Scoping](#composing-with-oauth-scoping)
11. [What Directories Could Do to Make This Layer Unnecessary](#what-directories-could-do-to-make-this-layer-unnecessary)
12. [Challenges and Open Problems](#challenges-and-open-problems)
13. [References](#references)

## The Decision Point: Trusting a Server You've Just Discovered

An agent — or a user, through an agent's UI — searches a directory for "search my Google Drive" or receives a recommendation from the directory's own ranking. A handful of results come back, each with a name, a short description, and a declared list of tools. At that exact moment, before any connection attempt, before any OAuth consent screen, before any tool has actually been called, a decision gets made: connect to this one, or not.

Nothing in the MCP protocol itself certifies that a server calling itself `gdrive-search-mcp` is the server your organization vetted six months ago, is affiliated with Google, or is meaningfully different in trustworthiness from a server that appeared in the directory last week under an almost identical name. The protocol specifies how a client and server negotiate capabilities once connected — it says nothing about who should get connected to in the first place. That gap is filled today, in practice, by whatever attention a human happens to pay to a directory listing before clicking connect, which for an agent auto-discovering servers on a user's behalf is often no attention at all.

This is worth stating plainly because it's easy to conflate with problems that are already reasonably well handled: MCP's authorization model (built on OAuth 2.1, RFC 8693 token exchange, and RFC 8707 resource indicators) is a serious answer to "what can this token do." It says nothing about "should this token have been issued to this server at all," which is a question that has to be answered *before* the authorization flow the token belongs to ever starts.

## Why This Is a Different Problem

It's worth being explicit about the boundary between this and two adjacent, better-covered problems, because all three get grouped under "MCP security" loosely enough to blur together.

**OAuth and token scoping** governs an *already-connected* relationship: given that an agent is going to talk to this server, what should the resulting token let it do — read-only, write-scoped, time-limited, revocable. This machinery does real work, but it assumes the "which server" decision has already been made correctly. A perfectly scoped token handed to an impersonating server is still a token handed to the wrong party; narrow scope limits the damage, it doesn't prevent the mistake.

**Tool-schema design** governs how a tool *you* built describes itself to a model — clear naming, unambiguous parameters, avoiding an overly broad `manage_record(action, ...)` surface that lets a model do more than any single call should. This is about the quality and safety of a schema authored by someone you already trust (yourself, or your own team), not about deciding whether to trust an unfamiliar author's schema in the first place.

**Trust verification**, the subject of this post, is upstream of both: it's the check that runs at discovery time, before a single tool call or OAuth redirect happens, to decide whether the server claiming to be `gdrive-search-mcp` is the one you — or your organization — already vetted, or a plausible-looking stranger wearing the same name.

The ordering matters because each layer's mitigations assume the layer above it did its job correctly. A well-designed OAuth scoping policy correctly limits what a malicious server *can* do with the token it receives — but it can't stop that token from being issued to the wrong server in the first place, because by the time scoping logic runs, the "which server" decision is already made. Fixing the upstream decision is strictly more valuable per unit of engineering effort than tightening the downstream one further, precisely because a mistake at the trust-verification layer propagates through every layer built on top of it, while a mistake at any one downstream layer is, at least in principle, contained by the layers still functioning correctly around it.

## The Ecosystem Reality: A Young, Unvetted Long Tail

The scale of the underlying risk is not speculative. **Endor Labs'** *State of Dependency Management 2025* report analyzed 10,663 MCP server repositories on GitHub and found the ecosystem to be, structurally, exactly the kind of environment where impersonation thrives: 75% of MCP servers are built by individual developers, often without enterprise-grade safeguards; 41% carry no license information at all, which limits even the most basic provenance and accountability signal a downstream user could check; and 82% use sensitive APIs that require careful security controls, meaning the majority of servers in this long tail aren't low-stakes toys — they're integrations with real access to real data, built by parties with no institutional vetting process behind them.

None of this means most MCP servers are malicious — the report itself notes that the *most popular* servers skew toward organizational maintainers (GitHub, Microsoft, and similar names account for a disproportionate share of the top 20). The risk is specifically in the long tail: a young ecosystem with no strong, scaled "verified publisher" signal comparable to what package registries and app stores have spent years building, growing fast enough that discovery increasingly happens by search rather than by referral. That combination — high growth, thin provenance, search-driven discovery — is precisely the environment where typosquatting and slopsquatting thrive in package registries, and MCP server directories are a newer, less mature version of the same trust surface, with the added wrinkle that an agent, not just a human, can be the one doing the searching and connecting.

The directories themselves are still consolidating. **Claude's** connector directory has existed for a while as the more mature of the two; **OpenAI's** side went through Plugins (2023), Actions (2024), and Connectors (2025) before a July 9, 2026 migration folded app discovery into a unified Plugin directory that now serves both ChatGPT and Codex from a single catalog. Consolidation is good for discoverability and bad, in the near term, for trust signal maturity — a newly merged catalog inherits whatever provenance metadata each predecessor system happened to collect, which for both ecosystems today is closer to "self-declared publisher name" than to a verified identity check.

### The Familiar Analogy, in a Newer Trust Surface

Package registries went through this exact progression years earlier, and the pattern is instructive precisely because it's already well understood. Typosquatting on npm and PyPI — publishing `reqeusts` or `python-dateutil-` hoping for a stray keystroke — is old news; registries responded with automated near-name detection, verified-publisher badges, and download-count-weighted search ranking that makes an established package hard to displace by name alone. The more recent variant, **slopsquatting**, exploits a newer weakness: coding agents that hallucinate plausible-but-nonexistent package names, which an attacker can then squat on so the *next* agent that hallucinates the same name gets a real, malicious package instead of an install error. Both variants share a structure: a young or automatable discovery mechanism, combined with a namespace that's cheap to register into, produces an incentive to look like something trusted rather than to build something trustworthy.

MCP server directories reproduce the same structure with none of the mitigations package registries eventually built. There is no scaled, cross-directory verified-publisher signal comparable to what npm or PyPI now offer. There is no established download-count or reputation-weighted ranking mature enough to reliably push an impersonator below the legitimate result. And, in the added wrinkle specific to agentic systems, the entity doing the "typing" that might produce a near-name collision is sometimes an agent's own search query or an LLM's paraphrase of what it's looking for, not a human's keystroke — which means the slopsquatting dynamic (a name an *agent* is statistically likely to guess or search for, rather than one a human is likely to mistype) is arguably a more natural fit for MCP discovery than for package installation, where a human is still usually the one typing the command.

## Design: An MCP Server Trust-Verification Layer

The layer this post develops sits client-side — inside the agent framework itself, or as a thin proxy in front of every connection attempt — and runs exactly once, at discovery time, before any OAuth authorization request is sent to the candidate server.

```
┌──────────────────────────────────────────────────────────────┐
│  Agent / user searches directory, gets candidate server(s)    │
│    │                                                           │
│    ▼                                                           │
│  Trust-Verification Gate                                       │
│    1. embed candidate's declared identity (name+description   │
│       +tool names)                                             │
│    2. query_points against trusted_servers  ────────────┐      │
│    3. embed candidate's declared tool list separately     │      │
│    4. query_points against trusted_servers (toolset)  ───┤      │
│    5. combine signals → TrustVerdict                       │      │
│    │                                                       │      │
│    ├── clear: proceed to OAuth authorization request       │      │
│    └── flagged: block, surface for explicit human approval  │      │
└──────────────────────────────────────────────────────────────┘
          │                                        ▲
          ▼                                        │
   OAuth / connection layer               Qdrant: trusted_servers
   (unaffected — still applies             (org-approved corpus +
    narrow scoping as usual)                optional public "verified" tier)
```

Two things are worth calling out about this placement. First, it runs *before* the OAuth flow, not instead of it — a server that clears trust verification still only gets whatever scope the connection logic decides to request, unchanged from today's practice. Second, it's advisory by default for the identity check and only hard-blocking in the narrowest case (an exact name collision with a different, unverified publisher) — most flags should route to an explicit human confirmation step rather than a silent auto-deny, because the base rate of "genuinely new and legitimate server that happens to look unfamiliar" is much higher than the base rate of "actual impersonation attempt," at least until an ecosystem's trusted corpus matures.

## Component One: The Trusted Corpus

The corpus is a Qdrant collection of previously-vetted, known-legitimate servers — seeded from an organization's own explicitly-approved server list, and optionally supplemented with whatever a directory's own "verified" tier exposes, where one exists. Each entry stores two separate embeddings, because the two checks in this design (identity similarity and tool-set similarity) are deliberately independent signals with different failure modes.

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

COLLECTION = "trusted_servers"
EMBED_DIM = 384  # e.g. all-MiniLM-L6-v2 or an equivalent small embedding model


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "identity": VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            "toolset": VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        },
    )
    for field_name, schema in [
        ("server_name", PayloadSchemaType.KEYWORD),
        ("publisher", PayloadSchemaType.KEYWORD),
        ("approval_source", PayloadSchemaType.KEYWORD),
        ("approved_at", PayloadSchemaType.FLOAT),
    ]:
        client.create_payload_index(
            collection_name=COLLECTION, field_name=field_name, field_schema=schema,
        )
```

The `identity` named vector embeds the server's self-presented identity — its name, description, and the names of its declared tools concatenated as a short summary — the text a directory listing shows a human before they decide to click connect. The `toolset` named vector embeds only the tool names and descriptions, deliberately excluding the server's own name and marketing description, so that a server trying to hide behind an innocuous name can still be caught on the basis of what it actually claims to do.

```python
def identity_text(server_name: str, description: str, tool_names: list[str]) -> str:
    return f"{server_name}: {description} tools: {', '.join(sorted(tool_names))}"


def toolset_text(declared_tools: list[dict]) -> str:
    """declared_tools: [{'name': ..., 'description': ...}, ...]"""
    return " | ".join(
        f"{t['name']}: {t['description']}" for t in sorted(declared_tools, key=lambda t: t["name"])
    )


def add_trusted_server(
    client: QdrantClient,
    embed_fn,
    server_name: str,
    description: str,
    publisher: str,
    declared_tools: list[dict],
    approval_source: str,   # e.g. "org_admin_review" | "directory_verified_tier"
    approved_at: float,
) -> None:
    import uuid
    tool_names = [t["name"] for t in declared_tools]
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "identity": embed_fn(identity_text(server_name, description, tool_names)),
                "toolset": embed_fn(toolset_text(declared_tools)),
            },
            payload={
                "server_name": server_name,
                "publisher": publisher,
                "approved_at": approved_at,
                "approval_source": approval_source,
                "declared_tools": tool_names,
            },
        )],
    )
```

`approval_source` matters for the same reason it matters in any provenance system: an entry seeded from an org's own hands-on admin review carries different weight than one pulled automatically from a directory's self-reported "verified" badge, and downstream logic (or a human reviewing a flag) should be able to see which kind of approval backs a given trusted entry rather than treating all trust sources as equivalent.

### Bootstrapping and Maintaining the Corpus

A corpus is only as useful as its coverage of the servers an organization's agents actually try to connect to, which means seeding it once at rollout and never revisiting it defeats the purpose almost as thoroughly as never building it. A practical bootstrap pulls from whatever list of already-approved integrations an organization maintains today — often a spreadsheet or an internal wiki page nobody has turned into structured data yet — and a lightweight batch job keeps it current as new approvals happen:

```python
def bootstrap_from_approved_list(
    client: QdrantClient,
    embed_fn,
    approved_servers: list[dict],   # each: name, description, publisher, tools, approved_at
) -> None:
    for entry in approved_servers:
        add_trusted_server(
            client, embed_fn,
            server_name=entry["name"],
            description=entry["description"],
            publisher=entry["publisher"],
            declared_tools=entry["tools"],
            approval_source="org_admin_review",
            approved_at=entry["approved_at"],
        )


def sync_directory_verified_tier(
    client: QdrantClient,
    embed_fn,
    directory_client,   # thin wrapper around whichever directory's listing API
) -> int:
    """Pull any publisher-verified listings a directory exposes and add them
    at a distinct approval_source, so they're never conflated with an org's
    own hands-on review when a flag is later evaluated by a human."""
    added = 0
    for listing in directory_client.list_verified_servers():
        add_trusted_server(
            client, embed_fn,
            server_name=listing["name"],
            description=listing["description"],
            publisher=listing["publisher"],
            declared_tools=listing["tools"],
            approval_source="directory_verified_tier",
            approved_at=listing["verified_at"],
        )
        added += 1
    return added
```

Keeping `sync_directory_verified_tier` as a distinct, periodic job (rather than a one-time import) matters because a directory's verified tier is not static — servers get added to it, and in principle a verification could later be revoked, neither of which an org's corpus should silently miss.

## Component Two: The Lookalike-Identity Check

This is the core signal: a newly discovered server that describes itself almost identically to an already-trusted one, but comes from a different, unverified publisher. The check is precise about what it flags and why — high similarity alone is not the problem (two legitimately different servers can coincidentally describe overlapping functionality), and a different publisher alone is not the problem (that's just... a different, possibly perfectly fine, server). The combination — high similarity *and* an unmatched publisher — is what makes something worth a human's attention.

```python
from dataclasses import dataclass, field


@dataclass
class TrustSignal:
    kind: str                 # "lookalike_identity" | "toolset_overlap"
    matched_server: str
    matched_publisher: str
    similarity: float
    severity: str              # "review" | "block"


def check_lookalike_identity(
    client: QdrantClient,
    embed_fn,
    candidate_name: str,
    candidate_description: str,
    candidate_tool_names: list[str],
    candidate_publisher: str,
    *,
    similarity_threshold: float = 0.90,
    top_k: int = 5,
) -> list[TrustSignal]:
    query_vec = embed_fn(identity_text(candidate_name, candidate_description, candidate_tool_names))
    hits = client.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        using="identity",
        limit=top_k,
        with_payload=True,
    ).points

    signals = []
    for h in hits:
        if h.score < similarity_threshold:
            continue
        trusted_publisher = h.payload["publisher"]
        if trusted_publisher == candidate_publisher:
            continue  # same publisher re-listing or re-approving its own server — not a lookalike
        exact_name_collision = h.payload["server_name"] == candidate_name
        signals.append(TrustSignal(
            kind="lookalike_identity",
            matched_server=h.payload["server_name"],
            matched_publisher=trusted_publisher,
            similarity=h.score,
            severity="block" if exact_name_collision else "review",
        ))
    return signals
```

`similarity_threshold=0.90` is a starting point, not a universal constant — it should be calibrated against the actual embedding model in use and revisited as the trusted corpus grows, the same way any similarity-based threshold needs calibration against real data rather than a value chosen in the abstract. The distinction between `"block"` (an exact name collision with a mismatched publisher — there's no legitimate reason two different publishers register the literal same server name) and `"review"` (high similarity, different name, different publisher — plausibly a lookalike, but also plausibly a coincidentally similar description) is the load-bearing design choice here: only the narrowest, least ambiguous case should ever auto-block, everything softer routes to a human.

## Component Three: The Tool-Set Overlap Check

A subtler variant of the same risk: a server with a generic, unassuming name — nothing that would trip the identity check above — whose declared tool list closely matches a sensitive trusted server's tool list. This is capability impersonation rather than identity impersonation, and it deserves a separate, lighter-weight check precisely because it's a weaker signal on its own — tool names and descriptions naturally converge across servers doing genuinely similar, legitimate things (plenty of file-search servers will have a tool called something like `search_files`), so this check is scored and surfaced, not blocked, even at high similarity.

```python
def check_toolset_overlap(
    client: QdrantClient,
    embed_fn,
    candidate_declared_tools: list[dict],
    *,
    similarity_threshold: float = 0.85,
    top_k: int = 5,
) -> list[TrustSignal]:
    query_vec = embed_fn(toolset_text(candidate_declared_tools))
    hits = client.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        using="toolset",
        limit=top_k,
        with_payload=True,
    ).points

    return [
        TrustSignal(
            kind="toolset_overlap",
            matched_server=h.payload["server_name"],
            matched_publisher=h.payload["publisher"],
            similarity=h.score,
            severity="review",  # never auto-block on toolset overlap alone
        )
        for h in hits if h.score >= similarity_threshold
    ]
```

The severity is hardcoded to `"review"` because toolset overlap, unlike an identity-plus-publisher mismatch, has no equivalent of the "exact name collision" case that would justify an automatic block — the strongest this signal should ever do on its own is put a candidate server in front of a human before an OAuth request goes out, flagged as "declares tools very similar to `<trusted sensitive server>`, worth a second look before granting access."

Which trusted servers are worth running this check against at all is itself a judgment call worth making explicit rather than leaving implicit. Running it against every entry in the corpus indiscriminately works at small scale; at a larger scale, scoping the toolset-overlap check to a `sensitive` flag on trusted entries — set for servers with write access, financial data access, or credential-adjacent capabilities — keeps the check's signal-to-noise ratio higher by only comparing against the servers where an impersonation attempt would actually matter.

### A Cheap Pre-Filter: Name-Distance Heuristics

Before either embedding-based check runs, a plain edit-distance comparison against known server names catches the narrowest and most classic case — a dropped hyphen, a transposed character, a pluralization — at negligible cost, and is worth running unconditionally rather than relying on the embedding model to happen to place near-identical strings close together (most embedding models do, but a direct string check is cheaper and more interpretable for this specific pattern):

```python
def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            current[j] = min(
                previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb),
            )
        previous = current
    return previous[-1]


def near_name_matches(candidate_name: str, trusted_names: list[str], max_distance: int = 2) -> list[str]:
    normalized = candidate_name.lower().replace("-", "").replace("_", "")
    matches = []
    for name in trusted_names:
        trusted_normalized = name.lower().replace("-", "").replace("_", "")
        if normalized != trusted_normalized and levenshtein(normalized, trusted_normalized) <= max_distance:
            matches.append(name)
    return matches
```

Normalizing away hyphens and underscores before comparing is deliberate — `gdrive-search-mcp` and `gdrive-searchmcp` differ by a single character once separators are stripped, which is exactly the pattern this pre-filter exists to catch cheaply, ahead of (and in addition to) whatever the embedding-based identity check separately finds through semantic rather than lexical similarity. The two checks catch different things: this one catches lexical near-misses even when a description is worded completely differently; the embedding check catches semantically similar descriptions even under a completely different name.

## Putting It Together: The Pre-Connection Gate

```python
@dataclass
class TrustVerdict:
    allow_without_review: bool
    signals: list[TrustSignal] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(s.severity == "block" for s in self.signals)

    @property
    def needs_review(self) -> bool:
        return bool(self.signals) and not self.blocked


def pre_connection_check(
    client: QdrantClient,
    embed_fn,
    candidate_name: str,
    candidate_description: str,
    candidate_publisher: str,
    candidate_declared_tools: list[dict],
) -> TrustVerdict:
    tool_names = [t["name"] for t in candidate_declared_tools]
    signals = check_lookalike_identity(
        client, embed_fn, candidate_name, candidate_description,
        tool_names, candidate_publisher,
    )
    signals += check_toolset_overlap(client, embed_fn, candidate_declared_tools)
    return TrustVerdict(allow_without_review=not signals, signals=signals)
```

The calling code in the agent framework is a single gate check inserted before whatever already initiates an OAuth authorization request:

```python
verdict = pre_connection_check(
    qdrant_client, embed_fn,
    candidate_name="gdrive-searchmcp",
    candidate_description="Search files and folders in your Google Drive",
    candidate_publisher="unknown-dev-234",
    candidate_declared_tools=[
        {"name": "search_files", "description": "Search Drive files by query"},
        {"name": "read_file_content", "description": "Read the contents of a Drive file"},
    ],
)

if verdict.blocked:
    raise ConnectionBlocked(verdict.signals)
if verdict.needs_review:
    surface_for_human_confirmation(verdict.signals)   # halts here until approved
else:
    proceed_to_oauth_authorization()
```

Nothing about this gate touches the OAuth flow itself — it's a strict precondition. A server that passes cleanly proceeds exactly as it would today; a server that trips a signal stops before any authorization request, token, or tool call exists at all.

### Logging Every Decision, Not Just the Flags

A trust gate that only logs when it flags something makes it impossible to later answer "how many servers did we connect to without review this month, and were any of them subsequently found to be a problem." Every gate evaluation — clean pass or flag — is worth a lightweight audit record, kept separately from the trusted corpus itself so that reviewing decision history never requires mutating the corpus:

```python
@dataclass
class TrustDecisionRecord:
    candidate_name: str
    candidate_publisher: str
    verdict: str            # "clean" | "reviewed_approved" | "reviewed_rejected" | "blocked"
    signals: list[TrustSignal]
    decided_at: float
    decided_by: str          # "system" for a clean auto-pass, else the reviewing user/admin id


def log_decision(store, record: TrustDecisionRecord) -> None:
    store.append(record)  # append-only log: a durable table, not the Qdrant corpus itself
```

The distinction between the corpus (what's currently trusted) and the decision log (what was ever decided, and by whom) matters because they answer different questions later: the corpus answers "should a new candidate be flagged against this entry," while the log answers "who approved this server, and when, and on what evidence" — the question that actually comes up during an incident review, and one a corpus that only stores current state can't answer once an entry has been superseded or removed.

## Worked Example: gdrive-search-mcp vs. gdrive-searchmcp

An organization approved `gdrive-search-mcp` months ago — a well-established, actively maintained server published by `Acme Data Co`, reviewed by an internal admin, and added to the trusted corpus with `approval_source="org_admin_review"`. Its declared tools are `search_files`, `read_file_content`, and `list_folders`.

A user's agent, searching the newly-consolidated Plugin directory for "search my drive," surfaces `gdrive-searchmcp` — no hyphen between "search" and "mcp" — published by `unknown-dev-234`, first appearing in the directory the previous week. Its description reads almost identically to the trusted server's: "Search files and folders in your Google Drive," with the same two core tools declared.

Running `pre_connection_check` against this candidate:

1. **Identity check.** The candidate's identity text ("gdrive-searchmcp: Search files and folders in your Google Drive tools: read_file_content, search_files") embeds to a vector with similarity 0.94 against the trusted `gdrive-search-mcp` entry — well above the 0.90 threshold. The publishers don't match (`unknown-dev-234` vs. `Acme Data Co`), and the names aren't an *exact* collision (missing hyphen), so this resolves to a `TrustSignal(kind="lookalike_identity", severity="review", similarity=0.94)`.
2. **Toolset check.** The declared tools overlap almost completely with the trusted server's tool set, producing a second signal at similarity 0.91 — consistent with, and reinforcing, the identity signal rather than adding new independent information in this particular case.
3. **Verdict.** `TrustVerdict(allow_without_review=False, signals=[...])`, `needs_review=True`, `blocked=False` (no exact name collision, so this doesn't hard-block).

| Property | `gdrive-search-mcp` (trusted) | `gdrive-searchmcp` (candidate) |
|---|---|---|
| Publisher | Acme Data Co | unknown-dev-234 |
| First seen / approved | 2026-03-14, `org_admin_review` | Directory listing, last week |
| Declared tools | `search_files`, `read_file_content`, `list_folders` | `search_files`, `read_file_content` |
| Identity similarity to trusted entry | — | 0.94 |
| Toolset similarity to trusted entry | — | 0.91 |
| Verdict | n/a (this is the baseline) | **flagged for human review** |

The agent halts before sending any OAuth authorization request to `gdrive-searchmcp` and surfaces the two matched signals to the user: "this looks a lot like `gdrive-search-mcp`, which your organization already trusts, but comes from a different, unrecognized publisher — connect anyway, or use the known server instead?" Whether the candidate turns out to be an intentional lookalike or simply an independent developer who happened to build something similar and named it unfortunately closely, the decision gets made with the relevant information in front of a human, rather than by whichever result happened to rank first in a directory search.

It's worth extending the example slightly to show the gate isn't just a two-server special case. Suppose the same directory search actually returns four candidates:

| Candidate | Publisher | Identity similarity to trusted `gdrive-search-mcp` | Name-distance pre-filter | Gate outcome |
|---|---|---|---|---|
| `gdrive-search-mcp` | Acme Data Co | — (this is the trusted entry) | — | n/a, already trusted |
| `gdrive-searchmcp` | unknown-dev-234 | 0.94 | distance 0 after normalization | **flagged for review** |
| `drive-file-assistant` | Acme Data Co | 0.71 | no near match | clean (same publisher, moderate similarity — a second legitimate tool from a publisher you already trust) |
| `cloud-storage-helper` | random-oss-contributor | 0.38 (identity) / 0.89 (toolset) | no near match | **flagged for review** (toolset-overlap signal only) |

The third candidate illustrates why the identity check gates on publisher match, not just similarity: `drive-file-assistant` scores meaningfully similar to the trusted server and shares its publisher, so it passes cleanly as, most likely, a second legitimate offering from an already-vetted source rather than an impersonation attempt. The fourth candidate illustrates the toolset-overlap check earning its keep independently of the identity check — its name and description don't resemble the trusted server's at all, but its declared tools (a generic-sounding file search and read pair) score highly against `gdrive-search-mcp`'s toolset vector, which is exactly the "impersonating capability rather than identity" pattern Component Three exists to catch.

## Composing With OAuth Scoping

Trust verification and OAuth scoping answer different questions, and a server clearing this check should not receive any more trust downstream than it would have otherwise. Trust verification answers "should I connect to this server at all" — a yes/no (or yes-with-review) gate that runs once, at discovery time. Scoping answers "what should this server be allowed to do, given that I am connecting to it" — a per-session, per-task decision about token audience, requested permissions, and lifetime that applies every time a connection is actually used, independent of how well the server scored on the trust check.

Concretely: even `gdrive-search-mcp`, the long-trusted server in the worked example above, should still only ever receive a token scoped to read access for the specific Drive folder relevant to the current task, not a standing, broadly-scoped credential — passing trust verification is not a reason to relax scoping discipline. The two layers are complementary and deliberately non-substitutable: a narrowly-scoped token handed to an impersonating server still leaks whatever that narrow scope covers, and a broadly-scoped token handed to a legitimately trusted server is still an unnecessarily large blast radius if that server is later compromised. Trust verification reduces the odds of connecting to the wrong party in the first place; scoping bounds the damage on every connection, trusted or not.

One practical wrinkle worth flagging explicitly: the pre-connection gate runs once, at first discovery, not on every subsequent call to an already-connected server. Caching a "clean" verdict for the lifetime of a session (or until the candidate's declared name, publisher, or tool list changes) is the right default — re-running the full check on every tool invocation would add latency for no additional signal, since nothing about a server's declared identity changes between two consecutive calls in the same session. What *should* invalidate a cached verdict is any change to the server's declared metadata itself — a directory listing update, a re-fetched tool manifest with new entries — which is a reasonable trigger to re-run the gate rather than silently trusting the cached result indefinitely.

## What Directories Could Do to Make This Layer Unnecessary

Everything in this post is a client-side mitigation for a gap that a directory itself is better positioned to close. It's worth being explicit about what that would look like, both because it clarifies why the client-side layer is a stopgap rather than a permanent architecture, and because it gives a concrete target for what "this problem is basically solved" would mean.

A directory-level verified-publisher system, in the shape npm and PyPI eventually converged on, would tie a server listing to a checked identity — a domain the publisher demonstrably controls, an organization account with its own verification history, or a cryptographic signature over the server's manifest that a client can check without needing its own trusted corpus at all. Paired with edit-distance-aware search ranking that actively suppresses near-name collisions to unverified publishers (rather than ranking purely on relevance or recency), a directory could make the lookalike pattern in the worked example above structurally harder to pull off — not just harder to fall for.

None of that exists at meaningful scale across MCP directories today. Until it does, the trust-verification layer in this post is the pragmatic answer: a client-side corpus an organization builds and maintains itself, because waiting for the ecosystem-wide version to arrive means shipping agents that connect to unverified servers by name in the meantime.

## Challenges and Open Problems

**Cold start is a structural property, not a bug.** A genuinely new, never-before-seen, and entirely legitimate server will, by construction, look "unverified" the first time anyone encounters it — it has no entry in the trusted corpus to compare against, and depending on how the identity check is tuned, it may not even trigger a signal (nothing to be similar *to* yet), which means the system's default posture toward brand-new servers is silence, not suspicion. The real risk this creates is on the other side: a system that only ever knows how to flag things as suspicious, with no correspondingly easy path for a human to explicitly approve and add a new, legitimate server to the trusted corpus, will accumulate friction and get bypassed. The `add_trusted_server` function in Component One needs to be a first-class, low-friction workflow — ideally a single action available right from the same review-confirmation UI that surfaces a flag — not a separate, heavyweight admin process that only a few people know exists.

**Publisher-identity signals are thin across current directories.** The lookalike-identity check leans on a `publisher` field that, in most MCP directories today, is self-declared by whoever submitted the server rather than independently verified — closer to a GitHub username than to a notarized identity. This limits how strong a "different publisher" signal can really be: an attacker attempting a genuine impersonation could, in principle, also fabricate a publisher string designed to look similar to the target's. Until directories themselves adopt stronger publisher verification (comparable to a package registry's verified-maintainer badge, or a domain-ownership check), this check is a meaningfully useful filter, not a cryptographic guarantee.

**Tool-set overlap checks need pre-connection visibility into a candidate's tool list.** For the most cautious posture — never even opening a connection to an unverified server before deciding whether to trust it — the tool-set check has to work off whatever metadata a directory listing provides *before* connection, not what the server actually exposes once you're already talking to it over MCP's own tool-discovery mechanism. This is fine when a directory's listing includes an accurate, complete tool manifest (increasingly common as directories mature), and a real limitation when it doesn't: a server that under-declares its tools in its public listing but exposes more once connected defeats a check that only ever looked at the listing. Closing this gap fully requires either directories committing to accurate, complete tool manifests as a listing requirement, or accepting a lighter-weight initial connection (with no data access yet) purely for tool-list inspection before the trust decision is finalized — a real tradeoff this design does not resolve.

**A widely-used legitimate category — many small, independently-built servers offering genuinely similar functionality — will generate review flags at a rate that risks training users to click through them without reading.** Unlike the classic phishing-link-warning problem, where most warnings genuinely correspond to danger, a large fraction of toolset-overlap and even identity-similarity flags in a mature MCP ecosystem will correspond to nothing more sinister than two developers independently building similar file-search integrations. Keeping the false-positive rate low enough that a flag still carries meaning — via a well-calibrated similarity threshold, and via making the `"review"` interaction fast and specific enough that skimming it takes seconds rather than minutes — is an ongoing tuning problem, not a one-time calibration exercise, and it gets harder rather than easier as the trusted corpus and the surrounding ecosystem both grow.

## References

- Endor Labs. [State of Dependency Management 2025](https://www.endorlabs.com/learn/state-of-dependency-management-2025). Analysis of 10,663 MCP server repositories: 75% built by individual developers, 41% with no license information, 82% using sensitive APIs.
- Model Context Protocol. [Specification](https://modelcontextprotocol.io/specification). The protocol's own scope — capability negotiation between an already-selected client and server — and why server discovery and trust are outside it.
- Model Context Protocol. [Authorization Specification](https://modelcontextprotocol.io/specification/draft/basic/authorization). The OAuth 2.1 / RFC 8693 / RFC 8707-based token layer this post's trust-verification gate runs strictly before.
- OpenAI. ChatGPT Release Notes, July 9, 2026 — the App Directory to Plugin Directory migration serving both ChatGPT and Codex from a unified catalog (background context; see OpenAI's help center release notes for the primary source).
- Spracklen, J., et al. *We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs.* USENIX Security 2025. The research quantifying the package-hallucination phenomenon that "slopsquatting" (coined by Python Software Foundation developer-in-residence Seth Larson) exploits — the closest documented analogue to the lookalike-server risk this post addresses.
- Qdrant. [Vector Database Documentation](https://qdrant.tech/documentation/). Reference for named vectors, `query_points`, and payload filtering used throughout this post.
