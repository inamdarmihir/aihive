---
title: "Designing Tool Schemas Agents Don't Misuse: Lessons from Production MCP Servers"
date: 2026-07-30
description: "A tool's name, description, and parameter schema function as a prompt the model reads before every decision to use it — get that prompt wrong and the agent doesn't fail loudly, it succeeds at the wrong thing. This post covers the misuse patterns that recur in production MCP servers, OWASP's Tool Misuse & Exploitation category, and a Qdrant-backed linter that catches overlapping or near-duplicate tool definitions before they ship."
tags: ["agents", "mcp", "tool-design", "security", "schema", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

An MCP tool doesn't fail the way a function call with a type error fails. If a tool's schema is ambiguous, the agent doesn't throw an exception — it calls the tool with a plausible-but-wrong argument, gets a valid-looking response back, and proceeds confidently on bad input. Anthropic's own engineering write-up on tool design is direct about the mechanism: tool descriptions are prompts, and every word in a tool's name, description, and parameter documentation shapes how the model decides to use it — refinements to tool descriptions alone produced a measurable jump on SWE-bench Verified, with no change to the model or the underlying capability.

This post is about the schema-design failure surface specifically — not tool-*response* validation (covered by the companion post on [silent MCP failures](/posts/silent-failures-mcp-agents/), which explicitly scopes prompt injection and schema-misuse out of its own coverage) and not duplicate-*issue* detection in a bug tracker (a related but distinct problem covered in the [Qdrant GitHub triage post](/posts/qdrant-duplicate-issue-triage/)). Here the concern is a tool registry that grows past a handful of tools — the point where two engineers on different teams independently ship tools that do almost the same thing, phrased just differently enough that an agent picks between them inconsistently.

## Table of Contents

1. [A Tool Definition Is a Prompt, Not Documentation](#a-tool-definition-is-a-prompt-not-documentation)
2. [Misuse Patterns That Recur in Production](#misuse-patterns-that-recur-in-production)
3. [OWASP's Tool Misuse and Exploitation Category](#owasps-tool-misuse-and-exploitation-category)
4. [A Design Checklist](#a-design-checklist)
5. [An Eval Harness for Tool Descriptions](#an-eval-harness-for-tool-descriptions)
6. [Scoping Tools to the Privilege They Actually Need](#scoping-tools-to-the-privilege-they-actually-need)
7. [A Qdrant-Backed Pre-Deployment Tool Linter](#a-qdrant-backed-pre-deployment-tool-linter)
8. [A Worked Example](#a-worked-example)
9. [Challenges and Open Problems](#challenges-and-open-problems)
10. [References](#references)

## A Tool Definition Is a Prompt, Not Documentation

Traditional API design treats a function signature as a contract for a compiler or a human reading docs — the machine enforces types, the human infers intent from good naming. Neither assumption transfers cleanly to a model deciding whether and how to call a tool: there's no compiler rejecting a semantically-wrong-but-type-valid argument, and the "human" reading the docs is doing so probabilistically, under time pressure (mid-generation, one tool call among several it needs to make), not with the deliberate care a developer brings to reading an SDK reference. A parameter named `user` invites the model to guess whether it wants an ID, a username, or a full name; a parameter named `user_id` closes off two of those three guesses before the model ever has to make one.

## Misuse Patterns That Recur in Production

**Ambiguous parameter naming.** `user`, `id`, `date`, `type` — any parameter name that requires inferring format or referent from context rather than stating it. The fix is boring and effective: `user_id` (not `user`), `due_date_iso8601` (not `date`), `record_type_enum` (not `type`).

**Overly broad tools that invite scope creep.** A single `manage_record(action, id, data)` tool where `action` can be `"read"`, `"update"`, or `"delete"` puts the entire access-control decision inside a string parameter the model chooses, rather than in the tool boundary itself. Splitting into `read_record`, `update_record`, `delete_record` as separate tools — each independently scopable, independently permissioned, independently auditable — moves the safety boundary from "hope the model picks the right action string" to "the delete capability simply isn't exposed to callers who shouldn't have it."

**Free-text parameters where an enum would do.** A `status` parameter typed as a free string invites the model to pass `"done"`, `"Done"`, `"completed"`, or an invented value that happens to look plausible, none of which your backend may actually recognize. Constraining to an explicit enum in the JSON Schema removes an entire class of silently-wrong calls — the model either picks a valid value or the call fails loudly at the schema-validation layer, which is a far better failure mode than a backend silently no-op'ing on an unrecognized status string.

**Unbounded queries.** A `search(query)` tool with no `limit` parameter, or one where `limit` has no enforced maximum, invites a call that returns an enormous payload — Anthropic's own guidance is explicit that agents reason better with a filtered, paginated result set than with everything at once, and an unbounded tool is both a UX problem (context window bloat) and occasionally a real resource-exhaustion problem on the backend.

**Tools that silently accept invalid parameter combinations.** A tool that accepts both `start_date` and `days_ago` and picks one arbitrarily when both are present, rather than rejecting the ambiguous call, produces a result the model has no way of knowing was computed from the "wrong" of the two parameters it supplied.

## OWASP's Tool Misuse and Exploitation Category

The OWASP Top 10 for Agentic Applications (2026) names this class of problem **ASI02: Tool Misuse & Exploitation** directly: an agent using a legitimately-held tool in an unsafe or unintended way — chaining a harmless tool with a sensitive API, or forwarding unvalidated output from one tool call into a powerful downstream command. The framing worth internalizing: the risk isn't that the agent acquires unauthorized tools, it's that poor scoping, ambiguous descriptions, or unsafe delegation lets it misuse tools it already legitimately has. A `manage_record` tool with a `delete` action string is not a security bug in the traditional sense — every call is authenticated, every call is to a tool the caller is entitled to use — and it's exactly the shape ASI02 describes, because the *unsafe use* is baked into how narrowly (or not) the tool boundary was drawn in the first place.

## A Design Checklist

Distilled from the patterns above and Anthropic's own published guidance:

- **Prefix tools by domain** (`search_contacts`, `search_documents`) rather than generic verbs, for clarity as the registry scales past a handful of tools.
- **Prefer narrow, high-leverage tools over thin wrappers around every API endpoint you have.** A registry of forty near-identical CRUD wrappers is worse for an agent than five well-scoped tools that correspond to actual workflows.
- **Return human-readable context, not raw IDs**, in tool responses — an agent reasons better over `{"assignee": "Priya Shah"}` than `{"assignee_id": 48213}`, and a raw ID invites the model to fabricate a plausible-looking name if it needs one for a follow-up action.
- **Enforce pagination and hard limits server-side**, not just as a documented convention the model might ignore.
- **Constrain to enums wherever the value space is closed**, and reserve free text for genuinely open-ended fields.
- **Write a small eval suite (5–10 real or synthetic call scenarios) per tool** and actually run it — Anthropic's own account of iterating on tool design treats the agent's own behavior against a fixed eval set as the feedback loop, not developer intuition alone.

## An Eval Harness for Tool Descriptions

"Write a small eval suite and run it" is easy advice to nod at and skip; making it concrete is what makes it actually happen. A minimal harness: a fixed set of realistic call scenarios per tool, each with an expected argument shape, run against the actual model that will use the tool in production (not a proxy or a cheaper model — tool-calling behavior is not reliably transferable across model families or even model versions):

```python
@dataclass
class ToolCallScenario:
    prompt: str                    # a realistic user request that should trigger this tool
    expected_tool: str
    expected_args_subset: dict      # keys that must match; extra args are fine

def run_tool_eval(scenarios: list[ToolCallScenario], agent) -> dict:
    results = []
    for sc in scenarios:
        call = agent.get_first_tool_call(sc.prompt)
        correct_tool = call is not None and call.tool_name == sc.expected_tool
        correct_args = correct_tool and all(
            call.args.get(k) == v for k, v in sc.expected_args_subset.items()
        )
        results.append({
            "prompt": sc.prompt, "correct_tool": correct_tool, "correct_args": correct_args,
            "actual_call": call,
        })
    n = len(results)
    return {
        "tool_accuracy": sum(r["correct_tool"] for r in results) / n,
        "arg_accuracy": sum(r["correct_args"] for r in results) / n,
        "failures": [r for r in results if not r["correct_args"]],
    }
```

A ten-scenario suite for `search_documents` would include the obvious case ("find documents about Q3 revenue"), a case designed to probe the ambiguous-naming failure mode directly ("look up user 4821's recent activity" — does the model correctly map "user" to `user_id`, or does it hallucinate a `user` parameter that doesn't exist in the schema?), and a case testing whether the model respects the declared `limit` bound rather than requesting an unbounded result set. Run this suite every time the tool's description or parameters change, not just at initial ship — a description edit made to fix one failure mode can easily introduce a regression in a scenario the eval suite would have caught immediately and a human reviewer, skimming a diff, would not.

Running a ten-scenario suite like this before and after applying the checklist's naming and bounding fixes to a representative ambiguous tool definition (illustrative numbers, shaped by the mechanism described rather than a specific measured deployment):

| Metric | Before (ambiguous naming, no enum, no limit cap) | After (checklist applied) |
|---|---|---|
| Tool-selection accuracy (right tool called at all) | 90% | 100% |
| Argument accuracy (correct tool, correct args) | 60% | 90% |
| Unbounded-query attempts (no `limit` respected) | 3 of 10 scenarios | 0 of 10 scenarios |
| Enum-violation attempts (invented `status` value) | 2 of 10 scenarios | 0 of 10 scenarios |

Tool-selection accuracy moving least (90% → 100%) and argument accuracy moving most (60% → 90%) matches the mechanism described earlier: an ambiguous tool is usually still the *right* tool to reach for — the model correctly identifies intent — the failure concentrates in *how* it's called once selected, which is exactly the layer that parameter naming, enums, and bounds operate on. A team that only tracks whether the right tool got called (a coarser, easier-to-instrument metric) would see a nearly-perfect score both before and after and conclude the tool definition was already fine, missing the argument-level failures entirely.

## Scoping Tools to the Privilege They Actually Need

The `manage_record(action, ...)` anti-pattern from earlier in this post is really a special case of a more general principle: a tool's *declared capability surface*, independent of any runtime auth check, is itself a security boundary, because it determines what an agent can even attempt to do before any permission system gets a say. This is distinct from the token-scoping problem covered in the [OAuth for agents](/posts/oauth-for-agents/) post — that post is about which *user* a token authorizes an agent to act as; this is about which *actions* a tool exposes at all, regardless of who's calling it.

Concretely, this means resisting the urge to build one flexible, parameterized tool that covers every variation of an operation a backend supports, in favor of several narrower tools that each map to one clearly-scoped capability:

- A `read_only` tool family (`search_documents`, `get_record`) that a lower-trust calling context — a public-facing assistant, say — can be granted without any risk of state mutation, because the mutation capability doesn't exist in that tool's schema at all, not because a permission check happened to deny it at runtime.
- A `write` tool family (`create_record`, `update_record`) gated to contexts that genuinely need mutation, kept structurally separate so a misconfigured permission grant on the read-only family can't accidentally carry write capability along with it.
- A `destructive` tool family (`delete_record`) exposed to as few calling contexts as the product genuinely requires, ideally paired with the human-confirmation gate pattern discussed for browser agents elsewhere on this blog, since a destructive action reached via a tool call carries the same asymmetric downside as one reached via a browser click.

This three-tier split costs more up-front design work than one flexible `manage_record` tool and pays for itself the first time a permission-grant mistake happens — granting the `read_only` family to a context that shouldn't have write access is a no-op by construction, where granting a flexible `manage_record` tool to the same context and forgetting to also restrict its `action` parameter is a live vulnerability that a schema review might well miss, since the schema itself gives no visual signal that `action="delete"` is even a possibility a reviewer should be checking for.

## A Qdrant-Backed Pre-Deployment Tool Linter

The checklist above catches problems within a single tool's design. It doesn't catch the problem that shows up once a registry has dozens of tools shipped by different people over time: two tools that overlap enough in purpose that an agent picks between them inconsistently, without either tool individually looking wrong. Embedding each tool's full definition (name, description, parameter schema) and checking new submissions against the existing registry by similarity catches this before deployment, the same nearest-neighbor-duplicate mechanism used for GitHub issue triage elsewhere on this blog, applied to a different corpus with a different purpose — linting a registry pre-deploy, not deduplicating user reports post-hoc:

```python
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class ToolRegistryLinter:
    def __init__(self, client: QdrantClient, embed_fn, collection: str = "tool_registry"):
        self.client, self.embed_fn, self.collection = client, embed_fn, collection
        names = {c.name for c in client.get_collections().collections}
        if collection not in names:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def _definition_text(self, name: str, description: str, params: dict) -> str:
        param_str = ", ".join(f"{k}: {v}" for k, v in params.items())
        return f"{name}({param_str}): {description}"

    def check_before_deploy(self, name: str, description: str, params: dict,
                             threshold: float = 0.88, top_k: int = 5):
        text = self._definition_text(name, description, params)
        hits = self.client.search(
            collection_name=self.collection, query_vector=self.embed_fn(text), limit=top_k,
        )
        overlaps = [h for h in hits if h.score >= threshold]
        return {
            "safe_to_deploy": not overlaps,
            "overlapping_tools": [
                {"score": h.score, "definition": h.payload["text"]} for h in overlaps
            ],
        }

    def register(self, name: str, description: str, params: dict):
        text = self._definition_text(name, description, params)
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=str(uuid.uuid4()), vector=self.embed_fn(text),
                                 payload={"name": name, "text": text})],
        )
```

Running `check_before_deploy` in CI, ahead of merging a new tool into a shared MCP server, turns "did anyone happen to notice this overlaps with an existing tool" from a code-review hope into an automated gate — flagged overlaps go to a human for a real decision (consolidate, differentiate the descriptions more sharply, or confirm the overlap is intentional and scoped differently enough to keep both), rather than being silently discovered months later when an agent starts calling the wrong one of the two roughly half the time.

## A Worked Example

Team A ships `get_customer_orders(customer_id, limit)`. Six months later, Team B, unaware of Team A's tool, ships `fetch_orders_for_customer(customer_id, max_results)` — functionally near-identical, different names, different parameter name for the same limit concept. Run through the linter at merge time, the two definitions embed at a cosine similarity around 0.91 against the threshold of 0.88, flagging the overlap before it merges. A human reviewer either deletes the redundant tool, or — if there's a genuine reason both need to exist (different backing services, different auth scopes) — renames both to make the distinction explicit in the description itself (`get_customer_orders_v2_billing_system` vs. `get_customer_orders_legacy_crm`) rather than leaving an agent to guess between two similarly-described tools at call time.

## Challenges and Open Problems

**Embedding similarity measures description overlap, not behavioral overlap.** Two tools can read as nearly identical in their descriptions while having meaningfully different side effects (one is read-only, one triggers a downstream webhook), or read as very different while functionally overlapping (a `refund_order` tool and a `void_transaction` tool, described in domain-specific jargon that doesn't textually resemble each other but resolve to the same customer-facing outcome). The linter is a triage signal for human review, not an automated merge gate that can safely run without a human in the loop on flagged pairs.

**There's a real tension between "few high-leverage tools" and organizational reality.** Anthropic's own advice — prefer a small number of high-leverage tools over thin wrappers around every endpoint — is correct and also hard to enforce once a tool registry is shared across teams that don't coordinate tightly. The linter catches overlap after the fact; it doesn't solve the organizational problem of teams not knowing what already exists before they start building, which is arguably the more fundamental issue.

**A threshold tuned for one domain doesn't necessarily transfer to another.** The 0.88 similarity threshold in the worked example was chosen for illustration — the right threshold depends on how verbose and jargon-heavy your organization's tool descriptions tend to be, and needs calibration against your own registry's false-positive rate (flagging genuinely distinct tools) versus false-negative rate (missing real overlaps written in dissimilar language) rather than being taken as a universal constant.

## References

- Anthropic. *Writing effective tools for AI agents — using AI agents.* [anthropic.com/engineering/writing-tools-for-agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Claude Platform Docs. *Define tools.* [platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools)
- OWASP Gen AI Security Project. *OWASP Top 10 for Agentic Applications (2026).* [genai.owasp.org](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
- Model Context Protocol. *Specification.* [modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification/draft/basic/authorization)
- Qdrant. *Qdrant Vector Database Documentation.* [qdrant.tech/documentation](https://qdrant.tech/documentation)
