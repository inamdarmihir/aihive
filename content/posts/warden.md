---
title: "Warden: A Pre-Install Gate for Autonomous Coding Agents"
date: 2026-07-26
description: "LLMs have hallucinated package names for years, but the execution model just changed: an autonomous coding agent now runs pip install and npm install itself, with no human eyeballing the name first. Warden is a layered pre-install gate for that new reality — a pip-installable hook that plugs into Claude Code, Cursor, and Copilot's tool-execution loop — where a plain registry-existence check is necessary but not sufficient, and Qdrant does two structurally different jobs: catching typosquats an attacker got to first, and proactively predicting which conflation-style names your own agents are likely to invent next."
tags: ["agents", "security", "supply-chain", "coding-agents", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Code-generating LLMs have invented package names that don't exist since the first models were fine-tuned on code. That fact alone was never the interesting part — a hallucinated import in a ChatGPT response sitting in a browser tab is inert until a human copies it into a terminal. What changed in the last cycle of agent tooling is the step that used to sit between hallucination and execution: Claude Code, Cursor's agent mode, OpenAI Codex, and GitHub Copilot's workspace agents now run `pip install` and `npm install` themselves, inside an auto-accept or bypass-permissions loop, as a routine part of finishing a task. The eyeball check that used to catch an obviously-wrong package name before anything happened is gone, not because the underlying hallucination rate got worse, but because the human is no longer in the loop at the exact moment it matters.

This post is about defending that specific moment — the interval between an agent deciding to install a package and the shell actually running the command. It's also the implementation spec for **Warden**, a small pip-installable library that closes that moment: a `PreToolUse`-style hook you register directly with an existing agent framework (Claude Code, GitHub Copilot's workspace agents, or any harness that exposes a pre-tool-call interception point), backed by a layered check pipeline and a Qdrant-held trusted corpus. I cover why checking whether a proposed package name exists in the registry is a necessary first move but has a real structural blind spot, the four-layer design that closes it, and where Qdrant earns its place in that pipeline as one signal among several rather than the whole design. It does not cover post-install runtime sandboxing, supply-chain attacks that don't involve a hallucinated name (typosquats a human fat-fingered, or compromised maintainer accounts), or SBOM generation — those are different, well-covered problems. Familiarity with how agent frameworks expose pre-tool-use hooks (Claude Code's `PreToolUse`, in particular) is assumed but not required to follow the design.

## Table of Contents

1. [The Term, and Why the Threat Model Just Changed](#the-term-and-why-the-threat-model-just-changed)
2. [What Spracklen et al. Actually Measured](#what-spracklen-et-al-actually-measured)
3. [Why Existence Checking Alone Fails: A Race-Condition Argument](#why-existence-checking-alone-fails-a-race-condition-argument)
4. [The react-codeshift Incident](#the-react-codeshift-incident)
5. [Prior Art: Registry Scanners and the Agentinel Pattern](#prior-art-registry-scanners-and-the-agentinel-pattern)
6. [Design: Warden's Layered Pre-Install Gate](#design-wardens-layered-pre-install-gate)
   - [Layer 1: Registry Existence](#layer-1-registry-existence)
   - [Layer 2: Trusted-Corpus Near-Miss Detection](#layer-2-trusted-corpus-near-miss-detection)
   - [Layer 3: Proactive Conflation-Risk Prediction](#layer-3-proactive-conflation-risk-prediction)
   - [Layer 4: Metadata Heuristics](#layer-4-metadata-heuristics)
   - [Structured Feedback Into the Agent's Context](#structured-feedback-into-the-agents-context)
7. [Installing and Using Warden](#installing-and-using-warden)
8. [Worked Example: react-codeshift Through the Pipeline](#worked-example-react-codeshift-through-the-pipeline)
9. [Mapping to the Five Eyes Agentic AI Advisory](#mapping-to-the-five-eyes-agentic-ai-advisory)
10. [Challenges and Open Problems](#challenges-and-open-problems)
11. [References](#references)

## The Term, and Why the Threat Model Just Changed

**Slopsquatting** was coined in April 2025 by Seth Larson, Developer-in-Residence at the **Python Software Foundation**, and popularized by Andrew Nesbitt, creator of **Ecosyste.ms** — the practice of registering a package name that doesn't exist but that an LLM tends to hallucinate, betting that a developer or an agent will eventually install it. The mechanics are the same as typosquatting's older cousin: register a name close enough to something legitimate-sounding, publish a payload, wait. What's different is the source of the name. A typosquat exploits a human's fat finger; a slopsquat exploits a model's confident, repeatable mistake.

The reason this deserves a fresh look in 2026 specifically is not that the hallucination rate changed — it's that the *execution model* did. A developer pasting AI-suggested code into a terminal in 2024 was, whether they thought about it that way or not, a manual verification gate: eyes on the package name, a half-second of "wait, is that real?" before hitting enter. An agent running in auto-accept mode, with shell access as one of its standard tools, doesn't have that half-second. It proposes a package, and unless something intercepts the specific `Bash` tool call before the shell executes it, the install just happens. The **Cloud Security Alliance**'s April 2026 research note on slopsquatting is blunt about this shift: the threat moved from theoretical to confirmed specifically because agentic tooling removed the human checkpoint that made hallucinated names mostly harmless for two years.

The scale this plays out against is not small. **Sonatype**'s 2026 State of the Software Supply Chain report logged more than 454,600 new malicious packages across open-source registries in 2025 alone — a 75% year-over-year increase, bringing the cumulative known-and-blocked total past 1.2 million. Not all of that volume is slopsquatting specifically (plenty is old-fashioned typosquatting and automated flooding campaigns), but it's the ambient environment a pre-install gate operates in: registries where "a package with a plausible name exists" is a rapidly weakening signal of "a package with a plausible name is safe."

**Endor Labs**'s "State of Dependency Management 2025" gives the sharpest picture yet of what agents specifically are doing wrong. Analyzing 10,663 MCP server repositories alongside large-scale testing of AI-generated dependency recommendations across PyPI, npm, Maven, and NuGet, the report found that 49% of dependency versions imported by AI coding agents carry known vulnerabilities, 34% are outright hallucinated — they don't exist in any registry at all — and, putting those together, only about 1 in 5 dependency recommendations from an AI coding assistant are actually safe to use. The finding that matters most for this post's design: when agents were equipped with even simple verification tools — `curl`, `grep`, something that lets the agent actually check a registry before trusting its own suggestion — the safe-dependency rate roughly tripled, from about 20% to 57%. That's the entire thesis of a pre-install gate stated as an empirical result rather than an argument: verification, not a better model, is what moves the safety number.

## What Spracklen et al. Actually Measured

Before designing a defense it's worth being precise about what's actually being defended against, and the most rigorous public measurement remains **Spracklen et al., "We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code-Generating LLMs"** ([USENIX Security 2025, arXiv:2406.10279](https://arxiv.org/abs/2406.10279)). The study tested 16 code-generation models, commercial and open source, across Python and JavaScript, generating 576,000 code samples total.

The headline numbers:

| Finding | Value |
|---|---|
| Overall hallucination rate across all models | 19.7% of recommended packages didn't exist |
| Open-source model average | 21.7% |
| Commercial model average | 5.2% |
| Best commercial model (GPT-4 Turbo) | 3.59% |
| Worst models (CodeLlama 7B / 34B) | hallucinated in over a third of outputs |
| Unique hallucinated package names found | 205,474 |

Two further findings matter more for design than the headline rate does. First, **repeatability**: the authors reran 500 hallucination-triggering prompts 10 times each against the same model. 43% of hallucinated packages reappeared in *all 10* reruns; 39% never reappeared at all; 58% repeated more than once across the 10 runs. That's a bimodal distribution, not a smooth one — a given hallucination is either highly stable (the model will produce it again and again for a semantically similar prompt) or effectively noise. The stable half is the actionable threat, because it's predictable enough for an attacker to target.

Second, **plausibility**, measured via Levenshtein-distance analysis against real package names: only about 13% of hallucinated names were simple off-by-one typos of something real. Roughly 38% showed moderate string similarity — naming or conceptual overlap with real packages, the conflation pattern. The remaining ~49% were highly dissimilar from any existing package name: fully fabricated, but still contextually plausible enough that a developer skimming a diff wouldn't blink. A fourth, smaller but structurally important finding: 8.7% of hallucinated Python package recommendations were actually valid npm package names — the model reached into the wrong ecosystem's namespace entirely. Finally, on self-detection: GPT-4 Turbo and DeepSeek could correctly identify their *own* just-generated hallucinated package names with over 75% accuracy when explicitly asked to review the output — a model can often tell you it lied, if you ask it to check.

Every layer of the design below maps to one of these findings. Pure fabrications (the ~49%) are what a plain existence check catches trivially. The 38% conflation bucket is what needs a semantic, not lexical, detector. The bimodal repeatability is what turns hallucination from an annoyance into an attack surface. The cross-ecosystem 8.7% is a limitation I come back to at the end, because it breaks an assumption the rest of the design quietly makes.

It's also worth noting what Spracklen et al. tried as mitigations on the model side, because it clarifies why this post's design deliberately sits outside the model rather than trying to fix it there. The authors evaluated Retrieval-Augmented Generation, self-detected feedback, and supervised fine-tuning, and reported that fine-tuning produced the largest reduction, pushing one model's hallucination rate below 3%. Those are real, complementary mitigations — and a team training or fine-tuning its own coding model should absolutely use them. But they're mitigations against the *rate*, not guarantees against a specific instance, and none of them help at all against the race-condition risk described next: even a model with a hallucination rate near zero still occasionally produces a name that happens to already be squatted, and no amount of training-time correction protects against a name that another developer's prompt, against another deployment of the same model, exposed to an attacker last month. Model-side mitigation and an execution-time gate are answering different questions — "how often does this happen" versus "what happens the one time it does" — and a serious defense needs both, not one instead of the other.

## Why Existence Checking Alone Fails: A Race-Condition Argument

The obvious first defense — before an install runs, check whether the named package actually exists in the target registry — catches the ~49% of pure fabrications for free and costs one cheap API call. It is necessary. It is not sufficient, and the reason is structural rather than a matter of tuning: **if an attacker has already registered the exact hallucinated name, the existence check passes and the attack succeeds.** The check can only tell you "this name resolves to something," not "the something it resolves to is legitimate."

This would be a minor edge case if hallucinations were independently random across developers. They're not — that's the whole point of the repeatability finding above. A modal hallucination that a popular commercial model produces for a common prompt pattern (say, "give me a package for exponential backoff retries in Python") is a shared target, not a private one. Every developer or agent issuing a semantically similar prompt against the same model is drawing from the same small set of likely outputs. An attacker needs to observe that pattern surface exactly once — in a public GitHub issue, a Stack Overflow answer, a shared agent skill file, a blog post with a code sample — to register the name before *your* agent ever rolls the dice.

This is worth a short, deliberately informal probabilistic sketch, because "worth taking seriously" and "provably quantified" are different claims and I want to be honest about which one this is. Model the repeatability rate $r$ from Spracklen et al. (roughly $0.43$ to $0.58$, depending on which tier of the bimodal distribution you're looking at) as an approximation of the probability that a semantically similar prompt against the same model reproduces the same modal hallucinated name. If $n$ independent developers issue prompts from the same cluster against that model, a rough birthday-style approximation for the probability that *at least one* of them hits the shared modal name is:

$$P(\text{at least one exposure}) \approx 1 - (1 - r)^n$$

Plugging in the low end of the observed range, $r = 0.43$, and a genuinely small population, $n = 10$:

$$1 - (1 - 0.43)^{10} = 1 - (0.57)^{10} \approx 1 - 0.0034 \approx 0.997$$

With only ten independent developers or agents drawing from the same prompt pattern against the same popular model, there's better than a 99% chance at least one of them produces the shared hallucination — and the attacker only needs that single public exposure to register the name ahead of everyone else who hasn't hit it yet. This is not a rigorous statistical model of real-world prompt diversity (real prompts aren't drawn i.i.d. from a fixed distribution, and $r$ isn't literally a per-developer independent probability), but it captures the shape of the actual risk correctly: your defense isn't a bet against your own agent's roll of the dice, it's a bet against every other developer worldwide using the same handful of popular models. Law of large numbers favors the attacker, not you, and that's exactly the case a name-exists check has no way to catch — it was designed to catch fabrication, not front-running.

## The react-codeshift Incident

The clearest real demonstration of both the conflation pattern and the distribution mechanism happened in January 2026. Security researcher Charlie Eriksen at **Aikido Security**, while extending earlier research on unclaimed package names, scraped GitHub for references to `npx` commands and kept finding one name recurring across an unusual number of repositories: `react-codeshift`.

The name is a textbook conflation hallucination, blending two real, well-known packages — `jscodeshift` (Facebook's codemod runner) and `react-codemod` (React-specific transform scripts built on it). No such package had ever existed. Eriksen traced its origin to a single commit of 47 LLM-generated **Agent Skills** — shareable Markdown/YAML instruction files that coding agents load and execute as part of a task — committed with no human review. The skills were forked, translated into Japanese (which forked them further), and had propagated to more than 237 GitHub repositories by the time Eriksen found them, each inheriting the same hallucinated reference with no human at any point in the chain having deliberately introduced it.

The part worth sitting with is who was actually running `npx react-codeshift`. It wasn't developers copy-pasting a suggestion — it was autonomous agents loading a forked skill file and following its instructions verbatim, faithfully attempting to install a package that had never existed, without ever checking. Eriksen described watching "a persistent trickle of 1-4 downloads per day" against the placeholder he registered to claim the name defensively before an attacker did ([Aikido Security, "Agent Skills Spreading Hallucinated npx Commands"](https://www.aikido.dev/blog/agent-skills-spreading-hallucinated-npx-commands)). Those download attempts are the race-condition argument from the previous section made concrete: 237 independent exposures of the same hallucination, and the only reason it stayed benign was that a researcher happened to claim the name first. It could just as easily have been an attacker.

## Prior Art: Registry Scanners and the Agentinel Pattern

Commercial scanning already exists here — Socket.dev, Snyk, and Aikido all offer registry-aware dependency scanning that can flag a newly-registered or suspicious package. What's more directly relevant to the design below is a category of small, open-source tools built specifically to sit inside an agent's tool-execution loop rather than run as a separate CI step. **Agentinel** is the clearest example: it registers as a pre-tool-use hook in Claude Code (and similarly for GitHub Copilot's execution model), intercepts `npm install` / `pip install` calls before the shell runs them, and checks the named package against a bundled OSV database of over 200,000 known-malicious packages plus lightweight heuristics — package age under 30 days, zero or near-zero downloads.

The detail worth borrowing wholesale, independent of Agentinel's specific detection logic, is *the shape of its output*. Rather than a blocking error or a line in a log a human might read hours later, a rejected install returns structured JSON directly into the agent's own context:

```json
{
  "blocked": true,
  "reason": "Package does not exist on npm (Likely hallucination).",
  "suggestion": "Search for an existing alternative package."
}
```

This matters because the audience for the rejection is not primarily a human — it's the agent itself, mid-task. A plain shell error ("command failed, exit code 1") gives a coding agent almost nothing to work with; a structured verdict that says *why* lets the agent reason "I hallucinated a package name, let me search for the real one" and self-correct within the same turn, instead of blindly retrying the identical install or surfacing a dead end to the user. Any pre-install gate built for an agent, not a human reviewer, should return feedback in this shape. The pipeline below does.

## Design: Warden's Layered Pre-Install Gate

**Warden** sits as a `PreToolUse`-style hook matched against `Bash` tool calls, inspecting the command before it reaches the shell. It runs four checks in increasing order of cost and decreasing order of certainty, short-circuiting as soon as one produces a definitive verdict.

```
proposed install command
        │
        ▼
┌───────────────────┐   exists? ──No──►  BLOCK  (pure fabrication)
│ Layer 1: Registry  │
│ existence check    │──Yes────────────────────────┐
└───────────────────┘                               │
                                                      ▼
                                        ┌────────────────────────┐
                                        │ Layer 2: near-miss      │──flagged──► HOLD (human approval)
                                        │ vs. trusted corpus      │
                                        └────────────────────────┘
                                                      │ clean
                                                      ▼
                                        ┌────────────────────────┐
                                        │ Layer 4: metadata       │──suspicious──► HOLD (human approval)
                                        │ heuristics (age, dl's)  │
                                        └────────────────────────┘
                                                      │ clean
                                                      ▼
                                                   ALLOW

Layer 3 runs asynchronously, out of the hot path, updating the
watchlist and trusted corpus that Layer 2 reads from.
```

### Layer 1: Registry Existence

This is the cheapest check and the one that catches the ~49% pure-fabrication bucket from Spracklen et al. with a single lookup:

```python
import httpx

async def check_registry_existence(name: str, ecosystem: str) -> bool:
    """Returns True if the package name resolves on the target registry."""
    if ecosystem == "pypi":
        url = f"https://pypi.org/pypi/{name}/json"
    elif ecosystem == "npm":
        url = f"https://registry.npmjs.org/{name}"
    else:
        raise ValueError(f"unsupported ecosystem: {ecosystem}")

    async with httpx.AsyncClient(timeout=5) as client:
        resp = await client.get(url)
        return resp.status_code == 200
```

If this returns `False`, the pipeline blocks immediately — there is no ambiguity, and no reason to spend a vector search on a name that flatly doesn't exist. The interesting design work starts on the branch where this returns `True`, because that's the branch a registered-first attacker also passes through cleanly.

### Layer 2: Trusted-Corpus Near-Miss Detection

This is the first of two places Qdrant does real work, and its job is narrow: catch the case where Layer 1 passes a name that technically exists but is a close semantic match to something well-known, without being an exact string match to it. That gap is exactly where an attacker who got there first hides.

The collection is built from two sources: every `package@version` your organization has actually resolved successfully in a lockfile (a genuine, historically-verified trust signal), plus a periodic snapshot of the top-N most-downloaded packages per ecosystem (a proxy for "well-known enough that a near-miss against it is suspicious"). Each point embeds the package's name plus a short description pulled from its registry metadata.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

client = QdrantClient(url="https://your-cluster.qdrant.io", api_key="...")

client.create_collection(
    collection_name="trusted_corpus",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
client.create_payload_index(
    collection_name="trusted_corpus", field_name="ecosystem", field_schema="keyword",
)
client.create_payload_index(
    collection_name="trusted_corpus", field_name="package_name", field_schema="keyword",
)
```

Populating it is a straightforward upsert per package, tagging where the trust came from:

```python
import uuid

def upsert_trusted_package(
    client: QdrantClient, embed_fn, name: str, ecosystem: str,
    description: str, source: str,  # "lockfile" | "top_n_snapshot"
):
    vector = embed_fn(f"{name}: {description}")
    client.upsert(
        collection_name="trusted_corpus",
        points=[PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{ecosystem}:{name}")),
            vector=vector,
            payload={
                "package_name": name,
                "ecosystem": ecosystem,
                "description": description,
                "source": source,
            },
        )],
    )
```

The check itself runs only after Layer 1 has confirmed the proposed name exists. It first tests for an exact match in the trusted corpus (cheap, filter-only, no vector math needed); if there isn't one, it runs a filtered vector search and inspects whether the top hit is suspiciously close despite not being a string match:

```python
NEAR_MISS_THRESHOLD = 0.90

def check_near_miss(
    client: QdrantClient, embed_fn, name: str, ecosystem: str, description: str,
) -> dict | None:
    exact = client.query_points(
        collection_name="trusted_corpus",
        query=embed_fn(f"{name}: {description}"),
        query_filter=Filter(must=[
            FieldCondition(key="ecosystem", match=MatchValue(value=ecosystem)),
            FieldCondition(key="package_name", match=MatchValue(value=name)),
        ]),
        limit=1,
    )
    if exact.points:
        return None  # already a known-trusted package, nothing to flag

    candidates = client.query_points(
        collection_name="trusted_corpus",
        query=embed_fn(f"{name}: {description}"),
        query_filter=Filter(must=[
            FieldCondition(key="ecosystem", match=MatchValue(value=ecosystem)),
        ]),
        limit=1,
    )
    if not candidates.points:
        return None

    top = candidates.points[0]
    if top.score >= NEAR_MISS_THRESHOLD and top.payload["package_name"] != name:
        return {
            "flag": "near_miss",
            "proposed": name,
            "closest_trusted": top.payload["package_name"],
            "similarity": top.score,
        }
    return None
```

The key property this check has that Layer 1 doesn't: it does not care whether the proposed name exists in the registry. It cares whether it's suspiciously close, in embedding space, to something well-established, while not being that thing. A brand-new, never-before-seen but perfectly legitimate package will usually *not* trip this — it either won't be close to anything in the corpus, or its description will genuinely differ from anything nearby. A registered-first typosquat or conflation of two well-known trusted packages will.

**Backfilling the trusted corpus** is a one-time bulk load rather than a steady-state concern, and it's worth being explicit about the two sources feeding it, because they carry different confidence levels. Lockfile-derived entries (`source="lockfile"`) are the stronger signal — they represent packages your own build has actually resolved and shipped successfully, so a near-miss against one of these means "close to something *you* trust," not just "close to something popular." Top-N snapshot entries (`source="top_n_snapshot"`) are weaker but broaden coverage to well-known packages your org hasn't happened to depend on yet, which matters because an attacker conflating two extremely popular, widely-recognized packages (as in the `react-codeshift` case) doesn't care whether your specific lockfile references them:

```python
async def backfill_from_lockfiles(client: QdrantClient, embed_fn, lockfile_paths: list[str]):
    for path in lockfile_paths:
        for name, version, ecosystem in parse_lockfile(path):  # your own lockfile parser
            description = await fetch_registry_summary(name, ecosystem)
            upsert_trusted_package(client, embed_fn, name, ecosystem, description, source="lockfile")

async def backfill_top_n(client: QdrantClient, embed_fn, ecosystem: str, n: int = 5_000):
    for name, description in await fetch_top_n_packages(ecosystem, n):
        upsert_trusted_package(client, embed_fn, name, ecosystem, description, source="top_n_snapshot")
```

Re-running the top-N backfill on a periodic schedule (weekly is reasonable) keeps the corpus current as package popularity shifts; lockfile-derived entries update naturally as a side effect of the normal `PreToolUse` hook logging every successful install.

**Threshold calibration** on `NEAR_MISS_THRESHOLD` is the one tuning knob in this layer that actually matters, and the tradeoff is symmetric and unforgiving in both directions. Set it too low (say, 0.80) and a meaningful fraction of genuinely distinct, legitimately-named packages that happen to serve an adjacent purpose to something trusted — a competing HTTP client library, a different but related linter plugin — get flagged for human approval they don't need, and a gate that cries wolf on ordinary installs trains developers to rubber-stamp the `hold` verdict without reading it, which defeats the entire point. Set it too high (0.95+) and the gap between "flags obvious conflations like `react-codeshift`" and "flags nothing at all" narrows to the point where the layer only catches near-exact string variants Layer 1 or a simple edit-distance check would have caught anyway, making the vector search redundant.

| `NEAR_MISS_THRESHOLD` | Behavior | Risk |
|---|---|---|
| 0.80 | Flags moderately-related packages, high recall | Alert fatigue; `hold` verdicts get rubber-stamped |
| 0.90 (default above) | Flags close conflations and typosquats of well-known names | Some genuinely adjacent-but-distinct packages held unnecessarily |
| 0.95+ | Flags only near-exact semantic duplicates | Redundant with cheaper lexical checks; misses moderate-similarity conflations (the ~38% CSA bucket) |

There's no threshold that's correct in the abstract — it depends on how big and how diverse your trusted corpus is (a corpus with thousands of adjacent-but-distinct tooling packages needs a higher bar than a narrow, curated one) and how much friction your team tolerates on the `hold` path. Starting at 0.90 and adjusting based on a few weeks of real `hold`-verdict outcomes (were they overridden as false alarms, or did they catch something) is the practical path; treating 0.90 as a permanent default without that feedback loop is how thresholds drift stale.

### Layer 3: Proactive Conflation-Risk Prediction

Layer 2 is reactive by construction — it can only flag a near-miss against packages *already* in the trusted corpus, after a name has been proposed. Layer 3 does something mechanically different: instead of waiting for a proposal, it walks the trusted corpus itself looking for pairs of packages that are semantically close to each other, generates the conflation names an LLM would plausibly invent for that pair, and checks whether those names are still unclaimed — building a watchlist before any agent has proposed anything.

The first step is neighbor discovery within the corpus. For each trusted package, query its own stored vector against the same collection, restricted to the same ecosystem, and look at what sits close to it:

```python
CONFLATION_THRESHOLD = 0.85

def find_adjacent_pairs(client: QdrantClient, ecosystem: str) -> list[tuple[dict, dict, float]]:
    """Returns (package_a, package_b, similarity) for semantically adjacent
    trusted packages — candidates for name conflation."""
    all_points, _ = client.scroll(
        collection_name="trusted_corpus",
        scroll_filter=Filter(must=[FieldCondition(key="ecosystem", match=MatchValue(value=ecosystem))]),
        with_vectors=True,
        limit=10_000,
    )

    pairs = []
    seen = set()
    for point in all_points:
        neighbors = client.query_points(
            collection_name="trusted_corpus",
            query=point.vector,
            query_filter=Filter(must=[FieldCondition(key="ecosystem", match=MatchValue(value=ecosystem))]),
            limit=6,  # top 5 real neighbors + itself
        )
        for hit in neighbors.points:
            if hit.id == point.id:
                continue
            pair_key = frozenset({point.id, hit.id})
            if pair_key in seen or hit.score < CONFLATION_THRESHOLD:
                continue
            seen.add(pair_key)
            pairs.append((point.payload, hit.payload, hit.score))
    return pairs
```

`jscodeshift` and `react-codemod` are exactly the kind of pair this surfaces: both do codemod tooling for the JavaScript/React ecosystem, so their description embeddings sit close together despite sharing almost no substring. That's the case a plain exact-match or fuzzy-string index would never connect — the similarity is conceptual, not lexical, which is why this needs a vector search rather than a Levenshtein pass.

Once a pair is flagged as adjacent, the next step generates plausible blended names using a lightweight token-splicing heuristic, checks each candidate against the real registry, and records the ones that are still unclaimed:

```python
import re

def tokenize(name: str) -> list[str]:
    """Splits a package name on common delimiters and case boundaries."""
    parts = re.split(r"[-_./]", name)
    tokens = []
    for part in parts:
        tokens.extend(re.findall(r"[a-z]+|[A-Z][a-z]*|[0-9]+", part))
    return [t.lower() for t in tokens if t]

def generate_conflation_candidates(name_a: str, name_b: str) -> set[str]:
    tokens_a, tokens_b = tokenize(name_a), tokenize(name_b)
    if not tokens_a or not tokens_b:
        return set()

    candidates = {
        tokens_b[0] + tokens_a[-1],   # e.g. "react" + "codeshift"
        tokens_a[0] + tokens_b[-1],   # e.g. "js" + "codemod"
        tokens_b[0] + "-" + tokens_a[-1],
        tokens_a[0] + "-" + tokens_b[-1],
    }
    return {c for c in candidates if c not in (name_a, name_b) and len(c) > 3}

# generate_conflation_candidates("jscodeshift", "react-codemod")
# -> {"reactcodeshift", "jscodemod", "react-codeshift", "js-codemod"}
```

`react-codeshift` falls directly out of this heuristic applied to exactly the pair that produced it in the real incident — which is the intended validation, not a coincidence I'm claiming for the general case. The heuristic is deliberately simple (token-boundary splicing, not a generative model) precisely because the goal is a cheap, deterministic sweep over the trusted corpus's neighbor graph, not a second LLM in the loop generating its own hallucinations about what a hallucination might look like.

Each generated candidate then gets a Layer-1-style existence check. Anything unclaimed goes onto a watchlist collection:

```python
def update_watchlist(client: QdrantClient, embed_fn, candidate: str, ecosystem: str,
                      source_pair: tuple[str, str]):
    exists = check_registry_existence_sync(candidate, ecosystem)
    if exists:
        return  # already claimed by someone — not a live risk to preempt

    client.upsert(
        collection_name="conflation_watchlist",
        points=[PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{ecosystem}:{candidate}")),
            vector=embed_fn(candidate),
            payload={
                "candidate_name": candidate,
                "ecosystem": ecosystem,
                "source_pair": list(source_pair),
                "status": "watching",
            },
        )],
    )
```

The last piece is prioritization, and it's what keeps this from being a purely theoretical exercise generating thousands of never-hallucinated candidate names. Every time Layer 1 blocks a proposal for not existing, that rejected name gets logged, embedded, and stored in a separate rejection-log collection — an audit trail that also doubles as ground truth about what your *own* agents actually hallucinate in practice, not what's theoretically possible:

```python
def cross_reference_watchlist(client: QdrantClient) -> list[dict]:
    """Prioritizes watchlist entries that match names your own agents have
    actually proposed and had rejected, over purely theoretical candidates."""
    watchlist, _ = client.scroll(collection_name="conflation_watchlist", limit=10_000)
    prioritized = []
    for entry in watchlist:
        hits = client.query_points(
            collection_name="rejection_log",
            query=entry.vector,
            query_filter=Filter(must=[
                FieldCondition(key="ecosystem", match=MatchValue(value=entry.payload["ecosystem"])),
            ]),
            limit=1,
        )
        actually_hallucinated = bool(hits.points) and hits.points[0].score >= 0.95
        prioritized.append({**entry.payload, "confirmed_hallucination": actually_hallucinated})
    return sorted(prioritized, key=lambda e: e["confirmed_hallucination"], reverse=True)
```

Watchlist entries confirmed against your own rejection log are the ones worth acting on first — either registering defensively with a benign placeholder, exactly as Eriksen did manually for `react-codeshift`, or simply monitoring more closely. Doing this systematically, ahead of an incident, is the difference between Layer 3 and the reactive pattern the entire industry currently relies on.

### Layer 4: Metadata Heuristics

A secondary signal, not the centerpiece: for a package that exists and clears the near-miss check, a handful of registry-metadata heuristics catch packages that look opportunistic even without a name collision — age under 30-90 days, near-zero download counts, no publisher history, or a description that reads as generic filler. None of this is specific to hallucination detection; it's the same signal Agentinel and similar tools already use, included here because it's cheap and catches a different failure mode than the semantic checks above (a genuinely novel, non-conflated malicious package registered under a plausible but unrelated name).

### Why None of These Checks Are Themselves an LLM Call

It's worth being explicit about a design choice that's easy to reach for and wrong here: none of the four layers above ask a model anything. Registry existence is an HTTP status code. Near-miss and conflation detection are vector similarity scores against an embedding index. Metadata heuristics are field comparisons on JSON the registry already returned. This is deliberate, not an oversight — a gate built to catch a model's confident mistake shouldn't itself route its verdict through another model call that can be confidently wrong in a new way. Asking an LLM "does this package name look hallucinated?" is a plausible-sounding idea that reintroduces exactly the failure mode the gate exists to prevent: a second model, with no ground truth beyond its own training distribution, guessing. A registry lookup and a cosine similarity threshold are deterministic and reproducible — the same input always produces the same verdict, which matters both for debugging false positives and for the kind of auditable decision trail the Five Eyes advisory calls for. Where a model *does* legitimately help — generating the token-splicing candidates in Layer 3, or embedding a description for the vector search — its output is checked against a real, external, non-model source of truth (the registry) before it ever produces a verdict, rather than being trusted on its own.

### Structured Feedback Into the Agent's Context

Every layer above ultimately produces one of three verdicts — allow, block, or hold for human approval — and all three get returned to the agent as structured JSON, following the pattern Agentinel established, not a bare shell error:

```python
def evaluate_install(name: str, ecosystem: str) -> dict:
    if not check_registry_existence(name, ecosystem):
        log_rejection(name, ecosystem)  # feeds Layer 3's cross-reference
        return {
            "verdict": "block",
            "reason": f"'{name}' does not exist on {ecosystem}. Likely hallucination.",
            "suggestion": "Search the registry for the correct package name before retrying.",
        }

    near_miss = check_near_miss(client, embed_fn, name, ecosystem, description="")
    if near_miss:
        return {
            "verdict": "hold",
            "reason": (
                f"'{name}' exists but is a close semantic match to trusted package "
                f"'{near_miss['closest_trusted']}' (similarity {near_miss['similarity']:.2f}) "
                f"without matching its name exactly."
            ),
            "suggestion": "Requires human approval before install — possible typosquat or conflation.",
        }

    return {"verdict": "allow", "reason": "Package passed existence and near-miss checks."}
```

The `block` case gives the agent exactly what it needs to self-correct within the same turn. The `hold` case is deliberately different in kind — it doesn't tell the agent to retry, because there's nothing wrong with the *name* on its face; it tells the agent (and, downstream, a human reviewer) that a package which technically exists needs eyes on it before anything installs, which is a fundamentally different verdict from "you made this up."

**Latency budget.** A pre-install hook sits directly in the agent's task loop, so added latency is a real cost, not an afterthought. A rough per-install accounting for the synchronous path (Layers 1, 2, and 4 — Layer 3 runs out-of-band against the corpus, not per-install):

| Step | Typical latency | Notes |
|---|---|---|
| Layer 1: registry existence check | 50–150ms | Single HTTP GET against PyPI/npm; cacheable for repeated names within a session |
| Layer 2: near-miss embedding + query | 20–40ms | One embedding call plus two filtered `query_points` calls against a corpus in the tens-of-thousands-of-points range |
| Layer 4: metadata heuristics | 50–150ms | Reuses the registry response already fetched in Layer 1 where possible, rather than a second round-trip |
| **Total (allow path)** | **~150–300ms** | Adds a sub-second pause to an install the agent was about to run anyway |

This is small enough to be a non-issue for a single install, but an agent that's resolving a dozen transitive dependencies in one task can accumulate a few seconds of added latency — acceptable for the security tradeoff, but worth caching Layer 1's registry lookups within a session (the same name is often checked multiple times as an agent iterates) rather than re-fetching on every retry.

## Installing and Using Warden

Warden ships as a single pip-installable package with no required external service beyond a Qdrant instance for Layers 2 and 3 (Layer 1 and Layer 4 run against public registries directly, and work with Warden's install-time defaults even without Qdrant configured):

```bash
pip install warden-agents
# or, with a local FastEmbed model instead of an external embedding API:
pip install "warden-agents[fastembed]"
```

The package is four modules, each corresponding to a section above:

```
warden/
├── registry.py     # Layer 1: check_registry_existence
├── corpus.py        # Layer 2: TrustedCorpus, check_near_miss, backfill_from_lockfiles
├── conflation.py     # Layer 3: find_adjacent_pairs, generate_conflation_candidates, watchlist
├── heuristics.py     # Layer 4: metadata heuristics (age, downloads, publisher history)
└── hook.py           # evaluate_install(), wired to your framework's PreToolUse equivalent
```

Wiring Warden into Claude Code is a one-line hook registration in `settings.json`, pointing at the same `evaluate_install` function shown earlier in this post:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": "python -m warden.hook"}]
      }
    ]
  }
}
```

```python
from warden.hook import evaluate_install
from warden.corpus import TrustedCorpus

corpus = TrustedCorpus(url="https://your-cluster.qdrant.io", api_key="...")
verdict = evaluate_install(corpus, name="react-codeshift", ecosystem="npm")
# {"verdict": "hold", "reason": "...", "suggestion": "..."}
```

For a framework without a native pre-tool-use hook, the same `evaluate_install` call wraps directly around any function your agent loop already uses to shell out — a `subprocess.run` wrapper, a LangChain `ShellTool`, or a custom tool-calling middleware — since Warden's public interface is a plain function call, not a framework-specific plugin API.

## Worked Example: react-codeshift Through the Pipeline

Walking `react-codeshift` through this pipeline at two different points in time makes the layering concrete.

**Before January 14, 2026** (before Eriksen claimed the name, and assuming no attacker had beaten him to it): an agent executing one of the 47 forked skill files proposes `npx react-codeshift`. Layer 1 issues a registry lookup, gets a 404, and blocks immediately with `"Package does not exist on npm (Likely hallucination)."` This is the overwhelming majority case for pure fabrications and would have stopped every one of the 237 repositories' agents cold, if any of them had this gate installed. No later layer ever runs.

**In the counterfactual where an attacker registered `react-codeshift` first** (which is exactly the scenario Eriksen's defensive registration was racing against): Layer 1's existence check now returns `True` — the name resolves, the package is real, nothing about a plain registry lookup distinguishes it from any other legitimate install. This is the case a name-exists check structurally cannot catch, and it's where Layer 2 does its job: the near-miss detector embeds `react-codeshift`'s (attacker-authored, probably thin) description, searches the trusted corpus restricted to the npm ecosystem, and finds `jscodeshift` and `react-codemod` sitting close by in embedding space at a similarity comfortably above the 0.90 threshold, while the string `react-codeshift` matches neither exactly. Verdict: `hold`, routed to a human, with the specific trusted packages named in the reason field.

**Running proactively, before either scenario ever happens**: Layer 3's neighbor sweep over the trusted corpus finds `jscodeshift` and `react-codemod` as an adjacent pair (both codemod tooling for the JS/React ecosystem, similarity above the 0.85 conflation threshold) independent of any agent ever proposing anything. The token-splicing heuristic applied to that pair generates `react-codeshift` as a candidate, a registry check confirms it's unclaimed, and it lands on the watchlist — with the option to register it defensively with a placeholder before anyone, attacker or otherwise, gets there. This is the systematic version of exactly what Eriksen did by hand after the fact, run against your own dependency graph before an incident rather than in response to one.

**Layer 4**, for completeness: if the attacker-registered package had instead slipped past Layer 2 (say, if `jscodeshift` or `react-codemod` weren't in your trusted corpus yet — the cold-start problem discussed below), the package's actual metadata — registered days ago, zero prior downloads, no publisher history — would still have been a secondary signal worth surfacing, even without the semantic match.

## Mapping to the Five Eyes Agentic AI Advisory

On May 1, 2026, **CISA**, the **NSA**, and their counterparts in Australia (**ASD's ACSC**), Canada (**Canadian Centre for Cyber Security**), New Zealand (**NCSC-NZ**), and the UK (**NCSC-UK**) — jointly, the Five Eyes alliance — published [*Careful Adoption of Agentic AI Services*](https://www.cyber.gov.au/business-government/secure-design/artificial-intelligence/careful-adoption-of-agentic-ai-services), the first coordinated multigovernment guidance specifically addressing agentic AI security. Several of its recommendations map directly onto specific layers of the design above, which is a useful sanity check that the layering isn't arbitrary:

| Advisory recommendation | Implementing layer |
|---|---|
| Treat AI agents as untrusted components by default, not extensions of trusted developer judgment | The gate as a whole — every proposed install is checked regardless of which agent or model produced it |
| Maintain a trusted, pre-approved registry of third-party components and restrict agents to it | Layer 2's trusted corpus, though as a near-miss detector rather than a hard allowlist — see the cold-start tradeoff below |
| Require human approval before any agent takes a high-impact action, explicitly including installing a new dependency | The `hold` verdict path from Layers 2 and 4 |
| Give agents the minimum tool access required for their task | The hook architecture itself — the agent's `Bash` tool is mediated, not raw, at the specific point where it could install arbitrary code |
| Invest in monitoring and logging, since agentic decision processes are hard to inspect after the fact | The rejection log that Layer 3 cross-references doubles as exactly this audit trail |

The advisory's "restrict to a pre-approved registry" recommendation, read literally, would mean an agent can only ever install packages someone has already vetted — which is safer but also unworkable for teams whose agents legitimately need new, never-before-used packages regularly. Layer 2 is a deliberately softer implementation of the same principle: not a hard allowlist, but a near-miss detector that treats "close to something trusted, without being that thing" as the specific pattern worth gating on, while letting genuinely novel packages through the existence and metadata checks alone.

## Challenges and Open Problems

**Warden's trusted corpus has a real cold-start problem.** A genuinely new, legitimate package — something published last week that nobody's lockfile has ever referenced and that isn't yet in the top-N snapshot for its ecosystem — has no reason to sit close to anything in the corpus, so it neither trips the near-miss detector nor benefits from it. It sails through as a clean install, correctly, but the corpus provides zero signal about it either way. This is a genuine trust vacuum, not something the current design papers over.

**Conflation prediction is a heuristic that only catches adjacent-pair hallucinations.** Layer 3 depends on the LLM's hallucination itself being a conflation of two things close together in the trusted corpus's own embedding space. The roughly 51% of hallucinations that are purely fabricated rather than conflated (per the CSA research note's classification) have no adjacent-pair structure to predict from — Layer 3 has nothing to say about them, and they fall entirely to Layer 1's existence check plus whatever an attacker hasn't yet claimed.

**Cross-language confusion breaks the ecosystem-scoping assumption everywhere in this design.** The 8.7% of hallucinated Python packages that turned out to be valid npm names in Spracklen et al.'s data means a corpus and watchlist scoped separately per ecosystem — which is how both Layer 2's filters and Layer 3's neighbor search are built above — will simply miss a hallucination that crosses ecosystems. Catching that class would require either a shared embedding space across ecosystems (with the attendant risk of degrading precision within each one) or an explicit secondary check: does this name that doesn't exist on the requested registry exist on a *different* one, which is itself a distinct signal worth a dedicated check this design doesn't currently include.

**A single organization's install-time gate does nothing about the distribution vector that made react-codeshift a 237-repository problem in the first place.** The hallucination didn't spread because 237 different agents independently made the same mistake — it spread because one commit of skill files got forked repeatedly, with no review at any fork point. A pre-install gate stops the *execution* of a bad name; it does nothing to stop a skill file containing that name from being forked into your organization's own agent tooling next week. That's a governance problem — skill files and shared agent instructions need the same pre-adoption scrutiny a new dependency gets, before they're ever forked — not something a runtime hook can fix after the fact.

**Defensive registration at scale raises its own questions the design doesn't fully resolve.** Eriksen claiming `react-codeshift` with a benign placeholder was a one-off act by a researcher stopping a specific, already-observed incident. Layer 3 turns that into a standing recommendation to preemptively claim every unclaimed name your own heuristic generates, across every adjacent package pair in a large trusted corpus — which, run naively, could mean registering hundreds of placeholder packages per ecosystem. Registries have finite tolerance for bulk, automated namespace claiming, and a policy of registering everything the watchlist produces risks looking indistinguishable from the squatting behavior the gate exists to prevent. The more defensible version of Layer 3 treats the watchlist as a prioritized list for monitoring and, selectively, for registration of the highest-confidence candidates (the ones cross-referenced against real rejection-log hits), not a blanket claim-everything policy — but where exactly that line sits is a judgment call this post doesn't settle.

## References

- Spracklen, J., Wijewickrama, R., Sakib, A.H.M.N., Maiti, A., Viswanath, B., Jadliwala, M. "We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code-Generating LLMs." USENIX Security Symposium, 2025. [arxiv.org/abs/2406.10279](https://arxiv.org/abs/2406.10279)
- Aikido Security. "Agent Skills Are Spreading Hallucinated npx Commands." [aikido.dev/blog/agent-skills-spreading-hallucinated-npx-commands](https://www.aikido.dev/blog/agent-skills-spreading-hallucinated-npx-commands)
- Cloud Security Alliance. "Slopsquatting: AI Code Hallucinations Fuel Supply Chain Attacks." Research Note, April 19, 2026. [labs.cloudsecurityalliance.org/research/csa-research-note-slopsquatting-ai-supply-chain-20260419-csa](https://labs.cloudsecurityalliance.org/research/csa-research-note-slopsquatting-ai-supply-chain-20260419-csa/)
- Endor Labs. "State of Dependency Management 2025: Security in the AI-Code Era." [endorlabs.com/learn/state-of-dependency-management-2025](https://www.endorlabs.com/learn/state-of-dependency-management-2025)
- CISA, NSA, ASD's ACSC, Canadian Centre for Cyber Security, NCSC-NZ, NCSC-UK. "Careful Adoption of Agentic AI Services." Joint Guidance, April 30, 2026. [cyber.gov.au/business-government/secure-design/artificial-intelligence/careful-adoption-of-agentic-ai-services](https://www.cyber.gov.au/business-government/secure-design/artificial-intelligence/careful-adoption-of-agentic-ai-services)
- Sonatype. "2026 State of the Software Supply Chain: Open Source Malware." [sonatype.com/state-of-the-software-supply-chain/2026/open-source-malware](https://www.sonatype.com/state-of-the-software-supply-chain/2026/open-source-malware)
- Wikipedia. "Slopsquatting." [en.wikipedia.org/wiki/Slopsquatting](https://en.wikipedia.org/wiki/Slopsquatting)
- Qdrant. Qdrant Vector Database Documentation. [qdrant.tech/documentation](https://qdrant.tech/documentation)
