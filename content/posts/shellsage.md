---
title: "ShellSage: The Shell Translation Layer That Keeps AI Agents Sane on Windows"
date: 2026-06-15
description: "How a hybrid BM25 + dense retrieval layer with Reciprocal Rank Fusion prevents context rot in Claude Code sessions by intercepting bash→PowerShell mismatches before they ever reach the LLM."
tags: ["ai-agents", "mcp", "rag", "claude-code", "shell", "retrieval"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Large language models are increasingly used as autonomous coding agents — tools like Claude Code, GitHub Copilot, and Cursor are taking over complete software engineering workflows. These agents don't just write code; they *run* it. They `ls`, they `grep`, they `rm -rf` (carefully, one hopes). But there is a silent operational hazard lurking in these agentic pipelines that is rarely discussed: **shell heterogeneity**.

A model trained predominantly on Linux bash idioms will write `ls -la`, pipe through `grep`, and chain commands with `&&`. On a Windows developer machine, these commands fail. The agent reads the error, reasons about it, retries, fails again, inflates the context window with stack traces, and eventually halts or hallucinates a path forward. This is not a hypothetical — it is a daily tax paid by Windows developers using AI coding agents.

[ShellSage](https://github.com/inamdarmihir/shellsage) is an interception layer that translates bash→PowerShell syntax mismatches *before* the command hits the shell, preventing the retry spiral from starting. This post is a technical examination of how it works, why the problem is harder than it looks, and what makes the hybrid retrieval + rule-based approach a principled engineering choice.

## The problem: shell mismatch as a token tax

To appreciate ShellSage's design, it helps to understand precisely what goes wrong in a naive agentic session on Windows.

Consider a Claude Code session where the model is asked to "list all Python files modified in the last day." The model generates:

```bash
find . -name "*.py" -mtime -1
```

This is correct bash. On Windows, `find` is a completely different utility (it searches for strings in files, not files on disk). The shell returns an error. Claude Code reads the error into context (~2–3k tokens of error trace), reasons about it, generates a correction, which might be another bash idiom, fails again, and so on.

Each retry cycle in an agentic loop is not a flat-cost operation. Retry loops compound because each subsequent API call carries the *full accumulated context* as input tokens. A retry at turn 40 does not cost what a retry at turn 1 costs — it costs 40 turns of tokens multiplied by the retry overhead. A single multi-turn session with several failed commands can easily burn 100k+ tokens that deliver zero useful output.

The failure mode has a name in the field: **context rot**. Each irrelevant token doesn't just waste space — it dilutes the signal the model is attending to, and the degradation accelerates as the window fills. Shell error traces are particularly bad offenders: they are verbose, contain no information the model can usefully act on without retrying, and they permanently inflate every future turn in the session.

ShellSage's value proposition is containment: intercept the bad command before it produces an error trace, translate it, and deliver a correct command. **The LLM context never sees the failure.**

## Architecture overview

ShellSage sits between the AI agent and the target shell as a *pre-execution hook*. In Claude Code's terminology, this is a `PreToolUse` hook on Bash tool invocations.

```
Agent / LLM client (Claude Code, Cursor, GitHub Copilot)
                    │  bash command (potentially wrong)
                    ▼
    ┌──────────────────────────────────────────────┐
    │  ShellSage PreToolUse Hook                   │
    │  ──────────────────────────────────────────  │
    │  1. Qdrant Hybrid Search                     │
    │     └─ Dense (all-MiniLM-L6-v2, 384-dim)    │
    │     └─ Sparse (BM25, local)                  │
    │     └─ Fused via RRF                         │
    │  2. Rule-Based Fallback (60+ regex rules)    │
    │  3. Passthrough (already valid)              │
    └──────────────────────────────────────────────┘
                    │  corrected PowerShell command
                    ▼
              Target shell (PowerShell)

    PostToolUse Hook:
    └─ Record outcome → Qdrant (self-correcting memory)
```

The resolution order matters. Qdrant hybrid search runs first because it handles semantic paraphrase — the agent might write `ls -l` or `dir /a` or "list files verbosely," but all three should resolve to `Get-ChildItem -Force`. The rule-based fallback provides a zero-dependency cold-start. Passthrough handles commands already valid for the target shell.

## Component 1: rule-based fallback

ShellSage ships with 60+ pre-mapped bash→PowerShell translation rules in `rules.py`. These are deterministic regex rewrites — no model inference, no network call, no latency.

| Bash | PowerShell |
|---|---|
| `ls -la` | `Get-ChildItem -Force` |
| `grep -r "pattern" .` | `Select-String -Pattern "pattern" -Recurse` |
| `cat file.txt` | `Get-Content file.txt` |
| `rm -rf dir/` | `Remove-Item -Recurse -Force dir/` |
| `find . -name "*.py"` | `Get-ChildItem -Recurse -Filter "*.py"` |
| `echo $HOME` | `echo $env:USERPROFILE` |
| `&&` | `;` |

The rule engine is the *cold start* story. Before any user-contributed translations exist in Qdrant, the system has a useful baseline available immediately after `shellsage init`.

The limitation of pure rules is composability. A command like:

```bash
find . -name "*.py" -mtime -1 | xargs grep "import os"
```

involves multiple constructs chained together. A regex ruleset that handles each element independently may not compose correctly into a valid PowerShell pipeline. This is where hybrid retrieval becomes necessary.

## Component 2: hybrid semantic retrieval

ShellSage embeds shell command pairs (bash → PowerShell) using `all-MiniLM-L6-v2`, a 22 MB sentence transformer that produces 384-dimensional dense vectors. These are stored in Qdrant alongside a local BM25 index over the same corpus.

The choice of `all-MiniLM-L6-v2` is deliberate: it is CPU-optimized, small enough to load once per process, and performs well on short technical text. For shell command retrieval, the embedding task is to recognize that `find . -name "*.py"` and "search for python files recursively" are semantically similar — something BM25 would miss entirely (no shared tokens), but a sentence transformer handles naturally.

**BM25** (Best Matching 25) is a probabilistic sparse retrieval model scoring documents by term frequency and inverse document frequency. For shell commands, it excels at exact token matching: if the agent writes exactly `grep -rn`, BM25 will surface training examples containing those tokens with high confidence. Where BM25 fails is paraphrase: `grep -rn` versus "recursive grep with line numbers" share no tokens but mean the same thing.

The two modalities are complementary:

- **Dense (semantic):** handles paraphrase, natural language descriptions, approximate matches
- **Sparse (BM25):** handles exact token matches, precise flag names, known command strings

Neither alone is sufficient for a real agentic workload.

## Component 3: Reciprocal Rank Fusion

Combining dense and sparse retrieval results is non-trivial. Their scores live on incompatible scales: BM25 produces unbounded positive scores, while cosine similarity over unit-norm embeddings lives in $[-1, 1]$. Naive weighted averaging produces unstable results that depend heavily on corpus statistics.

**Reciprocal Rank Fusion (RRF)** solves this by operating entirely on *ranks*, not scores. For a set of retrieval systems $R$ and a document $d$:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

where $k$ is a smoothing constant (typically 60) and $\text{rank}_r(d)$ is the position of $d$ in retriever $r$'s ranked list.

Intuitively: a document appearing at position 1 in *both* the dense and sparse results scores $\frac{1}{61} + \frac{1}{61} \approx 0.033$. A document at position 1 in dense but position 100 in sparse scores $\frac{1}{61} + \frac{1}{160} \approx 0.023$. Documents that score highly in both systems naturally float to the top; neither retriever can monopolize the ranking.

The practical advantage for ShellSage: adding a new retrieval modality does not require re-tuning score weights. The rank-based fusion handles modality additions gracefully.

## Component 4: self-correcting memory via Qdrant

The most interesting architectural decision in ShellSage is the `PostToolUse` hook: after a translated command executes, the outcome is recorded back into Qdrant.

ShellSage maintains three Qdrant collections:

1. **`translations`** — successful bash→PowerShell pairs, from seeds and learned successes
2. **`failures`** — commands that failed even after translation, with their error patterns
3. **`context`** — agent session context for disambiguation

The feedback loop:

1. Agent generates `ls -la ./src`
2. ShellSage translates to `Get-ChildItem -Force ./src`
3. Command executes successfully
4. PostToolUse hook records `(bash_cmd, ps_cmd, SUCCESS)` to the `translations` collection
5. The dense embedding of the bash command is stored alongside

Over time, the corpus accumulates project-specific knowledge. If a repository has non-standard path conventions or uses Git Bash alongside PowerShell, the agent's actual successful translations become future retrieval signal. The corpus grows with use, and retrieval quality improves monotonically.

The failure collection is equally important. If ShellSage's translation still fails (wrong rule, unusual context), that failure is recorded and can flag future similar queries rather than triggering another bad translation attempt.

## Component 5: the MCP server and Claude Code hooks

ShellSage exposes two integration surfaces.

### MCP server

Started via `shellsage mcp`, the server exposes four tools to any MCP-compatible client:

```python
@mcp.tool()
def translate_command(bash_command: str) -> str: ...

@mcp.tool()
def lookup_translation(query: str) -> list[Translation]: ...

@mcp.tool()
def record_outcome(command: str, success: bool, error: str | None): ...

@mcp.tool()
def get_stats() -> dict: ...
```

### Claude Code hooks (preferred)

After `shellsage hooks install`, two Python scripts land in `.claude/hooks/`. The configuration in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse":  [{"matcher": "Bash", "hooks": [{"type": "command", "command": "python .claude/hooks/pre_tool_use.py"}]}],
    "PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "python .claude/hooks/post_tool_use.py"}]}]
  }
}
```

This is *transparent* to the agent. Claude Code writes bash; the hook translates it to PowerShell; PowerShell executes it; success is reported back. From the model's perspective, it just wrote valid shell commands.

This transparency is the critical design choice. If the agent knew translation was happening, it might attempt to compensate — writing hybrid bash/PowerShell, or second-guessing its own commands. By keeping the translation invisible, the model's bash generation stays clean and uncontaminated by Windows idioms it is not optimized to produce.

## The cold start problem

Any retrieval-augmented system faces the cold start problem: the corpus is empty at initialization, so early queries have nothing to retrieve. ShellSage addresses this with its 60-translation seed corpus, loaded via `shellsage init`.

But cold start has a subtler dimension here. Seed translations are *general-purpose* — they cover common Unix→Windows mappings. A project using domain-specific tooling (`kubectl`, `aws-cli`, custom scripts) will have translation patterns absent from the seed set. Until project-specific translations accumulate via the PostToolUse loop, the system falls back to rules or passthrough.

ShellSage's performance on a new project follows a learning curve. The first session uses only seeds and rules. By session 10, the agent's successful translations have accumulated, and retrieval quality for that project's idioms has improved substantially. The system degrades gracefully (passthrough) rather than catastrophically.

## Efficiency analysis

ShellSage's design makes latency a first-class constraint. A translation layer adding 500ms per command would be worse than no layer at all.

The system achieves ~1.5ms average query resolution through:

- **Local-only inference.** `all-MiniLM-L6-v2` is loaded once per process. Subsequent embeddings are pure CPU SIMD — no network, no GPU initialization.
- **Qdrant local mode.** Vector search is a local socket call with microsecond latency, not an HTTP request to a cloud endpoint.
- **BM25 is trivially fast.** Inverted-index lookup is $O(|q|)$ in query length. RRF fusion over rank lists adds negligible overhead.
- **Lazy model loading.** For exact-match commands, regex rules resolve the query without touching the embedding model at all.

Against this, the cost of *not* having ShellSage is measured in LLM tokens. ShellSage's benchmarks claim 100% token savings on failed-command scenarios — accurate for the case where translation succeeds. A 3-retry cycle that would cost ~45k tokens in LLM API calls costs 0 additional tokens when translation intercepts the failure.

| Scenario | Without ShellSage | With ShellSage | Savings |
|---|---|---|---|
| 1 failed PS command | ~3 retries × ~15k tokens = **45k tokens** | **0 extra tokens** | 100% |
| 10-command session (3 failures) | ~135k wasted tokens | **0 wasted tokens** | 100% |
| Error enters context | Yes — bloats all future turns | Never | Compounding |

## Open problems and limitations

ShellSage is a practical and well-engineered tool, but several limitations are worth understanding clearly.

**Multi-command pipelines.** Bash pipelines like `find . -name "*.py" | xargs grep "TODO" | sort -u` involve multiple constructs composed together. ShellSage translates individual command patterns; composing translated components into a valid PowerShell pipeline is harder. PowerShell's object pipeline (`Where-Object`, `ForEach-Object`) has different semantics from Unix text pipelines — correct composition requires understanding the data flow, not just the individual commands.

**Environment-specific context.** The same bash command can have different correct translations depending on context. `rm -rf ./dist` in a web project versus a C++ project with active file handles in `./dist` may warrant different handling. The context collection in Qdrant attempts to capture this, but disambiguation from session context alone is difficult.

**Adversarial commands.** ShellSage fixes *accidental* shell mismatches, not malicious ones. A prompt-injected instruction generating a destructive bash command will be translated faithfully into equally destructive PowerShell. Safety is entirely external to ShellSage.

**One-directional.** The current implementation assumes bash → PowerShell/CMD. The inverse — a Windows-native agent writing PowerShell that needs to run on Linux CI — is not addressed.

**Docker dependency.** The hybrid search path requires Qdrant running as a local Docker container. If the container is down, the system falls back to rules only. This is a real friction point on environments where Docker is unavailable or restricted.

## Summary

ShellSage addresses a concrete, underappreciated problem in production agentic coding: shell heterogeneity causes AI agents to generate syntactically correct but environmentally wrong commands, producing error traces that inflate LLM context and degrade session quality over time. The architectural choices — hybrid semantic+lexical retrieval, RRF fusion, a rule-based cold-start corpus, and transparent Claude Code hooks — are well-matched to the problem constraints.

What is worth generalizing: ShellSage frames a retrieval problem through the lens of *context budgets* rather than retrieval benchmarks. The primary metric is not translation accuracy in isolation, but tokens saved per agent session. As agentic systems grow in scope and cost, this framing — treating every token in the context window as a resource to be conserved, not a log to be appended — is a perspective the field would benefit from adopting more broadly.

## Further reading

- [ShellSage on GitHub](https://github.com/inamdarmihir/shellsage)
- [Reciprocal Rank Fusion (Cormack et al. 2009)](https://dl.acm.org/doi/10.1145/1571941.1572114)
- [Hybrid Search Explained — Weaviate](https://weaviate.io/blog/hybrid-search-explained)
- [The Hidden Cost Driver in Agentic Coding Sessions — Vantage](https://www.vantage.sh/blog/agentic-coding-costs)
- [Context Rot in AI Coding Agents — MindStudio](https://www.mindstudio.ai/blog/context-rot-ai-coding-agents-how-to-prevent)
