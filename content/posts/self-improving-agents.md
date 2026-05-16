-----

## title: Architecting Self-Improving Agents
date: 2026-05-16
description: A proposed architecture for agents that learn from experience — episodic memory in Markdown, selective context reveal, MCP as wiring, and SME-driven evals that close the fine-tuning loop.
tags: [agents, fine-tuning, memory, mcp, lora, self-improvement]
author: Mihir Inamdar
showToc: true
math: true

Most agents deployed in production are amnesiac. Each run begins from the same base model, the same system prompt, and zero memory of what happened last time. The SME who corrected a bad output on Tuesday has no guarantee the agent behaves differently on Thursday. The engineer who noticed a systematic failure pattern writes it in a Slack thread, not anywhere the agent can read.

This post proposes a concrete architecture to fix that. The core idea is to treat every agent run as a structured learning event — writing it to a Markdown episodic memory store, surfacing relevant prior episodes selectively at inference time, routing curation and training through MCP servers, and periodically closing the loop with a LoRA fine-tune over SME-approved traces. I’ll describe each component in enough detail to implement it.

I’m focusing on the *specialized* case: an agent that does a particular class of tasks repeatedly (LLM eval scoring, risk assessment, technical review, document extraction — anything where there’s a domain expert who can tell good from bad). The approach is less interesting for one-off general assistants and most interesting for agents that accumulate hundreds of runs on a narrow task family.

## The Problem with Amnesiac Agents

<#the-problem-with-amnesiac-agents>

Before describing the architecture, it’s worth being precise about what’s broken.

**No durable lesson storage.** **Reflexion** ([Shinn et al., 2023](https://arxiv.org/abs/2303.11366)) demonstrated that verbal self-reflection stored in a memory buffer can improve performance across trials. But the buffer is in-process and ephemeral. When the process ends, the lessons end. Every production deployment I’m aware of throws this away.

**No selective reveal.** Vector-store RAG retrieves the most embedding-similar chunks. This is the wrong retrieval criterion for episodic memory. What you want is: *which prior episode is most analogous to the current task, and how much of its detail does the agent need right now?* A failed run from last week should surface — but maybe only its summary and the SME’s correction note, not the full 40-turn trace. The current chunk-level similarity approach has no notion of episode structure, outcome, or relevance budget.

**No curation loop.** Self-play and self-training methods (**ReST**, [Gulcehre et al., 2023](https://arxiv.org/abs/2308.08998); **STaR**, [Zelikman et al., 2022](https://arxiv.org/abs/2203.14465)) filter traces via a reward model or verifier before training. But in specialized domains, no reward model exists — a domain expert exists. The architecture needs to route SME attention efficiently, not assume it away.

**Fine-tuning cadence mismatch.** LoRA adapters are cheap enough ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) that a weekly fine-tune is realistic. But nothing in standard agent frameworks triggers this, formats the data, or runs the safety checks that prevent catastrophic forgetting.

The architecture below addresses all four.

## System Overview

<#system-overview>

Five components, wired by MCP:

1. **Episodic Memory Store** — a Git-backed directory of structured Markdown files, one per agent run.
1. **Selective-Reveal Retriever** — a two-stage retriever that first scans episode summaries, then loads detail on demand.
1. **MCP Wiring** — three servers (memory, eval, training) that expose these components as tools and resources to any MCP-compatible agent host.
1. **SME Eval Loop** — a rubric-driven judge that scores traces, queues borderline cases for SME review, and promotes approved traces into the training set.
1. **Gated LoRA Fine-Tuning** — a batched fine-tune over curated traces, gated by a non-regression check before promotion to production.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Host (MCP client)                  │
│                                                                 │
│   Task In ──► [Selective Retriever] ──► [Agent + LoRA Adapter]  │
│                      │  ▲                        │              │
│                      │  │ load episode detail    │ trace out    │
│                      ▼  │                        ▼              │
│           ┌──────────────────┐      ┌──────────────────────┐    │
│           │  memory-server   │      │    eval-server       │    │
│           │  (MCP Resources) │      │  (MCP Tools+Prompts) │    │
│           └──────────────────┘      └──────────────────────┘    │
│                   │ write episode              │ promote         │
│                   ▼                            ▼                 │
│           ┌──────────────────────────────────────────────┐      │
│           │          Episodic Memory Store (Git)         │      │
│           │    episodes/  ←──── SME edits (PR/commit)    │      │
│           └──────────────────────────────────────────────┘      │
│                                    │ training set               │
│                                    ▼                             │
│                          ┌──────────────────┐                   │
│                          │  training-server │                   │
│                          │  (LoRA adapter)  │                   │
│                          └──────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

*Overview of the self-improving agent architecture. Three MCP servers expose memory, eval, and training as protocol-native resources and tools.*

## Component One: Episodic Memory as Markdown

<#component-one-episodic-memory-as-markdown>

### The Episode File Format

<#the-episode-file-format>

Each agent run produces one episode file. The directory structure is:

```
memory/
  episodes/
    2026-05-01T14:32:00-task-merchant-risk-eval/
      EPISODE.md          ← always loaded when this episode is retrieved
      references/
        full_trace.md     ← only loaded on demand
        tool_calls.json
        sme_corrections.md
```

`EPISODE.md` has a mandatory frontmatter block and a structured body:

```markdown
---
task_type: merchant-risk-assessment
outcome: partial_success
sme_score: 3
sme_score_max: 5
tags: [merchant-risk, false-negative, missing-signal]
timestamp: 2026-05-01T14:32:00Z
description: >
  Risk assessment for a mid-tier payments merchant. Agent missed
  the chargeback velocity signal and rated risk as MEDIUM when
  SME determined HIGH. Correction: always check 90d chargeback
  rate before finalizing risk tier.
---

## What was attempted

The agent ran a standard risk evaluation using the merchant's
transaction history and public filings. Three signals were
checked: revenue trend, industry classification, and customer
complaint rate.

## What went wrong

The 90-day chargeback velocity signal was not retrieved. The
tool `get_chargeback_history(lookback=90)` returned data but
the agent interpreted the 2.1% rate as within normal range.
The SME noted that for this merchant category (digital goods),
2.1% is a HIGH-risk indicator, not MEDIUM.

## Lesson

For merchants in category_code 7372–7379 (software/digital goods),
chargeback thresholds are: LOW < 0.5%, MEDIUM 0.5–1.5%, HIGH > 1.5%.
The global threshold table does not apply to these categories.

## SME correction note

> "The chargeback rate is category-sensitive. The model needs
> to look up the category-specific table, not the global one.
> I've added the lookup to `references/sme_corrections.md`."
```

The frontmatter `description` field is the key design decision. It’s written to be *retrievable* — a 1–3 sentence summary of the task and the salient outcome, compact enough that hundreds of them can be scanned in a single context window. The body is loaded only after the episode is selected for retrieval.

### Progressive Disclosure via Frontmatter Routing

<#progressive-disclosure-via-frontmatter-routing>

This pattern is adapted from Anthropic’s Agent Skills format. The retriever operates at three resolution levels:

|Level      |What’s loaded                       |Tokens (approx.)|When                                       |
|-----------|------------------------------------|----------------|-------------------------------------------|
|Description|Frontmatter `description` field only|~50             |Always, during initial scan                |
|Summary    |Full `EPISODE.md` body              |~400–800        |After episode selected by retriever        |
|Detail     |`references/full_trace.md` + others |~5,000–40,000   |Only if episode body explicitly requests it|

The agent never sees the `references/` directory unless `EPISODE.md` contains a line like `See references/sme_corrections.md for the corrected threshold table.` — at which point the retriever loads it. This keeps token expenditure proportional to how analogous a prior episode actually is.

### Writing Episodes After a Run

<#writing-episodes-after-a-run>

Episode writing happens as a post-run tool call, not inline. The agent calls `memory_server.write_episode(trace)` after the task completes. The memory server runs a lightweight extraction pass using a secondary LLM call:

```
Given this agent trace, produce a structured EPISODE.md with:
- frontmatter: task_type (classify from taxonomy), outcome
  (success/partial_success/failure), tags (3–5 relevant concepts),
  description (1–3 sentences: what was the task, what was the key
  outcome or failure)
- body sections: "What was attempted", "What went wrong" (if
  applicable), "Lesson" (if a clear lesson is extractable)

Do not speculate about lessons you cannot derive directly from
the trace. If no lesson is clear, leave the Lesson section empty.

Trace:
{trace}
```

This separation matters: the agent that ran the task is not the same process that writes the episode. A failed agent run can still produce a well-structured episode. The episode writer has full access to the trace and produces a summary at a fixed cost (~300–600 tokens for the extraction call), not proportional to the complexity of what the agent did.

## Component Two: Selective Context Reveal at Inference Time

<#component-two-selective-context-reveal-at-inference-time>

### Two-Stage Retrieval

<#two-stage-retrieval>

When a new task arrives, retrieval runs in two stages before the agent acts.

**Stage 1: Description scan.** The retriever loads only the `description` field from all episodes in the relevant task family (filtered by `task_type` from the frontmatter). At 50 tokens per episode, a store of 1,000 episodes costs 50,000 tokens — feasible in a single context window pass. The LLM scores each description for analogical relevance to the current task and returns a ranked list of episode IDs.

**Stage 2: Body loading.** For the top-K episodes (K=3 by default, configurable), the retriever loads the full `EPISODE.md` body. The agent reads these before acting. If any episode body references a `references/` file that’s clearly needed (e.g., a correction table), the retriever loads that file too.

This two-stage design is not novel in isolation — it’s essentially a coarse-to-fine retrieval. What’s novel is the *unit of retrieval*: a structured episode with outcome metadata, not a text chunk. The `outcome` and `sme_score` fields let the retriever weight analogous *failures* more heavily than analogous *successes* when the current task resembles a historically tricky pattern.

### Budget-Aware Loading

<#budget-aware-loading>

The retriever tracks remaining context budget. If the agent’s task description plus tool definitions already consume 80% of the context window, stage 2 is compressed: only the frontmatter + “Lesson” section of each episode is loaded, not the full body. The retriever signals this compression to the agent via a note in the context: `[Memory: 3 episodes loaded in summary mode due to context constraints. Full bodies available via memory_server.get_episode(id).]`

### The Relevance Scoring Function

<#the-relevance-scoring-function>

The scoring prompt used in stage 1:

```
You are scoring episodic memory entries for relevance to a current task.

Current task:
{current_task_description}

Rate each episode description below on two dimensions:
1. Task similarity (0–3): how similar is the episode's task to the current task?
2. Outcome utility (0–2): would knowing about this episode's outcome/lesson
   change how the agent approaches the current task?

Return a ranked JSON list: [{"id": "...", "task_sim": N, "outcome_util": N}, ...]

Episode descriptions:
{descriptions}
```

The composite score is `task_sim + outcome_util`. Ties are broken by recency. This is simple enough to tune: a domain team can look at a set of retrievals, find the mis-rankings, and adjust the prompt.

## Component Three: MCP as the Wiring Layer

<#component-three-mcp-as-the-wiring-layer>

The architecture exposes memory, evaluation, and training as three independent MCP servers. The agent host connects to all three via standard MCP client configuration.

### Memory Server

<#memory-server>

The memory server exposes:

**Resources** (app-controlled, readable by the agent):

- `memory://episodes/{task_type}` — returns all episode descriptions for a task type, formatted for stage-1 scanning
- `memory://episodes/{id}/body` — returns the full `EPISODE.md` for a specific episode
- `memory://episodes/{id}/references/{filename}` — returns a specific reference file

**Tools** (model-controlled, callable by the agent):

- `write_episode(trace: str, task_type: str) → episode_id` — runs the extraction pass and writes the episode file
- `update_episode_lesson(episode_id: str, lesson: str)` — lets the agent or SME append a lesson post-hoc
- `search_episodes(query: str, task_type: str, top_k: int) → list[EpisodeSummary]` — the stage-1 retrieval call

The memory server is a thin filesystem wrapper. The `episodes/` directory is a Git repo; the server calls `git commit` on each `write_episode` call with the episode ID as the commit message. Episode history is therefore `git log`.

```python
class MemoryServer:
    def __init__(self, episodes_dir: Path):
        self.episodes_dir = episodes_dir
        self.repo = git.Repo(episodes_dir)

    def write_episode(self, trace: str, task_type: str) -> str:
        episode_id = f"{datetime.utcnow().isoformat()}-{task_type}"
        episode_dir = self.episodes_dir / episode_id
        episode_dir.mkdir()

        episode_md = self._extract_episode(trace)
        (episode_dir / "EPISODE.md").write_text(episode_md)

        refs_dir = episode_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "full_trace.md").write_text(trace)

        self.repo.index.add([str(episode_dir)])
        self.repo.index.commit(f"episode: {episode_id}")
        return episode_id
```

### Eval Server

<#eval-server>

The eval server exposes:

**Prompts** (user-controlled templates):

- `rubric/{task_type}` — the current evaluation rubric for this task type, loaded from `rubrics/{task_type}.md`

**Tools**:

- `judge_trace(trace: str, task_type: str) → JudgmentResult` — runs the LLM-as-judge pass and returns a score + reasoning
- `submit_for_sme_review(episode_id: str, reason: str)` — queues an episode for human review
- `record_sme_judgment(episode_id: str, score: int, corrections: str, promote: bool)` — writes SME judgment back to the episode file and opt
