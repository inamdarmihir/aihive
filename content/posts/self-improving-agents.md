-----

## title: “Architecting Self-Improving Agents”
date: 2026-05-16
description: “A proposed architecture for agents that learn from experience - episodic memory in Markdown, selective context reveal, MCP as wiring, and SME-driven evals that close the fine-tuning loop.”
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
      EPISODE.md
      references/
        full_trace.md
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
tool get_chargeback_history(lookback=90) returned data but the
agent interpreted the 2.1% rate as within normal range. The SME
noted that for this merchant category (digital goods), 2.1% is
a HIGH-risk indicator, not MEDIUM.

## Lesson

For merchants in category_code 7372-7379 (software/digital goods),
chargeback thresholds are: LOW < 0.5%, MEDIUM 0.5-1.5%, HIGH > 1.5%.
The global threshold table does not apply to these categories.

## SME correction note

The chargeback rate is category-sensitive. The model needs to look
up the category-specific table, not the global one. Lookup added to
references/sme_corrections.md.
```

The frontmatter `description` field is the key design decision. It’s written to be *retrievable* — a 1–3 sentence summary compact enough that hundreds can be scanned in a single context window. The body is loaded only after the episode is selected.

### Progressive Disclosure via Frontmatter Routing

<#progressive-disclosure-via-frontmatter-routing>

This pattern is adapted from Anthropic’s Agent Skills format. The retriever operates at three resolution levels:

|Level      |What’s loaded                       |Tokens (approx.)|When                                       |
|-----------|------------------------------------|----------------|-------------------------------------------|
|Description|Frontmatter `description` field only|~50             |Always, during initial scan                |
|Summary    |Full `EPISODE.md` body              |~400–800        |After episode selected by retriever        |
|Detail     |`references/full_trace.md` + others |~5,000–40,000   |Only if episode body explicitly requests it|

The agent never sees the `references/` directory unless `EPISODE.md` contains a line like `See references/sme_corrections.md for the corrected threshold table.` This keeps token expenditure proportional to how analogous a prior episode actually is.

### Writing Episodes After a Run

<#writing-episodes-after-a-run>

Episode writing happens as a post-run tool call, not inline. The agent calls `memory_server.write_episode(trace)` after the task completes. The memory server runs a lightweight extraction pass using a secondary LLM call:

```
Given this agent trace, produce a structured EPISODE.md with:
- frontmatter: task_type, outcome (success/partial_success/failure),
  tags (3-5 concepts), description (1-3 sentences)
- body: "What was attempted", "What went wrong" (if applicable),
  "Lesson" (if clearly derivable)

Do not speculate. If no lesson is clear, leave that section empty.

Trace: {trace}
```

This separation matters: the agent that ran the task is not the same process that writes the episode. A failed run can still produce a well-structured episode. The extraction cost is fixed (~300–600 tokens), not proportional to what the agent did.

## Component Two: Selective Context Reveal at Inference Time

<#component-two-selective-context-reveal-at-inference-time>

### Two-Stage Retrieval

<#two-stage-retrieval>

When a new task arrives, retrieval runs in two stages before the agent acts.

**Stage 1: Description scan.** The retriever loads only the `description` field from all episodes in the relevant task family (filtered by `task_type`). At 50 tokens per episode, a store of 1,000 episodes costs 50,000 tokens — feasible in a single context window pass. The LLM scores each description for analogical relevance and returns a ranked list of episode IDs.

**Stage 2: Body loading.** For the top-K episodes (K=3 by default), the retriever loads the full `EPISODE.md` body. If any body references a `references/` file that’s clearly needed, the retriever loads that too.

What’s novel is the *unit of retrieval*: a structured episode with outcome metadata, not a text chunk. The `outcome` and `sme_score` fields let the retriever weight analogous *failures* more heavily than analogous *successes* when the current task resembles a historically tricky pattern.

### Budget-Aware Loading

<#budget-aware-loading>

The retriever tracks remaining context budget. If the agent’s task description plus tool definitions already consume 80% of the context window, stage 2 is compressed: only the frontmatter + “Lesson” section of each episode is loaded. The retriever signals this via a note: `[Memory: 3 episodes loaded in summary mode. Full bodies available via memory_server.get_episode(id).]`

### The Relevance Scoring Function

<#the-relevance-scoring-function>

The scoring prompt used in stage 1:

```
You are scoring episodic memory entries for relevance to a current task.

Current task: {current_task_description}

Rate each episode on two dimensions:
1. Task similarity (0-3): how similar is the episode's task?
2. Outcome utility (0-2): would this episode's lesson change how
   the agent approaches the current task?

Return a ranked JSON list: [{"id": "...", "task_sim": N, "outcome_util": N}, ...]

Episode descriptions: {descriptions}
```

The composite score is `task_sim + outcome_util`. Ties are broken by recency. A domain team can look at a set of retrievals, find mis-rankings, and adjust the prompt.

## Component Three: MCP as the Wiring Layer

<#component-three-mcp-as-the-wiring-layer>

The architecture exposes memory, evaluation, and training as three independent MCP servers. The agent host connects to all three via standard MCP client configuration.

### Memory Server

<#memory-server>

**Resources** (app-controlled, readable by the agent):

- `memory://episodes/{task_type}` — all episode descriptions for a task type
- `memory://episodes/{id}/body` — the full `EPISODE.md` for a specific episode
- `memory://episodes/{id}/references/{filename}` — a specific reference file

**Tools** (model-controlled, callable by the agent):

- `write_episode(trace: str, task_type: str) -> episode_id`
- `update_episode_lesson(episode_id: str, lesson: str)`
- `search_episodes(query: str, task_type: str, top_k: int) -> list[EpisodeSummary]`

The memory server is a thin filesystem wrapper. The `episodes/` directory is a Git repo; the server calls `git commit` on each `write_episode`. Episode history is `git log`.

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

**Prompts** (user-controlled templates):

- `rubric/{task_type}` — the current evaluation rubric, loaded from `rubrics/{task_type}.md`

**Tools**:

- `judge_trace(trace: str, task_type: str) -> JudgmentResult`
- `submit_for_sme_review(episode_id: str, reason: str)`
- `record_sme_judgment(episode_id: str, score: int, corrections: str, promote: bool)`

### Training Server

<#training-server>

**Tools**:

- `enqueue_for_finetuning(episode_id: str)`
- `trigger_training_run(adapter_name: str)`
- `get_adapter_status() -> AdapterStatus`
- `rollback_adapter(version: str)`

**Resources**:

- `training://pending_batch` — episodes queued for training
- `training://adapter_versions` — version history of all adapters

### Why MCP and Not a Custom API

<#why-mcp-and-not-a-custom-api>

Every MCP-compatible agent host (Claude Desktop, Claude Code, Cursor, Copilot, Gemini CLI as of late 2025) connects to these servers without integration code. The agent discovers tools at startup via the standard `tools/list` call. Swapping the LoRA backend from Axolotl to unsloth is a training-server internal change; the agent interface doesn’t change.

The deeper reason is that MCP’s **roots** and **sampling** primitives are useful here. Roots let the memory server expose `episodes/` as a permission-scoped filesystem resource so the agent can navigate subdirectories as code rather than making round-trip resource calls for each file — the pattern Anthropic documented for large tool sets ([Anthropic Engineering, 2025](https://www.anthropic.com/engineering/code-execution-with-mcp)). Sampling lets the eval server make LLM calls back through the agent host without managing its own API key.

## Component Four: SME-Driven Evaluation

<#component-four-sme-driven-evaluation>

### The Rubric as a Markdown Artifact

<#the-rubric-as-a-markdown-artifact>

The rubric for a task type lives in `rubrics/{task_type}.md` — a versioned Markdown file in the same Git repo as the episodes:

```markdown
---
version: 2.1
last_updated: 2026-04-15
---

# Rubric: Merchant Risk Assessment

## Criteria

### 1. Signal Completeness (0-3)
- 3: All mandatory signals retrieved (revenue trend, chargeback rate
     with category-specific threshold, complaint rate, industry class)
- 2: 3 of 4 signals retrieved
- 1: 2 of 4 signals retrieved
- 0: Fewer than 2 signals

### 2. Threshold Accuracy (0-2)
- 2: Category-specific thresholds applied correctly
- 1: Global thresholds applied (wrong for digital goods)
- 0: No threshold reasoning

### 3. Rationale Quality (0-2)
- 2: Risk tier explicitly justified by named signals and thresholds
- 1: Tier stated without full justification
- 0: No rationale

### 4. Completeness of Response (0-1)
- 1: Response includes recommended next actions
- 0: Response terminates at risk tier only

## Passing Thresholds

Auto-promote: total >= 7 | SME review: 4-6 | Discard: < 4
```

The `version` field lets the eval server record which rubric version was used for any given judgment, which matters when you want to re-judge old episodes under a revised rubric.

### LLM-as-Judge Calibration

<#llm-as-judge-calibration>

The judge prompt:

```
You are evaluating an agent's output against the rubric below.
Score each criterion independently. Show your reasoning before
stating the score. Do not let your overall impression affect
individual criterion scores.

Rubric: {rubric_body}
Agent trace: {trace}

Return JSON: {"criterion_scores": {...}, "total": N, "reasoning": "..."}
```

Calibration follows the **SMELL** framework ([Quotient AI, 2024](https://blog.quotientai.co/subject-matter-expert-language-liaison-smell)): an SME labels 30–50 traces before the judge is deployed. The labeled set is split 70/30 into calibration and held-out. The judge prompt is iterated until Krippendorff’s alpha between judge and SME labels exceeds 0.7 on the held-out set.

### The Promotion Decision

<#the-promotion-decision>

The promotion decision is a three-way split:

- **Auto-promote** (total >= threshold): episode written to the pending training batch immediately.
- **SME queue** (score in review band): episode metadata, judge reasoning, and a link to the full trace are surfaced in the SME review interface — a Markdown-rendered list of queued episodes with inline edit capability.
- **Discard** (below discard threshold): episode retained in the memory store (failures are valuable for retrieval) but excluded from the training set.

SMEs edit `EPISODE.md` directly and call `eval_server.record_sme_judgment(...)`. The corrections field is written into `references/sme_corrections.md` and linked from the episode body. From this point the lesson includes the SME’s direct language, not an LLM paraphrase.

## Component Five: Gated LoRA Fine-Tuning

<#component-five-gated-lora-fine-tuning>

### Trace Formatting for SFT

<#trace-formatting-for-sft>

Each promoted episode is formatted as a (system, user, assistant) training example: original task as the user message, corrected agent output as the assistant message, base system prompt as the system message. If the SME only wrote a correction note and didn’t produce a full corrected output, a secondary LLM pass generates the ideal response:

```
Given this agent trace and the SME correction note, write the
ideal assistant response. Reflect the SME correction exactly.

Original task: {task}
Agent response: {agent_response}
SME correction: {sme_correction}

Write only the ideal response. Do not explain the correction.
```

This synthetic ideal-response generation is the weakest link in the pipeline. The mitigation is to have the SME spot-check a random 10% of generated ideal responses before each training run.

### The Training Gate

<#the-training-gate>

Fine-tuning runs on a fixed cadence — weekly for high-volume tasks, monthly for low-volume. Before any adapter is promoted to production, it must clear two gates:

**Gate 1: Domain improvement.** The new adapter must score higher than the current production adapter on a held-out SME eval set. If it does not improve, the cycle is skipped and traces accumulate for the next run.

**Gate 2: Base capability non-regression.** LoRA learns less and forgets less than full fine-tuning ([Biderman et al., 2024](https://arxiv.org/abs/2405.09673)), but “forgets less” is not “forgets nothing.” If any general-capability benchmark drops more than 2% relative, the adapter is rejected.

```python
def gate_adapter(new_adapter, current_adapter, held_out_evals, base_benchmarks):
    domain_score_new = run_domain_eval(new_adapter, held_out_evals)
    domain_score_current = run_domain_eval(current_adapter, held_out_evals)

    if domain_score_new <= domain_score_current:
        return GateResult.REJECT, "No domain improvement"

    for benchmark in base_benchmarks:
        score_new = run_benchmark(new_adapter, benchmark)
        score_current = run_benchmark(current_adapter, benchmark)
        relative_drop = (score_current - score_new) / score_current
        if relative_drop > benchmark.tolerance:
            return GateResult.REJECT, f"Regression on {benchmark.name}"

    return GateResult.PROMOTE, "Passed both gates"
```

### Adapter Versioning and Rollback

<#adapter-versioning-and-rollback>

Every adapter version is tagged with a manifest:

```json
{
  "adapter_id": "merchant-risk-v7",
  "base_model": "claude-3-haiku-20240307",
  "training_episodes": ["2026-04-01T...", "2026-04-08T...", "..."],
  "domain_score": 0.84,
  "base_benchmark_deltas": {"reasoning": -0.01, "instruction": 0.00},
  "promoted_at": "2026-05-01T09:00:00Z"
}
```

Rollback is a single `training_server.rollback_adapter(version)` call. Because each manifest records which episodes were used, a regression can be traced to a specific batch and the adapter retrained from the remaining history.

## The Three Time-Scales of Improvement

<#the-three-time-scales-of-improvement>

The architecture operates at three time-scales, each with its own failure mode and fallback:

**Per-turn (in-context).** The agent retrieves relevant episodes and acts. No weight updates, no persistence after the run. This is the floor — it works on the first task and improves as the memory store grows. Failure mode: memory store is empty or retrieval quality is poor. Fallback: base model without memory.

**Per-episode (memory store update).** After each run, a new episode is written. The agent is immediately better at analogous tasks on the next run, without waiting for a fine-tune. Failure mode: episode writer produces low-quality summaries or the judge is miscalibrated. Fallback: SME edits `EPISODE.md` directly in the Git repo.

**Per-cycle (adapter refresh).** Every week or month, a new LoRA adapter is trained and gated. This bakes stable lessons into weights, reducing context load for very common patterns. Failure mode: adapter fails both gates. Fallback: previous adapter stays in production; memory store continues accumulating.

The three layers degrade gracefully. If fine-tuning is disabled, memory still works. If memory is empty, the base model still functions. If the judge is miscalibrated, the SME queue catches the damage before it reaches training.

## End-to-End Data Flow

<#end-to-end-data-flow>

A complete cycle, from task to trained adapter:

1. **Task arrives.** Agent host receives a new merchant risk assessment request.
1. **Memory scan.** Retriever calls `search_episodes(query, task_type, top_k=3)`. Returns three episode summaries.
1. **Body loading.** Retriever loads full `EPISODE.md` for each. One references `sme_corrections.md`; retriever loads it. Total context addition: ~2,500 tokens.
1. **Agent acts.** Agent runs the task with the current LoRA adapter and retrieved context.
1. **Episode written.** Agent calls `write_episode(trace, task_type)`. Memory server extracts and commits.
1. **Judge runs.** Eval server scores the trace: 6/8 — in the SME review band.
1. **SME queued.** Eval server calls `submit_for_sme_review(episode_id, reason="Score 6/8; threshold accuracy scored 1/2.")`.
1. **SME reviews.** SME edits the lesson section and records judgment: score 7/8, promote=True.
1. **Training batch updated.** Episode joins the pending batch via `enqueue_for_finetuning(episode_id)`.
1. **Weekly training run.** LoRA trained on 47 promoted episodes from the past week. Both gates clear. Adapter promoted.
1. **Next task.** New agent run uses the updated adapter and memory store.

## Implementation Notes

<#implementation-notes>

**Use a smaller model for episode extraction.** The episode writer runs on every trace. A 7B model handles the structured extraction prompt reliably. Save the larger model for the task itself and for stage-1 relevance scoring, which requires more nuanced analogical judgment.

**Git is the right storage backend, not a vector database.** The memory store doesn’t need millisecond retrieval latency — it runs once per task, upstream of the agent’s actual work. Git gives you history, diffability, branch-based experimentation, and the ability for SMEs to submit corrections as PRs. A vector index can be layered on top for the stage-1 scan if the store exceeds ~5,000 episodes, but don’t start there.

**Pin rubric versions to judge runs.** When the rubric changes, old scores become incomparable to new scores. Store `rubric_version` alongside every score. If you re-run the judge under a new rubric, record it as a new judgment rather than overwriting the old one.

**Keep adapter batch sizes small.** A weekly run on 30–50 promoted traces is preferable to a quarterly run on 500. Small batches keep the feedback loop fast and make it easier to attribute regressions to specific episodes.

**Don’t train on retrieved context, only on output.** Training examples should be `(task_description, ideal_output)` pairs, not `(task_description + retrieved_episodes, ideal_output)`. The fine-tune should bake the *lesson* into weights so the model needs less scaffolding over time.

## Challenges and Open Problems

<#challenges-and-open-problems>

**When to bake a lesson into weights vs. keep it in memory.** There’s no principled answer. The heuristic I use: if the same correction appears in 5+ consecutive episodes, the lesson has stabilized enough for fine-tuning. If it’s appearing in ~20% of episodes, it’s a systematic gap worth training on. If it’s appearing in fewer than 5% of cases, leave it in memory — it’s probably an edge case that a fine-tune might over-fit to.

**Catastrophic forgetting under continual adaptation.** LoRA forgets less than full fine-tuning ([Biderman et al., 2024](https://arxiv.org/abs/2405.09673)), and **CURLoRA** ([Fawi, 2024](https://arxiv.org/abs/2408.14572)) and **I-LoRA** ([Chen et al., 2024](https://arxiv.org/abs/2402.18865)) push further via matrix decomposition and interpolation regularization. A system fine-tuning weekly for a year is still uncharted territory; maintaining a rollback-to-base capability at all times is the safe default.

**Privacy of episodic memory.** A trace contains everything the agent saw — potentially user PII, proprietary data, partial credentials from tool outputs. Mitigations: run a redaction pass at write time; store memory in per-tenant directories behind auth-scoped MCP roots; render redacted versions in the SME UI. None of these are complete solutions; the right answer depends heavily on deployment context.

**Model collapse under recursive self-curation.** If the judge model is from the same family as the task model, and the task model is fine-tuned on judge-curated data, you’re creating a feedback loop that can amplify biases and erode output diversity. Safeguards: keep an SME-labeled ground-truth eval set that is never used for training; measure output diversity metrics over adapter versions; re-anchor the rubric to SME labels every N cycles.

**Cold start.** A new domain with no episodes needs either human demonstrations or a low-stakes bootstrapping period with heavy SME review. For enterprise domains, the SME often has a backlog of past human-handled cases that can seed the memory store before the agent runs a single live task.

## References

<#references>

- Shinn et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- Wang et al. (2023). *Voyager: An Open-Ended Embodied Agent with Large Language Models*. [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)
- Gulcehre et al. (2023). *Reinforced Self-Training (ReST) for Language Modeling*. [arXiv:2308.08998](https://arxiv.org/abs/2308.08998)
- Zelikman et al. (2022). *STaR: Bootstrapping Reasoning With Reasoning*. [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Biderman et al. (2024). *LoRA Learns Less and Forgets Less*. [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
- Fawi (2024). *CURLoRA: Stable LLM Continual Fine-Tuning*. [arXiv:2408.14572](https://arxiv.org/abs/2408.14572)
- Chen et al. (2024). *I-LoRA: Continual Domain Adaptation for LLMs*. [arXiv:2402.18865](https://arxiv.org/abs/2402.18865)
- Park et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. UIST 2023. [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
- Packer et al. (2023). *MemGPT: Towards LLMs as Operating Systems*. [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- Quotient AI (2024). *SMELL: Subject-Matter Expert Language Liaison*. [blog.quotientai.co](https://blog.quotientai.co/subject-matter-expert-language-liaison-smell)
- Anthropic Engineering (2025). *Code Execution with MCP*. [anthropic.com](https://www.anthropic.com/engineering/code-execution-with-mcp)
