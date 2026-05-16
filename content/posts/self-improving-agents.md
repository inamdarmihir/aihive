# Architecting Self-Improving Agents

Date: May 2026 | Estimated Reading Time: 35 min | Author: [Your Name]

-----

**Table of Contents**

- [The Problem with Amnesiac Agents](#the-problem-with-amnesiac-agents)
- [System Overview](#system-overview)
- [Component One: Episodic Memory as Markdown](#component-one-episodic-memory-as-markdown)
  - [The Episode File Format](#the-episode-file-format)
  - [Progressive Disclosure via Frontmatter Routing](#progressive-disclosure-via-frontmatter-routing)
  - [Writing Episodes After a Run](#writing-episodes-after-a-run)
- [Component Two: Selective Context Reveal at Inference Time](#component-two-selective-context-reveal-at-inference-time)
  - [Two-Stage Retrieval](#two-stage-retrieval)
  - [Budget-Aware Loading](#budget-aware-loading)
  - [The Relevance Scoring Function](#the-relevance-scoring-function)
- [Component Three: MCP as the Wiring Layer](#component-three-mcp-as-the-wiring-layer)
  - [Memory Server](#memory-server)
  - [Eval Server](#eval-server)
  - [Training Server](#training-server)
  - [Why MCP and Not a Custom API](#why-mcp-and-not-a-custom-api)
- [Component Four: SME-Driven Evaluation](#component-four-sme-driven-evaluation)
  - [The Rubric as a Markdown Artifact](#the-rubric-as-a-markdown-artifact)
  - [LLM-as-Judge Calibration](#llm-as-judge-calibration)
  - [The Promotion Decision](#the-promotion-decision)
- [Component Five: Gated LoRA Fine-Tuning](#component-five-gated-lora-fine-tuning)
  - [Trace Formatting for SFT](#trace-formatting-for-sft)
  - [The Training Gate](#the-training-gate)
  - [Adapter Versioning and Rollback](#adapter-versioning-and-rollback)
- [The Three Time-Scales of Improvement](#the-three-time-scales-of-improvement)
- [End-to-End Data Flow](#end-to-end-data-flow)
- [Implementation Notes](#implementation-notes)
- [Challenges and Open Problems](#challenges-and-open-problems)
- [Citation](#citation)

-----

Most agents deployed in production are amnesiac. Each run begins from the same base model, the same system prompt, and zero memory of what happened last time. The SME who corrected a bad output on Tuesday has no guarantee the agent behaves differently on Thursday. The engineer who noticed a systematic failure pattern writes it in a Slack thread, not anywhere the agent can read.

This post proposes a concrete architecture to fix that. The core idea is to treat every agent run as a structured learning event — writing it to a Markdown episodic memory store, surfacing relevant prior episodes selectively at inference time, routing curation and training through MCP servers, and periodically closing the loop with a LoRA fine-tune over SME-approved traces. I’ll describe each component in enough detail to implement it.

I’m focusing on the *specialized* case: an agent that does a particular class of tasks repeatedly (LLM eval scoring, risk assessment, technical review, document extraction — anything where there’s a domain expert who can tell good from bad). The approach is less interesting for one-off general assistants and most interesting for agents that accumulate hundreds of runs on a narrow task family.

-----

## The Problem with Amnesiac Agents

Before describing the architecture, it’s worth being precise about what’s broken.

**No durable lesson storage.** Reflexion ([Shinn et al., 2023](https://arxiv.org/abs/2303.11366)) demonstrated that verbal self-reflection stored in a memory buffer can improve performance across trials. But the buffer is in-process and ephemeral. When the process ends, the lessons end. Every production deployment I’m aware of throws this away.

**No selective reveal.** Vector-store RAG retrieves the most embedding-similar chunks. This is the wrong retrieval criterion for episodic memory. What you want is: *which prior episode is most analogous to the current task, and how much of its detail does the agent need right now?* A failed run from last week should surface — but maybe only its summary and the SME’s correction note, not the full 40-turn trace. The current chunk-level similarity approach has no notion of episode structure, outcome, or relevance budget.

**No curation loop.** Self-play and self-training methods (ReST, [Gulcehre et al., 2023](https://arxiv.org/abs/2308.08998); STaR, [Zelikman et al., 2022](https://arxiv.org/abs/2203.14465)) filter traces via a reward model or verifier before training. But in specialized domains, no reward model exists — a domain expert exists. The architecture needs to route SME attention efficiently, not assume it away.

**Fine-tuning cadence mismatch.** LoRA adapters are cheap enough ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) that a weekly fine-tune is realistic. But nothing in standard agent frameworks triggers this, formats the data, or runs the safety checks that prevent catastrophic forgetting.

The architecture below addresses all four.

-----

## System Overview

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

-----

## Component One: Episodic Memory as Markdown

### The Episode File Format

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
tools `get_chargeback_history(lookback=90)` returned data but
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

This pattern is adapted from Anthropic’s Agent Skills format. The retriever operates at three resolution levels:

|Level      |What’s loaded                       |Tokens (approx.)|When                                                                                      |
|-----------|------------------------------------|----------------|------------------------------------------------------------------------------------------|
|Description|Frontmatter `description` field only|~50             |Always, during initial scan                                                               |
|Summary    |Full `EPISODE.md` body              |~400–800        |After episode selected by retriever                                                       |
|Detail     |`references/full_trace.md` + others |~5,000–40,000   |Only if episode body explicitly requests it, or agent needs tool-call-level reconstruction|

The agent never sees the `references/` directory unless `EPISODE.md` contains a line like `See references/sme_corrections.md for the corrected threshold table.` — at which point the retriever loads it. This keeps token expenditure proportional to how analogous a prior episode actually is.

### Writing Episodes After a Run

Episode writing happens as a post-run tool call, not inline. The agent calls `memory_server.write_episode(trace)` after the task completes. The memory server runs a lightweight extraction pass using a secondary LLM call with the following prompt structure:

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

-----

## Component Two: Selective Context Reveal at Inference Time

### Two-Stage Retrieval

When a new task arrives, retrieval runs in two stages before the agent acts.

**Stage 1: Description scan.** The retriever loads only the `description` field from all episodes in the relevant task family (filtered by `task_type` from the frontmatter). At 50 tokens per episode, a store of 1,000 episodes costs 50,000 tokens — feasible in a single context window pass. The LLM scores each description for analogical relevance to the current task and returns a ranked list of episode IDs.

**Stage 2: Body loading.** For the top-K episodes (K=3 by default, configurable), the retriever loads the full `EPISODE.md` body. The agent reads these before acting. If any episode body references a `references/` file that’s clearly needed (e.g., a correction table), the retriever loads that file too.

This two-stage design is not novel in isolation — it’s essentially a coarse-to-fine retrieval. What’s novel is the *unit of retrieval*: a structured episode with outcome metadata, not a text chunk. The `outcome` and `sme_score` fields let the retriever weight analogous *failures* more heavily than analogous *successes* when the current task resembles a historically tricky pattern.

### Budget-Aware Loading

The retriever tracks remaining context budget. If the agent’s task description plus tool definitions already consume 80% of the context window, stage 2 is compressed: only the frontmatter + “Lesson” section of each episode is loaded, not the full body. The retriever signals this compression to the agent via a note in the context: `[Memory: 3 episodes loaded in summary mode due to context constraints. Full bodies available via memory_server.get_episode(id).]`

### The Relevance Scoring Function

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

-----

## Component Three: MCP as the Wiring Layer

The architecture exposes memory, evaluation, and training as three independent MCP servers. The agent host connects to all three via standard MCP client configuration.

### Memory Server

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
# Simplified memory server implementation
class MemoryServer:
    def __init__(self, episodes_dir: Path):
        self.episodes_dir = episodes_dir
        self.repo = git.Repo(episodes_dir)

    def write_episode(self, trace: str, task_type: str) -> str:
        episode_id = f"{datetime.utcnow().isoformat()}-{task_type}"
        episode_dir = self.episodes_dir / episode_id
        episode_dir.mkdir()

        # Extraction pass via secondary LLM call
        episode_md = self._extract_episode(trace)
        (episode_dir / "EPISODE.md").write_text(episode_md)

        # Write full trace to references/
        refs_dir = episode_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "full_trace.md").write_text(trace)

        self.repo.index.add([str(episode_dir)])
        self.repo.index.commit(f"episode: {episode_id}")
        return episode_id
```

### Eval Server

The eval server exposes:

**Prompts** (user-controlled templates):

- `rubric/{task_type}` — the current evaluation rubric for this task type, loaded from `rubrics/{task_type}.md`

**Tools**:

- `judge_trace(trace: str, task_type: str) → JudgmentResult` — runs the LLM-as-judge pass and returns a score + reasoning
- `submit_for_sme_review(episode_id: str, reason: str)` — queues an episode for human review
- `record_sme_judgment(episode_id: str, score: int, corrections: str, promote: bool)` — writes SME judgment back to the episode file and optionally promotes to the training set

### Training Server

The training server exposes:

**Tools**:

- `enqueue_for_finetuning(episode_id: str)` — adds a promoted episode to the pending training batch
- `trigger_training_run(adapter_name: str)` — initiates a LoRA fine-tune over the current pending batch
- `get_adapter_status() → AdapterStatus` — returns current production adapter version, A/B test results, rollback history
- `rollback_adapter(version: str)` — promotes a previous adapter version back to production

**Resources**:

- `training://pending_batch` — the current set of episodes queued for training
- `training://adapter_versions` — version history of all adapters

### Why MCP and Not a Custom API

The alternative is writing a bespoke HTTP API that the agent calls. The MCP argument is practical: every MCP-compatible agent host (Claude Desktop, Claude Code, Cursor, Copilot, Gemini CLI as of late 2025) can connect to these servers without any integration code. The agent discovers tools at startup via the standard `tools/list` call. Swapping the LoRA backend from Axolotl to unsloth is a training-server internal change; the agent interface doesn’t change.

The deeper reason is that MCP’s **roots** and **sampling** primitives are useful for this architecture. Roots let the memory server expose the `episodes/` directory as a permission-scoped filesystem resource, so the agent can navigate episode subdirectories as code rather than making round-trip resource calls for each file — the pattern Anthropic documented for large tool sets ([Anthropic Engineering, 2025](https://www.anthropic.com/engineering/code-execution-with-mcp)). Sampling lets the eval server make LLM calls back through the agent host without managing its own API key.

-----

## Component Four: SME-Driven Evaluation

### The Rubric as a Markdown Artifact

The rubric for a task type is a Markdown file in `rubrics/{task_type}.md`. It follows an analytic rubric structure with named criteria, score ranges, and brief anchors:

```markdown
# Rubric: Merchant Risk Assessment
version: 2.1
last_updated: 2026-04-15
last_updated_by: [SME name]

## Criteria

### 1. Signal Completeness (0–3)
- 3: All mandatory signals retrieved and interpreted (revenue trend,
     chargeback rate with category-specific threshold, complaint rate,
     industry classification)
- 2: 3 of 4 mandatory signals retrieved
- 1: 2 of 4 mandatory signals retrieved
- 0: Fewer than 2 signals

### 2. Threshold Accuracy (0–2)
- 2: Category-specific thresholds applied correctly
- 1: Global thresholds applied (wrong for digital goods categories)
- 0: No threshold reasoning present

### 3. Rationale Quality (0–2)
- 2: Final risk tier is explicitly justified by named signals and thresholds
- 1: Tier stated without full justification
- 0: No rationale

### 4. Completeness of Response (0–1)
- 1: Response includes recommended next actions
- 0: Response terminates at risk tier only

## Passing threshold

Auto-promote to training set: total score ≥ 7 (out of 8)
Queue for SME review: total score 4–6
Discard: total score < 4
```

The rubric is versioned in the same Git repo as the episodes. The `version` field in the frontmatter lets the eval server record which rubric version was used for any given judgment — relevant because rubrics evolve and you may want to re-judge old episodes under a new rubric.

### LLM-as-Judge Calibration

The judge prompt loads the rubric and the trace:

```
You are evaluating an agent's output against the rubric below.
Score each criterion independently. Show your reasoning for each
criterion before stating the score. Do not let your overall
impression of the response affect individual criterion scores.

Rubric:
{rubric_body}

Agent trace:
{trace}

Return JSON: {"criterion_scores": {"signal_completeness": N, ...},
              "total": N, "reasoning": "..."}
```

Calibration follows the SMELL framework approach ([Quotient AI, 2024](https://blog.quotientai.co/subject-matter-expert-language-liaison-smell)): for each new task type, an SME labels 30–50 traces using the rubric before the judge is deployed. The labeled set is split 70/30 into calibration and held-out. The judge prompt is iterated until Krippendorff’s α between judge and SME labels exceeds 0.7 on the held-out set. This bar is achievable in practice for structured analytic rubrics; holistic single-score rubrics rarely get there without more labeled data.

### The Promotion Decision

The promotion decision is a three-way split:

- **Auto-promote** (total ≥ threshold): episode written to the pending training batch immediately.
- **SME queue** (score in review band): episode metadata, judge reasoning, and a link to the full trace are surfaced in the SME review interface — a simple Markdown-rendered list of queued episodes, each with inline edit capability.
- **Discard** (score below discard threshold): episode retained in the memory store (failures are valuable for retrieval) but excluded from the training set. Its `sme_score` field is set to the judge score; it will still surface during retrieval for analogous tasks.

The SME interface is thin by design. SMEs edit `EPISODE.md` directly (or via a UI that writes to the file) and call `eval_server.record_sme_judgment(episode_id, score, corrections, promote=True/False)`. The corrections field is written into `references/sme_corrections.md` and linked from `EPISODE.md`‘s body. From this point, the episode’s lesson includes the SME’s direct language, not an LLM paraphrase of it.

-----

## Component Five: Gated LoRA Fine-Tuning

### Trace Formatting for SFT

Each promoted episode is formatted as a (system, user, assistant) training example. The transformation is straightforward: take the original task as the user message, take the *corrected* agent output (post-SME edits, if any) as the assistant message, and use the base system prompt as the system message. If the SME only wrote a correction note and didn’t produce a full corrected output, a secondary LLM pass generates the ideal response conditioned on the SME note:

```
Given this agent trace and the SME correction note below,
write the ideal assistant response for this task. The response
should reflect the SME correction exactly.

Original task: {task}
Agent's response: {agent_response}
SME correction note: {sme_correction}

Write only the ideal response. Do not explain the correction.
```

This synthetic ideal-response generation is the weakest link in the pipeline and the most important one to monitor. If the synthetic ideal responses diverge from what the SME actually wanted, the fine-tuning signal is corrupted. The mitigation is to have the SME spot-check a random 10% of generated ideal responses before each training run.

### The Training Gate

Fine-tuning runs on a fixed cadence — weekly for high-volume tasks, monthly for low-volume. Before any adapter is promoted to production, it must clear two gates:

**Gate 1: Domain improvement.** The new adapter must score higher than the current production adapter on a held-out SME eval set for the target task type. The eval set is never used for training and is refreshed with new SME-labeled examples each cycle. If the new adapter does not improve, the cycle is skipped and traces are accumulated for the next run.

**Gate 2: Base capability non-regression.** LoRA learns less and forgets less than full fine-tuning ([Biderman et al., 2024](https://arxiv.org/abs/2405.09673)), but “forgets less” is not “forgets nothing.” Before promotion, the new adapter is evaluated on a fixed suite of general-capability benchmarks (reasoning, instruction following, refusal behavior). If any benchmark drops more than a configurable tolerance (default: 2% relative), the adapter is rejected. The tolerance should be domain-sensitive: a risk-assessment agent can probably tolerate more drift on creative writing benchmarks than on reasoning benchmarks.

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

Every adapter version is tagged in the training server with a manifest:

```json
{
  "adapter_id": "merchant-risk-v7",
  "base_model": "claude-3-haiku-20240307",
  "training_episodes": ["2026-04-01T...", "2026-04-08T...", ...],
  "domain_score": 0.84,
  "base_benchmark_deltas": {"reasoning": -0.01, "instruction": 0.00},
  "promoted_at": "2026-05-01T09:00:00Z",
  "gate_results": {...}
}
```

Rollback is a single `training_server.rollback_adapter(version)` call. The system records which episodes were used to train each adapter, so if a regression is traced to a specific batch of bad SME corrections, those episodes can be removed from the store and the adapter retrained from the remaining history.

-----

## The Three Time-Scales of Improvement

The architecture operates at three time-scales, each with its own failure mode and fallback:

**Per-turn (in-context).** The agent retrieves relevant episodes and acts. No weight updates, no persistence after the run. This is the floor — it works on the first task and improves as the memory store grows. Failure mode: memory store is empty or retrieval quality is poor. Fallback: base model without memory.

**Per-episode (memory store update).** After each run, a new episode is written. The agent is immediately better at analogous tasks on the next run, without waiting for a fine-tune. Failure mode: episode writer produces low-quality summaries; judge has poor calibration. Fallback: SME can manually edit `EPISODE.md` in the Git repo.

**Per-cycle (adapter refresh).** Every week/month, a new LoRA adapter is trained and gated. This bakes stable lessons into weights, reducing context load for very common patterns. Failure mode: adapter fails both gates; synthetic ideal-response generation is off. Fallback: previous adapter stays in production; memory store continues accumulating.

The three layers degrade gracefully. If fine-tuning is disabled, memory still works. If memory is empty, the base model still functions. If the judge is miscalibrated, the SME queue catches the damage before it reaches training.

-----

## End-to-End Data Flow

A complete cycle, from task to trained adapter:

1. **Task arrives.** Agent host receives a new merchant risk assessment request.
1. **Memory scan.** Retriever calls `memory_server.search_episodes(query=task_description, task_type="merchant-risk-assessment", top_k=3)`. Returns three episode summaries.
1. **Body loading.** Retriever loads full `EPISODE.md` for each. One episode references `sme_corrections.md`; retriever loads it. Total context addition: ~2,500 tokens.
1. **Agent acts.** Agent runs the task with the current LoRA adapter and retrieved context. Produces risk tier + rationale.
1. **Episode written.** Agent calls `memory_server.write_episode(trace, task_type="merchant-risk-assessment")`. Memory server runs extraction pass; commits `EPISODE.md` to Git.
1. **Judge runs.** Eval server calls `eval_server.judge_trace(trace, task_type)`. Returns score 6/8 — in the SME review band.
1. **SME queued.** Eval server calls `eval_server.submit_for_sme_review(episode_id, reason="Score 6/8; threshold accuracy criterion scored 1/2.")`.
1. **SME reviews.** SME reads the episode, edits the lesson section to clarify the correct threshold table, and records judgment: score 7/8, corrections written, promote=True.
1. **Training batch updated.** Eval server calls `training_server.enqueue_for_finetuning(episode_id)`. Episode joins the pending batch.
1. **Weekly training run.** `training_server.trigger_training_run("merchant-risk-v8")` fires. LoRA is trained on 47 promoted episodes from the past week. Both gates clear. Adapter promoted.
1. **Next task.** New agent run uses adapter v8 and the updated memory store.

-----

## Implementation Notes

A few things that are non-obvious in practice:

**The extraction LLM should be smaller than the task LLM.** The episode writer runs on every trace. Use a smaller, cheaper model for extraction — the structured output prompt is simple enough that a 7B model handles it reliably. Save the larger model for the task itself and for stage-1 relevance scoring (which requires more nuanced analogical judgment).

**Git is the right storage backend, not a vector database.** The memory store doesn’t need millisecond retrieval latency — it runs once per task, upstream of the agent’s actual work. Git gives you history, diffability, branch-based experimentation (try a new rubric on a feature branch before merging), and the ability for SMEs to submit corrections as PRs. A vector index can be layered on top for the stage-1 scan if the store exceeds ~5,000 episodes, but don’t start there.

**Rubric versions should be pinned to judge runs.** When the rubric changes, old scores become incomparable to new scores. The `judge_run` record should store `rubric_version` alongside the score. If you re-run the judge under a new rubric, record it as a new judgment rather than overwriting the old one. This matters when you want to understand why an adapter trained on old rubric scores behaves differently than expected under the new rubric.

**Keep adapter batch sizes small.** The first instinct is to wait until hundreds of traces accumulate before fine-tuning. This makes the feedback loop slow. A weekly run on 30–50 promoted traces is preferable to a quarterly run on 500. Small batches also make it easier to attribute regressions to specific episodes.

**Don’t train on the retrieved context, only on the output.** The training examples should be `(task_description, ideal_output)` pairs, not `(task_description + retrieved_episodes, ideal_output)`. You’re training the model to produce better outputs, not to depend on a specific retrieval result. The retrieval is in-context scaffolding; the fine-tune should bake the *lesson* into weights so the model needs less scaffolding over time.

-----

## Challenges and Open Problems

**When to bake a lesson into weights vs. keep it in memory.** There’s no principled answer here. The heuristic I use: if the same correction appears in 5+ consecutive episodes, the lesson has stabilized enough for fine-tuning. If it’s appearing in ~20% of episodes for a task type, it’s a systematic gap worth training on. If it’s appearing sporadically in <5% of cases, leave it in memory — it’s probably an edge case that a fine-tune might over-fit to.

**Catastrophic forgetting under continual adaptation.** LoRA forgets less than full fine-tuning ([Biderman et al., 2024](https://arxiv.org/abs/2405.09673)), and CURLoRA ([Fawi, 2024](https://arxiv.org/abs/2408.14572)) and I-LoRA ([Chen et al., 2024](https://arxiv.org/abs/2402.18865)) push further on this axis via matrix decomposition and interpolation regularization. The safe choice is gate 2 above — benchmark every adapter before promoting. A system fine-tuning weekly for a year is still uncharted territory; I’d recommend maintaining a “rollback-to-base” capability at all times.

**Privacy of episodic memory.** A trace contains everything the agent saw. In practice this may include user PII, proprietary data, partial credentials that appeared in tool outputs. Mitigations: (1) run a redaction pass at write time using a secondary LLM call before the episode is committed; (2) store memory in per-tenant directories exposed only through auth-scoped MCP roots; (3) consider whether the SME review UI should render redacted versions. None of these are complete solutions; the right answer depends heavily on deployment context.

**Model collapse under recursive self-curation.** If the judge model is from the same family as the task model, and the task model is fine-tuned on judge-curated data, you’re creating a feedback loop that can amplify biases and erode output diversity. The safeguards are: (a) keep an SME-labeled ground-truth eval set that is never used for training, (b) measure output diversity metrics (n-gram entropy, semantic coverage) over adapter versions, (c) re-anchor the rubric to SME labels every N cycles rather than letting it drift on LLM-generated corrections alone.

**Cold start.** A new domain with no episodes needs either human demonstrations or a low-stakes bootstrapping period with heavy SME review. Voyager’s curriculum approach ([Wang et al., 2023](https://arxiv.org/abs/2305.16291)) — start with achievable tasks, expand the curriculum as the skill library grows — is the right model. For enterprise domains, the SME often has a backlog of past human-handled cases that can be formatted into the episode store to seed the memory with institutional knowledge before the agent runs a single live task.

-----

## Citation

```bibtex
@misc{self-improving-agents-2026,
  title   = {Architecting Self-Improving Agents},
  year    = {2026},
  note    = {Blog post}
}
```

## References

- Shinn et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- Wang et al. (2023). *Voyager: An Open-Ended Embodied Agent with Large Language Models*. [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)
- Gulcehre et al. (2023). *Reinforced Self-Training (ReST) for Language Modeling*. [arXiv:2308.08998](https://arxiv.org/abs/2308.08998)
- Zelikman et al. (2022). *STaR: Bootstrapping Reasoning With Reasoning*. [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Biderman et al. (2024). *LoRA Learns Less and Forgets Less*. [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
- Fawi (2024). *CURLoRA: Stable LLM Continual Fine-Tuning and Catastrophic Forgetting Mitigation*. [arXiv:2408.14572](https://arxiv.org/abs/2408.14572)
- Park et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. UIST 2023. [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
- Packer et al. (2023). *MemGPT: Towards LLMs as Operating Systems*. [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- Quotient AI (2024). *SMELL: Subject-Matter Expert Language Liaison*. [blog.quotientai.co](https://blog.quotientai.co/subject-matter-expert-language-liaison-smell)
- Anthropic Engineering (2025). *Code Execution with MCP: Building More Efficient AI Agents*. [anthropic.com](https://www.anthropic.com/engineering/code-execution-with-mcp)
