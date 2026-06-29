---
title: "Self-Healing Code Agents for CI Regression Repair"
date: 2026-06-29
description: "A concrete walkthrough of building a LangGraph agent that diagnoses failing CI tests in Python services, generates unified-diff patches, gates them with LLM self-reflection, and executes in a Docker sandbox -- with structured post-mortems when the regression cannot be fixed automatically."
tags: ["agents", "langgraph", "ci-cd", "code-repair", "docker", "swe-bench", "tree-sitter"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Picture a CI pipeline that catches a regression, opens a GitHub issue, and then just... waits. A developer gets paged, context-switches from whatever they were doing, reads the traceback, writes a patch, and pushes it. That loop takes anywhere from 20 minutes to several hours depending on the complexity of the bug and the developer's availability.

The question is whether an agent can close that loop automatically: read the failing test output, trace which files are relevant, generate a patch, verify it in an isolated environment, and open a PR -- all before the developer has finished reading the Slack alert. For well-scoped regressions in Python services (a changed function signature, a missed edge case in a utility, a broken import after a refactor), this is a realistic target.

**Self-healing-agent**, by Mihir Inamdar, is a worked implementation of that loop. It uses LangGraph to orchestrate a multi-node pipeline that diagnoses failing tests, generates patches as unified diffs, gates each patch with a second LLM self-reflection call, executes in a locked-down Docker sandbox, and produces a structured post-mortem on failure. This post covers the architecture, the design decisions, and what this approach means for teams who want autonomous repair in their CI pipelines.

This targets Python services with pytest-based test suites running in Docker-compatible CI environments. Familiarity with LangGraph, Docker, and basic pytest is assumed. Some knowledge of tree-sitter for program analysis is useful but not required.

## The Automated Program Repair Problem

**Automated program repair (APR)** has a long research history predating language models: GenProg (Le Goues et al., 2012) used genetic programming to search the patch space, and template-based systems like TBar (Liu et al., 2019) applied fix patterns from human-written code. These systems worked on constrained problem classes and required running tests to evaluate candidate patches, often evaluating thousands of candidates per bug.

LLMs changed the economics. A model can read a failing test, understand what it expects, look at the implementation, and generate a plausible patch in one forward pass, with no evolutionary search or template library required. The question is no longer whether LLMs can generate patches, but whether they can do so reliably in a closed loop without human oversight.

**SWE-bench** (Jimenez et al., 2023) operationalized this as a benchmark: given a real GitHub issue and the repository at the time it was filed, generate a patch that makes the failing tests pass without breaking the rest of the test suite. SWE-bench Lite contains 300 such tasks drawn from 12 popular Python repositories. Scores are reported as **pass@1**, the fraction of tasks where the agent's first attempt passes all tests.

The state of the art on SWE-bench Lite as of mid-2025 has moved quickly but remains well below full resolution: SWE-agent achieved a 12.5% resolution rate on SWE-bench, with AutoCodeRover reaching approximately 19% on SWE-bench Lite through AST-based code search. The harder SWE-bench Pro benchmark, designed to resist data contamination with enterprise-scale tasks, shows performance below 25% pass@1 across widely used models, with the highest score at 23.3%. The benchmark ceiling is real and meaningful: bugs that require multi-file changes, compile-time dependencies, or C extensions are systematically hard in ways current architectures do not address.

For the CI use case, the harder end of SWE-bench is not the primary target. Most CI regressions are scoped: a single failing test caused by a code change that is already understood. The architecture here is a clean, inspectable implementation of the full repair loop -- context retrieval, patch generation, self-reflection, sandboxed execution, and structured failure analysis -- that works well on that class of bug.

---

## Where Existing Agents Break Down

Before the implementation, here are the failure modes the design is responding to.

**Context retrieval is the first bottleneck.** A repository might have hundreds of files. Pasting everything into the context window is not feasible. Including only the file containing the failing test misses imports, parent classes, and shared utilities that are actually relevant to the bug. Getting the right context is both important and hard to do mechanically. Agents that read files indiscriminately burn many tokens on irrelevant code.

**One-shot patches have high variance.** An LLM generating a patch from a failing test and some context produces the most likely-looking fix according to its training distribution. If the actual bug is uncommon or the fix requires a non-obvious change, the first patch fails. Without a principled retry mechanism that incorporates what went wrong, the agent is sampling from the same distribution again with a slightly different prompt.

**Sandbox safety is often an afterthought.** Running agent-generated code without isolation is genuinely dangerous. A faulty patch might `rm -rf` something, make network requests, or exhaust system memory. Most demo agents that run code directly on the host machine are betting that the LLM will not produce destructive code, which is not a reliable bet.

**Failure is opaque without structured analysis.** When an agent fails on a task, what have you learned? If the failure mode is just "tests still failing," you do not know whether the agent misidentified the root cause, generated a syntactically invalid patch, got blocked by missing test fixtures, or encountered a C extension. Structured failure analysis turns opaque failures into actionable information.

---

## Architecture: Five Nodes, One Sandbox

The agent is a LangGraph `StateGraph` with five nodes and conditional routing based on test outcomes and iteration count:

```
        START
          │
  ┌───────▼────────┐
  │ context_builder │  tree-sitter import graph → relevant files
  └───────┬─────────┘
          │
  ┌───────▼────────┐
  │ patch_generator │  Claude Sonnet 4.6, temp=0.2, unified diff output
  └───────┬─────────┘
          │
  ┌───────▼────────┐
  │  self_reflector │  second LLM call: APPROVED / REVISE: [critique]
  └───────┬─────────┘
          │
┌─────────┴───────────┐
│ not approved         │ approved (or iter >= 2)
│ AND iter < 2         │
└──► patch_generator   │
                       ▼
              ┌────────────────┐
              │ sandbox_executor│  Docker, net=none, 512MB, 60s timeout
              └────────┬───────┘
                       │
      ┌────────────────┼──────────────────┐
      │ passed         │ failed            │ max_iterations
      ▼                ▼                   ▼
 ┌──────────┐  ┌────────────────┐  ┌────────────────────┐
 │ pr_opener│  │ patch_generator│  │ postmortem_generator│
 └────┬─────┘  └────────────────┘  └──────────┬─────────┘
      │                                        │
     END                                      END
```

The complete state carried through the graph:

```python
# agent/state.py -- single source of truth
@dataclass
class AgentState:
    task_id: str
    repo_path: str
    failing_tests: list[str]
    issue_description: str
    
    # built by context_builder
    relevant_files: dict[str, str]
    import_graph: dict[str, list[str]]
    
    # generated and revised by patch_generator / self_reflector
    current_patch: str | None
    patch_history: list[PatchAttempt]
    
    # runtime counters
    iteration: int
    max_iterations: int
    
    # execution results
    last_test_output: str | None
    last_exit_code: int | None
    
    # observability
    total_tokens: int
    total_llm_calls: int
    total_cost_usd: float
    
    # terminal outputs
    pr_url: str | None
    postmortem: PostMortem | None
```

Every node reads from and writes to this state. LangGraph manages the transitions; each node is a pure function from `AgentState` to a partial state update. This makes the graph straightforward to test: each node can be unit-tested with a mocked state and inspected mid-run.

---

## Context Building with tree-sitter

The first node, `context_builder`, identifies which files in the repository are actually relevant to the failing tests. This is a hard problem: a test that imports `from mylib.utils import format_date` might depend on `format_date` calling `parse_iso8601`, which calls `_validate_timezone`, all defined in different files.

The solution is to build an **import graph** using **tree-sitter** ([tree-sitter.github.io](https://tree-sitter.github.io)), a fast incremental parsing library that can parse Python source code into concrete syntax trees without executing it. Tree-sitter's Python grammar can identify `import` and `from ... import` statements with high reliability across Python 2 and 3 syntax variations.

The algorithm:

1. Parse each failing test file with tree-sitter, extract direct imports
2. Resolve each import to a file path within the repository (handling `__init__.py`, package roots, relative imports)
3. Parse each resolved file, extract its imports, recurse
4. Stop at the repository boundary (do not follow stdlib or third-party packages)
5. The resulting transitive closure is the import graph; the files in it become `relevant_files`

This is static analysis, not execution; it does not handle dynamic imports (`importlib.import_module(name)`), conditional imports (`if sys.platform == 'win32': import winreg`), or plugin systems. For the majority of bugs in well-structured Python services, static import tracing catches the relevant files.

Why tree-sitter rather than Python's own `ast` module? Two reasons: tree-sitter handles syntax errors gracefully (returning partial trees rather than raising), which matters because you are often analyzing code with bugs; and tree-sitter supports incremental reparsing, which is efficient if you need to reparse after applying a patch.

The context window budget is finite. After building the import graph, `context_builder` selects files by relevance -- files directly imported by the failing tests rank higher than transitively imported files -- and truncates at a configurable token limit.

---

## Patch Generation at Low Temperature

`patch_generator` produces patches in **unified diff format**, the standard `git diff` output format with `---`, `+++`, and `@@ ... @@` headers. This is a deliberate choice over asking the model to produce full file contents or Python AST manipulations.

Unified diff has several advantages here:

**Verbosity scales with change size.** A one-line fix produces a short diff. A ten-line refactor produces a longer one. Full file output is always the file size, regardless of how small the change is.

**Parseable and applicable mechanically.** `patch_generator` produces text; `sandbox_executor` applies it with `subprocess.run(["patch", "-p1", ...])`. The patch utility handles offset matching, rejects cleanly if context does not match, and reports exactly which hunks succeeded or failed.

**Reviewable by humans and LLMs.** Unified diffs are what code review tools display. The self-reflector node reads the diff and can reason about what changed and what did not, which would be harder with full file output.

The model is called at `temperature=0.2`, low but not zero. Zero temperature produces the single most likely token at each step, which can sometimes produce locally consistent but globally incoherent patches (a common failure mode is fixing the immediate assertion while missing a related invariant elsewhere). A small amount of temperature adds diversity without sacrificing coherence. On retry after a failed test, the temperature is unchanged: the patch_generator already has the test failure output in its context, which is more informative than sampling variation.

The prompt structure for `patch_generator`:

```
System: You are an expert Python engineer. You will be given:
1. A failing test with its full output
2. The relevant source files (identified from the import graph)
3. The issue description
4. [On retry] The previous patch and why it failed

Produce a unified diff that fixes the failing test without breaking others.
Output ONLY the diff, no explanation.

[context_builder output: relevant files]
[failing test output]
[issue description]
[previous attempt if iteration > 0]
```

The "output ONLY the diff" instruction matters because the subsequent `patch` command needs clean diff text. Any explanation before or after the diff will cause the patch application to fail.

---

## Self-Reflection as a Pre-Execution Gate

Before any patch reaches the Docker sandbox, it passes through `self_reflector`, a second LLM call that reads the proposed patch and critiques it. The node outputs one of two structured responses:

```
APPROVED
```

or

```
REVISE: [one-sentence critique]
```

If `REVISE` and `iteration < 2`, the critique is appended to the patch_generator context and the graph loops back. If `iteration >= 2`, the patch is sent to the sandbox regardless. At that point, sandbox execution results are more informative than another LLM opinion.

This two-call structure is a lightweight version of the **Reflexion** pattern (Shinn et al., 2023). In Reflexion, an actor generates an output, an evaluator scores it, and a self-reflection module diagnoses the issue and proposes a fix, appending both the failed output and the suggested correction to the input context. The self-healing-agent's `self_reflector` is the evaluator step applied *before* execution rather than after: a pre-execution filter rather than a post-execution teacher.

The `self_reflector` prompt is structured to look for specific failure modes:

```
Given this patch, check for:
1. Does the fix address the root cause stated in the issue, or just the symptom?
2. Are there any obvious regression risks -- tests that currently pass that this change might break?
3. Is the diff syntactically valid and coherent?

If all three look reasonable, output exactly: APPROVED
Otherwise, output exactly: REVISE: [one sentence critique]
```

The value of this gate is asymmetric. A patch that looks wrong to the reflector before execution often *is* wrong: catching it saves a Docker container startup (5-10 seconds) and keeps the test failure output from entering the context window. A patch the reflector approves that still fails in the sandbox provides useful new information (the sandbox output) for the next attempt. The gate filters obvious mistakes cheaply; the sandbox catches the rest.

One design note on iteration bound: the reflector runs on iterations 0 and 1, then the patch goes straight to the sandbox from iteration 2 onward. By iteration 2, the LLM has seen two test failure outputs with their full tracebacks. That is richer information than another LLM opinion. Continuing to apply the pre-execution gate past iteration 2 would slow convergence without improving it.

---

## The Docker Sandbox

Every patch execution happens inside a Docker container. The container configuration:

| Parameter | Value | Rationale |
|---|---|---|
| Network | `--network none` | Prevents any network access from agent-generated code |
| Memory | `--memory 512m` | Hard cap, prevents OOM from runaway test fixtures |
| CPU | `--cpus 1` | Prevents starvation of the host system |
| Timeout | 60 seconds hard kill | Prevents infinite loops in test suites |
| Repo mount | Read-only | Agent cannot modify the repository itself |
| Write path | In-container tmpdir | Patch is applied and tests run in a fresh copy |

The container lifecycle per execution:

```
docker run \
  --network none \
  --memory 512m \
  --cpus 1 \
  --rm \
  -v /path/to/repo:/repo:ro \
  self-healing-agent-sandbox:latest \
  /bin/sh -c "
    cp -r /repo /tmp/repo_copy && \
    cd /tmp/repo_copy && \
    patch -p1 < /patch.diff && \
    python -m pytest {failing_tests} --timeout=30 -x -q 2>&1
  "
```

The `--rm` flag ensures containers do not accumulate. The read-only repo mount means even a destructive patch cannot modify the source; it is applied to a copy in `/tmp`. The 60-second host-level `docker kill` runs separately from pytest's `--timeout=30` to handle cases where pytest itself hangs rather than the test code.

The exit code from Docker is the primary signal: 0 means all failing tests now pass, non-zero means failure. The full stdout/stderr is captured and passed back into the agent state as `last_test_output`, which becomes context for the next `patch_generator` call.

This is the loop: the sandbox output is the ground truth. A patch that looks correct to both the generator and the reflector but fails in the sandbox gets its failure traceback appended to the context. The next patch_generator call has the original test failure, the first patch, the first test output, and now the second test output. The context accumulates real execution evidence rather than LLM speculation.

**Why Docker over safer alternatives?** The lighter-weight alternative is executing in a `subprocess` with `seccomp` filters or using a Python sandbox like `RestrictedPython`. Docker is heavier but complete: it provides an environment that closely matches what the tests expect (filesystem layout, installed packages, Python version), which matters for CI use cases where test infrastructure often depends on the exact package environment. A seccomp-filtered subprocess can still see the host filesystem; Docker's mount model eliminates that entirely.

For CI integration, the sandbox image can be the same image your CI already uses for the test suite. This eliminates the "tests pass in sandbox but not in CI" problem, because they are the same environment.

---

## Post-Mortems on Failure

When the agent exhausts its iteration budget without passing the tests, it does not just return a failure code. The final node, `postmortem_generator`, produces a structured analysis:

```python
@dataclass
class PostMortem:
    task_id: str
    attempts_made: int
    patches_tried: list[str]      # the actual diffs
    test_outputs: list[str]       # one per sandbox execution
    
    root_cause_hypothesis: str    # labelled as hypothesis, not fact
    why_each_attempt_failed: list[str]
    
    next_steps: list[str]         # exactly three, actionable
    
    # observability
    total_tokens: int
    total_llm_calls: int
    total_cost_usd: float
```

The `root_cause_hypothesis` field is explicitly labelled as a hypothesis in both the schema and the prompt. The agent is instructed to express appropriate uncertainty and not overstate confidence about bugs it could not fix. An agent that confidently asserts an incorrect root cause is worse than one that honestly reports "hypothesis: the issue may be in the timezone handling, but I was not able to confirm this."

The `next_steps` field produces exactly three concrete items for a human engineer. The constraint is deliberate: open-ended lists of "potential next steps" are often useless. Three specific items is enough to be actionable without being overwhelming.

A realistic post-mortem might look like:

```
Task: astropy/astropy -- separability_matrix wrong for nested CompoundModels

Attempts: 3
- Attempt 1: Modified _separable() to handle nested composition -- still failed
             (separability propagation logic for n>2 nesting broken)
- Attempt 2: Rewrote CompoundModel.separability_matrix() directly -- still failed
             (did not account for non-square separability matrices)
- Attempt 3: Added recursive case to _combine_separability() -- still failed
             (assumption about matrix dimensions wrong for depth > 2)

Root-cause hypothesis (unconfirmed): The separability matrix propagation
does not correctly handle the case where the n-th level CompoundModel has
components that are themselves CompoundModels. This appears to require
tracking the full model tree, not just the immediate composition.

Next steps:
1. Add a test for triple-composition (A ∘ B ∘ C) to isolate the dimension issue
2. Review the original separability_matrix() algorithm in the Astropy paper;
   the recursion base case may be incorrect for nested operators
3. Consider adding a debug mode to _separable() that prints intermediate
   matrix dimensions to identify where the shape assumption breaks
```

This kind of output is genuinely useful to a human engineer picking up where the agent left off. In a CI context, the post-mortem gets attached to the issue automatically so the on-call engineer has a starting point rather than a bare stack trace.

---

## Benchmarking: SWE-bench Lite and the pass@k Estimator

The evaluation harness runs the agent against SWE-bench Lite's 300 tasks. Each task provides a repository snapshot, a problem statement, and hidden test cases. The agent receives the problem statement and has access to the repository; it does not see the hidden tests.

Metrics tracked per task:

- `passed`: bool (did all hidden tests pass)
- `iterations`: int (how many sandbox executions)
- `llm_calls`: int (total LLM API calls, including reflector)
- `cost_usd`: float (at current API pricing)

Aggregate metrics use the **unbiased pass@k estimator** from Chen et al. 2021 (the Codex paper):

$$\text{pass@k} = \mathbb{E}_{\text{problems}} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right]$$

where $n$ is the number of samples generated per problem, $c$ is the number that pass, and $k$ is the pass budget. For this agent, $n = \text{max\_iterations}$ and $c \in \{0, 1\}$ (the agent either solves it or does not). At $n = 5, k = 1$, this reduces to `pass@1 = c/n`, which is just the fraction of tasks solved. The unbiased estimator matters at $k > 1$: the naive "fraction of k-sample runs that include a pass" is biased upward when $c > 0$ and $k < n$.

The evaluation script runs tasks sequentially by default and saves results incrementally to `eval/benchmark_results.json` after each task, making it safe to interrupt and resume:

```bash
# Smoke test: first 5 tasks
python scripts/run_swebench.py --limit 5

# Full evaluation
python scripts/run_swebench.py
```

The README is honest about the current state of the benchmark results: no evaluation has been run yet on the full 300 tasks. The benchmark infrastructure exists and is correct; the numbers are unpopulated. Shipping an evaluation harness without pre-computed numbers is unusual for a research implementation, but it means the reader knows exactly what has and has not been validated.

---

## Challenges and Open Problems

**Multi-file bugs are structurally hard.** The primary expected failure mode: the agent modifies one file but the fix spans two or more interdependent modules. The patch applies cleanly but the tests still fail. The import graph helps identify relevant files, but `patch_generator` generates a single unified diff. Bugs that require coordinated changes across files -- an interface change that requires updating both the implementation and all callers -- need multi-file patch support. This is non-trivial: the patch format handles multi-file diffs but the model needs to reason about consistency across file boundaries simultaneously.

**Test infrastructure that does not exist in the sandbox.** Some tasks have tests that depend on external fixtures: databases that need to be seeded, network calls that need to be mocked, file system paths that need to exist. The sandbox's `--network none` flag catches some of these early (network-dependent tests fail immediately), but filesystem fixture problems are harder. The agent sees a test failure that is not caused by its patch at all, which is confusing and wasteful.

**Self-reflection reliability.** The reflector is called with the same base model as the generator (`claude-sonnet-4-6`). This is the standard LLM-as-judge critique: a model evaluating its own outputs has an inherent bias toward approving them. Empirically, the reflector catches obvious mistakes (syntax errors, obviously unrelated changes), but it is unlikely to catch subtle logical errors that it would also make in generation. Using a stronger or differently-prompted judge model, or cross-checking against a lightweight static analysis pass, would improve the gate's effectiveness.

**Context window accumulation.** The state carries the full history of attempts across iterations: previous patches, previous test outputs, critique texts. At `max_iterations=5`, the context for the fifth attempt includes four failed patches plus their sandbox outputs, which can easily total 20,000+ tokens. This crowds out the relevant source files. Selective context compression, summarizing earlier iterations rather than including them verbatim, would help, but requires deciding what to summarize and what to preserve.

**The PR opener is optimistic.** When a task passes, `pr_opener` creates a GitHub pull request via PyGithub. But applying this agent to a real CI pipeline rather than the benchmark's sandboxed snapshots introduces complexity that the benchmark is specifically designed to eliminate: conflicting in-flight changes, repository-specific CI requirements, review processes, and the possibility that the "passing tests" criterion is necessary but not sufficient for correctness. The gap between "passes the specific failing test" and "is correct to merge" is significant. For production use, the PR should be treated as a draft for human review, not an automatic merge.

**No cross-task learning.** Each task starts fresh. If the agent failed on one regression because it misidentified the root cause of a numpy shape mismatch, nothing about that failure informs the next regression with an identical failure pattern in a different service. The post-mortem lives in `results/` but nothing reads it. A shared failure library, indexed by failure category and retrieved for similar tasks, would let the agent learn from its own history, which is exactly the kind of compound improvement that makes long-horizon agents more capable over time.

The self-healing-agent architecture is a coherent answer to what "closing the loop" actually means for code repair: pre-execution critique to filter obvious mistakes, sandboxed execution to get ground truth, test output fed back as context for revision, and structured failure analysis when the budget runs out. The components each do one thing and hand off cleanly to the next. That makes it a useful reference implementation for CI regression repair regardless of where the benchmark numbers eventually land.

---

## References

- Inamdar, Mihir. *self-healing-agent: Self Healing AI Agent in Sandbox*. [github.com/inamdarmihir/self-healing-agent](https://github.com/inamdarmihir/self-healing-agent)
- Jimenez, Carlos E. et al. (2023). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* arXiv:2310.06770
- Shinn, Noah et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023.
- Chen, Mark et al. (2021). *Evaluating Large Language Models Trained on Code*. arXiv:2107.03374
- Yang, John et al. (2024). *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*. NeurIPS 2024.
- Le Goues, Claire et al. (2012). *GenProg: A Generic Method for Automatic Software Repair*. IEEE TSE.
