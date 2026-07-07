---
title: "Self-Healing Code Agents for CI Regression Repair"
date: 2026-06-27
description: "A concrete walkthrough of building a LangGraph agent that diagnoses failing CI tests in Python services, generates unified-diff patches, gates them with LLM self-reflection, and executes in a Docker sandbox -- with structured post-mortems when the regression cannot be fixed automatically."
tags: ["agents", "langgraph", "ci-cd", "code-repair", "docker", "swe-bench", "tree-sitter", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

A CI pipeline that catches a regression and then simply waits for a human is leaving money on the table. The developer gets paged, context-switches, reads the traceback, writes a patch, and pushes it. That loop takes anywhere from twenty minutes to several hours depending on bug complexity and availability.

The question is whether an agent can close that loop automatically: read the failing test output, trace which files are relevant, generate a patch, verify it in an isolated environment, and open a PR, all before the developer has finished reading the Slack alert. For well-scoped regressions in Python services (a changed function signature, a missed edge case in a utility, a broken import after a refactor), this is a realistic target.

**Self-healing-agent**, by Mihir Inamdar, is a worked implementation of that loop. It uses LangGraph to orchestrate a multi-node pipeline that diagnoses failing tests, generates patches as unified diffs, gates each patch with a second LLM self-reflection call, executes in a locked-down Docker sandbox, and produces a structured post-mortem on failure. This post covers the architecture, the design decisions, and the role Qdrant plays as the agent's long-term regression memory.

This post targets Python services with pytest-based test suites running in Docker-compatible CI environments. Familiarity with LangGraph, Docker, and basic pytest is assumed. Some knowledge of tree-sitter for program analysis is useful but not required.

---

## Table of Contents

1. [The Automated Program Repair Problem](#the-automated-program-repair-problem)
2. [Where Existing Agents Break Down](#where-existing-agents-break-down)
3. [Architecture: Six Nodes, One Sandbox](#architecture-six-nodes-one-sandbox)
4. [Context Building with tree-sitter](#context-building-with-tree-sitter)
5. [Qdrant as Few-Shot Regression Memory](#qdrant-as-few-shot-regression-memory)
   - [Collection Schema and Named Vectors](#collection-schema-and-named-vectors)
   - [Payload Design and Filtering](#payload-design-and-filtering)
   - [Upserting Successful Repairs](#upserting-successful-repairs)
   - [Retrieval at Repair Time](#retrieval-at-repair-time)
   - [Why Qdrant over Simpler Alternatives](#why-qdrant-over-simpler-alternatives)
6. [Patch Generation with Few-Shot Context](#patch-generation-with-few-shot-context)
7. [Self-Reflection as a Pre-Execution Gate](#self-reflection-as-a-pre-execution-gate)
8. [The Docker Sandbox](#the-docker-sandbox)
9. [Post-Mortems on Failure](#post-mortems-on-failure)
10. [Benchmarking: SWE-bench Lite and the pass@k Estimator](#benchmarking-swe-bench-lite-and-the-passk-estimator)
11. [Challenges and Open Problems](#challenges-and-open-problems)
12. [References](#references)

---

## The Automated Program Repair Problem

**Automated program repair (APR)** has a long research history predating language models. GenProg (Le Goues et al., 2012) used genetic programming to search the patch space; template-based systems like TBar (Liu et al., 2019) applied fix patterns from human-written code. These systems worked on constrained problem classes and required running tests to evaluate candidate patches, often evaluating thousands of candidates per bug.

LLMs changed the economics. A model can read a failing test, understand what it expects, look at the implementation, and generate a plausible patch in one forward pass, with no evolutionary search or template library required. The question is no longer whether LLMs can generate patches, but whether they can do so reliably in a closed loop without human oversight.

**SWE-bench** (Jimenez et al., 2023) operationalized this as a benchmark: given a real GitHub issue and the repository at the time it was filed, generate a patch that makes the failing tests pass without breaking the rest of the test suite. SWE-bench Lite contains 300 such tasks drawn from 12 popular Python repositories. Scores are reported as **pass@1**, the fraction of tasks where the agent's first attempt passes all tests.

The state of the art on SWE-bench Lite as of mid-2025 has moved quickly but remains well below full resolution. SWE-agent achieved a 12.5% resolution rate on SWE-bench; AutoCodeRover reached approximately 19% on SWE-bench Lite through AST-based code search. The harder SWE-bench Pro benchmark, designed to resist data contamination with enterprise-scale tasks, shows performance below 25% pass@1 across widely used models, with the highest score at 23.3%. The benchmark ceiling is real and meaningful: bugs that require multi-file changes, compile-time dependencies, or C extensions are systematically hard in ways current architectures do not address.

For the CI use case, the harder end of SWE-bench is not the primary target. Most CI regressions are scoped: a single failing test caused by a code change that is already understood. The architecture here is a clean, inspectable implementation of the full repair loop (context retrieval, patch generation, self-reflection, sandboxed execution, and structured failure analysis) that works well on that class of bug.

---

## Where Existing Agents Break Down

Before the implementation, here are the failure modes the design is responding to.

**Context retrieval is the first bottleneck.** A repository might have hundreds of files. Pasting everything into the context window is not feasible. Including only the file containing the failing test misses imports, parent classes, and shared utilities that are actually relevant to the bug. Getting the right context is both important and hard to do mechanically. Agents that read files indiscriminately burn many tokens on irrelevant code.

**One-shot patches have high variance.** An LLM generating a patch from a failing test and some context produces the most likely-looking fix according to its training distribution. If the actual bug is uncommon or the fix requires a non-obvious change, the first patch fails. Without a principled retry mechanism that incorporates what went wrong, the agent is sampling from the same distribution again with a slightly different prompt.

**No memory across tasks.** Each regression is treated as a completely novel event. If the agent previously fixed a timezone normalization bug in one service by stripping the UTC offset before formatting, nothing about that solution is available when an identical pattern appears in a different service three weeks later. Relevant past repairs exist in post-mortem logs, but nothing retrieves them.

**Sandbox safety is often an afterthought.** Running agent-generated code without isolation is genuinely dangerous. A faulty patch might `rm -rf` something, make network requests, or exhaust system memory. Most demo agents that run code directly on the host machine are betting that the LLM will not produce destructive code, which is not a reliable bet.

**Failure is opaque without structured analysis.** When an agent fails on a task, what have you learned? If the failure mode is just "tests still failing," you do not know whether the agent misidentified the root cause, generated a syntactically invalid patch, got blocked by missing test fixtures, or encountered a C extension. Structured failure analysis turns opaque failures into actionable information.

---

## Architecture: Six Nodes, One Sandbox

The agent is a LangGraph `StateGraph` with six nodes and conditional routing based on test outcomes and iteration count. The sixth node, `qdrant_retriever`, was added to provide the patch generator with few-shot examples from past successful repairs:

```
        START
          │
  ┌───────▼────────┐
  │ context_builder │  tree-sitter import graph → relevant files
  └───────┬─────────┘
          │
  ┌───────▼────────┐
  │qdrant_retriever │  top-3 similar past repairs from ci_regressions collection
  └───────┬─────────┘
          │
  ┌───────▼────────┐
  │ patch_generator │  Claude Sonnet 4.6, temp=0.2, unified diff + few-shot context
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
  store repair                             store postmortem
  in Qdrant                                in Qdrant (patch_succeeded=False)
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

    # retrieved by qdrant_retriever
    few_shot_examples: list[dict] | None  # top-3 similar past repairs

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

This is static analysis, not execution. It does not handle dynamic imports (`importlib.import_module(name)`), conditional imports (`if sys.platform == 'win32': import winreg`), or plugin systems. For the majority of bugs in well-structured Python services, static import tracing catches the relevant files.

Why tree-sitter rather than Python's own `ast` module? Two reasons: tree-sitter handles syntax errors gracefully (returning partial trees rather than raising), which matters because you are often analyzing code with bugs; and tree-sitter supports incremental reparsing, which is efficient if you need to reparse after applying a patch.

The context window budget is finite. After building the import graph, `context_builder` selects files by relevance (files directly imported by the failing tests rank higher than transitively imported files) and truncates at a configurable token limit.

---

## Qdrant as Few-Shot Regression Memory

The `qdrant_retriever` node addresses the "no memory across tasks" failure mode. After `context_builder` runs, the retriever queries a Qdrant collection called `ci_regressions` for the three most similar past repair cases. These are passed to `patch_generator` as few-shot examples.

The core intuition: if the agent has previously seen and fixed a bug where `test_serialize_datetime` failed with `AssertionError: '2024-01-15T10:00:00+05:30' != '2024-01-15T10:00:00Z'`, and the fix was to strip timezone information before formatting, then when a structurally identical failure appears in a different service, the model benefits from seeing that (failure pattern, patch, root cause) triple in its context. This is in-context learning with empirically grounded examples rather than abstract instruction.

### Collection Schema and Named Vectors

The collection uses two **named vectors** per point, each 1536 dimensions (suitable for OpenAI `text-embedding-3-small` or a code-specific model like `nomic-embed-code`):

| Vector name | What it encodes | Query use |
|---|---|---|
| `signature` | The structured test failure text: test name, module path, error type, assertion message | Primary search: find failures that *look* similar |
| `root_cause` | The natural-language root cause hypothesis from PostMortem | Secondary search: find failures with *similar causes* even if the test surface differs |

The *failing test signature* is a structured text block built from the test's pytest node ID, the exception class, and the first line of the assertion error:

```
test: tests/utils/test_date.py::test_format_date_timezone_aware
module: mylib.utils.date
error_type: AssertionError
error_msg: '2024-01-15T10:00:00+05:30' != '2024-01-15T10:00:00'
exit_code: 1
```

Embedding this structured representation rather than raw test output gives the embedding model a consistent schema to work with. Raw pytest output includes timestamps, file paths, and terminal colors that vary across runs and degrade embedding quality.

Named vectors in Qdrant allow both embedding spaces to live in one point without maintaining separate collections. A single `client.search()` call specifying `query_vector=NamedVector(name="signature", ...)` searches only the signature space, leaving `root_cause` available for independent queries or future multi-vector scoring.

**Collection creation:**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="ci_regressions",
    vectors_config={
        "signature": VectorParams(size=1536, distance=Distance.COSINE),
        "root_cause": VectorParams(size=1536, distance=Distance.COSINE),
    },
)
```

### Payload Design and Filtering

Each point in `ci_regressions` carries a payload with the fields needed for both filtering and downstream prompting:

| Field | Type | Purpose |
|---|---|---|
| `task_id` | `str` | Links back to the original CI run |
| `language` | `str` | `"python"` for now; enables multi-language expansion |
| `framework` | `str` | `"pytest"`, `"unittest"`, `"nose"` |
| `exit_code` | `int` | pytest exit code at time of failure |
| `failing_test_signature` | `str` | The structured text used for the embedding |
| `patch_diff` | `str` | The unified diff that resolved the regression |
| `root_cause_text` | `str` | The PostMortem root cause hypothesis |
| `patch_succeeded` | `bool` | False for post-mortem-only entries (agent failed) |
| `repo_name` | `str` | Repository identifier, for informational display |

**Why `exit_code` filtering is non-trivial:** pytest exit codes encode meaningfully different failure classes:

- `exit_code=1`: Tests ran but assertions failed. This is the standard "logic bug" case.
- `exit_code=2`: pytest collection error. The test file itself has an import error or syntax error; the bug is usually in a module import, not a logic path.
- `exit_code=3`: Internal pytest error.
- `exit_code=4`: Command-line usage error.
- `exit_code=137`: OOM kill from the Docker memory limit.

A few-shot example drawn from an `exit_code=2` run showed a patch that added a missing `__init__.py` and fixed a relative import. That example is actively misleading when the current failure is `exit_code=1` (a failed assertion about datetime formatting). Qdrant's payload filtering ensures the retrieved examples come from the same failure class:

```python
query_filter=Filter(
    must=[
        FieldCondition(key="language", match=MatchValue(value="python")),
        FieldCondition(key="framework", match=MatchValue(value="pytest")),
        FieldCondition(key="exit_code", match=MatchValue(value=1)),
        FieldCondition(key="patch_succeeded", match=MatchValue(value=True)),
    ]
)
```

The `patch_succeeded=True` filter is also important: post-mortem entries (where the agent failed) are stored with `patch_succeeded=False` and excluded from few-shot retrieval. Showing the model examples of failed repair attempts as positive few-shot context would be counterproductive.

### Upserting Successful Repairs

After `pr_opener` creates a pull request, the agent stores the successful repair in Qdrant. The upsert happens outside the LangGraph graph (as a side-effect of the success path) to keep the graph nodes pure:

```python
import uuid
from qdrant_client.models import NamedVector, PointStruct


def store_repair(
    task_id: str,
    failing_sig: str,        # structured text: "test:...\nmodule:...\nerror:..."
    patch_diff: str,
    root_cause_text: str,
    language: str,
    framework: str,
    exit_code: int,
    repo_name: str,
    embed_fn,                # callable[[str], list[float]]
) -> None:
    """Persist a successful repair triple to the ci_regressions collection."""
    client.upsert(
        collection_name="ci_regressions",
        points=[
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, task_id)),
                vector={
                    "signature": embed_fn(failing_sig),
                    "root_cause": embed_fn(root_cause_text),
                },
                payload={
                    "task_id": task_id,
                    "language": language,
                    "framework": framework,
                    "exit_code": exit_code,
                    "failing_test_signature": failing_sig,
                    "patch_diff": patch_diff,
                    "root_cause_text": root_cause_text,
                    "patch_succeeded": True,
                    "repo_name": repo_name,
                },
            )
        ],
    )
```

`uuid.uuid5` with the task ID as the name produces a deterministic ID: re-running a task after a re-run does not create a duplicate point; it overwrites the existing one. This is the correct behavior for a repair that was retried.

### Retrieval at Repair Time

The `qdrant_retriever` node runs after `context_builder` and before `patch_generator`. It extracts the failing test signature from `AgentState`, embeds it, and queries the collection:

```python
def retrieve_few_shot(
    failing_sig: str,
    language: str,
    framework: str,
    exit_code: int,
    embed_fn,
    top_k: int = 3,
) -> list[dict]:
    """Return top-k similar past successful repairs, filtered by execution context."""
    results = client.search(
        collection_name="ci_regressions",
        query_vector=NamedVector(name="signature", vector=embed_fn(failing_sig)),
        query_filter=Filter(
            must=[
                FieldCondition(key="language", match=MatchValue(value=language)),
                FieldCondition(key="framework", match=MatchValue(value=framework)),
                FieldCondition(key="exit_code", match=MatchValue(value=exit_code)),
                FieldCondition(key="patch_succeeded", match=MatchValue(value=True)),
            ]
        ),
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "failing_test_signature": r.payload["failing_test_signature"],
            "patch_diff": r.payload["patch_diff"],
            "root_cause_text": r.payload["root_cause_text"],
            "similarity": r.score,
        }
        for r in results
    ]
```

The search uses COSINE distance on the `signature` vector. Qdrant's HNSW index makes this fast even at scale: a collection of 50,000 historical CI failures returns results in under 5ms at ef=128.

When the collection is empty (a fresh deployment), `retrieve_few_shot` returns an empty list and `few_shot_examples` in `AgentState` is `None`. The `patch_generator` prompt checks for this and simply omits the few-shot section. The agent degrades gracefully to its baseline behavior.

### Why Qdrant over Simpler Alternatives

The obvious alternative is a key-value store or a full-text index. Both fall short in specific ways.

**Key-value stores** (Redis, DynamoDB) match exact keys. "Find past repairs for bugs similar to this one" has no exact key; the similarity is in embedding space, not in literal strings.

**Full-text search** (Elasticsearch, SQLite FTS) matches keyword overlap. A regression in `test_parse_iso_date` and a structurally identical one in `test_format_utc_timestamp` share almost no keyword overlap, even though they represent the same underlying failure pattern. Embedding-based search captures semantic and structural similarity that surface-level text matching misses.

**Qdrant specifically** is chosen over other ANN stores for three reasons:

1. **Payload filtering with HNSW pre-filtering:** Qdrant's filtered HNSW search applies the payload filter during the graph traversal, not as a post-filter over the full result set. This matters when the filter is selective (only 10% of repairs match `exit_code=2`): pre-filtering avoids retrieving and scoring hundreds of irrelevant points.

2. **Named vectors per point:** A single collection holds both the `signature` and `root_cause` vector spaces. Without named vectors, you would need two separate collections with manual ID synchronization, or a single concatenated vector that mixes the two spaces (which degrades both).

3. **On-disk indexing:** CI repair histories grow over months. Qdrant's `on_disk_payload` and `memmap_threshold` options allow collections to exceed available RAM without performance collapse, which matters if you're indexing five years of CI failures from a large codebase.

---

## Patch Generation with Few-Shot Context

`patch_generator` produces patches in **unified diff format**, the standard `git diff` output format with `---`, `+++`, and `@@ ... @@` headers. This is a deliberate choice over asking the model to produce full file contents or Python AST manipulations.

Unified diff has several properties that matter here:

**Verbosity scales with change size.** A one-line fix produces a short diff. A ten-line refactor produces a longer one. Full file output is always the file size, regardless of how small the change is.

**Parseable and applicable mechanically.** `patch_generator` produces text; `sandbox_executor` applies it with `subprocess.run(["patch", "-p1", ...])`. The patch utility handles offset matching, rejects cleanly if context does not match, and reports exactly which hunks succeeded or failed.

**Reviewable by humans and LLMs.** Unified diffs are what code review tools display. The self-reflector node reads the diff and can reason about what changed and what did not, which would be harder with full file output.

The model is called at `temperature=0.2`, low but not zero. Zero temperature produces the single most likely token at each step, which can sometimes produce locally consistent but globally incoherent patches (a common failure mode is fixing the immediate assertion while missing a related invariant elsewhere). A small amount of temperature adds diversity without sacrificing coherence. On retry after a failed test, the temperature is unchanged: the `patch_generator` already has the test failure output in its context, which is more informative than sampling variation.

The prompt structure for `patch_generator` now includes the Qdrant-retrieved few-shot examples when available:

```
System: You are an expert Python engineer. You will be given:
1. A failing test with its full output
2. The relevant source files (identified from the import graph)
3. The issue description
4. [If available] Up to 3 similar past regressions with their fixes
5. [On retry] The previous patch and why it failed

Produce a unified diff that fixes the failing test without breaking others.
Output ONLY the diff, no explanation.

[context_builder output: relevant files]
[failing test output]
[issue description]
[few-shot examples from Qdrant, if any]
[previous attempt if iteration > 0]
```

The few-shot section, when present, looks like this in the prompt:

```
--- Similar past repairs (retrieved from regression history) ---

Example 1 (similarity: 0.91):
Failing test signature:
  test: tests/api/test_serializers.py::test_serialize_created_at
  module: myapp.api.serializers
  error_type: AssertionError
  error_msg: '2024-01-15T10:00:00+05:30' != '2024-01-15T10:00:00Z'
  exit_code: 1

Root cause: format_datetime() did not normalize to UTC before calling
  isoformat(), so timezone-aware datetimes retained the local offset.

Fix:
--- a/myapp/api/serializers.py
+++ b/myapp/api/serializers.py
@@ -14,7 +14,7 @@
 def format_datetime(dt: datetime) -> str:
-    return dt.isoformat()
+    return dt.astimezone(timezone.utc).isoformat()
```

The "output ONLY the diff" instruction matters because the subsequent `patch` command needs clean diff text. Any explanation before or after the diff will cause the patch application to fail.

The few-shot examples benefit the generator in two ways: they provide a concrete patch format to imitate, and when the similarity score is high, they directly suggest the fix pattern. A model that has seen the UTC normalization fix once in context is substantially more likely to apply it correctly on a structurally identical regression.

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

If `REVISE` and `iteration < 2`, the critique is appended to the `patch_generator` context and the graph loops back. If `iteration >= 2`, the patch is sent to the sandbox regardless. At that point, sandbox execution results are more informative than another LLM opinion.

This two-call structure is a lightweight version of the **Reflexion** pattern (Shinn et al., 2023). In Reflexion, an actor generates an output, an evaluator scores it, and a self-reflection module diagnoses the issue and proposes a fix, appending both the failed output and the suggested correction to the input context. The `self_reflector` here is the evaluator step applied *before* execution rather than after: a pre-execution filter rather than a post-execution teacher.

The `self_reflector` prompt looks for specific failure modes:

```
Given this patch, check for:
1. Does the fix address the root cause stated in the issue, or just the symptom?
2. Are there any obvious regression risks: tests that currently pass that this
   change might break?
3. Is the diff syntactically valid and coherent?

If all three look reasonable, output exactly: APPROVED
Otherwise, output exactly: REVISE: [one sentence critique]
```

The value of this gate is asymmetric. A patch that looks wrong to the reflector before execution often *is* wrong: catching it saves a Docker container startup (5-10 seconds) and keeps the test failure output from entering the context window. A patch the reflector approves that still fails in the sandbox provides useful new information (the sandbox output) for the next attempt. The gate filters obvious mistakes cheaply; the sandbox catches the rest.

One design note on the iteration bound: the reflector runs on iterations 0 and 1, then the patch goes straight to the sandbox from iteration 2 onward. By iteration 2, the LLM has seen two test failure outputs with their full tracebacks. That is richer information than another LLM opinion. Continuing to apply the pre-execution gate past iteration 2 would slow convergence without improving it.

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

This is the core feedback loop: the sandbox output is the ground truth. A patch that looks correct to both the generator and the reflector but fails in the sandbox gets its failure traceback appended to the context. The next `patch_generator` call has the original test failure, the first patch, the first test output, and now the second test output. The context accumulates real execution evidence rather than LLM speculation.

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

The `next_steps` field produces exactly three concrete items for a human engineer. The constraint is deliberate: open-ended lists of potential next steps are often useless. Three specific items is enough to be actionable without being overwhelming.

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

Post-mortem entries are also stored in the Qdrant collection with `patch_succeeded=False`. They are excluded from few-shot retrieval (the filter requires `patch_succeeded=True`), but they serve a separate purpose: offline analysis. By querying for all failed repairs grouped by `exit_code` and `root_cause_text` similarity, a team can identify classes of bugs the agent systematically fails on and invest in targeted improvements (better context retrieval for multi-file bugs, better prompting for C extension failures, etc.).

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

The few-shot retrieval from Qdrant is a cold-start problem on the benchmark: the `ci_regressions` collection is empty at the beginning of a run, so the first tasks receive no few-shot examples. As the run progresses and successful repairs accumulate, later tasks benefit from retrieved examples. A fair evaluation requires either pre-seeding the collection from a prior run or reporting pass@1 separately for early tasks (cold) and late tasks (warm). This warm/cold distinction is worth tracking explicitly, as it measures the empirical value of the retrieval layer.

---

## Challenges and Open Problems

**Multi-file bugs are structurally hard.** The primary expected failure mode: the agent modifies one file but the fix spans two or more interdependent modules. The patch applies cleanly but the tests still fail. The import graph helps identify relevant files, but `patch_generator` generates a single unified diff. Bugs that require coordinated changes across files (an interface change that requires updating both the implementation and all callers) need multi-file patch support. This is non-trivial: the patch format handles multi-file diffs but the model needs to reason about consistency across file boundaries simultaneously.

**Test infrastructure that does not exist in the sandbox.** Some tasks have tests that depend on external fixtures: databases that need to be seeded, network calls that need to be mocked, filesystem paths that need to exist. The sandbox's `--network none` flag catches some of these early (network-dependent tests fail immediately), but filesystem fixture problems are harder. The agent sees a test failure that is not caused by its patch at all, which is confusing and wasteful.

**Self-reflection reliability.** The reflector is called with the same base model as the generator (`claude-sonnet-4-6`). This is the standard LLM-as-judge critique: a model evaluating its own outputs has an inherent bias toward approving them. Empirically, the reflector catches obvious mistakes (syntax errors, obviously unrelated changes), but it is unlikely to catch subtle logical errors that it would also make in generation. Using a stronger or differently-prompted judge model, or cross-checking against a lightweight static analysis pass, would improve the gate's effectiveness.

**Context window accumulation.** The state carries the full history of attempts across iterations: previous patches, previous test outputs, critique texts. At `max_iterations=5`, the context for the fifth attempt includes four failed patches plus their sandbox outputs, which can easily total 20,000+ tokens. This crowds out the relevant source files. Selective context compression, summarizing earlier iterations rather than including them verbatim, would help, but requires deciding what to summarize and what to preserve.

**Qdrant cold start and embedding quality.** The few-shot retrieval is most valuable after the collection has accumulated hundreds of diverse repairs. For a team deploying this for the first time, the collection starts empty and the retrieval layer adds latency without benefit for the first few dozen tasks. One mitigation is to pre-seed the collection from SWE-bench resolutions or from the team's historical issue tracker. The other open question is embedding model choice: general-purpose text embeddings (text-embedding-3-small) handle natural language root cause descriptions well, but code-specific models (voyage-code-2, nomic-embed-code) may produce better signal for the `signature` vector, which contains module paths and assertion strings. This is worth empirically measuring once the collection has sufficient data.

**Qdrant payload staleness.** Repairs stored months ago may use outdated API patterns, deprecated library calls, or outdated language constructs. A retrieved few-shot example that patches `datetime.utcnow()` (deprecated in Python 3.12) into code is technically correct but introduces a new warning. The collection should include a `stored_at` timestamp in the payload to allow recency-weighted retrieval or time-bounded filters.

**The PR opener is optimistic.** When a task passes, `pr_opener` creates a GitHub pull request via PyGithub. But applying this agent to a real CI pipeline rather than the benchmark's sandboxed snapshots introduces complexity that the benchmark is specifically designed to eliminate: conflicting in-flight changes, repository-specific CI requirements, review processes, and the possibility that the "passing tests" criterion is necessary but not sufficient for correctness. The gap between "passes the specific failing test" and "is correct to merge" is significant. For production use, the PR should be treated as a draft for human review, not an automatic merge.

The self-healing-agent architecture is a coherent answer to what "closing the loop" actually means for code repair: static import analysis to retrieve relevant context, vector-based regression memory to provide empirically grounded few-shot examples, pre-execution critique to filter obvious mistakes, sandboxed execution to get ground truth, test output fed back as context for revision, and structured failure analysis when the budget runs out. The components each do one thing and hand off cleanly to the next.

---

## References

- Inamdar, Mihir. *self-healing-agent: Self Healing AI Agent in Sandbox*. [github.com/inamdarmihir/self-healing-agent](https://github.com/inamdarmihir/self-healing-agent)
- Jimenez, Carlos E. et al. (2023). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* arXiv:2310.06770
- Shinn, Noah et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023.
- Chen, Mark et al. (2021). *Evaluating Large Language Models Trained on Code*. arXiv:2107.03374
- Yang, John et al. (2024). *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*. NeurIPS 2024.
- Le Goues, Claire et al. (2012). *GenProg: A Generic Method for Automatic Software Repair*. IEEE TSE.
- Liu, Kui et al. (2019). *TBar: Revisiting Template-Based Automated Program Repair*. ISSTA 2019.
- Qdrant. *Qdrant Vector Database Documentation*. [qdrant.tech/documentation](https://qdrant.tech/documentation)
