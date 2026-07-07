---
title: "Building a Local Code-Runner Agent with DeepAgents"
date: 2026-06-11
description: "A concrete walkthrough of building a sandboxed code-execution agent using the DeepAgents framework, covering the sandbox-as-tool pattern, local shell backends, filesystem tooling, and the design decisions that make autonomous code execution safe enough to ship."
tags: ["agents", "deepagents", "sandbox", "langgraph", "code-execution", "qdrant"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Most tutorials on coding agents stop at the point where the agent writes code. The harder problem is running it: safely, observably, and without letting the agent touch your host filesystem, environment variables, or network in ways you did not intend. This post walks through a concrete implementation of a local code-runner agent built on the **DeepAgents** framework ([langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)), which wraps LangGraph with a filesystem backend, context management, and first-class sandbox support.

Two orthogonal problems make code-execution agents non-trivial. The first is *isolation*: the agent needs a contained environment where its commands cannot damage the host. The second is *memory*: the agent's tool list grows with task complexity, its context window fills with tool outputs, and each invocation starts with no knowledge of prior runs. I will use Qdrant as the layer that addresses the memory side: a `tool_registry` collection for semantic tool discovery, and a `session_memory` collection for persistent, retrievable execution context.

I will focus on implementation decisions rather than framework overview. The post assumes familiarity with LangGraph's graph and state primitives and basic Python async patterns. It does not assume prior experience with sandboxing, container isolation, or vector databases.

## Table of Contents

1. [Why Code Execution Is Different from Tool Use](#why-code-execution-is-different-from-tool-use)
2. [The DeepAgents Sandbox Architecture](#the-deepagents-sandbox-architecture)
3. [The Local Shell Backend](#the-local-shell-backend)
4. [The System Prompt and Tool Visibility](#the-system-prompt-and-tool-visibility)
5. [Qdrant as Tool Registry](#qdrant-as-tool-registry)
6. [Qdrant as Session Memory](#qdrant-as-session-memory)
7. [The Agent Loop with Retrieval-Augmented Context](#the-agent-loop-with-retrieval-augmented-context)
8. [Context Management and Token Budget](#context-management-and-token-budget)
9. [File Transfer: Seeding and Retrieval](#file-transfer-seeding-and-retrieval)
10. [Security Considerations for Local Sandboxes](#security-considerations-for-local-sandboxes)
11. [Running the Agent](#running-the-agent)
12. [What This Pattern Enables](#what-this-pattern-enables)
13. [Challenges and Open Problems](#challenges-and-open-problems)
14. [References](#references)

## Why Code Execution Is Different from Tool Use

Standard tool-use agents call deterministic, bounded functions: search the web, query a database, send a message. Code execution differs in two fundamental ways.

First, the action space is unbounded. An agent calling `execute("rm -rf /")` and an agent calling `execute("pytest tests/")` are structurally identical from the framework's perspective. The difference is intent, which cannot be verified statically.

Second, code execution is stateful in a way that ordinary tool calls are not. Installing a package changes the environment for all subsequent commands. Writing a file persists across tool calls. This statefulness is useful (it is what lets the agent iterate on a failing test rather than starting from scratch) but it means the execution environment accumulates state that both the agent and the developer must reason about.

The sandbox abstraction addresses isolation: it bounds the unbounded by confining execution to a disposable environment that disappears when the task ends. Memory retrieval, the focus of the Qdrant sections below, addresses the complementary problem of what the agent *knows* when it begins: which tools are available and what happened in prior steps of the same session.

## The DeepAgents Sandbox Architecture

DeepAgents represents sandboxes as *backends* (the same abstraction used for filesystems and stores, but with one additional capability: the `execute` tool). When `create_deep_agent()` detects that the configured backend implements `SandboxBackendProtocol`, it automatically adds `execute` to the agent's tool list. When the backend does not implement the protocol, `execute` is filtered out and the agent never sees it. This conditional availability is re-evaluated on every model call, not just at startup.

The only method a sandbox provider must implement is `execute(command: str) -> ExecuteResult`. Every other filesystem operation (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) is built on top of `execute()` by the `BaseSandbox` base class, which constructs shell scripts and runs them in the sandbox via this single method. This design means adding a new provider requires implementing one method.

There are two architectural patterns for integrating agents with sandboxes.

**Agent in sandbox.** The agent process runs inside the sandbox. You communicate with it over the network via WebSocket or HTTP. API keys must live inside the sandbox, updates require rebuilding images, and you need to manage the communication infrastructure separately. This pattern mirrors local development closely but has significant security implications: credentials inside the sandbox are accessible to a context-injected agent.

**Sandbox as tool.** The agent runs on your machine or server. When it needs to execute code, it calls sandbox tools that invoke the provider's API remotely. API keys stay outside the sandbox, agent logic can be updated without rebuilding images, and sandbox failures do not lose agent state. This is the recommended pattern for most use cases and the one this post follows.

The `deepagents-local-sandbox` repo implements sandbox-as-tool using a local shell backend (the sandbox is a subprocess on your machine with a scoped working directory, not a remote VM). The same agent code works with Modal, Runloop, or Daytona by swapping the backend.

## The Local Shell Backend

The local backend creates a scoped workspace directory and executes all commands relative to it. "Scoped" means the agent can read and write files within the workspace freely but cannot escape to the parent filesystem without an explicit `..` traversal, which the system prompt discourages.

```python
from deepagents import create_deep_agent
from deepagents.backends import LocalShellSandbox
from langchain_anthropic import ChatAnthropic
from pathlib import Path

workspace = Path("./sandbox-workspace")
workspace.mkdir(exist_ok=True)

backend = LocalShellSandbox(working_directory=str(workspace))

agent = create_deep_agent(
    model=ChatAnthropic(model="claude-sonnet-4-6", temperature=0),
    system_prompt=SYSTEM_PROMPT,
    backend=backend,
)
```

`LocalShellSandbox` wraps Python's `subprocess` module. Each `execute()` call spawns a subprocess with `cwd` set to the workspace and captures combined stdout and stderr. The exit code is returned alongside the output so the agent can distinguish success from failure without parsing output text.

One non-obvious behavior: the local shell backend does *not* maintain shell state across `execute()` calls. Environment variables set in one call (`export FOO=bar`) are not visible in the next call. This is intentional (it prevents implicit state from accumulating invisibly) but it means the agent must pass state between commands explicitly: via files, or by chaining commands with `&&`.

## The System Prompt and Tool Visibility

The system prompt does more work in a code-execution agent than in most agents. It tells the model what environment it is operating in, what the working directory structure is, and what the conventions are for iteration.

```python
SYSTEM_PROMPT = """You are a coding assistant with access to a sandboxed shell environment.

### Environment
- Working directory: /workspace (all paths relative to here)
- Python 3.11 available, pip available
- You can install packages with pip install
- Git is available

### Iteration conventions
1. Write code to a file first, then execute it
2. On failure, read the error output carefully before retrying
3. Check exit codes -- a non-zero exit means the command failed
4. Verify your work: run the code, check the output matches what was asked

### What you should NOT do
- Do not attempt to access paths outside /workspace
- Do not install packages you do not need for the task
- Do not leave temporary files in /workspace when the task is complete
"""
```

The DeepAgents harness adapts the system prompt automatically when a sandbox backend is detected, appending context about the remote execution environment. For the local backend, this adaptation is minimal since the agent is running in the same environment as its workspace.

The `execute` tool the agent sees has the following signature:

```
execute(command: str) -> str

Run a shell command in the sandbox working directory.
Returns combined stdout/stderr and exit code.
Large outputs are automatically saved to a file.
```

The large-output truncation is handled by the harness, not the backend. When `execute()` returns more than 20,000 tokens of output, the harness saves the full output to a file in the workspace and substitutes a truncated preview plus a path reference. The agent can then use `read_file` to access the full output incrementally.

## Qdrant as Tool Registry

A production code-execution agent may expose dozens of tools: the core sandbox tools (`execute`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`), task-specific utilities (a code linter, a type checker, a test-coverage tool), and domain tools (a database query interface, an API client). Sending all tool descriptors to the model on every call is wasteful: it consumes context budget for tools unlikely to be relevant, and degrades the model's ability to select precisely when many irrelevant tools are present (Li et al. 2023).

A more tractable approach: store all tool descriptors in a vector database and at each step retrieve the top-$K$ most semantically relevant tools by embedding the current task description. Qdrant is a good fit here because its payload filtering lets you express structured constraints (tool category, required permissions, availability in the current backend) alongside semantic similarity, without a separate metadata index.

### Collection Schema: `tool_registry`

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    PointStruct, PayloadSchemaType,
)

client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="tool_registry",
    vectors_config=VectorParams(
        size=1536,          # text-embedding-3-small output dim
        distance=Distance.COSINE,
    ),
)

# Payload indexes for fast filtering during ANN scan
client.create_payload_index(
    collection_name="tool_registry",
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD,
)
client.create_payload_index(
    collection_name="tool_registry",
    field_name="requires_sandbox",
    field_schema=PayloadSchemaType.BOOL,
)
client.create_payload_index(
    collection_name="tool_registry",
    field_name="backend_compat",
    field_schema=PayloadSchemaType.KEYWORD,
)
```

Each point in the collection represents one tool. The vector is the embedding of the tool's name concatenated with its description (and optionally a representative example invocation). The payload carries the structured metadata needed for filtering and for constructing the tool spec sent to the model:

```python
from openai import OpenAI

openai_client = OpenAI()

def embed(text: str) -> list[float]:
    resp = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return resp.data[0].embedding

TOOLS = [
    {
        "name": "execute",
        "description": (
            "Run a shell command in the sandbox working directory. "
            "Returns combined stdout/stderr and exit code."
        ),
        "input_schema": {"command": "str"},
        "category": "sandbox",
        "requires_sandbox": True,
        "backend_compat": ["local", "modal", "runloop", "daytona"],
    },
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file in the workspace. "
            "Supports byte-range reads for large files."
        ),
        "input_schema": {"path": "str", "start_byte": "int?", "end_byte": "int?"},
        "category": "filesystem",
        "requires_sandbox": False,
        "backend_compat": ["local", "modal", "runloop", "daytona"],
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file in the workspace, "
            "creating parent directories as needed."
        ),
        "input_schema": {"path": "str", "content": "str"},
        "category": "filesystem",
        "requires_sandbox": False,
        "backend_compat": ["local", "modal", "runloop", "daytona"],
    },
    # additional tools follow the same schema
]

points = []
for i, tool in enumerate(TOOLS):
    embed_text = f"{tool['name']}: {tool['description']}"
    points.append(PointStruct(
        id=i,
        vector=embed(embed_text),
        payload={
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["input_schema"],
            "category": tool["category"],
            "requires_sandbox": tool["requires_sandbox"],
            "backend_compat": tool["backend_compat"],
        },
    ))

client.upsert(collection_name="tool_registry", points=points)
```

The payload schema for each point:

| Field | Type | Description |
|---|---|---|
| `name` | `keyword` | Tool name; unique identifier |
| `description` | `text` | Human-readable description used for embedding |
| `input_schema` | `object` | Parameter names and types |
| `category` | `keyword` | One of: `sandbox`, `filesystem`, `analysis`, `domain` |
| `requires_sandbox` | `bool` | Whether the tool requires a sandbox backend |
| `backend_compat` | `keyword[]` | Backends on which the tool is available |

### Tool Retrieval at Each Agent Step

At the start of each ReAct step, before the model call, the agent embeds the current task description and retrieves the top-$K$ tools with an optional backend filter:

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

def retrieve_tools(
    task_description: str,
    backend_name: str,
    top_k: int = 8,
) -> list[dict]:
    query_vector = embed(task_description)

    results = client.search(
        collection_name="tool_registry",
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="backend_compat",
                    match=MatchValue(value=backend_name),
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
    )

    return [hit.payload for hit in results]
```

The `backend_compat` filter ensures the agent only sees tools available for its current backend, without a separate lookup. For a task like "run pytest on the test suite," the retrieval surfaces `execute`, `read_file`, `glob`, and potentially a coverage tool from the `analysis` category. A task like "add type hints to a module" surfaces `read_file`, `write_file`, `edit_file`, and perhaps a type-checker tool.

Why Qdrant for this rather than a simple dictionary lookup? Two reasons. First, semantic search handles vocabulary mismatch: the agent's internal description of the current step may use different words than the tool's documentation, and COSINE similarity finds the right tool even when the phrasing differs. Second, Qdrant's payload filtering lets you express constraints (backend compatibility, permission level, category) as structured filters applied *during* the ANN scan, not as a post-filter on the full result set. For a registry with hundreds of tools across multiple backends, this matters for latency.

Qdrant's HNSW index keeps retrieval latency below 5 ms at this scale, well within the budget of a ReAct loop that already pays 200-500 ms for the model call. The tool registry collection fits comfortably in memory for any realistic number of tools (a registry with 1,000 tools at 1,536 dimensions occupies roughly 6 MB of vector data).

## Qdrant as Session Memory

Each invocation of the agent starts with no knowledge of prior runs by default. For a one-shot task this is acceptable. For iterative or multi-session workflows (refactoring a codebase over several hours, building a data pipeline step by step), the agent should be able to retrieve relevant context from earlier steps without carrying that context in the full context window.

The session memory design stores every significant agent step as a Qdrant point: code written, outputs observed, variable bindings established. At the start of each new step, the agent retrieves the most relevant past steps by semantic similarity to the current task, filtered by session and optionally by recency.

### Collection Schema: `session_memory`

```python
client.create_collection(
    collection_name="session_memory",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE,
    ),
)

# Payload indexes for efficient per-session and per-type filtering
client.create_payload_index(
    collection_name="session_memory",
    field_name="session_id",
    field_schema=PayloadSchemaType.KEYWORD,
)
client.create_payload_index(
    collection_name="session_memory",
    field_name="step_idx",
    field_schema=PayloadSchemaType.INTEGER,
)
client.create_payload_index(
    collection_name="session_memory",
    field_name="type",
    field_schema=PayloadSchemaType.KEYWORD,
)
client.create_payload_index(
    collection_name="session_memory",
    field_name="timestamp",
    field_schema=PayloadSchemaType.INTEGER,
)
```

The payload schema for each session memory point:

| Field | Type | Description |
|---|---|---|
| `session_id` | `keyword` | Opaque session identifier, e.g. a UUID |
| `step_idx` | `integer` | Zero-indexed step within the session |
| `type` | `keyword` | One of: `code`, `output`, `variable`, `observation` |
| `content` | `text` | Actual content: code, output text, or variable assignment |
| `tool_name` | `keyword` | Which tool produced or consumed this step |
| `exit_code` | `integer` | For `output` type: the exit code of the command |
| `timestamp` | `integer` | Unix timestamp in milliseconds |
| `tags` | `keyword[]` | Semantic tags: `error`, `success`, `install`, `test` |

The vector for each point is the embedding of the `content` field (for code and output) or a structured description (for variable bindings). For long outputs that exceed the embedding model's token limit, embed a truncated prefix plus a one-sentence summary.

### Writing Steps to Session Memory

```python
import uuid
import time

SESSION_ID = str(uuid.uuid4())

def store_step(
    step_idx: int,
    step_type: str,
    content: str,
    tool_name: str,
    exit_code: int | None = None,
    tags: list[str] | None = None,
) -> None:
    vector = embed(content[:2000])  # truncate before embedding
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "session_id": SESSION_ID,
            "step_idx": step_idx,
            "type": step_type,
            "content": content,
            "tool_name": tool_name,
            "exit_code": exit_code,
            "timestamp": int(time.time() * 1000),
            "tags": tags or [],
        },
    )
    client.upsert(collection_name="session_memory", points=[point])
```

The agent calls `store_step` after each tool invocation, categorizing the result by type. Writing a file produces a `code` point. Running a command and observing its output produces an `output` point (with `exit_code`). Binding a variable in a script produces a `variable` point.

### Retrieving Relevant Past Steps

```python
from qdrant_client.models import Range

def retrieve_context(
    current_task: str,
    session_id: str,
    top_k: int = 5,
    max_step_age: int | None = None,
) -> list[dict]:
    query_vector = embed(current_task)

    must_conditions = [
        FieldCondition(
            key="session_id",
            match=MatchValue(value=session_id),
        )
    ]

    if max_step_age is not None:
        cutoff_ts = int(time.time() * 1000) - max_step_age * 1000
        must_conditions.append(
            FieldCondition(
                key="timestamp",
                range=Range(gte=cutoff_ts),
            )
        )

    results = client.search(
        collection_name="session_memory",
        query_vector=query_vector,
        query_filter=Filter(must=must_conditions),
        limit=top_k,
        with_payload=True,
    )

    # Sort by step_idx to reconstruct execution order
    hits = sorted(
        [r.payload for r in results],
        key=lambda p: p["step_idx"],
    )
    return hits
```

The retrieval returns the top-$K$ past steps most semantically similar to the current task, filtered to the current session and optionally to a recency window. Sorting by `step_idx` before injecting into the context window gives the model a coherent causal narrative rather than a temporally scrambled list ordered by similarity score.

A representative retrieval for the task "add error handling to the divide function" might surface: the `write_file` step where `ops.py` was first written (type: `code`), the `execute` step where the test suite failed with a `ZeroDivisionError` (type: `output`, tags: `error`, `test`), and the `write_file` step where the guard was added (type: `code`). These three steps give the model the provenance it needs without replaying the full conversation history.

### Why Qdrant for Session Memory

The practical alternative to a vector store for session memory is full conversation replay: include every prior step in the context window. For short sessions this is acceptable. For long sessions (50+ steps with large code files and verbose test outputs), it consumes most of the context budget and degrades model performance as the window fills.

Qdrant's payload filtering addresses two structural problems that a pure FAISS index or in-memory embedding search cannot handle. First, session isolation: a single Qdrant collection can serve multiple concurrent agents, with `session_id` filtering ensuring each agent's retrieval is scoped to its own history. Second, structured constraints: filtering by `type = "output"` and `tags contains "error"` lets the agent retrieve specifically the failed runs from its history, not merely the semantically closest steps. These filters apply inside the HNSW scan, so the cost is proportional to the filtered result count rather than the total collection size.

For scalar quantization (Qdrant supports `ScalarQuantization(type="int8")` natively), session memory vectors compress 4x with minimal recall degradation, typically below 1% at top-5. For a long-running agent accumulating thousands of steps, this keeps the collection footprint manageable without a separate compression pipeline.

## The Agent Loop with Retrieval-Augmented Context

With both the tool registry and session memory in Qdrant, the agent loop has an explicit retrieval phase before each model call:

```python
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage

async def run_agent_step(
    task: str,
    session_id: str,
    step_idx: int,
    backend_name: str,
) -> dict:
    # 1. Retrieve relevant tools for this task
    tools = retrieve_tools(task, backend_name=backend_name, top_k=8)
    tool_specs = [format_tool_spec(t) for t in tools]

    # 2. Retrieve relevant past steps from this session
    past_steps = retrieve_context(task, session_id=session_id, top_k=5)
    context_block = format_context_block(past_steps)

    # 3. Build the prompt with retrieved context appended to system prompt
    messages = [
        SystemMessage(content=SYSTEM_PROMPT + context_block),
        HumanMessage(content=task),
    ]

    # 4. Model call with retrieved tool subset only
    response = await model.ainvoke(messages, tools=tool_specs)

    # 5. Execute tool calls and store results in session_memory
    for tool_call in response.tool_calls:
        result = await execute_tool(tool_call, backend=backend)
        store_step(
            step_idx=step_idx,
            step_type=classify_result(tool_call.name, result),
            content=result.output,
            tool_name=tool_call.name,
            exit_code=result.exit_code,
            tags=extract_tags(tool_call.name, result),
        )
        step_idx += 1

    return {"response": response, "next_step_idx": step_idx}
```

`format_context_block` converts the retrieved past steps into a structured block injected at the end of the system prompt:

```
### Relevant context from this session
[step 3 | write_file | ops.py]
def divide(a, b):
    return a / b

[step 5 | execute | python -m pytest tests/ -v]
FAILED tests/test_ops.py::test_divide_by_zero
ZeroDivisionError: division by zero
exit_code: 1
tags: error, test
```

This gives the model a grounded narrative: it sees what was tried, what failed, and can reason about what to do next without consuming the full message history.

A typical successful execution on the task "create a Python package with pytest tests, then run them" looks like:

1. Agent retrieves tools: `write_file`, `execute`, `glob` (top-3 by cosine similarity to the task)
2. Agent creates `mathutils/__init__.py` and `mathutils/ops.py` via `write_file`; both stored as `code` steps
3. Agent creates `tests/test_ops.py` via `write_file`
4. Agent calls `execute("pip install pytest -q")`; stored as `output` step with tag `install`
5. Agent calls `execute("python -m pytest tests/ -v")`; one test fails; stored with tags `error`, `test`
6. Next step retrieval surfaces step 5 (high similarity to "fix failing test")
7. Agent edits `mathutils/ops.py` to add a zero-division guard; stored as `code` step
8. Agent calls `execute("python -m pytest tests/ -v")` again; all tests pass; stored with tag `success`
9. Agent creates `README.md` and returns a summary

Steps 5 through 7 are the iteration the sandbox makes safe. Without isolation, a failing test that triggered an unintended side effect (writing to a config file, making a network request) would be invisible and potentially harmful. With the local sandbox, the blast radius is the workspace directory. With Qdrant session memory, the agent retains the provenance of the failure and fix across subsequent steps without replaying the full context.

## Context Management and Token Budget

Code-execution agents consume context quickly. A multi-step task that installs packages, writes files, and runs tests can accumulate thousands of tokens of tool outputs before the task finishes. DeepAgents handles this with automatic context offloading.

When a tool input or result exceeds 20,000 tokens, the harness offloads the content to the backend filesystem and substitutes a file reference with a 10-line preview. The agent sees:

```
[Output truncated -- full content saved to /workspace/.deepagents/tool_outputs/execute_1.txt]
Preview:
...
```

The agent can retrieve the full content with `read_file` when it needs it. This keeps the active context window lean while preserving full observability.

The interaction between Qdrant session memory and context offloading is complementary rather than redundant. Offloading handles large outputs within a single session run: the full content is on disk, the agent reads it incrementally via `read_file`. Qdrant session memory handles cross-step retrieval: the agent recovers semantically relevant steps from earlier in the session without holding them in the context window. Together they allow long-horizon tasks that would otherwise hit the context limit.

There is a subtle interaction between context offloading and multi-step iteration. If the agent offloads a large pytest output, then iterates on the code, then runs pytest again, it now has two references in its context but the second supersedes the first. The agent generally attends to the most recent tool result, but for tasks with many iterations it is worth monitoring whether the agent reads stale offloaded files.

## File Transfer: Seeding and Retrieval

There are two distinct planes of file access in a DeepAgents sandbox. The *agent filesystem tools* (`read_file`, `write_file`, `edit_file`) are what the LLM calls during execution; they go through `execute()` inside the sandbox. The *file transfer APIs* (`upload_files`, `download_files`) are what your application code calls to move files between the host and the sandbox before or after the agent runs.

For the code-runner-agent, the seed pattern is useful when you want the agent to work on an existing codebase:

```python
# Seed the workspace with source files before the agent starts
with open("existing_module.py", "rb") as f:
    backend.upload_files([
        ("/workspace/existing_module.py", f.read()),
        ("/workspace/requirements.txt", b"pytest==8.1.0\nnumpy==1.26.0\n"),
    ])

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Add type hints to existing_module.py and fix any issues pytest finds.",
    }]
})

# Retrieve the modified file after the agent finishes
artifacts = backend.download_files(["/workspace/existing_module.py"])
modified_source = artifacts[0].content.decode()
```

The `upload_files` and `download_files` methods use the provider's native file transfer APIs, not shell commands. For the local backend, they are direct filesystem reads and writes. For remote providers (Modal, Runloop, Daytona), they use the provider's SDK file transfer methods, which are more efficient than encoding large files as base64 in shell commands.

An extension worth noting: when the agent finishes a task, store the produced artifacts as Qdrant points in `session_memory` with `type="code"` and `tool_name="upload"`. This makes the output retrievable in future sessions without requiring the developer to know which files were produced.

## Security Considerations for Local Sandboxes

The local shell backend provides *workspace scoping* but not *process isolation*. The agent's subprocess can read files outside the workspace if it knows the path. It can make network requests. It can install packages that affect the host Python environment (unless you use a virtualenv in the workspace). For development this is usually acceptable. For production or untrusted inputs it is not.

The threat model for code-execution agents is **context injection**: an attacker who controls part of the agent's input (a document it is asked to analyze, a codebase it is asked to review) can embed instructions that the agent executes as commands. The sandbox limits the blast radius of a successful injection but does not prevent it.

Three practical mitigations for the local backend:

**Virtualenv isolation.** Create a fresh virtualenv in the workspace before the agent starts. Point `VIRTUAL_ENV` and `PATH` into the workspace. Package installs go into the workspace virtualenv, not the host environment.

```python
import subprocess
subprocess.run(["python", "-m", "venv", str(workspace / ".venv")], check=True)
```

**Human-in-the-loop on shell commands.** DeepAgents supports HITL middleware that intercepts tool calls before execution. For untrusted inputs, require human approval before any `execute` call. This is slow but safe.

**Output filtering.** Use LangChain middleware to filter or redact sensitive patterns in tool outputs before they enter the agent's context. This prevents a context injection that reads a credential file from successfully exfiltrating it via the agent's final output.

A Qdrant-specific consideration: the `session_memory` collection may accumulate sensitive outputs (error messages containing paths, environment variable dumps, partial API responses). These should be stored with payload-level encryption or excluded from the collection entirely if the session processes untrusted inputs. Qdrant's payload filtering can mark sensitive steps with a `redacted` tag and exclude them from retrieval in shared-session scenarios.

For anything stronger (blocking network access, preventing filesystem escapes, resource limits), you need a proper sandbox provider (Modal, Daytona, Runloop) rather than the local shell backend.

## Running the Agent

The repo is structured to run with minimal setup:

```bash
git clone https://github.com/inamdarmihir/deepagents-local-sandbox
cd deepagents-local-sandbox/examples/code-runner-agent
pip install -r requirements.txt
# Start Qdrant locally via Docker
docker run -p 6333:6333 qdrant/qdrant
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...   # for text-embedding-3-small
python agent.py
```

The workspace directory is created fresh on each run and left in place afterward for inspection. The Qdrant collections are created on first run if they do not exist; subsequent runs append to the same `session_memory` collection, so the agent accumulates context across invocations within the same session ID.

On a task like "write a Python function that computes the Levenshtein distance between two strings, with tests," a typical run completes in 8-12 model calls over 30-60 seconds (depending on install latency for pytest). The tool retrieval and session memory lookups add roughly 10-20 ms per step, dominated by the embedding call rather than the Qdrant search itself.

The recursion limit of 50 is generous. Most tasks complete in under 20 steps. If the agent hits the limit, it is usually stuck in a retry loop on a failing test or a missing dependency. The fix is to strengthen the iteration conventions in the system prompt or to add an explicit convention for when to stop retrying and report failure.

## What This Pattern Enables

The local code-runner-agent with Qdrant-backed tool discovery and session memory is a building block for more ambitious systems. The same architecture, with a remote sandbox backend swapped in, becomes deployable in production. The same filesystem tooling works for data analysis, document processing, or infrastructure scripting: any task where the agent needs to write, run, and iterate.

The sandbox-as-tool pattern keeps agent state outside the execution environment, which means you can run multiple agents in parallel against separate sandboxes without interference. Qdrant's session isolation (the `session_id` payload filter) extends this naturally: a single Qdrant cluster serves all concurrent agents, with each agent's retrieval scoped to its own history.

The tool registry pattern scales to large tool sets without degrading model selection accuracy. As the number of tools grows from a handful to hundreds, semantic retrieval preserves precision in a way that passing the full tool list to the model does not. Combined with Qdrant's payload filtering for backend compatibility and permission level, the pattern generalizes to multi-tenant and multi-backend deployments without schema changes.

## Challenges and Open Problems

**Shell state across calls.** The stateless `execute()` model is safe but awkward for multi-step tasks that depend on shell state. Common workarounds: source a shell script at the top of each command, write state to a file and read it back, or chain commands with `&&`. None of these is as clean as a persistent shell session, which some providers (E2B, Daytona) now support. Session memory partially compensates: the agent can retrieve variable bindings from prior steps and re-establish them in the current command.

**Dependency installation latency.** For the local backend, `pip install` on every run is slow (5-30 seconds for a typical scientific stack). The right fix for production is a pre-built sandbox image with dependencies pre-installed. For local development, persisting the workspace across runs and storing a `variable` point in session memory for each installed package lets the agent skip reinstallation on subsequent runs.

**Observability of long-running commands.** `execute()` is synchronous: the agent blocks until the command completes. For a `pytest` run that takes 30 seconds, the agent has no visibility into progress and the user sees nothing. DeepAgents supports streaming tool outputs for providers that implement it, but the local shell backend does not yet support real-time streaming of subprocess output.

**Session memory staleness.** As the `session_memory` collection accumulates steps across many sessions, old steps may become misleading. A step that records "install numpy 1.26.0" may be retrieved for a task involving NumPy even though the current session has already installed a different version. Mitigations include expiring old steps via timestamp filtering in the retrieval query and storing a `superseded_by` field in the payload to allow explicit invalidation.

**Embedding quality for code.** General-purpose text embedding models (`text-embedding-3-small`, `text-embedding-ada-002`) produce reasonable embeddings for code, but code-specific embeddings (CodeBERT, UniXcoder) are measurably better for retrieval tasks over large codebases (Husain et al. 2019). For a tool registry described in natural language, general embeddings are adequate. For session memory that stores raw code, switching to a code-specific embedding model would improve retrieval precision at the cost of a second embedding infrastructure to maintain.

**Context injection surface area.** The threat is real and the mitigations (HITL, output filtering, virtualenv isolation) are partial. The community does not yet have a satisfying solution for code-execution agents that need to process untrusted inputs at scale. Sandbox providers are building network proxies that intercept and filter HTTP requests from inside the sandbox; this feature is not yet widely available.

**Tool registry synchronization.** The `tool_registry` collection is written once at startup. If a tool's description or input schema changes (a new parameter is added to `execute`), the registry must be updated and the affected points re-embedded. For a slowly-changing tool set this is negligible. For a dynamic tool set (where tools are registered and deregistered at runtime), you need an upsert-based registration workflow and a mechanism to invalidate embeddings that depend on outdated tool descriptions.

## References

- langchain-ai/deepagents. *DeepAgents: The batteries-included agent harness*. [github.com](https://github.com/langchain-ai/deepagents)
- LangChain Blog (2026). *Execute Code with Sandboxes for Deep Agents*. [blog.langchain.com](https://blog.langchain.com/execute-code-with-sandboxes-for-deepagents/)
- LangChain Blog (2026). *The Two Patterns by Which Agents Connect Sandboxes*. [blog.langchain.com](https://blog.langchain.com/the-two-patterns-by-which-agents-connect-sandboxes/)
- LangChain Docs. *Sandboxes -- Deep Agents*. [docs.langchain.com](https://docs.langchain.com/oss/python/deepagents/sandboxes)
- Packer et al. (2023). **MemGPT: Towards LLMs as Operating Systems**. [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- Li et al. (2023). **API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs**. [arXiv:2304.08244](https://arxiv.org/abs/2304.08244)
- Husain et al. (2019). **CodeSearchNet Challenge: Evaluating the State of Semantic Code Search**. [arXiv:1909.09436](https://arxiv.org/abs/1909.09436)
