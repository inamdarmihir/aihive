---
title: "Building a Local Code-Runner Agent with DeepAgents"
date: 2026-05-19
description: "A concrete walkthrough of building a sandboxed code-execution agent using the DeepAgents framework, covering the sandbox-as-tool pattern, local shell backends, filesystem tooling, and the design decisions that make autonomous code execution safe enough to ship."
tags: ["agents", "deepagents", "sandbox", "langgraph", "code-execution"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Most tutorials on coding agents stop at the point where the agent writes code. The harder problem is running it -- safely, observably, and without letting the agent touch your host filesystem, environment variables, or network in ways you did not intend. This post walks through a concrete implementation of a local code-runner agent built on the **DeepAgents** framework ([langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)), which wraps LangGraph with a filesystem backend, context management, and first-class sandbox support.

I will focus on the implementation decisions rather than the framework overview. The post assumes familiarity with LangGraph's graph and state primitives and basic Python async patterns. It does not assume prior experience with sandboxing or container isolation.

## Why Code Execution is Different from Tool Use

[#why-code-execution-is-different-from-tool-use](#why-code-execution-is-different-from-tool-use)

Standard tool-use agents call deterministic, bounded functions: search the web, query a database, send a message. Code execution is different in two ways.

First, the space of possible actions is unbounded. An agent calling `execute("rm -rf /")` and an agent calling `execute("pytest tests/")` are structurally identical from the framework's perspective. The only difference is intent, which you cannot verify statically.

Second, code execution is stateful in a way tool calls are not. Installing a package changes the environment for all subsequent commands. Writing a file persists across tool calls. This statefulness is useful -- it is what lets the agent iterate on a failing test rather than starting from scratch -- but it means the execution environment accumulates state that the agent (and the developer) must reason about.

The sandbox abstraction addresses both. It bounds the unbounded by isolating execution from the host. It makes statefulness safe by confining it to a disposable environment that disappears when the task ends.

## The DeepAgents Sandbox Architecture

[#the-deepagents-sandbox-architecture](#the-deepagents-sandbox-architecture)

DeepAgents represents sandboxes as *backends* -- the same abstraction used for filesystems and stores, but with one additional capability: the `execute` tool. When `create_deep_agent()` detects that the configured backend implements `SandboxBackendProtocol`, it automatically adds `execute` to the agent's tool list. When the backend does not implement the protocol, `execute` is filtered out and the agent never sees it. This conditional availability is handled on every model call, not just at startup.

The only method a sandbox provider must implement is `execute(command: str) -> ExecuteResult`. Every other filesystem operation -- `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` -- is built on top of `execute()` by the `BaseSandbox` base class, which constructs shell scripts and runs them in the sandbox via this single method. This design means adding a new provider is implementing one method.

There are two architecture patterns for integrating agents with sandboxes:

**Agent in sandbox.** The agent process runs inside the sandbox. You communicate with it over the network via a WebSocket or HTTP server. API keys must live inside the sandbox, updates require rebuilding images, and you need to manage the communication infrastructure yourself. This pattern mirrors local development closely but has significant security implications -- credentials inside the sandbox are accessible to a context-injected agent.

**Sandbox as tool.** The agent runs on your machine or server. When it needs to execute code, it calls sandbox tools that invoke the provider's API remotely. API keys stay outside the sandbox, agent logic can be updated without rebuilding images, and sandbox failures do not lose agent state. This is the recommended pattern for most use cases and the one the code-runner-agent uses.

The `deepagents-local-sandbox` repo implements the sandbox-as-tool pattern using a local shell backend -- the sandbox is a subprocess on your machine with a scoped working directory, not a remote VM. This is the right starting point for development; the same agent code works with Modal, Runloop, or Daytona by swapping the backend.

## The Local Shell Backend

[#the-local-shell-backend](#the-local-shell-backend)

The local backend creates a scoped workspace directory and executes all commands relative to it. The key design decision is what "scoped" means: the agent can read and write files within the workspace freely, but cannot escape to the parent filesystem without an explicit `..` traversal, which the agent's system prompt discourages.

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

One non-obvious behavior: the local shell backend does *not* maintain shell state across `execute()` calls. Environment variables set in one call (`export FOO=bar`) are not visible in the next call. This is intentional -- it prevents implicit state from accumulating invisibly -- but it means the agent must explicitly pass state between commands (via files, or by chaining commands with `&&`).

## The System Prompt and Tool Visibility

[#the-system-prompt-and-tool-visibility](#the-system-prompt-and-tool-visibility)

The system prompt does more work in a code-execution agent than in most agents. It needs to tell the model what environment it is operating in, what the working directory structure is, and what the conventions are for iteration.

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

The `execute` tool the agent sees has the following signature from the model's perspective:

```
execute(command: str) -> str

Run a shell command in the sandbox working directory.
Returns combined stdout/stderr and exit code.
Large outputs are automatically saved to a file.
```

The large-output truncation is handled by the harness, not the backend. When `execute()` returns more than 20,000 tokens of output, the harness saves the full output to a file in the workspace and substitutes a truncated preview plus a path reference. The agent can then use `read_file` to access the full output incrementally.

## The Agent Loop

[#the-agent-loop](#the-agent-loop)

The core agent loop in DeepAgents is a standard ReAct loop built on LangGraph, with context management layered on top. The agent receives a task, reasons about it, calls tools, observes results, and repeats until it produces a final answer or hits the recursion limit.

```python
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": """Create a Python package called 'mathutils' with:
                - A module with add, subtract, multiply, divide functions
                - Comprehensive pytest tests
                - A README
Then run the tests and confirm they pass."""
            }
        ]
    },
    config={"recursion_limit": 50}
)
```

A typical successful execution on this task looks like:

1. Agent creates `mathutils/__init__.py` and `mathutils/ops.py` via `write_file`
2. Agent creates `tests/test_ops.py` via `write_file`
3. Agent calls `execute("pip install pytest -q")` -- exit code 0
4. Agent calls `execute("python -m pytest tests/ -v")` -- one test fails (divide by zero not handled)
5. Agent reads the pytest output, identifies the failure, edits `mathutils/ops.py` to add a guard
6. Agent calls `execute("python -m pytest tests/ -v")` again -- all tests pass
7. Agent creates `README.md`, returns a summary

Steps 4-6 are the iteration the sandbox makes safe. Without isolation, a failing test that triggered an unintended side effect (writing to a config file, making a network request, modifying an installed package) would be invisible and potentially harmful. With the local sandbox, the blast radius is the workspace directory.

## Context Management and Token Budget

[#context-management-and-token-budget](#context-management-and-token-budget)

Code-execution agents are token-hungry. A multi-step task that installs packages, writes files, and runs tests can accumulate thousands of tokens of tool outputs in the conversation history before the task is done. DeepAgents handles this with automatic context offloading.

When a tool input or result exceeds 20,000 tokens, the harness offloads the content to the backend filesystem and substitutes a file reference with a 10-line preview. The agent sees:

```
[Output truncated -- full content saved to /workspace/.deepagents/tool_outputs/execute_1.txt]
Preview:
...
```

The agent can retrieve the full content with `read_file` when it needs it. This keeps the active context window lean while preserving full observability. The offload threshold is configurable but 20,000 tokens is a reasonable default for most code-execution tasks.

There is a subtle interaction between context offloading and multi-step iteration. If the agent offloads a large pytest output, then iterates on the code, then runs pytest again, it now has two references in its context but the second one supersedes the first. The agent generally handles this correctly by attending to the most recent tool result, but for tasks with many iterations it is worth monitoring whether the agent is reading stale offloaded files.

## File Transfer: Seeding and Retrieval

[#file-transfer-seeding-and-retrieval](#file-transfer-seeding-and-retrieval)

There are two distinct planes of file access in a DeepAgents sandbox. The *agent filesystem tools* (`read_file`, `write_file`, `edit_file`) are what the LLM calls during execution -- they go through `execute()` inside the sandbox. The *file transfer APIs* (`upload_files`, `download_files`) are what your application code calls to move files between the host and the sandbox before or after the agent runs.

For the code-runner-agent, the seed pattern is useful when you want the agent to work on an existing codebase:

```python
# Seed the workspace with source files before the agent starts
with open("existing_module.py", "rb") as f:
    backend.upload_files([
        ("/workspace/existing_module.py", f.read()),
        ("/workspace/requirements.txt", b"pytest==8.1.0\nnumpy==1.26.0\n"),
    ])

result = agent.invoke({
    "messages": [{"role": "user", "content": "Add type hints to existing_module.py and fix any issues pytest finds."}]
})

# Retrieve the modified file after the agent finishes
artifacts = backend.download_files(["/workspace/existing_module.py"])
modified_source = artifacts[0].content.decode()
```

The `upload_files` and `download_files` methods use the provider's native file transfer APIs, not shell commands. For the local backend, they are direct filesystem reads and writes. For remote providers (Modal, Runloop, Daytona), they use the provider's SDK file transfer methods, which are typically more efficient than encoding large files as base64 in shell commands.

## Security Considerations for Local Sandboxes

[#security-considerations-for-local-sandboxes](#security-considerations-for-local-sandboxes)

The local shell backend provides *workspace scoping* but not *process isolation*. The agent's subprocess can read files outside the workspace if it knows the path. It can make network requests. It can install packages that affect the host Python environment (unless you use a virtualenv in the workspace). For development, this is usually acceptable. For production or untrusted inputs, it is not.

The threat model for code-execution agents is **context injection** -- an attacker who controls part of the agent's input (a document it is asked to analyze, a codebase it is asked to review) can embed instructions that the agent executes as commands. The sandbox limits the blast radius of a successful injection, but does not prevent it.

Three practical mitigations for the local backend:

**Virtualenv isolation.** Create a fresh virtualenv in the workspace before the agent starts. Point `VIRTUAL_ENV` and `PATH` into the workspace. Package installs go into the workspace virtualenv, not the host environment.

```python
import subprocess
subprocess.run(["python", "-m", "venv", str(workspace / ".venv")], check=True)
```

**Human-in-the-loop on shell commands.** DeepAgents supports HITL middleware that intercepts tool calls before execution. For untrusted inputs, require human approval before any `execute` call. This is slow but safe.

**Output filtering.** Use LangChain middleware to filter or redact sensitive patterns in tool outputs before they enter the agent's context. This prevents a context injection that reads a credential file from successfully exfiltrating it via the agent's final output.

For anything stronger -- blocking network access, preventing filesystem escapes, resource limits -- you need a proper sandbox provider (Modal, Daytona, Runloop) rather than the local shell backend.

## Running the Agent

[#running-the-agent](#running-the-agent)

The repo is structured to be runnable with minimal setup:

```bash
git clone https://github.com/inamdarmihir/deepagents-local-sandbox
cd deepagents-local-sandbox/examples/code-runner-agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python agent.py
```

The workspace directory is created fresh on each run and left in place afterward so you can inspect what the agent produced. On a task like "write a Python function that computes the Levenshtein distance between two strings, with tests," a typical run completes in 8-12 model calls over 30-60 seconds (depending on install latency for pytest).

The recursion limit of 50 is generous. Most tasks complete in under 20 steps. If the agent is hitting the limit, it is usually stuck in a retry loop on a failing test or a missing dependency -- the fix is to read the tool output more carefully in the system prompt, or to add explicit conventions for when to stop retrying and report failure.

## What This Pattern Enables

[#what-this-pattern-enables](#what-this-pattern-enables)

The local code-runner-agent is a useful building block. The same agent architecture, with a remote sandbox backend swapped in, becomes deployable in production. The same filesystem tooling works for data analysis, document processing, or infrastructure scripting -- any task where the agent needs to write, run, and iterate. The sandbox-as-tool pattern keeps agent state outside the execution environment, which means you can run multiple agents in parallel against separate sandboxes without them interfering with each other.

What the pattern does *not* solve is long-horizon memory. Each invocation of the agent starts fresh. If you want the agent to remember that a particular approach to a problem worked last week, you need to layer an episodic memory store on top -- which is a different architecture problem entirely.

## Challenges and Open Problems

[#challenges-and-open-problems](#challenges-and-open-problems)

**Shell state across calls.** The stateless `execute()` model is safe but awkward for multi-step tasks that depend on shell state. Common workarounds: source a shell script at the top of each command, write state to a file and read it back, or chain commands with `&&`. None of these is as clean as a persistent shell session, which some providers (E2B, Daytona) now support.

**Dependency installation latency.** For the local backend, `pip install` on every run is slow (5-30 seconds for a typical scientific stack). The right fix for production is a pre-built sandbox image with dependencies pre-installed. For local development, persisting the workspace across runs lets the agent skip reinstallation if the environment has not changed.

**Observability of long-running commands.** `execute()` is synchronous -- the agent blocks until the command completes. For a `pytest` run that takes 30 seconds, the agent has no visibility into progress and the user sees nothing. DeepAgents supports streaming tool outputs for providers that implement it, but the local shell backend does not yet support real-time streaming of subprocess output.

**Context injection surface area.** The threat is real and the mitigations (HITL, output filtering, virtualenv isolation) are partial. The community does not yet have a satisfying solution for code-execution agents that need to process untrusted inputs at scale. Sandbox providers are building network proxies that intercept and filter HTTP requests from inside the sandbox, which would help with exfiltration; this feature is not yet widely available.

## References

[#references](#references)

- langchain-ai/deepagents. *DeepAgents: The batteries-included agent harness*. [github.com](https://github.com/langchain-ai/deepagents)
- LangChain Blog (2026). *Execute Code with Sandboxes for Deep Agents*. [blog.langchain.com](https://blog.langchain.com/execute-code-with-sandboxes-for-deepagents/)
- LangChain Blog (2026). *The Two Patterns by Which Agents Connect Sandboxes*. [blog.langchain.com](https://blog.langchain.com/the-two-patterns-by-which-agents-connect-sandboxes/)
- LangChain Docs. *Sandboxes -- Deep Agents*. [docs.langchain.com](https://docs.langchain.com/oss/python/deepagents/sandboxes)
- Packer et al. (2023). *MemGPT: Towards LLMs as Operating Systems*. [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
