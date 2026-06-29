---
title: "Building a Stateful AI Co-worker: Lessons from Specter"
date: 2026-06-25
description: "How Specter wires together LangGraph orchestration, Qdrant vector memory, SSE streaming, and headless browser access to build an AI agent that actually persists context across sessions."
tags: ["agents", "langgraph", "qdrant", "memory", "streaming", "browser-automation"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Most "AI assistant" demos share a silent assumption: the conversation is ephemeral. Ask a question, get an answer, close the tab. Everything resets. This is convenient for demos and catastrophic for anything that resembles real work -- where context accumulates, tasks span multiple sessions, and the assistant is supposed to actually *know* you by now.

Specter is a small but architecturally interesting personal agent that takes the opposite stance. Built by Mihir Inamdar, it wires together **LangGraph** for agent orchestration, **Qdrant** for semantic long-term memory, **FastAPI** with SSE streaming for the backend, and a **Next.js 14** chat UI -- plus an optional headless browser for web interaction. This post walks through how those pieces fit together, where the interesting design decisions live, and what this kind of stack implies for building agents that feel less like toys.

Pre-reads: familiarity with LLM tool-calling is assumed. Some knowledge of graph-based agent frameworks (LangGraph, or conceptually similar) will help.

## The Core Problem: Statefulness in Agents

The dominant pattern for LLM applications is stateless: a user sends a message, the model produces a response, done. What makes it feel stateful is usually just stuffing the conversation history into the context window. This works until it doesn't -- context windows have limits, full conversation replay gets expensive fast, and there's no structural distinction between what the model *knows* and what it merely saw in the last few turns.

Real work sessions look different. You mention your name and your job once, and you expect the assistant to still know both of those things six sessions later. You ask it to browse a site for you, read the output, and remember the relevant parts. You want it to evaluate whether its own response was actually good.

These are distinct problems with distinct solutions:

- **Cross-session memory** → vector store with semantic retrieval
- **Live web access** → headless browser automation
- **Output quality** → structured self-evaluation

Specter assembles these into a single agent without making any of them the entire point. That's what makes it worth studying.

## Architecture Overview

```
┌──────────────────────────────────────────────┐
│  Next.js 14 (port 3000)                      │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ ChatPanel│  │MemoryPanel│  │SuccessCrit │ │
│  └────┬─────┘  └──────────┘  └────────────┘ │
│       │ fetch (SSE)                          │
│  ┌────▼───────────────────────────────────┐  │
│  │  /api/chat   /api/memory  (Next routes)│  │
└──┴────┬────────────────────────────────────┴──┘
        │ HTTP proxy
┌───────▼──────────────────────────────────────┐
│  FastAPI (port 8000)                          │
│                                               │
│  POST /api/chat  ──►  SpectorAgent            │
│                        │                      │
│                   LangGraph StateGraph         │
│                   ┌────┴──────┐               │
│                   │   agent   │◄── GPT-4o-mini │
│                   └────┬──────┘               │
│                   ┌────▼──────┐               │
│                   │   tools   │               │
│                   │ ┌────────┐│               │
│                   │ │memory  ││ remember/recall│
│                   │ ├────────┤│               │
│                   │ │browser ││ open/read/     │
│                   │ │        ││ snapshot/click │
│                   │ └────────┘│               │
│                   └────┬──────┘               │
│                   ┌────▼──────┐               │
│                   │ evaluate  │ structured-out │
│                   └────┬──────┘               │
│                        │                      │
│  GET /api/memory ──►  QdrantStore             │
│                   AsyncQdrantClient           │
│                   (:memory: or remote)        │
│                                               │
│  browser tools ──►  agent-browser daemon      │
│                      Chrome (headless)        │
└──────────────────────────────────────────────┘
```

The data flow is simple to reason about. The Next.js frontend maintains a SSE connection to its own API routes, which proxy to the FastAPI backend. The backend runs a LangGraph `StateGraph` that can call memory tools, browser tools, or both -- then pipes the token stream back up. The Qdrant store runs in-memory by default (`:memory:`), or points at a remote server for persistence.

This separation is clean. The frontend knows nothing about LangGraph. The backend knows nothing about the React component tree. They talk through SSE events with three message types: `token`, `done`, and `error`.

## LangGraph as the Orchestration Backbone

**LangGraph** ([LangChain AI, 2023](https://github.com/langchain-ai/langgraph)) is a graph-based agent runtime built on top of LangChain. The key idea is representing agent execution as a directed graph of nodes and edges, where nodes are computation steps and edges control flow -- including cycles. This matters because real tool-calling agents loop: the model calls a tool, sees the result, decides whether to call another tool or stop, and so on.

In Specter's implementation, the graph has three nodes:

1. **`agent`** -- calls GPT-4o-mini with the current message and tool schemas, decides whether to use tools or generate a final response
2. **`tools`** -- executes whichever tool the agent selected (memory operations or browser actions), feeds the result back
3. **`evaluate`** -- after a final response is ready, scores it against any success criteria the user provided

The execution cycle for a simple turn looks like:

```
agent → tools → agent → tools → ... → agent → evaluate → END
```

Each loop iteration adds new messages to the graph's shared state. This is LangGraph's `MessagesState` -- a list of `BaseMessage` objects that accumulates tool calls and tool results alongside user and assistant turns.

What's nice about this approach is that the graph handles the conditional routing automatically. The `agent` node returns control to `tools` if there's a pending tool call, or routes to `evaluate` (and then `END`) when the model produces a final answer. You define the routing logic once as a conditional edge function, and LangGraph calls it after every node execution.

One thing Specter doesn't use but is worth noting: LangGraph's **checkpointing** mechanism, which lets you persist graph state across execution runs to a durable store. This would let the agent resume interrupted tasks. Specter instead offloads persistence to Qdrant, which is fine for memory but wouldn't help you resume a half-finished web browsing task.

## Memory via Qdrant: Remember and Recall

The memory system is where Specter distinguishes itself from a basic chat wrapper. It exposes two tools to the agent: `remember` and `recall`.

**`remember(content, user_id, session_id)`** -- Embeds `content` using `text-embedding-3-small` and stores it in Qdrant with the user and session IDs as metadata payload.

**`recall(query, user_id, session_id)`** -- Performs a nearest-neighbor search in the Qdrant collection for vectors similar to the embedded `query`. Returns the top-$k$ results, ranked by cosine similarity.

The agent decides autonomously when to call these tools. If you say "my name is Mihir," the agent calls `remember`. If you later ask "what do you know about me?", it calls `recall`.

The Qdrant client is configured as `AsyncQdrantClient(":memory:")` by default, which spins up an in-process vector store with no external dependencies. Swapping to a remote Qdrant instance for true persistence is a one-line config change:

```python
# backend/.env
QDRANT_URL=http://localhost:6333
```

The embedding dimension depends on the model. `text-embedding-3-small` produces 1536-dimensional vectors by default (configurable down to 512). Qdrant's HNSW index handles approximate nearest-neighbor search at this dimensionality easily, with queries completing well under 10ms at typical personal-assistant scales.

One design note worth calling out: the memory is scoped by `user_id` and `session_id`, but recall is filtered only by `user_id`. This means facts from previous sessions are retrievable. The session boundary doesn't reset what the agent knows -- it just organizes when things were stored.

The practical effect is shown in the README's example:

```
Turn 1 -- You: My name is Mihir and I work in AI.
Turn 2 -- You: What do you know about me?

Specter: You're Mihir, a software engineer working in AI.
         [recalled from Qdrant vector memory]
```

This works even across sessions, as long as you're hitting the same Qdrant instance (non-ephemeral mode).

### The retrieval boundary problem

One thing the current implementation doesn't address: how the agent decides *what* is worth remembering. Right now, the agent calls `remember` based on prompt instructions and its own judgment. This is flexible but unpredictable. More structured approaches define explicit "what to remember" categories -- entities, preferences, constraints -- and store them separately from conversational noise. **MemGPT** ([Packer et al. 2023](https://arxiv.org/abs/2310.08560)) explored a related design where memory was divided into core memory (always in context) and archival memory (retrieved on demand). Specter's approach is simpler and probably sufficient for a personal co-worker at this scale, but would need tightening for multi-user or production deployments.

## Server-Sent Events and Streaming

The streaming model is worth spending a moment on, because it's where most "AI chat" implementations make a choice they later regret.

**Polling** is simple but creates visible lag -- you wait for the whole response, then it appears. **WebSockets** are real-time but stateful connections that complicate load balancing and reconnection. **Server-Sent Events (SSE)** sit in between: a long-lived HTTP response where the server pushes newline-delimited data chunks as they're available. It's unidirectional (server to client only), which is fine here because the user doesn't need to send data while the stream is in progress.

Specter's backend uses `sse-starlette` to yield events from FastAPI. Each event has a `type` field and a `content` field:

```json
{"type": "token",  "content": "<text chunk>"}
{"type": "done",   "content": ""}
{"type": "error",  "content": "<message>"}
```

The frontend's `ChatPanel` component reads this stream using the browser's built-in `EventSource` API (or a fetch-based polyfill for more control). Tokens accumulate in React state and render progressively -- users see words arriving as the model generates them, rather than waiting for the full response.

The LangGraph streaming integration uses `astream_events`, which emits events at each node transition. Specter filters for `on_chat_model_stream` events to extract individual tokens from the underlying LLM stream. Other event types (tool calls, tool results) could be surfaced to the UI for a more transparent "thinking" display, but Specter keeps this simple.

One practical issue with SSE: reconnection. If the connection drops mid-stream, `EventSource` reconnects automatically, but the stream is now gone. The backend doesn't have a mechanism to resume a stream mid-generation. For a personal tool running on localhost this rarely matters; in production, you'd want a session store that can replay missed events or a client-side buffer.

## Browser Automation as a Tool

The browser integration is handled by **agent-browser** (Vercel Labs), a CLI tool that wraps headless Chrome and exposes it over a local API. Specter wraps each operation as an async Python tool using `asyncio.to_thread`, since the CLI calls are blocking.

The agent gets access to 10 tools:

| Tool | What it does |
|---|---|
| `browser_open` | Navigate to a URL |
| `browser_read` | Fetch readable text from a URL (no browser launch) |
| `browser_snapshot` | Get accessibility tree with element refs (`@e1`, `@e2`, …) |
| `browser_click` | Click an element by ref or CSS selector |
| `browser_fill` | Clear and fill an input field |
| `browser_get_text` | Get visible text from an element |
| `browser_screenshot` | Take a screenshot, returns file path |
| `browser_scroll` | Scroll in any direction |
| `browser_wait` | Wait N milliseconds |
| `browser_close` | Close the browser session |

The agent-browser daemon keeps Chrome alive between tool calls, so startup overhead (~5 s) only hits on the first `browser_open`. Subsequent operations complete in 1–2 s.

The accessibility tree output from `browser_snapshot` is particularly useful. Rather than dumping raw HTML (which is verbose and LLM-unfriendly), it produces a structured representation of the page's interactive elements with stable refs like `@e1`, `@e2`. The agent can then click `@e1` rather than trying to generate a fragile CSS selector.

The README shows a representative interaction:

```
You: Open https://example.com and summarize what you find on the page.

Specter: The page is titled "Example Domain" and states that this domain
         is intended for use in documentation examples...

Specter autonomously called browser_open → navigated → read content → responded.
Total: ~9 s.
```

That's real-world latency: one browser_open call plus an LLM inference pass. For a personal agent where you're thinking in seconds anyway, this is fine.

### What the browser can't do reliably

The hard cases for browser automation haven't changed much: CAPTCHA challenges, login flows with MFA, sites that fingerprint headless browsers, and highly dynamic single-page apps where the accessibility tree is empty until JavaScript hydrates. `browser_read` sidesteps some of this by using a plain HTTP fetch to extract readable text without launching Chrome at all -- faster and more robust for read-only tasks on standard pages.

## Self-Evaluation: Agents Grading Their Own Outputs

Every response optionally passes through a structured self-evaluation step. The user can supply a `success_criteria` string with their message, and after the agent generates a response, Specter submits both the response and the criteria to GPT-4o-mini as a second call with `structured output` enabled.

The output schema looks roughly like:

```python
class EvaluationResult(BaseModel):
    success: bool
    score: float  # 0.0 to 1.0
    reasoning: str
```

The score and success flag appear alongside the response in the UI. The README example shows a weekly planning response scoring `0.95` against the implied criteria.

This is a lightweight version of what the research community calls *LLM-as-judge* -- using a language model to evaluate another language model's output. **G-Eval** ([Liu et al. 2023](https://arxiv.org/abs/2303.16634)) formalized this approach with chain-of-thought prompting for more reliable scoring. **MT-Bench** ([Zheng et al. 2023](https://arxiv.org/abs/2306.05685)) showed that strong models can score outputs at near-human agreement on many dimensions, though they inherit known biases: preferring longer responses, favoring their own outputs, and struggling with factual accuracy verification.

Specter's self-evaluation is simpler than any of these -- a single score from the same model that generated the response. The limitations are obvious: the model can't verify its own factual claims, and there's an inherent conflict of interest. But for the personal co-worker use case, it's still useful. A planner that scores its own output at 0.3 is telling you something.

A more robust approach would separate the generator and evaluator: use a smaller model for generation and a stronger model for evaluation, or use a different prompt with explicit chain-of-thought scoring criteria. Neither is hard to add here.

## Frontend Design: The Memory Side Panel

The frontend is Next.js 14 with React 18, TypeScript, and Tailwind CSS. The component structure is clean:

```
components/
├── ChatPanel.tsx         # Main chat orchestrator + SSE reader
├── MessageList.tsx
├── MessageBubble.tsx
├── InputBar.tsx
├── MemoryPanel.tsx       # Live view of recalled memories
└── SuccessCriteria.tsx   # Optional input for self-eval criteria
```

`ChatPanel` owns the SSE connection and the message state. It reads the token stream, accumulates text, and updates the message list in real time. The SSE connection is managed with a `useEffect` cleanup that closes the connection when the component unmounts -- important for avoiding stale streams on navigation.

The **MemoryPanel** is the most distinctive UI element. It hits `GET /api/memory/search?q=&user_id=&limit=5` to show what facts the agent has retrieved (or recently stored). This makes memory visible and auditable, which matters: if the agent has the wrong fact about you, you want to see it before it affects ten more turns.

The `SuccessCriteria` component is a simple textarea that, when filled in, gets included in the `/api/chat` request body. It's optional and unintrusive -- if you don't fill it in, evaluation is skipped.

## Challenges and Open Problems

**Memory staleness.** Qdrant stores facts but has no built-in mechanism for expiration or conflict resolution. If you tell the agent you work in AI on Monday and you change careers on Friday, both facts are in the vector store with equal weight. Retrieval will surface whichever is more semantically similar to the query, which may not be the newer one. Systems like **A-MEM** ([Xu et al. 2024](https://arxiv.org/abs/2404.00573)) address this with explicit memory update operations, but Specter currently has no `forget` or `update` primitive.

**No checkpointing.** LangGraph supports graph-level checkpointing, which would let Specter recover an interrupted multi-step task. As it stands, if the browser task takes 30 seconds and the connection drops at second 25, there's no resume path -- the user has to re-issue the request.

**Single-agent architecture.** The LangGraph graph has one agent node. For complex multi-step tasks -- "research this topic, write a summary, save it to a doc, email it to me" -- a single agent calling all tools sequentially works but lacks parallelism and specialization. A multi-agent architecture (separate researcher, writer, and executor subgraphs) would be better suited to this kind of workflow, though it adds significant coordination complexity.

**Self-evaluation reliability.** The current evaluator uses the same model that generated the response. This is fast and cheap, but the model can't catch its own factual errors and may score its outputs higher than warranted. Even a simple cross-check against retrieved memory (does the response contradict known facts?) would improve reliability.

**Browser tool surface area.** Ten browser tools cover most common interactions, but there are gaps: file downloads, JavaScript execution, cookie management, and anything requiring authenticated sessions. The Vercel agent-browser daemon abstracts Chrome well but exposes only a subset of the full DevTools Protocol.

Despite these limitations, Specter is a well-constructed example of what a personal agent with memory, browsing, and self-evaluation actually looks like when you wire it together -- not in a prototype that glosses over the plumbing, but in something that runs, has a real API, and shows the seams. That's most of what matters at this stage of the field.

## Further reading

- [Specter on GitHub](https://github.com/inamdarmihir/specter)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [MemGPT: Towards LLMs as Operating Systems (Packer et al. 2023)](https://arxiv.org/abs/2310.08560)
- [G-Eval: NLG Evaluation using GPT-4 (Liu et al. 2023)](https://arxiv.org/abs/2303.16634)
- [A-MEM: Agentic Memory for LLM Agents (Xu et al. 2024)](https://arxiv.org/abs/2404.00573)
- [Server-Sent Events -- MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
