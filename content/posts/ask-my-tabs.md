---
title: "Ask My Tabs: A Local Agentic RAG Layer Over Your Open Browser Tabs"
date: 2026-08-14
description: "A Chrome extension that answers questions across a working set of open tabs, with retrieval and generation both running in-browser via WebGPU. The interesting part isn't the agent loop, it's a retrieval bug that only showed up once the embedding model got quantized -- and never threw an error."
tags: ["chrome-extension", "rag", "agents", "webgpu", "quantization", "browser-ai"]
author: "Mihir Inamdar"
showToc: true
math: false
---

The specific moment this is built for: six tabs open, comparing something, and the actual
answer requires clicking between all of them to check one fact against another. A search
engine can't help here, the information isn't public, it's scattered across pages a specific
person happened to open. [Ask My Tabs](https://github.com/inamdarmihir/ask-my-tabs) is a
Chrome extension that answers a question across a working set of open tabs and cites which
tab each part of the answer came from.

The constraint that shaped every design decision: nothing leaves the machine. Embeddings and
generation both run in-browser, via WebGPU. No server, no API key, no network call carrying
tab content anywhere. That constraint is also why the interesting finding in this post exists
at all — it forced picking a small model and a quantization setting, and one of those settings
turned out to be silently wrong in a way that a functional test would never have caught.

## Architecture Overview

```
Browser tabs
    │  chrome.scripting: extract readable text
    ▼
┌─────────────────────────────────────────────────────────────┐
│  background.js (MV3 service worker)                         │
│  ────────────────────────────────────────────────────────  │
│  thin router: reads tab content, keeps the offscreen        │
│  document alive, otherwise stays out of the way             │
└─────────────────────────────────────────────────────────────┘
    │  page text
    ▼
┌─────────────────────────────────────────────────────────────┐
│  offscreen.js  (chrome.offscreen document)                  │
│  ────────────────────────────────────────────────────────  │
│  chunk.js       → 180-word overlapping chunks, SHA-256      │
│                    content hash for reuse detection          │
│  embeddings.js  → bge-small-en-v1.5 via transformers.js,    │
│                    WebGPU, fp16                               │
│  vectorstore.js → flat cosine search over IndexedDB          │
│  llm.js         → Qwen2.5-1.5B-Instruct via WebLLM            │
│  agent.js       → query planning, sufficiency check,          │
│                    one bounded refinement hop, synthesis      │
└─────────────────────────────────────────────────────────────┘
    │  answer + numbered citations back to specific tabs
    ▼
popup.js  (renders streaming tokens, click-through citations)
```

Everything that actually does work — both models, the vector store, the agent loop — lives in
the offscreen document, not the popup and not the service worker. That placement isn't
arbitrary, and it's worth explaining why, because the obvious choices don't work.

## Why `chrome.offscreen`, Not the Popup or the Service Worker

A popup is destroyed the instant the user clicks away from it. If model state lived there,
every toolbar click would mean reloading close to a gigabyte of weights from scratch. That
rules out the popup as anything but a UI shell.

The service worker looks like the obvious fallback — it's the thing in Manifest V3 designed
to persist across popup opens and closes. It's also ephemeral by design (Chrome can and does
kill it after roughly 30 seconds of inactivity) and has historically inconsistent WebGPU
access, since it isn't a DOM context. Neither property is compatible with holding a loaded
embedding model and a loaded LLM in memory.

`chrome.offscreen` is Chrome's actual answer to "I need a persistent context with full DOM
and GPU access, independent of the popup's and the service worker's lifecycles." An offscreen
document is invisible, has a real DOM, and survives exactly as long as the extension chooses
to keep it around. `background.js` creates it once and otherwise limits itself to the two
things an offscreen document can't do itself: reading tab content via `chrome.scripting`, and
creating the offscreen document in the first place. Everything else — both models, the vector
store, the whole agent loop — runs there.

## The Embedding Pipeline, and a Bug That Only Showed Up Under Test

Each tab's readable text gets extracted, chunked into 180-word overlapping windows (word-based
rather than character-based, so chunk size stays meaningful across languages that don't
tokenize like English), and embedded with `bge-small-en-v1.5` — a 33M-parameter,
384-dimensional retrieval-tuned encoder, run through
[transformers.js](https://huggingface.co/docs/transformers.js) (`@huggingface/transformers`,
the successor to `@xenova/transformers`):

```js
import { pipeline, env } from "@huggingface/transformers";

env.allowLocalModels = false;
const MODEL_ID = "Xenova/bge-small-en-v1.5";

const extractor = await pipeline("feature-extraction", MODEL_ID, {
  device: "webgpu",
  dtype: "fp16",
  progress_callback: onProgress,
});

const output = await extractor(texts, { pooling: "mean", normalize: true });
```

`fp16` is the second value that ever occupied that `dtype` field. The first was `q8`, 8-bit
quantization, chosen for the obvious reason: a smaller download and a faster load on hardware
without a GPU. It worked, in the sense that it loaded without error and produced 384-dimensional
vectors of the right shape. Whether those vectors meant anything was a separate question, and
the only way to answer it was to actually check — not with a functional test, since a
functional test only confirms the shape came back correct, but by embedding two real sentence
pairs, one genuinely similar and one genuinely unrelated, and checking that similarity ranked
them correctly.

It didn't. At `q8`, the similar pair scored 0.62 cosine similarity; the unrelated pair scored
0.69. The unrelated pair, ranked as *more* similar. At `fp16`, the same two pairs scored 0.59
and 0.30 — correctly separated, and still meaningfully smaller than the full `fp32` weights.

Nothing about the `q8` failure looked like a failure from the outside. No exception, no
console warning, no dropped dimension. The model loaded three times faster than `fp32` and
handed back vectors that were the right size and the right dtype. The only way this surfaces
is by checking whether the ranking the vectors produce is actually correct, and that's a step
a lot of "does the model load" testing skips, because a loaded model that returns numbers
*looks* like success.

[An earlier post here](https://inamdarmihir.github.io/aihive/posts/why-your-rag-keeps-repeating-itself/)
on retrieval redundancy made this point in the abstract — "the same trap as judging a
compressed embedding model by whether it loads fast instead of whether its rankings are still
correct." This is that scenario, not hypothetically: a real quantization flag, in a real
extension, that would have shipped silently broken to anyone who installed it, if the only
test run against it had been "does it load and return a vector."

## The Agent Loop

A plain RAG pipeline embeds the question once, retrieves once, stuffs the top-k chunks into a
prompt. That's a reasonable default and it falls over on exactly the query this extension
exists for: "compare X and Y," where a single embedding of the whole question sits somewhere
between X's cluster and Y's cluster in vector space and retrieves neither well. `agent.js`
runs a short, explicitly bounded loop instead of a single retrieval pass:

```js
const MAX_HOPS = 2;
const TOP_K_PER_QUERY = 5;
const SUFFICIENCY_SCORE_FLOOR = 0.45;

export async function answerQuestion(question, { tabIds, tabTitles }, onStatus, onToken) {
  const queries = await planQueries(question, tabTitles);   // 1-3 sub-queries, LLM-planned

  let allHits = [];
  for (const q of queries) {
    const [qEmbedding] = await embed([q]);
    allHits.push(...await search(qEmbedding, { tabIds, topK: TOP_K_PER_QUERY }));
  }
  allHits = dedupeHits(allHits);

  const { sufficient, refinedQuery } = await checkSufficiency(question, allHits);
  if (!sufficient) {
    const [qEmbedding] = await embed([refinedQuery]);
    allHits = dedupeHits([...allHits, ...await search(qEmbedding, { tabIds, topK: TOP_K_PER_QUERY })]);
  }

  // ...top 8 hits get handed to the LLM as numbered context, cited inline as [1], [2], ...
}
```

`planQueries` asks the LLM to break the question into one to three short searches — a
comparison question becomes one query per thing being compared, instead of one blurry query
that muddles both. `checkSufficiency` runs after retrieval and decides, first on a cheap
numeric floor (best score under 0.45 almost never means "the answer's actually in here"), then
on the model's own read of whether the retrieved snippets cover the question, whether it's
worth spending one more search before answering. `MAX_HOPS` caps that at exactly one refinement.
This is a deliberate ceiling, not a missing feature: enough structure to help on a genuine
comparison question, not enough rope for a 1.5B model to talk itself into an open-ended search
loop it can't reliably terminate.

Both `planQueries` and `checkSufficiency` call the LLM through `chatJSON`, which asks WebLLM
for `response_format: { type: "json_object" }`. Small local models are inconsistent about
actually returning valid JSON under load, so `safeParseJSON` falls back to treating the raw
question as a single query, or the retrieved snippets as sufficient, rather than letting a
malformed response take down the whole request.

## Why No WASM Vector Database

The retrieval store is brute-force cosine similarity over a plain array, not an ANN index:

```js
function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // embeddings are pre-normalized, so dot product == cosine similarity
}

export async function search(queryEmbedding, { tabIds = null, topK = 5 } = {}) {
  const all = await getAllChunks();
  const pool = tabIds ? all.filter((c) => tabIds.includes(c.tabId)) : all;
  return pool
    .map((c) => ({ ...c, score: cosine(queryEmbedding, c.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}
```

The original plan called for something like Voy or Orama, a WASM-compiled ANN index, on the
theory that a vector store should have one. A research session realistically has a handful of
open tabs and a few hundred chunks. Brute-force cosine over a plain JavaScript array is
microseconds at that size — there is nothing for an approximate index to buy back, and it's
one fewer WASM binary to package, version, and debug. IndexedDB exists here purely so the
working set survives the offscreen document being torn down between sessions; it has nothing
to do with query speed. This is the boring answer, and at this scale it's also the correct one:
added complexity that doesn't pay for itself is still a cost, even when the thing it would have
sped up wasn't slow.

Re-adding a tab whose content hasn't changed is a no-op, checked via a SHA-256 hash of the
extracted text computed with `crypto.subtle.digest` — the same "don't recompute what hasn't
changed" instinct that made sense for a separate project's table-extraction cache, applied here
to a much cheaper operation.

## What's Actually Verified

The embedding half of this was tested live in a real browser tab, not just written and assumed
correct: `bge-small-en-v1.5` loading via transformers.js, running on WebGPU without falling
back to WASM, and — the part that mattered — correctly separating a similar sentence pair from
an unrelated one once `fp16` replaced `q8`.

What isn't verified end to end is the WebLLM half, running inside an actual `chrome.offscreen`
document in a loaded extension. transformers.js was testable directly in a plain browser tab
with no extension APIs involved at all; `chrome.offscreen`, `chrome.scripting`, and a
several-hundred-megabyte WebLLM model download aren't something a sandboxed test page can stand
in for. Loading this unpacked in real Chrome is the first genuine test of that path — which is
exactly why `chatJSON`'s malformed-response fallback exists rather than being an edge case
handled after the fact: small local models under real load are inconsistent about following a
"JSON only" instruction, and the agent loop needed to survive that from the first real run, not
after the first crash report.

## Challenges and Open Problems

**Page-text extraction is blunt.** It strips scripts and obvious chrome (nav, header, footer)
but isn't real Readability-style content detection. Sites with heavy client-side rendering,
paywalls, or content gated behind interaction extract poorly or come back empty. A proper
readability pass, or falling back to the accessibility tree for pages that resist plain DOM
extraction, is the natural next step.

**The working set doesn't survive a tab closing.** Chunks are tracked by `tabId`, and Chrome
reuses tab IDs after a tab closes. Closing a tab that was part of the working set leaves a
stale entry pointing at nothing; the only cleanup right now is removing it by hand. A
`chrome.tabs.onRemoved` listener that calls `deleteTab` automatically is a small, obvious fix
that just hasn't been built yet.

**No re-indexing on navigation.** Add a tab, then navigate it somewhere else, and the original
content stays indexed under that `tabId` until it's manually removed and re-added. The content
hash check in `chunk.js` catches unchanged content efficiently; it does nothing for content
that changed out from under an already-indexed `tabId`. A `chrome.tabs.onUpdated` listener
keyed on URL change would close this the same way the closed-tab fix would.

**Retrieval and generation are both approximate by construction.** `SUFFICIENCY_SCORE_FLOOR`
at 0.45 and `MAX_HOPS` at 2 are reasonable starting points, not tuned constants — there's no
labeled query set here to sweep them against, only the two sentence pairs used to validate the
embedding model itself. A real eval set of tab-comparison questions with known-correct answers
would turn these from defensible guesses into numbers with evidence behind them.

## Further Reading

- [Ask My Tabs on GitHub](https://github.com/inamdarmihir/ask-my-tabs)
- [transformers.js documentation](https://huggingface.co/docs/transformers.js)
- [bge-small-en-v1.5 (BAAI)](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [WebLLM](https://github.com/mlc-ai/web-llm)
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [chrome.offscreen API — Chrome for Developers](https://developer.chrome.com/docs/extensions/reference/api/offscreen)
