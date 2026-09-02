---
title: "Static embeddings lose to BM25. Here's the middle ground nobody tested."
date: 2026-08-26
description: "A Qdrant DevRel benchmark showed static, no-transformer embeddings losing to BM25 on quality despite a 30x speed edge. It never tested ternary quantization, a BitNet-style approach that keeps the transformer and just quantizes its weights. I built and measured that middle point on real CodeSearchNet data."
tags: ["qdrant", "embeddings", "quantization", "vector-search"]
author: "Mihir Inamdar"
showToc: false
---

A Qdrant DevRel benchmark I ran into tested what happens when you strip the
transformer out of an embedding model entirely: static, model2vec-style
embeddings, looked up from a table instead of computed by attention layers.
The result was clear. Static embeddings deliver on speed, about 9,100
docs/sec against 31 docs/sec for `bge-small` ONNX at batch 32. They do not
deliver on quality. The best static model tested scored 0.2887 NDCG@10 on
CodeSearchNet Python, worse than plain BM25's 0.2955, and far behind
`bge-small`'s 0.6742
([full results](https://github.com/Dylancouzon/static-embeddings)).

That benchmark tests two ends of a spectrum, keep the full transformer, or
remove it entirely. Nobody had tested the middle: a model that keeps the
transformer, attention and all, but quantizes its weights down to three
values, `{-1, 0, +1}`. That's ternary quantization, the BitNet-style
approach behind [Ternlight](https://github.com/soycaporal/ternlight), and
it's a different tradeoff than deleting the transformer outright. I built
and measured that middle point on a real task: given a docstring, find the
function it documents, on a held-out sample of 500 real CodeSearchNet
Python pairs, indexed and queried through Qdrant.

## What I tested

Three embedders, one task, real Qdrant collections for all three so the
comparison isn't confounded by anything outside the model itself:

| Embedder | Params | Weights | recall@1 | recall@10 | Doc embed | Query latency |
|---|---:|---|---:|---:|---:|---:|
| Domain-tuned ternary | 9.5M | ternary | 0.872 | 0.964 | 357.5 docs/sec | 1.64 ms |
| Generic ternary | n/a | ternary | 0.902 | 0.974 | 19.3 docs/sec | 20.34 ms |
| `bge-small-en-v1.5` | 33M | fp32 | 0.982 | 0.996 | 1.7 docs/sec* | 410.66 ms |

*Measured with FastEmbed's default settings, no thread tuning, one machine.
Slower than commonly published `bge-small` numbers; I'm reporting this run's
real result, not adjusting it toward an expected one. Full method and raw
results: [qdrant-ternlight-techdocs](https://github.com/inamdarmihir/qdrant-ternlight-techdocs).

Setting this up meant indexing the same 500 CodeSearchNet pairs three
separate times, once per embedder, into three separate Qdrant collections,
then running the identical docstring-to-function queries against each.
That's more bookkeeping than it sounds like: ternary weights mean `{-1, 0,
+1}` instead of full floating point, so the embedding vectors themselves
come out a different shape than what `bge-small` produces, and I had to
make sure the collection configs (distance metric, dimensionality) matched
what each model actually emits rather than copy-pasting one config three
times.

Both ternary models land within 8-11 points of the full fp32 model's
recall@1, and within 2-5 points by recall@10. That's a real middle ground.
The static-embeddings benchmark, on the same dataset, shows a
no-transformer model losing to BM25 outright. Keeping the transformer and
just quantizing its weights buys back most of the quality that removing it
entirely gives up, while still cutting per-query latency by 25-200x against
the full-size model.

## The part that didn't go as expected

I fine-tuned a version of Ternlight specifically on CodeSearchNet Python,
expecting the domain specialization to beat the generic, general-purpose
release on this task. It didn't. My domain-tuned checkpoint is faster (18x
higher document throughput, 12x lower query latency than the generic model)
but scores 2-3 points lower on recall, not higher.

My best guess is scale, not architecture: I trained on 25,000 domain
samples for 30 epochs, a realistic single-session budget, against whatever
larger, broader corpus the generic release trained on. Ternary quantization
plus a much smaller training run seems to have traded some of the generic
model's quality for speed, rather than trading generic quality for domain
quality the way I'd hoped. That's just what this specific run showed, and
it's worth stating plainly rather than burying: at this data and epoch
scale, specialization didn't close the gap.

## What this actually means

If a use case can tolerate an 8-11 point recall@1 gap against a full-size
transformer, ternary quantization is a real option that a no-transformer
static embedding isn't. It's the middle ground between "fast but wrong" and
"correct but slow" that the static-embeddings benchmark left open. What's
still unresolved is whether more training data and epochs closes the
distance between a domain-tuned ternary model and its generic counterpart,
or whether ternary's reduced capacity caps what specialization alone can
buy. I'd want to run that before drawing a firmer conclusion.

Where I'd actually reach for this: anything where query volume is high
enough that `bge-small`'s per-query latency is the bottleneck, and where a
10-ish point recall gap is a real cost but not a disqualifying one, code
search over a large internal monorepo, say, or a first-pass retrieval
stage ahead of a slower reranker. I wouldn't reach for it as a drop-in
`bge-small` replacement for anything recall-critical without first running
this same eval against your own data, because 500 CodeSearchNet pairs is a
narrow slice and your corpus almost certainly behaves differently.

Sources: [static-embeddings benchmark](https://github.com/Dylancouzon/static-embeddings) · [Ternlight](https://github.com/soycaporal/ternlight) · [qdrant-ternlight-techdocs](https://github.com/inamdarmihir/qdrant-ternlight-techdocs), full method, raw logs, and reproduction steps.
