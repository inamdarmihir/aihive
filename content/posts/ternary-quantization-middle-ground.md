---
title: "Static embeddings lose to BM25. Here's the middle ground Qdrant didn't test."
date: 2026-08-26
description: "A Qdrant DevRel benchmark showed static, no-transformer embeddings losing to BM25 on quality despite a 30x speed edge. It never tested ternary quantization, a BitNet-style approach that keeps the transformer and just quantizes its weights. We built and measured that middle point on real CodeSearchNet data."
tags: ["qdrant", "embeddings", "quantization", "vector-search"]
author: "Mihir Inamdar"
showToc: false
---

A Qdrant DevRel benchmark tested what happens when you strip the transformer
out of an embedding model entirely: static, model2vec-style embeddings,
looked up from a table instead of computed by attention layers. The result
was clear. Static embeddings deliver on speed, about 9,100 docs/sec against
31 docs/sec for `bge-small` ONNX at batch 32. They do not deliver on
quality. The best static model tested scored 0.2887 NDCG@10 on CodeSearchNet
Python, worse than plain BM25's 0.2955, and far behind `bge-small`'s 0.6742
([full results](https://github.com/Dylancouzon/static-embeddings)).

That benchmark tested two ends of a spectrum: keep the full transformer, or
remove it entirely. It didn't test the middle: a model that keeps the
transformer, attention and all, but quantizes its weights to three values,
`{-1, 0, +1}`. That's ternary quantization, the BitNet-style approach behind
[Ternlight](https://github.com/soycaporal/ternlight), and it's a genuinely
different tradeoff than deleting the transformer. We built and measured
that middle point for a real task: given a docstring, find the function it
documents, on a held-out sample of 500 real CodeSearchNet Python pairs,
indexed and queried through Qdrant.

## What we tested

Three embedders, one task, real Qdrant collections for all three so the
comparison isn't confounded by anything outside the model itself:

| Embedder | Params | Weights | recall@1 | recall@10 | Doc embed | Query latency |
|---|---:|---|---:|---:|---:|---:|
| Domain-tuned ternary | 9.5M | ternary | 0.872 | 0.964 | 357.5 docs/sec | 1.64 ms |
| Generic ternary | n/a | ternary | 0.902 | 0.974 | 19.3 docs/sec | 20.34 ms |
| `bge-small-en-v1.5` | 33M | fp32 | 0.982 | 0.996 | 1.7 docs/sec* | 410.66 ms |

*Measured with FastEmbed's default settings, no thread tuning, one machine.
Slower than commonly published `bge-small` numbers; reported as this run's
real result, not adjusted toward an expected one. Full method and raw
results: [qdrant-ternlight-techdocs](https://github.com/inamdarmihir/qdrant-ternlight-techdocs).

Both ternary models land within 8-11 points of the full fp32 model's
recall@1, and within 2-5 points by recall@10. That's a real middle ground.
The static-embeddings benchmark's numbers, on the same dataset, show a
no-transformer model losing to BM25 outright. Keeping the transformer and
just quantizing its weights clearly buys back most of the quality that
removing it entirely gives up, while still cutting per-query latency by
25-200x against the full-size model.

## The part that didn't go as expected

We fine-tuned a version of Ternlight specifically on CodeSearchNet Python,
expecting the domain specialization to beat the generic, general-purpose
release on this task. It didn't. Our domain-tuned checkpoint is faster
(18x higher document throughput, 12x lower query latency than the generic
model) but scores 2-3 points lower on recall, not higher.

The likely reason is scale, not architecture: we trained on 25,000 domain
samples for 30 epochs, a realistic single-session budget, against whatever
larger, broader corpus the generic release was trained on. Ternary
quantization plus a much smaller training run seems to have traded some of
the generic model's quality for speed, rather than trading generic quality
for domain quality the way we'd hoped. That's the honest result of this
specific run. It's a real limitation to state plainly, not a negative
result to bury: at this data and epoch scale, specialization didn't close
the gap.

## What this actually means

If a use case can tolerate an 8-11 point recall@1 gap against a full-size
transformer, ternary quantization is a real option that a
no-transformer static embedding is not. It's the actual middle ground
between "fast but wrong" and "correct but slow" that the static-embeddings
benchmark left open. The gap that's still open: whether more training
data and epochs closes the distance between a domain-tuned ternary model
and its generic counterpart, or whether ternary's reduced capacity caps
what specialization alone can buy. That's the natural next experiment,
not this one.

Sources: [static-embeddings benchmark](https://github.com/Dylancouzon/static-embeddings) · [Ternlight](https://github.com/soycaporal/ternlight) · [qdrant-ternlight-techdocs](https://github.com/inamdarmihir/qdrant-ternlight-techdocs), full method, raw logs, and reproduction steps.
