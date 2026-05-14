---
title: "Attention Is All You Need — Paper Breakdown"
date: 2026-05-10
description: "A plain-English walkthrough of the 2017 Transformer paper that changed everything in deep learning."
tags: ["transformers", "attention", "nlp", "paper-breakdown"]
author: "Mihir Inamdar"
showToc: true
math: true
---

The 2017 paper *Attention Is All You Need* by Vaswani et al. introduced the **Transformer** architecture — the foundation of every large language model today, from BERT to GPT-4 to Claude.

This post breaks it down from scratch.

## The problem before Transformers

Before 2017, sequence-to-sequence tasks (translation, summarization, etc.) were dominated by **RNNs** and **LSTMs**. These process tokens one at a time, left to right.

Two big problems:
1. **Sequential bottleneck** — you can't parallelize training across tokens
2. **Vanishing gradients** — long-range dependencies (e.g., a verb agreeing with a noun 20 words earlier) get diluted

## The core idea: Self-Attention

Instead of processing tokens sequentially, Transformers process the entire sequence at once using **attention** — every token looks at every other token and decides how much to "attend" to each one.

For a token at position `i`, attention computes a weighted sum over all positions:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- **Q** (Query) — what the current token is looking for
- **K** (Key) — what each token offers
- **V** (Value) — the actual content to aggregate

The dot product `QK^T` measures similarity. Dividing by `√d_k` stabilizes gradients (avoids huge dot products in high dimensions). Softmax turns similarities into a probability distribution.

## Multi-Head Attention

Instead of one attention computation, the paper runs **8 attention heads** in parallel, each learning to focus on different types of relationships (syntax, coreference, position, etc.).

The results are concatenated and projected:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

## Positional Encoding

Attention has no notion of order — it treats the sequence as a set. To inject position information, the authors add a **positional encoding** to each token embedding:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

This sinusoidal encoding lets the model distinguish positions and generalize to sequence lengths not seen during training.

## The full architecture

The Transformer is an **encoder-decoder** model:

- **Encoder** — 6 identical layers of (Multi-Head Self-Attention → Feed-Forward Network), each with residual connections + layer norm
- **Decoder** — same, but with an extra cross-attention layer that attends to encoder outputs, and masked self-attention (so tokens can't look ahead)

## Why it mattered

The Transformer enabled:
- Full parallelization during training → massive speedup
- Better long-range dependency modeling
- Scaling to billions of parameters

Every major model since 2018 — BERT, GPT, T5, LLaMA, Claude — is a Transformer variant.

## Further reading

- [Original paper (arXiv)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer — Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
- [Annotated Transformer — Harvard NLP](http://nlp.seas.harvard.edu/annotated-transformer/)
