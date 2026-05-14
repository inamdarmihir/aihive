---
title: "LLM Sampling Strategies: Temperature, Top-k, and Top-p Explained"
date: 2026-05-05
description: "A practical explanation of how language models decide what token to generate next — and why it matters."
tags: ["llm", "inference", "sampling", "explainer"]
author: "Mihir Inamdar"
showToc: true
---

When a language model generates text, it doesn't just pick the single most likely next word. It *samples* from a distribution — and the sampling strategy dramatically affects output quality, creativity, and coherence.

Here's a breakdown of the three main strategies.

## How output generation works

At each step, the model outputs a **logit** for every token in its vocabulary (~50k tokens for GPT-style models). These logits are converted to probabilities via softmax.

The question is: how do you pick one token from 50,000 probabilities?

## 1. Greedy Decoding

Always pick the highest-probability token.

**Problem**: Leads to repetitive, boring output. The model gets stuck in loops ("The cat sat on the mat. The cat sat on the mat.").

## 2. Temperature Sampling

Temperature `T` scales the logits before softmax:

```
scaled_logit = logit / T
```

- `T < 1` → sharper distribution → more confident, predictable output
- `T > 1` → flatter distribution → more random, creative output
- `T = 1` → unchanged distribution

**Rule of thumb**: Use `T ≈ 0.7` for factual tasks, `T ≈ 1.0–1.2` for creative writing.

## 3. Top-k Sampling

After computing probabilities, keep only the top `k` tokens and redistribute probability mass among them.

If `k = 50`, the model only ever picks from the 50 most likely tokens — preventing wild low-probability choices.

**Problem**: `k` is fixed regardless of the distribution shape. Sometimes the top 50 are all reasonable; sometimes only 3 are.

## 4. Top-p (Nucleus) Sampling

Instead of a fixed count, keep the smallest set of tokens whose cumulative probability exceeds `p`.

If `p = 0.9`, you include tokens until you've covered 90% of the probability mass.

- When the model is confident → nucleus is small (maybe 3–5 tokens)
- When the model is uncertain → nucleus is larger (maybe 50+ tokens)

**This is more adaptive than top-k** and tends to produce better output. Most modern APIs (including Anthropic's) use top-p.

## What to use in practice

| Use Case | Recommended Settings |
|---|---|
| Factual Q&A | T=0.2, top-p=0.9 |
| Code generation | T=0.1–0.3 |
| Creative writing | T=0.9–1.1, top-p=0.95 |
| Chatbot | T=0.7, top-p=0.9 |

Most production systems combine **temperature + top-p**, with top-k disabled or set very high.

## Further reading

- [How do LLMs sample? (Hugging Face blog)](https://huggingface.co/blog/how-to-generate)
- [Nucleus Sampling paper — Holtzman et al. 2019](https://arxiv.org/abs/1904.09751)
