---
title: "The Retrieval Granularity Problem in RAG"
date: 2026-05-17
description: "Why indexing at a fixed chunk size breaks RAG, and how a multi-resolution index in Qdrant fixes it by deferring the granularity decision to query time."
tags: ["rag", "retrieval", "qdrant", "vector-search", "nlp"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Every RAG system makes a silent assumption at indexing time: that the unit of storage is also the unit of retrieval relevance. You split documents into chunks, embed those chunks, and later retrieve whichever chunks are most similar to the query. The chunk boundary becomes the atomic unit of knowledge for the entire system.

This assumption is wrong in ways that are hard to debug.

## The problem with fixed-granularity indexing

Before describing the architecture, it's worth being precise about what breaks and why. There are three structural failure modes.

**1. Specificity mismatch** — a specific factual query needs a precise, narrow retrieval unit. The answer is a single sentence buried in paragraph 7 of a 3,000-word policy document. The chunk containing that sentence also contains four other sentences about unrelated topics. The dense embedding of that chunk is pulled toward the aggregate topic, not toward the specific fact. A more topically coherent chunk from a different section may rank higher, even though it doesn't contain the answer.

**2. Context collapse at fine granularity** — the inverse problem: if you index at sentence-level to improve precision, you lose the context that makes individual sentences interpretable. A sentence like *"The threshold for this category is 1.5%"* has no retrievable referent for "this category" when embedded in isolation. **Dense X Retrieval** ([Chen et al. 2024](https://arxiv.org/abs/2312.06648)) coins the term *context collapse* for this failure mode.

**3. Granularity-query mismatch is invisible at indexing time** — the decision about granularity is made during ingestion, without access to the query distribution. **Standard RAG** ([Lewis et al. 2020](https://arxiv.org/abs/2005.11401)) indexes a corpus into fixed-size windows, and the retrieval objective becomes:

$$\hat{c} = \arg\max_{c \in \mathcal{C}} \cos(\mathbf{q}, \mathbf{e}(c))$$

where $\mathbf{q}$ is the query embedding, $\mathbf{e}(c)$ is the embedding of chunk $c$, and $\mathcal{C}$ is the full corpus. As ([Zhong et al. 2024](https://arxiv.org/abs/2406.00456)) shows empirically, the optimal granularity is query-dependent, and no single level dominates across query types.

## Prior work on granularity

Several lines of work have addressed retrieval granularity, each solving part of the problem.

**Proposition-level indexing.** **Dense X Retrieval** ([Chen et al. 2024](https://arxiv.org/abs/2312.06648)) proposes *propositions* as a new retrieval unit — atomic, self-contained natural language statements of a single fact. Propositions are extracted from source documents using an LLM and indexed as independent units. Proposition-level indexing outperforms passage-level retrieval on open-domain QA benchmarks, but proposition-only systems underperform on synthesis queries where the answer requires multiple related facts in coherent prose.

**Small-to-large retrieval.** The **parent-document retriever** pattern (popularized in LangChain and LlamaIndex) maintains two indices: a *child* index of small chunks for retrieval precision and a *parent* index of larger windows for context. Retrieval identifies child chunks, then expands to return the parent window. This is directionally correct but commits to a fixed expansion ratio — a static lookup, not a query-conditioned decision.

**Mix-of-Granularity.** **Mix-of-Granularity** (MoG) ([Zhong et al. 2024](https://arxiv.org/abs/2406.00456)) pre-indexes documents at multiple granularities and uses a learned MLP router to select the granularity level per query. MoG demonstrates that adaptive selection beats any single fixed level, but the router commits to one granularity per query — an oversimplification for multi-part queries that need different levels for different sub-questions.

Taken together, these works establish that (a) granularity matters, (b) the right granularity is query-dependent, and (c) coarse and fine retrievals are complementary. What they don't do is provide a unified index that supports *simultaneous retrieval at multiple granularities for the same underlying text*. That is the gap this post addresses.

## The multi-resolution index

The core idea is to index each semantic unit of a document at three resolution levels simultaneously, linking them by provenance, and let the retrieval layer — not the ingestion pipeline — decide which level to use per query.

### Three levels, one point

For each source document $d$, the ingestion pipeline produces three granularity tiers:

| Level | Unit | Typical size | Purpose |
|---|---|---|---|
| **Proposition** | Atomic, self-contained factoid | 20–60 tokens | Precision retrieval; specific facts and figures |
| **Passage** | Semantically coherent paragraph | 150–400 tokens | Standard retrieval; balanced precision/context |
| **Section** | Document section or logical block | 400–1200 tokens | Synthesis retrieval; thematic queries |

Every proposition is linked to its parent passage, and every passage to its parent section, via a `parent_id` payload field. The granularity decision at query time determines which embedding to search, but finer retrievals can always expand upward at minimal cost.

### Named vectors in Qdrant

Qdrant supports *named vectors* per point — multiple independent vector spaces attached to the same stored object. This is the enabling primitive. Each text segment is stored once as a Qdrant point, with three named vector fields carrying embeddings of the proposition, passage, and section text, respectively.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="knowledge_base",
    vectors_config={
        "proposition": VectorParams(size=1536, distance=Distance.COSINE),
        "passage":     VectorParams(size=1536, distance=Distance.COSINE),
        "section":     VectorParams(size=1536, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "passage_sparse": SparseVectorParams(),
    }
)
```

The `passage_sparse` vector enables hybrid search (dense + sparse) at the passage level — important for queries involving exact identifiers, codes, and domain-specific terminology that dense embeddings handle poorly.

### Collection schema

Each Qdrant point represents one *proposition* and carries embeddings at all three levels:

```python
from qdrant_client.models import PointStruct, SparseVector

point = PointStruct(
    id=proposition_id,
    vector={
        "proposition": proposition_embedding,
        "passage":     passage_embedding,
        "section":     section_embedding,
    },
    sparse_vector={
        "passage_sparse": SparseVector(
            indices=bm25_indices,
            values=bm25_weights
        )
    },
    payload={
        "proposition_text": "Digital goods merchants with a 90-day ...",
        "passage_text":     "Risk tiering for digital goods follows ...",
        "section_text":     "Section 4: Category-Specific Risk Thresholds ...",
        "doc_id":           "risk-policy-v2.3",
        "section_id":       "sec_4",
        "passage_index":    7,
        "proposition_index": 2,
        "doc_type":         "policy",
        "domain":           "merchant-risk",
        "last_updated":     "2024-11-15",
    }
)
```

The critical design choice is that the *point* represents the proposition, not the passage or section. This keeps the index at maximum precision by default, while storing coarser-grained embeddings as additional named vectors on the same record. A search over `proposition` returns specific facts; a search over `section` returns thematic content — both hit the same collection, and either result expands to any granularity via payload lookup.

## Query-adaptive granularity selection

At query time, a lightweight classifier determines which named vector — or combination — to use for retrieval. The classifier runs inline before the Qdrant query.

### Query type classification

Queries fall into three broad types that map to retrieval granularity:

| Query type | Examples | Preferred level |
|---|---|---|
| **Lookup** | "What is the chargeback threshold for digital goods?" | `proposition` |
| **Synthesis** | "Explain how the company tiers merchant risk." | `section` |
| **Hybrid** | "Summarize the threshold rules and their justification." | `passage` + rerank |

Classification uses a small prompt:

```
Classify this query into one of three retrieval types:
- "lookup": asks for a specific fact, number, definition, or rule
- "synthesis": asks for a summary, explanation, or comparison across multiple ideas
- "hybrid": combines both (e.g., give details AND context)

Query: {query}

Return only: lookup | synthesis | hybrid
```

This runs on a small, fast model (a 7B model handles this reliably). Total overhead: ~50ms. The classification need not be perfect — the system degrades gracefully if a lookup query is misclassified as hybrid, retrieving passage-level results at the cost of some recall precision.

### Routing logic

```python
def route_query(query: str, query_type: str, k: int = 5):
    query_embedding = embed(query)

    if query_type == "lookup":
        return client.search(
            collection_name="knowledge_base",
            query_vector=NamedVector(name="proposition", vector=query_embedding),
            limit=k,
            with_payload=True
        )

    elif query_type == "synthesis":
        return client.search(
            collection_name="knowledge_base",
            query_vector=NamedVector(name="section", vector=query_embedding),
            limit=k,
            with_payload=True
        )

    elif query_type == "hybrid":
        sparse_query = encode_sparse(query)
        return client.query_points(
            collection_name="knowledge_base",
            prefetch=[
                models.Prefetch(
                    query=NamedVector(name="passage", vector=query_embedding),
                    limit=20
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        name="passage_sparse",
                        vector=sparse_query
                    ),
                    limit=20
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
        )
```

### The hybrid reranking pass

For `lookup` queries, the raw proposition search may miss adjacent propositions that are collectively relevant. A post-retrieval expansion step loads the parent passages for the top-3 proposition hits and runs a cross-encoder reranker over (query, passage) pairs. The final context is assembled from the top-ranked passages, with the specific proposition highlighted inline.

For `synthesis` queries, the section-level search returns thematically relevant sections, but sections may be 1,000+ tokens. A summarization pass — or selective loading of only the highest-scoring passages within each section — keeps the context window tight.

## Ingestion pipeline

The ingestion pipeline is the most expensive part of this architecture. It involves two LLM passes per document: one for proposition extraction, one for section boundary detection if the document lacks structural markup.

### Proposition extraction

For each passage $p$ (a paragraph or 250-token window), the extraction prompt generates a list of atomic propositions:

```
Given the passage below, extract all distinct factual propositions.
Each proposition must be:
1. Self-contained: understandable without reading the passage
2. Atomic: exactly one claim, no conjunctions of separate facts
3. Grounded: no information not present in the passage

Format: one proposition per line.

Passage: {passage_text}
```

**PropRAG** ([Wang & Han 2025](https://aclanthology.org/2025.emnlp-main.317)) demonstrates that propositions generated this way can be linked into reasoning chains for multi-hop retrieval, outperforming both passage-level RAG and knowledge graph-based approaches. The key advantage over triples is that propositions preserve natural language context — conditional clauses, hedges, exceptions — that triples discard.

For a 1,000-word document, proposition extraction produces 20–50 propositions and costs ~1,000 input tokens plus ~200 output tokens per LLM call. At current API pricing, roughly $0.001–0.003 per document page.

### Linking propositions to their parents

Each proposition carries three text representations in its payload: its own text, its parent passage, and its parent section. These are embedded separately:

```python
def embed_proposition(prop_text, passage_text, section_text, model):
    prop_emb    = model.embed(prop_text)
    passage_emb = model.embed(passage_text)
    section_emb = model.embed(section_text)
    sparse_emb  = bm25_encode(passage_text)
    return prop_emb, passage_emb, section_emb, sparse_emb
```

The embedding calls are the dominant cost at ingestion: 3× the cost of single-level indexing. For knowledge-intensive, fact-dense domains — legal, medical, financial, technical documentation — the tradeoff is favorable. For general-purpose conversational RAG, the overhead may not be justified.

### Indexing cost analysis

For a corpus of $N$ propositions (each document page produces roughly 30 propositions):

| Component | Cost per proposition |
|---|---|
| LLM extraction | ~$0.0001 |
| Embedding (×3) | ~$0.00003 |
| Qdrant storage | ~2 KB per point |

A 100,000-page corpus produces roughly 3 million propositions, requiring ~6 GB of vector storage at 1536 dimensions with float32, or ~1.5 GB with int8 scalar quantization. Qdrant's quantization reduces memory footprint by up to 4× with minimal recall degradation.

## When this helps and when it doesn't

The multi-resolution approach solves the granularity mismatch problem but introduces costs that are not always justified.

It helps when:
1. **The corpus is fact-dense** — legal contracts, financial filings, technical specifications, medical guidelines where the query distribution is mixed between lookup and synthesis
2. **The same corpus serves heterogeneous queries** — a policy document answering both "what is the exact threshold?" from compliance tooling and "explain the risk framework" from onboarding docs
3. **Recall failures are costly** — in high-stakes domains, missing the specific fact is a worse failure mode than added retrieval latency or storage cost

It doesn't help when:
1. **The corpus is narrative or conversational** — blog posts, emails, support transcripts where passage-level retrieval already captures most relevant content and proposition extraction produces noisy output
2. **The query distribution is uniform** — if all queries are lookups, use proposition-level only; if all are synthesis, use section-level only
3. **The corpus exceeds ~50 million pages** — the 3× embedding overhead becomes a real barrier; dynamic proposition extraction at query time is more practical, though it adds latency

## Challenges and open problems

**Granularity classifier calibration.** The three-way classification (lookup, synthesis, hybrid) is a coarse taxonomy. Many queries are ambiguous or multi-part, requiring different granularities for different sub-questions. A compositional architecture that decomposes queries and routes each sub-question independently would be more powerful, at the cost of complexity. **IRCoT** ([Trivedi et al. 2022](https://arxiv.org/abs/2212.10509)) demonstrates that interleaving retrieval with chain-of-thought reasoning enables multi-hop retrieval without explicit decomposition, but doesn't address granularity selection.

**Proposition extraction quality.** Downstream retrieval quality is bounded by extraction quality. Current prompts produce propositions of varying quality — some introduce subtle paraphrase errors or omit qualifications from the source. No reliable automatic quality metric exists that doesn't itself require a ground-truth proposition corpus. **ChunkRAG** ([Raina & Gales 2024](https://aclanthology.org/2024.fever-1.25)) uses a secondary LLM pass to filter low-quality retrieved units — a similar mechanism is needed at the extraction stage.

**Cross-granularity consistency.** When a proposition and its parent passage both rank highly for the same query, the LLM receives the same information at two levels of detail. This is sometimes useful (proposition provides the precise figure, passage provides interpretive context) and sometimes harmful (the LLM must reconcile two representations of the same fact). A deduplication pass would help, but the collapse criterion is nontrivial.

**Temporal consistency of propositions.** When source documents are updated, propositions from old versions persist in the index. A proposition true in Q4 2024 may be false after a Q1 2025 policy update. Versioned indexing — retaining propositions with validity timestamps and filtering at query time — is straightforward in Qdrant via payload filters but requires discipline in the ingestion pipeline.

**Cold start for the granularity classifier.** For a new domain, no labeled query examples exist. Few-shot prompting with generic examples works acceptably but calibrates less well than domain-specific fine-tuning. A self-supervised approach — observing which granularity level the LLM actually attends to in mixed-granularity context and using that as a training signal — is plausible but untested.

## Further reading

- [Dense X Retrieval: What Retrieval Granularity Should We Use? (Chen et al. 2024)](https://arxiv.org/abs/2312.06648)
- [Mix-of-Granularity (Zhong et al. 2024)](https://arxiv.org/abs/2406.00456)
- [PropRAG: Guiding Retrieval with Beam Search over Proposition Paths (Wang & Han 2025)](https://aclanthology.org/2025.emnlp-main.317)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al. 2020)](https://arxiv.org/abs/2005.11401)
- [Qdrant Named Vectors documentation](https://qdrant.tech/documentation/concepts/vectors/#named-vectors)
- [IRCoT: Interleaving Retrieval with Chain-of-Thought (Trivedi et al. 2022)](https://arxiv.org/abs/2212.10509)
