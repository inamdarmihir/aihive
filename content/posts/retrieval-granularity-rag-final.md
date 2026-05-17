# The Retrieval Granularity Problem in RAG

Every RAG system makes a silent assumption at indexing time: that the unit of storage is also the unit of retrieval relevance. You split documents into chunks, embed those chunks, and later retrieve whichever chunks are most similar to the query. The chunk boundary becomes the atomic unit of knowledge for the entire system.

This assumption is wrong in ways that are hard to debug.

This post describes an architecture that defers the granularity decision to query time, using a *multi-resolution index* in Qdrant where each document position is indexed simultaneously at three granularities, and a lightweight classifier routes each query to the right level. I'll describe each component in enough detail to implement it.

I'm focusing on knowledge-intensive, fact-dense corpora — legal contracts, financial filings, technical specifications, medical guidelines — where the query distribution is mixed between precise factual lookups and broader synthesis questions. The approach is less interesting for conversational or narrative corpora where passage-level retrieval already captures most relevant content.

---

**Table of Contents**

- [The Problem with Fixed-Granularity Indexing](#the-problem-with-fixed-granularity-indexing)
- [Prior Work on Granularity](#prior-work-on-granularity)
  - [Proposition-Level Indexing](#proposition-level-indexing)
  - [Small-to-Large Retrieval](#small-to-large-retrieval)
  - [Mix-of-Granularity](#mix-of-granularity)
  - [What These Approaches Leave Unsolved](#what-these-approaches-leave-unsolved)
- [The Multi-Resolution Index](#the-multi-resolution-index)
  - [Three Levels, One Point](#three-levels-one-point)
  - [Named Vectors in Qdrant](#named-vectors-in-qdrant)
  - [Collection Schema](#collection-schema)
- [Query-Adaptive Granularity Selection](#query-adaptive-granularity-selection)
  - [Query Type Classification](#query-type-classification)
  - [Routing Logic](#routing-logic)
  - [The Hybrid Reranking Pass](#the-hybrid-reranking-pass)
- [Ingestion Pipeline](#ingestion-pipeline)
  - [Proposition Extraction](#proposition-extraction)
  - [Linking Propositions to Their Parents](#linking-propositions-to-their-parents)
  - [Indexing Cost Analysis](#indexing-cost-analysis)
- [When This Helps and When It Doesn't](#when-this-helps-and-when-it-doesnt)
- [Challenges and Open Problems](#challenges-and-open-problems)
- [Citation](#citation)

---

## The Problem with Fixed-Granularity Indexing

Before describing the architecture, it's worth being precise about what breaks and why. There are three structural failure modes, and they have different causes.

**Specificity mismatch.** A specific factual query — "what is the chargeback threshold applied to digital goods merchants in Q4 2024?" — needs a precise, narrow retrieval unit. The answer is a single sentence buried in paragraph 7 of a 3,000-word policy document. The chunk containing that sentence also contains four other sentences about unrelated merchant categories. The dense embedding of that chunk is pulled toward the aggregate topic, not toward the specific fact. A more topically coherent chunk about payment risk from a different section may rank higher, even though it doesn't contain the answer. The precision of dense retrieval degrades as the gap between what the query asks and what the chunk contains increases ([Chen et al. 2024](https://arxiv.org/abs/2312.06648)).

**Context collapse at fine granularity.** The inverse problem: if you index at sentence-level to improve precision, you lose the context that makes individual sentences interpretable. A sentence like *"The threshold for this category is 1.5%"* has no retrievable referent for "this category" when the sentence is embedded in isolation. **Dense X Retrieval** ([Chen et al. 2024](https://arxiv.org/abs/2312.06648)) coins the term *context collapse* for this failure mode and proposes propositions — atomic, self-contained factoids — as a mitigation. Propositions work for factoid retrieval but impose heavy indexing overhead and still underperform on synthesis queries.

**Granularity-query mismatch is invisible at indexing time.** The fundamental issue is architectural: the decision about granularity is made during ingestion, without access to the query distribution. **Standard RAG** ([Lewis et al. 2020](https://arxiv.org/abs/2005.11401)) indexes a corpus by chunking documents into fixed-size windows — typically 256 to 512 tokens with some overlap — and the retrieval objective becomes:

$$\hat{c} = \arg\max_{c \in \mathcal{C}} \cos(\mathbf{q}, \mathbf{e}(c))$$

where $\mathbf{q}$ is the query embedding, $\mathbf{e}(c)$ is the embedding of chunk $c$, and $\mathcal{C}$ is the full corpus. The retrieval unit is fixed at indexing time. As ([Zhong et al. 2024](https://arxiv.org/abs/2406.00456)) shows empirically, the optimal granularity is query-dependent, and no single level dominates across query types.

The architecture below addresses all three.

---

## Prior Work on Granularity

Several lines of work have addressed retrieval granularity, each solving part of the problem.

### Proposition-Level Indexing

**Dense X Retrieval** ([Chen et al. 2024](https://arxiv.org/abs/2312.06648)) proposes *propositions* as a new retrieval unit. A proposition is an atomic, self-contained natural language statement of a single fact, e.g., *"Digital goods merchants with a 90-day chargeback rate above 1.5% are classified as HIGH risk."* Propositions are extracted from source documents using an LLM and indexed as independent units.

The results are compelling for factoid retrieval: proposition-level indexing outperforms passage-level retrieval on open-domain QA benchmarks. The key insight is that proposition embeddings carry higher information density per token — the embedding captures a single, unambiguous claim rather than the centroid of a paragraph.

The limitation is twofold. First, proposition extraction is expensive at ingestion time (one LLM call per paragraph, at minimum). Second, proposition-only systems perform worse than passage systems on synthesis queries, where the answer requires multiple related facts presented in coherent prose rather than a list of atomic claims.

### Small-to-Large Retrieval

The **parent-document retriever** pattern (popularized in LangChain and LlamaIndex) addresses context collapse by maintaining two indices: a *child* index of small chunks for retrieval precision, and a *parent* index of larger windows for context. Retrieval identifies child chunks, then expands to return the parent window.

This is directionally correct but commits to a fixed expansion ratio. The child-to-parent mapping is a static lookup, not a query-conditioned decision. For a query that genuinely only needs the child chunk, the parent expansion wastes context budget and may introduce noise.

### Mix-of-Granularity

**Mix-of-Granularity** (MoG) ([Zhong et al. 2024](https://arxiv.org/abs/2406.00456)) takes a routing approach: documents are pre-indexed at multiple granularities (token, sentence, passage, document), and a learned router selects the granularity level per query using a multilayer perceptron (MLP) trained on query-granularity pairs. At inference time, the router assigns a granularity level, and retrieval proceeds at that level.

MoG demonstrates that adaptive granularity selection is measurably better than any single fixed level. However, the router commits to one granularity per query. This is still an oversimplification: a query about a complex policy may need precise retrieval for one sub-question and passage-level context for another. And the granularity-per-document segmentation is independent of document structure; a sentence boundary in one document may straddle a conceptual unit in another.

### What These Approaches Leave Unsolved

Taken together, these works establish that (a) granularity matters, (b) the right granularity is query-dependent, and (c) coarse and fine retrievals are complementary. What they don't do is provide a unified index that supports *simultaneous retrieval at multiple granularities for the same underlying text*, with query-time assembly of the final context window from whichever granularity is most useful per query.

That is the gap this post addresses.

---

## The Multi-Resolution Index

The core idea is to index each semantic unit of a document at three resolution levels simultaneously, linking them by provenance, and let the retrieval layer — not the ingestion pipeline — decide which level to use per query.

### Three Levels, One Point

For each source document $d$, the ingestion pipeline produces three granularity tiers:

| Level | Unit | Typical size | Purpose |
|---|---|---|---|
| **Proposition** | Atomic, self-contained factoid | 20–60 tokens | Precision retrieval; specific facts and figures |
| **Passage** | Semantically coherent paragraph | 150–400 tokens | Standard retrieval; balanced precision/context |
| **Section** | Document section or logical block | 400–1200 tokens | Synthesis retrieval; thematic queries |

Every proposition is linked to its parent passage, and every passage is linked to its parent section, via a `parent_id` payload field. When a proposition is retrieved, the full passage — and the full section — are one lookup away. The granularity decision at query time determines which embedding to search, but finer retrievals can always expand upward at minimal cost.

### Named Vectors in Qdrant

Qdrant supports *named vectors* per point — multiple independent vector spaces attached to the same stored object. This is the enabling primitive. Each text segment is stored once as a Qdrant point, with three named vector fields carrying embeddings of the proposition, passage, and section text, respectively.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams,
    NamedVector, NamedSparseVector
)

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

### Collection Schema

Each Qdrant point represents one *proposition* and carries embeddings at all three levels:

```python
from qdrant_client.models import PointStruct, SparseVector

point = PointStruct(
    id=proposition_id,
    vector={
        "proposition": proposition_embedding,   # embed(proposition_text)
        "passage":     passage_embedding,       # embed(parent_passage_text)
        "section":     section_embedding,       # embed(parent_section_text)
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

The critical design choice is that the *point* represents the proposition, not the passage or section. This keeps the index at maximum precision by default, while storing the coarser-grained embeddings as additional named vectors on the same record. A search over the `proposition` vector returns specific facts; a search over the `section` vector returns thematic content — but both operations hit the same underlying collection, and either result can be expanded to any granularity via payload lookup.

---

## Query-Adaptive Granularity Selection

At query time, a lightweight classifier determines which named vector — or combination of named vectors — to use for retrieval. The classifier runs inline before the Qdrant query.

### Query Type Classification

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

This runs on a small, fast model (a 7B model handles this reliably). Total overhead: ~50ms. The classification need not be perfect — the system degrades gracefully if a lookup query is misclassified as hybrid (it retrieves passage-level results, which contain the fact plus context, at the cost of some recall precision).

### Routing Logic

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

### The Hybrid Reranking Pass

For `lookup` queries, the raw proposition search returns precise results but may miss adjacent propositions that are collectively relevant. A post-retrieval expansion step loads the parent passages for the top-3 proposition hits and runs a cross-encoder reranker over (query, passage) pairs. The final context presented to the LLM is assembled from the top-ranked passages, with the specific proposition highlighted inline.

For `synthesis` queries, the section-level search returns thematically relevant sections, but sections may be 1,000+ tokens. A summarization pass — or selective loading of only the highest-scoring passages within each section — keeps the context window tight.

---

## Ingestion Pipeline

The ingestion pipeline is the most expensive part of this architecture. It involves two LLM passes per document: one for proposition extraction, one for section boundary detection if the document doesn't have structural markup.

### Proposition Extraction

For each passage $p$ (defined as a paragraph or 250-token window), the extraction prompt generates a list of atomic propositions:

```
Given the passage below, extract all distinct factual propositions.
Each proposition must be:
1. Self-contained: understandable without reading the passage
2. Atomic: exactly one claim, no conjunctions of separate facts
3. Grounded: no information not present in the passage

Format: one proposition per line.

Passage: {passage_text}
```

**PropRAG** ([Wang & Han 2025](https://aclanthology.org/2025.emnlp-main.317)) demonstrates that propositions generated this way can be linked into reasoning chains for multi-hop retrieval, outperforming both passage-level RAG and knowledge graph-based approaches on multi-hop QA benchmarks. The key advantage over triples is that propositions preserve natural language context — conditional clauses, hedges, exceptions — that triples discard.

For a 1,000-word document, proposition extraction typically produces 20–50 propositions and costs ~1,000 input tokens plus ~200 output tokens per LLM call. At current API pricing, this is roughly $0.001–0.003 per document page — manageable for most corpora, though non-trivial at millions of documents.

### Linking Propositions to Their Parents

Each proposition carries three text representations in its payload: its own text, the text of its parent passage, and the text of its parent section. These are embedded separately:

```python
def embed_proposition(prop_text, passage_text, section_text, model):
    prop_emb    = model.embed(prop_text)
    passage_emb = model.embed(passage_text)
    section_emb = model.embed(section_text)
    sparse_emb  = bm25_encode(passage_text)
    return prop_emb, passage_emb, section_emb, sparse_emb
```

The embedding calls are the dominant cost at ingestion: 3× the embedding cost of single-level indexing. For a corpus where the retrieval quality gain is material — knowledge-intensive, fact-dense domains like legal, medical, financial, or technical documentation — the tradeoff is favorable. For general-purpose conversational RAG, the overhead may not be justified.

### Indexing Cost Analysis

For a corpus of $N$ propositions (each document page produces roughly 30 propositions):

| Component | Cost per proposition |
|---|---|
| LLM extraction | ~$0.0001 |
| Embedding (×3) | ~$0.00003 |
| Qdrant storage | ~2 KB per point |

A 100,000-page corpus produces roughly 3 million propositions, requiring ~6 GB of vector storage at 1536 dimensions with float32, or ~1.5 GB with int8 scalar quantization. Qdrant's quantization reduces memory footprint by up to 4× with minimal recall degradation, making this feasible on a single machine for most enterprise corpora.

---

## When This Helps and When It Doesn't

The multi-resolution approach solves the granularity mismatch problem but introduces costs that are not always justified.

**It helps when the corpus is fact-dense.** Legal contracts, financial filings, technical specifications, medical guidelines — these documents contain many specific facts that passage embeddings dilute, and the query distribution is mixed between lookup and synthesis. The same corpus answering heterogeneous queries is the canonical case: a risk policy document serving both "what is the exact threshold?" queries from compliance tooling and "explain the risk framework" queries from onboarding documentation.

**It helps when recall failures are costly.** In high-stakes domains, missing the specific fact that answers the query is a worse failure mode than the retrieval latency or storage cost this architecture adds.

**It doesn't help when the corpus is narrative or conversational.** Blog posts, emails, support ticket transcripts — the information density is low enough that passage-level retrieval captures most relevant content, and proposition extraction produces noisy output.

**It doesn't help when the query distribution is uniform.** If all queries are lookups, use proposition-level indexing. If all are synthesis, use section-level. The multi-resolution index adds cost without benefit when the distribution doesn't require adaptive granularity.

**It becomes impractical above ~50 million pages.** At that scale, the 3× embedding cost at ingestion time is a real barrier. Dynamic proposition extraction — extracting propositions at query time from the top-$k$ coarse results, then reranking — is more practical, though it adds latency.

---

## Challenges and Open Problems

**Granularity classifier calibration.** The three-way classification (lookup, synthesis, hybrid) is a coarse taxonomy. In practice, many queries are ambiguous or multi-part, requiring different granularities for different sub-questions. A compositional architecture that decomposes queries and routes each sub-question independently would be more powerful, at the cost of complexity. **IRCoT** ([Trivedi et al. 2022](https://arxiv.org/abs/2212.10509)) demonstrates that interleaving retrieval with chain-of-thought reasoning enables multi-hop retrieval without explicit query decomposition, but this approach doesn't address granularity selection.

**Proposition extraction quality.** The quality of downstream retrieval is bounded by the quality of proposition extraction. Current extraction prompts produce propositions of varying quality: some are genuinely atomic and self-contained, others introduce subtle paraphrase errors or omit qualifications present in the source. No reliable automatic quality metric for proposition extraction exists that doesn't itself require a reference corpus of ground-truth propositions. **ChunkRAG** ([Raina & Gales 2024](https://aclanthology.org/2024.fever-1.25)) uses chunk-level filtering with a secondary LLM pass to remove low-quality retrieved units — a similar quality control mechanism is needed at the extraction stage.

**Cross-granularity consistency.** When a proposition and its parent passage both rank highly for the same query, the LLM receives the same information at two levels of detail. This redundancy is sometimes useful (proposition provides the precise figure, passage provides the interpretive context) and sometimes harmful (the LLM has to reconcile two representations of the same fact). A deduplication pass that collapses a proposition into its parent passage when both are retrieved would reduce this, but the collapse criterion is nontrivial — collapsing always would defeat the purpose of proposition-level retrieval.

**Temporal consistency of propositions.** When source documents are updated, propositions extracted from old versions persist in the index. A proposition that was true in Q4 2024 ("the threshold is 1.5%") may be false after a policy update in Q1 2025. Passage-level indices suffer from the same problem, but it is more acute for propositions because their specificity makes staleness immediately harmful. Versioned indexing — retaining propositions with validity timestamps and filtering at query time — is straightforward to implement in Qdrant via payload filters but requires discipline in the ingestion pipeline.

**Cold start for the granularity classifier.** The classifier is prompted on query-type examples from the target domain. For a new domain, no labeled examples exist. Few-shot prompting with generic examples works acceptably but calibrates less well than domain-specific fine-tuning. A self-supervised approach — observe which granularity level the LLM actually attends to when presented with mixed-granularity context, and use that as a training signal — is plausible but untested.

---

## Citation

```bibtex
@misc{inamdar2026retrievalgranularity,
  title   = {The Retrieval Granularity Problem in RAG},
  author  = {Inamdar, Mihir},
  journal = {AI Hive},
  year    = {2026},
  url     = {https://inamdarmihir.github.io/aihive/posts/retrieval-granularity-rag/}
}

@inproceedings{lewis2020rag,
  title     = {Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks},
  author    = {Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and others},
  booktitle = {NeurIPS},
  year      = {2020},
  url       = {https://arxiv.org/abs/2005.11401}
}

@inproceedings{chen2024densex,
  title     = {Dense X Retrieval: What Retrieval Granularity Should We Use?},
  author    = {Chen, Tong and Wang, Hongwei and Chen, Sihao and others},
  booktitle = {EMNLP},
  year      = {2024},
  url       = {https://arxiv.org/abs/2312.06648}
}

@inproceedings{zhong2024mog,
  title     = {Mix-of-Granularity: Optimize the Chunking Granularity for Retrieval-Augmented Generation},
  author    = {Zhong, Zhangshen and others},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.00456}
}

@inproceedings{wang2025proprag,
  title     = {PropRAG: Guiding Retrieval with Beam Search over Proposition Paths},
  author    = {Wang, Jingjin and Han, Jiawei},
  booktitle = {EMNLP},
  year      = {2025},
  url       = {https://aclanthology.org/2025.emnlp-main.317}
}

@inproceedings{raina2024atomic,
  title     = {Question-Based Retrieval using Atomic Units for Enterprise RAG},
  author    = {Raina, Vatsal and Gales, Mark},
  booktitle = {FEVER Workshop at EMNLP},
  year      = {2024},
  url       = {https://aclanthology.org/2024.fever-1.25}
}

@article{trivedi2022ircot,
  title  = {Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions},
  author = {Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  year   = {2022},
  url    = {https://arxiv.org/abs/2212.10509}
}
```
