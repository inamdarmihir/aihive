---
title: "Driftsync: Incremental Embedding Synchronization for Living Corpora"
date: 2026-07-26
description: "A source-agnostic sync engine that re-embeds exactly what changed in a living corpus — handling content edits, moves, deletes, and embedding-pipeline drift — without nightly full rebuilds, using Qdrant as the reference vector store."
tags: ["qdrant", "embeddings", "rag", "synchronization", "vector-search", "system-design"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Every retrieval-augmented system eventually runs into the same quiet failure mode: the documents change, and the embeddings don't. A page gets rewritten, a section moves, a doc gets deleted — and the vector index keeps serving the old text with full confidence, because nothing told it otherwise. This post is about closing that gap properly: not "re-embed everything nightly," but a general, source-agnostic synchronization engine that re-embeds exactly what changed, survives changes to the embedding pipeline itself, and stays cheap at millions of chunks. I'll use Qdrant as the reference vector database throughout, but the design has no dependency on it.

I would like to narrow the scope up front. This post does not cover *how* to chunk documents well, nor does it cover retrieval quality — it assumes you already have a chunking and embedding pipeline and asks only: given that pipeline, how do you keep a vector index consistent with a source corpus that keeps changing underneath it, without re-processing the world on every run?

## Table of Contents

1. [Why This Is Harder Than It Looks](#why-this-is-harder-than-it-looks)
2. [Two Kinds of Identity](#two-kinds-of-identity)
3. [The Missing Dimension: Pipeline Drift](#the-missing-dimension-pipeline-drift)
4. [Diffing at Scale: A Merge-Join](#diffing-at-scale-a-merge-join)
5. [Detecting Moves and Duplicates](#detecting-moves-and-duplicates)
6. [A Source-Agnostic Adapter Interface](#a-source-agnostic-adapter-interface)
7. [The Sync Engine](#the-sync-engine)
8. [Making It Public](#making-it-public)
9. [Challenges and Open Problems](#challenges-and-open-problems)
10. [Citation](#citation)

## Why This Is Harder Than It Looks

The naive fix — recompute every embedding on a schedule — works fine at small scale and fails in two independent ways as a system matures. First, it becomes computationally wasteful: re-embedding a corpus that's 99.9% unchanged to catch the 0.1% that isn't is pure overhead, and that overhead grows linearly with corpus size while the actual amount of change usually doesn't. Second, and less obviously, it stops being well-defined once you're indexing more than one source. "Everything" is easy to say about a single docs site; it's ambiguous once your index spans a git repo, a wiki, and a PDF archive that update on different schedules.

A better pattern has started showing up in individual tutorials, Qdrant's own write-up on syncing their internal docs among them: give every chunk of text a content fingerprint and a stable identity, diff incoming chunks against a record of what's already indexed, and touch only what changed. The idea itself borrows, whether or not the authors intended it, from a much older piece of systems work: Tridgell and Mackerras's **rsync algorithm** ([Tridgell & Mackerras, 1996](https://rsync.samba.org/tech_report/)), which synchronizes files across a network by hashing blocks and transferring only the blocks whose checksums don't match. Embedding sync is the same idea one layer up the stack — checksums over chunks instead of checksums over file blocks — and it's worth being explicit about that lineage, because the failure modes are the same ones rsync's designers already had to solve: what happens when block boundaries shift, and how do you diff efficiently without shipping the whole file to compare it.

What existing write-ups get right is the diffing idea itself. What they don't address is what happens when the *pipeline*, not just the corpus, changes — and how to diff efficiently once "corpus" means millions of chunks across several heterogeneous sources rather than one crawl of one docs site. Those are the two gaps this post tries to close.

## Two Kinds of Identity

Every chunk needs two independent identifiers, because they answer two different questions that get conflated if you only track one:

- A **content hash** $h = \text{SHA-256}(\text{normalize}(\text{text}))$, answering *"have I seen this exact text before, anywhere?"*
- A **deterministic point ID** $\text{id} = f(\text{source\_id}, \text{location}, \text{chunk\_index})$, answering *"is this still the same slot in the corpus?"*

Comparing an incoming stream of $(\text{id}, h)$ pairs against a stored manifest of the same shape sorts every chunk into exactly one of five outcomes:

| Manifest has id? | Hash matches? | Outcome |
|---|---|---|
| yes | yes | unchanged — skip |
| yes | no | content edited in place — re-embed |
| no (but $h$ seen elsewhere) | — | moved or duplicated — reuse vector |
| no | — | genuinely new — embed and insert |
| id present in manifest, absent from incoming | — | deleted — remove |

This table is the correct core of the idea, and it's the part that transfers cleanly from the rsync lineage: a fixed identity plus a content checksum is enough to classify change without ever comparing the actual text. The open questions are what identity and checksum mean once the pipeline itself is not static, and how the comparison is executed once it can't fit in a single request-response cycle. Those are addressed next.

## The Missing Dimension: Pipeline Drift

Existing designs implicitly treat the corpus as the only thing that drifts and the pipeline as fixed forever. That assumption breaks the first time a team upgrades its embedding model or changes chunk size from, say, 512 to 800 tokens. Both changes invalidate every existing vector even though the underlying text hasn't moved at all — and a pure content-hash diff will classify all of it as "unchanged" and silently skip it. The index becomes internally consistent and simultaneously wrong.

The fix is to widen the manifest's notion of identity to include pipeline state:

$$
\text{row} = (\text{id},\ h,\ v_{\text{embed}},\ v_{\text{chunk}},\ \text{location},\ \text{run\_id})
$$

where $v_{\text{embed}}$ and $v_{\text{chunk}}$ are version tags for the embedding model and the chunking algorithm respectively. A chunk only counts as unchanged when $h$, $v_{\text{embed}}$, and $v_{\text{chunk}}$ all match the stored row. Bumping either version tag turns a model or chunker upgrade into an ordinary, first-class sync event: every affected row is flagged for re-embedding, and the engine can stage that work — rate-limited, prioritized, resumable — instead of either ignoring the drift or triggering an uncontrolled full rebuild. This is a small change to the schema and, as far as I can tell, absent from every public write-up of this pattern so far.

## Diffing at Scale: A Merge-Join

A point-existence check against the vector database for every incoming chunk is the natural first implementation, and it's fine for a few thousand chunks. It stops being fine once round-trip latency dominates: at $n$ chunks and one network call per chunk, wall-clock time grows as $O(n \cdot d)$ where $d$ is per-call latency, and $d$ does not shrink as $n$ grows.

The standard database answer to this class of problem is a **sort-merge join** ([Selinger et al., 1979](https://dl.acm.org/doi/10.1145/582095.582099)), and it applies directly here:

1. Bulk-load the existing manifest for a given source, sorted by point ID, in one pass (a single scroll/filter query rather than $n$ point lookups).
2. Stream incoming chunks from the source adapter, sorted the same way.
3. Walk both sorted sequences together in lock-step, classifying each row in $O(1)$ against the table above.
4. Batch the resulting inserts, updates, and deletes rather than issuing them one at a time.

This turns an $O(n \cdot d)$ network-bound diff into an $O(n \log n)$ in-memory sort (or $O(n)$ if the source can already yield chunks in ID order) plus a constant number of bulk reads and batched writes. In practice this is the difference between a sync run measured in minutes versus one measured in hours once a corpus crosses roughly $10^5$–$10^6$ chunks.

## Detecting Moves and Duplicates

The merge-join above only catches moves and duplicates *within* a single source's crawl — it says nothing about content that migrated from one source to another, or that was copied verbatim into a second document. Catching that requires a second index: a corpus-wide map from content hash to point ID, built once and consulted whenever the merge-join produces a "new chunk, no ID match" result. If the hash is already present anywhere in the index, the existing vector is reused and only the manifest row is updated — no embedding call is made.

Exact-hash matching has an obvious blind spot: a paragraph that's copied and lightly edited gets a completely different hash and is treated as brand new, even though reusing its neighbor's vector (or skipping re-embedding, if the edit is truly cosmetic) would often be the more correct behavior. A cheap way to extend the same mechanism to near-duplicates is **SimHash** ([Charikar, 2002](https://dl.acm.org/doi/10.1145/509907.509965)), which produces fixed-length fingerprints such that similar inputs produce fingerprints with small Hamming distance — unlike a cryptographic hash, where a single-character edit is designed to change the output completely. Layering a SimHash index alongside the exact-hash index would let the engine flag "probably-the-same-paragraph, re-embedding optional" as a sixth outcome, at the cost of tolerating occasional false positives. I have not built this layer yet; it's listed under open problems below.

## A Source-Agnostic Adapter Interface

Everything above is independent of where chunks come from. The only thing a new source needs to implement is a narrow protocol:

```python
from dataclasses import dataclass
from typing import Iterator, Protocol

@dataclass
class Chunk:
    location: str        # stable path / URL / block ID — not the content
    content: str
    chunk_index: int
    metadata: dict

class SourceAdapter(Protocol):
    source_id: str

    def iter_chunks(self) -> Iterator[Chunk]:
        """Yield every chunk currently present at the source."""
```

A git-docs adapter walks tracked Markdown files; a sitemap adapter crawls and strips boilerplate HTML; a Confluence adapter paginates the REST API. None of them need to know anything about hashing, diffing, or the vector database — that logic lives once, in the sync engine, and every adapter inherits it for free.

## The Sync Engine

Putting the pieces together:

```python
def sync(adapter: SourceAdapter, embed_version: str, chunk_version: str):
    incoming = sorted(
        (make_point_id(adapter.source_id, c.location, c.chunk_index), c)
        for c in adapter.iter_chunks()
    )
    manifest = load_manifest(adapter.source_id)   # bulk-loaded, sorted
    hash_index = load_hash_index()                # corpus-wide, for move/dup detection

    for point_id, chunk in incoming:
        h = sha256(chunk.content)
        row = manifest.get(point_id)

        if row and row.matches(h, embed_version, chunk_version):
            continue                                          # unchanged
        elif row:
            reembed_in_place(point_id, chunk)                  # content or pipeline drift
        elif h in hash_index:
            reuse_vector(hash_index[h], point_id, chunk)       # moved or duplicated
        else:
            embed_and_insert(point_id, chunk)                  # genuinely new

    delete_orphans(manifest, seen_ids={pid for pid, _ in incoming})
    commit_manifest(adapter.source_id, run_id=new_run_id())
```

Every step is idempotent by construction, because point IDs are pure functions of source, location, and chunk index rather than incrementing counters. A crash mid-run simply means the next run reclassifies the same chunks the same way; the manifest commit happens last, so a partial run can never leave the index in a state the manifest doesn't already agree with — worst case, some work is redone, never lost or double-applied.

## Making It Public

The plan is a small, dependency-light, MIT-licensed Python package:

- A backend-agnostic core: the manifest schema, the merge-join diff, and the five-way (soon six-way) classification logic.
- A Qdrant reference backend — payload-based manifest storage for smaller corpora, with an optional external SQLite/Postgres manifest for corpora large enough that bulk scroll itself becomes a bottleneck.
- Reference adapters for git/Markdown, sitemap-crawled HTML, Confluence, and Notion.
- A CLI (`driftsync run --source config.yaml`) so common sources need zero custom code to wire into a nightly job or CI step.

## Challenges and Open Problems

**Chunk boundary stability.** If the chunker's output shifts whenever unrelated nearby content changes, chunks that are semantically identical get new hashes for no real reason, and the diff loses most of its benefit. This is precisely the problem **content-defined chunking** was built to solve in the file-systems literature — Muthitacharoen et al.'s LBFS ([Muthitacharoen et al., 2001](https://dl.acm.org/doi/10.1145/502034.502052)) uses a rolling hash to pick chunk boundaries based on local content rather than fixed offsets, so an edit in the middle of a file only perturbs the chunks near it. An equivalent idea — boundary selection driven by local structure (headings, sentence boundaries) rather than fixed token counts — would make the identity scheme in this post considerably more robust, and is not yet implemented.

**Near-duplicate detection.** As noted above, exact hashing is blind to lightly-edited duplicates. A SimHash or MinHash ([Broder, 1997](https://ieeexplore.ieee.org/document/666900)) layer alongside the exact-hash index is the natural extension and is left for future work.

**Cross-source identity.** The same content published to a docs site and re-exported as a PDF currently looks like two unrelated corpora unless an operator explicitly merges their `source_id` namespaces. Resolving this automatically would require the near-duplicate layer above plus some notion of canonical source precedence, and I don't have a design for that yet that I'd trust in production.

## Citation

```bibtex
@techreport{tridgell1996rsync,
  title={The rsync algorithm},
  author={Tridgell, Andrew and Mackerras, Paul},
  year={1996},
  institution={Australian National University}
}

@inproceedings{selinger1979access,
  title={Access path selection in a relational database management system},
  author={Selinger, P. Griffiths and Astrahan, M. M. and Chamberlin, D. D. and Lorie, R. A. and Price, T. G.},
  booktitle={Proceedings of the 1979 ACM SIGMOD International Conference on Management of Data},
  year={1979}
}

@inproceedings{charikar2002similarity,
  title={Similarity estimation techniques from rounding algorithms},
  author={Charikar, Moses S.},
  booktitle={Proceedings of the 34th Annual ACM Symposium on Theory of Computing},
  year={2002}
}

@inproceedings{muthitacharoen2001low,
  title={A low-bandwidth network file system},
  author={Muthitacharoen, Athicha and Chen, Benjie and Mazi{\`e}res, David},
  booktitle={Proceedings of the 18th ACM Symposium on Operating Systems Principles},
  year={2001}
}

@article{broder1997resemblance,
  title={On the resemblance and containment of documents},
  author={Broder, Andrei Z.},
  journal={Proceedings of Compression and Complexity of Sequences},
  year={1997}
}
```
