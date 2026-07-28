---
title: "Driftsync: Incremental Embedding Synchronization for Living Corpora"
date: 2026-07-15
description: "A source-agnostic sync engine that re-embeds exactly what changed in a living corpus — handling content edits, moves, deletes, and embedding-pipeline drift — without nightly full rebuilds, using Qdrant as the reference vector store."
tags: ["qdrant", "embeddings", "rag", "synchronization", "vector-search", "system-design"]
author: "Mihir Inamdar"
showToc: true
math: true
---

Every retrieval-augmented system eventually hits the same quiet failure: documents change and embeddings don't. A page is rewritten, a section moves, a doc is deleted — and the vector index keeps serving stale text with full confidence. This post closes that gap properly: not "re-embed everything nightly," but a source-agnostic sync engine that re-embeds exactly what changed, survives embedding-pipeline changes, and stays cheap at millions of chunks. I'll use Qdrant as the reference vector store; the design does not depend on it.

I would like to narrow the scope up front. This post does not cover chunking quality or retrieval quality — it assumes you have a chunking and embedding pipeline and asks only how to keep a vector index consistent with a living corpus without re-processing the world on every run. Out of scope: multi-modal embeddings, cross-lingual alignment, and query-time hybrid ranking.

## Table of Contents

1. [Why This Is Harder Than It Looks](#why-this-is-harder-than-it-looks)
2. [Two Kinds of Identity](#two-kinds-of-identity)
3. [The Missing Dimension: Pipeline Drift](#the-missing-dimension-pipeline-drift)
4. [Diffing at Scale: A Merge-Join](#diffing-at-scale-a-merge-join)
5. [Detecting Moves and Duplicates](#detecting-moves-and-duplicates)
6. [A Source-Agnostic Adapter Interface](#a-source-agnostic-adapter-interface)
7. [Qdrant Collection Schema](#qdrant-collection-schema)
8. [Manifest Storage: Payload vs External SQLite](#manifest-storage-payload-vs-external-sqlite)
9. [The Sync Engine](#the-sync-engine)
10. [Rate-Limited Re-embedding on Pipeline Bump](#rate-limited-re-embedding-on-pipeline-bump)
11. [CLI and Nightly Job Integration](#cli-and-nightly-job-integration)
12. [Evaluation / Worked Example](#evaluation--worked-example)
13. [Challenges and Open Problems](#challenges-and-open-problems)
14. [Citation](#citation)

## Why This Is Harder Than It Looks

The naive fix — recompute every embedding on a schedule — works at small scale and fails two ways as systems mature. First, it is wasteful: re-embedding a corpus that is 99.9% unchanged is overhead that grows with $N$ while change usually does not. Second, "everything" is ambiguous once the index spans a git repo, a wiki, and a PDF archive on different schedules.

A better pattern appears in tutorials (including Qdrant's docs sync write-up): fingerprint each chunk, diff against what's indexed, touch only what changed. The idea borrows from Tridgell and Mackerras's **rsync algorithm** ([Tridgell & Mackerras, 1996](https://rsync.samba.org/tech_report/)) — checksums over chunks instead of file blocks, with the same failure modes: shifting boundaries, and diffing without shipping the whole corpus.

For $N$ chunks, daily change fraction $c$, per-chunk embed cost $e$, and sync overhead $s$:

$$
C_{\text{full}} = N \cdot e, \qquad C_{\text{inc}} = N \cdot s + c \cdot N \cdot e
$$

so $C_{\text{inc}} / C_{\text{full}} = s/e + c$. Typically $s/e \sim 10^{-3}$–$10^{-4}$ and $c \in [0.001, 0.05]$. At $N = 10^6$, $e = 50\text{ms}$, $c = 0.01$, a full rebuild is ~14 hours of serial embedding; incremental is ~8 minutes of embedding plus seconds of hashing.

Existing write-ups get the diff right. They miss pipeline drift and efficient multi-source diffing at millions of chunks — the gaps this post closes.

## Two Kinds of Identity

Every chunk needs two independent identifiers.

A **content hash** answers *"have I seen this exact text before?"*:

$$
h = \text{SHA-256}(\text{normalize}(\text{text}))
$$

Normalization collapses whitespace. Without it, a trailing space forces a pointless re-embed.

A **deterministic point ID** answers *"is this still the same corpus slot?"*:

$$
\text{id} = \text{UUID5}\big(\text{ns},\ \text{source\_id}\ \|\ \text{location}\ \|\ \text{chunk\_index}\big)
$$

UUID5 works natively as a Qdrant point ID. The ID must not depend on content — otherwise edits look like delete-plus-insert and moves become invisible.

Comparing incoming $(\text{id}, h)$ pairs against the manifest yields:

| Manifest has id? | Hash matches? | Pipeline versions match? | Outcome |
|---|---|---|---|
| yes | yes | yes | unchanged — skip |
| yes | no | — | content edited — re-embed |
| yes | yes | no | pipeline drift — staged re-embed |
| no (but $h$ seen elsewhere) | — | — | moved / duplicated — reuse vector |
| no | — | — | genuinely new — embed and insert |
| id in manifest, absent from incoming | — | — | deleted — remove |

This table is the core of the idea: identity plus checksum classifies change without comparing text. The pipeline-version column is next.

```python
import hashlib, uuid
from dataclasses import dataclass

DRIFTSYNC_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

def content_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()

def make_point_id(source_id: str, location: str, chunk_index: int) -> str:
    return str(uuid.uuid5(DRIFTSYNC_NS, f"{source_id}|{location}|{chunk_index}"))

@dataclass(frozen=True)
class ManifestRow:
    point_id: str; content_hash: str; embed_version: str; chunk_version: str
    location: str; source_id: str; run_id: str

    def matches(self, h: str, embed_v: str, chunk_v: str) -> bool:
        return (self.content_hash == h and self.embed_version == embed_v
                and self.chunk_version == chunk_v)
```

## The Missing Dimension: Pipeline Drift

Existing designs treat the corpus as the only thing that drifts. That breaks on the first embedding-model upgrade or chunk-size change from 512 to 800 tokens: every vector is invalid though the text hasn't moved, and a pure content-hash diff skips it all. The index is consistent and wrong.

Widen the manifest to include pipeline state:

$$
\text{row} = (\text{id},\ h,\ v_{\text{embed}},\ v_{\text{chunk}},\ \text{location},\ \text{run\_id})
$$

Unchanged requires $h$, $v_{\text{embed}}$, and $v_{\text{chunk}}$ all match. Bumping either tag turns a model/chunker upgrade into a first-class, rate-limited sync event instead of a silent skip or uncontrolled rebuild.

Version tags change when pipeline *semantics* change. I use `text-embedding-3-small@1536` and `markdown-heading-v2@512` — the `@` suffix captures dimension, window, overlap.

```python
@dataclass(frozen=True)
class PipelineConfig:
    embed_version: str; chunk_version: str; embed_dim: int; model_name: str
    def bump_embed(self, model_name: str, dim: int) -> "PipelineConfig":
        return PipelineConfig(f"{model_name}@{dim}", self.chunk_version, dim, model_name)
```

This is a small schema change absent from public write-ups. Pipeline upgrades become incremental jobs instead of weekend rebuilds.

## Diffing at Scale: A Merge-Join

A point-existence check per chunk is fine for thousands of chunks and fails once latency dominates: $O(n \cdot d)$ wall-clock. At $d = 5\text{ms}$, $n = 10^6$, that is ~1.4 hours of network waits before any embedding.

The database answer is a **sort-merge join** ([Selinger et al., 1979](https://dl.acm.org/doi/10.1145/582095.582099)):

1. Bulk-load the manifest for a source, sorted by point ID.
2. Stream incoming chunks sorted the same way.
3. Walk both lock-step, classifying each row in $O(1)$.
4. Batch writes.

This turns $O(n \cdot d)$ into $O(n \log n)$ sorting (or $O(n)$ if already ID-ordered) plus bulk reads and batched writes. Memory is $O(n_{\text{source}})$ per source slice.

```python
from enum import Enum, auto
from typing import Iterator

class SyncAction(Enum):
    SKIP = auto(); REEMBED = auto(); REUSE = auto()
    INSERT = auto(); DELETE = auto()

def merge_join(
    incoming: list[tuple[str, "Chunk", str]],  # (point_id, chunk, hash)
    manifest: dict[str, ManifestRow],
    hash_index: dict[str, str],                # content_hash -> point_id
    embed_version: str,
    chunk_version: str,
) -> Iterator[tuple[SyncAction, str, object]]:
    seen: set[str] = set()
    for point_id, chunk, h in sorted(incoming, key=lambda x: x[0]):
        seen.add(point_id)
        row = manifest.get(point_id)
        if row and row.matches(h, embed_version, chunk_version):
            yield SyncAction.SKIP, point_id, chunk
        elif row:
            yield SyncAction.REEMBED, point_id, chunk
        elif h in hash_index:
            yield SyncAction.REUSE, point_id, (chunk, hash_index[h])
        else:
            yield SyncAction.INSERT, point_id, chunk
    for orphan_id in sorted(set(manifest) - seen):
        yield SyncAction.DELETE, orphan_id, None
```

In practice this is minutes vs hours past $10^5$–$10^6$ chunks. Checkpoint after each write batch on the last `point_id`; resume from that cursor on crash.

## Detecting Moves and Duplicates

The merge-join catches moves when the point ID changes (e.g. file rename). Cross-source copies need a corpus-wide `content_hash → point_id` map: on "new chunk, no ID match," reuse the vector if the hash exists.

```python
def build_hash_index(client, collection: str) -> dict[str, str]:
    index: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=1000, offset=offset,
            with_payload=["content_hash"], with_vectors=False,
        )
        for p in points:
            h = p.payload.get("content_hash")
            if h and h not in index:
                index[h] = str(p.id)
        if offset is None:
            break
    return index
```

Exact-hash matching is blind to lightly-edited duplicates. A cheap extension is **SimHash** ([Charikar, 2002](https://dl.acm.org/doi/10.1145/509907.509965)), which produces fixed-length fingerprints where similar inputs have small Hamming distance — unlike a cryptographic hash, where one character flips the entire output:

$$
\text{SimHash}(d) = \text{sign}\Big(\sum_{t \in d} w_t \cdot \text{hashbits}(t)\Big)
$$

Each token contributes a weighted $\pm 1$ vector from its hash; summing and taking the sign of each bit yields a fingerprint whose Hamming distance tracks bag-of-tokens similarity. Layering SimHash alongside the exact-hash index would add a sixth outcome — "probably the same paragraph, re-embedding optional" — at the cost of occasional false positives. I have not built this layer yet. **MinHash** ([Broder, 1997](https://ieeexplore.ieee.org/document/666900)) is the natural alternative when the goal is Jaccard similarity over shingles.

## A Source-Agnostic Adapter Interface

Everything above is independent of where chunks come from. A new source implements a narrow protocol; hashing, diffing, and the vector store live once in the sync engine.

```python
from dataclasses import dataclass, field
from typing import Iterator, Protocol

@dataclass
class Chunk:
    location: str; content: str; chunk_index: int
    metadata: dict = field(default_factory=dict)

class SourceAdapter(Protocol):
    source_id: str
    def iter_chunks(self) -> Iterator[Chunk]: ...
```

A git-docs adapter walks tracked Markdown and chunks on headings; a sitemap adapter crawls HTML; a Confluence adapter paginates the REST API:

```python
from pathlib import Path
import subprocess, re, requests
from bs4 import BeautifulSoup

class GitMarkdownAdapter:
    def __init__(self, repo: Path, source_id: str = "git-docs"):
        self.repo, self.source_id = repo, source_id

    def iter_chunks(self) -> Iterator[Chunk]:
        files = subprocess.check_output(
            ["git", "-C", str(self.repo), "ls-files", "*.md"], text=True).splitlines()
        for rel in files:
            text = (self.repo / rel).read_text(encoding="utf-8")
            for i, section in enumerate(self._split(text)):
                yield Chunk(rel, section, i, {"format": "markdown"})

    @staticmethod
    def _split(text: str) -> list[str]:
        return [p.strip() for p in re.split(r"(?=^#{1,3} )", text, flags=re.M) if p.strip()]

class SitemapAdapter:
    def __init__(self, sitemap_url: str, source_id: str = "web-docs"):
        self.sitemap_url, self.source_id = sitemap_url, source_id

    def iter_chunks(self) -> Iterator[Chunk]:
        urls = [loc.text for loc in BeautifulSoup(
            requests.get(self.sitemap_url, timeout=30).text, "xml").find_all("loc")]
        for url in urls:
            text = BeautifulSoup(requests.get(url, timeout=30).text,
                                 "html.parser").get_text("\n", strip=True)
            for i, start in enumerate(range(0, max(len(text), 1), 1050)):
                piece = text[start:start + 1200]
                if piece.strip():
                    yield Chunk(url, piece, i)

class ConfluenceAdapter:
    def __init__(self, base_url, space_key, token, source_id="confluence"):
        self.base_url, self.space_key = base_url.rstrip("/"), space_key
        self.token, self.source_id = token, source_id

    def iter_chunks(self) -> Iterator[Chunk]:
        start = 0
        while True:
            resp = requests.get(
                f"{self.base_url}/rest/api/content",
                params={"spaceKey": self.space_key, "expand": "body.storage",
                        "start": start, "limit": 50},
                headers={"Authorization": f"Bearer {self.token}"}, timeout=60)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                break
            for page in results:
                text = BeautifulSoup(page["body"]["storage"]["value"],
                                     "html.parser").get_text("\n", strip=True)
                for i, section in enumerate(GitMarkdownAdapter._split(text) or [text]):
                    yield Chunk(f"page:{page['id']}", section, i,
                                {"title": page.get("title")})
            start += 50
```

The subtle contract: `location` and `chunk_index` must be stable across runs for unchanged content. If an adapter renumbers chunks when an unrelated page is added, every point ID changes and sync degrades to a full rebuild. Heading-based chunking is more stable than fixed token windows; content-defined chunking is better still (see open problems).

## Qdrant Collection Schema

The index lives in a single collection `corpus_chunks`. Each chunk is one point: a dense vector plus a structured payload that doubles as the sync manifest for corpora that fit in a scroll. I focus on the schema decisions that make incremental sync efficient.

### Collection Creation and Payload Fields

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PayloadSchemaType, PointStruct,
    Filter, FieldCondition, MatchValue,
)
import time

client = QdrantClient(url="http://localhost:6333")

def ensure_collection(name: str, dim: int) -> None:
    if name not in {c.name for c in client.get_collections().collections}:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            optimizers_config={"indexing_threshold": 20_000},
        )

def make_chunk_point(point_id, vector, chunk, source_id, content_hash,
                     embed_version, chunk_version, run_id) -> PointStruct:
    return PointStruct(id=point_id, vector=vector, payload={
        "source_id": source_id, "location": chunk.location,
        "chunk_index": chunk.chunk_index, "content_hash": content_hash,
        "embed_version": embed_version, "chunk_version": chunk_version,
        "run_id": run_id, "text": chunk.content, "metadata": chunk.metadata,
        "sync_status": "active",  # active | pending_reembed | tombstone
        "updated_at": int(time.time()),
    })
```

`source_id` / `location` / `chunk_index` reconstruct identity; the three version fields drive the unchanged check; `run_id` ties points to a sync run; `sync_status` allows staged re-embeds without dropping points mid-migration; `text` makes re-embedding without re-fetching trivial (drop for large corpora, keep text in object storage keyed by hash).

### Payload Indexes

```python
def ensure_payload_indexes(collection: str) -> None:
    for field_name, schema in [
        ("source_id", PayloadSchemaType.KEYWORD),
        ("location", PayloadSchemaType.KEYWORD),
        ("content_hash", PayloadSchemaType.KEYWORD),
        ("embed_version", PayloadSchemaType.KEYWORD),
        ("chunk_version", PayloadSchemaType.KEYWORD),
        ("run_id", PayloadSchemaType.KEYWORD),
        ("sync_status", PayloadSchemaType.KEYWORD),
        ("updated_at", PayloadSchemaType.INTEGER),
        ("chunk_index", PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(collection, field_name, field_schema=schema)

def load_manifest(collection: str, source_id: str) -> dict[str, ManifestRow]:
    manifest, offset = {}, None
    while True:
        points, offset = client.scroll(
            collection, scroll_filter=Filter(must=[
                FieldCondition(key="source_id", match=MatchValue(value=source_id))
            ]),
            limit=1000, offset=offset, with_payload=True, with_vectors=False,
        )
        for p in points:
            pl = p.payload
            manifest[str(p.id)] = ManifestRow(
                str(p.id), pl["content_hash"], pl["embed_version"],
                pl["chunk_version"], pl["location"], pl["source_id"], pl["run_id"])
        if offset is None:
            break
    return manifest
```

## Manifest Storage: Payload vs External SQLite

For corpora up to a few hundred thousand chunks per source, storing the manifest in Qdrant payload works well: one filtered scroll loads everything, and there is a single source of truth. Beyond that, scroll latency grows with payload size, and the sync engine wants a reverse index from `content_hash` to `point_id` that avoids a full collection scan.

| Corpus size (chunks) | Manifest backend | Notes |
|---|---|---|
| $< 2 \times 10^5$ | Qdrant payload only | Simplest ops; one system to back up |
| $2 \times 10^5$ – $2 \times 10^6$ | Payload + SQLite hash index | Payload as SoT; SQLite accelerates move detection |
| $> 2 \times 10^6$ | External SQLite/Postgres as SoT | Payload holds retrieval fields; sync reads/writes SQL |

```python
import sqlite3
from pathlib import Path

class SqliteManifest:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS manifest (
                point_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                location TEXT NOT NULL, chunk_index INTEGER NOT NULL,
                content_hash TEXT NOT NULL, embed_version TEXT NOT NULL,
                chunk_version TEXT NOT NULL, run_id TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_src ON manifest(source_id, point_id);
            CREATE INDEX IF NOT EXISTS idx_hash ON manifest(content_hash);
        """)

    def load_source(self, source_id: str) -> dict[str, ManifestRow]:
        rows = self.conn.execute(
            "SELECT point_id, content_hash, embed_version, chunk_version, "
            "location, source_id, run_id FROM manifest WHERE source_id=? "
            "ORDER BY point_id", (source_id,)).fetchall()
        return {r[0]: ManifestRow(*r) for r in rows}

    def lookup_hash(self, h: str) -> str | None:
        row = self.conn.execute(
            "SELECT point_id FROM manifest WHERE content_hash=? LIMIT 1", (h,)
        ).fetchone()
        return row[0] if row else None

    def upsert_many(self, rows: list[ManifestRow]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO manifest VALUES (?,?,?,?,?,?,?,?)",
            [(r.point_id, r.source_id, r.location, 0, r.content_hash,
              r.embed_version, r.chunk_version, r.run_id) for r in rows])
        self.conn.commit()
```

When SQL is SoT, commit SQL *after* Qdrant upsert. Crash between leaves Qdrant ahead; next run repairs SQL. The opposite order leaves dangling rows pointing at missing vectors, so I prefer Qdrant-first.

## The Sync Engine

Putting the pieces together into a class that owns batching, idempotency, and crash recovery:

```python
from dataclasses import dataclass
from typing import Callable
import uuid

EmbedFn = Callable[[list[str]], list[list[float]]]

@dataclass
class SyncStats:
    skipped: int = 0; reembedded: int = 0; reused: int = 0
    inserted: int = 0; deleted: int = 0

class SyncEngine:
    def __init__(self, client, collection, pipeline, embed_fn,
                 batch_size=64, manifest=None):
        self.client, self.collection = client, collection
        self.pipeline, self.embed_fn = pipeline, embed_fn
        self.batch_size, self.manifest_store = batch_size, manifest
        ensure_collection(collection, pipeline.embed_dim)

    def sync(self, adapter: SourceAdapter) -> SyncStats:
        run_id, stats = str(uuid.uuid4()), SyncStats()
        incoming = sorted(
            [(make_point_id(adapter.source_id, c.location, c.chunk_index),
              c, content_hash(c.content)) for c in adapter.iter_chunks()],
            key=lambda x: x[0])
        if self.manifest_store:
            manifest = self.manifest_store.load_source(adapter.source_id)
            hash_index: dict[str, str] = {}
        else:
            manifest = load_manifest(self.collection, adapter.source_id)
            hash_index = build_hash_index(self.client, self.collection)

        to_embed, to_reuse, to_delete = [], [], []
        for action, pid, payload in merge_join(
            incoming, manifest, hash_index,
            self.pipeline.embed_version, self.pipeline.chunk_version,
        ):
            if action is SyncAction.SKIP:
                stats.skipped += 1
            elif action is SyncAction.REEMBED:
                to_embed.append((pid, payload, content_hash(payload.content)))
            elif action is SyncAction.INSERT:
                h = content_hash(payload.content)
                donor = (self.manifest_store.lookup_hash(h)
                         if self.manifest_store else hash_index.get(h))
                (to_reuse if donor else to_embed).append(
                    (pid, payload, h, donor) if donor else (pid, payload, h))
            elif action is SyncAction.REUSE:
                chunk, donor = payload
                to_reuse.append((pid, chunk, content_hash(chunk.content), donor))
            elif action is SyncAction.DELETE:
                to_delete.append(pid)

        self._flush_embeds(to_embed, adapter.source_id, run_id, stats)
        self._flush_reuses(to_reuse, adapter.source_id, run_id, stats)
        self._flush_deletes(to_delete, stats)
        return stats

    def _flush_embeds(self, items, source_id, run_id, stats) -> None:
        pv = self.pipeline
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            vecs = self.embed_fn([c.content for _, c, _ in batch])
            self.client.upsert(self.collection, points=[
                make_chunk_point(pid, v, c, source_id, h,
                                 pv.embed_version, pv.chunk_version, run_id)
                for (pid, c, h), v in zip(batch, vecs)])
            if self.manifest_store:
                self.manifest_store.upsert_many([
                    ManifestRow(pid, h, pv.embed_version, pv.chunk_version,
                                c.location, source_id, run_id)
                    for pid, c, h in batch])
            stats.inserted += len(batch)

    def _flush_reuses(self, items, source_id, run_id, stats) -> None:
        pv = self.pipeline
        for pid, chunk, h, donor_id in items:
            donor = self.client.retrieve(self.collection, ids=[donor_id],
                                         with_vectors=True)[0]
            self.client.upsert(self.collection, points=[
                make_chunk_point(pid, donor.vector, chunk, source_id, h,
                                 pv.embed_version, pv.chunk_version, run_id)])
            stats.reused += 1

    def _flush_deletes(self, ids, stats) -> None:
        if ids:
            self.client.delete(self.collection, points_selector=ids)
            stats.deleted += len(ids)
```

Every step is idempotent: point IDs are pure functions of source, location, and chunk index. A crash mid-run means the next run reclassifies the same way. Batches commit to Qdrant before optional SQL — worst case redo, never incorrect double-apply.

Failure modes: **API timeout mid-batch** — retry; upserts overwrite by ID. **Non-deterministic `chunk_index`** — looks like a full rebuild; alert when `deleted + inserted` spike. **Hash collision** — verify a content prefix on reuse if it matters. **Dimension mismatch after model bump** — new collection or named vector plus staged cutover.


## Rate-Limited Re-embedding on Pipeline Bump

When $v_{\text{embed}}$ changes, every row is stale, but re-embedding millions of chunks at once is rarely acceptable. I stage migration with a priority queue ordered by retrieval popularity (`hit_count`) when available, otherwise by oldest `updated_at`:

```python
import heapq  # ReembedTask ordering
from dataclasses import dataclass
from pathlib import Path

@dataclass(order=True)
class ReembedTask:
    priority: float; point_id: str; text: str

class PipelineMigrator:
    def __init__(self, engine: SyncEngine, max_per_minute: int = 120):
        self.engine, self.max_per_minute = engine, max_per_minute

    def plan(self, old_version: str) -> list[ReembedTask]:
        tasks, offset = [], None
        filt = Filter(must=[FieldCondition(
            key="embed_version", match=MatchValue(value=old_version))])
        while True:
            points, offset = self.engine.client.scroll(
                self.engine.collection, scroll_filter=filt, limit=500,
                offset=offset, with_payload=True, with_vectors=False)
            for p in points:
                tasks.append(ReembedTask(-float(p.payload.get("hit_count", 0)),
                                         str(p.id), p.payload.get("text", "")))
            if offset is None: break
        return tasks

    def run(self, tasks, new_pipeline, resume_from=0) -> int:
        import time as _time
        done, t0, n, batch = resume_from, _time.time(), 0, []
        for task in sorted(tasks)[resume_from:]:
            if not task.text: done += 1; continue
            batch.append(task)
            if len(batch) < self.engine.batch_size: continue
            self._commit(batch, new_pipeline)
            done += len(batch); n += len(batch); batch = []
            if n >= self.max_per_minute:
                _time.sleep(max(0.0, 60.0 - (_time.time() - t0))); t0, n = _time.time(), 0
            Path(".driftsync_migrate_cursor").write_text(str(done))
        if batch:
            self._commit(batch, new_pipeline); done += len(batch)
            Path(".driftsync_migrate_cursor").write_text(str(done))
        return done

    def _commit(self, batch, pipeline) -> None:
        vecs = self.engine.embed_fn([t.text for t in batch])
        self.engine.client.upsert(self.engine.collection, points=[
            PointStruct(id=t.point_id, vector=v, payload={
                "embed_version": pipeline.embed_version,
                "sync_status": "active", "updated_at": int(time.time()),
            }) for t, v in zip(batch, vecs)])
```

Query filters can accept either `embed_version` when spaces are compatible. When not (1536-d → 3072-d), dual-write a parallel collection and cut over at 100%. Resume is one integer on disk; lost cursor only costs re-planning.

## CLI and Nightly Job Integration

A sync engine that requires a custom script per source will not get run. The packaging goal is a small CLI that reads YAML and exits non-zero on partial failure so cron or CI can alert.

```yaml
# driftsync.yaml
collection: corpus_chunks
qdrant_url: http://localhost:6333
pipeline:
  embed_version: text-embedding-3-small@1536
  chunk_version: markdown-heading-v2@512
  embed_dim: 1536
  model_name: text-embedding-3-small
manifest: .driftsync/manifest.sqlite   # omit for payload-only
sources:
  - {type: git_markdown, source_id: product-docs, repo: /data/repos/product-docs}
  - {type: sitemap, source_id: help-center, sitemap_url: https://help.example.com/sitemap.xml}
  - {type: confluence, source_id: eng-wiki, base_url: https://example.atlassian.net/wiki,
     space_key: ENG, token_env: CONFLUENCE_TOKEN}
```

```python
# driftsync/cli.py
import argparse, os, sys, yaml

ADAPTERS = {
    "git_markdown": lambda c: GitMarkdownAdapter(Path(c["repo"]), c["source_id"]),
    "sitemap": lambda c: SitemapAdapter(c["sitemap_url"], c["source_id"]),
    "confluence": lambda c: ConfluenceAdapter(
        c["base_url"], c["space_key"], os.environ[c["token_env"]], c["source_id"]),
}

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="driftsync")
    sub = p.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("-c", "--config", default="driftsync.yaml")
    run_p.add_argument("--source")
    mig_p = sub.add_parser("migrate")
    mig_p.add_argument("-c", "--config", default="driftsync.yaml")
    mig_p.add_argument("--from-version", required=True)
    mig_p.add_argument("--to-model", required=True)
    mig_p.add_argument("--to-dim", type=int, required=True)
    mig_p.add_argument("--max-per-minute", type=int, default=120)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    pipeline = PipelineConfig(**cfg["pipeline"])
    manifest = SqliteManifest(Path(cfg["manifest"])) if cfg.get("manifest") else None
    engine = SyncEngine(QdrantClient(url=cfg["qdrant_url"]), cfg["collection"],
                        pipeline, build_embedder(pipeline), manifest=manifest)
    if args.cmd == "run":
        rc = 0
        for src in cfg["sources"]:
            if args.source and src["source_id"] != args.source: continue
            try: print(f"[{src['source_id']}] {engine.sync(ADAPTERS[src['type']](src))}")
            except Exception as exc:
                print(f"[{src['source_id']}] FAILED: {exc}", file=sys.stderr); rc = 1
        return rc
    new_p = pipeline.bump_embed(args.to_model, args.to_dim)
    cur = Path(".driftsync_migrate_cursor")
    PipelineMigrator(engine, args.max_per_minute).run(
        PipelineMigrator(engine).plan(args.from_version), new_p,
        int(cur.read_text()) if cur.exists() else 0)
    return 0
```

Nightly: `15 3 * * * driftsync run -c /etc/driftsync.yaml`. CI uses `--source` for the changed repo. Prefer nightly full-source plus per-PR scoped sync: PR catches edits quickly; nightly backstops deletes and webhook-less sources.

## Evaluation / Worked Example

I ran the engine against a mixed corpus: product Markdown in git ($42{,}000$ chunks), help-center sitemap ($18{,}000$), Confluence ($27{,}000$) — $87{,}000$ total, 1536-d `text-embedding-3-small`, Qdrant 1.9 single-node, payload-only manifest, batch size 64.

| Scenario | Change set | Wall clock | Embedding calls |
|---|---|---|---|
| Cold initial sync | 100% new | 48 min | 87{,}000 |
| Nightly, quiet day | 0.4% edit, 0.1% new, 0.05% del | 3.1 min | ~480 |
| Nightly, busy release | 4.2% edit, 1.1% new, 0.3% del | 11 min | ~4{,}900 |
| File rename storm (200 files) | IDs change, hashes stable | 2.4 min | 0 (all REUSE) |
| Pipeline bump (staged, 120/min) | 87k pending | ~12 hours | 87{,}000 |

With $c \approx 0.005$ on the quiet day, observed $3.1 / 48 \approx 0.065$; the gap versus $s/e + c$ is cold-sync indexing and the quiet-day full manifest scroll. At $10\times$ corpus size the ratio should improve toward $s/e + c$ as scroll moves to SQLite.

One failure: the sitemap adapter renumbered `chunk_index` after boilerplate-stripping shortened page text, flipping thousands of chunks to DELETE+INSERT. The fix was content-offset buckets rather than dense indices — adapter stability is part of the sync contract.

## Challenges and Open Problems

**Chunk boundary stability.** If the chunker's output shifts whenever nearby content changes, semantically identical chunks get new hashes and the diff loses its benefit. This is the problem **content-defined chunking** solved in the file-systems literature — Muthitacharoen et al.'s **LBFS** ([Muthitacharoen et al., 2001](https://dl.acm.org/doi/10.1145/502034.502052)) uses a rolling hash (Rabin fingerprints) so a mid-file edit only perturbs nearby chunks. An equivalent for text — headings, sentences, or Rabin cut points — is not yet implemented.

**Near-duplicate detection.** Exact hashing is blind to lightly-edited duplicates. A SimHash or MinHash layer is the natural extension. Silently reusing a near-duplicate is wrong for legal/medical corpora; an optional sixth action with an audit log is probably the right default.

**Cross-source identity.** The same content on a docs site and re-exported as a PDF looks like two corpora unless `source_id` namespaces are merged. Automatic resolution needs near-duplicates plus canonical source precedence; I don't yet have a production-ready design.

**Dimension-changing upgrades.** In-place updates work when vector size is unchanged. Switching dimensionality forces a new collection or named-vector migration, dual-read, and a cutover with recall checks — operational work I have only sketched.

**Manifest / index divergence.** Operators can delete points or restore backups out of order. A periodic `driftsync doctor` that recomputes hashes from stored `text` (or re-fetches from adapters) would close this loop; it does not exist yet.

**Multi-tenant isolation.** Every scroll/search must filter on `tenant_id`, and point IDs must incorporate the tenant — otherwise collisions overwrite another tenant's vectors. Easy to add to `make_point_id`, easy to forget in adapters.

## Citation

```bibtex
@techreport{tridgell1996rsync,
  title={The rsync algorithm}, author={Tridgell, Andrew and Mackerras, Paul},
  year={1996}, institution={Australian National University},
  url={https://rsync.samba.org/tech_report/}}
@inproceedings{selinger1979access,
  title={Access path selection in a relational database management system},
  author={Selinger, P. G. and Astrahan, M. M. and Chamberlin, D. D.
          and Lorie, R. A. and Price, T. G.},
  booktitle={Proc. 1979 ACM SIGMOD}, year={1979}, doi={10.1145/582095.582099}}
@inproceedings{charikar2002similarity,
  title={Similarity estimation techniques from rounding algorithms},
  author={Charikar, Moses S.}, booktitle={Proc. 34th ACM STOC},
  year={2002}, doi={10.1145/509907.509965}}
@inproceedings{muthitacharoen2001low,
  title={A low-bandwidth network file system},
  author={Muthitacharoen, Athicha and Chen, Benjie and Mazi{\`e}res, David},
  booktitle={Proc. 18th ACM SOSP}, year={2001}, doi={10.1145/502034.502052}}
@article{broder1997resemblance,
  title={On the resemblance and containment of documents},
  author={Broder, Andrei Z.},
  journal={Proc. Compression and Complexity of Sequences},
  year={1997}, doi={10.1109/SEQUEN.1997.666900}}
@techreport{rabin1981fingerprinting,
  title={Fingerprinting by random polynomials}, author={Rabin, Michael O.},
  year={1981}, institution={Harvard University}, note={TR-15-81}}
@misc{qdrant2024docs,
  title={Qdrant Documentation: Payload Indexes and Filtered Search},
  author={{Qdrant}}, year={2024},
  url={https://qdrant.tech/documentation/concepts/indexing/}}
```
