# Hadith Semantic Search Benchmark

A **local, modular, research-grade semantic search system** for Hadith retrieval.
The goal is to compare different vector-search algorithms on a Hadith dataset,
measure their speed and retrieval quality, and identify the best approach for
eventual deployment in an Android application.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Architecture](#2-architecture)
3. [How Embeddings Work](#3-how-embeddings-work)
4. [Search Algorithms](#4-search-algorithms)
   - [Brute Force](#brute-force)
   - [IVF](#ivf--inverted-file-index)
   - [HNSW](#hnsw--hierarchical-navigable-small-worlds)
   - [PQ](#pq--product-quantization)
   - [IVF + PQ](#ivf--pq-combined)
5. [Why ANN is Approximate](#5-why-ann-is-approximate)
6. [Speed-Memory-Recall Tradeoffs](#6-speed-memory-recall-tradeoffs)
7. [Getting Started](#7-getting-started)
8. [CLI Reference](#8-cli-reference)
9. [Benchmarking Guide](#9-benchmarking-guide)
10. [Interpreting Benchmark Results](#10-interpreting-benchmark-results)
11. [Design Patterns](#11-design-patterns)
12. [Logging Architecture](#12-logging-architecture)
13. [Configuration](#13-configuration)
14. [Android Portability](#14-android-portability)
15. [Project Structure](#15-project-structure)

---

## 1. What This Project Does

This system:

1. Loads a Hadith dataset (JSON/JSONL)
2. Generates sentence embeddings using a multilingual model
3. Caches embeddings so they are computed only once
4. Builds different vector-search indexes
5. Accepts Arabic natural-language queries
6. Searches for semantically similar Hadiths
7. Benchmarks algorithms by speed, recall, and storage

The primary use case is **research and benchmarking** before committing to an
algorithm for mobile deployment. You can answer questions like:

> "For 50,000 Hadiths at 384-dimensional embeddings, is HNSW with ef_search=64 fast
> enough and accurate enough to run comfortably on a mid-range Android device?"

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Configuration (YAML)                      │
│              src/config/config.py — single source of truth       │
└──────────────────────────────────────────────────────────────────┘
                               │
         ┌─────────────────────┼──────────────────────┐
         ▼                     ▼                      ▼
  ┌─────────────┐    ┌──────────────────┐   ┌──────────────────┐
  │  HadithDataset  │    │  EmbeddingModel  │   │ VectorSearchIndex │
  │  (abstract)     │    │  (abstract)      │   │  (abstract)       │
  │  JSONHadithDataset  │    │  SentenceTransformerModel  │   │  BruteForceIndex  │
  └─────────────┘    └──────────────────┘   │  IVFIndex         │
                               │             │  HNSWIndex        │
                               ▼             │  PQIndex          │
                     ┌──────────────────┐   │  IVFPQIndex       │
                     │  EmbeddingCache  │   └──────────────────┘
                     │  (npz + JSON)    │            │
                     └──────────────────┘            │
                               │                     │
                               └──────┬──────────────┘
                                      ▼
                          ┌─────────────────────┐
                          │  HadithSearchEngine  │
                          │  (high-level API)    │
                          └─────────────────────┘
                                      │
                          ┌─────────────────────┐
                          │  BenchmarkRunner     │
                          │  metrics / CSV / JSON│
                          └─────────────────────┘
```

**Key design principles:**

- **No algorithm-specific logic outside the factory** — the engine never does `if algorithm == "hnsw"`
- **Single config object** — no global state, all parameters flow through `Config`
- **Lazy model loading** — the sentence-transformer is loaded only when first used
- **Structured logging** — no raw `print()` in library code; all logs go through `logging`

---

## 3. How Embeddings Work

Each Hadith text is converted into a **dense vector** (embedding) by a neural model.
Semantically similar texts produce vectors that are close together in the embedding space.

**Model used:** `paraphrase-multilingual-MiniLM-L12-v2`
- Supports 50+ languages including Arabic
- 384-dimensional output vectors
- ~120 MB on disk
- Designed for **symmetric retrieval** (query and document use same representation)

**L2 normalization:** All embeddings are L2-normalized so that:
- cosine similarity = dot product (faster to compute)
- Scores are in [-1, 1], with 1.0 meaning identical

**Generation pipeline:**

```
Hadith text
    ↓
Tokenization (WordPiece)
    ↓
BERT-style encoder (12 layers)
    ↓
Mean pooling of token embeddings
    ↓
L2 normalization
    ↓
384-dim float32 vector
```

Embeddings are computed **once**, saved to `data/processed/embeddings.npz`, and
reloaded on every subsequent run. If you change the model, the cache detects the
mismatch and prompts you to regenerate.

---

## 4. Search Algorithms

### Brute Force

**How it works:**  
For every query, compute cosine similarity against ALL N stored vectors.
Sort by score and return top-K.

**Complexity:**  
- Build: O(1) — just stores the matrix  
- Search: O(N·D) per query  
- Storage: N·D·4 bytes (float32)

**Recall:** Always 1.0 — examines every vector so nothing is missed.

**When to use:**  
Baseline and ground truth for benchmarking. Practical for small datasets (<10K on mobile).

```
embeddings matrix (N×D)
    ↓
query · embeddings^T   [dot product = cosine for normalized vectors]
    ↓
argsort descending
    ↓
Top-K results
```

---

### IVF — Inverted File Index

**How it works:**  
1. **Build:** Cluster all N vectors into K groups using K-Means. Store each vector in its cluster's "inverted list."  
2. **Search:** Find the `nprobe` nearest clusters to the query, then scan only those clusters.

**Key parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_clusters` | Number of Voronoi cells. Rule: ≈√N | 256 |
| `nprobe` | Clusters searched per query | 8 |

**Complexity:**  
- Build: O(N·K·D·iters) for K-Means  
- Search: O(K·D + N/K·nprobe·D)  
- Storage: Same as brute force (no compression)

**Recall vs speed tradeoff:**  
Increasing `nprobe` improves recall at the cost of search time. `nprobe=K` equals brute force.

```
Query
  ↓
Distance to K centroids: O(K·D)
  ↓
Select nprobe nearest centroids
  ↓
Scan nprobe inverted lists: O(N·nprobe/K·D)
  ↓
Top-K results
```

---

### HNSW — Hierarchical Navigable Small Worlds

**How it works:**  
HNSW builds a multi-layer proximity graph:

- **Layer 0:** All N vectors, densely connected (M edges per node)
- **Layer 1:** A random subset, sparser
- **Layer L (top):** A small number of "highway" nodes

**Search navigation:**
1. Start at the single entry point in the top layer
2. Greedily hop to the nearest neighbor in that layer
3. Descend to the layer below and repeat
4. At layer 0, perform beam search (keeping `ef_search` candidates)
5. Return top-K from the beam

**Key parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `M` | Max bidirectional edges per node | 16 |
| `ef_construction` | Build-time beam width (quality) | 200 |
| `ef_search` | Query-time beam width (recall vs latency) | 64 |

**`ef_search` effect on recall:**

| ef_search | Approx. Recall@10 | Notes |
|-----------|-------------------|-------|
| 16 | ~0.70–0.80 | Fastest |
| 64 | ~0.90–0.95 | Balanced |
| 200 | ~0.98+ | Near-exact |
| ≥ N | 1.00 | Equivalent to brute force |

**Complexity:**  
- Build: O(N·M·log N) time, O(N·M) extra for edges  
- Search: O(log N · ef_search · M)  
- Storage: ~(N·M·8) bytes for edge graph + N·D·4 bytes for vectors

**When to use:**  
Best recall/latency tradeoff for medium to large datasets. Recommended for Android deployment.

---

### PQ — Product Quantization

**How it works:**  
PQ compresses each D-dimensional vector into M bytes:

1. **Split:** Divide each vector into M subspaces of size D/M
2. **Train codebooks:** Run K-Means on each subspace independently, producing M codebooks each with K=2^bits centroids
3. **Encode:** Replace each subvector with the index of its nearest centroid (one byte)
4. **Search (ADC):** For a query, precompute a distance table (M × K), then score each database vector by summing M table lookups

**Key parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `subquantizers` (M) | Number of subspaces | 8 |
| `bits_per_code` | Bits per centroid index (8 → 256 centroids) | 8 |

**Storage comparison (384-dim, N=50K):**

| Format | Size |
|--------|------|
| Float32 (uncompressed) | 72 MB |
| PQ M=8 | 0.4 MB (180× smaller) |
| PQ M=16 | 0.8 MB |

**When to use:**  
Memory-constrained environments (e.g., Android). Excellent compression, moderate recall.

---

### IVF + PQ Combined

**How it works:**  
Combines IVF partitioning with PQ compression:

```
Query
  ↓ embed
Compute distance to IVF centroids
  ↓
Select nprobe clusters
  ↓
ADC search over PQ-compressed vectors in those clusters
  ↓
Top-K results
```

IVF reduces the number of vectors to score from N to ~N·nprobe/K.  
PQ makes each comparison cheap (byte table lookups instead of float dot products).

**When to use:**  
Large datasets (>100K) where you need both fast search AND small memory footprint.
This is the combination used by FAISS in production.

---

## 5. Why ANN is Approximate

Brute force finds the **exact** top-K nearest neighbors. ANN algorithms trade
**some recall** for **much faster search**:

- **IVF:** The true nearest neighbor may be in a cluster that isn't one of the
  `nprobe` searched. It's exact within the searched clusters.

- **HNSW:** The greedy graph traversal may get stuck in a local optimum and
  miss the true global nearest neighbor. Increasing `ef_search` reduces this risk.

- **PQ:** Each vector is replaced by its quantized approximation. Two vectors
  that are close together may be assigned to the same centroid, losing fine-grained distance info.

**The key insight:** In practice, for ranked retrieval the top-10 from an ANN
algorithm at Recall@10=0.95 is good enough — the 5% miss rate rarely matters
for the user's experience.

---

## 6. Speed-Memory-Recall Tradeoffs

| Algorithm | Build Time | Search Latency | Memory | Recall@10 | Android? |
|-----------|-----------|----------------|--------|-----------|----------|
| Brute Force | ≈0s | O(N) slow | N·D·4B | 1.00 | ✅ small N |
| IVF | seconds | fast | N·D·4B | ~0.85-0.95 | ✅ |
| HNSW | minutes | very fast | N·D·4B + graph | ~0.90-0.98 | ✅ recommended |
| PQ | seconds | fast | N·M bytes | ~0.70-0.90 | ✅ best size |
| IVF+PQ | seconds | fast | N·M bytes | ~0.80-0.92 | ✅ |

**Rule of thumb:** Start with HNSW at ef_search=64. Increase ef_search if recall is
too low, decrease it if latency is too high.

---

## 7. Getting Started

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt
```

### Full workflow

```bash
# Step 1: Prepare a sample dataset (500 Hadiths)
python app.py prepare

# Step 2: Generate embeddings (downloads model ~120MB on first run)
python app.py embeddings

# Step 3: Build indexes
python app.py build --algorithm brute_force
python app.py build --algorithm hnsw
python app.py build --algorithm all  # builds all 5

# Step 4: Search
python app.py search --algorithm hnsw --query "حديث عن الأعمال بالنية"

# Step 5: Benchmark
python app.py benchmark --algorithm all
```

### Quick test (no real model)

```bash
# Run the test suite with synthetic embeddings
pytest tests/ -v
```

---

## 8. CLI Reference

### `prepare` — Prepare dataset

```bash
python app.py prepare
```

Creates `data/raw/hadiths.json` with 500 sample Hadiths.
Replace with your actual corpus by placing it at the same path.

---

### `embeddings` — Generate embeddings

```bash
python app.py embeddings
python app.py embeddings --force   # Regenerate even if cached
```

Downloads model, encodes all Hadiths, saves to `data/processed/embeddings.npz`.

---

### `build` — Build index

```bash
python app.py build --algorithm hnsw
python app.py build --algorithm all
python app.py build --algorithm hnsw --set hnsw.M=32 --set hnsw.ef_construction=400
```

---

### `search` — Search

```bash
python app.py search --algorithm hnsw --query "حديث عن النية"
python app.py search --algorithm brute_force --query "فضل العلم" --top-k 20
python app.py search --algorithm ivf --query "الصلاة" --brief
```

---

### `benchmark` — Benchmark

```bash
python app.py benchmark --algorithm hnsw
python app.py benchmark --algorithm all
python app.py benchmark --algorithm all --rebuild   # Force rebuild
python benchmark.py --algorithm hnsw               # Equivalent shortcut
```

---

### `stats` — Index statistics

```bash
python app.py stats --algorithm hnsw
python app.py stats --algorithm all
```

---

### Global options

```
--config PATH        Custom YAML config file
--log-level LEVEL    DEBUG | INFO | WARNING | ERROR
--set KEY=VALUE      Override config values (repeatable)
```

---

## 9. Benchmarking Guide

### What the benchmark does

1. Loads cached embeddings
2. Selects `n_queries` random vectors from the dataset as benchmark queries
3. Computes **Brute Force exact results** as ground truth
4. For each algorithm: builds (or loads) the index, runs `warmup_queries` warm-up queries, then times the benchmark queries
5. Computes Recall@K, Precision@K, P50/P95/P99 latency
6. Prints a comparison table and saves JSON + CSV results

### Running a comparison

```bash
python app.py benchmark --algorithm all
```

Example output:

```
==================================================================
 BENCHMARK RESULTS
==================================================================
Algorithm       Build     Avg ms     P50 ms     P95 ms   Recall@K    Size MB
------------------------------------------------------------------
brute_force      0.00s       2.45       2.31       3.87     1.0000       2.29
ivf              8.32s       0.74       0.71       0.98     0.8850       2.29
hnsw            12.51s       0.31       0.29       0.45     0.9550       1.43
pq               6.14s       1.12       1.09       1.38     0.7300       0.02
ivf_pq          14.23s       0.52       0.49       0.71     0.8200       0.03
------------------------------------------------------------------
```

### Tuning for recall

If Recall@K is too low, try:

| Algorithm | Tuning |
|-----------|--------|
| IVF | Increase `nprobe` |
| HNSW | Increase `ef_search` |
| PQ | Increase `subquantizers` |
| IVF+PQ | Increase `nprobe` and `subquantizers` |

```bash
python app.py benchmark --algorithm hnsw --set hnsw.ef_search=128
```

### Saving benchmark results

All benchmark runs save to `data/benchmark_results/benchmark_YYYYMMDD_HHMMSS.json`
and `.csv` for reproducibility.

---

## 10. Interpreting Benchmark Results

### Recall@K

```
Recall@K = |ground_truth_ids ∩ candidate_ids| / K
```

- **1.00** = perfect (found all K true nearest neighbors)
- **0.95** = excellent for most retrieval tasks
- **0.80** = acceptable, may miss 2/10 relevant results
- Below 0.70 — consider tuning or switching algorithms

### Latency

- **Avg ms** — mean search latency (includes outliers)
- **P50** — median (half of queries faster, half slower)
- **P95** — 95th percentile (worst-case for 95% of queries)

For Android, target P95 < 50ms for a good user experience.

### Index size

- Brute Force and IVF store raw float32 vectors — no compression
- PQ and IVF+PQ compress dramatically (>100× smaller)
- HNSW adds ~M·8·N bytes for the graph structure on top of the vectors

### Decision framework

```
Dataset size < 10K:  Brute Force (exact, simple)
Dataset < 100K, mobile storage OK:  HNSW (best recall/latency)
Dataset > 100K, storage constrained:  IVF+PQ
Batch indexing, accuracy critical:  IVF + high nprobe
```

---

## 11. Design Patterns

### Factory Pattern

The `VectorSearchFactory` is the only place that maps names to classes:

```python
# app.py / search engine — never touches specific index classes
index = VectorSearchFactory.create("hnsw", config)  # Returns HNSWIndex
index.build(embeddings, ids)
result = index.search(query_vector, top_k=10)

# Adding a new algorithm:
# 1. Implement VectorSearchIndex in src/indexes/my_algo.py
# 2. Add to _REGISTRY in src/indexes/factory.py
# Done — no changes anywhere else.
```

### Abstract Interface

```python
class VectorSearchIndex(ABC):
    def build(self, embeddings, hadith_ids): ...
    def search(self, query_vector, top_k) -> SearchResult: ...
    def save(self, path): ...
    def load(self, path): ...
    def get_stats(self) -> IndexStats: ...
```

Every algorithm implements this. The search engine calls only these methods.

### Dependency Injection

`HadithSearchEngine` receives a `Config` object — it never calls `load_config()` itself.
This makes testing easy: inject a test config pointing to a temp directory.

---

## 12. Logging Architecture

All modules get a logger via:

```python
from src.logging.logger import get_logger
logger = get_logger(__name__)

logger.info("Loaded %d Hadiths", n)
logger.debug("Query shape: %s", query.shape)
logger.warning("top_k=%d exceeds dataset size %d", top_k, n)
logger.error("Failed to parse record %d: %s", i, exc)
```

Log format:
```
2026-08-29 19:32:11 | INFO     | src.indexes.hnsw | Index built in 2.41s
2026-08-29 19:32:16 | WARNING  | src.search.engine | top_k=100 > dataset size 50
```

Configure at startup:
```python
configure_logging(level="DEBUG")   # See all messages
configure_logging(level="WARNING") # Only problems
```

Or via CLI: `python app.py search --log-level DEBUG ...`

---

## 13. Configuration

Edit `configs/default.yaml` to change any default parameter.

Override at runtime with `--set`:

```bash
# Use a different model
python app.py embeddings --set embedding.model=sentence-transformers/paraphrase-arabic-xlm-r-roberta-base

# Tune HNSW for better recall
python app.py benchmark --algorithm hnsw --set hnsw.ef_search=128

# Tune IVF for a large dataset
python app.py build --algorithm ivf --set ivf.n_clusters=1024 --set ivf.nprobe=32
```

---

## 14. Android Portability

This Python system is for **offline research and index construction**.
For Android deployment, you carry the pre-built assets to the device:

### Portable components

| Component | How to port |
|-----------|-------------|
| Embedding model | Export to ONNX, run with ONNX Runtime Mobile |
| HNSW index | hnswlib binary file + hnswlib C++ via Android NDK |
| PQ index | Codebooks + byte codes = plain arrays → Kotlin/Java |
| IVF centroids | NumPy array → load as float array in Kotlin |
| BruteForce (small N) | Float matrix → simple dot product in Java |

### ONNX export (embedding model)

```python
from sentence_transformers import SentenceTransformer
from optimum.exporters.onnx import main_export

# After pip install optimum[onnxruntime]
main_export("paraphrase-multilingual-MiniLM-L12-v2", output="models/onnx/")
```

Then copy the ONNX model + the index binary to Android assets.

### NOT portable (offline only)

- Embedding generation pipeline (`scripts/generate_embeddings.py`)
- K-Means training for IVF/PQ codebooks
- Benchmark framework
- Python test suite

### Android runtime components (future)

```
Android App
  ├── assets/hnsw_index.bin       ← built offline
  ├── assets/embeddings_meta.json ← model info
  ├── assets/model.onnx           ← embedding model
  └── lib/
      ├── hnswlib.so              ← compiled C++ lib
      └── onnxruntime.aar         ← ONNX Runtime
```

---

## 15. Project Structure

```
hadith-search/
├── app.py                    ← Main CLI entry point
├── benchmark.py              ← Thin wrapper for benchmark subcommand
├── requirements.txt
├── configs/
│   └── default.yaml          ← Central configuration
├── data/
│   ├── raw/hadiths.json      ← Input dataset
│   ├── processed/            ← Cached embeddings (.npz + meta JSON)
│   ├── indexes/              ← Persisted index files (one dir per algo)
│   └── benchmark_results/    ← JSON + CSV benchmark outputs
├── models/                   ← Downloaded model cache
├── scripts/
│   ├── prepare_dataset.py    ← Create/validate dataset
│   └── generate_embeddings.py← Encode and cache embeddings
├── src/
│   ├── config/config.py      ← Typed config dataclasses + YAML loader
│   ├── logging/logger.py     ← Centralized logging setup
│   ├── profiling/timer.py    ← Timer context manager + @timed decorator
│   ├── datasets/
│   │   ├── base.py           ← HadithDataset ABC
│   │   └── json_loader.py    ← JSONHadithDataset
│   ├── embeddings/
│   │   ├── base.py           ← EmbeddingModel ABC
│   │   ├── sentence_transformer.py ← SentenceTransformer wrapper
│   │   ├── factory.py        ← EmbeddingFactory
│   │   └── cache.py          ← EmbeddingCache (npz + meta)
│   ├── indexes/
│   │   ├── base.py           ← VectorSearchIndex ABC, SearchResult, IndexStats
│   │   ├── brute_force.py    ← Exact cosine search
│   │   ├── ivf.py            ← K-Means + inverted lists
│   │   ├── hnsw.py           ← hnswlib-backed HNSW
│   │   ├── pq.py             ← Custom PQ with ADC search
│   │   ├── ivf_pq.py         ← IVF + PQ combined
│   │   └── factory.py        ← VectorSearchFactory registry
│   ├── search/engine.py      ← HadithSearchEngine (high-level API)
│   └── benchmarking/
│       ├── metrics.py        ← Recall@K, Precision@K, latency stats
│       └── benchmark.py      ← BenchmarkRunner
└── tests/
    ├── unit/                 ← Unit tests (no real model needed)
    └── integration/          ← End-to-end pipeline tests
```

---

## License

This codebase is a research prototype. The sample Hadith texts are sourced from
authentic collections for educational purposes. Ensure compliance with your dataset's
licensing terms in production.
