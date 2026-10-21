# Heavier HNSW build settings at ~100k people

**Question.** With `m=16, ef_construct=100`, HNSW lost about 4 points of top-1 on the 99,499-person synthetic gallery (see [`../synthetic_100k`](../synthetic_100k/)). Can heavier settings fix it, and at what cost?

**Method.** The 99,499 gallery vectors are loaded into Qdrant (cosine) with several `(m, ef_construct, default_segment_number)` settings, `indexing_threshold=1` to force the graph. For each build, the same 10,000-probe sample
(same seed as `synth_100k.py`) is searched at `hnsw_ef` = 64, 128, 256, 512, 1024 (top-1) and compared with Qdrant's exact search; latency = 200 single queries (p50 / p95); build time is wall-clock until the index is green.

**Files:** [`hnsw_tuning.py`](hnsw_tuning.py), [`hnsw_tuning.csv`](hnsw_tuning.csv).
```bash
SYNTH_WORK=work_synth100k python -m experiments.hnsw_tuning.hnsw_tuning     # from the repo root; needs Qdrant on localhost:6333 and the synth_100k embeddings
```

**Result** (exact scan: 82.05% top-1, 8.3 ms):

| Build settings | `hnsw_ef` | Same top-1 as exact | p50 | Build |
|---|--:|--:|--:|--:|
| m=16, ef_construct=100 | 128 | 95.4% | 3.6 ms | 58 s |
| m=16, ef_construct=100 | 512 | 99.3% | 7.0 ms | 58 s |
| **m=32, ef_construct=200** | 128 | 99.2% | 4.5 ms | 124 s |
| m=32, ef_construct=200 | 256 | 99.8% | 6.5 ms | 124 s |
| m=48, ef_construct=400 | 128 | 99.6% | 5.4 ms | 220 s |
| m=48, ef_construct=400 | 256 | 99.95% | 7.9 ms | 220 s |

A heavier build removes most of the loss (`m=32, ef_construct=200, hnsw_ef=128` agrees with exact on 99.2% of probes at about half the exact latency). Raising only `hnsw_ef` also recovers accuracy but its latency approaches the exact scan.
**Caveats:** `default_segment_number=1` did not change the reported segment count (2 in all runs), so the segment effect is not isolated; one build per setting; synthetic faces; single queries over local HTTP (Qdrant in a Colima VM); only ~100k vectors.
