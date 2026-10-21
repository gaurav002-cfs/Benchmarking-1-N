# HNSW vs exact search (8k gallery)

Question: how much accuracy does HNSW lose against an exact scan, and at what `hnsw_ef`?

**Method.** The 8,000 gallery embeddings of one model (default: `glintr100-antelopev2`) are loaded into a Qdrant collection (cosine, `m=16`, `ef_construct=100`).
`indexing_threshold=1` forces Qdrant to build the HNSW graph even though the collection is small (by default it does exact search below its threshold). All 24,000 probes are searched with
`exact=True` and with `hnsw_ef` = 16 / 64 / 128 / 256 (top-10). Latency = 300 single queries (p50 / p95).

**Files:** [`hnsw_sweep.py`](hnsw_sweep.py) (needs Qdrant on `localhost:6333` and the cached embeddings from `python -m benchmark.run glintr100-antelopev2`), [`hnsw_vs_exact.csv`](hnsw_vs_exact.csv).

```bash
python -m experiments.hnsw_vs_exact.hnsw_sweep [slug]     # from the repo root
```

**Result:** `ef >= 128` matches the exact search on 99.9% of probes; `ef=16` loses about 2 points of top-1. See the table in the [main README](../../README.md#hnsw-vs-exact-search-8k-gallery--experimentshnsw_vs_exact).
**Caveat:** this is 8,000 vectors on real faces; at 100k synthetic faces HNSW lost more (see [`synthetic_100k`](../synthetic_100k/)). Latency includes a local HTTP round trip to Qdrant (running in Colima).
