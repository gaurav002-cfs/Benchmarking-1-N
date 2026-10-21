# Search latency vs collection size

**Method.** Real 8k gallery vectors (baseline embeddings) padded with random unit vectors to 25,000 and 100,000, in Qdrant (cosine, `m=16`, `ef_construct=100`, graph forced with `indexing_threshold=1`).
150 single queries (top-20) after a 10-query warm-up; exact scan vs `hnsw_ef=128`. Also times one baseline embedding (ONNX CPU, batch of 1, 8 threads).

**Files:** [`latency_scaling.py`](latency_scaling.py), [`latency_scaling.csv`](latency_scaling.csv).
```bash
python -m experiments.latency_scaling.latency_scaling     # from the repo root; needs Qdrant on localhost:6333
```

**Findings.** Exact scan grows about linearly (fixed ~2.5 ms + ~0.07 ms per 1,000 vectors); HNSW stays almost flat (~3 ms). One embedding takes about 94 ms on CPU, so it dominates the request at these sizes.

**Caveats:** the padding vectors are random, so recall at 25k / 100k is *not* measured here (see [`synthetic_100k`](../synthetic_100k/) for that); Apple M2 Pro, Qdrant in a Colima VM, local HTTP round trip included; 1M-vector behaviour is not measured.
