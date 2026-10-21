"""HNSW (hnsw_ef sweep) vs exact search on the 8k gallery, top-1..top-10 accuracy + latency.
Needs Qdrant on localhost:6333 and the cached embeddings of one model (run `python -m benchmark.run glintr100-antelopev2` first).
   python -m experiments.hnsw_vs_exact.hnsw_sweep [slug]   (from the repo root)      ->  experiments/hnsw_vs_exact/hnsw_vs_exact.csv
"""
import csv, sys, time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from benchmark.common import ROOT, WORK, split

slug = sys.argv[1] if len(sys.argv) > 1 else "glintr100-antelopev2"
sp = split(); n_g, n_p = int((sp.split == "train").sum()), int((sp.split == "test").sum())
e = np.load(WORK / f"emb_{slug}.npz")["e"]; G, P = e[:n_g], e[n_g:n_g + n_p]
gl = sp[sp.split == "train"].person.to_numpy(dtype=object); pl = sp[sp.split == "test"].person.to_numpy(dtype=object)
pos = {p: i for i, p in enumerate(gl)}; ti = np.array([pos[p] for p in pl])
q = QdrantClient(url="http://localhost:6333", timeout=300); C = "bench_hnsw_8k"
if q.collection_exists(C): q.delete_collection(C)
q.create_collection(C, vectors_config=models.VectorParams(size=G.shape[1], distance=models.Distance.COSINE),
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100), optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1))  # force the graph to be built
q.upload_collection(C, vectors=G, ids=range(n_g), batch_size=500, parallel=1)
t0 = time.time()
while time.time() - t0 < 600:
    i = q.get_collection(C)
    if i.status == models.CollectionStatus.GREEN and i.indexed_vectors_count >= n_g: break
    time.sleep(1)
print("indexed:", i.indexed_vectors_count, "/", i.points_count)
cfg = [("exact", models.SearchParams(exact=True))] + [(f"hnsw ef={x}", models.SearchParams(hnsw_ef=x, exact=False)) for x in (16, 64, 128, 256)]
rows, ex1 = [], None
for name, sp_ in cfg:
    ids = []
    for s in range(0, n_p, 100):
        for r in q.query_batch_points(C, requests=[models.QueryRequest(query=v.tolist(), limit=10, params=sp_) for v in P[s:s + 100]]): ids.append([p.id for p in r.points])
    ids = np.array(ids); hit = ids == ti[:, None]
    if ex1 is None: ex1 = ids[:, 0]
    lat = []
    for v in P[:300]:
        t1 = time.perf_counter(); q.query_points(C, query=v.tolist(), limit=10, search_params=sp_); lat.append((time.perf_counter() - t1) * 1000)
    rows.append([name] + [round(float(hit[:, :k].any(1).mean()), 4) for k in range(1, 11)] + [round(float((ids[:, 0] == ex1).mean()), 4), round(float(np.percentile(lat, 50)), 2), round(float(np.percentile(lat, 95)), 2)])
    print(rows[-1], flush=True)
out = Path(__file__).parent / "hnsw_vs_exact.csv"
csv.writer(open(out, "w", newline="")).writerows([["search"] + [f"top{k}" for k in range(1, 11)] + ["same_top1_as_exact", "p50_ms", "p95_ms"]] + rows)
q.delete_collection(C); print("wrote", out)
