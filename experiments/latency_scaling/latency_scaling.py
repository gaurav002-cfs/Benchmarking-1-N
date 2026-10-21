"""Search latency vs collection size (exact vs HNSW ef=128) and single-image embedding latency.
The 25k / 100k collections are the real 8k gallery vectors padded with RANDOM unit vectors: only latency is meaningful, not accuracy.
Needs Qdrant on localhost:6333 and cached baseline embeddings.   python -m experiments.latency_scaling.latency_scaling   (from the repo root)  ->  latency_scaling.csv"""
import csv, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from qdrant_client import QdrantClient, models

from benchmark.common import WORK, WEIGHTS, load_images, split

sp = split(); n_g, n_p = int((sp.split == "train").sum()), int((sp.split == "test").sum())
e = np.load(WORK / "emb_glintr100-antelopev2.npz")["e"]; G, P = e[:n_g], e[n_g:n_g + n_p]; rng = np.random.default_rng(0)
q = QdrantClient(url="http://localhost:6333", timeout=600); rows = []


def lat(C, sp_, n=150):
    for v in P[:10]: q.query_points(C, query=v.tolist(), limit=20, search_params=sp_)
    d = []
    for v in P[:n]:
        t = time.perf_counter(); q.query_points(C, query=v.tolist(), limit=20, search_params=sp_); d.append((time.perf_counter() - t) * 1000)
    return round(float(np.percentile(d, 50)), 2), round(float(np.percentile(d, 95)), 2)


for N in (8000, 25000, 100000):
    C = f"bench_scale_{N}"
    if q.collection_exists(C): q.delete_collection(C)
    q.create_collection(C, vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE), hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
                        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1))
    X = rng.standard_normal((N - n_g, 512)).astype(np.float32) if N > n_g else np.zeros((0, 512), np.float32)
    if len(X): X /= np.linalg.norm(X, axis=1, keepdims=True)
    q.upload_collection(C, vectors=np.vstack([G, X]), ids=range(N), batch_size=500, parallel=1)
    t0 = time.time()
    while time.time() - t0 < 900:
        i = q.get_collection(C)
        if i.status == models.CollectionStatus.GREEN and i.indexed_vectors_count >= 0.99 * N: break
        time.sleep(2)
    ex, hn = lat(C, models.SearchParams(exact=True)), lat(C, models.SearchParams(hnsw_ef=128)); rows.append([N, *ex, *hn]); print(rows[-1], flush=True); q.delete_collection(C)

# single-image embedding latency of the baseline on CPU (batch of 1, 8 threads)
so = ort.SessionOptions(); so.intra_op_num_threads = 8
s = ort.InferenceSession(str(next((WEIGHTS / "antelopev2").rglob("glintr100.onnx"))), sess_options=so, providers=["CPUExecutionProvider"]); inp = s.get_inputs()[0].name
x = ((np.asarray(load_images()[:1])[..., ::-1].astype(np.float32) - 127.5) / 127.5).transpose(0, 3, 1, 2).copy()
for _ in range(3): s.run(None, {inp: x})
d = []
for _ in range(30): t = time.perf_counter(); s.run(None, {inp: x}); d.append((time.perf_counter() - t) * 1000)
emb = [round(float(np.percentile(d, 50))), round(float(np.percentile(d, 95)))]; print("embed 1 image CPU p50/p95 ms:", emb)
out = Path(__file__).parent / "latency_scaling.csv"
w = csv.writer(open(out, "w", newline="")); w.writerow(["collection_size", "exact_p50_ms", "exact_p95_ms", "hnsw_ef128_p50_ms", "hnsw_ef128_p95_ms"]); w.writerows(rows); w.writerow([]); w.writerow(["embed_one_image_cpu_p50_ms", "p95_ms"]); w.writerow(emb)
