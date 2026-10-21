"""Can heavier HNSW build settings fix the accuracy loss at ~100k people (synthetic gallery)?
Builds the 99,499-vector gallery in Qdrant with several (m, ef_construct, segments) settings and, for each, searches a fixed 10,000-probe sample at several hnsw_ef values,
comparing top-1 and agreement with exact search. Needs Qdrant on localhost:6333 and the embeddings written by ../synthetic_100k/synth_100k.py.
   SYNTH_WORK=work_synth100k python -m experiments.hnsw_tuning.hnsw_tuning     ->  experiments/hnsw_tuning/hnsw_tuning.csv
"""
import csv, glob, os, time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

WD = os.environ.get("SYNTH_WORK", "work_synth100k")
def load(n):
    fs = sorted(glob.glob(f"{WD}/emb_{n}_*.npz"), key=lambda f: int(f.split("_")[-1].split(".")[0])); return np.vstack([np.load(f)["v"] for f in fs])
G, Pv = load("gal"), load("prb"); z = np.load(f"{WD}/results.npz", allow_pickle=True); gp, qp = z["gp"], z["qp"]
samp = np.random.default_rng(1).choice(len(Pv), 10000, replace=False); QS = Pv[samp]            # same sample as synth_100k.py
pos = {p: i for i, p in enumerate(gp)}; truth = np.array([pos[p] for p in qp[samp]])
q = QdrantClient(url="http://localhost:6333", timeout=1800); C = "bench_hnsw_tuning"
# (label, m, ef_construct, default_segment_number)   segments=0 -> Qdrant default (several segments, one graph each)
CONFIGS = [("m=16 efc=100 (previous settings)", 16, 100, 0), ("m=16 efc=100, 1 segment", 16, 100, 1), ("m=32 efc=200", 32, 200, 0), ("m=32 efc=200, 1 segment", 32, 200, 1), ("m=48 efc=400", 48, 400, 0)]
EFS = (64, 128, 256, 512, 1024)


def search(sp_, n=len(QS)):
    ids = []
    for s in range(0, n, 100):
        for r in q.query_batch_points(C, requests=[models.QueryRequest(query=v.tolist(), limit=1, params=sp_) for v in QS[s:s + 100]]): ids.append(r.points[0].id)
    return np.array(ids)


def latency(sp_, n=200):
    d = []
    for v in QS[:n]:
        t = time.perf_counter(); q.query_points(C, query=v.tolist(), limit=1, search_params=sp_); d.append((time.perf_counter() - t) * 1000)
    return round(float(np.percentile(d, 50)), 2), round(float(np.percentile(d, 95)), 2)


rows, exact_ids = [], None
out = Path(__file__).parent / "hnsw_tuning.csv"; hdr = ["config", "m", "ef_construct", "segments", "hnsw_ef", "top1", "same_top1_as_exact", "p50_ms", "p95_ms", "build_seconds", "indexed_vectors"]
for label, m, efc, seg in CONFIGS:
    if q.collection_exists(C): q.delete_collection(C)
    opt = dict(indexing_threshold=1, **({"default_segment_number": seg} if seg else {}))
    q.create_collection(C, vectors_config=models.VectorParams(size=G.shape[1], distance=models.Distance.COSINE), hnsw_config=models.HnswConfigDiff(m=m, ef_construct=efc),
                        optimizers_config=models.OptimizersConfigDiff(**opt))
    t0 = time.time(); q.upload_collection(C, vectors=G, ids=range(len(G)), batch_size=500, parallel=1)
    while time.time() - t0 < 5400:
        i = q.get_collection(C)
        if i.status == models.CollectionStatus.GREEN and i.indexed_vectors_count >= 0.99 * len(G): break
        time.sleep(3)
    build = round(time.time() - t0); print(f"[{label}] built in {build}s, indexed {i.indexed_vectors_count}/{i.points_count}, segments={i.segments_count}", flush=True)
    if exact_ids is None:
        exact_ids = search(models.SearchParams(exact=True)); ex_lat = latency(models.SearchParams(exact=True))
        rows.append(["exact scan (reference)", "", "", "", "", round(float((exact_ids == truth).mean()), 4), 1.0, *ex_lat, "", ""]); print(rows[-1], flush=True)
    for ef in EFS:
        sp_ = models.SearchParams(hnsw_ef=ef, exact=False); ids = search(sp_); lat = latency(sp_)
        rows.append([label, m, efc, i.segments_count, ef, round(float((ids == truth).mean()), 4), round(float((ids == exact_ids).mean()), 4), *lat, build, i.indexed_vectors_count]); print(rows[-1], flush=True)
    csv.writer(open(out, "w", newline="")).writerows([hdr] + rows)
q.delete_collection(C); print("wrote", out)
