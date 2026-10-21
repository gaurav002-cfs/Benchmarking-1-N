# NOTE: adapted for portability (env vars for paths) AFTER the reported run, which used the original local paths. Not re-run in this form (about 100 min).
"""Synthetic ~100k-identity test: enroll image 0 of each identity, probe with images 1-4, 500 held-out unknown identities.
Embeddings via CoreML (resumable, chunked). Exact search via numpy; Qdrant exact vs HNSW checked on a probe sample."""
import glob, json, os, time

import cv2
import numpy as np
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
from insightface.model_zoo import get_model
from qdrant_client import QdrantClient, models

WD = os.environ.get("SYNTH_WORK", "work_synth100k"); os.makedirs(WD, exist_ok=True)
SRC = os.path.expanduser(os.environ.get("SYNTH_ROOT", "~/Downloads")); BS, CH, N_UNK, SEED = 64, 16384, 500, 42   # folder holding subjects_*_5_imgs/
roots = [r for r in sorted(glob.glob(f"{SRC}/subjects_*_5_imgs")) if os.path.isdir(r)]
id2dir = {p: os.path.join(r, p) for r in roots for p in os.listdir(r) if p.isdigit()}
ids = np.array(sorted(id2dir, key=int)); perm = np.random.default_rng(SEED).permutation(len(ids))
unk, enr = ids[perm[:N_UNK]], ids[perm[N_UNK:]]
print(f"identities={len(ids)} enrolled={len(enr)} unknown={len(unk)}", flush=True)

gal = [f"{id2dir[i]}/0.png" for i in enr]
prb = [f"{id2dir[i]}/{k}.png" for i in enr for k in range(1, 5)]; prb_person = np.repeat(enr, 4).astype(int)
unp = [f"{id2dir[i]}/{k}.png" for i in unk for k in range(5)]

P = os.path.abspath(os.environ.get("GLINTR100_ONNX", "weights/antelopev2/glintr100.onnx")); provs = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
rec = get_model(P, providers=provs); rec.prepare(ctx_id=-1)
so = ort.SessionOptions(); so.intra_op_num_threads = 8
rec.session = ort.InferenceSession(P, sess_options=so, providers=provs)
print("providers:", rec.session.get_providers(), flush=True)


def embed_stage(name, paths):
    out = np.zeros((len(paths), 512), np.float32); ok = np.zeros(len(paths), bool)
    with ThreadPoolExecutor(8) as ex:
        for c0 in range(0, len(paths), CH):
            cf = f"{WD}/emb_{name}_{c0 // CH}.npz"; sub = paths[c0:c0 + CH]
            if os.path.exists(cf):
                z = np.load(cf); out[c0:c0 + len(sub)] = z["v"]; ok[c0:c0 + len(sub)] = z["ok"]; continue
            t0 = time.time(); imgs = list(ex.map(cv2.imread, sub))
            v = np.zeros((len(sub), 512), np.float32); o = np.zeros(len(sub), bool)
            for b in range(0, len(sub), BS):
                batch = imgs[b:b + BS]; good = [j for j, im in enumerate(batch) if im is not None]
                if not good: continue
                arr = [batch[j] for j in good]; n = len(arr)
                arr += [arr[-1]] * (BS - n)                      # fixed batch shape for CoreML
                f = rec.get_feat(arr)[:n]; f = f / np.linalg.norm(f, axis=1, keepdims=True)
                for k, j in enumerate(good): v[b + j] = f[k]; o[b + j] = True
            np.savez(cf, v=v, ok=o); out[c0:c0 + len(sub)] = v; ok[c0:c0 + len(sub)] = o
            print(f"{name} {c0 + len(sub)}/{len(paths)} ({len(sub) / (time.time() - t0):.0f} img/s)", flush=True)
    return out, ok


gv, gok = embed_stage("gal", gal); uv, uok = embed_stage("unk", unp); pv, pok = embed_stage("prb", prb)
print(f"unreadable: gallery={int((~gok).sum())} probes={int((~pok).sum())} unknown={int((~uok).sum())}", flush=True)
G, gp = gv[gok], enr[gok].astype(int); Q, qp = pv[pok], prb_person[pok]; U = uv[uok]
tix = {p: i for i, p in enumerate(gp)}; ti = np.array([tix.get(p, -1) for p in qp]); keep = ti >= 0
Q, qp, ti = Q[keep], qp[keep], ti[keep]


def topk(X, k=10, ti=None):
    I, S, own, rank = [], [], [], []
    for s in range(0, len(X), 1000):
        M = X[s:s + 1000] @ G.T; part = np.argpartition(-M, k, axis=1)[:, :k]
        o = np.take_along_axis(M, part, 1).argsort(1)[:, ::-1]; idx = np.take_along_axis(part, o, 1)
        I.append(idx); S.append(np.take_along_axis(M, idx, 1))
        if ti is not None:
            ow = M[np.arange(len(M)), ti[s:s + 1000]]; own.append(ow); rank.append((M > ow[:, None]).sum(1) + 1)
    return (np.vstack(I), np.vstack(S)) + ((np.concatenate(own), np.concatenate(rank)) if ti is not None else ())


t0 = time.time(); tI, tS, own, rank = topk(Q, ti=ti); uI, uS = topk(U)
print(f"exact numpy search over {len(G)} gallery x {len(Q)} probes: {time.time() - t0:.0f}s", flush=True)
right = gp[tI[:, 0]] == qp; s1 = tS[:, 0]; us = uS[:, 0]
res = {"gallery": int(len(G)), "probes": int(len(Q)), "unknown_photos": int(len(U)),
       "top1": float((rank <= 1).mean()), "top5": float((rank <= 5).mean()), "top10": float((rank <= 10).mean()),
       "rank_gt_100": float((rank > 100).mean()), "median_true_sim_when_wrong": float(np.median(own[~right]))}
rows = []
for t in np.round(np.arange(0.20, 0.701, 0.05), 2):
    ok_ = (right & (s1 >= t)).sum(); bad = (~right & (s1 >= t)).sum(); unkacc = (us >= t).sum()
    rows.append(dict(threshold=float(t), found=ok_ / len(Q), confident_wrong=bad / len(Q), correct_rejected=(right & (s1 < t)).sum() / len(Q),
                     unknown_accepted=unkacc / len(U), precision_known=ok_ / max(ok_ + bad, 1)))
res["thresholds"] = rows
np.savez(f"{WD}/results.npz", tI=tI, tS=tS, own=own, rank=rank, qp=qp, gp=gp, uI=uI, uS=uS)
json.dump(res, open(f"{WD}/summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "thresholds"}), flush=True)
for r in rows: print({k: round(v, 4) for k, v in r.items()}, flush=True)

# ---- Qdrant: store gallery, compare exact vs HNSW on a sample ----
q = QdrantClient(url="http://localhost:6333", timeout=900); C = "synth_100k"
if q.collection_exists(C): q.delete_collection(C)
q.create_collection(C, vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100), optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1))
t0 = time.time(); q.upload_collection(C, vectors=G, ids=range(len(G)), batch_size=500, parallel=1)
while time.time() - t0 < 1800:
    i = q.get_collection(C)
    if i.status == models.CollectionStatus.GREEN and i.indexed_vectors_count >= 0.99 * len(G): break
    time.sleep(3)
print(f"qdrant: {i.points_count} points, hnsw-indexed {i.indexed_vectors_count}, upload+index {time.time() - t0:.0f}s", flush=True)
samp = np.random.default_rng(1).choice(len(Q), 10000, replace=False); QS = Q[samp]; exp1 = tI[samp, 0]; truth1 = qp[samp]
q_res = {}
for name, sp in [("exact", models.SearchParams(exact=True))] + [(f"hnsw_ef{e}", models.SearchParams(hnsw_ef=e)) for e in (16, 64, 128, 256)]:
    top = []
    for s in range(0, len(QS), 100):
        for r in q.query_batch_points(C, requests=[models.QueryRequest(query=v.tolist(), limit=1, params=sp) for v in QS[s:s + 100]]):
            top.append(r.points[0].id)
    top = np.array(top); lat = []
    for v in QS[:300]:
        t1 = time.perf_counter(); q.query_points(C, query=v.tolist(), limit=10, search_params=sp); lat.append((time.perf_counter() - t1) * 1000)
    q_res[name] = dict(top1=float((gp[top] == truth1).mean()), same_as_numpy_exact=float((top == exp1).mean()),
                       p50_ms=float(np.percentile(lat, 50)), p95_ms=float(np.percentile(lat, 95)))
    print(name, {k: round(v, 4) for k, v in q_res[name].items()}, flush=True)
res["qdrant_sample10k"] = q_res; json.dump(res, open(f"{WD}/summary.json", "w"), indent=1)
print("DONE", flush=True)
