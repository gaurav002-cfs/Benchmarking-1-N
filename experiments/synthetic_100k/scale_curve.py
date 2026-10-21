"""Same synthetic data, gallery sizes 500 ... 99,499 (10,000 random probes each): how accuracy falls as the gallery grows.
Reads the embeddings written by synth_100k.py.   python experiments/synthetic_100k/scale_curve.py  ->  scale_curve.csv"""
import csv, glob, os, time
from pathlib import Path

import numpy as np

WD = os.environ.get("SYNTH_WORK", "work_synth100k")
def load(n):
    fs = sorted(glob.glob(f"{WD}/emb_{n}_*.npz"), key=lambda f: int(f.split("_")[-1].split(".")[0])); return np.vstack([np.load(f)["v"] for f in fs])
G, U, Pv = load("gal"), load("unk"), load("prb"); rng = np.random.default_rng(0); NG = len(G)


def run(N, nprobe=10000):
    gi = rng.permutation(NG)[:N]; Gs = G[gi]
    pool = np.array([4 * j + k for j in gi for k in range(4)]); pi = rng.choice(pool, min(nprobe, len(pool)), replace=False)   # probes of gallery identity j = rows 4j..4j+3
    pos = {g: i for i, g in enumerate(gi)}; ti = np.array([pos[p // 4] for p in pi]); Q = Pv[pi]
    own, rank = [], []
    for s in range(0, len(Q), 1000):
        M = Q[s:s + 1000] @ Gs.T; o = M[np.arange(len(M)), ti[s:s + 1000]]; own.append(o); rank.append((M > o[:, None]).sum(1) + 1)
    own, rank = np.concatenate(own), np.concatenate(rank)
    us = np.concatenate([(U[s:s + 1000] @ Gs.T).max(1) for s in range(0, len(U), 1000)])
    return [N, round((rank <= 1).mean(), 4), round((rank <= 5).mean(), 4), round((rank <= 10).mean(), 4), round(float(np.median(own)), 3), round(float(np.median(us)), 3),
            round((us >= .4).mean(), 3), round((us >= .5).mean(), 3), round((us >= .6).mean(), 3)]


rows = [run(N) for N in (500, 2000, 8000, 32000, NG)]
out = Path(__file__).parent / "scale_curve.csv"
csv.writer(open(out, "w", newline="")).writerows([["gallery_size", "top1", "top5", "top10", "median_genuine_sim", "median_impostor_best_sim", "unknown_accepted_at_0.40", "at_0.50", "at_0.60"]] + rows)
for r in rows: print(r)
