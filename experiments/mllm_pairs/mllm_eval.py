# NOTE: paths adapted for the repo. Run from the repo root:  python -m experiments.mllm_pairs.mllm_eval <pairs.jsonl>
"""Compare an MLLM's 'same person?' score with specialist embedding models on the SAME sampled pairs.
  python mllm_eval.py work_bench/llava15_pairs.jsonl
"""
import glob, json, os, sys

import numpy as np
import pandas as pd

rows = [json.loads(l) for l in open(sys.argv[1])]
y = np.array([r["y"] for r in rows]); n1, n0 = int(y.sum()), int((1 - y).sum())
import os
from benchmark.common import WORK, split
sp = split()
row_of = {p: i for i, p in enumerate(sp.relpath)}


def auc(s, y):                                   # rank-based AUC (Mann-Whitney)
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return (r[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum()))


def report(name, s):
    imp = np.sort(s[y == 0]); gen = s[y == 1]; ts = np.unique(s); acc = max(((s >= t) == y).mean() for t in ts)
    tar = lambda far: float((gen > np.quantile(imp, 1 - far)).mean())
    return dict(model=name, AUC=auc(s, y), best_threshold_acc=acc, TAR_at_FAR10=tar(0.10), TAR_at_FAR1=tar(0.01))


out = [report(f"MLLM: {os.path.basename(sys.argv[1]).split('_pairs')[0]}", np.array([r["score"] for r in rows]))]
saying_yes = np.mean([r["text"].strip().lower().startswith("yes") for r in rows])
for f in sorted(glob.glob(str(WORK / "emb_*.npz"))):
    E = np.load(f)["e"]; name = os.path.basename(f)[4:-4]
    out.append(report(name, np.array([float(E[row_of[r["a"]]] @ E[row_of[r["b"]]]) for r in rows])))
df = pd.DataFrame(out).sort_values("AUC", ascending=False)
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60)
print(f"pairs: {len(rows)} ({n1} same-person, {n0} different-person) | MLLM answered 'Yes' first on {saying_yes:.0%} of pairs")
print(df.round(3).to_string(index=False))
df.to_csv(sys.argv[1].replace(".jsonl", "_vs_specialists.csv"), index=False)
