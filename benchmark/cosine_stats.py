"""Cosine-similarity analysis per model: how large are the scores for the same person, for different people and for strangers, and which cosine
threshold gives which error rate. Needs the cached embeddings (python -m benchmark.run <slug>).

   python -m benchmark.cosine_stats [slug ...]     ->  models/<slug>/cosine.json, models/<slug>/cosine_distribution.png, results/cosine_stats.csv
   python -m benchmark.cosine_stats --photo-top5 slug [slug ...]   ->  results/photo_top5/<slug>.csv  (per probe photo: top-5 gallery photos with cosine / match %)

Scores are cosine similarities of L2-normalised embeddings (range -1..1; "match %" = cosine x 100). They are NOT probabilities and are model-specific.
"""
import csv, json, sys
from pathlib import Path

import numpy as np

from benchmark.common import ROOT, WORK, split

PCT = (1, 5, 25, 50, 75, 95, 99)


def parts(slug):
    sp = split(); n_g, n_p = int((sp.split == "train").sum()), int((sp.split == "test").sum())
    e = np.load(WORK / f"emb_{slug}.npz")["e"]; G, P, U = e[:n_g], e[n_g:n_g + n_p], e[n_g + n_p:]
    gl = sp[sp.split == "train"].person.to_numpy(dtype=object); pl = sp[sp.split == "test"].person.to_numpy(dtype=object)
    pos = {p: i for i, p in enumerate(gl)}; ti = np.array([pos[p] for p in pl])
    return sp, n_g, n_p, G, P, U, ti


def stats(slug):
    sp, n_g, n_p, G, P, U, ti = parts(slug)
    own = np.einsum("nd,nd->n", P, G[ti])                                         # same person: probe vs its own enrolled photo
    rng = np.random.default_rng(0); imp = np.empty((n_p, 200), np.float32)         # different people: 200 random impostors per probe (as in evaluate())
    for s in range(0, n_p, 2000):
        idx = rng.integers(0, n_g, (len(P[s:s + 2000]), 200)); idx = np.where(idx == ti[s:s + 2000, None], (idx + 1) % n_g, idx)
        imp[s:s + 2000] = np.einsum("nd,nkd->nk", P[s:s + 2000], G[idx])
    imp = imp.ravel(); unk = (U @ G.T).max(1)                                      # strangers: best match of each never-enrolled photo
    pc = lambda a: {f"p{p}": round(float(np.percentile(a, p)), 4) for p in PCT}
    d = (own.mean() - imp.mean()) / np.sqrt(0.5 * (own.var() + imp.var()))
    out = dict(same_person=pc(own), different_person=pc(imp), stranger_best_match=pc(unk), mean_same=round(float(own.mean()), 4), mean_different=round(float(imp.mean()), 4),
               dprime=round(float(d), 3),
               cosine_threshold_at_far={k: round(float(np.quantile(imp, 1 - f)), 4) for k, f in (("1e-2", 1e-2), ("1e-3", 1e-3), ("1e-4", 1e-4))},
               cosine_threshold_at_unknown_accept={k: round(float(np.quantile(unk, 1 - f)), 4) for k, f in (("1%", .01), ("5%", .05), ("10%", .10))})
    f = ROOT / "models" / slug; (f / "cosine.json").write_text(json.dumps(out, indent=1))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.6)); b = np.linspace(-0.2, 1.0, 100)
        ax.hist(imp, b, density=True, alpha=.55, label="different people (random pairs)", color="#d9534f"); ax.hist(unk, b, density=True, alpha=.55, label="strangers: best match in gallery", color="#888888")
        ax.hist(own, b, density=True, alpha=.55, label="same person", color="#2e9e5b"); t = out["cosine_threshold_at_unknown_accept"]["1%"]
        ax.axvline(t, color="k", ls="--"); ax.text(t, ax.get_ylim()[1] * .92, f" threshold at 1% stranger-accept = {t:.2f}", fontsize=8)
        ax.set_xlabel("cosine similarity (match % = x100)"); ax.set_ylabel("density"); ax.set_title(f"{slug}: cosine similarity distributions"); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(f / "cosine_distribution.png", dpi=110); plt.close(fig)
    except Exception as e:
        print("figure skipped:", e)
    return out


def photo_top5(slug):
    sp, n_g, n_p, G, P, U, ti = parts(slug)
    gname = sp[sp.split == "train"].relpath.to_numpy(); gp = sp[sp.split == "train"].person.to_numpy(dtype=object)
    pname = sp[sp.split == "test"].relpath.to_numpy(); pp = sp[sp.split == "test"].person.to_numpy(dtype=object)
    out = ROOT / "results/photo_top5"; out.mkdir(parents=True, exist_ok=True)
    hdr = ["probe_image", "true_person"] + [x for k in range(1, 6) for x in (f"top{k}_person", f"top{k}_image", f"top{k}_cosine", f"top{k}_match_pct")] + ["true_person_rank", "top1_correct"]
    with open(out / f"{slug}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr)
        for s in range(0, n_p, 2000):
            M = P[s:s + 2000] @ G.T; own = M[np.arange(len(M)), ti[s:s + 2000]]; rank = (M > own[:, None]).sum(1) + 1
            part = np.argpartition(-M, 5, axis=1)[:, :5]; o = np.take_along_axis(M, part, 1).argsort(1)[:, ::-1]; idx = np.take_along_axis(part, o, 1)
            for j in range(len(M)):
                i = s + j; row = [pname[i], pp[i]]
                for k in range(5): c = float(M[j, idx[j, k]]); row += [gp[idx[j, k]], gname[idx[j, k]], round(c, 4), round(c * 100, 1)]
                w.writerow(row + [int(rank[j]), "Y" if rank[j] == 1 else "N"])
    print("wrote", out / f"{slug}.csv")


def aggregate():
    rows = []
    for p in sorted((ROOT / "models").iterdir()):
        f = p / "cosine.json"
        if f.exists():
            c = json.loads(f.read_text()); rows.append([p.name, c["same_person"]["p50"], c["different_person"]["p50"], c["stranger_best_match"]["p50"], c["dprime"],
                                                        c["cosine_threshold_at_far"]["1e-2"], c["cosine_threshold_at_far"]["1e-3"], c["cosine_threshold_at_far"]["1e-4"],
                                                        c["cosine_threshold_at_unknown_accept"]["1%"], c["cosine_threshold_at_unknown_accept"]["5%"], c["same_person"]["p5"], c["same_person"]["p95"]])
    (ROOT / "results").mkdir(exist_ok=True)
    csv.writer(open(ROOT / "results/cosine_stats.csv", "w", newline="")).writerows([["slug", "median_same_person", "median_different_person", "median_stranger_best_match", "dprime",
                                                                                     "cos_thr_FAR_1e-2", "cos_thr_FAR_1e-3", "cos_thr_FAR_1e-4", "cos_thr_stranger_accept_1pct", "cos_thr_stranger_accept_5pct", "same_person_p5", "same_person_p95"]] + rows)
    print("results/cosine_stats.csv:", len(rows), "models")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--photo-top5"]:
        for s in a[1:]: photo_top5(s)
    else:
        for s in (a or sorted(p.name for p in (ROOT / "models").iterdir() if (WORK / f"emb_{p.name}.npz").exists())):
            c = stats(s); print(f"{s:34s} same={c['same_person']['p50']:.3f} diff={c['different_person']['p50']:.3f} stranger-best={c['stranger_best_match']['p50']:.3f} d'={c['dprime']:.2f} "
                                f"thr@FAR1e-3={c['cosine_threshold_at_far']['1e-3']:.3f} thr@1%={c['cosine_threshold_at_unknown_accept']['1%']:.3f}", flush=True)
        aggregate()
