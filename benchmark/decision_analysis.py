"""Decision-oriented numbers per model: what happens to the 24,000 enrolled-person probes and 1,500 stranger photos at a given cosine threshold.

   python -m benchmark.decision_analysis      ->  results/decision_analysis.csv  and  results/three_band_policy.csv

For each model (its own cosine scale), two operating points on the top-1 match score:
  A) 'standard':  threshold = cosine at which 1% of stranger photos are accepted
  B) 'tight':     smallest threshold at which confident-wrong (wrong top-1 accepted) is at most 0.2% of enrolled probes
Outcomes for enrolled probes: auto-accepted correct / CONFIDENT WRONG (wrong person accepted) / not auto-accepted but the right person is in the top-5 (a reviewer could recover it)
/ not auto-accepted and not in the top-5 (missed). Strangers: share accepted.
"""
import csv, json

import numpy as np

from benchmark.cosine_stats import parts
from benchmark.common import ROOT

meta = json.load(open(ROOT / "results/model_meta.json")); rows = []
for slug in meta:
    sp, n_g, n_p, G, P, U, ti = parts(slug)
    s1 = np.empty(n_p, np.float32); a1 = np.empty(n_p, int); rank = np.empty(n_p, int)
    for s in range(0, n_p, 2000):
        M = P[s:s + 2000] @ G.T; o = M[np.arange(len(M)), ti[s:s + 2000]]; s1[s:s + 2000] = M.max(1); a1[s:s + 2000] = M.argmax(1); rank[s:s + 2000] = (M > o[:, None]).sum(1) + 1
    right = a1 == ti; unk = (U @ G.T).max(1)

    def at(T):
        acc = s1 >= T; ok = acc & right; bad = acc & ~right; rest = ~acc
        return dict(threshold=round(float(T), 4), auto_correct=ok.mean(), confident_wrong=bad.mean(), review_recoverable_top5=(rest & (rank <= 5)).mean(), missed=(rest & (rank > 5)).mean(),
                    stranger_accepted=(unk >= T).mean(), precision_of_accepted=ok.sum() / max(acc.sum(), 1))
    A = at(np.quantile(unk, 0.99))                                                  # 1% of strangers accepted
    cand = np.sort(np.unique(s1[~right]))[::-1]                                     # wrong-top-1 scores, high to low
    k = int(np.floor(0.002 * n_p)); T_tight = (cand[k] + 1e-6) if len(cand) > k else cand[-1]   # allow at most k confident-wrong
    B = at(T_tight)
    rows.append(dict(slug=slug, name=meta[slug]["name"], top1=right.mean(), top5=(rank <= 5).mean(), top10=(rank <= 10).mean(),
                     **{f"A_{k_}": v for k_, v in A.items()}, **{f"B_{k_}": v for k_, v in B.items()}))
    print(f"{slug:32s} top1={right.mean():.4f} | A thr={A['threshold']:.3f} auto={A['auto_correct']:.3f} wrong={A['confident_wrong']:.4f} review={A['review_recoverable_top5']:.3f} miss={A['missed']:.3f} | "
          f"B thr={B['threshold']:.3f} auto={B['auto_correct']:.3f} wrong={B['confident_wrong']:.4f} stranger={B['stranger_accepted']:.3f}", flush=True)
w = csv.DictWriter(open(ROOT / "results/decision_analysis.csv", "w", newline=""), fieldnames=list(rows[0]))
w.writeheader(); [w.writerow({k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}) for r in rows]
print("wrote results/decision_analysis.csv")


# ---- three-band policy (accept / review / reject) for the leading models: accept at the cosine that lets 1% of strangers through,
# review down to the cosine that lets 10% through, reject below
band = []
for slug in ["glintr100-antelopev2", "lvface-s", "lvface-t", "adaface-ir101-webface12m", "lafs-part-fvit-wf4m-lafs", "lvface-b"]:
    sp, n_g, n_p, G, P, U, ti = parts(slug)
    s1 = np.empty(n_p, np.float32); a1 = np.empty(n_p, int)
    for s in range(0, n_p, 2000):
        M = P[s:s + 2000] @ G.T; s1[s:s + 2000] = M.max(1); a1[s:s + 2000] = M.argmax(1)
    right = a1 == ti; unk = (U @ G.T).max(1); c = json.load(open(ROOT / "models" / slug / "cosine.json"))["cosine_threshold_at_unknown_accept"]
    hi, lo = c["1%"], c["10%"]; acc = s1 >= hi; rev = (s1 < hi) & (s1 >= lo); rej = s1 < lo
    band.append(dict(slug=slug, T_accept=hi, T_review=lo, enrolled_auto_correct=(acc & right).mean(), enrolled_confident_wrong=(acc & ~right).mean(), enrolled_review=rev.mean(),
                     enrolled_review_top1_right=(rev & right).mean(), enrolled_rejected=rej.mean(), enrolled_rejected_top1_right=(rej & right).mean(),
                     stranger_accepted=(unk >= hi).mean(), stranger_review=((unk < hi) & (unk >= lo)).mean(), stranger_rejected=(unk < lo).mean()))
w = csv.DictWriter(open(ROOT / "results/three_band_policy.csv", "w", newline=""), fieldnames=list(band[0])); w.writeheader()
[w.writerow({k: (round(float(v), 5) if not isinstance(v, str) else v) for k, v in r.items()}) for r in band]
print("wrote results/three_band_policy.csv")
