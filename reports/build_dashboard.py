"""Build FaceMatch_Dashboard.html: exact vs HNSW(ef) charts, Top-1..Top-10, match-% distributions. Self-contained (inline plotly)."""
import glob, json, time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qdrant_client import QdrantClient, models

TH, K, C = 0.40, 10, "hnsw_test"  # hnsw_test = the 8k gallery with a real HNSW graph (m=16, ef_construct=100)
sp = pd.read_csv("work8k/split.csv")
tr = sp[sp.split == "train"].reset_index(drop=True); te = sp[sp.split == "test"].reset_index(drop=True)
un = sp[sp.split == "unknown"].reset_index(drop=True)
emb = {np.load(f)["out"].shape[0]: np.load(f)["out"] for f in glob.glob("work8k/emb_*.npz")}
P, U = emb[len(te)], emb[len(un)]
gp = tr.person.to_numpy(dtype=object); truth = te.person.to_numpy(dtype=object)
q = QdrantClient(url="http://localhost:6333", timeout=300)
assert q.get_collection(C).indexed_vectors_count == len(tr), "HNSW graph not built"

# measured earlier in this session (single-query p50 ms, 3,000 probes) - see HNSW vs Exact sheet
LAT = {"Exact": 3.0, "ef=16": 2.5, "ef=64": 3.0, "ef=128": 3.3, "ef=256": 4.1}
CFG = {"Exact": models.SearchParams(exact=True), **{f"ef={e}": models.SearchParams(hnsw_ef=e, exact=False) for e in (16, 64, 128, 256)}}


def search(Q, sparams):
    ids, sc = [], []
    for s in range(0, len(Q), 100):
        res = q.query_batch_points(C, requests=[models.QueryRequest(query=v.tolist(), limit=K, params=sparams) for v in Q[s:s + 100]])
        for r in res:
            ids.append([p.id for p in r.points]); sc.append([p.score for p in r.points])
    return np.array(ids), np.array(sc)


R = {}
for name, spar in CFG.items():
    t0 = time.time(); ids, sc = search(P, spar); uids, usc = search(U, spar)
    hit = gp[ids] == truth[:, None]                      # (n, K) true person at each rank
    topk = [float(hit[:, :k].any(1).mean()) for k in range(1, K + 1)]
    R[name] = dict(ids=ids, sc=sc, usc=usc[:, 0], topk=topk, right=hit[:, 0])
    print(f"{name}: top1={topk[0]:.4f} top5={topk[4]:.4f} top10={topk[9]:.4f} ({time.time() - t0:.0f}s)", flush=True)
ex = R["Exact"]
for name, r in R.items():
    r["agree"] = float((r["ids"][:, 0] == ex["ids"][:, 0]).mean())
    r["r10"] = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(r["ids"], ex["ids"])]))  # top-10 overlap with exact

# ---------- figures ----------
BLUE, GREEN, RED, GREY, ORG = "#1f77b4", "#2ca02c", "#d62728", "#7f7f7f", "#ff7f0e"
ks = list(range(1, K + 1)); figs = {}


def layout(fig, h=420, t=None):
    fig.update_layout(template="plotly_white", height=h, title=t, margin=dict(t=60, b=40, l=50, r=20), legend=dict(orientation="h", y=-0.2))
    return fig


def pct_hist(fig, r, row, col):
    right = r["right"]; s1 = r["sc"][:, 0] * 100
    for lab, x, c in (("Correct top-1", s1[right], GREEN), ("Wrong top-1", s1[~right], RED), ("Unknown person", r["usc"] * 100, GREY)):
        fig.add_trace(go.Histogram(x=x, name=lab, marker_color=c, opacity=.65, xbins=dict(start=-10, end=100, size=2.5),
                                   histnorm="percent", legendgroup=lab, showlegend=(row == 1 and col == 1)), row=row, col=col)
    fig.add_vline(x=TH * 100, line_dash="dash", line_color="black", row=row, col=col)
    fig.update_xaxes(title_text="Match % (cosine x 100)", row=row, col=col); fig.update_yaxes(title_text="% of probes in group", row=row, col=col)


# 1. headline: top-K exact, 500 vs 8k
s500 = json.load(open("work/summary.json"))
f = go.Figure()
f.add_bar(x=["Top-1", "Top-5", "Top-10"], y=[s500["top1"] * 100, s500["top5"] * 100, s500["top10"] * 100], name="500 people", marker_color="#9ecae1",
          text=[f"{v * 100:.1f}%" for v in (s500["top1"], s500["top5"], s500["top10"])], textposition="outside")
f.add_bar(x=["Top-1", "Top-5", "Top-10"], y=[ex["topk"][0] * 100, ex["topk"][4] * 100, ex["topk"][9] * 100], name="8,000 people", marker_color=BLUE,
          text=[f"{ex['topk'][i] * 100:.1f}%" for i in (0, 4, 9)], textposition="outside")
layout(f, 380, "Accuracy vs gallery size (exact search, 1 enrolled photo/person)"); f.update_yaxes(range=[90, 98], title="% probes where true person is in top-K")
figs["scale"] = f

# 2. exact top-1..10 curve
f = go.Figure(go.Bar(x=[f"Top-{k}" for k in ks], y=[v * 100 for v in ex["topk"]], marker_color=BLUE, text=[f"{v * 100:.2f}%" for v in ex["topk"]], textposition="outside"))
layout(f, 380, "Exact search: Top-1 to Top-10 accuracy (8,000 people, 24,000 probes)"); f.update_yaxes(range=[90, 96], title="% probes with true person in top-K")
figs["topk"] = f

# 3. threshold tradeoff (exact)
n = len(te); s1 = ex["sc"][:, 0]; right = ex["right"]; us = ex["usc"]
ths = np.round(np.arange(0.20, 0.701, 0.025), 3)
f = go.Figure()
f.add_scatter(x=ths * 100, y=[(right & (s1 >= t)).mean() * 100 for t in ths], name="Found & accepted (correct)", line=dict(color=GREEN, width=3))
f.add_scatter(x=ths * 100, y=[(~right & (s1 >= t)).mean() * 100 for t in ths], name="Confident WRONG match", line=dict(color=RED, width=3))
f.add_scatter(x=ths * 100, y=[(us >= t).mean() * 100 for t in ths], name="Unknown person wrongly accepted", line=dict(color=GREY, width=3, dash="dot"))
f.add_vline(x=TH * 100, line_dash="dash", line_color="black", annotation_text="operating point 40%")
layout(f, 420, "Threshold trade-off (exact): how many accepted vs how many mistakes"); f.update_xaxes(title="Match % threshold"); f.update_yaxes(title="% of probes")
figs["thresh"] = f

# 4. match % distribution (exact) bigger
f = make_subplots(rows=1, cols=1); pct_hist(f, ex, 1, 1); f.update_layout(barmode="overlay")
layout(f, 420, "Match % of the top-1 result (exact): correct vs wrong vs unknown people")
figs["dist"] = f

# 5. per-config panels
panels = []
for name, r in R.items():
    f = make_subplots(rows=1, cols=2, subplot_titles=("Top-1 ... Top-10 accuracy", "Top-1 Match % distribution"), column_widths=[.45, .55])
    f.add_bar(x=[f"Top-{k}" for k in ks], y=[v * 100 for v in r["topk"]], marker_color=BLUE if name == "Exact" else ORG, showlegend=False,
              text=[f"{v * 100:.1f}" for v in r["topk"]], textposition="outside", row=1, col=1)
    f.update_yaxes(range=[85, 97], title_text="% probes", row=1, col=1)
    pct_hist(f, r, 1, 2); f.update_layout(barmode="overlay")
    layout(f, 360)
    kpi = (f"top-1 {r['topk'][0] * 100:.2f}% &middot; top-5 {r['topk'][4] * 100:.2f}% &middot; top-10 {r['topk'][9] * 100:.2f}% &middot; "
           f"same top-1 as exact {r['agree'] * 100:.2f}% &middot; top-10 overlap with exact {r['r10'] * 100:.1f}% &middot; p50 latency {LAT[name]} ms")
    panels.append((name, kpi, f))

# 6. comparison across configs
names = list(R)
f = go.Figure()
for name, c in zip(names, [BLUE, "#fdae6b", "#fd8d3c", "#e6550d", "#a63603"]):
    f.add_scatter(x=[f"Top-{k}" for k in ks], y=[v * 100 for v in R[name]["topk"]], name=name, mode="lines+markers", line=dict(color=c, width=3 if name == "Exact" else 2))
layout(f, 420, "Top-K accuracy: exact vs each HNSW ef (zoomed)"); f.update_yaxes(range=[89, 95.5], title="% probes")
figs["cmp"] = f
f = make_subplots(specs=[[{"secondary_y": True}]])
f.add_bar(x=names, y=[R[n_]["agree"] * 100 for n_ in names], name="Same top-1 as exact (%)", marker_color=BLUE, text=[f"{R[n_]['agree'] * 100:.2f}" for n_ in names], textposition="outside")
f.add_scatter(x=names, y=[LAT[n_] for n_ in names], name="p50 latency (ms)", mode="lines+markers", line=dict(color=RED, width=3), secondary_y=True)
f.update_yaxes(range=[90, 101], title_text="% identical to exact", secondary_y=False); f.update_yaxes(range=[0, 6], title_text="ms", secondary_y=True)
layout(f, 420, "Fidelity vs speed: higher ef = closer to exact, slightly slower"); figs["tradeoff"] = f
# 7. latency scaling
f = go.Figure()
f.add_scatter(x=[8000, 25000, 100000], y=[3.0, 4.1, 9.1], name="Exact scan", mode="lines+markers", line=dict(color=BLUE, width=3))
f.add_scatter(x=[8000, 25000, 100000], y=[3.3, 3.4, 3.6], name="HNSW ef=128", mode="lines+markers", line=dict(color=ORG, width=3))
layout(f, 380, "Search latency vs collection size (measured p50; 25k/100k padded with random vectors)"); f.update_xaxes(title="vectors in collection", type="log"); f.update_yaxes(title="ms", range=[0, 11])
figs["lat"] = f

# ---------- html ----------
first = [True]
def div(fig):
    h = fig.to_html(full_html=False, include_plotlyjs="inline" if first[0] else False); first[0] = False; return h


r_, tp = ex["right"], ex["sc"][:, 0]
cards = [("Top-1", f"{ex['topk'][0] * 100:.1f}%", "right person ranked first"), ("Top-5", f"{ex['topk'][4] * 100:.1f}%", "in first 5"), ("Top-10", f"{ex['topk'][9] * 100:.1f}%", "in first 10"),
         ("Found @40%", f"{(r_ & (tp >= TH)).mean() * 100:.1f}%", "correct + accepted"),
         ("Precision @40%", f"{(r_ & (tp >= TH)).sum() / (tp >= TH).sum() * 100:.2f}%", "of accepted matches, % correct (known probes)"),
         ("Confident wrong", f"{(~r_ & (tp >= TH)).mean() * 100:.2f}%", "wrong person, score >= 40%"), ("Unknown accepted", f"{(ex['usc'] >= TH).mean() * 100:.1f}%", "never-enrolled scoring >= 40%"),
         ("Embed latency", "~102 ms", "per face, CPU"), ("Search latency", "~3 ms", "8k vectors")]
card_html = "".join(f"<div class=card><div class=v>{v}</div><div class=t>{t}</div><div class=s>{s}</div></div>" for t, v, s in cards)
html = f"""<!doctype html><html><head><meta charset=utf-8><title>Face Match - current position</title><style>
body{{font-family:-apple-system,Segoe UI,Arial;margin:0;background:#f5f7fa;color:#222}}.w{{max-width:1180px;margin:auto;padding:24px}}
h1{{color:#1F3864;margin:0}}h2{{color:#1F3864;border-bottom:2px solid #d9e2f3;padding-bottom:6px;margin-top:36px}}.sub{{color:#666}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}}.card{{background:#fff;border-radius:10px;padding:14px 18px;min-width:170px;box-shadow:0 1px 4px #0002}}
.v{{font-size:28px;font-weight:700;color:#1F3864}}.t{{font-weight:600}}.s{{font-size:12px;color:#777}}.box{{background:#fff;border-radius:10px;padding:14px;margin:14px 0;box-shadow:0 1px 4px #0002}}
.note{{background:#fff8e1;border-left:4px solid #f5b400;padding:10px 14px;margin:10px 0;border-radius:4px}}.kpi{{font-size:13px;color:#444;margin:4px 0 0 8px}}li{{margin:4px 0}}</style></head><body><div class=w>
<h1>Face Match (1:N) - where we are</h1>
<div class=sub>antelopev2 (SCRFD + glintr100, 512-d) &middot; Qdrant on Colima &middot; VGGFace2 112x112 &middot; 8,000 enrolled people (1 photo each), 24,000 test photos, 1,500 photos of 500 unknown people</div>
<div class=cards>{card_html}</div>
<div class=note><b>Match %</b> = cosine similarity x 100. It is <b>not</b> the probability that the match is right. Threshold shown = 40%.
Results below are on VGGFace2 (clean, pre-aligned) - the easy case; real Cashfree ID-vs-selfie photos will be harder.</div>
<h2>1. Exact search (full scan) - the reference</h2>
<div class=box>{div(figs['topk'])}</div><div class=box>{div(figs['scale'])}</div>
<div class=box>{div(figs['dist'])}<ul><li>Green (correct) clusters high, red (wrong) and grey (unknown) cluster low; the dashed line is the 40% threshold.</li>
<li>Overlap between the groups near the line is where mistakes and review cases come from.</li></ul></div>
<div class=box>{div(figs['thresh'])}<ul><li>Raising the threshold cuts confident-wrong matches and unknown-accepted, but loses correct matches.</li>
<li>At 40%: ~91% found, ~0.4% confident wrong, ~2.3% of unknown people accepted. At 45-50%, wrong accepts drop below 1%.</li></ul></div>
<h2>2. HNSW at different <code>ef</code> (one panel each)</h2>
<div class=note>HNSW is approximate: <b>ef</b> = how many candidates it keeps while searching. Low ef is fast but can miss the true nearest face; ef &ge; 128 is practically identical to exact. Measured on a real HNSW graph (m=16, ef_construct=100) over the same 8,000 faces and 24,000 probes.</div>
{"".join(f"<div class=box><h3 style='margin:0 0 4px 8px'>{n_}</h3><div class=kpi>{k}</div>{div(f)}</div>" for n_, k, f in panels)}
<h2>3. Side by side</h2>
<div class=box>{div(figs['cmp'])}</div><div class=box>{div(figs['tradeoff'])}</div>
<h2>4. Latency and scale</h2>
<div class=box>{div(figs['lat'])}<ul><li>Exact scan grows linearly (~0.066 ms per 1,000 vectors); HNSW stays almost flat.</li>
<li>At 8k-100k vectors search is under 10 ms - embedding (~100 ms) dominates, so HNSW isn't needed for speed yet. 1M-vector behaviour was not measured.</li></ul></div>
<h2>Caveats</h2><ul><li>Single random 8k gallery/seed; VGGFace2 is cleaner than real KYC photos; misses likely hard/mislabeled photos (not verified by eye).</li>
<li>Latency values were measured earlier in this session (3,000-probe test), not in this run. Face detection (SCRFD) not timed - VGGFace2 is pre-aligned.</li>
<li>HNSW recall at sizes above 8k was not measured. Weights and VGGFace2 are non-commercial research licensed.</li></ul>
</div></body></html>"""
open("FaceMatch_Dashboard.html", "w").write(html)
print("saved FaceMatch_Dashboard.html", len(html) // 1024, "KB")
