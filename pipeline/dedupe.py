"""Embed every FIRST/SECOND image, index all in Qdrant, write each image's top-5 nearest *other-ID* faces to CSV.

  python dedupe.py [N]     # N = sample of IDs (default all); embeddings cached in embeddings.npz (resumable)
"""
import csv, sys
from pathlib import Path

import numpy as np
from qdrant_client import models

from facematch import embed, qd

SRC = Path.home() / "Downloads"
COL, CACHE, TOPK = "all", Path("embeddings.npz"), 5
DUP, REVIEW = 0.6, 0.4  # >=DUP: likely same person; >=REVIEW: manual check (tune after eyeballing pairs)
n = int(sys.argv[1]) if len(sys.argv) > 1 else None

ids = sorted(p.name.split("_")[0] for p in (SRC / "first").glob("*_FIRST_IMAGE.jpg"))
if n:
    ids = [str(i) for i in np.random.default_rng(0).choice(ids, n, replace=False)]
files = [(i, k, SRC / d / f"{i}_{k.upper()}_IMAGE.jpg") for i in ids for k, d in (("first", "first"), ("second", "second"))]
files = [f for f in files if f[2].exists()]

cache = dict(np.load(CACHE, allow_pickle=True)["d"].item()) if CACHE.exists() else {}  # key -> vec or None (no face)
for j, (i, k, p) in enumerate(files):
    key = f"{i}_{k}"
    if key not in cache:
        try:
            cache[key] = embed(p)
        except ValueError:
            cache[key] = None
    if j % 200 == 0:
        print(j, "/", len(files), flush=True)
        np.savez(CACHE, d=np.array(cache, dtype=object))
np.savez(CACHE, d=np.array(cache, dtype=object))

pts = [(i, k, cache[f"{i}_{k}"]) for i, k, _ in files if cache[f"{i}_{k}"] is not None]
nofaces = [f"{i}_{k}" for i, k, _ in files if cache[f"{i}_{k}"] is None]

qd.recreate_collection(COL, vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE))
for s in range(0, len(pts), 256):
    qd.upsert(COL, [models.PointStruct(id=s + j, vector=v.tolist(), payload={"id": i, "kind": k})
                    for j, (i, k, v) in enumerate(pts[s:s + 256], 0)])

rows = []
for i, k, v in pts:
    flt = models.Filter(must_not=[models.FieldCondition(key="id", match=models.MatchValue(value=i))])
    for r, h in enumerate(qd.query_points(COL, query=v.tolist(), query_filter=flt, limit=TOPK).points, 1):
        band = "duplicate" if h.score >= DUP else "review" if h.score >= REVIEW else "clear"
        rows.append([i, k, r, h.payload["id"], h.payload["kind"], round(h.score, 4), band])

with open("top5_matches.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "kind", "rank", "match_id", "match_kind", "score", "band"])
    w.writerows(rows)
Path("no_face.txt").write_text("\n".join(nofaces))

dup_ids = {r[0] for r in rows if r[6] == "duplicate"}
rev_ids = {r[0] for r in rows if r[6] == "review"} - dup_ids
print(f"\n{len(pts)} images indexed, {len(nofaces)} no-face (no_face.txt)")
print(f"IDs with a likely duplicate (>={DUP}): {len(dup_ids)} | review only ({REVIEW}-{DUP}): {len(rev_ids)} of {len(ids)}")
print("wrote top5_matches.csv")
