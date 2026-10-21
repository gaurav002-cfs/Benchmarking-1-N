"""Face match: enroll images into Qdrant, search by image, or verify two images.

  python facematch.py enroll faces/            # name = file stem
  python facematch.py search probe.jpg [k]
  python facematch.py verify a.jpg b.jpg
"""
import sys, uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient, models

COLLECTION = "faces"
THRESHOLD = 0.4  # cosine; starting point for glintr100 -- tune on your own labeled pairs

app = FaceAnalysis(name="antelopev2", root=str(Path(__file__).parent),
                   allowed_modules=["detection", "recognition"],  # skip landmark/genderage models
                   providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
qd = QdrantClient(url="http://localhost:6333")


def embed(path):
    """Largest face -> normalized 512-d embedding (SCRFD detect + 5-pt align + glintr100)."""
    img = np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))[:, :, ::-1]  # BGR
    faces = app.get(img)
    if not faces:
        raise ValueError(f"no face in {path}")
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return f.normed_embedding


def enroll(folder):
    if not qd.collection_exists(COLLECTION):
        qd.create_collection(COLLECTION, vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE))
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        try:
            vec = embed(p)
        except ValueError as e:
            print("skip:", e)
            continue
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, p.name))  # re-enroll = overwrite, not duplicate
        qd.upsert(COLLECTION, [models.PointStruct(id=pid, vector=vec.tolist(), payload={"name": p.stem})])
        print("enrolled:", p.name)


def search(path, k=3):
    hits = qd.query_points(COLLECTION, query=embed(path).tolist(), limit=k).points
    for h in hits:
        print(f"{h.payload['name']:<30} {h.score:.3f} {'MATCH' if h.score >= THRESHOLD else '-'}")


def verify(a, b):
    s = float(embed(a) @ embed(b))
    print(f"similarity={s:.3f} -> {'SAME' if s >= THRESHOLD else 'DIFFERENT'}")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:] or ["help"]
    {"enroll": enroll, "search": lambda p, k=3: search(p, int(k)), "verify": verify}.get(
        cmd, lambda *_: print(__doc__))(*a)
