"""Face match: enroll images into Qdrant, search by image, or verify two images.

  python facematch.py enroll faces/            # name = file stem
  python facematch.py search probe.jpg [k]
  python facematch.py verify a.jpg b.jpg
  python facematch.py rotcheck photo.jpg       # self-test: rotate the photo and show the cosine to the unrotated embedding

Needs InsightFace's antelopev2 models in <INSIGHTFACE_ROOT>/models/antelopev2 (INSIGHTFACE_ROOT defaults to ~/.insightface) and Qdrant on localhost:6333.
Rotated photos are handled: the face is chosen over the four right-angle orientations (see robust_face), because a plain detect-and-align step can
silently produce a wrong crop for photos rotated by more than about 60 degrees (measured in the two-photo pipeline test: cosine to the unrotated embedding ~0.04).
"""
import math, os, sys, uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient, models

COLLECTION = "faces"
THRESHOLD = 0.4  # cosine; glintr100 benchmark thresholds: 0.24 = 0.1% false accepts on random pairs, 0.43 = 1% of strangers accepted (8k gallery). Tune on your own labeled pairs

app = FaceAnalysis(name="antelopev2", root=os.path.expanduser(os.environ.get("INSIGHTFACE_ROOT", "~/.insightface")),
                   allowed_modules=["detection", "recognition"],  # skip landmark/genderage models
                   providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
qd = QdrantClient(url="http://localhost:6333")


def roll_deg(f):
    """Tilt of the eye line in degrees (0 = upright face)."""
    (x0, y0), (x1, y1) = f.kps[0], f.kps[1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def robust_face(img):
    """Face for a BGR image that may be rotated: try the 4 right-angle orientations, keep the largest face of each if it is confident (score >= 0.5)
    and near-upright (eye line within 60 degrees of level), then take the most confident. None if there is no such face.
    (Picking the most confident detection alone is NOT safe: it can choose a sideways orientation for an upright photo.)"""
    best = None
    for k in range(4):
        faces = app.get(np.ascontiguousarray(np.rot90(img, k)))
        if not faces:
            continue
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if f.det_score >= 0.5 and abs(roll_deg(f)) <= 60 and (best is None or f.det_score > best.det_score):
            best = f
    return best


def embed(path):
    """Largest near-upright face -> normalized 512-d embedding (SCRFD detect + 5-pt align + glintr100), robust to rotated photos."""
    img = np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))[:, :, ::-1]  # BGR
    f = robust_face(img)
    if f is None:
        raise ValueError(f"no near-upright face in {path}")
    return f.normed_embedding


def rotcheck(path):
    """Self-test: rotate the photo and show how close each embedding stays to the unrotated one (should stay high at every angle)."""
    import cv2
    img = np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))[:, :, ::-1].copy()
    ref = robust_face(img).normed_embedding
    for ang in (0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 135, -135, 180):
        h, w = img.shape[:2]; M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        cs, sn = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * sn + w * cs), int(h * cs + w * sn); M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
        f = robust_face(cv2.warpAffine(img, M, (nw, nh)))
        print(f"rotated {ang:>5} deg: " + ("NO FACE" if f is None else f"cosine to unrotated = {float(f.normed_embedding @ ref):.3f}"))


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
    {"enroll": enroll, "search": lambda p, k=3: search(p, int(k)), "verify": verify, "rotcheck": rotcheck}.get(
        cmd, lambda *_: print(__doc__))(*a)
