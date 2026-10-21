"""Shared protocol, helpers and evaluation for the 1:N face-embedding benchmark.

Protocol (data/split_8k.csv): 8,000 enrolled people (1 photo each = gallery), 24,000 probes (3 other photos of the same people),
1,500 photos of 500 people who are NOT enrolled (open-set). Images: VGGFace2 112x112 aligned crops. Exact search (numpy).
"""
import os, subprocess, sys, time, zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("VGGFACE2_ROOT", ROOT / "vggface2_112x112"))   # folder containing id_XXXX/ sub-folders
WEIGHTS = Path(os.environ.get("WEIGHTS_DIR", ROOT / "weights"))                # downloaded model files (git-ignored)
WORK = Path(os.environ.get("WORK_DIR", ROOT / "work"))                         # cached images / embeddings (git-ignored)
BS = 64


def split():
    sp = pd.read_csv(ROOT / "data/split_8k.csv")
    sp["split"] = pd.Categorical(sp.split, ["train", "test", "unknown"])
    return sp.sort_values("split", kind="stable").reset_index(drop=True)          # row order: gallery, probes, unknown


def load_images():
    """All benchmark images as uint8 BGR (N,112,112,3), cached to WORK/images_u8.npy."""
    WORK.mkdir(parents=True, exist_ok=True); f = WORK / "images_u8.npy"
    if f.exists(): return np.load(f, mmap_mode="r")
    from concurrent.futures import ThreadPoolExecutor
    def rd(p):
        im = cv2.imread(str(DATA_ROOT / p)); assert im is not None, f"missing image {p} (set VGGFACE2_ROOT)"
        return im if im.shape[:2] == (112, 112) else cv2.resize(im, (112, 112))
    with ThreadPoolExecutor(8) as ex: arr = np.stack(list(ex.map(rd, split().relpath.tolist())))
    np.save(f, arr); return np.load(f, mmap_mode="r")


# ---------------- downloads ----------------
def fetch(url, rel):
    """Download url to WEIGHTS/rel if missing (curl, resumable)."""
    dest = WEIGHTS / rel
    if dest.exists(): return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(dest), url], check=True)
    return dest


def fetch_zip(url, rel):
    """Download a zip and extract it to WEIGHTS/rel; returns the folder."""
    d = WEIGHTS / rel
    if not d.exists():
        z = fetch(url, rel + ".zip"); d.mkdir(parents=True); zipfile.ZipFile(z).extractall(d)
    return d


# ---------------- generic model wrappers: fn(uint8 BGR (n,112,112,3)) -> (n,d) ----------------
def onnx_arcface(path):
    """InsightFace-style ONNX recognizer: RGB, (x-127.5)/127.5, 112x112. Uses CoreML/CUDA if available, else CPU."""
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 8
    avail = ort.get_available_providers()
    def build(provs): return ort.InferenceSession(str(path), sess_options=so, providers=provs)
    s = build([p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider") if p in avail] + ["CPUExecutionProvider"]); inp = s.get_inputs()[0].name
    try: s.run(None, {inp: np.zeros((BS, 3, 112, 112), np.float32)})
    except Exception: s = build(["CPUExecutionProvider"])                           # accelerator cannot run every model; fall back to CPU
    def fn(x):
        b = ((x[..., ::-1].astype(np.float32) - 127.5) / 127.5).transpose(0, 3, 1, 2)
        return np.concatenate([s.run(None, {inp: np.ascontiguousarray(b[i:i + BS])})[0] for i in range(0, len(b), BS)])
    return fn


def torch_model(model, mean, std, size=None, rgb=True):
    """PyTorch wrapper: RGB (BGR if rgb=False), (x/255-mean)/std, optional bilinear resize to size x size. Device: cuda > mps > cpu."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(dev).eval()
    m = torch.tensor(mean, device=dev).view(1, 3, 1, 1); sd = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    @torch.no_grad()
    def fn(x):
        out = []
        for i in range(0, len(x), 128):
            t = torch.from_numpy(np.ascontiguousarray(x[i:i + 128][..., ::-1] if rgb else x[i:i + 128])).to(dev).permute(0, 3, 1, 2).float() / 255
            t = (t - m) / sd
            if size: t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
            out.append(model(t.contiguous()).float().cpu().numpy())
        return np.concatenate(out)
    return fn


# ---------------- embedding cache + evaluation ----------------
def embed_model(slug, build, imgs):
    WORK.mkdir(parents=True, exist_ok=True); cf = WORK / f"emb_{slug}.npz"
    if cf.exists():
        z = np.load(cf); return z["e"], float(z["ips"])
    fn = build(); fn(np.asarray(imgs[:BS]))                                          # build + warm up
    t0 = time.time(); parts = []
    for i in range(0, len(imgs), 1024):
        parts.append(fn(np.asarray(imgs[i:i + 1024])).astype(np.float32))
        print(f"  {slug}: {min(i + 1024, len(imgs))}/{len(imgs)} ({(i + 1024) / (time.time() - t0):.1f} img/s)", flush=True)
    e = np.concatenate(parts); ips = len(imgs) / (time.time() - t0)
    e /= np.linalg.norm(e, axis=1, keepdims=True); np.savez(cf, e=e, ips=ips); return e, ips


def evaluate(e, ips=0.0):
    """e: L2-normalised embeddings in split() row order. Returns the metrics dict used throughout the repo."""
    sp = split(); n_g, n_p, n_u = [int((sp.split == s).sum()) for s in ("train", "test", "unknown")]
    gl = sp[sp.split == "train"].person.to_numpy(dtype=object); pl = sp[sp.split == "test"].person.to_numpy(dtype=object)
    G, P, U = e[:n_g], e[n_g:n_g + n_p], e[n_g + n_p:]
    pos = {p: i for i, p in enumerate(gl)}; ti = np.array([pos[p] for p in pl])
    own = np.empty(n_p, np.float32); rank = np.empty(n_p, int); s1 = np.empty(n_p, np.float32); a1 = np.empty(n_p, int)
    for s in range(0, n_p, 2000):
        M = P[s:s + 2000] @ G.T; o = M[np.arange(len(M)), ti[s:s + 2000]]
        own[s:s + 2000] = o; rank[s:s + 2000] = (M > o[:, None]).sum(1) + 1; s1[s:s + 2000] = M.max(1); a1[s:s + 2000] = M.argmax(1)
    us = (U @ G.T).max(1); right = a1 == ti
    r = dict(dim=int(e.shape[1]), img_per_s=round(ips), top1=float((rank <= 1).mean()), top5=float((rank <= 5).mean()), top10=float((rank <= 10).mean()),
             genuine_sim_median=float(np.median(own)), impostor_best_sim_median=float(np.median(us)))
    for f in (0.01, 0.05):                       # open-set: share of enrolled probes found at a fixed unknown-accept rate
        thr = np.quantile(us, 1 - f); r[f"found@unknown_accept_{int(f * 100)}%"] = float((right & (s1 >= thr)).mean()); r[f"thr@unknown_accept_{int(f * 100)}%"] = float(thr)
    rng = np.random.default_rng(0); imp = np.empty((n_p, 200), np.float32)   # 1:1 verification: genuine pairs vs 200 random impostors per probe
    for s in range(0, n_p, 2000):
        idx = rng.integers(0, n_g, (len(P[s:s + 2000]), 200)); idx = np.where(idx == ti[s:s + 2000, None], (idx + 1) % n_g, idx)
        imp[s:s + 2000] = np.einsum("nd,nkd->nk", P[s:s + 2000], G[idx])
    imp = imp.ravel()
    for far in (1e-3, 1e-4): r[f"TAR@FAR={far:g}"] = float((own >= np.quantile(imp, 1 - far)).mean())
    far_ = np.logspace(-5, -0.5, 400); frr = np.array([(own < t).mean() for t in np.quantile(imp, 1 - far_)])
    r["EER"] = float(far_[np.argmin(np.abs(far_ - frr))])
    return r
