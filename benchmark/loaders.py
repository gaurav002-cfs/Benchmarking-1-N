"""Loaders for the PyTorch / HuggingFace models (imports are lazy so ONNX-only users do not need torch)."""
import os, sys

from benchmark.common import WEIGHTS, fetch, torch_model


def edgeface(name, ckpt):
    """EdgeFace (Idiap). Model code is fetched from github.com/otroshi/edgeface (not vendored: see its licence)."""
    import torch
    raw = "https://raw.githubusercontent.com/otroshi/edgeface/main/backbones/"
    for f in ("__init__.py", "timmfr.py"): fetch(raw + f, f"edgeface_code/backbones/{f}")
    sys.path.insert(0, str(WEIGHTS / "edgeface_code")); from backbones import get_model
    m = get_model(name); m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return torch_model(m, [.5] * 3, [.5] * 3)                                   # RGB, (x-0.5)/0.5


def timm_hub(repo):
    """timm model from the HuggingFace hub; mean/std come from the model's own pretrained_cfg."""
    import timm
    m = timm.create_model(f"hf_hub:{repo}", pretrained=True); c = m.pretrained_cfg
    return torch_model(m, c.get("mean", [.5] * 3), c.get("std", [.5] * 3))


def facenet(weights):
    """facenet-pytorch InceptionResnetV1 (weights download automatically). Expects 160x160, (x-127.5)/128; our 112x112 crops are upsampled."""
    from facenet_pytorch import InceptionResnetV1
    return torch_model(InceptionResnetV1(pretrained=weights).eval(), [127.5 / 255] * 3, [128 / 255] * 3, size=160)


def clip_vision(repo):
    """CLIP image tower only (general-purpose encoder, not face-specific). 224x224, CLIP mean/std."""
    import torch
    from transformers import CLIPModel
    class W(torch.nn.Module):
        def __init__(self): super().__init__(); self.m = CLIPModel.from_pretrained(repo)
        def forward(self, x): return self.m.get_image_features(pixel_values=x)
    return torch_model(W(), [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711], size=224)


def cvlface(repo):
    """CVLface checkpoints (minchul/*). Custom model code is downloaded from the HF repo; needs transformers==4.44.2, fvcore, omegaconf, einops.
    Run each CVLface model in its own process: they share a top-level python package called `models`."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel
    path = os.path.expanduser(f"~/.cvlface_cache/{repo}"); os.makedirs(path, exist_ok=True)
    hf_hub_download(repo, "files.txt", local_dir=path)
    for f in [x for x in open(f"{path}/files.txt").read().split("\n") if x] + ["config.json", "wrapper.py", "model.safetensors"]:
        if not os.path.exists(f"{path}/{f}"): hf_hub_download(repo, f, local_dir=path)
    cwd = os.getcwd(); os.chdir(path); sys.path.insert(0, path)
    try: m = AutoModel.from_pretrained(path, trust_remote_code=True)
    finally: os.chdir(cwd); sys.path.pop(0)
    class W(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): o = self.m(x); return o[0] if isinstance(o, (tuple, list)) else o
    return torch_model(W(m), [.5] * 3, [.5] * 3)                                 # RGB, (x-0.5)/0.5
