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


def _strip(sd, *prefixes):
    for pre in prefixes: sd = {(k[len(pre):] if k.startswith(pre) else k): v for k, v in sd.items()}
    return sd


def _load(model, sd, what):
    """Load a state dict and say what did not match (an incomplete match would silently give a bad model)."""
    r = model.load_state_dict(sd, strict=False)
    print(f"  [{what}] missing keys: {len(r.missing_keys)}  unexpected keys: {len(r.unexpected_keys)}", flush=True)
    assert not r.missing_keys, f"{what}: missing {r.missing_keys[:5]}"
    return model


def arcface_torch_net(raw, code_dir, name, ckpt, **kw):
    """InsightFace arcface_torch-style `backbones` package (LVFace, ViT-8R, ...): code fetched from `raw` (a raw.githubusercontent.com URL), model = get_model(name, **kw).
    RGB, (x/255-0.5)/0.5. Run one such model per process (they share the package name `backbones`)."""
    import torch
    for f in ("__init__.py", "iresnet.py", "mobilefacenet.py", "vit.py"): fetch(raw + f"backbones/{f}", f"{code_dir}/backbones/{f}")
    sys.path.insert(0, str(WEIGHTS / code_dir)); from backbones import get_model
    net = get_model(name, fp16=False, **kw)
    sd = torch.load(ckpt, map_location="cpu"); sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    return torch_model(_load(net, _strip(sd, "module."), name), [.5] * 3, [.5] * 3)


def elasticface(ckpt):
    """ElasticFace-Cos+ iResNet-100 (MS1MV2). Same iResNet definition as arcface_torch (fetched from the LVFace repo, MIT)."""
    import torch
    fetch("https://raw.githubusercontent.com/bytedance/LVFace/main/backbones/iresnet.py", "lvface_code/backbones/iresnet.py")
    sys.path.insert(0, str(WEIGHTS / "lvface_code" / "backbones")); from iresnet import iresnet100
    net = iresnet100(False, fp16=False)
    return torch_model(_load(net, _strip(torch.load(ckpt, map_location="cpu"), "module."), "elasticface iresnet100"), [.5] * 3, [.5] * 3)   # RGB, (x/255-0.5)/0.5


def magface(ckpt):
    """MagFace iResNet-100 (MS1MV2). Input as in the MagFace repo: BGR, x/255, no mean/std. Only the embedding direction is used (its norm is a quality score)."""
    import torch
    fetch("https://raw.githubusercontent.com/IrvingMeng/MagFace/main/models/iresnet.py", "magface_code/iresnet.py")
    sys.path.insert(0, str(WEIGHTS / "magface_code")); from iresnet import iresnet100
    net = iresnet100(pretrained=False, num_classes=512)
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    sd = {".".join(k.split(".")[2:]): v for k, v in sd.items() if k.startswith("features.module.")}     # features.module.conv1.weight -> conv1.weight
    return torch_model(_load(net, sd, "magface iresnet100"), [0.] * 3, [1.] * 3, rgb=False)


def lafs(ckpt):
    """LAFS Part-fViT (WebFace4M). Code fetched from github.com/szlbiubiubiu/lafs_cvpr2024 (MIT). RGB, x/255 - 0.5 (no division, as in the repo's IJB evaluation).
    The landmark branch is part of the model, so no external landmarks are needed. The checkpoint also holds the training classification head (205,990 classes)."""
    import torch
    raw = "https://raw.githubusercontent.com/szlbiubiubiu/lafs_cvpr2024/main/face_pre_pro/"
    for f in ("ViT_face.py", "mobilenet.py"): fetch(raw + f, f"lafs_code/face_pre_pro/{f}")
    sys.path.insert(0, str(WEIGHTS / "lafs_code")); from face_pre_pro.ViT_face import ViT_face_landmark_patch8
    net = ViT_face_landmark_patch8(loss_type="CosFace", GPU_ID="0", num_class=205990, image_size=112, patch_size=8, dim=768, depth=12, heads=11, mlp_dim=2048,
                                   dropout=0.1, emb_dropout=0.1, with_land=True)
    return torch_model(_load(net, _strip(torch.load(ckpt, map_location="cpu"), "module."), "lafs part-fvit"), [.5] * 3, [1.] * 3)   # (x/255 - 0.5) / 1
